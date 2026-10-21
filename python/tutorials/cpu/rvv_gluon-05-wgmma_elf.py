"""
Warp-Group MMA in Gluon (RISC-V ELF)
====================================

The kernels of python/tutorials/gluon/05-wgmma.py, compiled for riscv64 by the CPU backend,
deployed to the board and checked there in FP16 (FP32 accumulation):

* small_mma_kernel: D = A @ B + C for a single tile, one program,
* blocked_matmul_kernel: C = A @ B, one program per BLOCK_M x BLOCK_N output tile, looping over
  K, with B given either as [K, N] or transposed as [N, K] (TRANSPOSE_B),
* blocked_matmul_pipelined_kernel: the same loop with the loads of the next K block issued
  before the MMA of the current one (two buffers).

Differences from the GPU tutorial:

* WGMMA is Hopper-only, and the CPU path of Gluon (gluon-strip-encodings) rejects it, so every
  warpgroup_mma (+ warpgroup_mma_init / warpgroup_mma_wait) is gl.dot_fma, Gluon's generic
  FMA dot, which becomes a plain tt.dot for the CPU backend. Its operands are converted to
  DotOperandLayout(k_width=0) of the accumulator's BlockedLayout, which gl.dot_fma requires
  (the NVMMADistributedLayout / pick_wgmma_layout of the tutorial are not used).
* There are no TMA descriptors, shared memory or mbarriers: A, B, C (and D) are plain pointers
  with explicit strides, every tma.async_load into shared memory is a masked gl.load with
  other=0 (TMA zero-fills out-of-bounds elements), and every
  smem.store + fence_async_shared + tma.async_store is a masked gl.store. mbarrier.init /
  expect / wait / invalidate and the phase bookkeeping go away.
* TRANSPOSE_B: the tutorial makes a transposed view of the B tile in shared memory
  (b_smem.permute((1, 0))); here the [BLOCK_N, BLOCK_K] tile is loaded into registers and
  transposed with gl.permute.
* The pipelined kernel keeps the tutorial's double buffering with loop-carried register tiles
  in place of the two shared-memory buffers. Nothing is asynchronous on the CPU, so this only
  changes the order of the loads and the dot.
* small_mma's LHS_IN_REG and INSTR_SHAPE_N knobs only select WGMMA operand sources and
  instruction shapes, so they are not ported. num_warps is not a kernel argument, and all
  layouts use one warp, since the standalone ELF helpers compile with num_warps = 1.
* Smaller problems than the tutorial's (208 x 416 x 304, 2000 x 1000 x 2000) with 32 x 32 x 32
  blocks, so the inputs fit in the generated runner; no dimension is a multiple of its block
  size, so the masked paths are exercised. No occupancy analysis, config search or benchmarks.
"""

import torch

import triton
from triton.backends.compiler import GPUTarget
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon._runtime import GluonASTSource


@gluon.jit
def small_mma_kernel(a_ptr, b_ptr, c_ptr, d_ptr,  #
                     M: gl.constexpr, N: gl.constexpr, K: gl.constexpr):
    # The accumulator's layout; the operands use DotOperandLayouts of it.
    c_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 1], [1, 0])
    a_reg_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=c_layout, k_width=0)
    b_reg_layout: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=c_layout, k_width=0)

    offs_m = gl.arange(0, M, layout=gl.SliceLayout(1, c_layout))
    offs_n = gl.arange(0, N, layout=gl.SliceLayout(0, c_layout))
    offs_ka = gl.arange(0, K, layout=gl.SliceLayout(0, c_layout))
    offs_kb = gl.arange(0, K, layout=gl.SliceLayout(1, c_layout))

    # GPU: tma.async_load of A, B and C into shared memory, waited on with an mbarrier.
    # A has shape [M, K].
    a = gl.load(a_ptr + offs_m[:, None] * K + offs_ka[None, :])
    # B has shape [K, N].
    b = gl.load(b_ptr + offs_kb[:, None] * N + offs_n[None, :])
    # C has shape [M, N].
    c_offs = offs_m[:, None] * N + offs_n[None, :]
    c = gl.load(c_ptr + c_offs)

    # GPU: d = warpgroup_mma(a, b_smem, c, is_async=True, use_acc=True), then
    # warpgroup_mma_wait(num_outstanding=0, deps=(d, )).
    d = gl.dot_fma(gl.convert_layout(a, a_reg_layout), gl.convert_layout(b, b_reg_layout), c)

    # GPU: d_smem.store(d), fence_async_shared, tma.async_store, tma.store_wait.
    gl.store(d_ptr + c_offs, d)


@gluon.jit
def blocked_matmul_kernel(a_ptr, b_ptr, c_ptr,  #
                          M, N, K,  #
                          stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,  #
                          BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,  #
                          TRANSPOSE_B: gl.constexpr):
    dtype: gl.constexpr = a_ptr.dtype.element_ty

    # The block of C this program is processing is (pid_m, pid_n).
    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    # GPU: the WGMMA layout from pick_wgmma_layout.
    mma_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 1], [1, 0])
    a_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma_layout, k_width=0)
    b_layout: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma_layout, k_width=0)
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mma_layout)

    rows_m = off_m + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, mma_layout))
    cols_n = off_n + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mma_layout))
    rows_n = off_n + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, mma_layout))
    rows_k = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(1, mma_layout))
    cols_k = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, mma_layout))

    for k in range(0, K, BLOCK_K):
        # Load tiles of A and B. GPU: tma.async_load into a_smem / b_smem + mbarrier.wait.
        a_mask = (rows_m < M)[:, None] & (k + cols_k < K)[None, :]
        a = gl.load(a_ptr + rows_m[:, None] * stride_am + (k + cols_k)[None, :] * stride_ak, mask=a_mask, other=0.0)
        if TRANSPOSE_B:
            # B is [N, K]: load the [BLOCK_N, BLOCK_K] tile at [off_n, k].
            b_mask = (rows_n < N)[:, None] & (k + cols_k < K)[None, :]
            b = gl.load(b_ptr + rows_n[:, None] * stride_bn + (k + cols_k)[None, :] * stride_bk, mask=b_mask,
                        other=0.0)
            # GPU: a transposed view over the tile of B in shared memory, b_smem.permute((1, 0)).
            b = gl.permute(b, (1, 0))
        else:
            # B is [K, N]: load the [BLOCK_K, BLOCK_N] tile at [k, off_n].
            b_mask = (k + rows_k < K)[:, None] & (cols_n < N)[None, :]
            b = gl.load(b_ptr + (k + rows_k)[:, None] * stride_bk + cols_n[None, :] * stride_bn, mask=b_mask,
                        other=0.0)

        # GPU: warpgroup_mma(a_smem, b, acc, is_async=True) + warpgroup_mma_wait(num_outstanding=0).
        acc = gl.dot_fma(gl.convert_layout(a, a_layout), gl.convert_layout(b, b_layout), acc)

    # Downcast accumulator and store tile of C. GPU: c_smem.store + fence_async_shared + tma.async_store.
    c_mask = (rows_m < M)[:, None] & (cols_n < N)[None, :]
    gl.store(c_ptr + rows_m[:, None] * stride_cm + cols_n[None, :] * stride_cn, acc.to(dtype), mask=c_mask)


@gluon.jit
def load_tiles(k, a_ptr, b_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, rows_m, cols_n, rows_k, cols_k):
    # GPU: tma.async_load into a_smem.index(index) / b_smem.index(index) + mbarrier.wait.
    a_mask = (rows_m < M)[:, None] & (k + cols_k < K)[None, :]
    a = gl.load(a_ptr + rows_m[:, None] * stride_am + (k + cols_k)[None, :] * stride_ak, mask=a_mask, other=0.0)
    b_mask = (k + rows_k < K)[:, None] & (cols_n < N)[None, :]
    b = gl.load(b_ptr + (k + rows_k)[:, None] * stride_bk + cols_n[None, :] * stride_bn, mask=b_mask, other=0.0)
    return a, b


@gluon.jit
def blocked_matmul_pipelined_kernel(a_ptr, b_ptr, c_ptr,  #
                                    M, N, K,  #
                                    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,  #
                                    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr):
    dtype: gl.constexpr = a_ptr.dtype.element_ty

    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    mma_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 1], [1, 0])
    a_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma_layout, k_width=0)
    b_layout: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma_layout, k_width=0)
    # GPU: warpgroup_mma_init(gl.zeros(...)).
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mma_layout)

    rows_m = off_m + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, mma_layout))
    cols_n = off_n + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mma_layout))
    rows_k = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(1, mma_layout))
    cols_k = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, mma_layout))

    # GPU: 2 shared-memory buffers each for A and B, selected by `index ^= 1`. Here the current
    # tiles are loop-carried registers, and the first ones are loaded before the loop.
    a, b = load_tiles(0, a_ptr, b_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, rows_m, cols_n, rows_k,
                      cols_k)
    for k in range(BLOCK_K, K, BLOCK_K):
        # Issue the loads of the next K block, then the MMA of the current one.
        a_next, b_next = load_tiles(k, a_ptr, b_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, rows_m,
                                    cols_n, rows_k, cols_k)
        acc = gl.dot_fma(gl.convert_layout(a, a_layout), gl.convert_layout(b, b_layout), acc)
        a = a_next
        b = b_next
    # The MMA of the last K block.
    acc = gl.dot_fma(gl.convert_layout(a, a_layout), gl.convert_layout(b, b_layout), acc)

    c_mask = (rows_m < M)[:, None] & (cols_n < N)[None, :]
    gl.store(c_ptr + rows_m[:, None] * stride_cm + cols_n[None, :] * stride_cn, acc.to(dtype), mask=c_mask)


triton.runtime.driver.set_active_to_cpu()

from triton.backends.cpu import riscv
from triton.backends.cpu.riscv import compile_deploy_and_run

# The standalone ELF helpers build a Triton ASTSource; build a Gluon one instead.
riscv.ASTSource = lambda fn, signature, constexprs: GluonASTSource(fn, signature, constexprs,
                                                                   target=GPUTarget("cpu", 0, 0))

RISCV_HOST = "chlee@140.114.78.64"
RISCV_REMOTE_DIR = "~/triton-riscv-elf"


def run_on_board(kernel, arguments, grid, constexprs, signature, expected, atol, name):
    result = compile_deploy_and_run(kernel, arguments, grid, f"artifacts/riscv/{name}.elf", RISCV_HOST,
                                     constexprs=constexprs, signature=signature, expected=expected, atol=atol,
                                     remote_dir=RISCV_REMOTE_DIR)
    print(f"{name}:")
    print(result.stdout, end="")
    print(result.stderr, end="")


torch.manual_seed(0)

# small_mma: D (FP32) = A @ B + C, the tutorial's first test shape plus a larger one.
for SM, SN, SK in [(64, 32, 32), (64, 128, 64)]:
    a = torch.randn(SM, SK, dtype=torch.float16)
    b = torch.randn(SK, SN, dtype=torch.float16)
    c = torch.randn(SM, SN, dtype=torch.float32)
    d_ref = a.float() @ b.float() + c
    run_on_board(small_mma_kernel, {
        "a_ptr": a.float().flatten().tolist(),
        "b_ptr": b.float().flatten().tolist(),
        "c_ptr": c.flatten().tolist(),
        "d_ptr": [0.0] * (SM * SN),
    }, (1, ), {"M": SM, "N": SN, "K": SK}, {"a_ptr": "*fp16", "b_ptr": "*fp16", "c_ptr": "*fp32", "d_ptr": "*fp32"},
                 {"d_ptr": d_ref.flatten().tolist()}, 1e-3, f"rvv-gluon-small-mma-{SM}x{SN}x{SK}")

# blocked_matmul / blocked_matmul_pipelined: C (FP16) = A @ B.
M, N, K = 104, 208, 152
BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32
FP16 = {"a_ptr": "*fp16", "b_ptr": "*fp16", "c_ptr": "*fp16"}
BLOCKS = {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K}
grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

a = torch.randn((M, K), dtype=torch.float16)
b = torch.randn((K, N), dtype=torch.float16)
c_ref = torch.matmul(a.float(), b.float())
# One FP16 ulp at the largest |c|: C is rounded to FP16 on the store.
ATOL = 2.0**(torch.log2(c_ref.abs().max()).floor().item() - 10)
expected = {"c_ptr": c_ref.flatten().tolist()}


def matmul_arguments(b_values, stride_bk, stride_bn):
    return {
        "a_ptr": a.float().flatten().tolist(), "b_ptr": b_values, "c_ptr": [0.0] * (M * N),  #
        "M": M, "N": N, "K": K,  #
        "stride_am": K, "stride_ak": 1, "stride_bk": stride_bk, "stride_bn": stride_bn,  #
        "stride_cm": N, "stride_cn": 1,
    }


# B as [K, N].
b_kn = matmul_arguments(b.float().flatten().tolist(), N, 1)
# B transposed, stored as [N, K].
b_nk = matmul_arguments(b.T.contiguous().float().flatten().tolist(), 1, K)

run_on_board(blocked_matmul_kernel, b_kn, grid, {**BLOCKS, "TRANSPOSE_B": False}, FP16, expected, ATOL,
             "rvv-gluon-blocked-matmul")
run_on_board(blocked_matmul_kernel, b_nk, grid, {**BLOCKS, "TRANSPOSE_B": True}, FP16, expected, ATOL,
             "rvv-gluon-blocked-matmul-transpose-b")
run_on_board(blocked_matmul_pipelined_kernel, b_kn, grid, BLOCKS, FP16, expected, ATOL,
             "rvv-gluon-blocked-matmul-pipelined")
