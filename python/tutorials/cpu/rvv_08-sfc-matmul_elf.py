"""
Space-Filling-Curve Matrix Multiplication, BF16 (RISC-V ELF)
============================================================

The kernels of 08-sfc-matmul.py, compiled for riscv64, deployed to the board
and checked there:

* block_transpose_pack_kernel: block-packs A and block-and-VNNI-packs B, in
  the order of a generalized Hilbert curve over the blocks,
* sfc_kernel over all of K (BLOCKING_FACTOR_K = 1),
* sfc_kernel split in two along K (BLOCKING_FACTOR_K = 2): the first half
  stores its partial sum, the second loads it and finishes.

The board runner launches one kernel per executable, so each run gets the
inputs the previous kernel would have produced, computed with torch from the
layouts the packing kernel is checked against.

The block sizes are chosen for RVV rather than the AMX/AVX-512 ones of the
x86 tutorial.
"""

import functools

import torch

import triton
import triton.language as tl

from gilbert_d2xy import gilbert_d2xy


# Transforms the A matrix into a tensor of shape:
#
#  (BLOCKS_M, BLOCKS_K, BLOCK_SIZE_M, BLOCK_SIZE_K)
#
# and the B matrix into a tensor of shape:
#
#  (BLOCKS_N, BLOCKS_K, BLOCK_SIZE_K, BLOCK_SIZE_N)
#
# Data is block-packed into contiguous chunks of memory. Neighboring blocks in
# the K dimension will also be neighboring in memory. In addition, the B matrix
# is also packed in VNNI format.
@triton.jit
def block_transpose_pack_kernel(a_in_ptr, a_out_ptr, a_sfc_map_ptr, b_in_ptr, b_out_ptr, b_sfc_map_ptr, M, N, K,
                                BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    VNNI: tl.constexpr = 32 // b_in_ptr.type.element_ty.primitive_bitwidth

    pid = tl.program_id(axis=0)

    BLOCKS_M = M // BLOCK_SIZE_M
    BLOCKS_N = N // BLOCK_SIZE_N
    BLOCKS_K = K // BLOCK_SIZE_K

    # Block-pack A
    if pid < BLOCKS_M * BLOCKS_K:
        block_m = tl.load(a_sfc_map_ptr + 2 * pid)
        block_k = tl.load(a_sfc_map_ptr + 2 * pid + 1)

        a_in_desc = tl.make_tensor_descriptor(base=a_in_ptr, shape=(M, K), strides=(K, 1),
                                              block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K))
        a_out_desc = tl.make_tensor_descriptor(base=a_out_ptr, shape=(BLOCKS_M, BLOCKS_K, BLOCK_SIZE_M, BLOCK_SIZE_K),
                                               strides=(BLOCK_SIZE_M * K, BLOCK_SIZE_M * BLOCK_SIZE_K, BLOCK_SIZE_K, 1),
                                               block_shape=(1, 1, BLOCK_SIZE_M, BLOCK_SIZE_K))

        block = a_in_desc.load((block_m * BLOCK_SIZE_M, block_k * BLOCK_SIZE_K)).reshape(
            (1, 1, BLOCK_SIZE_M, BLOCK_SIZE_K))
        a_out_desc.store((block_m, block_k, 0, 0), block)

    # Block-and-VNNI-pack B
    if pid < BLOCKS_K * BLOCKS_N:
        block_k = tl.load(b_sfc_map_ptr + 2 * pid)
        block_n = tl.load(b_sfc_map_ptr + 2 * pid + 1)

        b_in_desc = tl.make_tensor_descriptor(base=b_in_ptr, shape=(K, N), strides=(N, 1),
                                              block_shape=(1, BLOCK_SIZE_N))
        b_out_desc = tl.make_tensor_descriptor(
            base=b_out_ptr, shape=(BLOCKS_N, BLOCKS_K, BLOCK_SIZE_K // VNNI, BLOCK_SIZE_N * VNNI),
            strides=(BLOCK_SIZE_N * K, BLOCK_SIZE_K * BLOCK_SIZE_N, BLOCK_SIZE_N * VNNI, 1),
            block_shape=(1, 1, 1, BLOCK_SIZE_N * VNNI))
        for i in tl.range(0, BLOCK_SIZE_K // VNNI):
            row1 = b_in_desc.load((block_k * BLOCK_SIZE_K + i * VNNI, block_n * BLOCK_SIZE_N)).reshape((BLOCK_SIZE_N, ))
            if VNNI > 1:
                row2 = b_in_desc.load((block_k * BLOCK_SIZE_K + i * VNNI + 1, block_n * BLOCK_SIZE_N)).reshape(
                    (BLOCK_SIZE_N, ))
                if VNNI > 2:
                    row3 = b_in_desc.load((block_k * BLOCK_SIZE_K + i * VNNI + 2, block_n * BLOCK_SIZE_N)).reshape(
                        (BLOCK_SIZE_N, ))
                    row4 = b_in_desc.load((block_k * BLOCK_SIZE_K + i * VNNI + 3, block_n * BLOCK_SIZE_N)).reshape(
                        (BLOCK_SIZE_N, ))
                    row1 = tl.ravel(tl.join(row1, row3))
                    row2 = tl.ravel(tl.join(row2, row4))
                row1 = tl.ravel(tl.join(row1, row2))
            b_out_desc.store((block_n, block_k, i, 0), row1.reshape((1, 1, 1, BLOCK_SIZE_N * VNNI)))


# Matmul kernel using the space curve filling approach in https://arxiv.org/abs/2601.16294v1,
# based on the generalized hilbert curve implementation from https://github.com/jakubcerveny/gilbert
#
# Each program computes a single output tile with the 2D coordinates derived from the precomputed SFC mapping.
# If `BLOCKING_FACTOR_K == 1`, then program handles all `BLOCKS_K = K // BLOCK_SIZE_K` blocks along the common dimension,
# otherwise the program performs a partial accumulation of `BLOCKS_K_PER_PROG = ⌈BLOCKS_K / BLOCKING_FACTOR_K⌉` blocks,
# starting at `ik * BLOCKS_K_PER_PROG` and ending at `min((ik + 1) * BLOCKS_K_PER_PROG, BLOCKS_K)`. The partial
# accumulation is stored in `c_tmp_ptr` and will be loaded and accumulated in the next iteration of the outer loop.
#
@triton.jit
def sfc_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    c_tmp_ptr,
    sfc_map_ptr,
    M,
    N,
    K,
    ik,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    DTYPE: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    BLOCKING_FACTOR_K: tl.constexpr,
    IS_FIRST_K_BLOCK: tl.constexpr,
    IS_LAST_K_BLOCK: tl.constexpr,
):
    VNNI: tl.constexpr = 32 // b_ptr.type.element_ty.primitive_bitwidth

    BLOCKS_M = M // BLOCK_SIZE_M
    BLOCKS_N = N // BLOCK_SIZE_N
    BLOCKS_K = K // BLOCK_SIZE_K
    BLOCKS_K_PER_PROG = tl.cdiv(BLOCKS_K, BLOCKING_FACTOR_K)

    pid = tl.program_id(axis=0)
    block_m = tl.load(sfc_map_ptr + 2 * pid)
    block_n = tl.load(sfc_map_ptr + 2 * pid + 1)
    block_k = ik * BLOCKS_K_PER_PROG

    a_desc = tl.make_tensor_descriptor(base=a_ptr, shape=(BLOCKS_M, BLOCKS_K, BLOCK_SIZE_M, BLOCK_SIZE_K),
                                       strides=(BLOCK_SIZE_M * K, BLOCK_SIZE_M * BLOCK_SIZE_K, BLOCK_SIZE_K, 1),
                                       block_shape=(1, 1, BLOCK_SIZE_M, BLOCK_SIZE_K))

    b_desc = tl.make_tensor_descriptor(base=b_ptr,
                                       shape=(BLOCKS_N, BLOCKS_K, BLOCK_SIZE_K // VNNI, BLOCK_SIZE_N * VNNI),
                                       strides=(BLOCK_SIZE_N * K, BLOCK_SIZE_K * BLOCK_SIZE_N, BLOCK_SIZE_N * VNNI, 1),
                                       block_shape=(1, 1, BLOCK_SIZE_K // VNNI, BLOCK_SIZE_N * VNNI))

    c_desc = tl.make_tensor_descriptor(base=c_ptr, shape=(BLOCKS_M, BLOCKS_N, BLOCK_SIZE_M, BLOCK_SIZE_N),
                                       strides=(BLOCK_SIZE_M * N, BLOCK_SIZE_N, N, 1),
                                       block_shape=(1, 1, BLOCK_SIZE_M, BLOCK_SIZE_N))

    if BLOCKING_FACTOR_K > 1:
        c_tmp_desc = tl.make_tensor_descriptor(base=c_tmp_ptr, shape=(BLOCKS_M, BLOCKS_N, BLOCK_SIZE_M, BLOCK_SIZE_N),
                                               strides=(BLOCK_SIZE_M * N, BLOCK_SIZE_M * BLOCK_SIZE_N, BLOCK_SIZE_N, 1),
                                               block_shape=(1, 1, BLOCK_SIZE_M, BLOCK_SIZE_N))

    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ACC_DTYPE)

    for block_ki in range(block_k, min(block_k + BLOCKS_K_PER_PROG, BLOCKS_K)):
        a = a_desc.load([block_m, block_ki, 0, 0]).reshape((BLOCK_SIZE_M, BLOCK_SIZE_K))
        b = b_desc.load([block_n, block_ki, 0, 0]).reshape((BLOCK_SIZE_K // VNNI, BLOCK_SIZE_N * VNNI))

        b = tl.extra.cpu.vnni_decode(b)

        c = tl.dot(a, b, acc=c, out_dtype=ACC_DTYPE)

    if not IS_FIRST_K_BLOCK:
        c_tmp = c_tmp_desc.load([block_m, block_n, 0, 0]).reshape((BLOCK_SIZE_M, BLOCK_SIZE_N))
        c += c_tmp

    if not IS_LAST_K_BLOCK:
        c = c.reshape((1, 1, BLOCK_SIZE_M, BLOCK_SIZE_N))
        c_tmp_desc.store([block_m, block_n, 0, 0], c)
        return

    c = c.to(DTYPE).reshape((1, 1, BLOCK_SIZE_M, BLOCK_SIZE_N))
    c_desc.store([block_m, block_n, 0, 0], c)


@functools.lru_cache
def make_sfc_tensor(x, y, dtype=torch.int32, device='cpu'):
    gilbert = (gilbert_d2xy(i, x, y) for i in range(x * y))
    return torch.tensor([c for xy in gilbert for c in xy], dtype=dtype, device=device)


# %%
# Unit Test
# ---------

torch.manual_seed(0)
triton.runtime.driver.set_active_to_cpu()

BLOCK_SIZE_M = 16
BLOCK_SIZE_N = 32
BLOCK_SIZE_K = 32
M = N = K = 64
VNNI = 2  # BF16
BLOCKS_M, BLOCKS_N, BLOCKS_K = M // BLOCK_SIZE_M, N // BLOCK_SIZE_N, K // BLOCK_SIZE_K

a = torch.randn((M, K), device='cpu', dtype=torch.bfloat16)
b = torch.randn((K, N), device='cpu', dtype=torch.bfloat16)
torch_output = torch.mm(a.float(), b.float())

# The layouts block_transpose_pack_kernel produces:
#   a_packed[mb, kb, i, j]           = a[mb * BM + i, kb * BK + j]
#   b_packed[nb, kb, i, j * VNNI + v] = b[kb * BK + i * VNNI + v, nb * BN + j]
a_packed = a.view(BLOCKS_M, BLOCK_SIZE_M, BLOCKS_K, BLOCK_SIZE_K).permute(0, 2, 1, 3).contiguous()
b_packed = b.view(BLOCKS_K, BLOCK_SIZE_K // VNNI, VNNI, BLOCKS_N, BLOCK_SIZE_N).permute(3, 0, 1, 4, 2).contiguous()
# The partial sum of the first half of K, in the blocked layout of c_tmp:
#   c_tmp[mb, nb, i, j] = partial[mb * BM + i, nb * BN + j]
half_k = BLOCK_SIZE_K * triton.cdiv(BLOCKS_K, 2)
partial = torch.mm(a[:, :half_k].float(), b[:half_k].float())
c_tmp_blocked = partial.view(BLOCKS_M, BLOCK_SIZE_M, BLOCKS_N, BLOCK_SIZE_N).permute(0, 2, 1, 3).contiguous()

sfc_map_mn = make_sfc_tensor(BLOCKS_M, BLOCKS_N)
sfc_map_mk = make_sfc_tensor(BLOCKS_M, BLOCKS_K)
sfc_map_kn = make_sfc_tensor(BLOCKS_K, BLOCKS_N)

from triton.backends.cpu.riscv import compile_deploy_and_run

RISCV_HOST = "chlee@140.114.78.64"
RISCV_REMOTE_DIR = "~/triton-riscv-elf"
# One BF16 ulp (8 bits of mantissa) at the largest output.
BF16_ATOL = 2.0**(torch.log2(torch_output.abs().max()).floor().item() - 7)


def flat(t):
    return t.float().flatten().tolist() if t.is_floating_point() else t.flatten().tolist()


def run_on_board(kernel, arguments, grid, constexprs, signature, expected, name, atol):
    result = compile_deploy_and_run(kernel, arguments, grid, f"artifacts/riscv/{name}.elf", RISCV_HOST,
                                     constexprs=constexprs, signature=signature, expected=expected, atol=atol,
                                     remote_dir=RISCV_REMOTE_DIR)
    print(f"{name}:")
    print(result.stdout, end="")
    print(result.stderr, end="")


run_on_board(
    block_transpose_pack_kernel, {
        "a_in_ptr": flat(a), "a_out_ptr": [0.0] * (M * K), "a_sfc_map_ptr": flat(sfc_map_mk),  #
        "b_in_ptr": flat(b), "b_out_ptr": [0.0] * (K * N), "b_sfc_map_ptr": flat(sfc_map_kn),  #
        "M": M, "N": N, "K": K,
    }, (max(BLOCKS_M * BLOCKS_K, BLOCKS_K * BLOCKS_N), ),
    {"BLOCK_SIZE_M": BLOCK_SIZE_M, "BLOCK_SIZE_N": BLOCK_SIZE_N, "BLOCK_SIZE_K": BLOCK_SIZE_K}, {
        "a_in_ptr": "*bf16", "a_out_ptr": "*bf16", "a_sfc_map_ptr": "*i32",  #
        "b_in_ptr": "*bf16", "b_out_ptr": "*bf16", "b_sfc_map_ptr": "*i32",
    }, {"a_out_ptr": flat(a_packed), "b_out_ptr": flat(b_packed)}, "rvv-sfc-pack", atol=0)

sfc_signature = {"a_ptr": "*bf16", "b_ptr": "*bf16", "c_ptr": "*bf16", "c_tmp_ptr": "*fp32", "sfc_map_ptr": "*i32"}


def sfc_constexprs(blocking_factor_k, first, last):
    return {
        "BLOCK_SIZE_M": BLOCK_SIZE_M, "BLOCK_SIZE_N": BLOCK_SIZE_N, "BLOCK_SIZE_K": BLOCK_SIZE_K,  #
        "DTYPE": tl.bfloat16, "ACC_DTYPE": tl.float32, "BLOCKING_FACTOR_K": blocking_factor_k,  #
        "IS_FIRST_K_BLOCK": first, "IS_LAST_K_BLOCK": last,
    }


def sfc_arguments(ik, c_tmp):
    return {
        "a_ptr": flat(a_packed), "b_ptr": flat(b_packed), "c_ptr": [0.0] * (M * N), "c_tmp_ptr": c_tmp,  #
        "sfc_map_ptr": flat(sfc_map_mn), "M": M, "N": N, "K": K, "ik": ik,
    }


grid = (BLOCKS_M * BLOCKS_N, )
run_on_board(sfc_kernel, sfc_arguments(0, [0.0] * (M * N)), grid, sfc_constexprs(1, True, True), sfc_signature,
             {"c_ptr": flat(torch_output.to(torch.bfloat16))}, "rvv-sfc-matmul", atol=BF16_ATOL)
run_on_board(sfc_kernel, sfc_arguments(0, [0.0] * (M * N)), grid, sfc_constexprs(2, True, False), sfc_signature,
             {"c_tmp_ptr": flat(c_tmp_blocked)}, "rvv-sfc-matmul-splitk-first", atol=1e-3)
run_on_board(sfc_kernel, sfc_arguments(1, flat(c_tmp_blocked)), grid, sfc_constexprs(2, False, True), sfc_signature,
             {"c_ptr": flat(torch_output.to(torch.bfloat16))}, "rvv-sfc-matmul-splitk-last", atol=BF16_ATOL)
