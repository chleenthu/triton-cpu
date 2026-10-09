"""
Blocked Matrix Multiplication (RISC-V ELF)
==========================================

The kernels of 04-blocked-matmul.py, compiled for riscv64, deployed to the
board and checked there:

* matmul_kernel on row-major A and B,
* block_transpose_combined_kernel, which re-encodes A as transposed blocks
  and B as transposed (column-of-blocks contiguous) blocks,
* matmul_kernel on those blocked encodings.

The board runner launches one kernel per executable, so the blocked inputs of
the last run are computed with torch from the layouts the encoding kernel is
checked against.
"""

import torch

import triton
import triton.language as tl
import os

DTYPE = os.getenv("DTYPE", "float32")
in_dtype = getattr(torch, DTYPE)
out_dtype = torch.float32 if in_dtype.is_floating_point else torch.int32
# Choose block size depending on dtype. We have more register
# capacity for bfloat16/float16 compared to float32.
BLOCK_SIZE_M = 8 if DTYPE == "float32" else 32
BLOCK_SIZE_N = 32
BLOCK_SIZE_K = 8 if DTYPE == "float32" else 64 // in_dtype.itemsize
GROUP_SIZE_M = 8


# This kernel is used for blocked encoding of input tensors for matmul.
#
# Blocked encoding is used to transform 2D tensor [M, N] into 4D tensor
# [M / BLOCK_SIZE_M, N / BLOCK_SIZE_N, BLOCK_SIZE_M, BLOCK_SIZE_N].
# This makes following access to blocks in matmul more efficient because
# each block is placed into a contiguous memory fragment and is likely
# to fit a single memory page.
#
# If TRANSPOSED_B is set to True then head dimensions of the RHS
# tensor are transposed. It provides contiguos placement for a column
# of blocks.
#
# If PACKED_B is set to True then B is VNNI encoded. Only works when
# BLOCKED_B is True.
#
# If TRANSPOSED_BLOCK_A is set to True then tail dimensions of the LHS
# tensor are transposed. Transposed LHS block better matches FMA lowering
# used by Triton CPU backend which processes RHS block row-by-row and LHS
# block column-by-column.
@triton.jit
def block_transpose_combined_kernel(in_a, out_a, in_b, out_b, M, N, K, BLOCK_SIZE_M: tl.constexpr,
                                    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
                                    BLOCKED_A: tl.constexpr, TRANSPOSED_BLOCK_A: tl.constexpr, BLOCKED_B: tl.constexpr,
                                    TRANSPOSED_B: tl.constexpr, PACKED_B: tl.constexpr):
    tl.static_assert(BLOCKED_A or not TRANSPOSED_BLOCK_A)
    tl.static_assert(BLOCKED_B or not TRANSPOSED_B)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    in_block_m = first_pid_m + (pid % group_size_m)
    in_block_n = (pid % num_pid_in_group) // group_size_m

    if BLOCKED_A:
        a_out_block_m = in_block_m
        A_OUT_BLOCK_SIZE_M: tl.constexpr = BLOCK_SIZE_K if TRANSPOSED_BLOCK_A else BLOCK_SIZE_M
        A_OUT_BLOCK_SIZE_K: tl.constexpr = BLOCK_SIZE_M if TRANSPOSED_BLOCK_A else BLOCK_SIZE_K
        A_OUT_BLOCKS_M = M // BLOCK_SIZE_M
        A_OUT_BLOCKS_K = K // BLOCK_SIZE_K
        A_OUT_STRIDE_M: tl.constexpr = A_OUT_BLOCK_SIZE_K
        A_OUT_STRIDE_BLOCK_M = BLOCK_SIZE_M * K
        A_OUT_STRIDE_BLOCK_K: tl.constexpr = BLOCK_SIZE_M * BLOCK_SIZE_K
        for in_block_k in tl.range(in_block_n, A_OUT_BLOCKS_K, N // BLOCK_SIZE_N):
            a_out_block_k = in_block_k
            a_in_desc = tl.make_tensor_descriptor(base=in_a, shape=(M, K), strides=(K, 1),
                                                  block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K))
            a_out_desc = tl.make_tensor_descriptor(
                base=out_a, shape=(A_OUT_BLOCKS_M, A_OUT_BLOCKS_K, A_OUT_BLOCK_SIZE_M, A_OUT_BLOCK_SIZE_K),
                strides=(A_OUT_STRIDE_BLOCK_M, A_OUT_STRIDE_BLOCK_K, A_OUT_STRIDE_M, 1),
                block_shape=(1, 1, A_OUT_BLOCK_SIZE_M, A_OUT_BLOCK_SIZE_K))
            val = a_in_desc.load((in_block_m * BLOCK_SIZE_M, in_block_k * BLOCK_SIZE_K))
            if TRANSPOSED_BLOCK_A:
                val = val.T
            val = tl.reshape(val, (1, 1, A_OUT_BLOCK_SIZE_M, A_OUT_BLOCK_SIZE_K))
            a_out_desc.store((a_out_block_m, a_out_block_k, 0, 0), val)

    if BLOCKED_B:
        B_PACKED_NUM: tl.constexpr = 32 // in_b.type.element_ty.primitive_bitwidth if PACKED_B else 1
        PACKED_BLOCK_SIZE_K: tl.constexpr = BLOCK_SIZE_K // B_PACKED_NUM if PACKED_B else BLOCK_SIZE_K
        PACKED_BLOCK_SIZE_N: tl.constexpr = BLOCK_SIZE_N * B_PACKED_NUM if PACKED_B else BLOCK_SIZE_N
        B_OUT_BLOCKS_K = N // BLOCK_SIZE_N if TRANSPOSED_B else K // BLOCK_SIZE_K
        B_OUT_BLOCKS_N = K // BLOCK_SIZE_K if TRANSPOSED_B else N // BLOCK_SIZE_N
        B_OUT_STRIDE_BLOCK_K = (K * BLOCK_SIZE_N if TRANSPOSED_B else BLOCK_SIZE_K * N)
        B_OUT_STRIDE_BLOCK_N: tl.constexpr = BLOCK_SIZE_K * BLOCK_SIZE_N
        for in_block_k in tl.range(in_block_m, K // BLOCK_SIZE_K, M // BLOCK_SIZE_M):
            b_out_block_k = in_block_n if TRANSPOSED_B else in_block_k
            b_out_block_n = in_block_k if TRANSPOSED_B else in_block_n
            b_in_desc = tl.make_tensor_descriptor(base=in_b, shape=(K, N), strides=(N, 1),
                                                  block_shape=(1, BLOCK_SIZE_N))
            b_out_desc = tl.make_tensor_descriptor(
                base=out_b, shape=(B_OUT_BLOCKS_K, B_OUT_BLOCKS_N, PACKED_BLOCK_SIZE_K, PACKED_BLOCK_SIZE_N),
                strides=(B_OUT_STRIDE_BLOCK_K, B_OUT_STRIDE_BLOCK_N, PACKED_BLOCK_SIZE_N, 1),
                block_shape=(1, 1, 1, PACKED_BLOCK_SIZE_N))
            for i in tl.range(0, BLOCK_SIZE_K // B_PACKED_NUM):
                row1 = b_in_desc.load(
                    (in_block_k * BLOCK_SIZE_K + i * B_PACKED_NUM, in_block_n * BLOCK_SIZE_N)).reshape((BLOCK_SIZE_N, ))
                if B_PACKED_NUM > 1:
                    row2 = b_in_desc.load(
                        (in_block_k * BLOCK_SIZE_K + i * B_PACKED_NUM + 1, in_block_n * BLOCK_SIZE_N)).reshape(
                            (BLOCK_SIZE_N, ))
                    if B_PACKED_NUM > 2:
                        row3 = b_in_desc.load(
                            (in_block_k * BLOCK_SIZE_K + i * B_PACKED_NUM + 2, in_block_n * BLOCK_SIZE_N)).reshape(
                                (BLOCK_SIZE_N, ))
                        row4 = b_in_desc.load(
                            (in_block_k * BLOCK_SIZE_K + i * B_PACKED_NUM + 3, in_block_n * BLOCK_SIZE_N)).reshape(
                                (BLOCK_SIZE_N, ))
                        row1 = tl.ravel(tl.join(row1, row3))
                        row2 = tl.ravel(tl.join(row2, row4))
                    row1 = tl.ravel(tl.join(row1, row2))
                b_out_desc.store((b_out_block_k, b_out_block_n, i, 0), row1.reshape((1, 1, 1, PACKED_BLOCK_SIZE_N)))


# Matmul kernel that computes a single output block [BLOCK_SIZE_M, BLOCK_SIZE_N]. LHS can be in the
# rowmajor, blocked, or blocked transposed encoding. RHS can be in rowmajor, blocked, or transposed
# blocked encoding.
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                  BLOCK_SIZE_K: tl.constexpr,
                  # number of blocks in a group
                  GROUP_SIZE_M: tl.constexpr, BLOCKED_A: tl.constexpr, TRANSPOSED_BLOCK_A: tl.constexpr,
                  BLOCKED_B: tl.constexpr, TRANSPOSED_B: tl.constexpr, PACKED_B: tl.constexpr, OUT_DTYPE: tl.constexpr):
    # TRANSPOSED_BLOCK_A means that each block in A is transposed.
    # It is allowed only for blocked input.
    assert (BLOCKED_A or not TRANSPOSED_BLOCK_A)
    # TRANSPOSED_B means that blocks of B are reordered but blocks
    # itself are not transpoed. It is allowed only for blocked input.
    assert (BLOCKED_B or not TRANSPOSED_B)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    block_m = first_pid_m + (pid % group_size_m)
    block_n = (pid % num_pid_in_group) // group_size_m

    A_BLOCK_SIZE_M: tl.constexpr = BLOCK_SIZE_K if TRANSPOSED_BLOCK_A else BLOCK_SIZE_M
    A_BLOCK_SIZE_K: tl.constexpr = BLOCK_SIZE_M if TRANSPOSED_BLOCK_A else BLOCK_SIZE_K
    A_BLOCKS_M = M // BLOCK_SIZE_M
    A_BLOCKS_K = K // BLOCK_SIZE_K
    a_stride_k: tl.constexpr = 1
    a_stride_m = A_BLOCK_SIZE_K if BLOCKED_A else K
    a_stride_block_k = A_BLOCK_SIZE_M * A_BLOCK_SIZE_K if BLOCKED_A else A_BLOCK_SIZE_K
    a_stride_block_m = BLOCK_SIZE_M * K

    B_PACKED_NUM: tl.constexpr = 32 // b_ptr.type.element_ty.primitive_bitwidth if PACKED_B else 1
    PACKED_BLOCK_SIZE_K: tl.constexpr = BLOCK_SIZE_K // B_PACKED_NUM if PACKED_B else BLOCK_SIZE_K
    PACKED_BLOCK_SIZE_N: tl.constexpr = BLOCK_SIZE_N * B_PACKED_NUM if PACKED_B else BLOCK_SIZE_N
    assert BLOCKED_B or not TRANSPOSED_B
    b_stride_n: tl.constexpr = 1
    b_stride_k = PACKED_BLOCK_SIZE_N if BLOCKED_B else N * B_PACKED_NUM
    if TRANSPOSED_B:
        b_stride_block_n = BLOCK_SIZE_N * K
        b_stride_block_k = BLOCK_SIZE_K * BLOCK_SIZE_N
    else:
        b_stride_block_n = BLOCK_SIZE_K * BLOCK_SIZE_N if BLOCKED_B else PACKED_BLOCK_SIZE_N
        b_stride_block_k = BLOCK_SIZE_K * N

    a_desc = tl.make_tensor_descriptor(base=a_ptr, shape=(A_BLOCKS_M, A_BLOCKS_K, A_BLOCK_SIZE_M, A_BLOCK_SIZE_K),
                                       strides=(a_stride_block_m, a_stride_block_k, a_stride_m, a_stride_k),
                                       block_shape=(1, 1, A_BLOCK_SIZE_M, A_BLOCK_SIZE_K))
    b_desc = tl.make_tensor_descriptor(
        base=b_ptr, shape=(K // BLOCK_SIZE_K, N // BLOCK_SIZE_N, PACKED_BLOCK_SIZE_K, PACKED_BLOCK_SIZE_N),
        strides=(b_stride_block_k, b_stride_block_n, b_stride_k, b_stride_n),
        block_shape=(1, 1, PACKED_BLOCK_SIZE_K, PACKED_BLOCK_SIZE_N))
    c_desc = tl.make_tensor_descriptor(base=c_ptr, shape=(M, N), strides=(N, 1),
                                       block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N))

    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=OUT_DTYPE)
    for block_k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = a_desc.load((block_m, block_k, 0, 0)).reshape((A_BLOCK_SIZE_M, A_BLOCK_SIZE_K))
        b = b_desc.load((block_k, block_n, 0, 0)).reshape((PACKED_BLOCK_SIZE_K, PACKED_BLOCK_SIZE_N))

        if TRANSPOSED_BLOCK_A:
            a = a.T

        if PACKED_B:
            b = tl.extra.cpu.vnni_decode(b)

        c += tl.dot(a, b, out_dtype=OUT_DTYPE)

    c_desc.store((block_m * BLOCK_SIZE_M, block_n * BLOCK_SIZE_N), c)


# %%
# Unit Test
# ---------

torch.manual_seed(0)
triton.runtime.driver.set_active_to_cpu()

size = 128
M = N = K = size
if in_dtype.is_floating_point:
    a = torch.randn((M, K), device='cpu', dtype=in_dtype)
    b = torch.randn((K, N), device='cpu', dtype=in_dtype)
else:
    a = torch.randint(0, 5, (M, K), device='cpu', dtype=in_dtype)
    b = torch.randint(0, 5, (K, N), device='cpu', dtype=in_dtype)
torch_output = torch.matmul(a.to(out_dtype), b.to(out_dtype))

# The encodings block_transpose_combined_kernel produces with BLOCKED_A,
# TRANSPOSED_BLOCK_A, BLOCKED_B and TRANSPOSED_B set:
#   a_blocked[mb, kb, i, j] = a[mb * BM + j, kb * BK + i]   (each block transposed)
#   b_blocked[nb, kb, i, j] = b[kb * BK + i, nb * BN + j]   (blocks of a column contiguous)
a_blocked = a.view(M // BLOCK_SIZE_M, BLOCK_SIZE_M, K // BLOCK_SIZE_K, BLOCK_SIZE_K).permute(0, 2, 3, 1).contiguous()
b_blocked = b.view(K // BLOCK_SIZE_K, BLOCK_SIZE_K, N // BLOCK_SIZE_N, BLOCK_SIZE_N).permute(2, 0, 1, 3).contiguous()

from triton.backends.cpu.riscv import compile_deploy_and_run

RISCV_HOST = "chlee@140.114.78.64"
RISCV_REMOTE_DIR = "~/triton-riscv-elf"
TL_DTYPE = {"float32": "fp32", "bfloat16": "bf16", "float16": "fp16", "int8": "i8"}[DTYPE]
TL_OUT_DTYPE = "fp32" if in_dtype.is_floating_point else "i32"
ATOL = 1e-3 if DTYPE == "float32" else 1e-1
grid = ((M // BLOCK_SIZE_M) * (N // BLOCK_SIZE_N), )


def flat(t):
    return t.flatten().tolist()


def run_on_board(kernel, arguments, constexprs, signature, expected, name, atol=ATOL):
    result = compile_deploy_and_run(kernel, arguments, grid, f"artifacts/riscv/{name}.elf", RISCV_HOST,
                                     constexprs=constexprs, signature=signature, expected=expected, atol=atol,
                                     remote_dir=RISCV_REMOTE_DIR)
    print(f"{name}:")
    print(result.stdout, end="")
    print(result.stderr, end="")


def matmul_constexprs(blocked):
    return {
        "BLOCK_SIZE_M": BLOCK_SIZE_M, "BLOCK_SIZE_N": BLOCK_SIZE_N, "BLOCK_SIZE_K": BLOCK_SIZE_K,  #
        "GROUP_SIZE_M": GROUP_SIZE_M,  #
        "BLOCKED_A": blocked, "TRANSPOSED_BLOCK_A": blocked,  #
        "BLOCKED_B": blocked, "TRANSPOSED_B": blocked, "PACKED_B": False,  #
        "OUT_DTYPE": tl.float32 if in_dtype.is_floating_point else tl.int32,
    }


matmul_signature = {"a_ptr": f"*{TL_DTYPE}", "b_ptr": f"*{TL_DTYPE}", "c_ptr": f"*{TL_OUT_DTYPE}"}

# Row-major inputs.
run_on_board(matmul_kernel, {
    "a_ptr": flat(a), "b_ptr": flat(b), "c_ptr": [0] * (M * N), "M": M, "N": N, "K": K
}, matmul_constexprs(False), matmul_signature, {"c_ptr": flat(torch_output)}, "rvv-blocked-matmul-rowmajor")

# Blocked encoding of A and B.
run_on_board(
    block_transpose_combined_kernel, {
        "in_a": flat(a), "out_a": [0] * (M * K), "in_b": flat(b), "out_b": [0] * (K * N),  #
        "M": M, "N": N, "K": K,
    }, {
        "BLOCK_SIZE_M": BLOCK_SIZE_M, "BLOCK_SIZE_N": BLOCK_SIZE_N, "BLOCK_SIZE_K": BLOCK_SIZE_K,  #
        "GROUP_SIZE_M": GROUP_SIZE_M,  #
        "BLOCKED_A": True, "TRANSPOSED_BLOCK_A": True, "BLOCKED_B": True, "TRANSPOSED_B": True, "PACKED_B": False,
    }, {"in_a": f"*{TL_DTYPE}", "out_a": f"*{TL_DTYPE}", "in_b": f"*{TL_DTYPE}", "out_b": f"*{TL_DTYPE}"},
    {"out_a": flat(a_blocked), "out_b": flat(b_blocked)}, "rvv-blocked-matmul-encode", atol=0)

# Blocked inputs.
run_on_board(matmul_kernel, {
    "a_ptr": flat(a_blocked), "b_ptr": flat(b_blocked), "c_ptr": [0] * (M * N), "M": M, "N": N, "K": K
}, matmul_constexprs(True), matmul_signature, {"c_ptr": flat(torch_output)}, "rvv-blocked-matmul-blocked")
