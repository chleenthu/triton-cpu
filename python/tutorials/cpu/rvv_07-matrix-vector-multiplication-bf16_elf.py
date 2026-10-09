"""
Matrix-Vector Multiplication, BF16 (RISC-V ELF)
===============================================

The BF16 GEMV kernel of 07-matrix-vector-multiplication-bf16.py, compiled for
riscv64, deployed to the board and checked there.

The reference reproduces the kernel's own BF16 arithmetic on this backend,
which truncates (rounds toward zero, `vand` with 0xffff0000) every BF16
intermediate: each product a * x and each step of the per-block tl.sum
butterfly (pairwise halving over BLOCK_SIZE_N lanes) is truncated to BF16,
the blocks are accumulated in FP32 in order, and the final conversion to BF16
also truncates. With that, the result is checked exactly (atol 0).
torch.matmul(weight, x) in BF16, the original tutorial's reference, rounds
differently and does not match on this backend.
"""

import torch

import triton
import triton.language as tl

BLOCK_SIZE_M = 16
BLOCK_SIZE_N = 64
"""
Kernel for computing Y = A @ X, where A is a dense matrix with
M rows and N columns.
- Input X has shape (N,)
- A has shape (M, N)
- Output has shape (M,)
"""


@triton.jit
def gemv_kernel(
    Y,
    A,
    X,
    M,
    N,
    stride_am,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    rm = start_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = tl.arange(0, BLOCK_SIZE_N)

    A = A + (rm[:, None] * stride_am + rn[None, :])
    X = X + rn

    acc = tl.zeros((BLOCK_SIZE_M, ), dtype=tl.float32)
    for n in range(N, 0, -BLOCK_SIZE_N):
        a = tl.load(A)
        x = tl.load(X)
        acc += tl.sum(a * x[None, :], axis=1)
        A += BLOCK_SIZE_N
        X += BLOCK_SIZE_N

    y = acc.to(tl.bfloat16)
    Y = Y + rm
    tl.store(Y, y)


# %%
# Unit Test
# ---------

torch.manual_seed(0)
triton.runtime.driver.set_active_to_cpu()

M, N = 128, 1024
assert M % BLOCK_SIZE_M == 0 and N % BLOCK_SIZE_N == 0, "Masking currently not supported"
weight = torch.randn((M, N), device='cpu', dtype=torch.bfloat16)
x = torch.randn((N), device='cpu', dtype=torch.bfloat16)

from triton.backends.cpu.riscv import compile_deploy_and_run

RISCV_HOST = "chlee@140.114.78.64"
RISCV_REMOTE_DIR = "~/triton-riscv-elf"


def run_on_board(kernel, constexprs, name):
    def trunc(t):  # FP32 -> BF16 precision, rounding toward zero
        return (t.view(torch.int32) & -65536).view(torch.float32)

    p = trunc(weight.float() * x.float()[None, :]).view(M, N // BLOCK_SIZE_N, BLOCK_SIZE_N)
    while p.shape[-1] > 1:
        h = p.shape[-1] // 2
        p = trunc(p[..., :h] + p[..., h:])
    acc = torch.zeros(M)
    for k in range(N // BLOCK_SIZE_N):
        acc = acc + p[:, k, 0]
    torch_output = trunc(acc).to(torch.bfloat16)  # exact: already a BF16 value
    atol = 0.0
    arguments = {
        "Y": [0.0] * M, "A": weight.float().flatten().tolist(), "X": x.float().tolist(),  #
        "M": M, "N": N, "stride_am": weight.stride(0),
    }
    grid = (triton.cdiv(M, BLOCK_SIZE_M), )
    result = compile_deploy_and_run(kernel, arguments, grid, f"artifacts/riscv/{name}.elf", RISCV_HOST,
                                     constexprs=constexprs, signature={"Y": "*bf16", "A": "*bf16", "X": "*bf16"},
                                     expected={"Y": torch_output.float().tolist()}, atol=atol,
                                     remote_dir=RISCV_REMOTE_DIR)
    print(f"{name} (atol {atol}):")
    print(result.stdout, end="")
    print(result.stderr, end="")


run_on_board(gemv_kernel, {"BLOCK_SIZE_M": BLOCK_SIZE_M, "BLOCK_SIZE_N": BLOCK_SIZE_N}, "rvv-gemv-bf16")
