"""
Matrix-Vector Multiplication (RISC-V ELF)
=========================================

The FP32 GEMV kernel of 06-matrix-vector-multiplication.py, compiled for
riscv64, deployed to the board and checked there against torch.matmul.
"""

import torch

import triton
import triton.language as tl

BLOCK_SIZE_M = 1
BLOCK_SIZE_N = 512
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

    Y = Y + rm
    tl.store(Y, acc)


# %%
# Unit Test
# ---------

torch.manual_seed(0)
triton.runtime.driver.set_active_to_cpu()

M, N = 128, 1024
assert M % BLOCK_SIZE_M == 0 and N % BLOCK_SIZE_N == 0, "Masking currently not supported"
weight = torch.randn((M, N), device='cpu', dtype=torch.float32)
x = torch.randn((N), device='cpu', dtype=torch.float32)

from triton.backends.cpu.riscv import compile_deploy_and_run

RISCV_HOST = "chlee@140.114.78.64"
RISCV_REMOTE_DIR = "~/triton-riscv-elf"


def run_on_board(kernel, constexprs, name):
    # torch.matmul selects lower precision kernels on some CPUs if x is 1-d.
    torch_output = torch.matmul(weight, x[:, None]).reshape(-1)
    arguments = {
        "Y": [0.0] * M, "A": weight.flatten().tolist(), "X": x.tolist(),  #
        "M": M, "N": N, "stride_am": weight.stride(0),
    }
    grid = (triton.cdiv(M, BLOCK_SIZE_M), )
    result = compile_deploy_and_run(kernel, arguments, grid, f"artifacts/riscv/{name}.elf", RISCV_HOST,
                                     constexprs=constexprs, expected={"Y": torch_output.tolist()}, atol=1e-4,
                                     remote_dir=RISCV_REMOTE_DIR)
    print(f"{name}:")
    print(result.stdout, end="")
    print(result.stderr, end="")


run_on_board(gemv_kernel, {"BLOCK_SIZE_M": BLOCK_SIZE_M, "BLOCK_SIZE_N": BLOCK_SIZE_N}, "rvv-gemv")
