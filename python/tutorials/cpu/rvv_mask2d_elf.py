"""
2D Tail Mask
============

A 2D tiled `C = alpha * A + B` over an M x N row-major fp32 matrix, where neither M nor N is a
multiple of the tile size. Each program owns one BLOCK_M x BLOCK_N tile and guards its loads and
store with a box-shaped mask:

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rm[:, None] < M) & (rn[None, :] < N)

Tiles on the bottom edge cut rows, tiles on the right edge cut columns, and the bottom-right tile
cuts both. This is the shape AnalyzeTailMasks recognizes (cmpi of range+offset against a splat
bound, expand_dims, broadcast, andi), so with TRITON_VSETVL_MINE=1 the masks become
triton_cpu.tail_mask / vector.create_mask and the masked loads/stores lower to vsetvli loops.
Used to compare the IR with and without TRITON_VSETVL_MINE on the RVV backend.
"""

import numpy as np
import torch

import triton
import triton.language as tl

BLOCK_M = 8
BLOCK_N = 16


@triton.jit
def mask2d_kernel(a_ptr,  # *Pointer* to input matrix A (M x N, row-major).
                  b_ptr,  # *Pointer* to input matrix B (M x N, row-major).
                  c_ptr,  # *Pointer* to output matrix C (M x N, row-major).
                  M, N,  # Matrix dimensions.
                  stride_m,  # Row stride (elements) shared by A, B and C.
                  alpha,  # Scalar multiplier.
                  BLOCK_M: tl.constexpr,  # Rows per tile.
                  BLOCK_N: tl.constexpr,  # Columns per tile.
                  ):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    offs = rm[:, None] * stride_m + rn[None, :]

    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    c = alpha * a + b
    tl.store(c_ptr + offs, c, mask=mask)


# %%
# Test against a numpy reference. M = 37 and N = 45 leave a 5-row and a 13-column remainder for
# the 8 x 16 tiles, so every kind of edge tile is exercised.
torch.manual_seed(0)
M, N = 37, 45
alpha = 1.5
triton.runtime.driver.set_active_to_cpu()
a = torch.rand((M, N), dtype=torch.float32)
b = torch.rand((M, N), dtype=torch.float32)
a_list = a.flatten().tolist()
b_list = b.flatten().tolist()

from triton.backends.cpu.riscv import compile_deploy_and_run

RISCV_HOST = "chlee@140.114.78.64"
RISCV_REMOTE_DIR = "~/triton-riscv-elf"


def run_on_board(kernel, constexprs, name):
    expected = (np.float32(alpha) * a.numpy() + b.numpy()).astype(np.float32)

    arguments = {
        "a_ptr": a_list,
        "b_ptr": b_list,
        "c_ptr": [0.0] * (M * N),
        "M": M,
        "N": N,
        "stride_m": N,
        "alpha": alpha,
    }
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    result = compile_deploy_and_run(kernel, arguments, grid, f"artifacts/riscv/{name}.elf", RISCV_HOST,
                                     constexprs=constexprs, expected={"c_ptr": expected.flatten().tolist()},
                                     remote_dir=RISCV_REMOTE_DIR)
    print(f"{name}:")
    print(result.stdout, end="")
    print(result.stderr, end="")


run_on_board(mask2d_kernel, {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N}, "rvv-mask2d")
