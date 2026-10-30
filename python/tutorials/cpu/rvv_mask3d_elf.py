"""
3D Tail Mask
============

A 3D tiled `C = alpha * A + B` over a BATCH x M x N row-major fp32 tensor, where none of BATCH, M
and N is a multiple of the tile size. Each program owns one BLOCK_B x BLOCK_M x BLOCK_N tile and
guards its loads and store with a box-shaped mask:

    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rb[:, None, None] < BATCH) & (rm[None, :, None] < M) & (rn[None, None, :] < N)

The tail mask has three bounds (BATCH - rb, M - rm, N - rn), so it has two leading dimensions:
ConvertMemoryOps' getTailRowMask builds each row's mask as
create_mask((BATCH - rb > i0) && (M - rm > i1) ? N - rn : 0), with a real arith.andi of the two
leading-dimension checks (in the 2D case it folds away). Used to compare the IR with and without
TRITON_VSETVL_MINE on the RVV backend.
"""

import numpy as np
import torch

import triton
import triton.language as tl

BLOCK_B = 2
BLOCK_M = 8
BLOCK_N = 16


@triton.jit
def mask3d_kernel(a_ptr,  # *Pointer* to input tensor A (BATCH x M x N, row-major).
                  b_ptr,  # *Pointer* to input tensor B (BATCH x M x N, row-major).
                  c_ptr,  # *Pointer* to output tensor C (BATCH x M x N, row-major).
                  BATCH, M, N,  # Tensor dimensions.
                  stride_b,  # Batch stride (elements) shared by A, B and C.
                  stride_m,  # Row stride (elements) shared by A, B and C.
                  alpha,  # Scalar multiplier.
                  BLOCK_B: tl.constexpr,  # Batches per tile.
                  BLOCK_M: tl.constexpr,  # Rows per tile.
                  BLOCK_N: tl.constexpr,  # Columns per tile.
                  ):
    pid_b = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)
    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rb[:, None, None] < BATCH) & (rm[None, :, None] < M) & (rn[None, None, :] < N)
    offs = rb[:, None, None] * stride_b + rm[None, :, None] * stride_m + rn[None, None, :]

    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    c = alpha * a + b
    tl.store(c_ptr + offs, c, mask=mask)


# %%
# Test against a numpy reference. BATCH = 5, M = 37 and N = 45 leave a 1-batch, a 5-row and a
# 13-column remainder for the 2 x 8 x 16 tiles, so every kind of edge tile is exercised.
torch.manual_seed(0)
BATCH, M, N = 5, 37, 45
alpha = 1.5
triton.runtime.driver.set_active_to_cpu()
a = torch.rand((BATCH, M, N), dtype=torch.float32)
b = torch.rand((BATCH, M, N), dtype=torch.float32)
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
        "c_ptr": [0.0] * (BATCH * M * N),
        "BATCH": BATCH,
        "M": M,
        "N": N,
        "stride_b": M * N,
        "stride_m": N,
        "alpha": alpha,
    }
    grid = (triton.cdiv(BATCH, BLOCK_B), triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    result = compile_deploy_and_run(kernel, arguments, grid, f"artifacts/riscv/{name}.elf", RISCV_HOST,
                                     constexprs=constexprs, expected={"c_ptr": expected.flatten().tolist()},
                                     remote_dir=RISCV_REMOTE_DIR)
    print(f"{name}:")
    print(result.stdout, end="")
    print(result.stderr, end="")


run_on_board(mask3d_kernel, {"BLOCK_B": BLOCK_B, "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N}, "rvv-mask3d")
