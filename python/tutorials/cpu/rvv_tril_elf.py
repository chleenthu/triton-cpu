"""
Lower Triangle (tril)
=====================

`C = tril(A)` over an M x N row-major fp32 matrix, tiled like rvv_mask2d_elf.py: each program owns
one BLOCK_M x BLOCK_N tile, M and N are not multiples of the tile, and the edge tiles are guarded by
the usual box mask. The lower triangle is folded into the load mask with a plain compare:

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    box = (rm[:, None] < M) & (rn[None, :] < N)
    tri = rn[None, :] <= rm[:, None]         # element (r, c) is kept iff c <= r
    a = tl.load(a_ptr + offs, mask=box & tri, other=0.0)
    tl.store(c_ptr + offs, a, mask=box)

Per row r the load mask is still a prefix (columns [0, r - pid_n * BLOCK_N + 1) of the tile), so
on RVV each row could be one vsetvli + vle32. But the bound of `tri` is `rm[:, None]`, a range
along axis 0, not a splat, so the mask is a triangle and not a box. With TRITON_VSETVL_MINE=1:

  - the store mask `box` is recognized and becomes triton_cpu.tail_mask -> vsetvli per row;
  - the load mask `box & tri` is not: AnalyzeTailMasks fails on `tri` (analyzeMask: `is CmpIOp
    sle`, then the bound side analyzeInt returns `IntInfo Range axis=0` and the compare is rejected
    because `bound->isRange`), and the failure of one andi operand fails the whole mask. The load
    keeps the default masked lowering.

Run with TRITON_LLVM_DEBUG_ONLY=analyze-tail-masks to see the trace, and compare the dumps with and
without TRITON_VSETVL_MINE (see g.sh).
"""

import numpy as np
import torch

import triton
import triton.language as tl

BLOCK_M = 8
BLOCK_N = 16


@triton.jit
def tril_kernel(a_ptr,  # *Pointer* to input matrix A (M x N, row-major).
                c_ptr,  # *Pointer* to output matrix C (M x N, row-major).
                M, N,  # Matrix dimensions.
                stride_m,  # Row stride (elements) shared by A and C.
                BLOCK_M: tl.constexpr,  # Rows per tile.
                BLOCK_N: tl.constexpr,  # Columns per tile.
                ):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    box = (rm[:, None] < M) & (rn[None, :] < N)
    tri = rn[None, :] <= rm[:, None]
    offs = rm[:, None] * stride_m + rn[None, :]

    a = tl.load(a_ptr + offs, mask=box & tri, other=0.0)
    tl.store(c_ptr + offs, a, mask=box)


# %%
# Test against torch.tril. M = 37 and N = 45 leave a 5-row and a 13-column remainder for the
# 8 x 16 tiles, and N > M so some tiles are fully above the diagonal (all-off load mask).
torch.manual_seed(0)
M, N = 37, 45
triton.runtime.driver.set_active_to_cpu()
a = torch.rand((M, N), dtype=torch.float32)
a_list = a.flatten().tolist()

from triton.backends.cpu.riscv import compile_deploy_and_run

RISCV_HOST = "chlee@140.114.78.64"
RISCV_REMOTE_DIR = "~/triton-riscv-elf"


def run_on_board(kernel, constexprs, name):
    expected = torch.tril(a).numpy().astype(np.float32)

    arguments = {
        "a_ptr": a_list,
        "c_ptr": [-1.0] * (M * N),  # poison: every element must be written
        "M": M,
        "N": N,
        "stride_m": N,
    }
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    result = compile_deploy_and_run(kernel, arguments, grid, f"artifacts/riscv/{name}.elf", RISCV_HOST,
                                     constexprs=constexprs, expected={"c_ptr": expected.flatten().tolist()},
                                     remote_dir=RISCV_REMOTE_DIR)
    print(f"{name}:")
    print(result.stdout, end="")
    print(result.stderr, end="")


run_on_board(tril_kernel, {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N}, "rvv-tril")
