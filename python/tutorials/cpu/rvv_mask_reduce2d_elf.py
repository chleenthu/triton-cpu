"""
Masked 2D Row Reduction (RISC-V ELF)
====================================

Row sums and row maxima of a [BLOCK_M, BLOCK_N] FP32 tile whose rows are only
partly valid (N < BLOCK_N columns), reduced as
tl.sum(tl.where(mask, x, 0), 1) and tl.max(tl.where(mask, x, -inf), 1): the
masked row reductions of fused attention and layer norm. See rvv_reduce2d_elf.py
for full rows.

    python rvv_mask_reduce2d_elf.py                         # default lowering
    TRITON_VSETVL_REDUCE=1 python rvv_mask_reduce2d_elf.py  # multi-dim reduction

By default, ConvertReductionOp lowers a reduction along one axis of a 2-D
tensor as a shuffle butterfly per row. TRITON_VSETVL_REDUCE=1 also enables its
multi-dimensional lowering (vector.multi_reduction). Its masked path
(llvm.vp.reduce over the valid lanes only) applies to a reduction along a
tensor's only non-unit axis, so these 2-D row reductions still reduce full
BLOCK_N rows with the identity selected into the masked-off lanes. Both are
checked against torch on the board.
"""

import torch

import triton
import triton.language as tl


@triton.jit
def mask_reduce2d_kernel(x_ptr, sum_ptr, max_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    mask = cols[None, :] < N  # [1, BLOCK_N]: the valid columns of every row
    x = tl.load(x_ptr + rows[:, None] * N + cols[None, :], mask=mask, other=0.0)  # [BLOCK_M, BLOCK_N]
    tl.store(sum_ptr + rows, tl.sum(tl.where(mask, x, 0.0), axis=1))
    tl.store(max_ptr + rows, tl.max(tl.where(mask, x, -float("inf")), axis=1))


# %%
# Unit Test
# ---------

torch.manual_seed(0)
triton.runtime.driver.set_active_to_cpu()

M, N = 64, 50
BLOCK_M, BLOCK_N = 16, 64  # 50 of 64 columns valid in every row
x = torch.randn(M, N)

from triton.backends.cpu.riscv import compile_deploy_and_run

RISCV_HOST = "chlee@140.114.78.64"
RISCV_REMOTE_DIR = "~/triton-riscv-elf"


def run_on_board(kernel, constexprs, name):
    arguments = {"x_ptr": x.flatten().tolist(), "sum_ptr": [0.0] * M, "max_ptr": [0.0] * M, "M": M, "N": N}
    # The sum may be added in another order than torch's: allow a few FP32 ulps.
    result = compile_deploy_and_run(kernel, arguments, (M // BLOCK_M, ), f"artifacts/riscv/{name}.elf", RISCV_HOST,
                                     constexprs=constexprs,
                                     expected={"sum_ptr": x.sum(dim=1).tolist(), "max_ptr": x.max(dim=1).values.tolist()},
                                     atol=1e-5, remote_dir=RISCV_REMOTE_DIR)
    print(f"{name}:")
    print(result.stdout, end="")
    print(result.stderr, end="")


run_on_board(mask_reduce2d_kernel, {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N}, "rvv-mask-reduce2d")
