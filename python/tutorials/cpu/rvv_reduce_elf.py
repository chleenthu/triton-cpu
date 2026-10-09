"""
Masked Reduction (RISC-V ELF)
=============================

Row sums and row maxima of a matrix whose rows are shorter than the block, in
the form TRITON_VSETVL_REDUCE rewrites: a reduction of
tl.where(tail mask, x, identity) along the block's only axis.

    TRITON_VSETVL_REDUCE=1 python rvv_reduce_elf.py

With the variable set, ConvertReductionOp (mapToMaskedReduction) turns each
reduction into a masked vector.reduction over x, which LLVM lowers to
llvm.vp.reduce.fadd / llvm.vp.reduce.fmax limited to the first n_cols lanes,
and the select disappears. Without it, the identity is selected into the
masked-off lanes and all BLOCK_SIZE lanes are reduced (llvm.vector.reduce.*).
Both are checked against torch on the board; the masked float sum is
unordered (reassoc), so the sums get a small tolerance.
"""

import torch

import triton
import triton.language as tl


@triton.jit
def masked_reduce_kernel(x_ptr, sum_ptr, max_ptr, row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols  # the tail mask
    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0)
    # select(tail mask, x, identity) reduced along the only axis.
    row_sum = tl.sum(tl.where(mask, x, 0.0), axis=0)
    row_max = tl.max(tl.where(mask, x, -float("inf")), axis=0)
    tl.store(sum_ptr + row, row_sum)
    tl.store(max_ptr + row, row_max)


# %%
# Unit Test
# ---------

torch.manual_seed(0)
triton.runtime.driver.set_active_to_cpu()

n_rows, n_cols = 64, 781  # rows shorter than the block: 781 of 1024 lanes
BLOCK_SIZE = triton.next_power_of_2(n_cols)
x = torch.randn(n_rows, n_cols)

from triton.backends.cpu.riscv import compile_deploy_and_run

RISCV_HOST = "chlee@140.114.78.64"
RISCV_REMOTE_DIR = "~/triton-riscv-elf"


def run_on_board(kernel, constexprs, name):
    arguments = {
        "x_ptr": x.flatten().tolist(), "sum_ptr": [0.0] * n_rows, "max_ptr": [0.0] * n_rows,  #
        "row_stride": x.stride(0), "n_cols": n_cols,
    }
    # The sum may be added in any order (reassoc): allow a few FP32 ulps of a
    # 781-element sum. The maxima are exact.
    result = compile_deploy_and_run(kernel, arguments, (n_rows, ), f"artifacts/riscv/{name}.elf", RISCV_HOST,
                                     constexprs=constexprs,
                                     expected={"sum_ptr": x.sum(dim=1).tolist(), "max_ptr": x.max(dim=1).values.tolist()},
                                     atol=1e-4, remote_dir=RISCV_REMOTE_DIR)
    print(f"{name}:")
    print(result.stdout, end="")
    print(result.stderr, end="")


run_on_board(masked_reduce_kernel, {"BLOCK_SIZE": BLOCK_SIZE}, "rvv-reduce")
