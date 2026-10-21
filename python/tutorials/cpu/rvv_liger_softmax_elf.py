"""
Liger Softmax
=============

The single-block forward and backward kernels of Liger-Kernel's Softmax (liger_kernel/ops/softmax.py), run
on the board with a torch reference, one program per row:

softmax_fwd -- _softmax_single_block_forward_kernel: the row in one block, y = exp(x - max) / sum.
softmax_bwd -- _softmax_single_block_backward_kernel: dx = y * (dy - sum(dy * y)).

The multi-block kernels (for rows wider than the largest block) are left out: they start from
tl.float32(-float("inf")), which this Triton's frontend rejects ("'dtype' object is not callable").

An irregular number of columns so the masked tail is tested.
"""

import os
import sys

sys.path.insert(0, os.path.expanduser("~/Liger-Kernel/src"))

import torch

import triton
import triton.language as tl

from liger_kernel.ops.softmax import _softmax_single_block_backward_kernel
from liger_kernel.ops.softmax import _softmax_single_block_forward_kernel

torch.manual_seed(0)
triton.runtime.driver.set_active_to_cpu()

M, N = 8, 500
BLOCK_SIZE = triton.next_power_of_2(N)

x = torch.randn(M, N, dtype=torch.float32)
dy = torch.randn(M, N, dtype=torch.float32)

y = torch.softmax(x, dim=1)
dx = y * (dy - (dy * y).sum(dim=1, keepdim=True))

from triton.backends.cpu.riscv import compile_deploy_and_run

RISCV_HOST = "chlee@140.114.78.64"
RISCV_REMOTE_DIR = "~/triton-riscv-elf"


def flat(t):
    return t.detach().flatten().tolist()


def run_on_board(kernel, arguments, grid, constexprs, expected, name, atol=1e-4, repeatable=True):
    # The runner launches the kernel many times before it checks the result. A kernel that overwrites
    # its own inputs is checked after a single launch, and timed in a second run without a check.
    runs = [(name, expected, None)] if repeatable else [(name, expected, "0"), (f"{name}-time", None, None)]
    for run_name, run_expected, iters in runs:
        saved = {var: os.environ.get(var) for var in ("TRITON_BENCH_ITERS", "TRITON_BENCH_WARMUP")}
        if iters is not None:
            os.environ["TRITON_BENCH_ITERS"] = iters
            os.environ["TRITON_BENCH_WARMUP"] = iters
        print(f"=== {run_name} {kernel.__name__}", flush=True)
        try:
            result = compile_deploy_and_run(kernel, arguments, grid, f"artifacts/riscv/{run_name}.elf", RISCV_HOST,
                                             constexprs=constexprs, expected=run_expected, atol=atol,
                                             remote_dir=RISCV_REMOTE_DIR)
        finally:
            for var, value in saved.items():
                if value is None:
                    os.environ.pop(var, None)
                else:
                    os.environ[var] = value
        print(f"{run_name}:")
        print(result.stdout, end="")
        print(result.stderr, end="", flush=True)

run_on_board(_softmax_single_block_forward_kernel, {
    "Y_ptr": [0.0] * (M * N),
    "Y_row_stride": N,
    "X_ptr": flat(x),
    "X_row_stride": N,
    "n_cols": N,
}, (M, ), {"BLOCK_SIZE": BLOCK_SIZE}, {"Y_ptr": flat(y)}, "rvv-liger-softmax-fwd")

run_on_board(_softmax_single_block_backward_kernel, {
    "dy_ptr": flat(dy),
    "dy_stride": N,
    "y_ptr": flat(y),
    "y_stride": N,
    "dx_ptr": [0.0] * (M * N),
    "dx_stride": N,
    "n_cols": N,
}, (M, ), {"BLOCK_SIZE": BLOCK_SIZE}, {"dx_ptr": flat(dx)}, "rvv-liger-softmax-bwd")
