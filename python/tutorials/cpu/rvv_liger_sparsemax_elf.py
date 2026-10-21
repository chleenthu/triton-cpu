"""
Liger Sparsemax
===============

The forward and backward kernels of Liger-Kernel's Sparsemax (liger_kernel/ops/sparsemax.py), run on the
board with a torch reference, one program per row:

sparsemax_fwd -- _sparsemax_forward_kernel: from the row sorted in descending order (sorted by torch, as
                 Liger does), the support size k and threshold tau via a cumulative sum, y = max(x - tau, 0).
sparsemax_bwd -- _sparsemax_backward_kernel: on the support, dx = dy - mean of dy over the support; 0 elsewhere.

An irregular number of columns so the masked tail is tested.
"""

import os
import sys

sys.path.insert(0, os.path.expanduser("~/Liger-Kernel/src"))

import torch

import triton
import triton.language as tl

from liger_kernel.ops.sparsemax import _sparsemax_backward_kernel
from liger_kernel.ops.sparsemax import _sparsemax_forward_kernel

torch.manual_seed(0)
triton.runtime.driver.set_active_to_cpu()

M, N = 8, 500
BLOCK_SIZE = triton.next_power_of_2(N)

x = torch.randn(M, N, dtype=torch.float32)
dy = torch.randn(M, N, dtype=torch.float32)

z = torch.sort(x, dim=1, descending=True).values
cssv = z.cumsum(dim=1)
r = torch.arange(1, N + 1, dtype=torch.float32)
k = ((1 + r * z) > cssv).sum(dim=1)
tau = (cssv.gather(1, (k - 1)[:, None]) - 1) / k[:, None]
y = torch.clamp(x - tau, min=0)
supp = y > 0
dx = torch.where(supp, dy - (dy * supp).sum(dim=1, keepdim=True) / supp.sum(dim=1, keepdim=True), 0.0)

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

run_on_board(_sparsemax_forward_kernel, {
    "x_ptr": flat(x),
    "x_stride_row": N,
    "sorted_x_ptr": flat(z),
    "sorted_x_stride_row": N,
    "o_ptr": [0.0] * (M * N),
    "o_stride_row": N,
    "n_cols": N,
}, (M, ), {"BLOCK_SIZE": BLOCK_SIZE, "num_warps": 1}, {"o_ptr": flat(y)}, "rvv-liger-sparsemax-fwd")

run_on_board(_sparsemax_backward_kernel, {
    "o_ptr": flat(y),
    "go_ptr": flat(dy),
    "gi_ptr": [0.0] * (M * N),
    "stride": N,
    "n_cols": N,
}, (M, ), {"BLOCK_SIZE": BLOCK_SIZE, "num_warps": 1}, {"gi_ptr": flat(dx)}, "rvv-liger-sparsemax-bwd")
