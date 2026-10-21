"""
Liger Total Variation Distance
==============================

Liger-Kernel's TVD kernel (liger_kernel/ops/tvd.py), which computes the loss and its gradient together,
run on the board with a torch reference, one program per row:

tvd -- _tv_distance_kernel with the batchmean reduction and no labels: loss[row] = sum(|p - q| / 2) * scale
       and grad = sign(p - q) / 2 * scale, scale = 1 / n_rows.

An irregular number of columns so the masked tail is tested.
"""

import os
import sys

sys.path.insert(0, os.path.expanduser("~/Liger-Kernel/src"))

import torch

import triton
import triton.language as tl

from liger_kernel.ops.tvd import _tv_distance_kernel

torch.manual_seed(0)
triton.runtime.driver.set_active_to_cpu()

M, N = 8, 500
BLOCK_SIZE = triton.next_power_of_2(N)
REDUCTION_BATCHMEAN = 3
scale = 1.0 / M

p = torch.softmax(torch.randn(M, N, dtype=torch.float32), dim=1)
q = torch.softmax(torch.randn(M, N, dtype=torch.float32), dim=1)

loss = (0.5 * (p - q).abs()).sum(dim=1) * scale
grad = 0.5 * torch.sign(p - q) * scale

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

run_on_board(_tv_distance_kernel, {
    "p_ptr": flat(p),
    "p_stride": N,
    "q_ptr": flat(q),
    "q_stride": N,
    "loss_ptr": [0.0] * M,
    "loss_stride": 1,
    "grads_ptr": [0.0] * (M * N),
    "grads_stride": N,
    "label_ptr": [0] * M,
    "n_cols": N,
    "scale": scale,
}, (M, ), {"ignore_index": -100, "BLOCK_SIZE": BLOCK_SIZE, "HAS_LABEL": False, "reduction": REDUCTION_BATCHMEAN}, {
    "loss_ptr": flat(loss),
    "grads_ptr": flat(grad),
}, "rvv-liger-tvd")
