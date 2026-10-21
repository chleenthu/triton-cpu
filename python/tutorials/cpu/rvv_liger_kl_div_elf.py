"""
Liger KL divergence
===================

The forward and backward kernels of Liger-Kernel's KLDivLoss (liger_kernel/ops/kl_div.py), run on the
board with a torch reference, one program per row:

kl_div_fwd -- _kldiv_kernel_forward with a summed reduction: loss[row] = sum(t * (log(max(t, eps)) - y)),
              y the prediction in log space, t the target (not in log space).
kl_div_bwd -- _kldiv_kernel_backward: grad = -t.

An irregular number of columns so the masked tail is tested.
"""

import os
import sys

sys.path.insert(0, os.path.expanduser("~/Liger-Kernel/src"))

import torch

import triton
import triton.language as tl

from liger_kernel.ops.kl_div import _kldiv_kernel_backward
from liger_kernel.ops.kl_div import _kldiv_kernel_forward

torch.manual_seed(0)
triton.runtime.driver.set_active_to_cpu()

M, N = 8, 500
BLOCK_SIZE = triton.next_power_of_2(N)
REDUCTION_SUM = 1
eps = 1e-10

y = torch.log_softmax(torch.randn(M, N, dtype=torch.float32), dim=1)
t = torch.softmax(torch.randn(M, N, dtype=torch.float32), dim=1)

loss = (t * (torch.log(torch.clamp(t, min=eps)) - y)).sum(dim=1)
grad = -t

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

run_on_board(_kldiv_kernel_forward, {
    "y_ptr": flat(y),
    "y_stride": N,
    "gt_ptr": flat(t),
    "gt_stride": N,
    "loss_ptr": [0.0] * M,
    "loss_stride": 1,
    "n_cols": N,
    "eps": eps,
}, (M, ), {"BLOCK_SIZE": BLOCK_SIZE, "reduction": REDUCTION_SUM, "log_target": False}, {"loss_ptr": flat(loss)},
             "rvv-liger-kl-div-fwd")

run_on_board(_kldiv_kernel_backward, {
    "target_ptr": flat(t),
    "target_stride": N,
    "new_grads_ptr": [0.0] * (M * N),
    "new_grads_stride": N,
    "n_cols": N,
}, (M, ), {"BLOCK_SIZE": BLOCK_SIZE, "log_target": False}, {"new_grads_ptr": flat(grad)}, "rvv-liger-kl-div-bwd")
