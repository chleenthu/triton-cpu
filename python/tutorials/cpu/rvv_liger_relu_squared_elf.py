"""
Liger ReLU squared
==================

The forward and backward kernels of Liger-Kernel's ReLU squared (liger_kernel/ops/relu_squared.py), run on
the board with a torch reference, one program per row:

relu_squared_fwd -- _relu_squared_forward_kernel: y = relu(x)^2.
relu_squared_bwd -- _relu_squared_backward_kernel: dx = 2 * dy * relu(x).

An irregular number of columns so the masked tail is tested.
"""

import os
import sys

sys.path.insert(0, os.path.expanduser("~/Liger-Kernel/src"))

import torch

import triton
import triton.language as tl

from liger_kernel.ops.relu_squared import _relu_squared_backward_kernel
from liger_kernel.ops.relu_squared import _relu_squared_forward_kernel

torch.manual_seed(0)
triton.runtime.driver.set_active_to_cpu()

M, N = 8, 500
BLOCK_SIZE = triton.next_power_of_2(N)

x = torch.randn(M, N, dtype=torch.float32)
dy = torch.randn(M, N, dtype=torch.float32)

y = torch.relu(x)**2
dx = 2 * dy * torch.relu(x)

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

run_on_board(_relu_squared_forward_kernel, {
    "Y_ptr": [0.0] * (M * N),
    "Y_stride": N,
    "X_ptr": flat(x),
    "X_stride": N,
}, (M, ), {"n_cols": N, "BLOCK_SIZE": BLOCK_SIZE}, {"Y_ptr": flat(y)}, "rvv-liger-relu-squared-fwd")

run_on_board(_relu_squared_backward_kernel, {
    "dX_ptr": [0.0] * (M * N),
    "dX_stride": N,
    "dY_ptr": flat(dy),
    "dY_stride": N,
    "X_ptr": flat(x),
    "X_stride": N,
}, (M, ), {"n_cols": N, "BLOCK_SIZE": BLOCK_SIZE}, {"dX_ptr": flat(dx)}, "rvv-liger-relu-squared-bwd")
