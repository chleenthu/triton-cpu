"""
Liger SwiGLU
============

The forward and backward kernels of Liger-Kernel's SwiGLU (liger_kernel/ops/swiglu.py), run on the board
with a torch reference, one program per row:

swiglu_fwd -- _swiglu_forward_kernel: c = silu(a) * b.
swiglu_bwd -- _swiglu_backward_kernel: da = dc * (silu(a) * (1 - sigmoid(a)) + sigmoid(a)) * b and
              db = dc * silu(a), written over a and b. Because it overwrites its inputs, it is checked
              after one launch and timed in a second run.

gate_multiplier 1, an irregular number of columns so the masked tail is tested.
"""

import os
import sys

sys.path.insert(0, os.path.expanduser("~/Liger-Kernel/src"))

import torch

import triton
import triton.language as tl

from liger_kernel.ops.swiglu import _swiglu_backward_kernel
from liger_kernel.ops.swiglu import _swiglu_forward_kernel

torch.manual_seed(0)
triton.runtime.driver.set_active_to_cpu()

M, N = 8, 500
BLOCK_SIZE = triton.next_power_of_2(N)

a = torch.randn(M, N, dtype=torch.float32)
b = torch.randn(M, N, dtype=torch.float32)
dc = torch.randn(M, N, dtype=torch.float32)

sig = torch.sigmoid(a)
silu_a = a * sig
c = silu_a * b
da = dc * (silu_a * (1 - sig) + sig) * b
db = dc * silu_a

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

run_on_board(_swiglu_forward_kernel, {
    "a_ptr": flat(a),
    "b_ptr": flat(b),
    "c_ptr": [0.0] * (M * N),
    "stride": N,
    "gate_multiplier": 1.0,
}, (M, ), {"n_cols": N, "BLOCK_SIZE": BLOCK_SIZE}, {"c_ptr": flat(c)}, "rvv-liger-swiglu-fwd")

run_on_board(_swiglu_backward_kernel, {
    "dc_ptr": flat(dc),
    "a_ptr": flat(a),
    "b_ptr": flat(b),
    "stride": N,
    "gate_multiplier": 1.0,
}, (M, ), {"n_cols": N, "BLOCK_SIZE": BLOCK_SIZE}, {
    "a_ptr": flat(da),
    "b_ptr": flat(db),
}, "rvv-liger-swiglu-bwd", repeatable=False)
