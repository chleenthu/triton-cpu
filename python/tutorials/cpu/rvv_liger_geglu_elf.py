"""
Liger GEGLU
===========

The forward and backward kernels of Liger-Kernel's GEGLU with the tanh approximation of GELU
(liger_kernel/ops/geglu.py), run on the board with a torch reference, one program per row:

geglu_fwd -- _geglu_tanh_forward_kernel: c = gelu_tanh(a) * b.
geglu_bwd -- _geglu_tanh_backward_kernel: da = dc * b * gelu_tanh'(a) and db = dc * gelu_tanh(a),
             written over a and b. Because it overwrites its inputs, it is checked after one launch
             and timed in a second run.

An irregular number of columns so the masked tail is tested.
"""

import os
import sys

sys.path.insert(0, os.path.expanduser("~/Liger-Kernel/src"))

import torch

import triton
import triton.language as tl

from liger_kernel.ops.geglu import _geglu_tanh_backward_kernel
from liger_kernel.ops.geglu import _geglu_tanh_forward_kernel

torch.manual_seed(0)
triton.runtime.driver.set_active_to_cpu()

M, N = 8, 500
BLOCK_SIZE = triton.next_power_of_2(N)

a = torch.randn(M, N, dtype=torch.float32)
b = torch.randn(M, N, dtype=torch.float32)
dc = torch.randn(M, N, dtype=torch.float32)

k = 0.7978845608028654  # sqrt(2 / pi)
t = torch.tanh(k * (a + 0.044715 * a**3))
gelu_a = 0.5 * a * (1 + t)
c = gelu_a * b
da = dc * b * (0.5 * (1 + t) + 0.5 * a * (1 - t * t) * (k * (1 + 3 * 0.044715 * a * a)))
db = dc * gelu_a

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

run_on_board(_geglu_tanh_forward_kernel, {
    "a": flat(a),
    "b": flat(b),
    "c": [0.0] * (M * N),
    "stride": N,
}, (M, ), {"n_cols": N, "BLOCK_SIZE": BLOCK_SIZE}, {"c": flat(c)}, "rvv-liger-geglu-fwd")

run_on_board(_geglu_tanh_backward_kernel, {
    "dc": flat(dc),
    "a": flat(a),
    "b": flat(b),
    "stride": N,
}, (M, ), {"n_cols": N, "BLOCK_SIZE": BLOCK_SIZE}, {
    "a": flat(da),
    "b": flat(db),
}, "rvv-liger-geglu-bwd", repeatable=False)
