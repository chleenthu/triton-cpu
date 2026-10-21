"""
Liger PolyNorm
==============

The forward and backward kernels of Liger-Kernel's PolyNorm (liger_kernel/ops/poly_norm.py), run on the
board with a torch reference:

poly_norm_fwd -- _poly_norm_forward_kernel, one program per row:
                 y = w0 * norm(x^3) + w1 * norm(x^2) + w2 * norm(x) + b, norm(u) = u / sqrt(mean(u^2) + eps),
                 and the three rstd values are stored per row.
poly_norm_bwd -- _poly_norm_backward_kernel, each program handles rows_per_program rows: dx, and the
                 program's partial dW (3 values) and dB.

An irregular number of columns so the masked tail is tested.
"""

import os
import sys

sys.path.insert(0, os.path.expanduser("~/Liger-Kernel/src"))

import torch

import triton
import triton.language as tl

from liger_kernel.ops.poly_norm import _poly_norm_backward_kernel
from liger_kernel.ops.poly_norm import _poly_norm_forward_kernel

torch.manual_seed(0)
triton.runtime.driver.set_active_to_cpu()

M, N = 16, 500
ROWS_PER_PROGRAM = 4
BLOCK_SIZE = triton.next_power_of_2(N)
eps = 1e-6

x = torch.randn(M, N, dtype=torch.float32, requires_grad=True)
w = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float32, requires_grad=True)
b = torch.tensor([0.1], dtype=torch.float32, requires_grad=True)
dy = torch.randn(M, N, dtype=torch.float32)

rstds = [1 / torch.sqrt((x**p * x**p).mean(dim=1) + eps) for p in (3, 2, 1)]
y = sum(wp * x**p * r[:, None] for wp, p, r in zip(w, (3, 2, 1), rstds)) + b
y.backward(dy)
rstd = torch.stack(rstds, dim=1).detach()
groups = M // ROWS_PER_PROGRAM
# Per program: dW_p = sum over its rows of rstd_p * sum(dy * x^p); dB = sum of dy.
xd = x.detach()
dw = torch.stack([(rstds[i].detach()[:, None] * dy * xd**p).sum(dim=1).reshape(groups, -1).sum(dim=1)
                  for i, p in enumerate((3, 2, 1))], dim=1)
db = dy.sum(dim=1).reshape(groups, -1).sum(dim=1)

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

run_on_board(_poly_norm_forward_kernel, {
    "Y_ptr": [0.0] * (M * N),
    "Y_row_stride": N,
    "X_ptr": flat(x),
    "X_row_stride": N,
    "W_ptr": flat(w),
    "B_ptr": flat(b),
    "RSTD_ptr": [0.0] * (M * 3),
    "RSTD_row_stride": 3,
    "n_cols": N,
    "eps": eps,
}, (M, ), {"BLOCK_SIZE": BLOCK_SIZE}, {
    "Y_ptr": flat(y),
    "RSTD_ptr": flat(rstd),
}, "rvv-liger-poly-norm-fwd", atol=1e-3)

run_on_board(_poly_norm_backward_kernel, {
    "dY_ptr": flat(dy),
    "dY_row_stride": N,
    "dX_ptr": [0.0] * (M * N),
    "dX_row_stride": N,
    "X_ptr": flat(x),
    "X_row_stride": N,
    "W_ptr": flat(w),
    "RSTD_ptr": flat(rstd),
    "RSTD_row_stride": 3,
    "dW_ptr": [0.0] * (groups * 3),
    "dW_row_stride": 3,
    "dB_ptr": [0.0] * groups,
    "n_rows": M,
    "n_cols": N,
}, (groups, ), {"rows_per_program": ROWS_PER_PROGRAM, "BLOCK_SIZE": BLOCK_SIZE}, {
    "dX_ptr": flat(x.grad),
    "dW_ptr": flat(dw),
    "dB_ptr": flat(db),
}, "rvv-liger-poly-norm-bwd", atol=1e-2)
