"""
Liger LayerNorm
===============

The forward and backward kernels of Liger-Kernel's LayerNorm (liger_kernel/ops/layer_norm.py), run on the
board with a torch reference:

layer_norm_fwd -- _layer_norm_forward_kernel, one program per row: y = (x - mean) * rstd * w + b, and
                  mean and rstd are stored per row.
layer_norm_bwd -- _layer_norm_backward_kernel, each program handles rows_per_program rows:
                  dx = (w * dy - (xhat * c1 + c2)) * rstd with c1 = mean(xhat * w * dy), c2 = mean(w * dy),
                  and the program's partial dW and dB are stored in its own row.

An irregular number of columns so the masked tail is tested.
"""

import os
import sys

sys.path.insert(0, os.path.expanduser("~/Liger-Kernel/src"))

import torch

import triton
import triton.language as tl

from liger_kernel.ops.layer_norm import _layer_norm_backward_kernel
from liger_kernel.ops.layer_norm import _layer_norm_forward_kernel

torch.manual_seed(0)
triton.runtime.driver.set_active_to_cpu()

M, N = 16, 500
ROWS_PER_PROGRAM = 4
BLOCK_SIZE = triton.next_power_of_2(N)
eps = 1e-6

x = torch.randn(M, N, dtype=torch.float32)
w = torch.rand(N, dtype=torch.float32)
b = torch.rand(N, dtype=torch.float32)
dy = torch.randn(M, N, dtype=torch.float32)

mean = x.mean(dim=1)
rstd = 1 / torch.sqrt(x.var(dim=1, unbiased=False) + eps)
xhat = (x - mean[:, None]) * rstd[:, None]
y = xhat * w + b
wdy = w * dy
c1 = (xhat * wdy).mean(dim=1, keepdim=True)
c2 = wdy.mean(dim=1, keepdim=True)
dx = (wdy - (xhat * c1 + c2)) * rstd[:, None]
dw = (dy * xhat).reshape(M // ROWS_PER_PROGRAM, ROWS_PER_PROGRAM, N).sum(dim=1)
db = dy.reshape(M // ROWS_PER_PROGRAM, ROWS_PER_PROGRAM, N).sum(dim=1)

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

run_on_board(_layer_norm_forward_kernel, {
    "Y_ptr": [0.0] * (M * N),
    "Y_row_stride": N,
    "X_ptr": flat(x),
    "X_row_stride": N,
    "W_ptr": flat(w),
    "W_row_stride": 1,
    "B_ptr": flat(b),
    "B_row_stride": 1,
    "Mean_ptr": [0.0] * M,
    "Mean_row_stride": 1,
    "RSTD_ptr": [0.0] * M,
    "RSTD_row_stride": 1,
    "n_cols": N,
    "eps": eps,
}, (M, ), {"BLOCK_SIZE": BLOCK_SIZE}, {
    "Y_ptr": flat(y),
    "Mean_ptr": flat(mean),
    "RSTD_ptr": flat(rstd),
}, "rvv-liger-layer-norm-fwd")

run_on_board(_layer_norm_backward_kernel, {
    "X_ptr": flat(x),
    "stride_x": N,
    "W_ptr": flat(w),
    "Mean_ptr": flat(mean),
    "stride_mean": 1,
    "RSTD_ptr": flat(rstd),
    "stride_rstd": 1,
    "DX_ptr": [0.0] * (M * N),
    "stride_dx": N,
    "DW_ptr": [0.0] * (M // ROWS_PER_PROGRAM * N),
    "stride_dw": N,
    "DB_ptr": [0.0] * (M // ROWS_PER_PROGRAM * N),
    "stride_db": N,
    "DY_ptr": flat(dy),
    "stride_dy": N,
    "n_rows": M,
    "n_cols": N,
}, (M // ROWS_PER_PROGRAM, ), {
    "rows_per_program": ROWS_PER_PROGRAM, "BLOCK_SIZE": BLOCK_SIZE, "COMPUTE_DW": True, "COMPUTE_DB": True
}, {
    "DX_ptr": flat(dx),
    "DW_ptr": flat(dw),
    "DB_ptr": flat(db),
}, "rvv-liger-layer-norm-bwd")
