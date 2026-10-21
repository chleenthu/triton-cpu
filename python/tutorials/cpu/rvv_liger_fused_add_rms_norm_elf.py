"""
Liger fused add + RMSNorm
=========================

The forward and backward kernels of Liger-Kernel's fused residual add + RMSNorm
(liger_kernel/ops/fused_add_rms_norm.py), run on the board with a torch reference:

fused_add_rms_norm_fwd -- _fused_add_rms_norm_forward_kernel, one program per row: s = x + r is stored as
                          the new residual, y = s * rstd * (offset + w), and rstd is stored per row.
fused_add_rms_norm_bwd -- _fused_add_rms_norm_backward_kernel, each program handles rows_per_program rows:
                          the RMSNorm gradient of s plus the residual's own gradient dS_out, and the
                          program's partial dW in its own row.

Llama casting mode (fp32 compute), offset 0, an irregular number of columns so the masked tail is tested.
"""

import os
import sys

sys.path.insert(0, os.path.expanduser("~/Liger-Kernel/src"))

import torch

import triton
import triton.language as tl

from liger_kernel.ops.fused_add_rms_norm import _fused_add_rms_norm_backward_kernel
from liger_kernel.ops.fused_add_rms_norm import _fused_add_rms_norm_forward_kernel

torch.manual_seed(0)
triton.runtime.driver.set_active_to_cpu()

M, N = 16, 500
ROWS_PER_PROGRAM = 4
BLOCK_SIZE = triton.next_power_of_2(N)
eps = 1e-6
CASTING_MODE_LLAMA = 0

x = torch.randn(M, N, dtype=torch.float32)
r = torch.randn(M, N, dtype=torch.float32)
w = torch.rand(N, dtype=torch.float32)
dy = torch.randn(M, N, dtype=torch.float32)
ds_out = torch.randn(M, N, dtype=torch.float32)

s = x + r
rstd = 1 / torch.sqrt((s * s).mean(dim=1) + eps)
y = s * rstd[:, None] * w
m = dy * w
dx = rstd[:, None] * m - (rstd**3 / N * (m * s).sum(dim=1))[:, None] * s + ds_out
dw = (dy * s * rstd[:, None]).reshape(M // ROWS_PER_PROGRAM, ROWS_PER_PROGRAM, N).sum(dim=1)

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

run_on_board(_fused_add_rms_norm_forward_kernel, {
    "Y_ptr": [0.0] * (M * N),
    "Y_row_stride": N,
    "S_ptr": [0.0] * (M * N),
    "S_row_stride": N,
    "X_ptr": flat(x),
    "X_row_stride": N,
    "R_ptr": flat(r),
    "R_row_stride": N,
    "W_ptr": flat(w),
    "W_row_stride": 1,
    "RSTD_ptr": [0.0] * M,
    "RSTD_row_stride": 1,
    "n_cols": N,
    "eps": eps,
    "offset": 0.0,
}, (M, ), {"casting_mode": CASTING_MODE_LLAMA, "BLOCK_SIZE": BLOCK_SIZE}, {
    "Y_ptr": flat(y),
    "S_ptr": flat(s),
    "RSTD_ptr": flat(rstd),
}, "rvv-liger-fused-add-rms-norm-fwd")

run_on_board(_fused_add_rms_norm_backward_kernel, {
    "dY_ptr": flat(dy),
    "dY_row_stride": N,
    "dS_out_ptr": flat(ds_out),
    "dS_out_row_stride": N,
    "dX_ptr": [0.0] * (M * N),
    "dX_row_stride": N,
    "X_ptr": flat(s),
    "X_row_stride": N,
    "W_ptr": flat(w),
    "W_row_stride": 1,
    "RSTD_ptr": flat(rstd),
    "RSTD_row_stride": 1,
    "dW_ptr": [0.0] * (M // ROWS_PER_PROGRAM * N),
    "dW_row_stride": N,
    "n_rows": M,
    "n_cols": N,
    "offset": 0.0,
}, (M // ROWS_PER_PROGRAM, ), {
    "X_dtype": tl.float32, "rows_per_program": ROWS_PER_PROGRAM, "casting_mode": CASTING_MODE_LLAMA,
    "BLOCK_SIZE": BLOCK_SIZE, "has_dS_out": True, "COMPUTE_DW": True
}, {
    "dX_ptr": flat(dx),
    "dW_ptr": flat(dw),
}, "rvv-liger-fused-add-rms-norm-bwd")
