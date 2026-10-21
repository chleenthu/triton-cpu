"""
Liger RMSNorm
=============

The forward and backward kernels of Liger-Kernel's RMSNorm (liger_kernel/ops/rms_norm.py), run on the
board with a torch reference:

rms_norm_fwd -- _rms_norm_forward_kernel, one program per row: y = x * rstd * (offset + w),
                rstd = 1 / sqrt(mean(x^2) + eps), and rstd is stored per row.
rms_norm_bwd -- _rms_norm_backward_kernel, each program handles rows_per_program rows:
                dx = rstd * m - rstd^3 / n_cols * sum(m * x) * x with m = dy * (w + offset), and the
                program's partial dW (sum of dy * x * rstd over its rows) is stored in its own row.

Llama casting mode (fp32 compute), offset 0, an irregular number of columns so the masked tail is tested.
"""

import os
import sys

sys.path.insert(0, os.path.expanduser("~/Liger-Kernel/src"))

import torch

import triton
import triton.language as tl

from liger_kernel.ops.rms_norm import _rms_norm_backward_kernel
from liger_kernel.ops.rms_norm import _rms_norm_forward_kernel

torch.manual_seed(0)
triton.runtime.driver.set_active_to_cpu()

M, N = 16, 500
ROWS_PER_PROGRAM = 4
BLOCK_SIZE = triton.next_power_of_2(N)
eps = 1e-6
CASTING_MODE_LLAMA = 0

x = torch.randn(M, N, dtype=torch.float32)
w = torch.rand(N, dtype=torch.float32)
dy = torch.randn(M, N, dtype=torch.float32)

rstd = 1 / torch.sqrt((x * x).mean(dim=1) + eps)
y = x * rstd[:, None] * w
m = dy * w
dx = rstd[:, None] * m - (rstd**3 / N * (m * x).sum(dim=1))[:, None] * x
dw = (dy * x * rstd[:, None]).reshape(M // ROWS_PER_PROGRAM, ROWS_PER_PROGRAM, N).sum(dim=1)

from triton.backends.cpu.riscv import compile_deploy_and_run

RISCV_HOST = "chlee@140.114.78.64"
RISCV_REMOTE_DIR = "~/triton-riscv-elf"


def flat(t):
    return t.detach().flatten().tolist()


def run_on_board(kernel, arguments, grid, constexprs, expected, name, atol=1e-4):
    print(f"=== {name} {kernel.__name__}", flush=True)
    result = compile_deploy_and_run(kernel, arguments, grid, f"artifacts/riscv/{name}.elf", RISCV_HOST,
                                     constexprs=constexprs, expected=expected, atol=atol,
                                     remote_dir=RISCV_REMOTE_DIR)
    print(f"{name}:")
    print(result.stdout, end="")
    print(result.stderr, end="", flush=True)


run_on_board(_rms_norm_forward_kernel, {
    "Y_ptr": [0.0] * (M * N),
    "Y_row_stride": N,
    "X_ptr": flat(x),
    "X_row_stride": N,
    "W_ptr": flat(w),
    "W_row_stride": 1,
    "RSTD_ptr": [0.0] * M,
    "RSTD_row_stride": 1,
    "n_cols": N,
    "eps": eps,
    "offset": 0.0,
}, (M, ), {"casting_mode": CASTING_MODE_LLAMA, "elementwise_affine": True, "BLOCK_SIZE": BLOCK_SIZE}, {
    "Y_ptr": flat(y),
    "RSTD_ptr": flat(rstd),
}, "rvv-liger-rms-norm-fwd")

run_on_board(_rms_norm_backward_kernel, {
    "dY_ptr": flat(dy),
    "dY_row_stride": N,
    "dX_ptr": [0.0] * (M * N),
    "dX_row_stride": N,
    "X_ptr": flat(x),
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
    "rows_per_program": ROWS_PER_PROGRAM,
}, (M // ROWS_PER_PROGRAM, ), {
    "X_dtype": tl.float32, "casting_mode": CASTING_MODE_LLAMA, "elementwise_affine": True, "BLOCK_SIZE": BLOCK_SIZE,
    "COMPUTE_DW": True
}, {
    "dX_ptr": flat(dx),
    "dW_ptr": flat(dw),
}, "rvv-liger-rms-norm-bwd")
