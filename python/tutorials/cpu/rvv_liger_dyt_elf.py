"""
Liger DyT
=========

The forward and backward kernels of Liger-Kernel's Dynamic Tanh (liger_kernel/ops/dyt.py), run on the
board with a torch reference:

dyt_fwd -- _dyt_fwd_kernel (the autotuned kernel's own function, with a fixed BLOCK_N), one program
           per (column block, row): y = tanh(alpha * x) * gamma + beta.
dyt_bwd -- _dyt_bwd_kernel (the autotuned kernel's own function, with a fixed BLOCK_N), ROW_PROGRAMS
           programs per column block, each looping over the rows start, start + ROW_PROGRAMS, ...:
           dx = alpha * (1 - tanh^2) * dy * gamma, and the program's partial dgamma, dbeta and dalpha.

An irregular number of columns so the masked tail is tested.
"""

import os
import sys

sys.path.insert(0, os.path.expanduser("~/Liger-Kernel/src"))

import torch

import triton
import triton.language as tl

from liger_kernel.ops.dyt import _dyt_bwd_kernel
from liger_kernel.ops.dyt import _dyt_fwd_kernel

torch.manual_seed(0)
triton.runtime.driver.set_active_to_cpu()

M, N = 8, 500
BLOCK_N = 512
ROW_PROGRAMS = 4

x = torch.randn(M, N, dtype=torch.float32)
alpha = torch.tensor([0.7], dtype=torch.float32)
gamma = torch.rand(N, dtype=torch.float32)
beta = torch.rand(N, dtype=torch.float32)
dy = torch.randn(M, N, dtype=torch.float32)

t = torch.tanh(alpha * x)
y = t * gamma + beta
tmp = (1 - t * t) * dy * gamma
dx = alpha * tmp
# Program r handles rows r, r + ROW_PROGRAMS, ...
dg = torch.stack([(dy * t)[r::ROW_PROGRAMS].sum(dim=0) for r in range(ROW_PROGRAMS)])
db = torch.stack([dy[r::ROW_PROGRAMS].sum(dim=0) for r in range(ROW_PROGRAMS)])
da = torch.stack([(x * tmp)[r::ROW_PROGRAMS].sum() for r in range(ROW_PROGRAMS)])

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

run_on_board(_dyt_fwd_kernel.fn, {
    "X": flat(x),
    "Y": [0.0] * (M * N),
    "Alpha": flat(alpha),
    "Gamma": flat(gamma),
    "Beta": flat(beta),
}, (triton.cdiv(N, BLOCK_N), M), {"HAVE_BETA": True, "N": N, "BLOCK_N": BLOCK_N}, {"Y": flat(y)},
             "rvv-liger-dyt-fwd")

run_on_board(_dyt_bwd_kernel.fn, {
    "DY": flat(dy),
    "DX": [0.0] * (M * N),
    "DA": [0.0] * (ROW_PROGRAMS * triton.cdiv(N, 512)),
    "DG": [0.0] * (ROW_PROGRAMS * N),
    "DB": [0.0] * (ROW_PROGRAMS * N),
    "X": flat(x),
    "Alpha": flat(alpha),
    "Gamma": flat(gamma),
    "M": M,
}, (triton.cdiv(N, BLOCK_N), ROW_PROGRAMS), {"HAVE_BETA": True, "N": N, "BLOCK_N": BLOCK_N}, {
    "DX": flat(dx),
    "DG": flat(dg),
    "DB": flat(db),
    "DA": flat(da),
}, "rvv-liger-dyt-bwd", atol=1e-3)
