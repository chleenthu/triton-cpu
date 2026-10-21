"""
Liger GroupNorm
===============

The forward and backward kernels of Liger-Kernel's GroupNorm (liger_kernel/ops/group_norm.py), run on the
board with a torch reference, one program per (batch, group):

group_norm_fwd -- _group_norm_forward_kernel: mean and rstd over the group's channels * hidden elements,
                  y = (x - mean) * rstd * w[channel] + b[channel], and mean and rstd are stored.
group_norm_bwd -- _group_norm_backward_kernel: dx, and dW / dB per channel accumulated with atomic adds
                  across the batch. Because it accumulates into its outputs, it is checked after one
                  launch and timed in a second run. The float atomic add becomes a call to libatomic
                  (__atomic_load_4 in a compare-and-swap loop), which the kernel .so is not linked
                  against, so the runner is linked with -latomic to load it.

A hidden size that is not a power of 2, so the masked tail is tested.
"""

import os
import sys

sys.path.insert(0, os.path.expanduser("~/Liger-Kernel/src"))

import torch

import triton
import triton.language as tl

from liger_kernel.ops.group_norm import _group_norm_backward_kernel
from liger_kernel.ops.group_norm import _group_norm_forward_kernel

torch.manual_seed(0)
triton.runtime.driver.set_active_to_cpu()

B, C, G, H = 4, 8, 2, 100
CPG = C // G
eps = 1e-6

x = torch.randn(B, C, H, dtype=torch.float32, requires_grad=True)
w = torch.rand(C, dtype=torch.float32, requires_grad=True)
bias = torch.rand(C, dtype=torch.float32, requires_grad=True)
dy = torch.randn(B, C, H, dtype=torch.float32)

y = torch.nn.functional.group_norm(x, G, w, bias, eps)
y.backward(dy)
xg = x.detach().reshape(B, G, -1)
mean = xg.mean(dim=2)
rstd = 1 / torch.sqrt(xg.var(dim=2, unbiased=False) + eps)

from triton.backends.cpu.riscv import Toolchain
from triton.backends.cpu.riscv import compile_deploy_and_run

RISCV_HOST = "chlee@140.114.78.64"
RISCV_REMOTE_DIR = "~/triton-riscv-elf"


def flat(t):
    return t.detach().flatten().tolist()


class AtomicToolchain(Toolchain):
    # Links the runner with libatomic even though the runner itself does not call it, so the
    # kernel .so's libatomic calls resolve when the board loads it.
    def compile_command(self, sources, output, extra_args=()):
        return super().compile_command(sources, output, [*extra_args, "-Wl,--no-as-needed", "-latomic"])


def run_on_board(kernel, arguments, grid, constexprs, expected, name, atol=1e-4, repeatable=True, toolchain=None):
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
                                             remote_dir=RISCV_REMOTE_DIR, toolchain=toolchain)
        finally:
            for var, value in saved.items():
                if value is None:
                    os.environ.pop(var, None)
                else:
                    os.environ[var] = value
        print(f"{run_name}:")
        print(result.stdout, end="")
        print(result.stderr, end="", flush=True)

run_on_board(_group_norm_forward_kernel, {
    "Y_ptr": [0.0] * (B * C * H),
    "Y_row_stride": C * H,
    "Y_col_stride": CPG * H,
    "X_ptr": flat(x),
    "X_row_stride": C * H,
    "X_col_stride": CPG * H,
    "Mean_ptr": [0.0] * (B * G),
    "Mean_row_stride": G,
    "Mean_col_stride": 1,
    "RSTD_ptr": [0.0] * (B * G),
    "RSTD_row_stride": G,
    "RSTD_col_stride": 1,
    "W_ptr": flat(w),
    "B_ptr": flat(bias),
    "hidden_size": CPG * H,
    "channels_per_group": CPG,
    "eps": eps,
}, (B, G), {"BLOCK_SIZE": triton.next_power_of_2(CPG * H)}, {
    "Y_ptr": flat(y),
    "Mean_ptr": flat(mean),
    "RSTD_ptr": flat(rstd),
}, "rvv-liger-group-norm-fwd")

run_on_board(_group_norm_backward_kernel, {
    "X_ptr": flat(x),
    "X_row_stride": C * H,
    "X_col_stride": H,
    "W_ptr": flat(w),
    "Mean_ptr": flat(mean),
    "Mean_ptr_row_stride": G,
    "Mean_ptr_col_stride": 1,
    "RSTD_ptr": flat(rstd),
    "DX_ptr": [0.0] * (B * C * H),
    "DW_ptr": [0.0] * C,
    "DB_ptr": [0.0] * C,
    "UPSTREAM_ptr": flat(dy),
}, (B, G), {"hidden_size": H, "channels_per_group": CPG, "BLOCK_SIZE": triton.next_power_of_2(H), "dtype": tl.float32}, {
    "DX_ptr": flat(x.grad),
    "DW_ptr": flat(w.grad),
    "DB_ptr": flat(bias.grad),
}, "rvv-liger-group-norm-bwd", atol=1e-3, repeatable=False, toolchain=AtomicToolchain.from_env())
