"""
Liger cross entropy
===================

Liger-Kernel's cross-entropy kernel (liger_kernel/ops/cross_entropy.py), which computes the loss and writes
the gradient over the logits in the same pass, run on the board with a torch reference, one program per row:

cross_entropy -- liger_cross_entropy_kernel with the mean reduction, no class weights, no label smoothing,
                 no z-loss and no softcapping: loss[row] = (logsumexp(x) - x[y]) / n_non_ignore and
                 x = (softmax(x) - onehot(y)) / n_non_ignore. One row's label is ignore_index, so its
                 gradient row is zeroed and its loss left at 0. Because it overwrites its input, it is
                 checked after one launch and timed in a second run.

A vocabulary that is not a power of 2, so the masked tail is tested.
"""

import os
import sys

sys.path.insert(0, os.path.expanduser("~/Liger-Kernel/src"))

import torch

import triton
import triton.language as tl

from liger_kernel.ops.cross_entropy import liger_cross_entropy_kernel

torch.manual_seed(0)
triton.runtime.driver.set_active_to_cpu()

M, V = 8, 1000
BLOCK_SIZE = triton.next_power_of_2(V)
IGNORE_INDEX = -100

x = torch.randn(M, V, dtype=torch.float32)
target = torch.randint(0, V, (M, ), dtype=torch.int64)
target[3] = IGNORE_INDEX
valid = target != IGNORE_INDEX
n_non_ignore = int(valid.sum())

loss = torch.zeros(M)
grad = torch.zeros(M, V)
lse = torch.logsumexp(x, dim=1)
for r in range(M):
    if valid[r]:
        loss[r] = (lse[r] - x[r, target[r]]) / n_non_ignore
        grad[r] = torch.softmax(x[r], dim=0) / n_non_ignore
        grad[r, target[r]] -= 1.0 / n_non_ignore

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

run_on_board(liger_cross_entropy_kernel, {
    "X_ptr": flat(x),
    "X_stride": V,
    "Y_ptr": flat(target),
    "Y_stride": 1,
    "weight_ptr": [0.0],
    "loss_ptr": [0.0] * M,
    "z_loss_ptr": [0.0],
    "loss_stride": 1,
    "token_accuracy_ptr": [0.0],
    "token_accuracy_stride": 0,
    "predicted_tokens_ptr": [0],
    "predicted_tokens_stride": 0,
    "n_cols": V,
    "n_non_ignore": float(n_non_ignore),
    "sum_non_ignore_weight": float(n_non_ignore),
    "weight_sum": 0.0,
    "ignore_index": IGNORE_INDEX,
    "softcap": 0.0,
}, (M, ), {
    "lse_square_scale": 0.0, "label_smoothing": 0.0, "reduction": "mean", "RETURN_Z_LOSS": False,
    "RETURN_TOKEN_ACCURACY": False, "RETURN_PREDICTED_TOKENS": False, "BLOCK_SIZE": BLOCK_SIZE,
    "HAS_WEIGHT": False, "HAS_SOFTCAPPING": False, "HAS_GRADIENTS": True
}, {
    "X_ptr": flat(grad),
    "loss_ptr": flat(loss),
}, "rvv-liger-cross-entropy", repeatable=False)
