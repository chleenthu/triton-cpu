"""
Liger RoPE
==========

Liger-Kernel's rotary position embedding kernel (liger_kernel/ops/rope.py), forward and backward, run on the
board with a torch reference, one program per token (batch * sequence position), over all q and k heads:

rope_fwd -- _triton_rope: q1, q2 = the two halves of each head, q1' = q1 * cos - q2 * sin and
            q2' = q2 * cos + q1 * sin, the same for k, written over q and k.
rope_bwd -- _triton_rope with BACKWARD_PASS: the inverse rotation (sin negated), written over dq and dk.

Both overwrite their inputs, so each is checked after one launch and timed in a second run.
"""

import os
import sys

sys.path.insert(0, os.path.expanduser("~/Liger-Kernel/src"))

import torch

import triton
import triton.language as tl

from liger_kernel.ops.rope import _triton_rope

torch.manual_seed(0)
triton.runtime.driver.set_active_to_cpu()

BS, SL, N_QH, N_KH, HD = 1, 8, 4, 2, 64
RD = HD
PAD_N_QH, PAD_N_KH, PAD_HD, PAD_RD = (triton.next_power_of_2(v) for v in (N_QH, N_KH, HD, RD))
BLOCK_SIZE = max(PAD_N_QH, PAD_N_KH)

# (batch, seq, heads, head_dim), the layout the kernel sees after Liger's transpose.
q = torch.randn(BS, SL, N_QH, HD, dtype=torch.float32)
k = torch.randn(BS, SL, N_KH, HD, dtype=torch.float32)
theta = torch.arange(SL, dtype=torch.float32)[:, None] / 10000**(torch.arange(0, HD, 2) / HD)
cos = torch.cat([theta.cos(), theta.cos()], dim=-1)[None]  # (1, seq, head_dim)
sin = torch.cat([theta.sin(), theta.sin()], dim=-1)[None]


def rotate(t, sign):
    c = cos[0, :, None, :HD // 2]
    s = sign * sin[0, :, None, :HD // 2]
    t1, t2 = t[..., :HD // 2], t[..., HD // 2:]
    return torch.cat([t1 * c - t2 * s, t2 * c + t1 * s], dim=-1)


q_fwd, k_fwd = rotate(q, 1.0), rotate(k, 1.0)
q_bwd, k_bwd = rotate(q, -1.0), rotate(k, -1.0)

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

for backward, q_ref, k_ref, name in [(False, q_fwd, k_fwd, "rvv-liger-rope-fwd"),
                                     (True, q_bwd, k_bwd, "rvv-liger-rope-bwd")]:
    run_on_board(_triton_rope, {
        "q_ptr": flat(q),
        "q_row_stride": N_QH * HD,
        "k_ptr": flat(k),
        "k_row_stride": N_KH * HD,
        "cos": flat(cos),
        "cos_row_stride": HD,
        "sin": flat(sin),
        "sin_row_stride": HD,
        "sl": SL,
    }, (BS * SL, ), {
        "bs": BS, "cos_bs": 1, "n_qh": N_QH, "n_kh": N_KH, "hd": HD, "pad_n_qh": PAD_N_QH, "pad_n_kh": PAD_N_KH,
        "pad_hd": PAD_HD, "pad_rd": PAD_RD, "rd": RD, "BLOCK_SIZE": BLOCK_SIZE, "BACKWARD_PASS": backward
    }, {
        "q_ptr": flat(q_ref),
        "k_ptr": flat(k_ref),
    }, name, repeatable=False)
