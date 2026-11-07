"""
Histogram
=========

A tail-masked tl.histogram. Each program loads BLOCK_SIZE int32 values in [0, NUM_BINS) and stores
its own histogram to row pid of a [num_programs, NUM_BINS] output, so no atomics are needed.

hist_kernel -- tail mask: offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), m = offs < n, with n
               not a multiple of BLOCK_SIZE. x = tl.load(..., mask=m, other=0) and
               tl.histogram(x, NUM_BINS, mask=m). The last program has only n % BLOCK_SIZE valid
               lanes; its off lanes hold 0, so a lowering that drops the mask over-counts bin 0.
               This is the case AnalyzeTailMasks can turn into triton_cpu.tail_mask, so the
               histogram loop could stop at the tail length.
"""

import numpy as np
import torch

import triton
import triton.language as tl

BLOCK_SIZE = 64
NUM_BINS = 16


@triton.jit
def hist_kernel(x_ptr,  # *Pointer* to input values.
                z_ptr,  # *Pointer* to [num_programs, NUM_BINS] output.
                n_elements,  # Number of input values (not a multiple of BLOCK_SIZE).
                BLOCK_SIZE: tl.constexpr, NUM_BINS: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = offs < n_elements
    x = tl.load(x_ptr + offs, mask=m, other=0)
    h = tl.histogram(x, NUM_BINS, mask=m)
    tl.store(z_ptr + pid * NUM_BINS + tl.arange(0, NUM_BINS), h)


def ref_hist(x, n_programs):
    """Per-program histogram of x (numpy int32)."""
    out = np.zeros((n_programs, NUM_BINS), dtype=np.int32)
    for p in range(n_programs):
        out[p] = np.bincount(x[p * BLOCK_SIZE:(p + 1) * BLOCK_SIZE], minlength=NUM_BINS)
    return out


# %%
# Test against a numpy reference.
torch.manual_seed(0)
triton.runtime.driver.set_active_to_cpu()

from triton.backends.cpu.riscv import compile_deploy_and_run

RISCV_HOST = "chlee@140.114.78.64"
RISCV_REMOTE_DIR = "~/triton-riscv-elf"
CONSTEXPRS = {"BLOCK_SIZE": BLOCK_SIZE, "NUM_BINS": NUM_BINS}


def run_on_board(kernel, arguments, grid, expected, name):
    result = compile_deploy_and_run(kernel, arguments, grid, f"artifacts/riscv/{name}.elf", RISCV_HOST,
                                     constexprs=CONSTEXPRS, expected=expected, remote_dir=RISCV_REMOTE_DIR)
    print(f"{name}:")
    print(result.stdout, end="")
    print(result.stderr, end="")


# Tail mask: the last program has n_tail % BLOCK_SIZE = 40 valid lanes.
n_tail = 1000
n_prog_tail = triton.cdiv(n_tail, BLOCK_SIZE)
x_tail = torch.randint(0, NUM_BINS, (n_tail, ), dtype=torch.int32)
run_on_board(hist_kernel, {
    "x_ptr": x_tail,
    "z_ptr": torch.zeros(n_prog_tail * NUM_BINS, dtype=torch.int32),
    "n_elements": n_tail,
}, (n_prog_tail, ), {"z_ptr": ref_hist(x_tail.numpy(), n_prog_tail).ravel().tolist()}, "rvv-histogram-tail")
