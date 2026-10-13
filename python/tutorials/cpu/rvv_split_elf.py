"""
Split
=====

One kernel that keeps y live across a tl.split, used to see whether -custom-reverse rebuilds y from
its two halves (ReverseRemat[zip-of-deinterleave]):

split_live -- each program loads 2 * BLOCK_SIZE int32 values y, and splits them into the even and odd
              lanes with lo, hi = tl.split(tl.reshape(y, (BLOCK_SIZE, 2))), i.e. lo[i] = y[2 * i] and
              hi[i] = y[2 * i + 1]. The split is lowered as an unzip by narrowing shifts: y is read as
              64-bit lanes, lo = vnsrl.wi y, 0 (low halves) and hi = vnsrl.wx y, 32 (high halves).
              Then lo and hi are multiplied by loaded weights, r = (lo * w0) * (hi * w1) is stored, and
              only then y is used again: out2 = y * 3. -custom-reverse kills y after the unzip and
              rebuilds it right before the multiply as zip(lo, hi), vwaddu.vv lo, hi followed by
              vwmaccu.vx -1, hi. The rebuild needs both halves, so it goes where lo and hi are still
              live.

All sizes are exact multiples of the tile sizes, so no masks are involved.
"""

import numpy as np
import torch

import triton
import triton.language as tl

BLOCK_LIVE = 32


@triton.jit
def split_live_kernel(y_ptr,  # *Pointer* to input vector of 2 * n_elements.
                      w_ptr,  # *Pointer* to 2 weight vectors of n_elements, stored back to back.
                      out_ptr,  # *Pointer* to output vector of n_elements.
                      out2_ptr,  # *Pointer* to output vector of 2 * n_elements.
                      n_elements,  # Size of lo and hi (multiple of BLOCK_SIZE).
                      BLOCK_SIZE: tl.constexpr,  # Number of elements of lo and hi each program processes.
                      ):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs2 = pid * 2 * BLOCK_SIZE + tl.arange(0, 2 * BLOCK_SIZE)
    y = tl.load(y_ptr + offs2)
    t = tl.reshape(y, (BLOCK_SIZE, 2))
    lo, hi = tl.split(t)
    w0 = tl.load(w_ptr + offs)
    w1 = tl.load(w_ptr + n_elements + offs)
    p0 = lo * w0
    p1 = hi * w1
    r = p0 * p1
    tl.store(out_ptr + offs, r)
    out2 = y * 3
    tl.store(out2_ptr + offs2, out2)


# %%
# Test against a numpy reference.
torch.manual_seed(0)
size = 1024
triton.runtime.driver.set_active_to_cpu()
y_live = torch.randint(-100, 100, (2 * size, ), dtype=torch.int32)
w_live = torch.randint(-100, 100, (2 * size, ), dtype=torch.int32)

from triton.backends.cpu.riscv import compile_deploy_and_run

RISCV_HOST = "chlee@140.114.78.64"
RISCV_REMOTE_DIR = "~/triton-riscv-elf"


def run_on_board(kernel, arguments, grid, constexprs, expected, name):
    result = compile_deploy_and_run(kernel, arguments, grid, f"artifacts/riscv/{name}.elf", RISCV_HOST,
                                     constexprs=constexprs, expected=expected, remote_dir=RISCV_REMOTE_DIR)
    print(f"{name}:")
    print(result.stdout, end="")
    print(result.stderr, end="")


# int32 arithmetic wraps around the same way in numpy and on the board, so the check is exact.
lo_live, hi_live = y_live.numpy()[0::2], y_live.numpy()[1::2]
p_live = [lo_live * w_live.numpy()[:size], hi_live * w_live.numpy()[size:]]
run_on_board(split_live_kernel, {
    "y_ptr": y_live,
    "w_ptr": w_live,
    "out_ptr": torch.zeros(size, dtype=torch.int32),
    "out2_ptr": torch.zeros(2 * size, dtype=torch.int32),
    "n_elements": size,
}, (size // BLOCK_LIVE, ), {"BLOCK_SIZE": BLOCK_LIVE}, {
    "out_ptr": (p_live[0] * p_live[1]).tolist(),
    "out2_ptr": (y_live.numpy() * 3).tolist(),
}, "rvv-split-live")
