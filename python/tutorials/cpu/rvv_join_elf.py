"""
Join
====

One kernel that keeps a live across a tl.join, used to see whether -custom-reverse rebuilds a from the
joined copy (ReverseRemat[deinterleave-of-zip]):

join_live -- each program loads BLOCK_SIZE int32 values a and b, and interleaves them with
             y = tl.reshape(tl.join(a, b), (2 * BLOCK_SIZE, )), i.e. y[2 * i] = a[i] and
             y[2 * i + 1] = b[i]. The interleave is lowered as a zip through widening arithmetic:
             vwaddu.vv a, b followed by vwmaccu.vx -1, b, which puts a in the low and b in the high
             half of each 64-bit lane. Then two products of y with loaded weights are live at once,
             r = (y * w0) * (y * w1) is stored, and only then a is used again: out2 = a - b.
             -custom-reverse kills a after the zip and rebuilds it right before the vsub as the low
             half of y, vnsrl.wi y, 0.
interleave_live -- the same as join_live, but y = tl.interleave(a, b), to check that tl.interleave
                   lowers the same way (tt.join + tt.reshape, then vector.interleave).

All sizes are exact multiples of the tile sizes, so no masks are involved.
"""

import numpy as np
import torch

import triton
import triton.language as tl

BLOCK_LIVE = 32


@triton.jit
def join_live_kernel(a_ptr,  # *Pointer* to first input vector.
                     b_ptr,  # *Pointer* to second input vector.
                     w_ptr,  # *Pointer* to 2 weight vectors of 2 * n_elements, stored back to back.
                     out2_ptr,  # *Pointer* to output vector of 2 * n_elements.
                     out_ptr,  # *Pointer* to output vector of n_elements.
                     n_elements,  # Size of a and b (multiple of BLOCK_SIZE).
                     BLOCK_SIZE: tl.constexpr,  # Number of elements of a and b each program processes.
                     ):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs2 = pid * 2 * BLOCK_SIZE + tl.arange(0, 2 * BLOCK_SIZE)
    a = tl.load(a_ptr + offs)
    b = tl.load(b_ptr + offs)
    t = tl.join(a, b)
    y = tl.reshape(t, (2 * BLOCK_SIZE, ))
    w0 = tl.load(w_ptr + offs2)
    w1 = tl.load(w_ptr + 2 * n_elements + offs2)
    p0 = y * w0
    p1 = y * w1
    r = p0 * p1
    tl.store(out2_ptr + offs2, r)
    out = a - b
    tl.store(out_ptr + offs, out)


@triton.jit
def interleave_live_kernel(a_ptr,  # *Pointer* to first input vector.
                           b_ptr,  # *Pointer* to second input vector.
                           w_ptr,  # *Pointer* to 2 weight vectors of 2 * n_elements, stored back to back.
                           out2_ptr,  # *Pointer* to output vector of 2 * n_elements.
                           out_ptr,  # *Pointer* to output vector of n_elements.
                           n_elements,  # Size of a and b (multiple of BLOCK_SIZE).
                           BLOCK_SIZE: tl.constexpr,  # Number of elements of a and b each program processes.
                           ):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs2 = pid * 2 * BLOCK_SIZE + tl.arange(0, 2 * BLOCK_SIZE)
    a = tl.load(a_ptr + offs)
    b = tl.load(b_ptr + offs)
    y = tl.interleave(a, b)
    w0 = tl.load(w_ptr + offs2)
    w1 = tl.load(w_ptr + 2 * n_elements + offs2)
    p0 = y * w0
    p1 = y * w1
    r = p0 * p1
    tl.store(out2_ptr + offs2, r)
    out = a - b
    tl.store(out_ptr + offs, out)


# %%
# Test against a numpy reference.
torch.manual_seed(0)
size = 1024
triton.runtime.driver.set_active_to_cpu()
a_live = torch.randint(-100, 100, (size, ), dtype=torch.int32)
b_live = torch.randint(-100, 100, (size, ), dtype=torch.int32)
w_live = torch.randint(-100, 100, (4 * size, ), dtype=torch.int32)

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
y_live = np.stack([a_live.numpy(), b_live.numpy()], 1).reshape(-1)
p_live = [y_live * w_live.numpy()[k * 2 * size:(k + 1) * 2 * size] for k in range(2)]
run_on_board(join_live_kernel, {
    "a_ptr": a_live,
    "b_ptr": b_live,
    "w_ptr": w_live,
    "out2_ptr": torch.zeros(2 * size, dtype=torch.int32),
    "out_ptr": torch.zeros(size, dtype=torch.int32),
    "n_elements": size,
}, (size // BLOCK_LIVE, ), {"BLOCK_SIZE": BLOCK_LIVE}, {
    "out2_ptr": (p_live[0] * p_live[1]).tolist(),
    "out_ptr": (a_live.numpy() - b_live.numpy()).tolist(),
}, "rvv-join-live")

run_on_board(interleave_live_kernel, {
    "a_ptr": a_live,
    "b_ptr": b_live,
    "w_ptr": w_live,
    "out2_ptr": torch.zeros(2 * size, dtype=torch.int32),
    "out_ptr": torch.zeros(size, dtype=torch.int32),
    "n_elements": size,
}, (size // BLOCK_LIVE, ), {"BLOCK_SIZE": BLOCK_LIVE}, {
    "out2_ptr": (p_live[0] * p_live[1]).tolist(),
    "out_ptr": (a_live.numpy() - b_live.numpy()).tolist(),
}, "rvv-interleave-live")
