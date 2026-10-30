"""
Mask to vector
==============

One kernel that keeps a mask m live across its conversion to a vector, used to see whether
-custom-reverse rebuilds m from the vector (ReverseRemat[mask-of-merge]):

mask_live -- each program loads BLOCK_SIZE int32 values x and makes the mask m = x > 0, then the vector
             y = tl.where(m, -1, 0): vmv.v.i 0 and vmerge.vim y, 0, -1, m, so y[i] = -1 where m[i] and
             0 elsewhere. Then two products of y with loaded weights are live at once,
             r = (y * w0) * (y * w1), and only then m is used again: out = tl.where(m, r, w1). x is not
             used again, so m cannot be recomputed from x. -custom-reverse kills m after the merge and
             rebuilds it from y right before the second select as vmsne.vi y, 0.

All sizes are exact multiples of the tile sizes, so no masks are involved in the loads and stores.
"""

import numpy as np
import torch

import triton
import triton.language as tl

BLOCK_LIVE = 64


@triton.jit
def mask_live_kernel(a_ptr,  # *Pointer* to input vector.
                     w_ptr,  # *Pointer* to 2 weight vectors, stored back to back.
                     out_ptr,  # *Pointer* to output vector.
                     n_elements,  # Size of each vector (multiple of BLOCK_SIZE).
                     BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
                     ):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(a_ptr + offs)
    m = x > 0
    y = tl.where(m, -1, 0)
    w0 = tl.load(w_ptr + offs)
    w1 = tl.load(w_ptr + n_elements + offs)
    p0 = y * w0
    p1 = y * w1
    r = p0 * p1
    out = tl.where(m, r, w1)
    tl.store(out_ptr + offs, out)


# %%
# Test against a numpy reference.
torch.manual_seed(0)
size = 1024
triton.runtime.driver.set_active_to_cpu()
a_live = torch.randint(-100, 100, (size, ), dtype=torch.int32)
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
m_live = a_live.numpy() > 0
y_live = np.where(m_live, -1, 0).astype(np.int32)
w0_live, w1_live = w_live.numpy()[:size], w_live.numpy()[size:]
r_live = (y_live * w0_live) * (y_live * w1_live)
out_live = np.where(m_live, r_live, w1_live)
run_on_board(mask_live_kernel, {
    "a_ptr": a_live,
    "w_ptr": w_live,
    "out_ptr": torch.zeros(size, dtype=torch.int32),
    "n_elements": size,
}, (size // BLOCK_LIVE, ), {"BLOCK_SIZE": BLOCK_LIVE}, {"out_ptr": out_live.tolist()}, "rvv-mask-to-vector-live")
