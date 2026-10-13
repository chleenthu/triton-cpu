"""
Cumsum
======

One kernel that keeps x live across a tl.cumsum, used to see what a prefix sum lowers to and whether
-custom-reverse rebuilds x from the prefix sum:

cumsum_live -- each program loads BLOCK_SIZE int32 values x, and takes their prefix sum,
               y = tl.cumsum(x, 0), so y[i] = x[0] + ... + x[i]. Then two products of y with loaded
               weights are live at once, and only then x is used again: out = ((y * w0) * (y * w1)) - x.
               A prefix sum is exactly invertible, x[i] = y[i] - y[i - 1] (vslide1up y, 0 and vsub),
               so x could be rebuilt from y; no plan matches this yet.

All sizes are exact multiples of the tile sizes, so no masks are involved.
"""

import numpy as np
import torch

import triton
import triton.language as tl

BLOCK_LIVE = 64


@triton.jit
def cumsum_live_kernel(a_ptr,  # *Pointer* to input vector.
                       w_ptr,  # *Pointer* to 2 weight vectors, stored back to back.
                       out_ptr,  # *Pointer* to output vector.
                       n_elements,  # Size of each vector (multiple of BLOCK_SIZE).
                       BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
                       ):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(a_ptr + offs)
    y = tl.cumsum(x, 0)
    w0 = tl.load(w_ptr + offs)
    w1 = tl.load(w_ptr + n_elements + offs)
    p0 = y * w0
    p1 = y * w1
    r = p0 * p1
    out = r - x
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
y_live = np.cumsum(a_live.numpy().reshape(-1, BLOCK_LIVE), axis=1, dtype=np.int32).reshape(-1)
p_live = [y_live * w_live.numpy()[k * size:(k + 1) * size] for k in range(2)]
out_live = (p_live[0] * p_live[1]) - a_live.numpy()
run_on_board(cumsum_live_kernel, {
    "a_ptr": a_live,
    "w_ptr": w_live,
    "out_ptr": torch.zeros(size, dtype=torch.int32),
    "n_elements": size,
}, (size // BLOCK_LIVE, ), {"BLOCK_SIZE": BLOCK_LIVE}, {"out_ptr": out_live.tolist()}, "rvv-cumsum-live")
