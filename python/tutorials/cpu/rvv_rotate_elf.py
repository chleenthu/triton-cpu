"""
Rotate
======

One kernel that keeps x live across a lane rotation, used to see whether -custom-reverse rebuilds x
from the rotated copy (ReverseRemat[inverse-slide-rotation]):

rotate_live -- each program loads BLOCK_SIZE int32 values x, and rotates them by k = BLOCK_SIZE / 2
               lanes, y = tl.reshape(tl.flip(tl.reshape(x, (2, BLOCK_SIZE // 2)), 0), (BLOCK_SIZE, )),
               i.e. the two halves of x swap places. Triton has no rotate-by-k op (tl.gather fails to
               lower on the CPU backend), and swapping the halves is the rotation it can express. It is
               lowered as a slide pair: vslidedown by k and vslideup by n - k. Then y is combined with
               loaded weights, r = (y * w0) * (y * w1) + (y + w0) * (y + w1), and only then x is used
               again: out = r - x. The scheduler places the vslideup (x's second read) late, so the
               body has to be this long for x to sit unused for the 6 instructions -custom-reverse
               needs; with only (y * w0) * (y * w1) it does not fire. -custom-reverse kills x after the
               rotation and rebuilds it as the opposite rotation of y (down n - k, up k).

All sizes are exact multiples of the tile sizes, so no masks are involved.
"""

import numpy as np
import torch

import triton
import triton.language as tl

BLOCK_LIVE = 64


@triton.jit
def rotate_live_kernel(a_ptr,  # *Pointer* to input vector.
                       w_ptr,  # *Pointer* to 2 weight vectors, stored back to back.
                       out_ptr,  # *Pointer* to output vector.
                       n_elements,  # Size of each vector (multiple of BLOCK_SIZE).
                       BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
                       ):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(a_ptr + offs)
    t = tl.reshape(x, (2, BLOCK_SIZE // 2))
    t = tl.flip(t, 0)
    y = tl.reshape(t, (BLOCK_SIZE, ))
    w0 = tl.load(w_ptr + offs)
    w1 = tl.load(w_ptr + n_elements + offs)
    p0 = y * w0
    p1 = y * w1
    q0 = y + w0
    q1 = y + w1
    r = p0 * p1 + q0 * q1
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
y_live = np.roll(a_live.numpy().reshape(-1, BLOCK_LIVE), BLOCK_LIVE // 2, axis=1).reshape(-1)
p_live = [y_live * w_live.numpy()[k * size:(k + 1) * size] for k in range(2)]
q_live = [y_live + w_live.numpy()[k * size:(k + 1) * size] for k in range(2)]
out_live = (p_live[0] * p_live[1] + q_live[0] * q_live[1]) - a_live.numpy()
run_on_board(rotate_live_kernel, {
    "a_ptr": a_live,
    "w_ptr": w_live,
    "out_ptr": torch.zeros(size, dtype=torch.int32),
    "n_elements": size,
}, (size // BLOCK_LIVE, ), {"BLOCK_SIZE": BLOCK_LIVE}, {"out_ptr": out_live.tolist()}, "rvv-rotate-live")
