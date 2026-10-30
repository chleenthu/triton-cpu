"""
Widen by k
==========

One kernel that keeps an int16 vector v live across a widening add of a constant, used to see whether
-custom-reverse rebuilds v from the widened copy (ReverseRemat[narrow-of-widen], widen by k):

widenk_live -- each program loads BLOCK_SIZE int16 values v and widens them with y = v.to(tl.int32) + K,
               which LLVM selects as vwadd.vx v, k. Then y is combined with loaded int32 weights,
               r = (y * w0) * (y * w1) + (y + w0) * (y + w1) is stored, and only then v is used again:
               out2 = v * 3 (int16). -custom-reverse kills v after the widen and rebuilds it right before
               the multiply as narrow(y) - k: vnsrl.wi y, 0 then vsub.vx k at v's SEW. The low 16 bits of
               y are v + k whatever k's upper bits, so the rebuild is exact.

All sizes are exact multiples of the tile sizes, so no masks are involved.
"""

import numpy as np
import torch

import triton
import triton.language as tl

BLOCK_LIVE = 64
K = 5


@triton.jit
def widenk_live_kernel(v_ptr,  # *Pointer* to int16 input vector.
                       w_ptr,  # *Pointer* to 2 int32 weight vectors, stored back to back.
                       out_ptr,  # *Pointer* to int32 output vector.
                       out2_ptr,  # *Pointer* to int16 output vector.
                       n_elements,  # Size of each vector (multiple of BLOCK_SIZE).
                       K: tl.constexpr,  # The constant added by the widen.
                       BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
                       ):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    v = tl.load(v_ptr + offs)
    y = v.to(tl.int32) + K
    w0 = tl.load(w_ptr + offs)
    w1 = tl.load(w_ptr + n_elements + offs)
    p0 = y * w0
    p1 = y * w1
    q0 = y + w0
    q1 = y + w1
    r = p0 * p1 + q0 * q1
    tl.store(out_ptr + offs, r)
    out2 = v * 3
    tl.store(out2_ptr + offs, out2)


# %%
# Test against a numpy reference.
torch.manual_seed(0)
size = 1024
triton.runtime.driver.set_active_to_cpu()
v_live = torch.randint(-100, 100, (size, ), dtype=torch.int16)
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


# int32 / int16 arithmetic wraps around the same way in numpy and on the board, so the check is exact.
y_live = v_live.numpy().astype(np.int32) + K
w0_live, w1_live = w_live.numpy()[:size], w_live.numpy()[size:]
r_live = (y_live * w0_live) * (y_live * w1_live) + (y_live + w0_live) * (y_live + w1_live)
run_on_board(widenk_live_kernel, {
    "v_ptr": v_live,
    "w_ptr": w_live,
    "out_ptr": torch.zeros(size, dtype=torch.int32),
    "out2_ptr": torch.zeros(size, dtype=torch.int16),
    "n_elements": size,
}, (size // BLOCK_LIVE, ), {"K": K, "BLOCK_SIZE": BLOCK_LIVE}, {
    "out_ptr": r_live.tolist(),
    "out2_ptr": (v_live.numpy() * 3).astype(np.int16).tolist(),
}, "rvv-widenk-live")
