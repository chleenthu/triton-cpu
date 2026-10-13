"""
Permute
=======

Two kernels with a tl.permute: one to see what a permute lowers to on its own, and one that keeps x
live across a constant in-register permutation, used to see whether -custom-reverse rebuilds x from
the permuted copy (ReverseRemat[inverse-permutation-gather]):

permute      -- each program loads a PERM_M x PERM_N int32 tile x (row-major), transposes it with
                y = tl.permute(x, (1, 0)) and stores the PERM_N x PERM_M tile y row-major, so
                out[i * PERM_M + j] = a[j * PERM_N + i]. Nothing else happens, so the dumped IR shows
                only what tl.permute itself becomes.
permute_live -- each program loads BLOCK_SIZE = PERM_M * PERM_N int32 values x, and permutes them
                with y = tl.reshape(tl.permute(tl.reshape(x, (PERM_M, PERM_N)), (1, 0)), (BLOCK_SIZE, )),
                i.e. y[i * PERM_M + j] = x[j * PERM_N + i] (a transpose of the block seen as a
                PERM_M x PERM_N tile). That is a constant shuffle, lowered to vrgather(ei16).vv with an
                index loaded from the constant pool. With PERM_M != PERM_N the permutation is not its
                own inverse, so the rebuild has to load the inverse index table. Then two products of
                y with loaded weights are live at once, and only then x is used again:
                out = ((y * w0) * (y * w1)) - x. -custom-reverse kills x after the permutation and
                rebuilds it as a gather of y by the inverse permutation right before the xor. With
                one product, x is not live long enough to fire; this is the smallest form that does.

All sizes are exact multiples of the tile sizes, so no masks are involved.
"""

import numpy as np
import torch

import triton
import triton.language as tl

PERM_M = 4
PERM_N = 16
BLOCK_LIVE = PERM_M * PERM_N


@triton.jit
def permute_kernel(a_ptr,  # *Pointer* to input vector.
                   out_ptr,  # *Pointer* to output vector.
                   BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
                   PERM_M: tl.constexpr,  # Rows of the tile x is loaded as.
                   PERM_N: tl.constexpr,  # Columns of that tile (PERM_M * PERM_N == BLOCK_SIZE).
                   ):
    pid = tl.program_id(axis=0)
    offs_in = pid * BLOCK_SIZE + tl.arange(0, PERM_M)[:, None] * PERM_N + tl.arange(0, PERM_N)[None, :]
    offs_out = pid * BLOCK_SIZE + tl.arange(0, PERM_N)[:, None] * PERM_M + tl.arange(0, PERM_M)[None, :]
    v = tl.load(a_ptr + offs_in)
    y = tl.permute(v, (1, 0))
    tl.store(out_ptr + offs_out, y)


@triton.jit
def permute_live_kernel(a_ptr,  # *Pointer* to input vector.
                        w_ptr,  # *Pointer* to 2 weight vectors, stored back to back.
                        out_ptr,  # *Pointer* to output vector.
                        n_elements,  # Size of each vector (multiple of BLOCK_SIZE).
                        BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
                        PERM_M: tl.constexpr,  # Rows of the tile the block is transposed as.
                        PERM_N: tl.constexpr,  # Columns of that tile (PERM_M * PERM_N == BLOCK_SIZE).
                        ):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    v = tl.load(a_ptr + offs)
    t = tl.permute(tl.reshape(v, (PERM_M, PERM_N)), (1, 0))
    y = tl.reshape(t, (BLOCK_SIZE, ))
    w0 = tl.load(w_ptr + offs)
    w1 = tl.load(w_ptr + n_elements + offs)
    p0 = y * w0
    p1 = y * w1
    r = p0 * p1
    out = r - v
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


#y_live = a_live.numpy().reshape(-1, PERM_M, PERM_N).transpose(0, 2, 1).reshape(-1)
#run_on_board(permute_kernel, {
#    "a_ptr": a_live,
#    "out_ptr": torch.zeros(size, dtype=torch.int32),
#}, (size // BLOCK_LIVE, ), {"BLOCK_SIZE": BLOCK_LIVE, "PERM_M": PERM_M, "PERM_N": PERM_N},
#             {"out_ptr": y_live.tolist()}, "rvv-permute")

# int32 arithmetic wraps around the same way in numpy and on the board, so the check is exact.
y_live = a_live.numpy().reshape(-1, PERM_M, PERM_N).transpose(0, 2, 1).reshape(-1)
p_live = [y_live * w_live.numpy()[k * size:(k + 1) * size] for k in range(2)]
out_live = (p_live[0] * p_live[1]) - a_live.numpy()
run_on_board(permute_live_kernel, {
    "a_ptr": a_live,
    "w_ptr": w_live,
    "out_ptr": torch.zeros(size, dtype=torch.int32),
    "n_elements": size,
}, (size // BLOCK_LIVE, ), {"BLOCK_SIZE": BLOCK_LIVE, "PERM_M": PERM_M, "PERM_N": PERM_N},
             {"out_ptr": out_live.tolist()}, "rvv-permute-live")
