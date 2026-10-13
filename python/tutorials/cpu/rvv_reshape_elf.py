"""
Reshape
=======

Three kernels with a plain tl.reshape, used to see what a reshape lowers to and whether
-custom-reverse has anything to rebuild there:

reshape      -- each program loads BLOCK_SIZE = RESHAPE_M * RESHAPE_N int32 values x, reshapes them to
                a RESHAPE_M x RESHAPE_N tile and stores the tile row-major, so out == a. Nothing else
                happens, so the dumped IR shows only what tl.reshape itself becomes. It is free:
                the four row stores take slices of one load to consecutive addresses, and the
                backend merges them back into one vle32 + one vse32 at m8.
reshape_add  -- the smallest kernel where the reshape is not free: the same as reshape, but each row
                is computed on (+ 1) before the row-major store. The rows are no longer plain slices
                of the load, so the backend has to split the m8 group into four m2 rows with three
                vslidedown. The split is a slide, not a register rename, because the pipeline only
                pins VLEN >= 256 (-riscv-v-vector-bits-min=256): row i starts 16 * i lanes in, which
                is a register boundary only when VLEN is exactly 256.
reshape_live -- each program loads BLOCK_SIZE = RESHAPE_M * RESHAPE_N int32 values x, and reshapes
                them to a RESHAPE_M x RESHAPE_N tile, y = tl.reshape(x, (RESHAPE_M, RESHAPE_N)), so
                y[i, j] = x[i * RESHAPE_N + j]. The four weight vectors are loaded as tiles of the same
                shape, and the four products of y with them are all live at once
                (r = p0*p1*p2*p3, s = p0+p1+p2+p3). Only then x is used again, back in 1D:
                out = (reshape(r + s) ^ x) - reshape(y). The xor keeps LLVM from reassociating x into
                an early add, and y is used last, so it is live wherever x is needed again.

A row-major reshape does not move any lane in the tensor. What decides its cost is whether a 1D
value and its 2D rows both have to exist in registers: TTCIR lowers 2D loads, stores and ops row by
row, so a 2D value that is computed on and meets a 1D value (or a 2D memory op) needs the rows split
out with vslidedown, and a 2D value turned back into 1D needs them joined with vslideup. When the
rows only pass between contiguous memory, or an elementwise op sits between a reshape and its
reshape back, LLVM folds the slices away.

All sizes are exact multiples of the tile sizes, so no masks are involved.
"""

import numpy as np
import torch

import triton
import triton.language as tl

RESHAPE_M = 4
RESHAPE_N = 16
BLOCK_LIVE = RESHAPE_M * RESHAPE_N


@triton.jit
def reshape_kernel(a_ptr,  # *Pointer* to input vector.
                   out_ptr,  # *Pointer* to output vector.
                   BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
                   RESHAPE_M: tl.constexpr,  # Rows of the tile x is reshaped to.
                   RESHAPE_N: tl.constexpr,  # Columns of that tile (RESHAPE_M * RESHAPE_N == BLOCK_SIZE).
                   ):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs2d = pid * BLOCK_SIZE + tl.arange(0, RESHAPE_M)[:, None] * RESHAPE_N + tl.arange(0, RESHAPE_N)[None, :]
    x = tl.load(a_ptr + offs)
    y = tl.reshape(x, (RESHAPE_M, RESHAPE_N))
    tl.store(out_ptr + offs2d, y)


@triton.jit
def reshape_add_kernel(a_ptr,  # *Pointer* to input vector.
                       out_ptr,  # *Pointer* to output vector.
                       BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
                       RESHAPE_M: tl.constexpr,  # Rows of the tile x is reshaped to.
                       RESHAPE_N: tl.constexpr,  # Columns of that tile (RESHAPE_M * RESHAPE_N == BLOCK_SIZE).
                       ):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs2d = pid * BLOCK_SIZE + tl.arange(0, RESHAPE_M)[:, None] * RESHAPE_N + tl.arange(0, RESHAPE_N)[None, :]
    x = tl.load(a_ptr + offs)
    y = tl.reshape(x, (RESHAPE_M, RESHAPE_N))
    tl.store(out_ptr + offs2d, y + 1)


@triton.jit
def reshape_live_kernel(a_ptr,  # *Pointer* to input vector.
                        w_ptr,  # *Pointer* to 4 weight vectors, stored back to back.
                        out_ptr,  # *Pointer* to output vector.
                        n_elements,  # Size of each vector (multiple of BLOCK_SIZE).
                        BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
                        RESHAPE_M: tl.constexpr,  # Rows of the tile x is reshaped to.
                        RESHAPE_N: tl.constexpr,  # Columns of that tile (RESHAPE_M * RESHAPE_N == BLOCK_SIZE).
                        ):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs2d = pid * BLOCK_SIZE + tl.arange(0, RESHAPE_M)[:, None] * RESHAPE_N + tl.arange(0, RESHAPE_N)[None, :]
    x = tl.load(a_ptr + offs)
    y = tl.reshape(x, (RESHAPE_M, RESHAPE_N))
    w0 = tl.load(w_ptr + offs2d)
    w1 = tl.load(w_ptr + n_elements + offs2d)
    w2 = tl.load(w_ptr + 2 * n_elements + offs2d)
    w3 = tl.load(w_ptr + 3 * n_elements + offs2d)
    p0 = y * w0
    p1 = y * w1
    p2 = y * w2
    p3 = y * w3
    r = p0 * p1 * p2 * p3
    s = p0 + p1 + p2 + p3
    tl.store(out_ptr + offs, (tl.reshape(r + s, (BLOCK_SIZE, )) ^ x) - tl.reshape(y, (BLOCK_SIZE, )))


# %%
# Test against a numpy reference.
torch.manual_seed(0)
size = 1024
triton.runtime.driver.set_active_to_cpu()
a_live = torch.randint(-100, 100, (size, ), dtype=torch.int32)
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


run_on_board(reshape_kernel, {
    "a_ptr": a_live,
    "out_ptr": torch.zeros(size, dtype=torch.int32),
}, (size // BLOCK_LIVE, ), {"BLOCK_SIZE": BLOCK_LIVE, "RESHAPE_M": RESHAPE_M, "RESHAPE_N": RESHAPE_N},
             {"out_ptr": a_live.tolist()}, "rvv-reshape")

run_on_board(reshape_add_kernel, {
    "a_ptr": a_live,
    "out_ptr": torch.zeros(size, dtype=torch.int32),
}, (size // BLOCK_LIVE, ), {"BLOCK_SIZE": BLOCK_LIVE, "RESHAPE_M": RESHAPE_M, "RESHAPE_N": RESHAPE_N},
             {"out_ptr": (a_live + 1).tolist()}, "rvv-reshape-add")

# int32 arithmetic wraps around the same way in numpy and on the board, so the check is exact.
# A row-major reshape keeps the element order, so flattened y is x itself.
#y_live = a_live.numpy()
#p_live = [y_live * w_live.numpy()[k * size:(k + 1) * size] for k in range(4)]
#out_live = ((p_live[0] * p_live[1] * p_live[2] * p_live[3] + sum(p_live)) ^ a_live.numpy()) - y_live
#run_on_board(reshape_live_kernel, {
#    "a_ptr": a_live,
#    "w_ptr": w_live,
#    "out_ptr": torch.zeros(size, dtype=torch.int32),
#    "n_elements": size,
#}, (size // BLOCK_LIVE, ), {"BLOCK_SIZE": BLOCK_LIVE, "RESHAPE_M": RESHAPE_M, "RESHAPE_N": RESHAPE_N},
#             {"out_ptr": out_live.tolist()}, "rvv-reshape-live")
