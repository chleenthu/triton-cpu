"""
Flip / Permute
==============

Four kernels that permute data within a register-resident tile, used to see how vector reverse
and transpose get lowered (and rematerialized) on the RVV backend:

1. flip1d   -- each program loads BLOCK_SIZE contiguous fp32 values, reverses them in-register with
               tl.flip(x, 0), and stores the block at the mirrored position, so the whole array ends
               up reversed (expected = a[::-1]).
2. flip2d   -- each program loads a BLOCK_M x BLOCK_N tile, reverses each row with tl.flip(x, 1),
               and stores it at the mirrored column tile (expected = np.flip(a, 1)).
3. trans2d  -- each program loads a BLOCK_M x BLOCK_N tile, transposes it with tl.permute(x, (1, 0)),
               and stores it at the transposed tile position (expected = a.T).
4. flip_live -- x stays live across the flip: y = tl.flip(x, 0), then four products of y with
               loaded weights that are all live at once (r = p0*p1*p2*p3, s = p0+p1+p2+p3), and only
               then x is used again: out = ((r + s) ^ x) - y. The xor keeps LLVM from reassociating
               x into an early add, and y is used last, so it is live wherever x is needed again.
               At BLOCK_LIVE = 64 int32 (LMUL8) this peaks at 56 VR units, above the 32 available,
               so -custom-reverse can kill x after the flip and rebuild it as flip(y) right before
               the xor (ReverseRemat[flip-of-flip]: 56 -> 48 VP).

Note that tl.flip is not a dedicated op: the frontend (triton/language/standard.py) bitcasts to int,
reshapes the flipped dim to (2, 2, ..., 2) and applies log2(n) xor-sum steps. With
TRITON_CPU_FLIP_TO_SHUFFLE=1, ConvertFlipToShuffle turns that chain back into one vector.shuffle,
which LLVM lowers to vid / vrsub / vrgather.vv (by default the xor chain is kept). tl.permute, in
contrast, is a tt.trans op.

All sizes are exact multiples of the tile sizes, so no masks are involved.
"""

import numpy as np
import torch

import triton
import triton.language as tl

BLOCK_SIZE = 16
BLOCK_M = 8
BLOCK_N = 16
BLOCK_LIVE = 64


@triton.jit
def flip1d_kernel(a_ptr,  # *Pointer* to input vector.
                  out_ptr,  # *Pointer* to output vector.
                  n_elements,  # Size of the vectors (multiple of BLOCK_SIZE).
                  BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
                  ):
    pid = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK_SIZE)
    x = tl.load(a_ptr + pid * BLOCK_SIZE + offs)
    y = tl.flip(x, 0)
    # Block pid lands at the mirrored block, so the whole array is reversed.
    tl.store(out_ptr + (n_elements - (pid + 1) * BLOCK_SIZE) + offs, y)


@triton.jit
def flip2d_kernel(a_ptr,  # *Pointer* to input matrix (M x N, row-major).
                  out_ptr,  # *Pointer* to output matrix (M x N, row-major).
                  M, N,  # Matrix dimensions (multiples of BLOCK_M / BLOCK_N).
                  BLOCK_M: tl.constexpr,  # Rows per tile.
                  BLOCK_N: tl.constexpr,  # Columns per tile.
                  ):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    x = tl.load(a_ptr + rm[:, None] * N + rn[None, :])
    y = tl.flip(x, 1)
    # Mirrored column tile: columns [N - (pid_n + 1) * BLOCK_N, N - pid_n * BLOCK_N).
    rn_out = (N - (pid_n + 1) * BLOCK_N) + tl.arange(0, BLOCK_N)
    tl.store(out_ptr + rm[:, None] * N + rn_out[None, :], y)


@triton.jit
def trans2d_kernel(a_ptr,  # *Pointer* to input matrix (M x N, row-major).
                   out_ptr,  # *Pointer* to output matrix (N x M, row-major).
                   M, N,  # Matrix dimensions (multiples of BLOCK_M / BLOCK_N).
                   BLOCK_M: tl.constexpr,  # Rows per input tile.
                   BLOCK_N: tl.constexpr,  # Columns per input tile.
                   ):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    x = tl.load(a_ptr + rm[:, None] * N + rn[None, :])
    y = tl.permute(x, (1, 0))
    tl.store(out_ptr + rn[:, None] * M + rm[None, :], y)


@triton.jit
def flip_live_kernel(a_ptr,  # *Pointer* to input vector.
                     w_ptr,  # *Pointer* to 4 weight vectors, stored back to back.
                     out_ptr,  # *Pointer* to output vector.
                     n_elements,  # Size of each vector (multiple of BLOCK_SIZE).
                     BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
                     ):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(a_ptr + offs)
    y = tl.flip(x, 0)
    w0 = tl.load(w_ptr + offs)
    w1 = tl.load(w_ptr + n_elements + offs)
    w2 = tl.load(w_ptr + 2 * n_elements + offs)
    w3 = tl.load(w_ptr + 3 * n_elements + offs)
    p0 = y * w0
    p1 = y * w1
    p2 = y * w2
    p3 = y * w3
    r = p0 * p1 * p2 * p3
    s = p0 + p1 + p2 + p3
    tl.store(out_ptr + offs, ((r + s) ^ x) - y)


# %%
# Test against numpy references.
torch.manual_seed(0)
size = 1024
M, N = 32, 64
triton.runtime.driver.set_active_to_cpu()
a1d = torch.rand(size, dtype=torch.float32)
a2d = torch.rand((M, N), dtype=torch.float32)
a2d_list = a2d.flatten().tolist()
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


run_on_board(flip1d_kernel, {
    "a_ptr": a1d.tolist(),
    "out_ptr": [0.0] * size,
    "n_elements": size,
}, (size // BLOCK_SIZE, ), {"BLOCK_SIZE": BLOCK_SIZE}, {"out_ptr": np.flip(a1d.numpy()).tolist()}, "rvv-flip1d")

#run_on_board(flip2d_kernel, {
#    "a_ptr": a2d_list,
#    "out_ptr": [0.0] * (M * N),
#    "M": M,
#    "N": N,
#}, (M // BLOCK_M, N // BLOCK_N), {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N},
#             {"out_ptr": np.flip(a2d.numpy(), 1).flatten().tolist()}, "rvv-flip2d")

#run_on_board(trans2d_kernel, {
#    "a_ptr": a2d_list,
#    "out_ptr": [0.0] * (M * N),
#    "M": M,
#    "N": N,
#}, (M // BLOCK_M, N // BLOCK_N), {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N},
#             {"out_ptr": a2d.numpy().T.flatten().tolist()}, "rvv-trans2d")

# int32 arithmetic wraps around the same way in numpy and on the board, so the check is exact.
y_live = np.flip(a_live.numpy().reshape(-1, BLOCK_LIVE), 1).reshape(-1)
p_live = [y_live * w_live.numpy()[k * size:(k + 1) * size] for k in range(4)]
out_live = ((p_live[0] * p_live[1] * p_live[2] * p_live[3] + sum(p_live)) ^ a_live.numpy()) - y_live
run_on_board(flip_live_kernel, {
    "a_ptr": a_live,
    "w_ptr": w_live,
    "out_ptr": torch.zeros(size, dtype=torch.int32),
    "n_elements": size,
}, (size // BLOCK_LIVE, ), {"BLOCK_SIZE": BLOCK_LIVE}, {"out_ptr": out_live.tolist()}, "rvv-flip-live")
