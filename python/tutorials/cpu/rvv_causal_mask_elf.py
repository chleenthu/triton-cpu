"""
Causal Mask (bitmask form)
==========================

The causal mask from the Gluon attention example (python/examples/gluon/01-attention-forward.py,
_mask_scalar / _apply_causal_mask), applied to an M x N fp32 score matrix tile by tile:

    col_limit_right = (offs_m - start_n + 1)[:, None]
    s = offs_n & ~0xf                       # start of the lane's 16-column group
    i = offs_n & 0xf                        # lane inside the group
    mask = -1 << max(col_limit_right - s, 0)
    keep = (mask & (1 << i)) == 0           # bit i of mask clear <=> i < col_limit_right - s
    qk = where(keep, qk, -inf)

On NVIDIA the 16-bit mask word lets ptxas use one R2P instruction per 16 lanes. Per lane it is the
tail mask offs_n < col_limit_right, i.e. column <= row: element (r, c) is kept iff c <= r.

Deviations from the example: _mask_scalar is applied to the whole tile instead of through
gl.map_elementwise (no CPU lowering for tt.map_elementwise), max(...) becomes gl.maximum, and the
shift is capped at 16 (see _mask_scalar).

Each program owns one BLOCK_M x BLOCK_N tile; M and N are multiples of the block sizes, so the
loads and stores need no mask. The CPU backend strips the Gluon layouts, so the layout only has to
be valid for the one warp the standalone ELF helper compiles with.
"""

import torch

import triton
from triton.backends.compiler import GPUTarget
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon._runtime import GluonASTSource

M, N = 128, 128
BLOCK_M, BLOCK_N = 16, 64


@gluon.jit
def _mask_scalar(qk, col_limit_right, s, i):
    col_lim_right_s = col_limit_right - s
    # Only bits 0..15 are tested (i < 16), so cap the shift at 16: -1 << 16 keeps all 16 lanes.
    # The example relies on PTX shl clamping shifts >= 32 to give 0; in LLVM such a shift is
    # poison and RVV vsll uses only the low 5 bits (-1 << 32 == -1), which would mask every lane.
    col_lim_right_cur = gl.minimum(gl.maximum(col_lim_right_s, 0), 16)
    mask = -1 << col_lim_right_cur
    mask_i_bit = (mask & (1 << i)) == 0
    return gl.where(mask_i_bit, qk, -float("inf"))


@gluon.jit
def causal_mask_kernel(qk_ptr, out_ptr, stride_m,  #
                       BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, layout: gl.constexpr):
    pid_m = gl.program_id(0)
    pid_n = gl.program_id(1)
    start_n = pid_n * BLOCK_N
    offs_m = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(dim=1, parent=layout))
    offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(dim=0, parent=layout))[None, :]
    offsets = offs_m[:, None] * stride_m + (start_n + offs_n)
    qk = gl.load(qk_ptr + offsets)

    # _apply_causal_mask, with the column limit of the attention softmax loop.
    col_limit_right = (offs_m - start_n + 1)[:, None]
    s = offs_n & ~0xf
    i = offs_n & 0xf
    # The example runs _mask_scalar per element through gl.map_elementwise, which the CPU backend
    # cannot lower (tt.map_elementwise reaches LLVM translation). Its body is all elementwise ops,
    # so applying it to the whole tile gives the same result per lane.
    qk = _mask_scalar(qk, col_limit_right, s, i)

    gl.store(out_ptr + offsets, qk)


# %%
# Test against a torch reference.
triton.runtime.driver.set_active_to_cpu()

from triton.backends.cpu import riscv
from triton.backends.cpu.riscv import compile_deploy_and_run

# The standalone ELF helpers build a Triton ASTSource; build a Gluon one instead.
riscv.ASTSource = lambda fn, signature, constexprs: GluonASTSource(fn, signature, constexprs,
                                                                   target=GPUTarget("cpu", 0, 0))

RISCV_HOST = "chlee@140.114.78.64"
RISCV_REMOTE_DIR = "~/triton-riscv-elf"


def run_on_board(kernel, arguments, grid, constexprs, expected, name):
    result = compile_deploy_and_run(kernel, arguments, grid, f"artifacts/riscv/{name}.elf", RISCV_HOST,
                                     constexprs=constexprs, expected=expected, remote_dir=RISCV_REMOTE_DIR)
    print(f"{name}:")
    print(result.stdout, end="")
    print(result.stderr, end="")


torch.manual_seed(0)
qk = torch.randn(M, N, dtype=torch.float32)
# Element (r, c) is kept iff c <= r.
expected = torch.where(torch.ones(M, N, dtype=torch.bool).tril(), qk, torch.tensor(float("-inf")))
layout = gl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[1, 32], warps_per_cta=[1, 1], order=[1, 0])
run_on_board(causal_mask_kernel, {
    "qk_ptr": qk.flatten(),  # row-major M x N
    "out_ptr": torch.zeros(M * N, dtype=torch.float32),
    "stride_m": N,
}, (M // BLOCK_M, N // BLOCK_N), {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "layout": layout},
             {"out_ptr": expected.flatten().tolist()}, "rvv-causal-mask")
