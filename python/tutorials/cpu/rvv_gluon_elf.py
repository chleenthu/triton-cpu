"""
Gluon on RVV
============

Kernels from the Gluon tutorials (python/tutorials/gluon/01-intro.py, 02-layouts.py) compiled by
the CPU backend and run on the RISC-V board:

1. copy_scalar -- one program copies a single fp32 value (01-intro).
2. memcpy_1d   -- 1D masked copy with an explicit BlockedLayout (02-layouts).
3. memcpy_2d   -- 2D masked copy that loads in one layout and stores in another through
                  gl.convert_layout (02-layouts, memcpy_2d_inout_kernel).
4. row_sum     -- gl.sum over the rows of a 2D tile.

The CPU backend only consumes TTIR, so for a CPU target GluonASTSource runs the usual Gluon layout
passes and then gluon-strip-encodings, which drops every layout and folds gl.convert_layout away.
Layouts therefore do not change the generated code here; they only have to be valid for the
num_warps / threads-per-warp (32) the frontend verifies them against; the standalone ELF
helpers compile with num_warps=1, so every layout here uses one warp. Ops without a TTIR
counterpart (shared memory, warp specialization, MMA, TMA, ...) are rejected.

xnumel / ynumel are not multiples of the block sizes, so the masked paths are exercised.
"""

import torch

import triton
from triton.backends.compiler import GPUTarget
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon._runtime import GluonASTSource


@gluon.jit
def copy_scalar_kernel(in_ptr, out_ptr):
    value = gl.load(in_ptr)
    gl.store(out_ptr, value)


@gluon.jit
def memcpy_1d_kernel(in_ptr, out_ptr, xnumel, XBLOCK: gl.constexpr, layout: gl.constexpr):
    pid = gl.program_id(0)
    start = pid * XBLOCK
    indices = gl.arange(0, XBLOCK, layout=layout)
    offsets = start + indices
    mask = offsets < xnumel
    value = gl.load(in_ptr + offsets, mask=mask)
    gl.store(out_ptr + offsets, value, mask=mask)


@gluon.jit
def get_mask_and_offsets(start_x, start_y, xnumel, ynumel, xstride, ystride,  #
                         XBLOCK: gl.constexpr, YBLOCK: gl.constexpr, layout: gl.constexpr):
    indices_x = start_x + gl.arange(0, XBLOCK, layout=gl.SliceLayout(dim=1, parent=layout))
    indices_y = start_y + gl.arange(0, YBLOCK, layout=gl.SliceLayout(dim=0, parent=layout))
    mask = (indices_x[:, None] < xnumel) & (indices_y[None, :] < ynumel)
    offsets = xstride * indices_x[:, None] + ystride * indices_y[None, :]
    return mask, offsets


@gluon.jit
def memcpy_2d_inout_kernel(in_ptr, out_ptr,  #
                           xnumel, ynumel, xstride_in, ystride_in, xstride_out, ystride_out,  #
                           layout_in: gl.constexpr, layout_out: gl.constexpr,  #
                           XBLOCK: gl.constexpr, YBLOCK: gl.constexpr):
    pid_x = gl.program_id(0)
    pid_y = gl.program_id(1)
    start_x = pid_x * XBLOCK
    start_y = pid_y * YBLOCK
    mask_in, in_offsets = get_mask_and_offsets(start_x, start_y, xnumel, ynumel, xstride_in, ystride_in,  #
                                               XBLOCK, YBLOCK, layout_in)
    mask_out, out_offsets = get_mask_and_offsets(start_x, start_y, xnumel, ynumel, xstride_out, ystride_out,  #
                                                 XBLOCK, YBLOCK, layout_out)
    value = gl.load(in_ptr + in_offsets, mask=mask_in)
    value = gl.convert_layout(value, layout_out)
    gl.store(out_ptr + out_offsets, value, mask=mask_out)


@gluon.jit
def row_sum_kernel(in_ptr, out_ptr, YNUMEL: gl.constexpr, XBLOCK: gl.constexpr, layout: gl.constexpr):
    pid = gl.program_id(0)
    xs = pid * XBLOCK + gl.arange(0, XBLOCK, layout=gl.SliceLayout(dim=1, parent=layout))
    ys = gl.arange(0, YNUMEL, layout=gl.SliceLayout(dim=0, parent=layout))
    x = gl.load(in_ptr + xs[:, None] * YNUMEL + ys[None, :])
    gl.store(out_ptr + xs, gl.sum(x, axis=1))


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

run_on_board(copy_scalar_kernel, {
    "in_ptr": [42.0],
    "out_ptr": [0.0],
}, (1, ), {}, {"out_ptr": [42.0]}, "rvv-gluon-copy-scalar")

XBLOCK_1D = 256
xnumel_1d = 1000
a1d = torch.randn(xnumel_1d)
run_on_board(memcpy_1d_kernel, {
    "in_ptr": a1d.tolist(),
    "out_ptr": [0.0] * xnumel_1d,
    "xnumel": xnumel_1d,
}, (triton.cdiv(xnumel_1d, XBLOCK_1D), ), {
    "XBLOCK": XBLOCK_1D,
    "layout": gl.BlockedLayout([1], [32], [1], [0]),
}, {"out_ptr": a1d.tolist()}, "rvv-gluon-memcpy-1d")

# Row-major in, column-major out: out viewed as (ynumel, xnumel) row-major is a.T.
XBLOCK_2D, YBLOCK_2D = 32, 32
xnumel_2d, ynumel_2d = 50, 70
a2d = torch.randn(xnumel_2d, ynumel_2d)
run_on_board(memcpy_2d_inout_kernel, {
    "in_ptr": a2d.flatten().tolist(),
    "out_ptr": [0.0] * (xnumel_2d * ynumel_2d),
    "xnumel": xnumel_2d,
    "ynumel": ynumel_2d,
    "xstride_in": ynumel_2d,
    "ystride_in": 1,
    "xstride_out": 1,
    "ystride_out": xnumel_2d,
}, (triton.cdiv(xnumel_2d, XBLOCK_2D), triton.cdiv(ynumel_2d, YBLOCK_2D)), {
    "layout_in": gl.BlockedLayout([1, 1], [1, 32], [1, 1], [1, 0]),
    "layout_out": gl.BlockedLayout([1, 1], [32, 1], [1, 1], [0, 1]),
    "XBLOCK": XBLOCK_2D,
    "YBLOCK": YBLOCK_2D,
}, {"out_ptr": a2d.T.flatten().tolist()}, "rvv-gluon-memcpy-2d")

XBLOCK_SUM, YNUMEL_SUM = 16, 64
xnumel_sum = 4 * XBLOCK_SUM
a_sum = torch.randn(xnumel_sum, YNUMEL_SUM)
run_on_board(row_sum_kernel, {
    "in_ptr": a_sum.flatten().tolist(),
    "out_ptr": [0.0] * xnumel_sum,
}, (xnumel_sum // XBLOCK_SUM, ), {
    "YNUMEL": YNUMEL_SUM,
    "XBLOCK": XBLOCK_SUM,
    "layout": gl.BlockedLayout([1, 1], [1, 32], [1, 1], [1, 0]),
}, {"out_ptr": a_sum.sum(dim=1).tolist()}, "rvv-gluon-row-sum")
