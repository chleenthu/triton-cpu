"""
Async Copy in Gluon (RISC-V ELF)
================================

The kernels of python/tutorials/gluon/03-async-copy.py, compiled for riscv64 by the CPU backend,
deployed to the board and checked there:

* memcpy_1d_cpasync_kernel: 1D masked copy,
* elementwise_add_kernel: c = a + b over 2D tiles, one row block per program, looping over the
  column blocks (ported unchanged),
* elementwise_add_cpasync_kernel: the same loop with the A / B tiles staged through shared memory,
* elementwise_add_pipelined_kernel: the software-pipelined version (prologue, overlapped steady
  state, drain), with two stages.

Differences from the GPU tutorial:

* The CPU has no shared memory and no cp.async, and the CPU path of Gluon (gluon-strip-encodings)
  rejects ttg.local_alloc and the async copy ops. Every
  allocate_shared_memory + cp.async_load + commit_group / wait_group + smem.load sequence is
  therefore a plain gl.load into registers.
* The pipelined kernel keeps the tutorial's structure, with the shared-memory ring buffer
  replaced by loop-carried register tiles: the loads for column block i + 1 are issued before the
  add / store of block i. The load and the add are not actually asynchronous on the CPU. Only the
  double-buffered case (num_buffers = 2) is ported; num_buffers = 1 is
  elementwise_add_cpasync_kernel, and a ring of three register tiles cannot be indexed by
  copy_idx % num_buffers.
* All layouts use one warp (warps_per_cta = 1), since the standalone ELF helpers compile with
  num_warps = 1. The CPU backend drops the layouts in any case.
* Smaller problems than the tutorial's (1000 x 2000, 4000 x 120), so the inputs fit in the
  generated runner, chosen so that neither dimension is a multiple of its block size and the
  masked paths are exercised. No benchmarks against the GPU numbers.
"""

import torch

import triton
from triton.backends.compiler import GPUTarget
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon._runtime import GluonASTSource


@gluon.jit
def memcpy_1d_cpasync_kernel(in_ptr, out_ptr, xnumel, XBLOCK: gl.constexpr):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    offsets = pid * XBLOCK + gl.arange(0, XBLOCK, layout=layout)
    mask = offsets < xnumel
    # GPU: allocate_shared_memory + cp.async_load + commit_group + wait_group(0) + smem.load.
    value = gl.load(in_ptr + offsets, mask=mask)
    gl.store(out_ptr + offsets, value, mask=mask)


@gluon.jit
def elementwise_add_kernel(  #
        a_ptr, b_ptr, c_ptr, xnumel, ynumel,  #
        xstride_a, ystride_a, xstride_b, ystride_b, xstride_c, ystride_c,  #
        XBLOCK: gl.constexpr, YBLOCK: gl.constexpr,  #
):
    pid = gl.program_id(0)
    # Compute the offset to the row this program will process.
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 1], [1, 0])
    xoffs = pid * XBLOCK + gl.arange(0, XBLOCK, gl.SliceLayout(1, layout))

    a_ptrs = a_ptr + xstride_a * xoffs[:, None]
    b_ptrs = b_ptr + xstride_b * xoffs[:, None]
    c_ptrs = c_ptr + xstride_c * xoffs[:, None]

    for yoff in range(0, ynumel, YBLOCK):
        # Offset to the column block.
        yoffs = yoff + gl.arange(0, YBLOCK, gl.SliceLayout(0, layout))
        mask = (xoffs < xnumel)[:, None] & (yoffs < ynumel)[None, :]

        a_val = gl.load(a_ptrs + ystride_a * yoffs[None, :], mask=mask)
        b_val = gl.load(b_ptrs + ystride_b * yoffs[None, :], mask=mask)

        c_val = a_val + b_val
        gl.store(c_ptrs + ystride_c * yoffs[None, :], c_val, mask=mask)


@gluon.jit
def elementwise_add_cpasync_kernel(  #
        a_ptr, b_ptr, c_ptr, xnumel, ynumel,  #
        xstride_a, ystride_a, xstride_b, ystride_b, xstride_c, ystride_c,  #
        XBLOCK: gl.constexpr, YBLOCK: gl.constexpr,  #
):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 1], [1, 0])
    xoffs = pid * XBLOCK + gl.arange(0, XBLOCK, gl.SliceLayout(1, layout))
    a_ptrs = a_ptr + xstride_a * xoffs[:, None]
    b_ptrs = b_ptr + xstride_b * xoffs[:, None]
    c_ptrs = c_ptr + xstride_c * xoffs[:, None]

    # GPU: a_smem / b_smem = gl.allocate_shared_memory(dtype, [XBLOCK, YBLOCK], layout=smem_layout).
    for yoff in range(0, ynumel, YBLOCK):
        yoffs = yoff + gl.arange(0, YBLOCK, gl.SliceLayout(0, layout))
        mask = (xoffs < xnumel)[:, None] & (yoffs < ynumel)[None, :]

        # GPU: cp.async_load both tiles, commit_group, wait_group(0), then a_smem / b_smem.load.
        a_val = gl.load(a_ptrs + ystride_a * yoffs[None, :], mask=mask)
        b_val = gl.load(b_ptrs + ystride_b * yoffs[None, :], mask=mask)

        c_val = a_val + b_val
        gl.store(c_ptrs + ystride_c * yoffs[None, :], c_val, mask=mask)


@gluon.jit
def issue_loads(copy_idx, a_ptrs, ystride_a, b_ptrs, xmask, ynumel, y_idx, ystride_b, YBLOCK: gl.constexpr):
    # Masking the loads by yoffs < ynumel handles the prefetch past the last block.
    yoffs = copy_idx * YBLOCK + y_idx
    mask = xmask & (yoffs < ynumel)[None, :]
    # GPU: cp.async_load into a_smem / b_smem.index(copy_idx % num_buffers), then commit_group.
    a_val = gl.load(a_ptrs + ystride_a * yoffs[None, :], mask)
    b_val = gl.load(b_ptrs + ystride_b * yoffs[None, :], mask)
    return copy_idx + 1, a_val, b_val


@gluon.jit
def perform_add(read_idx, a_val, b_val, c_ptrs, ynumel, ystride_c, y_idx, xmask, YBLOCK: gl.constexpr):
    # GPU: a_val / b_val = a_smem / b_smem.index(read_idx % num_buffers).load(layout).
    c_val = a_val + b_val
    yoffs = read_idx * YBLOCK + y_idx
    mask = xmask & (yoffs < ynumel)[None, :]
    gl.store(c_ptrs + ystride_c * yoffs[None, :], c_val, mask=mask)
    return read_idx + 1


@gluon.jit
def elementwise_add_pipelined_kernel(  #
        a_ptr, b_ptr, c_ptr, xnumel, ynumel,  #
        xstride_a, ystride_a, xstride_b, ystride_b, xstride_c, ystride_c,  #
        XBLOCK: gl.constexpr, YBLOCK: gl.constexpr,  #
):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 1], [1, 0])
    xoffs = pid * XBLOCK + gl.arange(0, XBLOCK, gl.SliceLayout(1, layout))
    a_ptrs = a_ptr + xstride_a * xoffs[:, None]
    b_ptrs = b_ptr + xstride_b * xoffs[:, None]
    c_ptrs = c_ptr + xstride_c * xoffs[:, None]
    y_idx = gl.arange(0, YBLOCK, gl.SliceLayout(0, layout))
    xmask = (xoffs < xnumel)[:, None]

    copy_idx = 0
    read_idx = 0

    # Prologue (num_buffers - 1 = 1 peeled copy): fill the pipeline.
    copy_idx, a_val, b_val = issue_loads(copy_idx, a_ptrs, ystride_a, b_ptrs, xmask, ynumel, y_idx, ystride_b, YBLOCK)

    # Steady state: issue the copy of the next block, then process the current one.
    for _ in range(gl.cdiv(ynumel, YBLOCK) - 1):
        copy_idx, a_next, b_next = issue_loads(copy_idx, a_ptrs, ystride_a, b_ptrs, xmask, ynumel, y_idx, ystride_b,
                                               YBLOCK)
        read_idx = perform_add(read_idx, a_val, b_val, c_ptrs, ynumel, ystride_c, y_idx, xmask, YBLOCK)
        a_val = a_next
        b_val = b_next

    # Drain the last block.
    read_idx = perform_add(read_idx, a_val, b_val, c_ptrs, ynumel, ystride_c, y_idx, xmask, YBLOCK)


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

# The tutorial's memcpy test sizes.
for xnumel, XBLOCK in [(200, 128), (1000, 256)]:
    a = torch.randn(xnumel)
    run_on_board(memcpy_1d_cpasync_kernel, {
        "in_ptr": a.tolist(),
        "out_ptr": [0.0] * xnumel,
        "xnumel": xnumel,
    }, (triton.cdiv(xnumel, XBLOCK), ), {"XBLOCK": XBLOCK}, {"out_ptr": a.tolist()},
                 f"rvv-gluon-memcpy-1d-cpasync-{xnumel}")

XBLOCK, YBLOCK = 32, 64
for xnumel, ynumel in [(100, 200), (400, 120)]:
    a = torch.randn(xnumel, ynumel)
    b = torch.randn(xnumel, ynumel)
    arguments = {
        "a_ptr": a.flatten().tolist(),
        "b_ptr": b.flatten().tolist(),
        "c_ptr": [0.0] * (xnumel * ynumel),
        "xnumel": xnumel,
        "ynumel": ynumel,
        "xstride_a": ynumel,
        "ystride_a": 1,
        "xstride_b": ynumel,
        "ystride_b": 1,
        "xstride_c": ynumel,
        "ystride_c": 1,
    }
    expected = {"c_ptr": (a + b).flatten().tolist()}
    for kernel, name in [(elementwise_add_kernel, "elementwise-add"),
                         (elementwise_add_cpasync_kernel, "elementwise-add-cpasync"),
                         (elementwise_add_pipelined_kernel, "elementwise-add-pipelined")]:
        run_on_board(kernel, arguments, (triton.cdiv(xnumel, XBLOCK), ), {"XBLOCK": XBLOCK, "YBLOCK": YBLOCK},
                     expected, f"rvv-gluon-{name}-{xnumel}x{ynumel}")
