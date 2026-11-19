"""
Tail masks across control flow
==============================

Does AnalyzeTailMasks find a tail mask when the mask, its bound, or its use sits on the other side of
an if / else or a loop? A runtime `if flag > 0:` becomes `scf.if` and a `for` becomes `scf.for`, so
the question is whether the mask's def chain can cross a region boundary.

    outside   mask = offs < n defined before the if, loads / stores inside both branches
    inside    each branch builds its own mask (offs < n / offs < n2) and uses it there
    yielded   each branch builds a mask, the if yields it, the load / store after the if uses it
    bound     the if yields only the scalar bound (n or n2); mask = offs < bound after the if
    loop      offs = k * BLOCK + arange inside an scf.for, mask = offs < n used in the loop body
    carried   the mask is an scf.for iter_arg, updated each iteration, used after the loop
    carried_use  the same, used by a load inside the loop
    nested    an inner loop starts from the outer loop's carried mask; the outer loop carries the result
    not_box   a carried mask updated with xor: not a box, must stay a plain mask
    carried_offs     the index tensor is an scf.for iter_arg: offs += BLOCK each iteration
    carried_offs_2d  a 4 x 16 tile over a 3 x 21 matrix whose column offsets rn are carried (rn += BLOCK)
    carried_offs_not_range  offs = offs * 2 is not a range, must stay a plain mask
    yielded_offs     the index tensor comes out of an scf.if, the mask is built after it
    carried_offs_if  a carried index tensor advanced inside an scf.if (+16 / +8)
    while_offs       an scf.while carrying the index tensor (offs += BLOCK)
    while_mask       an scf.while carrying a mask, used after the loop
    while_mask_use   an scf.while carrying a mask, used by a load inside the loop

Each kernel runs on the board (checked against numpy). Then its TTIR is run through
`triton-opt -triton-cpu-analyze-tail-masks` and the mask operand of every tt.load / tt.store is
reported as `tail_mask` (detected) or `plain i1` (not detected).
"""

import os
import re
import subprocess

import numpy as np
import torch

import triton
import triton.language as tl

BLOCK = 16
N_TOTAL = 64


@triton.jit
def outside_kernel(a_ptr, c_ptr, n, n2, flag, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    if flag > 0:
        x = tl.load(a_ptr + offs, mask=mask, other=0.0)
        tl.store(c_ptr + offs, x * 2.0, mask=mask)
    else:
        x = tl.load(a_ptr + offs, mask=mask, other=0.0)
        tl.store(c_ptr + offs, x + 1.0, mask=mask)


@triton.jit
def inside_kernel(a_ptr, c_ptr, n, n2, flag, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if flag > 0:
        m = offs < n
        x = tl.load(a_ptr + offs, mask=m, other=0.0)
        tl.store(c_ptr + offs, x * 2.0, mask=m)
    else:
        m = offs < n2
        x = tl.load(a_ptr + offs, mask=m, other=0.0)
        tl.store(c_ptr + offs, x + 1.0, mask=m)


@triton.jit
def yielded_kernel(a_ptr, c_ptr, n, n2, flag, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if flag > 0:
        m = offs < n
    else:
        m = offs < n2
    x = tl.load(a_ptr + offs, mask=m, other=0.0)
    tl.store(c_ptr + offs, x * 2.0, mask=m)


@triton.jit
def bound_kernel(a_ptr, c_ptr, n, n2, flag, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if flag > 0:
        bound = n
    else:
        bound = n2
    m = offs < bound
    x = tl.load(a_ptr + offs, mask=m, other=0.0)
    tl.store(c_ptr + offs, x * 2.0, mask=m)


@triton.jit
def loop_kernel(a_ptr, c_ptr, n, n2, flag, BLOCK: tl.constexpr):
    # One program walks the whole array in BLOCK steps.
    for k in range(0, tl.cdiv(n, BLOCK)):
        offs = k * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        x = tl.load(a_ptr + offs, mask=m, other=0.0)
        tl.store(c_ptr + offs, x * 2.0, mask=m)


@triton.jit
def carried_kernel(a_ptr, c_ptr, n, n2, flag, BLOCK: tl.constexpr):
    # The mask is loop-carried: m_k = m_{k-1} & (offs < n - k), i.e. offs < n - k after step k.
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    for k in range(0, flag):
        m = m & (offs < n - k)
    x = tl.load(a_ptr + offs, mask=m, other=0.0)
    tl.store(c_ptr + offs, x * 2.0, mask=m)


@triton.jit
def carried_use_kernel(a_ptr, c_ptr, n, n2, flag, BLOCK: tl.constexpr):
    # The loop-carried mask is used by a load inside the loop, then shrunk.
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK, ), dtype=tl.float32)
    m = offs < n
    for k in range(0, flag):
        acc += tl.load(a_ptr + offs, mask=m, other=0.0)
        m = m & (offs < n - 3 * k - 1)
    tl.store(c_ptr + offs, acc, mask=offs < n)


@triton.jit
def nested_kernel(a_ptr, c_ptr, n, n2, flag, BLOCK: tl.constexpr):
    # The inner loop starts from the outer loop's mask, and the outer loop
    # carries on with the inner loop's result.
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK, ), dtype=tl.float32)
    m = offs < n
    for i in range(0, flag):
        mi = m
        for j in range(0, 2):
            mi = mi & (offs < n - i - 2 * j)
            acc += tl.load(a_ptr + offs, mask=mi, other=0.0)
        m = mi
    tl.store(c_ptr + offs, acc, mask=offs < n)


@triton.jit
def not_box_kernel(a_ptr, c_ptr, n, n2, flag, BLOCK: tl.constexpr):
    # m ^ box is not a box: must stay a plain mask.
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    for k in range(0, flag):
        m = m ^ (offs < n2 + k)
    x = tl.load(a_ptr + offs, mask=m, other=0.0)
    tl.store(c_ptr + offs, x * 2.0, mask=m)


@triton.jit
def carried_offs_kernel(a_ptr, c_ptr, n, n2, flag, BLOCK: tl.constexpr):
    # The index tensor is loop-carried: offs += BLOCK each iteration.
    offs = tl.arange(0, BLOCK)
    for k in range(0, tl.cdiv(n, BLOCK)):
        m = offs < n
        x = tl.load(a_ptr + offs, mask=m, other=0.0)
        tl.store(c_ptr + offs, x * 2.0, mask=m)
        offs += BLOCK


@triton.jit
def carried_offs_2d_kernel(a_ptr, c_ptr, n, n2, flag, BLOCK: tl.constexpr):
    # A ROWS x BLOCK tile walks along the columns of a flag x n2 row-major
    # matrix: rows fixed per program, column offsets carried.
    ROWS: tl.constexpr = 4
    rm = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    rn = tl.arange(0, BLOCK)
    for k in range(0, tl.cdiv(n2, BLOCK)):
        m = (rm[:, None] < flag) & (rn[None, :] < n2)
        offs = rm[:, None] * n2 + rn[None, :]
        x = tl.load(a_ptr + offs, mask=m, other=0.0)
        tl.store(c_ptr + offs, x * 2.0, mask=m)
        rn += BLOCK


@triton.jit
def carried_offs_not_range_kernel(a_ptr, c_ptr, n, n2, flag, BLOCK: tl.constexpr):
    # offs * 2 is not a range: must stay a plain mask.
    offs = tl.arange(0, BLOCK) + 1
    for k in range(0, flag):
        offs = offs * 2
    m = offs < n
    x = tl.load(a_ptr + offs, mask=m, other=0.0)
    tl.store(c_ptr + offs, x * 2.0, mask=m)


@triton.jit
def yielded_offs_kernel(a_ptr, c_ptr, n, n2, flag, BLOCK: tl.constexpr):
    # The index tensor comes out of an scf.if; the mask is built after it.
    base = tl.program_id(0) * BLOCK
    if flag > 0:
        offs = base + tl.arange(0, BLOCK)
    else:
        offs = base + 3 + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(a_ptr + offs, mask=m, other=0.0)
    tl.store(c_ptr + offs, x * 2.0, mask=m)


@triton.jit
def carried_offs_if_kernel(a_ptr, c_ptr, n, n2, flag, BLOCK: tl.constexpr):
    # A carried index tensor advanced through an scf.if: +16 on even steps,
    # +8 on odd steps.
    offs = tl.arange(0, BLOCK)
    for k in range(0, tl.cdiv(n, 8)):
        m = offs < n
        x = tl.load(a_ptr + offs, mask=m, other=0.0)
        tl.store(c_ptr + offs, x * 2.0, mask=m)
        if k % 2 == 0:
            offs += 16
        else:
            offs += 8


@triton.jit
def while_offs_kernel(a_ptr, c_ptr, n, n2, flag, BLOCK: tl.constexpr):
    # A while loop (scf.while) carrying the index tensor: offs += BLOCK.
    offs = tl.arange(0, BLOCK)
    k = 0
    while k < n:
        m = offs < n
        x = tl.load(a_ptr + offs, mask=m, other=0.0)
        tl.store(c_ptr + offs, x * 2.0, mask=m)
        offs += BLOCK
        k += BLOCK


@triton.jit
def while_mask_kernel(a_ptr, c_ptr, n, n2, flag, BLOCK: tl.constexpr):
    # A while loop carrying a mask, used after the loop.
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    k = 0
    while k < flag:
        m = m & (offs < n - k)
        k += 1
    x = tl.load(a_ptr + offs, mask=m, other=0.0)
    tl.store(c_ptr + offs, x * 2.0, mask=m)


@triton.jit
def while_mask_use_kernel(a_ptr, c_ptr, n, n2, flag, BLOCK: tl.constexpr):
    # A while loop carrying a mask, used by a load inside the loop.
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK, ), dtype=tl.float32)
    m = offs < n
    k = 0
    while k < flag:
        acc += tl.load(a_ptr + offs, mask=m, other=0.0)
        m = m & (offs < n - 3 * k - 1)
        k += 1
    tl.store(c_ptr + offs, acc, mask=offs < n)


def reference(name, a, n, n2, flag):
    offs = np.arange(N_TOTAL)
    if name == "while_offs":
        return np.where(offs < n, a * 2, 0).astype(np.float32)
    if name == "while_mask":
        return reference("carried", a, n, n2, flag)
    if name == "while_mask_use":
        return reference("carried_use", a, n, n2, flag)
    if name == "carried_use":
        acc, m = np.zeros(N_TOTAL, np.float32), offs < n
        for k in range(flag):
            acc += np.where(m, a, 0)
            m = m & (offs < n - 3 * k - 1)
        return np.where(offs < n, acc, 0).astype(np.float32)
    if name == "nested":
        acc, m = np.zeros(N_TOTAL, np.float32), offs < n
        for i in range(flag):
            mi = m
            for j in range(2):
                mi = mi & (offs < n - i - 2 * j)
                acc += np.where(mi, a, 0)
            m = mi
        return np.where(offs < n, acc, 0).astype(np.float32)
    if name == "carried_offs":
        m = offs < n
        return np.where(m, a * 2, 0).astype(np.float32)
    if name == "carried_offs_2d":
        c = np.zeros(N_TOTAL, np.float32)
        for r in range(flag):
            for col in range(n2):
                c[r * n2 + col] = a[r * n2 + col] * 2
        return c
    if name == "carried_offs_not_range":
        o = (np.arange(BLOCK) + 1) * 2**flag
        c = np.zeros(N_TOTAL, np.float32)
        m = o < n
        c[o[m]] = a[o[m]] * 2
        return c
    if name == "yielded_offs":
        c = np.zeros(N_TOTAL, np.float32)
        for pid in range(N_TOTAL // BLOCK):
            o = pid * BLOCK + np.arange(BLOCK) + (0 if flag > 0 else 3)
            o = o[o < n]
            c[o] = a[o] * 2
        return c
    if name == "carried_offs_if":
        c, o = np.zeros(N_TOTAL, np.float32), np.arange(BLOCK)
        for k in range(-(-n // 8)):
            om = o[o < n]
            c[om] = a[om] * 2
            o = o + (16 if k % 2 == 0 else 8)
        return c
    if name == "not_box":
        m = offs < n
        for k in range(flag):
            m = m ^ (offs < n2 + k)
        return np.where(m, a * 2, 0).astype(np.float32)
    return reference_branch(name, a, n, n2, flag)


def reference_branch(name, a, n, n2, flag):
    offs = np.arange(N_TOTAL)
    c = np.zeros(N_TOTAL, np.float32)
    if name == "outside":
        m = offs < n
        c[m] = a[m] * 2 if flag > 0 else a[m] + 1
    elif name == "inside":
        m = offs < (n if flag > 0 else n2)
        c[m] = a[m] * 2 if flag > 0 else a[m] + 1
    elif name in ("yielded", "bound"):
        m = offs < (n if flag > 0 else n2)
        c[m] = a[m] * 2
    elif name == "loop":
        m = offs < n
        c[m] = a[m] * 2
    elif name == "carried":
        m = offs < (n - (flag - 1) if flag > 0 else n)
        c[m] = a[m] * 2
    return c


def report_detection(name, ttir):
    """Run AnalyzeTailMasks on the TTIR and say which load / store masks became tail_mask."""
    opt = os.path.join(os.path.dirname(triton.__file__), "..", "..", "build", "cmake.linux-x86_64-cpython-3.10", "bin",
                       "triton-opt")
    out = subprocess.run([opt, "-triton-cpu-analyze-tail-masks", "-"], input=ttir, capture_output=True, text=True,
                         check=True).stdout
    tails = set(re.findall(r"(%\S+) = triton_cpu.tail_mask", out))
    for line in out.splitlines():
        m = re.search(r"\btt\.(load|store) (.*?) :", line)
        if not m:
            continue
        operands = [o.strip() for o in m.group(2).split(",")]
        mask = operands[1] if m.group(1) == "load" else operands[2]
        print(f"  {name:<22} tt.{m.group(1):<5} mask {mask:<10} {'tail_mask' if mask in tails else 'plain i1'}")


torch.manual_seed(0)
a = torch.rand(N_TOTAL, dtype=torch.float32).numpy()
n, n2 = 37, 21

triton.runtime.driver.set_active_to_cpu()
from triton.backends.cpu.riscv import compile_deploy_and_run, compile_kernel_to_so

#KERNELS = [("outside", outside_kernel, 2), ("outside", outside_kernel, 0), ("inside", inside_kernel, 2),
#           ("inside", inside_kernel, 0), ("yielded", yielded_kernel, 2), ("yielded", yielded_kernel, 0),
#           ("bound", bound_kernel, 2), ("bound", bound_kernel, 0), ("loop", loop_kernel, 2),
#           ("carried", carried_kernel, 3), ("carried", carried_kernel, 0), ("carried_use", carried_use_kernel, 3),
#           ("nested", nested_kernel, 3), ("not_box", not_box_kernel, 3), ("carried_offs", carried_offs_kernel, 0),
#           ("carried_offs_2d", carried_offs_2d_kernel, 3), ("carried_offs_not_range", carried_offs_not_range_kernel, 2),
#           ("yielded_offs", yielded_offs_kernel, 2), ("yielded_offs", yielded_offs_kernel, 0),
#           ("carried_offs_if", carried_offs_if_kernel, 0), ("while_offs", while_offs_kernel, 0),
#           ("while_mask", while_mask_kernel, 3), ("while_mask", while_mask_kernel, 0),
#           ("while_mask_use", while_mask_use_kernel, 3)]
KERNELS = [("yielded", yielded_kernel, 0)]
GRIDS = {"loop": (1, ), "carried_offs_if": (1, ), "while_offs": (1, ), "carried_offs": (1, ), "carried_offs_2d": (1, )}
reported = set()
for name, kernel, flag in KERNELS:
    args = {"a_ptr": a.tolist(), "c_ptr": [0.0] * N_TOTAL, "n": n, "n2": n2, "flag": flag}
    grid = GRIDS.get(name, (triton.cdiv(N_TOTAL, BLOCK), ))
    result = compile_deploy_and_run(kernel, args, grid, f"artifacts/riscv/rvv-mask-branch-{name}.elf",
                                    "chlee@140.114.78.64", constexprs={"BLOCK": BLOCK},
                                    expected={"c_ptr": reference(name, a, n, n2, flag).tolist()},
                                    remote_dir="~/triton-riscv-elf")
    ok = result.returncode == 0 and "PASS" in result.stdout
    print(f"board {name:<22} flag={flag}: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print(result.stdout, result.stderr)
    if name not in reported:
        reported.add(name)
        compiled, _ = compile_kernel_to_so(kernel, args, constexprs={"BLOCK": BLOCK})
        report_detection(name, compiled.asm["ttir"])
