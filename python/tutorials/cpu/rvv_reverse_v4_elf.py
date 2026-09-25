"""
Reverse Dependency Chain v4
============================

This kernel implements the "reverse_v4" micro-benchmark from ~/tsvc/reverse_v4.txt:

    offset = pid * BLOCK_SIZE
    %33 = offset + tl.arange(0, BLOCK_SIZE)
    mask = %33 < n_elements
    A = tl.load(a, mask=mask)
    B = A + 5
    C = B << 3
    D = C * scalar
    E = D >> 1 (srli)
    F = D >> 1 (srai)
    G = E | F
    H = G + D
    I = H & C
    J = I - B
    K = J - A
    tl.store(K, k, mask=mask)

Like reverse_v3.txt, the whole chain is integer end to end and typed tl.uint32 via a `signature=`
override, so E's ">> 1" is a logical shift (lshr / srli) for free. But this time D is shifted by the
*same* amount two different ways: E wants srli (unsigned/logical) and F wants srai (signed/
arithmetic) -- the same bit pattern, two different shifts, which a single dtype can't give both of
at once (Triton's `>>` picks lshr or ashr from the *operand's* signedness -- see core.py's
`tensor.__rshift__`). So F re-derives its own signed view of D via an int32 bitcast round-trip
(`d.to(tl.int32, bitcast=True) >> 1`, then bitcast back to tl.uint32 to rejoin the unsigned chain at
G), the same technique reverse_v2.txt used to splice a differently-typed op into an otherwise
single-dtype chain. Used to exercise register allocation / scheduling on the RVV backend, matching
the register table in ~/tsvc/"TSVC - reverse.csv".
"""

import numpy as np
import torch

import triton
import triton.language as tl

CPU_BLOCK_SIZE = 128


@triton.jit
def reverse_v4_kernel(a_ptr,  # *Pointer* to input vector (tl.uint32).
                       output_ptr,  # *Pointer* to output vector (tl.uint32).
                       n_elements,  # Size of the vectors.
                       scalar,  # Scalar multiplier (tl.uint32).
                       BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
                       ):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE
    idx = offset + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements

    a = tl.load(a_ptr + idx, mask=mask)
    b = a + 5
    c = b << 3
    d = c * scalar
    e = d >> 1  # a/d/scalar are tl.uint32, so this is a logical shift right (srli).
    f_signed = d.to(tl.int32, bitcast=True) >> 1  # signed view of the same bits -> arithmetic shift (srai).
    f = f_signed.to(tl.uint32, bitcast=True)
    g = e | f
    h = g + d
    i = h & c
    j = i - b
    k = j - a
    tl.store(output_ptr + idx, k, mask=mask)


# %%
# Let's use the above kernel and test its correctness against a plain numpy uint32 reference
# implementation of the same dependency chain (numpy gives exact, wraparound 32-bit unsigned
# arithmetic; the int32 `.view()` for F reproduces the kernel's signed-bitcast round-trip, giving
# an arithmetic ">>" matching the kernel's srai bit-for-bit):
torch.manual_seed(0)
size = 98432
scalar = 3
triton.runtime.driver.set_active_to_cpu()
a = torch.randint(0, 2**31, (size, ), dtype=torch.int64)
a_list = a.tolist()

SIGNATURE = {"a_ptr": "*u32", "output_ptr": "*u32", "scalar": "u32"}

from triton.backends.cpu.riscv import compile_deploy_and_run

RISCV_HOST = "chlee@140.114.78.64"
RISCV_REMOTE_DIR = "~/triton-riscv-elf"


def run_on_board(kernel, constexprs, name):
    au = np.array(a_list, dtype=np.uint32)
    b_ref = (au + np.uint32(5)).astype(np.uint32)
    c_ref = (b_ref << np.uint32(3)).astype(np.uint32)
    d_ref = (c_ref * np.uint32(scalar)).astype(np.uint32)
    e_ref = (d_ref >> np.uint32(1)).astype(np.uint32)  # numpy >> on uint32 is a logical shift.
    f_ref = (d_ref.view(np.int32) >> 1).view(np.uint32).astype(np.uint32)  # signed view -> arithmetic shift.
    g_ref = (e_ref | f_ref).astype(np.uint32)
    h_ref = (g_ref + d_ref).astype(np.uint32)
    i_ref = (h_ref & c_ref).astype(np.uint32)
    j_ref = (i_ref - b_ref).astype(np.uint32)
    expected = (j_ref - au).astype(np.uint32)

    arguments = {
        "a_ptr": a_list,
        "output_ptr": [0] * size,
        "n_elements": size,
        "scalar": scalar,
    }
    grid = (triton.cdiv(size, CPU_BLOCK_SIZE), )
    result = compile_deploy_and_run(kernel, arguments, grid, f"artifacts/riscv/{name}.elf", RISCV_HOST,
                                     constexprs=constexprs, signature=SIGNATURE,
                                     expected={"output_ptr": expected.tolist()}, remote_dir=RISCV_REMOTE_DIR)
    print(f"{name}:")
    print(result.stdout, end="")
    print(result.stderr, end="")


run_on_board(reverse_v4_kernel, {"BLOCK_SIZE": CPU_BLOCK_SIZE}, "rvv-reverse-v4")
