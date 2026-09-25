"""
Reverse Dependency Chain v3
============================

This kernel implements the "reverse_v3" micro-benchmark from ~/tsvc/reverse_v3.txt:

    offset = pid * BLOCK_SIZE
    %33 = offset + tl.arange(0, BLOCK_SIZE)
    mask = %33 < n_elements
    A = tl.load(a, mask=mask)
    B = A + 5
    C = B << 2
    D = C * scalar
    E = D >> 1 (srli)
    F = D & 0xFFFFFFFF
    G = E | F
    H = G + D
    I = H & C
    J = I - B
    K = J - A
    tl.store(K, k, mask=mask)

Unlike reverse.txt/reverse_v2.txt (a float32 dependency chain, with one bitwise AND spliced in via
an int32 bitcast round-trip), this chain is genuinely integer end to end: it mixes shifts (<<, >>)
with &, |, +, - and *, and the source explicitly calls out ">> 1" as "(srli)" -- a *logical* (unsigned)
right shift, not the arithmetic (sign-extending) shift a signed type would get. So the whole chain
(A..K) is typed tl.uint32 via an explicit `signature=` override (Triton's `>>` lowers to lshr for
unsigned dtypes and ashr for signed ones -- see core.py's `tensor.__rshift__`), which makes the
shift, and every bitwise op, well-defined without any bitcast tricks; `D & 0xFFFFFFFF` is then a
literal all-ones mask on an already-32-bit value (an identity: F == D), exercising the AND
instruction the same way reverse_v2.txt's bitcast trick did, but for the natural reason this time.
Used to exercise register allocation / scheduling on the RVV backend, matching the register table
in ~/tsvc/"TSVC - reverse.csv".
"""

import numpy as np
import torch

import triton
import triton.language as tl

CPU_BLOCK_SIZE = 128


@triton.jit
def reverse_v3_kernel(a_ptr,  # *Pointer* to input vector (tl.uint32).
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
    c = b << 2
    d = c * scalar
    e = d >> 1  # a/d/scalar are tl.uint32, so this is a logical shift right (srli).
    f = d & 0xFFFFFFFF  # identity: an all-ones mask on an already-32-bit value.
    g = e | f
    h = g + d
    i = h & c
    j = i - b
    k = j - a
    tl.store(output_ptr + idx, k, mask=mask)


# %%
# Let's use the above kernel and test its correctness against a plain numpy uint32 reference
# implementation of the same dependency chain (numpy gives exact, wraparound 32-bit unsigned
# arithmetic and a logical ">>", matching the kernel's tl.uint32 chain bit-for-bit):
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
    c_ref = (b_ref << np.uint32(2)).astype(np.uint32)
    d_ref = (c_ref * np.uint32(scalar)).astype(np.uint32)
    e_ref = (d_ref >> np.uint32(1)).astype(np.uint32)  # numpy >> on uint32 is a logical shift.
    f_ref = (d_ref & np.uint32(0xFFFFFFFF)).astype(np.uint32)
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


run_on_board(reverse_v3_kernel, {"BLOCK_SIZE": CPU_BLOCK_SIZE}, "rvv-reverse-v3")
