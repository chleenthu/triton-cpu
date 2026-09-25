"""
Reverse Dependency Chain v2
============================

This kernel implements the "reverse_v2" micro-benchmark from ~/tsvc/reverse_v2.txt:

    offset = pid * BLOCK_SIZE
    %33 = offset + tl.arange(0, BLOCK_SIZE)
    mask = %33 < n_elements
    A = tl.load(a, mask=mask)
    B = A + 5
    C = B - 3
    D = C * scalar
    E = D + 1
    F = D & 0xFFFFFFFF
    G = E + F
    H = G + D
    I = H + C
    J = I - B
    K = J + A
    tl.store(K, k, mask=mask)

A single-input, longer dependency chain than reverse.txt (A -> B -> C -> D -> E,F -> G -> H -> I -> J -> K),
used to exercise register allocation / scheduling on the RVV backend, matching the register table in
~/tsvc/"TSVC - reverse.csv".

`D & 0xFFFFFFFF` is a bitwise AND on a float32 value; Triton requires integer operands for `&`, so F is
computed by bitcasting D to int32, ANDing with the all-ones 32-bit mask (`-1`, the same bit pattern as
0xFFFFFFFF in signed int32 -- an identity on the bit pattern), and bitcasting back to float32 -- i.e.
F == D numerically, while still exercising the int/float bitcast + integer ALU op on the RVV backend.
"""

import torch

import triton
import triton.language as tl

CPU_BLOCK_SIZE = 128


@triton.jit
def reverse_v2_kernel(a_ptr,  # *Pointer* to input vector.
                       output_ptr,  # *Pointer* to output vector.
                       n_elements,  # Size of the vectors.
                       scalar,  # Scalar multiplier.
                       BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
                       ):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE
    idx = offset + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements

    a = tl.load(a_ptr + idx, mask=mask)
    b = a + 5
    c = b - 3
    d = c * scalar
    e = d + 1
    # 0xFFFFFFFF does not fit in signed int32; -1 has the identical all-ones bit pattern.
    f = (d.to(tl.int32, bitcast=True) & -1).to(tl.float32, bitcast=True)
    g = e + f
    h = g + d
    i = h + c
    j = i - b
    k = j + a
    tl.store(output_ptr + idx, k, mask=mask)


# %%
# Let's use the above kernel and test its correctness against a plain torch/python
# reference implementation of the same dependency chain:
torch.manual_seed(0)
size = 98432
scalar = 2.0
triton.runtime.driver.set_active_to_cpu()
a = torch.rand(size, device=torch.device('cpu'))
from triton.backends.cpu.riscv import compile_deploy_and_run

RISCV_HOST = "chlee@140.114.78.64"
RISCV_REMOTE_DIR = "~/triton-riscv-elf"


def run_on_board(kernel, constexprs, name):
    b_ref = a + 5
    c_ref = b_ref - 3
    d_ref = c_ref * scalar
    e_ref = d_ref + 1
    f_ref = d_ref  # AND with 0xFFFFFFFF is a bit-pattern identity on float32.
    g_ref = e_ref + f_ref
    h_ref = g_ref + d_ref
    i_ref = h_ref + c_ref
    j_ref = i_ref - b_ref
    expected = j_ref + a
    arguments = {
        "a_ptr": a.tolist(),
        "output_ptr": [0.0] * size,
        "n_elements": size,
        "scalar": scalar,
    }
    grid = (triton.cdiv(size, CPU_BLOCK_SIZE), )
    result = compile_deploy_and_run(kernel, arguments, grid, f"artifacts/riscv/{name}.elf", RISCV_HOST,
                                     constexprs=constexprs, expected={"output_ptr": expected.tolist()},
                                     remote_dir=RISCV_REMOTE_DIR)
    print(f"{name}:")
    print(result.stdout, end="")
    print(result.stderr, end="")


run_on_board(reverse_v2_kernel, {"BLOCK_SIZE": CPU_BLOCK_SIZE}, "rvv-reverse-v2")
