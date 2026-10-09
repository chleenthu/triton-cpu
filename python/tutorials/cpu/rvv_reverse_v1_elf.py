"""
Reverse Dependency Chain
========================

This kernel implements the "reverse" micro-benchmark from ~/tsvc/reverse.txt:

    offset = pid * BLOCK_SIZE
    %33 = offset + tl.arange(0, BLOCK_SIZE)
    D = %33 < n_elements
    A = tl.load(a, mask=D)
    B = tl.load(b, mask=D)
    C = A + B
    F = C * scalar
    I = C + F
    J = I - B
    K = A + J
    tl.store(K, k, mask=D)

It is a long dependency chain (A,B -> C -> F -> I -> J -> K) used to exercise
register allocation / scheduling on the RVV backend, matching the register
table in ~/tsvc/"TSVC - reverse.csv".
"""

import torch

import triton
import triton.language as tl

CPU_BLOCK_SIZE = 128


@triton.jit
def reverse_kernel(a_ptr,  # *Pointer* to first input vector.
                    b_ptr,  # *Pointer* to second input vector.
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
    b = tl.load(b_ptr + idx, mask=mask)
    c = a + b
    f = c * scalar
    i = c + f
    j = i - b
    k = a + j
    tl.store(output_ptr + idx, k, mask=mask)


# %%
# Let's use the above kernel and test its correctness against a plain torch/python
# reference implementation of the same dependency chain:
torch.manual_seed(0)
size = 98432
scalar = 2.0
triton.runtime.driver.set_active_to_cpu()
a = torch.rand(size, device=torch.device('cpu'))
b = torch.rand(size, device=torch.device('cpu'))
from triton.backends.cpu.riscv import compile_deploy_and_run

RISCV_HOST = "chlee@140.114.78.64"
RISCV_REMOTE_DIR = "~/triton-riscv-elf"


def run_on_board(kernel, constexprs, name):
    c_ref = a + b
    f_ref = c_ref * scalar
    i_ref = c_ref + f_ref
    j_ref = i_ref - b
    expected = a + j_ref
    arguments = {
        "a_ptr": a.tolist(),
        "b_ptr": b.tolist(),
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


run_on_board(reverse_kernel, {"BLOCK_SIZE": CPU_BLOCK_SIZE}, "rvv-reverse")
