"""
Reverse Remat Paper, Fig. 4 / 9 Sequence (M=8, f32)
===================================================
Bahi & Eisenbeis, "Register Reverse Rematerialization" Fig. 4 (IJPP 2014 Fig. 9):
    for (i = 0; i < M-1; i++) A[i+1] = A[i] + (i+1);
    B = A[M-1];
    for (i = M-2; i >= 0; i--) B = B * A[i];
fully unrolled with M = 8. Without remat all M values are live at once; reverse
remat recomputes A[i] = A[i+1] - (i+1) on the way back, so only 2 are.
Vector version: one lane per element (as one GPU thread per lattice site in
the paper), BLOCK_SIZE = 64 f32/i32 = one LMUL8 register group per value at
VLEN 256, so every value in the paper's DAG costs 8 of the 32 vector registers.
Inputs are small integers stored as FP32, so +/-k and *2 or /2 are exact and
the reversed values are bit-identical. FP32 rather than int32: LLVM's
middle-end folds an int32 chain A[i+1] = A[i] + (i+1) into A[i] = A[0] + c_i
and reassociates the product into a tree, which leaves every A[i] with a
single use and nothing to rematerialize; float adds and multiplies are not
reassociated (no fast-math), so the chain reaches the backend as written.
"""

import torch

import triton
import triton.language as tl

CPU_BLOCK_SIZE = 64

@triton.jit
def paper_seq8_f32_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements
    a = tl.load(a_ptr + idx, mask=mask)
    a0 = a
    a1 = a0 + 1
    a2 = a1 + 2
    a3 = a2 + 3
    a4 = a3 + 4
    a5 = a4 + 5
    a6 = a5 + 6
    a7 = a6 + 7
    p = a7
    p = p * a6
    p = p * a5
    p = p * a4
    p = p * a3
    p = p * a2
    p = p * a1
    p = p * a0
    tl.store(output_ptr + idx, p, mask=mask)


torch.manual_seed(0)
size = 98432
triton.runtime.driver.set_active_to_cpu()
a = torch.randint(0, 10, (size, ), dtype=torch.int32).to(torch.float32)
from triton.backends.cpu.riscv import compile_deploy_and_run

RISCV_HOST = "chlee@140.114.78.64"
RISCV_REMOTE_DIR = "~/triton-riscv-elf"


def reference():
    a0 = a
    a1 = a0 + 1
    a2 = a1 + 2
    a3 = a2 + 3
    a4 = a3 + 4
    a5 = a4 + 5
    a6 = a5 + 6
    a7 = a6 + 7
    p = a7
    p = p * a6
    p = p * a5
    p = p * a4
    p = p * a3
    p = p * a2
    p = p * a1
    p = p * a0
    return p


def run_on_board(kernel, constexprs, name):
    expected = reference()
    arguments = {
        "a_ptr": a.tolist(),
        "output_ptr": [0.0] * size,
        "n_elements": size,
    }
    grid = (triton.cdiv(size, CPU_BLOCK_SIZE), )
    result = compile_deploy_and_run(kernel, arguments, grid, f"artifacts/riscv/{name}.elf", RISCV_HOST,
                                     constexprs=constexprs, expected={"output_ptr": expected.tolist()},
                                     remote_dir=RISCV_REMOTE_DIR)
    print(f"{name}:")
    print(result.stdout, end="")
    print(result.stderr, end="")


run_on_board(paper_seq8_f32_kernel, {"BLOCK_SIZE": CPU_BLOCK_SIZE}, "rvv-paper-seq8-f32")
