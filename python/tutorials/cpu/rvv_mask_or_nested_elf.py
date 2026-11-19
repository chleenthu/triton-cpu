"""
OR of nested boxes
==================

A 2D `C = A + B` whose mask is the OR of two boxes, one inside the other:

    inner = (rm[:, None] < M) & (rn[None, :] < N)      # box [M - rm, N - rn]
    outer = rm[:, None] <= M                            # box [M - rm + 1, BLOCK_N]
    mask  = inner | outer                               # = outer

The two boxes differ in both axes, so AnalyzeTailMasks used to give up on the OR. The containment
check (provablyLE in AnalyzeTailMasks.cpp) proves M - rm <= M - rm + 1 and N - rn <= BLOCK_N, so the
mask becomes triton_cpu.tail_mask of the outer box. With TRITON_VSETVL_MINE=1 the loads and the
store are then tail-masked (vsetvli) accesses. The kernel stays in bounds: rows up to M (inclusive)
exist because the matrices are allocated with M + 1 rows.
"""

import numpy as np
import torch

import triton
import triton.language as tl

BLOCK_M = 8
BLOCK_N = 16


@triton.jit
def mask_or_nested_kernel(a_ptr, b_ptr, c_ptr, M, N, stride_m, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    inner = (rm[:, None] < M) & (rn[None, :] < N)
    outer = rm[:, None] <= M
    mask = inner | outer
    offs = rm[:, None] * stride_m + rn[None, :]
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    tl.store(c_ptr + offs, a + b, mask=mask)


torch.manual_seed(0)
M, N = 21, 11  # rows 0..M are on (M + 1 rows); every column of the 16-wide tile is on
ROWS = triton.cdiv(M + 1, BLOCK_M) * BLOCK_M
a = torch.rand((ROWS, BLOCK_N), dtype=torch.float32)
b = torch.rand((ROWS, BLOCK_N), dtype=torch.float32)
expected = np.zeros((ROWS, BLOCK_N), np.float32)
expected[:M + 1] = (a.numpy() + b.numpy())[:M + 1]

triton.runtime.driver.set_active_to_cpu()
from triton.backends.cpu.riscv import compile_deploy_and_run

result = compile_deploy_and_run(
    mask_or_nested_kernel, {
        "a_ptr": a.flatten().tolist(), "b_ptr": b.flatten().tolist(), "c_ptr": [0.0] * (ROWS * BLOCK_N), "M": M, "N":
        N, "stride_m": BLOCK_N
    }, (triton.cdiv(M + 1, BLOCK_M), ), "artifacts/riscv/rvv-mask-or-nested.elf", "chlee@140.114.78.64",
    constexprs={"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N}, expected={"c_ptr": expected.flatten().tolist()},
    remote_dir="~/triton-riscv-elf")
print("rvv-mask-or-nested:")
print(result.stdout, end="")
print(result.stderr, end="")
