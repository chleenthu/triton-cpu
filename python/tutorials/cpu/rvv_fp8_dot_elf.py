"""
FP8 Dot (RISC-V ELF)
====================

FP8 kernels compiled for riscv64, deployed to the board and checked there, for
e4m3 (fp8e4nv) and e5m2 (fp8e5):

* fp8_matmul_kernel: tl.dot on FP8 inputs with an FP32 accumulator,
* fp8_scale_kernel: an FP8 output, y = fp8(x * scale).

The board runner stores FP8 buffers as bytes (generate_runner in
third_party/cpu/backend/riscv.py): the arguments are ordinary float values,
encoded to FP8 on the host, and FP8 outputs are decoded on the board and
compared with the expected values rounded to the same FP8 format.

There are no FP8 instructions in RVV: the kernels convert FP8 to FP32 and
compute in FP32.
"""

import sys

import numpy as np
import torch

import triton
import triton.language as tl


@triton.jit
def fp8_matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,  #
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)  # FP8
        b = tl.load(b_ptrs)  # FP8
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)


@triton.jit
def fp8_scale_kernel(x_ptr, y_ptr, n_elements, scale, BLOCK_SIZE: tl.constexpr):
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    tl.store(y_ptr + offs, (x * scale).to(y_ptr.dtype.element_ty), mask=mask)


triton.runtime.driver.set_active_to_cpu()
from triton.backends.cpu.riscv import compile_deploy_and_run

RISCV_HOST = "chlee@140.114.78.64"
RISCV_REMOTE_DIR = "~/triton-riscv-elf"

M, N, K = 64, 64, 128
BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 64
FORMATS = {"e4m3": ("fp8e4nv", torch.float8_e4m3fn), "e5m2": ("fp8e5", torch.float8_e5m2)}


def report(name, result):
    print(f"{name}:")
    print(result.stdout, end="")
    print(result.stderr, end="")


for fmt in sys.argv[1:] or list(FORMATS):
    ty, torch_dtype = FORMATS[fmt]
    rng = np.random.default_rng(0)
    # Values that are exact in FP8, so the inputs are what the reference uses.
    a = torch.from_numpy(rng.standard_normal((M, K)).astype(np.float32)).to(torch_dtype)
    b = torch.from_numpy(rng.standard_normal((K, N)).astype(np.float32)).to(torch_dtype)
    c_ref = a.double() @ b.double()
    # FP8 x FP8 products are exact in FP32; only the order of the FP32 sum differs.
    atol = float(c_ref.abs().max()) * 2.0**-20
    result = compile_deploy_and_run(
        fp8_matmul_kernel, {
            "a_ptr": a.float().flatten().tolist(), "b_ptr": b.float().flatten().tolist(),
            "c_ptr": [0.0] * (M * N), "M": M, "N": N, "K": K,
        }, (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), ), f"artifacts/riscv/fp8-dot-{fmt}.elf",
        RISCV_HOST, constexprs={"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K},
        signature={"a_ptr": f"*{ty}", "b_ptr": f"*{ty}", "c_ptr": "*fp32"}, expected={"c_ptr": c_ref.flatten().tolist()},
        atol=atol, remote_dir=RISCV_REMOTE_DIR)
    report(f"fp8-dot-{fmt} (atol {atol:.3g})", result)

    # FP8 output: y = fp8(x * scale). The runner rounds the expected values to the
    # same FP8 format, so the check is exact.
    n, scale = 1000, 1.5
    x = torch.from_numpy(rng.standard_normal(n).astype(np.float32)).to(torch_dtype).float()
    result = compile_deploy_and_run(
        fp8_scale_kernel, {"x_ptr": x.tolist(), "y_ptr": [0.0] * n, "n_elements": n, "scale": scale},
        (triton.cdiv(n, 256), ), f"artifacts/riscv/fp8-scale-{fmt}.elf", RISCV_HOST,
        constexprs={"BLOCK_SIZE": 256}, signature={"x_ptr": f"*{ty}", "y_ptr": f"*{ty}", "scale": "fp32"},
        expected={"y_ptr": (x * scale).tolist()}, atol=0.0, remote_dir=RISCV_REMOTE_DIR)
    report(f"fp8-scale-{fmt}", result)
