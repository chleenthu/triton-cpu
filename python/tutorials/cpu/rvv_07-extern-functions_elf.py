"""
Libdevice (`tl.extra.libdevice`) function (RISC-V ELF)
======================================================

The asin kernel of python/tutorials/07-extern-functions.py, compiled for
riscv64, deployed to the board and checked there against torch.asin.

On the CPU backend, triton.language.extra.libdevice resolves to the CPU
implementation (triton/language/extra/cpu/libdevice.py), so no extern_libs
(the CUDA libdevice / ROCm ocml bitcode of the GPU tutorial) are passed.
"""

import torch

import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def asin_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x = libdevice.asin(x)
    tl.store(y_ptr + offsets, x, mask=mask)


# %%
# Unit Test
# ---------

torch.manual_seed(0)
triton.runtime.driver.set_active_to_cpu()

size = 9843  # not a multiple of BLOCK_SIZE, so the last block is masked
BLOCK_SIZE = 1024
x = torch.rand(size)

from triton.backends.cpu.riscv import compile_deploy_and_run

RISCV_HOST = "chlee@140.114.78.64"
RISCV_REMOTE_DIR = "~/triton-riscv-elf"


def run_on_board(kernel, constexprs, name):
    output_torch = torch.asin(x)
    arguments = {"x_ptr": x.tolist(), "y_ptr": [0.0] * size, "n_elements": size}
    grid = (triton.cdiv(size, BLOCK_SIZE), )
    result = compile_deploy_and_run(kernel, arguments, grid, f"artifacts/riscv/{name}.elf", RISCV_HOST,
                                     constexprs=constexprs, expected={"y_ptr": output_torch.tolist()}, atol=1e-6,
                                     remote_dir=RISCV_REMOTE_DIR)
    print(f"{name}:")
    print(result.stdout, end="")
    print(result.stderr, end="")


run_on_board(asin_kernel, {"BLOCK_SIZE": BLOCK_SIZE}, "rvv-asin")
