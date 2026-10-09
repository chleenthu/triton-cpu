"""
Fused Softmax (RISC-V ELF)
==========================

The row-wise softmax kernel of 02-fused-softmax.py, compiled for riscv64,
deployed to the board and checked against torch.softmax there.

The matrix has an irregular number of columns, so the masked load/store
padding of each row is exercised.
"""

import torch

import triton
import triton.language as tl


@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # The rows of the softmax are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    # Subtract maximum for numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)


# %%
# Unit Test
# ---------
#
# A matrix with an irregular number of rows and columns, so the padding of
# each row to BLOCK_SIZE is tested.

torch.manual_seed(0)
triton.runtime.driver.set_active_to_cpu()

n_rows, n_cols = 67, 781
x = torch.randn(n_rows, n_cols, device='cpu')
BLOCK_SIZE = triton.next_power_of_2(n_cols)

from triton.backends.cpu.riscv import compile_deploy_and_run

RISCV_HOST = "chlee@140.114.78.64"
RISCV_REMOTE_DIR = "~/triton-riscv-elf"


def run_on_board(kernel, constexprs, name):
    torch_output = torch.softmax(x, axis=1)
    arguments = {
        "output_ptr": [0.0] * (n_rows * n_cols), "input_ptr": x.flatten().tolist(),  #
        "input_row_stride": x.stride(0), "output_row_stride": x.stride(0), "n_cols": n_cols,
    }
    grid = (n_rows, )
    result = compile_deploy_and_run(kernel, arguments, grid, f"artifacts/riscv/{name}.elf", RISCV_HOST,
                                     constexprs=constexprs, expected={"output_ptr": torch_output.flatten().tolist()},
                                     atol=1e-5, remote_dir=RISCV_REMOTE_DIR)
    print(f"{name}:")
    print(result.stdout, end="")
    print(result.stderr, end="")


run_on_board(softmax_kernel, {"BLOCK_SIZE": BLOCK_SIZE}, "rvv-softmax")
