"""
Low-Memory Dropout (RISC-V ELF)
===============================

The two kernels of python/tutorials/04-low-memory-dropout.py, compiled for
riscv64, deployed to the board and checked there:

* _dropout: dropout with an explicit keep mask,
* _seeded_dropout: dropout whose keep mask comes from tl.rand(seed, offsets).

tl.rand is Triton's Philox generator, so the reference for the seeded kernel
reimplements it (triton/language/random.py: philox with 10 rounds on 32-bit
lanes, then uint_to_uniform_float) and the keep decisions must match exactly.
"""

import numpy as np
import torch

import triton
import triton.language as tl


@triton.jit
def _dropout(
    x_ptr,  # pointer to the input
    x_keep_ptr,  # pointer to a mask of 0s and 1s
    output_ptr,  # pointer to the output
    n_elements,  # number of elements in the `x` tensor
    p,  # probability that an element of `x` is changed to zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    # The line below is the crucial part, described in the paragraph above!
    output = tl.where(x_keep, x / (1 - p), 0.0)
    # Write-back output
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def _seeded_dropout(
    x_ptr,
    output_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    # compute memory offsets of elements handled by this instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # load data from x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # randomly prune it
    random = tl.rand(seed, offsets)
    x_keep = random > p
    # write-back
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


# %%
# Reference for tl.rand
# ---------------------


def philox_rand(seed: int, offsets: np.ndarray, n_rounds: int = 10) -> np.ndarray:
    """tl.rand(seed, offsets) for int32 offsets and a seed below 2**32."""
    mask32 = np.uint64(0xFFFFFFFF)
    c0 = offsets.astype(np.uint64) & mask32
    c1 = np.zeros_like(c0)
    c2 = np.zeros_like(c0)
    c3 = np.zeros_like(c0)
    k0 = np.uint64(seed & 0xFFFFFFFF)
    k1 = np.uint64((seed >> 32) & 0xFFFFFFFF)
    A, B = np.uint64(0xD2511F53), np.uint64(0xCD9E8D57)
    for _ in range(n_rounds):
        _c0, _c2 = c0, c2
        c0 = ((B * _c2) >> np.uint64(32)) ^ c1 ^ k0
        c2 = ((A * _c0) >> np.uint64(32)) ^ c3 ^ k1
        c1 = (B * _c2) & mask32
        c3 = (A * _c0) & mask32
        k0 = (k0 + np.uint64(0x9E3779B9)) & mask32
        k1 = (k1 + np.uint64(0xBB67AE85)) & mask32
    # uint_to_uniform_float on 32 bits: bitcast to int32, x < 0 -> ~x, x * scale.
    x = c0.astype(np.uint32).view(np.int32)
    x = np.where(x < 0, ~x, x)
    return x.astype(np.float32) * np.float32(4.6566127342e-10)


def dropout_reference(x: np.ndarray, keep: np.ndarray, p: float) -> np.ndarray:
    x = x.astype(np.float32)
    return np.where(keep, x / (np.float32(1) - np.float32(p)), np.float32(0)).astype(np.float32)


# %%
# Unit Test
# ---------

torch.manual_seed(0)
triton.runtime.driver.set_active_to_cpu()

n_elements = 1000  # not a multiple of BLOCK_SIZE, so the last block is masked
BLOCK_SIZE = 256
p = 0.5
x = torch.randn(n_elements, dtype=torch.float32)
x_keep = (torch.rand(n_elements) > p).to(torch.int32)

from triton.backends.cpu.riscv import compile_deploy_and_run

RISCV_HOST = "chlee@140.114.78.64"
RISCV_REMOTE_DIR = "~/triton-riscv-elf"
grid = (triton.cdiv(n_elements, BLOCK_SIZE), )


def run_on_board(kernel, arguments, signature, expected, name):
    result = compile_deploy_and_run(kernel, arguments, grid, f"artifacts/riscv/{name}.elf", RISCV_HOST,
                                     constexprs={"BLOCK_SIZE": BLOCK_SIZE}, signature=signature, expected=expected,
                                     atol=1e-6, remote_dir=RISCV_REMOTE_DIR)
    print(f"{name}:")
    print(result.stdout, end="")
    print(result.stderr, end="")


run_on_board(
    _dropout, {
        "x_ptr": x.tolist(), "x_keep_ptr": x_keep.tolist(), "output_ptr": [0.0] * n_elements,  #
        "n_elements": n_elements, "p": p,
    }, {"x_keep_ptr": "*i32"}, {"output_ptr": dropout_reference(x.numpy(), x_keep.numpy() != 0, p).tolist()},
    "rvv-dropout")

for seed in (123, 512):
    keep = philox_rand(seed, np.arange(n_elements, dtype=np.int32)) > np.float32(p)
    run_on_board(
        _seeded_dropout, {
            "x_ptr": x.tolist(), "output_ptr": [0.0] * n_elements,  #
            "n_elements": n_elements, "p": p, "seed": seed,
        }, None, {"output_ptr": dropout_reference(x.numpy(), keep, p).tolist()}, f"rvv-seeded-dropout-{seed}")
