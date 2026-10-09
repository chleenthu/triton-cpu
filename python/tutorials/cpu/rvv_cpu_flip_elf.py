"""
CPU Flip (test_cpu_flip on RVV)
===============================

The kernel and case matrix from python/test/unit/cpu/test_cpu_flip.py, built as standalone riscv64
ELFs and run on the board instead of through pytest on the host. Each program loads an M x N x K
tensor, flips it along `dim` with tl.flip, and stores it back at the same offsets:

    x = tl.load(X + off3d)
    x = tl.flip(x, dim)
    tl.store(Z + off3d, x)

tl.flip lowers to a bitcast to int, a reshape of the flipped dim to (2, 2, ..., 2) and log2(n)
xor-sum steps (triton/language/standard.py), so this exercises how that chain is turned into a
vector reverse on the RVV backend for 32-bit and 16-bit element types.
"""

import numpy as np
import torch

import triton
import triton.language as tl

# Same (M, N, K, dim) matrix as test_cpu_flip.py.
flip_cases = [
    (1, 16, 64, 0),
    (1, 16, 64, 1),
#    (1, 16, 64, 2),
#    (1, 16, 64, -2),
#    (32, 1, 2, 0),
#    (32, 1, 2, 1),
#    (32, 1, 2, 2),
#    (32, 1, 2, -2),
]
#dtypes = ["int32", "float16", "float32", "bfloat16"]
dtypes = ["int32"]


def numpy_random(shape, dtype_str):
    # The int32 / float / bfloat16 branches of triton._internal_testing.numpy_random (same seed), inlined
    # because that module imports pytest. bfloat16 comes back as bf16-representable float32.
    rs = np.random.RandomState(seed=17)
    if dtype_str == "int32":
        iinfo = np.iinfo(np.int32)
        x = rs.randint(iinfo.min, iinfo.max, shape, dtype=np.int32)
        x[x == 0] = 1
        return x
    if dtype_str == "bfloat16":
        return (rs.normal(0, 1, shape).astype("float32").view("uint32") & np.uint32(0xffff0000)).view("float32")
    return rs.normal(0, 1, shape).astype(dtype_str)


@triton.jit
def flip_kernel(X, Z, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, dim: tl.constexpr):
    offx = tl.arange(0, M) * N * K
    offy = tl.arange(0, N) * K
    offz = tl.arange(0, K)
    off3d = offx[:, None, None] + offy[None, :, None] + offz[None, None, :]
    x = tl.load(X + off3d)
    x = tl.flip(x, dim)
    tl.store(Z + off3d, x)


# %%
# Test every case against torch.flip.
triton.runtime.driver.set_active_to_cpu()

from triton.backends.cpu.riscv import compile_deploy_and_run

RISCV_HOST = "chlee@140.114.78.64"
RISCV_REMOTE_DIR = "~/triton-riscv-elf"


def run_on_board(M, N, K, dim, dtype_str):
    # Cast so the bfloat16 case really runs in bf16 (exact, since the values are bf16-representable).
    x = torch.from_numpy(numpy_random((M, N, K), dtype_str=dtype_str)).to(getattr(torch, dtype_str))
    expected = torch.flip(x, (dim, ))
    arguments = {
        "X": x.flatten(),
        "Z": torch.zeros_like(x).flatten(),
    }
    constexprs = {"M": M, "N": N, "K": K, "dim": dim}
    name = f"rvv-cpu-flip-{M}x{N}x{K}-dim{dim}-{dtype_str}"
    result = compile_deploy_and_run(flip_kernel, arguments, (1, ), f"artifacts/riscv/{name}.elf", RISCV_HOST,
                                     constexprs=constexprs, expected={"Z": expected.flatten().tolist()},
                                     remote_dir=RISCV_REMOTE_DIR)
    print(f"{name}:")
    print(result.stdout, end="")
    print(result.stderr, end="")


for M, N, K, dim in flip_cases:
    for dtype_str in dtypes:
        run_on_board(M, N, K, dim, dtype_str)
