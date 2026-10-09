"""
Group GEMM (RISC-V ELF)
=======================

The grouped GEMM kernel of python/tutorials/08-grouped-gemm.py, compiled for
riscv64, deployed to the board and checked there: several FP16 GEMMs of
different sizes computed by one launch of NUM_SM programs, each walking the
tiles of all groups in turn.

Difference from the GPU tutorial: there, each group's A, B and C are separate
device tensors whose addresses are passed in tables (group_a_ptrs, ...) and
loaded in the kernel with `.to(tl.pointer_type(tl.float16))`. The board runner
declares every buffer as its own C array and fills buffers with literal values
only, so a table of runtime addresses cannot be passed. Here all groups' A
matrices are concatenated into one buffer (likewise B and C), and the tables
hold each group's element offset into it: a_ptr = group_a + offset. The rest
of the kernel (sizes and leading dimensions per group, the tile walk, the
blocked GEMM) is unchanged. The TMA variant (grouped_matmul_tma_kernel) and
autotuning are not ported.
"""

import torch

import triton
import triton.language as tl


@triton.jit
def grouped_matmul_kernel(
    # all groups' A, B and C matrices, concatenated
    group_a,
    group_b,
    group_c,
    # element offset of each group's matrix in group_a / group_b / group_c
    group_a_offs,
    group_b_offs,
    group_c_offs,
    # device tensor of gemm sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <M, N, K> of each gemm
    group_gemm_sizes,
    # device tensor of leading dimension sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <lda, ldb, ldc> of each gemm
    g_lds,
    # number of gemms
    group_size,
    # number of virtual SM
    NUM_SM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        # get the gemm size of the current problem
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        # iterate through the tiles in the current gemm problem
        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            # pick up a tile from the current gemm problem
            k = gk
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)
            a_ptr = group_a + tl.load(group_a_offs + g)
            b_ptr = group_b + tl.load(group_b_offs + g)
            c_ptr = group_c + tl.load(group_c_offs + g)
            # figure out tile coordinates
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            # do regular gemm here
            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
            b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                # hint to Triton compiler to do proper loop pipelining
                tl.multiple_of(a_ptrs, [16, 16])
                tl.multiple_of(b_ptrs, [16, 16])
                # assume full tile for now
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K
                b_ptrs += BLOCK_SIZE_K * ldb
            c = accumulator.to(tl.float16)

            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_cn[None, :]

            # assumes full tile for now
            tl.store(c_ptrs, c)

            # go to the next tile by advancing NUM_SM
            tile_idx += NUM_SM

        # get ready to go to the next gemm problem
        last_problem_end = last_problem_end + num_tiles


# %%
# Unit Test
# ---------

torch.manual_seed(0)
triton.runtime.driver.set_active_to_cpu()

GPU_BLOCK_SIZE_M, GPU_BLOCK_SIZE_N, GPU_BLOCK_SIZE_K = 128, 128, 32
CPU_BLOCK_SIZE_M, CPU_BLOCK_SIZE_N, CPU_BLOCK_SIZE_K = 16, 16, 32
BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = CPU_BLOCK_SIZE_M, CPU_BLOCK_SIZE_N, CPU_BLOCK_SIZE_K
NUM_SM = 4
# <M, N, K> of each GEMM; full tiles only, as the kernel assumes.
group_m = [64, 96, 32, 128]
group_n = [64, 32, 96, 64]
group_k = [64, 32, 96, 32]
group_size = len(group_m)

group_A, group_B, group_C_ref = [], [], []
a_offs, b_offs, c_offs, g_sizes, g_lds = [], [], [], [], []
a_total = b_total = c_total = 0
for M, N, K in zip(group_m, group_n, group_k):
    A = torch.rand((M, K), dtype=torch.float16)
    B = torch.rand((K, N), dtype=torch.float16)
    group_A.append(A)
    group_B.append(B)
    group_C_ref.append(torch.matmul(A.float(), B.float()))
    a_offs.append(a_total)
    b_offs.append(b_total)
    c_offs.append(c_total)
    a_total += M * K
    b_total += K * N
    c_total += M * N
    g_sizes += [M, N, K]
    g_lds += [A.stride(0), B.stride(0), N]

from triton.backends.cpu.riscv import compile_deploy_and_run

RISCV_HOST = "chlee@140.114.78.64"
RISCV_REMOTE_DIR = "~/triton-riscv-elf"


def flat(ts):
    return torch.cat([t.float().flatten() for t in ts]).tolist()


def run_on_board(kernel, name):
    arguments = {
        "group_a": flat(group_A), "group_b": flat(group_B), "group_c": [0.0] * c_total,  #
        "group_a_offs": a_offs, "group_b_offs": b_offs, "group_c_offs": c_offs,  #
        "group_gemm_sizes": g_sizes, "g_lds": g_lds, "group_size": group_size,
    }
    signature = {
        "group_a": "*fp16", "group_b": "*fp16", "group_c": "*fp16",  #
        "group_a_offs": "*i32", "group_b_offs": "*i32", "group_c_offs": "*i32",  #
        "group_gemm_sizes": "*i32", "g_lds": "*i32",
    }
    constexprs = {"NUM_SM": NUM_SM, "BLOCK_SIZE_M": BLOCK_SIZE_M, "BLOCK_SIZE_N": BLOCK_SIZE_N,
                  "BLOCK_SIZE_K": BLOCK_SIZE_K}
    # The output is FP16: allow one FP16 ulp (10 bits of mantissa) at the largest value.
    c_max = max(c.abs().max().item() for c in group_C_ref)
    atol = 2.0**(torch.log2(torch.tensor(c_max)).floor().item() - 10)
    result = compile_deploy_and_run(kernel, arguments, (NUM_SM, ), f"artifacts/riscv/{name}.elf", RISCV_HOST,
                                     constexprs=constexprs, signature=signature,
                                     expected={"group_c": flat(group_C_ref)}, atol=atol,
                                     remote_dir=RISCV_REMOTE_DIR)
    print(f"{name}:")
    print(result.stdout, end="")
    print(result.stderr, end="")


run_on_board(grouped_matmul_kernel, "rvv-grouped-gemm")
