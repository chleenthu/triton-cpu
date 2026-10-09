"""
Block Scaled Matrix Multiplication (RISC-V ELF)
===============================================

The microscaled (MX) matmul of python/tutorials/10-block-scaled-matmul.py,
compiled for riscv64, deployed to the board and checked there, for the
tutorial's three format combinations: mxfp8 x mxfp8, mxfp4 x mxfp4, and the
mixed mxfp8 x mxfp4.

The GPU tutorial computes each K step with tl.dot_scaled, which maps to the
block-scaled MMA instructions of Blackwell / CDNA4. The RISC-V CPU backend
cannot lower tl.dot_scaled ("failed to translate module to LLVM IR"), so this
version does the same math by hand:

* elements are stored as raw bytes: e4m3 (one per byte, mxfp8) or e2m1 (two
  per byte, mxfp4; the low nibble holds the even element, as
  triton.tools.mxfp packs them),
* each group of 32 consecutive elements along K shares one e8m0 scale,
  2**(scale - 127),
* the kernel decodes the elements to FP32, multiplies every group by its
  scale, and accumulates with a regular tl.dot.

The pointer-based structure follows block_scaled_matmul_kernel_cdna4, but the
scales use a plain [rows, K / 32] layout (no MFMA shuffle), and the output is
FP32.
"""

import numpy as np
import torch

import triton
import triton.language as tl


@triton.jit
def _e8m0_to_f32(s):
    # 2**(s - 127): the scale byte is exactly the exponent field of an FP32.
    return (s.to(tl.uint32) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _e2m1_to_f32(x):
    # e2m1: sign, 2 exponent bits (bias 1), 1 mantissa bit; exponent 0 is
    # subnormal. Values: 0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives.
    sign = (x >> 3) & 1
    e = (x >> 1) & 3
    m = (x & 1).to(tl.float32)
    mag = tl.where(e == 0, m * 0.5, (1.0 + m * 0.5) * tl.exp2((e - 1).to(tl.float32)))
    return tl.where(sign == 1, -mag, mag)


@triton.jit
def _load_a(a_ptrs, A_FMT: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    raw = tl.load(a_ptrs)  # [BLOCK_M, BLOCK_K // pack] bytes
    if A_FMT == "e4m3":
        return raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    else:
        lo = _e2m1_to_f32(raw & 0xF)
        hi = _e2m1_to_f32(raw >> 4)
        return tl.join(lo, hi).reshape(BLOCK_M, BLOCK_K)


@triton.jit
def _load_b(b_ptrs, B_FMT: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
    raw = tl.load(b_ptrs)  # [BLOCK_K // pack, BLOCK_N] bytes, packed along K
    if B_FMT == "e4m3":
        return raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    else:
        lo = _e2m1_to_f32(raw & 0xF)
        hi = _e2m1_to_f32(raw >> 4)
        return tl.join(lo, hi).permute(0, 2, 1).reshape(BLOCK_K, BLOCK_N)


@triton.jit
def block_scaled_matmul_kernel(a_ptr, b_ptr, c_ptr, a_scales_ptr, b_scales_ptr, M, N, K,  #
                               stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,  #
                               stride_asm, stride_ask, stride_bsn, stride_bsk,  #
                               A_FMT: tl.constexpr, B_FMT: tl.constexpr,  #
                               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """C = A x B with A, B in MX formats. A is (M, K), B is (K, N), both packed
    along K; A_scales is (M, K / 32) and B_scales is (N, K / 32), e8m0."""
    SCALE_GROUP_SIZE: tl.constexpr = 32
    A_PACK: tl.constexpr = 2 if A_FMT == "e2m1" else 1
    B_PACK: tl.constexpr = 2 if B_FMT == "e2m1" else 1
    tl.static_assert(BLOCK_K % SCALE_GROUP_SIZE == 0)

    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_ka = tl.arange(0, BLOCK_K // A_PACK)
    offs_kb = tl.arange(0, BLOCK_K // B_PACK)
    offs_ks = tl.arange(0, BLOCK_K // SCALE_GROUP_SIZE)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_ka[None, :] * stride_ak
    b_ptrs = b_ptr + offs_kb[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    a_scale_ptrs = a_scales_ptr + offs_am[:, None] * stride_asm + offs_ks[None, :] * stride_ask
    b_scale_ptrs = b_scales_ptr + offs_bn[:, None] * stride_bsn + offs_ks[None, :] * stride_bsk

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = _load_a(a_ptrs, A_FMT, BLOCK_M, BLOCK_K)  # [BLOCK_M, BLOCK_K]
        b = _load_b(b_ptrs, B_FMT, BLOCK_K, BLOCK_N)  # [BLOCK_K, BLOCK_N]
        a_scales = _e8m0_to_f32(tl.load(a_scale_ptrs))  # [BLOCK_M, BLOCK_K // 32]
        b_scales = _e8m0_to_f32(tl.load(b_scale_ptrs))  # [BLOCK_N, BLOCK_K // 32]

        # Apply each scale to its group of 32 elements along K.
        a = (a.reshape(BLOCK_M, BLOCK_K // SCALE_GROUP_SIZE, SCALE_GROUP_SIZE) *
             a_scales[:, :, None]).reshape(BLOCK_M, BLOCK_K)
        b = (b.reshape(BLOCK_K // SCALE_GROUP_SIZE, SCALE_GROUP_SIZE, BLOCK_N) *
             tl.trans(b_scales)[:, None, :]).reshape(BLOCK_K, BLOCK_N)
        accumulator = tl.dot(a, b, accumulator)

        a_ptrs += (BLOCK_K // A_PACK) * stride_ak
        b_ptrs += (BLOCK_K // B_PACK) * stride_bk
        a_scale_ptrs += (BLOCK_K // SCALE_GROUP_SIZE) * stride_ask
        b_scale_ptrs += (BLOCK_K // SCALE_GROUP_SIZE) * stride_bsk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# %%
# Inputs and reference
# --------------------

E2M1_VALUES = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float64)


def random_mx(rows, cols, fmt, rng):
    """Random elements in fmt; returns (codes as stored, decoded values)."""
    if fmt == "e4m3":
        x = torch.from_numpy(rng.standard_normal((rows, cols)).astype(np.float32)).to(torch.float8_e4m3fn)
        return x.view(torch.uint8).numpy(), x.float().numpy().astype(np.float64)
    codes = rng.integers(0, 16, size=(rows, cols), dtype=np.uint8)
    return codes, E2M1_VALUES[codes]


def pack_e2m1(codes, axis):
    """Two e2m1 codes per byte along axis: the low nibble holds the even element."""
    codes = np.moveaxis(codes, axis, -1)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    return np.moveaxis(packed, -1, axis).astype(np.uint8)


def make_case(M, N, K, a_fmt, b_fmt, seed=0):
    rng = np.random.default_rng(seed)
    a_codes, a_vals = random_mx(M, K, a_fmt, rng)  # (M, K)
    b_codes, b_vals = random_mx(K, N, b_fmt, rng)  # (K, N)
    # e8m0 scales near 1: 2**-3 .. 2**3.
    a_scale = rng.integers(124, 131, size=(M, K // 32), dtype=np.uint8)
    b_scale = rng.integers(124, 131, size=(N, K // 32), dtype=np.uint8)
    a_deq = a_vals * np.repeat(2.0**(a_scale.astype(np.float64) - 127), 32, axis=1)
    b_deq = b_vals * np.repeat(2.0**(b_scale.astype(np.float64) - 127), 32, axis=1).T
    a_bytes = pack_e2m1(a_codes, 1) if a_fmt == "e2m1" else a_codes
    b_bytes = pack_e2m1(b_codes, 0) if b_fmt == "e2m1" else b_codes
    return a_bytes, b_bytes, a_scale, b_scale, a_deq @ b_deq


# %%
# Unit Test
# ---------

triton.runtime.driver.set_active_to_cpu()

M, N, K = 64, 64, 128
GPU_BLOCK_M, GPU_BLOCK_N, GPU_BLOCK_K = 128, 256, 128
CPU_BLOCK_M, CPU_BLOCK_N, CPU_BLOCK_K = 16, 16, 64
BLOCK_M, BLOCK_N, BLOCK_K = CPU_BLOCK_M, CPU_BLOCK_N, CPU_BLOCK_K

from triton.backends.cpu.riscv import compile_deploy_and_run

RISCV_HOST = "chlee@140.114.78.64"
RISCV_REMOTE_DIR = "~/triton-riscv-elf"


def run_on_board(a_fmt, b_fmt, name):
    a_bytes, b_bytes, a_scale, b_scale, c_ref = make_case(M, N, K, a_fmt, b_fmt)
    arguments = {
        "a_ptr": a_bytes.flatten().tolist(), "b_ptr": b_bytes.flatten().tolist(), "c_ptr": [0.0] * (M * N),  #
        "a_scales_ptr": a_scale.flatten().tolist(), "b_scales_ptr": b_scale.flatten().tolist(),  #
        "M": M, "N": N, "K": K,  #
        "stride_am": a_bytes.shape[1], "stride_ak": 1, "stride_bk": b_bytes.shape[1], "stride_bn": 1,  #
        "stride_cm": N, "stride_cn": 1,  #
        "stride_asm": K // 32, "stride_ask": 1, "stride_bsn": K // 32, "stride_bsk": 1,
    }
    signature = {"a_ptr": "*u8", "b_ptr": "*u8", "c_ptr": "*fp32", "a_scales_ptr": "*u8", "b_scales_ptr": "*u8"}
    constexprs = {"A_FMT": a_fmt, "B_FMT": b_fmt, "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K}
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )
    # The products are exact in FP32; only the order of the FP32 sum differs.
    atol = max(1e-6, float(np.abs(c_ref).max()) * 2.0**-20)
    result = compile_deploy_and_run(block_scaled_matmul_kernel, arguments, grid, f"artifacts/riscv/{name}.elf",
                                     RISCV_HOST, constexprs=constexprs, signature=signature,
                                     expected={"c_ptr": c_ref.flatten().tolist()}, atol=atol,
                                     remote_dir=RISCV_REMOTE_DIR)
    print(f"{name} (atol {atol:.3g}):")
    print(result.stdout, end="")
    print(result.stderr, end="")


run_on_board("e4m3", "e4m3", "rvv-block-scaled-mxfp8")
# run_on_board("e2m1", "e2m1", "rvv-block-scaled-mxfp4")
# run_on_board("e4m3", "e2m1", "rvv-block-scaled-mixed")
