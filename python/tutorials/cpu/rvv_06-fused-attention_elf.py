"""
Fused Attention, forward (RISC-V ELF)
=====================================

The forward kernel of python/tutorials/06-fused-attention.py (Flash Attention
v2), compiled for riscv64, deployed to the board and checked there, for
non-causal (STAGE = 1) and causal (STAGE = 3) attention in FP16.

Differences from the GPU tutorial:

* q, k, v and o are passed as plain pointers, so _maybe_make_tensor_desc builds
  the tensor descriptors inside the kernel (the path the tutorial takes when
  host TMA descriptors are not supported).
* One fixed config (BLOCK_M = 64, BLOCK_N = 32) instead of autotuning, with
  warp_specialize = False and IS_HOPPER = False, and N_CTX passed as a
  constexpr. The FP8 path and the backward kernels are not ported.

The reference computes softmax(q k^T * sm_scale) v in FP32 from the FP16 inputs;
M is the per-row log-sum-exp of the scaled scores in base 2, which the kernel
stores for the backward pass.
"""

import math

import torch

import triton
import triton.language as tl


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    desc_k, desc_v,  #
                    offset_y, dtype: tl.constexpr, start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, warp_specialize: tl.constexpr, IS_HOPPER: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    offsetk_y = offset_y + lo
    if dtype == tl.float8e5:
        offsetv_y = offset_y * HEAD_DIM + lo
    else:
        offsetv_y = offset_y + lo
    # loop over k, v and update accumulator
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = desc_k.load([offsetk_y, 0]).T
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        # -- compute correction factor
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        # -- update output accumulator --
        if not IS_HOPPER and warp_specialize and BLOCK_M == 128 and HEAD_DIM == 128:
            BM: tl.constexpr = acc.shape[0]
            BN: tl.constexpr = acc.shape[1]
            acc0, acc1 = acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            acc = tl.join(acc0, acc1).permute(0, 2, 1).reshape([BM, BN])
        else:
            acc = acc * alpha[:, None]
        # prepare p and v for the dot
        if dtype == tl.float8e5:
            v = desc_v.load([0, offsetv_y]).T
        else:
            v = desc_v.load([offsetv_y, 0])
        p = p.to(dtype)
        # note that this non transposed v for FP8 is only supported on Blackwell
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        # place this at the end of the loop to reduce register pressure
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N
    return acc, l_i, m_i


@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


@triton.jit
def _attn_fwd(sm_scale, M,  #
              Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              FP8_OUTPUT: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              warp_specialize: tl.constexpr,  #
              IS_HOPPER: tl.constexpr,  #
              ):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(desc_q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])
    if FP8_OUTPUT:
        desc_v = _maybe_make_tensor_desc(desc_v, shape=[HEAD_DIM, y_dim], strides=[N_CTX, 1],
                                         block_shape=[HEAD_DIM, BLOCK_N])
    else:
        desc_v = _maybe_make_tensor_desc(desc_v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                         block_shape=[BLOCK_N, HEAD_DIM])
    desc_k = _maybe_make_tensor_desc(desc_k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N, HEAD_DIM])
    desc_o = _maybe_make_tensor_desc(desc_o, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = desc_q.load([qo_offset_y, 0])
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,  #
                                        desc_k, desc_v,  #
                                        offset_y, dtype, start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX,  #
                                        warp_specialize, IS_HOPPER)
    # stage 2: on-band
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,  #
                                        desc_k, desc_v,  #
                                        offset_y, dtype, start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX,  #
                                        warp_specialize, IS_HOPPER)
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    desc_o.store([qo_offset_y, 0], acc.to(dtype))


# %%
# Unit Test
# ---------

torch.manual_seed(20)
triton.runtime.driver.set_active_to_cpu()

Z, H, N_CTX, HEAD_DIM = 1, 2, 128, 64
BLOCK_M, BLOCK_N = 64, 32
sm_scale = 0.5
q = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=torch.float16).normal_(mean=0.0, std=0.5)
k = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=torch.float16).normal_(mean=0.0, std=0.5)
v = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=torch.float16).normal_(mean=0.0, std=0.5)


def attention_reference(causal):
    s = torch.matmul(q.float(), k.float().transpose(2, 3)) * sm_scale
    if causal:
        keep = torch.tril(torch.ones((N_CTX, N_CTX), dtype=torch.bool))
        s = s.masked_fill(~keep, float("-inf"))
    o = torch.matmul(torch.softmax(s, dim=-1), v.float())
    m = torch.logsumexp(s, dim=-1) / math.log(2)
    return o, m


from triton.backends.cpu.riscv import compile_deploy_and_run

RISCV_HOST = "chlee@140.114.78.64"
RISCV_REMOTE_DIR = "~/triton-riscv-elf"


def run_on_board(causal, name):
    o_ref, m_ref = attention_reference(causal)
    numel = Z * H * N_CTX * HEAD_DIM
    arguments = {
        "sm_scale": sm_scale, "M": [0.0] * (Z * H * N_CTX), "Z": Z, "H": H,  #
        "desc_q": q.float().flatten().tolist(), "desc_k": k.float().flatten().tolist(),  #
        "desc_v": v.float().flatten().tolist(), "desc_o": [0.0] * numel,
    }
    constexprs = {
        "N_CTX": N_CTX, "HEAD_DIM": HEAD_DIM, "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N,  #
        "FP8_OUTPUT": False, "STAGE": 3 if causal else 1, "warp_specialize": False, "IS_HOPPER": False,
    }
    signature = {"sm_scale": "fp32", "M": "*fp32", "desc_q": "*fp16", "desc_k": "*fp16", "desc_v": "*fp16",
                 "desc_o": "*fp16"}
    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H)
    # FP16 output (p is rounded to FP16 before the second dot): atol 1e-2.
    # The runner checks one atol for all buffers, so M uses the same one.
    result = compile_deploy_and_run(_attn_fwd, arguments, grid, f"artifacts/riscv/{name}.elf", RISCV_HOST,
                                     constexprs=constexprs, signature=signature,
                                     expected={"desc_o": o_ref.flatten().tolist(), "M": m_ref.flatten().tolist()},
                                     atol=1e-2, remote_dir=RISCV_REMOTE_DIR)
    print(f"{name}:")
    print(result.stdout, end="")
    print(result.stderr, end="")


run_on_board(False, "rvv-attention-fwd")
run_on_board(True, "rvv-attention-fwd-causal")
