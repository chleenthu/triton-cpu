"""
Causal GQA Attention (fused QKV)
================================

Causal grouped-query attention that reads q, k and v straight from a fused QKV buffer
(M, (Hq + 2*Hkv) * D) and writes (M, Hq * D), the layout an o_proj consumes. One program owns
BLOCK_M query rows of one query head; query head h reads kv head h // (Hq // Hkv).

The masks it builds are tail masks of three shapes:

    q / out:  m_mask[:, None]            rows offs_m < M          (BLOCK_M x HEAD_DIM)
    k:        c_mask[None, :]            cols  cols   < M          (HEAD_DIM x BLOCK_N, transposed)
    v:        c_mask[:, None]            rows  cols   < M          (BLOCK_N x HEAD_DIM)

plus the causal tl.where((offs_m[:, None] >= cols[None, :]) & c_mask[None, :], qk, -inf) on the
scores, which is not box-shaped (its bound differs per row). M = 37 is not a multiple of the
16-row blocks, so the last query block and the last kv block are both partial.

Used to check how far AnalyzeTailMasks / TRITON_VSETVL_CHUNK / TRITON_VSETVL_REDUCE handle the
masks of a real attention kernel.

Data types follow the fused Qwen2.5 prefill pipeline this kernel comes from: q, k, v and out are
fp16; the scores, softmax statistics and accumulator are fp32 (tl.dot of fp16 accumulates in
fp32), and p is cast to fp16 before the p @ v dot. The reference does the same in torch (fp32
attention on the fp16 inputs) and rounds the result to fp16.
"""

import math

import torch

import triton
import triton.language as tl


@triton.jit
def _attn_fwd(
    qkv_ptr, out_ptr, M, stride_qkv, stride_o, qk_scale_log2,
    NUM_Q_HEADS: tl.constexpr, NUM_KV_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Causal GQA attention reading q/k/v straight from the fused QKV buffer
    (M, (Hq + 2*Hkv) * D) and writing (M, Hq * D), the layout o_proj consumes."""
    start_m = tl.program_id(0)
    h = tl.program_id(1)
    kvh = h // (NUM_Q_HEADS // NUM_KV_HEADS)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    m_mask = offs_m < M

    q = tl.load(
        qkv_ptr + offs_m[:, None] * stride_qkv + h * HEAD_DIM + offs_d[None, :],
        mask=m_mask[:, None], other=0.0,
    )
    k_base = qkv_ptr + (NUM_Q_HEADS + kvh) * HEAD_DIM
    v_base = qkv_ptr + (NUM_Q_HEADS + NUM_KV_HEADS + kvh) * HEAD_DIM

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    hi = tl.minimum((start_m + 1) * BLOCK_M, M)
    for start_n in range(0, hi, BLOCK_N):
        cols = start_n + offs_n
        c_mask = cols < M
        k = tl.load(
            k_base + cols[None, :] * stride_qkv + offs_d[:, None], mask=c_mask[None, :], other=0.0
        )
        qk = tl.dot(q, k) * qk_scale_log2
        qk = tl.where((offs_m[:, None] >= cols[None, :]) & c_mask[None, :], qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp2(qk - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(
            v_base + cols[:, None] * stride_qkv + offs_d[None, :], mask=c_mask[:, None], other=0.0
        )
        acc = tl.dot(p.to(v.dtype), v, acc * alpha[:, None])
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(
        out_ptr + offs_m[:, None] * stride_o + h * HEAD_DIM + offs_d[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=m_mask[:, None],
    )


# %%
# Test against a torch reference. M = 37 leaves a 5-row remainder for the 16-row blocks.
torch.manual_seed(0)
M = 37
NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM = 4, 2, 16
BLOCK_M, BLOCK_N = 16, 16
triton.runtime.driver.set_active_to_cpu()

qkv = torch.randn(M, (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM, dtype=torch.float16)
sm_scale = 1.0 / math.sqrt(HEAD_DIM)

q, k, v = qkv.float().split([NUM_Q_HEADS * HEAD_DIM, NUM_KV_HEADS * HEAD_DIM, NUM_KV_HEADS * HEAD_DIM], dim=1)
q = q.view(M, NUM_Q_HEADS, HEAD_DIM).transpose(0, 1)  # (Hq, M, D)
k = k.view(M, NUM_KV_HEADS, HEAD_DIM).transpose(0, 1).repeat_interleave(NUM_Q_HEADS // NUM_KV_HEADS, 0)
v = v.view(M, NUM_KV_HEADS, HEAD_DIM).transpose(0, 1).repeat_interleave(NUM_Q_HEADS // NUM_KV_HEADS, 0)
ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, scale=sm_scale)
# The kernel stores fp16, so round the fp32 reference the same way.
expected = ref.transpose(0, 1).reshape(M, NUM_Q_HEADS * HEAD_DIM).to(torch.float16).float()

from triton.backends.cpu.riscv import compile_deploy_and_run

RISCV_HOST = "chlee@140.114.78.64"
RISCV_REMOTE_DIR = "~/triton-riscv-elf"


def run_on_board(kernel, constexprs, name):
    arguments = {
        "qkv_ptr": qkv.flatten(),
        "out_ptr": torch.zeros(M * NUM_Q_HEADS * HEAD_DIM, dtype=torch.float16),
        "M": M,
        "stride_qkv": (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM,
        "stride_o": NUM_Q_HEADS * HEAD_DIM,
        "qk_scale_log2": sm_scale * math.log2(math.e),
    }
    grid = (triton.cdiv(M, BLOCK_M), NUM_Q_HEADS)
    result = compile_deploy_and_run(kernel, arguments, grid, f"artifacts/riscv/{name}.elf", RISCV_HOST,
                                     constexprs=constexprs, signature={"qkv_ptr": "*fp16", "out_ptr": "*fp16"},
                                     expected={"out_ptr": expected.flatten().tolist()},
                                     # FP16 output, p rounded to FP16 before the second dot: atol 1e-2,
                                     # as in rvv_06-fused-attention_elf.py.
                                     atol=1e-2, remote_dir=RISCV_REMOTE_DIR)
    print(f"{name}:")
    print(result.stdout, end="")
    print(result.stderr, end="")


run_on_board(_attn_fwd, {
    "NUM_Q_HEADS": NUM_Q_HEADS, "NUM_KV_HEADS": NUM_KV_HEADS, "HEAD_DIM": HEAD_DIM,
    "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N,
}, "rvv-gqa-attn")
