# Standalone repro of vLLM's unified-attention Triton kernels, as run by
# ../model.sh (Qwen2.5-0.5B, fp16, TRITON_ATTN backend, sm121).
#
#   python unified_attention_repro.py                 # real 128-token prefill (2D)
#                                                     #  + first 3D decode-warmup call
#   python unified_attention_repro.py --case all      # every captured call
#   python unified_attention_repro.py --list
#
# Kernel bodies = only the code paths that were compiled in model.sh (dead
# constexpr branches removed). The signature keeps the same runtime args as
# vLLM (incl. unused ones) so the TTIR is identical to
# ../triton_cache/{LEN53FFT..., NARPBJGB...}/ apart from #loc.
#
# Launch args are the exact kwargs vLLM passed, captured with
# capture_sitecustomize.py into captured/*.pt (tensors + scalars + constexprs
# + grid). The output is checked bit-for-bit against what vLLM produced.
#
# vLLM call chain:
#   Qwen2Attention.forward -> Attention.forward
#   -> torch.ops.vllm.unified_attention_with_output
#   -> TritonAttentionImpl.forward (v1/attention/backends/triton_attn.py)
#   -> unified_attention()        (v1/attention/ops/triton_unified_attention.py)
#   -> kernel_unified_attention[grid](...)  [+ reduce_segments[grid](...) in 3D]
#
# "# L123" / "# H123" = line in vLLM's triton_unified_attention.py /
# triton_attention_helpers.py.
import argparse
import glob
import os

import torch
import triton
import triton.language as tl

HERE = os.path.dirname(os.path.abspath(__file__))
CAPTURED = os.path.join(HERE, "captured")
# GPU if present, else CPU (triton-cpu backend)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# helpers (triton_attention_helpers.py)
# ---------------------------------------------------------------------------
@triton.jit
def cdiv_fn(x, y):  # H22
    return (x + y - 1) // y


@triton.jit
def find_seq_idx(query_start_len_ptr, target_idx, num_seqs,  # H83
                 BLOCK_Q: tl.constexpr, use_q_block_mode: tl.constexpr):
    # binary search over cu_seqlens_q
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val
        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid
    return left - 1


@triton.jit
def softmax_step(S, M, L):  # H462  online-softmax update
    m_j = tl.maximum(M, tl.max(S, axis=1))
    safe_m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
    P = tl.exp(S - safe_m_j[:, None])
    l_j = tl.sum(P, axis=1)
    alpha = tl.exp(M - safe_m_j)
    L_new = L * alpha + l_j
    return m_j, L_new, P, alpha


# ---------------------------------------------------------------------------
# main kernel (L295)
# ---------------------------------------------------------------------------
@triton.jit
def kernel_unified_attention(
    output_ptr, query_ptr, key_cache_ptr, value_cache_ptr,
    block_tables_ptr, seq_lens_ptr,
    scale,
    k_scale, v_scale, out_scale, softcap,          # unused here; kept for same signature
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64, query_stride_1: tl.int64,
    output_stride_0: tl.int64, output_stride_1: tl.int64,
    qq_bias_stride_0: tl.int64,                    # unused
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    rswa_prefix_lens_ptr,                          # unused
    stride_k_cache_0: tl.int64, stride_k_cache_1: tl.int64,
    stride_k_cache_2: tl.int64, stride_k_cache_3: tl.constexpr,
    stride_v_cache_0: tl.int64, stride_v_cache_1: tl.int64,
    stride_v_cache_2: tl.int64, stride_v_cache_3: tl.constexpr,
    query_start_len_ptr,
    BLOCK_Q: tl.constexpr,
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
    IS_3D: tl.constexpr,
    segm_output_ptr=None, segm_max_ptr=None, segm_expsum_ptr=None,
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2) if IS_3D else 0

    # ---- which sequence / q-block am I (resolve_seq_and_query_len, H49) ----
    seq_idx = find_seq_idx(query_start_len_ptr, q_block_global_idx, num_seqs,
                           BLOCK_Q, True)                                   # H70
    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx
    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_stop = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_stop - cur_batch_in_all_start_index
    seq_len = tl.load(seq_lens_ptr + seq_idx)                               # H78

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:                 # L435
        return

    if IS_3D:
        tiles_per_segment = cdiv_fn(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)  # L439
        if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
            return
    else:
        tiles_per_segment = 0                                              # L443

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv  # L456

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (query_offset_0[:, None] * query_stride_0
                    + query_offset_1[:, None] * query_stride_1
                    + offs_d[None, :])

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)                                       L485
    Q = tl.load(query_ptr + query_offset,
                mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
                other=0.0)

    block_table_offset = seq_idx * block_table_stride
    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)               # H133
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    context_len = seq_len - cur_batch_query_len                            # L515

    # ---- tile-loop bounds (compute_tile_loop_bounds, H146; causal, no SWA) --
    max_seq_prefix_len = (context_len + q_block_local_idx * BLOCK_Q
                          + (BLOCK_M - 1) // num_queries_per_kv + 1)        # H191
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)          # H204
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)                     # H206
    if IS_3D:
        loop_lo = max(segm_idx * tiles_per_segment, 0)                     # H276
        loop_hi = min((segm_idx + 1) * tiles_per_segment, num_tiles)       # H277
    else:
        loop_lo = 0
        loop_hi = num_tiles

    for j in range(loop_lo, loop_hi):                                      # L541
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)                                                     # L545

        v_offset = (physical_block_idx[:, None] * stride_v_cache_0         # L590
                    + kv_head_idx * stride_v_cache_2
                    + offs_d[None, :] * stride_v_cache_3
                    + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1)
        k_offset = (physical_block_idx[None, :] * stride_k_cache_0
                    + kv_head_idx * stride_k_cache_2
                    + offs_d[:, None] * stride_k_cache_3
                    + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1)
        # K : (HEAD_SIZE, TILE_SIZE)                                        L608
        K = tl.load(key_cache_ptr + k_offset,
                    mask=dim_mask[:, None] & tile_mask[None, :], other=0.0)
        # V : (TILE_SIZE, HEAD_SIZE)                                        L613
        V = tl.load(value_cache_ptr + v_offset,
                    mask=dim_mask[None, :] & tile_mask[:, None], other=0.0)
        K = K.to(Q.dtype)                                                  # L635 (_cast_kv_tile)
        V = V.to(Q.dtype)                                                  # L636

        # causal mask (compute_kv_seq_mask, H359)
        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = seq_offset[None, :] <= query_abs_pos

        # S : (BLOCK_M, TILE_SIZE)
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        S += scale * tl.dot(Q, K)                                          # L663
        S = tl.where(query_mask_1[:, None] & query_mask_0[:, None] & seq_mask,
                     S, float("-inf"))                                     # L668

        M, L, P, alpha = softmax_step(S, M, L)                             # L682
        acc = acc * alpha[:, None]
        acc += tl.dot(P.to(V.dtype), V)                                    # L712 (p_scale=1.0)

    # ---- epilogue -----------------------------------------------------------
    if IS_3D:
        # per-segment partial output; finalized by reduce_segments         L747
        segm_output_offset = (
            query_offset_0[:, None].to(tl.int64)
            * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
            + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
            + segm_idx * HEAD_SIZE_PADDED
            + tl.arange(0, HEAD_SIZE_PADDED)[None, :])
        tl.store(segm_output_ptr + segm_output_offset, acc,
                 mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None])
        # store_segm_reduce_scalars (H306)
        segm_offset = (query_offset_0.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
                       + query_offset_1 * NUM_SEGMENTS_PER_SEQ + segm_idx)
        tl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0 & query_mask_1)     # H311
        tl.store(segm_expsum_ptr + segm_offset, L, mask=query_mask_0 & query_mask_1)  # H312
    else:
        acc = acc / L[:, None]                                             # L773
        output_offset = (query_offset_0[:, None] * output_stride_0         # L802
                         + query_offset_1[:, None] * output_stride_1
                         + offs_d[None, :])
        tl.store(output_ptr + output_offset, acc,
                 mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None])


# ---------------------------------------------------------------------------
# 3D finalize: combine per-segment partials (L814)
# ---------------------------------------------------------------------------
@triton.jit
def reduce_segments(
    output_ptr, segm_output_ptr, segm_max_ptr, segm_expsum_ptr,
    seq_lens_ptr, num_seqs,
    num_query_heads: tl.constexpr,
    out_scale_inv,                                 # unused; kept for same signature
    output_stride_0: tl.int64, output_stride_1: tl.int64,
    block_table_stride: tl.int64,                  # unused
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    query_start_len_ptr,
    BLOCK_Q: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    if query_token_idx >= tl.load(query_start_len_ptr + num_seqs):        # L841 (graph padding)
        return
    seq_idx = find_seq_idx(query_start_len_ptr, query_token_idx, num_seqs,
                           BLOCK_Q, False)
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    if seq_len <= 0:
        return

    tiles_per_segment = cdiv_fn(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
    act_num_segments = cdiv_fn(seq_len, tiles_per_segment * TILE_SIZE)
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(
        [NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32)
    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1, 0).to(tl.int1)

    segm_offset = (query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
                   + query_head_idx * NUM_SEGMENTS_PER_SEQ
                   + tl.arange(0, NUM_SEGMENTS_PER_SEQ))
    segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))
    segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
    valid_segment = segm_mask & (segm_expsum > 0) & (segm_max != float("-inf"))
    segm_max = tl.where(valid_segment, segm_max, float("-inf"))
    overall_max = tl.max(segm_max)

    segment_scale = tl.where(valid_segment, tl.exp(segm_max - overall_max), 0.0)
    segm_expsum = segm_expsum * segment_scale
    overall_expsum = tl.sum(segm_expsum)

    segm_output_offset = (
        query_token_idx.to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :])
    segm_output = tl.load(segm_output_ptr + segm_output_offset,
                          mask=segm_mask[:, None] & dim_mask[None, :], other=0.0)
    segm_output *= segment_scale[:, None]
    acc_sum = tl.sum(segm_output, axis=0)
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

    output_offset = (query_token_idx * output_stride_0
                     + query_head_idx * output_stride_1
                     + tl.arange(0, HEAD_SIZE_PADDED))
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)             # L911


KERNELS = {"kernel_unified_attention": kernel_unified_attention,
           "reduce_segments": reduce_segments}

# The real 128-token prompt (2D), and the first 3D call -- the one that
# compiled triton_cache/NARPBJGB... (3D only ran in vLLM's warmup dummy runs,
# since model.sh generates max_tokens=1).
DEFAULT_CASES = [
    "kernel_unified_attention_2d_q128_seqlen128",
    "kernel_unified_attention_3d_q1_seqlen3",
    "reduce_segments_q1_seqlen3",
]


# ---------------------------------------------------------------------------
# replay captured vLLM launches
# ---------------------------------------------------------------------------
def load_case(name):
    rec = torch.load(os.path.join(CAPTURED, name + ".pt"), weights_only=False)
    args = {}
    for k, v in rec["kwargs"].items():
        if isinstance(v, dict) and "storage" in v:      # KV cache: exact strided view
            buf = v["storage"].to(DEV)
            v = torch.as_strided(buf, v["shape"], v["stride"], 0)
        elif isinstance(v, torch.Tensor):
            v = v.to(DEV)
        args[k] = v
    return rec, args


def run_case(name, prev_outputs):
    rec, args = load_case(name)
    fn = KERNELS[rec["name"]]

    # reduce_segments consumes the 3D kernel's segm_* outputs: chain them
    # if that kernel was replayed just before (else use vLLM's captured ones).
    if rec["name"] == "reduce_segments":
        for k in ("segm_output_ptr", "segm_max_ptr", "segm_expsum_ptr"):
            if k in prev_outputs:
                args[k] = prev_outputs[k]

    dropped = {k: v for k, v in args.items() if k not in fn.arg_names}
    kw = {k: v for k, v in args.items() if k in fn.arg_names}
    print(f"== {name}  grid={rec['grid']}")
    print("   args:", ", ".join(
        f"{k}={tuple(v.shape) if isinstance(v, torch.Tensor) else v}" for k, v in kw.items()))
    print("   dropped (dead in this config):", ", ".join(
        f"{k}={v}" for k, v in dropped.items() if not isinstance(v, torch.Tensor)))

    fn[rec["grid"]](**kw)
    if DEV == "cuda":
        torch.cuda.synchronize()

    ok = True
    for k, ref in rec["after"].items():
        got = kw[k].cpu()
        same = torch.equal(got, ref)
        ok &= same
        diff = (got.float() - ref.float()).abs().max().item()
        print(f"   {k}: {'bit-exact' if same else f'max|diff|={diff:.3g}'} vs vLLM (GPU)")
    return ok, {k: kw[k] for k in rec["after"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", nargs="*", default=DEFAULT_CASES,
                    help="captured case names, or 'all'")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    allcases = sorted(os.path.basename(p)[:-3] for p in glob.glob(os.path.join(CAPTURED, "*.pt")))
    if a.list:
        print("\n".join(allcases))
        return
    cases = allcases if a.case == ["all"] else a.case
    if a.case == ["all"]:  # run each 3D kernel right before its reduce_segments
        cases = sorted(cases, key=lambda c: (c.split("_seqlen")[-1], "reduce" in c))

    ok, prev = True, {}
    for c in cases:
        r, prev = run_case(c, prev)
        ok &= r
    print(f"device={DEV}:", "ALL BIT-EXACT vs vLLM" if ok else "not bit-exact vs vLLM GPU run (expected on CPU)")


if __name__ == "__main__":
    main()
