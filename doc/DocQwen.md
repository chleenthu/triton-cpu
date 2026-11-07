# Qwen2.5-0.5B through vLLM + triton-cpu

Workload: `scripts/qwen05b_prefill_seq128.py`, one 128-token prefill, 1 generated
token. Setup and run commands are in `DocVllm.md` ("Simple setup").

- **CPU:** x86 host, bf16, Inductor `cpu_backend="triton"`, `TRITON_CPU_TARGET=native`,
  vLLM's C++ `CPU_ATTN`.
- **GPU:** `chlee@140.114.78.168:~/serve`, RTX 3060 Ti, fp16, `TRITON_ATTN`.

# Where each op runs (CPU)

Only the glue ops are Triton. Counted from the call function of the compiled
graph: `torch.ops._C.onednn_mm` is the only matmul, plus
`torch.ops.vllm.unified_attention_with_output` (24×) and
`unified_kv_cache_update` (48×).

| Op | Where it runs |
|---|---|
| GEMM (qkv, o_proj, gate_up, down in all 24 layers) | oneDNN `torch.ops._C.onednn_mm` (C++) |
| `lm_head` | torch matmul, outside the compiled graph (`compute_logits`) |
| Attention | vLLM `CPU_ATTN` (C++) |
| RMSNorm, RoPE, SiLU-mul, embedding | **Triton** (Inductor → triton-cpu) |
| Input preparation and sampling | **Triton** (vLLM's own kernels → triton-cpu) |
| MoE / Mamba | not in Qwen2.5-0.5B (dense model) |

- **GEMM:** vLLM picks each linear's backend once at load time in
  `dispatch_cpu_unquantized_gemm` (`vllm/model_executor/layers/utils.py`): small
  layers → `weight_packed_linear_cpu` (sgl-kernel, C++), otherwise `onednn_mm`,
  fallback `torch.nn.functional.linear`. `onednn_mm` is opaque to Inductor, so it
  fuses RMSNorm / SiLU-mul / residual around the call (hence `onednn_mm` in the
  Triton kernel names) but the GEMM itself stays C++.
- **MoE / Mamba:** for models that have them, vLLM on CPU also uses C++/torch:
  MoE only offers the C++ CPU backend
  (`vllm/model_executor/layers/fused_moe/oracle/unquantized.py:97`), and Mamba
  swaps in CPU versions (`mamba_ssm.py:852`, `causal_conv1d.py:1282`,
  `ssd_combined.py:232`, `ssu_dispatch.py:484`).
- Routing all four through Triton is described in `DocVllm.md` ("Routing
  Attention, GEMM, MoE and Mamba through Triton").

# Compiled kernels

A *source* is one `@triton.jit` function; a *compiled kernel* is one entry in the
Triton cache (one per autotune config or argument specialization). Counted from
the x86 run's cache entries (each `.so` checked to be x86-64).

| | GPU | CPU |
|---|---|---|
| Inductor sources → compiled | 9 → 14 | 11 → 14 |
| vLLM kernels compiled | 29 | 11 |
| **Total compiled** | **43** | **25** |

## Inductor (CPU): 11 sources → 14 compiled

| Source | Kind | Configs (XBLOCK) | Picked on x86 |
|---|---|---|---|
| RMSNorm + embedding `…_embedding_…_rsqrt_0/3/5/6` | `persistent_reduction` | 1 | 1 |
| RMSNorm + residual `…_mean_mul_…_rsqrt_7/8/9/10` | `persistent_reduction` | 1 | 1 |
| RoPE `…_index_select_…_view_1` | `pointwise` | 1024 / 512 | 512 |
| RoPE `…_index_select_…_view_2` | `pointwise` | 512 / 256 | 256 |
| SiLU-mul `triton_poi_fused_mul_onednn_mm_silu_slice_4` | `pointwise` | 1024 / 512 | 512 |

8 reductions × 1 + 3 pointwise × 2 = 14. The 3 extra compiles come only from
pointwise autotuning (GPU: 5 pointwise × 2 + 4 RMSNorm × 1 = 14).

## vLLM (CPU): 9 kernels → 11 compiled

`_apply_write_kernel` is compiled 3 times (argument specializations); the other 8
once each: `_combine_sampled_and_draft_tokens`, `_compute_slot_mappings`,
`_gather_block_tables`, `_get_num_sampled_and_rejected`, `_gumbel_sample`,
`_post_update`, `_prepare_pos_seq_lens`, `_prepare_prefill_inputs`.

## Why the GPU has more

| GPU only | Why |
|---|---|
| `kernel_unified_attention`, `reduce_segments`, `reshape_and_cache_kernel_flash` | `TRITON_ATTN`; the CPU platform only allows C++ `CPU_ATTN` (`vllm/platforms/cpu.py:172`) |
| `_topk_topp`, `_temperature`, `_min_p`, `_penalties`, `_bias`, `_bad_words`, `_bincount`, `_ranks`, `_prompt_logprobs_token_ids`, `_apply_grammar_bitmask`, `_topk_log_softmax` | compiled by the GPU worker's `_dummy_sampler_run` warmup (`vllm/v1/worker/gpu_worker.py:900`); the CPU worker has no such warmup, and the real run (`temperature=0`) never launches them |
| more variants of the shared vLLM kernels | different specializations on GPU |

The Inductor kernels do the same jobs on both (RMSNorm, RoPE, SiLU-mul) but fuse
differently: on CPU vLLM's oneDNN linear sits inside the graph (`onednn_mm` in
the names), on GPU the GEMMs are separate calls (`fused_add_rms_norm_*`).

# Pointwise autotuning and cross-compiling

- `config.triton.autotune_pointwise` defaults to `True`
  (`torch/_inductor/config.py:2059`). The default pointwise heuristic
  (`torch/_inductor/heuristics/triton_codegen/pointwise.py`) gives 1D kernels 2
  configs (`bs`, `bs/2`), 2D kernels 6, 3D kernels 1 (more only with
  `max_autotune`). `persistent_reduction` kernels here get 1 config.
- Inductor benchmarks the configs on first launch on the machine it runs on
  (`triton_heuristics.py:1563`). With cross-compiling that is the x86 host, so the
  XBLOCK used for riscv64 was chosen by x86 timing.
- **Correctness:** not affected. Every config computes the same result, and
  `qwen_rv_run.py` compiles exactly the captured constexprs, so the board runs the
  variant that produced the x86 reference.
- **Performance:** the x86 choice (XBLOCK 256–512) means `<256 x …>`–`<512 x …>`
  vectors on RVV, which compile slowly in `llc` and run slowly on the board.
- **Compile time:** every tuned kernel is compiled once per config on x86.
- **Simple fixes:** tune on the board (compile and time every config, plus smaller
  XBLOCKs such as 64–128, in `qwen_rv_run.py`), and optionally set
  `"autotune_pointwise": False` in `inductor_compile_config` so x86 compiles one
  config per kernel.

# Board results (SpacemiT K1, `chlee@140.114.78.64`)

`scripts/qwen_capture.py` keeps the last launch of every kernel (the real
prefill); `scripts/qwen_rv_run.py` compiles it for riscv64, runs it once on the
board with the captured inputs, compares every buffer the x86 launch changed
(`rtol = atol = 1e-2`), then times it (mean of 10, single thread).

| Kernel | Check | ms | Max error |
|---|---|---|---|
| `_apply_write_kernel` | PASS | 0.002 | 0 |
| `_combine_sampled_and_draft_tokens_kernel` | PASS | 0.000 | 0 |
| `_compute_slot_mappings_kernel` | crash | – | – |
| `_gather_block_tables_kernel` | crash | – | – |
| `_get_num_sampled_and_rejected_kernel` | PASS | 0.000 | 0 |
| `_gumbel_sample_kernel` | PASS | 1.867 | 0 |
| `_post_update_kernel` | PASS | 0.000 | 0 |
| `_prepare_pos_seq_lens_kernel` | PASS | 0.006 | 0 |
| `_prepare_prefill_inputs_kernel` | PASS | 0.002 | 0 |
| RMSNorm + embedding `rsqrt_0` | PASS | 0.885 | 0.0078 |
| RMSNorm + embedding `rsqrt_3` | PASS | 1.569 | 0.0625 |
| RMSNorm + embedding `rsqrt_5` | PASS | 2.396 | 0.0625 |
| RMSNorm + embedding `rsqrt_6` | PASS | 3.375 | 0.0625 |
| RMSNorm + residual `rsqrt_7` | PASS | 1.365 | 1 |
| RMSNorm + residual `rsqrt_8` | PASS | 2.311 | 0.125 |
| RMSNorm + residual `rsqrt_9` | PASS | 3.236 | 0.125 |
| RMSNorm + residual `rsqrt_10` | PASS | 4.633 | 0.25 |
| RoPE `view_1` | pending | | |
| RoPE `view_2` | pending | | |
| SiLU-mul `silu_slice_4` | pending | | |

- **Max error** is absolute, in bf16. The larger values (0.25, 1) are on large
  residual-stream values, where one bf16 rounding step is that big; they are
  within the relative tolerance.
- **Crashes:** `_compute_slot_mappings_kernel` and `_gather_block_tables_kernel`
  read `*_block_table_ptrs`, tables of raw x86 addresses, which are invalid on the
  board. This is a limit of replaying one launch in isolation, not of triton-cpu.
