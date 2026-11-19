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

## Cache entries, compiled kernels and launched kernels

| Run | Device | Cache entries | Compiled kernels | Actual (launched) |
|---|---|---|---|---|
| vLLM (`qwen05b_prefill_seq128.py`) | CPU (x86) | 42 | 25 | 20 |
| | GPU (RTX 3060 Ti) | 44 | 43 | 19 |
| TorchInductor (`qwen_forward_seq128.py`) | CPU (x86) | 35 | 28 | 18 |
| | GPU (RTX 3060 Ti) | 21 | 20 | 14 |

- **Cache entries** = every directory in `TRITON_CACHE_DIR`. Besides one entry
  per compiled kernel, the GPU cache always has one `cuda_utils` driver module,
  and the CPU cache has one `__triton_cpu_launcher.so` per launch signature (17
  for vLLM, 7 for TorchInductor).
- **Compiled kernels** = cache entries that hold a kernel binary (the same as the
  dump entries in `TRITON_DUMP_DIR`), one per autotune config or argument
  specialization.
- **Actual** = distinct kernels launched in the real run, counted with a
  `triton.knobs.runtime.launch_enter_hook` (GPU) or `qwen_capture.py` (CPU).
  vLLM's warmup launches (`_dummy_run`, `profile_run`, `_dummy_sampler_run`,
  `compile_or_warm_up_model`) are excluded; only `LLM.generate` counts.
- **vLLM GPU actual** was measured on the GX10 (GB10, `-p 54323
  chlee@140.114.78.56`; 42 entries / 41 compiled there): the RTX 3060 Ti did not
  have enough free memory. The 19 are 8 Inductor kernels, the same 9
  input-preparation/sampling kernels as on CPU, and `kernel_unified_attention` +
  `reshape_and_cache_kernel_flash`. `reduce_segments` and the sampler kernels
  (`_topk_topp`, `_temperature`, …) are compiled by warmup only; the 128-token
  prefill uses the 2D attention kernel (`MAX_3D_QUERY_LEN = 8`).
- **TorchInductor** is the GPU box's `scripts/qwen_forward.py` (one forward pass,
  `use_cache=False`, HF eager attention, 128 random tokens); the x86 port is
  `scripts/qwen_forward_seq128.py` (bf16 instead of fp16). The 18 vs 14 launched
  kernels differ only in RoPE: CUDA fuses q/k RoPE into one kernel, the CPU Triton
  backend emits five.
- A cache shared by several runs over-counts: `model.sh` runs `qwen_capture.py`
  and `qwen_rv_run.py` with the same `TRITON_CACHE_DIR`, so `scripts/triton_cache`
  also collects the RISC-V builds. Count from a fresh cache.

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
| `_compute_slot_mappings_kernel` | run failed (exit 255) | – | – |
| `_gather_block_tables_kernel` | skipped | – | – |
| `_get_num_sampled_and_rejected_kernel` | PASS | 0.000 | 0 |
| `_gumbel_sample_kernel` | PASS | 1.874 | 0 |
| `_post_update_kernel` | PASS | 0.000 | 0 |
| `_prepare_pos_seq_lens_kernel` | PASS | 0.008 | 0 |
| `_prepare_prefill_inputs_kernel` | PASS | 0.002 | 0 |
| RMSNorm + embedding `rsqrt_0` | PASS | 1.211 | 0 |
| RMSNorm + embedding `rsqrt_3` | PASS | 1.985 | 0 |
| RMSNorm + embedding `rsqrt_5` | PASS | 2.848 | 0 |
| RMSNorm + embedding `rsqrt_6` | PASS | 3.927 | 0 |
| RMSNorm + residual `rsqrt_7` | PASS | 1.750 | 0 |
| RMSNorm + residual `rsqrt_8` | PASS | 2.852 | 0 |
| RMSNorm + residual `rsqrt_9` | PASS | 3.639 | 0 |
| RMSNorm + residual `rsqrt_10` | PASS | 5.171 | 0 |
| RoPE `view_1` | PASS | 1475.972 | 0 |
| RoPE `view_2` | PASS | 207.292 | 0 |
| SiLU-mul `silu_slice_4` | PASS | 8185.040 | 0 |

- **Max error** is absolute, in bf16. In this run every checked kernel matches the
  x86 result exactly.
- **`_compute_slot_mappings_kernel`:** the board run exits with status 255 and no
  stderr. The kernel reads `*_block_table_ptrs`, tables of raw x86 addresses, which
  are invalid on the board. This is a limit of replaying one launch in isolation,
  not of triton-cpu.
- **`_gather_block_tables_kernel`** is skipped: its captured x86 launch changed no
  buffer, so there is nothing to compare, and `qwen_rv_run.py` does not run it.

# Masked loads/stores replaced by `TRITON_VSETVL_CHUNK`

Per launched kernel: how many `vector.maskedload` / `vector.maskedstore` reach
`MemoryOpToLLVM` (blue), and how many `TRITON_VSETVL_CHUNK` lowers to
`vp.load` / `vp.store` instead of `llvm.masked.load` / `llvm.masked.store`
(orange). Solid = loads, hatched = stores.

Counted with `scripts/qwen_rv_masks.py`: it compiles every captured kernel for
riscv64 with `TRITON_VSETVL_CHUNK=1 TRITON_VSETVL_REDUCE=1 TRITON_VSETVL_STATS=1`.
With `TRITON_VSETVL_STATS` set, `MemoryOpToLLVM` prints the masked op counts
before and after its conversion. Plotted with `~/tex/png_script/kernel_mask_count.py`.

| Flow | Kernels | Masked loads/stores | Replaced |
|---|---|---|---|
| vLLM | 20 | 60 | 60 |
| TorchInductor | 18 | 48 | 48 |

- **Every masked op is replaced.** No kernel keeps an `llvm.masked.load` or
  `llvm.masked.store`.
- **RMSNorm** has 3–8 per kernel, and **softmax** has 2.
- **RoPE** (both flows) and **TorchInductor SiLU** have none, so any change in their
  board time is not from mask removal. vLLM SiLU has 1 masked store.

# fp16, bf16 and the RISC-V vector extensions

## Which extensions are fp16 and which are bf16

The fp16 and bf16 extensions are separate families. bf16 has no "full arithmetic"
extension like Zvfh: the ratified set only converts bf16 to/from f32 and does the
widening multiply-add that matmul needs, and all other math is done in f32.

| | fp16 (IEEE half) | bf16 |
|---|---|---|
| scalar, conversions only | Zfhmin | Zfbfmin |
| scalar, full arithmetic | Zfh | none |
| vector, conversions only (also makes f16 vectors a legal type) | Zvfhmin | Zvfbfmin |
| vector, full arithmetic | Zvfh | none ratified |
| vector, widening multiply-add into f32 | covered by Zvfh | Zvfbfwma (`vfwmaccbf16`) |

There is no "Zvfbf". A proposal for more bf16 vector arithmetic (Zvfbfa) exists but is
not ratified.

The board (`chlee@140.114.78.64`, SpacemiT X60) reports in `/proc/cpuinfo`:
`zfh_zfhmin` and `zvfh_zvfhmin`, i.e. **all of fp16, none of bf16** (no `zfbfmin`,
`zvfbfmin` or `zvfbfwma`).

How the backend handles this today:

- **bf16:** `compiler.py` (`promote_bf16_to_fp32`, `decompose_bf16_conv`) does bf16
  memory ops on `i16` and decomposes bf16↔f32 conversions on riscv64, so no bf16
  vector instruction is needed.
- **Codegen target features** are hard-coded in `third_party/cpu/llvm.cc:192` as
  `{"+m", "+f", "+d", "+v", "+zvfbfmin"}`. This **claims Zvfbfmin, which the board
  does not have**, and **omits Zvfh/Zvfhmin, which it does**. The workaround above
  keeps the first from causing trouble; the second is what breaks fp16 with
  `TRITON_VSETVL_CHUNK` (below). The list that matches the board would be
  `{"+m", "+f", "+d", "+v", "+zfh", "+zvfh"}`.

## fp16 with `TRITON_VSETVL_CHUNK` fails in LLVM codegen

`python/tutorials/cpu/rvv_gqa_attn_elf.py` is the `_attn_fwd` kernel of the fused
Qwen2.5 prefill (causal GQA reading a fused QKV buffer), with fp16 I/O like that
pipeline:

| build | result |
|---|---|
| fp16, no `TRITON_VSETVL_*` | PASS on the board |
| fp16, `TRITON_VSETVL_CHUNK=1 TRITON_VSETVL_REDUCE=1` | `LLVM ERROR: Do not know how to scalarize the result of this operator!` (`vp_load v1f16`) |
| fp32 or bf16, `TRITON_VSETVL_CHUNK=1 TRITON_VSETVL_REDUCE=1` | PASS on the board |

CHUNK lowers the masked row loads/stores to `llvm.vp.load.v16f16` /
`llvm.vp.store.v16f16`. Without `+zvfhmin`, f16 vectors are not a legal type, so LLVM
splits the op down to single elements, and it cannot scalarize a VP op. Without CHUNK
the same loads are `llvm.masked.load` / `llvm.masked.store`, which LLVM can legalize
by splitting, so plain fp16 works. bf16 under CHUNK is fine because its
`vp.load` / `vp.store` are on `i16`.

Fixes (neither done yet): add `+zvfhmin` (or `+zvfh`) to the riscv64 feature list in
`llvm.cc`, which matches the board; or have CHUNK/LANE do f16 `vp.load` / `vp.store`
on `i16` and bitcast, as for bf16.

## Why `scripts/model.sh` and `scripts/model_forward.sh` don't hit it

- **Neither script sets any `TRITON_VSETVL_*` flag**, so the kernels they run on
  the board use `llvm.masked.load` / `llvm.masked.store`, which work for every dtype.
- **The kernels they capture are not fp16.** `model_forward.sh` loads the model in
  bf16 (`qwen_forward_seq128.py`: `dtype=torch.bfloat16`). `model.sh` runs
  `qwen05b_prefill_seq128.py`, which asks vLLM for `dtype="float16"` (since
  `c79213223`), yet the 20 kernels captured in `~/qwen_rv` on 2026-11-19 have only
  `*bf16`, `*fp32` and integer pointer arguments. Where vLLM's CPU path turns the
  requested fp16 into bf16 is not traced yet.
- That is also why the CHUNK counts in the previous section (from `qwen_rv_masks.py`,
  which compiles every captured kernel with `TRITON_VSETVL_CHUNK=1`) built without
  errors: there was no fp16 kernel among them. A real fp16 kernel compiled with
  CHUNK would fail as above until the feature list is fixed.
