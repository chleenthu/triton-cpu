# vLLM on CPU: what does not use Triton

| Op | CPU path |
|---|---|
| Attention | `CPU_ATTN` / `CPU_MLA` / `AMX_MLA`, all C++ (`vllm/platforms/cpu.py:138`) |
| Linear / GEMM | torch / oneDNN |
| MoE | `experts/cpu_moe.py`, `cpu_int4_moe.py` (C++/torch) |
| Mamba / SSM | explicit `is_cpu()` overrides in `mamba/ops/*` |
| Norm / RoPE / activation / fusions | vLLM C++ custom ops, or torch.compile with Inductor's C++ code generator (`cpp.*` options, `vllm/platforms/cpu.py:306`) |

# Captured triton-cpu kernel launches (Qwen2.5-0.5B-Instruct, bf16)

vLLM run with Inductor `cpu_backend="triton"`; one real 128-token prefill plus
decode steps were captured. A *launch* is one recorded call of a kernel with a
distinct argument shape (`__0` = first shape seen, `__1`/`__2` = later ones);
each launch is built into its own riscv64 ELF with its real inputs embedded.

## vLLM V2 model runner and sampler kernels (12 launches, 9 kernels)

| Launch | Status |
|---|---|
| `_apply_write_kernel` `__0` `__1` `__2` | built |
| `_combine_sampled_and_draft_tokens_kernel__0` | built |
| `_compute_slot_mappings_kernel` `__0` `__1` | built |
| `_gather_block_tables_kernel__0` | built |
| `_get_num_sampled_and_rejected_kernel__0` | built |
| `_gumbel_sample_kernel__0` (sampling over the 151,936-token vocab) | built |
| `_prepare_pos_seq_lens_kernel__0` | built |
| `_prepare_prefill_inputs_kernel__0` | built |
| `_post_update_kernel__0` | skipped: 39 M elements |

## Inductor-generated model kernels (22 launches, 11 kernels, each prefill + decode)

| Kernel family | Launches | Status |
|---|---|---|
| RMSNorm + residual add (`triton_per_…_mean_mul_…_rsqrt_7/8/9/10`) | 8 | built |
| RoPE (`triton_poi_…_index_select_…_view_1/2`) | 4 | built |
| SiLU-and-mul (`triton_poi_fused_mul_onednn_mm_silu_slice_4`) | 2 | built |
| RMSNorm fused with the embedding lookup (`…_embedding_…_0/3/5/6`) | 8 | skipped: ~136 M elements |

# Board results (SpacemiT K1, 25 runnable launches)

Each ELF runs once untimed to check its output against the x86 result, then is
timed (mean of 100 iterations after 10 warmup) with 1 and 8 OpenMP threads.
Times are ms per launch.

| Kernel | Launch | Check | 1 thread | 8 threads |
|---|---|---|---|---|
| `_apply_write_kernel` | `__0` / `__1` / `__2` | PASS | 0.0023 | 0.026 – 0.030 |
| `_combine_sampled_and_draft_tokens_kernel` | `__0` | PASS | 0.0002 | 0.030 |
| `_get_num_sampled_and_rejected_kernel` | `__0` | PASS | 0.0002 | 0.024 |
| `_gumbel_sample_kernel` | `__0` | PASS | 1.81 | 0.35 |
| `_prepare_pos_seq_lens_kernel` | `__0` | PASS | 0.0064 | 0.026 |
| `_prepare_prefill_inputs_kernel` | `__0` | PASS | 0.0023 | 0.025 |
| RMSNorm + residual `rsqrt_7` | prefill / decode | PASS | 1.91 / 0.089 | 0.47 / 0.094 |
| RMSNorm + residual `rsqrt_8` | prefill / decode | PASS | 2.50 / 0.010 | 0.47 / 0.026 |
| RMSNorm + residual `rsqrt_9` | prefill / decode | PASS | 3.47 / 0.014 | 0.56 / 0.027 |
| RMSNorm + residual `rsqrt_10` | prefill / decode | PASS | 4.65 / 0.018 | 0.75 / 0.023 |
| SiLU-and-mul `silu_slice_4` | prefill / decode | PASS | 7780 / 62.5 | 1117 / 14.1 |
| RoPE `view_1` | prefill / decode | PASS* | 1449 / 11.5 | 206 / 11.5 |
| RoPE `view_2` | prefill / decode | PASS* | 212 / 2.55 | 30.1 / 2.54 |
| `_compute_slot_mappings_kernel` | `__0` / `__1` | segfault | – | – |
| `_gather_block_tables_kernel` | `__0` | segfault | – | – |

- **RoPE PASS\*:** the generated `riscv.py` runner reported FAIL, but that was the
  replay, not triton-cpu. In vLLM `out_ptr1 == out_ptr0 + 32` (q/k halves
  interleaved in one buffer), so each output's captured region contains the
  other's writes in its gaps, while the runner gives every pointer its own array.
  Rerun with the real inputs and checked only on the elements each pointer writes
  (`x0 + 64*x3`), all 4 launches match x86 (0 wrong of 57,344 / 448 / 8,192 / 64,
  within bf16 rounding). Still slow on the board, and `llc -O3` on the prefill
  `view_2` IR (`<512 x …>` vectors) takes minutes.
- **SiLU-and-mul** is correct but very slow: 1.1 s for 622,592 elements at 8
  threads (x86 does a whole 128-token prefill in 82 ms).
- **Segfaults** are a limit of the standalone replay, not triton-cpu: these
  kernels read `*_block_table_ptrs`, tables of raw x86 addresses.
- Tiny decode kernels are faster single-threaded (OpenMP startup ≈ 20 µs).

# Routing Attention, GEMM, MoE and Mamba through Triton (x86, experiment)

vLLM already has Triton versions of all four; on CPU it just never selects them.
Each was forced on with monkeypatches (no vLLM source change) and run through
triton-cpu with `TRITON_CPU_TARGET=native`.

| Op | What was done | Result |
|---|---|---|
| Attention | Forced vLLM's Triton attention backend (`TRITON_ATTN`: `unified_attention` + `triton_reshape_and_cache_flash`) instead of `CPU_ATTN` | Works end to end in Qwen2.5-0.5B, coherent text ("Paris…"). Eager: 1.34 s vs 1.00 s (C++) for 2×32 tokens. |
| GEMM | `scripts/patch_inductor_cpu_triton_gemm.py` ported to torch 2.13, Linear switched to plain `torch.nn.functional.linear` | Every projection in vLLM's compiled graph is a Triton GEMM template, with neighbours fused in (q/k/v + RMSNorm + residual, gate/up + SiLU-mul, o_proj, down_proj). A 37×896 by 4864×896 linear matches torch within a few bf16 rounding steps. |
| MoE | Called vLLM's Triton fused MoE (`fused_experts_impl`) directly, with torch stand-ins for `moe_sum` / `moe_align_block_size` | Correct on 3 shapes (max error 0.001–0.007). About 80 s compile per shape. |
| Mamba | Triton ops compared with vLLM's CPU versions and a plain-torch reference | `selective_state_update` correct (2e-5); `causal_conv1d_update` exact. `mamba_chunk_scan` (prefill) not checked: see below. |

Attention + GEMM + Inductor all on Triton, Qwen2.5-0.5B bf16: engine init 835 s
(vs 21 s), generate 2×32 tokens 1.72 s (vs 1.00 s, C++ eager). Text is coherent;
wording drifts after ~15 tokens ("second largest" → "third largest"), as
expected from a different bf16 accumulation order. Accuracy not measured.

## What each needs to work with triton-cpu

- **Attention:** the CPU platform only ever returns `CPU_ATTN`
  (`vllm/platforms/cpu.py:172`); it needs a way to allow `TRITON_ATTN`. Also
  `triton_reshape_and_cache_flash.py:424` asks for the CUDA compute capability in
  its non-ROCm branch, which crashes on CPU; it should only run on CUDA.
- **GEMM:**
  - vLLM sends CPU linears to oneDNN (`vllm/model_executor/layers/utils.py:426`),
    an opaque op Inductor can't see into. It needs an option to use plain
    `F.linear`.
  - Inductor won't emit Triton GEMMs on CPU without the patch. For torch 2.13 two
    of its edits move: `heuristics/template/gemm.py` is
    `template_heuristics/gemm.py`, and `pick_deterministic_choice` starts with
    `assert len(choices) >= 2` instead of raising "expected at least 2 choices".
  - It also needs the `CpuInterface.device` no-op from `scripts/build_qwen_engine.py`.
  - `lm_head` runs outside the compiled graph (`compute_logits`), so it stays a
    torch matmul.
- **MoE:**
  - The CPU platform only offers the C++ MoE backend
    (`vllm/model_executor/layers/fused_moe/oracle/unquantized.py:97`); the Triton
    backend has to be allowed there.
  - The CPU build has no `_moe_C`, so `moe_sum` and `moe_align_block_size` need
    torch or Triton versions. `silu_and_mul` is already in the CPU `_C`.
- **Mamba:** the Triton versions exist, but vLLM swaps in CPU versions on CPU
  (`mamba_ssm.py:852`, `causal_conv1d.py:1282`, `ssd_combined.py:232`,
  `ssu_dispatch.py:484`).
  - `mamba_chunk_scan` is blocked by compile time: `_chunk_scan_fwd_kernel`
    autotunes over ~9 GPU-sized configs (`ssd_chunk_scan.py:17`), each 4–11 MB of
    LLVM IR; the third was still in LLVM codegen after 30 min. Like the GEMM patch,
    CPU needs one small fixed config instead of autotuning (the other SSD kernels
    autotune too).
- Qwen2.5-0.5B has no MoE or mamba layers, so those two were only checked with
  kernel-level tests.

# Simple setup: vLLM + triton-cpu in one venv

Everything runs from `~/triton-cpu/.venv` (torch `2.14.0+cpu`, triton-cpu
editable). `~/vllm/.venv` is no longer used.

## Environment

- **torch:** `torch/_inductor` and `torch/_dynamo` in the venv are symlinks to
  `~/pytorch/torch/...` (branch `triton-cpu-2.14.0` at tag `v2.14.0`), so Inductor
  edits take effect without reinstalling. The originals are kept as
  `_inductor.orig` / `_dynamo.orig`.
- **vLLM:** `~/vllm` (main) installed editable for CPU:

  ```bash
  source ~/triton-cpu/.venv/bin/activate
  cd ~/vllm && python use_existing_torch.py
  pip install -r requirements/build/cpu.txt -c constraints.txt --extra-index-url https://download.pytorch.org/whl/cpu
  pip install -r requirements/cpu.txt       -c constraints.txt --extra-index-url https://download.pytorch.org/whl/cpu
  CC=<gcc>=12.3> CXX=<g++>=12.3> CFLAGS=-I<inc> CXXFLAGS=-I<inc> LDFLAGS=-L<lib> \
    VLLM_TARGET_DEVICE=cpu MAX_JOBS=16 pip install -e . --no-build-isolation -c constraints.txt
  ```

  - `constraints.txt` pins `torch==2.14.0+cpu` and `triton==3.8.0+cpu.git9c1ee479`
    so pip cannot replace them.
  - The x86 build needs gcc ≥ 12.3; the system has 11.4, so gcc 13.4 from the
    conda env `~/.conda/envs/vllm` is used.
  - That compiler does not search `/usr/include`, so `<inc>` / `<lib>` hold links
    to only `numa.h`, `numaif.h`, `numacompat1.h`,
    `x86_64-linux-gnu/python3.10/pyconfig.h` and `libnuma.so`.
  - The built `_C*.so` need only `GLIBCXX_3.4.21`, so the system libstdc++ works.

## Running `scripts/qwen05b_prefill_seq128.py`

Changes from the GPU version: `inductor_compile_config={"cpu_backend": "triton"}`,
`dtype="bfloat16"`, no `gpu_memory_utilization`, `--attention-backend` defaults
to vLLM's CPU choice, and `TRITON_CPU_TARGET=native` is set before any import.

```bash
cd ~/triton-cpu/scripts
CC=~/llvm-project/install/bin/clang TRITON_LOCAL_LIBOMP_PATH=$HOME/.triton-native-libomp OMP_NUM_THREADS=8 VLLM_CPU_KVCACHE_SPACE=4 \
  ../.venv/bin/python qwen05b_prefill_seq128.py
```

- `CC`: native mode passes clang-only flags (`--target=…`, `-fuse-ld=lld`); with
  the system gcc the first kernel fails with
  `gcc: error: unrecognized command-line option '--target=x86_64-unknown-linux-gnu'`.
- `TRITON_LOCAL_LIBOMP_PATH`: the project clang ships no host `omp.h` / libomp;
  this points the launcher build at `~/.triton-native-libomp` (as in `g.sh`), and
  `OMP_NUM_THREADS` sets the launcher's thread count. `TRITON_DISABLE_OPENMP=1`
  also works, with single-threaded launches.
- `VLLM_CPU_KVCACHE_SPACE` (GB) replaces `gpu_memory_utilization`.

Result: 128 prompt tokens, 1 generated token. Engine init 484 s, almost all
triton-cpu kernel compilation; later runs reuse `~/.cache/vllm/torch_compile_cache`.

Without `TRITON_CPU_TARGET=native` the kernels and launcher are built as riscv64
ELFs, which the x86 process cannot load (`ImportError: …__triton_cpu_launcher.so:
cannot open shared object file`). triton-cpu's riscv64 target only produces
binaries; running them needs the board.

## Testing the kernels on the board

1. `scripts/qwen_capture.py` (x86): runs `qwen05b_prefill_seq128.main()` with a
   wrapped `CPULauncher` and keeps the last launch of every Triton kernel (the real
   prefill) in `~/qwen_rv/<kernel>/`: `kernel.py`, `meta.pkl` (grid, signature,
   scalar and constexpr values), `in<i>.bin` (every pointer argument before the
   launch), `ref<i>.bin` (every `out_ptr*` / `in_out_ptr*` after it).
2. `scripts/qwen_rv_run.py`: for each kernel, compiles it for riscv64 (default
   target), writes a small C runner that reads `in<i>.bin`, loops over the grid
   and writes `out<i>.bin`, builds it with `build_standalone_executable`, then
   `scp`/`ssh` to `chlee@140.114.78.64:~/qwen_rv/<kernel>/`, runs it, copies the
   outputs back and compares them with x86 (`rtol = atol = 1e-2`, bf16). Prints
   PASS + ms per launch (mean of 10) + max error, or the mismatch.

Buffers are passed as files, not C array literals as in `riscv.py`'s
`generate_runner`, so the 136 M-element embedding table works.

```bash
cd ~/triton-cpu/scripts
CC=~/llvm-project/install/bin/clang TRITON_LOCAL_LIBOMP_PATH=$HOME/.triton-native-libomp OMP_NUM_THREADS=8 VLLM_CPU_KVCACHE_SPACE=4 ../.venv/bin/python qwen_capture.py
CC=~/llvm-project/install/bin/clang ../.venv/bin/python qwen_rv_run.py [kernel ...]
```

## Kernels: CPU (bf16) vs GPU (`chlee@140.114.78.168:~/serve`, RTX 3060 Ti, fp16, `TRITON_ATTN`)

The GPU Triton cache has 44 entries, **31 unique kernels**; the CPU capture has
**20**. Shared: vLLM's input-preparation and sampling kernels
(`_apply_write_kernel`, `_compute_slot_mappings_kernel`, `_gumbel_sample_kernel`, …).

| GPU only | Why |
|---|---|
| `kernel_unified_attention`, `reduce_segments`, `reshape_and_cache_kernel_flash` | `TRITON_ATTN`; the CPU platform forces C++ `CPU_ATTN` (`vllm/platforms/cpu.py:172`) |
| `_topk_topp`, `_temperature`, `_min_p`, `_penalties`, `_bias`, `_bad_words`, `_bincount`, `_ranks`, `_prompt_logprobs_token_ids`, `_apply_grammar_bitmask`, `_topk_log_softmax` | compiled by the GPU worker's `_dummy_sampler_run` warmup (`gpu_worker.py:900`); the CPU worker has none, and the real run (`temperature=0`) never launches them. The GPU list counts compiled kernels, the CPU capture counts launched ones. |
| 8 Inductor kernels (`fused_add_rms_norm_*`, `mul_silu_slice_1`, RoPE `add_cat_index_select_…_1..4`, `embedding_rms_norm_0`) | same jobs as the 11 CPU Inductor kernels (`…_onednn_mm_pow_rsqrt_*`, `mul_onednn_mm_silu_slice_4`, RoPE `index_select_…_1/2`); on CPU vLLM's oneDNN linear sits inside the graph, so fusion and names differ |
