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
