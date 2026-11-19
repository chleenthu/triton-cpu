# Sinking vector loads to their first use (`TRITON_SINK_LOADS`)

## Motivation

vLLM RMSNorm kernel `triton_per_fused__to_copy_add_embedding_mean_mul_onednn_mm_pow_rsqrt_3`
on the SpacemiT K1 board (riscv64, VLEN 256), mean of 3 interleaved rounds. "Index loaded
first" is the same kernel source with two `tl.load` lines swapped by hand.

| Version | Baseline ms | CHUNK+REDUCE ms | Speedup over original baseline |
|---|---|---|---|
| Original order | 4.24 | 3.28 | 1.29 |
| Index loaded first | 3.61 | 2.14 | 1.98 |

With the loads swapped, the vLLM kernel matches its TorchInductor twin
(`..._embedding_mean_mul_pow_rsqrt_9`, 2.27 ms with CHUNK+REDUCE). That explains why
TorchInductor RMSNorm gains 1.6–2.3× from `TRITON_VSETVL_CHUNK=1 TRITON_VSETVL_REDUCE=1`
while vLLM RMSNorm gains only 1.05–1.34×.

## Cause

The two kernels compute the same thing (embedding gather + residual add, sum of squares,
`rsqrt`, multiply by weight; 128 rows of 896 bf16, `R0_BLOCK = 1024`). The vLLM source loads
the residual row before the per-row token index:

```python
tmp0 = tl.load(in_ptr0 + (r0_1 + 896*x0), r0_mask & xmask, ...)   # residual row, 1024 x bf16
tmp2 = tl.load(in_ptr1 + (x0), xmask, ...)                          # token index, scalar
...
tmp5 = tl.load(in_ptr2 + (r0_1 + 896*tmp3), r0_mask & xmask, ...)  # embedding row (gather)
tmp7 = tmp1 + tmp6
```

TorchInductor loads the index first, then the residual row.

The scalar index load is masked by `xmask`, so triton-cpu lowers it to an `scf.if`, which
becomes a branch in LLVM. In the vLLM order the residual row is loaded before that branch
and is live across it. As i16 the row is 1024 × 16 bits = 8 m8 register groups (64 vector
registers, against 32 in the machine). LLVM does not move the load below the branch, so most
of the row is spilled and reloaded with whole-register `vs8r.v` / `vl8r.v`.

## Evidence

riscv64 compile of each kernel with CHUNK+REDUCE, counting whole-register m8 spills and
reloads in the assembly. Only the order of the two loads changes:

| Kernel (CHUNK+REDUCE) | `vs8r` | `vl8r` |
|---|---|---|
| vLLM, original order | 281 | 320 |
| vLLM, index loaded first | 133 | 152 |
| TorchInductor, original order | 133 | 152 |
| TorchInductor, residual loaded first | 281 | 320 |

Figure: `~/tex/fig/sink_loads_spills.png` (plotted with
`~/tex/png_script/spill_order.py json/sink_loads_spills.json`).

Swapping the loads swaps the counts exactly, in both directions. These source differences
leave the vLLM count at 281 / 320:

- constant `xnumel = 128` (TorchInductor's kernel hardcodes it; vLLM passes it at run time);
- operand order of the residual add;
- operand order of the weight multiply;
- removing `tl.device_assert`.

`xnumel` was also checked on the board: hardcoding it changes neither kernel's speedup
beyond noise.

## Fix

`TRITON_SINK_LOADS=1` adds a step at the end of the `OptimizeMasks` pass
(`third_party/cpu/lib/TritonCPUTransforms/OptimizeMasks.cpp`, tttcir stage). For every block,
bottom-up, it moves each op that

- has no regions and at least one vector result, and
- is pure, or only reads memory,

to just before its first user in the same block. A reading op is never moved past an op
that may write memory (asserts, stores, calls, unknown ops). Going bottom-up lets a load
follow the conversions computed from it (`bf16` → `i16` load, then the shifts to `f32`), so
the whole chain lands next to the add that uses it. The `scf.if` of the index load then
comes before the residual row, as in the TorchInductor kernel.

The flag works on its own or together with the `TRITON_VSETVL_*` flags.

## tttcir before and after

The fix runs in `OptimizeMasks`, the first pass of `make_tttcir`, so its effect is visible
in the tttcir dump. Files in `~/tsvc/kernel/`, both compiled for riscv64 with
`TRITON_VSETVL_CHUNK=1 TRITON_VSETVL_REDUCE=1`; only `TRITON_SINK_LOADS` differs:

- `triton_per_fused__to_copy_add_embedding_mean_mul_onednn_mm_pow_rsqrt_3.py`: the kernel
  source, unchanged (the fix is in the compiler);
- `….before.tttcir`: without `TRITON_SINK_LOADS`;
- `….after.tttcir`: with `TRITON_SINK_LOADS=1`.

Who is who in the IR:

| Python | tttcir | What |
|---|---|---|
| `tmp0 = tl.load(in_ptr0 + (r0_1 + 896*x0), r0_mask & xmask)` | `vector.maskedload … memref<1024xi16> … into vector<1024xi16>` | residual row, 1024 × bf16 |
| `tmp2 = tl.load(in_ptr1 + (x0), xmask)` | `scf.if` around `tt.load … !tt.ptr<i32>` yielding `vector<1x1xi32>` | token index, one i32 per row |
| `tmp5 = tl.load(in_ptr2 + (r0_1 + 896*tmp3), r0_mask & xmask)` | `vector.maskedload`, address from the index | embedding row (gather) |
| `tmp20 = tl.load(in_ptr3 + (r0_1), r0_mask)` | `vector.maskedload` with a `vector.constant_mask` | RMSNorm weight row |

The index load becomes an `scf.if` because it is a scalar load masked by `xmask`:

```mlir
%tmp2_31 = scf.if %tmp2_30 -> (vector<1x1xi32>) {
  %tmp2_62 = tt.load %tmp2_28 {...} : !tt.ptr<i32>
  %tmp2_63 = vector.broadcast %tmp2_62 : i32 to vector<1x1xi32>
  scf.yield %tmp2_63 : vector<1x1xi32>
} else {
  scf.yield %tmp2 : vector<1x1xi32>
}
```

Position of each load (line numbers):

| | before.tttcir | after.tttcir |
|---|---|---|
| Residual row `tmp0` | 42 | 72 |
| Index `scf.if` `tmp2` | 52 | 40 |
| Weight row `tmp20` | 61 | 92 |
| Embedding gather `tmp5` | 79 | 65 |
| `tmp7 = tmp0 + tmp5` (`arith.addf`) | after the gather | 79 |

Both files contain the same 98 top-level ops; nothing is added, removed or changed, only
the order differs. The SSA names differ (e.g. `%tmp0_21` → `%tmp0_43`) because MLIR
renumbers values after ops move; the `tmp` prefixes and `loc` still trace each value to
its Python variable. Three groups of ops move:

1. **Residual row `tmp0`**: its `vector.maskedload` and the bf16 → f32 chain after it
   (`arith.bitcast`, `vector.shape_cast`, `arith.extui`, `arith.shli`, `arith.bitcast`), from
   before the index `scf.if` to just before the `tmp7` `arith.addf`. This removes the
   spills.
2. **Weight row `tmp20`**: its `vector.maskedload` and the same conversion chain, from
   before the gather to after the reduction, just before the final multiply. This
   shortens its live range too.
3. **Small vector ops computing addresses and masks**: a few `vector.create_mask`,
   `vector.shape_cast` and `vector.broadcast` ops and the unnamed `%0` and `%1`, each moved
   to just before its user. They are `vector<1x1>` values or masks and do not change
   register pressure.

Scalar ops (`tt.addptr`, integer address arithmetic, `tt.ptr_to_int` / `tt.int_to_ptr`,
`ptr_to_memref`) and the `scf.if` itself do not move; the `scf.if` only appears earlier
because the residual load moved down past it.

## Compile results

`vs8r` / `vl8r` per kernel, riscv64:

| Build | vLLM `rsqrt_3` | TorchInductor `rsqrt_9` |
|---|---|---|
| No flags | 385 / 455 | 339 / 401 |
| `TRITON_SINK_LOADS` | 339 / 400 | 339 / 400 |
| CHUNK+REDUCE | 281 / 320 | 133 / 152 |
| SINK + CHUNK+REDUCE | 133 / 152 | 133 / 152 |

With the flag, the vLLM kernel spills exactly as much as the TorchInductor kernel, and the
TorchInductor kernel is unchanged.

## Board results (20 vLLM kernels)

All 20 vLLM kernels, four variants, 3 interleaved rounds (each round runs baseline,
CHUNK+REDUCE, SINK, SINK + CHUNK+REDUCE back to back). Every kernel passes the x86 output
check in every round. Times are the mean of the 3 rounds; speedup is over baseline.

| Kernel | Baseline ms | CHUNK+REDUCE | SINK | SINK + CHUNK+REDUCE |
|---|---|---|---|---|
| RMSNorm `rsqrt_0` | 2.39 | 2.28 (1.05) | 2.42 (0.99) | 1.90 (**1.26**) |
| RMSNorm `rsqrt_3` | 4.25 | 3.23 (1.32) | 3.65 (1.17) | 2.13 (**2.00**) |
| RMSNorm `rsqrt_5` | 5.99 | 5.16 (1.16) | 5.33 (1.12) | 2.76 (**2.17**) |
| RMSNorm `rsqrt_6` | 7.72 | 5.99 (1.29) | 7.79 (0.99) | 4.51 (**1.71**) |
| RMSNorm `rsqrt_7` | 3.74 | 2.94 (1.28) | 3.88 (0.96) | 2.37 (**1.58**) |
| RMSNorm `rsqrt_8` | 5.66 | 4.32 (1.31) | 5.69 (0.99) | 2.87 (**1.97**) |
| RMSNorm `rsqrt_9` | 7.49 | 6.03 (1.24) | 7.57 (0.99) | 3.65 (**2.05**) |
| RMSNorm `rsqrt_10` | 9.04 | 8.00 (1.13) | 9.75 (0.93) | 4.57 (**1.98**) |
| `_gumbel_sample`\* | 2.23 | 1.66 (1.34) | 2.14 (1.04) | 1.67 (1.33) |
| SiLU `silu_slice_4` | 8,756.7 | 7,750.9 (1.13) | 8,694.2 (1.01) | 7,662.3 (1.14) |
| RoPE `view_1` | 1,506.2 | 1,505.3 (1.00) | 1,462.6 (1.03) | 1,480.9 (1.02) |
| RoPE `view_2` | 210.5 | 209.8 (1.00) | 205.0 (1.03) | 205.4 (1.02) |
| **Total (20 kernels)** | **10,521.9** | **9,505.6 (1.107)** | **10,410.1 (1.011)** | **9,375.1 (1.122)** |

\* `_gumbel_sample` CHUNK+REDUCE and SINK + CHUNK+REDUCE: mean of a separate 5-round
interleaved re-time of this kernel alone (see below); baseline and SINK are the 3-round means.

- **RMSNorm with SINK + CHUNK+REDUCE: 1.26–2.17×**, up from 1.05–1.32× with CHUNK+REDUCE
  alone. This is the range of the TorchInductor RMSNorm kernels (1.6–2.3×).
- **SINK alone barely helps (0.93–1.17×).** Without CHUNK the masked loads are still
  lowered to `llvm.masked.load` and the spills only drop from 385 / 455 to 339 / 400; the
  two optimizations work together.
- **SiLU and RoPE** are unchanged by SINK (1.13 → 1.14 for SiLU). They are pointwise
  kernels without a long-lived row, so sinking has nothing to shorten. Since SiLU is
  83% of the total time, the 20-kernel total only moves from 1.107× to 1.122×.
- **`_gumbel_sample` is not slower with SINK.** In the 3 rounds SINK + CHUNK+REDUCE measured
  2.509, 1.748 and 1.654 ms against 1.727, 1.688 and 1.654 for CHUNK+REDUCE; the first round
  was an outlier (likely other load on the shared board) and pulled the mean to 1.97 ms.
  A 5-round interleaved re-time of this kernel alone gives 1.663 ms for CHUNK+REDUCE and
  1.674 ms for SINK + CHUNK+REDUCE, within noise. SINK does cut its spills
  (`vs8r` / `vl8r` 1130 / 1772 → 910 / 1555), but the kernel is dominated by scalar code:
  2,048 libm calls (two logs per element for the Gumbel noise) and the scalar address
  arithmetic around them.
- The six bookkeeping kernels at about 0.01 ms or below (e.g. `_prepare_pos_seq_lens` at
  5.00×) are under the timer's resolution and are left out of the table.
