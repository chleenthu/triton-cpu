# Results of the VSETVL lowering fixes (board)

Three kernels from triton-cpu `python/tutorials/cpu/` went wrong or slow only with `TRITON_VSETVL_MINE=1 TRITON_VSETVL_REDUCE=1` (in `DocResult.md`'s tables): 07 gemv bf16 failed verification, 05 `_layer_norm_bwd_dwdb` blew up to 8391 spills and 3.07 ms, and lqcd hop_t_m / hop_t_p ran 1.53 ms / 0.925 ms. The causes and fixes are in triton-cpu, not in our pass:

- `REDUCE` turns on `useMultiDimReductionOp` (`third_party/cpu/backend/compiler.py:208`), so `ConvertReductionOp` maps `tl.sum` to `vector.multi_reduction` instead of the default shuffle lowering. Fixed in `mapToReductionOp` / `mapToMaskedReduction` (`third_party/cpu/lib/TritonToTritonCPU/ConvertReductionOp.cpp`).
- `MINE` moves every tail-masked load / store through a stack slot with a vsetvli copy loop. Fixed in `VsetvlMaskedLoadOpConversion` / `VsetvlMaskedStoreOpConversion` (`third_party/cpu/lib/TritonCPUToLLVM/MemoryOpToLLVM.cpp`).

The kernels are compiled through Triton (`TRITON_RISCV_LMUL=8`) without `--custom-*` flags unless stated, and run on the board. "VSETVL" is `TRITON_VSETVL_MINE=1 TRITON_VSETVL_REDUCE=1`; `MINE`, `REDUCE` and `LANE` are each `TRITON_VSETVL_*` variable alone. Max VP and SLIL are `ExpandPseudos`' own numbers; spills and reloads are `Folded Spill` / `Folded Reload` in the kernel's assembly. Two times are two runs. The first run after the board has been idle is slow, so every series starts with a warm-up run that is not reported.

| kernel | config | max VP | SLIL | spills / reloads | board |
|---|---|---|---|---|---|
| 07 gemv bf16 | baseline | 262 | 64985 | 178 / 240 | PASS 0.86 ms, PASS 0.89 ms |
| | VSETVL, before | 132 | 73499 | 1097 / 1116 | FAIL 1.47 ms |
| | VSETVL, after | 262 | 64985 | 178 / 240 | **PASS 0.81 ms** |
| 05 layer_norm_bwd_dwdb | baseline | 1050 | 206636 | 694 / 566 | PASS 0.142 ms |
| | VSETVL, before | 2032 | 20494772 | 8391 / 8884 | PASS 2.12 ms |
| | VSETVL, after | 2032 | 7072 | **574 / 476** | **PASS 0.216 ms** |
| hop_t_m | baseline | 409 | 111600 | 171 / 255 | PASS 1.02 ms |
| | VSETVL, before | 240 | 5488 | 128 / 215 | PASS 1.46 ms, PASS 1.55 ms |
| | VSETVL, after | 368 | 55808 | 138 / 194 | **PASS 1.00 ms, PASS 1.02 ms** |
| hop_t_p | baseline | 313 | 59184 | 136 / 201 | PASS 0.74 ms |
| | VSETVL, before | 224 | 5112 | 103 / 182 | PASS 0.89 ms, PASS 0.94 ms |
| | VSETVL, after | 320 | 34472 | 118 / 172 | **PASS 0.71 ms, PASS 0.68 ms** |

- 07 gemv bf16 passes again, and its VSETVL build is now the same assembly and LLVM IR as the baseline.
- 05 dwdb loses the 8391 spills: `REDUCE` alone is back at the baseline (0.142 ms). The remaining gap of VSETVL (0.216 ms against 0.142 ms) is `MINE`'s copy loops, which the `MINE` fix does not cover for dwdb's 128-element rows.
- hop_t_m and hop_t_p are back at baseline speed and keep 20–25% fewer spills / reloads than the baseline.
- With all `--custom-*` flags on top of VSETVL after the fixes: 07 bf16 PASS 1.04 ms (175 / 261, 41 reverse remats, max VP 262 → 148), 05 dwdb PASS 0.192 ms (same code as without the flags).
- The 2.48 ms baseline of 07 bf16 in `DocResult.md` was a slow first run: the same binary measured 0.86 and 0.89 ms when rerun.

## 07 gemv bf16: wrong result with `REDUCE`

`gemv_kernel` computes `acc += tl.sum(a * x[None, :], axis=1)` on `[16, 64]` BF16 tiles. The test checks the result exactly (`atol 0`) against a reference that models the backend's BF16 arithmetic: each product and each step of the per-block butterfly is truncated to BF16 (rounded toward zero, `vand 0xffff0000`). With VSETVL it failed with `Y[0]` got 67, expected 66.5, with or without the `--custom-*` flags.

| env | board | `fadd.s` / `__truncsfbf2` | `vfadd` |
|---|---|---|---|
| baseline | PASS 0.86 ms | 0 / 0 | 97 |
| `MINE` | PASS 0.81 ms | 0 / 0 | 97 |
| `REDUCE` | FAIL 1.47 ms | 1024 / 1024 | 1 |

- **Cause:** `REDUCE` makes the reduction a `vector.multi_reduction`, which becomes 16 × `llvm.vector.reduce.fadd.v64bf16` (reassoc). RVV has no BF16 vector arithmetic without Zvfbfa, so LLVM scalarizes each one into a sequential chain of `vslidedown` → `fadd.s` → `call __truncsfbf2` (1024 of each, 512 `vslidedown`). That changes the order (a linear sum instead of the halving butterfly) and the rounding (`__truncsfbf2` rounds to nearest even at every step; the default path truncates). The products `a * x` are the same in both builds (`vfmul` in FP32, then `vnsrl` by 16, which truncates).
- **Not a miscompile:** the result is a different, valid BF16 rounding. A torch model of the `REDUCE` arithmetic (sequential order, round to nearest even after each add) gives `Y[0] = 67` and differs from the test's reference in 119 of 128 rows. Used as the expected value on the board, the model passes all 128 outputs exactly. The scalar chain, with a library call per element, is also why `REDUCE` was slower.
- **Fix:** floats narrower than 32 bits (`isNarrowFloat`: BF16, F16) never take the `vector.multi_reduction` path or the masked `vector.reduction` path of `mapToMaskedReduction`. They fall back to the shuffle butterfly, which computes in FP32 vectors and keeps the backend's own rounding. The 1-D `vector.reduction` path used without `REDUCE` is unchanged, so the default build does not change.
- **Option not taken:** extending to FP32 before the reduction and rounding to BF16 once at the end. That vectorizes (`vfredusum`) and is more accurate, but it changes the result, so the test's reference would have to change too.

After the fix, `REDUCE` passes at 0.82 ms (was FAIL 1.47 ms), and the baseline, `REDUCE` and VSETVL builds are byte-for-byte the same assembly and LLVM IR. With all `--custom-*` flags on top, the 41 reverse remats lower max VP and SLIL, but reloads go up (240 → 261) and the kernel is slower (about 0.81 → 1.04 ms).

## 05 layer norm dwdb: 8391 spills with `REDUCE`

`_layer_norm_bwd_dwdb` accumulates two `[BLOCK_SIZE_M, BLOCK_SIZE_N] = [32, 128]` FP32 tiles, `dw` and `db`, and ends with `tl.sum(dw, axis=0)` and `tl.sum(db, axis=0)`, a reduction along the outer axis. Only this kernel runs; every run passes.

| env | max VP | SLIL | spills / reloads | asm lines | `shufflevector` / `reduce.fadd` (LLVM IR) | board |
|---|---|---|---|---|---|---|
| baseline | 1050 | 206636 | 694 / 566 | 8643 | 34 / 0 | PASS 0.137 ms |
| `MINE` | 2032 | 7072 | 574 / 476 | 10359 | 0 / 0 | PASS 0.202 ms |
| `REDUCE` | 2583 | 34656696 | **8580 / 8967** | 125411 | **15649 / 257** | PASS **2.46 ms** |
| VSETVL | 2032 | 20494772 | 8391 / 8884 | 157475 | 15616 / 257 | PASS 2.12 ms |

- **Cause:** by default, an axis-0 reduction goes through `lowerLeadingDimension`: 31 element-wise `vfadd`s of the 128-wide rows per tile, with nothing transposed. With `REDUCE` it becomes `vector.multi_reduction <add> … [0] : vector<32x128xf32> to vector<128xf32>`, and `LowerMultiReduction.cpp:40` lowers every multi-reduction with `VectorMultiReductionLowering::InnerReduction`. For an outer axis, MLIR first transposes `[32, 128]` to `[128, 32]` with shuffles so that the reduced axis is innermost, then does 128 separate 32-lane reductions per tile: about 15600 `shufflevector`s and 256 `llvm.vector.reduce.fadd.v32f32` over the two 16 KB tiles. Every element goes through the shuffle network, which drives max VP to 2583 and the spills to 8580.
- `InnerReduction` is the right strategy when the reduced axis is already innermost (softmax rows, the GEMV's `axis=1`). It is the wrong one for axis 0, where `InnerParallel`, or the default row-wise lowering, is just element-wise adds.
- **`MINE` costs 0.137 → 0.202 ms of its own:** each of the 64 masked loads becomes a vsetvli copy loop into a stack slot followed by a reload (`emitVsetvlCopyLoop`): `vse32` 5 → 198, `vle32` 128 → 194, branches 2 → 67. It spills less (694 → 574), but the extra memory traffic costs more than it saves. Its max VP of 2032 most likely comes from how the m8 copy loops are counted, not from real register pressure: SLIL drops to 7072.
- **Fix:** `vector.multi_reduction` is used only when the reduced axis is the innermost one (`axis == srcTy.getRank() - 1`). Other axes fall back to `lowerLeadingDimension`, as without `REDUCE`. Last-axis reductions are unchanged.
- **Option not taken:** choosing `InnerParallel` per op in `LowerMultiReduction.cpp`. That works too, but it means running two pattern sets filtered by axis.

| env | max VP | SLIL | spills / reloads | asm lines | `shufflevector` / `reduce.fadd` | board |
|---|---|---|---|---|---|---|
| baseline | 1050 | 206636 | 694 / 566 | 8643 | 34 / 0 | PASS 0.142 ms |
| `REDUCE` | 1033 | 186913 | 695 / 603 | 8687 | 33 / 0 | PASS **0.142 ms** |
| VSETVL | 2032 | 7072 | 574 / 476 | 10359 | 0 / 0 | PASS **0.216 ms** |
| VSETVL + all `--custom-*` | 2032 | 7072 | 574 / 476 | 10359 | 0 / 0 | PASS 0.192 ms |

`REDUCE` alone is now as fast as the baseline. The remaining gap of the VSETVL builds is `MINE`'s copy loops. The code is identical in the last two rows, so their difference in time is board noise.

## lqcd hop_t_m / hop_t_p: slow with `MINE`

hop_t_m and hop_t_p have no reductions, so `REDUCE` changes almost nothing; `MINE` alone causes the slowdown. `LANE` is `TRITON_VSETVL_LANE=1`, which lowers the same tail-masked accesses to `vp.load` / `vp.store` straight into registers.

| kernel | env | spills / reloads | `vle` / `vse` | `vsetvli` | branches | board |
|---|---|---|---|---|---|---|
| hop_t_m | baseline | 171 / 255 | 66 / 24 | 1 | 0 | PASS 1.06 ms, PASS 1.05 ms |
| | `REDUCE` | 163 / 246 | 66 / 24 | 1 | 0 | PASS 1.11 ms |
| | **`MINE`** | 128 / 215 | **156 / 114** | **179** | **91** | PASS **1.66 ms** |
| | VSETVL | 128 / 215 | 156 / 114 | 179 | 91 | PASS 1.46 ms, PASS 1.55 ms |
| | `LANE` | 138 / 194 | 66 / 24 | 165 | 1 | PASS 1.00 ms, PASS 1.01 ms |
| hop_t_p | baseline | 136 / 201 | 42 / 24 | 1 | 0 | PASS 0.67 ms, PASS 0.66 ms |
| | `REDUCE` | 131 / 218 | 42 / 24 | 1 | 0 | PASS 0.68 ms |
| | **`MINE`** | 103 / 182 | **108 / 90** | **131** | **67** | PASS **0.94 ms** |
| | VSETVL | 103 / 182 | 108 / 90 | 131 | 67 | PASS 0.89 ms, PASS 0.94 ms |
| | `LANE` | 118 / 172 | 42 / 24 | 95 | 1 | PASS 0.75 ms, PASS 0.67 ms |

- **Cause:** hop_t_m does 66 masked loads and 24 masked stores of 64 FP32 elements (256 B) each, all with the tail mask `site < n_sites`. `n_sites` (4096) is a runtime argument, so every access keeps its mask even though all 64 programs are full blocks. The baseline builds the mask once (`vmslt`) and does each access in one instruction (`vle32.v v24, (s11), v0.t`); the whole kernel has one `vsetvli`. `MINE` moves every load through a 256 B stack slot: a loop of `vsetvli` (vl = remaining), `vle32.v` from memory, `vse32.v` to the slot and a branch, then `vsetvli` back to 64 and `vle32.v` from the slot. A store goes the other way: the value is stored to a slot, then a loop copies it out.
- So each access costs three vector memory ops instead of one, plus two `vsetvli`s and a loop branch, and the reload waits on the store to the same address just before it. For hop_t_m that is 90 → 270 vector memory ops, 1 → 179 `vsetvli`s and 91 small loop blocks, about 90 ns per access over 64 programs × 90 accesses. `MINE` spills less (171 / 255 → 128 / 215), but saving about 40 spill/reload pairs does not pay for about 180 extra vector memory ops.
- `MINE`'s max VP / SLIL (hop_t_m SLIL 111600 → 5488) probably overstate the drop in pressure: the 91 copy-loop blocks split the kernel, and `ExpandPseudos` counts SLIL per block, so a value that is live across a loop block without being used in it is not counted. (Not checked directly.)
- `LANE` keeps the lower spills without the copies and runs at baseline speed.
- **Fix:** the pipeline compiles for VLEN ≥ 256 (`-riscv-v-vector-bits-min=256`, `third_party/cpu/llvm.cc`), so an m8 register group holds at least 2048 bits. When the whole vector fits in that (`fitsOneM8Group`: elements × element bits ≤ 2048), `MINE` now emits a single `vsetvli(EVL)` + `vp.load` / `vp.store` into the register, the same code as `LANE` (`emitVPLoad` / `emitVPStore`, now shared by both). Larger vectors still go through the stack slot and the copy loop.

| kernel | env | max VP | SLIL | spills / reloads | `vle` / `vse` | `vsetvli` | branches | board |
|---|---|---|---|---|---|---|---|---|
| hop_t_m | baseline | 409 | 111600 | 171 / 255 | 66 / 24 | 1 | 0 | PASS 1.02 ms |
| | `MINE` | 368 | 55808 | 138 / 194 | 66 / 24 | 165 | 1 | PASS **1.14 ms**, PASS **1.03 ms** |
| | VSETVL | 368 | 55808 | 138 / 194 | 66 / 24 | 165 | 1 | PASS **1.00 ms**, PASS **1.02 ms** |
| hop_t_p | baseline | 313 | 59184 | 136 / 201 | 42 / 24 | 1 | 0 | PASS 0.74 ms |
| | `MINE` | 320 | 34472 | 118 / 172 | 42 / 24 | 95 | 1 | PASS **0.67 ms**, PASS 2.02 ms |
| | VSETVL | 320 | 34472 | 118 / 172 | 42 / 24 | 95 | 1 | PASS **0.71 ms**, PASS **0.68 ms** |

- The assembly is the same as `LANE`'s: no stack slots, no extra memory ops, one loop-free block, and 20–25% fewer spills / reloads than the baseline. The board time is back at the baseline.
- The 2.02 ms hop_t_p run is an outlier: the identical binary measured 0.67 ms in the other run.
- **dwdb is not covered:** its masked loads are rows of `vector<128xf32>`, 4096 bits, two m8 groups, so `MINE` still copies them through the stack (PASS 0.200 ms, PASS 0.201 ms against 0.148 ms baseline, the same code as before the fix). `LANE` on dwdb, which also uses `vp.load` for vectors larger than one register group (LLVM splits them), runs PASS 0.130 ms, PASS 0.140 ms with 665 / 539 spills (baseline 694 / 566). So raising or dropping the size limit in `fitsOneM8Group` would most likely bring dwdb under `MINE` to the baseline too.

## Not yet checked

- The two reduction guards only change reductions of BF16 / F16 values or along an outer axis. 02 softmax and 05 `_layer_norm_fwd_fused` (FP32, last axis or 1-D) should not change, but they were not rerun after the fixes.
- The `MINE` fix affects every kernel with masked 1-D loads or stores of at most 2048 bits (64 FP32 or 128 BF16 elements, for example), not only hop_t. Of the other kernels, only hop_t_m / hop_t_p and dwdb were rerun after it.
