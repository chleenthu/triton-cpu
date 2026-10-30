# Tail masks: AnalyzeTailMasks and TailMaskToEVL

A *tail mask* is a mask whose lanes `[0, L)` are on and the rest off: the mask of the last, partial block of a Triton program (`offs < n`). On RISC-V such a mask can be replaced by an explicit vector length (EVL), so a masked load / store becomes `vsetvli` + `vp.load` / `vp.store` instead of a generic masked access.

Two passes do this work:

- **AnalyzeTailMasks** (TTIR, `third_party/cpu/lib/TritonToTritonCPU/AnalyzeTailMasks.cpp`) finds box-shaped masks and replaces them with `triton_cpu.tail_mask`, which carries the per-dimension bounds.
- **TailMaskToEVL** (late TTCIR, `third_party/cpu/lib/TritonCPUTransforms/TailMaskToEVL.cpp`) puts the masks of `vector.maskedload` / `vector.maskedstore` into `vector.create_mask` form and gives tail lanes that are never read a poison pass-through.

`MemoryOpToLLVM` (`third_party/cpu/lib/TritonCPUToLLVM/MemoryOpToLLVM.cpp`) then lowers the masked accesses that have a 1-D `create_mask` / `constant_mask` mask to EVL code.

Detection is done at the TTIR level. TailMaskToEVL only reads the result and handles the simple 1-D masks that did not come from a `tail_mask`.

## Environment variables

| Variable | Effect |
|---|---|
| `TRITON_VSETVL_MINE` | Tail-masked loads / stores use `vsetvli` + `vp.load` / `vp.store`. A vector that fits one m8 register group is moved straight in registers; a larger one goes through a stack slot with a `vsetvli` copy loop (`VsetvlMaskedLoadOpConversion` / `VsetvlMaskedStoreOpConversion`). |
| `TRITON_VSETVL_LANE` | Tail-masked loads / stores use `llvm.vp.load` / `llvm.vp.store` on the fixed-length vector, with `llvm.vp.merge` for the pass-through unless it is poison (`VectorMaskedLoadOpConversion` / `VectorMaskedStoreOpConversion`). Used only when `MINE` is not set. |
| `TRITON_VSETVL_REDUCE` | `ConvertReductionOp` turns a reduction of `select(tail mask, x, identity)` into a masked `vector.reduction` (lowered to `llvm.vp.reduce.*`). It also turns on `useMultiDimReductionOp`. |

Any of the three variables turns on both AnalyzeTailMasks and TailMaskToEVL (`_vsetvl_enabled()`, `third_party/cpu/backend/compiler.py:119`). The LLVM lowering of loads / stores depends on `MINE` / `LANE` only.

## Pipeline

| Stage | Pass | What happens to tail masks |
|---|---|---|
| TTIR → TTCIR, first | `add_analyze_tail_masks` (`compiler.py:195`) | Box-shaped masks of `tt.load`, `tt.store` and tensor `arith.select` become `triton_cpu.tail_mask`. |
| | `add_scalarize` | A scalarized access computes each element's mask from the bounds (`TailMaskScalarizeInterface`, `ScalarizeInterface.cpp:259`) instead of storing the mask to a temporary buffer. |
| | `add_convert_memory_ops` | Each row of an n-D access gets its own mask from the bounds (`getTailRowMask`, `ConvertMemoryOps.cpp:48`) instead of a `vector.extract` from the n-D mask. |
| | `add_convert_elementwise_ops` | `triton_cpu.tail_mask` → `vector.create_mask` with the same bounds (`TailMaskOpConversion`, `ConvertElementwiseOps.cpp:167`). |
| | `add_convert_reduction_op` | With `REDUCE`: `mapToMaskedReduction` (`ConvertReductionOp.cpp:210`) needs the `select` condition to be a `vector.create_mask`. |
| TTCIR, late | `add_tail_mask_to_evl` (`compiler.py:272`) | Runs after `add_decompose_fp_conversions`, so the bf16 → i16 memory rewrite has already produced the final loads. |
| TTCIR → LLVM | `MemoryOpToLLVM` | `getMaskEVL` (`MemoryOpToLLVM.cpp:117`) takes the EVL from a constant prefix mask, a 1-D `vector.constant_mask` or a 1-D `vector.create_mask`. Any other mask keeps the default masked lowering. |

## `triton_cpu.tail_mask`

Defined in `third_party/cpu/include/Dialect/TritonCPU/IR/TritonCPUOps.td:75`.

```mlir
%m = triton_cpu.tail_mask %b0, %b1 : tensor<1x1024xi1>
```

Element `(i_0, ..., i_{n-1})` is on iff `i_d < bounds[d]` for every `d`. Bounds are `i32` and are not clamped: a bound `<= 0` turns the whole mask off, and a bound `>=` the dimension size keeps that dimension fully on, the same semantics as `vector.create_mask`. The op is `Pure`. It exists only between AnalyzeTailMasks and ConvertElementwiseOps. A pass placed between them that handles masks must know about it.

## AnalyzeTailMasks (TTIR)

The pass collects the mask operands of `tt.load`, `tt.store` and every `arith.select` with a tensor condition (`AnalyzeTailMasks.cpp:422`). For each one, `analyzeMask` (`AnalyzeTailMasks.cpp:232`) tries to describe the mask as a box: one bound expression `B_d` per dimension. If it succeeds, the bounds are materialized as `i32` scalar ops in front of the user and the operand is replaced with `triton_cpu.tail_mask`. The analysis itself creates no IR, and masks that are already a `tail_mask` are left alone.

Masks it recognizes:

| Op | Meaning |
|---|---|
| `arith.cmpi slt/sle/sgt/sge` of (index range + offset) against a splat bound | `i_axis < bound - offset` (`+1` for `sle` / `sge`) |
| `arith.andi` | Per-dimension `min` of the two boxes |
| `arith.ori` | Per-dimension `max`, when the other dims agree |
| `tt.splat` of an `i1` | All on or all off |
| `tt.expand_dims`, `tt.broadcast`, `tt.trans`, `arith.constant` | Moves or reshapes the box |
| `triton_cpu.tail_mask` | Already a box |

Index tensors are described through `tt.make_range`, `tt.splat`, `tt.expand_dims`, `tt.broadcast`, `tt.trans`, `arith.addi` / `arith.subi` and `arith.constant`. A dimension of size 1 broadcast to `size` becomes `B >= 1 ? size : 0`.

Tests: `test/TritonCPU/analyze-tail-masks.mlir`, `test/TritonCPU/convert-tail-masks.mlir`.

## TailMaskToEVL (TTCIR)

The pass walks `vector.maskedload` / `vector.maskedstore` and analyzes each mask into a tail length `L` (`analyzeTailLength`, `TailMaskToEVL.cpp:188`). It then does two things:

1. A mask that is not yet a `create_mask` / `constant_mask` is replaced with `vector.create_mask L`.
2. A load whose lanes `>= L` are never observed gets a `ub.poison` pass-through, so the lowering can skip the merge (`tailLanesUnused`, `TailMaskToEVL.cpp:463`).

The analysis only accepts masks whose type is effectively 1-D: every dim is 1 except the last (`getEffective1DLength`, `TailMaskToEVL.cpp:33`). Masks it recognizes:

| Form | Length |
|---|---|
| Constant prefix mask | Number of leading `true` lanes |
| `vector.constant_mask` | Last dim size, or 0 if a leading dim is 0 |
| `vector.create_mask` (1-D, or n-D with unit leading dims) | Last operand, `min`ed with `lead > 0 ? n : 0` for each leading operand |
| `vector.shape_cast` | Same as its source |
| `vector.broadcast` of an `i1` | `c ? n : 0` |
| `arith.andi` / `arith.ori` | `min` / `max` of the two lengths |
| `arith.cmpi slt/sle/sgt/sge` of (iota + offset) against a splat bound | `bound - offset` (`+1` for `sle` / `sge`) |

Iota is a constant `start, start+1, ...` vector, a `vector.step`, or an `arith.addi` of one of those with a splat. The scalar operands of `create_mask` are read through constants, `select(c, x, 0)`, `arith.minsi` / `arith.maxsi`, `arith.index_cast` and `arith.extsi` (`analyzeScalarLength`, `TailMaskToEVL.cpp:144`), so lengths built from the same values compare equal and can be proven `<=` another length (`provablyLE`, `TailMaskToEVL.cpp:413`).

There is no case for `vector.extract`. A mask taken out of an n-D mask by `vector.extract` is not recognized.

`tailLanesUnused` follows the result of a load through lane-wise ops. The tail lanes are unused when every path ends in one of these:

- an `arith.select` that picks the value only where its condition (length `<= L`) is on;
- a `vector.maskedstore` whose mask has length `<= L`;
- a `vector.mask { vector.reduction }` whose mask has length `<= L`.

Test: `test/TritonCPU/tail-mask-to-evl.mlir`.

## What each pass covers

| Case | TailMaskToEVL alone | With AnalyzeTailMasks |
|---|---|---|
| 1-D mask: `offs = pid*B + tl.arange(0, B)`, `mask = offs < n` | Recognized: TTCIR still holds `cmpi(iota + splat, splat)` | Recognized: arrives as `create_mask`. Same EVL. |
| 2-D `[M, N]` load / store, `mask = (rows[:, None] < M) & (cols[None, :] < N)` | Not recognized: each row's mask is a `vector.extract` from the 2-D `andi(broadcast, broadcast)`. The default masked lowering is used. | Recognized: each row's mask is `create_mask(row inside ? N_bound : 0)` |
| `[1, N]` blocks with a 2-D mask | Only if the mask folds into the forms above | Recognized: shape-cast of an n-D `create_mask` |
| `tl.where(mask, x, 0)` next to a load with the same mask | Only if the condition matches the forms above | The condition becomes the same `tail_mask` as the load's mask, so the two compare equal |
| `TRITON_VSETVL_REDUCE` | No masked reduction: the `select` condition is not a `create_mask` | Masked `vector.reduction` |
| Scalarized masked access | Mask stored to a temporary buffer | Each element's mask computed from the bounds |

## Why detection is done at the TTIR level

- **Some passes need the bounds before TailMaskToEVL runs.** By the time TailMaskToEVL runs:
  - `ConvertMemoryOps` has split n-D accesses into rows;
  - `ConvertReductionOp` has lowered the reductions;
  - the scalarize pass has written masks to memory.

  Each of them needs to know that the mask is a tail mask while it runs.
- **The structure of the mask is still visible.** At TTIR a mask is one `make_range` / `splat` / `expand_dims` / `broadcast` / `cmpi` / `andi` chain over a shaped tensor. At TTCIR the same mask is spread over per-row `vector.extract`s of broadcasts, possibly rewritten further by canonicalization and `optimize_masks`.
- **The bounds stay attached as operands.** MLIR has no tensor-level `create_mask`, and an attribute on the load would be dropped more easily than an operand of a dedicated op.

TailMaskToEVL keeps its own `cmpi` / iota matching for 1-D masks that did not come through a `tail_mask`.
