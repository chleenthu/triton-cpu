# ScalarizeInterface in the `mask2d` example

`TailMaskScalarizeInterface` (`ScalarizeInterface.cpp`) has two entry points, `computeScalarValue` (constant indices) and `computeScalarValueForLoop` (loop indices). In `mask2d` **neither is called**, so there is no result to clean up later. Only two places would call them on a mask:

1. **The scalarize pass** (`ScalarizeUsingForOps.cpp`), which runs before `ConvertMemoryOps` (`add_scalarize` in `make_ttcir`). It only turns a load or store into an `scf.for` element loop if `shouldScalarizeOpGeneric` says so, and that check rejects contiguous accesses first:

   ```cpp
   auto axisInfo = axisAnalysis.getAxisInfo(ptr);
   if (isContiguousRowMajorAccess(axisInfo, scalarizeOp)) {
   ```

   The `mask2d` loads and stores are contiguous row-major (the same reason `ConvertMemoryOps` prints `contiguous row-major: one vector load per row`). So they're never scalarized, and `computeScalarValueForLoop` on the `tail_mask` is never reached.
2. **`ConvertMemoryOps::maybeStoreVecToTempBuf`**, which asks `canComputeScalarValue(mask)`. It's only used on the scalar and scatter paths, which this kernel doesn't take.

The scalarize interface does run in `mask2d`, just not on the mask. `ConvertMemoryOps::extractScalarPointer` calls `canComputeScalarValue(ptrs)` / `computeScalarValue(ptrs, indices)` with **constant** indices `[row, 0]`, using the generic `TritonOpScalarizeInterface` on the pointer tensor `%a_18 = tt.addptr …`. Those results are kept: they're the per-row scalar pointers in the TTCIR, e.g. `%a_11 = tt.addptr %a_ptr, %offs_10`. With `-debug-only=scalarize-interface`, that's the only thing printed for this kernel.

## Which version of `computeScalarValue` runs

`computeScalarValue` is a method of the `ScalarizeInterface`, and each op type attaches its own version. A call like:

```cpp
computeScalarValue(ptrs.getDefiningOp(), ptrs, indices, rewriter)
```

runs the version that belongs to the op that defines the value:

| op that defines the value | version that runs |
|---|---|
| `tt.addptr`, `arith.addi`, `arith.muli`, … | generic `TritonOpScalarizeInterface<OpTy>` |
| `tt.splat` | `TritonOpScalarizeInterface<SplatOp>` (returns the source) |
| `tt.make_range` | `TritonOpScalarizeInterface<MakeRangeOp>` |
| `tt.broadcast`, `tt.expand_dims`, `tt.trans` | their `ScalariztionFunctor` versions |
| `triton_cpu.tail_mask` | `TailMaskScalarizeInterface` (`computeScalarValue` / `computeScalarValueForLoop` → `build`) |

In `mask2d`, the only caller is `ConvertMemoryOps::extractScalarPointer`, and it scalarizes the pointer tensor. It starts at `%a_18 = tt.addptr …` and recurses through that pointer's operands:

```
%a_18 = tt.addptr %a, %offs_17            -> generic (addptr)
  %a       = tt.splat %a_ptr              -> SplatOp
  %offs_17 = arith.addi %offs_15, %offs_16 -> generic (addi)
    %offs_15 = tt.broadcast %offs_7        -> BroadcastOp
      %offs_7 = arith.muli %mask, %offs    -> generic (muli)
        %mask = tt.expand_dims %rm_2       -> ExpandDimsOp
          %rm_2 = arith.addi %rm_1, %rm_0  -> generic, then SplatOp / MakeRangeOp
        %offs = tt.splat %stride_m         -> SplatOp
    %offs_16 = tt.broadcast %mask_8 …      -> BroadcastOp, ExpandDimsOp, …
```

The `tail_mask` isn't part of the pointer computation, so the walk never reaches it, and `TailMaskScalarizeInterface`'s versions never run.

So, more precisely:
- **Called in `mask2d`:** the scalarize interface's versions for the pointer-building ops (`addptr`, `splat`, `addi`, `muli`, `broadcast`, `expand_dims`, `make_range`). Their results are the per-row scalar pointers kept in the TTCIR.
- **Not called in `mask2d`:** `TailMaskScalarizeInterface::computeScalarValue` and `computeScalarValueForLoop`. They would only run if a scalarized load or store asked for the mask's value per element, which this kernel never does.

# Why `arith.muli %mask, %offs`

In the `mask2d` TTIR:

```mlir
%mask   = tt.expand_dims %rm_2 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>   // rm[:, None]
%offs   = tt.splat %stride_m : i32 -> tensor<8x1xi32>                               // stride_m
%offs_7 = arith.muli %mask, %offs : tensor<8x1xi32>                                 // rm[:, None] * stride_m
```

It isn't multiplying the boolean mask. `%mask` here is an `i32` index tensor that happens to be named "mask". The kernel uses `rm[:, None]` twice:

```python
mask = (rm[:, None] < M) & (rn[None, :] < N)        # 1st use: inside `mask = ...`
offs = rm[:, None] * stride_m + rn[None, :]         # 2nd use: inside `offs = ...`
```

Both create the same `tt.expand_dims %rm_2 {axis = 1}`, and CSE merges them into one value. The printed name comes from the source location of the first one, which is on the `mask = …` line. So the shared `rm[:, None]` gets named `%mask`, even though it's just row indices. The `cmpi` result is a different value (`%mask_7` / `%mask_11`).

So `%offs_7` is `rm[:, None] * stride_m`, the element offset of each row's start (row index × row stride). Then `+ rn[None, :]` adds the column, giving `offs[i][j] = (rm + i) * stride_m + (rn + j)`, the flat index into the row-major matrix.
