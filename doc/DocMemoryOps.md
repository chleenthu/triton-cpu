# Two lowerings of `triton_cpu.tail_mask`

`triton_cpu.tail_mask` is lowered in two places, and they don't do the same job: each produces a different mask for a different consumer.

| | `ConvertElementwiseOps` (the `TailMaskOp` pattern) | `ConvertMemoryOps` (`getTailRowMask`) |
|---|---|---|
| what it lowers | the `triton_cpu.tail_mask` **value** itself | one **row's** mask for one split load or store |
| result | n-D `vector.create_mask %b0, %b1 : vector<8x16xi1>` | 1-D `vector.create_mask %len : vector<16xi1>`, with `%len = select(row on, last bound, 0)` |
| who uses it | ops that consume the whole mask as a vector, e.g. `arith.select` from `tl.where` | the per-row `vector.maskedload` / `maskedstore` |

`ConvertElementwiseOps`:

```cpp
SmallVector<Value> bounds;
for (Value b : adaptor.getBounds())
  bounds.push_back(arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), b));
rewriter.replaceOpWithNewOp<vector::CreateMaskOp>(op, vecTy, bounds);   // n-D create_mask
```

`ConvertMemoryOps` (`getTailRowMask`, end of the general path):

```cpp
Value len = arith::SelectOp::create(rewriter, loc, rowOn, bounds.back(), zero);
Value lenIdx = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), len);
return vector::CreateMaskOp::create(rewriter, loc, rowMaskTy, ValueRange{lenIdx});   // 1-D create_mask
```

## Why the memory ops can't just reuse the n-D mask

`ConvertMemoryOps` splits each 8×16 `tt.load` / `tt.store` into 8 row ops on `vector<16xf32>`, and each needs a 1-D `vector<16xi1>` mask. The generic way to get one is to extract row `i` from the n-D mask. That's exactly what the fallback does when the mask isn't a `tail_mask`:

```cpp
subMask = getTailRowMask(...);
if (!subMask)
  subMask = vector::ExtractOp::create(rewriter, loc, mask, subIndices);   // row i of the n-D mask
```

That would give the right lanes, but it loses the form later passes need:
- **`MemoryOpToLLVM::getMaskEVL`** only gets an EVL from a 1-D `vector.create_mask` with **one** operand (or a constant mask). A `vector.extract` of a 2-D `create_mask` doesn't match, so the load would fall back to an ordinary masked load: no `vp.load`, no `vsetvli`.
- **`TailMaskToEVL`** would first have to work out the row's length from the `extract` and rebuild it.

So `getTailRowMask` builds the row mask straight from the `tail_mask`'s bounds, in the exact form `getMaskEVL` reads:

```
row i:  create_mask(index_cast(select(bound0 > i, bound_last, 0)))
```

That's why it looks at the original op (`mask.getDefiningOp<triton::cpu::TailMaskOp>()`) rather than at the converted mask: it needs the scalar bounds, not a vector.

## How the two fit together

- **Shared mask:** when all leading dimensions are 1 (e.g. `[1, 16]`), `getTailRowMask` reuses the converted mask: `shape_cast(convertedMask)` turns the `vector<1x16xi1>` `create_mask` from `ConvertElementwiseOps` into `vector<16xi1>`. Then a `tl.where` and the load share one mask.
- **Unused n-D mask:** if the only users of a `tail_mask` are loads and stores (as in `mask2d`), every row mask comes from `getTailRowMask`, and the n-D `create_mask` has no users, so it's removed as dead code. That's why the `mask2d` `.ttcir` has no `vector<8x16xi1>` mask, only the 8 per-row 1-D `create_mask`s.

In short: `ConvertElementwiseOps` lowers the mask as a value for general use, and `ConvertMemoryOps` builds row masks for the memory ops in the 1-D form that the EVL lowering needs.
