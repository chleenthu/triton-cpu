# ConvertReductionOp: masked reductions (commit "add mask")

The "add mask" commit (`6bc1f1b38`) adds an optional masked reduction to `third_party/cpu/lib/TritonToTritonCPU/ConvertReductionOp.cpp`. It recognizes reductions like `tl.sum(tl.where(mask, x, 0))`, where the mask is a tail mask, and turns them into a reduction that only combines the active lanes. On RISC-V that lowers to a `llvm.vp.reduce.*` with an explicit vector length (EVL). It's off unless `TRITON_VSETVL_REDUCE` is set.

The diff has four parts.

## 1. A new option, `useMaskedReduction`

It's added to `ReduceOpConversion`'s constructor and member fields, to the `ConvertReductionOp` pass, and to `createConvertReductionOp(…, bool useMaskedReduction)`. `compiler.py` passes it in:

```python
cpu.passes.ttcpuir.add_convert_reduction_op(pm, True, bool(os.getenv("TRITON_VSETVL_REDUCE")),
                                            use_masked_reduction=bool(os.getenv("TRITON_VSETVL_REDUCE")))
```

## 2. `matchAndRewrite` tries the masked form first

```cpp
if (useMaskedReduction && succeeded(mapToMaskedReduction(op, rewriter)))
  return success();
// ... otherwise the existing lowerings, unchanged
```

If the pattern doesn't match, everything goes through the old path as before.

## 3. A refactor: `detectSimpleCombine`

The old code checked inline that the reduce body is a single binary op on its two block arguments (e.g. `arith.addf %acc, %x`), then mapped it to a `vector::CombiningKind`. That check is now its own function, `detectSimpleCombine(op, kind)`, so the old path and the new masked path share it. The old path's behavior is unchanged: `vector.reduction` for 1-D, `vector.multi_reduction` otherwise.

## 4. The new lowering: `mapToMaskedReduction`

**When it applies.** All of these must hold, or it returns `failure()` and the normal path is used:

| check | why |
|---|---|
| one operand, one result | simple reductions only |
| `detectSimpleCombine` succeeds | the combine must be a known kind (add, mul, max, …) |
| every dimension except `axis` has size 1 | the input must be effectively 1-D, e.g. `tensor<1x128>` reduced along axis 1, so one masked `vector.reduction` does the whole thing |
| the input (looking through `unrealized_conversion_cast`s) is an `arith.select` | the `tl.where(...)` |
| the `select`'s condition comes from a `vector.create_mask` | a tail mask, i.e. a lowered `triton_cpu.tail_mask` |
| the `select`'s false value is a splat constant equal to the kind's **identity** (`getIdentity`) | the masked-off lanes must contribute nothing |

**Why the identity condition matters.** `reduce(select(mask, x, identity))` equals `reduce(x over the mask's lanes only)`, because combining with the identity changes nothing: `+ 0`, `* 1`, `max(·, -inf)`, … So the `select` can be dropped, and the mask can be applied directly to the reduction.

**What it builds:**

```cpp
x    = vector.shape_cast sel.getTrueValue() : vector<n x T>   // flatten to 1-D
mask = vector.shape_cast sel.getCondition() : vector<n x i1>
acc  = arith.constant identity
red  = vector.reduction <kind>, x, acc                      // + fastmath<reassoc> for float add/mul
res  = vector.mask %mask { red }                            // only lanes where mask is on
// broadcast res back if the converted result type is a vector
```

For example, `tl.sum(tl.where(m, x, 0.0))` on `tensor<1x128xf32>` becomes:

```mlir
%xf = vector.shape_cast %x : vector<1x128xf32> to vector<128xf32>
%mf = vector.shape_cast %m : vector<1x128xi1> to vector<128xi1>
%r  = vector.mask %mf { vector.reduction <add>, %xf, %cst0 fastmath<reassoc> : vector<128xf32> into f32 }
```

Later, the `vector.mask` around the reduction lowers to a `llvm.vp.reduce.fadd` whose EVL comes from the `create_mask` length.

**The `reassoc` flag.** For float add and mul, the reduction is marked reassociative. That allows the unordered `vp.reduce` hardware reduction, but the last bits of the result may differ from the default tree of shuffles. The code comment notes this.

**`getIdentity`.** It returns the neutral element for each kind:

| kind | identity |
|---|---|
| ADD, OR, XOR, MAXUI | 0 |
| MUL | 1 |
| AND, MINUI | all ones |
| MAXSI / MINSI | signed min / signed max |
| MAXNUMF / MINNUMF | -inf / +inf |
| others (e.g. NaN-propagating `maximumf` / `minimumf`) | `nullptr`, so no masked form |

NaN-propagating min/max have no identity: there's no value `v` with `max(v, x) == x` for every `x` including NaN. So those reductions keep the old lowering.

## How it fits with the rest of the tail-mask work

This is the masked reduction that `TailMaskToEVL::tailLanesUnused` looks for. When a loaded vector's only use is `vector.mask %m { vector.reduction … }` with a mask no wider than the load's, the lanes past the tail are never read. So the load's pass-through can be poison. Without `TRITON_VSETVL_REDUCE`, `tl.sum(x)` reads all lanes, and the pass-through has to stay. That's the counter-example in `DocTailMaskToEVL.md`.

## Which changes depend on `TRITON_VSETVL_REDUCE`, and which on tail masks

In this commit, every new behavior in `ConvertReductionOp.cpp` is switched on by `TRITON_VSETVL_REDUCE`. Inside that, the tail-mask dependency is one specific check: the `select`'s condition must be a `vector.create_mask`. Only the tail-mask pipeline produces that form.

### How `TRITON_VSETVL_REDUCE` reaches the file

In the commit's `compiler.py`:

```python
cpu.passes.ttcpuir.add_convert_reduction_op(pm, True, False,
                                            use_masked_reduction=bool(os.getenv("TRITON_VSETVL_REDUCE")))
```

`triton_cpu.cc` gained the matching `use_masked_reduction` argument (default `false`). The same commit also makes `TRITON_VSETVL_REDUCE` one of the three variables that turn on `AnalyzeTailMasks`:

```python
def _vsetvl_enabled():
    return any(os.getenv(v) for v in ("TRITON_VSETVL_MINE", "TRITON_VSETVL_LANE", "TRITON_VSETVL_REDUCE"))
```

So setting `TRITON_VSETVL_REDUCE` does two things: it creates the tail masks (`AnalyzeTailMasks`), and it enables the masked reduction that consumes them.

(Later, `compiler.py` was changed to also pass `bool(os.getenv("TRITON_VSETVL_REDUCE"))` as the second argument, `use_multidim_reduction_op`. In this commit, the second argument was `False`.)

### The changes in `ConvertReductionOp.cpp`, by category

| change | related to `TRITON_VSETVL_REDUCE` | related to the tail-mask optimization |
|---|---|---|
| `useMaskedReduction` added to the constructor, members, pass and `createConvertReductionOp` | **yes**: it's how the variable gets in | no, just plumbing |
| `matchAndRewrite`: `if (useMaskedReduction && succeeded(mapToMaskedReduction(…)))` | **yes**: the only way the new code runs | no, just the entry point |
| `detectSimpleCombine` split out of the old code | no: a refactor shared by both paths, with no behavior change | no |
| `mapToMaskedReduction`, checking one operand, a known combine kind, and only one non-unit dimension | yes (only runs when it's on) | no, generic restrictions |
| `mapToMaskedReduction`, checking the `select`'s false value is the identity (`getIdentity`) | yes | no: what makes dropping the `select` correct, for any mask |
| **`mapToMaskedReduction`, checking the `select`'s condition comes from a `vector.create_mask`** | yes | **yes**: the tail-mask part, see below |
| building `vector.mask %mask { vector.reduction … }` | yes | **yes**: this form is what tail-mask codegen consumes later |
| `getIdentity` (neutral element per kind) | yes | no |

### Why the `create_mask` check is the tail-mask part

```cpp
auto sel = selVal.getDefiningOp<arith::SelectOp>();
if (!sel || sel.getType() != srcTy ||
    !sel.getCondition().getDefiningOp<vector::CreateMaskOp>())   // <- needs a tail mask
  return failure();
```

The condition is a `vector.create_mask` only when the mask went through the tail-mask pipeline: `AnalyzeTailMasks` turns `cols < n_cols` into `triton_cpu.tail_mask`, and `ConvertElementwiseOps` lowers that to `vector.create_mask`. Without `AnalyzeTailMasks`, the mask in TTCIR is an elementwise `arith.cmpi` vector, this check fails, and the reduction keeps the old lowering.

That's why the commit also makes `TRITON_VSETVL_REDUCE` enable `AnalyzeTailMasks`. On its own, the masked reduction would never find a `create_mask`.

The `vector.mask` it builds is also what the rest of the tail-mask work relies on:
- **`TailMaskToEVL::tailLanesUnused`** looks for `vector.mask %m { vector.reduction }`. When the mask is no wider than the load's, the load's pass-through can be poison.
- **Upstream `MaskedReductionOpConversion`** lowers it to `llvm.vp.reduce.*` with the `create_mask` as the lane mask.

In short:
- **`TRITON_VSETVL_REDUCE`:** the option plumbing and the new `mapToMaskedReduction` / `getIdentity` path.
- **Tail mask:** the `create_mask` condition check (it only matches masks from `AnalyzeTailMasks`) and the `vector.mask` it produces (consumed by `TailMaskToEVL`).
- **Neither:** the `detectSimpleCombine` refactor.

### The `create_mask` is kept and reused, not deleted

The check only confirms that the `select`'s condition is a `create_mask`. That same `create_mask` then becomes the mask operand of the new `vector.mask`. What disappears is the `arith.select`, and the `tt.reduce` is replaced.

**Before** `mapToMaskedReduction` (the reduce is still a `tt.reduce`; the rest is already vector code):

```mlir
%m  = vector.create_mask %len : vector<1024xi1>                    // the tail mask
%x  = vector.maskedload …, %m, %zeros …                            // load uses %m
%s  = arith.select %m, %x, %zeros : vector<1024xi1>, vector<1024xf32>   // tl.where(mask, x, 0.0)
%r  = "tt.reduce"(%s) ({ arith.addf … })                          // tl.sum(...)
```

**After:**

```mlir
%m  = vector.create_mask %len : vector<1024xi1>                    // unchanged, still here
%x  = vector.maskedload …, %m, %zeros …                            // unchanged
%r  = vector.mask %m { vector.reduction <add>, %x, %cst0 : vector<1024xf32> into f32 }
                                                                    // new: reuses %m
```

The compiled `rvv_reduce_elf.py` TTCIR shows the same: one `%x_3 = vector.create_mask`, used by both the masked load and the two `vector.mask`s.

| op | what happens |
|---|---|
| `vector.create_mask %m` | **kept**; becomes the mask of `vector.mask` (`sel.getCondition()`) |
| `arith.select %m, %x, %zeros` | **bypassed**; the reduction reads its true value `%x` directly (`sel.getTrueValue()`), and the `select` becomes unused and is removed by later cleanup |
| `tt.reduce` | **replaced** by `vector.mask %m { vector.reduction … }` (`rewriter.replaceOp(op, res)`) |

So the `create_mask` check isn't looking for something to remove. It's making sure the mask is a tail mask: lanes `[0, len)`, from `AnalyzeTailMasks`. The resulting `vector.mask` then has a tail-shaped mask that the later tail-mask passes (`TailMaskToEVL`) understand.

### Why `rvv_reduce_elf.py` keeps a "redundant" `tl.where`

In `rvv_reduce_elf.py`:

```python
x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0)
row_sum = tl.sum(tl.where(mask, x, 0.0), axis=0)
```

Here the `tl.where` is redundant for the result: `other=0.0` already makes the off lanes 0. The kernel writes it out anyway because `select(tail mask, x, identity)` → reduction is exactly the pattern `mapToMaskedReduction` looks for. With `TRITON_VSETVL_REDUCE=1`, that pattern becomes `vector.mask %m { vector.reduction <add>, %x, 0 }`: the `select` is gone, and only the first `n_cols` lanes are summed. Without the `tl.where`, the reduction would just sum all `BLOCK_SIZE` lanes (zeros included). The answer would be the same, but the masked lowering wouldn't be used.

## NaN-propagating min/max

There are two kinds of floating-point min/max. They differ only in how they treat **NaN**.

| | "NaN-ignoring" (`minnum` / `maxnum`) | **NaN-propagating** (`minimum` / `maximum`) |
|---|---|---|
| MLIR op | `arith.minnumf` / `arith.maxnumf` | `arith.minimumf` / `arith.maximumf` |
| `vector::CombiningKind` | `MINNUMF` / `MAXNUMF` | `MINIMUMF` / `MAXIMUMF` |
| LLVM intrinsic | `llvm.minnum` / `llvm.maxnum` | `llvm.minimum` / `llvm.maximum` |
| standard | IEEE 754-2008 `minNum` / `maxNum` | IEEE 754-2019 `minimum` / `maximum` |
| one input is NaN | returns the **other** input (NaN treated as "missing") | returns **NaN** |
| `max(-0.0, +0.0)` | either one | `+0.0` (`-0 < +0`) |

Examples:

```
maxnum (3.0, NaN) = 3.0          maximum(3.0, NaN) = NaN
maxnum (NaN, NaN) = NaN          maximum(NaN, NaN) = NaN
```

So as a reduction over `[1, NaN, 5]`: a `maxnum` reduction gives `5` (the NaN is skipped), and a `maximum` reduction gives `NaN` (one NaN poisons the result).

**In Triton:** `tl.max(x)` / `tl.maximum(a, b)` take a `propagate_nan` argument. With the default, `propagate_nan=NONE`, you get `maxnumf`, i.e. the `MAXNUMF` kind in `rvv_reduce_elf.py`'s `tl.max`. With `propagate_nan=ALL`, you get `maximumf` / `MAXIMUMF`.

### `getIdentity`'s comment is too strong

```cpp
// Identity element of `kind` for elemTy, or null if the masked form would
// not be equivalent (NaN-propagating min/max have no identity).
```

`-inf` is in fact an identity for `maximum`: `maximum(-inf, x) == x` for every `x`, including NaN, because `maximum(-inf, NaN)` is `NaN`. Likewise `+inf` is an identity for `minimum`. Upstream MLIR's `MaskedReductionOpConversion` handles masked `MAXIMUMF` / `MINIMUMF` too, but differently from the other kinds:

```cpp
case CombiningKind::MAXIMUMF:
  result = lowerMaskedReductionWithRegular<LLVM::vector_reduce_fmaximum,
                                           MaskNeutralFMaximum>(…, maskOp.getMask(), fmf);
```

It selects the neutral value into the masked-off lanes and then does an ordinary `llvm.vector.reduce.fmaximum`. That's the same work as the unmasked `select` + reduction, because no `vp.reduce.*` intrinsic is used. The other kinds, `MAXNUMF` included, get a real `llvm.vp.reduce.*`.

So the practical reason the masked form skips `maximumf` / `minimumf` is that there's no gain: they wouldn't become a `vp.reduce`. It's not that an identity doesn't exist. Leaving them out (`default: return nullptr`) is harmless; those reductions keep the old lowering, and the result is the same.
