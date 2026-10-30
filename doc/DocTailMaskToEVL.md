# TailMaskToEVL: poisoning the pass-through of masked loads

These two parts of `TailMaskToEVL` (`third_party/cpu/lib/TritonCPUTransforms/TailMaskToEVL.cpp`) make up one optimization: when nothing ever reads a masked load's off lanes, the load's **pass-through** value is replaced with `ub.poison`.

## Background: the pass-through operand

```mlir
%v = vector.maskedload %base[%i], %mask, %passthru : ... into vector<16xf32>
```

Lane `i` of `%v` is `mem[i]` when `mask[i]` is on, and `passthru[i]` when it's off. In Triton, `passthru` comes from `other=`:

```python
x = tl.load(ptrs, mask=m, other=0.0)    # off lanes are 0.0
```

On RISC-V, a tail-masked load becomes a vector load with an explicit vector length (EVL). The off lanes are then the ones at or past the length. Keeping `passthru` in them costs extra work, either a merge after the load or an instruction mode that leaves those lanes untouched ("tail-undisturbed"), which requires the pass-through to be in the destination register first. If no later code reads those lanes, that work is wasted.

## Background: RVV tail policies (tail-undisturbed vs tail-agnostic)

"Tail-undisturbed" is one of two settings RISC-V vector instructions use for the lanes past the vector length. It's the cheapest correct way to give a load "keep the pass-through in the off lanes" behavior, which is why it's one of the options above.

### RVV vector length, tail, and the two tail policies

Every RVV instruction works on the first `vl` elements of its registers, where `vl` is set by `vsetvli`. The elements at index `vl` and above are called the **tail**. `vsetvli` also sets how the instruction treats the tail of its destination register `vd`:

| policy | `vsetvli` flag | what happens to tail elements of `vd` |
|---|---|---|
| **tail-undisturbed** | `tu` | they keep whatever `vd` held before the instruction |
| **tail-agnostic** | `ta` | the hardware may keep them or overwrite them with all 1s; software must treat them as garbage |

There's an identical pair for lanes turned off by a mask inside `vl`: mask-undisturbed (`mu`) and mask-agnostic (`ma`). With an EVL load there's no mask inside `vl`, so only the tail policy matters here.

### Why tail-undisturbed implements the pass-through

A tail-masked load with EVL `len` means:

```
lane i = (i < len) ? mem[i] : passthru[i]
```

With `vl = len`:
- lanes `< len` are the active elements, which the load fills from memory.
- lanes `>= len` are the tail.

Under `tu`, the tail keeps the old contents of `vd`. So if `vd` already holds `passthru` before the load, one load instruction gives exactly the right result:

```asm
vmv.v.v   v8, v4                  # v8 = passthru (only if it isn't there already)
vsetvli   t0, a1, e32, m1, tu, ma # vl = len, tail-undisturbed
vle32.v   v8, (a0)                # lanes < len from memory, lanes >= len keep passthru
```

This is the "pass-through in the destination register first" option. In LLVM's RISC-V backend, the load's pass-through operand is tied to the destination register. A non-poison pass-through makes the backend choose `tu` and, if the pass-through is still needed elsewhere, insert a copy into `vd`.

The other option is the merge: load with `ta` into any register, then combine with the pass-through using a mask such as `lane < len` (e.g. `vmerge.vvm`). That costs an extra instruction, plus building the mask.

### Why `tu` costs more than `ta`

When the pass-through is `ub.poison`, the tail may be anything, so the backend can use `ta`:

```asm
vsetvli   t0, a1, e32, m1, ta, ma
vle32.v   v8, (a0)                # tail = garbage, nobody reads it
```

`tu` costs more in three ways:
- **The copy.** `passthru` has to be moved into `vd` first, unless it already happens to be there and isn't needed afterward. This also keeps an extra register busy.
- **A dependency on the old `vd`.** Under `tu`, the instruction's result includes the previous value of `vd`, so it has to wait for whatever last wrote `vd`. Out-of-order or register-renaming cores can't treat it as a fresh write. Some implementations are slower with `tu` for this reason.
- **Fewer scheduling choices.** Tied operands limit which register the result can go in.

With `ta`, the load depends only on the address and `vl`. That's the saving `TailMaskToEVL` gets by setting the pass-through to `ub.poison` when nothing reads the tail.

### Summary: ways to fill the off lanes

**1. Every `vle32` has a tail policy; the cheap cases use `ta`.** Every vector instruction runs under the `vsetvli` before it, which sets `tu` or `ta`, so no load is "policy-free". Poison and merge both use the cheap setting, `ta`, so the load doesn't care about the tail:

| method | the `vle32` | extra work |
|---|---|---|
| poison | `ta` | none |
| merge | `ta` | build a mask (`lane < len`) plus a `vmerge.vvm` after the load |
| pass-through in `vd` | `tu` | copy the pass-through into `vd` (if needed), plus a dependency on the old `vd` |

**2. The ranking is usually poison, then `tu`, then merge.** Merge needs an extra instruction and a mask register, while `tu` needs at most one register copy. LLVM's RISC-V backend prefers `tu` too: when it sees a load followed by a merge with the same `vl`, it folds them into a single `tu` load with the pass-through tied to `vd`.

**3. There's a fourth, masked variant.** Instead of shortening `vl`, keep `vl` at the full width and turn the off lanes off with a mask register `v0`, using **mask-undisturbed** (`mu`):

```asm
vsetvli  t0, zero, e32, m1, ta, mu   # vl = full width
vle32.v  v8, (a0), v0.t              # v0 = (lane < len); off lanes keep v8 = passthru
```

It's the same idea as `tu`, but the off lanes are chosen by a mask instead of by `vl`. That's roughly what a plain masked load becomes without EVL, and it costs a mask register plus building `v0`. `TailMaskToEVL` turns tail masks into an EVL to avoid exactly this.

In short: poison (`ta`), then pass-through via `tu`, then merge or the `mu`-masked load. Only the poison case lets the load skip the tail entirely.

## getTailLen: the entry point into the analysis

`getTailLen` answers one question for each masked load or store: **is its mask a tail mask, i.e. lanes `[0, L)` on and the rest off? If so, what is `L`?**

```cpp
// Tail length of a masked access, or null when its mask is not a tail mask.
template <typename OpTy> TailLenPtr getTailLen(OpTy op) {
  auto maskTy = cast<VectorType>(op.getMask().getType());
  if (maskTy.getRank() != 1 || maskTy.isScalable())
    return nullptr;                              // only plain 1-D fixed-width masks
  return analyzeTailLength(op.getMask(), 0);     // L as a TailLen tree, or null
}
```

It's a template, so it works for both `vector.maskedload` and `vector.maskedstore`:
1. **Shape check:** only 1-D, fixed-size masks (`vector<16xi1>`) qualify. n-D masks and scalable ones (`vector<[4]xi1>`) are rejected right away.
2. **Analysis:** it calls `analyzeTailLength` on the mask, which walks the ops that build it (`create_mask`, `cmpi`, `andi`, `broadcast`, …) and returns `L` as a symbolic tree, or `nullptr` if the mask isn't a tail mask.

**How `runOnOperation` uses the result:** it's the input to all three of the pass's decisions.

| use | what `len` is for |
|---|---|
| `if (TailLenPtr len = getTailLen(op))` | **Filter:** only tail-masked ops are considered at all. Others (gather-like or arbitrary masks) are left alone. |
| `tailLanesUnused(load.getResult(), *len, …)` | **Poison check:** `len` is what users' masks are compared against (`maskWithin` → `provablyLE`), to prove no lane `>= L` is ever read. |
| `replaceMask(op, *len)` | **Rewrite:** `buildTailLength(*len)` turns the tree into an `index` value, and a new `vector.create_mask` is built from it, which later lowers to the EVL. |

**In the `mask2d` example:** it's called 24 times (16 masked loads, 8 masked stores), and every call succeeds with `Min(AllOrNone n=16 cond=%a_<row>, Index %a_9)`, i.e. "row exists ? `N - rn` : 0". Then:
- **Filter:** all 24 pass.
- **Poison check:** fails for all 16 loads, because each loaded row's only user is `vector.to_elements`, which isn't lane-wise. So `len` ends up unused.
- **Rewrite:** skipped for all 24 (`isAlreadyEVLMask`: every mask is already a `vector.create_mask`), so `len` is unused here too.

So `getTailLen` does all the analysis, but in this kernel nothing downstream acts on its result, and the pass leaves the IR unchanged.

## Step 1: decide which loads qualify (analysis only)

```cpp
if (auto load = dyn_cast<vector::MaskedLoadOp>(op)) {
  if (TailLenPtr len = getTailLen(load)) {          // mask is a tail mask "lane < len"
    loads.emplace_back(load, len);                  // will get a create_mask later
    llvm::SmallPtrSet<Value, 16> visited;
    if (!load.getPassThru().getDefiningOp<ub::PoisonOp>() &&   // not poison already
        tailLanesUnused(load.getResult(), *len,
                        load.getVectorType().getNumElements(), visited))
      poisonTail.push_back(load);                   // safe to drop the pass-through
  }
}
```

1. **`getTailLen`** checks that the mask is a tail mask (lanes `[0, len)` on) and returns `len`. Otherwise the load is skipped entirely.
2. **`loads.emplace_back`** records the load so its mask can be rewritten as a `vector.create_mask` later.
3. **The poison check** skips loads whose pass-through is already `ub.poison`.
4. **`tailLanesUnused(result, len, …)`** follows every use of the loaded value. It returns true only if every path ends somewhere that ignores lanes `>= len`:
   - **Lane-wise ops**, such as elementwise `arith` ops and `shape_cast`, where lane `i` of the result depends only on lane `i` of the inputs. The check keeps following their results.
   - **`arith.select %cond, %v, …`** where `%cond`'s tail length is provably `<= len`. The select only takes `%v` in lanes it can actually have.
   - **`vector.maskedstore`** of the value with a mask whose length is `<= len`. The bad lanes never reach memory.
   - **A masked `vector.reduction`** (inside `vector.mask %m`) with `%m`'s length `<= len`.

   Any other use, such as an unmasked reduction, a shuffle, or a call, makes it return false.

   `visited` stops it from walking the same value twice when two paths join back together.

This check runs before any mask is rewritten. The comment above the loop explains why: whether a load's tail is unused is decided from its users' select and store masks, which the next step replaces.

## Step 2: apply it after the masks are rewritten

```cpp
// Lanes past the tail are never observed, so they need not hold the
// pass-through value; poison lets the lowering skip the merge.
for (vector::MaskedLoadOp load : poisonTail) {
  OpBuilder b(load);                                     // insert right before the load
  Value poison =
      ub::PoisonOp::create(b, load.getLoc(), load.getPassThru().getType());
  load.getPassThruMutable().assign(poison);              // pass-through := poison
}
```

For each load that qualified, the code creates a `ub.poison` of the same vector type and makes it the pass-through. Poison means "any value". So off lanes may now hold anything, and the backend can use its cheapest option, "tail-agnostic", with no merge and no copying of the pass-through first.

## Example

```python
x = tl.load(a_ptrs, mask=m, other=0.0)
y = x * 2.0
tl.store(b_ptrs, y, mask=m)
```

```mlir
%x = vector.maskedload %a[...], %m, %zeros     // passthru = 0.0
%y = arith.mulf %x, %two                        // lane-wise
vector.maskedstore %b[...], %m, %y              // same mask
```

`%x` is only used by `mulf`, which is lane-wise, and `%y` is only stored with the same mask. The off lanes of `%x` never matter, so `%zeros` becomes `ub.poison`. The result is the same, and the load is cheaper.

**Counter-example:**

```python
x = tl.load(ptrs, mask=m, other=0.0)
s = tl.sum(x)          # plain reduction over all 16 lanes
```

The sum reads every lane, so `other=0.0` really matters. Those lanes must be 0 or the sum is wrong. `tailLanesUnused` returns false, and the pass-through is kept.

The one exception is a masked reduction, from `ConvertReductionOp` with `TRITON_VSETVL_REDUCE` set. That reduction itself skips lanes `>= len`, so the load still qualifies.

## analyzeScalarLength: the `select` case

In the `mask2d` example (`c = alpha*a + b` on an 8×16 tile), `ConvertMemoryOps` lowers each row of `triton_cpu.tail_mask` to a 1-D mask whose length is a `select`. For row 0:

```mlir
%a_13 = arith.cmpi sgt, %a_8, %a : i32           // %a_8 = M - rm, %a = 0: row 0 exists
%a_14 = arith.select %a_13, %a_9, %a : i32       // %a_9 = N - rn
%a_15 = arith.index_cast %a_14 : i32 to index
%a_16 = vector.create_mask %a_15 : vector<16xi1>
```

`analyzeScalarLength` records `c ? x : 0` as `min(c ? n : 0, x)`, because `TailLen` has no general select node, only `AllOrNone` (`c ? n : 0`):

| identity | the IR |
|---|---|
| `c` | `%a_13` (`M - rm > 0`, row 0 exists) |
| `x` | `%a_9` (`N - rn`) |
| `0` | `%a` (constant 0) |
| `n` | 16 (lanes of `vector<16xi1>`) |
| `c ? x : 0` | `%a_14 = arith.select %a_13, %a_9, %a` |
| `min(c ? n : 0, x)` | `Min(AllOrNone n=16 cond=%a_13, Index %a_9)` |

The two can give different numbers but always the same mask, because a mask length is clamped: `create_mask` turns on lanes `j < L`, so `L <= 0` means none and `L >= n` means all `n`.

### What `n` and `x >= n` mean

`n` is 16, the number of lanes in each row vector (`vector<16xi1>`), i.e. the tile width `BLOCK_N = 16`. `x = %a_9 = N - rn` is the number of matrix columns left from this tile's first column `rn` onward.

So `x >= n` means "at least 16 columns are left", and the whole tile fits inside the matrix. `x < n` means this is the last, partial tile in the column direction.

**Example with `N = 45`** (tiles `pid_n = 0, 1, 2`, `rn = 16 * pid_n`), for a row that exists:

| `pid_n` | `rn` | `x = N - rn` | `x >= 16`? | `select` result | `min(16, x)` | lanes on |
|---|---|---|---|---|---|---|
| 0 | 0 | 45 | yes | 45 | 16 | all 16 (cols 0–15) |
| 1 | 16 | 29 | yes | 29 | 16 | all 16 (cols 16–31) |
| 2 | 32 | 13 | **no** | 13 | 13 | 13 (cols 32–44) |

- **`x >= 16`** (tiles 0 and 1): the numbers differ (45 vs 16, 29 vs 16), but both are at least 16, so `create_mask` turns all 16 lanes on either way. This is the case where the identity relies on clamping.
- **`x < 16`** (tile 2, the tail): `min(16, x) = x`, so both sides are exactly equal. Nothing needs clamping.
- **`x <= 0`** would mean a tile entirely past `N`. With the usual `cdiv(N, 16)` grid that doesn't happen, but if it did, both sides are `<= 0`, so all lanes are off.
- **Row doesn't exist** (`c` false): the `select` gives 0 and `min(0, x) <= 0`, so all lanes are off either way.

So in every case the two forms give the same mask. They only differ numerically when `x >= 16`, and there the difference doesn't matter.
