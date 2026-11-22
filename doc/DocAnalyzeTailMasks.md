# AnalyzeTailMasks walkthrough

## `mkBin`

`mkBin(kind, a, b)` builds one expression node, `a <kind> b`. For example, `mkBin(Expr::Sub, x, y)` builds the tree for `x - y`:

```
Sub
  x
  y
```

It simplifies as it builds:
- **Both operands constant:** it computes the result, so `mkBin(Add, Const 3, Const 4)` gives `Const 7`.
- **Adding or subtracting 0:** it returns the other operand, so `mkBin(Add, x, Const 0)` gives `x`.

## Worked example: `slt`

A typical 1-D tail mask:

```python
offs = pid * 128 + tl.arange(0, 128)
mask = offs < n
```

In TTIR (simplified):

```
%r    = tt.make_range {start = 0, end = 128}   // [0, 1, ..., 127]
%s    = tt.splat %base                          // %base = pid * 128, same in every element
%offs = arith.addi %s, %r
%ns   = tt.splat %n
%m    = arith.cmpi slt, %offs, %ns              // offs < n
```

**Choosing the sides:** the predicate is `slt`, i.e. `lhs < rhs`, so `rangeSide = %offs`, `boundSide = %ns`, and `extra = 0`.

**`range = analyzeInt(%offs)`**, built bottom-up:
- `%r` (make_range) gives `{isRange=true, axis=0, scalar=Const 0}`, meaning element `i` is `0 + i`.
- `%s` (splat) gives `{isRange=false, scalar=Leaf %base}`, meaning every element is `base`.
- `addi`: exactly one side is a range, so the result is a range with scalar `mkBin(Add, Leaf %base, Const 0)`, which simplifies to `Leaf %base`.
- So `range = {isRange=true, axis=0, scalar=Leaf %base}`, meaning element `i` is `base + i`.

**`bound = analyzeInt(%ns)`** gives `{isRange=false, scalar=Leaf %n}`, meaning every element is `n`.

**The bound for axis 0:**

```
base + i < n   <=>   i < n - base
box[0] = mkBin(Add, mkBin(Sub, Leaf %n, Leaf %base), Const 0)
       = Sub(Leaf %n, Leaf %base)
```

With `n = 300`, `pid = 2`: `base = 256`, so the bound is `300 - 256 = 44`. Elements 0–43 are on and 44–127 are off. That's a tail mask.

**The other predicates** all reduce to the same `i < something` form:

| predicate | means | rewritten | `extra` |
|---|---|---|---|
| `slt` | `off + i < n` | `i < n - off` | 0 |
| `sle` | `off + i <= n` | `i < n - off + 1` (integers) | 1 |
| `sgt` | `n > off + i` | same as `slt` with sides swapped | 0 |
| `sge` | `n >= off + i` | same as `sle` with sides swapped | 1 |

`eq`, `ne` and unsigned predicates aren't prefixes of the axis, so they fall to `default` and return nothing.

## Why give up when `bound->isRange`

A box needs one cutoff per axis that is the same everywhere else in the tensor. If the bound side is also a range, the cutoff changes from element to element. The classic case is the causal mask in attention:

```python
mask = offs_n[None, :] <= offs_m[:, None]     # j <= i
```

Here the bound (`offs_m`) varies along axis 0, so each row has a different cutoff (`1` = on):

```
      j=0 1 2 3
i=0     1 0 0 0
i=1     1 1 0 0
i=2     1 1 1 0
i=3     1 1 1 1
```

That's a triangle, not a box. No single `B_1` describes it, so the analysis returns `std::nullopt` and the mask stays a plain `i1` tensor.

## The other refusal: `!range->isRange`

If the "range" side turns out to be a splat too, e.g. `splat(a) < splat(b)`, the compare has the same result in every element, so the mask is either all on or all off. That is a box in principle: `fullBox` if true, `0` if false. The code doesn't build it, because that would need a bound like `a < b ? size : 0`, and `Select` takes an existing `i1` value as its condition, while this code creates no IR. So it returns `std::nullopt`, and the comment above it describes the case being skipped, not something the code handles.

## `expand_dims` of an `i32` tensor does not go through `analyzeMask`

Example: `%offs` is `tensor<128xi32>`, so `tt.expand_dims %offs {axis = 1}` produces an `i32` tensor. The `ExpandDimsOp` case in `analyzeMaskImpl` is never used for it, because `analyzeMaskImpl` rejects anything that isn't `i1` before it looks at the defining op (`AnalyzeTailMasks.cpp:357`):

```cpp
if (!shape || shape->empty() ||
    !getElementTypeOrSelf(mask.getType()).isInteger(1))
  return std::nullopt;
```

Also, nothing calls `analyzeMask` on an `i32` value. It's only called on mask operands of load/store/select and on operands of mask ops, all of which are `i1`. An `expand_dims` of an `i32` tensor goes through the `ExpandDimsOp` case of `analyzeInt` instead.

## Full 2-D example: `rm[:, None] < M & rn[None, :] < N`

Each `//` comment is what the analysis returns for that line's result. The `i32` tensor lines get an `IntInfo` from `analyzeInt`, and the `i1` lines get a `Box` from `analyzeMask`. Both functions return a `std::optional`; the comment shows the value inside it.

```
// rows: rm = pid_m * 8 + arange(0, 8)
%rm      = arith.muli %pid_m, %c8_i32 : i32                          // scalar: not analyzed, only used as Leaf %rm
%rm_0    = tt.make_range {start = 0, end = 8} : tensor<8xi32>        // {isRange=true,  axis=0, scalar=Const 0}
%rm_1    = tt.splat %rm : i32 -> tensor<8xi32>                       // {isRange=false, axis=0, scalar=Leaf %rm}
%rm_2    = arith.addi %rm_1, %rm_0 : tensor<8xi32>                   // {isRange=true,  axis=0, scalar=Leaf %rm}

// A: row mask rm[:, None] < M  (8x1 -> 8x16)
%mask    = tt.expand_dims %rm_2 {axis = 1} -> tensor<8x1xi32>        // {isRange=true,  axis=0, scalar=Leaf %rm}
%mask_6  = tt.splat %M : i32 -> tensor<8x1xi32>                      // {isRange=false, axis=0, scalar=Leaf %M}
%mask_7  = arith.cmpi slt, %mask, %mask_6 : tensor<8x1xi32>          // Box [Sub(%M, %rm), Const 1]
%mask_11 = tt.broadcast %mask_7 -> tensor<8x16xi1>                   // Box [Sub(%M, %rm), Const 16]

// B: column mask rn[None, :] < N  (1x16 -> 8x16)
//   (rn = pid_n * 16 + arange(0, 16) is built the same way as rm)
                                                                     // %rn_5 = {isRange=true, axis=0, scalar=Leaf %rn}
%mask_8  = tt.expand_dims %rn_5 {axis = 0} -> tensor<1x16xi32>       // {isRange=true,  axis=1, scalar=Leaf %rn}
%mask_9  = tt.splat %N : i32 -> tensor<1x16xi32>                     // {isRange=false, axis=0, scalar=Leaf %N}
%mask_10 = arith.cmpi slt, %mask_8, %mask_9 : tensor<1x16xi32>       // Box [Const 1, Sub(%N, %rn)]
%mask_12 = tt.broadcast %mask_10 -> tensor<8x16xi1>                  // Box [Const 8, Sub(%N, %rn)]

// C = A & B (8x16), the one mask used by both loads and the store
%mask_13 = arith.andi %mask_11, %mask_12 : tensor<8x16xi1>           // Box [Sub(%M, %rm), Sub(%N, %rn)]
```

Where each value comes from:

- **`%rm`:** a scalar `i32`, not a tensor, so neither function is called on it. It only appears as `Leaf %rm` inside `%rm_1`'s result. (Called on it, `analyzeInt` would fail at `getShape` and return `std::nullopt`.)
- **`%rm_2`:** the `addi` case keeps the range side's `isRange`/`axis`, and the scalar is `mkBin(Add, Leaf %rm, Const 0)`, which simplifies to `Leaf %rm`.
- **`%mask`:** this `expand_dims` is on an `i32` tensor, so it goes through the `ExpandDimsOp` case of `analyzeInt`, not `analyzeMask` (which rejects non-`i1` values). The expand axis is 1 and `0 >= 1` is false, so `axis` stays 0. Element `(i, 0)` is `rm + i`.
- **`%mask_8`:** the expand axis is 0 and `0 >= 0` is true, so `axis` becomes 1. Element `(0, j)` is `rn + j`.
- **`%mask_7`:** starts from `fullBox([8, 1])` = `[Const 8, Const 1]`, then `box[range axis 0] = Sub(%M, %rm) + 0`.
- **`%mask_10`:** starts from `fullBox([1, 16])` = `[Const 1, Const 16]`, then `box[range axis 1] = Sub(%N, %rn)`.
- **`%mask_11`:** dimension 1 is broadcast 1 → 16, and `mkAtLeastOne(Const 1, 16)` simplifies to `Const 16`.
- **`%mask_12`:** dimension 0 is broadcast 1 → 8, and `mkAtLeastOne(Const 1, 8)` simplifies to `Const 8`.
- **`%mask_13`:** the `andi` case takes a per-dimension min. In dimension 0, `r = Const 8` is full, so the result is `l`. In dimension 1, `l = Const 16` is full, so the result is `r`. Neither dimension needs a `Min` node.

The pass then replaces the mask operand of both loads and the store with `triton_cpu.tail_mask %b0, %b1`, where `%b0 = M - rm` and `%b1 = N - rn` are materialized as `i32` scalars.

The values are filled in bottom-up, but the calls run top-down. The pass calls `analyzeMask(%mask_13, 0)`, which calls `analyzeMask` on `%mask_11` and `%mask_12`, and so on down to the `make_range`s. Each result is computed when its call returns. With `-debug-only=analyze-tail-masks`, `printIntInfo` prints the `analyzeInt` results and the `analyzeMask depth=N { ... }` blocks print the `Box` results, showing this nesting.

## Bit-test masks: `(C & (1 << i)) != 0`

`(C & (1 << i)) != 0` reaches the pass as `arith.cmpi ne` of `andi(splat C, shli(splat 1, range))` against 0. AnalyzeTailMasks only matches `cmpi slt/sle/sgt/sge` of a range plus offset against a bound, so the trace shows `is CmpIOp ne` → `failed` regardless of the constant. Nothing earlier rewrites the expression into `i < n`, because no pass in the pipeline folds a bit test into a compare.

To make it work, you can either:

- **Write it as a compare:** `i < n`. This is detected today and produces the same code as `rvv_mask2d_elf.py`.
- **Teach `analyzeMask` the bit-test forms:**
  - `(C & (1 << i)) != 0` with a constant `C = 2^n - 1` gives bound `n`, where `n` is the number of trailing 1 bits. It must check that `C + 1` is a power of two; 25 would be rejected.
  - `(((1 << k) - 1) & (1 << i)) != 0` and `((-1 << k) & (1 << i)) == 0` with a scalar `k` give bound `k`, valid only when `k` is provably within 0..bit width (for example behind a `min`/`max` clamp, as in the causal mask). Otherwise the shift is poison.

## vLLM unified attention (2D): which masks become boxes

`kernel_unified_attention_2d_q128_seqlen128` from `scripts/attn_repro` (`./run_riscv.sh` with `TRITON_LLVM_DEBUG_ONLY=analyze-tail-masks`), with the trace annotated line by line in `Unified.txt`. The pass visits the mask uses in walk order:

| # | Mask use | Result | Why it stops |
|---|---|---|---|
| 1 | `select` of `tl.where(offs_d < HEAD_SIZE, 1, 0)` | Box 64 | Full width, so the mask is all on |
| 2 | `select` of query_mask_0 | failed | `query_pos = offs_m // 7`: `divsi` isn't handled |
| 3 | `select` of query_mask_1 | failed | `offs_m % 7`: `remsi` isn't handled |
| 4 | Q load (`%Q_53`) | failed | `dim_mask` is `cmpi ne` |
| 5 | K load | failed | `cmpi ne`; the `tile_mask` side is never reached |
| 6 | V load | failed | `cmpi ne`; the `tile_mask` side is never reached |
| 7 | Causal `select` (`%S_134`) | failed | `cmpi ne` through query_mask_1 |
| 8 | `select` of `safe_m_j` | failed | `cmpf ogt`, a float compare |
| 9 | Output store (same mask `%Q_53` as the Q load) | failed | Same as #4 |

vLLM writes its masks as `tl.where(cond, 1, 0).to(tl.int1)`, which reaches the TTIR as `arith.select %c, dense<1>, dense<0>` followed by `arith.cmpi ne, ..., dense<0>`. An `andi` is all or nothing: if one side fails, the other side is never analyzed. That is why K and V lose `tile_mask` even though it is a box.

### Current table

All nine mask uses, with the `cmpi`-of-`select` case in place (from the board run's trace):

| # | Mask use | Result | Why |
|---|---|---|---|
| 1 | `select` of `tl.where(offs_d < HEAD_SIZE, 1, 0)` | Box 64 | Full width, so the mask is all on |
| 2 | `select` of query_mask_0 | failed | `query_pos = offs_m // 7`: `divsi` isn't handled |
| 3 | `select` of query_mask_1 | failed | `offs_m % 7`: `remsi` isn't handled |
| 4 | Q load (`%Q_53`) | failed | `dim_mask` now gives a box, but query_mask_0 fails on `divsi`, so the `andi` fails |
| 5 | K load | **Box rank 2** | `dim_mask` (64, full) × `tile_mask` |
| 6 | V load | **Box rank 2** | `tile_mask` × `dim_mask` (64, full) |
| 7 | Causal `select` (`%S_134`) | failed | query_mask_1 fails on `remsi`; the causal `sle` side is never reached |
| 8 | `select` of `safe_m_j` | failed | `cmpf ogt`, a float compare |
| 9 | Output store (same mask `%Q_53` as the Q load) | failed | Same as #4 |

In #4 and #9, `dim_mask` resolves through the `tail_mask` that #1 inserted ("is TailMaskOp" in the trace). What still stops these masks is the `divsi`/`remsi` from vLLM's `// num_queries_per_kv` and `% num_queries_per_kv` indexing, not the `cmpi ne`.

### After handling `divsi` and `remsi`

`analyzeInt` now also describes `scalar + (start + i) / div`. Dividing a non-negative range by a positive constant `C` keeps it monotone, so `scalar + (start + i) / C < n` becomes `i < C * (n - scalar) - start` and the mask stays a box. `remsi` by a positive constant is handled only when it never wraps around (`start + size - 1 < C`), where `x % C` is just `x`. A wrapping remainder repeats every `C` elements, so it isn't a prefix.

| # | Mask use | Result | Why |
|---|---|---|---|
| 1 | `select` of `tl.where(offs_d < HEAD_SIZE, 1, 0)` | Box 64 | Full width, so the mask is all on |
| 2 | `select` of query_mask_0 | **Box rank 1** | `query_pos = %0 + i // 7` gives `i < 7 * (cur_batch_query_len - %0)` |
| 3 | `select` of query_mask_1 | failed | `offs_m % 7` over 16 rows wraps around |
| 4 | Q load (`%Q_53`) | failed | query_mask_1 is in the `andi` |
| 5 | K load | Box rank 2 | `dim_mask` (64, full) × `tile_mask` |
| 6 | V load | Box rank 2 | `tile_mask` × `dim_mask` (64, full) |
| 7 | Causal `select` (`%S_134`) | failed | query_mask_1 |
| 8 | `select` of `safe_m_j` | failed | `cmpf ogt`, a float compare |
| 9 | Output store (same mask `%Q_53` as the Q load) | failed | Same as #4 |

query_mask_1 is `h * 7 + i % 7 < 14`. It is all on for the two KV heads of this model, but the pass cannot know `h` at compile time, and in general the pattern repeats instead of forming a prefix. Getting the Q load, the output store and the causal `select` past it would need a mask that is not a single box.

### After handling a wrapping `remsi` in a compare

A wrapping `remsi` no longer fails: `analyzeInt` describes it as `scalar + (start + i) % C`, a value in `[scalar, scalar + C)`. In a `slt`/`sgt` compare against a bound `n`, if both `n` and `scalar` are provably multiples of `C` (`n = C*k`, `scalar = C*h`), then `C*h + r < C*k` holds for every `r` in `[0, C)` exactly when `h < k`. The mask is all on or all off, which is the box `k - h >= 1 ? size : 0` along that axis. `sle`/`sge` are not handled: with `k == h`, only `r == 0` would be on. `exactQuotient` proves the multiple of `C` for constants, an `arith.muli` by a constant multiple of `C`, and sums, differences and constant multiples of those.

| # | Mask use | Result | Why |
|---|---|---|---|
| 1 | `select` of `tl.where(offs_d < HEAD_SIZE, 1, 0)` | Box 64 | Full width, so the mask is all on |
| 2 | `select` of query_mask_0 | Box rank 1 | `i < 7 * (cur_batch_query_len - %0)` |
| 3 | `select` of query_mask_1 | **Box rank 1** | `kv_head_idx*7 + i % 7 < 14` gives `kv_head_idx < 2 ? 16 : 0` |
| 4 | Q load (`%Q_53`) | **Box rank 2** | `dim_mask` × query_mask_0 × query_mask_1 |
| 5 | K load | Box rank 2 | `dim_mask` (64, full) × `tile_mask` |
| 6 | V load | Box rank 2 | `tile_mask` × `dim_mask` (64, full) |
| 7 | Causal `select` (`%S_134`) | failed | The causal `sle` bound depends on the row, so it is not a box |
| 8 | `select` of `safe_m_j` | failed | `cmpf ogt`, a float compare |
| 9 | Output store (same mask `%Q_53` as the Q load) | **Box rank 2** | Same as #4 |

On the board (LMUL=1, 20 iterations, 3 warmup) the kernel takes 52.5 ms, against 52.6 ms before. The Q load and the output store run once per program, outside the KV loop, so boxing them barely changes the time. The remaining cost is in the loop: the K/V loads (already boxes) and the causal `select` (#7).

### Prototype: per-row bound for the causal `select`

The causal mask is not one box, but every row of it is a prefix: element `(i, j)` is on iff `seq_offset + j <= context_len + query_pos(i)`, so the key bound depends on the row. `analyzeRowBound` describes such a mask as a box plus one per-row bound: element `(i_0, ..., i_{n-1})` is on iff it is in the box and `i_col < base + (start + i_row) / div`. It comes from a compare whose range side runs along one axis (`col`) while its bound side runs along another (`row`, which may use the `divsi` form), and an `andi` combines it with boxes. For now it only runs under `-debug-only=analyze-tail-masks`, for masks `analyzeMask` rejects, and does not change the IR.

| # | Mask use | Result | Why |
|---|---|---|---|
| 1 | `select` of `tl.where(offs_d < HEAD_SIZE, 1, 0)` | Box 64 | Full width, so the mask is all on |
| 2 | `select` of query_mask_0 | Box rank 1 | `i < 7 * (cur_batch_query_len - %0)` |
| 3 | `select` of query_mask_1 | Box rank 1 | `kv_head_idx < 2 ? 16 : 0` |
| 4 | Q load (`%Q_53`) | Box rank 2 | `dim_mask` × query_mask_0 × query_mask_1 |
| 5 | K load | Box rank 2 | `dim_mask` (64, full) × `tile_mask` |
| 6 | V load | Box rank 2 | `tile_mask` × `dim_mask` (64, full) |
| 7 | Causal `select` (`%S_134`) | **Box + RowBound** (prototype, trace only) | Rows: `min(query_mask_1, query_mask_0)`; keys: `j < context_len + %0 - seq_offset + 1 + i // 7` |
| 8 | `select` of `safe_m_j` | failed | `cmpf ogt`, a float compare (also fails in `analyzeRowBound`) |
| 9 | Output store (same mask `%Q_53` as the Q load) | Box rank 2 | Same as #4 |

The `i // 7` is the GQA structure: rows 0–6, 7–13 and 14–15 each share one key bound, one more key per token. Using #7 would still need a `tail_mask` form with a per-row bound and a per-row lowering of the `select`.
