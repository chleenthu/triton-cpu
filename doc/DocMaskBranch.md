# Tail masks across control flow

Can AnalyzeTailMasks (`third_party/cpu/lib/TritonToTritonCPU/AnalyzeTailMasks.cpp`) detect a tail mask when the mask, its bound, or its use is on the other side of an `if` / `else` or a loop? A runtime `if flag > 0:` in a Triton kernel becomes `scf.if`, and a `for` becomes `scf.for`, so the question is whether the mask's definition chain can cross a region boundary.

## Test

`python/tutorials/cpu/rvv_mask_branch_elf.py`, run with:

```bash
KERNEL=mask_branch ./g.sh
```

The script:
- runs each kernel on the RISC-V board, with both branches of every `if`, against a numpy reference;
- passes the kernel's TTIR through `triton-opt -triton-cpu-analyze-tail-masks`;
- reports whether the mask of each `tt.load` / `tt.store` became a `triton_cpu.tail_mask`.

| Kernel | Shape | Before | Now |
|---|---|---|---|
| `outside` | `mask = offs < n` before the `if`, used in both branches | detected | detected |
| `inside` | each branch builds and uses its own mask | detected | detected |
| `bound` | the `if` yields only the scalar bound (`n` or `n2`), `mask = offs < bound` after it | detected | detected |
| `loop` | `offs = k * BLOCK + arange` and the mask built inside an `scf.for` body | detected | detected |
| `yielded` | each branch builds a mask, the `scf.if` yields it, the load / store after the `if` uses it | **not detected** | detected |
| `carried` | the mask is an `scf.for` iter_arg, `m = m & (offs < n - k)` each iteration, used after the loop (3 iterations and 0) | **not detected** | detected |
| `carried_use` | the same, used by a load inside the loop | **not detected** | detected |
| `nested` | an inner loop starts from the outer loop's carried mask, and the outer loop carries on with the inner loop's result | **not detected** | detected |
| `not_box` | a carried mask updated with `^`: not a box | not detected | not detected (correct) |
| `carried_offs` | the index tensor is an `scf.for` iter_arg: `offs += BLOCK` each iteration, `mask = offs < n` in the loop | **not detected** | detected |
| `carried_offs_2d` | a 4 x 16 tile whose column offsets `rn` are carried (`rn += BLOCK`), `mask = (rm < M) & (rn < N)` | **not detected** | detected |
| `carried_offs_not_range` | `offs = offs * 2` carried: not a range | not detected | not detected (correct) |
| `yielded_offs` | the index tensor comes out of an `scf.if` (`base + arange` / `base + 3 + arange`), the mask is built after it | **not detected** | detected |
| `carried_offs_if` | a carried index tensor advanced inside an `scf.if` (`+16` on even steps, `+8` on odd) | **not detected** | detected |
| `while_offs` | a `while` loop (`scf.while`) carrying the index tensor, `offs += BLOCK` | **not detected** | detected |
| `while_mask` | an `scf.while` carrying a mask, used after the loop (3 iterations and 0) | **not detected** | detected |
| `while_mask_use` | an `scf.while` carrying a mask, used by a load inside the loop | **not detected** | detected |

All runs PASS on the board before and after the change.

## Why most cases already worked

- **Masks inside regions are collected.** `runOnOperation` collects masks with `getOperation()->walk(...)`, which visits nested regions. Loads and stores inside `scf.if` / `scf.for` bodies are found.
- **The definition chain is plain SSA.** A value defined above a region is visible inside it. In `outside`, `inside` and `loop`, every op from the load's mask back to `make_range` / `splat` is reachable through `getDefiningOp()`.
- **A scalar bound can come from anywhere.** In `bound`, the bound is the `i32` result of an `scf.if`. The splat of it becomes a `Leaf`, and the analysis never needs to look at how a leaf was computed.

## Why `yielded` and `carried` failed before

`analyzeMask` and `analyzeInt` only follow `getDefiningOp()`:

- **`yielded`:** the load's mask is `%1 = scf.if %0 -> (tensor<16xi1>)`. Its defining op is the `scf.if`, which had no case, so the analysis returned `std::nullopt`. Canonicalization does not turn it into an `arith.select`: the dumped TTIR still has the `scf.if`.
- **`carried`:** inside the loop, the mask is a block argument (the iter_arg), so `getDefiningOp()` is null. After the loop it is an `scf.for` result, which also has no case.

## The `scf.if` case

`analyzeMaskImpl` now handles a mask that is a result of an `scf.if`:

```cpp
if (auto ifOp = dyn_cast<scf::IfOp>(def)) {
  if (ifOp.getElseRegion().empty())
    return std::nullopt;
  unsigned idx = cast<OpResult>(mask).getResultNumber();
  auto thenBox = analyzeMask(ifOp.thenYield().getOperand(idx), depth + 1);
  auto elseBox = thenBox ? analyzeMask(ifOp.elseYield().getOperand(idx), depth + 1)
                         : std::nullopt;
  if (!elseBox)
    return std::nullopt;
  Box box;
  for (int64_t d = 0; d < rank; ++d) {
    const ExprPtr &t = (*thenBox)[d], &e = (*elseBox)[d];
    box.push_back(sameExpr(t, e) ? t : mkSelect(ifOp.getCondition(), t, e));
  }
  return box;
}
```

If both yielded masks are boxes, the result is a box whose bound in each axis is `cond ? then-bound : else-bound`. If the two bounds are the same expression, the select is dropped. `Expr::Select` already existed (for `tt.splat` of an `i1`), so `materialize` emits it as `arith.select`. For `yielded`:

```mlir
%b = arith.select %cond, %bound_then, %bound_else : i32
%m = triton_cpu.tail_mask %b : tensor<16xi1>
```

### Dominance guard

The bounds are materialized in front of the mask's user. Before this case existed, every `Leaf` came from an op in the mask's definition chain, and that chain dominates the user, so every leaf was visible there. Looking through an `scf.if` breaks that: a branch can compute its bound inside the branch, for example `%n1 = arith.addi %n, %c1` in the then-region. That value does not dominate a load after the `if`.

`runOnOperation` therefore checks every `Leaf` and `Select` condition with `DominanceInfo::properlyDominates` before it materializes a box (`valuesDominate`). If any value is not visible at the user, the mask is left alone, and `-debug-only=analyze-tail-masks` prints `bounds not visible at the user; skipped`. The same mask can still be detected for a user inside the branch.

Before the guard is reached, many bounds that are computed inside a branch are already fine. `analyzeInt` builds the expression of a splat's scalar from its parts (`scalarExpr`): constants and `arith.addi` / `arith.subi` are looked through, and only other ops become leaves. So `base + 3` computed in a branch becomes `Add(Leaf base, Const 3)`, and both parts are visible after the `if`. The `addi` is rebuilt at the user, and CSE merges it with any copy that is already there. Triton puts such scalars inside the branch on its own (`yielded_offs`), so without this the guard would reject the common case.

The guard could be relaxed further: a non-dominating bound could be yielded out of the `scf.if` as an extra `i32` result, the way the `bound` kernel does by hand. It isn't done because that means rewriting the `scf.if`.

### Index tensors from an `scf.if`

`analyzeIntImpl` has the same case for `i32` tensors. Both yielded index tensors must have the same shape: a range along the same axis, or both splats. The result keeps that shape, and its scalar is `cond ? then-scalar : else-scalar`, or the common scalar when the two are the same expression. A mask built after the `if` from the yielded `offs` is then analyzed as usual. For `yielded_offs` the bound is `n - (cond ? base : base + 3)`. Mixed shapes, a range in one branch and a splat in the other, are rejected.

Combined with loop carrying, this also covers a carried `offs` advanced inside an `scf.if` (`carried_offs_if`). The yielded value is an `scf.if` result whose branches are `offs + 16` and `offs + 8`, so the carried scalar's update is `select(cond, o + 16, o + 8)`.

### Relation to LLVM's if/else → select folding

LLVM's SimplifyCFG has a similar fold. `FoldTwoEntryPHINode` turns an if/else diamond that ends in a `phi` into a `select` by running both sides unconditionally. Its cost limit is `-two-entry-phi-node-folding-threshold` (default 4); the one-sided case (`if` with no `else`) is limited by `-phi-node-folding-threshold`. From C, `clang -O2 -mllvm -two-entry-phi-node-folding-threshold=0 x.c` keeps the two basic blocks and the `phi`.

The `scf.if` case above is the same idea, but it happens earlier and on scalars instead of vectors:

- **It runs much earlier.** AnalyzeTailMasks works on TTIR, where the branch is still a structured `scf.if` that yields a `tensor<16xi1>`. SimplifyCFG only sees the program after it becomes LLVM IR.
- **What LLVM's fold would do.** Without AnalyzeTailMasks, the `scf.if` is lowered to two blocks, each computing a 16-lane vector compare, joined by a `phi` of `<16 x i1>`. SimplifyCFG then either runs both vector compares and `select`s between the masks (default threshold), or keeps both blocks and the `phi` (threshold 0). Either way the load gets an ordinary mask.
- **What AnalyzeTailMasks does instead.** Both branches' masks are boxes, so it picks between their *scalar bounds*: `select %cond, n - off, n2 - off` → `tail_mask`. The vector mask yielded by the `scf.if` then has no users, and canonicalization deletes the whole `scf.if`. For `yielded`, after the pass and canonicalize:

  ```mlir
  %5  = arith.cmpi sgt, %arg4, %c0_i32 : i32
  %10 = arith.select %5, %8, %9 : i32
  %11 = triton_cpu.tail_mask %10 : tensor<16xi1>
  ```

  The final `.llir` has no branches at all, just `%.v = select i1 %13, i32 %2, i32 %3`, so there is no diamond left for SimplifyCFG to fold or keep.

The asm still shows two blocks, for a different reason:

```asm
bgtz  a6, .LBB0_2
mv    a2, a3        # else: take n2
.LBB0_2:
subw  a3, a2, a4    # bound = chosen n - off
```

The RISC-V backend turns a scalar `select` into a short branch because the target has no conditional-move instruction: the Zicond extension (`czero.eqz` / `czero.nez`) is not enabled. That happens during instruction selection, not in SimplifyCFG, and it costs one scalar branch per program, not two vector compares.

So the SimplifyCFG threshold decides whether *vector masks* are computed under a branch or by running both sides. For box masks, AnalyzeTailMasks avoids that choice: it picks between the scalar bounds, and the vector mask is never built.

## Tests

`test/TritonCPU/analyze-tail-masks.mlir`:
- `@if_yield`: then yields `arange < n`, else yields `arange < m`. The load gets `tail_mask (select cond, n, m)`.
- `@if_yield_inner_bound`: the then-branch computes its bound `n * 2` inside the branch. Nothing is rewritten. (`n + 1` would be rebuilt from `n` and `1`, see above.)
- `@if_yield_offs`: then yields `arange + a`, else `arange + b`. The mask `offs < n` after the `if` gets `tail_mask (n - select(cond, a, b))`.
- `@if_yield_offs_mixed`: a range in one branch and a splat in the other. Nothing is rewritten.

## Loop-carried masks

A mask that is an `scf.for` iter_arg (or an `scf.while` argument, see below) has no closed-form bound: after the loop, `m & (offs < n - k)` over all `k` has the bound `min` over every iteration. Instead, the bound is computed at run time: AnalyzeTailMasks gives the loop one extra `i32` iter_arg per axis of the mask. Before the main rewrite, `runOnOperation` calls `carryLoopMasks`:

1. **Find candidates.** `findCarriedMasks` walks back from every collected load / store / select mask through `i1` tensors, `scf.if` yields, and loop inits and yields. It collects each `scf.for` iter_arg it reaches as a candidate.
2. **Check.** Every candidate's iter_arg and loop result is given a placeholder box, all at once. The init mask and the yielded mask must both analyze to boxes: their bounds must be visible at the loop (init) and at the `scf.yield` (yield). A candidate that fails is dropped, and the rest are checked again, because they may have relied on it. This repeats until none fails.
3. **Add the bound iter_args.** Each loop is rebuilt with `replaceWithAdditionalYields`, initially with dummy inits (0) and yields. The masks' boxes are recorded in `knownBoxes`: inside the loop, the box of the new iter_args; after the loop, the box of the new results.
4. **Fill in the bounds.** The new inits get the init mask's bounds, materialized before the loop. The new yields get the yielded mask's bounds, materialized before the `scf.yield`, in terms of the new iter_args.

`analyzeMaskImpl` returns the recorded box for any value in `knownBoxes` (`is loop-carried box` in the debug log). After that, the main rewrite treats the mask like any other.

**Why it is correct:** by induction, at the start of every iteration the mask equals the box of its carried bounds. This holds for the init, and the yield keeps it. After the loop, the result is the last yielded mask, or the init if the loop ran zero times, and either way it equals the box of the loop's bound results.

**Why steps 3 and 4 are separate:** in `nested`, the inner loop's init mask is the outer loop's iter_arg, and the outer loop yields the inner loop's result. Each loop's bounds refer to values of the other. Adding every loop's iter_args first, and filling them in afterwards, removes the ordering problem.

For `carried`:

```mlir
%b0 = arith.subi %n, %off : i32
%r:2 = scf.for %k = %c0 to %flag step %c1 iter_args(%m = %m0, %b = %b0) -> (tensor<16xi1>, i32) {
  ...
  %mn = arith.andi %m, %cur : tensor<16xi1>
  %bn = arith.minsi %b, %cur_bound : i32
  scf.yield %mn, %bn : tensor<16xi1>, i32
}
%tm = triton_cpu.tail_mask %r#1 : tensor<16xi1>
```

### Removing the dead mask

After the main rewrite, every load / store reads `tail_mask` of the carried bounds, but the original `i1` mask is still updated every iteration. Its only reader is its own update, which feeds back into it through the yield. Canonicalization cannot see that: the block argument has a use (the update), so it does not look dead. `removeDeadCarriedMasks` removes it:

1. **Liveness (`liveValues`).** The roots are the operands of every op with an effect or with regions, including `tt.return` and the `scf.yield` of an `scf.if`. Liveness then flows backwards through pure ops and around loops. An `scf.for` iter_arg or result is live only if something live reads it, and then its init and yielded operand are live. An `scf.while` before argument makes its init and after-region yield operand live. An after argument or result makes its `scf.condition` argument live.
2. **Cut the cycle.** For each carried mask whose values are all dead, the yield (the `scf.for` yield, or the `scf.while` after-region yield) gets the init value instead of the update.
3. **Erase the update ops (`eraseDeadOps`).** These are pure ops whose results are dead and unused, erased users-first.
4. **Drop the loop values.** `populateRegionBranchOpInterfaceCanonicalizationPatterns` (MLIR's generic patterns for `scf.for` / `scf.while`), applied to the touched loops only, removes the now-unused block arguments and results together with their init, yield and `scf.condition` operands.
5. **Erase the now-dead init masks before the loops.**

After it, `carried` is `scf.for ... iter_args(%b = n - off) -> (i32)`: no `i1` iter_arg and no `cmpi` / `andi` in the body. A mask that is still read keeps its iter_arg: `not_box` (the load reads the mask itself), and any carried mask some non-load user needs.


## `scf.while`

An `scf.while` carries values in two places, and both are handled (`CarryKind` in `AnalyzeTailMasks.cpp`):

| Kind | Values | Sources (materialized before) |
|---|---|---|
| `ForIter` | `scf.for` iter_arg `i` and result `i` | init (the loop), yielded value (the `scf.yield`) |
| `WhileBefore` | before-region argument `i` | init (the loop), after-region yielded value (its `scf.yield`) |
| `WhileAfter` | after-region argument `i` and result `i` | `scf.condition` argument `i` (the `scf.condition`) |

A `WhileAfter` value needs carried scalars of its own. The condition arguments are computed in the before region, and those values are not visible in the after region or after the loop. Their scalars are passed through `scf.condition` as extra arguments, which become extra after-region arguments and extra results. Usually the condition just forwards a before argument, and canonicalization then merges the two copies.

`replaceWithAdditionalYields` exists only for `scf.for`, so `addScalars(scf::WhileOp, ...)` rebuilds the `scf.while`: new inits and result types, both regions moved over with `takeBody` (ops and block arguments stay the same), extra block arguments added, and dummy zeros appended to the `scf.condition` and the after-region `scf.yield`. Step 3 then replaces the dummies, as for `scf.for`.

For `while_offs` (`offs += 16` while `k < n`):

```mlir
scf.while (%offs = %range, %k = %c0, %o = %c0) : (tensor<16xi32>, i32, i32) -> (...) {
  scf.condition(%go) %offs, %k, %o : tensor<16xi32>, i32, i32
} do {
^bb0(%offs: tensor<16xi32>, %k: i32, %o: i32):
  %m = triton_cpu.tail_mask (n - %o)
  ...
  scf.yield %offs_next, %k_next, (%o + 16)
}
```

## Loop-carried index tensors

The same machinery (`carryLoopValues`) handles `i32` tensor iter_args, the common `offs += BLOCK` loop. `analyzeInt` describes an index tensor as an `IntInfo`: `scalar + i_axis` (a range along `axis`) or `scalar` (a splat). A carried index tensor keeps its shape, range or splat and its axis, which is taken from the init value. Only its `scalar` changes, so it gets one extra `i32` iter_arg:

- **Shape first (step 0):** each candidate's shape comes from analyzing its init. An init that depends on another carried value (an inner loop starting from an outer loop's `offs`) is retried once that value has a shape. Candidates that never get one are dropped.
- **Check:** the yielded value must have the same shape (same `isRange`, same `axis`). `offs * 2` is not a range at all; `offs + a range` changes the shape. Both are rejected.
- **Record:** `knownInts` maps the iter_arg (inside the loop) and the loop result (after it) to `IntInfo{isRange, axis, Leaf(new scalar)}`. `analyzeIntImpl` returns that IntInfo (`is loop-carried index` in the debug log), and masks built from the index tensor are analyzed as usual.

`findCarried` follows `i32` tensors as well as `i1` tensors back from the collected masks, so index iter_args are found through the `cmpi`.

For `carried_offs` (`offs = arange`, `offs += 16`, `mask = offs < n`):

```mlir
%r:2 = scf.for %k = ... iter_args(%offs = %range, %o = %c0) -> (tensor<16xi32>, i32) {
  %b  = arith.subi %n, %o : i32
  %m  = triton_cpu.tail_mask %b : tensor<16xi1>
  ...
  %on = arith.addi %o, %c16 : i32
  scf.yield %offs_next, %on : tensor<16xi32>, i32
}
```

The original tensor iter_arg stays: unlike a mask, the index tensor is usually still needed for the addresses (`tt.addptr %p, %offs`). Only carried masks are removed when dead.

Tests in `test/TritonCPU/analyze-tail-masks.mlir`:
- `@for_carried_offs`: init 0, yield `o + 16`, the load's mask is `tail_mask (n - o)`.
- `@for_carried_offs_not_range`: `offs * 2` gets no extra iter_arg.
- `@for_carried`: a carried mask used inside the loop and after it. It gets an `i32` iter_arg with init `n` and yield `min(bound, n - k)`, and the dead `i1` iter_arg and its `andi` are removed.
- `@for_carried_not_box`: a mask updated with `xori` gets no extra iter_arg.
- `@while_carried_offs`: an `scf.while` carrying `offs`: an extra before argument (init 0), forwarded by `scf.condition`, and yielded as `o + 16`. The load in the after region gets `tail_mask (n - o)`.
- `@while_carried_mask`: an `scf.while` carrying a mask that is used after the loop. The bound comes out as an extra result, and the dead `i1` argument is removed from both regions.
