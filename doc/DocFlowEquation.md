# Flow Equations for Tail-Mask Codegen

Transcribed from `~/tex/pdf/tail-mask-codegen.pdf`, slides 12–14 ("Form the Flow Equation").

**Legend**

| Symbol | Meaning |
|---|---|
| U | uniform |
| A | affine |
| V | vector |
| B | box |
| SB | shifted box |
| MB | multi-box |
| D | divergent |

## 1. Mask logic (`i1`): AND / OR

| Operator | Source Operands | Destination Operand | Type |
|---|---|---|---|
| AND | B, B | B | i1 |
| AND | B, SB | SB | i1 |
| AND | B, MB | MB | i1 |
| AND | SB, SB | SB | i1 |
| AND | SB, MB | MB or SB | i1 |
| AND | MB, MB | MB or SB or B | i1 |
| OR | B, B | B | i1 |
| OR | B, SB | D or B | i1 |
| OR | B, MB | MB or B | i1 |
| OR | MB, MB | D | i1 |

## 2. Index arithmetic (`i32`)

| Operator | Source Operands | Destination Operand | Type |
|---|---|---|---|
| ADD / SUB | U, U | U | i32 |
| ADD / SUB | A, U | A | i32 |
| ADD / SUB | V, U | V | i32 |
| MUL / DIV / MOD | U, U | U | i32 |
| MUL | A, U | A | i32 |
| DIV / MOD | A, U | V | i32 |
| MUL / DIV / MOD | A, A | V | i32 |
| MUL / DIV / MOD | A, V | V | i32 |
| SHIFT | A, S | A | i32 |
| SHIFT | A, A | V | i32 |

## 3. Comparisons (`i1`)

| Operator | Source Operands | Destination Operand | Type |
|---|---|---|---|
| SLT / SLE / SGT / SGE | U, U | U | i1 |
| SLT / SLE / SGT / SGE | U, A | B | i1 |
| SLT / SLE / SGT / SGE | B, U | SB or B | i1 |
| SLT / SLE / SGT / SGE | A, A | D | i1 |
| SLT / SLE / SGT / SGE | B, A | ? | i1 |
| SLT / SLE / SGT / SGE | B, B | B | i1 |
| SLT / SLE / SGT / SGE | B, SB | SB or B | i1 |
| SLT / SLE / SGT / SGE | B, MB | ? | i1 |
| SLT / SLE / SGT / SGE | SB, SB | B | i1 |
| SLT / SLE / SGT / SGE | MB, MB | D | i1 |

## Verification on RVV

Three scripts in `python/tutorials/cpu/` check the tables (shared helpers in `flow_eq_common.py`). Run them through `g.sh`:

```bash
KERNEL=flow_andor ./g.sh    # table 1
KERNEL=flow_arith ./g.sh    # table 2
KERNEL=flow_cmp   ./g.sh    # table 3
```

Each script enumerates (or samples) operands of each class on 16 lanes, classifies every result on the host, and picks one witness per observed (entry, outcome). The witnesses run on the board, and the runner checks every output lane against numpy. All four board runs (`rvv-flow-andor`, `rvv-flow-arith`, `rvv-flow-cmp`, `rvv-flow-cmp2d`) PASS with `TRITON_VSETVL_MINE=1`, so each outcome below is produced by real RVV code.

Class definitions used (one axis): **B** `[0, L)`; **SB** one run `[a, b)` with `a > 0` (a suffix counts); **MB** two or more runs; **D** a 2-D mask that is not `row_set × col_set`. All-off / all-on results are degenerate and not counted. **U** is all lanes equal; **A** is `s + c·i` with `c ≠ 0` (mod 2³²); **V** is anything else.

Each entry is judged two ways, using the order B ≤ SB ≤ MB ≤ D and U ≤ A ≤ V:
- **Unsound:** some result is above every class the table names. The entry is wrong.
- **Inexact:** nothing goes above the claim, but the list leaves out reachable classes or names some that never occur.

### Wrong entries (unsound)

| Table | Entry | Table says | Observed | Witness / reason |
|---|---|---|---|---|
| 2 | SHIFT A, S | A | **A or U or V** for `>>` (A or U for `<<`) | `i >> 1` = 0,0,1,1,… is not affine. A right shift is a division, like DIV A, U. Only `<<` keeps A. |
| 3 | U, A | B | **B or SB** | `u < s + c·i` with `c > 0` is a suffix (SB). B comes only from `A < U` or a negative stride. Every predicate gives both. |
| 3 | B, B | B | **SB or MB** | i1 compare: `b1 < b2` = `¬b1 ∧ b2` = `[L1, L2)` (SB); `b1 <= b2` = `b1 → b2` = `[0, L2) ∪ [L1, N)` (MB). Never a non-degenerate B. |
| 3 | B, SB | SB or B | **SB or B or MB** | `[0,L) ∧ ¬[a,b)` = `[0,a) ∪ [b,L)` is MB. |
| 3 | SB, SB | B | **B or SB or MB** | Interval minus interval gives 0, 1 or 2 runs. |

For i1 vs i1 the result is the same whether i1 is read as unsigned (Triton's bool `<`, true = 1) or signed (`cmpi slt` on i1, true = −1). The two readings only swap which predicate gives which class.

### The `?` entries

| Entry | Observed | Why |
|---|---|---|
| B, A | **B or SB** | With zext or sext of the box, lanes inside and outside the box compare `A` against two thresholds that differ by 1. Both are half-lines in the same direction and at most one lane apart, so the result is still one half-line. |
| B, MB | **MB or SB or B** | Every predicate is a boolean function of two interval sets, which can give any of the three. |

### Correct but inexact

| Table | Entry | Table says | Observed | Note |
|---|---|---|---|---|
| 1 | AND B, MB | MB | MB or SB or B | Slides 9/10 row 3 show `f&(B, MB) = B`. SB occurs when L cuts the first run of an MB that starts after 0. |
| 1 | OR B, SB | D or B | MB or B (1-D) | A gap between `[0,L)` and `[a,b)` gives two runs, which is MB, not D. B, MB already says "MB or B". |
| 1 | OR MB, MB | D | MB or SB or B (1-D) | In 1-D any union of intervals is B/SB/MB. D only appears in 2-D. |
| 2 | DIV/MOD A, U; DIV/MOD/SHR A, A; DIV/MOD A, V | V | also U, A | V is the catch-all class, so these are fine. Example: `(4i) / 2` is A. |
| 2 | SHL A, U | A | A or U | U happens when the shift pushes every lane's bits out (≡ 0 mod 2³²). |
| 3 | A, A | D | B or SB on the same axis; D (or a product) across axes | `a_i < b_i` with both affine on one axis is linear in `i`, so it's a half-line. D needs the operands on different axes (`rows[:, None] < cols[None, :]`, the causal mask). |
| 3 | MB, MB | D | MB or SB or B | Same reason as OR MB, MB. |

Not in the tables but measured: **ADD/SUB A, A → A or U**; **OR SB, SB → MB or SB**; **OR SB, MB → MB or SB or B**.

**OR in 2-D:** OR is not a per-axis equation. Even OR B, B is D for 41% of random 2-D boxes (an L shape). It is B only when one box contains the other, as in slide 11 row 3. AND of 2-D products is always the product of the per-axis ANDs.
