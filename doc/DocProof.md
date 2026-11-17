# Reversible ⇔ Injective

## Definitions

An operation: y = f(v₁, v₂, …, vₙ)  (y may be a tuple of results, e.g. lo, hi)

**Notation.** v₋ᵢ means *all the other inputs*, every index except i:

  v₋ᵢ = (v₁, …, vᵢ₋₁, vᵢ₊₁, …, vₙ)

| i | vᵢ | v₋ᵢ |
|---|---|---|
| 1 | v₁ | (v₂, v₃) |
| 2 | v₂ | (v₁, v₃) |
| 3 | v₃ | (v₁, v₂) |

**Reversible.** f is reversible in vᵢ if there is a function gᵢ such that

  gᵢ(f(v), v₋ᵢ) = vᵢ  **for every choice of v₁, …, vₙ**.

- v₋ᵢ says *which* variables gᵢ may use (all operands except vᵢ).
- "For every choice" says *for which values* the equation must hold: all of them, with one fixed gᵢ.

**Injective.** With v₋ᵢ fixed, vᵢ ↦ f(vᵢ, v₋ᵢ) is injective (one-to-one: different inputs give different outputs):

  f(vᵢ, v₋ᵢ) = f(vᵢ′, v₋ᵢ)  ⇒  vᵢ = vᵢ′

Rule of thumb: an operation is injective in an input exactly when it never throws away information about that input.

## Why "for all inputs" matters

Example: y = a & b, asking whether it is reversible in a.

- If b = 0xFF, then y = a, so g(y, b) = y rebuilds a. ✅
- If b = 0, then y = 0 for every a, and no g can recover a. ❌

The equation holds for *some* inputs but not all, so `&` is **not** reversible in a. Without "for all inputs", the definition would wrongly accept it.

## Examples (8-bit wraparound integers)

| Operation | Injective in `a`? | Why |
|---|---|---|
| `y = a + b` | ✅ yes | `a = y − b` |
| `y = a ^ b` | ✅ yes | `a = y ^ b` |
| `y = a * 3` | ✅ yes | 3 is odd, so it has a multiplicative inverse mod 2⁸ |
| `y = a * 2` | ❌ no | `a=0` and `a=128` both give `y=0` |
| `y = a & b` | ❌ no | with `b=0`, every `a` gives `y=0` |
| `y = a >> 1` | ❌ no | `a=4` and `a=5` both give `2` |
| `(lo, hi) = a * b` (widening) | ✅ yes, if b ≠ 0 | nothing is lost; `a = (hi:lo) / b` |

## Proof: reversible ⇒ injective (by contradiction)

A proof by contradiction needs only two assumptions: the hypothesis (reversible) and the negated conclusion (not injective).

Negating an implication: "P ⇒ Q" is false exactly when P is true and Q is false.

**Assume** f is reversible in vᵢ. So there is a function gᵢ with

  gᵢ(f(vᵢ, v₋ᵢ), v₋ᵢ) = vᵢ  for every vᵢ and v₋ᵢ.  (★)

**Suppose, for contradiction,** that f is not injective in vᵢ. Then there exist v₋ᵢ and vᵢ ≠ vᵢ′ with

  f(vᵢ, v₋ᵢ) = f(vᵢ′, v₋ᵢ). Call this common value y.

**Apply (★) twice.** It holds for all inputs, so it holds for both of these:

- With input (vᵢ, v₋ᵢ): gᵢ(y, v₋ᵢ) = vᵢ
- With input (vᵢ′, v₋ᵢ): gᵢ(y, v₋ᵢ) = vᵢ′

**gᵢ is a function.** The same input (y, v₋ᵢ) can only give one output, so

  vᵢ = gᵢ(y, v₋ᵢ) = vᵢ′.

This contradicts vᵢ ≠ vᵢ′. So f must be injective in vᵢ. ∎

The "for all inputs" in the definition is what makes the proof work: it lets (★) be applied to both (vᵢ, v₋ᵢ) and (vᵢ′, v₋ᵢ).

## Proof: injective ⇒ reversible (direct)

Assume f is injective in vᵢ. Define gᵢ(y, v₋ᵢ):

- If some vᵢ satisfies f(vᵢ, v₋ᵢ) = y, it is **unique** by injectivity, so gᵢ returns that vᵢ.
- Otherwise (y is never produced), gᵢ returns anything, e.g. 0.

Then for every input, gᵢ(f(vᵢ, v₋ᵢ), v₋ᵢ) returns the unique preimage, which is vᵢ. So f is reversible in vᵢ. ∎

### Example: f(a, b) = a² + b

Let a ∈ {0, 1, 2, 3} and b ∈ {0, 1}. Is f reversible in a?

**Step 1: check injectivity in a.** Fix b and list the outputs:

| a | f(a, 0) | f(a, 1) |
|---|---|---|
| 0 | 0 | 1 |
| 1 | 1 | 2 |
| 2 | 4 | 5 |
| 3 | 9 | 10 |

In each column, the four outputs are all different, so f is injective in a for each fixed b.

The columns overlap (y = 1 shows up in both), and that's fine: injectivity is checked **with b fixed**, which is why gᵢ also takes b as input.

**Step 2: build g(y, b) by following the two rules.**

- **Rule 1:** if some a gives f(a, b) = y, return that a. It is unique by Step 1.
- **Rule 2:** otherwise, return 0.

| y | g(y, 0) | g(y, 1) |
|---|---|---|
| 0 | 0 | *0 (rule 2)* |
| 1 | 1 | 0 |
| 2 | *0 (rule 2)* | 1 |
| 3 | *0 (rule 2)* | *0 (rule 2)* |
| 4 | 2 | *0 (rule 2)* |
| 5 | *0 (rule 2)* | 2 |
| … | … | … |
| 9 | 3 | *0 (rule 2)* |
| 10 | *0 (rule 2)* | 3 |

The italic entries are values of y that f never produces for that b. Their output doesn't matter, because g is never asked about them.

**Step 3: verify g(f(a, b), b) = a for all 8 inputs.**

| (a, b) | y = f(a, b) | g(y, b) | = a? |
|---|---|---|---|
| (0, 0) | 0 | 0 | ✅ |
| (1, 0) | 1 | 1 | ✅ |
| (2, 0) | 4 | 2 | ✅ |
| (3, 0) | 9 | 3 | ✅ |
| (0, 1) | 1 | 0 | ✅ |
| (1, 1) | 2 | 1 | ✅ |
| (2, 1) | 5 | 2 | ✅ |
| (3, 1) | 10 | 3 | ✅ |

All inputs pass, so f is reversible in a. ∎

The proof never needed a formula like a = √(y − b). It only needed the lookup table, and injectivity guarantees that the table never has two answers in one cell.

### Counter-example: where the construction breaks

Take f(a, b) = a & b with b = 1:

| a | f(a, 1) |
|---|---|
| 0 | 0 |
| 1 | 1 |
| 2 | 0 |
| 3 | 1 |

For y = 0, rule 1 finds **two** candidates (a = 0 and a = 2). Whichever one g(0, 1) returns, it is wrong for the other. That is exactly the step that needs injectivity.

## Proof: injective ⇒ reversible (by contradiction)

Negating "reversible": "some gᵢ works for all inputs" is false means **every gᵢ fails on at least one input**:

  ¬(∃gᵢ ∀v: gᵢ(f(v), v₋ᵢ) = vᵢ)  ≡  ∀gᵢ ∃v: gᵢ(f(v), v₋ᵢ) ≠ vᵢ

(∃ becomes ∀, ∀ becomes ∃, = becomes ≠.)

**Assume** f is injective in vᵢ. For every v₋ᵢ and every vᵢ, vᵢ′:

  f(vᵢ, v₋ᵢ) = f(vᵢ′, v₋ᵢ)  ⇒  vᵢ = vᵢ′  (◆)

**Suppose, for contradiction,** that f is not reversible in vᵢ. Then every function gᵢ fails on some input:

  ∀gᵢ ∃(vᵢ, v₋ᵢ): gᵢ(f(vᵢ, v₋ᵢ), v₋ᵢ) ≠ vᵢ  (✗)

**Pick a specific gᵢ.** Define gᵢ(y, v₋ᵢ):

- If some w satisfies f(w, v₋ᵢ) = y, return such a w.
- Otherwise, return 0.

**By (✗), this gᵢ fails somewhere.** So there is an input (vᵢ\*, v₋ᵢ\*) with

  y\* = f(vᵢ\*, v₋ᵢ\*) and gᵢ(y\*, v₋ᵢ\*) = w ≠ vᵢ\*.

**What w is.** y\* is produced by vᵢ\*, so the first rule of gᵢ applies. That means gᵢ returned a w with

  f(w, v₋ᵢ\*) = y\* = f(vᵢ\*, v₋ᵢ\*).

**Apply (◆) to the pair (w, vᵢ\*).** Since (◆) is true, one of two cases must hold:

- **Case (1): f(w, v₋ᵢ\*) ≠ f(vᵢ\*, v₋ᵢ\*).** Impossible, because we just showed they are equal.
- **Case (2): f(w, v₋ᵢ\*) = f(vᵢ\*, v₋ᵢ\*) and w = vᵢ\*.** This contradicts w ≠ vᵢ\*.

Both cases fail. So f must be reversible in vᵢ. ∎

**Why gᵢ must still be built:** the negation (✗) only says every gᵢ fails somewhere. To get a contradiction, one concrete gᵢ has to be tested, and the preimage function is the one that injectivity makes correct. So the contradiction proof and the direct proof rest on the same idea. The direct proof is just shorter.

## Conclusion

**Reversible in vᵢ ⇔ injective in vᵢ (with v₋ᵢ fixed).**
