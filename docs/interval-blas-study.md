# Interval BLAS: verified enclosures

**Question** (issue [#12](https://github.com/stillwater-sc/mp-blas/issues/12)).
Issue #9 measured how *wrong* a dot product is. This measures how much it can
**prove**. Given interval element vectors, how tight an enclosure does each
accumulation strategy deliver — and does an exact accumulator make interval
arithmetic useful where it currently is not?

This is the third and original leg of the Kulisch argument. [Issue
#9](dot-product-characterization.md) established that the exact scalar product
was designed for **verified computing** — an exact residual makes defect
correction effective, which is what yields results with machine-proved bounds —
and then measured only accuracy and reproducibility. Interval arithmetic is the
consumer of that guarantee, and this is where it gets measured.

**Status: complete.** Containment gate; `dot`; `ger`/`gemv`; `rot` and wrapping;
`gemm` and rank-k; verified solve; and `nrm2` (§14), which was Phase 1's deferred
item and is now closed.

---

## 1. Correctness first: the Phase 0 gate

An interval result is only worth measuring if it is *correct*, and the failure
mode is counter-intuitive: an unsound implementation produces intervals that are
too **narrow**, which reads as impressive tightness. Universal's `interval` did
exactly that until [universal#1234](https://github.com/stillwater-sc/universal/issues/1234)
— `interval(0.1) * interval(0.1)` returned a **zero-width** interval around a
value the exact product does not equal.

So every width in this note sits behind
[`tests/level1/test_interval_containment.cpp`](../tests/level1/test_interval_containment.cpp),
which asserts the Fundamental Theorem of Interval Arithmetic

> for all `x ∈ X`, `y ∈ Y`: `x ∘ y ∈ fl(X ∘ Y)`

evaluated over the exact reals, across element types, operators, and the issue-#9
regimes. It includes a **negative control** that hand-builds the pre-#1234
zero-width interval and asserts the harness rejects it — a gate that cannot fail
is worthless, which is precisely the flaw that let #1234 and #1202 survive
Universal's own test suite.

Two portability lessons came out of building it, both worth carrying:

- **`long double` is not a usable high-precision reference.** It is 80-bit on
  x86-64 Linux but only 64-bit on ARM64 and MSVC. A reference built on it went
  *blind* on macOS — the negative control passed, meaning the gate accepted the
  unsound interval. The fix was to carry the reference as an unevaluated sum of
  two doubles built from error-free transformations, which are exact on any
  IEEE-754 implementation and deliver ~106 bits identically everywhere.
- **`relative_width` is meaningless when the enclosure brackets zero.** Two
  accumulators returned the same `[-64, 64]` with identical absolute width but
  relative widths six orders of magnitude apart, purely from where the midpoint
  landed. Use `absolute_width` there. This will matter in Phase 3 (`rot`) and
  Phase 5 (`solve`), where results near zero are the norm.

---

## 2. The hypotheses

```
H1   naive interval accumulation rounds both endpoints outward once per term,
     so its width GROWS with n and with cond -- it reports the error BOUND.

H2   exact accumulation in a quire pair, rounded outward exactly ONCE on the way
     out, has width set only by that final rounding -- so it is FLAT in both,
     and reports the actual uncertainty.
```

H2 is the substantive claim. If it holds, the exact dot product does not merely
buy a few digits; it is the difference between an enclosure that certifies
something and one that certifies nothing.

## 3. The interval quire accumulator

[`include/mtl/math/interval_quire_accumulator.hpp`](../include/mtl/math/interval_quire_accumulator.hpp)
specializes MTL5's `accumulator_traits` for an `interval<Scalar>` value type over
a **pair of quires** — one per endpoint. Both accumulate exactly; the single
outward rounding happens in `value()`. Per the architectural rule this lives in
mp-blas, not MTL5.

**Endpoint selection is the interesting part.** Interval multiplication picks its
endpoints from the four corner products `a·c, a·d, b·c, b·d`. In eight of the nine
sign cases *which* corners win is determined by the operand signs alone, so each
chosen corner goes straight to `quire_mul` and is accumulated exactly, with no
comparison needed.

The ninth case — both operands straddling zero — genuinely requires comparing
`a·d` against `b·c` and `a·c` against `b·d`. Comparing **rounded** products there
can pick the wrong corner when two candidates are near-equal, and the wrong corner
yields an enclosure that is too *narrow*: a containment violation. So that case
compares the candidates **exactly, in a scratch quire**, and is the only path that
pays for one.

**One deliberate looseness.** `value()` steps each endpoint outward
unconditionally rather than only when the conversion was inexact, because
`quire::convert_to<T>()` goes through `double` and is limited to 53 significand
bits — so for a quire holding more than that (exactly the interesting case)
neither the fact nor the direction of its rounding is observable from outside.
The cost is at most one ulp per endpoint, and it shows up as `R = 3` below rather
than `R = 1`. A conditional, tighter version needs an exactness query on the
quire — the same Stage-1/Stage-2 split Universal used for #1234 and #1248.

---

## 4. Measurements: `dot`

```bash
./build/applications/level1/interval_dot_study/interval_dot_study [nmax]
```

gcc and clang produce **byte-identical output**. All three strategies **deliver
`interval<double>`**, following the convention in
[level1-accumulator-study.md](level1-accumulator-study.md): fix the element
precision, vary only the accumulator, round every result out to the same width, so
the numbers isolate *accumulation* and are not silently capped by the output type.

> Getting this wrong is easy and I did it first: delivering the quire to
> `interval<Scalar>` caps it at the element type's ulp (`3.7e-09` for
> `posit<32,2>`) and understates it by seven orders of magnitude — a property of
> the output type, not of the accumulator.

`digits` = `-log10(relative width)` = decimal digits **certified**. Negative means
the enclosure is wider than its own midpoint: nothing is proved.

### 4.1 `interval<posit<32,2>>`, n = 4096

| regime | cond | naive | promoted | **quire** | dig(nv) | dig(pr) | **dig(qr)** |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform | 1.3e+02 | 2.8e-05 | 3.4e-13 | **2.2e-16** | 4.6 | 12.5 | **15.7** |
| positive | 2.0e+00 | 9.0e-05 | 3.4e-13 | **3.3e-16** | 4.0 | 12.5 | **15.5** |
| graded | 3.8e+01 | 6.5e-04 | 1.2e-15 | **3.4e-16** | 3.2 | 14.9 | **15.5** |
| cancel 1e-3 | 2.1e+06 | 1.1e+00 | 8.3e-09 | **4.3e-16** | −0.0 | 8.1 | **15.4** |
| cancel 1e-6 | 2.1e+09 | 6.1e+02 | 8.3e-06 | **4.2e-16** | −2.8 | 5.1 | **15.4** |
| cancel 1e-9 | 2.1e+12 | 3.9e+02 | 8.3e-03 | **4.1e-16** | −2.6 | 2.1 | **15.4** |
| kahan 1e-6 | 2.7e+19 | 1.6e+03 | 4.2e+02 | **4.2e-16** | −3.2 | −2.6 | **15.4** |

`interval<cfloat<32,8>>` behaves the same, shifted by its wider `u`; see the
program output.

### 4.2 Length scaling — H1 against H2 directly

`interval<cfloat<32,8>>`, regime `cancel 1e-6`, `cond ≈ 2e8` held fixed:

| n | naive w | quire w | naive/n₀ | **quire/n₀** |
|---:|---:|---:|---:|---:|
| 64 | 2.04e+01 | 4.24e-16 | 1.0 | **1.0** |
| 128 | 3.47e+01 | 4.24e-16 | 1.7 | **1.0** |
| 256 | 7.54e+01 | 4.24e-16 | 3.7 | **1.0** |
| 512 | 2.16e+02 | 4.24e-16 | 10.6 | **1.0** |
| 1024 | 2.91e+03 | 4.24e-16 | 142.2 | **1.0** |
| 2048 | 3.60e+03 | 4.24e-16 | 176.1 | **1.0** |
| 4096 | 1.14e+03 | 4.24e-16 | 55.6 | **1.0** |

The quire width is not approximately flat — it is **the same number at every
length**, to every printed digit.

---

## 5. Findings

1. **H2 holds, in its strongest form.** The quire column is `4.1e-16` – `4.3e-16`
   across *every* regime and *every* length: `cond` spanning **19 decades**
   (2.0 → 2.7e19) and `n` spanning 64 → 4096. It certifies **15.4–15.7 digits**
   throughout. The enclosure has stopped tracking the error bound and started
   tracking the actual uncertainty, which is the whole point.

2. **H1 holds.** Naive interval width grows with both axes and crosses into
   uselessness. At `cancel 1e-3 / n=4096` it certifies **0.0 digits**; by
   `kahan / n=4096` it certifies **−3.2** — an enclosure 1600× wider than the
   value it encloses. Containment holds the entire time, which is exactly the
   trap: a rigorous, correct, worthless answer.

3. **A promoted `double` interval accumulator is not a substitute — it fails at a
   predictable threshold.** It is excellent up to `cond ≈ 1e16` and then collapses:
   on `kahan` (`cond = 2.7e19`) it certifies **−2.6 digits**, proving nothing,
   while the quire certifies 15.4. The crossover is `cond ≈ 1/u_double`, which is
   **the same threshold issue #9 found for accuracy** (finding 3 there). The
   accuracy and enclosure stories land on the identical constant — evidence they
   are two views of one mechanism, not two separate results.

4. **The headline.** On `kahan 1e-6 / n=4096`, naive certifies −3.2 digits and
   promoted −2.6. Both prove *nothing at all*. The quire proves **15.4 digits**.
   That is not an accuracy improvement; it is the difference between a computation
   that produces a verified result and one that does not. This is the concrete
   form of Kulisch's claim, measured.

5. **`R = 3.0`, constant.** The quire enclosure is 3× the narrowest correct one,
   uniformly, from the unconditional outward step described in §3 — near-optimal
   and, importantly, *independent of the problem*. Naive's overestimation grows
   without bound.

6. **Tapered precision resurfaces.** Delivered to `interval<posit<32,2>>` rather
   than `interval<double>`, the quire's relative width drifts ~30× across the
   cancel ladder — not from conditioning but because the *result magnitude* shrinks
   with the residual and posit's relative ulp depends on magnitude. On
   fixed-precision `cfloat<32,8>` the same sweep is flat to within 5%. This is the
   third independent appearance of "posit's `u` is not a single number" (see
   [§4.3 of the characterization note](dot-product-characterization.md)), and the
   reason `tests/level1/test_interval_quire.cpp` states H2 on a fixed-precision
   type and holds posit to a weaker claim.

---

## 6. Measurements: level 2 — `ger` and `gemv`

```bash
./build/applications/level2/interval_matvec_study/interval_matvec_study [nmax]
```

Phase 1 showed exact accumulation collapses the enclosure width. Phase 2 asks
**which half of the arithmetic that win came from** — accumulation or
multiplication — by measuring the two level-2 operators that separate them. gcc
and clang produce byte-identical output.

### 6.1 `ger`, the control

`ger` computes `A += alpha·x·yᵀ`: one multiply-add per element and **no reduction
at all**. MTL5's `ger` accordingly has no `Accumulator` parameter, and that is
correct rather than an omission — there is nothing to accumulate.

`interval<posit<32,2>>`, worst element over the whole matrix:

| n | max abs width | max R |
|---:|---:|---:|
| 4 | 2.24e-08 | 4.50 |
| 16 | 2.24e-08 | 4.50 |
| 64 | 2.24e-08 | 4.50 |
| 256 | 2.24e-08 | 4.50 |

Identical to every digit, at every size. Interval multiplication contributes a
**fixed, small, problem-size-independent** width, within 4.5× of the narrowest
possible enclosure for a multiply-add.

That is the control result, and it settles the attribution: **Phase 1's win came
from removing accumulation rounding, not from anything about products.** Had this
column grown with `n`, the Phase 1 story would have needed rewriting.

### 6.2 Repeated `ger`: a reduction with nowhere to put an accumulator

A single `ger` has no reduction — but `k` successive rank-1 updates put `k`
products into every element of `A`. That *is* a reduction, spread across calls,
and MTL5's per-call interface has nowhere to hold an accumulator across them:

| k | 1 | 4 | 16 | 64 | 256 |
|---|---|---|---|---|---|
| max abs width | 2.24e-08 | 1.04e-07 | 4.62e-07 | 3.31e-06 | 2.57e-05 |
| vs k=1 | 1.0 | 4.7 | 20.7 | 148 | **1152** |

The width grows roughly linearly in `k` and **no accumulator seam reaches it**.
This is the same shape as the `trsv` limitation already noted in the roadmap: the
operator's interface, not its arithmetic, is what forecloses the fix. The
structural remedy is to express a rank-`k` update as a single `gemm`
(`A += X·Yᵀ`), which *does* expose the seam — i.e. it becomes a Phase 4 question.

### 6.3 `gemv`

`y = A x` is `ger`'s per-element widening plus one reduction per output row, and
`mult()`'s `Accumulator` seam takes the interval quire directly. Worst row
reported; all strategies deliver `interval<double>`.

`interval<posit<32,2>>`:

| regime | n | cond | naive | **quire** | dig(nv) | **dig(qr)** |
|---|---:|---:|---:|---:|---:|---:|
| uniform | 128 | 3.7e+01 | 1.5e-06 | **2.7e-16** | 5.8 | **15.6** |
| graded | 128 | 8.6e+00 | 4.4e-05 | **2.7e-16** | 4.4 | **15.6** |
| cancel 1e-3 | 128 | 6.1e+04 | 3.7e-03 | **4.3e-16** | 2.4 | **15.4** |
| cancel 1e-9 | 128 | 6.1e+10 | 6.1e+01 | **4.1e-16** | −1.8 | **15.4** |
| kahan 1e-6 | 16 | 7.9e+17 | 1.9e+07 | **4.2e-16** | **−7.3** | **15.4** |
| kahan 1e-6 | 128 | 1.1e+18 | 2.7e+02 | **4.2e-16** | −2.4 | **15.4** |

`interval<cfloat<32,8>>` behaves the same. `R(quire) = 3.0` throughout, exactly as
in Phase 1.

Phase 1 reproduces row-wise with no surprises — which is the expected and desired
outcome for `gemv`, since each output element *is* a dot product. The worst naive
case in the whole study appears here: `kahan / n=16` certifies **−7.3 digits**, an
enclosure ~10⁷× wider than the value it encloses, while the quire certifies 15.4.

---

## 7. Findings, level 2

7. **The attribution holds: the win is accumulation, not multiplication.** Single
   `ger` width is `2.24e-08` at n = 4, 16, 64 and 256 — identical to every digit —
   with `R` pinned at 4.50. Interval multiplication is well-behaved and
   size-independent; all of Phase 1's degradation lived in the reduction. This is
   the control that licenses the Phase 1 story.

8. **`gemv` is Phase 1, row-wise, with nothing new** — quire flat at
   `4.1e-16`–`4.3e-16` and `R = 3.0` across `cond` spanning 17 decades, naive
   crossing into negative certified digits. Worth stating plainly *because* it is
   unsurprising: each output element is a dot product, so a level-2 result that
   differed from level 1 would have indicated a bug in the seam rather than a
   discovery.

9. **A reduction can hide in an operator that has no reduction.** Repeated `ger`
   accumulates across *calls*, so its width grows ~linearly in the number of
   updates (1152× at k=256) and no per-call accumulator seam can reach it. The
   remedy is structural — re-express rank-`k` as one `gemm` — not arithmetic. Same
   shape as the `trsv` seam limitation: what forecloses the fix is the operator's
   interface, not its arithmetic.

---

## 8. Measurements: `rot` and the wrapping effect

```bash
./build/applications/level1/interval_rot_study/interval_rot_study [kmax]
```

Phases 1 and 2 showed exact accumulation collapses a reduction's enclosure width,
and that the win came from accumulation rather than multiplication. This phase
went looking for where that **stops** being true. `rot` was the predicted place,
on two independent grounds:

1. the reduction is length **two**, so there is almost no accumulation error for
   an exact accumulator to remove;
2. the dominant error is **wrapping** — a rotation maps an axis-aligned box to a
   tilted one, and storing the result as an interval pair takes the axis-aligned
   hull. That is a loss in the *representation*, not the arithmetic, and it
   compounds geometrically.

Confirming a predicted negative is what bounds the claims made in Phases 1–2, so
it was measured rather than asserted. gcc and clang agree byte-for-byte.

### 8.1 One rotation: the multiplier is geometry

A box of width `w` rotated by θ has an axis-aligned hull of width
`w·(|cos θ| + |sin θ|)`:

| θ | 0° | 5° | 15° | 30° | 45° | 60° | 75° | 90° |
|---|---|---|---|---|---|---|---|---|
| `\|c\|+\|s\|` | 1.0000 | 1.0834 | 1.2247 | 1.3660 | 1.4142 | 1.3660 | 1.2247 | 1.0000 |
| measured | 1.0112 | 1.0896 | 1.2332 | 1.3750 | 1.4291 | 1.3769 | 1.2351 | 1.0168 |

Agreement within 2% at every angle, including the endpoints where the rotation
maps axes to axes and there is no wrapping at all.

### 8.2 Repeated rotation: does an exact accumulator change the rate?

64 rotations from a point start, `interval<posit<32,2>>`, θ = 45°:

| k | naive w | quire w | nv/qr | naive rate | quire rate | predicted |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.49e-08 | 7.45e-09 | 2.00 | — | — | — |
| 8 | 7.00e-07 | 3.43e-07 | 2.04 | 1.7333 | 1.7280 | 1.4142 |
| 16 | 1.18e-05 | 5.74e-06 | 2.06 | 1.5603 | 1.5575 | 1.4142 |
| 32 | 3.03e-03 | 1.47e-03 | 2.06 | 1.4833 | 1.4820 | 1.4142 |
| 64 | 1.98e+02 | 9.66e+01 | 2.06 | **1.4478** | **1.4472** | **1.4142** |

The rate estimates start high and descend toward `|c|+|s|` because they are
measured from `k=1`, so the early steps still carry the rounding seed; the
asymptote is the wrapping factor.

Same shape at θ = 5° (rates 1.1249 vs 1.1252 against a prediction of 1.0834) and
for `interval<cfloat<32,8>>` (1.4478 vs 1.4455).

### 8.3 The angle decides, not the arithmetic

Identical arithmetic, identical 64 steps, `interval<posit<32,2>>`:

| θ | `\|c\|+\|s\|` | final width | mechanism |
|---|---|---|---|
| 90° | 1.000 | **1.8e-06** | bounded — rounding only |
| 45° | 1.414 | **2.0e+02** | geometric — wrapping |

Eight orders of magnitude apart, decided purely by the angle.

---

## 9. Findings: `rot`

10. **The predicted negative result holds: an exact accumulator changes the
    constant, not the rate.** Naive and quire growth rates agree to 3–4
    significant figures at every angle and element type (1.4478 vs 1.4472 at 45°;
    1.1249 vs 1.1252 at 5°), while the quire's benefit is a flat ~2× at every `k`.
    Wrapping is a loss in the representation, and no accumulator reaches it.

11. **The mechanism is confirmed as geometric, not arithmetic.** The measured
    per-step growth matches `|c| + |s|` to within 2% for a single rotation, and
    the repeated-rotation rate converges to the same constant from above. At
    `|c|+|s| = 1` the width stays bounded over 64 steps while at `|c|+|s| = 1.414`
    it grows by 10 orders of magnitude — same code, same element type, only the
    angle differs. So this is not "intervals always blow up"; it is a specific,
    predictable, angle-dependent effect.

12. **Two independent reasons the quire cannot help here**, and it is worth
    keeping them apart. The reduction is only 2 terms long, so there is nearly
    nothing to accumulate exactly; *and* the error is representational rather than
    arithmetic. Either alone would sink it. This is the clean counterexample to
    "the exact dot product makes interval arithmetic work" — it makes *reductions*
    work, which is a narrower and more defensible claim.

### The three kinds of limit

Phases 1–3 have now produced three structurally different obstacles, and
conflating them would overstate what the exact dot product does:

| limit | example | can an exact accumulator help? |
|---|---|---|
| **dependency** | `nrm2` as `sqrt(dot(x,x))` (§10) | **No** — mathematical |
| **interface** | repeated `ger`, `trsv` (§7) | Not as written; fixable by re-expressing the operation |
| **wrapping** | repeated `rot` (§9) | **No** — representational |

Only where the obstacle is *accumulation* does the quire win — and there it wins
completely (§5).

---

## 10. Measurements: level 3 — `gemm` and the rank-k update

```bash
./build/applications/level3/interval_matmul_study/interval_matmul_study [nmax]
```

Two jobs. The first is confirmatory; the second changes an entry in the
three-limits table above. gcc and clang agree byte-for-byte.

### 10.1 `gemm` reproduces Phase 1 element-wise

`interval<posit<32,2>>`, worst element:

| regime | n | cond | naive | **quire** | dig(nv) | **dig(qr)** |
|---|---:|---:|---:|---:|---:|---:|
| uniform | 32 | 1.1e+01 | 2.6e-07 | **2.3e-16** | 6.6 | **15.6** |
| graded | 32 | 3.2e+00 | 8.1e-06 | **3.1e-16** | 5.1 | **15.5** |
| cancel 1e-3 | 32 | 1.9e+04 | 2.6e-04 | **4.3e-16** | 3.6 | **15.4** |
| cancel 1e-9 | 32 | 1.9e+10 | 2.6e+02 | **4.1e-16** | −2.4 | **15.4** |
| kahan 1e-6 | 32 | 1.0e+18 | 3.6e+07 | **4.2e-16** | **−7.6** | **15.4** |

As at level 2, this is the expected outcome rather than a new one: each `C(i,j)`
*is* a dot product. It is worth asserting precisely because a level-3 result that
differed would indicate a broken accumulator seam, not a discovery.

### 10.2 The rank-k update: does re-expressing dissolve the interface limit?

Phase 2 found that `k` successive `ger` calls form a reduction spread across
*calls*, which no per-call accumulator seam can reach. The proposed remedy was
structural — express the same computation as one `gemm`, `A += X·Yᵀ`, so the sum
becomes a single k-term reduction the `gemm` seam *can* accumulate exactly.

All three formulations compute the same quantity, and all three are checked
against the same reference. Widths normalized by `max |A(i,j)|`, because a rank-k
update has many elements with midpoints near zero (see the `relative_width`
caveat in §1).

`interval<posit<32,2>>`, n = 8:

| k | ger × k | gemm naive | **gemm + quire** |
|---:|---:|---:|---:|
| 1 | 2.75e-08 | 1.84e-08 | **2.74e-16** |
| 4 | 4.92e-08 | 3.58e-08 | **2.66e-16** |
| 16 | 1.07e-07 | 1.07e-07 | **2.45e-16** |
| 64 | 4.32e-07 | 3.90e-07 | **2.50e-16** |
| 256 | 1.84e-06 | 1.80e-06 | **2.62e-16** |
| **growth** | **×67** | **×98** | **×0.96** |

`interval<cfloat<32,8>>` is the same shape (×67, ×97, ×0.96), with the final gap
reaching 1.1e+11.

---

## 11. Findings: level 3

13. **`gemm` is Phase 1 element-wise**, at 15.4–15.6 certified digits regardless
    of conditioning, while naive reaches **−7.6 digits** on `kahan` — an enclosure
    ~10⁷ times wider than the value it encloses. No level-3-specific effect.

14. **The interface limit is genuinely fixable — and the fix needs both halves.**
    Re-expressing `k` rank-1 updates as one `gemm` exposes the accumulator seam
    that a sequence of `ger` calls cannot. But **re-expression alone does
    nothing**: naive `gemm` degrades with `k` exactly as repeated `ger` does (×98
    vs ×67). It is re-expression *plus* exact accumulation that makes the width
    flat (×0.96 over a 256× increase in `k`), and the gap reaches 7e+09.

    The second half is the one that is easy to assume away, so the test asserts
    naive `gemm` **does** degrade — if that ever stops being true, this finding is
    wrong and should be revisited rather than quietly inherited.

### The three limits, revisited

Phase 4 updates the table from §9:

| limit | example | can an exact accumulator help? |
|---|---|---|
| **dependency** | `nrm2` as `sqrt(dot(x,x))` (§12) | **No** — mathematical |
| **interface** | repeated `ger`, `trsv` | **Yes, after re-expressing** — `A += X·Yᵀ` as one `gemm` restores it fully (§11.14) |
| **wrapping** | repeated `rot` (§9) | **No** — representational |

So of the three obstacles found, exactly one is an artifact of how the
computation was *written* rather than what it *is* — and that one dissolves
completely once the reduction is made visible to the accumulator. The other two
stand.

---

## 12. Measurements: verified solve — the exact residual

```bash
./build/applications/level2/verified_solve_study/verified_solve_study [nmax]
```

This is what the line of work was built toward. §1 established that Kulisch
designed the exact scalar product for **verified computing**: an exact residual
makes defect correction effective. Phases 1–4 built and characterized the
instrument; this applies it to the problem it was designed for.

### 12.1 The honest baseline: naive interval `trsv`

Interval arithmetic applied directly to elimination, `interval<posit<32,2>>`,
n = 16:

| off | −0.25 | −0.50 | −0.75 | −0.90 | −0.99 |
|---|---|---|---|---|---|
| max rel. width | 5.96e-07 | 1.55e-06 | 2.26e-06 | 3.65e-06 | 3.74e-06 |
| certified digits | 6.2 | 5.8 | 5.6 | 5.4 | 5.4 |

Degrading with conditioning, as expected. Included because it is the baseline the
literature says to *avoid*, and a study that skipped straight to the method that
works would be hiding the comparison.

### 12.2 The payoff: defect correction, working vs exact residual

Hilbert system, `posit<32,2>` (u = 3.7e-09), relative forward error per
correction step:

| n | cond | cond·u | residual | it0 | it1 | it2 | it3 | it4 |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 4 | 2.8e+04 | 1.1e-04 | working | 6.0e-06 | 9.6e-06 | 1.8e-05 | 1.8e-05 | **1.8e-05** |
| | | | **EXACT** | 6.0e-06 | 3.7e-09 | 3.7e-09 | 3.7e-09 | **3.7e-09** |
| 6 | 2.9e+07 | 1.1e-01 | working | 1.2e-02 | 1.9e-02 | 2.9e-02 | 1.7e-02 | **2.1e-02** |
| | | | **EXACT** | 1.2e-02 | 1.9e-05 | 3.1e-08 | 4.4e-09 | **3.0e-09** |
| 8 | 1.9e+10 | 7.0e+01 | working | 3.7e-01 | 1.0e-01 | 4.6e-02 | 1.1e+00 | 4.9e-01 |
| | | | EXACT | 3.7e-01 | 2.6e-01 | 1.8e-01 | 1.3e-01 | 9.2e-02 |

`cfloat<32,8>` (u = 6.0e-08) behaves identically: at n = 4 the working residual
stagnates at 1.2e-04 while the exact residual reaches **5.9e-08 = u**.

### Why a general system and not `trsv`

Triangular systems are solved far more accurately than their condition number
suggests (Higham, ASNA ch. 8). Measured while building this: back-substitution
lands within a few ulp, defect correction has nothing to correct, and the two
residuals are indistinguishable. A Phase 5 built on `trsv` would have passed
vacuously and concluded, wrongly, that the exact residual does not help. The
Hilbert matrix is used because its forward error genuinely tracks `cond·u`.

---

## 13. Findings: verified solve

15. **The exact residual decouples accuracy from conditioning.** With a
    working-precision residual, defect correction stagnates at ~`cond·u` — and at
    n = 4 it gets *worse* with each step (6.0e-06 → 1.8e-05), because it is
    correcting with noise. With an exact residual it reaches **`u` exactly**
    (3.7e-09 for posit<32,2>, 5.9e-08 for cfloat<32,8>).

16. **And it holds as conditioning grows.** From n = 4 to n = 6 the condition
    number rises **1000×** (2.8e+04 → 2.9e+07) and the final accuracy with an
    exact residual is *unchanged* at ~`u`, while the working-precision residual
    degrades from 1.8e-05 to 2.1e-02. That is the decoupling stated precisely:
    the exact residual removes `cond` from the achievable accuracy.

    This is the classical Wilkinson result on iterative refinement, and it is
    Kulisch's argument for the exact scalar product — here measured on posit and
    cfloat arithmetic.

17. **The claim is bounded, and the bound is not the residual.** Past
    `cond·u ≈ 1` (n = 8, `cond·u = 70`) the exact residual reaches only 9.2e-02.
    The working-precision *factorization* carries no correct digits at that point,
    and no residual can rescue it. So the exact residual removes the `cond`
    dependence **of the residual**, not of the factorization — which is the honest
    form of the claim and the one the test asserts.

### What the exact dot product is for

Across five phases the answer is narrow and specific:

| obstacle | can an exact accumulator help? |
|---|---|
| accumulation error in a reduction | **Yes — completely.** 15.4 digits where naive certifies none (§5) |
| a reduction hidden across calls | **Yes, after re-expressing** as one reduction (§11) |
| catastrophic cancellation in a residual | **Yes — decouples accuracy from `cond`** (§13) |
| dependency (`nrm2` as `dot(x,x)`) | No — mathematical (§14) |
| wrapping (repeated `rot`) | No — representational (§9) |
| a factorization that has already failed | No — outside its reach (§13.17) |

The exact dot product makes **reductions** exact. Everything it buys follows from
that one property, and everything it cannot do falls outside it. The verified
solve is the case where that single property is worth the most: it is the
difference between an answer whose accuracy degrades with the problem's
conditioning and one that does not.

---

## 14. `nrm2`: why it is not `dot(x,x)`, and what fixes it

This was Phase 1's deferred item. It is the one obstacle in this study that is
**mathematical** rather than structural, and it is worth stating carefully because
it bounds the claim made everywhere else.

### The problem

Computing `nrm2` as `sqrt(dot(x,x))` hands the same interval to both arguments,
and interval arithmetic cannot know they are the same object. For an element
`X = [a,b]` straddling zero, `dot` therefore evaluates the general product `X·X`,
whose lower endpoint is `min(a·b, b·a) = a·b < 0` — whereas the true square is

```
X² = [0, max(a², b²)]
```

which can never be negative. This is the **dependency problem**: interval
arithmetic overestimates whenever a variable occurs more than once. It is not an
accumulation error, so **no accumulator fixes it** — an exact quire will
faithfully and exactly accumulate the wrong corner products.

### What it costs — the lower bound, not the width

`interval<posit<32,2>>`, n = 16, half-width 0.5, varying how many elements
straddle zero. Both columns accumulate exactly, so the only difference is
square-vs-product:

| straddling | `sqrt(dot(x,x))` certifies | `Σ X²` certifies | width ratio |
|---:|---:|---:|---:|
| 0.00 | ‖x‖ ≥ 2.2933 | ‖x‖ ≥ 2.2933 | 1.00 |
| 0.25 | ‖x‖ ≥ 1.7323 | ‖x‖ ≥ 2.0002 | 1.08 |
| 0.50 | ‖x‖ ≥ 0.7120 | ‖x‖ ≥ 1.5834 | 1.29 |
| 0.75 | **‖x‖ ≥ 0** | **‖x‖ ≥ 1.2901** | 1.55 |
| 1.00 | ‖x‖ ≥ 0 | ‖x‖ ≥ 0 | 1.00 |

The width ratio peaks at a modest 1.55×, which understates the damage. The real
loss is the **lower bound**: at 75% straddling `sqrt(dot(x,x))` certifies
`‖x‖ ≥ 0`, which proves *nothing* — it cannot establish that the vector is
nonzero. A lower bound on `‖x‖` is what proves a vector non-degenerate, a residual
genuinely nonzero, or a matrix not rank-deficient, so losing it is a qualitative
failure rather than a precision one.

(At 100% straddling both certify 0 — correctly, since the zero vector really is
in the box. That is the sanity check, not a failure.)

### Two independent defects

`nrm2` needs **both** fixes, and they are separable — which is why
[`include/sw/mp_blas/interval_nrm2.hpp`](../include/sw/mp_blas/interval_nrm2.hpp)
provides a square *and* uses a quire pair. With degenerate inputs and no element
straddling zero, the dependency loss is zero and only accumulation remains:

| n | naive accumulation | quire accumulation |
|---:|---:|---:|
| 64 | 3.37e-07 | **2.24e-16** |
| 4096 | 5.03e-05 | **4.29e-16** |

The familiar Phase 1 picture: naive grows with `n`, the quire is flat.

### A detail worth recording

`square()` clamps its lower bound at zero, and that is not defensive
programming — it is necessary. Outward rounding drives an **exact zero** lower
endpoint just below zero (`round_down(0)` is the next value toward −∞), so both
`[0,m]·[0,m]` and `[0,b]·[0,b]` come back with a tiny *negative* lower bound. A
square is provably non-negative, so raising it to zero keeps the enclosure valid
and makes it tighter. Caught by the test asserting `square()` is never negative.

### Where this leaves the claim

`nrm2` is the case that shows the boundary of what the exact dot product does.
Its accumulation half is fixed completely by the quire, exactly as everywhere
else in this study. Its dependency half is not touched by the accumulator at all,
and is fixed only by changing the *expression* — evaluating a square instead of
asking `dot` for a product. That is the same shape as the interface limit of
§11.14, and the opposite of the wrapping limit of §9, which no re-expression
repairs.

## Related

- [dot-product-characterization.md](dot-product-characterization.md) — issue #9:
  `cond`, `n_eff`, and the accuracy/reproducibility measurements this builds on.
- [`include/mtl/math/interval_quire_accumulator.hpp`](../include/mtl/math/interval_quire_accumulator.hpp)
  — the accumulator; [`include/sw/mp_blas/interval_containment.hpp`](../include/sw/mp_blas/interval_containment.hpp)
  — containment and tightness metrics.
- `tests/level1/test_interval_containment.cpp` (Phase 0 gate),
  `tests/level1/test_interval_quire.cpp` (H1/H2).
- Universal: #1234 outward rounding (fixed), #1248/#1253/#1256 tightness,
  #1259 posit `to_native` (open).
