# Level-1 accumulator study: `dot` and `nrm2`

**Question.** For a low-precision *element* type, how wide an *accumulator* does
a level-1 reduction need as the length `n` grows — and where does an exact
super-accumulator (Universal `quire`) earn its cost over a merely-promoted
`double` accumulator?

Two reductions are swept: **`dot`** (`x·y`) and **`nrm2`** (`sqrt(x·x)`). The
accumulation error of `nrm2` lives entirely in its sum-of-squares dot, so both
reuse the same accumulator machinery (`nrm2` is computed as `sqrt` of the
sum-of-squares dot; MTL5's `two_norm<Acc>` delivers `value<Acc>` with no
`Result` seam, so it cannot round a quire out — the `dot(x,x)` formulation is
the portable mixed path).

The three precisions of a mixed-precision inner product are independent
(MTL5 `mtl/math/accumulator_traits.hpp`): the **element** precision (bandwidth
in), the **accumulator** precision (registers), and the **result** precision
(out). We fix a low-precision element type and sweep the accumulator across the
four strategies MTL5 exposes, delivering every result rounded to `double` so the
numbers isolate **accumulation** error, not storage rounding:

| strategy   | call                              | semantics                                    |
|------------|-----------------------------------|----------------------------------------------|
| `native`   | `dot<T, double>`                  | products summed in the element type `T`      |
| `fma`      | `dot<fma_accumulator<double>, double>` | fused multiply-add, one rounding per term |
| `promoted` | `dot<double, double>`             | products summed in `double`                  |
| `quire`    | `dot<quire<T>, double>`           | exact sum of products, single round-out      |

The `quire` accumulator is supplied by mp-blas's
[`include/mtl/math/quire_accumulator.hpp`](../include/mtl/math/quire_accumulator.hpp),
which specializes MTL5's `accumulator_traits` for the Universal quire of a
posit, cfloat, or lns element — MTL5 itself stays free of any Universal
dependency.

Error is relative to a `long double` reference over the **same quantized element
values**, so quantization error cancels and only the accumulator differs.
Reproduce with:

```bash
./build/applications/level1/dot_accumulator_study/dot_accumulator_study
```

## Results

Relative error vs the long-double reference (deterministic pseudo-random
elements in (-1, 1)).

### element type `posit<16,2>`

| n        | native   | fma      | promoted | quire    |
|---------:|---------:|---------:|---------:|---------:|
|      256 | 1.5e-03  | 0        | 0        | 0        |
|     1024 | 5.3e-04  | 0        | 0        | 0        |
|     4096 | 9.0e-03  | 0        | 0        | 0        |
|    16384 | 1.6e-03  | 0        | 0        | 0        |
|    65536 | 1.1e-02  | 0        | 0        | 0        |
|   262144 | 1.0e-01  | 0        | 0        | 0        |
|  1048576 | 2.5e-01  | 0        | 0        | 0        |

### element type `posit<32,2>`

| n        | native   | fma      | promoted | quire    |
|---------:|---------:|---------:|---------:|---------:|
|      256 | 1.3e-08  | 3.7e-16  | 3.8e-16  | 1.8e-16  |
|     1024 | 1.3e-08  | 5.6e-16  | 6.2e-16  | 9.1e-17  |
|     4096 | 3.5e-08  | 4.4e-15  | 2.4e-15  | 2.9e-17  |
|    16384 | 2.3e-07  | 1.6e-16  | 1.6e-16  | 4.2e-17  |
|    65536 | 9.9e-07  | 6.2e-15  | 5.9e-15  | 5.9e-17  |
|   262144 | 8.5e-07  | 2.1e-14  | 1.9e-14  | 1.2e-16  |
|  1048576 | 6.1e-07  | 3.7e-14  | 3.3e-14  | 1.9e-16  |

### element types `cfloat` and `lns` (dot, at n=65536)

The posit story repeats for `cfloat` — **with two quire defects the sweep
surfaced in Universal** (issues filed, see below). `cfloat<32,8>`'s quire is
exact (~1e-17); `cfloat<16,5>`'s quire **floors at ~1e-9** — worse than a
promoted `double`, which is exact for 16-bit elements:

| type / n=65536 | native   | fma / promoted | quire       |
|----------------|---------:|---------------:|------------:|
| `cfloat<16,5>` | 1.4e-02  | **0**          | 1.9e-09 ⚠️  |
| `cfloat<32,8>` | 2.6e-06  | 4.6e-16        | 3.7e-17 ✓   |
| `lns<16,8>`    | 8.0e-02  | 6.4e-15        | 1.7e-06 ⚠️  |
| `lns<32,16>`   | 2.3e-02  | 6.7e-15        | 6.1e-11 ⚠️  |

For **`lns` the "exact" quire is a liability**: its `quire_mul` routes the
product through `double`, so the quire floor (1.7e-6 for `lns<16,8>`) is far
*worse* than a promoted `double` accumulator (~6e-15). For lns, promote — do not
use the quire.

### `nrm2`

`nrm2 = sqrt(x·x)` mirrors `dot` on every element type (the sum-of-squares is
the same reduction). Native `nrm2` degrades even faster with `n` because the
squares are all positive — e.g. `posit<16,2>` native `nrm2` is **78% wrong at
n=65536** — while `promoted`/`quire` stay at the machine floor (posit, wide
cfloat) or the type's quire floor (narrow cfloat, lns). Run the tool for the
full `nrm2` tables.

## Findings

1. **Same-precision accumulation is unsafe at scale.** Summing the products in
   the element type (`native`) degrades with `n` — for `posit<16,2>` the dot
   product is **25% wrong at n≈10⁶**, because each new product is added to a
   running sum that has drifted far from the term's magnitude (swamping). This
   is the accumulator, not the elements: the *same* values summed in a wider
   accumulator are far more accurate.

2. **For 16-bit elements, a promoted `double` accumulator is already exact —
   the quire buys nothing.** A `posit<16,2>` product lands well inside
   `double`'s 53-bit significand, and even 10⁶ such terms don't exhaust it at
   this reference resolution, so `promoted` and `quire` are both at the floor.
   **Recommendation: for ≤16-bit elements, accumulate in `double`; skip the
   quire.**

3. **For 32-bit elements over long reductions, the quire earns its cost.** With
   `posit<32,2>` elements, `promoted`/`fma` double accumulation is good but
   **grows with `n`** (rounding error accumulates: ~3e-14 at n≈10⁶). The
   **quire stays flat at ~1e-16 regardless of `n`** — exact accumulation has no
   `n`-dependence. At n≈10⁶ the quire is **~200× more accurate** than a promoted
   double accumulator. **Recommendation: when the element type approaches the
   accumulator's width (32-bit posit vs double) and reductions are long, prefer
   the exact quire.**

4. **`fma` ≈ `promoted` here.** Fusing the product removes one rounding per term,
   but with a `double` accumulator over these operands the product was already
   near-exact, so the two track each other. The fused path matters more when the
   accumulator equals the element precision (no widening headroom).

## Takeaway

Choose the dot-product accumulator by the **ratio of element width to
accumulator width and the reduction length**, not by habit:

- ≤16-bit elements → promoted `double` is exact enough; the quire is wasted work.
- 32-bit elements, long reductions → the exact quire is flat in `n` where a
  promoted accumulator drifts.

> **Scope caveat, added by
> [dot-product-characterization.md](dot-product-characterization.md) (issue #9).**
> These rules are calibrated on *this page's* data — uniform pseudo-random
> elements in (-1, 1), which is a benign, well-conditioned distribution. They are
> statements about types only because the data is held fixed. On ill-conditioned
> input the first rule fails outright: a promoted `double` accumulator loses all
> correct digits once the dot-product condition number `cond = 2|x|'|y|/|x'y|`
> exceeds `~1/u_double ≈ 1e16`, regardless of how narrow the elements are. And
> even at `cond = 2` (its best possible value) a 16-bit *native* accumulator can
> be 25% wrong through swamping alone. The data-driven criterion — measure `cond`
> and `n_eff`, then choose — is developed there.

This mirrors the mp-spice KLU result that *the quire helps a direct solve but is
washed out once a cheaper mechanism (there, iterative refinement) already
absorbs the accumulation error* — here the cheaper mechanism is simply a wider
built-in accumulator, and it only runs out of headroom when the element type
approaches the accumulator's own width.

For **cfloat and lns the recommendation is currently even simpler — promote to
`double`** — because their quire is not yet dependably exact (below). Once the
Universal fixes land, cfloat should match posit (exact, flat in `n`); lns needs
an exact log-domain product before its quire is worth using at all.

## Universal gaps surfaced

Extending the sweep to cfloat and lns turned this study into a conformance test
for Universal's quire. Three requirements were generated and filed:

1. **[stillwater-sc/universal#1201]** — `cfloat.hpp` / `lns.hpp` do not include
   their `fdp.hpp`, so `quire_mul` is undeclared out of the box (unlike
   `posit.hpp`). The mp-blas adapter works around this by including the `fdp`
   headers explicitly.
2. **[stillwater-sc/universal#1202]** — `quire_traits<cfloat>::radix_point` is
   undersized for no-subnormals configs: it reserves room for `minpos²` but not
   for a *normal* small value's product, which needs `2·fbits` more fraction
   bits. Result: `cfloat<16,5>` quire dot floors at ~2⁻³¹ (~1e-9) instead of
   exact. Confirmed — turning subnormals on (which widens `radix_point`) makes
   it bit-exact; `cfloat<32,8>` is already wide enough.
3. **[stillwater-sc/universal#1203]** — `quire_mul(lns, lns)` forms the product
   in `double`, so the lns quire is not exact (single-product error 7.6e-6 for
   `lns<16,8>`) and is *worse* than a promoted `double` accumulator. Needs an
   exact log-domain product, or a documented non-exactness guarantee.

The mp-blas quire adapter and `test_dot_quire` encode current behavior: they
assert the quire is **never worse than native** for every element type, and
**exact** only where it genuinely is today (posit any width, wide cfloat). When
the fixes land, the exactness assertions can extend to cfloat<16,5> and lns.

[stillwater-sc/universal#1201]: https://github.com/stillwater-sc/universal/issues/1201
[stillwater-sc/universal#1202]: https://github.com/stillwater-sc/universal/issues/1202
[stillwater-sc/universal#1203]: https://github.com/stillwater-sc/universal/issues/1203
