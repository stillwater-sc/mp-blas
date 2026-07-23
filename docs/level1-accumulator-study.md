# Level-1 accumulator study: the dot product

**Question.** For a low-precision *element* type, how wide an *accumulator* does
a dot product need as the length `n` grows — and where does an exact
super-accumulator (Universal `quire`) earn its cost over a merely-promoted
`double` accumulator?

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
which specializes MTL5's `accumulator_traits` for a Universal posit quire —
MTL5 itself stays free of any Universal dependency.

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

This mirrors the mp-spice KLU result that *the quire helps a direct solve but is
washed out once a cheaper mechanism (there, iterative refinement) already
absorbs the accumulation error* — here the cheaper mechanism is simply a wider
built-in accumulator, and it only runs out of headroom when the element type
approaches the accumulator's own width.
