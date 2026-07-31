# Interval arithmetic, and the three ways it goes wrong

Interval arithmetic is the machinery underneath verified computing: replace each
number by an interval known to contain it, define arithmetic so containment is
preserved, and the final interval provably contains the exact answer.

Being *sound* is the easy half. Being **useful** — narrow enough to certify
something — is where it gets interesting, and there are exactly three ways to
lose it. This note develops the arithmetic from first principles, then treats
each failure mode in the detail it deserves, because they have genuinely
different cures and conflating them leads to reaching for the wrong tool.

It expands §5 of [verified-computing.md](verified-computing.md), which places
interval arithmetic in its historical context alongside Wilkinson and Kulisch.
The measurements referenced throughout are in
[interval-blas-study.md](interval-blas-study.md).

> **On the math.** Display formulas use ```` ```math ```` fenced blocks, because
> markdown preprocesses the contents of `$$…$$` and corrupts LaTeX silently — see
> the note at the top of [verified-computing.md](verified-computing.md) for the
> details and the rules for inline math.

---

## Contents

1. [The representation and the invariant](#1-the-representation-and-the-invariant)
2. [The four operations](#2-the-four-operations)
3. [Outward rounding: the part that makes it real](#3-outward-rounding)
4. [Inclusion isotonicity: why composition works](#4-inclusion-isotonicity)
5. [Interval extensions: the idea that explains everything after](#5-interval-extensions)
6. [Failure (a): accumulation](#6-failure-a-accumulation)
7. [Failure (b): dependency](#7-failure-b-dependency)
8. [Failure (c): wrapping](#8-failure-c-wrapping)
9. [Choosing the right cure](#9-choosing-the-right-cure)
10. [What we measured](#10-what-we-measured)
11. [References](#11-references)

---

## 1. The representation and the invariant

An interval is an ordered pair of representable numbers standing for the whole
closed set between them:

```math
X = [\underline{x}, \overline{x}] = \lbrace t \in \mathbb{R} : \underline{x} \le t \le \overline{x} \rbrace .
```

Two derived quantities are used constantly:

```math
\mathrm{mid}(X) = \tfrac{1}{2}(\underline{x} + \overline{x}),
\qquad
\mathrm{wid}(X) = \overline{x} - \underline{x} .
```

A **degenerate** (or *thin*) interval has `wid(X) = 0` and represents a single
number exactly. A **thick** interval carries genuine uncertainty.

The one invariant that matters — everything else is bookkeeping in service of it:

> If `X` contains the true value of `x`, then after any sequence of interval
> operations the result contains the true value of the corresponding real
> expression.

That is worth stating carefully because it is *weaker* than people often assume.
It guarantees **containment**, not **tightness**. `[-∞, +∞]` satisfies it
perfectly and tells you nothing. All three failure modes below are failures of
tightness, never of containment — which is exactly what makes them dangerous.
A wrong answer announces itself; a correct but vacuous one does not.

---

## 2. The four operations

Addition and subtraction are immediate. The extremes of a sum occur at the
extremes of the summands:

```math
X + Y = [\underline{x} + \underline{y},\; \overline{x} + \overline{y}],
\qquad
X - Y = [\underline{x} - \overline{y},\; \overline{x} - \underline{y}] .
```

Note the crossover in subtraction: the smallest possible difference uses the
smallest `x` and the **largest** `y`. Getting this backwards is the classic
beginner's bug, and it produces intervals that are too narrow — i.e. unsound.

Multiplication is not monotone in the same way, because signs interact. The
product of two intervals attains its extremes at one of the four **corner
products**:

```math
X \cdot Y = [\;\min(\underline{x}\underline{y},\, \underline{x}\overline{y},\, \overline{x}\underline{y},\, \overline{x}\overline{y}),\;\;
              \max(\underline{x}\underline{y},\, \underline{x}\overline{y},\, \overline{x}\underline{y},\, \overline{x}\overline{y})\;] .
```

Evaluating all four products and taking min/max always works, but in eight of the
nine sign cases the winners are determined by the **signs alone** — no comparison
needed. Only when both operands straddle zero must the products actually be
compared. That matters for an exact implementation: comparing *rounded* products
can pick the wrong corner and yield an interval that is too narrow, so the
straddling case needs an exact comparison. (This repo's interval quire does
precisely that; see `include/mtl/math/interval_quire_accumulator.hpp`.)

Division is multiplication by a reciprocal, and is only defined when the divisor
excludes zero:

```math
X / Y = X \cdot [1/\overline{y},\; 1/\underline{y}], \qquad 0 \notin Y .
```

When `0 ∈ Y` the exact result set is unbounded, and the only enclosure is
`[-∞, +∞]`. That requires the underlying number type to *have* an infinity — not
all do. Posit has NaR instead, and `numeric_limits<posit>::infinity()` returns
maxpos, a finite value; returning `[-maxpos, maxpos]` would be an enclosure that
excludes everything beyond maxpos while claiming to contain it. (Filed as
stillwater-sc/universal#1277.)

---

## 3. Outward rounding

Everything above is exact real arithmetic. On a computer the endpoints must be
representable, and this is where implementations get it wrong.

Each endpoint must be rounded **away from the interval**: the lower bound toward
`-∞`, the upper toward `+∞`.

```math
X + Y = [\;\nabla(\underline{x} + \underline{y}),\;\; \Delta(\overline{x} + \overline{y})\;]
```

where `∇` is round-toward-`-∞` and `Δ` is round-toward-`+∞`. Round to nearest
instead and each endpoint can move *inward* by up to half an ulp, so the interval
becomes too narrow and the invariant of §1 is lost.

This is not a hypothetical. Universal's `interval` did exactly this until
recently — `interval(0.1) * interval(0.1)` returned a **zero-width** interval
around a value the exact product does not equal (stillwater-sc/universal#1234).
The failure is quiet in the worst way: the intervals look impressively *tight*,
precisely because they are wrong.

Two implementation notes worth knowing:

- **Directed rounding vs. `nextafter`.** Changing the FPU rounding mode is fast
  but does not generalize: it has no effect on software number systems such as
  posit or cfloat. `nextafter`-based outward stepping is portable across every
  `Scalar`, at the cost of up to one ulp of extra width.
- **Conditional widening.** Stepping outward *unconditionally* costs an ulp even
  when the operation was exact. Error-free transformations recover this: TwoSum
  yields the exact rounding error, and its sign says which endpoint (if either)
  actually needs to move. That is the difference between a 1-ulp-optimal
  enclosure and a merely correct one.

---

## 4. Inclusion isotonicity

The property that makes composition work is **inclusion isotonicity**: the
operations are monotone with respect to set inclusion.

```math
X_1 \subseteq X_2 \;\text{ and }\; Y_1 \subseteq Y_2
\quad\Longrightarrow\quad
X_1 \circ Y_1 \subseteq X_2 \circ Y_2 .
```

From this, the **Fundamental Theorem of Interval Arithmetic** follows: if each
input interval contains its true value, the computed result contains the true
result, however long the computation. Containment composes.

This is the whole reason interval arithmetic is worth doing. It is also why
tightness does *not* compose: nothing in the theorem says the result stays
narrow, and §§6–8 are three different ways it fails to.

---

## 5. Interval extensions

Here is the concept that makes the next three sections make sense, and it is the
one most often skipped.

Given a real function `f`, its **interval extension** `F` is what you get by
replacing every operation in a particular *expression* for `f` with its interval
counterpart. The crucial word is *expression*: `F` depends on how `f` was
written, not just on what `f` is.

What you would *like* is the **united extension** — the exact range:

```math
f(X) = \lbrace f(t) : t \in X \rbrace .
```

What you *get* is an enclosure of it:

```math
f(X) \subseteq F(X) .
```

The gap between the two is the **overestimation**, and every failure mode below
is a different reason for that gap:

| failure | source of the gap |
|---|---|
| accumulation | rounding, injected once per operation |
| dependency | `F` treats repeated occurrences of a variable as independent |
| wrapping | the *result* of `F` must be stored as a box, and the true set is not one |

Note that the first is about arithmetic, the second about the expression, and the
third about the data structure. Three different levels — which is why they need
three different cures.

---

## 6. Failure (a): accumulation

**What it is.** Every operation rounds outward, adding roughly an ulp of width.
Do `n` of them and you have accumulated `n` ulps, regardless of whether the
underlying quantity was uncertain at all.

Reduce a vector of *degenerate* (zero-width) intervals — inputs carrying no
uncertainty whatsoever — and the result still comes out thick. All of that width
was manufactured by the arithmetic.

**Why it is the worst one in practice.** The width does not merely grow with `n`;
it grows with the **conditioning**. A dot product's interval width tracks the
Wilkinson error bound

```math
\mathrm{wid}\big(\mathrm{IA}(x^Ty)\big) \;\sim\; \gamma_n \cdot |x|^T|y|
\;=\; \gamma_n \cdot \tfrac{1}{2}\,\mathrm{cond}(x,y) \cdot |x^Ty| ,
```

so the enclosure reports the *error bound* rather than the actual uncertainty.
On an ill-conditioned reduction the bound is enormous, the enclosure is enormous,
and it certifies nothing — while remaining perfectly correct.

**The cure: exact accumulation.** If the entire reduction is accumulated without
rounding and rounded outward exactly once at the end, the width is set by that
single rounding — independent of `n` and of `cond`. This is Kulisch's exact dot
product, applied to both endpoints.

Measured on `interval<posit<32,2>>` at `n = 4096`
([study §4](interval-blas-study.md)):

| regime | cond | naive interval | exact quire |
|---|---:|---:|---:|
| uniform | 1.3e+02 | 2.8e-05 | **2.2e-16** |
| cancel 1e-9 | 2.1e+12 | 3.9e+02 | **4.1e-16** |
| kahan 1e-6 | 2.7e+19 | 1.6e+03 | **4.2e-16** |

The quire column is flat across nineteen decades of conditioning. The naive
column, at `cond = 2.7e19`, is an interval **1600× wider than the value it
encloses** — sound, and worthless.

This is the failure mode that yields completely, and it is the one this
repository is mostly about.

---

## 7. Failure (b): dependency

**What it is.** Interval arithmetic has no memory. Each occurrence of a variable
in an expression is treated as an independent quantity ranging over its interval.
So for `X = [1, 2]`:

```math
X - X = [1 - 2,\; 2 - 1] = [-1, 1] \qquad \text{not} \qquad [0, 0].
```

This is not a bug. `X - X` correctly computes `{a - b : a ∈ X, b ∈ X}`, which
really is `[-1, 1]`. It is simply not what you meant. You meant
`{a - a : a ∈ X} = {0}`, and the notation gave the machine no way to know.

Likewise `X / X ≠ [1,1]`, and for `X = [-1, 2]`:

```math
X \cdot X = [-2, 4] \qquad \text{whereas} \qquad X^2 = [0, 4].
```

The square can never be negative; the product of two independent draws from `X`
certainly can.

### 7.1 The rule

> An interval extension computes the exact range **if each variable occurs
> exactly once** in the expression (a *single-use expression*). Every additional
> occurrence is an opportunity for overestimation.

### 7.2 The cure: rewrite the expression

Because the overestimation comes from the *form* of the expression, changing the
form fixes it. The classic demonstration — evaluate `f(x) = x² - x` on `X = [0,1]`.

Naively, with two occurrences of `x`:

```math
X^2 - X = [0,1] - [0,1] = [-1, 1].
```

Complete the square so `x` appears **once**:

```math
f(x) = \left(x - \tfrac{1}{2}\right)^2 - \tfrac{1}{4}
\quad\Longrightarrow\quad
\left([0,1] - \tfrac{1}{2}\right)^2 - \tfrac{1}{4}
= [-\tfrac{1}{2}, \tfrac{1}{2}]^2 - \tfrac{1}{4}
= [0, \tfrac{1}{4}] - \tfrac{1}{4}
= [-\tfrac{1}{4}, 0].
```

And `[-1/4, 0]` is the **exact** range: `f` has its minimum `-1/4` at `x = 1/2`
and its maximum `0` at both endpoints. Identical function, identical arithmetic,
four times narrower — purely from writing it differently.

### 7.3 When single-use form is unreachable

Most expressions cannot be rewritten that way. The standard tools then are:

- **Mean-value / centered form.** Using the mean value theorem on `m = mid(X)`:

  ```math
  F_{\mathrm{mv}}(X) = f(m) + F'(X)\,(X - m).
  ```

  The width is now driven by `wid(X)²` rather than `wid(X)` for smooth `f`, so on
  narrow intervals it is dramatically tighter.

- **Subdivision.** Split `X` into `p` pieces, evaluate each, take the hull.
  Overestimation typically falls like `1/p` (naive form) or `1/p²` (centered
  form). Costs `p` evaluations.

- **Monotonicity.** If `F'(X)` does not contain zero, `f` is monotone on `X` and
  the exact range is just the interval spanned by the two endpoint values.

- **Taylor models.** Carry a polynomial approximation plus a small interval
  remainder; the dependency lives in the polynomial, where it cancels
  symbolically.

### 7.4 Why no accumulator helps

This is worth stating flatly, because the instinct after §6 is to reach for the
quire. An exact accumulator will accumulate the *wrong corner products* exactly.
The overestimation was introduced when the expression was written, before any
arithmetic happened. Better arithmetic cannot recover information the expression
discarded.

Measured instance: computing `nrm2` as `sqrt(dot(x,x))` hands the same interval
to both arguments, so a zero-straddling element contributes `a·b < 0` instead of
`0`. With 75% of elements straddling zero, that formulation certifies
`‖x‖ ≥ 0` — proving nothing at all — where a dedicated sum-of-squares certifies
`‖x‖ ≥ 1.29` ([study §14](interval-blas-study.md)).

---

## 8. Failure (c): wrapping

This is the subtlest of the three and the one most often misdiagnosed, so it gets
the longest treatment.

### 8.1 The paradox

Take a rotation. It is an isometry: it preserves lengths, areas, angles, and the
norm of every vector. It loses nothing, and it is perfectly conditioned
(`κ = 1`). Apply it to a set and you get a congruent copy of that set.

Now apply it to an interval box, in interval arithmetic, and the box **grows** —
geometrically, without bound, under repeated application. Eight 45° rotations
return every point to where it started, yet the computed enclosure is sixteen
times wider than it began.

No rounding is required for this. It happens in exact arithmetic.

### 8.2 Where the width comes from

A 2-D interval vector is an axis-aligned rectangle. Rotating it produces a
**tilted** rectangle — which is not an interval vector, and cannot be stored as
one. The only option is to store the smallest axis-aligned box containing it: the
**hull**. And the hull of a tilted rectangle is bigger than the rectangle.

Concretely, let `X × Y` have widths `w_x` and `w_y`, and rotate by `θ` with
`c = cos θ`, `s = sin θ`:

```math
\begin{pmatrix} x' \\ y' \end{pmatrix}
= \begin{pmatrix} c & -s \\ s & c \end{pmatrix}
  \begin{pmatrix} x \\ y \end{pmatrix} .
```

Interval arithmetic evaluates `x' = cX - sY` and `y' = sX + cY` independently. The
width of a sum of independent interval terms is the sum of the widths, so:

```math
\mathrm{wid}(X') = |c|\,w_x + |s|\,w_y ,
\qquad
\mathrm{wid}(Y') = |s|\,w_x + |c|\,w_y .
```

For a square box, `w_x = w_y = w`, both become

```math
w' = \big(|\cos\theta| + |\sin\theta|\big)\, w .
```

That factor is `1` only when `θ` is a multiple of 90° — where the rotation maps
axes onto axes and the tilted rectangle *is* axis-aligned — and reaches its
maximum `√2 ≈ 1.4142` at `θ = 45°`, where the tilt is worst.

### 8.3 Why it compounds

The damage is that the hull is taken **at every step**, and each step starts from
the previous step's inflated box rather than from the true set. After `k`
rotations of `θ`:

```math
w_k = \big(|\cos\theta| + |\sin\theta|\big)^k\, w_0 .
```

Geometric growth. Contrast this with the truth: `k` rotations by `θ` compose into
a *single* rotation by `kθ`, so the exact set is congruent to the original and its
hull is at most `√2` wider — **for any `k`**. The overestimation is therefore

```math
\frac{\big(|\cos\theta| + |\sin\theta|\big)^k}{|\cos k\theta| + |\sin k\theta|}
\;\ge\; \frac{\big(|\cos\theta| + |\sin\theta|\big)^k}{\sqrt{2}} .
```

**A worked case.** Take `θ = 45°`, so the growth factor is `√2` per step, and
`k = 8` — a full 360°.

- **Truth:** eight 45° rotations is the identity. The exact set is *exactly* the
  original box. Overestimation should be 1.
- **Interval arithmetic:** `(√2)⁸ = 16`. The enclosure is 16× wider, having
  gained nothing but wrapping.

At `k = 64` the factor is `(√2)⁶⁴ = 2³² ≈ 4.3 × 10⁹`, for a computation whose
exact answer is again the identity.

### 8.4 Why it is a representation problem

The three failure modes look similar from a distance — all three produce an
enclosure wider than the truth — but their causes sit at different levels, and
wrapping is the only one that is about the **data structure**.

- It is **not accumulation.** No rounding occurred. Redo the arithmetic exactly,
  in a quire, at unlimited precision, and the growth is unchanged: the exact hull
  of the exact tilted rectangle is still `(|c| + |s|)` times wider.
- It is **not dependency.** Each of `x` and `y` occurs once in `cX - sY`. The
  expression is already in single-use form; there is nothing to rewrite.

What is lost is that the set after one rotation is *tilted*, and the interval
vector cannot express tilt. The information destroyed is the **correlation**
between the components — and a product of intervals is, by definition, a set with
no correlation between its coordinates.

That is the whole diagnosis: **wrapping is what you pay for representing a
correlated set with an uncorrelated one.**

### 8.5 What actually fixes it

Since the problem is the representation, the cure is a different representation —
one that can express correlation. This is a real and well-developed area:

- **Parallelepiped method** (Moore). Store the set as `A·U + c` where `U` is a
  box and `A` a matrix. A rotation multiplies into `A`; no hull is taken, so no
  wrapping. Its weakness is that `A` becomes ill-conditioned over many steps and
  the enclosure degrades for a different reason.

- **Lohner's QR method.** The standard fix for the above: periodically
  re-factorize `A = QR` and keep the well-conditioned orthogonal part. This is
  what production interval ODE solvers use, and it makes long integrations
  practical.

- **Zonotopes / affine arithmetic.** Represent the set as a centre plus a sum of
  generators, `c + Σ εᵢ gᵢ` with `εᵢ ∈ [-1,1]`. Linear maps act exactly on the
  generators, so linear transformations wrap not at all. Affine arithmetic also
  tracks shared `εᵢ` between quantities, which incidentally cures much of §7's
  dependency problem too.

- **Ellipsoids.** Closed under linear maps, and rotations act on them exactly.

- **Taylor models.** Polynomial plus interval remainder; correlations live in the
  polynomial part.

Note what every one of these has in common: none of them is an improvement to the
*arithmetic*. They all change what is being stored.

### 8.6 Where it bites

Wrapping was identified by Moore in the 1960s in the setting where it hurts
most — **interval integration of ODEs**, where each step is roughly a linear map
applied to the previous enclosure, and thousands of steps compound. Without a
wrapping-aware representation, interval ODE solvers are useless beyond a few
steps; with Lohner's method they are practical.

The same mechanism appears wherever repeated linear transformations are applied
to an enclosure: **Givens/QR sweeps** (a rotation per step, by construction),
**Kalman filtering** and other recursive state estimation, and repeated
**change of basis**.

### 8.7 Measured

From [interval-blas-study.md §8](interval-blas-study.md), 64 rotations of
`interval<posit<32,2>>` from a point start:

| θ | `\|cos θ\| + \|sin θ\|` | measured per-step growth | final width |
|---|---:|---:|---:|
| 5° | 1.0834 | 1.1249 | 2.5e-05 |
| 45° | 1.4142 | 1.4478 | 2.0e+02 |
| 90° | 1.0000 | — | 1.8e-06 (bounded) |

Two things to read from this. The measured growth rate matches the geometric
prediction `|cos θ| + |sin θ|` to within a couple of percent, confirming the
mechanism is the hull and not the arithmetic. And at 90°, where the prediction is
exactly `1`, the width stays bounded over the same 64 steps — the blowup is
specifically angle-dependent geometry, not a general tendency of intervals to
explode.

The decisive measurement is what happens when the arithmetic is made exact:

| | growth rate at 45° |
|---|---:|
| naive interval | 1.4478 |
| exact quire | **1.4472** |

Identical to three significant figures. The quire halves the *constant* (a
smaller rounding seed) and leaves the *rate* untouched — exactly what §8.4
predicts. It is the cleanest available demonstration that an exact dot product
buys nothing here.

---

## 9. Choosing the right cure

The practical value of separating the three is that the diagnosis determines the
tool. Reaching for the wrong one wastes effort and, worse, can look like it is
working.

| symptom | likely cause | cure | wrong tool |
|---|---|---|---|
| width grows with reduction length `n` | accumulation | exact accumulator (quire) | rewriting the expression |
| width grows with `cond` at fixed `n` | accumulation | exact accumulator | subdivision |
| a variable appears more than once | dependency | rewrite to single-use; centered form; subdivision | exact accumulator |
| `X - X` is not zero | dependency | rewrite, or affine arithmetic | any accumulator |
| width grows under repeated *linear* maps | wrapping | change the representation: zonotope, parallelepiped + QR, Taylor model | accumulator **and** rewriting |
| enclosure is `[-∞, +∞]` after a division | divisor straddles zero | reformulate to avoid it, or verify a solution instead of computing one | — |

A useful diagnostic: **make the inputs degenerate** (zero width). Any width in
the output is then manufactured by the computation. If it grows with `n`, it is
accumulation. If it persists with exact arithmetic, it is dependency or wrapping.
If it survives rewriting to single-use form, it is wrapping.

---

## 10. What we measured

All three failure modes were measured on posit and cfloat arithmetic in this
repository, and separating them was the point of the exercise:

| failure | example measured | exact accumulator | re-expression |
|---|---|---|---|
| accumulation | `dot`, `gemv`, `gemm`, residual | **fixes it** | — |
| dependency | `nrm2` as `sqrt(dot(x,x))` | no | **fixes it** |
| wrapping | repeated plane rotation | no | no |

A fourth category showed up that is not a property of interval arithmetic at all,
but of an API: a reduction can be **hidden** from the accumulator by being spread
across calls — `k` successive rank-1 updates instead of one `gemm`. That looks
like accumulation and resists the accumulator, until the operation is re-expressed
so the reduction becomes visible. Re-expression alone does not help either; it
takes both ([study §11](interval-blas-study.md)).

Full details: [interval-blas-study.md](interval-blas-study.md). Theory and
history: [verified-computing.md](verified-computing.md).

---

## 11. References

- R. E. Moore, *Interval Analysis*, Prentice-Hall, 1966. **No DOI** — predates DOI assignment. The later SIAM volumes are the practical entry points: *Methods and Applications of Interval Analysis*, 1979, [doi:10.1137/1.9781611970906](https://doi.org/10.1137/1.9781611970906); and R. E. Moore, R. B. Kearfott, M. J. Cloud, *Introduction to Interval Analysis*, 2009, [doi:10.1137/1.9780898717716](https://doi.org/10.1137/1.9780898717716)
- R. Krawczyk, "Newton-Algorithmen zur Bestimmung von Nullstellen mit Fehlerschranken", *Computing*, 1969. [doi:10.1007/BF02234767](https://doi.org/10.1007/BF02234767)
- R. J. Lohner, "Enclosing the solutions of ordinary initial and boundary value problems", in *Computerarithmetic*, 1987 — the QR method for wrapping. **No DOI** located.
- S. M. Rump, "Verification methods: Rigorous results using floating-point arithmetic", *Acta Numerica*, 2010. [doi:10.1017/S096249291000005X](https://doi.org/10.1017/S096249291000005X)
- L. H. de Figueiredo and J. Stolfi, "Affine Arithmetic: Concepts and Applications", *Numerical Algorithms*, 2004. [doi:10.1023/B:NUMA.0000049462.70970.b6](https://doi.org/10.1023/B:NUMA.0000049462.70970.b6)
- K. Makino and M. Berz, "Taylor models and other validated functional inclusion methods", *International Journal of Pure and Applied Mathematics*, 2003. **No DOI** located.
- IEEE Std 1788-2015, *IEEE Standard for Interval Arithmetic*. [doi:10.1109/IEEESTD.2015.7140721](https://doi.org/10.1109/IEEESTD.2015.7140721)
- U. Kulisch, *Computer Arithmetic and Validity: Theory, Implementation, and Applications*, de Gruyter, 2nd ed., 2013. [doi:10.1515/9783110301793](https://doi.org/10.1515/9783110301793)
- N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, SIAM, 2nd ed., 2002. [doi:10.1137/1.9780898718027](https://doi.org/10.1137/1.9780898718027)
