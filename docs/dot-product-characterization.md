# Dot-product characterization

**Question** (issue [#9](https://github.com/stillwater-sc/mp-blas/issues/9)).
Before we hypothesize data collection and statistical metrics for the quire
studies, what is already known? What did Kulisch's super-accumulator research
base its design on — reproducibility only, or numerical analysis too? What
information do the input vectors carry in, what survives into the result, and
has anyone studied the properties of the dot products that actually arise in
engineering and scientific linear algebra?

This note answers those three questions from the literature, then converts the
answer into a measurement framework: a feature vector per dot-product instance,
implemented in
[`include/sw/mp_blas/dot_characterization.hpp`](../include/sw/mp_blas/dot_characterization.hpp)
and exercised by
[`applications/level1/dot_characterization`](../applications/level1/dot_characterization).

> **Citation note.** References below are given as author / title / venue / year.
> Titles and years are stated as accurately as I can; volume and page numbers are
> deliberately omitted where I am not certain of them. Verify before quoting this
> document in a paper.

---

## 1. What Kulisch's design was actually based on

**Not reproducibility.** Reproducibility is the *modern* re-motivation, adopted
roughly two decades after the fact. Kulisch's super-accumulator ("the complete
register", later "the exact dot product", EDP) came out of a program in
**verified computing**: producing results with rigorous, machine-proved error
bounds.

The argument, developed by Kulisch and Miranker (*Computer Arithmetic in Theory
and Practice*, Academic Press, 1981; "The Arithmetic of the Digital Computer: A
New Approach", *SIAM Review*, 1986) and restated in Kulisch's *Computer
Arithmetic and Validity: Theory, Implementation, and Applications* (de Gruyter,
2nd ed. 2013), runs:

1. A computer's arithmetic should implement each operation with a **single
   rounding** of the exact mathematical result. IEEE 754 does this for
   `+ − × ÷ √`.
2. The dot product is the operation from which essentially all of linear algebra
   is built — matrix product, residual, norm, projection, Gram matrix. It is
   *not* one of the five, and a chain of individually-correctly-rounded
   operations does **not** compose into a correctly-rounded dot product.
3. Therefore make the exact scalar product a **fifth basic arithmetic
   operation**, implemented once in hardware with a fixed-point accumulator wide
   enough to hold any sum of products of the format without loss. Cost is
   bounded and modest, because the exponent range of a fixed format is bounded.
4. The **payoff** is not aesthetic. With an EDP you can compute a residual
   `r = b − Ax` to full accuracy in the working precision. That makes **defect
   (residual) correction** effective, and defect correction plus interval
   arithmetic yields **verified enclosures** — results that come with a proof
   that the true answer lies inside a stated interval, produced automatically.

So the design basis was a numerical-analysis result: **the exact dot product is
the enabling primitive for automatic result verification.** Without it, the
residual of a nearly-consistent system is computed at a relative accuracy of
roughly `cond·u`, which for an ill-conditioned system is no accuracy at all, and
the verification loop does not converge. Rump's survey "Verification methods:
Rigorous results using floating-point arithmetic" (*Acta Numerica*, 2010) is the
standard modern account of that machinery.

**Where reproducibility entered.** Once large parallel machines made summation
order nondeterministic (dynamic scheduling, variable thread counts, atomics),
bit-identical results across runs became a practical demand — debugging,
regression testing, regulatory sign-off. An exact accumulator delivers it for
free, since an exact sum is order-independent by construction. That motivated
the reproducible-BLAS line: Demmel, Ahrens & Nguyen's **ReproBLAS** (indexed
summation with pre-rounding), and Iakymchuk, Collange, Defour & Graillat's
**ExBLAS** (a long-accumulator BLAS descended directly from Kulisch). Gustafson
& Yonemoto's "Beating Floating Point at its Own Game: Posit Arithmetic"
(*Supercomputing Frontiers and Innovations*, 2017) reintroduced the same
structure as the posit **quire**, this time as a mandatory part of the number
system rather than an optional library.

**Consequence for mp-blas.** There are two independent value propositions and
they should be measured separately, because a design can want one without the
other:

| claim | what it buys | how to measure |
|---|---|---|
| **accuracy** | error independent of conditioning | relative error vs an independent high-accuracy reference |
| **reproducibility** | result independent of summation order | spread of the result over permuted orders |

The harness reports both. They are *not* the same column: a promoted `double`
accumulator can be accurate on a given problem while remaining
order-dependent, and that is exactly what the data below shows.

---

## 2. What information goes in, and what survives

This is the part of the issue with a crisp textbook answer.

Wilkinson's classical bound for an inner product accumulated in a format with
unit roundoff `u` (see Higham, *Accuracy and Stability of Numerical Algorithms*,
2nd ed., SIAM 2002, ch. 3; originally Wilkinson, *Rounding Errors in Algebraic
Processes*, 1963) is

```
|fl(x'y) − x'y|  ≤  γ_n · |x|'|y|,        γ_n = n·u / (1 − n·u)
```

The error is bounded not by the *result* but by `|x|'|y|`, the sum of the
magnitudes of the products. Dividing through:

```
                    |fl(x'y) − x'y|         γ_n
relative error  =   ───────────────   ≤    ───── · cond(x,y)
                        |x'y|                2

                                  2 · |x|'|y|
                    cond(x,y)  =  ───────────
                                     |x'y|
```

**`cond(x,y)` is the answer to the issue's question.** `|x|'|y|` is the
information the input vectors carry in; `|x'y|` is the information that survives
into the result; their ratio is the information destroyed, and `log10(cond)` is
that loss counted in decimal digits. The name and the factor of 2 follow Ogita,
Rump & Oishi, "Accurate Sum and Dot Product" (*SIAM J. Sci. Comput.*, 2005),
which is also where the compensated algorithms `Sum2`/`Dot2` come from — a
`Dot2` result is as accurate as one computed in doubled precision, which is what
the harness uses as its reference. `cond ≥ 2` always, with equality exactly when
every product has the same sign.

### Why one scalar is not enough

`cond` is a statement about **cancellation**. It is invariant under permutation
of the terms, so by construction it cannot see anything about summation order —
and two order-dependent effects matter:

- **Swamping.** When the running sum is much larger than an incoming term, the
  term is absorbed and lost. This has nothing to do with cancellation: the
  all-positive reduction `x'x` has `cond = 2`, the best possible, and still
  loses 25% of its value in `posit<16,2>` at `n = 4096` (measured below).
  What the `γ_n` factor is standing in for.
- **Growth.** A reduction can climb to a partial sum far above its final value
  and come back down. Intermediate rounding is then committed against the large
  intermediate, not the small answer.

So the feature vector needs a cancellation axis, a dynamic-range axis, and an
order axis. The six metrics:

| metric | definition | reads as |
|---|---|---|
| `cond` | `2·\|x\|'\|y\| / \|x'y\|` | digits destroyed by cancellation = `log10(cond)` |
| `D` | `log2(max\|p_i\| / min\|p_i\|)` over products `p_i = x_i·y_i` | exponent spread; swamping *potential*, in bits |
| `rho` | `max_k \|Σ_{i≤k} p_i\| / \|x'y\|` | growth factor; how far the partial sum overshoots |
| `n_eff` | `#{ i : \|p_i\| ≥ u · max_k\|Σ_{i≤k} p_i\| }` | terms still able to move the accumulator at its peak; `n_eff ≪ n` *is* swamping |
| `beta` | `\|#pos − #neg\| / n` | sign balance; `0` = fully cancelling, `1` = one-signed |
| `perm` | `(max − min)/\|x'y\|` over random summation orders | **reproducibility**, measured directly |

`D` is the *potential* for swamping and `n_eff` is the *realized* swamping —
they differ because whether a small term is lost depends on how large the sum
has already grown, not merely on the spread of the inputs.

### Sharpness of the bound

`γ_n ≈ n·u` is a worst case that assumes every rounding error pushes the same
way. Higham & Mary ("A New Approach to Probabilistic Rounding Error Analysis",
*SIAM J. Sci. Comput.*, 2019) showed that under a model where rounding errors
behave like independent mean-zero random variables, the realistic constant is
`√n·u` rather than `n·u`. Blanchard, Higham & Mary ("A Class of Fast and
Accurate Summation Algorithms", *SIAM J. Sci. Comput.*, 2020) analyze blocked
and pairwise summation, where the growth is `log n` instead of `n`. The
measurements in §4 are consistent with the stronger statement that on these
instances the realized error tracks `cond·u` with **no visible `n` dependence at
all** — a caveat worth carrying, since `n`-free behavior is a property of the
data, not a theorem.

The exception is one-signed summation: when every error has the same sign there
is nothing to cancel, and the pessimistic `n·u` growth is real. That is why the
`positive` row below is the worst case for a narrow accumulator despite having
the best possible `cond`.

---

## 3. What is known about dot products in the wild

Directly answering "has anyone studied their properties": there is no single
cross-disciplinary survey of dot-product conditioning that I can point to. What
exists is domain-specific, and it is consistent enough to tabulate. Each row
below names the structural regime and the specific literature that established
it.

| domain | dot products that arise | regime | evidence |
|---|---|---|---|
| **Computational geometry** | orientation / in-circle determinants | `cond → ∞` by design; only the **sign** is wanted, and it is decided entirely by cancelled digits | Shewchuk, "Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates" (*Discrete & Comput. Geometry*, 1997) — the canonical case where exact accumulation is not an optimization but a correctness requirement |
| **Circuit simulation** (SPICE / MNA) | short sparse rows; conductances spanning 1e−12…1e12 | huge `D`, strong cancellation at KCL nodes (currents must sum to zero) | the motivating application for the sister repo [mp-spice](https://github.com/stillwater-sc/mp-spice) |
| **Krylov solvers** | orthogonalization and residual inner products | loss of orthogonality grows like `cond·u`; drives the need for re-orthogonalization | Paige's analysis of Lanczos; Greenbaum on GMRES/CG in finite precision |
| **Iterative refinement / mixed precision** | the residual `b − Ax` | `cond ≈ cond(A)` by construction: `Ax ≈ b` means near-total cancellation | Higham & Mary, "Mixed precision algorithms in numerical linear algebra" (*Acta Numerica*, 2022); Carson & Higham on three-precision refinement. **This is Kulisch's original target.** |
| **N-body / molecular dynamics** | force sums, Ewald summation | net force ≈ 0 while individual terms are large — cancellation is structural, from Newton's third law | long-standing folklore in MD energy-conservation studies |
| **Structural FEM** | stiffness assembly, energy inner products | mostly one-signed, long, moderate `D`; `cond(A)` grows like `h⁻²` under refinement | standard FEM conditioning results |
| **Machine learning** | GEMM, attention scores | benign `cond`, small `D`, but `n` large and precision very low (fp16/bf16/fp8) — the `√n·u` term dominates, not `cond` | Higham & Mary probabilistic analysis; industry practice of fp32 accumulation for fp16 multiplicands |

**The pattern.** The regimes cluster into two groups that want different things:

- **Cancellation-dominated** (geometry, circuits, refinement, N-body): `cond` is
  large and can be unbounded. A wider accumulator buys digits linearly and
  eventually runs out. Only an exact accumulator is *unconditionally* correct.
- **Length-dominated** (ML, FEM): `cond` is small, `n` is large, element
  precision is low. Modest promotion (fp16 → fp32) fixes it, and the quire is
  not worth its cost.

This is a sharper statement of Milestone 1's finding. That study concluded "the
quire earns its cost when the element width approaches the accumulator width" —
a statement about *types*. The correct criterion is about the *data*: the quire
earns its cost when `cond` approaches `1/u` of the widest accumulator you would
otherwise use. Element width is a proxy that happens to correlate on benchmark
distributions.

---

## 4. Measurements

Reproduce with:

```bash
./build/applications/level1/dot_characterization/dot_characterization [n]
```

gcc 13 and clang produce **bit-identical output** (verified by `diff`), which is
itself a small reproducibility result: the harness uses no `<random>` and no
parallel reduction.

Reference values come from `Dot2`, not from a quire — the yardstick must be
independent of the thing being measured. `tests/level1/test_dot_characterization`
pins this down by checking `Dot2` is *exactly* right on a construction whose
answer is known in closed form.

### 4.1 Regimes at `n = 4096`

Five generated regimes isolate one structural axis each; `cancel`/`kahan` place
all positive terms first and all their exact negations second, so the running
sum climbs to `|x|'|y|/2` before collapsing onto a small residual — the
configuration the EDP was designed to survive.

**element type `posit<16,2>`** (`u = 2.4e-04`)

| regime | cond | D | rho | n_eff | beta | native | promoted | quire | perm(nat) | perm(quire) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform     | 1.3e+02 | 17 | 1.3e+00 | 3979 | 0.03 | 9.0e-03 | 0 | 0 | 5.1e-02 | **0** |
| positive    | 2.0e+00 | 24 | 1.0e+00 | 1723 | 1.00 | **2.5e-01** | 0 | 0 | 0 | **0** |
| graded      | 3.4e+01 | 86 | 1.1e+00 | **457** | 0.03 | 1.5e-01 | 0 | 1.5e-16 | 1.9e+00 | **0** |
| cancel 1e-3 | 2.1e+06 | 13 | 2.6e+04 | 4030 | 0.00 | 7.0e+01 | 0 | 0 | 6.5e+02 | **0** |
| cancel 1e-6 | 2.1e+09 | 20 | 2.6e+07 | 4030 | 0.00 | 7.0e+04 | 0 | 0 | 6.5e+05 | **0** |
| cancel 1e-9 | 2.1e+12 | 30 | 2.6e+10 | 4030 | 0.00 | 7.1e+07 | 0 | 0 | 6.6e+08 | **0** |
| kahan 1e-6  | 2.8e+19 | 85 | 8.1e+17 | 480  | 0.00 | 1.4e+17 | **1.8e+03** | **0** | 1.4e+18 | **0** |

**element type `posit<32,2>`** (`u = 3.7e-09`)

| regime | cond | D | rho | n_eff | beta | native | promoted | quire | perm(nat) | perm(quire) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform     | 1.3e+02 | 17 | 1.3e+00 | 4096 | 0.03 | 3.5e-08 | 2.5e-15 | 0 | 1.1e-06 | **0** |
| positive    | 2.0e+00 | 24 | 1.0e+00 | 4084 | 1.00 | 1.8e-07 | 4.1e-15 | 0 | 6.9e-07 | **0** |
| graded      | 3.8e+01 | 86 | 1.1e+00 | 1288 | 0.03 | 4.3e-06 | 0 | 0 | 1.5e-04 | **0** |
| cancel 1e-3 | 2.1e+06 | 13 | 2.6e+04 | 4096 | 0.00 | 2.9e-03 | 1.7e-11 | 0 | 5.4e-03 | **0** |
| cancel 1e-6 | 2.1e+09 | 20 | 2.6e+07 | 4096 | 0.00 | 2.9e+00 | 1.7e-08 | 0 | 5.7e+00 | **0** |
| cancel 1e-9 | 2.1e+12 | 30 | 2.6e+10 | 4094 | 0.00 | 2.9e+03 | 1.7e-05 | 0 | 5.7e+03 | **0** |
| kahan 1e-6  | 2.7e+19 | 85 | 7.9e+17 | 1312 | 0.00 | 8.4e+12 | **2.4e+03** | **0** | 9.1e+13 | **0** |

(`cfloat<32,8>` behaves like `posit<32,2>` shifted by its wider `u = 6.0e-08`;
see the program output.)

### 4.2 Conditioning crossover, `posit<32,2>`, `n = 4096`

Sweeping `cond` over twenty decades at fixed length and element type:

| cond | native | promoted `double` | quire |
|---:|---:|---:|---:|
| 2.1e+03 | 2.9e-06 | 1.6e-14 | **0** |
| 2.1e+07 | 2.9e-02 | 1.7e-10 | **0** |
| 2.1e+11 | 2.9e+02 | 1.7e-06 | **0** |
| 2.1e+15 | 2.9e+06 | 1.7e-02 | **0** |
| 2.1e+19 | 2.9e+10 | 1.7e+02 | **0** |
| 2.1e+23 | 2.9e+14 | 1.7e+06 | **0** |

Both inexact strategies are **exactly linear in `cond`** across twenty decades:
native error `≈ 1.4e−9 · cond`, promoted error `≈ 8.1e−18 · cond`. The quire is
flat at zero throughout. This is the `γ_n·cond/2` bound behaving as an equality
in its `cond` dependence — and note the realized coefficients are *below* `u`
(`3.7e−9` and `1.1e−16` respectively), i.e. the `n = 4096` factor in `γ_n` does
not materialize on this data.

### 4.3 Length sweep: which constant does the error actually carry?

§4.2 hinted that the `n` in `γ_n` does not materialize. This settles it with two
*controlled* sweeps, because "does the error grow with `n`?" is only answerable
if `cond` is held fixed while `n` moves.

**(a) Conditioning-dominated.** `generate_at_cond` pins `cond ≈ 1e12` across a
256× length range. Normalizing by `u·cond` removes the conditioning axis, so any
`n` dependence left in the ratio is real.

| n | cond | native | nat/(u·cond) | promoted | prm/(u·cond) | √(n/n₀) |
|---:|---:|---:|---:|---:|---:|---:|
| 256   | 9.8e+11 | 4.4e+02 | 0.12 | 2.2e-05 | 0.21  | 1.0 |
| 1024  | 1.0e+12 | 2.4e+02 | 0.064| 1.9e-05 | 0.17  | 2.0 |
| 4096  | 1.0e+12 | 1.4e+03 | 0.37 | 8.0e-06 | 0.072 | 4.0 |
| 16384 | 1.0e+12 | 2.6e+02 | 0.071| 1.7e-06 | 0.015 | 8.0 |
| 65536 | 1.0e+12 | 2.0e+02 | 0.054| 1.3e-06 | 0.012 | 16.0 |

The normalized native column is **flat** — it scatters over 0.054…0.37 with no
trend, against a `√n` yardstick running 1→16. The promoted column *decreases*.
So in this regime the error is `c·u·cond` with **no `n` dependence at all**.

**Caveat that matters:** this construction cancels pairwise *exactly*, which
largely suppresses the accumulate-many-roundings mechanism `γ_n` is about. It
isolates the `cond` axis by design — it is not evidence about the length axis.

**(b) Length-dominated.** The `positive` regime has `cond ≡ 2` for every `n`, so
normalizing divides by a constant and **any** growth in the ratio is pure `n`
dependence. This is the controlled experiment for the other axis.

`posit<32,2>` (`u = 3.7e-09`):

| n | native | nat/(u·cond) | promoted | prm/(u·cond) | √(n/n₀) |
|---:|---:|---:|---:|---:|---:|
| 256   | 3.4e-09 | 0.46 | 5.0e-16 | 2.3  | 1.0 |
| 1024  | 3.0e-07 | 40   | 8.4e-16 | 3.8  | 2.0 |
| 4096  | 1.8e-07 | 24   | 4.1e-15 | 19   | 4.0 |
| 16384 | 3.6e-07 | 48   | 4.8e-15 | 22   | 8.0 |
| 65536 | 4.5e-06 | 610  | 3.0e-15 | 13   | 16.0 |

Two different behaviors in one table:

- **The `double` accumulator grows sublinearly.** `prm/(u·cond)` goes 2.3 → 13,
  a factor of ~5.6 over a 256× increase in `n`. Wilkinson's `n·u` would predict
  256×; Higham & Mary's `√n·u` predicts 16×. The measurement sits below the
  probabilistic prediction and far below the deterministic one — **consistent
  with `√n·u`, and a clear refutation of `n·u` as a realistic estimate** for a
  fixed-precision accumulator on this data.
- **The `posit` native accumulator grows *superlinearly*.** `nat/(u·cond)` goes
  0.46 → 610, ~1300× over the same 256× length increase. This is not a violated
  bound — Wilkinson's bound at `n = 65536` is `n·u·cond/2 = 4.9e-04` and the
  measured `4.5e-06` sits comfortably under it. It is that **`u` is not
  constant for a tapered format.** As the running sum climbs to `≈ n/3`, it
  moves away from 1.0, posit's regime field lengthens and its fraction field
  shrinks, so the *local* unit roundoff coarsens as the reduction proceeds. The
  classical analysis is written for fixed-precision formats and simply does not
  model this.

`posit<16,2>` on the same regime saturates rather than diverging — `nat/(u·cond)`
runs 0.63, 12, 520, 1700, 2000 — because the relative error itself reaches
`9.5e-01`, i.e. essentially 100%. A ratio cannot keep growing once the answer is
already entirely wrong.

**Answer to the open question.** Both, on different axes:

| axis | held fixed | realized growth | model |
|---|---|---|---|
| conditioning | `n` varies, `cond` pinned | none | `error = c·u·cond` |
| length, fixed-precision accumulator | `cond ≡ 2` | ~5.6× per 256× `n` | `√n·u` (Higham–Mary) ✓ |
| length, tapered-precision element | `cond ≡ 2` | ~1300× per 256× `n` | none of them — `u` varies with magnitude |

The third row is the one worth carrying forward: **for posit, "unit roundoff" is
not a single number**, so error models parameterized by a scalar `u` understate
long native reductions. It is a further argument for not accumulating in a
tapered format at all.

---

## 5. Findings

1. **`cond` predicts accumulation error to within a small constant, over twenty
   decades.** The relative error of any inexact accumulator is `≈ c·u·cond`,
   with `c` of order one. It is the right single statistic — and it is cheap: a
   `Dot2` pass over `|x|,|y|` costs a handful of flops per element.

2. **`cond` alone is not sufficient — swamping is a second, independent axis.**
   `positive` has the best possible conditioning (`cond = 2`) and is the *worst*
   case for `posit<16,2>` native accumulation (25% error), because `n_eff` falls
   to 1723 of 4096. `graded` is likewise well-conditioned (`cond = 34`) and
   loses 15%, with `n_eff = 457`. Any characterization reporting only `cond`
   would call both of these benign. Report `cond` **and** `n_eff`.

3. **A promoted `double` accumulator is not a substitute for an exact one — it
   is a strictly weaker device that fails at a predictable threshold.** In the
   `kahan` regime (`cond ≈ 2.8e19`) promoted `double` returns a relative error
   of `≈ 2e3` — a result with no correct digits, indeed the wrong order of
   magnitude — while the quire is exact. The threshold `cond ≈ 1/u_double ≈ 1e16`
   is where it happens, and 4.2 confirms it: promoted `double` passes `100%`
   relative error at `cond ≈ 1e17`. This is a **counterexample to the reading of
   Milestone 1's guidance as "≤16-bit elements → just promote to `double`"**:
   that rule is a statement about benign data, and it does not survive
   conditioning above `1/u_double`.

4. **Reproducibility is a separate property, and only the exact accumulator has
   it.** `perm(quire) = 0` in every row of every table — over 16 random
   summation orders, across 20 decades of conditioning, for every element type.
   `perm(native)` is nonzero everywhere it can be, and *larger than the error
   itself*: in `posit<16,2>/graded` the error is 15% but the order-to-order
   spread is 190%. Two runs of the same code on the same data can differ by more
   than either differs from the truth. This is Kulisch's guarantee, measured —
   and it is the one claim that no amount of extra accumulator width can buy,
   because it is a statement about exactness, not precision.

5. **The decision rule should be data-driven, not type-driven.** Milestone 1
   framed the quire's value in terms of element width vs accumulator width. The
   measurements say the operative variable is `cond`:

   | measured | use |
   |---|---|
   | `cond ≲ ε_tol/(u_elem)` and `n_eff ≈ n` | **native** — no promotion needed |
   | `cond ≲ ε_tol/u_double` | **promoted `double`** — cheap and sufficient |
   | `cond ≳ 1/u_double ≈ 1e16`, or **any** reproducibility requirement | **exact quire** — nothing else works |

   Element width enters only through `u_elem`, and is a proxy that happens to
   correlate on benchmark distributions.

---

## 6. Data-collection plan

What the issue asks to be designed. The framework above turns "characterize dot
products" into a concrete program:

1. **Instrument, don't guess.** `characterize()` is header-only and depends on
   neither MTL5 nor Universal — it takes two `std::vector<double>`. Any kernel
   in any sister repo can emit a feature vector for the dot products it actually
   performs.
2. **Collect from real workloads, not synthetic ones.** The regimes here are
   deliberately synthetic so that each axis is isolated; they establish the
   *response surface*, not the *operating point*. The operating point has to
   come from mp-spice (MNA rows), mp-iterative (Krylov orthogonalization and
   residuals), and mp-ode. Each should log `(n, cond, D, rho, n_eff, beta)` per
   dot product over a representative solve.
3. **The statistic to report is a distribution, not a mean.** A solve performs
   millions of dot products; the accumulator must be chosen for the tail. Report
   the **`cond` histogram and its upper quantiles** (p99, max). One
   catastrophically-conditioned residual in a refinement loop is what decides
   whether the loop converges — an average `cond` says nothing about it.
4. **Validate the predictor before trusting it.** For each collected instance,
   check the realized error against `c·u·cond`. Where the prediction fails,
   there is structure the feature vector does not capture, and that is the next
   metric.
5. **Score reproducibility separately** via `perm`. It does not follow from
   accuracy and must not be inferred from it.

### Open questions this framework does not answer

- **Cost.** Everything here is accuracy and reproducibility. The quire's
  throughput and area cost are not measured anywhere in mp-blas yet
  (Milestone 3's benchmark harness). A decision rule needs both halves.
- **`n_eff` depends on summation order** through the peak partial sum, so it is
  a property of the (instance, algorithm) pair rather than the instance alone.
  Pairwise or blocked summation would change it; the harness only measures
  sequential order.
- **Sparse structure** is not represented. The regimes are dense; MNA rows are
  short and sparse, and short reductions may be dominated by different effects
  than the `n = 4096` cases here.
- ~~The `n`-independence observed in 4.2 is an empirical property of these
  generators, not a theorem.~~ **Settled in §4.3**: `n`-independence holds on the
  conditioning axis; on the length axis a fixed-precision accumulator grows
  consistently with `√n·u`, and a tapered-precision one grows faster than any
  constant-`u` model predicts.
- **Tapered precision has no error model here.** §4.3(b) shows posit's native
  accumulation growing superlinearly in `n` because its local `u` coarsens as
  the running sum grows. Quantifying that — an effective `u(magnitude)` for
  posit, and a corresponding bound — is open, and would matter to anyone using
  posit as an accumulator rather than as storage.

---

## References

- U. Kulisch and W. L. Miranker, *Computer Arithmetic in Theory and Practice*, Academic Press, 1981.
- U. Kulisch and W. L. Miranker, "The Arithmetic of the Digital Computer: A New Approach", *SIAM Review*, 1986.
- U. Kulisch, *Computer Arithmetic and Validity: Theory, Implementation, and Applications*, de Gruyter, 2nd ed., 2013.
- J. H. Wilkinson, *Rounding Errors in Algebraic Processes*, 1963.
- N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002 (ch. 3 dot products, ch. 4 summation).
- T. Ogita, S. M. Rump, S. Oishi, "Accurate Sum and Dot Product", *SIAM J. Sci. Comput.*, 2005.
- S. M. Rump, "Ultimately Fast Accurate Summation", *SIAM J. Sci. Comput.*, 2009.
- J. Demmel and Y. Hida, "Accurate and Efficient Floating Point Summation", *SIAM J. Sci. Comput.*, 2003.
- J. Demmel, P. Ahrens, H. D. Nguyen, "Efficient Reproducible Floating Point Summation and BLAS" (ReproBLAS), 2016.
- R. Iakymchuk, S. Collange, D. Defour, S. Graillat, "ExBLAS: Reproducible and Accurate BLAS Library", 2015.
- N. J. Higham and T. Mary, "A New Approach to Probabilistic Rounding Error Analysis", *SIAM J. Sci. Comput.*, 2019.
- P. Blanchard, N. J. Higham, T. Mary, "A Class of Fast and Accurate Summation Algorithms", *SIAM J. Sci. Comput.*, 2020.
- N. J. Higham and T. Mary, "Mixed precision algorithms in numerical linear algebra", *Acta Numerica*, 2022.
- S. M. Rump, "Verification methods: Rigorous results using floating-point arithmetic", *Acta Numerica*, 2010.
- J. R. Shewchuk, "Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates", *Discrete & Computational Geometry*, 1997.
- J. Gustafson and I. Yonemoto, "Beating Floating Point at its Own Game: Posit Arithmetic", *Supercomputing Frontiers and Innovations*, 2017.
- IEEE Std 1788-2015, *IEEE Standard for Interval Arithmetic*.

## Related

- [docs/level1-accumulator-study.md](level1-accumulator-study.md) — the error
  measurements this note supplies the input-side characterization for.
- [include/sw/mp_blas/dot_characterization.hpp](../include/sw/mp_blas/dot_characterization.hpp)
  — metrics, `Dot2` reference, and regime generators.
- [applications/level1/dot_characterization](../applications/level1/dot_characterization)
  — the study; [tests/level1/test_dot_characterization.cpp](../tests/level1/test_dot_characterization.cpp)
  — the invariants.
