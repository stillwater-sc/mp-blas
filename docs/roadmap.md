# mp-blas roadmap

## Milestone 0: composition layer bootstrapped (done)

- CMake scaffold replicated from
  [mp-spice](https://github.com/stillwater-sc/mp-spice) /
  [mp-iterative](https://github.com/stillwater-sc/mp-iterative): INTERFACE
  library `sw::mp_blas`, find_package → FetchContent fallback for MTL5 +
  Universal, config-package install, CI matrix (MSVC/GCC/Clang/AppleClang).
- Repo organized by BLAS level: `level1/` (vector-vector), `level2/`
  (matrix-vector), `level3/` (matrix-matrix).
- Smoke tests per level: axpy/dot, gemv, gemm — exact integer results in
  `double`, `float`, `cfloat<16,5>`, and `posit<16,2>`.
- Demo application: `level1/dot_precision` — dot-product accumulation error,
  native vs double accumulator, across six number types.

## Milestone 1: mixed-precision level-1 kernels + accumulator study

- [x] Quire accumulator adapter (`include/mtl/math/quire_accumulator.hpp`)
  specializing MTL5's `accumulator_traits` for a Universal posit quire (mirrors
  mp-iterative's adapter; kept local so mp-blas has no cross-repo header
  dependency). Test: `tests/level1/test_dot_quire.cpp` (quire recovers terms
  native same-precision accumulation swamps).
- [x] Accumulator sweep for `dot`: native vs fma vs promoted `double` vs exact
  quire, as a function of vector length and element precision
  (`applications/level1/dot_accumulator_study`). Write-up:
  [docs/level1-accumulator-study.md](level1-accumulator-study.md).

  **Headline findings.** (1) Same-precision accumulation is unsafe at scale —
  `posit<16,2>` native dot is 25% wrong at n≈10⁶ (swamping). (2) For ≤16-bit
  elements a promoted `double` accumulator is already exact; the quire buys
  nothing. (3) For `posit<32,2>` over long reductions the exact quire stays flat
  at ~1e-16 while a promoted `double` accumulator drifts to ~3e-14 (~200× at
  n≈10⁶) — the quire earns its cost when the element width approaches the
  accumulator's own width.
- [x] Extend the sweep to `nrm2` and to `cfloat`/`lns` element types. The
  adapter now specializes the quire for posit, cfloat, and lns. `nrm2` mirrors
  `dot` (same sum-of-squares reduction); native degrades faster (all-positive
  squares) — `posit<16,2>` native `nrm2` is 78% wrong at n=65536.

  Extending to cfloat/lns turned the sweep into a conformance test for
  Universal's quire and surfaced three upstream requirements (filed):
  - stillwater-sc/universal#1201 — `cfloat.hpp`/`lns.hpp` don't include
    `fdp.hpp`, so `quire_mul` is undeclared out of the box.
  - stillwater-sc/universal#1202 — `quire_traits<cfloat>::radix_point`
    undersized for no-subnormals configs → `cfloat<16,5>` quire floors at ~1e-9
    instead of exact (confirmed: subnormals-on is bit-exact).
  - stillwater-sc/universal#1203 — `quire_mul(lns,lns)` routes through `double`,
    so the lns quire is not exact and is worse than a promoted `double`.

- [x] Close out the cfloat/lns quire gaps as far as mp-blas can. All three
  upstream issues are still **open** as of 2026-07-28 and both defects reproduce
  against current Universal, so the exactness fixes themselves remain blocked
  upstream — but the mp-blas-side work is done:
  - **cfloat: workaround found, measured, and pinned.** Enabling subnormals
    widens `quire_traits<cfloat>::radix_point` 28 → 48 and makes a
    `cfloat<16,5>` quire **bit-exact** where its no-subnormals twin floors at
    `~2^-28 ≈ 3.7e-9`; native/fma/promoted columns are bit-identical, so only
    the quire changes. Root cause diagnosed: `radix_point` is sized from the
    smallest product *scale*, but a product carries `2·(fbits+1)` significand
    bits extending below its leading bit. `dot_accumulator_study` gained `+subn`
    rows; `test_dot_quire` pins it on a non-dyadic construction (the original
    powers-of-two case could not see the defect).
  - **lns: the guidance is structural, not temporary.** An lns product
    `2^(k + m/2^rbits)` is irrational for `m ≠ 0`, so *no* linear fixed-point
    super-accumulator can hold it exactly. universal#1203's achievable goal is
    "round each product at quire precision instead of double's 53 bits", not
    "be exact". **lns → promote, permanently.**

  Current guidance: posit any width, wide cfloat, and **any subnormal-enabled
  cfloat** → quire is exact; narrow cfloat without subnormals → enable
  subnormals or promote; **lns → always promote**.

## Milestone 1b: dot-product characterization (issue #9)

- [x] Background survey + measurement framework:
  [docs/dot-product-characterization.md](dot-product-characterization.md).
  Establishes what Kulisch's super-accumulator was actually designed for
  (verified computing via exact residuals and defect correction — reproducibility
  is the later re-motivation, from ReproBLAS/ExBLAS), what the numerical-analysis
  literature says the input structure is (the dot-product condition number
  `cond = 2|x|'|y|/|x'y|`, from Wilkinson's bound via Higham ch.3 and
  Ogita-Rump-Oishi 2005), and how the regimes cluster by field.
- [x] Feature vector + generators, header-only and free of both MTL5 and
  Universal so any sister repo can instrument its own kernels:
  [`include/sw/mp_blas/dot_characterization.hpp`](../include/sw/mp_blas/dot_characterization.hpp)
  (`cond`, exponent spread `D`, growth `rho`, effective length `n_eff`, sign
  balance `beta`, plus a `Dot2` error-free-transformation reference).
  Study: `applications/level1/dot_characterization`; invariants:
  `tests/level1/test_dot_characterization`. gcc and clang emit bit-identical
  output.

  **Headline findings.** (1) `cond` predicts the error of any inexact
  accumulator to within a small constant across *twenty* decades — native error
  `≈1.4e-9·cond` and promoted-double error `≈8.1e-18·cond` for `posit<32,2>` at
  n=4096, while the quire stays exactly 0. (2) `cond` is **not** sufficient:
  swamping is an independent axis. The all-positive reduction has the best
  possible conditioning (`cond=2`) and is the *worst* case for `posit<16,2>`
  native accumulation (25% error, `n_eff` 1723 of 4096). Report `cond` **and**
  `n_eff`. (3) A promoted `double` accumulator fails at `cond ≳ 1/u_double ≈
  1e16` — in the `kahan` regime it returns a relative error of ~2e3 where the
  quire is exact. Milestone 1's "≤16-bit → promote to `double`" guidance is a
  statement about *benign data*, not about types. (4) Reproducibility is a
  distinct property only exactness buys: `perm(quire)=0` everywhere, while
  `perm(native)` exceeds the error itself (190% order-to-order spread vs 15%
  error on `posit<16,2>/graded`).
- [ ] Collect feature vectors from real workloads (mp-spice MNA rows,
  mp-iterative Krylov orthogonalization/residuals, mp-ode) and report the `cond`
  **distribution and upper quantiles** — the accumulator must be chosen for the
  tail, not the mean.
- [x] Length sweep at fixed `cond` — **settled** (§4.3 of the write-up). Two
  controlled sweeps, because the question is only answerable with one axis
  pinned: (a) `cond` held at 1e12 across a 256× length range → the normalized
  error is *flat*, so there is no `n` dependence on the conditioning axis;
  (b) the `positive` regime, where `cond ≡ 2` for every `n`, isolates the length
  axis → a fixed-precision `double` accumulator grows ~5.6× per 256× `n`,
  **consistent with Higham-Mary's `sqrt(n)*u` and refuting `n*u`** (which would
  predict 256×). Unexpected third result: **posit native accumulation grows
  *superlinearly*** (~1300× per 256× `n`) because a tapered format's local unit
  roundoff coarsens as the running sum grows — no constant-`u` model covers it,
  which is a further argument against accumulating in a tapered format.

## Milestone 2: mixed-precision level-2 kernels

- [x] `gemv` with promoted/exact accumulation per output element (via MTL5
  `mult`'s `Accumulator` parameter): `applications/level2/matvec_accumulator_study`
  sweeps native/promoted/quire on swamping rows; `tests/level2/test_gemv_quire`
  checks the quire is never worse and exact for posit / wide cfloat. Migrated
  from Universal `applications/reproducibility/blas/l2_fmv.cpp`.
- [x] `trsv` precision study: `applications/level2/trsv_precision` +
  `tests/level2/test_trsv_smoke` -- back-substitution error vs the triangular
  factor's conditioning x working precision (feeds the mp-spice factorization
  work). MTL5's trsv has no accumulator seam, so this is an element-precision
  study.

## Milestone 3: mixed-precision level-3 kernels + benchmarking

- [x] `gemm` exact/promoted accumulation:
  `applications/level3/matmul_accumulator_study` + `tests/level3/test_gemm_quire`
  (L3 analogue of the above). Migrated from Universal
  `applications/reproducibility/blas/l3_fmm.cpp`.
- `gemm` in the tensor-core idiom: low-precision multiplicands,
  higher-precision (or exact quire) accumulation, fused convert on writeback.
- Benchmark harness: accuracy AND throughput tables (compare against
  OpenBLAS/MKL reference runs where available); reproducible one-command
  sweeps written up under `docs/`.

## Milestone 4: interval BLAS -- verified enclosures (issue #12)

- [x] **Phase 0** -- containment gate. `include/sw/mp_blas/interval_containment.hpp`
  (`encloses`, overestimation `R`, `in_range`, `absolute_width`) +
  `tests/level1/test_interval_containment.cpp`, with a negative control that
  asserts the gate can actually fail. Surfaced two upstream bugs
  (stillwater-sc/universal#1234 outward rounding, since fixed) and two
  portability lessons: `long double` is not a usable reference (64-bit on ARM64
  and MSVC, so the gate went blind on macOS), and `relative_width` is meaningless
  for enclosures bracketing zero.
- [x] **Phase 1 `dot`** -- interval quire accumulator
  (`include/mtl/math/interval_quire_accumulator.hpp`): a quire per endpoint,
  exact accumulation, ONE outward rounding. Study:
  `applications/level1/interval_dot_study`; hypotheses pinned by
  `tests/level1/test_interval_quire.cpp`. Write-up:
  [docs/interval-blas-study.md](interval-blas-study.md).

  **Headline findings.** (1) The quire enclosure is `4.1e-16`-`4.3e-16` across
  *every* regime and length -- `cond` spanning **19 decades** and `n` spanning
  64-4096 -- certifying 15.4-15.7 digits throughout; it is the same number at
  every length, to every printed digit. (2) Naive interval accumulation degrades
  on both axes and certifies **negative digits** (an enclosure wider than its own
  midpoint) from `cancel 1e-3` onward -- rigorous, correct, worthless. (3) A
  promoted `double` interval accumulator fails at `cond ~ 1/u_double ~ 1e16`, the
  **same threshold issue #9 found for accuracy**, so the accuracy and enclosure
  stories land on one constant. (4) On `kahan 1e-6 / n=4096` naive and promoted
  both prove *nothing* while the quire proves 15.4 digits -- Kulisch's claim,
  measured.
- [ ] **Phase 1 `nrm2`** -- blocked on a structural issue, not on effort:
  `sqrt(dot(x,x))` hands the same interval to both arguments, so for a
  zero-straddling element it computes `X*X` (lower endpoint `a*b < 0`) instead of
  `X^2 = [0, max(a^2,b^2)]`. That is the dependency problem; **no accumulator
  fixes it**, and an exact quire will faithfully accumulate the over-wide corner
  products. Needs a dedicated squaring accumulation. First place in this work
  where the EDP provably cannot help.
- [x] **Phase 2 `ger` + `gemv`** -- separates interval multiplication from interval
  accumulation, so Phase 1's win can be attributed.
  `applications/level2/interval_matvec_study` + `tests/level2/test_interval_l2`.

  **Headline findings.** (1) **The control settles the attribution:** a single
  `ger` (one multiply-add per element, no reduction) has width `2.24e-08` at
  n = 4/16/64/256 -- identical to every digit -- with `R` pinned at 4.50. Interval
  multiplication is size-independent and well-behaved, so all of Phase 1's
  degradation lived in the reduction. (2) `gemv` reproduces Phase 1 row-wise with
  nothing new (quire flat at ~4e-16, `R = 3.0`, naive into negative certified
  digits) -- worth stating precisely *because* it is unsurprising: a level-2 result
  differing from level 1 would have meant a broken seam, not a discovery. (3) **A
  reduction can hide in an operator that has no reduction:** `k` successive rank-1
  updates accumulate across *calls*, so width grows ~linearly in `k` (1152x at
  k=256) and **no per-call accumulator seam reaches it**. The remedy is structural
  -- re-express rank-`k` as one `gemm` -- not arithmetic. Same shape as the `trsv`
  seam limitation: the operator interface forecloses the fix, not its arithmetic.
- [x] **Phase 3 `rot` (wrapping)** -- deliberately hunting for where exact
  accumulation STOPS helping. `include/sw/mp_blas/interval_rot.hpp` +
  `applications/level1/interval_rot_study` + `tests/level1/test_interval_rot`.

  **Headline findings.** (1) **The predicted negative result holds:** an exact
  accumulator changes the CONSTANT, not the RATE. Naive and quire growth rates
  agree to 3-4 significant figures at every angle and element type (1.4478 vs
  1.4472 at 45 deg), while the quire's benefit is a flat ~2x at every k.
  (2) The mechanism is confirmed geometric: measured per-step growth matches
  `|c|+|s|` to within 2% across angles, and at `|c|+|s| = 1` the width stays
  bounded (1.8e-06 after 64 steps) where at 1.414 it reaches 2.0e+02 -- same code,
  same element type, only the angle differs. Not "intervals always blow up" but a
  specific, predictable, angle-dependent effect. (3) Two independent reasons the
  quire cannot help: the reduction is only 2 terms long, AND the loss is
  representational rather than arithmetic. This is the clean counterexample to
  "the exact dot product makes interval arithmetic work" -- it makes REDUCTIONS
  work, which is narrower and more defensible.

  Phases 1-3 have now produced three structurally different obstacles, worth
  keeping apart: **dependency** (`nrm2`, mathematical, no fix), **interface**
  (repeated `ger` / `trsv`, fixable by re-expressing the operation), and
  **wrapping** (`rot`, representational, no fix). Only where the obstacle is
  ACCUMULATION does the quire win -- and there it wins completely.
- [x] **Phase 4 `gemm` + rank-k** -- `applications/level3/interval_matmul_study`
  + `tests/level3/test_interval_gemm`.

  **Headline findings.** (1) `gemm` reproduces Phase 1 element-wise: quire at
  15.4-15.6 certified digits regardless of conditioning, while naive reaches
  **-7.6 digits** on `kahan` (an enclosure ~1e7x wider than the value it
  encloses). No level-3-specific effect, which is the desired outcome. (2) **The
  interface limit is genuinely fixable, and the fix needs BOTH halves.**
  Re-expressing k rank-1 updates as one `gemm` (`A += X*Y^T`) exposes the
  accumulator seam that a sequence of `ger` calls cannot -- but re-expression
  ALONE does nothing: naive `gemm` degrades with k exactly as repeated `ger` does
  (x98 vs x67). It is re-expression *plus* exact accumulation that makes the
  width flat (x0.96 over a 256x increase in k), with the gap reaching 7e+09.

  So of the three obstacles found in Phases 1-4, exactly one is an artifact of
  how the computation was *written* rather than what it *is*, and that one
  dissolves once the reduction is made visible to the accumulator:

  | limit | example | exact accumulator helps? |
  |---|---|---|
  | dependency | `nrm2` = `sqrt(dot(x,x))` | **No** -- mathematical |
  | interface | repeated `ger`, `trsv` | **Yes, after re-expressing** |
  | wrapping | repeated `rot` | **No** -- representational |
- [x] **Phase 5 verified solve** -- `applications/level2/verified_solve_study` +
  `tests/level2/test_verified_solve`. What the whole line of work was built toward.

  **Headline findings.** (1) **The exact residual decouples accuracy from
  conditioning.** With a working-precision residual, defect correction stagnates
  at ~`cond*u` -- and at n=4 gets *worse* each step (6.0e-06 -> 1.8e-05), because
  it corrects with noise. With an exact residual it reaches **`u` exactly**
  (3.7e-09 for posit<32,2>, 5.9e-08 for cfloat<32,8>). (2) **It holds as
  conditioning grows:** from n=4 to n=6 `cond` rises **1000x** (2.8e4 -> 2.9e7)
  and the exact-residual accuracy is *unchanged* at ~`u`, while the
  working-precision residual degrades 1.8e-05 -> 2.1e-02. The classical Wilkinson
  refinement result and Kulisch's argument, measured on posit and cfloat.
  (3) **The claim is bounded, and not by the residual:** past `cond*u ~ 1` (n=8,
  `cond*u = 70`) the exact residual reaches only 9.2e-02, because the
  working-precision *factorization* carries no correct digits. The exact residual
  removes the `cond` dependence OF THE RESIDUAL, not of the factorization.

  Design finding worth keeping: `trsv` is the wrong vehicle for this experiment.
  Triangular systems are solved far more accurately than their condition number
  suggests (Higham ASNA ch. 8), so back-substitution lands within a few ulp,
  defect correction has nothing to correct, and the two residuals are
  indistinguishable -- a Phase 5 built on `trsv` would have passed vacuously and
  concluded the opposite. Hilbert is used because its forward error genuinely
  tracks `cond*u`.

  **What the exact dot product is for**, across all five phases: it makes
  REDUCTIONS exact. Everything it buys follows from that one property --
  15.4 certified digits where naive certifies none; rank-k restored once the
  reduction is re-expressed; accuracy decoupled from conditioning in defect
  correction -- and everything it cannot do (dependency, wrapping, a failed
  factorization) falls outside it.

## Related

- MTL5 mixed-precision `dot`/`mult` accumulator seam
  (`mtl::math::accumulator_traits`).
- mp-iterative quire accumulator adapter (exact dot product bridge).
- mp-spice mixed-precision KLU studies (iterative refinement, quire).
