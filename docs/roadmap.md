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

  Current guidance: posit/wide-cfloat → quire is exact; ≤16-bit → promote to
  `double`; **lns → always promote** (quire is a liability until #1203 lands).

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
- [ ] Length sweep at fixed `cond` to settle whether the realized constant is
  `u`, `sqrt(n)*u` (Higham-Mary probabilistic), or `n*u` (Wilkinson worst case).
  Section 4.2 shows no visible `n` dependence, but that is a property of these
  generators, not a theorem.

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

## Related

- MTL5 mixed-precision `dot`/`mult` accumulator seam
  (`mtl::math::accumulator_traits`).
- mp-iterative quire accumulator adapter (exact dot product bridge).
- mp-spice mixed-precision KLU studies (iterative refinement, quire).
