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
- [ ] Extend the sweep to `nrm2` and to `cfloat`/`lns` quire instances (the
  adapter is posit-only today).

## Milestone 2: mixed-precision level-2 kernels

- `gemv` with promoted/exact accumulation per output element (MTL5 `mult`
  already takes an `Accumulator` template parameter).
- `trsv` precision study: forward/backward substitution error at low
  precision — feeds the mp-spice factorization work.

## Milestone 3: mixed-precision level-3 kernels + benchmarking

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
