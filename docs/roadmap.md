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

- Accumulator sweep for `dot` / `nrm2`: native vs promoted (`float`/`double`)
  vs exact (Universal `quire`, single rounding), as a function of vector
  length and element precision. Reuse MTL5's `accumulator_traits` seam; the
  quire adapter mirrors mp-iterative's `quire_accumulator.hpp`.
- Error-vs-n growth curves; document where each 16-bit element type needs a
  wider accumulator.

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
