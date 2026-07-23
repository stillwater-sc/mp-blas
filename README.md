# mp-blas

[![CMake](https://github.com/stillwater-sc/mp-blas/actions/workflows/cmake.yml/badge.svg)](https://github.com/stillwater-sc/mp-blas/actions/workflows/cmake.yml)

**Mixed-precision Basic Linear Algebra Subroutines.** mp-blas composes two
header-only libraries — [MTL5](https://github.com/stillwater-sc/mtl5) for
linear algebra and [Universal](https://github.com/stillwater-sc/universal) for
parameterized number systems — to develop and benchmark mixed-precision BLAS
kernels under custom arithmetic (half precision, bfloat16, posits, quire
super-accumulators, ...).

MTL5 deliberately has **no dependency on Universal**: it is the general
linear-algebra layer. mp-blas is the integration layer where MTL5's kernels
meet Universal's number types.

## Build

```bash
# Dependencies (MTL5 + Universal) are pulled automatically via FetchContent.
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Run the smoke tests
ctest --test-dir build --output-on-failure

# Run the dot-product accumulation demo (optional arg: vector length)
./build/applications/level1/dot_precision/dot_precision
```

Using local checkouts instead of fetching from GitHub:

```bash
cmake -B build \
  -DFETCHCONTENT_SOURCE_DIR_MTL5=../mtl5 \
  -DFETCHCONTENT_SOURCE_DIR_UNIVERSAL=../universal
```

## Layout

The repo is organized by BLAS level: **level 1** vector-vector operations
(axpy, dot, nrm2, scal), **level 2** matrix-vector operations (gemv, trsv,
ger), and **level 3** matrix-matrix operations (gemm, trsm, syrk).

```
applications/
  level1/dot_precision/   # dot-product accumulation error across precisions
include/sw/mp_blas/       # shared composition-layer headers
tests/
  level1/                 # axpy + dot smoke test across number types
  level2/                 # gemv smoke test
  level3/                 # gemm smoke test
docs/roadmap.md           # milestones and known integration work
```

## License

MIT — see [LICENSE](LICENSE).
