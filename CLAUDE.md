# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Project Overview

mp-blas is the **integration layer** for mixed-precision BLAS routines.
It composes two header-only sister libraries:

- [MTL5](https://github.com/stillwater-sc/mtl5) — C++20 linear algebra
  (dense/sparse kernels: `dot`, `axpy`, `mult` with pluggable accumulators).
- [Universal](https://github.com/stillwater-sc/universal) — parameterized number
  systems (`cfloat`, `posit`, `quire`, ...).

**Architectural rule:** MTL5 is the general linear-algebra layer and MUST NOT
depend on Universal. All MTL5 + Universal coupling lives here in mp-blas.

## Build Commands

```bash
# Dependencies are pulled automatically via FetchContent.
cmake -B build -DCMAKE_BUILD_TYPE=Release -Wno-dev
cmake --build build -j
ctest --test-dir build --output-on-failure

# Use local sister checkouts instead of fetching from GitHub:
cmake -B build -DFETCHCONTENT_SOURCE_DIR_MTL5=../mtl5 \
               -DFETCHCONTENT_SOURCE_DIR_UNIVERSAL=../universal
```

## Architecture

- Header-only composition under `include/sw/mp_blas/`. Namespace: `sw::mp_blas`.
- CMake: INTERFACE library `sw::mp_blas` linking MTL5 + Universal. Options:
  `MPBLAS_BUILD_APPLICATIONS`, `MPBLAS_BUILD_TESTS`.
- `applications/` and `tests/` are organized by BLAS level, in this order:
  `level1/` (vector-vector: axpy, dot, nrm2, scal), `level2/` (matrix-vector:
  gemv, trsv, ger), `level3/` (matrix-matrix: gemm, trsm, syrk).
- `applications/` — demonstration/benchmark programs (each its own CMakeLists).
- `tests/` — lightweight self-checking executables (no external framework);
  register with `mpblas_add_test`.
- `docs/roadmap.md` — milestones and known integration work.

## Conventions

- C++20, header-only. Match the sister repos (mtl5, mp-spice, mp-iterative,
  mp-ode) for style and CMake structure.
- Conventional Commits. Feature branches + PRs to `main`; CI must pass.
- Never commit build artifacts or downloaded data.
