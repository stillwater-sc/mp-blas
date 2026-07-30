# Level-3 BLAS: matrix-matrix operations (gemm, trsm, syrk, ...).

# Smoke test: gemm through the MTL5 + Universal composition.
mpblas_add_test(test_gemm_smoke)

# Exact/reproducible matmul via a Universal quire (L3 analogue of test_dot_quire).
mpblas_add_test(test_gemm_quire)

# Interval level-3 (issue #12, Phase 4): gemm reproducing Phase 1 element-wise,
# and the rank-k result -- re-expressing k ger calls as one gemm dissolves the
# Phase 2 interface limit, but only when combined with exact accumulation.
mpblas_add_test(test_interval_gemm)
