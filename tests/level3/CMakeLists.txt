# Level-3 BLAS: matrix-matrix operations (gemm, trsm, syrk, ...).

# Smoke test: gemm through the MTL5 + Universal composition.
mpblas_add_test(test_gemm_smoke)

# Exact/reproducible matmul via a Universal quire (L3 analogue of test_dot_quire).
mpblas_add_test(test_gemm_quire)
