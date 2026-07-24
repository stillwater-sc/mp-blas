# Level-2 BLAS: matrix-vector operations (gemv, trsv, ger, ...).

# Smoke test: gemv through the MTL5 + Universal composition.
mpblas_add_test(test_gemv_smoke)

# Exact/reproducible matvec via a Universal quire (L2 analogue of test_dot_quire).
mpblas_add_test(test_gemv_quire)

# Triangular solve (trsv) recovers a known solution across number types.
mpblas_add_test(test_trsv_smoke)
