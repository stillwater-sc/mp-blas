# Level-3 BLAS: matrix-matrix operations (gemm, trsm, syrk, ...).

# Smoke test: gemm through the MTL5 + Universal composition.
mpblas_add_test(test_gemm_smoke)

# Exact/reproducible matmul via a Universal quire (L3 analogue of test_dot_quire).
mpblas_add_test(test_gemm_quire)

# Interval level-3 (issue #12, Phase 4): gemm reproducing Phase 1 element-wise,
# and the rank-k result -- re-expressing k ger calls as one gemm dissolves the
# Phase 2 interface limit, but only when combined with exact accumulation.
mpblas_add_test(test_interval_gemm)

# The configuration-3 (quire) rows of the accumulation validation matrix,
# stillwater-sc/mtl5#586 (issue #34): gemm and spgemm, plain / FMA /
# compensated / quire at float. Configuration 1 is "the product rounds", so the
# compiler must not contract acc += a*b into an FMA.
mpblas_add_test(test_accumulation_matrix_l3)
if(NOT MSVC)
	target_compile_options(test_accumulation_matrix_l3 PRIVATE -ffp-contract=off)
endif()
