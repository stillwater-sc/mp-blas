# Level-1 BLAS: vector-vector operations (axpy, dot, nrm2, scal, ...).

# Smoke test: axpy + dot solve through the MTL5 + Universal composition.
mpblas_add_test(test_axpy_dot_smoke)

# Exact dot product via a Universal quire super-accumulator (issue: L1
# accumulator study). The quire must recover terms that native same-precision
# accumulation swamps.
mpblas_add_test(test_dot_quire)

# Quantization signal-to-noise ratio (QSNR) of number types + mixed-precision dot.
mpblas_add_test(test_qsnr)
