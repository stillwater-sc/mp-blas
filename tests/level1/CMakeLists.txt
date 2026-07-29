# Level-1 BLAS: vector-vector operations (axpy, dot, nrm2, scal, ...).

# Smoke test: axpy + dot solve through the MTL5 + Universal composition.
mpblas_add_test(test_axpy_dot_smoke)

# Exact dot product via a Universal quire super-accumulator (issue: L1
# accumulator study). The quire must recover terms that native same-precision
# accumulation swamps.
mpblas_add_test(test_dot_quire)

# Quantization signal-to-noise ratio (QSNR) of number types + mixed-precision dot.
mpblas_add_test(test_qsnr)

# Dot-product characterization (issue #9): the structural feature vector
# (cond, D, rho, n_eff, beta), the Wilkinson/Higham error bound it implies, and
# the order-independence of exact quire accumulation.
mpblas_add_test(test_dot_characterization)

# Interval BLAS containment gate (issue #12, Phase 0). Guards the Fundamental
# Theorem of Interval Arithmetic across element types and operators. Every width
# measurement in issue #12's later phases is meaningless unless this is green.
mpblas_add_test(test_interval_containment)
