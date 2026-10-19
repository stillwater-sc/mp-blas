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

# Interval quire accumulator (issue #12, Phase 1): the H1/H2 tightness claims --
# naive interval width grows with n and cond, exact quire accumulation is flat in
# both. Depends on test_interval_containment staying green (correctness first).
mpblas_add_test(test_interval_quire)

# Interval rot and wrapping (issue #12, Phase 3). Pins a NEGATIVE result: exact
# accumulation changes the constant but not the geometric growth RATE, because
# wrapping is a loss in the representation rather than in the arithmetic.
mpblas_add_test(test_interval_rot)

# Interval nrm2 (issue #12): the dependency problem. sqrt(dot(x,x)) evaluates
# X*X, not X^2, for elements straddling zero -- and no accumulator fixes that.
# Also pins that accumulation is a separate defect needing the quire.
mpblas_add_test(test_interval_nrm2)

# The configuration-3 (quire) rows of the accumulation validation matrix,
# stillwater-sc/mtl5#586 (issue #34): dot, sum, norms and determinism, plain /
# FMA / compensated / quire at one element type. Configuration 1 is "the
# product rounds", so the compiler must not contract acc += a*b into an FMA.
mpblas_add_test(test_accumulation_matrix_l1)
if(NOT MSVC)
	target_compile_options(test_accumulation_matrix_l1 PRIVATE -ffp-contract=off)
endif()
