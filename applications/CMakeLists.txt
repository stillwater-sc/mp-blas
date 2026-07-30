# Applications (demonstration programs)
# Organized by BLAS level (level1, level2, level3); each leaf subdirectory
# contains its own CMakeLists.txt.

# Dot-product accumulation error across precisions and accumulators.
add_subdirectory(level1/dot_precision)

# Milestone 1: accumulator sweep (native/fma/promoted/quire) vs vector length.
add_subdirectory(level1/dot_accumulator_study)

# Level-2/3 accumulator sweeps (native/promoted/quire) on swamping problems.
add_subdirectory(level2/matvec_accumulator_study)
add_subdirectory(level3/matmul_accumulator_study)

# Triangular-solve accuracy across number precisions (feeds mp-spice).
add_subdirectory(level2/trsv_precision)

# Quantization QSNR + mixed-precision dot (migrated from Universal mpdot/mpfma).
add_subdirectory(level1/quantization_qsnr)

# Dot-product characterization: structural features (cond, D, rho, n_eff, beta)
# of an inner product vs the accumulation error and reproducibility they predict.
add_subdirectory(level1/dot_characterization)

# Interval BLAS Phase 1 (issue #12): what a dot product can PROVE. Naive vs
# promoted vs exact interval-quire enclosure width across the issue-9 regimes.
add_subdirectory(level1/interval_dot_study)

# Interval BLAS Phase 2 (issue #12): ger as the reduction-free control, gemv as
# Phase 1 row-wise -- separates interval multiplication from accumulation.
add_subdirectory(level2/interval_matvec_study)

# Interval BLAS Phase 3 (issue #12): the wrapping effect under repeated plane
# rotations -- where exact accumulation is predicted to stop helping.
add_subdirectory(level1/interval_rot_study)
