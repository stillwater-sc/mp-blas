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
