# Applications (demonstration programs)
# Organized by BLAS level (level1, level2, level3); each leaf subdirectory
# contains its own CMakeLists.txt.

# Dot-product accumulation error across precisions and accumulators.
add_subdirectory(level1/dot_precision)

# Milestone 1: accumulator sweep (native/fma/promoted/quire) vs vector length.
add_subdirectory(level1/dot_accumulator_study)
