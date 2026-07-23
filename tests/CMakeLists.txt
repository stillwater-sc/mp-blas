include(CTest)

# Helper: each test is a self-checking executable linked against sw::mp_blas.
function(mpblas_add_test test_name)
	add_executable(${test_name} ${test_name}.cpp)
	target_link_libraries(${test_name} PRIVATE sw::mp_blas)
	add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

# Tests are organized by BLAS level:
#   level1 -- vector-vector: axpy, dot, nrm2, scal, ...
#   level2 -- matrix-vector: gemv, trsv, ger, ...
#   level3 -- matrix-matrix: gemm, trsm, syrk, ...
add_subdirectory(level1)
add_subdirectory(level2)
add_subdirectory(level3)
