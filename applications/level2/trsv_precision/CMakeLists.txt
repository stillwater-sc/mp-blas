add_executable(trsv_precision trsv_precision.cpp)
target_link_libraries(trsv_precision PRIVATE sw::mp_blas)
