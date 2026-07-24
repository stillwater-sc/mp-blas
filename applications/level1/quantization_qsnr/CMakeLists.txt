add_executable(quantization_qsnr quantization_qsnr.cpp)
target_link_libraries(quantization_qsnr PRIVATE sw::mp_blas)
