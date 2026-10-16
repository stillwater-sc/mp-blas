add_executable(matmul_accumulator_bench matmul_accumulator_bench.cpp)
# MTL5's thread pool runs on std::thread; the header-only fetch of MTL5 doesn't
# bring its Threads link along.
find_package(Threads REQUIRED)
target_link_libraries(matmul_accumulator_bench PRIVATE sw::mp_blas Threads::Threads)
