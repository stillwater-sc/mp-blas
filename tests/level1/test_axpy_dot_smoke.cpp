// mp-blas level-1 smoke test: verify the MTL5 + Universal composition builds
// and that axpy / dot produce exact results in each number type. All values
// are small integers, exactly representable even in the 16-bit types, so any
// deviation is a real defect, not rounding. Returns non-zero on failure (no
// external test framework, matching the repo's lightweight style).
#include <cmath>
#include <cstddef>
#include <iostream>

#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/axpy.hpp>
#include <mtl/operation/dot.hpp>

// Universal number types: the kernels must run in these through the
// MTL5 + Universal composition.
#include <universal/number/posit/posit.hpp>
#include <universal/number/cfloat/cfloat.hpp>

namespace {

template <typename T>
bool level1_ok(const char* name) {
    constexpr std::size_t n = 8;
    mtl::vec::dense_vector<T> x(n, T(1));
    mtl::vec::dense_vector<T> y(n, T(2));

    // axpy: y <- 3*x + y = 5 everywhere
    mtl::axpy(T(3), x, y);
    for (std::size_t i = 0; i < n; ++i) {
        if (static_cast<double>(y(static_cast<int>(i))) != 5.0) {
            std::cerr << name << " axpy: y(" << i << ") = "
                      << static_cast<double>(y(static_cast<int>(i))) << ", expected 5\n";
            return false;
        }
    }

    // dot: x . y = 5*n = 40, exact in every type under test
    T d = mtl::dot(x, y);
    if (static_cast<double>(d) != 5.0 * n) {
        std::cerr << name << " dot: " << static_cast<double>(d)
                  << ", expected " << 5.0 * n << '\n';
        return false;
    }

    // mixed-precision dot: accumulate in double over low-precision elements
    double dm = mtl::dot<double>(x, y);
    if (dm != 5.0 * n) {
        std::cerr << name << " dot<double>: " << dm << ", expected " << 5.0 * n << '\n';
        return false;
    }

    return true;
}

} // namespace

int main() {
    int failures = 0;

    if (!level1_ok<double>("double")) ++failures;
    if (!level1_ok<float>("float")) ++failures;
    if (!level1_ok<sw::universal::cfloat<16, 5>>("cfloat<16,5>")) ++failures;
    if (!level1_ok<sw::universal::posit<16, 2>>("posit<16,2>")) ++failures;

    if (failures == 0) std::cout << "mp-blas level-1 smoke test passed\n";
    return failures == 0 ? 0 : 1;
}
