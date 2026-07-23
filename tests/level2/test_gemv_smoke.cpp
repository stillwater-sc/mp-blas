// mp-blas level-2 smoke test: verify gemv (y <- A*x) runs through the
// MTL5 + Universal composition in each number type. All values are small
// integers, exactly representable even in the 16-bit types, so results must
// be exact. Returns non-zero on failure (no external test framework).
#include <cstddef>
#include <iostream>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/mult.hpp>

#include <universal/number/posit/posit.hpp>
#include <universal/number/cfloat/cfloat.hpp>

namespace {

template <typename T>
bool gemv_ok(const char* name) {
    // A (2x3) * x (3) = y (2): [[1 2 3],[4 5 6]] * [1 1 1] = [6, 15]
    mtl::mat::dense2D<T> A{{T(1), T(2), T(3)},
                           {T(4), T(5), T(6)}};
    mtl::vec::dense_vector<T> x(3, T(1));
    mtl::vec::dense_vector<T> y(2, T(0));

    mtl::mult(A, x, y);

    const double expected[2] = {6.0, 15.0};
    for (int i = 0; i < 2; ++i) {
        if (static_cast<double>(y(i)) != expected[i]) {
            std::cerr << name << " gemv: y(" << i << ") = "
                      << static_cast<double>(y(i)) << ", expected " << expected[i] << '\n';
            return false;
        }
    }
    return true;
}

} // namespace

int main() {
    int failures = 0;

    if (!gemv_ok<double>("double")) ++failures;
    if (!gemv_ok<float>("float")) ++failures;
    if (!gemv_ok<sw::universal::cfloat<16, 5>>("cfloat<16,5>")) ++failures;
    if (!gemv_ok<sw::universal::posit<16, 2>>("posit<16,2>")) ++failures;

    if (failures == 0) std::cout << "mp-blas level-2 smoke test passed\n";
    return failures == 0 ? 0 : 1;
}
