// mp-blas level-3 smoke test: verify gemm (C <- A*B) runs through the
// MTL5 + Universal composition in each number type. All values are small
// integers, exactly representable even in the 16-bit types, so results must
// be exact. Returns non-zero on failure (no external test framework).
#include <cstddef>
#include <iostream>

#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/mult.hpp>

#include <universal/number/posit/posit.hpp>
#include <universal/number/cfloat/cfloat.hpp>

namespace {

template <typename T>
bool gemm_ok(const char* name) {
    // [[1 2],[3 4]] * [[5 6],[7 8]] = [[19 22],[43 50]]
    mtl::mat::dense2D<T> A{{T(1), T(2)},
                           {T(3), T(4)}};
    mtl::mat::dense2D<T> B{{T(5), T(6)},
                           {T(7), T(8)}};
    mtl::mat::dense2D<T> C(2, 2);

    mtl::mult(A, B, C);

    const double expected[2][2] = {{19.0, 22.0}, {43.0, 50.0}};
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
            if (static_cast<double>(C(r, c)) != expected[r][c]) {
                std::cerr << name << " gemm: C(" << r << "," << c << ") = "
                          << static_cast<double>(C(r, c)) << ", expected "
                          << expected[r][c] << '\n';
                return false;
            }
        }
    }
    return true;
}

} // namespace

int main() {
    int failures = 0;

    if (!gemm_ok<double>("double")) ++failures;
    if (!gemm_ok<float>("float")) ++failures;
    if (!gemm_ok<sw::universal::cfloat<16, 5>>("cfloat<16,5>")) ++failures;
    if (!gemm_ok<sw::universal::posit<16, 2>>("posit<16,2>")) ++failures;

    if (failures == 0) std::cout << "mp-blas level-3 smoke test passed\n";
    return failures == 0 ? 0 : 1;
}
