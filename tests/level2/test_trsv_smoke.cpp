// mp-blas level-2 test: triangular solve (trsv) through the MTL5 + Universal
// composition recovers a known solution in each number type, to the type's
// floor. Solves U x = b with U upper-triangular and x = ones. Returns non-zero
// on failure (no external framework).
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/trsv.hpp>

#include <universal/number/posit/posit.hpp>
#include <universal/number/cfloat/cfloat.hpp>

namespace {
template <typename T>
double trsv_error(std::size_t n) {
    // moderately conditioned upper triangular: unit diagonal, -0.5 above.
    mtl::mat::dense2D<T> U(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) U(i, j) = (i == j) ? T(1) : (j > i ? T(-0.5) : T(0));
    mtl::vec::dense_vector<T> b(n, T(0)), x(n, T(0));
    for (std::size_t i = 0; i < n; ++i) { T s(0); for (std::size_t j = i; j < n; ++j) s = s + U(i, j); b[i] = s; } // b = U*ones
    mtl::trsv(U, x, b, /*upper=*/true);
    double err = 0.0;
    for (std::size_t i = 0; i < n; ++i) err = std::max(err, std::abs(double(x[i]) - 1.0));
    return err;
}
template <typename T>
bool ok(const char* tag, double tol) {
    double e = trsv_error<T>(16);
    if (e > tol) { std::cerr << tag << " trsv failed: err=" << e << " tol=" << tol << '\n'; return false; }
    std::cout << tag << " trsv ok: err=" << e << '\n';
    return true;
}
} // namespace

int main() {
    using namespace sw::universal;
    int failures = 0;
    if (!ok<double>("double", 1e-12))                                      ++failures;
    if (!ok<float>("float", 1e-4))                                         ++failures;
    if (!ok<posit<32, 2>>("posit<32,2>", 1e-6))                            ++failures;
    if (!ok<cfloat<32, 8, std::uint32_t, true, false, false>>("cfloat<32,8>", 1e-4)) ++failures;
    if (failures == 0) std::cout << "test_trsv_smoke passed\n";
    return failures == 0 ? 0 : 1;
}
