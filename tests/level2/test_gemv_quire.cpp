// mp-blas level-2 test: reproducible / exact matrix-vector product via a
// Universal quire super-accumulator (the MTL5 + Universal coupling that must not
// live in MTL5). The L2 analogue of test_dot_quire: each row of A is a swamping
// vector (one large entry followed by many small ones), so accumulating A*x in
// the element type loses the small terms, while the quire accumulates every
// product exactly and rounds once -- recovering them.
//
//   * ALWAYS: the quire matvec is no worse than native same-precision matvec.
//   * EXACT:  for element types whose quire is genuinely exact (posit, wide
//             cfloat), the quire recovers the swamped terms to (near) zero error.
//
// Returns non-zero on failure (no external test framework).
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>

#include <mtl/math/quire_accumulator.hpp>   // must precede gemv.hpp's traits use
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/mult.hpp>

#include <universal/number/posit/posit.hpp>
#include <universal/number/cfloat/cfloat.hpp>

namespace {

// max relative error of a matvec y = A x (element type T, accumulator Acc)
// against the exact double reference, over a swamping matrix.
template <typename T, typename Acc>
double matvec_error(std::size_t n, std::size_t smalls) {
    const double large = 16777216.0;   // 2^24
    const double small = 1.0;
    mtl::mat::dense2D<T> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = (j == 0) ? T(large) : (j <= smalls ? T(small) : T(0));
    mtl::vec::dense_vector<T> x(n, T(1));

    // reference in double: y_ref[i] = large + smalls*small
    const double yref = large + double(smalls) * small;

    double err = 0.0;
    if constexpr (std::is_same_v<Acc, void>) {
        mtl::vec::dense_vector<T> y(n, T(0));
        mtl::mult(A, x, y);
        for (std::size_t i = 0; i < n; ++i) err = std::max(err, std::abs(double(y[i]) - yref) / yref);
    } else {
        mtl::vec::dense_vector<double> y(n, 0.0);
        mtl::mult<Acc>(A, x, y);
        for (std::size_t i = 0; i < n; ++i) err = std::max(err, std::abs(y[i] - yref) / yref);
    }
    return err;
}

template <typename T, typename Quire>
bool check(const char* tag, bool expect_exact) {
    const std::size_t n = 12, smalls = 8;
    double native = matvec_error<T, void>(n, smalls);
    double quire  = matvec_error<T, Quire>(n, smalls);
    std::cout << tag << ": native rel err = " << native << ", quire rel err = " << quire << '\n';
    bool ok = quire <= native * (1.0 + 1e-12) + 1e-300;      // quire never worse
    if (expect_exact) ok = ok && (quire <= 1e-14);           // and exact for posit / wide cfloat
    if (!ok) std::cerr << tag << ": FAIL (native=" << native << " quire=" << quire << ")\n";
    return ok;
}

} // namespace

int main() {
    using namespace sw::universal;
    int failures = 0;
    if (!check<posit<32, 2>, quire<posit<32, 2>>>("posit<32,2>", true))  ++failures;
    if (!check<cfloat<32, 8, std::uint32_t, true, false, false>, quire<cfloat<32, 8, std::uint32_t, true, false, false>>>("cfloat<32,8>", true)) ++failures;
    if (failures == 0) std::cout << "test_gemv_quire passed\n";
    return failures == 0 ? 0 : 1;
}
