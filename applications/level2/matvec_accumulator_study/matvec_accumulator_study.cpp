// mp-blas level-2 accumulator study: matrix-vector product y = A x.
//
// The L2 analogue of dot_accumulator_study. Fixes a low-precision ELEMENT type
// and sweeps the ACCUMULATOR (native / promoted-to-double / exact quire) for
// MTL5's mult(), delivering every result in double so the numbers isolate
// ACCUMULATION error on a swamping matrix (each row = one large entry + many
// small ones). The quire sums all products exactly and recovers the small terms.
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>

#include <mtl/math/quire_accumulator.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/mult.hpp>

#include <universal/number/posit/posit.hpp>
#include <universal/number/cfloat/cfloat.hpp>

namespace {

template <typename T, typename Acc>
double matvec_err(std::size_t n, std::size_t smalls) {
    const double large = 16777216.0, small = 1.0;
    mtl::mat::dense2D<T> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) A(i, j) = (j == 0) ? T(large) : (j <= smalls ? T(small) : T(0));
    mtl::vec::dense_vector<T> x(n, T(1));
    const double yref = large + double(smalls) * small;
    double err = 0.0;
    if constexpr (std::is_same_v<Acc, void>) {
        mtl::vec::dense_vector<T> y(n, T(0)); mtl::mult(A, x, y);
        for (std::size_t i = 0; i < n; ++i) err = std::max(err, std::abs(double(y[i]) - yref) / yref);
    } else {
        mtl::vec::dense_vector<double> y(n, 0.0); mtl::mult<Acc>(A, x, y);
        for (std::size_t i = 0; i < n; ++i) err = std::max(err, std::abs(y[i] - yref) / yref);
    }
    return err;
}

template <typename T>
void study(const std::string& tag, std::size_t n, std::size_t smalls) {
    using Quire = sw::universal::quire<T>;
    std::cout << "  " << std::left << std::setw(14) << tag
              << "  native="   << std::setw(12) << std::scientific << std::setprecision(3) << matvec_err<T, void>(n, smalls)
              << "  promoted=" << std::setw(12) << matvec_err<T, double>(n, smalls)
              << "  quire="    << std::setw(12) << matvec_err<T, Quire>(n, smalls) << '\n';
}

} // namespace

int main() {
    using namespace sw::universal;
    std::cout << "matrix-vector accumulator study -- relative error of y = A x (swamping rows)\n";
    std::cout << "  element type    native accum.   promoted (double)   quire (exact)\n";
    study<posit<16, 2>>("posit<16,2>", 12, 8);
    study<posit<32, 2>>("posit<32,2>", 12, 8);
    study<cfloat<32, 8, std::uint32_t, true, false, false>>("cfloat<32,8>", 12, 8);
    return 0;
}
