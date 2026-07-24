// mp-blas level-3 accumulator study: matrix-matrix product C = A B.
//
// The L3 analogue of dot_accumulator_study. Fixes a low-precision ELEMENT type
// and sweeps the ACCUMULATOR (native / promoted-to-double / exact quire) for
// MTL5's mult(), delivering every result in double so the numbers isolate
// ACCUMULATION error on a swamping product (each C(i,j) = one large product +
// many small ones). The quire sums all products exactly and recovers them.
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>

#include <mtl/math/quire_accumulator.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/mult.hpp>

#include <universal/number/posit/posit.hpp>
#include <universal/number/cfloat/cfloat.hpp>

namespace {

template <typename T, typename Acc>
double matmul_err(std::size_t n, std::size_t smalls) {
    const double large = 16777216.0;
    mtl::mat::dense2D<T> A(n, n), B(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t k = 0; k < n; ++k) {
            A(i, k) = (k == 0) ? T(large) : (k <= smalls ? T(1) : T(0));
            B(i, k) = T(1);
        }
    const double cref = large + double(smalls);
    double err = 0.0;
    if constexpr (std::is_same_v<Acc, void>) {
        mtl::mat::dense2D<T> C(n, n); mtl::mult(A, B, C);
        for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < n; ++j) err = std::max(err, std::abs(double(C(i, j)) - cref) / cref);
    } else {
        mtl::mat::dense2D<double> C(n, n); mtl::mult<Acc>(A, B, C);
        for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < n; ++j) err = std::max(err, std::abs(C(i, j) - cref) / cref);
    }
    return err;
}

template <typename T>
void study(const std::string& tag, std::size_t n, std::size_t smalls) {
    using Quire = sw::universal::quire<T>;
    std::cout << "  " << std::left << std::setw(14) << tag
              << "  native="   << std::setw(12) << std::scientific << std::setprecision(3) << matmul_err<T, void>(n, smalls)
              << "  promoted=" << std::setw(12) << matmul_err<T, double>(n, smalls)
              << "  quire="    << std::setw(12) << matmul_err<T, Quire>(n, smalls) << '\n';
}

} // namespace

int main() {
    using namespace sw::universal;
    std::cout << "matrix-matrix accumulator study -- relative error of C = A B (swamping entries)\n";
    std::cout << "  element type    native accum.   promoted (double)   quire (exact)\n";
    study<posit<16, 2>>("posit<16,2>", 8, 6);
    study<posit<32, 2>>("posit<32,2>", 8, 6);
    study<cfloat<32, 8, std::uint32_t, true, false, false>>("cfloat<32,8>", 8, 6);
    return 0;
}
