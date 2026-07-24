// trsv_precision: triangular-solve accuracy across number precisions.
//
// Back-substitution error grows with the triangular factor's conditioning and
// the working precision -- the accuracy question behind mixed-precision LU/LDL^T
// factorization (feeds the mp-spice sparse-direct work). Solves U x = b for an
// upper-triangular U with x = ones, sweeping the off-diagonal magnitude (which
// controls conditioning) and reporting the recovered-solution error per type.
// MTL5's trsv has no accumulator seam, so this is an element-precision study.
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/trsv.hpp>

#include <universal/number/posit/posit.hpp>
#include <universal/number/cfloat/cfloat.hpp>

namespace {
template <typename T>
double trsv_error(std::size_t n, double off) {
    // upper triangular: unit diagonal, `off` above (larger |off| -> worse cond)
    mtl::mat::dense2D<T> U(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) U(i, j) = (i == j) ? T(1) : (j > i ? T(off) : T(0));
    mtl::vec::dense_vector<T> b(n, T(0)), x(n, T(0));
    for (std::size_t i = 0; i < n; ++i) { T s(0); for (std::size_t j = i; j < n; ++j) s = s + U(i, j); b[i] = s; }
    mtl::trsv(U, x, b, /*upper=*/true);
    double err = 0.0;
    for (std::size_t i = 0; i < n; ++i) err = std::max(err, std::abs(double(x[i]) - 1.0));
    return err;
}
template <typename T>
void report(const std::string& tag, std::size_t n) {
    std::cout << "  " << std::left << std::setw(14) << tag << std::right;
    for (double off : { -0.5, -0.9, -0.99 })
        std::cout << "  off=" << std::setw(5) << off << " err=" << std::setw(10) << std::scientific << std::setprecision(2) << trsv_error<T>(n, off);
    std::cout << '\n';
}
} // namespace

int main(int argc, char** argv) {
    using namespace sw::universal;
    std::size_t n = (argc > 1) ? static_cast<std::size_t>(std::stoul(argv[1])) : 24;
    std::cout << "triangular solve U x = b (x = ones), n = " << n << " -- error vs off-diagonal / precision\n";
    report<double>("double", n);
    report<float>("float", n);
    report<posit<32, 2>>("posit<32,2>", n);
    report<posit<16, 2>>("posit<16,2>", n);
    report<cfloat<16, 5, std::uint16_t, true, false, false>>("cfloat<16,5>", n);
    return 0;
}
