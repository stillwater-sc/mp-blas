// mp-blas level-3 test: reproducible / exact matrix-matrix product via a
// Universal quire super-accumulator (the MTL5 + Universal coupling that must not
// live in MTL5). The L3 analogue of test_dot_quire / test_gemv_quire: each entry
// C(i,j) = sum_k A(i,k) B(k,j) is a swamping sum (one large product followed by
// many small ones), so accumulating C in the element type loses the small terms,
// while the quire accumulates every product exactly and rounds once.
//
//   * ALWAYS: the quire matmul is no worse than native same-precision matmul.
//   * EXACT:  for element types whose quire is exact (posit, wide cfloat), the
//             quire recovers the swamped terms to (near) zero error.
//
// Returns non-zero on failure (no external test framework).
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>

#include <mtl/math/quire_accumulator.hpp>   // must precede mult.hpp's traits use
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/mult.hpp>

#include <universal/number/posit/posit.hpp>
#include <universal/number/cfloat/cfloat.hpp>

namespace {

// max relative error of C = A B (element type T, accumulator Acc) against the
// exact double reference, over a swamping product.
template <typename T, typename Acc>
double matmul_error(std::size_t n, std::size_t smalls) {
    const double large = 16777216.0;   // 2^24
    // A(i,0) = large, A(i,k) = 1 for 1<=k<=smalls (else 0); B(k,j) = 1.
    // C(i,j) = large + smalls, a swamping sum in each output entry.
    mtl::mat::dense2D<T> A(n, n), B(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t k = 0; k < n; ++k) {
            A(i, k) = (k == 0) ? T(large) : (k <= smalls ? T(1) : T(0));
            B(i, k) = T(1);
        }
    const double cref = large + double(smalls);

    double err = 0.0;
    if constexpr (std::is_same_v<Acc, void>) {
        mtl::mat::dense2D<T> C(n, n);
        mtl::mult(A, B, C);
        for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < n; ++j) err = std::max(err, std::abs(double(C(i, j)) - cref) / cref);
    } else {
        mtl::mat::dense2D<double> C(n, n);
        mtl::mult<Acc>(A, B, C);
        for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < n; ++j) err = std::max(err, std::abs(C(i, j) - cref) / cref);
    }
    return err;
}

template <typename T, typename Quire>
bool check(const char* tag, bool expect_exact) {
    const std::size_t n = 8, smalls = 6;
    double native = matmul_error<T, void>(n, smalls);
    double quire  = matmul_error<T, Quire>(n, smalls);
    std::cout << tag << ": native rel err = " << native << ", quire rel err = " << quire << '\n';
    bool ok = quire <= native * (1.0 + 1e-12) + 1e-300;
    if (expect_exact) ok = ok && (quire <= 1e-14);
    if (!ok) std::cerr << tag << ": FAIL (native=" << native << " quire=" << quire << ")\n";
    return ok;
}

} // namespace

int main() {
    using namespace sw::universal;
    int failures = 0;
    if (!check<posit<32, 2>, quire<posit<32, 2>>>("posit<32,2>", true))  ++failures;
    if (!check<cfloat<32, 8, std::uint32_t, true, false, false>, quire<cfloat<32, 8, std::uint32_t, true, false, false>>>("cfloat<32,8>", true)) ++failures;
    if (failures == 0) std::cout << "test_gemm_quire passed\n";
    return failures == 0 ? 0 : 1;
}
