// mp-blas -- interval level-3 study: gemm and the rank-k update (issue #12, Phase 4).
//
// Two jobs, and the second is the interesting one.
//
// 1. gemm at scale. Each C(i,j) is a dot product, so Phase 1 should reproduce
//    element-wise exactly as it did for gemv in Phase 2. A level-3 result that
//    DIFFERED would indicate a broken seam rather than a discovery.
//
// 2. The rank-k update, which is a direct test of a question Phase 2 raised.
//    Phase 2 found that k successive `ger` calls accumulate k products into every
//    element of A -- a reduction spread across CALLS, which no per-call
//    accumulator seam can reach, so the width grew ~linearly in k. The proposed
//    remedy was structural: express the same computation as ONE gemm,
//
//        A += X * Y^T        X is n x k, Y^T is k x n
//
//    so that A(i,j) = sum_u X(i,u) * Y^T(u,j) becomes a single k-term reduction
//    that the gemm accumulator seam CAN accumulate exactly. Both formulations
//    compute the same mathematical quantity, so the comparison is fair, and the
//    study checks all three enclose the same reference.
//
//    This is the one entry in the three-limits table (docs/interval-blas-study.md)
//    where the answer might change -- the "interface" limit is the only one that
//    is not mathematical or representational.
//
// Usage: interval_matmul_study [nmax]     (default nmax = 64)
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <mtl/math/interval_quire_accumulator.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/operation/ger.hpp>

#include <sw/mp_blas/dot_characterization.hpp>
#include <sw/mp_blas/interval_containment.hpp>

#include <universal/number/posit/posit.hpp>
#include <universal/number/cfloat/cfloat.hpp>
#include <universal/number/interval/interval.hpp>

namespace {

using sw::mp_blas::exact_ref;
using sw::mp_blas::regime;

struct Case { regime r; double param; const char* label; };

const Case kRegimes[] = {
    { regime::uniform,  0.0,   "uniform"     },
    { regime::graded,   40.0,  "graded"      },
    { regime::cancel,   1e-3,  "cancel 1e-3" },
    { regime::cancel,   1e-9,  "cancel 1e-9" },
    { regime::kahan,    1e-6,  "kahan 1e-6"  },
};

// ---------------------------------------------------------------------------
// 1. gemm at scale
// ---------------------------------------------------------------------------

/// C = A B with every row of A and every column of B carrying the same issue-#9
/// regime instance, so each C(i,j) IS a Phase 1 dot product and the comparison
/// to level 1 is direct. All strategies deliver interval<double>.
template <typename Scalar>
void gemm_study(const std::string& name, std::size_t nmax) {
    using I  = sw::universal::interval<Scalar>;
    using ID = sw::universal::interval<double>;
    using QA = sw::mp_blas::interval_quire<Scalar>;

    std::cout << "\n=== gemm: C = A B, interval<" << name << "> ===\n"
              << "  worst element reported; each element is one Phase 1 dot product\n"
              << std::left << std::setw(13) << "regime" << std::right << std::setw(6) << "n"
              << std::setw(10) << "cond"
              << std::setw(11) << "naive" << std::setw(11) << "quire"
              << std::setw(9) << "dig(nv)" << std::setw(9) << "dig(qr)" << '\n';

    for (const auto& c : kRegimes) {
        for (std::size_t n = 8; n <= nmax; n *= 4) {
            auto [xv, yv] = sw::mp_blas::generate(c.r, n, c.param);

            mtl::mat::dense2D<I> A(n, n), B(n, n);
            for (std::size_t k = 0; k < n; ++k) {
                const Scalar sx(xv[k]), sy(yv[k]);
                xv[k] = static_cast<double>(sx);
                yv[k] = static_cast<double>(sy);
                for (std::size_t i = 0; i < n; ++i) A(i, k) = I(sx);
                for (std::size_t j = 0; j < n; ++j) B(k, j) = I(sy);
            }

            const double u = 0.5 * static_cast<double>(std::numeric_limits<Scalar>::epsilon());
            const auto f = sw::mp_blas::characterize(xv, yv, u);
            const auto truth = exact_ref::from(sw::mp_blas::dot2(xv, yv));

            mtl::mat::dense2D<I>  Cn(n, n);
            mtl::mat::dense2D<ID> Cq(n, n);
            mtl::mult(A, B, Cn);
            mtl::mult<QA>(A, B, Cq);

            double wn = 0.0, wq = 0.0;
            bool ok = true;
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = 0; j < n; ++j) {
                    const ID np(static_cast<double>(Cn(i, j).lower()),
                                static_cast<double>(Cn(i, j).upper()));
                    wn = std::max(wn, sw::mp_blas::relative_width(np));
                    wq = std::max(wq, sw::mp_blas::relative_width(Cq(i, j)));
                    ok = ok && sw::mp_blas::encloses(np, truth)
                            && sw::mp_blas::encloses(Cq(i, j), truth);
                }
            }

            std::cout << std::left << std::setw(13) << c.label << std::right << std::setw(6) << n
                      << std::scientific << std::setprecision(1) << std::setw(10) << f.cond
                      << std::setw(11) << wn << std::setw(11) << wq
                      << std::fixed << std::setprecision(1)
                      << std::setw(9) << -std::log10(wn) << std::setw(9) << -std::log10(wq)
                      << (ok ? "" : "   *** CONTAINMENT LOST ***")
                      << std::defaultfloat << '\n';
        }
    }
}

// ---------------------------------------------------------------------------
// 2. The rank-k update: does re-expressing dissolve the interface limit?
// ---------------------------------------------------------------------------

/// Compute A = sum_u x_u y_u^T three ways and compare enclosure width vs k.
///
/// Width is reported relative to max |A(i,j)| rather than per element. A rank-k
/// update has many elements with midpoints near zero, and relative_width divides
/// by the midpoint -- so a per-element relative measure is unstable here for
/// reasons that have nothing to do with the accumulator. Normalizing by the
/// largest magnitude in the matrix is stable and is the natural matrix analogue.
template <typename Scalar>
void rank_k_study(const std::string& name, std::size_t kmax) {
    using I  = sw::universal::interval<Scalar>;
    using ID = sw::universal::interval<double>;
    using QA = sw::mp_blas::interval_quire<Scalar>;
    const std::size_t n = 8;

    std::cout << "\n=== rank-k update, interval<" << name << ">, n = " << n << " ===\n"
              << "  the SAME quantity three ways:\n"
              << "    ger x k     : k successive rank-1 updates -- reduction across CALLS\n"
              << "    gemm naive  : one gemm, interval arithmetic\n"
              << "    gemm quire  : one gemm, exact accumulation of the k-term reduction\n"
              << std::right << std::setw(6) << "k"
              << std::setw(12) << "ger x k" << std::setw(12) << "gemm nv"
              << std::setw(12) << "gemm qr" << std::setw(12) << "ger/qr"
              << "   contained\n";

    for (std::size_t k = 1; k <= kmax; k *= 4) {
        std::vector<std::vector<double>> xs(k, std::vector<double>(n));
        std::vector<std::vector<double>> ys(k, std::vector<double>(n));
        for (std::size_t u = 0; u < k; ++u)
            for (std::size_t i = 0; i < n; ++i) {
                xs[u][i] = static_cast<double>(Scalar(sw::mp_blas::synth(i, 100 + u)));
                ys[u][i] = static_cast<double>(Scalar(sw::mp_blas::synth(i, 900 - u)));
            }

        // (a) k successive ger calls
        mtl::mat::dense2D<I> A(n, n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) A(i, j) = I(Scalar(0));
        for (std::size_t u = 0; u < k; ++u) {
            mtl::vec::dense_vector<I> x(n, I(Scalar(0))), y(n, I(Scalar(0)));
            for (std::size_t i = 0; i < n; ++i) {
                x[i] = I(Scalar(xs[u][i]));
                y[i] = I(Scalar(ys[u][i]));
            }
            mtl::ger(I(Scalar(1)), x, y, A);
        }

        // (b, c) one gemm: X is n x k, Yt is k x n
        mtl::mat::dense2D<I> X(n, k), Yt(k, n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t u = 0; u < k; ++u) X(i, u) = I(Scalar(xs[u][i]));
        for (std::size_t u = 0; u < k; ++u)
            for (std::size_t j = 0; j < n; ++j) Yt(u, j) = I(Scalar(ys[u][j]));

        mtl::mat::dense2D<I>  Cn(n, n);
        mtl::mat::dense2D<ID> Cq(n, n);
        mtl::mult(X, Yt, Cn);
        mtl::mult<QA>(X, Yt, Cq);

        // Normalizer: the largest magnitude in the exact result.
        double scale = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) {
                std::vector<double> px(k), py(k);
                for (std::size_t u = 0; u < k; ++u) { px[u] = xs[u][i]; py[u] = ys[u][j]; }
                scale = std::max(scale, std::abs(sw::mp_blas::dot2(px, py)));
            }
        if (scale == 0.0) scale = 1.0;

        double wg = 0.0, wn = 0.0, wq = 0.0;
        bool ok = true;
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                std::vector<double> px(k), py(k);
                for (std::size_t u = 0; u < k; ++u) { px[u] = xs[u][i]; py[u] = ys[u][j]; }
                const auto truth = exact_ref::from(sw::mp_blas::dot2(px, py));
                ok = ok && sw::mp_blas::encloses(A(i, j), truth)
                        && sw::mp_blas::encloses(Cn(i, j), truth)
                        && sw::mp_blas::encloses(Cq(i, j), truth);
                wg = std::max(wg, sw::mp_blas::absolute_width(A(i, j)));
                wn = std::max(wn, sw::mp_blas::absolute_width(Cn(i, j)));
                wq = std::max(wq, sw::mp_blas::absolute_width(Cq(i, j)));
            }
        }

        std::cout << std::right << std::setw(6) << k
                  << std::scientific << std::setprecision(2)
                  << std::setw(12) << wg / scale << std::setw(12) << wn / scale
                  << std::setw(12) << wq / scale
                  << std::setw(12) << (wq > 0.0 ? wg / wq : 0.0)
                  << "   " << (ok ? "yes" : "*** NO ***")
                  << std::defaultfloat << '\n';
    }
}

} // namespace

int main(int argc, char* argv[]) {
    std::size_t nmax = 64;
    if (argc > 1) nmax = static_cast<std::size_t>(std::atoll(argv[1]));

    std::cout << "Interval level-3 study: gemm and the rank-k update (issue #12, Phase 4)\n"
              << "  gemm   -- Phase 1 element-wise, at level 3\n"
              << "  rank-k -- does re-expressing k ger calls as one gemm dissolve the\n"
              << "            interface limit found in Phase 2?\n"
              << "Widths are rigorous enclosures; `dig` is decimal digits certified.\n";

    gemm_study<sw::universal::posit<32, 2>>("posit<32,2>", nmax);
    gemm_study<sw::universal::cfloat<32, 8>>("cfloat<32,8>", nmax);

    rank_k_study<sw::universal::posit<32, 2>>("posit<32,2>", 256);
    rank_k_study<sw::universal::cfloat<32, 8>>("cfloat<32,8>", 256);

    return 0;
}
