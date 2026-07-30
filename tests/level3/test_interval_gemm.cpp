// mp-blas level-3 test: interval gemm and the rank-k update (issue #12, Phase 4).
//
// Two claims are pinned here, and the second is the one that changes an entry in
// the three-limits table of docs/interval-blas-study.md.
//
// 1. gemm reproduces Phase 1 element-wise. Each C(i,j) is a dot product, so a
//    level-3 result that DIFFERED from level 1 would mean a broken accumulator
//    seam, not a discovery. Asserted so a seam regression is caught here rather
//    than inferred later from odd numbers.
//
// 2. THE RANK-K RESULT. Phase 2 found that k successive `ger` calls form a
//    reduction spread across CALLS, which no per-call accumulator seam can reach,
//    so the enclosure width grows with k. The proposed remedy was structural:
//    express the same computation as one gemm, A += X*Y^T, so the sum becomes a
//    single k-term reduction the gemm seam can accumulate exactly.
//
//    The measurement says the remedy works, but only in BOTH halves:
//      * re-expressing as gemm alone does NOT help -- naive gemm degrades with k
//        just like repeated ger;
//      * re-expressing AND accumulating exactly makes the width flat in k.
//
//    So the "interface" limit is genuinely fixable, and the fix is structural and
//    arithmetic together. The assertions below are shaped to catch either half
//    silently regressing.
//
// All three formulations compute the same mathematical quantity, and the test
// checks all three enclose the same reference -- otherwise the comparison would
// be meaningless.
//
// Returns non-zero on failure (no external framework).
#include <algorithm>
#include <cmath>
#include <cstddef>
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

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) { std::cerr << "FAIL: " << what << '\n'; ++failures; }
}

/// Rank-k update computed three ways; widths normalized by max |A(i,j)|.
///
/// Normalizing by the matrix's largest magnitude rather than per element: a
/// rank-k update has many elements with midpoints near zero, and relative_width
/// divides by the midpoint, so a per-element measure is unstable for reasons that
/// have nothing to do with the accumulator.
struct RankK {
    double ger = 0.0, gemm_naive = 0.0, gemm_quire = 0.0;
    bool contained = true;
};

template <typename Scalar>
RankK rank_k(std::size_t n, std::size_t k) {
    using I  = sw::universal::interval<Scalar>;
    using ID = sw::universal::interval<double>;
    using QA = sw::mp_blas::interval_quire<Scalar>;

    std::vector<std::vector<double>> xs(k, std::vector<double>(n));
    std::vector<std::vector<double>> ys(k, std::vector<double>(n));
    for (std::size_t u = 0; u < k; ++u)
        for (std::size_t i = 0; i < n; ++i) {
            xs[u][i] = static_cast<double>(Scalar(sw::mp_blas::synth(i, 100 + u)));
            ys[u][i] = static_cast<double>(Scalar(sw::mp_blas::synth(i, 900 - u)));
        }

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

    mtl::mat::dense2D<I> X(n, k), Yt(k, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t u = 0; u < k; ++u) X(i, u) = I(Scalar(xs[u][i]));
    for (std::size_t u = 0; u < k; ++u)
        for (std::size_t j = 0; j < n; ++j) Yt(u, j) = I(Scalar(ys[u][j]));

    mtl::mat::dense2D<I>  Cn(n, n);
    mtl::mat::dense2D<ID> Cq(n, n);
    mtl::mult(X, Yt, Cn);
    mtl::mult<QA>(X, Yt, Cq);

    RankK r;
    double scale = 0.0, wg = 0.0, wn = 0.0, wq = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            std::vector<double> px(k), py(k);
            for (std::size_t u = 0; u < k; ++u) { px[u] = xs[u][i]; py[u] = ys[u][j]; }
            const double exact = sw::mp_blas::dot2(px, py);
            const auto truth = exact_ref::from(exact);
            scale = std::max(scale, std::abs(exact));

            // All three must enclose the SAME value, or the comparison is void.
            r.contained = r.contained && sw::mp_blas::encloses(A(i, j), truth)
                                      && sw::mp_blas::encloses(Cn(i, j), truth)
                                      && sw::mp_blas::encloses(Cq(i, j), truth);
            wg = std::max(wg, sw::mp_blas::absolute_width(A(i, j)));
            wn = std::max(wn, sw::mp_blas::absolute_width(Cn(i, j)));
            wq = std::max(wq, sw::mp_blas::absolute_width(Cq(i, j)));
        }
    }
    if (scale == 0.0) scale = 1.0;
    r.ger = wg / scale; r.gemm_naive = wn / scale; r.gemm_quire = wq / scale;
    return r;
}

} // namespace

int main() {
    using namespace sw::universal;
    using P32  = posit<32, 2>;
    using CF32 = cfloat<32, 8>;
    using ID   = interval<double>;

    std::cout << "issue #12 Phase 4 -- interval gemm and the rank-k update\n";

    // -- 1. gemm reproduces Phase 1 element-wise -----------------------------
    {
        struct C { regime r; double p; const char* l; };
        const C cases[] = {
            { regime::uniform, 0.0,  "uniform"     },
            { regime::cancel,  1e-3, "cancel 1e-3" },
            { regime::kahan,   1e-6, "kahan 1e-6"  },
        };

        auto run = [&](auto tag, const char* tname, const C& c, std::size_t n) {
            using Scalar = decltype(tag);
            using I  = interval<Scalar>;
            using QA = sw::mp_blas::interval_quire<Scalar>;

            auto [xv, yv] = sw::mp_blas::generate(c.r, n, c.p);
            mtl::mat::dense2D<I> A(n, n), B(n, n);
            for (std::size_t k = 0; k < n; ++k) {
                const Scalar sx(xv[k]), sy(yv[k]);
                xv[k] = static_cast<double>(sx);
                yv[k] = static_cast<double>(sy);
                for (std::size_t i = 0; i < n; ++i) A(i, k) = I(sx);
                for (std::size_t j = 0; j < n; ++j) B(k, j) = I(sy);
            }
            const auto truth = exact_ref::from(sw::mp_blas::dot2(xv, yv));

            mtl::mat::dense2D<I>  Cn(n, n);
            mtl::mat::dense2D<ID> Cq(n, n);
            mtl::mult(A, B, Cn);
            mtl::mult<QA>(A, B, Cq);

            const std::string tg = std::string(tname) + "/" + c.l + "/n=" + std::to_string(n);
            double wn = 0.0, wq = 0.0;
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j) {
                    const ID np(static_cast<double>(Cn(i, j).lower()),
                                static_cast<double>(Cn(i, j).upper()));
                    check(sw::mp_blas::encloses(np, truth), tg + ": naive gemm lost containment");
                    check(sw::mp_blas::encloses(Cq(i, j), truth), tg + ": QUIRE gemm lost containment");
                    wn = std::max(wn, sw::mp_blas::relative_width(np));
                    wq = std::max(wq, sw::mp_blas::relative_width(Cq(i, j)));
                }
            check(wq <= wn * (1.0 + 1e-9), tg + ": quire gemm wider than naive");
            return std::pair<double, double>{ wn, wq };
        };

        for (const auto& c : cases)
            for (std::size_t n : {8u, 32u}) {
                run(P32{},  "posit<32,2>",  c, n);
                run(CF32{}, "cfloat<32,8>", c, n);
            }

        // Phase 1's signature at level 3: the quire lands on ~4e-16 regardless of
        // conditioning, while naive collapses on kahan.
        const auto benign = run(P32{}, "posit<32,2>", cases[0], 32);
        const auto hard   = run(P32{}, "posit<32,2>", cases[2], 32);
        check(hard.first > 1.0,
              "expected naive gemm to be useless on kahan; width " + std::to_string(hard.first));
        const double drift = hard.second / benign.second;
        check(drift > 0.25 && drift < 4.0,
              "gemm: quire width moved with cond (benign " + std::to_string(benign.second) +
                  " -> kahan " + std::to_string(hard.second) + ")");
        std::cout << "  gemm posit<32,2> n=32:  naive " << std::scientific << std::setprecision(2)
                  << benign.first << " -> " << hard.first << " (useless);  quire "
                  << benign.second << " -> " << hard.second << " (flat)\n" << std::defaultfloat;
    }

    // -- 2. THE RANK-K RESULT ------------------------------------------------
    {
        auto report = [&](auto tag, const char* tname) {
            using Scalar = decltype(tag);
            const std::size_t n = 8;
            const RankK r1   = rank_k<Scalar>(n, 1);
            const RankK r256 = rank_k<Scalar>(n, 256);

            check(r1.contained && r256.contained,
                  std::string(tname) + ": rank-k formulations do not all enclose the same "
                  "reference -- the comparison is not measuring one quantity");

            // (a) Phase 2's interface limit is still real: repeated ger degrades.
            check(r256.ger > 10.0 * r1.ger,
                  std::string(tname) + ": repeated ger did not degrade with k (" +
                      std::to_string(r1.ger) + " -> " + std::to_string(r256.ger) +
                      ") -- Phase 2's interface finding would need revisiting");

            // (b) Re-expressing as gemm is NOT sufficient on its own. This is the
            //     half that is easy to assume away, so it is asserted explicitly.
            check(r256.gemm_naive > 10.0 * r1.gemm_naive,
                  std::string(tname) + ": naive gemm did NOT degrade with k -- if "
                  "re-expression alone now suffices, the Phase 4 finding that the fix "
                  "needs both halves is wrong");

            // (c) Re-expression + exact accumulation dissolves it: flat in k.
            const double drift = r256.gemm_quire / r1.gemm_quire;
            check(drift > 0.25 && drift < 4.0,
                  std::string(tname) + ": gemm+quire width is not flat in k (" +
                      std::to_string(r1.gemm_quire) + " -> " + std::to_string(r256.gemm_quire) +
                      ", x" + std::to_string(drift) + ")");

            // (d) And the gap is enormous, not marginal.
            check(r256.ger / r256.gemm_quire > 1e6,
                  std::string(tname) + ": gemm+quire is not decisively tighter than repeated "
                  "ger at k=256 (x" + std::to_string(r256.ger / r256.gemm_quire) + ")");

            std::cout << "  rank-k " << std::left << std::setw(14) << tname
                      << " k=1 -> 256:  ger x" << std::fixed << std::setprecision(0)
                      << r256.ger / r1.ger
                      << ",  gemm naive x" << r256.gemm_naive / r1.gemm_naive
                      << ",  gemm quire x" << std::setprecision(2) << drift
                      << "   (ger/quire = " << std::scientific << std::setprecision(1)
                      << r256.ger / r256.gemm_quire << ")\n" << std::defaultfloat;
        };

        report(P32{},  "posit<32,2>");
        report(CF32{}, "cfloat<32,8>");
    }

    if (failures == 0) std::cout << "test_interval_gemm passed\n";
    return failures == 0 ? 0 : 1;
}
