// mp-blas level-2 test: interval ger and gemv (issue #12, Phase 2).
//
// Phase 2 exists to separate interval MULTIPLICATION from interval ACCUMULATION,
// so that Phase 1's result ("exact accumulation collapses the enclosure width")
// can be attributed to the right cause. The two level-2 operators do that split:
//
//   ger    A += alpha*x*y^T -- one multiply-add per element, NO reduction. MTL5's
//          ger has no Accumulator parameter and correctly so. This is the CONTROL:
//          if width here is small and independent of problem size, then Phase 1's
//          win came from removing accumulation rounding, not from products.
//
//   gemv   y = A x -- ger's widening plus one reduction per output row, and
//          mult()'s Accumulator seam takes the interval quire. Must reproduce
//          Phase 1 row-wise.
//
// Also pins a structural finding: a single ger has no reduction, but k successive
// rank-1 updates accumulate k products into every element of A -- a reduction
// spread across CALLS, which no per-call accumulator seam can reach.
//
// Depends on tests/level1/test_interval_containment (Phase 0) for correctness;
// these are tightness and attribution claims on top of it.
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

/// One rank-1 update; returns {max absolute element width, max overestimation R}.
/// Absolute width because many elements of x*y^T have midpoints near zero.
template <typename Scalar>
std::pair<double, double> ger_once(std::size_t n) {
    using I = sw::universal::interval<Scalar>;

    mtl::mat::dense2D<I> A(n, n);
    mtl::vec::dense_vector<I> x(n, I(Scalar(0))), y(n, I(Scalar(0)));
    std::vector<double> xd(n), yd(n);
    for (std::size_t i = 0; i < n; ++i) {
        const Scalar sx(sw::mp_blas::synth(i, 1)), sy(sw::mp_blas::synth(i, 2));
        xd[i] = static_cast<double>(sx);
        yd[i] = static_cast<double>(sy);
        x[i] = I(sx);
        y[i] = I(sy);
    }
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) A(i, j) = I(Scalar(0));

    mtl::ger(I(Scalar(1)), x, y, A);

    double wmax = 0.0, rmax = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            wmax = std::max(wmax, sw::mp_blas::absolute_width(A(i, j)));
            const auto truth = exact_ref::product(xd[i], yd[j]);
            check(sw::mp_blas::encloses(A(i, j), truth),
                  "ger/n=" + std::to_string(n) + ": element (" + std::to_string(i) + "," +
                      std::to_string(j) + ") does not enclose the exact product");
            if (sw::mp_blas::in_range<Scalar>(truth))
                rmax = std::max(rmax, sw::mp_blas::overestimation<I, Scalar>(A(i, j), truth));
        }
    }
    return { wmax, rmax };
}

} // namespace

int main() {
    using namespace sw::universal;
    using P32  = posit<32, 2>;
    using CF32 = cfloat<32, 8>;
    using ID   = interval<double>;

    std::cout << "issue #12 Phase 2 -- interval ger and gemv\n";

    // -- 1. THE CONTROL: single ger width is independent of problem size ------
    // No reduction, so nothing accumulates and the per-element width is set by
    // one multiply-add. If this were to grow with n, Phase 1's attribution would
    // be wrong and the story would be about products rather than sums.
    {
        const auto s4   = ger_once<P32>(4);
        const auto s16  = ger_once<P32>(16);
        const auto s64  = ger_once<P32>(64);

        check(std::abs(s64.first / s4.first - 1.0) < 1e-9 &&
              std::abs(s16.first / s4.first - 1.0) < 1e-9,
              "ger control: single-update width is NOT independent of n (4: " +
                  std::to_string(s4.first) + ", 16: " + std::to_string(s16.first) +
                  ", 64: " + std::to_string(s64.first) + ") -- Phase 1's attribution "
                  "of its win to accumulation would need revisiting");
        // One multiply-add should land within a small constant of optimal.
        check(s64.second < 16.0,
              "ger control: overestimation R=" + std::to_string(s64.second) +
                  " is larger than a single interval multiply-add should produce");
        std::cout << "  ger control  n=4/16/64: width " << std::scientific << std::setprecision(2)
                  << s4.first << " / " << s16.first << " / " << s64.first
                  << "   R=" << std::fixed << std::setprecision(2) << s64.second
                  << " (flat => Phase 1's win was accumulation)\n" << std::defaultfloat;
    }

    // -- 2. Repeated ger IS a reduction, with no seam to fix it ---------------
    // k successive rank-1 updates put k products into every element. MTL5's
    // per-call interface has nowhere to hold an accumulator across calls, so the
    // width grows and cannot be recovered at this level. Asserting the growth
    // pins the limitation so it cannot silently disappear (or silently worsen).
    {
        using I = interval<P32>;
        const std::size_t n = 16;
        auto width_after = [&](std::size_t k) {
            mtl::mat::dense2D<I> A(n, n);
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j) A(i, j) = I(P32(0));
            for (std::size_t u = 0; u < k; ++u) {
                mtl::vec::dense_vector<I> x(n, I(P32(0))), y(n, I(P32(0)));
                for (std::size_t i = 0; i < n; ++i) {
                    x[i] = I(P32(sw::mp_blas::synth(i, 100 + u)));
                    y[i] = I(P32(sw::mp_blas::synth(i, 900 - u)));
                }
                mtl::ger(I(P32(1)), x, y, A);
            }
            double w = 0.0;
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j)
                    w = std::max(w, sw::mp_blas::absolute_width(A(i, j)));
            return w;
        };
        const double w1 = width_after(1), w64 = width_after(64);
        check(w64 > 10.0 * w1,
              "repeated ger: width did not grow with the number of updates (1: " +
                  std::to_string(w1) + ", 64: " + std::to_string(w64) +
                  ") -- if an accumulator seam appeared, this note is stale");
        std::cout << "  repeated ger k=1 -> 64: width " << std::scientific << std::setprecision(2)
                  << w1 << " -> " << w64 << " (x" << std::fixed << std::setprecision(0)
                  << w64 / w1 << ", no seam reaches it)\n" << std::defaultfloat;
    }

    // -- 3. gemv reproduces Phase 1 row-wise ---------------------------------
    {
        struct C { regime r; double p; const char* l; };
        const C cases[] = {
            { regime::uniform,  0.0,  "uniform"     },
            { regime::graded,   40.0, "graded"      },
            { regime::cancel,   1e-3, "cancel 1e-3" },
            { regime::cancel,   1e-9, "cancel 1e-9" },
            { regime::kahan,    1e-6, "kahan 1e-6"  },
        };

        auto gemv = [&](auto tag, const char* tname, const C& c, std::size_t n) {
            using Scalar = decltype(tag);
            using I  = interval<Scalar>;
            using QA = sw::mp_blas::interval_quire<Scalar>;

            auto [xv, yv] = sw::mp_blas::generate(c.r, n, c.p);
            mtl::mat::dense2D<I> A(n, n);
            mtl::vec::dense_vector<I> x(n, I(Scalar(0)));
            for (std::size_t j = 0; j < n; ++j) {
                const Scalar sy(yv[j]); yv[j] = static_cast<double>(sy); x[j] = I(sy);
            }
            for (std::size_t j = 0; j < n; ++j) {
                const Scalar sx(xv[j]); xv[j] = static_cast<double>(sx);
                for (std::size_t i = 0; i < n; ++i) A(i, j) = I(sx);
            }
            const auto truth = exact_ref::from(sw::mp_blas::dot2(xv, yv));

            mtl::vec::dense_vector<I>  yn(n, I(Scalar(0)));
            mtl::vec::dense_vector<ID> yq(n, ID(0.0));
            mtl::mult(A, x, yn);
            mtl::mult<QA>(A, x, yq);

            const std::string tg = std::string(tname) + "/" + c.l + "/n=" + std::to_string(n);
            double wn = 0.0, wq = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                const ID np(static_cast<double>(yn[i].lower()),
                            static_cast<double>(yn[i].upper()));
                check(sw::mp_blas::encloses(np, truth), tg + ": naive gemv row lost containment");
                check(sw::mp_blas::encloses(yq[i], truth), tg + ": QUIRE gemv row lost containment");
                wn = std::max(wn, sw::mp_blas::relative_width(np));
                wq = std::max(wq, sw::mp_blas::relative_width(yq[i]));
            }
            check(wq <= wn * (1.0 + 1e-9), tg + ": quire row wider than naive");
            return std::pair<double, double>{ wn, wq };
        };

        for (const auto& c : cases) {
            for (std::size_t n : {16u, 128u}) {
                gemv(P32{},  "posit<32,2>",  c, n);
                gemv(CF32{}, "cfloat<32,8>", c, n);
            }
        }

        // H2 row-wise: the quire row width does not move with cond, while naive
        // crosses into certifying nothing.
        const auto benign = gemv(CF32{}, "cfloat<32,8>", cases[0], 128);
        const auto hard   = gemv(CF32{}, "cfloat<32,8>", cases[4], 128);
        check(hard.first > 1.0,
              "expected naive gemv to be useless on kahan/n=128; width " +
                  std::to_string(hard.first));
        const double drift = hard.second / benign.second;
        check(drift > 0.25 && drift < 4.0,
              "H2 row-wise: quire gemv width moved with cond (benign " +
                  std::to_string(benign.second) + " -> kahan " + std::to_string(hard.second) +
                  ", x" + std::to_string(drift) + ")");
        std::cout << "  gemv cfloat<32,8> n=128:  naive " << std::scientific << std::setprecision(2)
                  << benign.second << " -> " << hard.first << " (useless);  quire "
                  << benign.second << " -> " << hard.second << " (flat)\n" << std::defaultfloat;
    }

    if (failures == 0) std::cout << "test_interval_l2 passed\n";
    return failures == 0 ? 0 : 1;
}
