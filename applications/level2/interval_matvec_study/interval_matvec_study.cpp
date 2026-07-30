// mp-blas -- interval level-2 study: ger and gemv (issue #12, Phase 2).
//
// Phase 1 showed that for `dot`, exact accumulation collapses the enclosure width
// from "grows with n and cond" to "flat at ~1 ulp". Phase 2 asks what part of that
// was accumulation and what part was interval multiplication, by measuring the two
// level-2 operators that separate them.
//
//   ger    A += alpha*x*y^T -- ONE multiply-add per element, NO reduction.
//          MTL5's ger has no Accumulator parameter, and correctly so: there is
//          nothing to accumulate. This is the CONTROL. If enclosure width here is
//          small and independent of problem size, then Phase 1's win came from
//          removing accumulation rounding, not from anything about products.
//
//   gemv   y = A x -- ger's per-element widening PLUS one reduction per output
//          row. mult()'s Accumulator seam accepts the interval quire, so this
//          should reproduce Phase 1 row-wise.
//
// A third measurement, which the two-operator split makes visible: a single ger
// has no reduction, but a SEQUENCE of k rank-1 updates accumulates k products into
// every element of A -- a reduction spread across calls, where no accumulator seam
// can reach it. See the findings in docs/interval-blas-study.md.
//
// Every width below is a rigorous enclosure (Phase 0 gate), so `digits` is what
// the computation certifies.
//
// Usage: interval_matvec_study [nmax]     (default nmax = 512)
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
// ger: the control
// ---------------------------------------------------------------------------

/// A single rank-1 update. Reports the WORST element enclosure over all of A, so
/// growing the problem can only make the reported number worse -- if it stays
/// flat, that is meaningful rather than an artifact of averaging.
///
/// Absolute width, not relative: many elements of a rank-1 update have midpoints
/// near zero (x(i)*y(j) with either factor near zero), and relative_width divides
/// by the midpoint.
template <typename Scalar>
void ger_size_sweep(const std::string& name, std::size_t nmax) {
    using I = sw::universal::interval<Scalar>;

    std::cout << "\n=== ger control: single rank-1 update, interval<" << name << "> ===\n"
              << "  no reduction, so no accumulator can apply; width should be flat in n\n"
              << std::right << std::setw(9) << "n" << std::setw(15) << "max abs width"
              << std::setw(15) << "max R" << '\n';

    for (std::size_t n = 4; n <= nmax; n *= 4) {
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
                // Exact truth for one element: the product of two representable
                // values, available exactly from an error-free transformation.
                const auto truth = exact_ref::product(xd[i], yd[j]);
                if (!sw::mp_blas::encloses(A(i, j), truth)) {
                    std::cout << "  *** CONTAINMENT LOST at (" << i << "," << j << ") ***\n";
                    return;
                }
                if (sw::mp_blas::in_range<Scalar>(truth))
                    rmax = std::max(rmax, sw::mp_blas::overestimation<I, Scalar>(A(i, j), truth));
            }
        }
        std::cout << std::right << std::setw(9) << n
                  << std::scientific << std::setprecision(2) << std::setw(15) << wmax
                  << std::setw(15) << rmax << std::defaultfloat << '\n';
    }
}

/// k successive rank-1 updates into the same A. Each element then holds a sum of
/// k products -- a reduction, but one spread across k calls, so MTL5's per-call
/// interface has nowhere to put an accumulator. Width should grow with k, and
/// there should be no way to stop it at this level.
template <typename Scalar>
void ger_repeat_sweep(const std::string& name, std::size_t kmax) {
    using I = sw::universal::interval<Scalar>;
    const std::size_t n = 16;

    std::cout << "\n=== ger: k successive rank-1 updates, interval<" << name << ">, n = "
              << n << " ===\n"
              << "  a reduction spread across CALLS -- no accumulator seam reaches it\n"
              << std::right << std::setw(9) << "k" << std::setw(15) << "max abs width"
              << std::setw(12) << "vs k=1" << '\n';

    double w1 = 0.0;
    for (std::size_t k = 1; k <= kmax; k *= 4) {
        mtl::mat::dense2D<I> A(n, n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) A(i, j) = I(Scalar(0));

        for (std::size_t u = 0; u < k; ++u) {
            mtl::vec::dense_vector<I> x(n, I(Scalar(0))), y(n, I(Scalar(0)));
            for (std::size_t i = 0; i < n; ++i) {
                x[i] = I(Scalar(sw::mp_blas::synth(i, 100 + u)));
                y[i] = I(Scalar(sw::mp_blas::synth(i, 900 - u)));
            }
            mtl::ger(I(Scalar(1)), x, y, A);
        }

        double wmax = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                wmax = std::max(wmax, sw::mp_blas::absolute_width(A(i, j)));
        if (w1 == 0.0) w1 = wmax;

        std::cout << std::right << std::setw(9) << k
                  << std::scientific << std::setprecision(2) << std::setw(15) << wmax
                  << std::fixed << std::setprecision(1) << std::setw(12) << wmax / w1
                  << std::defaultfloat << '\n';
    }
}

// ---------------------------------------------------------------------------
// gemv: Phase 1, row-wise
// ---------------------------------------------------------------------------

/// y = A x, with every row of A carrying the same issue-#9 regime instance, so
/// each output element IS a Phase 1 dot product and the comparison is direct.
/// All strategies deliver interval<double> (the level1-accumulator-study
/// convention) so the numbers isolate accumulation, not the output type.
template <typename Scalar>
void gemv_study(const std::string& name, std::size_t nmax) {
    using I  = sw::universal::interval<Scalar>;
    using ID = sw::universal::interval<double>;
    using QA = sw::mp_blas::interval_quire<Scalar>;

    std::cout << "\n=== gemv: y = A x, interval<" << name << "> ===\n"
              << "  worst row reported; each row is one Phase 1 dot product\n"
              << std::left << std::setw(13) << "regime" << std::right << std::setw(7) << "n"
              << std::setw(10) << "cond"
              << std::setw(11) << "naive" << std::setw(11) << "quire"
              << std::setw(9) << "dig(nv)" << std::setw(9) << "dig(qr)"
              << std::setw(11) << "R(quire)" << '\n';

    for (const auto& c : kRegimes) {
        for (std::size_t n = 16; n <= nmax; n *= 8) {
            auto [xv, yv] = sw::mp_blas::generate(c.r, n, c.param);

            mtl::mat::dense2D<I> A(n, n);
            mtl::vec::dense_vector<I> x(n, I(Scalar(0)));
            for (std::size_t j = 0; j < n; ++j) {
                const Scalar sy(yv[j]);
                yv[j] = static_cast<double>(sy);
                x[j] = I(sy);
            }
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = 0; j < n; ++j) {
                    const Scalar sx(xv[j]);
                    A(i, j) = I(sx);
                    if (i == 0) xv[j] = static_cast<double>(sx);
                }
            }

            const double u = 0.5 * static_cast<double>(std::numeric_limits<Scalar>::epsilon());
            const auto f = sw::mp_blas::characterize(xv, yv, u);
            const auto truth = exact_ref::from(sw::mp_blas::dot2(xv, yv));

            mtl::vec::dense_vector<I>  yn(n, I(Scalar(0)));
            mtl::vec::dense_vector<ID> yq(n, ID(0.0));
            mtl::mult(A, x, yn);
            mtl::mult<QA>(A, x, yq);

            double wn = 0.0, wq = 0.0, rq = 0.0;
            bool ok = true;
            for (std::size_t i = 0; i < n; ++i) {
                const ID nprom(static_cast<double>(yn[i].lower()),
                               static_cast<double>(yn[i].upper()));
                wn = std::max(wn, sw::mp_blas::relative_width(nprom));
                wq = std::max(wq, sw::mp_blas::relative_width(yq[i]));
                rq = std::max(rq, sw::mp_blas::overestimation<ID, double>(yq[i], truth));
                ok = ok && sw::mp_blas::encloses(nprom, truth)
                        && sw::mp_blas::encloses(yq[i], truth);
            }

            std::cout << std::left << std::setw(13) << c.label << std::right << std::setw(7) << n
                      << std::scientific << std::setprecision(1) << std::setw(10) << f.cond
                      << std::setw(11) << wn << std::setw(11) << wq
                      << std::fixed << std::setprecision(1)
                      << std::setw(9) << -std::log10(wn) << std::setw(9) << -std::log10(wq)
                      << std::scientific << std::setprecision(1) << std::setw(11) << rq
                      << (ok ? "" : "   *** CONTAINMENT LOST ***")
                      << std::defaultfloat << '\n';
        }
    }
}

} // namespace

int main(int argc, char* argv[]) {
    std::size_t nmax = 512;
    if (argc > 1) nmax = static_cast<std::size_t>(std::atoll(argv[1]));

    std::cout << "Interval level-2 study: ger and gemv (issue #12, Phase 2)\n"
              << "Separates interval MULTIPLICATION from interval ACCUMULATION:\n"
              << "  ger  = one multiply-add per element, no reduction  -> the control\n"
              << "  gemv = ger's widening plus one reduction per row   -> Phase 1, row-wise\n"
              << "Widths are rigorous enclosures; `dig` is decimal digits certified.\n";

    ger_size_sweep<sw::universal::posit<32, 2>>("posit<32,2>", std::min<std::size_t>(nmax, 256));
    ger_repeat_sweep<sw::universal::posit<32, 2>>("posit<32,2>", 256);
    gemv_study<sw::universal::posit<32, 2>>("posit<32,2>", nmax);
    gemv_study<sw::universal::cfloat<32, 8>>("cfloat<32,8>", nmax);

    return 0;
}
