// mp-blas -- interval nrm2: the dependency problem, measured (issue #12).
//
// This closes Phase 1's open item. `nrm2` was deferred because computing it as
// sqrt(dot(x,x)) hands the same interval to both arguments, and interval
// arithmetic cannot know they are the same object -- the DEPENDENCY PROBLEM. For
// an element X = [a,b] straddling zero, `dot` evaluates the general product X*X
// (lower endpoint a*b < 0) instead of the true square X^2 = [0, max(a^2,b^2)].
//
// It is the one obstacle in docs/interval-blas-study.md that is mathematical
// rather than structural: no accumulator fixes it, because an exact quire will
// faithfully accumulate the wrong (over-wide) corner products. The fix is to stop
// asking `dot` -- see include/sw/mp_blas/interval_nrm2.hpp.
//
// Two sweeps, deliberately separating the two effects that both improve nrm2:
//
//   A. straddling fraction -- isolates DEPENDENCY. Fixed length, varying how many
//      elements contain zero. Exact accumulation is used by both methods, so the
//      only difference is square-vs-product.
//
//   B. vector length -- isolates ACCUMULATION. No element straddles zero, so
//      there is no dependency loss at all and the only question is whether the
//      sum of squares is accumulated exactly. A sum of squares is the `positive`
//      regime of issue #9 (cond at its floor of 2), so this is pure n*u drift.
//
// The headline is the LOWER bound rather than the width: a lower bound on ||x||
// is what proves a vector is non-degenerate, a residual is genuinely nonzero, or
// a matrix is not rank-deficient. Losing it is a qualitative failure, not a
// precision one.
//
// Usage: interval_nrm2_study [nmax]     (default nmax = 4096)
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <sw/mp_blas/interval_nrm2.hpp>
#include <sw/mp_blas/dot_characterization.hpp>
#include <sw/mp_blas/interval_containment.hpp>

#include <universal/number/posit/posit.hpp>
#include <universal/number/cfloat/cfloat.hpp>
#include <universal/number/interval/interval.hpp>

namespace {

using ID = sw::universal::interval<double>;

/// Build a test vector where `frac` of the elements are centered on zero (and so
/// straddle it), the rest well away from it.
template <typename Scalar>
mtl::vec::dense_vector<sw::universal::interval<Scalar>>
make_vector(std::size_t n, double frac, double half_width, double& true_lo, double& true_hi) {
    using I = sw::universal::interval<Scalar>;
    mtl::vec::dense_vector<I> x(n, I(Scalar(0)));
    true_lo = 0.0;
    true_hi = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double c = (static_cast<double>(i) < frac * static_cast<double>(n))
                             ? 0.0
                             : 1.0 + 0.5 * sw::mp_blas::synth(i, 11);
        const Scalar lo(c - half_width), hi(c + half_width);
        x[i] = I(lo, hi);
        const double a = static_cast<double>(lo), b = static_cast<double>(hi);
        true_lo += (a <= 0.0 && 0.0 <= b) ? 0.0 : std::min(a * a, b * b);
        true_hi += std::max(a * a, b * b);
    }
    true_lo = std::sqrt(true_lo);
    true_hi = std::sqrt(true_hi);
    return x;
}

// ---------------------------------------------------------------------------
// A. The dependency effect
// ---------------------------------------------------------------------------

template <typename Scalar>
void straddle_sweep(const std::string& name, std::size_t n) {
    std::cout << "\n=== A. dependency: nrm2 vs fraction of elements straddling zero,\n"
              << "       interval<" << name << ">, n = " << n << " ===\n"
              << "  both accumulate exactly; the only difference is square vs product\n"
              << std::right << std::setw(9) << "straddle"
              << std::setw(11) << "dot lo" << std::setw(11) << "sq lo"
              << std::setw(11) << "dot hi" << std::setw(11) << "width x"
              << std::setw(13) << "lower bound" << '\n';

    for (double frac : {0.0, 0.25, 0.5, 0.75, 1.0}) {
        double tl = 0.0, th = 0.0;
        auto x = make_vector<Scalar>(n, frac, 0.5, tl, th);

        const ID nd = sw::mp_blas::nrm2_via_dot<ID>(x);
        const ID ne = sw::mp_blas::nrm2_exact<ID, Scalar>(x);

        const bool ok = static_cast<double>(nd.lower()) <= tl && th <= static_cast<double>(nd.upper())
                     && static_cast<double>(ne.lower()) <= tl && th <= static_cast<double>(ne.upper());
        const double wd = sw::mp_blas::absolute_width(nd);
        const double we = sw::mp_blas::absolute_width(ne);

        std::cout << std::right << std::setw(9) << std::fixed << std::setprecision(2) << frac
                  << std::setw(11) << std::setprecision(4) << static_cast<double>(nd.lower())
                  << std::setw(11) << static_cast<double>(ne.lower())
                  << std::setw(11) << static_cast<double>(nd.upper())
                  << std::setw(11) << std::setprecision(2) << (we > 0.0 ? wd / we : 1.0)
                  << std::setw(13)
                  << (static_cast<double>(nd.lower()) <= 0.0 && static_cast<double>(ne.lower()) > 0.0
                          ? "LOST -> kept"
                          : "")
                  << (ok ? "" : "  *** CONTAINMENT ***")
                  << std::defaultfloat << '\n';
    }
    std::cout << "  `dot lo` is what sqrt(dot(x,x)) can certify about ||x||; `sq lo` is what\n"
              << "  the sum-of-squares formulation certifies. A lower bound of 0 proves nothing.\n";
}

// ---------------------------------------------------------------------------
// B. The accumulation effect, with dependency removed
// ---------------------------------------------------------------------------

/// Sum of squares accumulated in the ELEMENT type rather than a quire, so the
/// n*u drift of a long reduction is visible. No element straddles zero here, so
/// the dependency loss is zero and this isolates accumulation.
template <typename Interval, typename Scalar, typename Vector>
Interval nrm2_squares_naive(const Vector& x) {
    using sw::universal::sqrt;
    using I = sw::universal::interval<Scalar>;
    I acc(Scalar(0));
    for (std::size_t i = 0; i < static_cast<std::size_t>(x.size()); ++i)
        acc = acc + sw::mp_blas::square(x[i]);
    return sqrt(Interval(static_cast<double>(acc.lower()), static_cast<double>(acc.upper())));
}

template <typename Scalar>
void length_sweep(const std::string& name, std::size_t nmax) {
    std::cout << "\n=== B. accumulation: nrm2 vs length, interval<" << name << "> ===\n"
              << "  DEGENERATE inputs (zero-width elements, none straddling zero), so the\n"
              << "  dependency loss is zero AND the input contributes no uncertainty --\n"
              << "  every bit of the reported width is manufactured by the accumulation\n"
              << std::right << std::setw(8) << "n"
              << std::setw(13) << "naive acc" << std::setw(13) << "quire acc"
              << std::setw(11) << "ratio" << '\n';

    for (std::size_t n = 64; n <= nmax; n *= 4) {
        double tl = 0.0, th = 0.0;
        auto x = make_vector<Scalar>(n, 0.0, 0.0, tl, th);   // half-width 0: points

        const ID a = nrm2_squares_naive<ID, Scalar>(x);
        const ID q = sw::mp_blas::nrm2_exact<ID, Scalar>(x);

        const double wa = sw::mp_blas::relative_width(a);
        const double wq = sw::mp_blas::relative_width(q);
        std::cout << std::right << std::setw(8) << n
                  << std::scientific << std::setprecision(2)
                  << std::setw(13) << wa << std::setw(13) << wq
                  << std::fixed << std::setprecision(1) << std::setw(11) << wa / wq
                  << std::defaultfloat << '\n';
    }
}

} // namespace

int main(int argc, char* argv[]) {
    std::size_t nmax = 4096;
    if (argc > 1) nmax = static_cast<std::size_t>(std::atoll(argv[1]));

    std::cout << "Interval nrm2: the dependency problem (issue #12, Phase 1's open item)\n"
              << "  sqrt(dot(x,x)) evaluates X*X, not X^2, for elements straddling zero.\n"
              << "  No accumulator fixes that -- an exact quire accumulates the wrong\n"
              << "  corner products exactly. The fix is to evaluate the square directly.\n";

    straddle_sweep<sw::universal::posit<32, 2>>("posit<32,2>", 16);
    straddle_sweep<sw::universal::cfloat<32, 8>>("cfloat<32,8>", 16);

    length_sweep<sw::universal::posit<32, 2>>("posit<32,2>", nmax);

    return 0;
}
