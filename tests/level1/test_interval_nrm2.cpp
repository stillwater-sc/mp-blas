// mp-blas level-1 test: interval nrm2 and the dependency problem (issue #12).
//
// Closes Phase 1's open item. sqrt(dot(x,x)) hands the same interval to both
// arguments, and interval arithmetic cannot know they are the same object -- the
// DEPENDENCY PROBLEM. For X = [a,b] straddling zero, `dot` evaluates the general
// product X*X (lower endpoint a*b < 0) instead of the true square
// X^2 = [0, max(a^2,b^2)].
//
// nrm2 has TWO independent defects and needs both fixed. The test keeps them apart:
//   dependency    fixed by evaluating the square directly; NO accumulator helps,
//                 since an exact quire accumulates the wrong corners exactly.
//   accumulation  fixed by the quire, exactly as in Phase 1; visible even when no
//                 element straddles zero.
//
// The headline assertion is about the LOWER bound, not the width: a lower bound
// on ||x|| is what proves a vector non-degenerate or a residual genuinely
// nonzero. Losing it is a qualitative failure, not a precision one.
//
// Returns non-zero on failure (no external framework).
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>

#include <sw/mp_blas/interval_nrm2.hpp>
#include <sw/mp_blas/dot_characterization.hpp>
#include <sw/mp_blas/interval_containment.hpp>

#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/dot.hpp>

#include <universal/number/posit/posit.hpp>
#include <universal/number/cfloat/cfloat.hpp>
#include <universal/number/interval/interval.hpp>

namespace {

using ID = sw::universal::interval<double>;

int failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) { std::cerr << "FAIL: " << what << '\n'; ++failures; }
}

/// `frac` of the elements are centered on zero (so straddle it); the rest are
/// well away. Also returns the true nrm2 range over the box.
template <typename Scalar>
mtl::vec::dense_vector<sw::universal::interval<Scalar>>
make_vector(std::size_t n, double frac, double half_width, double& tl, double& th) {
    using I = sw::universal::interval<Scalar>;
    mtl::vec::dense_vector<I> x(n, I(Scalar(0)));
    tl = 0.0; th = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double c = (static_cast<double>(i) < frac * static_cast<double>(n))
                             ? 0.0 : 1.0 + 0.5 * sw::mp_blas::synth(i, 11);
        const Scalar lo(c - half_width), hi(c + half_width);
        x[i] = I(lo, hi);
        const double a = static_cast<double>(lo), b = static_cast<double>(hi);
        tl += (a <= 0.0 && 0.0 <= b) ? 0.0 : std::min(a * a, b * b);
        th += std::max(a * a, b * b);
    }
    tl = std::sqrt(tl); th = std::sqrt(th);
    return x;
}

} // namespace

int main() {
    using namespace sw::universal;
    using P32  = posit<32, 2>;
    using CF32 = cfloat<32, 8>;
    using I    = interval<P32>;

    std::cout << "issue #12 -- interval nrm2 and the dependency problem\n";

    // -- 1. square() is a square: never negative, and tight ------------------
    {
        struct C { double a, b, lo, hi; const char* l; };
        const C cases[] = {
            {  1.0,  2.0, 1.0, 4.0, "positive"              },
            { -2.0, -1.0, 1.0, 4.0, "negative"              },
            { -1.0,  2.0, 0.0, 4.0, "straddles"             },
            { -3.0,  1.0, 0.0, 9.0, "straddles, |lo| bigger"},
            {  0.0,  2.0, 0.0, 4.0, "touches 0"             },
        };
        for (const auto& c : cases) {
            const I s = sw::mp_blas::square(I(P32(c.a), P32(c.b)));
            const double lo = static_cast<double>(s.lower()), hi = static_cast<double>(s.upper());
            const std::string t = std::string("square(") + c.l + ")";
            check(lo >= 0.0, t + ": lower bound " + std::to_string(lo) + " is negative");
            check(lo <= c.lo + 1e-9 && hi >= c.hi - 1e-9,
                  t + ": [" + std::to_string(lo) + ", " + std::to_string(hi) +
                      "] does not enclose the true square");
            check(hi <= c.hi * 1.001 + 1e-9, t + ": upper bound is loose");
        }
    }

    // -- 2. The dependency problem, in one assertion -------------------------
    {
        const I straddling(P32(-1.0), P32(2.0));
        check(static_cast<double>((straddling * straddling).lower()) < 0.0,
              "the general product X*X did NOT go negative for a straddling X -- if "
              "interval multiplication now tracks dependency, this item is moot");
        check(static_cast<double>(sw::mp_blas::square(straddling).lower()) == 0.0,
              "square() of a straddling interval must have lower bound exactly 0");

        const I positive(P32(1.0), P32(2.0));
        check(static_cast<double>((positive * positive).lower()) ==
                  static_cast<double>(sw::mp_blas::square(positive).lower()),
              "square() and X*X must agree when X does not straddle zero");
    }

    // -- 3. THE HEADLINE: the certified lower bound on ||x|| -----------------
    {
        auto run = [&](auto tag, const char* tname) {
            using Scalar = decltype(tag);
            for (double frac : {0.0, 0.25, 0.5, 0.75, 1.0}) {
                double tl = 0.0, th = 0.0;
                auto x = make_vector<Scalar>(16, frac, 0.5, tl, th);
                const ID nd = sw::mp_blas::nrm2_via_dot<ID>(x);
                const ID ne = sw::mp_blas::nrm2_exact<ID, Scalar>(x);
                const std::string tg = std::string(tname) + "/straddle=" + std::to_string(frac);

                check(static_cast<double>(nd.lower()) <= tl && th <= static_cast<double>(nd.upper()),
                      tg + ": sqrt(dot(x,x)) does not enclose the true nrm2 range");
                check(static_cast<double>(ne.lower()) <= tl && th <= static_cast<double>(ne.upper()),
                      tg + ": exact-squares nrm2 does not enclose the true nrm2 range");
                check(static_cast<double>(ne.lower()) >= static_cast<double>(nd.lower()) - 1e-12,
                      tg + ": exact-squares lower bound is WORSE than sqrt(dot(x,x))");
                check(sw::mp_blas::absolute_width(ne) <=
                          sw::mp_blas::absolute_width(nd) * (1.0 + 1e-9),
                      tg + ": exact-squares enclosure is wider than sqrt(dot(x,x))");
            }

            // At 75% straddling sqrt(dot(x,x)) certifies NOTHING about ||x||: a
            // lower bound of 0 cannot prove the vector is nonzero.
            double tl = 0.0, th = 0.0;
            auto x = make_vector<Scalar>(16, 0.75, 0.5, tl, th);
            const ID nd = sw::mp_blas::nrm2_via_dot<ID>(x);
            const ID ne = sw::mp_blas::nrm2_exact<ID, Scalar>(x);
            check(static_cast<double>(nd.lower()) <= 0.0,
                  std::string(tname) + ": expected sqrt(dot(x,x)) to lose the lower bound "
                  "entirely at 75% straddling");
            check(static_cast<double>(ne.lower()) > 1.0,
                  std::string(tname) + ": expected the square formulation to certify a "
                  "positive lower bound at 75% straddling");

            std::cout << "  " << std::left << std::setw(14) << tname
                      << " 75% straddling:  sqrt(dot(x,x)) proves ||x|| >= "
                      << std::fixed << std::setprecision(4) << static_cast<double>(nd.lower())
                      << " (nothing),  squares prove ||x|| >= "
                      << static_cast<double>(ne.lower()) << '\n' << std::defaultfloat;
        };
        run(P32{},  "posit<32,2>");
        run(CF32{}, "cfloat<32,8>");
    }

    // -- 4. Accumulation is a SEPARATE defect --------------------------------
    // Degenerate inputs, none straddling: dependency loss is zero, so whatever is
    // measured here is accumulation alone. Confirms the two fixes are independent
    // and that nrm2 needs both.
    {
        auto naive_acc = [](const mtl::vec::dense_vector<I>& x) {
            I acc(P32(0));
            for (std::size_t i = 0; i < static_cast<std::size_t>(x.size()); ++i)
                acc = acc + sw::mp_blas::square(x[i]);
            return sqrt(ID(static_cast<double>(acc.lower()), static_cast<double>(acc.upper())));
        };
        double tl = 0.0, th = 0.0;
        auto x64   = make_vector<P32>(64,   0.0, 0.0, tl, th);
        auto x4096 = make_vector<P32>(4096, 0.0, 0.0, tl, th);

        const double a64   = sw::mp_blas::relative_width(naive_acc(x64));
        const double a4096 = sw::mp_blas::relative_width(naive_acc(x4096));
        const double q64   = sw::mp_blas::relative_width(sw::mp_blas::nrm2_exact<ID, P32>(x64));
        const double q4096 = sw::mp_blas::relative_width(sw::mp_blas::nrm2_exact<ID, P32>(x4096));

        check(a4096 > 10.0 * a64,
              "accumulation: naive sum-of-squares width did not grow with n");
        check(q4096 < 4.0 * q64,
              "accumulation: quire sum-of-squares width is not flat in n");
        check(a4096 / q4096 > 1e6,
              "accumulation: quire is not decisively tighter at n=4096");

        std::cout << "  accumulation (degenerate inputs, no straddling)  n=64 -> 4096:  naive "
                  << std::scientific << std::setprecision(2) << a64 << " -> " << a4096
                  << ",  quire " << q64 << " -> " << q4096 << '\n' << std::defaultfloat;
    }

    if (failures == 0) std::cout << "test_interval_nrm2 passed\n";
    return failures == 0 ? 0 : 1;
}
