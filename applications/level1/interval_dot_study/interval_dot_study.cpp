// mp-blas -- interval dot product study (issue #12, Phase 1).
//
// Issue #9 measured how WRONG a dot product is. This measures how much it can
// PROVE. Every number below is a rigorous enclosure of the exact result (the
// containment gate, tests/level1/test_interval_containment, is what licenses
// reading them that way), so `digits` is not an error estimate -- it is the
// number of decimal digits the computation certifies.
//
// Three accumulation strategies over interval element vectors:
//
//   naive     mtl::dot(x, y)                       endpoints rounded outward per term
//   promoted  mtl::dot<interval<double>, ...>      same, but accumulated in double
//   quire     mtl::dot<interval_quire<T>, ...>     exact accumulation, ONE outward round
//
// The hypotheses under test (see include/mtl/math/interval_quire_accumulator.hpp):
//   H1  naive width grows with n and with cond -- it tracks the error BOUND.
//   H2  quire width is set only by the single final rounding, so it is flat in
//       both -- it tracks the actual uncertainty.
//
// Reported per instance, alongside `cond` and `n_eff` from issue #9 so tightness
// can be regressed against input structure rather than reported standalone:
//
//   rel.w   relative width w/|midpoint|      (see the zero-midpoint caveat below)
//   digits  -log10(rel.w)                    decimal digits certified; <= 0 = none
//   R       width / narrowest correct width   overestimation; 1 = optimal
//
// Usage: interval_dot_study [nmax]     (default nmax = 4096)
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
#include <mtl/operation/dot.hpp>

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
    { regime::positive, 0.0,   "positive"    },
    { regime::graded,   40.0,  "graded"      },
    { regime::cancel,   1e-3,  "cancel 1e-3" },
    { regime::cancel,   1e-6,  "cancel 1e-6" },
    { regime::cancel,   1e-9,  "cancel 1e-9" },
    { regime::kahan,    1e-6,  "kahan 1e-6"  },
};

/// Promote an interval<Scalar> vector to interval<double>, endpoint by endpoint.
///
/// This exists because Universal's `interval<Scalar>` has NO cross-Scalar
/// converting constructor: `interval<double>(interval<posit<32,2>>)` does not
/// compile, so MTL5's promoted-accumulator seam -- `dot<interval<double>, ...>`
/// over interval<posit> elements -- cannot be used directly. That is exactly the
/// idiom a mixed-precision interval BLAS wants (narrow interval elements, wider
/// interval accumulator), so it is worth having upstream; filed as a Phase 1
/// finding.
///
/// Going through the two-argument constructor is safe rather than a shortcut: it
/// routes each endpoint through enclose_lo/enclose_hi, which round outward when
/// the conversion is inexact. For posit<32,2> and cfloat<32,8> the conversion to
/// double is exact anyway, so no widening occurs here.
template <typename Scalar>
mtl::vec::dense_vector<sw::universal::interval<double>>
promote(const mtl::vec::dense_vector<sw::universal::interval<Scalar>>& v) {
    using ID = sw::universal::interval<double>;
    const std::size_t n = v.size();
    mtl::vec::dense_vector<ID> out(n, ID(0.0));
    for (std::size_t i = 0; i < n; ++i)
        out[i] = ID(static_cast<double>(v[i].lower()), static_cast<double>(v[i].upper()));
    return out;
}

/// One row: all three strategies on one (element type, regime, n).
template <typename Scalar>
void row(const Case& c, std::size_t n) {
    using I  = sw::universal::interval<Scalar>;
    using ID = sw::universal::interval<double>;
    using QA = sw::mp_blas::interval_quire<Scalar>;

    auto [xd, yd] = sw::mp_blas::generate(c.r, n, c.param);

    mtl::vec::dense_vector<I> a(n, I(Scalar(0))), b(n, I(Scalar(0)));
    for (std::size_t i = 0; i < n; ++i) {
        const Scalar sx(xd[i]), sy(yd[i]);
        xd[i] = static_cast<double>(sx);
        yd[i] = static_cast<double>(sy);
        a[i] = I(sx);
        b[i] = I(sy);
    }

    const double u = 0.5 * static_cast<double>(std::numeric_limits<Scalar>::epsilon());
    const auto f = sw::mp_blas::characterize(xd, yd, u);
    const auto truth = exact_ref::from(sw::mp_blas::dot2(xd, yd));

    // All three DELIVER interval<double>, following the convention in
    // docs/level1-accumulator-study.md: fix the element precision, vary only the
    // accumulator, and round every result out to the same width -- so the numbers
    // isolate ACCUMULATION and are not silently capped by the output type.
    // Delivering the quire to interval<Scalar> instead would limit it to the
    // element type's ulp (3.7e-09 for posit<32,2>) and understate it by orders of
    // magnitude, which is not a property of the accumulator at all.
    //
    // posit<32,2> -> double and cfloat<32,8> -> double are exact, so promoting
    // naive's endpoints for reporting introduces no widening of its own.
    const I  naive_native = mtl::dot(a, b);
    const ID naive    = ID(static_cast<double>(naive_native.lower()),
                           static_cast<double>(naive_native.upper()));
    const ID promoted = mtl::dot(promote(a), promote(b));
    const ID quire    = mtl::dot<QA, ID>(a, b);            // exact accumulate, deliver double

    // Containment is a hard invariant, not a measurement. Flag loudly rather
    // than quietly printing a width for an enclosure that does not enclose.
    const bool ok = sw::mp_blas::encloses(naive, truth)
                 && sw::mp_blas::encloses(promoted, truth)
                 && sw::mp_blas::encloses(quire, truth);

    std::cout << std::left << std::setw(13) << c.label << std::right << std::setw(7) << n
              << std::scientific << std::setprecision(1)
              << std::setw(10) << f.cond
              << std::setw(8)  << f.n_eff
              << std::setw(11) << sw::mp_blas::relative_width(naive)
              << std::setw(11) << sw::mp_blas::relative_width(promoted)
              << std::setw(11) << sw::mp_blas::relative_width(quire)
              << std::fixed << std::setprecision(1)
              << std::setw(9)  << sw::mp_blas::certified_digits(naive)
              << std::setw(9)  << sw::mp_blas::certified_digits(promoted)
              << std::setw(9)  << sw::mp_blas::certified_digits(quire)
              << std::scientific << std::setprecision(1)
              << std::setw(11) << sw::mp_blas::overestimation<ID, double>(quire, truth)
              << (ok ? "" : "   *** CONTAINMENT LOST ***")
              << std::defaultfloat << '\n';
}

template <typename Scalar>
void study(const std::string& name, std::size_t nmax) {
    const double u = 0.5 * static_cast<double>(std::numeric_limits<Scalar>::epsilon());
    std::cout << "\n=== element interval<" << name << ">  (u = "
              << std::scientific << std::setprecision(1) << u << ") ===\n" << std::defaultfloat
              << std::left << std::setw(13) << "regime" << std::right << std::setw(7) << "n"
              << std::setw(10) << "cond" << std::setw(8) << "n_eff"
              << std::setw(11) << "naive" << std::setw(11) << "promoted" << std::setw(11) << "quire"
              << std::setw(9)  << "dig(nv)" << std::setw(9) << "dig(pr)" << std::setw(9) << "dig(qr)"
              << std::setw(11) << "R(quire)" << '\n';
    for (const auto& c : kRegimes)
        for (std::size_t n = 64; n <= nmax; n *= 8) row<Scalar>(c, n);
}

/// H1 vs H2 head to head: hold the regime fixed, sweep n, and show that one
/// strategy's width grows while the other's does not.
template <typename Scalar>
void length_scaling(const std::string& name, const Case& c, std::size_t nmax) {
    using I  = sw::universal::interval<Scalar>;
    using QA = sw::mp_blas::interval_quire<Scalar>;

    std::cout << "\n=== length scaling, interval<" << name << ">, regime " << c.label
              << " ===\n"
              << "  H1 predicts the naive column grows with n; H2 predicts the quire column does not.\n"
              << std::right << std::setw(9) << "n" << std::setw(12) << "naive w"
              << std::setw(12) << "quire w" << std::setw(11) << "naive/n0" << std::setw(11) << "quire/n0"
              << std::setw(12) << "ratio n/q" << '\n';

    double w0_naive = 0.0, w0_quire = 0.0;
    for (std::size_t n = 64; n <= nmax; n *= 2) {
        auto [xd, yd] = sw::mp_blas::generate(c.r, n, c.param);
        mtl::vec::dense_vector<I> a(n, I(Scalar(0))), b(n, I(Scalar(0)));
        for (std::size_t i = 0; i < n; ++i) {
            const Scalar sx(xd[i]), sy(yd[i]);
            a[i] = I(sx); b[i] = I(sy);
        }
        using ID2 = sw::universal::interval<double>;
        const auto nn = mtl::dot(a, b);
        const double wn = sw::mp_blas::relative_width(
            ID2(static_cast<double>(nn.lower()), static_cast<double>(nn.upper())));
        const double wq = sw::mp_blas::relative_width(mtl::dot<QA, ID2>(a, b));
        if (w0_naive == 0.0) { w0_naive = wn; w0_quire = wq; }

        std::cout << std::right << std::setw(9) << n
                  << std::scientific << std::setprecision(2)
                  << std::setw(12) << wn << std::setw(12) << wq
                  << std::fixed << std::setprecision(1)
                  << std::setw(11) << wn / w0_naive
                  << std::setw(11) << wq / w0_quire
                  << std::scientific << std::setprecision(1)
                  << std::setw(12) << (wq > 0.0 ? wn / wq : 0.0)
                  << std::defaultfloat << '\n';
    }
}

} // namespace

int main(int argc, char* argv[]) {
    std::size_t nmax = 4096;
    if (argc > 1) nmax = static_cast<std::size_t>(std::atoll(argv[1]));

    std::cout << "Interval dot product study (issue #12, Phase 1)\n"
              << "Every width below is a RIGOROUS enclosure of the exact result, so `dig` is the\n"
              << "number of decimal digits the computation certifies -- not an error estimate.\n"
              << "  naive    = mtl::dot over interval elements   (outward rounding per term)\n"
              << "  promoted = accumulated in interval<double>\n"
              << "  quire    = exact accumulation in a quire pair, ONE outward rounding\n"
              << "  R        = quire width / narrowest correct width (1 = optimal)\n"
              << "  cond/n_eff are the issue #9 structural features of the same instance.\n"
              << "CAVEAT: rel.w divides by the midpoint, so it is unstable when the enclosure\n"
              << "brackets zero; `positive` and the cancel ladder keep midpoints away from 0.\n";

    study<sw::universal::posit<32, 2>>("posit<32,2>", nmax);
    study<sw::universal::cfloat<32, 8>>("cfloat<32,8>", nmax);

    // cfloat<32,8> is fixed-precision, so its relative ulp does not depend on
    // magnitude -- the cleanest place to see H2 as a flat line.
    length_scaling<sw::universal::cfloat<32, 8>>("cfloat<32,8>", kRegimes[4], nmax);   // cancel 1e-6
    length_scaling<sw::universal::posit<32, 2>>("posit<32,2>", kRegimes[0], nmax);     // uniform

    return 0;
}
