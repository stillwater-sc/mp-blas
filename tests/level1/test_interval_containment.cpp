// mp-blas -- issue #12 Phase 0: the containment gate for interval BLAS.
//
// No interval WIDTH measurement in Phases 1-5 means anything unless containment
// holds first, because a type that violates containment produces intervals that
// are too NARROW to be correct -- which reads as impressive tightness. Universal's
// interval did exactly that before stillwater-sc/universal#1234: `interval(0.1) *
// interval(0.1)` returned a zero-width interval around a value the exact product
// does not equal. This test is the standing guard against any regression of that
// kind, and it must stay green for the rest of issue #12 to be meaningful.
//
// What is checked:
//   1. The original counterexample -- interval(0.1)*interval(0.1) encloses the
//      exact product, and does so tightly (Stage-2 EFT, universal#1248).
//   2. Scalar-op containment: + - * / over deliberately NON-DYADIC operands, for
//      every element type mp-blas cares about. Dyadic operands are useless here:
//      their results are exact, so no rounding occurs and the check is vacuous
//      (the same blind spot that let universal#1202 survive in test_dot_quire).
//   3. Reduction containment: mtl::dot over DEGENERATE intervals must enclose the
//      Dot2 reference, across the #9 regimes including the ill-conditioned ones.
//      Degenerate inputs contribute no uncertainty, so all width is manufactured
//      by the operator -- the cleanest possible gate.
//   4. Reduction containment with WIDTH: intervals with real width must enclose
//      the dot of every sampled interior point combination.
//   5. Tightness regression guard: a "fix" that returns [-inf, inf] would satisfy
//      containment trivially. Enclosures must also stay useful.
//   6. Sanity: overestimation R >= 1 always. R < 1 is arithmetically impossible
//      for a sound implementation and means containment was lost.
//
// Returns non-zero on failure (no external framework).
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <sw/mp_blas/dot_characterization.hpp>
#include <sw/mp_blas/interval_containment.hpp>

#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/dot.hpp>

#include <universal/number/posit/posit.hpp>
#include <universal/number/cfloat/cfloat.hpp>
#include <universal/number/interval/interval.hpp>

namespace {

using sw::mp_blas::reference_t;
using sw::mp_blas::regime;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) { std::cerr << "FAIL: " << what << '\n'; ++failures; }
}

/// Deliberately non-dyadic operand pairs: each has an inexact binary expansion,
/// so products and quotients genuinely round and the containment check bites.
const double kOperands[][2] = {
    { 0.1,   0.1   },
    { 0.1,   0.3   },
    { 1.0/3, 3.0   },
    { 0.7,   1.0/7 },
    { 1e-5,  1.0/3 },
    { 123.456, 0.001 },
    { -0.1,  0.3   },
    { 2.0/3, -0.7  },
};

/// Containment for the four arithmetic operators on degenerate intervals.
/// Degenerate inputs mean the exact result is a point, so FTIA reduces to
/// "the returned interval contains that point".
template <typename Scalar>
void scalar_ops(const std::string& name) {
    using I = sw::universal::interval<Scalar>;

    for (const auto& pr : kOperands) {
        // Quantize FIRST, then take the reference from the quantized values --
        // otherwise the check would be testing Scalar's conversion error rather
        // than the interval arithmetic.
        const Scalar sa(pr[0]), sb(pr[1]);
        const auto ra = static_cast<reference_t>(static_cast<double>(sa));
        const auto rb = static_cast<reference_t>(static_cast<double>(sb));
        if (ra == 0.0L || rb == 0.0L) continue;

        const I a(sa), b(sb);
        const std::string tag = name + " (" + std::to_string(pr[0]) + ", " + std::to_string(pr[1]) + ")";

        struct Op { const char* sym; I result; reference_t truth; };
        const Op ops[] = {
            { "+", a + b, ra + rb },
            { "-", a - b, ra - rb },
            { "*", a * b, ra * rb },
            { "/", a / b, ra / rb },
        };

        for (const auto& op : ops) {
            check(sw::mp_blas::encloses(op.result, op.truth),
                  tag + " op " + op.sym + ": enclosure [" +
                      std::to_string(static_cast<double>(op.result.lower())) + ", " +
                      std::to_string(static_cast<double>(op.result.upper())) +
                      "] does not contain " + std::to_string(static_cast<double>(op.truth)));

            // R < 1 is impossible for a sound implementation -- it would mean the
            // interval is narrower than the tightest correct enclosure. Skipped
            // when the exact result overflows Scalar's range (e.g. 123.456/0.001
            // in cfloat<16,5>): the narrowest enclosure is then [maxpos, +inf],
            // of infinite width, so the ratio is undefined. Containment above
            // still applies and is still checked.
            if (!sw::mp_blas::in_range<Scalar>(op.truth)) continue;
            const double R = sw::mp_blas::overestimation<I, Scalar>(op.result, op.truth);
            check(R >= 1.0 - 1e-9,
                  tag + " op " + op.sym + ": overestimation R=" + std::to_string(R) +
                      " < 1, so the enclosure is narrower than optimal (containment lost)");
        }
    }
}

/// Containment of mtl::dot over degenerate intervals, against the Dot2 reference.
/// Reported so a regression shows its magnitude, not just a boolean.
template <typename Scalar>
void reduction_degenerate(const std::string& name, regime r, double param, const char* label) {
    using I = sw::universal::interval<Scalar>;
    constexpr std::size_t n = 512;

    auto [xd, yd] = sw::mp_blas::generate(r, n, param);

    // Quantize to Scalar and read back, so the reference matches what the kernel
    // actually sees and only the interval arithmetic is under test.
    mtl::vec::dense_vector<I> a(n, I(Scalar(0))), b(n, I(Scalar(0)));
    for (std::size_t i = 0; i < n; ++i) {
        const Scalar sx(xd[i]), sy(yd[i]);
        xd[i] = static_cast<double>(sx);
        yd[i] = static_cast<double>(sy);
        a[i] = I(sx);
        b[i] = I(sy);
    }
    const auto truth = static_cast<reference_t>(sw::mp_blas::dot2(xd, yd));

    const I d = mtl::dot(a, b);
    const std::string tag = name + "/" + label;

    check(sw::mp_blas::encloses(d, truth),
          tag + ": dot enclosure [" + std::to_string(static_cast<double>(d.lower())) + ", " +
              std::to_string(static_cast<double>(d.upper())) + "] does not contain Dot2 reference " +
              std::to_string(static_cast<double>(truth)));

    // Tightness guard: containment alone is satisfiable by [-inf, inf], so the
    // enclosure must also stay useful. The bound has to be TYPE-AWARE -- naive
    // interval accumulation rounds outward ~1 ulp per term, so the expected
    // scale on a benign regime is n*u, which spans eight orders of magnitude
    // between double and cfloat<16,5>. A fixed threshold would just be a
    // disguised assertion about `double`.
    if (r == regime::uniform) {
        const double u = 0.5 * static_cast<double>(std::numeric_limits<Scalar>::epsilon());
        const double expected = static_cast<double>(n) * u;
        const double rw = sw::mp_blas::relative_width(d);
        check(std::isfinite(rw) && rw < 100.0 * expected,
              tag + ": relative width " + std::to_string(rw) + " exceeds 100x the expected n*u = " +
                  std::to_string(expected) + " (enclosure is far wider than outward rounding explains)");
    }

    std::cout << "  " << std::left << std::setw(26) << tag
              << " rel.width=" << std::scientific << std::setprecision(2) << sw::mp_blas::relative_width(d)
              << "  digits=" << std::fixed << std::setprecision(1) << sw::mp_blas::certified_digits(d)
              << std::defaultfloat << '\n';
}

/// Containment when the inputs have REAL width: the enclosure must contain the
/// dot of every sampled interior point combination, not merely the midpoint dot.
template <typename Scalar>
void reduction_with_width(const std::string& name) {
    using I = sw::universal::interval<Scalar>;
    constexpr std::size_t n = 128;

    auto [xd, yd] = sw::mp_blas::generate(regime::uniform, n);

    // Inflate each element into a genuine interval of half-width ~1e-3.
    mtl::vec::dense_vector<I> a(n, I(Scalar(0))), b(n, I(Scalar(0)));
    std::vector<double> xlo(n), xhi(n), ylo(n), yhi(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double dx = 1e-3 * (1.0 + 0.5 * std::abs(xd[i]));
        const double dy = 1e-3 * (1.0 + 0.5 * std::abs(yd[i]));
        const Scalar xl(xd[i] - dx), xh(xd[i] + dx), yl(yd[i] - dy), yh(yd[i] + dy);
        xlo[i] = static_cast<double>(xl); xhi[i] = static_cast<double>(xh);
        ylo[i] = static_cast<double>(yl); yhi[i] = static_cast<double>(yh);
        a[i] = I(xl, xh);
        b[i] = I(yl, yh);
    }
    const I d = mtl::dot(a, b);

    // Sample interior points deterministically, including all-lower and all-upper.
    for (std::size_t k = 0; k < 12; ++k) {
        std::vector<double> px(n), py(n);
        for (std::size_t i = 0; i < n; ++i) {
            const double t = (k == 0) ? 0.0 : (k == 1) ? 1.0
                                            : 0.5 * (1.0 + sw::mp_blas::synth(i, 100 + k));
            px[i] = xlo[i] + t * (xhi[i] - xlo[i]);
            py[i] = ylo[i] + (1.0 - t) * (yhi[i] - ylo[i]);
        }
        const auto truth = static_cast<reference_t>(sw::mp_blas::dot2(px, py));
        check(sw::mp_blas::encloses(d, truth),
              name + "/width: dot enclosure does not contain sampled interior dot (sample " +
                  std::to_string(k) + ", value " + std::to_string(static_cast<double>(truth)) + ")");
    }
}

} // namespace

int main() {
    using namespace sw::universal;
    using CF16subn = cfloat<16, 5, std::uint8_t, true, false, false>;

    std::cout << "issue #12 Phase 0 -- interval containment gate\n";

    // -- 1. The original counterexample from universal#1234 -----------------
    {
        using I = interval<double>;
        const double a = 0.1;
        const I p = I(a) * I(a);
        const auto truth = static_cast<reference_t>(a) * static_cast<reference_t>(a);

        check(sw::mp_blas::encloses(p, truth),
              "universal#1234 regression: interval(0.1)*interval(0.1) does not enclose the exact product");
        check(p.width() > 0.0,
              "universal#1234 regression: interval(0.1)*interval(0.1) has zero width, so it cannot "
              "enclose an inexact product");

        // Stage-2 EFT (universal#1248) should keep this at ~1 ulp, not blow it up.
        const double R = sw::mp_blas::overestimation<I, double>(p, truth);
        check(R >= 1.0 - 1e-9, "0.1*0.1: overestimation R < 1 (containment lost)");
        check(R <= 4.0, "0.1*0.1: overestimation R=" + std::to_string(R) +
                            " -- enclosure much wider than optimal, Stage-2 tightness regressed");
        std::cout << "  0.1*0.1: width=" << std::scientific << std::setprecision(2) << p.width()
                  << "  R=" << std::fixed << std::setprecision(2) << R << std::defaultfloat << '\n';
    }

    // -- 1b. NEGATIVE CONTROL: can this harness actually fail? --------------
    // A gate that cannot fail is worthless -- exactly the flaw that let
    // universal#1202 and #1234 survive Universal's own test suite. Hand-build the
    // too-narrow interval the pre-#1234 code returned and confirm the harness
    // rejects it, then an over-wide one and confirm containment passes while R
    // flags it. Both failure directions must be detectable.
    {
        using I = interval<double>;
        const double a = 0.1;
        const auto truth = static_cast<reference_t>(a) * static_cast<reference_t>(a);

        const I unsound(a * a, a * a);   // pre-#1234: round-to-nearest, zero width
        check(!sw::mp_blas::encloses(unsound, truth),
              "negative control: encloses() accepted a zero-width interval around an inexact "
              "product -- the containment gate is blind and every later phase is unguarded");
        check(sw::mp_blas::overestimation<I, double>(unsound, truth) < 1.0,
              "negative control: overestimation() did not report R < 1 for an unsound enclosure");

        const I overwide(a * a - 1e-3, a * a + 1e-3);
        check(sw::mp_blas::encloses(overwide, truth),
              "negative control: encloses() rejected a valid (if wide) enclosure");
        check(sw::mp_blas::overestimation<I, double>(overwide, truth) > 1e6,
              "negative control: overestimation() did not flag a grossly over-wide enclosure");
    }

    // -- 2. Scalar-op containment on non-dyadic operands --------------------
    scalar_ops<double>("double");
    scalar_ops<posit<32, 2>>("posit<32,2>");
    scalar_ops<posit<16, 2>>("posit<16,2>");
    scalar_ops<cfloat<32, 8>>("cfloat<32,8>");
    scalar_ops<CF16subn>("cfloat<16,5>+subn");

    // -- 3. Reduction containment over degenerate intervals -----------------
    // Includes the ill-conditioned regimes: cancellation is where an unsound
    // implementation is most likely to lose the true value.
    std::cout << "reduction over degenerate intervals (all width is manufactured by the operator):\n";
    reduction_degenerate<double>("double", regime::uniform, 0.0, "uniform");
    reduction_degenerate<double>("double", regime::graded, 40.0, "graded");
    reduction_degenerate<double>("double", regime::cancel, 1e-6, "cancel 1e-6");
    reduction_degenerate<double>("double", regime::kahan, 1e-6, "kahan 1e-6");
    reduction_degenerate<posit<32, 2>>("posit<32,2>", regime::uniform, 0.0, "uniform");
    reduction_degenerate<posit<32, 2>>("posit<32,2>", regime::cancel, 1e-6, "cancel 1e-6");
    reduction_degenerate<cfloat<32, 8>>("cfloat<32,8>", regime::uniform, 0.0, "uniform");
    reduction_degenerate<cfloat<32, 8>>("cfloat<32,8>", regime::cancel, 1e-6, "cancel 1e-6");

    // -- 4. Reduction containment with real input width ---------------------
    reduction_with_width<double>("double");
    reduction_with_width<posit<32, 2>>("posit<32,2>");

    // -- 5. Unbounded-result capability, per element type -------------------
    // Reachable only through division, so this restricts Phase 5 (solve) and
    // nothing in dot / nrm2 / ger / gemv / gemm / rot. Recorded, and asserted only
    // where an infinity genuinely exists.
    {
        std::cout << "unbounded-result capability (needed for Phase 5 solve only):\n";
        auto report = [](auto tag, const char* nm, bool expect) {
            using S = decltype(tag);
            const bool got = sw::mp_blas::can_represent_unbounded<S>();
            std::cout << "  " << std::left << std::setw(20) << nm
                      << (got ? "can signal unbounded" : "CANNOT -- saturates at max()") << '\n';
            if (expect) {
                check(got, std::string(nm) + ": expected to represent an unbounded interval but cannot");
            }
        };
        report(double{}, "double", true);
        report(cfloat<32, 8>{}, "cfloat<32,8>", true);
        // posit has NaR, not infinity, so interval<posit> returns [-maxpos, maxpos]
        // for a division whose true result set is unbounded. Not asserted -- this is
        // a documented restriction on Phase 5, tracked in issue #12 Phase 0.
        report(posit<32, 2>{}, "posit<32,2>", false);
        check(!sw::mp_blas::can_represent_unbounded<posit<32, 2>>(),
              "posit unexpectedly CAN represent unbounded intervals -- issue #12 Phase 0's "
              "restriction on solve is stale and should be removed");
    }

    if (failures == 0) std::cout << "test_interval_containment passed\n";
    return failures == 0 ? 0 : 1;
}
