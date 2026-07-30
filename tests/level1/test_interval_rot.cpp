// mp-blas level-1 test: interval plane rotation and wrapping (issue #12, Phase 3).
//
// Phases 1 and 2 established that exact accumulation collapses the enclosure
// width of a reduction, and that the win came from accumulation rather than
// multiplication. This test pins where that STOPS being true.
//
// The claim being pinned is a NEGATIVE one, and it matters as much as the
// positive results: it bounds what the exact dot product can be claimed to do.
// For a plane rotation the dominant error is WRAPPING -- the axis-aligned hull of
// a rotated box -- which is a loss in the REPRESENTATION, not the arithmetic. No
// accumulator can remove it. (The reduction is also only 2 terms long, so there
// is almost nothing for an exact accumulator to remove even in principle.)
//
// So the assertions are deliberately shaped to FAIL if someone later claims the
// quire fixes wrapping:
//   * both strategies must grow geometrically,
//   * their growth RATES must agree,
//   * the rate must match the geometric prediction |c| + |s|,
//   * the quire's benefit must be a bounded CONSTANT factor.
//
// Returns non-zero on failure (no external framework).
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>

#include <sw/mp_blas/interval_rot.hpp>
#include <sw/mp_blas/interval_containment.hpp>

#include <universal/number/posit/posit.hpp>
#include <universal/number/cfloat/cfloat.hpp>
#include <universal/number/interval/interval.hpp>

namespace {

using sw::mp_blas::exact_ref;

constexpr double kPi = 3.14159265358979323846;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) { std::cerr << "FAIL: " << what << '\n'; ++failures; }
}

struct Growth {
    double w_first    = 0.0;   ///< width after 1 rotation (the rounding seed)
    double w_last     = 0.0;   ///< width after k rotations
    double rate       = 0.0;   ///< measured per-step growth, seed removed
    bool   contained  = true;
};

/// Apply k rotations and measure the geometric growth of the enclosure width.
/// `use_quire` selects the accumulation, holding the kernel otherwise identical
/// so any difference is attributable to accumulation alone.
template <typename Scalar>
Growth sweep(double deg, std::size_t k, bool use_quire) {
    using I = sw::universal::interval<Scalar>;

    const double th = deg * kPi / 180.0;
    const Scalar c(std::cos(th)), s(std::sin(th));
    const double cd = static_cast<double>(c), sd = static_cast<double>(s);

    I x(Scalar(1)), y(Scalar(0));
    double rx = 1.0, ry = 0.0;
    Growth g;

    for (std::size_t i = 1; i <= k; ++i) {
        if (use_quire) sw::mp_blas::rot_quire(I(c), I(s), x, y);
        else           sw::mp_blas::rot_naive(I(c), I(s), x, y);

        const double t = cd * rx + sd * ry;   // same quantized rotation, in double
        ry = -sd * rx + cd * ry;
        rx = t;

        g.contained = g.contained && sw::mp_blas::encloses(x, exact_ref::from(rx));
        if (i == 1) g.w_first = sw::mp_blas::absolute_width(x);
        if (i == k) g.w_last  = sw::mp_blas::absolute_width(x);
    }
    // Rate measured from k=1 so the rounding seed cancels out of the ratio.
    g.rate = (k > 1 && g.w_first > 0.0)
                 ? std::pow(g.w_last / g.w_first, 1.0 / static_cast<double>(k - 1))
                 : 0.0;
    return g;
}

/// Run the full negative-result battery for one element type and angle.
template <typename Scalar>
void wrapping_case(const std::string& name, double deg) {
    const double th = deg * kPi / 180.0;
    const Scalar c(std::cos(th)), s(std::sin(th));
    const double wrap = sw::mp_blas::wrapping_factor(static_cast<double>(c),
                                                     static_cast<double>(s));
    const std::size_t k = 64;
    const std::string tag = name + "/" + std::to_string(static_cast<int>(deg)) + "deg";

    const Growth nv = sweep<Scalar>(deg, k, false);
    const Growth qr = sweep<Scalar>(deg, k, true);

    check(nv.contained, tag + ": naive rot lost containment");
    check(qr.contained, tag + ": quire rot lost containment");

    // 1. Both must actually grow geometrically -- otherwise there is no wrapping
    //    to reason about and the rest of the test is vacuous.
    check(nv.w_last > 100.0 * nv.w_first,
          tag + ": naive width did not grow geometrically (" +
              std::to_string(nv.w_first) + " -> " + std::to_string(nv.w_last) + ")");
    check(qr.w_last > 100.0 * qr.w_first,
          tag + ": QUIRE width did not grow geometrically -- if exact accumulation "
                "now suppresses wrapping, Phase 3's negative result is wrong and "
                "docs/interval-blas-study.md needs revisiting");

    // 2. THE NEGATIVE RESULT: the growth RATES must agree. An exact accumulator
    //    changes the constant, not the rate, because wrapping is representational.
    const double rate_ratio = nv.rate / qr.rate;
    check(std::abs(rate_ratio - 1.0) < 0.02,
          tag + ": naive and quire growth rates differ by more than 2% (naive " +
              std::to_string(nv.rate) + ", quire " + std::to_string(qr.rate) +
              ") -- accumulation is affecting the wrapping rate, contradicting Phase 3");

    // 3. The rate must match the GEOMETRIC prediction |c|+|s|, which is what
    //    identifies the mechanism as wrapping rather than accumulation. Measured
    //    from k=1 the estimate sits slightly above the asymptote because the early
    //    steps still carry the rounding seed, so allow a one-sided margin.
    check(nv.rate >= wrap * 0.98 && nv.rate <= wrap * 1.35,
          tag + ": measured growth rate " + std::to_string(nv.rate) +
              " does not match the wrapping prediction |c|+|s| = " + std::to_string(wrap));

    // 4. The quire's benefit is a bounded CONSTANT factor, not a rate change.
    const double ratio_first = nv.w_first / qr.w_first;
    const double ratio_last  = nv.w_last  / qr.w_last;
    check(ratio_first > 1.0 && ratio_first < 4.0,
          tag + ": unexpected quire benefit at k=1 (x" + std::to_string(ratio_first) + ")");
    check(ratio_last > 1.0 && ratio_last < 4.0,
          tag + ": quire benefit at k=64 (x" + std::to_string(ratio_last) +
              ") is not a bounded constant factor");

    std::cout << "  " << std::left << std::setw(20) << tag
              << " rate naive " << std::fixed << std::setprecision(4) << nv.rate
              << " vs quire " << qr.rate
              << "  (|c|+|s| = " << wrap << ")"
              << "   benefit x" << std::setprecision(2) << ratio_last << '\n'
              << std::defaultfloat;
}

} // namespace

int main() {
    using namespace sw::universal;
    using P32  = posit<32, 2>;
    using CF32 = cfloat<32, 8>;
    using I    = interval<P32>;

    std::cout << "issue #12 Phase 3 -- interval rot and the wrapping effect\n";

    // -- A single rotation multiplies width by |c|+|s|, by geometry ----------
    // Checked at the extremes: no wrapping at all on the axes, maximal at 45 deg.
    {
        for (double deg : {0.0, 30.0, 45.0, 90.0}) {
            const double th = deg * kPi / 180.0;
            const P32 c(std::cos(th)), s(std::sin(th));
            const double pred = sw::mp_blas::wrapping_factor(static_cast<double>(c),
                                                             static_cast<double>(s));
            const double half = 1e-6;
            I x(P32(1.0 - half), P32(1.0 + half)), y(P32(-half), P32(half));
            const double w0 = sw::mp_blas::absolute_width(x);
            sw::mp_blas::rot_naive(I(c), I(s), x, y);
            const double meas = std::max(sw::mp_blas::absolute_width(x),
                                         sw::mp_blas::absolute_width(y)) / w0;
            check(std::abs(meas / pred - 1.0) < 0.05,
                  "single rotation at " + std::to_string(static_cast<int>(deg)) +
                      " deg: measured width growth " + std::to_string(meas) +
                      " does not match |c|+|s| = " + std::to_string(pred));
        }
        std::cout << "  single-rotation growth matches |c|+|s| at 0/30/45/90 deg\n";
    }

    // -- The negative result, across angles and element types ----------------
    wrapping_case<P32>("posit<32,2>", 5.0);
    wrapping_case<P32>("posit<32,2>", 45.0);
    wrapping_case<CF32>("cfloat<32,8>", 45.0);

    // -- Rotation on the axes does NOT wrap ----------------------------------
    // |c|+|s| = 1 at 0 and 90 degrees, so there is no wrapping and the width can
    // only accumulate the per-step rounding: growth LINEAR in k (a few ulp per
    // step), not geometric. This is what shows the blowup elsewhere is
    // angle-dependent geometry rather than a generic "intervals always explode".
    //
    // The comparison is on ABSOLUTE final widths, not on growth ratios. At 90 deg
    // the first step is exact (multiply by 1 and by a near-zero cosine), so the
    // seed width is ~1e-20 and any ratio against it is meaningless. What is
    // meaningful is where the two land after the same number of steps with the
    // same arithmetic: microscopic vs enormous, decided purely by the angle.
    {
        const std::size_t k = 64;
        const Growth axis = sweep<P32>(90.0, k, false);
        const Growth tilt = sweep<P32>(45.0, k, false);
        check(axis.contained, "90 deg: lost containment");

        check(axis.w_last < 1e-3,
              "90 deg: final width " + std::to_string(axis.w_last) +
                  " is not bounded, though |c|+|s| = 1 predicts rounding-only growth");
        check(tilt.w_last > 1.0,
              "45 deg: final width " + std::to_string(tilt.w_last) +
                  " did not blow up, so there is no wrapping effect to attribute");
        check(tilt.w_last / axis.w_last > 1e6,
              "45 deg and 90 deg final widths are not separated, so the blowup is not "
              "attributable to the ANGLE (45 deg " + std::to_string(tilt.w_last) +
                  " vs 90 deg " + std::to_string(axis.w_last) + ")");

        std::cout << "  same arithmetic, same 64 steps, different angle:\n"
                  << "    90 deg (|c|+|s| = 1.000): final width "
                  << std::scientific << std::setprecision(1) << axis.w_last
                  << "  -- bounded, rounding only\n"
                  << "    45 deg (|c|+|s| = 1.414): final width " << tilt.w_last
                  << "  -- geometric, wrapping\n" << std::defaultfloat;
    }

    if (failures == 0) std::cout << "test_interval_rot passed\n";
    return failures == 0 ? 0 : 1;
}
