// mp-blas -- interval plane rotation: the wrapping effect (issue #12, Phase 3).
//
// Phases 1 and 2 established that exact accumulation collapses the enclosure
// width of a reduction, and that the win came from accumulation rather than from
// multiplication. This phase looks for where that STOPS being true.
//
// `rot` is the natural candidate on two independent grounds (see
// include/sw/mp_blas/interval_rot.hpp):
//   1. the reduction is length TWO, so there is almost no accumulation error for
//      an exact accumulator to remove;
//   2. the dominant error is WRAPPING -- forcing a rotated (tilted) box back into
//      an axis-aligned interval pair -- which is a loss in the REPRESENTATION,
//      not in the arithmetic, and compounds geometrically as (|c|+|s|)^k.
//
// The prediction is therefore a negative result: an exact accumulator should
// change the constant but NOT the growth rate. Confirming a predicted negative is
// what bounds the claim made in Phases 1-2, so it is worth measuring carefully
// rather than asserting.
//
// Yardstick. The reference is the same quantized rotation iterated in `double`
// from the same start. Over k <= 256 steps its own error stays ~1e-13 relative,
// while the interval widths under study reach 1e+2 -- so it is a valid witness by
// many orders of magnitude. (A rotation preserves the norm exactly in real
// arithmetic, so essentially all measured width growth is artifact; the quantized
// c, s do not satisfy c^2 + s^2 = 1 exactly, which is why the iterated reference
// is used rather than the analytic cos/sin of the total angle.)
//
// Usage: interval_rot_study [kmax]     (default kmax = 64)
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <sw/mp_blas/interval_rot.hpp>
#include <sw/mp_blas/interval_containment.hpp>

#include <universal/number/posit/posit.hpp>
#include <universal/number/cfloat/cfloat.hpp>
#include <universal/number/interval/interval.hpp>

namespace {

using sw::mp_blas::exact_ref;

constexpr double kPi = 3.14159265358979323846;

/// Iterate the same quantized rotation in double: the witness the enclosures
/// must contain.
void reference_rot(double c, double s, double& x, double& y) {
    const double t = c * x + s * y;
    y = -s * x + c * y;
    x = t;
}

// ---------------------------------------------------------------------------
// 1. Single rotation: tightness vs angle
// ---------------------------------------------------------------------------

/// One rotation applied to a box of known width, to isolate the per-step
/// wrapping multiplier from any compounding.
template <typename Scalar>
void angle_sweep(const std::string& name) {
    using I = sw::universal::interval<Scalar>;

    std::cout << "\n=== single rotation: width growth vs angle, interval<" << name << "> ===\n"
              << "  a box of width w becomes w*(|c|+|s|) -- 1 at multiples of 90 deg,\n"
              << "  sqrt(2) at 45 deg. This is geometry, not arithmetic.\n"
              << std::right << std::setw(9) << "theta" << std::setw(12) << "|c|+|s|"
              << std::setw(13) << "measured" << std::setw(12) << "pred/meas" << '\n';

    for (double deg : {0.0, 5.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0}) {
        const double th = deg * kPi / 180.0;
        const Scalar c(std::cos(th)), s(std::sin(th));
        const double pred = sw::mp_blas::wrapping_factor(static_cast<double>(c),
                                                         static_cast<double>(s));

        // A box wide enough that wrapping dominates the per-step rounding.
        const double half = 1e-6;
        I x(Scalar(1.0 - half), Scalar(1.0 + half));
        I y(Scalar(-half), Scalar(half));
        const double w0 = sw::mp_blas::absolute_width(x);

        sw::mp_blas::rot_naive(I(c), I(s), x, y);
        const double w1 = std::max(sw::mp_blas::absolute_width(x),
                                   sw::mp_blas::absolute_width(y));
        const double meas = w1 / w0;

        std::cout << std::right << std::setw(9) << std::fixed << std::setprecision(0) << deg
                  << std::setw(12) << std::setprecision(4) << pred
                  << std::setw(13) << meas
                  << std::setw(12) << std::setprecision(3) << pred / meas
                  << std::defaultfloat << '\n';
    }
}

// ---------------------------------------------------------------------------
// 2. Repeated rotation: does an exact accumulator change the rate?
// ---------------------------------------------------------------------------

template <typename Scalar>
void repeat_sweep(const std::string& name, double deg, std::size_t kmax) {
    using I = sw::universal::interval<Scalar>;

    const double th = deg * kPi / 180.0;
    const Scalar c(std::cos(th)), s(std::sin(th));
    const double cd = static_cast<double>(c), sd = static_cast<double>(s);
    const double wrap = sw::mp_blas::wrapping_factor(cd, sd);

    std::cout << "\n=== repeated rotation, interval<" << name << ">, theta = "
              << std::fixed << std::setprecision(0) << deg << " deg ===\n"
              << std::defaultfloat
              << "  wrapping factor |c|+|s| = " << std::fixed << std::setprecision(4) << wrap
              << "  =>  predicted width ~ (" << wrap << ")^k\n" << std::defaultfloat
              << "  starting from a POINT, so k=1's width is pure rounding; wrapping then\n"
              << "  amplifies whatever seed the arithmetic leaves.\n"
              << std::right << std::setw(6) << "k"
              << std::setw(12) << "naive w" << std::setw(12) << "quire w"
              << std::setw(11) << "nv/qr" << std::setw(13) << "naive rate"
              << std::setw(13) << "quire rate" << std::setw(11) << "predicted" << '\n';

    I xn(Scalar(1)), yn(Scalar(0));
    I xq(Scalar(1)), yq(Scalar(0));
    double rx = 1.0, ry = 0.0;           // double reference
    double w1n = 0.0, w1q = 0.0;
    bool contained = true;

    for (std::size_t k = 1; k <= kmax; ++k) {
        sw::mp_blas::rot_naive(I(c), I(s), xn, yn);
        sw::mp_blas::rot_quire(I(c), I(s), xq, yq);
        reference_rot(cd, sd, rx, ry);

        const auto tx = exact_ref::from(rx);
        contained = contained && sw::mp_blas::encloses(xn, tx) && sw::mp_blas::encloses(xq, tx);

        const double wn = sw::mp_blas::absolute_width(xn);
        const double wq = sw::mp_blas::absolute_width(xq);
        if (k == 1) { w1n = wn; w1q = wq; }

        const bool report = (k == 1 || k == 8 || k == 16 || k == 32 || k == kmax);
        if (report) {
            // Per-step growth rate, measured from k=1 so the seed cancels.
            const double e = (k > 1) ? 1.0 / static_cast<double>(k - 1) : 0.0;
            const double rate_n = (k > 1) ? std::pow(wn / w1n, e) : 0.0;
            const double rate_q = (k > 1) ? std::pow(wq / w1q, e) : 0.0;
            std::cout << std::right << std::setw(6) << k
                      << std::scientific << std::setprecision(2)
                      << std::setw(12) << wn << std::setw(12) << wq
                      << std::fixed << std::setprecision(2) << std::setw(11) << wn / wq;
            if (k > 1)
                std::cout << std::setprecision(4) << std::setw(13) << rate_n
                          << std::setw(13) << rate_q << std::setw(11) << wrap;
            else
                std::cout << std::setw(13) << "-" << std::setw(13) << "-" << std::setw(11) << "-";
            std::cout << std::defaultfloat << '\n';
        }
    }
    std::cout << "  containment held throughout: " << (contained ? "yes" : "*** NO ***") << '\n';
}

} // namespace

int main(int argc, char* argv[]) {
    std::size_t kmax = 64;
    if (argc > 1) kmax = static_cast<std::size_t>(std::atoll(argv[1]));

    std::cout << "Interval plane rotation: the wrapping effect (issue #12, Phase 3)\n"
              << "Looking for where exact accumulation STOPS helping. A rotation maps an\n"
              << "axis-aligned box to a tilted one, and storing it as an interval pair takes\n"
              << "the axis-aligned hull -- a loss in the REPRESENTATION, not the arithmetic.\n"
              << "Prediction: an exact accumulator changes the constant, not the rate.\n";

    angle_sweep<sw::universal::posit<32, 2>>("posit<32,2>");

    repeat_sweep<sw::universal::posit<32, 2>>("posit<32,2>", 5.0,  kmax);
    repeat_sweep<sw::universal::posit<32, 2>>("posit<32,2>", 45.0, kmax);
    repeat_sweep<sw::universal::cfloat<32, 8>>("cfloat<32,8>", 45.0, kmax);

    return 0;
}
