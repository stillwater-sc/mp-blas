// mp-blas level-1 test: the interval quire accumulator (issue #12, Phase 1).
//
// Pins the two hypotheses Phase 1 exists to test. Phase 0's containment gate
// (tests/level1/test_interval_containment.cpp) must be green for any of this to
// mean anything -- these are TIGHTNESS claims on top of correctness.
//
//   H1  naive interval accumulation rounds both endpoints outward once per term,
//       so its enclosure width GROWS with n and with the conditioning.
//   H2  exact accumulation in a quire pair, rounded outward exactly once on the
//       way out, has width set only by that final rounding -- so it is FLAT in n
//       and independent of the conditioning.
//
// H2 is the interesting one: it says the enclosure stops reporting the error
// BOUND and starts reporting the actual uncertainty. It is also the property
// Kulisch designed the exact scalar product for (see
// docs/dot-product-characterization.md section 1).
//
// Every check is also a containment check, because a tightness win obtained by
// losing containment is not a win.
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
#include <mtl/operation/dot.hpp>

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

struct Outcome {
    double naive_width = 0.0;
    double quire_width = 0.0;
    double cond        = 0.0;
    bool   naive_ok    = false;
    bool   quire_ok    = false;
};

/// Run one (element type, regime, n) instance through naive and quire
/// accumulation and report widths + containment.
template <typename Scalar>
Outcome run(regime r, double param, std::size_t n) {
    using I  = sw::universal::interval<Scalar>;
    using QA = sw::mp_blas::interval_quire<Scalar>;

    auto [xd, yd] = sw::mp_blas::generate(r, n, param);

    mtl::vec::dense_vector<I> a(n, I(Scalar(0))), b(n, I(Scalar(0)));
    for (std::size_t i = 0; i < n; ++i) {
        const Scalar sx(xd[i]), sy(yd[i]);
        xd[i] = static_cast<double>(sx);
        yd[i] = static_cast<double>(sy);
        a[i] = I(sx);
        b[i] = I(sy);
    }

    const double u = 0.5 * static_cast<double>(std::numeric_limits<Scalar>::epsilon());
    const auto features = sw::mp_blas::characterize(xd, yd, u);
    const auto truth = exact_ref::from(sw::mp_blas::dot2(xd, yd));

    const I naive = mtl::dot(a, b);
    const I quire = mtl::dot<QA, I>(a, b);

    Outcome o;
    o.cond        = features.cond;
    o.naive_width = sw::mp_blas::relative_width(naive);
    o.quire_width = sw::mp_blas::relative_width(quire);
    o.naive_ok    = sw::mp_blas::encloses(naive, truth);
    o.quire_ok    = sw::mp_blas::encloses(quire, truth);
    return o;
}

} // namespace

int main() {
    using namespace sw::universal;
    using P32 = posit<32, 2>;
    using CF32 = cfloat<32, 8>;

    std::cout << "issue #12 Phase 1 -- interval quire accumulator\n";

    // -- 0. Containment, everywhere, before any tightness claim --------------
    {
        struct C { regime r; double p; const char* l; };
        const C cases[] = {
            { regime::uniform,  0.0,  "uniform"     },
            { regime::positive, 0.0,  "positive"    },
            { regime::graded,   40.0, "graded"      },
            { regime::cancel,   1e-3, "cancel 1e-3" },
            { regime::cancel,   1e-9, "cancel 1e-9" },
            { regime::kahan,    1e-6, "kahan 1e-6"  },
        };
        for (const auto& c : cases) {
            for (std::size_t n : {64u, 512u, 4096u}) {
                const auto o = run<P32>(c.r, c.p, n);
                const std::string tag =
                    std::string("posit<32,2>/") + c.l + "/n=" + std::to_string(n);
                check(o.quire_ok, tag + ": interval QUIRE enclosure lost containment");
                check(o.naive_ok, tag + ": interval NAIVE enclosure lost containment");
                // A tighter enclosure that is also correct is the whole claim;
                // it must never be WIDER than naive.
                check(o.quire_width <= o.naive_width * (1.0 + 1e-9),
                      tag + ": quire enclosure (" + std::to_string(o.quire_width) +
                          ") is wider than naive (" + std::to_string(o.naive_width) + ")");
            }
        }
    }

    // -- 1. H1: naive width grows with n -------------------------------------
    // Measured on `uniform`, where cond is essentially fixed across n, so any
    // growth is the length axis alone.
    {
        const auto n64   = run<P32>(regime::uniform, 0.0, 64);
        const auto n4096 = run<P32>(regime::uniform, 0.0, 4096);
        check(n4096.naive_width > 4.0 * n64.naive_width,
              "H1: naive interval width did not grow with n (64: " +
                  std::to_string(n64.naive_width) + ", 4096: " +
                  std::to_string(n4096.naive_width) + ")");
        std::cout << "  H1 naive width  n=64 -> n=4096: " << std::scientific << std::setprecision(2)
                  << n64.naive_width << " -> " << n4096.naive_width
                  << "  (x" << std::fixed << std::setprecision(1)
                  << n4096.naive_width / n64.naive_width << ")\n" << std::defaultfloat;
    }

    // -- 2. H2: quire width is FLAT in n -------------------------------------
    // The exact accumulation introduces no rounding, so only the final outward
    // step contributes -- and that does not know how many terms there were.
    {
        const auto a = run<P32>(regime::cancel, 1e-6, 64);
        const auto b = run<P32>(regime::cancel, 1e-6, 512);
        const auto c = run<P32>(regime::cancel, 1e-6, 4096);
        const double lo = std::min({a.quire_width, b.quire_width, c.quire_width});
        const double hi = std::max({a.quire_width, b.quire_width, c.quire_width});
        check(hi <= 4.0 * lo,
              "H2: quire width is not flat in n (n=64: " + std::to_string(a.quire_width) +
                  ", n=512: " + std::to_string(b.quire_width) +
                  ", n=4096: " + std::to_string(c.quire_width) + ")");
        std::cout << "  H2 quire width  n=64/512/4096: " << std::scientific << std::setprecision(2)
                  << a.quire_width << " / " << b.quire_width << " / " << c.quire_width << '\n'
                  << std::defaultfloat;
    }

    // -- 3. H2: quire width is independent of conditioning -------------------
    // The decisive claim, and it needs the right element type to state cleanly.
    //
    // Measured on cfloat<32,8> (FIXED precision): relative ulp is
    // magnitude-independent, so a flat relative width is exactly H2.
    //
    // posit is measured too but held to a weaker claim on purpose. The cancel
    // ladder shrinks the surviving residual to raise cond, which also shrinks the
    // RESULT MAGNITUDE by the same factor -- and a tapered format's relative ulp
    // depends on magnitude. So a posit quire width that drifts across the ladder
    // is tapering, not conditioning, and asserting flatness there would be
    // asserting something false about the format. Same effect as the superlinear
    // posit growth in docs/dot-product-characterization.md section 4.3.
    {
        const auto f3 = run<CF32>(regime::cancel, 1e-3, 512);
        const auto f9 = run<CF32>(regime::cancel, 1e-9, 512);
        check(f9.cond > 100.0 * f3.cond,
              "the cancel ladder did not actually change cond -- test is not measuring what it claims");
        check(f9.naive_width > 100.0 * f3.naive_width,
              "H1: cfloat naive width did not degrade with cond (1e-3: " +
                  std::to_string(f3.naive_width) + ", 1e-9: " + std::to_string(f9.naive_width) + ")");
        const double fixed_ratio = f9.quire_width / f3.quire_width;
        check(fixed_ratio > 0.25 && fixed_ratio < 4.0,
              "H2: fixed-precision quire width moved with cond (cond " +
                  std::to_string(f3.cond) + " -> " + std::to_string(f9.cond) +
                  " changed width by x" + std::to_string(fixed_ratio) + ")");
        std::cout << "  H2 cfloat<32,8> cond " << std::scientific << std::setprecision(1)
                  << f3.cond << " -> " << f9.cond << ":  naive x" << std::fixed
                  << std::setprecision(0) << f9.naive_width / f3.naive_width
                  << ",  quire x" << std::setprecision(2) << fixed_ratio << " (flat)\n"
                  << std::defaultfloat;

        // posit: the quire must still beat naive by orders of magnitude, even
        // though tapering keeps it from being perfectly flat.
        const auto p3 = run<P32>(regime::cancel, 1e-3, 512);
        const auto p9 = run<P32>(regime::cancel, 1e-9, 512);
        const double posit_ratio = p9.quire_width / p3.quire_width;
        const double naive_ratio = p9.naive_width / p3.naive_width;
        check(posit_ratio < naive_ratio / 100.0,
              "H2: posit quire degraded comparably to naive across the cancel ladder "
              "(quire x" + std::to_string(posit_ratio) + " vs naive x" +
                  std::to_string(naive_ratio) + ")");
        std::cout << "  H2 posit<32,2>  same ladder:            naive x" << std::fixed
                  << std::setprecision(0) << naive_ratio << ",  quire x"
                  << std::setprecision(2) << posit_ratio << " (tapering, not cond)\n"
                  << std::defaultfloat;
    }

    // -- 4. The headline: a useful enclosure where naive delivers none -------
    // kahan at n=4096 is where naive interval arithmetic gives up entirely -- an
    // enclosure wider than its own midpoint certifies no digits at all.
    {
        const auto o = run<P32>(regime::kahan, 1e-6, 4096);
        check(o.naive_width > 1.0,
              "expected naive interval kahan/n=4096 to be useless (width > 1); got " +
                  std::to_string(o.naive_width));
        check(o.quire_width < 1e-4,
              "expected the quire to stay useful on kahan/n=4096; width " +
                  std::to_string(o.quire_width));
        check(o.quire_ok, "quire lost containment on kahan/n=4096");
        std::cout << "  kahan/n=4096:  naive width " << std::scientific << std::setprecision(2)
                  << o.naive_width << " (useless)  vs quire " << o.quire_width << '\n'
                  << std::defaultfloat;
    }

    // -- 5. Not posit-specific ----------------------------------------------
    {
        const auto o = run<CF32>(regime::cancel, 1e-6, 512);
        check(o.quire_ok, "cfloat<32,8>: quire lost containment");
        check(o.quire_width < o.naive_width,
              "cfloat<32,8>: quire not tighter than naive");
    }

    // -- 6. The straddle x straddle corner case ------------------------------
    // select_corners() resolves that case by comparing two products EXACTLY in a
    // scratch quire, because comparing rounded products could pick the wrong
    // corner and yield an enclosure that is too NARROW. Exercise it directly:
    // both operands straddle zero, with the two candidate products deliberately
    // near-equal.
    {
        using I  = interval<P32>;
        using QA = sw::mp_blas::interval_quire<P32>;
        const std::size_t n = 64;
        mtl::vec::dense_vector<I> a(n, I(P32(0))), b(n, I(P32(0)));
        std::vector<double> xlo(n), xhi(n), ylo(n), yhi(n);
        for (std::size_t i = 0; i < n; ++i) {
            // X = [-(1+d), 1], Y = [-1, 1+d]: a*d and b*c are both -(1+d), so the
            // lower-endpoint candidates tie, and a tiny perturbation decides.
            const double d = 1e-9 * static_cast<double>(i);
            const P32 xl(-(1.0 + d)), xh(1.0), yl(-1.0), yh(1.0 + d);
            xlo[i] = double(xl); xhi[i] = double(xh);
            ylo[i] = double(yl); yhi[i] = double(yh);
            a[i] = I(xl, xh);
            b[i] = I(yl, yh);
        }
        const I quire = mtl::dot<QA, I>(a, b);
        const I naive = mtl::dot(a, b);

        // Containment against every sampled interior combination.
        for (std::size_t k = 0; k < 8; ++k) {
            std::vector<double> px(n), py(n);
            for (std::size_t i = 0; i < n; ++i) {
                const double t = 0.5 * (1.0 + sw::mp_blas::synth(i, 500 + k));
                px[i] = xlo[i] + t * (xhi[i] - xlo[i]);
                py[i] = ylo[i] + (1.0 - t) * (yhi[i] - ylo[i]);
            }
            const auto truth = exact_ref::from(sw::mp_blas::dot2(px, py));
            check(sw::mp_blas::encloses(quire, truth),
                  "straddle x straddle: quire enclosure misses sampled interior dot (sample " +
                      std::to_string(k) + ")");
            check(sw::mp_blas::encloses(naive, truth),
                  "straddle x straddle: naive enclosure misses sampled interior dot (sample " +
                      std::to_string(k) + ")");
        }
        // ABSOLUTE width here, not relative: this sum brackets zero (the naive
        // midpoint is exactly 0, the quire's is -9.5e-07), and relative_width
        // divides by the midpoint. Both accumulators return [-64, 64] with
        // identical absolute width, but their relative widths differ by six
        // orders of magnitude purely from where the midpoint landed. See the
        // caveat on relative_width() in interval_containment.hpp.
        check(sw::mp_blas::absolute_width(quire) <=
                  sw::mp_blas::absolute_width(naive) * (1.0 + 1e-9),
              "straddle x straddle: quire absolute width " +
                  std::to_string(sw::mp_blas::absolute_width(quire)) +
                  " exceeds naive " + std::to_string(sw::mp_blas::absolute_width(naive)));
    }

    if (failures == 0) std::cout << "test_interval_quire passed\n";
    return failures == 0 ? 0 : 1;
}
