// mp-blas level-1 test: dot-product characterization (issue #9).
//
// Checks the invariants the characterization framework rests on, so the
// metrics in applications/level1/dot_characterization and the guidance in
// docs/dot-product-characterization.md cannot silently rot:
//
//   1. Dot2 reference is exact on a construction whose answer is known.
//   2. cond >= 2 always, and == 2 exactly for an all-positive reduction.
//   3. cond tracks the engineered cancellation knob (a decade of residual
//      shrinkage buys a decade of cond).
//   4. n_eff falls as the exponent spread D grows -- swamping is a distinct
//      axis from cancellation.
//   5. The Wilkinson/Higham bound holds: native relative error <= n*u*cond.
//   6. The quire is reproducible (zero spread over permuted summation orders)
//      where native accumulation is not -- Kulisch's guarantee, measured.
//   7. Accumulator quality is ordered: quire <= promoted <= native error.
//
// Returns non-zero on failure (no external framework).
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include <sw/mp_blas/dot_characterization.hpp>

#include <mtl/math/quire_accumulator.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/dot.hpp>

#include <universal/number/posit/posit.hpp>

namespace {

using sw::mp_blas::regime;
using sw::mp_blas::dot_features;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) { std::cerr << "FAIL: " << what << '\n'; ++failures; }
}

constexpr std::size_t kN = 2048;

/// Quantize to T and read back, so metrics describe what the kernel sees.
template <typename T>
void quantize(std::vector<double>& x, std::vector<double>& y,
              mtl::vec::dense_vector<T>& a, mtl::vec::dense_vector<T>& b) {
    for (std::size_t i = 0; i < x.size(); ++i) {
        a[i] = T(x[i]); b[i] = T(y[i]);
        x[i] = static_cast<double>(a[i]);
        y[i] = static_cast<double>(b[i]);
    }
}

/// Relative spread of the result over a few deterministic reorderings.
template <typename T, typename Acc>
double permutation_spread(const std::vector<double>& x, const std::vector<double>& y, double ref) {
    const std::size_t n = x.size();
    std::vector<std::size_t> p(n);
    std::iota(p.begin(), p.end(), std::size_t{0});
    double lo = std::numeric_limits<double>::infinity(), hi = -lo;
    for (std::size_t k = 0; k < 8; ++k) {
        // deterministic stride reorder: coprime strides visit every index once.
        const std::size_t stride = 2 * k + 3;
        std::vector<std::size_t> q(n);
        for (std::size_t i = 0, j = 0; i < n; ++i, j = (j + stride) % n) q[i] = p[j];
        mtl::vec::dense_vector<T> a(n, T(0)), b(n, T(0));
        for (std::size_t i = 0; i < n; ++i) { a[i] = T(x[q[i]]); b[i] = T(y[q[i]]); }
        const double v = mtl::dot<Acc, double>(a, b);
        lo = std::min(lo, v); hi = std::max(hi, v);
    }
    return (hi - lo) / ((ref != 0.0) ? std::abs(ref) : 1.0);
}

} // namespace

int main() {
    using namespace sw::universal;
    using P16 = posit<16, 2>;
    using P32 = posit<32, 2>;

    // -- 1. Dot2 is exact on a construction with a known answer -------------
    // The cancel regime sums to exactly the residual terms: every other term is
    // paired with its exact negation. Any reference weaker than doubled
    // precision would return garbage here, so this validates the yardstick
    // before it is used to score anything.
    {
        const double residual = 1e-9;
        auto [x, y] = sw::mp_blas::generate(regime::cancel, kN, residual);
        const std::size_t m = (kN - 1) / 2;
        const double expected = static_cast<double>(kN - 2 * m) * residual;
        const double ref = sw::mp_blas::dot2(x, y);
        check(ref == expected,
              "Dot2 reference not exact on the cancel construction (got " +
                  std::to_string(ref) + ", want " + std::to_string(expected) + ")");
    }

    // -- 2. cond >= 2, with equality for an all-positive reduction ----------
    {
        auto [x, y] = sw::mp_blas::generate(regime::positive, kN);
        const auto f = sw::mp_blas::characterize(x, y, 1e-16);
        check(std::abs(f.cond - 2.0) < 1e-12, "all-positive dot product has cond != 2");
        check(std::abs(f.sign_balance - 1.0) < 1e-12, "all-positive dot product is not sign-saturated");

        for (auto r : {regime::uniform, regime::graded, regime::cancel, regime::kahan}) {
            auto [xr, yr] = sw::mp_blas::generate(r, kN, (r == regime::uniform || r == regime::graded) ? 0.0 : 1e-6);
            const auto fr = sw::mp_blas::characterize(xr, yr, 1e-16);
            check(fr.cond >= 2.0 - 1e-12,
                  std::string("cond below its floor of 2 for regime ") + sw::mp_blas::to_string(r));
        }
    }

    // -- 3. cond tracks the engineered cancellation knob --------------------
    {
        double prev = 0.0;
        for (double residual : {1e-3, 1e-6, 1e-9}) {
            auto [x, y] = sw::mp_blas::generate(regime::cancel, kN, residual);
            const auto f = sw::mp_blas::characterize(x, y, 1e-16);
            check(f.cond > prev * 100.0,
                  "cond did not grow with shrinking residual (residual=" +
                      std::to_string(residual) + ", cond=" + std::to_string(f.cond) + ")");
            prev = f.cond;
        }
    }

    // -- 3b. generate_at_cond pins cond independently of n -------------------
    // The control the length sweep depends on: if cond drifted with n, the
    // sweep could not separate the length axis from the conditioning axis.
    {
        const double target = 1e12;
        for (std::size_t n : {256u, 1024u, 4096u, 16384u}) {
            auto [x, y] = sw::mp_blas::generate_at_cond(n, target);
            const auto f = sw::mp_blas::characterize(x, y, 1e-16);
            check(f.cond > target / 4.0 && f.cond < target * 4.0,
                  "generate_at_cond missed its target at n=" + std::to_string(n) +
                      " (cond=" + std::to_string(f.cond) + ", target=" + std::to_string(target) + ")");
        }
    }

    // -- 4. Swamping is a distinct axis: D up => n_eff down ------------------
    // Both instances are benign by cond, so any difference in n_eff is dynamic
    // range alone. This is why cond is not a sufficient statistic.
    {
        const double u16 = 0.5 * static_cast<double>(std::numeric_limits<P16>::epsilon());
        auto [xu, yu] = sw::mp_blas::generate(regime::uniform, kN);
        auto [xg, yg] = sw::mp_blas::generate(regime::graded, kN, 40.0);
        mtl::vec::dense_vector<P16> au(kN, P16(0)), bu(kN, P16(0)), ag(kN, P16(0)), bg(kN, P16(0));
        quantize<P16>(xu, yu, au, bu);
        quantize<P16>(xg, yg, ag, bg);
        const auto fu = sw::mp_blas::characterize(xu, yu, u16);
        const auto fg = sw::mp_blas::characterize(xg, yg, u16);
        check(fg.dynamic_range > fu.dynamic_range, "graded regime does not widen the exponent spread");
        check(fg.n_eff < fu.n_eff, "wider exponent spread did not reduce n_eff (no swamping detected)");
        check(fu.cond < 1e4 && fg.cond < 1e4, "uniform/graded regimes are supposed to be well-conditioned");
    }

    // -- 5, 6, 7. Bound, reproducibility, accumulator ordering --------------
    {
        struct Case { regime r; double param; const char* label; };
        const Case cases[] = {
            { regime::uniform,  0.0,  "uniform"     },
            { regime::positive, 0.0,  "positive"    },
            { regime::graded,   40.0, "graded"      },
            { regime::cancel,   1e-6, "cancel 1e-6" },
        };

        auto run = [&](auto tag, const char* tname, double u) {
            using T = decltype(tag);
            using Quire = quire<T>;
            for (const auto& c : cases) {
                auto [x, y] = sw::mp_blas::generate(c.r, kN, c.param);
                mtl::vec::dense_vector<T> a(kN, T(0)), b(kN, T(0));
                quantize<T>(x, y, a, b);
                const auto f = sw::mp_blas::characterize(x, y, u);
                const double denom = (f.ref != 0.0) ? std::abs(f.ref) : 1.0;
                auto rel = [&](double v) { return std::abs(v - f.ref) / denom; };

                const double e_native   = rel(mtl::dot<T, double>(a, b));
                const double e_promoted = rel(mtl::dot<double, double>(a, b));
                const double e_quire    = rel(mtl::dot<Quire, double>(a, b));
                const std::string tag_s = std::string(tname) + "/" + c.label;

                // 5. Wilkinson/Higham: |fl(x'y) - x'y| <= gamma_n |x|'|y|, i.e.
                //    relative error <= gamma_n * cond / 2 <= n*u*cond.
                const double bound = static_cast<double>(kN) * u * f.cond;
                check(e_native <= bound,
                      tag_s + ": native error " + std::to_string(e_native) +
                          " exceeds the n*u*cond bound " + std::to_string(bound));

                // 7. More accumulator buys no less accuracy. The slack absorbs
                //    the single round-out to double that all three share.
                check(e_quire <= e_promoted + 1e-15, tag_s + ": quire worse than promoted double");
                check(e_promoted <= e_native + 1e-15, tag_s + ": promoted double worse than native");

                // 6. Kulisch's guarantee: the exact accumulation is invariant
                //    under summation order; the native one is not.
                const double s_quire = permutation_spread<T, Quire>(x, y, f.ref);
                check(s_quire == 0.0,
                      tag_s + ": quire is not order-independent (spread " + std::to_string(s_quire) + ")");
                if (c.r == regime::cancel) {
                    const double s_native = permutation_spread<T, T>(x, y, f.ref);
                    check(s_native > 0.0,
                          tag_s + ": native accumulation is unexpectedly reproducible");
                }
            }
        };

        run(P16{}, "posit<16,2>", 0.5 * static_cast<double>(std::numeric_limits<P16>::epsilon()));
        run(P32{}, "posit<32,2>", 0.5 * static_cast<double>(std::numeric_limits<P32>::epsilon()));
    }

    if (failures == 0) std::cout << "test_dot_characterization passed\n";
    return failures == 0 ? 0 : 1;
}
