// mp-blas level-2 test: defect correction and the exact residual
// (issue #12, Phase 5).
//
// This pins the claim the whole line of work was built toward. Issue #9
// established that Kulisch designed the exact scalar product for VERIFIED
// COMPUTING: an exact residual makes defect correction effective. Phases 1-4
// built and characterized the instrument; this asserts it does the job.
//
//   residual in WORKING precision  ->  refinement stagnates at ~cond*u
//   residual computed EXACTLY      ->  refinement converges to ~u,
//                                       regardless of cond (while cond*u < 1)
//
// The exact residual does not buy "a few more digits". It decouples the
// achievable accuracy from the conditioning, up to the point where the
// factorization itself fails.
//
// NOTE ON THE TEST SYSTEM. Triangular systems are solved far more accurately
// than their condition number suggests (Higham, ASNA ch. 8), so back-substitution
// lands within a few ulp and defect correction has nothing to correct -- measured
// while building this, and it shows NO difference between the two residuals.
// A test built on trsv would therefore pass vacuously. The Hilbert matrix is used
// because its forward error genuinely tracks cond*u.
//
// Returns non-zero on failure (no external framework).
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/lu.hpp>

#include <sw/mp_blas/dot_characterization.hpp>

#include <universal/number/posit/posit.hpp>
#include <universal/number/cfloat/cfloat.hpp>

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) { std::cerr << "FAIL: " << what << '\n'; ++failures; }
}

struct Result {
    double first = 0.0;   ///< error of the initial solve
    double last  = 0.0;   ///< error after the correction steps
    double cond  = 0.0;
};

/// Hilbert solve in `Scalar` plus defect correction, with the residual formed
/// either in Scalar or exactly (Dot2 over the same quantized values).
template <typename Scalar>
Result refine(std::size_t n, bool exact_residual, int iters = 4) {
    mtl::mat::dense2D<Scalar> A(n, n);
    mtl::mat::dense2D<double> Ad(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            const Scalar v(1.0 / static_cast<double>(i + j + 1));
            A(i, j) = v;
            Ad(i, j) = static_cast<double>(v);
        }

    mtl::vec::dense_vector<Scalar> b(n, Scalar(0));
    mtl::vec::dense_vector<double> bd(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (std::size_t j = 0; j < n; ++j) s += Ad(i, j);
        b[i] = Scalar(s);
        bd[i] = static_cast<double>(b[i]);
    }

    mtl::mat::dense2D<double> LUd(Ad);
    std::vector<std::size_t> pd;
    mtl::lu_factor(LUd, pd);
    mtl::vec::dense_vector<double> xstar(n, 0.0);
    mtl::lu_solve(LUd, pd, xstar, bd);
    double xn = 0.0;
    for (std::size_t i = 0; i < n; ++i) xn = std::max(xn, std::abs(xstar[i]));

    Result out;
    {
        double an = 0.0, in = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            double t = 0.0;
            for (std::size_t j = 0; j < n; ++j) t += std::abs(Ad(i, j));
            an = std::max(an, t);
        }
        for (std::size_t j = 0; j < n; ++j) {
            mtl::vec::dense_vector<double> e(n, 0.0), c(n, 0.0);
            e[j] = 1.0;
            mtl::lu_solve(LUd, pd, c, e);
            double t = 0.0;
            for (std::size_t i = 0; i < n; ++i) t += std::abs(c[i]);
            in = std::max(in, t);
        }
        out.cond = an * in;
    }

    mtl::mat::dense2D<Scalar> LU(A);
    std::vector<std::size_t> pv;
    mtl::lu_factor(LU, pv);
    mtl::vec::dense_vector<Scalar> x(n, Scalar(0));
    mtl::lu_solve(LU, pv, x, b);

    auto err = [&]() {
        double e = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            e = std::max(e, std::abs(static_cast<double>(x[i]) - xstar[i]));
        return e / xn;
    };
    out.first = err();

    for (int it = 0; it < iters; ++it) {
        mtl::vec::dense_vector<Scalar> r(n, Scalar(0));
        for (std::size_t i = 0; i < n; ++i) {
            if (exact_residual) {
                std::vector<double> px(n + 1), py(n + 1);
                for (std::size_t j = 0; j < n; ++j) {
                    px[j] = static_cast<double>(A(i, j));
                    py[j] = -static_cast<double>(x[j]);
                }
                px[n] = 1.0;
                py[n] = static_cast<double>(b[i]);
                r[i] = Scalar(sw::mp_blas::dot2(px, py));
            } else {
                Scalar acc(0);
                for (std::size_t j = 0; j < n; ++j) acc = acc + A(i, j) * x[j];
                r[i] = b[i] - acc;
            }
        }
        mtl::vec::dense_vector<Scalar> d(n, Scalar(0));
        mtl::lu_solve(LU, pv, d, r);
        for (std::size_t i = 0; i < n; ++i) x[i] = x[i] + d[i];
    }
    out.last = err();
    return out;
}

/// The full claim, for one element type at one size in the regime where defect
/// correction is supposed to work (cond*u well below 1).
template <typename Scalar>
void claim(const std::string& name, std::size_t n) {
    const double u = 0.5 * static_cast<double>(std::numeric_limits<Scalar>::epsilon());
    const Result w = refine<Scalar>(n, false);
    const Result e = refine<Scalar>(n, true);
    const std::string tag = name + "/n=" + std::to_string(n);

    // The test is only meaningful inside the regime where the theory applies.
    check(w.cond * u < 0.5,
          tag + ": cond*u = " + std::to_string(w.cond * u) +
              " is past the point where defect correction can work at all, so this "
              "case cannot test the claim");

    // The initial solve must actually be inaccurate, or there is nothing to correct
    // and both residuals would trivially agree (this is exactly why a triangular
    // system cannot be used here).
    check(w.first > 100.0 * u,
          tag + ": the initial solve is already accurate to " + std::to_string(w.first) +
              " (~" + std::to_string(w.first / u) + " u), so defect correction has "
              "nothing to correct and the test is vacuous");

    // 1. EXACT residual reaches working precision.
    check(e.last < 20.0 * u,
          tag + ": with an EXACT residual, refinement reached " + std::to_string(e.last) +
              " rather than ~u = " + std::to_string(u));

    // 2. WORKING-precision residual does not.
    check(w.last > 100.0 * u,
          tag + ": a working-precision residual reached " + std::to_string(w.last) +
              ", which is ~u -- if refinement now converges without an exact residual, "
              "Phase 5's central claim is wrong");

    // 3. And the gap is large, not marginal.
    check(w.last / e.last > 100.0,
          tag + ": exact-vs-working final error gap is only x" +
              std::to_string(w.last / e.last));

    std::cout << "  " << std::left << std::setw(20) << tag
              << " cond " << std::scientific << std::setprecision(1) << w.cond
              << "  initial " << w.first
              << "  ->  working " << w.last << ", EXACT " << e.last
              << "  (u = " << u << ")\n" << std::defaultfloat;
}

} // namespace

int main() {
    using namespace sw::universal;

    std::cout << "issue #12 Phase 5 -- defect correction and the exact residual\n";

    // posit<32,2>: u = 3.7e-09. n=4 gives cond ~2.8e4, so cond*u ~1e-4 -- well
    // inside the regime where refinement is supposed to work.
    claim<posit<32, 2>>("posit<32,2>", 4);
    claim<posit<32, 2>>("posit<32,2>", 6);

    // cfloat<32,8>: u = 6.0e-08, so only the smaller system stays in regime.
    claim<cfloat<32, 8>>("cfloat<32,8>", 4);

    // Beyond cond*u ~ 1 NEITHER residual can help: the working-precision
    // factorization itself carries no correct digits. Asserted so the claim is not
    // read as unbounded -- the exact residual removes the cond dependence of the
    // RESIDUAL, not of the factorization.
    {
        const double u = 0.5 * static_cast<double>(std::numeric_limits<posit<32, 2>>::epsilon());
        const Result e = refine<posit<32, 2>>(8, true);
        check(e.cond * u > 1.0,
              "n=8 Hilbert was expected to be past cond*u = 1 for posit<32,2>");
        check(e.last > 100.0 * u,
              "n=8: refinement with an exact residual reached " + std::to_string(e.last) +
                  " past cond*u = 1, which would mean the exact residual also rescues a "
                  "failed factorization -- it should not");
        std::cout << "  " << std::left << std::setw(20) << "posit<32,2>/n=8"
                  << " cond*u = " << std::scientific << std::setprecision(1) << e.cond * u
                  << " > 1: exact residual reaches only " << e.last
                  << " -- the factorization, not the residual, is the limit\n"
                  << std::defaultfloat;
    }

    if (failures == 0) std::cout << "test_verified_solve passed\n";
    return failures == 0 ? 0 : 1;
}
