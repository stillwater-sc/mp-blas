// mp-blas -- verified solve: the exact residual and defect correction
// (issue #12, Phase 5).
//
// This is what the whole line of work was built toward. Issue #9 established
// that Kulisch designed the exact scalar product for VERIFIED COMPUTING -- an
// exact residual makes defect correction effective, which is what produces
// results with machine-proved bounds. Phases 1-4 built and characterized the
// instrument. This applies it to the problem it was designed for.
//
// Two parts.
//
// A. Naive interval triangular solve. Documented because it is the honest
//    baseline: interval arithmetic applied directly to elimination is known to
//    be catastrophic, and the study should show that rather than quietly skip
//    to the method that works.
//
// B. Defect correction with the residual computed two ways. The classical
//    Wilkinson result, and Kulisch's argument:
//
//      residual in WORKING precision  ->  refinement stagnates at ~cond*u
//      residual computed EXACTLY      ->  refinement converges to ~u,
//                                          regardless of cond (while cond*u < 1)
//
//    So the exact residual does not buy "a few more digits"; it decouples the
//    achievable accuracy from the conditioning entirely, up to the point where
//    the factorization itself fails.
//
// WHY A GENERAL SYSTEM AND NOT trsv. Triangular systems are solved far more
// accurately than their condition number suggests (Higham, ASNA ch. 8), so
// back-substitution lands within a few ulp and defect correction has essentially
// nothing to correct -- measured, and it shows no difference between the two
// residuals. The effect needs a system whose forward error genuinely tracks
// cond*u, so part B uses the Hilbert matrix, the classical ill-conditioned
// example.
//
// Usage: verified_solve_study [nmax]     (default nmax = 8)
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/lu.hpp>
#include <mtl/operation/trsv.hpp>

#include <sw/mp_blas/dot_characterization.hpp>
#include <sw/mp_blas/interval_containment.hpp>

#include <universal/number/posit/posit.hpp>
#include <universal/number/cfloat/cfloat.hpp>
#include <universal/number/interval/interval.hpp>

namespace {

// ---------------------------------------------------------------------------
// A. Naive interval triangular solve
// ---------------------------------------------------------------------------

/// Back-substitution over interval elements. Each x_i divides by U(i,i) and
/// subtracts a growing sum of interval products, so the enclosure width
/// compounds down the solve.
template <typename Scalar>
void interval_trsv_sweep(const std::string& name, std::size_t n) {
    using I = sw::universal::interval<Scalar>;

    std::cout << "\n=== A. naive interval trsv, interval<" << name << ">, n = " << n << " ===\n"
              << "  U has unit diagonal and `off` above; off -> -1 worsens conditioning.\n"
              << "  Interval arithmetic applied directly to elimination -- the honest baseline.\n"
              << std::right << std::setw(9) << "off" << std::setw(13) << "max rel w"
              << std::setw(11) << "digits" << '\n';

    for (double off : {-0.25, -0.5, -0.75, -0.9, -0.99}) {
        mtl::mat::dense2D<I> U(n, n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                U(i, j) = I(Scalar((i == j) ? 1.0 : (j > i ? off : 0.0)));

        mtl::vec::dense_vector<I> b(n, I(Scalar(0))), x(n, I(Scalar(0)));
        for (std::size_t i = 0; i < n; ++i)
            b[i] = I(Scalar(1.0 + 0.5 * sw::mp_blas::synth(i, 3)));

        mtl::trsv(U, x, b, /*upper=*/true);

        double w = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            w = std::max(w, sw::mp_blas::relative_width(x[i]));

        std::cout << std::right << std::setw(9) << std::fixed << std::setprecision(2) << off
                  << std::scientific << std::setprecision(2) << std::setw(13) << w
                  << std::fixed << std::setprecision(1) << std::setw(11) << -std::log10(w)
                  << std::defaultfloat << '\n';
    }
}

// ---------------------------------------------------------------------------
// B. Defect correction: working-precision vs exact residual
// ---------------------------------------------------------------------------

struct Refinement {
    std::vector<double> err;   ///< relative forward error per iteration
    double cond = 0.0;
};

/// Solve a Hilbert system in `Scalar`, then apply defect correction with the
/// residual formed either in Scalar or exactly (Dot2 over the same quantized
/// values). Ground truth is a `double` LU solve, which has ~7 decades of
/// headroom over posit<32,2>'s unit roundoff.
template <typename Scalar>
Refinement refine_hilbert(std::size_t n, bool exact_residual, int iters) {
    mtl::mat::dense2D<Scalar> A(n, n);
    mtl::mat::dense2D<double> Ad(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            const Scalar v(1.0 / static_cast<double>(i + j + 1));
            A(i, j)  = v;
            Ad(i, j) = static_cast<double>(v);
        }

    mtl::vec::dense_vector<Scalar> b(n, Scalar(0));
    mtl::vec::dense_vector<double> bd(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (std::size_t j = 0; j < n; ++j) s += Ad(i, j);
        b[i]  = Scalar(s);
        bd[i] = static_cast<double>(b[i]);
    }

    // Ground truth and a condition estimate, both from the double factorization.
    mtl::mat::dense2D<double> LUd(Ad);
    std::vector<std::size_t> pd;
    mtl::lu_factor(LUd, pd);
    mtl::vec::dense_vector<double> xstar(n, 0.0);
    mtl::lu_solve(LUd, pd, xstar, bd);

    double xn = 0.0;
    for (std::size_t i = 0; i < n; ++i) xn = std::max(xn, std::abs(xstar[i]));

    Refinement out;
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

    for (int it = 0; it <= iters; ++it) {
        double e = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            e = std::max(e, std::abs(static_cast<double>(x[i]) - xstar[i]));
        out.err.push_back(e / xn);
        if (it == iters) break;

        mtl::vec::dense_vector<Scalar> r(n, Scalar(0));
        for (std::size_t i = 0; i < n; ++i) {
            if (exact_residual) {
                // r_i = b_i - sum_j A(i,j) x_j, as ONE exactly-evaluated reduction
                // over the quantized values. This is the operation Kulisch wanted
                // the exact scalar product for.
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
    return out;
}

template <typename Scalar>
void defect_correction_study(const std::string& name, std::size_t nmax) {
    const double u = 0.5 * static_cast<double>(std::numeric_limits<Scalar>::epsilon());
    constexpr int kIters = 4;

    std::cout << "\n=== B. defect correction on a Hilbert system, " << name
              << " (u = " << std::scientific << std::setprecision(2) << u << ") ===\n"
              << std::defaultfloat
              << "  relative forward error per correction step; ground truth from a double LU\n"
              << std::right << std::setw(4) << "n" << std::setw(10) << "cond"
              << std::setw(10) << "cond*u" << std::setw(9) << "residual";
    for (int i = 0; i <= kIters; ++i) std::cout << std::setw(10) << ("it" + std::to_string(i));
    std::cout << '\n';

    for (std::size_t n = 4; n <= nmax; n += 2) {
        const Refinement w = refine_hilbert<Scalar>(n, false, kIters);
        const Refinement e = refine_hilbert<Scalar>(n, true,  kIters);

        for (int pass = 0; pass < 2; ++pass) {
            const Refinement& R = (pass == 0) ? w : e;
            if (pass == 0)
                std::cout << std::right << std::setw(4) << n
                          << std::scientific << std::setprecision(1)
                          << std::setw(10) << R.cond << std::setw(10) << R.cond * u;
            else
                std::cout << std::setw(4) << "" << std::setw(10) << "" << std::setw(10) << "";
            std::cout << std::setw(9) << (pass == 0 ? "working" : "EXACT");
            for (double v : R.err)
                std::cout << std::scientific << std::setprecision(1) << std::setw(10) << v;
            std::cout << std::defaultfloat << '\n';
        }
    }
    std::cout << "  (cond*u > 1 is past the point where the working-precision factorization\n"
              << "   itself carries no correct digits, so neither residual can rescue it.)\n";
}

} // namespace

int main(int argc, char* argv[]) {
    std::size_t nmax = 8;
    if (argc > 1) nmax = static_cast<std::size_t>(std::atoll(argv[1]));

    std::cout << "Verified solve: the exact residual and defect correction (issue #12, Phase 5)\n"
              << "  A. naive interval trsv -- the honest baseline\n"
              << "  B. defect correction with a working vs EXACT residual -- the payoff\n";

    interval_trsv_sweep<sw::universal::posit<32, 2>>("posit<32,2>", 16);

    defect_correction_study<sw::universal::posit<32, 2>>("posit<32,2>", nmax);
    defect_correction_study<sw::universal::cfloat<32, 8>>("cfloat<32,8>", nmax);

    return 0;
}
