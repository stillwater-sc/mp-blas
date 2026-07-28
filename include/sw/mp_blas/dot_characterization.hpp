#pragma once
// mp-blas -- dot-product characterization: structural metrics of an inner
// product instance, and generators for the canonical regimes (issue #9).
//
// WHY. The accumulator studies in this repo measure the ERROR a dot product
// incurs under native / promoted / quire accumulation. They do not measure the
// INPUT STRUCTURE that produces that error, so their findings read as anecdotes
// tied to one synthetic distribution. The numerical-analysis literature says
// most of the outcome is predicted by a small feature vector of the instance;
// this header computes it, so a quire study can correlate structure with
// accumulator choice instead of reporting isolated numbers.
//
// The governing result is Wilkinson's inner-product bound (Higham, "Accuracy
// and Stability of Numerical Algorithms", 2nd ed., ch. 3):
//
//     |fl(x'y) - x'y| <= gamma_n * |x|'|y|,     gamma_n = n*u / (1 - n*u)
//
// which, divided through by |x'y|, gives the RELATIVE error in terms of the
// dot-product condition number (Ogita, Rump & Oishi, "Accurate Sum and Dot
// Product", SIAM J. Sci. Comput. 2005):
//
//     cond(x,y) = 2 * |x|'|y| / |x'y|
//
// cond is the *information* measure the issue asks for: |x|'|y| is what the
// input vectors carry in, |x'y| is what survives into the result, and their
// ratio is the number of significant digits the reduction destroys. Everything
// else here refines that picture with the order-dependent effects a single
// scalar cannot express (swamping, growth, reproducibility).
//
// The reference value is computed with Dot2 (Ogita-Rump-Oishi Algorithm 5.3),
// an error-free-transformation dot product accurate as if evaluated in twice
// the working precision. It is deliberately NOT a quire: the reference must be
// independent of the accumulator under test.
//
// Namespace: sw::mp_blas. Header-only, depends only on the standard library --
// no MTL5 and no Universal, so the metrics can be applied to any element type
// once its values are read out as double.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace sw::mp_blas {

// ---------------------------------------------------------------------------
// Error-free transformations
// ---------------------------------------------------------------------------

/// Knuth's TwoSum: s = fl(a+b) and e = (a+b) - s exactly, for any a, b.
inline void two_sum(double a, double b, double& s, double& e) noexcept {
    s = a + b;
    const double bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}

/// TwoProduct via FMA: p = fl(a*b) and e = a*b - p exactly.
inline void two_product(double a, double b, double& p, double& e) noexcept {
    p = a * b;
    e = std::fma(a, b, -p);
}

/// Dot2 (Ogita-Rump-Oishi 2005, Algorithm 5.3): a dot product whose result is
/// accurate as if computed in doubled working precision and rounded once. Used
/// as the reference the accumulator strategies are scored against.
inline double dot2(const std::vector<double>& x, const std::vector<double>& y) {
    const std::size_t n = x.size();
    if (n == 0) return 0.0;
    double p, s;
    two_product(x[0], y[0], p, s);
    for (std::size_t i = 1; i < n; ++i) {
        double h, r, q;
        two_product(x[i], y[i], h, r);
        two_sum(p, h, p, q);
        s += q + r;
    }
    return p + s;
}

// ---------------------------------------------------------------------------
// Feature vector
// ---------------------------------------------------------------------------

/// Structural characterization of one dot-product instance. All fields are
/// computed on the values the kernel actually sees (i.e. AFTER quantization to
/// the element type), so they describe the realized problem, not the intent.
struct dot_features {
    std::size_t n            = 0;    ///< length
    double      ref          = 0.0;  ///< Dot2 reference value of x'y
    double      abs_sum      = 0.0;  ///< |x|'|y| -- information carried IN
    /// 2*abs_sum/|ref|. Bounded below by 2 (attained when every product has the
    /// same sign); log10(cond) is the decimal digits the reduction destroys.
    double      cond         = 0.0;
    double      dynamic_range= 0.0;  ///< D: log2(max|p_i| / min nonzero |p_i|)
    double      growth       = 0.0;  ///< rho: max partial |sum| / |ref|
    std::size_t n_eff        = 0;    ///< terms that can still move the accumulator
    double      sign_balance = 0.0;  ///< beta: |#pos - #neg| / n, 0 = balanced

    /// Decimal digits of cancellation implied by cond.
    double digits_lost() const {
        return (cond > 1.0 && std::isfinite(cond)) ? std::log10(cond) : 0.0;
    }
};

/// Compute the feature vector of x'y.
///
/// `u` is the unit roundoff of the accumulator whose swamping behavior `n_eff`
/// should describe (for a native reduction, the element type's u). A term can
/// only move an accumulator that currently holds M when |p_i| >= u*|M|; taking
/// M as the largest partial sum reached, `n_eff` counts the terms that are
/// still visible at the accumulator's worst moment. n_eff << n is the signature
/// of swamping, and it is what an exact quire removes.
inline dot_features characterize(const std::vector<double>& x,
                                 const std::vector<double>& y,
                                 double u) {
    dot_features f;
    f.n = x.size();
    if (f.n == 0 || y.size() != x.size()) return f;

    // |x|'|y| is an all-positive reduction, so Dot2 gives it to near machine
    // precision -- no cancellation to amplify.
    std::vector<double> ax(f.n), ay(f.n);
    for (std::size_t i = 0; i < f.n; ++i) { ax[i] = std::abs(x[i]); ay[i] = std::abs(y[i]); }
    f.abs_sum = dot2(ax, ay);
    f.ref     = dot2(x, y);

    f.cond = (f.ref != 0.0) ? 2.0 * f.abs_sum / std::abs(f.ref)
                            : std::numeric_limits<double>::infinity();

    // Dynamic range of the products, and sign balance.
    double pmax = 0.0, pmin = std::numeric_limits<double>::infinity();
    std::int64_t signs = 0;
    for (std::size_t i = 0; i < f.n; ++i) {
        const double p = std::abs(x[i] * y[i]);
        if (p > pmax) pmax = p;
        if (p > 0.0 && p < pmin) pmin = p;
        const double sp = x[i] * y[i];
        if (sp > 0.0) ++signs; else if (sp < 0.0) --signs;
    }
    f.dynamic_range = (pmax > 0.0 && std::isfinite(pmin) && pmin > 0.0)
                          ? std::log2(pmax / pmin) : 0.0;
    f.sign_balance = static_cast<double>(std::abs(signs)) / static_cast<double>(f.n);

    // Growth factor: the largest magnitude the running sum reaches, relative to
    // where it ends up. rho >> 1 means the reduction climbs a mountain and comes
    // back down -- the regime where intermediate rounding, not the final value,
    // sets the error. Order-dependent by construction (index order here).
    long double running = 0.0L, peak = 0.0L;
    for (std::size_t i = 0; i < f.n; ++i) {
        running += static_cast<long double>(x[i]) * static_cast<long double>(y[i]);
        peak = std::max(peak, std::abs(running));
    }
    const double peakd = static_cast<double>(peak);
    f.growth = (f.ref != 0.0) ? peakd / std::abs(f.ref)
                              : std::numeric_limits<double>::infinity();

    // Terms still visible at the accumulator's largest excursion.
    const double threshold = u * peakd;
    for (std::size_t i = 0; i < f.n; ++i)
        if (std::abs(x[i] * y[i]) >= threshold) ++f.n_eff;

    return f;
}

// ---------------------------------------------------------------------------
// Regime generators
// ---------------------------------------------------------------------------

/// Deterministic uniform value in (-1, 1): reproducible across platforms and
/// compilers, no <random> (matching dot_accumulator_study's `synth`).
inline double synth(std::size_t i, std::size_t salt) {
    std::uint64_t z = (i + 1) * 0x9E3779B97F4A7C15ull + salt * 0xBF58476D1CE4E5B9ull;
    z ^= z >> 30; z *= 0xBF58476D1CE4E5B9ull;
    z ^= z >> 27; z *= 0x94D049BB133111EBull;
    z ^= z >> 31;
    return 2.0 * (static_cast<double>(z >> 11) * 0x1.0p-53) - 1.0;
}

/// The canonical dot-product regimes. Each isolates one structural axis so a
/// study can attribute an error to a cause rather than to "the data".
enum class regime {
    uniform,   ///< x,y iid U(-1,1): the default benchmark distribution
    positive,  ///< x'x: all terms same sign, so cond attains its floor of 2
    graded,    ///< wide exponent spread, no engineered cancellation (swamping only)
    cancel,    ///< engineered cancellation: cond ~ 1/param (cancellation only)
    kahan      ///< graded AND cancelling: the adversarial composition
};

inline const char* to_string(regime r) {
    switch (r) {
        case regime::uniform:  return "uniform";
        case regime::positive: return "positive";
        case regime::graded:   return "graded";
        case regime::cancel:   return "cancel";
        case regime::kahan:    return "kahan";
    }
    return "?";
}

/// Generate an (x, y) pair for a regime.
///
/// `param` is regime-specific:
///   graded  -- half-width of the exponent spread in bits (default 40)
///   cancel  -- relative size of the surviving residual; cond ~ 4/param
///   kahan   -- same as cancel, applied on top of a graded magnitude profile
///
/// The `cancel` and `kahan` layouts place ALL positive terms first and all
/// their exact negations second, with the residual last. That ordering is
/// deliberate: pairing p with -p adjacently would let any accumulator cancel
/// them immediately, hiding the effect. Front-loading drives the running sum to
/// |x|'|y|/2 before it collapses, so the residual is maximally swamped -- the
/// configuration Kulisch's exact scalar product was designed to survive.
/// Because every term is negated exactly, the construction is invariant under
/// quantization to any symmetric number system: the pairs still cancel exactly
/// and the exact result remains the residual.
inline std::pair<std::vector<double>, std::vector<double>>
generate(regime r, std::size_t n, double param = 0.0) {
    std::vector<double> x(n, 0.0), y(n, 0.0);
    switch (r) {
        case regime::uniform:
            for (std::size_t i = 0; i < n; ++i) { x[i] = synth(i, 1); y[i] = synth(i, 2); }
            break;

        case regime::positive:
            for (std::size_t i = 0; i < n; ++i) { x[i] = synth(i, 1); y[i] = x[i]; }
            break;

        case regime::graded: {
            const double halfwidth = (param > 0.0) ? param : 40.0;
            for (std::size_t i = 0; i < n; ++i) {
                const double t = (n > 1) ? (2.0 * double(i) / double(n - 1) - 1.0) : 0.0;
                x[i] = std::ldexp(synth(i, 1), static_cast<int>(t * halfwidth));
                y[i] = synth(i, 2);
            }
            break;
        }

        case regime::cancel:
        case regime::kahan: {
            const double residual  = (param > 0.0) ? param : 1e-8;
            const double halfwidth = (r == regime::kahan) ? 40.0 : 0.0;
            const std::size_t m = (n - 1) / 2;   // m positive + m negative + 1 residual
            for (std::size_t j = 0; j < m; ++j) {
                double a = synth(j, 1), b = std::abs(synth(j, 2)) + 0.5;
                if (halfwidth > 0.0) {
                    const double t = (m > 1) ? (2.0 * double(j) / double(m - 1) - 1.0) : 0.0;
                    a = std::ldexp(a, static_cast<int>(t * halfwidth));
                }
                x[j]     = a; y[j]     =  b;   // + p_j
                x[m + j] = a; y[m + j] = -b;   // - p_j, exactly
            }
            // The one term that survives; everything before it sums to exactly zero.
            for (std::size_t i = 2 * m; i < n; ++i) { x[i] = residual; y[i] = 1.0; }
            break;
        }
    }
    return {std::move(x), std::move(y)};
}

} // namespace sw::mp_blas
