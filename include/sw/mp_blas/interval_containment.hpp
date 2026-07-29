#pragma once
// mp-blas -- interval enclosure quality: containment checks and tightness
// metrics for BLAS operators over Universal's interval<Scalar> (issue #12,
// Phase 0).
//
// WHY. An interval result is only worth measuring if it is CORRECT first. The
// whole point of an enclosure is the containment property -- the Fundamental
// Theorem of Interval Arithmetic:
//
//     for all x in X, y in Y:   x op y  in  fl(X op Y)
//
// evaluated over the exact reals, not over Scalar. Width measured against a
// type that violates containment is worse than no measurement: intervals that
// are too NARROW to be correct look impressively tight. (Universal's interval
// did exactly that before stillwater-sc/universal#1234 -- `interval(0.1) *
// interval(0.1)` returned a zero-width interval around a value the exact
// product does not equal.) So this header provides the gate first and the
// quality metric second, and every interval study in mp-blas must pass the gate
// before its widths mean anything.
//
// The tightness metric is deliberately NOT raw width. Raw width conflates two
// unrelated things: how uncertain the problem genuinely is, and how much
// information the arithmetic threw away. The separable quantity is the
// OVERESTIMATION RATIO
//
//     R = width(computed) / width(optimal)
//
// where width(optimal) is the narrowest interval with representable endpoints
// that still encloses the true result. R = 1 means the enclosure is as good as
// the format allows; R >> 1 means the arithmetic, not the problem, produced the
// width. R is the interval analogue of the accumulator error in
// docs/dot-product-characterization.md, and it is what should be regressed
// against `cond` / `n_eff` from dot_characterization.hpp.
//
// Header-only. Depends on the standard library and Universal's interval; the
// reduction helpers additionally use MTL5. Namespace: sw::mp_blas.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include <universal/number/interval/interval.hpp>

namespace sw::mp_blas {

// ---------------------------------------------------------------------------
// Reference type
// ---------------------------------------------------------------------------
//
// `long double` is the widest built-in reference available portably. On x86 it
// carries a 64-bit significand, which holds the product of two <=32-bit
// Universal values exactly; for `double` elements use dot2() from
// dot_characterization.hpp and pass its result in as the reference instead.
using reference_t = long double;

// ---------------------------------------------------------------------------
// Directed rounding on the reference/Scalar grid
// ---------------------------------------------------------------------------

/// Next representable Scalar strictly below x (saturating at the low end).
template <typename Scalar>
Scalar round_down(Scalar x) {
    using std::nextafter;
    const Scalar lo = -std::numeric_limits<Scalar>::infinity();
    const Scalar r = nextafter(x, lo);
    using std::isnan;
    return isnan(r) ? x : r;   // some Universal types return NaR past maxneg
}

/// Next representable Scalar strictly above x (saturating at the high end).
template <typename Scalar>
Scalar round_up(Scalar x) {
    using std::nextafter;
    const Scalar hi = std::numeric_limits<Scalar>::infinity();
    const Scalar r = nextafter(x, hi);
    using std::isnan;
    return isnan(r) ? x : r;
}

// ---------------------------------------------------------------------------
// The gate: containment
// ---------------------------------------------------------------------------

/// Does the interval enclose a single reference value? This is the FTIA check
/// for an operation whose exact result is a point (any op on degenerate inputs,
/// and every reduction over degenerate intervals).
template <typename Interval>
bool encloses(const Interval& iv, reference_t truth) {
    const auto lo = static_cast<reference_t>(iv.lower());
    const auto hi = static_cast<reference_t>(iv.upper());
    return lo <= truth && truth <= hi;
}

/// Does the interval enclose an entire reference range [truth_lo, truth_hi]?
/// Used when the inputs have real width, so the exact result is itself a range.
template <typename Interval>
bool encloses(const Interval& iv, reference_t truth_lo, reference_t truth_hi) {
    const auto lo = static_cast<reference_t>(iv.lower());
    const auto hi = static_cast<reference_t>(iv.upper());
    return lo <= truth_lo && truth_hi <= hi;
}

// ---------------------------------------------------------------------------
// The quality metric: overestimation
// ---------------------------------------------------------------------------

/// Is a reference value inside Scalar's finite dynamic range?
///
/// Matters because tightness is not assessable for a result the format cannot
/// hold: the narrowest enclosure of an overflowed value is [maxpos, +inf], whose
/// width is infinite, so the overestimation ratio is undefined. Containment is
/// still perfectly meaningful there -- [maxpos, +inf] does enclose the truth --
/// so callers should keep checking containment and skip R.
///
/// Compares against max() rather than testing isfinite() on the converted value,
/// because posit saturates to maxpos instead of overflowing to infinity, so a
/// converted out-of-range posit still looks finite.
template <typename Scalar>
bool in_range(reference_t v) {
    const auto mx = static_cast<reference_t>(std::numeric_limits<Scalar>::max());
    return std::abs(v) <= mx;
}

/// Width of the narrowest Scalar-representable interval enclosing [t_lo, t_hi].
/// Zero when both endpoints are exactly representable; otherwise ~1 ulp per
/// inexact endpoint. This is the best any implementation could do, so it is the
/// denominator R is measured against.
template <typename Scalar>
double optimal_width(reference_t t_lo, reference_t t_hi) {
    const auto s_lo = static_cast<Scalar>(t_lo);
    const auto s_hi = static_cast<Scalar>(t_hi);
    const Scalar lo = (static_cast<reference_t>(s_lo) <= t_lo) ? s_lo : round_down(s_lo);
    const Scalar hi = (static_cast<reference_t>(s_hi) >= t_hi) ? s_hi : round_up(s_hi);
    return static_cast<double>(static_cast<reference_t>(hi) - static_cast<reference_t>(lo));
}

/// Overestimation ratio R = width(computed) / width(optimal).
///
/// R == 1  : optimal -- the enclosure is as tight as the format permits
/// R >  1  : the arithmetic contributed width the problem did not require
/// R <  1  : IMPOSSIBLE for a sound implementation; indicates lost containment
///
/// Both widths are regularized by one ulp at the result. Without that, a truth
/// value that is exactly representable gives optimal width 0, and the ratio is
/// 0/0 -- yet a zero-width enclosure at exactly the right value is OPTIMAL, not
/// degenerate. (This is the common case, not a corner: `interval(x) +
/// interval(x)` is exact for any representable x, since doubling is exact.)
/// Regularizing makes R == 1 there, keeps R monotone in the computed width, and
/// preserves R >= 1 for every sound implementation.
template <typename Interval, typename Scalar = typename Interval::value_type>
double overestimation(const Interval& iv, reference_t t_lo, reference_t t_hi) {
    const double w_computed = static_cast<double>(
        static_cast<reference_t>(iv.upper()) - static_cast<reference_t>(iv.lower()));
    const double w_optimal = optimal_width<Scalar>(t_lo, t_hi);

    const auto s = static_cast<Scalar>(t_lo);
    const double ulp = static_cast<double>(
        static_cast<reference_t>(round_up(s)) - static_cast<reference_t>(s));
    const double eps = (ulp > 0.0) ? ulp : std::numeric_limits<double>::min();

    // Not assessable when the truth is outside Scalar's range: the narrowest
    // enclosure is then [maxpos, +inf] and both widths are infinite. Report
    // infinity rather than letting inf-inf produce a NaN that reads as "R < 1",
    // i.e. as lost containment. Guard with in_range() to skip instead.
    if (!std::isfinite(w_computed) || !std::isfinite(w_optimal) || !std::isfinite(eps))
        return std::numeric_limits<double>::infinity();

    return (w_computed + eps) / (w_optimal + eps);
}

template <typename Interval, typename Scalar = typename Interval::value_type>
double overestimation(const Interval& iv, reference_t truth) {
    return overestimation<Interval, Scalar>(iv, truth, truth);
}

/// Relative width w/|midpoint| -- the certified precision actually delivered.
template <typename Interval>
double relative_width(const Interval& iv) {
    const auto lo = static_cast<reference_t>(iv.lower());
    const auto hi = static_cast<reference_t>(iv.upper());
    const reference_t mid = 0.5L * (lo + hi);
    const reference_t denom = (mid != 0.0L) ? std::abs(mid) : 1.0L;
    return static_cast<double>((hi - lo) / denom);
}

/// Decimal digits the enclosure certifies.
template <typename Interval>
double certified_digits(const Interval& iv) {
    const double rw = relative_width(iv);
    if (rw <= 0.0) return std::numeric_limits<double>::infinity();
    return -std::log10(rw);
}

// ---------------------------------------------------------------------------
// Unbounded-result capability
// ---------------------------------------------------------------------------

/// Can `interval<Scalar>` represent an unbounded result?
///
/// Division by an interval containing zero has an unbounded exact result set,
/// and Universal's `interval` signals it with
/// `+/-numeric_limits<Scalar>::infinity()`. That only works when Scalar HAS an
/// infinity. It does not for posit (NaR only -- `infinity()` returns maxpos), so
/// `interval<posit>` returns the finite `[-maxpos, maxpos]`: saturated, not
/// unbounded, and therefore not an enclosure of the true set under IEEE 1788
/// semantics.
///
/// Only reachable through division, so this restricts `solve` (issue #12
/// Phase 5) and nothing in dot / nrm2 / ger / gemv / gemm / rot. Probe it rather
/// than assuming, so a study can skip element types that cannot express the
/// answer instead of silently reporting a bounded one.
template <typename Scalar>
bool can_represent_unbounded() {
    using std::isinf;
    const Scalar inf = std::numeric_limits<Scalar>::infinity();
    if (!std::numeric_limits<Scalar>::has_infinity) return false;
    // The decisive test is not the has_infinity flag but whether the value is
    // actually distinguishable from the largest finite magnitude.
    return inf > std::numeric_limits<Scalar>::max();
}

// ---------------------------------------------------------------------------
// Reduction helper
// ---------------------------------------------------------------------------

/// Widen a vector of point values into degenerate intervals [v, v]. A reduction
/// over these isolates the ARITHMETIC's width growth: the inputs contribute no
/// uncertainty, so every bit of the result's width was manufactured by the
/// operator. This is the cleanest Phase-0/1 configuration.
template <typename Interval, typename Scalar = typename Interval::value_type>
std::vector<Interval> degenerate(const std::vector<double>& v) {
    std::vector<Interval> out;
    out.reserve(v.size());
    for (double x : v) out.emplace_back(Interval(static_cast<Scalar>(x)));
    return out;
}

} // namespace sw::mp_blas
