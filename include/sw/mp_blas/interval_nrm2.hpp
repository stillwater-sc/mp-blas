#pragma once
// mp-blas -- interval nrm2: a sum of SQUARES, not a dot product with itself
// (issue #12, Phase 1's open item).
//
// THE PROBLEM. Computing nrm2 as sqrt(dot(x,x)) hands the same interval to both
// arguments, and interval arithmetic cannot know they are the same object. For an
// element X = [a,b] that straddles zero, `dot` therefore evaluates the general
// product X*X, whose lower endpoint is min(a*b, b*a) = a*b < 0 -- whereas the
// true square X^2 is [0, max(a^2,b^2)] and can never be negative.
//
// This is the DEPENDENCY PROBLEM: interval arithmetic overestimates whenever a
// variable occurs more than once in an expression. It is not an accumulation
// error, so no accumulator fixes it -- an exact quire will faithfully and exactly
// accumulate the wrong (over-wide) corner products. It is the one obstacle in
// docs/interval-blas-study.md that is mathematical rather than structural.
//
// The fix is to stop asking `dot` and evaluate the square directly, which is what
// this header does. Measured effect (see applications/level1/interval_nrm2_study):
// with every element straddling zero, sqrt(dot(x,x)) returns an enclosure exactly
// TWICE as wide as the truth with a negative lower bound; at 50% straddling the
// certified LOWER BOUND on ||x|| drops from 1.58 to 0.71.
//
// A nice property of the square that the general product does not have: the
// straddling case needs only a comparison of MAGNITUDES (|a| vs |b|), which is
// exact, where general interval multiplication needs a comparison of PRODUCTS and
// therefore a scratch quire (see interval_quire_accumulator.hpp). Squaring is
// monotone in |x|, so no rounding can flip the choice.

#include <cstddef>

#include <mtl/math/interval_quire_accumulator.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/dot.hpp>

#include <sw/mp_blas/interval_containment.hpp>

#include <universal/number/interval/interval.hpp>

namespace sw::mp_blas {

/// The exact square of an interval: X^2 = { x*x : x in X }.
///
///   a >= 0        -> [a^2, b^2]
///   b <= 0        -> [b^2, a^2]
///   a < 0 < b     -> [0,   max(a^2, b^2)]
///
/// Note the third case: the lower bound is exactly ZERO, not a*b. That is the
/// whole difference from the general product, and it is why nrm2 must not be
/// routed through dot(x,x).
template <typename Scalar>
sw::universal::interval<Scalar> square(const sw::universal::interval<Scalar>& X) {
    using I = sw::universal::interval<Scalar>;
    const Scalar a = X.lower(), b = X.upper(), zero(0);

    Scalar lo, hi;
    if (a >= zero || b <= zero) {
        // Endpoints share a sign, so the general product already picks the right
        // corners: [a^2, b^2] or [b^2, a^2].
        const I s = I(a, b) * I(a, b);
        lo = s.lower();
        hi = s.upper();
    } else {
        // Straddling: the general product would give a*b < 0 as the lower
        // endpoint. The true lower endpoint is exactly zero, and the upper is the
        // larger magnitude squared -- selected by an EXACT magnitude comparison,
        // since squaring is monotone in |x| and no rounding can flip the choice.
        const Scalar m = (-a > b) ? -a : b;    // negation is exact
        lo = zero;
        hi = (I(m) * I(m)).upper();
    }

    // A square is never negative, so a negative lower bound can be raised to zero
    // and the result is still a valid -- and tighter -- enclosure. This is not
    // hypothetical: outward rounding drives an EXACT zero lower endpoint just
    // below zero (round_down(0) is the next value toward -inf), so [0,m]*[0,m]
    // and even [0,b]*[0,b] come back with a tiny negative lower bound.
    if (lo < zero) lo = zero;
    return I(lo, hi);
}

/// Accumulate X^2 into a quire pair, exactly.
///
/// Every term is non-negative, so the sum of squares is the `positive` regime of
/// issue #9 -- cond is at its floor of 2 and there is no cancellation. The exact
/// accumulator is therefore not fighting conditioning here; it removes the n*u
/// drift of a long reduction, which is what a large nrm2 actually suffers from.
template <typename Scalar>
void add_square(sw::universal::quire<Scalar>& qlo,
                sw::universal::quire<Scalar>& qhi,
                const sw::universal::interval<Scalar>& X) {
    using sw::universal::quire_mul;
    const Scalar a = X.lower(), b = X.upper(), zero(0);

    if (a >= zero) {
        qlo += quire_mul(a, a);
        qhi += quire_mul(b, b);
    } else if (b <= zero) {
        qlo += quire_mul(b, b);
        qhi += quire_mul(a, a);
    } else {
        // Lower endpoint contributes exactly 0. Upper is the larger magnitude
        // squared -- chosen by an EXACT magnitude comparison, since squaring is
        // monotone in |x| and no rounding can flip it.
        const Scalar m = (-a > b) ? -a : b;
        qhi += quire_mul(m, m);
    }
}

/// nrm2 the way it is usually written: sqrt(dot(x, x)).
///
/// Provided so the study and tests can measure what the idiomatic formulation
/// costs, rather than asserting it.
template <typename Interval, typename Vector>
Interval nrm2_via_dot(const Vector& x) {
    using sw::universal::sqrt;
    const auto s = mtl::dot(x, x);
    const Interval sd(static_cast<double>(s.lower()), static_cast<double>(s.upper()));
    return sqrt(sd);
}

/// nrm2 as a sum of squares: dependency-free per element, accumulated exactly in
/// a quire pair, with a single outward rounding before the (outward-rounded)
/// square root.
template <typename Interval, typename Scalar, typename Vector>
Interval nrm2_exact(const Vector& x) {
    using sw::universal::quire;
    using sw::universal::sqrt;

    quire<Scalar> qlo{}, qhi{};
    qlo.clear();
    qhi.clear();
    for (std::size_t i = 0; i < static_cast<std::size_t>(x.size()); ++i)
        add_square(qlo, qhi, x[i]);

    double lo = qlo.template convert_to<double>();
    double hi = qhi.template convert_to<double>();
    if (!qlo.iszero()) lo = round_down(lo);
    if (!qhi.iszero()) hi = round_up(hi);
    if (lo < 0.0) lo = 0.0;                    // a sum of squares is never negative

    return sqrt(Interval(lo, hi));
}

} // namespace sw::mp_blas
