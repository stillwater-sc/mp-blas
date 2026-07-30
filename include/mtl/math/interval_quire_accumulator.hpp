#pragma once
// mp-blas -- accumulator_traits bridge for an INTERVAL quire: exact interval
// accumulation for MTL5's mixed-precision reductions (issue #12, Phase 1).
//
// WHY. A naive interval reduction rounds both endpoints outward at every step,
// so the enclosure widens monotonically with the reduction length -- the
// enclosure tracks the error BOUND rather than the error. Phase 0 measured this:
// at n = 512 over degenerate inputs, a benign `double` dot already delivers only
// 13.3 certified digits, and the `kahan` regime yields an enclosure spanning 47x
// its own midpoint. Containment holds; the answer is useless.
//
// An exact accumulator removes the mechanism entirely. Accumulate each endpoint
// in its own quire -- exactly, no rounding at all during the reduction -- and
// round outward exactly ONCE on the way out. The enclosure width then reflects
// only the final single rounding, independent of n and of the conditioning.
// That is Kulisch's exact-scalar-product argument applied to its original
// target: interval arithmetic with tight enclosures (see
// docs/dot-product-characterization.md section 1).
//
// This header lives in mp-blas, NOT in MTL5: MTL5 is the general linear-algebra
// layer and never depends on Universal, so all MTL5 + Universal coupling belongs
// in the composition layer. It keeps the <mtl/math/...> path so it sits next to
// the generic accumulator_traits it extends, mirroring quire_accumulator.hpp.
//
// Usage (pass BOTH the accumulator AND the result type, since dot's Result
// defaults to the accumulator, which here is a quire pair, not a number):
//
//   #include <mtl/math/interval_quire_accumulator.hpp>
//   using P  = sw::universal::posit<32,2>;
//   using I  = sw::universal::interval<P>;
//   using QA = sw::mp_blas::interval_quire<P>;
//   I d = mtl::dot<QA, I>(x, y);          // exact accumulate, one outward round
//
// PRECONDITION on the element type: the endpoint selection below is exact only
// if the Scalar's comparison and negation are exact and sign-symmetric, which
// holds for posit/cfloat/lns. It also requires quire_mul(Scalar, Scalar) to be
// exact -- true for posit and cfloat, NOT for lns (stillwater-sc/universal#1203
// routes the lns product through double), so an lns interval quire inherits that
// non-exactness. Containment is preserved either way; only tightness suffers.

#include <universal/number/posit/posit.hpp>
#include <universal/number/cfloat/cfloat.hpp>
#include <universal/number/interval/interval.hpp>

#include <mtl/math/accumulator_traits.hpp>

#include <sw/mp_blas/interval_containment.hpp>   // round_down / round_up

namespace sw::mp_blas {

/// A pair of quires: one accumulating the lower endpoint, one the upper. Both
/// exact, so the reduction introduces no rounding whatsoever; the single outward
/// rounding happens in accumulator_traits::value().
template <typename Scalar>
struct interval_quire {
    using scalar_type = Scalar;
    using quire_type  = sw::universal::quire<Scalar>;

    quire_type lo{};
    quire_type hi{};
};

namespace detail {

/// Which corner products bound the product of two intervals.
///
/// Interval multiplication picks its endpoints from the four corner products
/// a*c, a*d, b*c, b*d. Which corners win is determined by the operand SIGNS
/// alone in eight of the nine sign cases -- no comparison of products is needed,
/// so each chosen corner can be handed to quire_mul and accumulated exactly.
///
/// The ninth case, both operands straddling zero, genuinely needs a comparison:
/// lo = min(a*d, b*c) and hi = max(a*c, b*d). Comparing ROUNDED products there
/// could pick the wrong corner when the two are nearly equal, and picking the
/// wrong corner yields an enclosure that is too NARROW -- a containment
/// violation. So that case compares the two candidates EXACTLY, in a scratch
/// quire, and is the only path that pays for one.
template <typename Scalar>
struct corners {
    Scalar lo_x, lo_y;   ///< factors whose exact product is the lower endpoint
    Scalar hi_x, hi_y;   ///< factors whose exact product is the upper endpoint
};

/// Exact predicate: is x1*y1 < x2*y2? Evaluated in a quire, so no rounding can
/// flip the comparison.
template <typename Scalar>
bool product_less(const Scalar& x1, const Scalar& y1,
                  const Scalar& x2, const Scalar& y2) {
    sw::universal::quire<Scalar> t{};
    t.clear();
    t += sw::universal::quire_mul(x1, y1);
    t -= sw::universal::quire_mul(x2, y2);
    return !t.iszero() && t.isneg();
}

template <typename Scalar>
corners<Scalar> select_corners(const Scalar& a, const Scalar& b,
                               const Scalar& c, const Scalar& d) {
    const Scalar zero(0);

    if (a >= zero) {                       // X >= 0
        if (c >= zero)      return { a, c, b, d };   // Y >= 0
        else if (d <= zero) return { b, c, a, d };   // Y <= 0
        else                return { b, c, b, d };   // Y straddles
    }
    if (b <= zero) {                       // X <= 0
        if (c >= zero)      return { a, d, b, c };   // Y >= 0
        else if (d <= zero) return { b, d, a, c };   // Y <= 0
        else                return { a, d, a, c };   // Y straddles
    }
    // X straddles zero
    if (c >= zero)      return { a, d, b, d };
    else if (d <= zero) return { b, c, a, c };

    // Both straddle: the one case needing exact comparisons.
    corners<Scalar> r{};
    if (product_less(a, d, b, c)) { r.lo_x = a; r.lo_y = d; }
    else                          { r.lo_x = b; r.lo_y = c; }
    if (product_less(a, c, b, d)) { r.hi_x = b; r.hi_y = d; }
    else                          { r.hi_x = a; r.hi_y = c; }
    return r;
}

} // namespace detail
} // namespace sw::mp_blas

namespace mtl::math {

/// accumulator_traits specialization for an interval_quire over an
/// interval<Scalar> value type. The reduction is exact; value() performs the
/// single outward rounding that makes the result a valid enclosure.
template <typename Scalar>
struct accumulator_traits<sw::mp_blas::interval_quire<Scalar>,
                          sw::universal::interval<Scalar>> {
    using Value = sw::universal::interval<Scalar>;
    using Acc   = sw::mp_blas::interval_quire<Scalar>;

    static void clear(Acc& a) { a.lo.clear(); a.hi.clear(); }

    static void assign(Acc& a, const Value& v) {
        clear(a);
        a.lo += sw::universal::quire_mul(v.lower(), Scalar(1));
        a.hi += sw::universal::quire_mul(v.upper(), Scalar(1));
    }

    /// Round the exact accumulation out to an interval<S>, ONCE.
    ///
    /// The outward step is unconditional (except for an exactly-zero quire)
    /// rather than applied only when the conversion was inexact. That is
    /// deliberate: quire::convert_to<T>() is documented to go through double and
    /// is limited to 53 significand bits, so for a quire holding more than that
    /// -- exactly the interesting case -- neither the fact nor the direction of
    /// its rounding is observable from the outside. Stepping unconditionally
    /// costs at most one ulp per endpoint and guarantees containment; a
    /// conditional (tighter) version needs an exactness query on the quire,
    /// mirroring the Stage-1 / Stage-2 split Universal used for
    /// stillwater-sc/universal#1234 and #1248.
    template <typename Result = Value>
    static Result value(const Acc& a) {
        using S = typename Result::value_type;

        S lo = a.lo.template convert_to<S>();
        S hi = a.hi.template convert_to<S>();
        if (!a.lo.iszero()) lo = sw::mp_blas::round_down(lo);
        if (!a.hi.iszero()) hi = sw::mp_blas::round_up(hi);
        return Result(lo, hi);
    }

    /// The core operation: accumulate the product interval's two endpoints into
    /// their own quires, each as an EXACT product via quire_mul. No rounding
    /// occurs here at all -- that is the whole point.
    static void add_product(Acc& a, const Value& m, const Value& v) {
        const auto k = sw::mp_blas::detail::select_corners(
            m.lower(), m.upper(), v.lower(), v.upper());
        a.lo += sw::universal::quire_mul(k.lo_x, k.lo_y);
        a.hi += sw::universal::quire_mul(k.hi_x, k.hi_y);
    }
};

} // namespace mtl::math
