#pragma once
// mp-blas -- accumulator_traits bridge to Universal's quire super-accumulators.
//
// A quire is a fixed-size super-accumulator that captures sums of products of a
// fixed-size arithmetic type EXACTLY, deferring rounding to a single conversion
// at the end. The concept applies to any fixed-size arithmetic -- integer,
// fixpnt, posit, lns, cfloat -- not to posits specifically; posit is simply the
// first number system wired up here.
//
// Specializes mtl::math::accumulator_traits<Acc, Value> so that MTL5's
// mixed-precision BLAS kernels -- dot()/mult()/norms(), see
// mtl/math/accumulator_traits.hpp -- accumulate inner products in an exact
// quire and round once, instead of accumulating in the value type's arithmetic
// directly. This is the "super-accumulator" configuration (single-event
// rounding of an exact sum of products).
//
// This header lives in mp-blas, NOT in MTL5: MTL5 is the general linear-algebra
// layer and never depends on Universal, so all MTL5 + Universal coupling
// belongs here in the composition layer. It keeps the <mtl/math/...> path so
// the specialization sits next to the generic accumulator_traits it extends. No
// opt-in macro is needed -- if you are building against sw::mp_blas, the
// composition is what you asked for. (The adapter mirrors the one in the sister
// repo mp-iterative; kept local so mp-blas has no cross-repo header dependency.)
//
// Verified against the Universal source (github.com/stillwater-sc/universal,
// current `posit` module):
//   - quire_mul(posit<nbits,es,bt>, posit<nbits,es,bt>) returns a blocktriple
//     (include/sw/universal/number/posit/fdp.hpp)
//   - quire<NumberType, capacity, LimbType> is parameterized on the VALUE type
//     directly (capacity defaults from quire_traits<NumberType>), not on
//     nbits/es separately (include/sw/universal/number/quire/quire_impl.hpp)
//   - conversion back to a value is q.convert_to<TargetType>()
//
// Usage (round the exact accumulation out to the element type -- pass BOTH the
// quire accumulator AND the result type, since dot's Result defaults to the
// accumulator, which here is the quire, not a number):
//
//   #include <mtl/math/quire_accumulator.hpp>
//   using Posit = sw::universal::posit<32,2>;
//   using Quire = sw::universal::quire<Posit>;      // capacity auto-derived
//   Posit d = mtl::dot<Quire, Posit>(x, y);          // exact accumulate, round once
//   double dd = mtl::dot<Quire, double>(x, y);       // ... or round out to double
//
// Element types wired up here: posit, cfloat, lns (each admits the generic
// quire<NumberType> via its quire_traits). Universal's aggregator headers
// (cfloat.hpp / lns.hpp) do NOT pull in their own fdp.hpp the way posit.hpp
// does, so quire_mul is otherwise unavailable -- we include the fdp headers
// explicitly below (stillwater-sc/universal#1201).
//
// Known Universal limitations these element types inherit (the study
// applications/level1/dot_accumulator_study quantifies them):
//   - posit           : exact -- the quire delivers true single-event rounding.
//   - cfloat (no subn.): quire_traits<cfloat>::radix_point is undersized, so a
//                        quire dot floors at ~2^-radix_point (~1e-9 for
//                        cfloat<16,5>) instead of exact (stillwater-sc/universal#1202).
//   - lns             : quire_mul routes the product through double, so the lns
//                        quire is NOT exact and can be worse than a promoted
//                        double accumulator (stillwater-sc/universal#1203).

#include <universal/number/posit/posit.hpp>
#include <universal/number/posit/fdp.hpp>     // quire_mul(posit, posit) (posit.hpp already pulls this)
#include <universal/number/cfloat/cfloat.hpp>
#include <universal/number/cfloat/fdp.hpp>    // quire_mul(cfloat, cfloat) -- NOT pulled by cfloat.hpp
#include <universal/number/lns/lns.hpp>
#include <universal/number/lns/fdp.hpp>       // quire_mul(lns, lns) -- NOT pulled by lns.hpp

#include <mtl/math/accumulator_traits.hpp>

namespace mtl::math {

/// accumulator_traits specialization for the posit instance of the quire
/// pattern: Acc is a Universal quire parameterized on posit<nbits,es,bt>,
/// matching quire's real signature quire<NumberType, capacity, LimbType>.
/// Further specializations (cfloat, lns, fixpnt, integer quires) follow the
/// same shape as Universal exposes them. add_product uses quire_mul's exact
/// (unrounded) blocktriple product; value() rounds out once at the end via
/// convert_to<Result>() (the single-rounding semantics the generic template's
/// docstring describes).
///
/// NOTE: posit's third template parameter `bt` (limb type) defaults to
/// std::uint8_t in Universal's posit_impl.hpp. This specialization pattern binds
/// to that default -- if your project overrides `bt`, extend the pattern to
/// include it explicitly.
template <unsigned nbits, unsigned es, unsigned capacity, typename LimbType>
struct accumulator_traits<sw::universal::quire<sw::universal::posit<nbits, es>, capacity, LimbType>,
                           sw::universal::posit<nbits, es>> {
    using Value = sw::universal::posit<nbits, es>;
    using Acc   = sw::universal::quire<Value, capacity, LimbType>;

    static void clear(Acc& a) { a.clear(); }

    static void assign(Acc& a, const Value& v) {
        a.clear();
        a += sw::universal::quire_mul(v, Value(1));
    }

    template <typename Result = Value>
    static Result value(const Acc& a) {
        return a.template convert_to<Result>();
    }

    /// The core operation: a += m * v, accumulated exactly in the quire via
    /// Universal's quire_mul (returns an unrounded blocktriple product) -- no
    /// rounding until value() is called.
    static void add_product(Acc& a, const Value& m, const Value& v) {
        a += sw::universal::quire_mul(m, v);
    }
};

/// accumulator_traits specialization for the cfloat instance of the quire
/// pattern. cfloat uses blocktriple as its internal arithmetic engine, so
/// quire_mul returns the exact (unrounded) full-precision MUL blocktriple; the
/// quire accumulates a true exact sum of products and value() rounds out once.
template <unsigned nbits, unsigned es, typename bt,
          bool hasSubnormals, bool hasMaxExpValues, bool isSaturating,
          unsigned capacity, typename LimbType>
struct accumulator_traits<
        sw::universal::quire<sw::universal::cfloat<nbits, es, bt, hasSubnormals, hasMaxExpValues, isSaturating>,
                             capacity, LimbType>,
        sw::universal::cfloat<nbits, es, bt, hasSubnormals, hasMaxExpValues, isSaturating>> {
    using Value = sw::universal::cfloat<nbits, es, bt, hasSubnormals, hasMaxExpValues, isSaturating>;
    using Acc   = sw::universal::quire<Value, capacity, LimbType>;

    static void clear(Acc& a) { a.clear(); }

    static void assign(Acc& a, const Value& v) {
        a.clear();
        a += sw::universal::quire_mul(v, Value(1));
    }

    template <typename Result = Value>
    static Result value(const Acc& a) {
        return a.template convert_to<Result>();
    }

    static void add_product(Acc& a, const Value& m, const Value& v) {
        a += sw::universal::quire_mul(m, v);
    }
};

/// accumulator_traits specialization for the lns (logarithmic number system)
/// instance of the quire pattern.
///
/// CAVEAT: Universal's lns quire_mul currently forms the product by converting
/// both operands to double and multiplying there (number/lns/fdp.hpp), so the
/// per-term product is exact only to ~53 significand bits -- an lns whose
/// linear-domain product needs more than that is NOT accumulated exactly. The
/// quire still eliminates cross-term accumulation error (the sum of products is
/// exact given the double-rounded products), which is the dominant error on
/// long reductions, but this is not the single-event-rounding guarantee posit /
/// cfloat provide. Tracked upstream (stillwater-sc/universal#1203). `bt`
/// defaults to std::uint8_t and the pattern binds to Universal's single
/// optional `xtra`.
template <unsigned nbits, unsigned rbits, typename bt, auto... xtra,
          unsigned capacity, typename LimbType>
struct accumulator_traits<
        sw::universal::quire<sw::universal::lns<nbits, rbits, bt, xtra...>, capacity, LimbType>,
        sw::universal::lns<nbits, rbits, bt, xtra...>> {
    using Value = sw::universal::lns<nbits, rbits, bt, xtra...>;
    using Acc   = sw::universal::quire<Value, capacity, LimbType>;

    static void clear(Acc& a) { a.clear(); }

    static void assign(Acc& a, const Value& v) {
        a.clear();
        a += sw::universal::quire_mul(v, Value(1));
    }

    template <typename Result = Value>
    static Result value(const Acc& a) {
        return a.template convert_to<Result>();
    }

    static void add_product(Acc& a, const Value& m, const Value& v) {
        a += sw::universal::quire_mul(m, v);
    }
};

} // namespace mtl::math
