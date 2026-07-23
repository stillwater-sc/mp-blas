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

#include <universal/number/posit/posit.hpp>

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

} // namespace mtl::math
