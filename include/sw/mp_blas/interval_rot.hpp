#pragma once
// mp-blas -- plane rotation over interval elements (issue #12, Phase 3).
//
// BLAS `rot` applies a plane rotation to a pair of values:
//
//     [ x ]  <-  [  c  s ] [ x ]
//     [ y ]      [ -s  c ] [ y ]
//
// MTL5 has Givens rotations only in the GMRES-specific form
// (mtl/operation/givens.hpp, apply_stored_rotation), so the two-value kernel is
// spelled out here to be measured directly.
//
// WHY THIS OPERATOR. Phases 1 and 2 established that exact accumulation collapses
// the enclosure width of a reduction. `rot` is where that should STOP being true,
// for two independent reasons, and confirming a predicted negative is what bounds
// the claim:
//
//   1. The reduction is length TWO. Each output component is a 2-term dot
//      product, so there is essentially no accumulation error for an exact
//      accumulator to remove -- the quire's whole value is eliminating n-term
//      drift, and here n = 2.
//
//   2. The dominant error is WRAPPING, which is not an arithmetic error at all.
//      A rotation maps an axis-aligned box to a rotated box, and the result must
//      be stored as an axis-aligned box again -- so the enclosure is the hull of
//      a tilted rectangle. The hull of a box of width w rotated by theta has
//      width w*(|cos theta| + |sin theta|), which exceeds w for every angle that
//      is not a multiple of 90 degrees, peaking at sqrt(2) at 45 degrees. Applied
//      k times this compounds GEOMETRICALLY: (|c| + |s|)^k.
//
// No accumulator can fix (2): the loss happens in the representation, when a
// tilted set is forced back into an axis-aligned interval pair, not in the sum.
// The prediction is therefore that an exact accumulator changes the CONSTANT but
// not the RATE -- see docs/interval-blas-study.md.

#include <cstddef>

#include <mtl/math/interval_quire_accumulator.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/dot.hpp>

#include <universal/number/interval/interval.hpp>

namespace sw::mp_blas {

/// Naive interval `rot`: each component evaluated with interval arithmetic, so
/// every product and sum rounds outward.
template <typename Scalar>
void rot_naive(const sw::universal::interval<Scalar>& c,
               const sw::universal::interval<Scalar>& s,
               sw::universal::interval<Scalar>& x,
               sw::universal::interval<Scalar>& y) {
    using I = sw::universal::interval<Scalar>;
    const I zero(Scalar(0));
    const I t  = c * x + s * y;
    const I ny = (zero - s) * x + c * y;
    x = t;
    y = ny;
}

/// Exact-accumulation interval `rot`: each component is a 2-term dot product
/// accumulated in a quire pair and rounded outward exactly once.
///
/// This is deliberately the SAME kernel with only the accumulation swapped, so
/// any difference in the measured growth is attributable to accumulation alone.
template <typename Scalar>
void rot_quire(const sw::universal::interval<Scalar>& c,
               const sw::universal::interval<Scalar>& s,
               sw::universal::interval<Scalar>& x,
               sw::universal::interval<Scalar>& y) {
    using I  = sw::universal::interval<Scalar>;
    using QA = interval_quire<Scalar>;
    const I zero(Scalar(0));

    mtl::vec::dense_vector<I> row(2, zero), col(2, zero);
    col[0] = x; col[1] = y;

    row[0] = c;          row[1] = s;
    const I t = mtl::dot<QA, I>(row, col);

    row[0] = zero - s;   row[1] = c;
    const I ny = mtl::dot<QA, I>(row, col);

    x = t;
    y = ny;
}

/// The wrapping growth factor for one rotation: |c| + |s|.
///
/// This is the width multiplier the axis-aligned hull of a rotated box picks up,
/// independent of the arithmetic used. It is 1 only at multiples of 90 degrees
/// (where the rotation maps axes to axes and no wrapping occurs) and peaks at
/// sqrt(2) at 45 degrees.
inline double wrapping_factor(double c, double s) {
    return std::abs(c) + std::abs(s);
}

} // namespace sw::mp_blas
