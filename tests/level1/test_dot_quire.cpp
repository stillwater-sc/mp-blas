// mp-blas level-1 test: exact dot product via a Universal quire super-
// accumulator (the MTL5 + Universal coupling that must not live in MTL5).
//
// Constructs a dot product designed to expose accumulation error: a large term
// followed by many small terms whose running sum is lost to rounding when the
// products are accumulated in the element type itself ("swamping"). The quire
// accumulates all products exactly (posit/cfloat) and rounds once, so it must
// recover the small terms that naive same-precision accumulation drops.
//
// Element types: posit, cfloat, and lns. Two invariants are checked:
//   * ALWAYS: the quire is no worse than native same-precision accumulation
//     (holds for every element type; robust to the known Universal quire
//     limitations in cfloat<16,5>/lns -- see stillwater-sc/universal#1202/#1203).
//   * EXACT: for the types whose quire is genuinely exact (posit, wide cfloat),
//     the quire recovers the swamped terms to (near) zero error.
//
// Returns non-zero on failure (no external test framework).
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>

#include <mtl/math/quire_accumulator.hpp>   // must precede dot.hpp's use of the traits
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/dot.hpp>

#include <universal/number/posit/posit.hpp>
#include <universal/number/cfloat/cfloat.hpp>
#include <universal/number/lns/lns.hpp>

namespace {

// x = [1, eps, eps, ..., eps], y = all ones. Exact dot = 1 + n_small*eps, with
// eps below the ULP of 1.0 so native accumulation near 1.0 swamps the small
// terms. Returns {native_err, quire_err} vs the exact value, in double.
template <typename T>
std::pair<double, double> swamp_errors() {
    using Quire = sw::universal::quire<T>;
    constexpr std::size_t n_small = 64;
    constexpr std::size_t n = 1 + n_small;
    const double eps = 1.0 / 4096.0;   // below the ULP of 1.0 in the 16-bit types

    mtl::vec::dense_vector<T> x(n, T(0)), y(n, T(1));
    x(0) = T(1);
    for (std::size_t i = 1; i < n; ++i) x(static_cast<int>(i)) = T(eps);

    const double exact = 1.0 + n_small * eps;
    double native = static_cast<double>(mtl::dot(x, y));
    double quire  = static_cast<double>(mtl::dot<Quire, T>(x, y));
    return {std::abs(native - exact), std::abs(quire - exact)};
}

// Invariant that holds for every element type: the quire is no worse than
// native same-precision accumulation.
template <typename T>
bool quire_no_worse_than_native(const std::string& name) {
    auto [native_err, quire_err] = swamp_errors<T>();
    std::cout << name << ": native_err=" << native_err
              << " quire_err=" << quire_err << '\n';
    if (quire_err > native_err) {
        std::cerr << name << ": quire dot is LESS accurate than native -- adapter broken\n";
        return false;
    }
    return true;
}

// The exact-quire types: the quire delivers (near) zero error. For the 16-bit
// types this case also swamps native, so exactness additionally means a strict
// win over native; the wide types are already exact natively for this eps, so
// only exactness is asserted.
template <typename T>
bool quire_exact(const std::string& name) {
    auto [native_err, quire_err] = swamp_errors<T>();
    (void) native_err;
    if (quire_err > 1e-12) {
        std::cerr << name << ": quire dot not exact (err " << quire_err << ")\n";
        return false;
    }
    return true;
}

} // namespace

int main() {
    int failures = 0;

    // No-worse-than-native holds for every wired element type (robust to the
    // known Universal quire limitations in cfloat<16,5> / lns).
    if (!quire_no_worse_than_native<sw::universal::posit<16, 2>>("posit<16,2>")) ++failures;
    if (!quire_no_worse_than_native<sw::universal::cfloat<16, 5>>("cfloat<16,5>")) ++failures;
    if (!quire_no_worse_than_native<sw::universal::lns<16, 8>>("lns<16,8>")) ++failures;

    // Exactness holds where Universal's quire is exact today: posits (any width)
    // and wide cfloat. posit<16,2>/cfloat<16,5> above already show the strict
    // win over native (quire_err 0 vs native_err 0.0156) on this swamping case.
    if (!quire_exact<sw::universal::posit<16, 2>>("posit<16,2>")) ++failures;
    if (!quire_exact<sw::universal::posit<32, 2>>("posit<32,2>")) ++failures;
    if (!quire_exact<sw::universal::cfloat<32, 8>>("cfloat<32,8>")) ++failures;

    if (failures == 0) std::cout << "mp-blas level-1 quire test passed\n";
    return failures == 0 ? 0 : 1;
}
