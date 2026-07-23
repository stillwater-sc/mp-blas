// mp-blas level-1 test: exact dot product via a Universal quire super-
// accumulator (the MTL5 + Universal coupling that must not live in MTL5).
//
// Constructs a dot product designed to expose accumulation error: a large term
// followed by many small terms whose running sum is lost to rounding when the
// products are accumulated in the posit itself ("swamping"). The quire
// accumulates all products exactly and rounds once, so it must recover the
// small terms that naive same-precision accumulation drops.
//
// Returns non-zero on failure (no external test framework).
#include <cmath>
#include <cstddef>
#include <iostream>

#include <mtl/math/quire_accumulator.hpp>   // must precede dot.hpp's use of the traits
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/dot.hpp>

#include <universal/number/posit/posit.hpp>

namespace {

bool quire_recovers_swamped_terms() {
    using Posit = sw::universal::posit<16, 2>;
    using Quire = sw::universal::quire<Posit>;

    // x = [1, eps, eps, ..., eps], y = all ones. Exact dot = 1 + n_small*eps.
    // With eps below the ULP of 1.0 in posit<16,2>, native accumulation adds
    // each eps to a running sum near 1.0 and loses it; the quire keeps them.
    constexpr std::size_t n_small = 64;
    constexpr std::size_t n = 1 + n_small;
    const double eps = 1.0 / 4096.0;   // well below the ULP of 1.0 in posit<16,2>

    mtl::vec::dense_vector<Posit> x(n, Posit(0)), y(n, Posit(1));
    x(0) = Posit(1);
    for (std::size_t i = 1; i < n; ++i) x(static_cast<int>(i)) = Posit(eps);

    const double exact = 1.0 + n_small * eps;

    // native: accumulate products in the posit itself
    double native = static_cast<double>(mtl::dot(x, y));
    // quire: accumulate exactly, round once to the posit
    double quire = static_cast<double>(mtl::dot<Quire, Posit>(x, y));

    std::cout << "exact         = " << exact << '\n'
              << "native posit  = " << native << " (err " << std::abs(native - exact) << ")\n"
              << "quire  posit  = " << quire  << " (err " << std::abs(quire  - exact) << ")\n";

    // The quire must be at least as accurate, and materially better here:
    // native swamps the small terms, the quire recovers them.
    if (std::abs(quire - exact) > std::abs(native - exact)) {
        std::cerr << "quire dot is LESS accurate than native -- adapter broken\n";
        return false;
    }
    if (!(std::abs(quire - exact) < std::abs(native - exact))) {
        std::cerr << "quire dot did not improve on native on a swamping case\n";
        return false;
    }
    return true;
}

} // namespace

int main() {
    int failures = 0;
    if (!quire_recovers_swamped_terms()) ++failures;
    if (failures == 0) std::cout << "mp-blas level-1 quire test passed\n";
    return failures == 0 ? 0 : 1;
}
