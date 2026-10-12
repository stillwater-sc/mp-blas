// mp-blas level-1 test: exact dot product via a Universal quire super-
// accumulator (the MTL5 + Universal coupling that must not live in MTL5).
//
// Constructs a dot product designed to expose accumulation error: a large term
// followed by many small terms whose running sum is lost to rounding when the
// products are accumulated in the element type itself ("swamping"). The quire
// accumulates all products exactly (posit/cfloat) and rounds once, so it must
// recover the small terms that naive same-precision accumulation drops.
//
// Element types: posit, bposit, cfloat, and lns. Two invariants are checked:
//   * ALWAYS: the quire is no worse than native same-precision accumulation
//     (holds for every element type; robust to the known Universal quire
//     limitations in cfloat<16,5>/lns -- see stillwater-sc/universal#1202/#1203).
//   * EXACT: for the types whose quire is genuinely exact -- posit and bposit at any width,
//     wide cfloat, and ANY subnormal-enabled cfloat -- the quire recovers the
//     swamped terms to (near) zero error. Subnormals widen
//     quire_traits<cfloat>::radix_point enough to hold a product's full
//     significand, which is the workaround for universal#1202; a narrow cfloat
//     WITHOUT subnormals still only gets the no-worse-than-native guarantee.
//
// bposit additionally checks that the bridge rounds exactly as Universal's own
// fdp does: mtl::dot<quire<bposit>, bposit> must be bit-identical to
// sw::universal::fdp on the same vectors.
//
// Returns non-zero on failure (no external test framework).
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <mtl/math/quire_accumulator.hpp>   // must precede dot.hpp's use of the traits
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/dot.hpp>

#include <universal/number/posit/posit.hpp>
#include <universal/number/bposit/bposit.hpp>
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

// A DISCRIMINATING case for universal#1202. swamp_errors above uses exact
// powers of two (1 and 1/4096), whose products are single bits -- an undersized
// radix_point truncates nothing, so that construction cannot tell a correctly
// sized quire from a short one (and indeed cfloat<16,5> reports 0 there). Here
// the values carry full significands, so each product has ~2*fbits significand
// bits extending BELOW its leading bit, which is exactly what a short
// radix_point drops. Returns the quire's relative error vs a long-double
// reference over the same quantized values.
template <typename T>
double dense_quire_relerr() {
    using Quire = sw::universal::quire<T>;
    constexpr std::size_t n = 512;
    mtl::vec::dense_vector<T> x(n, T(0)), y(n, T(0));
    long double ref = 0.0L;
    for (std::size_t i = 0; i < n; ++i) {
        // deterministic, deliberately non-dyadic values in (-1, 1)
        std::uint64_t z = (i + 1) * 0x9E3779B97F4A7C15ull;
        z ^= z >> 30; z *= 0xBF58476D1CE4E5B9ull;
        z ^= z >> 27; z *= 0x94D049BB133111EBull; z ^= z >> 31;
        const double xv = 2.0 * (static_cast<double>(z >> 11) * 0x1.0p-53) - 1.0;
        z *= 0xD6E8FEB86659FD93ull; z ^= z >> 32;
        const double yv = 2.0 * (static_cast<double>(z >> 11) * 0x1.0p-53) - 1.0;

        const T xi(xv), yi(yv);
        x[i] = xi; y[i] = yi;
        ref += static_cast<long double>(static_cast<double>(xi))
             * static_cast<long double>(static_cast<double>(yi));
    }
    const double q = mtl::dot<Quire, double>(x, y);
    const long double denom = std::abs(ref) > 0.0L ? std::abs(ref) : 1.0L;
    return static_cast<double>(std::abs(static_cast<long double>(q) - ref) / denom);
}

// The bridge's round-to-Value path (quire_resolve) must agree bit for bit with
// Universal's own fused dot product. Random encodings over the whole range, so
// products reach both saturation ends and the quire's lowest bits.
template <typename T>
bool bridge_matches_fdp(const std::string& name) {
    using Quire = sw::universal::quire<T>;
    std::uint64_t state = 0x853C49E6748FEA9Bull;
    auto next = [&]() {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        T v; v.setbits(state >> (64 - T::nbits));
        if (v.isnar()) v.setzero();
        return v;
    };
    int mismatches = 0;
    for (int t = 0; t < 200; ++t) {
        const std::size_t n = 1 + static_cast<std::size_t>(t % 40);
        mtl::vec::dense_vector<T> x(n, T(0)), y(n, T(0));
        std::vector<T> sx(n), sy(n);
        for (std::size_t i = 0; i < n; ++i) {
            sx[i] = next(); sy[i] = next();
            x[i] = sx[i]; y[i] = sy[i];
        }
        const T bridged = mtl::dot<Quire, T>(x, y);
        const T direct  = sw::universal::fdp(sx, sy);
        if (bridged != direct) ++mismatches;
    }
    if (mismatches != 0) {
        std::cerr << name << ": bridge disagrees with sw::universal::fdp in " << mismatches << " of 200 dot products\n";
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
    if (!quire_no_worse_than_native<sw::universal::bposit16>("bposit16")) ++failures;
    if (!quire_no_worse_than_native<sw::universal::cfloat<16, 5>>("cfloat<16,5>")) ++failures;
    if (!quire_no_worse_than_native<sw::universal::lns<16, 8>>("lns<16,8>")) ++failures;

    // Exactness holds where Universal's quire is exact today: posits (any width)
    // and wide cfloat. posit<16,2>/cfloat<16,5> above already show the strict
    // win over native (quire_err 0 vs native_err 0.0156) on this swamping case.
    if (!quire_exact<sw::universal::posit<16, 2>>("posit<16,2>")) ++failures;
    if (!quire_exact<sw::universal::posit<32, 2>>("posit<32,2>")) ++failures;
    if (!quire_exact<sw::universal::bposit16>("bposit16")) ++failures;
    if (!quire_exact<sw::universal::bposit32>("bposit32")) ++failures;

    // bposit: the bridge rounds as Universal's fdp does, and a dense dot of
    // full-significand values is exact (a bposit16 product has at most 26
    // significand bits, so the long-double reference is itself exact).
    if (!bridge_matches_fdp<sw::universal::bposit16>("bposit16")) ++failures;
    if (!bridge_matches_fdp<sw::universal::bposit32>("bposit32")) ++failures;
    if (!bridge_matches_fdp<sw::universal::bposit64>("bposit64")) ++failures;   // wider than double: rounding through double would differ
    if (!bridge_matches_fdp<sw::universal::bposit<8, 4, 0, std::uint8_t>>("bposit<8,4,0>")) ++failures;
    {
        const double e = dense_quire_relerr<sw::universal::bposit16>();
        std::cout << "bposit16 dense quire relerr: " << e << '\n';
        if (!(e <= 1e-15)) {
            std::cerr << "bposit16 quire not exact on the dense dot (relerr " << e << ")\n";
            ++failures;
        }
    }
    if (!quire_exact<sw::universal::cfloat<32, 8>>("cfloat<32,8>")) ++failures;

    // The universal#1202 workaround, pinned on a construction that can actually
    // see it: a NARROW cfloat whose quire floors at ~2^-28 without subnormals
    // becomes bit-exact with them. Same nbits/es, same element arithmetic --
    // only radix_point (28 -> 48) differs. If the +subn assertion ever
    // regresses, docs/level1-accumulator-study.md's cfloat guidance is wrong and
    // the study's +subn rows are stale.
    {
        using Narrow     = sw::universal::cfloat<16, 5>;
        using NarrowSubn = sw::universal::cfloat<16, 5, std::uint8_t, true, false, false>;
        const double e_plain = dense_quire_relerr<Narrow>();
        const double e_subn  = dense_quire_relerr<NarrowSubn>();
        std::cout << "cfloat<16,5> dense quire relerr: no-subn=" << e_plain
                  << " +subn=" << e_subn << '\n';

        // Only the workaround is asserted. The no-subnormals value is reported
        // for contrast but deliberately NOT asserted to be bad -- when
        // universal#1202 lands it should drop to 0, and that must not fail here.
        if (!(e_subn <= 1e-15)) {
            std::cerr << "cfloat<16,5>+subn quire not exact (relerr " << e_subn
                      << ") -- universal#1202 workaround regressed\n";
            ++failures;
        }
    }

    if (failures == 0) std::cout << "mp-blas level-1 quire test passed\n";
    return failures == 0 ? 0 : 1;
}
