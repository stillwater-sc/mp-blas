// mp-blas -- dot-product characterization study (issue #9).
//
// The accumulator studies in this repo answer "how wrong is the reduction?".
// This one answers the prior question the quire studies need: "what about the
// INSTANCE made it wrong, and can we measure that before choosing an
// accumulator?".
//
// For each (regime, element type) the harness reports a structural feature
// vector of the inner product -- see include/sw/mp_blas/dot_characterization.hpp
// for the definitions and their provenance in the literature --
//
//   cond   2|x|'|y| / |x'y|    the dot-product condition number; log10(cond) is
//                              the number of decimal digits the reduction
//                              destroys (Higham ch.3; Ogita-Rump-Oishi 2005)
//   D      exponent spread     log2(max|p_i| / min|p_i|) over the products
//   rho    growth factor       max partial |sum| / |final|, in index order
//   n_eff  effective length    terms still >= u * (peak partial sum)
//   beta   sign balance        |#pos - #neg| / n; 0 = fully balanced
//
// alongside what those features PREDICT: the realized relative error of native,
// promoted-double and exact-quire accumulation, plus `perm`, the spread of the
// result over random reorderings of the summation. `perm` is the direct
// measurement of reproducibility -- the property Kulisch's exact scalar product
// guarantees by construction, and the one a native accumulator cannot offer at
// any width.
//
// Reference: Dot2 (error-free transformations), NOT a quire -- the reference
// must be independent of the accumulator under test.
//
// Usage: dot_characterization [n]     (default n = 4096)
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <sw/mp_blas/dot_characterization.hpp>

#include <mtl/math/quire_accumulator.hpp>   // must precede dot.hpp's traits use
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/dot.hpp>

#include <universal/number/posit/posit.hpp>
#include <universal/number/cfloat/cfloat.hpp>

namespace {

using sw::mp_blas::regime;
using sw::mp_blas::dot_features;

constexpr std::size_t kPermutations = 16;

/// Deterministic Fisher-Yates over a splitmix64 stream (no <random>, so the
/// permutations are identical on every platform and the table is reproducible).
std::vector<std::size_t> permutation(std::size_t n, std::size_t seed) {
    std::vector<std::size_t> p(n);
    std::iota(p.begin(), p.end(), std::size_t{0});
    std::uint64_t s = 0x243F6A8885A308D3ull + seed * 0x9E3779B97F4A7C15ull;
    auto next = [&]() {
        s += 0x9E3779B97F4A7C15ull;
        std::uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    };
    for (std::size_t i = n; i > 1; --i) p[i - 1] = std::exchange(p[next() % i], p[i - 1]);
    return p;
}

/// Relative spread (max-min)/|ref| of a dot product recomputed under
/// kPermutations random summation orders. Zero iff the accumulator is
/// order-independent, i.e. reproducible.
template <typename T, typename Acc>
double permutation_spread(const std::vector<double>& x, const std::vector<double>& y, double ref) {
    const std::size_t n = x.size();
    double lo = std::numeric_limits<double>::infinity(), hi = -lo;
    for (std::size_t k = 0; k < kPermutations; ++k) {
        const auto p = permutation(n, k);
        mtl::vec::dense_vector<T> a(n, T(0)), b(n, T(0));
        for (std::size_t i = 0; i < n; ++i) { a[i] = T(x[p[i]]); b[i] = T(y[p[i]]); }
        const double v = mtl::dot<Acc, double>(a, b);
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }
    const double denom = (ref != 0.0) ? std::abs(ref) : 1.0;
    return (hi - lo) / denom;
}

template <typename T>
void study(const std::string& name, std::size_t n) {
    using Quire = sw::universal::quire<T>;
    const double u = 0.5 * static_cast<double>(std::numeric_limits<T>::epsilon());

    std::cout << "\n=== element type " << name << ", n = " << n
              << " (accumulate, deliver double) ===\n"
              << "  u = " << std::scientific << std::setprecision(2) << u << std::defaultfloat << '\n'
              << std::right
              << std::setw(12) << "regime"
              << std::setw(11) << "cond"
              << std::setw(7)  << "D"
              << std::setw(11) << "rho"
              << std::setw(8)  << "n_eff"
              << std::setw(7)  << "beta"
              << std::setw(11) << "native"
              << std::setw(11) << "promoted"
              << std::setw(11) << "quire"
              << std::setw(11) << "perm(nat)"
              << std::setw(11) << "perm(qr)" << '\n';

    struct Case { regime r; double param; const char* label; };
    const Case cases[] = {
        { regime::uniform,  0.0,   "uniform"    },
        { regime::positive, 0.0,   "positive"   },
        { regime::graded,   40.0,  "graded"     },
        { regime::cancel,   1e-3,  "cancel 1e-3"},
        { regime::cancel,   1e-6,  "cancel 1e-6"},
        { regime::cancel,   1e-9,  "cancel 1e-9"},
        { regime::kahan,    1e-6,  "kahan 1e-6" },
    };

    for (const auto& c : cases) {
        auto [xd, yd] = sw::mp_blas::generate(c.r, n, c.param);

        // Quantize to the element type, then read back: every metric and every
        // error below is measured on the values the kernel actually sees, so
        // quantization error cancels out of the comparison and only the
        // accumulator differs.
        mtl::vec::dense_vector<T> a(n, T(0)), b(n, T(0));
        for (std::size_t i = 0; i < n; ++i) {
            a[i] = T(xd[i]); b[i] = T(yd[i]);
            xd[i] = static_cast<double>(a[i]);
            yd[i] = static_cast<double>(b[i]);
        }

        const dot_features f = sw::mp_blas::characterize(xd, yd, u);
        const double denom = (f.ref != 0.0) ? std::abs(f.ref) : 1.0;
        auto rel = [&](double v) { return std::abs(v - f.ref) / denom; };

        const double native   = mtl::dot<T, double>(a, b);
        const double promoted = mtl::dot<double, double>(a, b);
        const double quire    = mtl::dot<Quire, double>(a, b);

        std::cout << std::setw(12) << c.label
                  << std::scientific << std::setprecision(1)
                  << std::setw(11) << f.cond
                  << std::fixed << std::setprecision(0)
                  << std::setw(7)  << f.dynamic_range
                  << std::scientific << std::setprecision(1)
                  << std::setw(11) << f.growth
                  << std::setw(8)  << f.n_eff
                  << std::fixed << std::setprecision(2)
                  << std::setw(7)  << f.sign_balance
                  << std::scientific << std::setprecision(1)
                  << std::setw(11) << rel(native)
                  << std::setw(11) << rel(promoted)
                  << std::setw(11) << rel(quire)
                  << std::setw(11) << (permutation_spread<T, T>(xd, yd, f.ref))
                  << std::setw(11) << (permutation_spread<T, Quire>(xd, yd, f.ref))
                  << std::defaultfloat << '\n';
    }
}

/// Sweep the condition number over many decades at fixed length and element
/// type, to locate where each accumulator strategy stops delivering useful
/// digits. This is the decision rule the studies actually need: the theory says
/// a strategy with unit roundoff u fails once cond exceeds ~1/(n*u), so the
/// crossovers should land at predictable places rather than being discovered
/// per-problem.
template <typename T>
void crossover(const std::string& name, std::size_t n) {
    using Quire = sw::universal::quire<T>;
    const double u_elem = 0.5 * static_cast<double>(std::numeric_limits<T>::epsilon());
    const double u_dbl  = 0.5 * std::numeric_limits<double>::epsilon();

    std::cout << "\n=== conditioning crossover: element " << name << ", n = " << n << " ===\n"
              << "  predicted native   limit  cond ~ 1/(n*u_elem) = "
              << std::scientific << std::setprecision(1) << 1.0 / (double(n) * u_elem) << '\n'
              << "  predicted promoted limit  cond ~ 1/(n*u_dbl)  = "
              << 1.0 / (double(n) * u_dbl) << std::defaultfloat << '\n'
              << std::right
              << std::setw(11) << "cond"
              << std::setw(11) << "native"
              << std::setw(11) << "promoted"
              << std::setw(11) << "quire" << '\n';

    // Shrinking the surviving residual by a decade buys a decade of cond.
    for (int k = 0; k <= 20; k += 4) {
        auto [xd, yd] = sw::mp_blas::generate(regime::cancel, n, std::pow(10.0, -k));
        mtl::vec::dense_vector<T> a(n, T(0)), b(n, T(0));
        for (std::size_t i = 0; i < n; ++i) {
            a[i] = T(xd[i]); b[i] = T(yd[i]);
            xd[i] = static_cast<double>(a[i]);
            yd[i] = static_cast<double>(b[i]);
        }
        const dot_features f = sw::mp_blas::characterize(xd, yd, u_elem);
        if (f.ref == 0.0 || !std::isfinite(f.cond)) continue;   // residual vanished under quantization
        const double denom = std::abs(f.ref);
        auto rel = [&](double v) { return std::abs(v - f.ref) / denom; };

        std::cout << std::scientific << std::setprecision(1)
                  << std::setw(11) << f.cond
                  << std::setw(11) << rel(mtl::dot<T, double>(a, b))
                  << std::setw(11) << rel(mtl::dot<double, double>(a, b))
                  << std::setw(11) << rel(mtl::dot<Quire, double>(a, b))
                  << std::defaultfloat << '\n';
    }
}

/// Sweep the LENGTH at (approximately) fixed conditioning, to settle which
/// constant the realized error actually carries. Wilkinson's deterministic
/// bound says gamma_n ~ n*u; Higham & Mary's probabilistic analysis (2019) says
/// sqrt(n)*u is the realistic constant. Normalizing the measured error by
/// u*cond removes the conditioning axis, so whatever `n` dependence survives in
/// the ratio column IS the answer:
///
///   ratio flat in n        ->  error = c*u*cond            (no n dependence)
///   ratio grows like sqrt(n) ->  Higham-Mary probabilistic
///   ratio grows like n       ->  Wilkinson worst case
///
/// The `sqrt(n)/sqrt(n0)` column is printed alongside as the yardstick: if the
/// probabilistic model held, the normalized ratio would track it.
/// Two regimes are swept, because they exercise different mechanisms:
///   regime::cancel   -- cond pinned by generate_at_cond; cancellation-dominated.
///                       The pairs cancel EXACTLY, which largely suppresses the
///                       accumulate-many-roundings mechanism gamma_n describes,
///                       so this isolates the cond axis.
///   regime::positive -- cond == 2 identically for every n, so normalizing by
///                       u*cond divides by a constant and ANY growth in the
///                       ratio is pure n dependence. This is the controlled
///                       experiment for the length axis.
template <typename T>
void length_sweep(const std::string& name, regime r, double target_cond, std::size_t nmax) {
    const double u_elem = 0.5 * static_cast<double>(std::numeric_limits<T>::epsilon());
    const double u_dbl  = 0.5 * std::numeric_limits<double>::epsilon();

    std::cout << "\n=== length sweep, regime " << sw::mp_blas::to_string(r);
    if (r == regime::cancel) std::cout << " at fixed cond ~ " << std::scientific
                                       << std::setprecision(0) << target_cond;
    else                     std::cout << " (cond == 2 for every n)";
    std::cout << ", element " << name << " ===\n" << std::defaultfloat
              << "  normalized columns divide the error by u*cond; flat => no n dependence\n"
              << std::right
              << std::setw(10) << "n"
              << std::setw(11) << "cond"
              << std::setw(11) << "native"
              << std::setw(11) << "nat/u*cond"
              << std::setw(11) << "promoted"
              << std::setw(11) << "prm/u*cond"
              << std::setw(11) << "sqrt(n/n0)" << '\n';

    const std::size_t n0 = 256;
    for (std::size_t n = n0; n <= nmax; n *= 4) {
        auto [xd, yd] = (r == regime::cancel) ? sw::mp_blas::generate_at_cond(n, target_cond)
                                              : sw::mp_blas::generate(r, n, 0.0);
        mtl::vec::dense_vector<T> a(n, T(0)), b(n, T(0));
        for (std::size_t i = 0; i < n; ++i) {
            a[i] = T(xd[i]); b[i] = T(yd[i]);
            xd[i] = static_cast<double>(a[i]);
            yd[i] = static_cast<double>(b[i]);
        }
        const dot_features f = sw::mp_blas::characterize(xd, yd, u_elem);
        if (f.ref == 0.0 || !std::isfinite(f.cond)) continue;
        const double denom = std::abs(f.ref);
        auto rel = [&](double v) { return std::abs(v - f.ref) / denom; };

        const double e_nat = rel(mtl::dot<T, double>(a, b));
        const double e_prm = rel(mtl::dot<double, double>(a, b));

        std::cout << std::setw(10) << n
                  << std::scientific << std::setprecision(1)
                  << std::setw(11) << f.cond
                  << std::setw(11) << e_nat
                  << std::setw(11) << e_nat / (u_elem * f.cond)
                  << std::setw(11) << e_prm
                  << std::setw(11) << e_prm / (u_dbl * f.cond)
                  << std::fixed << std::setprecision(1)
                  << std::setw(11) << std::sqrt(double(n) / double(n0))
                  << std::defaultfloat << '\n';
    }
}

} // namespace

int main(int argc, char* argv[]) {
    std::size_t n = 4096;
    if (argc > 1) n = static_cast<std::size_t>(std::atoll(argv[1]));

    std::cout << "Dot-product characterization study (issue #9)\n"
              << "structural features of the instance, and the accumulation error they predict.\n"
              << "  cond  = 2|x|'|y| / |x'y|   (log10 cond = decimal digits destroyed)\n"
              << "  D     = log2(max|p_i| / min|p_i|)   exponent spread of the products, bits\n"
              << "  rho   = max partial |sum| / |x'y|   growth factor, index order\n"
              << "  n_eff = terms >= u * peak partial sum   (swamping: n_eff << n is bad)\n"
              << "  beta  = |#pos - #neg| / n           sign balance, 0 = fully cancelling\n"
              << "error columns are relative to a Dot2 (doubly-compensated) reference;\n"
              << "perm(*) is the spread of the result over " << kPermutations
              << " random summation orders (0 = reproducible).\n";

    study<sw::universal::posit<16, 2>>("posit<16,2>", n);
    study<sw::universal::posit<32, 2>>("posit<32,2>", n);
    study<sw::universal::cfloat<32, 8>>("cfloat<32,8>", n);

    crossover<sw::universal::posit<32, 2>>("posit<32,2>", n);

    // Milestone 1b: does the realized error carry u, sqrt(n)*u, or n*u? Two
    // controlled sweeps -- one isolating the cond axis, one the length axis.
    const std::size_t nmax = std::max<std::size_t>(n, 65536);
    length_sweep<sw::universal::posit<32, 2>>("posit<32,2>", regime::cancel, 1e12, nmax);
    length_sweep<sw::universal::posit<32, 2>>("posit<32,2>", regime::positive, 0.0, nmax);
    length_sweep<sw::universal::posit<16, 2>>("posit<16,2>", regime::positive, 0.0, nmax);

    return 0;
}
