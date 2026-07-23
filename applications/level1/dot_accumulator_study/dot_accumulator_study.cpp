// mp-blas Milestone 1 -- level-1 accumulator study: dot product and nrm2.
//
// The three precisions of a mixed-precision inner product are independent
// (mtl/math/accumulator_traits.hpp): element precision (bandwidth in),
// accumulator precision (registers), and result precision (out). This study
// fixes a low-precision ELEMENT type and sweeps the ACCUMULATOR across the four
// strategies MTL5 exposes, delivering every result rounded to double so the
// numbers isolate ACCUMULATION error (not storage rounding):
//
//   native   dot<T,      double>  -- sum products in the element type T
//   fma      dot<fma<double>, .. > -- fused multiply-add, one rounding per term
//   promoted dot<double,  double>  -- sum products in double
//   quire    dot<quire,   double>  -- exact sum of products, single round-out
//
// Two reductions are swept:
//   * dot  : x . y over deterministic pseudo-random x, y in (-1, 1).
//   * nrm2 : sqrt(x . x). The accumulation error lives entirely in the
//            sum-of-squares dot, so nrm2 reuses the same accumulator machinery
//            (computed as sqrt of dot<Acc, double>(x, x); MTL5's two_norm<Acc>
//            delivers value<Acc> with no Result seam, so it cannot round a quire
//            out -- the dot(x,x) formulation is the portable mixed path).
//
// Element types: posit, cfloat, and lns -- each admits the generic Universal
// quire via mp-blas's quire_accumulator.hpp adapter. Error is relative to a
// long-double reference over the SAME quantized element values, so quantization
// error cancels and only the accumulator differs.
//
// Usage: dot_accumulator_study [nmax]   (default nmax = 1<<20)
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include <mtl/math/quire_accumulator.hpp>   // must precede dot.hpp's traits use
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/dot.hpp>

#include <universal/number/posit/posit.hpp>
#include <universal/number/cfloat/cfloat.hpp>
#include <universal/number/lns/lns.hpp>

namespace {

// Deterministic values in (-1, 1): reproducible across platforms, no <random>.
double synth(std::size_t i, std::size_t salt) {
    std::uint64_t z = (i + 1) * 0x9E3779B97F4A7C15ull + salt * 0xBF58476D1CE4E5B9ull;
    z ^= z >> 30; z *= 0xBF58476D1CE4E5B9ull;
    z ^= z >> 27; z *= 0x94D049BB133111EBull;
    z ^= z >> 31;
    return 2.0 * (static_cast<double>(z >> 11) * 0x1.0p-53) - 1.0;
}

enum class Kind { dot, nrm2 };

template <typename T>
void study(Kind kind, const std::string& name, std::size_t nmax) {
    using sw::universal::quire;
    using Quire = quire<T>;
    using mtl::math::fma_accumulator;

    std::cout << "\n=== " << (kind == Kind::dot ? "dot  " : "nrm2 ")
              << "element type " << name << " (accumulate, deliver double) ===\n"
              << std::right
              << std::setw(10) << "n"
              << std::setw(13) << "native"
              << std::setw(13) << "fma"
              << std::setw(13) << "promoted"
              << std::setw(13) << "quire" << '\n';

    for (std::size_t n = 256; n <= nmax; n *= 4) {
        mtl::vec::dense_vector<T> x(n, T(0)), y(n, T(0));
        long double ref = 0.0L;   // dot: sum xi*yi ; nrm2: sum xi*xi (sqrt applied below)
        for (std::size_t i = 0; i < n; ++i) {
            T xi = T(synth(i, 1));
            T yi = (kind == Kind::dot) ? T(synth(i, 2)) : xi;
            x(static_cast<int>(i)) = xi;
            y(static_cast<int>(i)) = yi;
            ref += static_cast<long double>(static_cast<double>(xi))
                 * static_cast<long double>(static_cast<double>(yi));
        }
        if (kind == Kind::nrm2) ref = std::sqrt(ref);
        const long double denom = std::abs(ref) > 0.0L ? std::abs(ref) : 1.0L;

        auto deliver = [&](double sum_of_products) {
            return kind == Kind::nrm2 ? std::sqrt(sum_of_products) : sum_of_products;
        };
        auto rel = [&](double v) {
            return static_cast<double>(std::abs(static_cast<long double>(v) - ref) / denom);
        };

        double native   = deliver(mtl::dot<T, double>(x, y));
        double fma       = deliver(mtl::dot<fma_accumulator<double>, double>(x, y));
        double promoted  = deliver(mtl::dot<double, double>(x, y));
        double quirev    = deliver(mtl::dot<Quire, double>(x, y));

        std::cout << std::setw(10) << n
                  << std::scientific << std::setprecision(3)
                  << std::setw(13) << rel(native)
                  << std::setw(13) << rel(fma)
                  << std::setw(13) << rel(promoted)
                  << std::setw(13) << rel(quirev)
                  << std::defaultfloat << '\n';
    }
}

template <typename T>
void both(const std::string& name, std::size_t nmax) {
    study<T>(Kind::dot, name, nmax);
    study<T>(Kind::nrm2, name, nmax);
}

} // namespace

int main(int argc, char* argv[]) {
    std::size_t nmax = std::size_t{1} << 20;
    if (argc > 1) nmax = static_cast<std::size_t>(std::atoll(argv[1]));

    std::cout << "Level-1 accumulator study (dot and nrm2)\n"
              << "relative error vs a long-double reference over the same quantized values\n"
              << "columns: native (element-precision) | fma(double) | promoted(double) | quire(exact)\n";

    both<sw::universal::posit<16, 2>>("posit<16,2>", nmax);
    both<sw::universal::posit<32, 2>>("posit<32,2>", nmax);
    both<sw::universal::cfloat<16, 5>>("cfloat<16,5>", nmax);
    both<sw::universal::cfloat<32, 8>>("cfloat<32,8>", nmax);
    both<sw::universal::lns<16, 8>>("lns<16,8>", nmax);
    both<sw::universal::lns<32, 16>>("lns<32,16>", nmax);

    return 0;
}
