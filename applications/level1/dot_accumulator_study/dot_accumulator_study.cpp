// mp-blas Milestone 1 -- level-1 accumulator study: dot product.
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
// Error is relative to a long-double reference over the SAME quantized element
// values, so quantization error cancels and only the accumulator differs.
//
// The story this seeds: for a low-precision element type, how wide an
// accumulator does a dot product actually need as the length n grows, and where
// does the exact quire earn its cost over a merely-promoted double accumulator?
//
// Usage: dot_accumulator_study [nmax]   (default nmax = 1<<20)
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <mtl/math/quire_accumulator.hpp>   // must precede dot.hpp's traits use
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/dot.hpp>

#include <universal/number/posit/posit.hpp>

namespace {

// Deterministic values in (-1, 1): reproducible across platforms, no <random>.
double synth(std::size_t i, std::size_t salt) {
    std::uint64_t z = (i + 1) * 0x9E3779B97F4A7C15ull + salt * 0xBF58476D1CE4E5B9ull;
    z ^= z >> 30; z *= 0xBF58476D1CE4E5B9ull;
    z ^= z >> 27; z *= 0x94D049BB133111EBull;
    z ^= z >> 31;
    return 2.0 * (static_cast<double>(z >> 11) * 0x1.0p-53) - 1.0;
}

template <typename Posit>
void study(const std::string& name, std::size_t nmax) {
    using sw::universal::quire;
    using Quire = quire<Posit>;
    using mtl::math::fma_accumulator;

    std::cout << "\n=== element type " << name << " (accumulate, deliver double) ===\n"
              << std::right
              << std::setw(10) << "n"
              << std::setw(13) << "native"
              << std::setw(13) << "fma"
              << std::setw(13) << "promoted"
              << std::setw(13) << "quire" << '\n';

    for (std::size_t n = 256; n <= nmax; n *= 4) {
        mtl::vec::dense_vector<Posit> x(n, Posit(0)), y(n, Posit(0));
        long double ref = 0.0L;
        for (std::size_t i = 0; i < n; ++i) {
            Posit xi = Posit(synth(i, 1)), yi = Posit(synth(i, 2));
            x(static_cast<int>(i)) = xi;
            y(static_cast<int>(i)) = yi;
            ref += static_cast<long double>(static_cast<double>(xi))
                 * static_cast<long double>(static_cast<double>(yi));
        }
        const long double denom = std::abs(ref) > 0.0L ? std::abs(ref) : 1.0L;
        auto rel = [&](double v) {
            return static_cast<double>(std::abs(static_cast<long double>(v) - ref) / denom);
        };

        double native   = mtl::dot<Posit, double>(x, y);
        double fma       = mtl::dot<fma_accumulator<double>, double>(x, y);
        double promoted  = mtl::dot<double, double>(x, y);
        double quirev    = mtl::dot<Quire, double>(x, y);

        std::cout << std::setw(10) << n
                  << std::scientific << std::setprecision(3)
                  << std::setw(13) << rel(native)
                  << std::setw(13) << rel(fma)
                  << std::setw(13) << rel(promoted)
                  << std::setw(13) << rel(quirev)
                  << std::defaultfloat << '\n';
    }
}

} // namespace

int main(int argc, char* argv[]) {
    std::size_t nmax = std::size_t{1} << 20;
    if (argc > 1) nmax = static_cast<std::size_t>(std::atoll(argv[1]));

    std::cout << "Level-1 dot-product accumulator study\n"
              << "relative error vs a long-double reference over the same quantized values\n"
              << "columns: native (element-precision) | fma(double) | promoted(double) | quire(exact)\n";

    study<sw::universal::posit<16, 2>>("posit<16,2>", nmax);
    study<sw::universal::posit<32, 2>>("posit<32,2>", nmax);

    return 0;
}
