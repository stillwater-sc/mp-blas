// mp-blas demo: dot-product accumulation error across precisions.
//
// For each number type T, quantize two deterministic pseudo-random vectors to
// T, then compare two accumulation strategies against a double reference
// computed from the SAME quantized values (isolating accumulation error from
// quantization error):
//
//   native  -- mtl::dot(x, y): products summed in T itself
//   mixed   -- mtl::dot<double>(x, y): products summed in double
//
// This is the level-1 seed of the mixed-precision BLAS story: how much
// accuracy does a wider accumulator buy at each element precision, and how
// does it grow with vector length n?
//
// Usage: dot_precision [n]    (default n = 65536)
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/dot.hpp>

#include <universal/number/posit/posit.hpp>
#include <universal/number/cfloat/cfloat.hpp>

namespace {

// Deterministic values in (-1, 1): reproducible across platforms, no <random>.
double synth(std::size_t i, std::size_t salt) {
    // xorshift-style hash -> [0,1) -> (-1,1)
    std::uint64_t z = (i + 1) * 0x9E3779B97F4A7C15ull + salt * 0xBF58476D1CE4E5B9ull;
    z ^= z >> 30; z *= 0xBF58476D1CE4E5B9ull;
    z ^= z >> 27; z *= 0x94D049BB133111EBull;
    z ^= z >> 31;
    return 2.0 * (static_cast<double>(z >> 11) * 0x1.0p-53) - 1.0;
}

struct Row {
    std::string type;
    double ref;         // double dot over the quantized values
    double err_native;  // |native - ref| / |ref|
    double err_mixed;   // |mixed  - ref| / |ref|
};

template <typename T>
Row measure(const std::string& name, std::size_t n) {
    mtl::vec::dense_vector<T> x(n, T(0)), y(n, T(0));

    // Quantize to T, and build the double reference from the quantized values.
    double ref = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        T xi = T(synth(i, 1)), yi = T(synth(i, 2));
        x(static_cast<int>(i)) = xi;
        y(static_cast<int>(i)) = yi;
        ref += static_cast<double>(xi) * static_cast<double>(yi);
    }

    double native = static_cast<double>(mtl::dot(x, y));
    double mixed  = mtl::dot<double>(x, y);

    const double denom = std::abs(ref) > 0.0 ? std::abs(ref) : 1.0;
    return {name, ref, std::abs(native - ref) / denom, std::abs(mixed - ref) / denom};
}

} // namespace

int main(int argc, char* argv[]) {
    std::size_t n = 65536;
    if (argc > 1) n = static_cast<std::size_t>(std::atoll(argv[1]));

    std::cout << "dot-product accumulation error, n = " << n << '\n'
              << "(relative error vs a double reference over the same quantized values)\n\n";

    Row rows[] = {
        measure<double>("double", n),
        measure<float>("float", n),
        measure<sw::universal::cfloat<16, 8>>("bfloat16", n),
        measure<sw::universal::cfloat<16, 5>>("cfloat<16,5>", n),
        measure<sw::universal::posit<16, 2>>("posit<16,2>", n),
        measure<sw::universal::posit<32, 2>>("posit<32,2>", n),
    };

    std::cout << std::left << std::setw(14) << "type"
              << std::right << std::setw(14) << "reference"
              << std::setw(14) << "native acc"
              << std::setw(14) << "double acc" << '\n';
    for (const Row& r : rows) {
        std::cout << std::left << std::setw(14) << r.type
                  << std::right << std::scientific << std::setprecision(3)
                  << std::setw(14) << r.ref
                  << std::setw(14) << r.err_native
                  << std::setw(14) << r.err_mixed << '\n';
    }
    return 0;
}
