// quantization_qsnr: quantization signal-to-noise ratio of number types.
//
// Migrated from Universal benchmark/accuracy/quantization/{mpdot,mpfma}.cpp. For
// Gaussian data, the QSNR (dB) measures how much quantization noise a number
// type introduces when it represents the data:
//     QSNR = 10 log10( E[x^2] / E[(Q(x) - x)^2] )
// and the mixed-precision dot columns show how the ACCUMULATOR (native vs exact
// quire) affects the dot-product error once the elements are quantized.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <mtl/math/quire_accumulator.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/dot.hpp>

#include <universal/number/posit/posit.hpp>
#include <universal/number/cfloat/cfloat.hpp>

namespace {

std::vector<double> gaussian(std::size_t n, double stddev) {
    std::vector<double> v(n);
    std::uint64_t s = 0x243F6A8885A308D3ull;
    auto u01 = [&]() { s += 0x9E3779B97F4A7C15ull; std::uint64_t z = s; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; z ^= z >> 31; return (double(z >> 11) + 0.5) * 0x1.0p-53; };
    for (std::size_t i = 0; i < n; i += 2) {
        double u1 = u01(), u2 = u01();
        double r = stddev * std::sqrt(-2.0 * std::log(u1 <= 0.0 ? 1e-300 : u1)), th = 6.283185307179586 * u2;
        v[i] = r * std::cos(th);
        if (i + 1 < n) v[i + 1] = r * std::sin(th);
    }
    return v;
}

template <typename T>
double qsnr(const std::vector<double>& data) {
    double noise = 0.0, signal = 0.0;
    for (double x : data) { double d = double(T(x)) - x; noise += d * d; signal += x * x; }
    if (noise == 0.0) noise = std::numeric_limits<double>::min();
    return 10.0 * std::log10(signal / noise);
}

template <typename T>
void report(const std::string& tag, const std::vector<double>& data) {
    // quantized dot vs a double reference: native vs quire accumulation
    const std::size_t n = data.size();
    mtl::vec::dense_vector<T> a(n, T(0)), b(n, T(0));
    for (std::size_t i = 0; i < n; ++i) { a[i] = T(data[i]); b[i] = T(data[(i + 7) % n]); }
    double ref = 0.0; for (std::size_t i = 0; i < n; ++i) ref += double(a[i]) * double(b[i]);
    double naive = double(mtl::dot(a, b));
    double quire = mtl::dot<sw::universal::quire<T>, double>(a, b);

    std::cout << "  " << std::left << std::setw(14) << tag << std::right
              << "  QSNR=" << std::setw(7) << std::fixed << std::setprecision(1) << qsnr<T>(data) << " dB"
              << "   dot err: naive=" << std::setw(10) << std::scientific << std::setprecision(2) << std::abs(naive - ref)
              << " quire=" << std::setw(10) << std::abs(quire - ref) << '\n';
}

} // namespace

int main(int argc, char** argv) {
    using namespace sw::universal;
    std::size_t n = (argc > 1) ? static_cast<std::size_t>(std::stoul(argv[1])) : 4096;
    auto data = gaussian(n, 1.0);
    std::cout << "quantization QSNR + mixed-precision dot on N(0,1) data, N = " << n << "\n";
    std::cout << "  number type     representation      accumulation error\n";
    report<posit<32, 2>>("posit<32,2>", data);
    report<posit<16, 2>>("posit<16,2>", data);
    report<posit<8, 2>>("posit<8,2>", data);
    report<cfloat<16, 5, std::uint16_t, true, false, false>>("cfloat<16,5>", data);
    report<cfloat<8, 4, std::uint8_t, true, false, false>>("cfloat<8,4>", data);
    return 0;
}
