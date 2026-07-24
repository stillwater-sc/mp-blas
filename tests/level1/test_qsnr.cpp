// mp-blas level-1 test: quantization signal-to-noise ratio (QSNR) of number
// types on Gaussian data, and its monotonicity -- wider mantissas quantize with
// less noise, so a higher-precision type must have a higher QSNR (dB). Also
// checks the mixed-precision dot accumulation via the quire is no worse than
// native. Returns non-zero on failure (no external framework).
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include <mtl/math/quire_accumulator.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/dot.hpp>

#include <universal/number/posit/posit.hpp>
#include <universal/number/cfloat/cfloat.hpp>

namespace {

// deterministic standard-normal samples via Box-Muller on a splitmix64 stream.
std::vector<double> gaussian(std::size_t n) {
    std::vector<double> v(n);
    std::uint64_t s = 0x243F6A8885A308D3ull;
    auto u01 = [&]() { s += 0x9E3779B97F4A7C15ull; std::uint64_t z = s; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; z ^= z >> 31; return (double(z >> 11) + 0.5) * 0x1.0p-53; };
    for (std::size_t i = 0; i < n; i += 2) {
        double u1 = u01(), u2 = u01();
        double r = std::sqrt(-2.0 * std::log(u1 <= 0.0 ? 1e-300 : u1)), th = 6.283185307179586 * u2;
        v[i] = r * std::cos(th);
        if (i + 1 < n) v[i + 1] = r * std::sin(th);
    }
    return v;
}

// QSNR in dB: 10 log10( E[x^2] / E[(Q(x)-x)^2] ). Higher = less quantization noise.
template <typename T>
double qsnr(const std::vector<double>& data) {
    double noise = 0.0, signal = 0.0;
    for (double x : data) { double d = double(T(x)) - x; noise += d * d; signal += x * x; }
    if (noise == 0.0) noise = std::numeric_limits<double>::min();
    return 10.0 * std::log10(signal / noise);
}

} // namespace

int main() {
    using namespace sw::universal;
    auto data = gaussian(4096);
    int failures = 0;

    double q32 = qsnr<posit<32, 2>>(data);
    double q16 = qsnr<posit<16, 2>>(data);
    double q8  = qsnr<posit<8, 2>>(data);
    std::cout << "posit QSNR (dB): p32=" << q32 << " p16=" << q16 << " p8=" << q8 << '\n';
    if (!(q32 > q16 && q16 > q8)) { std::cerr << "QSNR not monotonic in precision\n"; ++failures; }
    if (!(q8 > 0.0)) { std::cerr << "posit8 QSNR non-positive\n"; ++failures; }

    // mixed-precision dot: quire accumulation is no worse than native for posit16
    {
        using P = posit<16, 2>;
        const std::size_t n = 512;
        mtl::vec::dense_vector<P> a(n, P(0)), b(n, P(0));
        for (std::size_t i = 0; i < n; ++i) { a[i] = P(data[i]); b[i] = P(data[(i + 7) % n]); }
        double ref = 0.0; for (std::size_t i = 0; i < n; ++i) ref += double(a[i]) * double(b[i]);
        double naive = double(mtl::dot(a, b));
        double quire = mtl::dot<sw::universal::quire<P>, double>(a, b);
        std::cout << "dot posit16: naive err=" << std::abs(naive - ref) << " quire err=" << std::abs(quire - ref) << '\n';
        if (!(std::abs(quire - ref) <= std::abs(naive - ref) + 1e-12)) { std::cerr << "quire dot worse than naive\n"; ++failures; }
    }

    if (failures == 0) std::cout << "test_qsnr passed\n";
    return failures == 0 ? 0 : 1;
}
