// mp-blas level-1 test: the configuration-3 (quire) rows of the accumulation
// validation matrix, stillwater-sc/mtl5#586 (issue #34). Each test case names the
// MTL5 tag it mirrors, so the matrix reads across repositories:
//
//   [586:dot]          mtl5 tests/unit/operation/test_accumulation_configs_mt.cpp
//   [586:sum]          mtl5 tests/unit/operation/test_sum_trace_accumulator.cpp
//   [586:norms]        mtl5 tests/unit/operation/test_accumulation_configs_mt.cpp
//   [586:determinism]  mtl5 tests/unit/operation/test_accumulation_configs_mt.cpp
//
// The same MTL5 routine runs under each accumulation configuration, at one
// element type T (float or double), so the comparison isolates the accumulation:
//
//   1.  plain        Accumulator = T                      product rounds, then the sum
//   2.  FMA          mtl::math::fma_accumulator<T>        one rounding per term
//   2c. compensated  mtl::math::compensated_accumulator<T> Dot2, as if in twice the precision
//   3.  quire        sw::universal::quire<T>              one rounding at the end
//
// MTL5 plays configuration 3 with its own Kulisch accumulator,
// mtl::math::exact_accumulator<T>. Here that accumulator is the independent
// REFERENCE: every error is measured exactly, as |exact - computed| rounded once
// to double. Both it and the quire are correctly rounded, so configuration 3 must
// equal the reference's round-out bit for bit -- a cross-check of the two
// implementations as well as of MTL5's plumbing.
//
// What is asserted (the honest statement, mtl5#586 "Finding"):
//   * configuration 3 is the correctly rounded result, on every instance;
//   * err(3) <= min(err(1), err(2), err(2c)) on every instance;
//   * 1 and 2 within gamma_n sum|x_i y_i|; 2c within the Dot2 bound
//     u|x.y| + gamma_n^2 sum|x_i y_i|;
//   * NOT err(2) <= err(1): on ill-conditioned sums that ordering flips with the
//     instance, because the error is the cancellation in the running sum, which
//     FMA does not touch.
//
// Measured (GenDot, 40 instances of n = 100 per anticipated cond; geometric mean
// of the relative error, plain / fma / compensated / quire). The achieved cond is
// typically 10x the anticipated one, as in Ogita-Rump-Oishi Table 6.1. The numbers
// match MTL5's, whose battery uses the same generator and seeds:
//
//         float   cond 1e4    2.8e-3  2.9e-3  1.3e-8  1.3e-8
//                 cond 1e8    9.6     11      1.7e-6  1.5e-8
//                 cond 1e12   1.0e5   1.1e5   1.5e-2  1.6e-8
//         double  cond 1e8    2.8e-8  3.8e-8  2.9e-17 2.9e-17
//                 cond 1e16   4.8     4.3     9.5e-16 3.5e-17
//                 cond 1e24   1.1e8   1.3e8   2.6e-8  2.8e-17
//
// The batteries up to cond 1e16 are seeded by cond, as MTL5's are, and give its
// numbers; past 2^64 that conversion is undefined behaviour, so cond 1e20 and up
// are seeded by log10(cond) and are different instances from MTL5's.
// [586:sum]: plain misses the correctly rounded sum in 17 of the 20 cancelling
// 1000-term float sums, the quire in none. The output is bit-identical with GCC
// and Clang.
//
// Built with -ffp-contract=off (tests/level1/CMakeLists.txt): a compiler that
// contracts acc += a*b into an FMA turns configuration 1 into configuration 2.
//
// Sets MTL5_NUM_THREADS before the thread pool's first use, so the long
// reductions split and merge. Returns non-zero on failure.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <type_traits>
#include <utility>

#include <universal/number/quire/quire.hpp>

#include <mtl/detail/thread_pool.hpp>
#include <mtl/generators/ill_conditioned.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/math/exact_accumulator.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/sum.hpp>
#include <mtl/vec/dense_vector.hpp>

namespace {

const int g_set_threads = [] {
#if defined(_WIN32)
    _putenv_s("MTL5_NUM_THREADS", "4");
#else
    setenv("MTL5_NUM_THREADS", "4", /*overwrite=*/1);
#endif
    return 0;
}();

template <typename T> using quire_acc = sw::universal::quire<T>;
template <typename T> using exact_acc = mtl::math::exact_accumulator<T>;
template <typename T> using fma_acc   = mtl::math::fma_accumulator<T>;
template <typename T> using comp_acc  = mtl::math::compensated_accumulator<T>;
template <typename T> using vec       = mtl::vec::dense_vector<T>;

static_assert(mtl::math::adl_accumulator<quire_acc<float>, float>);
static_assert(mtl::math::adl_accumulator<quire_acc<double>, double>);
static_assert(mtl::math::mergeable_accumulator<quire_acc<float>, float>);

int g_failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++g_failures;
    }
}

template <typename T>
bool same_bits(T a, T b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }

/// A cleared reference accumulator.
template <typename T>
exact_acc<T> zero() {
    exact_acc<T> a;
    accumulator_clear(a);
    return a;
}

template <typename T>
T round_out(const exact_acc<T>& a) { return accumulator_value(a, std::type_identity<T>{}); }

template <typename T>
double round_out_double(const exact_acc<T>& a) {
    return accumulator_value(a, std::type_identity<double>{});
}

/// |exact - computed|, rounded once to double; non-finite for a non-finite
/// `computed`, so that it fails every `<=` it meets.
template <typename T>
double err(const exact_acc<T>& exact, T computed) {
    if (!std::isfinite(computed))
        return std::isnan(computed) ? std::numeric_limits<double>::quiet_NaN()
                                    : std::numeric_limits<double>::infinity();
    exact_acc<T> d = exact;
    accumulator_add_product(d, static_cast<T>(-computed), T(1));
    return std::fabs(round_out_double(d));
}

template <typename T>
exact_acc<T> exact_dot(const vec<T>& x, const vec<T>& y) {
    auto a = zero<T>();
    for (std::size_t i = 0; i < x.size(); ++i) accumulator_add_product(a, x(i), y(i));
    return a;
}

/// A portable random source: the raw mt19937_64 stream is specified by the
/// standard; the distributions are not.
struct rng {
    std::mt19937_64 g;
    explicit rng(std::uint64_t seed) : g(seed) {}
    double uniform() { return static_cast<double>(g() >> 11) * 0x1p-53; }   // [0, 1)
    double symmetric() { return 2.0 * uniform() - 1.0; }                     // [-1, 1)
    int integer(int lo, int hi) {
        return lo + static_cast<int>(g() % static_cast<std::uint64_t>(hi - lo + 1));
    }
    template <typename V>
    void shuffle(V& x, V& y) {
        for (std::size_t i = x.size(); i > 1; --i) {
            const std::size_t j = static_cast<std::size_t>(g() % i);
            std::swap(x(i - 1), x(j));
            std::swap(y(i - 1), y(j));
        }
    }
};

/// Ogita, Rump and Oishi's GenDot (Algorithm 6.1), from the MTL5 library.
template <typename T>
void gen_dot(std::size_t n, double cond, rng& r, vec<T>& x, vec<T>& y) {
    auto d = mtl::generators::gen_dot<T>(n, cond, r.g);
    x = std::move(d.x);
    y = std::move(d.y);
}

/// gamma_{n+1} sum|x_i y_i|: the bound of a recursively summed dot product,
/// configurations 1 and 2 (the +1 absorbs the rounding of the exact sum).
template <typename T>
double dot_error_bound(const vec<T>& x, const vec<T>& y) {
    auto a = zero<T>();
    for (std::size_t i = 0; i < x.size(); ++i) accumulator_add_product(a, std::fabs(x(i)), std::fabs(y(i)));
    const double u = std::numeric_limits<T>::epsilon() / 2;
    const double nu = static_cast<double>(x.size() + 1) * u;
    return nu / (1.0 - nu) * round_out_double(a);
}

/// u|x.y| + gamma_{n+1}^2 sum|x_i y_i|, times (1 + 4u): the Dot2 bound, 2c.
template <typename T>
double dot2_error_bound(const vec<T>& x, const vec<T>& y) {
    auto a = zero<T>();
    for (std::size_t i = 0; i < x.size(); ++i) accumulator_add_product(a, std::fabs(x(i)), std::fabs(y(i)));
    const double u = std::numeric_limits<T>::epsilon() / 2;
    const double nu = static_cast<double>(x.size() + 1) * u;
    const double g = nu / (1.0 - nu);
    const double dot = std::fabs(round_out_double(exact_dot(x, y)));
    return (u * dot + g * g * round_out_double(a)) * (1.0 + 4.0 * u);
}

// ---------------------------------------------------------------------------
// [586:dot]
// ---------------------------------------------------------------------------

/// The FMA-wins construction: dot([-fl(ab), a], [1, b]) is the rounding error of
/// a*b. Configuration 1 returns 0, 2 and 3 the exact value.
void dot_fma_removes_the_product_rounding() {
    rng r(4);
    int nonzero = 0;
    bool ok1 = true, ok2 = true, ok3 = true;
    for (int i = 0; i < 2000; ++i) {
        const float a = static_cast<float>(r.symmetric()), b = static_cast<float>(r.symmetric());
        const float p = a * b;
        vec<float> x(2), y(2);
        x(0) = -p; y(0) = 1.0f;
        x(1) = a;  y(1) = b;
        const auto exact = exact_dot(x, y);
        ok1 = ok1 && mtl::dot<float, float>(x, y) == 0.0f;
        const float c2 = mtl::dot<fma_acc<float>, float>(x, y);
        ok2 = ok2 && err(exact, c2) == 0.0;
        ok3 = ok3 && err(exact, mtl::dot<quire_acc<float>, float>(x, y)) == 0.0;
        if (c2 != 0.0f) ++nonzero;
    }
    check(ok1, "[586:dot] plain returns 0 for the rounding error of a product");
    check(ok2, "[586:dot] fma returns the rounding error of a product exactly");
    check(ok3, "[586:dot] quire returns the rounding error of a product exactly");
    check(nonzero > 1000, "[586:dot] most products of random floats are inexact");
}

struct battery_means { double plain, fma, comp, quire; };

template <typename T>
battery_means dot_battery(double cond, std::size_t n, int instances, const char* type) {
    // Seeded by cond, as MTL5's battery is, so the numbers agree; past 2^64 the
    // conversion would be undefined, so those batteries are seeded by log10(cond).
    const std::uint64_t seed = cond < 0x1p64 ? static_cast<std::uint64_t>(cond)
                                             : static_cast<std::uint64_t>(std::log10(cond));
    rng r(seed);
    vec<T> x, y;
    battery_means m{0, 0, 0, 0};
    bool rounded = true, dominates = true, bound12 = true, bound2c = true;
    for (int s = 0; s < instances; ++s) {
        gen_dot<T>(n, cond, r, x, y);
        const auto exact = exact_dot(x, y);
        const T c1  = mtl::dot<T, T>(x, y);
        const T c2  = mtl::dot<fma_acc<T>, T>(x, y);
        const T c2c = mtl::dot<comp_acc<T>, T>(x, y);
        const T c3  = mtl::dot<quire_acc<T>, T>(x, y);
        rounded = rounded && same_bits(c3, round_out(exact));
        const double e1 = err(exact, c1), e2 = err(exact, c2), e2c = err(exact, c2c);
        const double e3 = err(exact, c3);
        dominates = dominates && e3 <= e1 && e3 <= e2 && e3 <= e2c;
        const double bound = dot_error_bound(x, y);
        bound12 = bound12 && e1 <= bound && e2 <= bound;
        bound2c = bound2c && e2c <= dot2_error_bound(x, y);
        // Relative errors, floored at u / 2^20 so an exact result does not send
        // the geometric mean to zero.
        const double scale = std::fabs(round_out_double(exact));
        const double floor = std::numeric_limits<T>::epsilon() * 0x1p-21;
        auto rel = [&](double e) { return std::log(std::max(e / scale, floor)); };
        m.plain += rel(e1); m.fma += rel(e2); m.comp += rel(e2c); m.quire += rel(e3);
    }
    for (double* v : {&m.plain, &m.fma, &m.comp, &m.quire}) *v = std::exp(*v / instances);
    std::printf("[586:dot] %-6s cond %7.0e   plain %8.2g  fma %8.2g  comp %8.2g  quire %8.2g\n",
                type, cond, m.plain, m.fma, m.comp, m.quire);
    check(rounded, "[586:dot] quire is the correctly rounded dot on every instance");
    check(dominates, "[586:dot] err(quire) <= min(err(plain), err(fma), err(comp)) on every instance");
    check(bound12, "[586:dot] plain and fma within gamma_n sum|x_i y_i|");
    check(bound2c, "[586:dot] compensated within the Dot2 bound");
    return m;
}

void dot_ill_conditioned() {
    for (double cond : {1e2, 1e4, 1e6, 1e8, 1e12, 1e16, 1e20})
        dot_battery<float>(cond, 100, 40, "float");
    for (double cond : {1e4, 1e8, 1e12, 1e16, 1e24, 1e32, 1e40})
        dot_battery<double>(cond, 100, 40, "double");
}

// ---------------------------------------------------------------------------
// [586:sum]
// ---------------------------------------------------------------------------

/// n float terms over 2^-12 .. 2^12, half of them negated, so the sum cancels.
vec<float> cancelling(std::size_t n, std::uint64_t seed) {
    rng r(seed);
    vec<float> x(n);
    for (std::size_t i = 0; i < n; ++i) {
        // One draw per statement: the order of a call's arguments is unspecified.
        const double mant = 1.0 + r.uniform();
        const int e = r.integer(-12, 12);
        const float v = static_cast<float>(std::ldexp(mant, e));
        x(i) = (r.g() & 1u) ? v : -v;
    }
    return x;
}

void sum_correctly_rounded() {
    std::size_t plain_wrong = 0;
    bool ok = true;
    for (std::uint64_t seed = 10; seed < 30; ++seed) {
        const auto x = cancelling(1000, seed);
        auto a = zero<float>();
        for (std::size_t i = 0; i < x.size(); ++i) accumulator_add_product(a, x(i), 1.0f);
        const float want = round_out(a);
        ok = ok && same_bits(mtl::sum<quire_acc<float>>(x), want);
        plain_wrong += mtl::sum<float>(x) != want;
    }
    std::printf("[586:sum] plain misses the correctly rounded sum in %zu of 20; quire in none\n",
                plain_wrong);
    check(ok, "[586:sum] quire returns the correctly rounded sum on every instance");
    check(plain_wrong > 0, "[586:sum] the battery makes plain summation round");

    // Constructed: 2^24 + 1 - 2^24. Plain (and so fma) absorbs the 1.
    vec<float> c(3);
    c(0) = 16777216.0f; c(1) = 1.0f; c(2) = -16777216.0f;
    check(mtl::sum<float>(c) == 0.0f, "[586:sum] plain absorbs 1 into 2^24");
    check(mtl::sum<quire_acc<float>>(c) == 1.0f, "[586:sum] quire keeps 1 next to 2^24");
}

// ---------------------------------------------------------------------------
// [586:norms]
// ---------------------------------------------------------------------------

/// sqrt of the sum of squares, rounded once to double, then the sqrt.
double expect_norm(const vec<float>& v) {
    auto a = zero<float>();
    for (std::size_t i = 0; i < v.size(); ++i) accumulator_add_product(a, v(i), v(i));
    return std::sqrt(round_out_double(a));
}

void norms_wide_range() {
    // In float the square of 1e30 overflows and the square of 1e-30 underflows.
    // Configurations 1 and 2 at float form or round that square in float; the
    // quire does not, and rounds the sum of squares once, to double through
    // Result, before the square root.
    {
        vec<float> v(1000);
        for (std::size_t i = 0; i < v.size(); ++i)
            v(i) = (i % 3 ? 1e30f : -3e29f) * static_cast<float>(1 + i % 7);
        check(std::isinf(mtl::two_norm<float, double>(v)), "[586:norms] plain two_norm overflows");
        check(std::isinf(mtl::two_norm<fma_acc<float>, double>(v)), "[586:norms] fma two_norm overflows");
        const double n3 = mtl::two_norm<quire_acc<float>, double>(v);
        check(n3 == expect_norm(v), "[586:norms] quire two_norm: no overflow, the sum of squares correctly rounded");
    }
    {
        vec<float> v(1000);
        for (std::size_t i = 0; i < v.size(); ++i) v(i) = 1e-30f * static_cast<float>(1 + i % 5);
        check(mtl::two_norm<float, double>(v) == 0.0, "[586:norms] plain two_norm underflows to 0");
        check(mtl::two_norm<fma_acc<float>, double>(v) == 0.0, "[586:norms] fma two_norm underflows to 0");
        const double n3 = mtl::two_norm<quire_acc<float>, double>(v);
        check(n3 > 0.0 && n3 == expect_norm(v), "[586:norms] quire two_norm: no underflow");
    }
    {
        mtl::mat::dense2D<float> m(40, 30);
        vec<float> flat(40 * 30);
        for (std::size_t r = 0; r < 40; ++r)
            for (std::size_t c = 0; c < 30; ++c) {
                const float s = (r + c) % 4 == 0 ? 1e25f : ((r + c) % 4 == 1 ? 1.0f : 1e-28f);
                m(r, c) = s * static_cast<float>(1 + (r * 30 + c) % 11);
                flat(r * 30 + c) = m(r, c);
            }
        check(std::isinf(mtl::frobenius_norm<float, double>(m)), "[586:norms] plain frobenius_norm overflows");
        check(std::isinf(mtl::frobenius_norm<fma_acc<float>, double>(m)), "[586:norms] fma frobenius_norm overflows");
        check(mtl::frobenius_norm<quire_acc<float>, double>(m) == expect_norm(flat),
              "[586:norms] quire frobenius_norm: no overflow, the sum of squares correctly rounded");
    }
}

// ---------------------------------------------------------------------------
// [586:determinism]
// ---------------------------------------------------------------------------

void determinism_summation_order() {
    // Shuffle one ill-conditioned dot ten ways: the quire gives the same bits
    // every time; plain and fma do not.
    rng r(7);
    vec<float> x, y;
    gen_dot<float>(2000, 1e8, r, x, y);
    const float c3 = mtl::dot<quire_acc<float>, float>(x, y);
    const float c1 = mtl::dot<float, float>(x, y);
    const float c2 = mtl::dot<fma_acc<float>, float>(x, y);
    bool same3 = true, c1_moved = false, c2_moved = false;
    for (int s = 0; s < 10; ++s) {
        r.shuffle(x, y);
        same3 = same3 && same_bits(mtl::dot<quire_acc<float>, float>(x, y), c3);
        c1_moved = c1_moved || !same_bits(mtl::dot<float, float>(x, y), c1);
        c2_moved = c2_moved || !same_bits(mtl::dot<fma_acc<float>, float>(x, y), c2);
    }
    check(same3, "[586:determinism] quire is independent of summation order");
    check(c1_moved, "[586:determinism] plain depends on summation order");
    check(c2_moved, "[586:determinism] fma depends on summation order");
}

void determinism_thread_count() {
    // Threaded, dot splits the terms into chunks and merges the partials. The
    // quire's merge is exact, so the threaded result is the serial one at any
    // MTL5_NUM_THREADS; plain and fma merge with a rounding add.
    constexpr std::size_t n = 300000;   // long enough for 65536-term chunks
    rng r(8);
    vec<float> x(n), y(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double mant = r.symmetric();
        x(i) = static_cast<float>(std::ldexp(mant, r.integer(-20, 20)));
        y(i) = static_cast<float>(r.symmetric());
    }
    check(same_bits(mtl::dot<quire_acc<float>, float>(x, y), round_out(exact_dot(x, y))),
          "[586:determinism] threaded quire dot equals the correctly rounded serial sum");

    float s1 = 0.0f;
    for (std::size_t i = 0; i < n; ++i) s1 += x(i) * y(i);
    const float t1 = mtl::dot<float, float>(x, y);
    const unsigned threads = mtl::detail::thread_pool::instance().size();
    std::printf("[586:determinism] thread pool: %u workers\n", threads);
    if (threads >= 2)
        check(!same_bits(t1, s1), "[586:determinism] threaded plain dot regroups the sum");
    else
        check(same_bits(t1, s1), "[586:determinism] single-threaded plain dot is the serial sum");
}

} // namespace

int main() {
    dot_fma_removes_the_product_rounding();
    dot_ill_conditioned();
    sum_correctly_rounded();
    norms_wide_range();
    determinism_summation_order();
    determinism_thread_count();
    if (g_failures != 0) {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "test_accumulation_matrix_l1: PASS\n";
    return 0;
}
