// mp-blas level-3 test: the configuration-3 (quire) rows of the accumulation
// validation matrix, stillwater-sc/mtl5#586 (issue #34), for the matrix products.
// Each test case names the MTL5 tag it mirrors:
//
//   [586:gemm]    mtl5 tests/unit/operation/test_accumulation_configs_mt.cpp
//   [586:spgemm]  mtl5 tests/unit/operation/test_spgemm_accumulator.cpp
//
// Configurations, at element type float (see tests/level1/test_accumulation_matrix_l1.cpp):
//   1 plain (Accumulator = float), 2 fma_accumulator<float>,
//   2c compensated_accumulator<float>, 3 sw::universal::quire<float>.
// mtl::math::exact_accumulator is the independent reference.
//
// What is asserted:
//   * gemm: every entry of configuration 3 is correctly rounded; err(3) is at
//     most err(1), err(2) and err(2c) at every entry; 1 and 2 within
//     gamma_K sum|a_ik b_kj|, 2c within the Dot2 bound.
//   * spgemm: every stored entry of configuration 3 is correctly rounded,
//     against a binary64 Kulisch reference fed whole products; 1 and 2 are not,
//     somewhere, or the inputs do not exercise rounding.
//
// Measured: gemm 24 x 64 x 24 with an ill-conditioned (cond 1e6) dot on each
// diagonal entry; spgemm 40 x 60 * 60 x 40 random cancelling, density 0.3. The
// test prints the counts; they are the same with GCC and Clang:
//   gemm    576 entries, not correctly rounded: plain 429, fma 416, comp 7, quire 0
//   spgemm  1594 entries, not correctly rounded: plain 635, fma 502, quire 0
// (MTL5's spgemm row draws its inputs from std::uniform_real_distribution, whose
// stream is implementation-defined, so its counts -- 584 and 446 of 1589 -- are
// of a different matrix.)
//
// Built with -ffp-contract=off (tests/level3/CMakeLists.txt). Returns non-zero
// on failure.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <type_traits>
#include <utility>

#include <universal/number/quire/quire.hpp>

#include <mtl/generators/ill_conditioned.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/math/exact_accumulator.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/operation/spgemm.hpp>
#include <mtl/vec/dense_vector.hpp>

namespace {

using quire_f = sw::universal::quire<float>;
using fma_f   = mtl::math::fma_accumulator<float>;
using comp_f  = mtl::math::compensated_accumulator<float>;
template <typename T> using exact_acc = mtl::math::exact_accumulator<T>;

int g_failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++g_failures;
    }
}

bool same_bits(float a, float b) { return std::memcmp(&a, &b, sizeof(float)) == 0; }

template <typename T>
exact_acc<T> zero() {
    exact_acc<T> a;
    accumulator_clear(a);
    return a;
}

double round_out_double(const exact_acc<float>& a) {
    return accumulator_value(a, std::type_identity<double>{});
}

/// |exact - computed|, rounded once to double.
double err(const exact_acc<float>& exact, float computed) {
    if (!std::isfinite(computed)) return std::numeric_limits<double>::infinity();
    exact_acc<float> d = exact;
    accumulator_add_product(d, -computed, 1.0f);
    return std::fabs(round_out_double(d));
}

/// A portable random source (the raw mt19937_64 stream is specified).
struct rng {
    std::mt19937_64 g;
    explicit rng(std::uint64_t seed) : g(seed) {}
    double uniform() { return static_cast<double>(g() >> 11) * 0x1p-53; }   // [0, 1)
};

// ---------------------------------------------------------------------------
// [586:gemm]
// ---------------------------------------------------------------------------

void gemm_rows() {
    // Row i of A and column i of B are one ill-conditioned dot, so the diagonal
    // of C is the dot battery through mult<Acc>; the off-diagonal entries are
    // ordinary random dots.
    constexpr std::size_t M = 24, K = 64;
    rng r(5);
    mtl::mat::dense2D<float> A(M, K), B(K, M);
    for (std::size_t i = 0; i < M; ++i) {
        auto d = mtl::generators::gen_dot<float>(K, 1e6, r.g);
        for (std::size_t k = 0; k < K; ++k) { A(i, k) = d.x(k); B(k, i) = d.y(k); }
    }
    mtl::mat::dense2D<float> C1(M, M), C2(M, M), C2c(M, M), C3(M, M);
    mtl::mult<float>(A, B, C1);
    mtl::mult<fma_f>(A, B, C2);
    mtl::mult<comp_f>(A, B, C2c);
    mtl::mult<quire_f>(A, B, C3);

    bool rounded = true, dominates = true, bound12 = true, bound2c = true;
    std::size_t wrong1 = 0, wrong2 = 0, wrong2c = 0, wrong3 = 0;
    const double u = std::numeric_limits<float>::epsilon() / 2;
    const double nu = static_cast<double>(K + 1) * u;
    const double g = nu / (1.0 - nu);
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t j = 0; j < M; ++j) {
            auto exact = zero<float>();
            auto absum = zero<float>();
            for (std::size_t k = 0; k < K; ++k) {
                accumulator_add_product(exact, A(i, k), B(k, j));
                accumulator_add_product(absum, std::fabs(A(i, k)), std::fabs(B(k, j)));
            }
            const float want = accumulator_value(exact, std::type_identity<float>{});
            rounded = rounded && same_bits(C3(i, j), want);
            wrong1 += C1(i, j) != want;
            wrong2 += C2(i, j) != want;
            wrong2c += C2c(i, j) != want;
            wrong3 += C3(i, j) != want;
            const double e1 = err(exact, C1(i, j)), e2 = err(exact, C2(i, j));
            const double e2c = err(exact, C2c(i, j)), e3 = err(exact, C3(i, j));
            dominates = dominates && e3 <= e1 && e3 <= e2 && e3 <= e2c;
            const double sabs = round_out_double(absum);
            bound12 = bound12 && e1 <= g * sabs && e2 <= g * sabs;
            const double dot2 = (u * std::fabs(round_out_double(exact)) + g * g * sabs) * (1.0 + 4.0 * u);
            bound2c = bound2c && e2c <= dot2;
        }
    std::printf("[586:gemm] %zu entries, not correctly rounded: plain %zu, fma %zu, comp %zu, quire %zu\n",
                M * M, wrong1, wrong2, wrong2c, wrong3);
    check(rounded, "[586:gemm] every quire entry is correctly rounded");
    check(dominates, "[586:gemm] err(quire) <= min(err(plain), err(fma), err(comp)) at every entry");
    check(bound12, "[586:gemm] plain and fma within gamma_K sum|a_ik b_kj|");
    check(bound2c, "[586:gemm] compensated within the Dot2 bound");
    check(wrong1 > 0 && wrong2 > 0, "[586:gemm] the inputs make plain and fma round");
}

// ---------------------------------------------------------------------------
// [586:spgemm]
// ---------------------------------------------------------------------------

using sparse = mtl::mat::compressed2D<float>;

/// Random m x n sparse matrix, values of mixed sign over 2^-12 .. 2^12, so the
/// sums cancel and round.
sparse random_sparse(std::size_t m, std::size_t n, double density, std::uint64_t seed) {
    rng r(seed);
    sparse A(m, n);
    {
        mtl::mat::inserter<sparse> ins(A);
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < n; ++j)
                if (r.uniform() < density) {
                    // One draw per statement: the order of a call's arguments is unspecified.
                    const double mant = 1.0 + r.uniform();
                    const int e = static_cast<int>(r.uniform() * 24.0) - 12;
                    const double mag  = std::ldexp(mant, e);
                    const double sign = r.uniform() < 0.5 ? -1.0 : 1.0;
                    ins[i][j] << static_cast<float>(sign * mag);
                }
    }
    return A;
}

/// The correctly rounded C(i,j): each float*float product formed exactly in
/// double and summed in a binary64 Kulisch accumulator, rounded once to float.
/// Independent of the code under test.
float exact_entry(const sparse& A, const sparse& B, std::size_t i, std::size_t j) {
    auto e = zero<double>();
    const auto& as = A.ref_major();
    const auto& ai = A.ref_minor();
    const auto& ad = A.ref_data();
    const auto& bs = B.ref_major();
    const auto& bi = B.ref_minor();
    const auto& bd = B.ref_data();
    for (auto ka = as[i]; ka < as[i + 1]; ++ka) {
        const std::size_t k = ai[ka];
        for (auto kb = bs[k]; kb < bs[k + 1]; ++kb)
            if (bi[kb] == j) accumulator_add_product(e, double(ad[ka]) * double(bd[kb]), 1.0);
    }
    return accumulator_value(e, std::type_identity<float>{});
}

void spgemm_rows() {
    const auto A = random_sparse(40, 60, 0.3, 7);
    const auto B = random_sparse(60, 40, 0.3, 8);
    const auto C1 = mtl::spgemm<float>(A, B);
    const auto C2 = mtl::spgemm<fma_f>(A, B);
    const auto C3 = mtl::spgemm<quire_f>(A, B);
    const bool same_structure = C1.ref_major() == C3.ref_major() && C1.ref_minor() == C3.ref_minor()
                             && C2.ref_major() == C3.ref_major() && C2.ref_minor() == C3.ref_minor();
    check(same_structure, "[586:spgemm] every configuration gives the same structure");
    if (!same_structure) return;

    std::size_t wrong1 = 0, wrong2 = 0, wrong3 = 0, entries = 0;
    const auto& s = C3.ref_major();
    const auto& c = C3.ref_minor();
    for (std::size_t i = 0; i < C3.num_rows(); ++i)
        for (auto k = s[i]; k < s[i + 1]; ++k) {
            const float want = exact_entry(A, B, i, c[k]);
            wrong1 += !same_bits(C1.ref_data()[k], want);
            wrong2 += !same_bits(C2.ref_data()[k], want);
            wrong3 += !same_bits(C3.ref_data()[k], want);
            ++entries;
        }
    std::printf("[586:spgemm] %zu entries, not correctly rounded: plain %zu, fma %zu, quire %zu\n",
                entries, wrong1, wrong2, wrong3);
    check(wrong3 == 0, "[586:spgemm] every quire entry is correctly rounded");
    check(wrong1 > 0 && wrong2 > 0, "[586:spgemm] the inputs make plain and fma round");

    // Constructed: (2^24, 1, -2^24) . (1, 1, 1). Plain and fma absorb the 1.
    sparse R(1, 3), S(3, 1);
    {
        mtl::mat::inserter<sparse> ir(R), is(S);
        ir[0][0] << 16777216.0f; ir[0][1] << 1.0f; ir[0][2] << -16777216.0f;
        is[0][0] << 1.0f;        is[1][0] << 1.0f; is[2][0] << 1.0f;
    }
    check(mtl::spgemm<float>(R, S).ref_data()[0] == 0.0f, "[586:spgemm] plain absorbs 1 into 2^24");
    check(mtl::spgemm<fma_f>(R, S).ref_data()[0] == 0.0f, "[586:spgemm] fma absorbs 1 into 2^24");
    check(mtl::spgemm<quire_f>(R, S).ref_data()[0] == 1.0f, "[586:spgemm] quire keeps 1 next to 2^24");
}

} // namespace

int main() {
    gemm_rows();
    spgemm_rows();
    if (g_failures != 0) {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "test_accumulation_matrix_l3: PASS\n";
    return 0;
}
