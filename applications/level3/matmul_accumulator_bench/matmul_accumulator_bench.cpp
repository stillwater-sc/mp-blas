// mp-blas level-3 accumulator benchmark: the cost of C = A B under the three
// accumulation configurations of MTL5's accumulator_traits.
//
// matmul_accumulator_study measures what each accumulator does to the ERROR of
// a product. This program measures what it costs in TIME, and how that cost
// scales across MTL5's thread pool:
//
//   default    mtl::mult(A, B, C)                    MTL5's own dispatch
//   1 plain    mtl::mult<T>(A, B, C)                 product rounds, then the sum
//   2 fma      mtl::mult<fma_accumulator<T>>(...)    one rounding per term
//   3 quire    mtl::mult<quire<T>>(...)              one rounding per C entry
//
// Every configuration delivers C in the element type T, so they differ only in
// how each inner product is accumulated. Configuration 3 is exact up to its
// final rounding, so it is also the reference: each row reports how many C
// entries the other configurations get wrong, and the normwise error.
//
// THREADS. mtl::mult partitions the rows of C across mtl::detail::thread_pool.
// Each C entry is summed by exactly one thread in ascending k, so every
// configuration gives the same bits at any thread count; the checksum column
// shows it. The pool's size is fixed when it is first used, from
// MTL5_NUM_THREADS (default 1), so one run measures one thread count. To sweep
// thread counts, or to place the threads on P-cores or E-cores of a hybrid CPU,
// use run_sweep.sh next to this file.
//
// The quire is orders of magnitude slower than the native configurations, so an
// n that keeps it to seconds leaves the others at a fraction of a millisecond,
// too short to time thread scaling. --configs picks a subset, so the native
// configurations can be timed at a larger n on their own. Without configuration
// 3 there is no reference, and the mismatch and error columns are left empty.
//
// Usage: matmul_accumulator_bench [--n N] [--reps R] [--type double|float|posit32|all]
//                                 [--configs default,plain,fma,quire] [--csv]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <mtl/detail/thread_pool.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/operation/mult.hpp>

#include <universal/number/posit/posit.hpp>
#include <universal/number/quire/quire.hpp>

namespace {

// A quire<T> runs through MTL5's ADL hook only when Universal defines the
// accumulator_* functions for it. For posit that is Universal v5.7.0 and
// later; for float and double it needs stillwater-sc/universal#1703. Without
// them MTL5 would fall back to the primary accumulator_traits and round every
// product, so configuration 3 is skipped instead of being measured wrongly.
template <typename Acc, typename T>
concept adl_accumulator = requires(Acc& a, const T& v) {
    accumulator_clear(a);
    accumulator_add_product(a, v, v);
};

struct Options {
    std::size_t n = 128;
    unsigned reps = 3;
    std::string type = "all";
    std::string configs = "default,plain,fma,quire";
    bool csv = false;
    bool runs(const std::string& c) const { return ("," + configs + ",").find("," + c + ",") != std::string::npos; }
};

struct Result {
    std::string config;
    bool ran = false;
    double min_ms = 0, median_ms = 0;
    std::uint64_t checksum = 0;
    double mismatch_pct = 0;   // C entries that differ from configuration 3
    double rel_err = 0;        // max |C - C3| / max |C3|
};

template <typename T>
using Matrix = mtl::mat::dense2D<T>;

// FNV-1a over C's entries as doubles (exact for every T used here).
template <typename T>
std::uint64_t checksum(const Matrix<T>& C) {
    std::uint64_t h = 1469598103934665603ull;
    for (std::size_t i = 0; i < C.num_rows(); ++i)
        for (std::size_t j = 0; j < C.num_cols(); ++j) {
            const double d = double(C(i, j));
            unsigned char b[sizeof(double)];
            std::memcpy(b, &d, sizeof d);
            for (unsigned char c : b) h = (h ^ c) * 1099511628211ull;
        }
    return h;
}

template <typename T, typename Acc>
Result time_config(const std::string& name, const Matrix<T>& A, const Matrix<T>& B,
                   Matrix<T>& C, unsigned reps) {
    Result r;
    r.config = name;
    r.ran = true;
    auto once = [&] {
        if constexpr (std::is_void_v<Acc>) mtl::mult(A, B, C);
        else mtl::mult<Acc>(A, B, C);
    };
    once();   // warm-up: page in C, start the pool's workers
    std::vector<double> ms;
    for (unsigned k = 0; k < reps; ++k) {
        const auto t0 = std::chrono::steady_clock::now();
        once();
        const auto t1 = std::chrono::steady_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    r.min_ms = ms.front();
    r.median_ms = ms[ms.size() / 2];
    r.checksum = checksum(C);
    return r;
}

template <typename T>
void compare(Result& r, const Matrix<T>& C, const Matrix<T>& exact) {
    std::size_t diff = 0;
    double err = 0, scale = 0;
    for (std::size_t i = 0; i < C.num_rows(); ++i)
        for (std::size_t j = 0; j < C.num_cols(); ++j) {
            const double c = double(C(i, j)), e = double(exact(i, j));
            if (c != e) ++diff;
            err = std::max(err, std::abs(c - e));
            scale = std::max(scale, std::abs(e));
        }
    r.mismatch_pct = 100.0 * double(diff) / double(C.num_rows() * C.num_cols());
    r.rel_err = scale > 0 ? err / scale : err;
}

template <typename T>
void bench(const std::string& tag, const Options& opt, unsigned threads) {
    using Quire = sw::universal::quire<T>;
    const std::size_t n = opt.n;
    Matrix<T> A(n, n), B(n, n), C(n, n), exact(n, n);
    std::mt19937_64 g(20261016);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) { A(i, j) = T(d(g)); B(i, j) = T(d(g)); }

    // Configuration 3 first: it is the reference for the others.
    Result quire;
    quire.config = "3 quire";
    bool quire_supported = false;
    if constexpr (adl_accumulator<Quire, T>) {
        quire_supported = true;
        if (opt.runs("quire")) quire = time_config<T, Quire>("3 quire", A, B, exact, opt.reps);
    }
    std::vector<Result> rows;
    if (opt.runs("default")) {
        rows.push_back(time_config<T, void>("default", A, B, C, opt.reps));
        if (quire.ran) compare(rows.back(), C, exact);
    }
    if (opt.runs("plain")) {
        rows.push_back(time_config<T, T>("1 plain", A, B, C, opt.reps));
        if (quire.ran) compare(rows.back(), C, exact);
    }
    if (opt.runs("fma")) {
        rows.push_back(time_config<T, mtl::math::fma_accumulator<T>>("2 fma", A, B, C, opt.reps));
        if (quire.ran) compare(rows.back(), C, exact);
    }
    if (opt.runs("quire") && !quire_supported) rows.push_back(quire);   // reported as skipped
    else if (quire.ran) rows.push_back(quire);

    const double macs = double(n) * double(n) * double(n);
    for (const auto& r : rows) {
        const double mmacs = r.ran ? macs / (r.min_ms * 1e3) : 0.0;   // million multiply-adds per second
        if (opt.csv) {
            std::cout << tag << ',' << r.config << ',' << threads << ',' << n << ',';
            if (r.ran) {
                std::cout << std::fixed << std::setprecision(3) << r.min_ms << ',' << r.median_ms << ','
                          << std::setprecision(2) << mmacs << ',' << std::hex << r.checksum << std::dec << ','
                          ;
                if (quire.ran) {
                    std::cout << std::setprecision(2) << r.mismatch_pct << ',' << std::scientific
                              << std::setprecision(3) << r.rel_err << std::defaultfloat;
                } else {
                    std::cout << ',';
                }
                std::cout << '\n';
            } else {
                std::cout << "skipped,,,,,\n";
            }
            continue;
        }
        std::cout << "  " << std::left << std::setw(12) << tag << std::setw(10) << r.config << std::right;
        if (!r.ran) {
            std::cout << "  skipped: this Universal has no accumulator_* protocol for quire<" << tag << ">\n";
            continue;
        }
        std::cout << std::fixed << std::setprecision(2) << std::setw(11) << r.min_ms << std::setw(11)
                  << r.median_ms << std::setw(12) << mmacs << "  " << std::hex << std::setw(16)
                  << std::setfill('0') << r.checksum << std::setfill(' ') << std::dec;
        if (quire.ran) {
            std::cout << std::setw(9) << std::setprecision(2) << r.mismatch_pct << "%" << std::setw(12)
                      << std::scientific << std::setprecision(2) << r.rel_err;
        }
        std::cout << std::defaultfloat << '\n';
    }
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::cerr << a << " needs a value\n"; std::exit(2); }
            return argv[++i];
        };
        if (a == "--n") o.n = std::stoul(next());
        else if (a == "--reps") o.reps = std::max(1u, static_cast<unsigned>(std::stoul(next())));
        else if (a == "--type") o.type = next();
        else if (a == "--configs") o.configs = next();
        else if (a == "--csv") o.csv = true;
        else {
            std::cerr << "usage: " << argv[0]
                      << " [--n N] [--reps R] [--type double|float|posit32|all]"
                         " [--configs default,plain,fma,quire] [--csv]\n";
            std::exit(a == "--help" || a == "-h" ? 0 : 2);
        }
    }
    return o;
}

} // namespace

int main(int argc, char** argv) {
    const Options opt = parse(argc, argv);
    const unsigned threads = mtl::detail::thread_pool::instance().size();
    if (opt.csv) {
        std::cout << "type,config,threads,n,min_ms,median_ms,mmac_per_s,checksum,mismatch_pct,rel_err\n";
    } else {
        std::cout << "matmul accumulator benchmark: C = A B, " << opt.n << " x " << opt.n << ", best of "
                  << opt.reps << ", " << threads << " thread(s) (MTL5_NUM_THREADS; "
                  << std::thread::hardware_concurrency() << " hardware threads)\n"
                  << "  mismatch and error are against configuration 3, the exact product rounded once\n\n"
                  << "  " << std::left << std::setw(12) << "type" << std::setw(10) << "config" << std::right
                  << std::setw(11) << "min ms" << std::setw(11) << "median ms" << std::setw(12) << "MMAC/s"
                  << "  " << std::left << std::setw(16) << "checksum" << std::right << std::setw(10)
                  << "mismatch" << std::setw(12) << "rel err" << '\n';
    }
    const bool all = opt.type == "all";
    if (all || opt.type == "double") bench<double>("double", opt, threads);
    if (all || opt.type == "float") bench<float>("float", opt, threads);
    if (all || opt.type == "posit32") bench<sw::universal::posit<32, 2>>("posit32", opt, threads);
    return 0;
}
