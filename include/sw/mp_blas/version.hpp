#pragma once
// mp-blas -- mixed-precision BLAS (MTL5 + Universal)
//
// Header-only composition layer. As shared utilities emerge (mixed-precision
// kernel wrappers, quire-backed accumulator adapters, benchmark harnesses),
// they live under sw::mp_blas. For now this header carries only version
// metadata.

namespace sw::mp_blas {

inline constexpr int version_major = 0;
inline constexpr int version_minor = 1;
inline constexpr int version_patch = 0;

} // namespace sw::mp_blas
