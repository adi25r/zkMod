// Force-included (-include) into every zkML HIP translation unit when it is compiled with nvcc.
// nvcc: sqrt(<integer>) is ambiguous between std::sqrt(float), std::sqrt(long double) and ::sqrt(double);
// the clang-based HIP compilers the code was written for promote an integer argument to double.
// These overloads do the same promotion, so unmodified call sites such as neuralNetwork.cpp's sqrt(B)
// (B a long long) compile. Exact-match overloads are preferred, so double/float calls are unaffected.
#pragma once
// HIP's NVIDIA backend redefines assert as `{ if (!COND) { abort_(); } }` (hip/nvidia_detail/nvidia_hip_runtime.h),
// with no parentheses around COND, so assert(a == b) becomes (!a) == b and fails to compile for class types such as
// the field element. Include the HIP header first, then restore the standard assert; later includes of the HIP
// header are guarded no-ops, so its assert is never re-applied.
#include <hip/hip_runtime.h>
#undef assert
#include <assert.h>
#include <cmath>

inline double sqrt(long long x)          { return ::sqrt(static_cast<double>(x)); }
inline double sqrt(unsigned long long x) { return ::sqrt(static_cast<double>(x)); }
inline double sqrt(long x)               { return ::sqrt(static_cast<double>(x)); }
inline double sqrt(unsigned long x)      { return ::sqrt(static_cast<double>(x)); }
inline double sqrt(int x)                { return ::sqrt(static_cast<double>(x)); }
inline double sqrt(unsigned x)           { return ::sqrt(static_cast<double>(x)); }
// zkML's `ll` is __int128 (src/typedef.hpp): the one that actually occurs.
inline double sqrt(__int128 x)           { return ::sqrt(static_cast<double>(x)); }
inline double sqrt(unsigned __int128 x)  { return ::sqrt(static_cast<double>(x)); }
