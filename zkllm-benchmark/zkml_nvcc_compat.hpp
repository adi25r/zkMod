// Force-included
#pragma once
#include <hip/hip_runtime.h>
#undef assert
#include <assert.h>
#include <cmath>

inline double sqrt(long long x) {
    return ::sqrt(static_cast<double>(x));
}
inline double sqrt(unsigned long long x) {
    return ::sqrt(static_cast<double>(x));
}
inline double sqrt(long x) {
    return ::sqrt(static_cast<double>(x));
}
inline double sqrt(unsigned long x) {
    return ::sqrt(static_cast<double>(x));
}
inline double sqrt(int x) {
    return ::sqrt(static_cast<double>(x));
}
inline double sqrt(unsigned x) {
    return ::sqrt(static_cast<double>(x));
}
// zkML's `ll` is __int128 (src/typedef.hpp): the one that actually occurs
inline double sqrt(__int128 x) {
    return ::sqrt(static_cast<double>(x));
}
inline double sqrt(unsigned __int128 x) {
    return ::sqrt(static_cast<double>(x));
}
