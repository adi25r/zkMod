#pragma once
// Tiny host-side profiling helpers shared by the sweep binaries. No HIP deps so
// it compiles into any TU. Wall clock in microseconds since epoch (so an
// external GPU-memory poller can attribute samples to a [t0,t1] window) and
// host memory from /proc/self/status (VmRSS = current, VmHWM = process peak).
#include <cstdio>
#include <fstream>
#include <string>
#include <sys/time.h>

inline double prof_now_us() {
    timeval tv;
    gettimeofday(&tv, nullptr);
    return (double)tv.tv_sec * 1e6 + (double)tv.tv_usec;
}

inline long prof_status_kb(const char *key) {
    std::ifstream f("/proc/self/status");
    std::string line;
    size_t klen = std::string(key).size();
    while (std::getline(f, line)) {
        if (line.compare(0, klen, key) == 0) {
            long v = 0;
            sscanf(line.c_str() + klen, "%ld", &v);
            return v;  // kB
        }
    }
    return -1;
}

// Pre: none. Post: VmHWM == current VmRSS (verified on kernel 6.8.0-137).
inline void prof_reset_peak_rss() {
    std::ofstream f("/proc/self/clear_refs");
    if (f) f << "5";
}
