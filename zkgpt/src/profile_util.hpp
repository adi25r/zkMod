#pragma once
// Host-side profiling helpers
#include <cstdio>
#include <fstream>
#include <string>
#include <sys/time.h>

inline double prof_now_us() {
    timeval tv;
    gettimeofday(&tv, nullptr);
    return (double)tv.tv_sec * 1e6 + (double)tv.tv_usec;
}

inline long prof_status_kb(const char* key) {
    std::ifstream f("/proc/self/status");
    std::string line;
    size_t klen = std::string(key).size();
    while (std::getline(f, line)) {
        if (line.compare(0, klen, key) == 0) {
            long v = 0;
            sscanf(line.c_str() + klen, "%ld", &v);
            return v;
        }
    }
    return -1;
}

// Pre: none
inline void prof_reset_peak_rss() {
    std::ofstream f("/proc/self/clear_refs");
    if (f)
        f << "5";
}
