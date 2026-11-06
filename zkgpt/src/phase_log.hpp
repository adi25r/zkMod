#pragma once
#ifndef ZKGPT_PHASE_LOG_HPP
#define ZKGPT_PHASE_LOG_HPP

// Usage: zkgpt_phase::set_part(i)

#include "roctx_scope.hpp"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <unistd.h>

namespace zkgpt_phase {

inline bool on() {
    static const bool v = [] {
        const char* e = getenv("GKR_PHASE_LOG");
        return !(e && e[0] == '0' && e[1] == '\0');
    }();
    return v;
}

inline double now_us() {
    timeval tv;
    gettimeofday(&tv, nullptr);
    return (double)tv.tv_sec * 1e6 + (double)tv.tv_usec;
}

inline int& part_ref() {
    static int p = -1;
    return p;
}
inline void set_part(long long i) {
    part_ref() = (int)i;
}
inline int& depth_ref() {
    static int d = 0;
    return d;
}

// One fprintf per event
extern "C" double zkgpt_phase_h2d_gib() __attribute__((weak));
inline bool xfer_on() {
    static const bool v = getenv("GKR_PHASE_XFER") != nullptr;
    return v;
}

inline void emit(const char* ev, const char* phase, int depth, double t_us, double dur_s, bool have_dur) {
    if (!on())
        return;
    if (have_dur && xfer_on() && zkgpt_phase_h2d_gib)
        fprintf(stderr, "PHASE part=%d pid=%d ev=%-5s phase=%-22s d=%d t_us=%.0f dur_s=%.4f h2d_gib=%.3f\n",
                part_ref(), (int)getpid(), ev, phase, depth, t_us, dur_s, zkgpt_phase_h2d_gib());
    else if (xfer_on() && zkgpt_phase_h2d_gib)
        fprintf(stderr, "PHASE part=%d pid=%d ev=%-5s phase=%-22s d=%d t_us=%.0f h2d_gib=%.3f\n", part_ref(),
                (int)getpid(), ev, phase, depth, t_us, zkgpt_phase_h2d_gib());
    else if (have_dur)
        fprintf(stderr, "PHASE part=%d pid=%d ev=%-5s phase=%-22s d=%d t_us=%.0f dur_s=%.4f\n", part_ref(),
                (int)getpid(), ev, phase, depth, t_us, dur_s);
    else
        fprintf(stderr, "PHASE part=%d pid=%d ev=%-5s phase=%-22s d=%d t_us=%.0f\n", part_ref(),
                (int)getpid(), ev, phase, depth, t_us);
}

inline void mark(const char* phase) {
    if (!on())
        return;
    emit("mark", phase, depth_ref(), now_us(), 0, false);
}

// mark() with a formatted name
inline void markf(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
inline void markf(const char* fmt, ...) {
    if (!on())
        return;
    char buf[64];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    emit("mark", buf, depth_ref(), now_us(), 0, false);
}

class Scope {
public:
    explicit Scope(const char* phase) : phase_(phase) {
        if (!on())
            return;
        t0_ = now_us();
        d_ = depth_ref()++;
        emit("begin", phase_, d_, t0_, 0, false);
    }
    ~Scope() {
        if (!on())
            return;
        const double t1 = now_us();
        --depth_ref();
        emit("end", phase_, d_, t1, (t1 - t0_) / 1e6, true);
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    const char* phase_;
    double t0_ = 0;
    int d_ = 0;
};

}  // namespace zkgpt_phase

#define ZKGPT_PHASE_CAT2(a, b) a##b
#define ZKGPT_PHASE_CAT(a, b) ZKGPT_PHASE_CAT2(a, b)
#define ZKGPT_PHASE(name) ::zkgpt_phase::Scope ZKGPT_PHASE_CAT(_zkphase_, __LINE__)(name)
#define ZKGPT_PHASE_MARK(name) ::zkgpt_phase::mark(name)
#define ZKGPT_PHASE_MARKF(...) ::zkgpt_phase::markf(__VA_ARGS__)

// One region, both marker systems
#define ZKGPT_PHASE_SCOPE(name) \
    ZKGPT_ROCTX_SCOPE(name);    \
    ZKGPT_PHASE(name)

#endif
