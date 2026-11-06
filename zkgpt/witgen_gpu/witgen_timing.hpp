#pragma once
#ifndef ZKGPT_WITGEN_TIMING_HPP
#define ZKGPT_WITGEN_TIMING_HPP

// Phase accumulators for witness generation

#include <cstdio>
#include <cstdlib>
#include <sys/resource.h>
#include <sys/time.h>

#include "profile_util.hpp"

namespace zkgpt_witgen {

struct Accum {
    double create_s = 0;  // whole neuralNetwork::create() call
    double fc_s = 0;      // fullyConnLayer only (the ~97% of field mults)
    double gates_s = 0;   // all other layer builders + calcNormalLayer
    double h2d_s = 0;     // reserved: staging, when the GPU path is active

    // Per-builder breakdown
    double ln1_s = 0, ln2_s = 0, ln3_s = 0;
    double round_s = 0, qk_s = 0;
    double sm1_s = 0, sm2_s = 0, sm3_s = 0;
    double gelu1_s = 0, gelu2_s = 0, gelu3_s = 0;
    double readln_s = 0, sparsity_s = 0;
    double merge_s = 0, subset_s = 0, copyin_s = 0, init_s = 0;

    // Boundary timers for the TIMING_SPLIT line
    double inlayer_s = 0, values_s = 0, carry_s = 0, sync_s = 0, misc_s = 0, teardown_s = 0;

    // calcNormalLayer, summed over every builder that calls it
    double calcnorm_s = 0;

    // Host CPU time consumed by witness generation
    double cpu_user_s = 0, cpu_sys_s = 0, wall_s = 0;

    void clear() {
        create_s = fc_s = gates_s = h2d_s = 0;
        ln1_s = ln2_s = ln3_s = 0;
        round_s = qk_s = 0;
        sm1_s = sm2_s = sm3_s = 0;
        gelu1_s = gelu2_s = gelu3_s = 0;
        readln_s = sparsity_s = 0;
        merge_s = subset_s = copyin_s = init_s = 0;
        inlayer_s = values_s = carry_s = sync_s = misc_s = teardown_s = 0;
        calcnorm_s = 0;
        cpu_user_s = cpu_sys_s = wall_s = 0;
    }
};

// GKR_WITGEN_PROF=1
inline bool prof_on() {
    static const bool on = getenv("GKR_WITGEN_PROF") != nullptr;
    return on;
}

// One set of counters per process
inline Accum& acc() {
    static Accum a;
    return a;
}

// getrusage(RUSAGE_SELF) user+sys, in seconds
inline void cpu_times(double* user_s, double* sys_s) {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    *user_s = ru.ru_utime.tv_sec + ru.ru_utime.tv_usec / 1e6;
    *sys_s = ru.ru_stime.tv_sec + ru.ru_stime.tv_usec / 1e6;
}

// Scoped stopwatch: adds its lifetime to the given field on destruction
class Scope {
public:
    explicit Scope(double& sink) : sink_(sink), t0_(prof_now_us()) {}
    ~Scope() { sink_ += (prof_now_us() - t0_) / 1e6; }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    double& sink_;
    double t0_;
};

// One line per partition, stderr
inline void report() {
    const Accum& a = acc();
    fprintf(stderr,
            "WITGEN_BREAKDOWN create_s=%.4f fc_s=%.4f calcnorm_s=%.4f "
            "ln1_s=%.4f ln2_s=%.4f ln3_s=%.4f round_s=%.4f qk_s=%.4f "
            "sm1_s=%.4f sm2_s=%.4f sm3_s=%.4f "
            "gelu1_s=%.4f gelu2_s=%.4f gelu3_s=%.4f "
            "readln_s=%.4f subset_s=%.4f copyin_s=%.4f init_s=%.4f "
            "cpu_user_s=%.4f cpu_sys_s=%.4f cpu_wall_s=%.4f cpu_cores=%.3f\n",
            a.create_s, a.fc_s, a.calcnorm_s, a.ln1_s, a.ln2_s, a.ln3_s, a.round_s, a.qk_s, a.sm1_s, a.sm2_s,
            a.sm3_s, a.gelu1_s, a.gelu2_s, a.gelu3_s, a.readln_s, a.subset_s, a.copyin_s, a.init_s,
            a.cpu_user_s, a.cpu_sys_s, a.wall_s,
            a.wall_s > 0 ? (a.cpu_user_s + a.cpu_sys_s) / a.wall_s : 0.0);
}

// Scope that only runs when GKR_WITGEN_PROF is set
class OptScope {
public:
    explicit OptScope(double& sink) : sink_(sink), t0_(prof_on() ? prof_now_us() : 0.0) {}
    ~OptScope() {
        if (prof_on())
            sink_ += (prof_now_us() - t0_) / 1e6;
    }
    OptScope(const OptScope&) = delete;
    OptScope& operator=(const OptScope&) = delete;

private:
    double& sink_;
    double t0_;
};

}  // namespace zkgpt_witgen

#endif  // ZKGPT_WITGEN_TIMING_HPP
