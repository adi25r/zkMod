#pragma once
#ifndef ZKGPT_WITGEN_TIMING_HPP
#define ZKGPT_WITGEN_TIMING_HPP

// Phase accumulators for witness generation.
//
// `circuit_s` does NOT measure witness generation: circuit_initialization_timer is
// started in neuralNetwork::create() but only stopped at the top of verifier::prove(),
// so it spans the gate-build loop PLUS the verifier ctor, the KV standalone commits,
// rp.init/build/prove and executeCommit. It overlaps `time_s`.
//
// Deliberately raw prof_now_us() deltas rather than the `timer` class:
// timer::elapse_sec() returns a CUMULATIVE total, not the last interval, so
// accumulating it inside a loop produces a triangular sum. It also asserts
// the timer is stopped, which makes it awkward to sample mid-phase.

#include <cstdio>
#include <cstdlib>
#include <sys/resource.h>
#include <sys/time.h>

#include "profile_util.hpp"

namespace zkgpt_witgen {

struct Accum {
    double create_s = 0;   // whole neuralNetwork::create() call
    double fc_s     = 0;   // fullyConnLayer only (the ~97% of field mults)
    double gates_s  = 0;   // all other layer builders + calcNormalLayer
    double h2d_s    = 0;   // reserved: staging, when the GPU arm is active

    // Per-builder breakdown, enabled with GKR_WITGEN_PROF=1. fc_s above is the only
    // one that is always on; the rest attribute the "gates_s" (build_s - fc_s) residual.
    // Each builder both CONSTRUCTS GATE LISTS and COMPUTES WITNESS VALUES in one pass,
    // so these totals are not by themselves an offloadable quantity -- see the _val
    // counters, which time only the value loops.
    double ln1_s = 0, ln2_s = 0, ln3_s = 0;
    double round_s = 0, qk_s = 0;
    double sm1_s = 0, sm2_s = 0, sm3_s = 0;
    double gelu1_s = 0, gelu2_s = 0, gelu3_s = 0;
    double readln_s = 0, sparsity_s = 0;
    double merge_s = 0, subset_s = 0, copyin_s = 0, init_s = 0;

    // Boundary timers for the TIMING_SPLIT line (witness / input layer / circuit).
    // init_s covers only the array setup before inputLayer(); the input layer's own
    // work (inputLayer + weight fill + final initLayer of layer 0) is inlayer_s, and
    // copyInputWitness stays copyin_s. values_s is the whole builder loop's wall time.
    // carry_s is the early-carry publish, sync_s the gate-readback drain, misc_s the
    // gate/value counting, teardown_s everything after copyInputWitness (arena release,
    // reports, witgen_teardown, witness-cache save).
    double inlayer_s = 0, values_s = 0, carry_s = 0, sync_s = 0, misc_s = 0, teardown_s = 0;

    // calcNormalLayer, summed over every builder that calls it. This is a SUBSET
    // of the per-builder totals above -- each builder's time is
    // (its own value + gate-emission loops) + (its share of calcnorm_s) -- so it
    // is the one number that says how much of the residual the gate replay is.
    double calcnorm_s = 0;

    // Host CPU time consumed by witness generation, from getrusage deltas across
    // the whole of create(). user+sys over wall is the mean number of cores busy:
    // ~1.0 means the phase is single-threaded (neuralNetwork.cpp has no OpenMP),
    // and on the GPU arms it says how much host work is left rather than how fast
    // the kernels are.
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

// GKR_WITGEN_PROF=1. Off by default: the per-builder Scopes are called 48 times per
// partition around work that can be sub-millisecond, and prof_now_us() is a syscall.
inline bool prof_on() {
    static const bool on = getenv("GKR_WITGEN_PROF") != nullptr;
    return on;
}

// One set of counters per process. The split runner builds partitions
// sequentially, so a single instance is unambiguous; it is cleared at the top of
// each build_partition().
inline Accum& acc() {
    static Accum a;
    return a;
}

// getrusage(RUSAGE_SELF) user+sys, in seconds. Process-wide across all threads,
// so it counts the OpenMP workers the prover spawns as well -- take deltas over
// a phase, not absolutes.
inline void cpu_times(double *user_s, double *sys_s) {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    *user_s = ru.ru_utime.tv_sec + ru.ru_utime.tv_usec / 1e6;
    *sys_s  = ru.ru_stime.tv_sec + ru.ru_stime.tv_usec / 1e6;
}

// Scoped stopwatch: adds its lifetime to the given field on destruction.
class Scope {
public:
    explicit Scope(double& sink) : sink_(sink), t0_(prof_now_us()) {}
    ~Scope() { sink_ += (prof_now_us() - t0_) / 1e6; }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
private:
    double& sink_;
    double  t0_;
};

// One line per partition, stderr. The per-builder fields are ALWAYS on: they
// wrap whole layer builders (about 21 calls per partition), not inner loops, so
// the clock reads are unmeasurable.
inline void report() {
    const Accum &a = acc();
    fprintf(stderr,
            "WITGEN_BREAKDOWN create_s=%.4f fc_s=%.4f calcnorm_s=%.4f "
            "ln1_s=%.4f ln2_s=%.4f ln3_s=%.4f round_s=%.4f qk_s=%.4f "
            "sm1_s=%.4f sm2_s=%.4f sm3_s=%.4f "
            "gelu1_s=%.4f gelu2_s=%.4f gelu3_s=%.4f "
            "readln_s=%.4f subset_s=%.4f copyin_s=%.4f init_s=%.4f "
            "cpu_user_s=%.4f cpu_sys_s=%.4f cpu_wall_s=%.4f cpu_cores=%.3f\n",
            a.create_s, a.fc_s, a.calcnorm_s,
            a.ln1_s, a.ln2_s, a.ln3_s, a.round_s, a.qk_s,
            a.sm1_s, a.sm2_s, a.sm3_s,
            a.gelu1_s, a.gelu2_s, a.gelu3_s,
            a.readln_s, a.subset_s, a.copyin_s, a.init_s,
            a.cpu_user_s, a.cpu_sys_s, a.wall_s,
            a.wall_s > 0 ? (a.cpu_user_s + a.cpu_sys_s) / a.wall_s : 0.0);
}

// Scope that only runs when GKR_WITGEN_PROF is set.
class OptScope {
public:
    explicit OptScope(double &sink)
        : sink_(sink), t0_(prof_on() ? prof_now_us() : 0.0) {}
    ~OptScope() { if (prof_on()) sink_ += (prof_now_us() - t0_) / 1e6; }
    OptScope(const OptScope &) = delete;
    OptScope &operator=(const OptScope &) = delete;
private:
    double &sink_;
    double  t0_;
};

}  // namespace zkgpt_witgen

#endif  // ZKGPT_WITGEN_TIMING_HPP
