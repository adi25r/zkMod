#pragma once
#ifndef ZKGPT_PHASE_ACCT_HPP
#define ZKGPT_PHASE_ACCT_HPP

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

namespace zkgpt_acct {

enum Bucket {
    A_VERIFY_GKR,
    A_VERIFY_LASSO,
    A_OPEN_COMMIT,  // verifier::prove's 3 stages
    A_BF_EVAL,      // brute_force_compute_eval
    A_BF_LR,        // brute_force_compute_LR
    A_HYRAX_OPEN,   // hyrax::open whole call
    A_RT,
    A_TPRIME,
    A_FOLD,  // its three parts
    A_N
};

inline const char* name(int b) {
    static const char* n[A_N] = {"verifyGKR",
                                 "verifyLasso",
                                 "openCommit",
                                 "brute_force_compute_eval",
                                 "brute_force_compute_LR",
                                 "hyrax::open",
                                 "  open:compute_RT",
                                 "  open:compute_Tprime",
                                 "  open:prove_dot_product"};
    return n[b];
}

struct State {
    double s[A_N] = {0};
    uint64_t calls[A_N] = {0};
    uint64_t elems[A_N] = {0};  // domain size, where meaningful
};

inline State& acc() {
    static State s;
    return s;
}
inline bool on() {
    static const bool v = getenv("GKR_ACCT") != nullptr;
    return v;
}

inline double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Adds its lifetime to bucket b
class Scope {
public:
    explicit Scope(int b, uint64_t n = 0) : b_(b), t0_(on() ? now() : 0), n_(n) {}
    ~Scope() {
        if (!on())
            return;
        acc().s[b_] += now() - t0_;
        acc().calls[b_] += 1;
        acc().elems[b_] += n_;
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    int b_;
    double t0_;
    uint64_t n_;
};

inline void reset() {
    if (on())
        acc() = State();
}

inline void report(const char* tag, int idx) {
    if (!on())
        return;
    const State& a = acc();
    for (int i = 0; i < A_N; ++i) {
        if (a.calls[i] == 0)
            continue;
        fprintf(stderr, "[acct] %s idx=%d %-26s %8.3fs calls=%-5llu elems=%llu\n", tag, idx, name(i), a.s[i],
                (unsigned long long)a.calls[i], (unsigned long long)a.elems[i]);
    }
}

}  // namespace zkgpt_acct
#endif
