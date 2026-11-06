//
// Created by 69029 on 3/9/2021.
//

#ifndef ZKCNN_CONVVERIFIER_HPP
#define ZKCNN_CONVVERIFIER_HPP

#include "prover.hpp"
#include "hyrax.hpp"
#include "timer.hpp"
using namespace hyrax;
using std::unique_ptr;
// Peak-release semaphore (main_gkr_split --overlapped -1): a partition's host AND VRAM
// peaks both land inside verify_lasso. When this hook is set, prove() frees
// what openCommit no longer needs right after verifyLasso -- the gate outputs val[1..],
// the Lasso multiplicity and beta tables -- returns the pages (malloc_trim), and calls
// the hook, which tells the parent this prover is past its peak.
#include <functional>
inline std::function<void()> g_past_peak_hook;

class verifier 
{
public:
    prover *p;
    const layeredCircuit &C;
    int max_threads;

    verifier(prover *pr, const layeredCircuit &cir);

    bool prove(int commit_thread, timer &circuit_initialization_timer);
    void range_prove(double range_prover_time_, timer_range range = {}) {
        range_prover_time = range_prover_time_;
        range_prover_range = range;
    };

    // Emits PROVER_BREAKDOWN + PROVER_TOTALS. Call AFTER range_prove(), else the
    // range-inclusive total is wrong.
    static void print_totals(double core, const double* phase, double range_s);
    void report_totals() const { print_totals(prover_time, phase_s, range_prover_time); }

    // Phase attribution. Every prover_time contribution goes through add_prover(),
    // so the buckets sum to prover_time by construction -- report_totals() prints
    // the residual as a self-check, and a non-zero residual means an untagged site.
    // PH_GKR_MATMUL is split out of verifyGKR's ptimer using mat_timer/matrix_time, so
    // generic carries the remainder and the two still sum to the region total.
    enum PPhase { PH_GKR_GENERIC, PH_GKR_MATMUL, PH_LASSO,
                  PH_HYRAX_COMMIT, PH_HYRAX_OPEN, PH_N };
    double phase_s[PH_N] = {0};
    inline void add_prover(int ph, double t) { prover_time += t; phase_s[ph] += t; }

    timer total_timer{true}, total_slow_timer{true};
    double verifier_time;
    double prover_time;
    double range_prover_time = 0;
    timer_range range_prover_range{};
    double matrix_time;
    Commit_return comm;
    Commit_return comm_input;
    bool separate_input = false;
    bool input_not_committed = false;
    bool input_committed_externally = false;
    bool output_private = false;
    
private:
    vector<vector<F>> r_u, r_v;
    vector<F> final_claim_u0, final_claim_v0;
    bool verifyGKR();
    bool verifyLasso();
    bool openCommit();
    

    vector<F> beta_g;
    vector<F> beta_u;
    vector<F> beta_v;
    void betaInitPhase1(int depth, const F &alpha, const F &beta, const vector<F>::const_iterator &r_0, const vector<F>::const_iterator &r_1, const F &relu_rou);
    void betaInitPhase2(int depth);

    F uni_value[2];
    F bin_value[3];
    void predicatePhase1(int layer_id);
    // Host reference paths, kept as the A/B target for the GPU predicate kernels.
    void predicatePhase1_cpu(int layer_id, F out[2]);
    void predicatePhase2_cpu(int layer_id, F out[3]);
    // Set in betaInitPhase1: initBetaTable zero-fills when beta==0 and early-returns
    // when alpha==0, so beta_g is identically zero exactly then and BOTH predicate
    // loops (and their gate scatters) can be skipped entirely.
    bool beta_g_zero = false;
    void predicatePhase2(int layer_id);

    F getFinalValue(const F &claim_u0, const F &claim_u1, const F &claim_v0, const F &claim_v1);

    F eval_in;
};


#endif //ZKCNN_CONVVERIFIER_HPP
