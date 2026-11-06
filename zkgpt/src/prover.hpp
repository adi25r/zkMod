#ifndef ZKCNN_PROVER_HPP
#define ZKCNN_PROVER_HPP

#include "global_var.hpp"
#include "circuit.h"
#include "polynomial.h"
#include "hyrax.hpp"
using std::unique_ptr;

class neuralNetwork;
struct Commit_return {
    int l;
    Fr* w;
    ll* ww;
    G1 G, *g;
    G1* comm;
};

struct ExternalInputSegment {
    size_t start;
    size_t len;
    Commit_return* commit;
    string label;
};

struct InputCommitmentSegment {
    size_t start;
    size_t len;
    Commit_return commit;
    Commit_return* external_commit;
    string label;
};

Commit_return commit_values_standalone(const vector<F>& vals, int thread_n = 1, timer* T = nullptr);

// Sub-phase accounting for sumcheckLassoInit
struct LassoInitTimers {
    double beta = 0;     // initBetaTable calls
    double scatter = 0;  // lasso_mult_v[u] += beta_g[hu] loops
    long scattered = 0;  // elements scattered, for a rate
    // verifyLasso's binding-check tail, which produces `gr`
    double bind_beta_g = 0;
    double bind_beta_uv = 0;
    double bind_gather = 0;
    long gathered = 0;
    // The two halves of verifyLasso's GPU region
    double lasso_sc_call = 0;
    double lasso_barrier = 0;
    // sumcheckLassoInit's PROLOGUE
    double lasso_alloc = 0;
    // verifyGKR's matrix_time region
    double mm_pa = 0;  // init_book_keeping (serial)
    double mm_pb = 0;  // init_book_keeping_fast (threaded)
    double mm_sc = 0;  // gpu_matmul_product_sumcheck + barrier
    long mm_layers = 0;
    void reset() {
        beta = scatter = 0;
        scattered = 0;
        bind_beta_g = bind_beta_uv = bind_gather = 0;
        gathered = 0;
        lasso_sc_call = lasso_barrier = lasso_alloc = 0;
        mm_pa = mm_pb = mm_sc = 0;
        mm_layers = 0;
    }
};
extern LassoInitTimers g_lasso_init_timers;

class prover {
public:
    ~prover();
    // Per-FCONN-layer metadata
    static constexpr int kMaxFC = 1024;
    int fc_row[kMaxFC], fc_col[kMaxFC], fc_input_row[kMaxFC], fc_input_col[kMaxFC], max_threads;
    // fc_start_id / fc_input_id are ABSOLUTE offsets into val[0]
    i64 fc_start_id[kMaxFC], fc_input_id[kMaxFC];
    int fc_real_row[kMaxFC], fc_real_col[kMaxFC], fc_real_input_row[kMaxFC],
        fc_real_input_col[kMaxFC];  // padded
    u64 total[2], total_size[2];    // u64: the Lasso domain reaches 2^32 (opt350m len256 splits=1)
    vector<linear_poly> mult_array[2];
    // Per-prover scratch for the u-partitioned gate scatters (prover.cpp build_u_partition)
    struct UPartitionScratch {
        std::vector<u32> order;
        std::vector<size_t> start;
        int T = 0;
    };
    UPartitionScratch upart_scratch[3];
    vector<linear_poly> V_mult[2];
    vector<linear_poly> tmp_mult_array[2];
    vector<linear_poly> tmp_V_mult[2];
    int** mat_val;
    vector<F> beta_u;
    void init();

    void sumcheckInitAll(const vector<F>::const_iterator& r_0_from_v);
    void sumcheckInit(const F& alpha_0, const F& beta_0);
    void sumcheckDotProdInitPhase1();
    void sumcheckInitPhase1(const F& relu_rou_0);
    void sumcheckInitPhase2();

    cubic_poly sumcheckDotProdUpdate1(const F& previous_random);
    quadratic_poly sumcheckUpdate1(const F& previous_random);
    quadratic_poly sumcheckUpdate2(const F& previous_random);

    F Vres(const vector<F>::const_iterator& r, u32 output_size, u8 r_size, int layer_id, int start = 0);
    vector<F> lasso_mult_v, lasso_v_mult;
    // Past-peak release
    void release_lasso_state();
    void sumcheckDotProdFinalize1(const F& previous_random, F& claim_1);
    void sumcheckFinalize1(const F& previous_random, F& claim_0, F& claim_1);
    void sumcheckFinalize2(const F& previous_random, F& claim_0, F& claim_1);
    void sumcheck_lasso_Finalize(const F& previous_random, F& claim_1);

    void sumcheckLassoInit(const vector<F>& s_u, const vector<F>& s_v, const vector<vector<F>>& r_uu,
                           const vector<vector<F>>& r_vv);
    quadratic_poly sumcheckLassoUpdate(const F& previous_random);
    quadratic_poly sumcheckUpdateEach_Lasso(const F& previous_random, bool idx);
    void commitInput(const vector<G1>& gens, int thread_n, timer* T);
    void commitInputSeparate_image(const vector<G1>& gens, int thread_n, timer* T);
    void commitInputSeparate_predicate(const vector<G1>& gens, int thread_n, timer* T);
    // keep_ww=false drops the host __int128 mirror of the committed vector
    void commitOutputPrivate(const vector<G1>& gens, int thread_n = 1, bool keep_ww = true);
    void copyInputWitness();
    void executeCommit(int thread_n = 1);
    void addExternalInputSegment(size_t start, size_t len, Commit_return* commit, const string& label);
    void clearExternalInputSegments();
    timer_range proveTimerRange() const { return prove_timer.range(); }
    timer_range commitTimerRange() const { return commit_range; }
    timer_range outputCommitTimerRange() const { return output_commit_range; }

    timer prove_timer{};
    double prove_time = 0;
    double open_time = 0;
    double proveTime() const { return 0; }
    double proofSize() const { return 0; }
    double polyProverTime() const { return 0; }
    double polyProofSize() const { return 0; }

    layeredCircuit C;
    vector<vector<F>> val;  // the output of each gate
    F getCirValue(int layer_id, const vector<i64>& ori, u32 u);
    u64 proof_size;
    vector<G> gens;
    vector<G> gens_output;
    // Value-initialised
    Commit_return cc{};
    Commit_return cc_input{};
    Commit_return cc_output{};
    vector<F> input_copy;
    bool separate_input = false;
    bool input_not_committed = false;         // indicates there is no input commitment at all
    bool input_committed_externally = false;  // indicates cc_input is provided externally
    // cc_input ownership
    bool cc_input_owned = false;      // owns cc_input.w / .ww / .comm
    bool cc_input_owns_gens = false;  // additionally owns cc_input.g
    bool output_private = false;
    // How many leading entries of the output layer are real
    i64 output_real_len = 0;
    int zkmod_padded_eff_bits(int layer);
    Commit_return commitInputPrefixFast(size_t n, int thr, timer* T);
    bool include_input_commit_size = true;
    vector<ExternalInputSegment> external_input_segments;
    vector<InputCommitmentSegment> input_commit_segments;
    void release_input_commit_segments();  // frees each segment's owned Commit_return
    timer_range commit_range{}, output_commit_range{};
    i64 inp_end_id = 0;
    F eval_img, eval_pred;

private:
    quadratic_poly sumcheckUpdateEach(const F& previous_random, bool idx);
    quadratic_poly sumcheckUpdate(const F& previous_random, vector<F>& r_arr);

    vector<F>::iterator r_0, r_1;  // current positions
    vector<vector<F>> r_u, r_v;    // next positions

    vector<F> beta_g;

    F add_term;

    F V_u0, V_u1;

    F alpha, beta, relu_rou;

    u8 round;         // step within a sumcheck
    int sumcheck_id;  // the level

    friend neuralNetwork;
};
__int128 convert(Fr x);

#endif  // ZKCNN_PROVER_HPP
