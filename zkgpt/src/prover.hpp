#ifndef ZKCNN_PROVER_HPP
#define ZKCNN_PROVER_HPP

#include "global_var.hpp"
#include "circuit.h"
#include "polynomial.h"
#include "hyrax.hpp"
using std::unique_ptr;

class neuralNetwork;
struct Commit_return
{
    int l;
    Fr* w;
    ll* ww;
    G1 G,*g;
    G1* comm;
};

struct ExternalInputSegment {
    size_t start;
    size_t len;
    Commit_return *commit;
    string label;
};

struct InputCommitmentSegment {
    size_t start;
    size_t len;
    Commit_return commit;
    Commit_return *external_commit;
    string label;
};

Commit_return commit_values_standalone(const vector<F> &vals, int thread_n=1, timer *T=nullptr);

// Sub-phase accounting for sumcheckLassoInit, the largest single host-side item in
// the prover. Emitted as LASSO_INIT under GKR_MM_PROF=1; always accumulated (two
// clock reads per layer is free next to a 2^bit_length table build).
struct LassoInitTimers {
    double beta = 0;      // initBetaTable calls
    double scatter = 0;   // lasso_mult_v[u] += beta_g[hu] loops
    long   scattered = 0; // elements scattered, for a rate
    // verifyLasso's binding-check tail, which produces `gr`. Split three ways because
    // the fixes are independent: the 2^n beta_g build, the per-layer beta_u/beta_v
    // builds, and the serial random-gather reduction into `gr`.
    double bind_beta_g = 0;
    double bind_beta_uv = 0;
    double bind_gather = 0;
    long   gathered = 0;
    // The two halves of verifyLasso's GPU region. `sc_call` is the sumcheck itself
    // (should track MATMUL_BREAKDOWN host_wall_s); `barrier` is the
    // hipDeviceSynchronize that follows it. A large `barrier` means work launched by
    // an EARLIER phase was still in flight and is being charged to PH_LASSO.
    double lasso_sc_call = 0;
    double lasso_barrier = 0;
    // sumcheckLassoInit's PROLOGUE: resize + zero-fill of lasso_mult_v (2^n elements)
    // and the val[0] pad. Separate from `beta`/`scatter`, which only cover the
    // function's body.
    double lasso_alloc = 0;
    // verifyGKR's matrix_time region (PH_GKR_MATMUL): besides the GPU product-sumcheck,
    // these two host-side bookkeeping table builds. init_book_keeping takes no thread
    // count; init_book_keeping_fast does.
    double mm_pa = 0;      // init_book_keeping   (serial)
    double mm_pb = 0;      // init_book_keeping_fast (threaded)
    double mm_sc = 0;      // gpu_matmul_product_sumcheck + barrier
    long   mm_layers = 0;
    void reset() {
        beta = scatter = 0; scattered = 0;
        bind_beta_g = bind_beta_uv = bind_gather = 0; gathered = 0;
        lasso_sc_call = lasso_barrier = lasso_alloc = 0;
        mm_pa = mm_pb = mm_sc = 0; mm_layers = 0;
    }
};
extern LassoInitTimers g_lasso_init_timers;

class prover {
public:
    ~prover();
    // Per-FCONN-layer metadata, indexed by FC ordinal across the WHOLE circuit
    // being built (4 FCs per transformer block, so a monolithic 32-layer model has 128).
    // These are adjacent members, so an out-of-range index would silently alias the
    // next array; neuralNetwork checks full_conn.size() against kMaxFC and aborts if
    // exceeded. witness_cache.hpp dumps these with sizeof(), so resizing is a
    // cache-format change (bump its kVersion).
    static constexpr int kMaxFC = 1024;
    int fc_row[kMaxFC],fc_col[kMaxFC], fc_input_row[kMaxFC],fc_input_col[kMaxFC],max_threads;
    // fc_start_id / fc_input_id are ABSOLUTE offsets into val[0] (weight_start_id,
    // q_offset), which exceeds 2^31 for a monolithic opt2b7 -- i64, not int.
    i64 fc_start_id[kMaxFC], fc_input_id[kMaxFC];
    int fc_real_row[kMaxFC],fc_real_col[kMaxFC],fc_real_input_row[kMaxFC],fc_real_input_col[kMaxFC]; // padded
    u64 total[2], total_size[2];      // u64: the Lasso domain reaches 2^32 (opt350m len256 splits=1)
    vector<linear_poly> mult_array[2];
    // Per-prover scratch for the u-partitioned gate scatters (prover.cpp
    // build_u_partition). A member, not a function static: a static is shared by
    // every prover thread in one process (modular-zkp's demo driver, mode 0) and a
    // thread_local is invisible to the OpenMP workers that consume it.
    struct UPartitionScratch { std::vector<u32> order; std::vector<size_t> start; int T = 0; };
    UPartitionScratch upart_scratch[3];
    vector<linear_poly> V_mult[2];
    vector<linear_poly> tmp_mult_array[2];
    vector<linear_poly> tmp_V_mult[2];
    int** mat_val;
    vector<F> beta_u;
    void init();

    void sumcheckInitAll(const vector<F>::const_iterator &r_0_from_v);
    void sumcheckInit(const F &alpha_0, const F &beta_0);
    void sumcheckDotProdInitPhase1();
    void sumcheckInitPhase1(const F &relu_rou_0);
    void sumcheckInitPhase2();

    cubic_poly sumcheckDotProdUpdate1(const F &previous_random);
    quadratic_poly sumcheckUpdate1(const F &previous_random);
    quadratic_poly sumcheckUpdate2(const F &previous_random);

    F Vres(const vector<F>::const_iterator &r, u32 output_size, u8 r_size,int layer_id,int start=0) ;
    vector<F> lasso_mult_v,lasso_v_mult;
    // Past-peak release (verifier.hpp g_past_peak_hook): drop the state only the GKR
    // sumcheck and Lasso use -- gate outputs val[1..] (openCommit reads val[0]), the
    // Lasso multiplicities and the beta tables.
    void release_lasso_state();
    void sumcheckDotProdFinalize1(const F &previous_random, F &claim_1);
    void sumcheckFinalize1(const F &previous_random, F &claim_0, F &claim_1);
    void sumcheckFinalize2(const F &previous_random, F &claim_0, F &claim_1);
    void sumcheck_lasso_Finalize(const F &previous_random, F &claim_1);

    void sumcheckLassoInit(const vector<F> &s_u, const vector<F> &s_v,const vector<vector<F>>& r_uu, const vector<vector<F>>& r_vv);
    quadratic_poly sumcheckLassoUpdate(const F &previous_random);
    quadratic_poly sumcheckUpdateEach_Lasso(const F &previous_random, bool idx) ;
    void commitInput(const vector<G1> &gens,int thread_n, timer *T);
    void commitInputSeparate_image(const vector<G1> &gens,int thread_n, timer *T);
    void commitInputSeparate_predicate(const vector<G1> &gens,int thread_n, timer *T);
    // keep_ww=false drops the host __int128 mirror of the committed vector. It is
    // write-only after the MSM everywhere except OwnedCommitment::capture, so the default keeps it and only
    // split_utils.hpp's boundary commit opts out. On the GPU path dropping it also
    // removes the host array entirely -- see gpu_fr_scalars.hpp.
    void commitOutputPrivate(const vector<G1> &gens,int thread_n=1,bool keep_ww=true);
    void copyInputWitness();
    void executeCommit(int thread_n=1);
    void addExternalInputSegment(size_t start, size_t len, Commit_return *commit, const string &label);
    void clearExternalInputSegments();
    timer_range proveTimerRange() const { return prove_timer.range(); }
    timer_range commitTimerRange() const { return commit_range; }
    timer_range outputCommitTimerRange() const {return output_commit_range;}

    timer prove_timer{};
    double prove_time = 0;
    double open_time = 0;
    double proveTime() const{ return 0;}
    double proofSize() const { return 0;}
    double polyProverTime() const { return 0;}
    double polyProofSize() const { return 0;}
    
    layeredCircuit C;
    vector<vector<F>> val;        // the output of each gate
    F getCirValue(int layer_id, const vector<i64> &ori, u32 u);
    u64 proof_size;
    vector<G> gens;
    vector<G> gens_output;
    // Value-initialised: prover has no ctor and init() runs only from the verifier
    // ctor, so a prover that is built and destroyed without one (the split runner's
    // probe pass) would otherwise delete[] stack garbage in ~prover.
    Commit_return cc{};
    Commit_return cc_input{};
    Commit_return cc_output{};
    vector<F> input_copy;
    bool separate_input = false;
    bool input_not_committed = false;           // indicates there is no input commitment at all
    bool input_committed_externally = false;    // indicates cc_input is provided externally
    // cc_input ownership. It is NOT one flag: commit_segment aliases the caller's
    // generator vector (ret.g = gens.data()) while commit_values_standalone new[]s
    // its own, so the arrays and the basis are owned independently. Both default to
    // false, which means "never free cc_input" for every assignment site that does
    // not opt in (notably where cc_input aliases an OwnedCommitment's vectors and
    // must never be delete[]d).
    bool cc_input_owned = false;        // owns cc_input.w / .ww / .comm
    bool cc_input_owns_gens = false;    // additionally owns cc_input.g
    bool output_private = false;
    // How many leading entries of the output layer are real, before pad_output_layer
    // padded it out to the successor's input-half size. 0 = never padded, treat the
    // whole layer as real. Only the commit's scalar conversion reads this; the
    // commitment is still over the full padded domain, which is what the successor
    // opens against.
    i64 output_real_len = 0;
    int zkmod_padded_eff_bits(int layer);
    Commit_return commitInputPrefixFast(size_t n, int thr, timer *T);
    bool include_input_commit_size = true;
    vector<ExternalInputSegment> external_input_segments;
    vector<InputCommitmentSegment> input_commit_segments;
    void release_input_commit_segments();   // frees each segment's owned Commit_return
    timer_range commit_range{}, output_commit_range{};
    i64 inp_end_id = 0;
    F eval_img, eval_pred;
    
private:
    quadratic_poly sumcheckUpdateEach(const F &previous_random, bool idx);
    quadratic_poly sumcheckUpdate(const F &previous_random, vector<F> &r_arr);
    

    vector<F>::iterator r_0, r_1;         // current positions
    vector<vector<F>> r_u, r_v;             // next positions

    vector<F> beta_g;
    
    F add_term;
    

    F V_u0, V_u1;

    F alpha, beta, relu_rou;

    

    
    u8 round;          // step within a sumcheck
    int sumcheck_id;   // the level

    friend neuralNetwork;
};
__int128 convert(Fr x)	;

#endif //ZKCNN_PROVER_HPP
