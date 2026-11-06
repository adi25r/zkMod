#ifndef ZKCNN_RANGE_PROVER_HPP
#define ZKCNN_RANGE_PROVER_HPP

#include "global_var.hpp"
#include "circuit.h"
#include "polynomial.h"
#include "hyrax_rp.hpp"
using std::unique_ptr;

class neuralNetwork;
enum class NonlinearOpType { Attention, LayerNorm, GELU, NonLinear };

class Constraint {
public:
    int query_size = 0;
    int range_size = 0;
    int inputs_capacity = 0;
    ll* inputs = nullptr;
    void prepare();
};

class OP {
public:
    NonlinearOpType op_type;
    vector<Constraint> constraints;
};

class range_prover {
public:
    // Constructor
    range_prover(int layer_num = 0, int head_num = 0, int head_dim = 0, int attn_dim = 0, int linear_dim = 0,
                 int seq_len = 0, int threads = 32, int merged = 0)
        : LayerNum(layer_num),
          HeadNum(head_num),
          HeadDim(head_dim),
          AttnDim(attn_dim),
          LinearDim(linear_dim),
          seq_len(seq_len),
          thread_num(threads),
          merged(merged) {}
    ~range_prover();
    range_prover(const range_prover&) = delete;
    range_prover& operator=(const range_prover&) = delete;

    struct SC_Return {
        Fr* random;
        Fr claim_f;
        Fr claim_g;
    };
    int merged = 0;
    SC_Return sumcheck_deg1(int l, Fr* f, Fr S);
    SC_Return sumcheck_deg3(int l, Fr* r, Fr* f, Fr* g, Fr S);
    int LayerNum, HeadNum, HeadDim, AttnDim, LinearDim, seq_len;  // llm related parameters
    int thread_num;
    vector<OP> ops;
    G1* g = nullptr;
    G1 GG;
    Fr* r = nullptr;
    ll* inputs = nullptr;
    void init();
    void push_back(NonlinearOpType op_type, const std::vector<std::pair<int, int>>& constraint_params);
    timer prove_timer;
    timer prepare_timer;
    void range_prove(ll* inputs, int range, int query_size, int thread_num, int k_stages);
    void logup(ll* f, ll* t, int m, int n, int thread_num, int k_stages);
    double prove();
    void build();

private:
    void release_allocations();

    int generator_count_ = 0;
    int r_count_ = 0;
    int inputs_capacity_ = 0;
    bool built_ = false;
    bool initialized_ = false;
};
__int128 convert(Fr x);

#endif  // ZKCNN_RANGE_PROVER_HPP
