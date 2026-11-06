//
// Created by 69029 on 3/16/2021.
//

#ifndef ZKCNN_NEURALNETWORK_HPP
#define ZKCNN_NEURALNETWORK_HPP

#include <vector>
#include <functional>
#include <fstream>
#include "circuit.h"
#include "prover.hpp"

using std::vector;
using std::tuple;
using std::pair;

struct SplitBlockCarryState {
    vector<F> input_data;
    i64 input_offset = 0;
    i64 len = 0;
    i64 feature_dim = 0;
    int input_c = 1;
    int input_e = 0;
};

enum convType {
    FFT, NAIVE, NAIVE_FAST
};

struct convKernel {
    convType ty;
    i64 channel_out, channel_in, size, stride_bl, padding, weight_start_id, bias_start_id;
    convKernel(convType _ty, i64 _channel_out, i64 _channel_in, i64 _size, i64 _log_stride, i64 _padding) :
            ty(_ty), channel_out(_channel_out), channel_in(_channel_in), size(_size), stride_bl(_log_stride), padding(_padding) {
    }

    convKernel(convType _ty, i64 _channel_out, i64 _channel_in, i64 _size) :
            convKernel(_ty, _channel_out, _channel_in, _size, 0, _size >> 1) {
    }
};

struct fconKernel {
    i64 channel_out, channel_in, weight_start_id, bias_start_id;
    fconKernel(i64 _channel_out, i64 _channel_in):
        channel_out(_channel_out), channel_in(_channel_in) {}
};

enum poolType {
    AVG, MAX, NONE
};

enum actType {
    RELU_ACT
};

struct poolKernel {
    poolType ty;
    i64 size, stride_bl, dcmp_start_id, max_start_id, max_dcmp_start_id;
    poolKernel(poolType _ty, i64 _size, i64 _log_stride):
            ty(_ty), size(_size), stride_bl(_log_stride) {}
};


class neuralNetwork 
{
public:
    int headnum, headdim, attn_dim, linear_dim;
    explicit neuralNetwork(i64 psize_x, i64 psize_y, i64 pchannel, i64 pparallel, const string &i_filename,
                           const string &c_filename, const string &o_filename, bool i_llm=false,
                           int headnum=0, int headdim=0, int attn_dim=0, int linear_dim=0);
    explicit neuralNetwork(i64 psize_x, i64 psize_y, i64 pchannel, i64 pparallel, const string &i_filename,
                           const string &c_filename, const string &o_filename, bool i_llm=false,
                           int headnum=0, int headdim=0, int attn_dim=0, int linear_dim=0, 
                           bool output_private=false, bool input_private=false, bool input_separate=false);
    void setTotalInputWitnessSize(i64 total_input_witness_size);
    i64 getTotalInputSize() const;
    void initStateFromPreviousBlock(prover &pr_prev_block, neuralNetwork &previous_block);
    void setCustomInput(vector<F> *data, i64 offset, i64 feature_dim);
    SplitBlockCarryState exportSplitBlockCarry(const prover &pr) const;
    // Where exportSplitBlockCarry's slice lives in val[0]: [off, off + n). False
    // when it would fall back to shipping all of val[0] (no window to link).
    bool carryWindow(const prover &pr, i64 &off, i64 &n) const;
    void initStateFromCarry(SplitBlockCarryState &carry);
    // Called by create() right after the builder loop, when this block's output
    // values (the next partition's carry) are final -- before merge_layer,
    // initSubset, copyInputWitness and the cache save. Not called on a cache hit.
    std::function<void(const prover &)> on_values_ready;
    // PROTOTYPE (GKR_PREBUILD): called after readFconWeight, before the layer loop;
    // may replace *custom_input_data and must return the carry's input_c/input_e.
    std::function<void(int &, int &)> on_before_values;

    neuralNetwork(i64 psize, i64 pchannel, i64 pparallel, i64 kernel_size, i64 sec_size, i64 fc_size,
                  i64 start_channel, poolType pool_ty, bool output_private, bool input_private, bool input_separate);
    neuralNetwork(i64 psize, i64 pchannel, i64 pparallel, i64 kernel_size, i64 sec_size, i64 fc_size,
                  i64 start_channel, poolType pool_ty);

    // part_idx: this block's split-partition index, or -1 for a single-block run / the
    // shape-only boundary probe. Threaded through to the witness cache's key so
    // differently-keyed partitions of the same depth/len never collide -- see
    // witness_cache.hpp.
    void create(prover &pr, bool merge, timer &circuit_initialization_timer, int part_idx = -1);
    // Set at the top of create() from pr.max_threads (already the correct per-split-mode
    // budget -- the full thread count monolithically, or main_gkr_split.cpp's tpc under
    // --concurrent/--batched/--overlapped). Read by the CPU witgen value loops and by
    // calcNormalLayer_cpu's bucketed evaluator; never omp_get_num_procs(), to avoid
    // oversubscribing forked children.
    int cpu_threads = 1;
    int layer_norm_w_c[30], layer_norm_w_e[30],layer_norm_b_c[30], layer_norm_b_e[30];
    i64 layer_norm_w_q_start[30],layer_norm_b_q_start[30];
    int layer_norm_c1[30], layer_norm_e1[30], layer_norm_c2[30], layer_norm_e2[30];
    int channel_in, channel_out;
    int layer_num;
    int* mat_values[300];
    bool output_private;
    bool input_private;
    bool input_separate;
    i64 total_input_witness_size = 0;
    i64 weight_start_offset = 0;
    bool custom_input = false;
    i64 custom_input_offset = 0;
    bool custom_input_feature_dim_explicit = false;
    i64 input_feature_dim = 0;
    i64 input_slot_size = 0;
    vector<F>* custom_input_data = nullptr;
protected:

    void initParam(prover& pr,int d);
    // Causal-attention geometry: query i attends keys 0..i, scores stored row-major
    // in a triangle of len*(len+1)/2 entries per head.
    i64 causalScoreCount() const;
    i64 causalRowStart(i64 query_id) const;
    i64 causalKeyCount(i64 query_id) const;

    int getNextBit(int layer_id);

    void calcSizeAfterPool(const poolKernel &p);

    void refreshFCParam(const fconKernel &fc);

    [[nodiscard]] i64 getFFTLen() const;

    [[nodiscard]] i8 getFFTBitLen() const;


    void prepareDecmpBit(i64 layer_id, i64 idx, i64 dcmp_id, i64 bit_shift);

    void prepareFieldBit(const F &data, i64 dcmp_id, i64 bit_shift);

    void prepareSignBit(i64 layer_id, i64 idx, i64 dcmp_id);

    void prepareMax(i64 layer_id, i64 idx, i64 max_id);

    void calcInputLayer(layer &circuit);

    void calcNormalLayer(const layer &circuit, i64 layer_id,bool output=false);
    // Host reference path for calcNormalLayer's gate replay, writing into a
    // caller-supplied buffer so GKR_GATES_CHECK can compare it against the GPU
    // result. Pre: out has circuit.size elements.
    void calcNormalLayer_cpu(const layer &circuit, i64 layer_id, F *out);
    void merge_layer(prover &pr,i64 layer_id) ;
    void checkNormalLayer(const layer &circuit, i64 layer_id,const vector<vector<F> > & vvv);

    void calcDotProdLayer(const layer &circuit, i64 layer_id);

    void calcFFTLayer(const layer &circuit, i64 layer_id);

    vector<vector<convKernel>> conv_section;
    vector<poolKernel> pool;
    poolType pool_ty;
    i64 pool_bl, pool_sz;
    i64 pool_stride_bl, pool_stride;
    i64 pool_layer_cnt, act_layer_cnt, conv_layer_cnt;
    actType act_ty;

    vector<fconKernel> full_conn;

    i64 pic_size_x, pic_size_y, pic_channel, pic_parallel;
    i64 SIZE;
    const i64 NCONV_FAST_SIZE, NCONV_SIZE, FFT_SIZE, AVE_POOL_SIZE, FC_SIZE, RELU_SIZE;
    i64 T;
    const i64 Q = 9;
    i64 Q_MAX;
    const i64 Q_BIT_SIZE = 220;

    
    i64 wc_reserved_in_size_ = 0;   // GKR_CHECK_RESERVE only
    i64 total_in_size, total_para_size, total_relu_in_size, total_ave_in_size, total_max_in_size;
    int x_bit, w_bit, x_next_bit;

    vector<vector<F>>::iterator val;

    bool is_llm;

    int input_e,input_c;
    i64 q_offset;
    i64 gelu_aux_start;
    i64 ln_aux_start;
    i64 softmax_aux_start;
    int len;
    int table[655360];
   
    int positive_check;
    int exp_check;

    void read_layer_norm(int ln_id);
    void inputLayer(layer &circuit);
    void compute_e_table();
    void addBiasLayer(layer &circuit, i64 &layer_id, i64 first_bias_id);

    void roundLayer(layer &circuit, i64 &layer_id, float scale,bool* sparsity_map=NULL);
    // Host reference path for roundLayer's per-element value loop. Pre: out
    // has 2*block_len capacity. Post: out[0..block_len) = the "qq" (rounded
    // value) half, out[block_len..2*block_len) = the "s" (delta-check) half
    // -- same layout as val[0][qq_base..qq_base+2*block_len) in roundLayer.
    void roundLayer_cpu_values(i64 layer_id, i64 block_len, F* out,
                               int c, int m, int M);
    void roundLayer_cpu_gates(std::vector<uniGate> &uni, std::vector<binGate> &bin,
                              i64 orgsize, i64 block_len, int c, int m, int M,
                              bool *sparsity_map);

    void multi_head_matrix_QK(layer &circuit, i64 &layer_id);
    void multi_head_matrix_QK_cpu_gates(std::vector<uniGate> &uni,
                                        std::vector<binGate> &bin, i64 score_count);

    void softmax_layer_1(layer &circuit, i64 &layer_id,float SQ,float SK,float Sv, float Sy);
    void softmax_layer_1_cpu_values(F *out, i64 layer_id, i64 orgsize,
                                    i64 score_count, int c1, int e1, int eprime);
    void softmax_layer_1_cpu_gates(std::vector<uniGate> &uni, std::vector<binGate> &bin,
                                   i64 layer_id, i64 orgsize, i64 score_count,
                                   int c1, int e1, int eprime);
    void softmax_layer_2(layer &circuit, i64 &layer_id,float SQ,float SK,float Sv, float Sy);
    void softmax_layer_2_cpu(std::vector<uniGate> &uni, std::vector<binGate> &bin,
                             F *v0, i64 layer_id, i64 orgsize, i64 score_count,
                             int c1, int e1, int eprime);
    void softmax_layer_3(layer &circuit, i64 &layer_id,float SQ,float SK,float Sv, float Sy);
    void softmax_layer_3_cpu_gates(std::vector<uniGate> &uni,
                                   std::vector<binGate> &bin,
                                   i64 orgsize, i64 score_count);

    void ln_checker_layer1(layer &circuit, i64 &layer_id, int ln_id, int ey,int cy,int cn_in,bool* sparsity_map=NULL);
    void ln_checker_layer1_cpu_values(F *out, i64 orgsize, i64 block_len,
                                      int real_cn_in, i64 qw_off, i64 qb_off,
                                      int c1, int e1, int c2, int e2, int m,
                                      bool *sparsity_map);
    void ln_checker_layer1_cpu_gates(std::vector<uniGate> &uni,
                                     std::vector<binGate> &bin, i64 orgsize,
                                     i64 block_len, int real_cn_in,
                                     bool *sparsity_map);
    
    void ln_checker_layer2(layer &circuit, i64 &layer_id, int ln_id, int ey,int cy,bool* sparsity_map=NULL);
    void ln_checker_layer2_cpu_gates(std::vector<uniGate> &uni,
                                     std::vector<binGate> &bin,
                                     i64 orgsize, i64 block_len, int m,
                                     int c1, int e1, int c2, int e2,
                                     i64 qw_off, i64 qb_off, bool *sparsity_map);
    void ln_checker_layer3(layer &circuit, i64 &layer_id, int ln_id, int ey,int cy,int cn_in,bool* sparsity_map=NULL);
    void ln_checker_layer3_cpu_gates(std::vector<uniGate> &uni,
                                     std::vector<binGate> &bin,
                                     i64 orgsize, i64 block_len, int real_cn_in);

    void gelu_checker_layer1(layer &circuit, i64 &layer_id, int real_cn_out, int ea,int ca,int eb,int cb,int ec,int cc,int ed,int cd,int ex,int cx,int ey,int cy);
    // Host reference path for gelu_checker_layer1's value loop, writing into a
    // caller-supplied advice block so GKR_GELU_CHECK can compare it against the
    // GPU result. Pre: out has 10 + 3*block_len + 3*len*real_cn_out elements,
    // zeroed (padding columns are skipped, not written).
    // Host reference path for gelu_checker_layer1's gate emission. Appends in the
    // original order, which initSubset depends on.
    void gelu_checker_layer1_cpu_gates(std::vector<uniGate> &uni,
                                       std::vector<binGate> &bin,
                                       i64 orgsize, i64 block_len,
                                       int real_cn_out, ll C6, ll C7);
    void gelu_checker_layer1_cpu_values(F *out, i64 block_len, int real_cn_out,
                                        ll C1, ll C2, ll C3, ll C4, ll C5, ll C6, ll C7,
                                        int ca, int cb, int cc, int cx, int cy,
                                        int ea, int eb, int ec, int ex, int ey);
    void gelu_checker_layer2(layer &circuit, i64 &layer_id, int real_cn_out, int ea,int ca,int eb,int cb,int ec,int cc,int ed,int cd,int ex,int cx,int ey,int cy);
    void gelu_checker_layer2_cpu_gates(std::vector<uniGate> &uni,
                                       std::vector<binGate> &bin,
                                       i64 orgsize, i64 layer_id, i64 block_len,
                                       int real_cn_out,
                                       ll C1, ll C2, ll C3, ll C4, ll C5);
    // val[layer_id-1] for the GPU builders; aliases val[0] when there is none.
    const F *wgPrevData(i64 layer_id) const;
    size_t   wgPrevSize(i64 layer_id) const;
    void gelu_checker_layer3(layer &circuit, i64 &layer_id, int real_cn_out, int ea,int ca,int eb,int cb,int ec,int cc,int ed,int cd,int ex,int cx,int ey,int cy);
    void gelu_checker_layer3_cpu_gates(std::vector<uniGate> &uni,
                                       std::vector<binGate> &bin,
                                       i64 orgsize, i64 block_len, int real_cn_out);

    void fullyConnLayer(layer &circuit, i64 &layer_id, i64 first_fc_id, i64 x_offset,int x_layer);
    // Host reference path for fullyConnLayer, writing into a caller-supplied
    // buffer so GKR_WITGEN_CHECK can compare it against the GPU result.
    void fullyConnLayer_cpu(F *out, i64 first_fc_id, i64 x_offset, int x_layer);

    // fc->round residency handoff: fullyConnLayer's GPU path leaves its output in
    // VRAM for roundLayer to consume in place. void* keeps this header HIP-free
    // (real type zkgpt_gpu::gpu_fe*). Whoever clears fc_resident_dev frees it;
    // layer/elems are the guard, so roundLayer fuses only on an exact match.
    void *fc_resident_dev = nullptr;
    i64   fc_resident_elems = 0;
    i64   fc_resident_layer = -1;
    // Idempotent.
    void releaseFcResident();

    static void printLayerInfo(const layer &circuit, i64 layer_id);

    void readBias(i64 first_bias_id);

    void readFconWeight(i64 first_fc_id,int real_r,int real_c,int id);



    void printLayerValues(prover &pr);

    void printInfer(prover &pr);
};


#endif //ZKCNN_NEURALNETWORK_HPP
