// Created by 69029 on 3/16/2021
#undef NDEBUG
#include "neuralNetwork.hpp"
#include "utils.hpp"
#include "global_var.hpp"
#include "witness_cache.hpp"
#include "gpu_witgen.hpp"
#include "witgen_timing.hpp"
#include "config.hpp"  // gkr_config.witgen_gpu (runtime-settable, see below)
#include <polynomial.h>
#include <circuit.h>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <sstream>
#include <cstdio>
#include <cstdlib>

using std::cerr;
using std::endl;
using std::ifstream;
using std::max;
using std::ofstream;

ifstream in;
ifstream conf;
ofstream out;
namespace {
i64 align_up_i64(i64 value, i64 alignment) {
    if (alignment <= 1)
        return value;
    return ((value + alignment - 1) / alignment) * alignment;
}

i64 next_power_of_two_i64(i64 value) {
    i64 power = 1;
    while (power < value)
        power <<= 1;
    return power;
}
}  // namespace

namespace multi_max {
template <class T>
T max(T head) {
    return head;
}
template <class T, typename... Args>
T max(T head, Args... args) {
    T t = max<T>(args...);
    return (head > t) ? head : t;
}
}  // namespace multi_max

neuralNetwork::neuralNetwork(i64 psize_x, i64 psize_y, i64 pchannel, i64 pparallel, const string& i_filename,
                             const string& c_filename, const string& o_filename, bool i_llm, int headnum,
                             int headdim, int attn_dim, int linear_dim)
    : pic_size_x(psize_x),
      pic_size_y(psize_y),
      pic_channel(pchannel),
      pic_parallel(pparallel),
      SIZE(0),
      NCONV_FAST_SIZE(1),
      NCONV_SIZE(2),
      FFT_SIZE(5),
      AVE_POOL_SIZE(1),
      FC_SIZE(1),
      RELU_SIZE(2),
      act_ty(RELU_ACT),
      headnum(headnum),
      headdim(headdim),
      attn_dim(attn_dim),
      linear_dim(linear_dim),
      output_private(false),
      input_private(false),
      custom_input(false),
      custom_input_offset(0),
      input_feature_dim(attn_dim),
      input_slot_size(0),
      custom_input_data(nullptr) {
    is_llm = i_llm;
    in.open(i_filename);
    conf.open(c_filename);
    input_c = 1;
    input_e = 0;
}
neuralNetwork::neuralNetwork(i64 psize_x, i64 psize_y, i64 pchannel, i64 pparallel, const string& i_filename,
                             const string& c_filename, const string& o_filename, bool i_llm, int headnum,
                             int headdim, int attn_dim, int linear_dim, bool output_private,
                             bool input_private, bool input_separate)
    : pic_size_x(psize_x),
      pic_size_y(psize_y),
      pic_channel(pchannel),
      pic_parallel(pparallel),
      SIZE(0),
      NCONV_FAST_SIZE(1),
      NCONV_SIZE(2),
      FFT_SIZE(5),
      AVE_POOL_SIZE(1),
      FC_SIZE(1),
      RELU_SIZE(2),
      act_ty(RELU_ACT),
      headnum(headnum),
      headdim(headdim),
      attn_dim(attn_dim),
      linear_dim(linear_dim),
      output_private(output_private),
      input_private(input_private),
      input_separate(input_separate),
      custom_input(false),
      custom_input_offset(0),
      input_feature_dim(attn_dim),
      input_slot_size(0),
      custom_input_data(nullptr) {
    is_llm = i_llm;
    in.open(i_filename);
    conf.open(c_filename);
    input_c = 1;
    input_e = 0;
}

// val[layer_id-1] for the gate evaluator
const F* neuralNetwork::wgPrevData(i64 layer_id) const {
    return (layer_id > 0 && !val[layer_id - 1].empty()) ? val[layer_id - 1].data() : val[0].data();
}
size_t neuralNetwork::wgPrevSize(i64 layer_id) const {
    return (layer_id > 0 && !val[layer_id - 1].empty()) ? val[layer_id - 1].size() : val[0].size();
}

// Compare a GPU-emitted gate list against the host reference, elementwise
static void wg_check_gates(const char* tag, const std::vector<uniGate>& ru, const std::vector<uniGate>& gu,
                           const std::vector<binGate>& rb, const std::vector<binGate>& gb) {
    if (ru.size() != gu.size() || rb.size() != gb.size()) {
        fprintf(stderr, "[%s] GATE COUNT uni %zu vs %zu, bin %zu vs %zu\n", tag, ru.size(), gu.size(),
                rb.size(), gb.size());
        abort();
    }
    for (size_t i = 0; i < ru.size(); ++i)
        if (ru[i].g != gu[i].g || ru[i].u != gu[i].u || ru[i].lu != gu[i].lu || ru[i].sc != gu[i].sc) {
            fprintf(stderr, "[%s] UNI GATE MISMATCH at %zu\n", tag, i);
            abort();
        }
    for (size_t i = 0; i < rb.size(); ++i)
        if (rb[i].g != gb[i].g || rb[i].u != gb[i].u || rb[i].v != gb[i].v || rb[i].l != gb[i].l ||
            rb[i].sc != gb[i].sc) {
            fprintf(stderr, "[%s] BIN GATE MISMATCH at %zu\n", tag, i);
            abort();
        }
}

void neuralNetwork::setTotalInputWitnessSize(i64 total_input_witness_size) {
    auto bitlen = ceilPow2BitLength64(total_input_witness_size);
    this->total_input_witness_size = 1LL << bitlen;
    std::cout << "Total Input Witness Size (in setTotalInputWitnessSize): " << total_input_witness_size
              << std::endl;
    if (total_input_witness_size > 0)
        weight_start_offset = this->total_input_witness_size >> 1;
    else
        weight_start_offset = 0;
}

// Total MLE size when splitting inputs: return 2 * max(input_half, weight_half)
i64 neuralNetwork::getTotalInputSize() const {
    auto size = total_in_size;
    std::cout << "Total Input Size (in getTotalInputSize): " << size << std::endl;
    std::cout << "Weight Start Offset (in getTotalInputSize): " << weight_start_offset << std::endl;
    auto input_pred_size = size - weight_start_offset;
    std::cout << "Input Pred Size (in getTotalInputSize): " << input_pred_size << std::endl;
    auto input_part_size = (input_pred_size > weight_start_offset) ? input_pred_size : weight_start_offset;
    std::cout << "Input Part Size (in getTotalInputSize): " << input_part_size << std::endl;
    std::cout << "Total Input Size (in getTotalInputSize): "
              << (1LL << (1 + ceilPow2BitLength64(input_part_size))) << std::endl;
    return 1LL << (1 + ceilPow2BitLength64(input_part_size));
}

void neuralNetwork::initStateFromPreviousBlock(prover& pr_prev_block, neuralNetwork& previous_block) {
    custom_input = true;
    custom_input_data = &pr_prev_block.val[0];
    custom_input_offset = previous_block.q_offset;
    len = previous_block.len;
    input_feature_dim = previous_block.channel_out;
    custom_input_feature_dim_explicit = true;
    input_slot_size = 0;
    input_c = previous_block.input_c;
    input_e = previous_block.input_e;
}

void neuralNetwork::setCustomInput(vector<F>* data, i64 offset, i64 feature_dim) {
    custom_input = true;
    custom_input_data = data;
    custom_input_offset = offset;
    input_feature_dim = feature_dim;
    custom_input_feature_dim_explicit = true;
    input_slot_size = 0;
}

SplitBlockCarryState neuralNetwork::exportSplitBlockCarry(const prover& pr) const {
    SplitBlockCarryState carry;
    // The consumer reads only [input_offset
    const i64 need = static_cast<i64>(len) * channel_out;
    if (!pr.val.empty() && q_offset >= 0 && need > 0 &&
        static_cast<i64>(q_offset) + need <= static_cast<i64>(pr.val[0].size())) {
        carry.input_data.assign(pr.val[0].begin() + q_offset, pr.val[0].begin() + q_offset + need);
        carry.input_offset = 0;
    } else {
        if (!pr.val.empty())
            carry.input_data = pr.val[0];
        carry.input_offset = q_offset;
    }
    carry.len = len;
    carry.feature_dim = channel_out;
    carry.input_c = input_c;
    carry.input_e = input_e;
    return carry;
}

bool neuralNetwork::carryWindow(const prover& pr, i64& off, i64& n) const {
    const i64 need = static_cast<i64>(len) * channel_out;
    if (pr.val.empty() || q_offset < 0 || need <= 0 ||
        static_cast<i64>(q_offset) + need > static_cast<i64>(pr.val[0].size()))
        return false;
    off = q_offset;
    n = need;
    return true;
}

void neuralNetwork::initStateFromCarry(SplitBlockCarryState& carry) {
    custom_input = true;
    custom_input_data = &carry.input_data;
    custom_input_offset = carry.input_offset;
    len = static_cast<int>(carry.len);
    input_feature_dim = carry.feature_dim;
    custom_input_feature_dim_explicit = true;
    input_slot_size = 0;
    input_c = carry.input_c;
    input_e = carry.input_e;
}

i64 neuralNetwork::causalScoreCount() const {
    return len * (len + 1) / 2;
}

i64 neuralNetwork::causalRowStart(i64 query_id) const {
    return query_id * (query_id + 1) / 2;
}

i64 neuralNetwork::causalKeyCount(i64 query_id) const {
    return query_id + 1;
}

void Out_group(Fr a, Fr b, Fr c) {
    char flaga = ' ', flagb = ' ', flagc = ' ';
    if (a > 1e10) {
        flaga = '-';
        a = -a;
    }

    if (b > 1e10) {
        b = -b;
        flagb = '-';
    }
    if (c > 1e10) {
        flagc = '-';
        c = -c;
    }
}

void neuralNetwork::initParam(prover& pr, int depth) {
    total_in_size = 0;
    total_para_size = 0;
    total_relu_in_size = 0;
    total_ave_in_size = 0;
    total_max_in_size = 0;
    pr.clearExternalInputSegments();
    // data: first reserve space for the (padded) user input so it can be committed separately
    i64 real_len = len;
    i64 padded_len = 1LL << ceilPow2BitLength(real_len);
    len = padded_len;
    i64 feature_dim =
        input_feature_dim > 0 ? input_feature_dim : (full_conn.empty() ? 1024 : full_conn.front().channel_in);
    if (custom_input && custom_input_data) {
        if (custom_input_offset < 0 || custom_input_offset > (i64)custom_input_data->size()) {
            throw std::runtime_error("custom input offset is outside the source witness");
        }
        i64 available = (i64)custom_input_data->size() - custom_input_offset;
        if (custom_input_feature_dim_explicit) {
            if (feature_dim <= 0) {
                throw std::runtime_error("custom input feature dimension must be positive");
            }
            i64 required = len * feature_dim;
            if (available < required) {
                throw std::runtime_error("custom input source is shorter than len * feature_dim");
            }
        } else {
            if (available <= 0 || available % len != 0) {
                throw std::runtime_error("cannot infer custom input feature dimension from source witness");
            }
            feature_dim = available / len;
        }
    }
    if (feature_dim <= 0)
        feature_dim = 1;
    input_feature_dim = feature_dim;
    input_slot_size = 1LL << ceilPow2BitLength(input_feature_dim);
    i64 input_slots = len * input_slot_size;

    i64 pos;
    if (input_private || input_separate) {
        if (!total_input_witness_size) {
            i64 half = 1LL << ceilPow2BitLength(input_slots);
            total_input_witness_size = half << 1;
            weight_start_offset = half;
        }
        pos = weight_start_offset;
        pr.inp_end_id = weight_start_offset;
    } else {
        weight_start_offset = 1LL << ceilPow2BitLength(input_slots);
        pos = weight_start_offset;
        pr.inp_end_id = pos;
    }
    if (full_conn.size() > (size_t)prover::kMaxFC) {
        fprintf(stderr,
                "[initParam] %zu FCONN layers exceed prover::kMaxFC=%d -- raise it "
                "in prover.hpp (and bump witness_cache kVersion)\n",
                full_conn.size(), prover::kMaxFC);
        abort();
    }
    for (int i = 0; i < full_conn.size(); ++i) {
        auto& fc = full_conn[i];
        refreshFCParam(fc);
        // fc_kernel
        pr.fc_real_input_row[i] = real_len;
        len = pr.fc_input_row[i] = 1 << ceilPow2BitLength(len);
        pr.fc_real_input_col[i] = fc.channel_in;
        pr.fc_real_row[i] = fc.channel_in;
        pr.fc_input_col[i] = 1 << ceilPow2BitLength(fc.channel_in);

        fc.channel_in = pr.fc_row[i] = 1 << ceilPow2BitLength(fc.channel_in);
        pr.fc_real_col[i] = fc.channel_out;
        fc.channel_out = pr.fc_col[i] = 1 << ceilPow2BitLength(fc.channel_out);

        fc.weight_start_id = pos;
        pr.fc_start_id[i] = fc.weight_start_id;
        u32 para_size = pr.fc_row[i] * pr.fc_col[i];
        pos += para_size;
        total_para_size += para_size;
    }
    std::cout << "initParam: weight_start_offset=" << weight_start_offset << " pos_end=" << pos
              << " total_input_witness_size=" << total_input_witness_size << std::endl;
    if (total_input_witness_size) {
        pr.inp_end_id = weight_start_offset;
        if (pos > total_input_witness_size) {
            fprintf(stderr, "Configured total_input_witness_size (%lld) too small, needs at least %lld\n",
                    total_input_witness_size, pos);
            exit(EXIT_FAILURE);
        }
        total_in_size = total_input_witness_size;
    } else {
        total_in_size = pos;
    }
    std::cout << "total_in_size: " << total_in_size << std::endl;
    vector<string> layers = {"l1",        "l2",       "l3",    "fcon",   "round", "MHA_QK", "softmax*V",
                             "softmax*v", "soft_end", "fcon2", "round2", "l1",    "l2",     "l3",
                             "fcon3",     "round",    "gelu1", "g2",     "g3",    "fcon4",  "round"};
    SIZE = 1 + layers.size() * depth;
}
void neuralNetwork::merge_layer(prover& pr, i64 layer_id) {
    int cntp = 0;
    for (int i = 0; i < layer_id; i++) {
        if ((int)pr.C.circuit[i].ty == 4 || (int)pr.C.circuit[i].ty == 11) {
            ++cntp;
            layer v;
            v.ty = pr.C.circuit[i].ty;
            pr.C.circuit.push_back(v);
            swap(pr.C.circuit[i], pr.C.circuit[pr.C.circuit.size() - 1]);
            vector<F> f;
            pr.val.push_back(f);
            swap(pr.val[i], pr.val[pr.val.size() - 1]);
        }
    }
    vector<vector<F> >::iterator itf = pr.val.begin();
    int cnt2 = 0;
    for (vector<layer>::iterator it = pr.C.circuit.begin(); it < pr.C.circuit.end();) {
        if ((int)it->ty == 4 || (int)it->ty == 11) {
            if (cnt2 < cntp) {
                it = pr.C.circuit.erase(it);
                itf = pr.val.erase(itf);
                cnt2++;
            } else {
                ++it;
                ++itf;
            }
        } else {
            ++it;
            ++itf;
        }
    }
    int offset[5] = {0, 0, 0, 0, 0};
    int lazy_offset[5] = {0, 0, 0, 0, 0};
    for (int i = 4; i < layer_id; i++) {
        auto t = pr.C.circuit[i].ty;
        int os = 0;
        if (t == layerType::MHA_QK || t == layerType::GELU_1 || t == layerType::LAYER_NORM_1)
            os = 1;
        else if (t == layerType::SOFTMAX_1 || t == layerType::GELU_2 || t == layerType::LAYER_NORM_2)
            os = 2;
        else if (t == layerType::SOFTMAX_2 || t == layerType::GELU_3 || t == layerType::LAYER_NORM_3)
            os = 3;
        else if (t == layerType::SOFTMAX_3)
            os = 4;
        else
            break;
        assert(os != 0);
        assert(pr.val[i].size() == pr.C.circuit[i].size);

        for (int j = 0; j < pr.C.circuit[i].size; j++) {
            pr.val[os].emplace_back(pr.val[i][j]);
        }
    }
    for (int i = 1; i < 4; i++) {
        auto t = pr.C.circuit[i].ty;
        int os = 0, os1 = 0;
        if (t == layerType::MHA_QK || t == layerType::GELU_1 || t == layerType::LAYER_NORM_1)
            os = 1;
        else if (t == layerType::SOFTMAX_1 || t == layerType::GELU_2 || t == layerType::LAYER_NORM_2)
            os = 2;
        else if (t == layerType::SOFTMAX_2 || t == layerType::GELU_3 || t == layerType::LAYER_NORM_3)
            os = 3;
        else if (t == layerType::SOFTMAX_3)
            os = 4;
        if (os == i) {
            pr.C.circuit[os].uni_interval.emplace_back(make_pair(0, pr.C.circuit[i].uni_gates.size()));
            pr.C.circuit[os].bin_interval.emplace_back(make_pair(0, pr.C.circuit[i].bin_gates.size()));
        }
    }
    for (int i = 4; i < layer_id; i++) {
        auto t = pr.C.circuit[i].ty;
        int os = 0, os1 = 0;
        if (t == layerType::MHA_QK || t == layerType::GELU_1 || t == layerType::LAYER_NORM_1)
            os = 1;
        else if (t == layerType::SOFTMAX_1 || t == layerType::GELU_2 || t == layerType::LAYER_NORM_2)
            os = 2;
        else if (t == layerType::SOFTMAX_2 || t == layerType::GELU_3 || t == layerType::LAYER_NORM_3)
            os = 3;
        else if (t == layerType::SOFTMAX_3)
            os = 4;
        else
            break;
        assert(os != 0);

        auto t2 = pr.C.circuit[i - 1].ty;
        if (t2 == layerType::MHA_QK || t2 == layerType::GELU_1 || t2 == layerType::LAYER_NORM_1)
            os1 = 1;
        else if (t2 == layerType::SOFTMAX_1 || t2 == layerType::GELU_2 || t2 == layerType::LAYER_NORM_2)
            os1 = 2;
        else if (t2 == layerType::SOFTMAX_2 || t2 == layerType::GELU_3 || t2 == layerType::LAYER_NORM_3)
            os1 = 3;
        else if (t2 == layerType::SOFTMAX_3)
            os1 = 4;

        assert(i != os);
        offset[os1] = offset[os] = 0;
        for (int j = 1; j < i; j++) {
            auto tp = pr.C.circuit[j].ty;
            int tos = 0;
            if (tp == layerType::MHA_QK || tp == layerType::GELU_1 || tp == layerType::LAYER_NORM_1)
                tos = 1;
            else if (tp == layerType::SOFTMAX_1 || tp == layerType::GELU_2 || tp == layerType::LAYER_NORM_2)
                tos = 2;
            else if (tp == layerType::SOFTMAX_2 || tp == layerType::GELU_3 || tp == layerType::LAYER_NORM_3)
                tos = 3;
            else if (tp == layerType::SOFTMAX_3)
                tos = 4;
            if (tos == os)
                offset[os] += pr.C.circuit[j].size;
        }
        for (int j = 1; j < i - 1; j++) {
            auto tp = pr.C.circuit[j].ty;
            int tos = 0;
            if (tp == layerType::MHA_QK || tp == layerType::GELU_1 || tp == layerType::LAYER_NORM_1)
                tos = 1;
            else if (tp == layerType::SOFTMAX_1 || tp == layerType::GELU_2 || tp == layerType::LAYER_NORM_2)
                tos = 2;
            else if (tp == layerType::SOFTMAX_2 || tp == layerType::GELU_3 || tp == layerType::LAYER_NORM_3)
                tos = 3;
            else if (tp == layerType::SOFTMAX_3)
                tos = 4;
            if (tos == os1)
                offset[os1] += pr.C.circuit[j].size;
        }
        pr.C.circuit[os].uni_interval.emplace_back(
            make_pair(pr.C.circuit[os].uni_gates.size(),
                      pr.C.circuit[os].uni_gates.size() + pr.C.circuit[i].uni_gates.size()));
        for (auto g = pr.C.circuit[i].uni_gates.begin(); g != pr.C.circuit[i].uni_gates.end(); g++) {
            if ((int)g->lu == 0)
                pr.C.circuit[os].uni_gates.emplace_back(g->g + offset[os], g->u, 0, g->sc);
            else {
                pr.C.circuit[os].uni_gates.emplace_back(g->g + offset[os], g->u + offset[os1], os1, g->sc);
                assert(os == os1 + 1);
                assert(g->u < pr.C.circuit[i - 1].size);
                assert(g->u + offset[os1] < pr.val[os1].size());
            }
        }
        pr.C.circuit[os].bin_interval.emplace_back(
            make_pair(pr.C.circuit[os].bin_gates.size(),
                      pr.C.circuit[os].bin_gates.size() + pr.C.circuit[i].bin_gates.size()));
        for (auto g = pr.C.circuit[i].bin_gates.begin(); g != pr.C.circuit[i].bin_gates.end(); g++) {
            if ((int)g->l == 0)
                pr.C.circuit[os].bin_gates.emplace_back(g->g + offset[os], g->u, g->v, g->sc, g->l);
            else if ((int)g->l == 1)
                pr.C.circuit[os].bin_gates.emplace_back(g->g + offset[os], g->u + offset[os1],
                                                        g->v + offset[os1], g->sc, g->l);
            else if ((int)g->l == 2)
                pr.C.circuit[os].bin_gates.emplace_back(g->g + offset[os], g->u + offset[os1], g->v, g->sc,
                                                        g->l);
            assert(g->g + offset[os] < pr.val[os].size());
        }
    }
    pr.C.circuit.erase(pr.C.circuit.begin() + 5, pr.C.circuit.begin() + layer_id - cntp);
    pr.val.erase(pr.val.begin() + 5, pr.val.begin() + layer_id - cntp);

    pr.C.size = pr.C.circuit.size();
    for (int i = 1; i < pr.C.size; i++) {
        initLayer(pr.C.circuit[i], pr.val[i].size(), pr.C.circuit[i].ty);
        if (pr.C.circuit[i].ty != layerType::FCONN)
            checkNormalLayer(pr.C.circuit[i], i, pr.val);
    }
}
void neuralNetwork::create(prover& pr, bool merge, timer& circuit_initialization_timer, int part_idx) {
    // Same budget prover.cpp/hyrax_rp.cpp/range_prover.cpp pass to num_threads(max_threads)
    cpu_threads = (pr.max_threads > 0) ? pr.max_threads : 1;

    // Hash the FCONN shape BEFORE initParam
    unsigned long long wc_fc_hash = 0;
    for (const auto& fc : full_conn) {
        wc_fc_hash = zkgpt_wc::mix64(wc_fc_hash, (unsigned long long)fc.channel_in);
        wc_fc_hash = zkgpt_wc::mix64(wc_fc_hash, (unsigned long long)fc.channel_out);
    }
    const zkgpt_wc::Key wc_key{
        layer_num,  len,      (int)output_private, (int)input_private, (int)input_separate,
        (int)merge, part_idx, wc_fc_hash};
    {
        zkgpt_wc::Derived d;
        int nfc = 0;
        if (zkgpt_wc::load(wc_key, pr, nfc, d)) {
            // The non-cache path starts this further down
            circuit_initialization_timer.start();
            q_offset = (int)d.q_offset;
            len = (int)d.len;
            channel_out = (int)d.channel_out;
            input_feature_dim = d.input_feature_dim;
            input_slot_size = d.input_slot_size;
            weight_start_offset = d.weight_start_offset;
            pr.inp_end_id = d.inp_end_id;
            input_c = d.input_c;
            input_e = d.input_e;
            pr.separate_input = input_private;
            pr.output_private = output_private;
            pr.input_not_committed = input_separate;
            pr.copyInputWitness();
            std::cout << "[witness_cache] HIT depth=" << layer_num << " len=" << len
                      << " part_idx=" << part_idx << std::endl;
            return;
        }
    }

    // Everything before the layer loop
    const double _init_t0 = prof_now_us();
    double _cpu_u0, _cpu_s0;
    zkgpt_witgen::cpu_times(&_cpu_u0, &_cpu_s0);
    compute_e_table();
    std::cout << "starting initParam" << std::endl;
    initParam(pr, layer_num);
    pr.C.init(Q_BIT_SIZE, SIZE);
    pr.val.resize(SIZE);
    val = pr.val.begin();

    // Pre-size val[0] from circuit shape
    {
        const i64 score_count = causalScoreCount();
        i64 precomputed = total_in_size;
        for (int i = 0; i < (int)full_conn.size(); ++i) {
            const i64 c_in = full_conn[i].channel_in;
            const i64 c_out = full_conn[i].channel_out;
            const i64 real_out = pr.fc_real_col[i];
            precomputed += 2 * c_out;
            precomputed += 20 + 2 * (len * c_out);
            if (i % 4 == 0 || i % 4 == 2)
                precomputed += 10 + 2 * (len * c_in) + 4 * len;
            if (i % 4 == 0)
                precomputed +=
                    10 + headnum * 2 * len + 4 * headnum * score_count + headnum * len * headdim + len * c_in;
            if (i % 4 == 2)
                precomputed += 10 + 3 * (len * c_out) + 3 * len * real_out;
        }
        val[0].reserve((size_t)precomputed);
        if (getenv("GKR_CHECK_RESERVE"))
            wc_reserved_in_size_ = precomputed;
    }

    i64 layer_id = 0;
    zkgpt_witgen::acc().init_s += (prof_now_us() - _init_t0) / 1e6;
    const double _inlayer_t0 = prof_now_us();
    std::cout << "starting inputLayer" << std::endl;
    inputLayer(pr.C.circuit[layer_id++]);
    std::cout << "starting readFconWeight" << std::endl;
    for (int i = 0; i < full_conn.size(); ++i) {
        auto& fc = full_conn[i];
        refreshFCParam(fc);
        readFconWeight(fc.weight_start_id, pr.fc_real_row[i], pr.fc_real_col[i], i);
    }
    int c = 0;
    pr.separate_input = input_private;
    pr.output_private = output_private;
    pr.input_not_committed = input_separate;
    // Full input witness is copied after auxiliary values are appended
    zkgpt_witgen::acc().inlayer_s += (prof_now_us() - _inlayer_t0) / 1e6;
    if (on_before_values) {
        on_before_values(input_c, input_e);
        auto& src = *custom_input_data;
        const i64 fdim = input_feature_dim;
        const i64 pdim = input_slot_size ? input_slot_size : (1LL << ceilPow2BitLength(fdim));
        if (custom_input_offset + len * fdim > (i64)src.size())
            throw std::runtime_error("prebuild: carry shorter than len*feature_dim");
        for (i64 i = 0; i < len; ++i)
            for (i64 j = 0; j < fdim; ++j)
                val[0][i * pdim + j] = src[custom_input_offset + i * fdim + j];
    }
    cout << "Start initiating circuit" << endl;
    circuit_initialization_timer.start();
    fprintf(stderr, "WITGEN_SIGNAL partition=%d phase=start t_us=%.0f\n", part_idx, prof_now_us());
    const double _values_t0 = prof_now_us();
    for (int i = 0; i < full_conn.size(); ++i) {
        auto& fc = full_conn[i];
        refreshFCParam(fc);
        if (i == 0) {
            q_offset = 0;
        }
        {
            zkgpt_witgen::Scope _s(zkgpt_witgen::acc().readln_s);
            read_layer_norm(0);
        }
        bool* sparsity = new bool[pr.fc_input_row[i] * pr.fc_input_col[i]];
        memset(sparsity, 0, sizeof(bool) * pr.fc_input_row[i] * pr.fc_input_col[i]);
        int cnt = 0;
        for (int j = 0; j < pr.fc_real_input_row[i]; j++)
            for (int k = 0; k < pr.fc_real_input_col[i]; k++) {
                ++cnt;
                sparsity[j * pr.fc_input_col[i] + k] = true;
            }
        if (i % 4 == 0 || i % 4 == 2) {
            {
                zkgpt_witgen::Scope _s(zkgpt_witgen::acc().ln1_s);
                ln_checker_layer1(pr.C.circuit[layer_id], layer_id, 0, input_e, input_c,
                                  pr.fc_real_input_col[0], sparsity);
            }
            {
                zkgpt_witgen::Scope _s(zkgpt_witgen::acc().ln2_s);
                ln_checker_layer2(pr.C.circuit[layer_id], layer_id, 0, input_e, input_c, sparsity);
            }
            {
                zkgpt_witgen::Scope _s(zkgpt_witgen::acc().ln3_s);
                ln_checker_layer3(pr.C.circuit[layer_id], layer_id, 0, input_e, input_c,
                                  pr.fc_real_input_col[0], sparsity);
            }
        }
        pr.fc_input_id[i] = q_offset;
        fullyConnLayer(pr.C.circuit[layer_id], layer_id, fc.weight_start_id, q_offset, 0);
        int cx = 7, ex = -8, cy = 3, ey = -8;
        float c_A, e_A, c_B = 1, e_B = -10, c_C = 7, e_C = -8;
        c_A = input_c;
        e_A = input_e;

        {
            zkgpt_witgen::Scope _s(zkgpt_witgen::acc().round_s);
            roundLayer(pr.C.circuit[layer_id], layer_id, (float)c_A * c_B / c_C * pow(2, e_A + e_B - e_C));
        }
        if (i % 4 == 0) {
            {
                zkgpt_witgen::Scope _s(zkgpt_witgen::acc().qk_s);
                multi_head_matrix_QK(pr.C.circuit[layer_id], layer_id);
            }
            {
                zkgpt_witgen::Scope _s(zkgpt_witgen::acc().sm1_s);
                softmax_layer_1(pr.C.circuit[layer_id], layer_id, pow(2, e_C) * c_C, pow(2, e_C) * c_C,
                                pow(2, e_C) * c_C, pow(1, -8));
            }
            {
                zkgpt_witgen::Scope _s(zkgpt_witgen::acc().sm2_s);
                softmax_layer_2(pr.C.circuit[layer_id], layer_id, pow(2, e_C) * c_C, pow(2, e_C) * c_C,
                                pow(2, e_C) * c_C, pow(1, -8));
            }
            {
                zkgpt_witgen::Scope _s(zkgpt_witgen::acc().sm3_s);
                softmax_layer_3(pr.C.circuit[layer_id], layer_id, pow(2, e_C) * c_C, pow(2, e_C) * c_C,
                                pow(2, e_C) * c_C, pow(1, -8));
            }
        }
        if (i % 4 == 2) {
            {
                zkgpt_witgen::Scope _s(zkgpt_witgen::acc().gelu1_s);
                gelu_checker_layer1(pr.C.circuit[layer_id], layer_id, pr.fc_real_col[i], -8, 48, -8, 217, -8,
                                    252, -8, 615, ex, cx, ey, cy);
            }
            {
                zkgpt_witgen::Scope _s(zkgpt_witgen::acc().gelu2_s);
                gelu_checker_layer2(pr.C.circuit[layer_id], layer_id, pr.fc_real_col[i], -8, 48, -8, 217, -8,
                                    252, -8, 615, ex, cx, ey, cy);
            }
            {
                zkgpt_witgen::Scope _s(zkgpt_witgen::acc().gelu3_s);
                gelu_checker_layer3(pr.C.circuit[layer_id], layer_id, pr.fc_real_col[i], -8, 48, -8, 217, -8,
                                    252, -8, 615, ex, cx, ey, cy);
            }
        }
    }

    zkgpt_witgen::acc().values_s += (prof_now_us() - _values_t0) / 1e6;

    if (on_values_ready) {
        zkgpt_witgen::Scope _s(zkgpt_witgen::acc().carry_s);
        on_values_ready(pr);
    }

    if (merge) {
        zkgpt_witgen::Scope _s(zkgpt_witgen::acc().merge_s);
        merge_layer(pr, layer_id);
    }

    total_in_size += total_max_in_size + total_ave_in_size + total_relu_in_size;
    {
        zkgpt_witgen::Scope _s(zkgpt_witgen::acc().inlayer_s);
        initLayer(pr.C.circuit[0], total_in_size, layerType::INPUT);
    }
    assert(total_in_size == pr.val[0].size());
    if (wc_reserved_in_size_ && wc_reserved_in_size_ != total_in_size)
        fprintf(stderr, "WITGEN_SIGNAL partition=%d reserve_mismatch reserved=%lld actual=%lld\n", part_idx,
                (long long)wc_reserved_in_size_, (long long)total_in_size);

    // The builders' gate readbacks are async on the copy stream
    {
        zkgpt_witgen::Scope _s(zkgpt_witgen::acc().sync_s);
        zkgpt_gpu::witgen_sync_gates();
    }
    {
        zkgpt_witgen::Scope _s(zkgpt_witgen::acc().subset_s);
        pr.C.initSubset();
    }

    const double _misc_t0 = prof_now_us();
    int cnt = 0;
    for (int i = 0; i < pr.C.size; i++)
        cnt += pr.val[i].size();
    int bin = 0, uni = 0;
    for (int i = 0; i < pr.C.size; i++) {
        bin += pr.C.circuit[i].bin_gates.size();
        uni += pr.C.circuit[i].uni_gates.size();
    }
    pr.mat_val = mat_values;
    zkgpt_witgen::acc().misc_s += (prof_now_us() - _misc_t0) / 1e6;
    {
        zkgpt_witgen::Scope _s(zkgpt_witgen::acc().copyin_s);
        pr.copyInputWitness();
    }
    fprintf(stderr, "WITGEN_SIGNAL partition=%d phase=end t_us=%.0f val0_size=%zu\n", part_idx, prof_now_us(),
            pr.val[0].size());

    const double _teardown_t0 = prof_now_us();
    // val[0] alone is ~1.1 GiB on gpt2
    releaseFcResident();
    {
        double u1, s1;
        zkgpt_witgen::cpu_times(&u1, &s1);
        zkgpt_witgen::acc().cpu_user_s = u1 - _cpu_u0;
        zkgpt_witgen::acc().cpu_sys_s = s1 - _cpu_s0;
        zkgpt_witgen::acc().wall_s = (prof_now_us() - _init_t0) / 1e6;
    }
    zkgpt_witgen::report();
    zkgpt_gpu::witgen_report();
    zkgpt_gpu::witgen_teardown();

    {
        const zkgpt_wc::Derived d{
            q_offset,      len,     channel_out, input_feature_dim, input_slot_size, weight_start_offset,
            pr.inp_end_id, input_c, input_e};
        zkgpt_wc::save(wc_key, pr, (int)full_conn.size(), d);
    }
    zkgpt_witgen::acc().teardown_s += (prof_now_us() - _teardown_t0) / 1e6;
}

void neuralNetwork::inputLayer(layer& circuit) {
    initLayer(circuit, total_in_size, layerType::INPUT);

    // NO GATE LIST FOR LAYER 0

    calcInputLayer(circuit);
}

pair<int, int> search(double scale) {
    double mindiff = 1e9;
    int best_e, best_c;
    for (int e = -10; e <= 10; e++)
        for (int c = 1; c <= 800; c++) {
            double s = pow(2, e) * c;
            if (abs(s - scale) < mindiff) {
                mindiff = abs(s - scale);
                best_e = e;
                best_c = c;
            }
        }
    return make_pair(best_e, best_c);
}

void neuralNetwork::read_layer_norm(int ln_id) {
    i64 orgsize = val[0].size();
    val[0].resize(orgsize + 2 * channel_out);
    total_in_size += 2 * channel_out;
    layer_norm_w_c[ln_id] = 1;
    layer_norm_w_e[ln_id] = 0;
    layer_norm_b_c[ln_id] = 1;
    layer_norm_b_e[ln_id] = -8;
    layer_norm_w_q_start[ln_id] = orgsize;
    layer_norm_b_q_start[ln_id] = orgsize + channel_out;

    for (int i = 0; i < channel_out; i++)
        val[0][i + layer_norm_w_q_start[ln_id]] = 1;
    for (int i = 0; i < channel_out; i++)
        val[0][i + layer_norm_b_q_start[ln_id]] = 1;
}
std::ostream& operator<<(std::ostream& os, __int128_t value) {
    if (value < 0) {
        os << '-';
        value = -value;
    }
    // save flags to restore them
    std::ios_base::fmtflags flags(os.flags());
    // set zero fill
    os << std::setfill('0') << std::setw(13);

    // 128-bit number has at most 39 digits, so the below loop will run at most 3 times
    const int64_t modulus = 10000000000000;  // 10**13
    do {
        int64_t val = value % modulus;
        value /= modulus;
        if (value == 0) {
            os.flags(flags);
            return os << val;
        }
        os << val;
    } while (1);
}
// Host reference paths for ln_checker_layer1
void neuralNetwork::ln_checker_layer1_cpu_values(F* out, i64 orgsize, i64 block_len, int real_cn_in,
                                                 i64 qw_off, i64 qb_off, int c1, int e1, int c2, int e2,
                                                 int m, bool* sparsity_map) {
    out[0] = 1;
    std::vector<ll> qw(channel_in), qb(channel_in);
    for (i64 co = 0; co < channel_in; ++co) {
        qw[co] = convert(val[0][qw_off + co]);
        qb[co] = convert(val[0][qb_off + co]);
    }
    __int128 mn = 1;
// qx/a were shared scratch reused across rows in the original serial loop
#pragma omp parallel for schedule(static) num_threads(cpu_threads)
    for (i64 i = 0; i < len; i++) {
        std::vector<ll> qx(channel_in, 0), a(channel_in, 0);
        ll sum = 0, B = 1, sigma;
        if (!sparsity_map[i * channel_in]) {
            continue;
        }
        for (i64 co = 0; co < channel_in; ++co) {
            i64 g = matIdx(i, co, channel_in);
            qx[co] = convert(val[0][q_offset + g]);
            sum += qx[co];
        }
        for (i64 co = 0; co < channel_in; ++co) {
            if (!sparsity_map[i * channel_in + co])
                continue;
            a[co] = qx[co] * real_cn_in - sum;
            B += a[co] * a[co];
        }
        assert(B != 0);
        sigma = round(sqrt(B));
        assert(sigma != 0);
        assert(sigma * sigma + sigma + 1 - B < (1ll << 32) && sigma * sigma + sigma + 1 - B > 0);
        assert(B - sigma * sigma + sigma < (1ll << 32) && B - sigma * sigma + sigma > 0);
        Fr delta1 = Fr(sigma * sigma + sigma + 1 - B) * Fr(B - sigma * sigma + sigma);
        for (i64 co = 0; co < channel_in; ++co) {
            i64 g = matIdx(i, co, channel_in);
            if (!sparsity_map[g]) {
                continue;
            }
            ll qy = round(pow(2, e1) * c1 * qw[co] * a[co] / sigma + pow(2, e2) * c2 * qb[co]);

            i64 y_off = orgsize + 10 + g;
            // Must index with real_cn_in
            i64 d2_off = orgsize + 10 + block_len + len * 4 + matIdx(i, co, real_cn_in);
            out[y_off - orgsize] = qy;
            ll term1, term2;
            term1 = (2 * qy + 1) * (1ll << (m - 1)) * sigma + 1 - (1ll << (e1 + m)) * c1 * qw[co] * a[co] -
                    (1ll << (e2 + m)) * c2 * qb[co] * sigma;
            term2 = (1ll << (e1 + m)) * c1 * qw[co] * a[co] + (1ll << (e2 + m)) * c2 * qb[co] * sigma -
                    (2 * qy - 1) * (1ll << (m - 1)) * sigma + 1;
            assert(term1 > 0 && term2 > 0);
            out[d2_off - orgsize] = Fr(term1) * Fr(term2);
        }
        i64 sum_off = orgsize + 10 + block_len + i;
        out[sum_off - orgsize] = sum;
        i64 b_off = orgsize + 10 + block_len + len + i;
        out[b_off - orgsize] = B;
        i64 sig_off = orgsize + 10 + block_len + len * 2 + i;
        out[sig_off - orgsize] = sigma;
        i64 d1_off = orgsize + 10 + block_len + len * 3 + i;
        out[d1_off - orgsize] = delta1;
    }
}

void neuralNetwork::ln_checker_layer1_cpu_gates(std::vector<uniGate>& uni, std::vector<binGate>& bin,
                                                i64 orgsize, i64 block_len, int real_cn_in,
                                                bool* sparsity_map) {
    Fr SUM = 0;
    for (i64 i = 0; i < len; i++) {
        if (!sparsity_map[i * channel_in]) {
            continue;
        }
        i64 sum_off = orgsize + 10 + block_len + i;
        i64 sig_off = orgsize + 10 + block_len + len * 2 + i;
        i64 b_off = orgsize + 10 + block_len + len + i;
        for (i64 co = 0; co < channel_in; ++co) {
            int g = matIdx(i, co, channel_in);
            if (!sparsity_map[g]) {
                continue;
            }
            uni.emplace_back(g, q_offset + g, 0, real_cn_in);  // verify a with x and sum
            uni.emplace_back(g, sum_off, 0, -1);
            if (i == 2) {
                SUM += val[0][q_offset + g];
            }
            uni.emplace_back(block_len + len * 2 + i, q_offset + g, 0, 1);  // +xi
        }
        bin.emplace_back(block_len + i, sig_off, sig_off, 1, 0);  // sigma^2
        uni.emplace_back(block_len + i, sig_off, 0, 1);           // sigma
        uni.emplace_back(block_len + i, orgsize, 0, 1);
        uni.emplace_back(block_len + i, b_off, 0, -1);

        bin.emplace_back(block_len + len + i, sig_off, sig_off, -1, 0);  // sigma^2
        uni.emplace_back(block_len + len + i, sig_off, 0, 1);            // sigma
        uni.emplace_back(block_len + len + i, b_off, 0, 1);
        uni.emplace_back(block_len + len * 2 + i, sum_off, 0, -1);  // SUM
    }
}

void neuralNetwork::ln_checker_layer1(layer& circuit, i64& layer_id, int ln_id, int ey, int cy,
                                      int real_cn_in, bool* sparsity_map) {
    int cw = layer_norm_w_c[ln_id];
    int ew = layer_norm_w_e[ln_id];
    int cb = layer_norm_b_c[ln_id];  // scale b
    int eb = layer_norm_b_e[ln_id];
    double sw = pow(2, ew) * cw, sb = pow(2, eb) * cb;
    double sy = pow(2, ey) * cy;
    i64 qw_off = layer_norm_w_q_start[ln_id];  // place of w vector
    i64 qb_off = layer_norm_b_q_start[ln_id];  // place of b vector

    int c1, e1, c2, e2;
    pair<int, int> S1, S2;
    S1 = search(sw * sqrt(real_cn_in) / sy);
    S2 = search(sb / sy);
    e1 = S1.first;
    c1 = S1.second;
    e2 = S2.first;
    c2 = S2.second;
    layer_norm_c1[ln_id] = c1;
    layer_norm_e1[ln_id] = e1;
    layer_norm_c2[ln_id] = c2;
    layer_norm_e2[ln_id] = e2;
    int m = multi_max::max(1, -e1, -e2);
    i64 block_len = len * channel_in;
    int output_size = block_len + 3 * len;
    initLayer(circuit, output_size, layerType::LAYER_NORM_1);
    circuit.need_phase2 = true;
    circuit.zero_start_id = block_len + 2 * len;
    i64 orgsize = val[0].size();
    const size_t ln1_alloc = 10 + (size_t)block_len * 2 + 4 * (size_t)len;
    val[0].resize(orgsize + ln1_alloc);
    ln_aux_start = orgsize;
    total_relu_in_size += 10 + block_len * 2 + 4 * len;
    if (gkr_config.witgen_full || gkr_config.witgen_full_check) {
        // Same rectangle assumption as ln_checker_layer2, verified the same way
        int real_row = 0, real_col = 0;
        while (real_col < channel_in && sparsity_map[real_col])
            ++real_col;
        while (real_row < len && sparsity_map[(i64)real_row * channel_in])
            ++real_row;
        for (i64 i = 0; i < len; ++i)
            for (i64 co = 0; co < channel_in; ++co)
                assert(sparsity_map[i * channel_in + co] == (i < real_row && co < real_col));

        std::vector<F> ref_v;
        std::vector<uniGate> ref_u;
        std::vector<binGate> ref_b;
        if (gkr_config.witgen_full_check) {
            ref_v.assign(ln1_alloc, F_ZERO);
            ln_checker_layer1_cpu_values(ref_v.data(), orgsize, block_len, real_cn_in, qw_off, qb_off, c1, e1,
                                         c2, e2, m, sparsity_map);
            ln_checker_layer1_cpu_gates(ref_u, ref_b, orgsize, block_len, real_cn_in, sparsity_map);
        }
        val[layer_id].resize(circuit.size);
        zkgpt_gpu::gpu_ln1(val[0].data(), (size_t)orgsize, ln1_alloc, wgPrevData(layer_id),
                           wgPrevSize(layer_id), val[layer_id].data(), circuit.size, circuit.uni_gates,
                           circuit.bin_gates, q_offset, block_len, (int)len, channel_in, real_row, real_col,
                           real_cn_in, qw_off, qb_off, c1, e1, c2, e2, m);
        if (gkr_config.witgen_full_check) {
            for (size_t k = 0; k < ln1_alloc; ++k)
                if (ref_v[k] != val[0][orgsize + k]) {
                    fprintf(stderr, "[ln1-check] VALUE MISMATCH idx=%zu\n", k);
                    abort();
                }
            zkgpt_gpu::witgen_sync_gates();
            wg_check_gates("ln1-check", ref_u, circuit.uni_gates, ref_b, circuit.bin_gates);
        }
        positive_check += real_row * (real_col + 1);
    } else {
        for (i64 i = orgsize; i < (i64)val[0].size(); i++)
            val[0][i] = 0;
        val[0][orgsize] = 1;
        ln_checker_layer1_cpu_values(&val[0][orgsize], orgsize, block_len, real_cn_in, qw_off, qb_off, c1, e1,
                                     c2, e2, m, sparsity_map);
        ln_checker_layer1_cpu_gates(circuit.uni_gates, circuit.bin_gates, orgsize, block_len, real_cn_in,
                                    sparsity_map);
        calcNormalLayer(circuit, layer_id);
    }
    for (i64 i = 0; i < len; i++) {
        assert(val[layer_id][block_len + len * 2 + i].isZero());
    }

    layer_id++;
}

// Host reference path for ln_checker_layer2's gate emission
void neuralNetwork::ln_checker_layer2_cpu_gates(std::vector<uniGate>& uni, std::vector<binGate>& bin,
                                                i64 orgsize, i64 block_len, int m, int c1, int e1, int c2,
                                                int e2, i64 qw_off, i64 qb_off, bool* sparsity_map) {
    for (i64 i = 0; i < len; i++) {
        i64 sum_off = orgsize + 10 + block_len + i;
        i64 sig_off = orgsize + 10 + block_len + len * 2 + i;
        i64 b_off = orgsize + 10 + block_len + len + i;

        for (i64 co = 0; co < channel_in; ++co) {
            int g = matIdx(i, co, channel_in);
            i64 y_off = orgsize + 10 + g;
            if (!sparsity_map[g]) {
                continue;
            }
            bin.emplace_back(g, g, qw_off + co, -(1ll << (e1 + m)) * c1, 2);  // qw[co]*a[co]
            bin.emplace_back(g, sig_off, qb_off + co, -(1ll << (e2 + m)) * c2, 0);
            bin.emplace_back(g, sig_off, y_off, 1 << m, 0);
            uni.emplace_back(g, orgsize, 0, 1);
            uni.emplace_back(g, sig_off, 0, (1 << (m - 1)));

            bin.emplace_back(g + block_len, g, qw_off + co, (1ll << (e1 + m)) * c1, 2);  // qw[co]*a[co]
            bin.emplace_back(g + block_len, sig_off, qb_off + co, (1ll << (e2 + m)) * c2, 0);
            bin.emplace_back(g + block_len, sig_off, y_off, -(1 << m), 0);
            uni.emplace_back(g + block_len, orgsize, 0, 1);
            uni.emplace_back(g + block_len, sig_off, 0, (1 << (m - 1)));

            bin.emplace_back(2 * block_len + len + i, g, g, 1, 1);
            // No sum of squares is accumulated here: it would emit no gate
        }
        if (!sparsity_map[i * channel_in]) {
            continue;
        }
        ll P = convert(val[0][b_off]);
        bin.emplace_back(2 * block_len + i, block_len + i, block_len + len + i, 1, 1);
        i64 d1_off = orgsize + 10 + block_len + len * 3 + i;
        uni.emplace_back(2 * block_len + i, d1_off, 0, -1);
        uni.emplace_back(2 * block_len + len + i, orgsize, 0, 1);
        uni.emplace_back(2 * block_len + len + i, b_off, 0, -1);  // b=sigma^2+1
    }
}

void neuralNetwork::ln_checker_layer2(layer& circuit, i64& layer_id, int ln_id, int ey, int cy,
                                      bool* sparsity_map) {
    int cw = layer_norm_w_c[ln_id];
    int ew = layer_norm_w_e[ln_id];
    int cb = layer_norm_b_c[ln_id];  // scale b
    int eb = layer_norm_b_e[ln_id];
    i64 qw_off = layer_norm_w_q_start[ln_id];  // place of w vector
    i64 qb_off = layer_norm_b_q_start[ln_id];  // place of b vector
    int c1, e1, c2, e2;
    c1 = layer_norm_c1[ln_id];
    e1 = layer_norm_e1[ln_id];
    c2 = layer_norm_c2[ln_id];
    e2 = layer_norm_e2[ln_id];
    int m = multi_max::max(-e1, -e2, 1);
    i64 block_len = len * channel_in;
    int output_size = 2 * block_len + 2 * len;
    initLayer(circuit, output_size, layerType::LAYER_NORM_2);
    circuit.need_phase2 = true;
    circuit.zero_start_id = 2 * block_len;
    i64 orgsize = ln_aux_start;
    if (gkr_config.witgen_full || gkr_config.witgen_full_check) {
        // The sparsity map is a rectangle by construction
        int real_row = 0, real_col = 0;
        while (real_col < channel_in && sparsity_map[real_col])
            ++real_col;
        while (real_row < len && sparsity_map[(i64)real_row * channel_in])
            ++real_row;
        for (i64 i = 0; i < len; ++i)
            for (i64 co = 0; co < channel_in; ++co)
                assert(sparsity_map[i * channel_in + co] == (i < real_row && co < real_col));

        std::vector<uniGate> ref_u;
        std::vector<binGate> ref_b;
        if (gkr_config.witgen_full_check)
            ln_checker_layer2_cpu_gates(ref_u, ref_b, orgsize, block_len, m, c1, e1, c2, e2, qw_off, qb_off,
                                        sparsity_map);
        val[layer_id].resize(circuit.size);
        zkgpt_gpu::gpu_ln2_gates(val[0].data(), val[0].size(), wgPrevData(layer_id), wgPrevSize(layer_id),
                                 val[layer_id].data(), circuit.size, circuit.uni_gates, circuit.bin_gates,
                                 orgsize, block_len, channel_in, real_row, real_col, (int)len, qw_off, qb_off,
                                 (1ll << (e1 + m)) * (ll)c1, (1ll << (e2 + m)) * (ll)c2, (ll)(1 << m),
                                 (ll)(1 << (m - 1)));
        if (gkr_config.witgen_full_check) {
            zkgpt_gpu::witgen_sync_gates();
            wg_check_gates("ln2-check", ref_u, circuit.uni_gates, ref_b, circuit.bin_gates);
        }
    } else {
        ln_checker_layer2_cpu_gates(circuit.uni_gates, circuit.bin_gates, orgsize, block_len, m, c1, e1, c2,
                                    e2, qw_off, qb_off, sparsity_map);
        calcNormalLayer(circuit, layer_id);
    }
    for (i64 i = 0; i < block_len; i++) {
        assert(!val[layer_id][i].isNegative());  // only need to check the sparse items of these
        assert(!val[layer_id][block_len + i].isNegative());
        if (i < block_len) {
            if (sparsity_map[i] == false)
                assert(val[layer_id][i].isZero());
            if (sparsity_map[i] == false)
                assert(val[layer_id][i + block_len].isZero());
        }
    }
    for (i64 i = 0; i < len; i++) {
        i64 b_off = orgsize + 10 + block_len + len + i;
        assert(val[layer_id][2 * block_len + len + i].isZero());
        assert(val[layer_id][2 * block_len + i].isZero());
    }

    layer_id++;
}

// Host reference path for ln_checker_layer3's gate emission
void neuralNetwork::ln_checker_layer3_cpu_gates(std::vector<uniGate>& uni, std::vector<binGate>& bin,
                                                i64 orgsize, i64 block_len, int real_cn_in) {
    for (i64 i = 0; i < len; i++) {
        for (i64 co = 0; co < real_cn_in; ++co) {
            int g = matIdx(i, co, channel_in);
            int p = matIdx(i, co, real_cn_in);
            i64 d2_off = orgsize + 10 + block_len + len * 4 + p;
            uni.emplace_back(p, d2_off, 0, -1);
            bin.emplace_back(p, g, g + block_len, 1, 1);
        }
    }
}

void neuralNetwork::ln_checker_layer3(layer& circuit, i64& layer_id, int ln_id, int ey, int cy,
                                      int real_cn_in, bool* sparsity_map) {
    i64 block_len = len * channel_in;
    int output_size = len * real_cn_in;
    initLayer(circuit, output_size, layerType::LAYER_NORM_3);
    circuit.need_phase2 = true;
    circuit.zero_start_id = 0;
    i64 orgsize = ln_aux_start;  // use previous layer's org size

    if (gkr_config.witgen_full || gkr_config.witgen_full_check) {
        std::vector<uniGate> ref_u;
        std::vector<binGate> ref_b;
        if (gkr_config.witgen_full_check)
            ln_checker_layer3_cpu_gates(ref_u, ref_b, orgsize, block_len, real_cn_in);
        val[layer_id].resize(circuit.size);
        zkgpt_gpu::gpu_ln3(val[0].data(), val[0].size(), wgPrevData(layer_id), wgPrevSize(layer_id),
                           val[layer_id].data(), circuit.size, circuit.uni_gates, circuit.bin_gates, orgsize,
                           block_len, (int)len, channel_in, real_cn_in);
        if (gkr_config.witgen_full_check) {
            zkgpt_gpu::witgen_sync_gates();
            wg_check_gates("ln3-check", ref_u, circuit.uni_gates, ref_b, circuit.bin_gates);
        }
    } else {
        ln_checker_layer3_cpu_gates(circuit.uni_gates, circuit.bin_gates, orgsize, block_len, real_cn_in);
        calcNormalLayer(circuit, layer_id);
    }
    for (int g = 0; g < output_size; g++) {
        assert(val[layer_id][g].isZero());  // assert will fail without UNDEF NDEBUG on cmake
    }
    layer_id++;
    q_offset = ln_aux_start + 10;  // get the rounded result for next computation
}

// Host reference path for gelu_checker_layer1's per-element value loop
void neuralNetwork::gelu_checker_layer1_cpu_values(F* out, i64 block_len, int real_cn_out, ll C1, ll C2,
                                                   ll C3, ll C4, ll C5, ll C6, ll C7, int ca, int cb, int cc,
                                                   int cx, int cy, int ea, int eb, int ec, int ex, int ey) {
    const int y_off = 10;
    const int abs_off = 10 + block_len;
    const int t_off = 10 + block_len * 2;
    const int d1_off = 10 + block_len * 3;
    const int d2_off = 10 + block_len * 3 + len * real_cn_out;
    const int d3_off = 10 + block_len * 3 + len * real_cn_out * 2;

// Per-g independent
#pragma omp parallel for schedule(static) num_threads(cpu_threads)
    for (i64 g = 0; g < block_len; ++g) {
        if (g % channel_out >= real_cn_out)
            continue;
        ll qx = convert(val[0][g + q_offset]);

        ll abs, t;
        if (qx < 0)
            out[abs_off + g] = abs = -qx;
        else
            out[abs_off + g] = abs = qx;
        if (C6 >= C7 * abs) {
            out[t_off + g] = t = 1;
        } else {
            out[t_off + g] = t = 0;
        }
        int gp = g / channel_out * real_cn_out + g % channel_out;
        out[d1_off + gp] = abs + 1;
        out[d2_off + gp] = t + (1 - 2 * t) * (C7 * abs - C6);
        assert(!out[d1_off + gp].isNegative());
        assert(!out[d2_off + gp].isNegative());
        double inner = (double)ca * cx * cx * qx * qx * pow(2, ea + 2 * ex) -
                       cb * cx * abs * pow(2, eb + ex) + cc * pow(2, ec);
        double middle = (double)qx + abs - abs * t * inner;
        double final = (double)cx * pow(2, ex - 1 - ey) * middle / cy;
        ll y = round(final);
        out[y_off + g] = (ll)y;
        ll term1 = (2 * y + 1) * C5 + 1 - C1 * qx - C1 * abs + C2 * t * abs * abs * abs - C3 * t * abs * abs +
                   C4 * t * abs;
        ll term2 = C1 * qx + C1 * abs - C2 * t * abs * abs * abs + C3 * t * abs * abs - C4 * t * abs -
                   (2 * y - 1) * C5 + 1;
        assert(term1 > 0);
        assert(term2 > 0);
        out[d3_off + gp] = Fr(term1) * Fr(term2);
    }
    out[0] = 1;
}

// Host reference path for gelu_checker_layer1's gate emission
void neuralNetwork::gelu_checker_layer1_cpu_gates(std::vector<uniGate>& uni, std::vector<binGate>& bin,
                                                  i64 orgsize, i64 block_len, int real_cn_out, ll C6, ll C7) {
    for (i64 g = 0; g < block_len; ++g) {
        if (g % channel_out >= real_cn_out)
            continue;
        int gp = g / channel_out * real_cn_out + g % channel_out;

        i64 abs_off = orgsize + 10 + block_len + g;

        i64 t_off = orgsize + 10 + block_len * 2 + g;

        i64 d1_off = orgsize + 10 + block_len * 3 + gp;
        i64 d2_off = orgsize + 10 + block_len * 3 + len * real_cn_out + gp;

        i64 c1 = orgsize;

        i64 q_off = g + q_offset;

        int abs_mult_t_off = g;
        int q_square_off = g + block_len;
        i64 dt1_off = g + block_len * 2;
        i64 dt2_off = g + block_len * 3;
        int t2_off = g + block_len * 4;
        int abs_check_off = g + block_len * 5;
        bin.emplace_back(abs_mult_t_off, abs_off, t_off, 1, 0);  // sc, layer

        bin.emplace_back(q_square_off, q_off, q_off, 1, 0);  // sc, layer

        uni.emplace_back(dt1_off, abs_off, 0, 1);  // g,u,lu,sc
        uni.emplace_back(dt1_off, c1, 0, 1);       // g,u,lu,sc
        uni.emplace_back(dt1_off, d1_off, 0, -1);  // g,u,lu,sc

        uni.emplace_back(dt2_off, c1, 0, -C6);
        uni.emplace_back(dt2_off, t_off, 0, 2ll * C6 + 1);
        uni.emplace_back(dt2_off, abs_off, 0, C7);
        bin.emplace_back(dt2_off, abs_off, t_off, -2ll * C7, 0);
        uni.emplace_back(dt2_off, d2_off, 0, -1);  // g,u,lu,sc

        bin.emplace_back(t2_off, t_off, t_off, 1, 0);  // sc, layer
        uni.emplace_back(t2_off, t_off, 0, -1);        // g,u,lu,sc

        bin.emplace_back(abs_check_off, q_off, q_off, 1, 0);       // sc, layer
        bin.emplace_back(abs_check_off, abs_off, abs_off, -1, 0);  // sc, layer
    }
}

void neuralNetwork::gelu_checker_layer1(layer& circuit, i64& layer_id, int real_cn_out, int ea, int ca,
                                        int eb, int cb, int ec, int cc, int ed, int cd, int ex, int cx,
                                        int ey, int cy) {
    int m1 = multi_max::max(0, -ec, -eb - ex, -ea - 2 * ex, ex - ey);
    int m2 = multi_max::max(0, -ed, -ex);
    ll C1 = cx * (1ll << (m1));
    ll C2 = (1ll << (ea + 2 * ex + m1)) * cx * ca * cx * cx;
    ll C3 = (1ll << (eb + ex + m1)) * cx * cb * cx;
    ll C4 = cx * cc * (1ll << (ec + m1));
    ll C5 = (1ll << (ey - ex + m1)) * cy;

    ll C6 = cd * (1ll << (ed + m2));
    ll C7 = cx * (1ll << (ex + m2));

    i64 block_len = len * channel_out;
    int output_size = 6 * block_len;
    initLayer(circuit, output_size, layerType::GELU_1);
    circuit.need_phase2 = true;
    circuit.zero_start_id = block_len * 2;
    i64 orgsize = val[0].size();
    const size_t gelu1_alloc = 10 + (size_t)block_len * 3 + (size_t)len * real_cn_out * 3;
    val[0].resize(orgsize + gelu1_alloc);  // Const
    gelu_aux_start = orgsize;
    positive_check += len * real_cn_out * 3;  // add positive check
    total_relu_in_size += 10 + len * real_cn_out * 3 + block_len * 3;

    const bool wg_gpu = gkr_config.witgen_full;
    const bool wg_check = gkr_config.witgen_full_check;

    if (wg_gpu || wg_check) {
        std::vector<uniGate> ref_uni;
        std::vector<binGate> ref_bin;
        if (wg_check) {
            // Snapshot the host paths BEFORE the GPU overwrites val[0], then compare
            std::vector<F> ref(gelu1_alloc);
            for (auto& x : ref)
                x.clear();
            gelu_checker_layer1_cpu_values(ref.data(), block_len, real_cn_out, C1, C2, C3, C4, C5, C6, C7, ca,
                                           cb, cc, cx, cy, ea, eb, ec, ex, ey);
            gelu_checker_layer1_cpu_gates(ref_uni, ref_bin, orgsize, block_len, real_cn_out, C6, C7);

            zkgpt_gpu::gpu_gelu1(val[0].data(), (size_t)orgsize, gelu1_alloc, q_offset, block_len, (int)len,
                                 channel_out, real_cn_out, C1, C2, C3, C4, C5, C6, C7, ca, cb, cc, cx, cy, ea,
                                 eb, ec, ex, ey,
                                 /*emit=*/true, &circuit.uni_gates, &circuit.bin_gates);

            for (size_t i = 0; i < gelu1_alloc; ++i)
                if (ref[i] != val[0][orgsize + i]) {
                    fprintf(stderr, "[gelu-check] VALUE MISMATCH idx=%zu\n", i);
                    abort();
                }
            zkgpt_gpu::witgen_sync_gates();
            wg_check_gates("gelu1-check", ref_uni, circuit.uni_gates, ref_bin, circuit.bin_gates);
        } else {
            zkgpt_gpu::gpu_gelu1(val[0].data(), (size_t)orgsize, gelu1_alloc, q_offset, block_len, (int)len,
                                 channel_out, real_cn_out, C1, C2, C3, C4, C5, C6, C7, ca, cb, cc, cx, cy, ea,
                                 eb, ec, ex, ey,
                                 /*emit=*/true, &circuit.uni_gates, &circuit.bin_gates);
        }
    } else {
        for (i64 i = orgsize; i < (i64)val[0].size(); i++)
            val[0][i] = 0;
        gelu_checker_layer1_cpu_values(&val[0][orgsize], block_len, real_cn_out, C1, C2, C3, C4, C5, C6, C7,
                                       ca, cb, cc, cx, cy, ea, eb, ec, ex, ey);
        gelu_checker_layer1_cpu_gates(circuit.uni_gates, circuit.bin_gates, orgsize, block_len, real_cn_out,
                                      C6, C7);
    }
    calcNormalLayer(circuit, layer_id);

    for (int g = 0; g < block_len; g++) {
        assert(!val[layer_id][g].isNegative());  // assert will fail without UNDEF NDEBUG on cmake
        assert(!val[layer_id][g + block_len].isNegative());
        if (g % channel_out >= real_cn_out) {
            assert(val[layer_id][g].isZero());
            assert(val[layer_id][g + block_len].isZero());
        }
        assert(val[layer_id][g + block_len * 2].isZero());
        assert(val[layer_id][g + block_len * 3].isZero());
        assert(val[layer_id][g + block_len * 4].isZero());
        assert(val[layer_id][g + block_len * 5].isZero());
    }
    layer_id++;
}

// Host reference path for gelu_checker_layer2's gate emission
void neuralNetwork::gelu_checker_layer2_cpu_gates(std::vector<uniGate>& uni, std::vector<binGate>& bin,
                                                  i64 orgsize, i64 layer_id, i64 block_len, int real_cn_out,
                                                  ll C1, ll C2, ll C3, ll C4, ll C5) {
    for (i64 g = 0; g < block_len; ++g) {
        if (g % channel_out >= real_cn_out)
            continue;
        i64 y_off = orgsize + 10 + g;
        i64 abs_off = orgsize + 10 + block_len + g;

        i64 t_off = orgsize + 10 + block_len * 2 + g;
        i64 c1 = orgsize;

        i64 q_off = g + q_offset;

        int term1_off = g;
        int term2_off = g + block_len;

        int abs_mult_t_off = g;
        int q_square_off = g + block_len;

        uni.emplace_back(term1_off, c1, 0, C5 + 1);     // C5+1
        uni.emplace_back(term1_off, y_off, 0, C5 * 2);  // 2*C5*y
        uni.emplace_back(term1_off, abs_off, 0, -C1);
        uni.emplace_back(term1_off, q_off, 0, -C1);
        uni.emplace_back(term1_off, abs_mult_t_off, layer_id - 1, C4);
        bin.emplace_back(term1_off, abs_mult_t_off, q_square_off, C2, 1);
        bin.emplace_back(term1_off, q_square_off, t_off, -C3, 2);

        uni.emplace_back(term2_off, c1, 0, C5 + 1);      // C5+1
        uni.emplace_back(term2_off, y_off, 0, -C5 * 2);  // 2*C5*y
        uni.emplace_back(term2_off, abs_off, 0, C1);
        uni.emplace_back(term2_off, q_off, 0, C1);
        uni.emplace_back(term2_off, abs_mult_t_off, layer_id - 1, -C4);
        bin.emplace_back(term2_off, abs_mult_t_off, q_square_off, -C2, 1);
        bin.emplace_back(term2_off, q_square_off, t_off, C3, 2);
    }
}

void neuralNetwork::gelu_checker_layer2(layer& circuit, i64& layer_id, int real_cn_out, int ea, int ca,
                                        int eb, int cb, int ec, int cc, int ed, int cd, int ex, int cx,
                                        int ey, int cy) {
    int m1 = multi_max::max(0, -ec, -eb - ex, -ea - 2 * ex, ex - ey);
    int m2 = multi_max::max(0, -ed, -ex);
    ll C1 = cx * (1ll << (m1));
    ll C2 = (1ll << (ea + 2 * ex + m1)) * cx * ca * cx * cx;
    ll C3 = (1ll << (eb + ex + m1)) * cx * cb * cx;
    ll C4 = cx * cc * (1ll << (ec + m1));
    ll C5 = (1ll << (ey - ex + m1)) * cy;

    ll C6 = cd * (1ll << (ed + m2));
    ll C7 = cx * (1ll << (ex + m2));
    i64 block_len = len * channel_out;
    int output_size = 2 * block_len;
    initLayer(circuit, output_size, layerType::GELU_2);
    circuit.need_phase2 = true;
    i64 orgsize = gelu_aux_start;  // use previous layer's org size

    if (gkr_config.witgen_full || gkr_config.witgen_full_check) {
        std::vector<uniGate> ref_u;
        std::vector<binGate> ref_b;
        if (gkr_config.witgen_full_check)
            gelu_checker_layer2_cpu_gates(ref_u, ref_b, orgsize, layer_id, block_len, real_cn_out, C1, C2, C3,
                                          C4, C5);
        val[layer_id].resize(circuit.size);
        zkgpt_gpu::gpu_gelu2(val[0].data(), val[0].size(), wgPrevData(layer_id), wgPrevSize(layer_id),
                             val[layer_id].data(), circuit.size, circuit.uni_gates, circuit.bin_gates,
                             q_offset, orgsize, block_len, (int)len, channel_out, real_cn_out, (int)layer_id,
                             C1, C2, C3, C4, C5);
        if (gkr_config.witgen_full_check) {
            zkgpt_gpu::witgen_sync_gates();
            wg_check_gates("gelu2-check", ref_u, circuit.uni_gates, ref_b, circuit.bin_gates);
            std::vector<F> refv(circuit.size);
            calcNormalLayer_cpu(circuit, layer_id, refv.data());
            for (u32 i = 0; i < circuit.size; ++i)
                if (refv[i] != val[layer_id][i]) {
                    fprintf(stderr, "[gelu2-check] VALUE MISMATCH at %u\n", i);
                    abort();
                }
        }
    } else {
        gelu_checker_layer2_cpu_gates(circuit.uni_gates, circuit.bin_gates, orgsize, layer_id, block_len,
                                      real_cn_out, C1, C2, C3, C4, C5);
        calcNormalLayer(circuit, layer_id);
    }
    for (int g = 0; g < block_len; g++) {
        if (g % channel_out >= real_cn_out) {
            assert(val[layer_id][g].isZero());
            assert(val[layer_id][g + block_len].isZero());
        }
        assert(!val[layer_id][g].isNegative());  // assert will fail without UNDEF NDEBUG on cmake
        assert(!val[layer_id][g + block_len].isNegative());
    }
    layer_id++;
}

// Host reference path for gelu_checker_layer3's gate emission
void neuralNetwork::gelu_checker_layer3_cpu_gates(std::vector<uniGate>& uni, std::vector<binGate>& bin,
                                                  i64 orgsize, i64 block_len, int real_cn_out) {
    for (i64 g = 0; g < block_len; ++g) {
        if (g % channel_out >= real_cn_out)
            continue;
        int gp = g / channel_out * real_cn_out + g % channel_out;

        i64 d3_off = orgsize + 10 + block_len * 3 + 2 * real_cn_out * len + gp;

        uni.emplace_back(gp, d3_off, 0, -1);
        bin.emplace_back(gp, g, g + block_len, 1, 1);
    }
}

void neuralNetwork::gelu_checker_layer3(layer& circuit, i64& layer_id, int real_cn_out, int ea, int ca,
                                        int eb, int cb, int ec, int cc, int ed, int cd, int ex, int cx,
                                        int ey, int cy) {
    i64 block_len = len * channel_out;
    int output_size = len * real_cn_out;
    initLayer(circuit, output_size, layerType::GELU_3);
    circuit.need_phase2 = true;
    circuit.zero_start_id = 0;
    i64 orgsize = gelu_aux_start;  // use previous layer's org size

    if (gkr_config.witgen_full || gkr_config.witgen_full_check) {
        std::vector<uniGate> ref_u;
        std::vector<binGate> ref_b;
        if (gkr_config.witgen_full_check)
            gelu_checker_layer3_cpu_gates(ref_u, ref_b, orgsize, block_len, real_cn_out);
        val[layer_id].resize(circuit.size);
        zkgpt_gpu::gpu_gelu3(val[0].data(), val[0].size(), wgPrevData(layer_id), wgPrevSize(layer_id),
                             val[layer_id].data(), circuit.size, circuit.uni_gates, circuit.bin_gates,
                             orgsize, block_len, (int)len, channel_out, real_cn_out);
        if (gkr_config.witgen_full_check) {
            zkgpt_gpu::witgen_sync_gates();
            wg_check_gates("gelu3-check", ref_u, circuit.uni_gates, ref_b, circuit.bin_gates);
        }
    } else {
        gelu_checker_layer3_cpu_gates(circuit.uni_gates, circuit.bin_gates, orgsize, block_len, real_cn_out);
        calcNormalLayer(circuit, layer_id);
    }
    for (int g = 0; g < output_size; g++)
        assert(val[layer_id][g].isZero());
    layer_id++;
    q_offset = gelu_aux_start + 10;
}

// Idempotent
void neuralNetwork::releaseFcResident() {
    if (fc_resident_dev == nullptr)
        return;
    HIP_CHECK(hipFree(fc_resident_dev));
    fc_resident_dev = nullptr;
    fc_resident_elems = 0;
    fc_resident_layer = -1;
}

void neuralNetwork::roundLayer_cpu_values(i64 layer_id, i64 block_len, F* out, int c, int m, int M) {
// Per-g independent: reads val[layer_id-1][g], writes out[g]/out[block_len+g] only
#pragma omp parallel for schedule(static) num_threads(cpu_threads)
    for (i64 g = 0; g < block_len; ++g) {
        ll p = convert(val[layer_id - 1][g]);
        ll q = round(p * c * pow(2, m));
        out[g] = q;  // compute non-linear round
        out[block_len + g] = Fr(p * c * (1ll << (m + M + 1)) + (1 << M) - q * (1 << (M + 1))) *
                             Fr(q * (1 << (M + 1)) + (1 << M) - c * (1ll << (m + M + 1)) * p);
        assert(!out[block_len + g].isNegative());
    }
}

// Host reference path for roundLayer's gate emission
void neuralNetwork::roundLayer_cpu_gates(std::vector<uniGate>& uni, std::vector<binGate>& bin, i64 orgsize,
                                         i64 block_len, int c, int m, int M, bool* sparsity_map) {
    for (i64 g = 0; g < block_len; ++g) {
        int p = g;
        i64 q = g + orgsize + 20;
        i64 c1 = orgsize;
        i64 s = q + block_len;
        if (sparsity_map) {
            if (!sparsity_map[g])
                continue;
        }
        bin.emplace_back(g, p, p, -(1ll << (2 * m + 2 * M + 2)) * c * c, 1);
        bin.emplace_back(g, p, q, (1ll << (m + 2 * M + 3)) * c, 2);
        bin.emplace_back(g, q, q, -(1 << (2 * M + 2)), 0);
        uni.emplace_back(g, c1, 0, (1ll << (2 * M)));  // this public input is one
        uni.emplace_back(g, s, 0, -1);
    }
}

void neuralNetwork::roundLayer(layer& circuit, i64& layer_id, float scale, bool* sparsity_map) {
    i64 block_len = len * channel_out;
    int c, m;
    pair<int, int> pm = search(scale);
    m = pm.first;
    c = pm.second;
    float virtual_scale = c * pow(2, m);
    i64 size = block_len;
    initLayer(circuit, size, layerType::RELU);
    circuit.need_phase2 = true;
    circuit.zero_start_id = 0;
    i64 orgsize = val[0].size();
    val[0].resize(orgsize + 20 + block_len * 2);  // Const

    q_offset = orgsize + 20;
    total_relu_in_size += 20 + block_len * 2;
    val[0][orgsize] = 1;
    int M = max(-m, 0);

    const bool round_gpu = gkr_config.round_gpu;
    const bool round_check = gkr_config.round_check;
    F* const qq_base = &val[0][orgsize + 20];

    if (round_gpu || round_check) {
        // Take the FCONN output in place if it is still resident, else stage
        gpu_fe* d_prev = static_cast<gpu_fe*>(fc_resident_dev);
        const bool fused =
            (d_prev != nullptr && fc_resident_layer == layer_id - 1 && fc_resident_elems == block_len);
        if (fused) {
            fc_resident_dev = nullptr;  // ownership moves to the callee
            fc_resident_elems = 0;
            fc_resident_layer = -1;
        } else {
            releaseFcResident();
            d_prev = nullptr;
            HIP_CHECK(GKR_DMALLOC(&d_prev, (size_t)block_len * sizeof(gpu_fe)));
            HIP_CHECK(zkgpt_gpu::gkr_stage_h2d(d_prev, reinterpret_cast<const gpu_fe*>(&val[layer_id - 1][0]),
                                               (size_t)block_len * sizeof(gpu_fe),
                                               zkgpt_gpu::gkr_copy_stream()));
            HIP_CHECK(hipStreamSynchronize(zkgpt_gpu::gkr_copy_stream()));
        }
        zkgpt_gpu::gpu_round_layer_from_device(d_prev, qq_base, block_len, c, m, M,
                                               /*free_prev=*/true);

        if (round_check) {
            // Field equality is exact
            std::vector<F> ref(2 * block_len);
            roundLayer_cpu_values(layer_id, block_len, ref.data(), c, m, M);
            for (i64 g = 0; g < 2 * block_len; ++g) {
                if (ref[g] != qq_base[g]) {
                    fprintf(stderr,
                            "[round-check] MISMATCH layer=%lld idx=%lld "
                            "(block_len=%lld c=%d m=%d M=%d)\n",
                            (long long)layer_id, (long long)g, (long long)block_len, c, m, M);
                    abort();
                }
            }
        }
    } else {
        releaseFcResident();  // CPU path reads val[] on the host
        roundLayer_cpu_values(layer_id, block_len, qq_base, c, m, M);
    }

    // create() never passes a sparsity map to roundLayer
    if ((gkr_config.witgen_full || gkr_config.witgen_full_check) && !sparsity_map) {
        std::vector<uniGate> ref_u;
        std::vector<binGate> ref_b;
        if (gkr_config.witgen_full_check)
            roundLayer_cpu_gates(ref_u, ref_b, orgsize, block_len, c, m, M, sparsity_map);
        val[layer_id].resize(circuit.size);
        zkgpt_gpu::gpu_round_gates(val[0].data(), val[0].size(), wgPrevData(layer_id), wgPrevSize(layer_id),
                                   val[layer_id].data(), circuit.size, circuit.uni_gates, circuit.bin_gates,
                                   orgsize, block_len, -(1ll << (2 * m + 2 * M + 2)) * (ll)c * c,
                                   (1ll << (m + 2 * M + 3)) * (ll)c, -(ll)(1 << (2 * M + 2)),
                                   (1ll << (2 * M)));
        if (gkr_config.witgen_full_check) {
            zkgpt_gpu::witgen_sync_gates();
            wg_check_gates("round-check", ref_u, circuit.uni_gates, ref_b, circuit.bin_gates);
        }
    } else {
        roundLayer_cpu_gates(circuit.uni_gates, circuit.bin_gates, orgsize, block_len, c, m, M, sparsity_map);
        calcNormalLayer(circuit, layer_id);
    }
    for (int i = 0; i < block_len; i++) {
        int p = i;
        i64 q = i + orgsize + 20;
        assert(val[layer_id][i].isZero());
    }
    layer_id++;
}

// Host reference path for multi_head_matrix_QK's gate emission
void neuralNetwork::multi_head_matrix_QK_cpu_gates(std::vector<uniGate>& uni, std::vector<binGate>& bin,
                                                   i64 score_count) {
    const int HEAD = headnum;
    const int HSIZE = headdim;
    (void)uni;
    for (int head = 0; head < HEAD; head++) {
        for (int i = 0; i < len; i++) {
            i64 row_start = causalRowStart(i);
            i64 key_count = causalKeyCount(i);
            for (i64 key = 0; key < key_count; ++key) {
                int targ_gate = head * score_count + row_start + key;
                for (int k = 0; k < HSIZE; k++) {
                    int q_feature = head * HSIZE + k;
                    i64 gate_i = q_offset + i * channel_out + q_feature;
                    i64 gate_j = q_offset + key * channel_out + attn_dim + q_feature;
                    bin.emplace_back(targ_gate, gate_i, gate_j, 1, 0);
                }
            }
        }
    }
}

void neuralNetwork::multi_head_matrix_QK(layer& circuit, i64& layer_id) {
    const int HEAD = headnum;
    const int HSIZE = headdim;
    assert(attn_dim == HEAD * HSIZE);
    const i64 score_count = causalScoreCount();
    int output_size = HEAD * score_count;
    initLayer(circuit, output_size, layerType::MHA_QK);
    circuit.need_phase2 = true;
    if (gkr_config.witgen_full || gkr_config.witgen_full_check) {
        std::vector<uniGate> ref_u;
        std::vector<binGate> ref_b;
        if (gkr_config.witgen_full_check)
            multi_head_matrix_QK_cpu_gates(ref_u, ref_b, score_count);

        // Both the sequence and non-sequence forms of gate_j are affine in the feature index
        std::vector<long long> qbase(len), kbase, row_start(len);
        std::vector<int> row_of(score_count);
        for (i64 i = 0; i < len; ++i) {
            qbase[i] = q_offset + i * channel_out;
            row_start[i] = causalRowStart(i);
            for (i64 key = 0; key < causalKeyCount(i); ++key)
                row_of[row_start[i] + key] = (int)i;
        }
        const i64 n_keys = causalKeyCount(len - 1);
        kbase.resize(n_keys);
        for (i64 key = 0; key < n_keys; ++key)
            kbase[key] = q_offset + key * channel_out + attn_dim;

        val[layer_id].resize(circuit.size);
        zkgpt_gpu::gpu_qk(val[0].data(), val[0].size(), wgPrevData(layer_id), wgPrevSize(layer_id),
                          val[layer_id].data(), circuit.size, circuit.uni_gates, circuit.bin_gates, qbase,
                          kbase, row_start, row_of, HEAD, HSIZE, score_count);
        if (gkr_config.witgen_full_check) {
            zkgpt_gpu::witgen_sync_gates();
            wg_check_gates("qk-check", ref_u, circuit.uni_gates, ref_b, circuit.bin_gates);
        }
    } else {
        multi_head_matrix_QK_cpu_gates(circuit.uni_gates, circuit.bin_gates, score_count);
        calcNormalLayer(circuit, layer_id);
    }
    layer_id++;
}

void neuralNetwork::compute_e_table() {
    double St = pow(2, -9), Se = pow(2, -20);
    for (int i = 0; i < 655360; i++) {
        int t = round(exp(-St * i) / Se);
        table[i] = max(t, 1);
    }
}

// Host reference paths for softmax_layer_1, kept as the A/B target for the GPU kernels
void neuralNetwork::softmax_layer_1_cpu_values(F* out, i64 layer_id, i64 orgsize, i64 score_count, int c1,
                                               int e1, int eprime) {
    const int HEAD = headnum;
    const int HSIZE = headdim;
    out[0] = 1;
// (head,i) pairs are independent
#pragma omp parallel for collapse(2) schedule(static) num_threads(cpu_threads)
    for (int head = 0; head < HEAD; head++) {
        for (int i = 0; i < len; i++) {
            i64 sum_E_offset = orgsize + 10 + head * len + i;
            i64 pmax_offset = orgsize + 10 + HEAD * len + head * len + i;
            out[sum_E_offset - orgsize] = 0;
            ll mx = 0;
            i64 row_start = causalRowStart(i);
            i64 key_count = causalKeyCount(i);
            for (i64 key = 0; key < key_count; key++) {
                int offset = head * score_count + row_start + key;
                ll S = convert(val[layer_id - 1][offset]);
                mx = max(mx, S);
            }
            out[pmax_offset - orgsize] = Fr(mx);
            for (i64 key = 0; key < key_count; key++) {
                int offset = head * score_count + row_start + key;
                i64 dt1_off = orgsize + 10 + HEAD * 2 * len + offset;
                i64 dt2_off = orgsize + 10 + HEAD * 2 * len + HEAD * score_count + offset;
                i64 t_off = orgsize + 10 + HEAD * 2 * len + 2 * HEAD * score_count + offset;
                i64 E_off = orgsize + 10 + HEAD * 2 * len + 3 * HEAD * score_count + offset;
                out[dt1_off - orgsize] = out[pmax_offset - orgsize] - val[layer_id - 1][offset];
                ll pj_ = convert(out[dt1_off - orgsize]);
                ll tj = round(c1 * pow(2, e1 + eprime + 1) * pj_ / pow(2, eprime + 1));
                out[t_off - orgsize] = tj;
                if (tj < 0) {
                    cerr << "softmax table index is negative: tj=" << tj << " pj=" << pj_ << " head=" << head
                         << " row=" << i << " key=" << key << " score_count=" << score_count << " c1=" << c1
                         << " e1=" << e1 << " eprime=" << eprime << endl;
                }
                assert(tj >= 0);
                int exp_value = tj < 655360 ? table[tj] : 1;
                out[E_off - orgsize] = exp_value;
                out[dt2_off - orgsize] =
                    Fr(c1 * (1 << (e1 + eprime + 1)) * pj_ + (1 << eprime) - tj * (1 << (eprime + 1))) *
                    Fr(-c1 * (1 << (e1 + eprime + 1)) * pj_ + (1 << eprime) + tj * (1 << (eprime + 1)));
                out[sum_E_offset - orgsize] += exp_value;
            }
        }
    }
}

void neuralNetwork::softmax_layer_1_cpu_gates(std::vector<uniGate>& uni, std::vector<binGate>& bin,
                                              i64 layer_id, i64 orgsize, i64 score_count, int c1, int e1,
                                              int eprime) {
    const int HEAD = headnum;
    const int HSIZE = headdim;
    for (int head = 0; head < HEAD; head++) {
        for (int i = 0; i < len; i++) {
            i64 row_start = causalRowStart(i);
            i64 key_count = causalKeyCount(i);
            for (int j = 0; j < HSIZE; j++) {
                int out_ij = head * len * HSIZE + i * HSIZE + j;
                for (i64 key = 0; key < key_count; key++) {
                    i64 Vkj_offset = q_offset + channel_out * key + 2 * attn_dim + head * HSIZE + j;
                    i64 E_off = orgsize + 10 + HEAD * 2 * len + 3 * HEAD * score_count + head * score_count +
                                row_start + key;
                    bin.emplace_back(out_ij, Vkj_offset, E_off, 1, 0);
                }
            }
        }

        for (int i = 0; i < len; i++) {
            i64 pmax_offset = orgsize + 10 + HEAD * len + head * len + i;
            i64 sum_E_offset = orgsize + 10 + head * len + i;
            i64 row_start = causalRowStart(i);
            i64 key_count = causalKeyCount(i);
            for (i64 key = 0; key < key_count; key++) {
                int offset = head * score_count + row_start + key;
                i64 dt1_off = orgsize + 10 + HEAD * 2 * len + offset;
                int check_seg1_offset = 2 * HEAD * score_count + HEAD * len * HSIZE + offset;
                uni.emplace_back(check_seg1_offset, dt1_off, 0, -1);
                uni.emplace_back(check_seg1_offset, offset, layer_id - 1, -1);
                uni.emplace_back(check_seg1_offset, pmax_offset, 0, 1);
                int term1_offset = HEAD * len * HSIZE + offset;
                int term2_offset = HEAD * score_count + HEAD * len * HSIZE + offset;
                i64 t_off = orgsize + 10 + HEAD * 2 * len + 2 * HEAD * score_count + offset;
                uni.emplace_back(term1_offset, dt1_off, 0, c1 * (1 << (e1 + 1 + eprime)));
                uni.emplace_back(term1_offset, orgsize, 0, 1 << eprime);
                uni.emplace_back(term1_offset, t_off, 0, -(1 << (eprime + 1)));

                uni.emplace_back(term2_offset, dt1_off, 0, -c1 * (1 << (e1 + 1 + eprime)));
                uni.emplace_back(term2_offset, orgsize, 0, 1 << eprime);
                uni.emplace_back(term2_offset, t_off, 0, (1 << (eprime + 1)));
            }
            int sum_e_check = 3 * HEAD * score_count + HEAD * len * HSIZE + head * len + i;
            uni.emplace_back(sum_e_check, sum_E_offset, 0, -1);
            for (i64 key = 0; key < key_count; key++) {
                int offset = head * score_count + row_start + key;
                i64 E_off = orgsize + 10 + HEAD * 2 * len + 3 * HEAD * score_count + offset;
                uni.emplace_back(sum_e_check, E_off, 0, 1);
            }
        }
    }
}

void neuralNetwork::softmax_layer_1(layer& circuit, i64& layer_id, float SQ, float SK, float Sv, float Sy) {
    const int HEAD = headnum;
    const int HSIZE = headdim;
    const i64 score_count = causalScoreCount();
    i64 orgsize = val[0].size();
    val[0].resize(orgsize + 10 + HEAD * 2 * len + 4 * HEAD * score_count + HEAD * len * HSIZE +
                  len * channel_in);  // sumE,pmax,delta1,delta2,t,E,delta3,Y
    softmax_aux_start = orgsize;
    positive_check += 2 * HEAD * score_count + HEAD * len * HSIZE;
    exp_check += HEAD * score_count;
    total_relu_in_size +=
        10 + 2 * HEAD * len + 4 * HEAD * score_count + HEAD * len * HSIZE + len * channel_in;
    int output_size = 3 * HEAD * score_count + HEAD * len + HEAD * len * HSIZE;
    initLayer(circuit, output_size, layerType::SOFTMAX_1);
    circuit.need_phase2 = true;
    circuit.zero_start_id = 2 * HEAD * score_count + HEAD * len * HSIZE;
    int e1, c1;
    const float St = pow(2, -9);
    const float Se = pow(2, -16);
    pair<int, int> pm = search(SQ * SK / St);
    e1 = pm.first;
    c1 = pm.second;
    int eprime = max(-e1 - 1, 0);
    if (gkr_config.witgen_full || gkr_config.witgen_full_check) {
        // Per-key and per-row index tables
        std::vector<long long> vbase, row_start(len), key_count(len);
        std::vector<int> row_of(score_count);
        for (i64 i = 0; i < len; ++i) {
            row_start[i] = causalRowStart(i);
            key_count[i] = causalKeyCount(i);
            for (i64 k = 0; k < key_count[i]; ++k)
                row_of[row_start[i] + k] = (int)i;
        }
        const i64 n_keys = key_count[len - 1];
        vbase.resize(n_keys);
        for (i64 key = 0; key < n_keys; ++key)
            vbase[key] = q_offset + channel_out * key + 2 * attn_dim;

        std::vector<F> ref_v;
        std::vector<uniGate> ref_u;
        std::vector<binGate> ref_b;
        const size_t alloc = val[0].size() - (size_t)orgsize;
        if (gkr_config.witgen_full_check) {
            ref_v.assign(alloc, F_ZERO);
            softmax_layer_1_cpu_values(ref_v.data(), layer_id, orgsize, score_count, c1, e1, eprime);
            softmax_layer_1_cpu_gates(ref_u, ref_b, layer_id, orgsize, score_count, c1, e1, eprime);
        }
        val[layer_id].resize(circuit.size);
        zkgpt_gpu::gpu_sm1(val[0].data(), (size_t)orgsize, alloc, wgPrevData(layer_id), wgPrevSize(layer_id),
                           val[layer_id].data(), circuit.size, circuit.uni_gates, circuit.bin_gates, table,
                           vbase, row_start, key_count, row_of, HEAD, (int)len, HSIZE, score_count,
                           (int)layer_id, c1, e1, eprime);
        if (gkr_config.witgen_full_check) {
            for (size_t k = 0; k < alloc; ++k)
                if (ref_v[k] != val[0][orgsize + k]) {
                    fprintf(stderr, "[sm1-check] VALUE MISMATCH idx=%zu\n", k);
                    abort();
                }
            zkgpt_gpu::witgen_sync_gates();
            wg_check_gates("sm1-check", ref_u, circuit.uni_gates, ref_b, circuit.bin_gates);
        }
    } else {
        for (i64 i = orgsize; i < (i64)val[0].size(); i++)
            val[0][i].clear();
        val[0][orgsize] = 1;
        softmax_layer_1_cpu_values(&val[0][orgsize], layer_id, orgsize, score_count, c1, e1, eprime);
        softmax_layer_1_cpu_gates(circuit.uni_gates, circuit.bin_gates, layer_id, orgsize, score_count, c1,
                                  e1, eprime);
        calcNormalLayer(circuit, layer_id);
    }
    for (int i = circuit.zero_start_id; i < val[layer_id].size(); i++) {
        assert(val[layer_id][i].isZero());
    }
    for (int i = HEAD * len * HSIZE; i < circuit.zero_start_id; i++) {
        assert(!val[layer_id][i].isNegative());
    }
    layer_id++;
}

// Host reference path for softmax_layer_2
void neuralNetwork::softmax_layer_2_cpu(std::vector<uniGate>& uni, std::vector<binGate>& bin, F* v0,
                                        i64 layer_id, i64 orgsize, i64 score_count, int c1, int e1,
                                        int eprime) {
    const int HEAD = headnum;
    const int HSIZE = headdim;
    ll f = 0, f2 = 0;
    for (int head = 0; head < HEAD; head++) {
        for (int i = 0; i < len; i++) {
            i64 sum_E_offset = orgsize + 10 + head * len + i;
            ll sumE = convert(val[0][sum_E_offset]);
            assert(sumE != 0);
            for (int j = 0; j < HSIZE; j++) {
                int out_ij = head * len * HSIZE + i * HSIZE + j;
                int term1_off = out_ij;
                int term2_off = out_ij + HEAD * len * HSIZE;
                i64 s_ij = orgsize + 10 + HEAD * 2 * len + 4 * HEAD * score_count + HEAD * len * HSIZE +
                           i * channel_in + head * HSIZE + j;
                i64 d3_off = orgsize + 10 + HEAD * 2 * len + 4 * HEAD * score_count + head * len * HSIZE +
                             i * HSIZE + j;
                ll Qij = convert(val[layer_id - 1][out_ij]);
                ll S = (ll)round(Qij * c1 * pow(2, e1) / sumE);
                v0[s_ij] = S;
                v0[d3_off] = Fr(sumE * (1 << (eprime - 1)) + Qij * c1 * (1 << (eprime + e1)) -
                                S * sumE * (1 << eprime)) *
                             Fr((sumE * (1 << (eprime - 1)) - Qij * c1 * (1 << (eprime + e1)) +
                                 S * sumE * (1 << eprime)));
                f2 = min(f2, S);
                uni.emplace_back(term1_off, out_ij, layer_id - 1, c1 * (1ll << (e1 + eprime)));
                uni.emplace_back(term1_off, sum_E_offset, 0, 1ll << (eprime - 1));

                bin.emplace_back(term1_off, s_ij, sum_E_offset, -(1 << eprime), 0);
                uni.emplace_back(term2_off, out_ij, layer_id - 1, -c1 * (1ll << (e1 + eprime)));
                uni.emplace_back(term2_off, sum_E_offset, 0, 1ll << (eprime - 1));
                bin.emplace_back(term2_off, s_ij, sum_E_offset, (1 << eprime), 0);
            }
        }
    }
    for (int head = 0; head < HEAD; head++) {
        for (int i = 0; i < len; i++) {
            i64 row_start = causalRowStart(i);
            i64 key_count = causalKeyCount(i);
            for (i64 key = 0; key < key_count; key++) {
                int offset = head * score_count + row_start + key;
                i64 dt2_off = orgsize + 10 + HEAD * 2 * len + HEAD * score_count + offset;
                int term1_offset = HEAD * len * HSIZE + offset;
                int term2_offset = HEAD * score_count + HEAD * len * HSIZE + offset;
                int now_off = 2 * HEAD * len * HSIZE + offset;
                bin.emplace_back(now_off, term1_offset, term2_offset, 1, 1);
                uni.emplace_back(now_off, dt2_off, 0, -1);
            }
        }
    }
}

void neuralNetwork::softmax_layer_2(layer& circuit, i64& layer_id, float SQ, float SK, float Sv, float Sy) {
    const int HEAD = headnum;
    const int HSIZE = headdim;
    const i64 score_count = causalScoreCount();
    i64 orgsize = softmax_aux_start;

    int output_size = 2 * HEAD * len * HSIZE + HEAD * score_count;
    initLayer(circuit, output_size, layerType::SOFTMAX_2);
    circuit.need_phase2 = true;
    circuit.zero_start_id = 2 * HEAD * len * HSIZE;
    int e1, c1;
    pair<int, int> pm = search(Sv / Sy);
    e1 = pm.first;
    c1 = pm.second;
    int eprime = max(-e1, 1);
    const size_t sm2_lo = (size_t)(orgsize + 10 + HEAD * 2 * len + 4 * HEAD * score_count);
    if (gkr_config.witgen_full || gkr_config.witgen_full_check) {
        std::vector<F> ref_v;
        std::vector<uniGate> ref_u;
        std::vector<binGate> ref_b;
        if (gkr_config.witgen_full_check) {
            ref_v.assign(val[0].begin() + sm2_lo, val[0].end());
            softmax_layer_2_cpu(ref_u, ref_b, &ref_v[0] - sm2_lo, layer_id, orgsize, score_count, c1, e1,
                                eprime);
        }
        val[layer_id].resize(circuit.size);
        zkgpt_gpu::gpu_sm2(val[0].data(), val[0].size(), sm2_lo, val[0].size(), wgPrevData(layer_id),
                           wgPrevSize(layer_id), val[layer_id].data(), circuit.size, circuit.uni_gates,
                           circuit.bin_gates, orgsize, HEAD, (int)len, HSIZE, channel_in, score_count,
                           (int)layer_id, c1, e1, eprime);
        if (gkr_config.witgen_full_check) {
            for (size_t k = sm2_lo; k < val[0].size(); ++k)
                if (ref_v[k - sm2_lo] != val[0][k]) {
                    fprintf(stderr, "[sm2-check] VALUE MISMATCH idx=%zu\n", k);
                    abort();
                }
            zkgpt_gpu::witgen_sync_gates();
            wg_check_gates("sm2-check", ref_u, circuit.uni_gates, ref_b, circuit.bin_gates);
        }
    } else {
        softmax_layer_2_cpu(circuit.uni_gates, circuit.bin_gates, val[0].data(), layer_id, orgsize,
                            score_count, c1, e1, eprime);
        // s_ij and d3_off land inside softmax_layer_1's advice block
        zkgpt_gpu::witgen_val0_push(val[0].data(), sm2_lo, val[0].size());
    }
    calcNormalLayer(circuit, layer_id);
    for (int i = circuit.zero_start_id; i < val[layer_id].size(); i++) {
        assert(val[layer_id][i].isZero());
    }
    for (int i = 0; i < circuit.zero_start_id; i++) {
        assert(!val[layer_id][i].isNegative());
    }
    layer_id++;
}
// Host reference path for softmax_layer_3's gate emission
void neuralNetwork::softmax_layer_3_cpu_gates(std::vector<uniGate>& uni, std::vector<binGate>& bin,
                                              i64 orgsize, i64 score_count) {
    const int HEAD = headnum;
    const int HSIZE = headdim;
    for (int head = 0; head < HEAD; head++) {
        for (int i = 0; i < len; i++) {
            for (int j = 0; j < HSIZE; j++) {
                int now = head * len * HSIZE + i * HSIZE + j;
                int term1_off = now;
                int term2_off = now + HEAD * len * HSIZE;
                i64 d3_off = orgsize + 10 + HEAD * 2 * len + 4 * HEAD * score_count + head * len * HSIZE +
                             i * HSIZE + j;

                bin.emplace_back(now, term1_off, term2_off, 1, 1);
                uni.emplace_back(now, d3_off, 0, -1);
            }
        }
    }
}

void neuralNetwork::softmax_layer_3(layer& circuit, i64& layer_id, float SQ, float SK, float Sv, float Sy) {
    const int HEAD = headnum;
    const int HSIZE = headdim;
    const i64 score_count = causalScoreCount();
    i64 orgsize = softmax_aux_start;

    int output_size = HEAD * len * HSIZE;
    initLayer(circuit, output_size, layerType::SOFTMAX_3);
    circuit.need_phase2 = true;
    circuit.zero_start_id = 0;
    // sumE != 0 is a property of softmax_layer_1's advice
    for (int head = 0; head < HEAD; head++)
        for (int i = 0; i < len; i++) {
            i64 sum_E_offset = orgsize + 10 + head * len + i;
            assert(convert(val[0][sum_E_offset]) != 0);
        }

    if (gkr_config.witgen_full || gkr_config.witgen_full_check) {
        std::vector<uniGate> ref_u;
        std::vector<binGate> ref_b;
        if (gkr_config.witgen_full_check)
            softmax_layer_3_cpu_gates(ref_u, ref_b, orgsize, score_count);
        val[layer_id].resize(circuit.size);
        zkgpt_gpu::gpu_sm3(val[0].data(), val[0].size(), wgPrevData(layer_id), wgPrevSize(layer_id),
                           val[layer_id].data(), circuit.size, circuit.uni_gates, circuit.bin_gates, orgsize,
                           HEAD, (int)len, HSIZE, score_count);
        if (gkr_config.witgen_full_check) {
            zkgpt_gpu::witgen_sync_gates();
            wg_check_gates("sm3-check", ref_u, circuit.uni_gates, ref_b, circuit.bin_gates);
        }
    } else {
        softmax_layer_3_cpu_gates(circuit.uni_gates, circuit.bin_gates, orgsize, score_count);
        calcNormalLayer(circuit, layer_id);
    }
    for (int i = circuit.zero_start_id; i < val[layer_id].size(); i++) {
        assert(val[layer_id][i].isZero());
    }
    layer_id++;
    q_offset = orgsize + 10 + HEAD * 2 * len + 4 * HEAD * score_count + HEAD * len * HSIZE;
}
// Host reference path, kept as the A/B target for the GPU kernel
void neuralNetwork::fullyConnLayer_cpu(F* out, i64 first_fc_id, i64 x_offset, int x_layer) {
// Rows are independent
#pragma omp parallel for schedule(static) num_threads(cpu_threads)
    for (i64 i = 0; i < len; i++) {
        for (i64 co = 0; co < channel_out; ++co) {
            i64 g = matIdx(i, co, channel_out);
            out[g] = 0;
            // circuit.uni_gates.emplace_back(g, first_bias_id + co, 0, 1)
            for (i64 ci = 0; ci < channel_in; ++ci) {
                i64 u = x_offset + matIdx(i, ci, channel_in);
                i64 v =
                    first_fc_id + matIdx(co, ci, channel_in);  // the matrix is distributed as (i,ci)*(co,ci)
                out[g] += val[x_layer][u] * val[0][v];
            }
        }
    }
}

void neuralNetwork::fullyConnLayer(layer& circuit, i64& layer_id, i64 first_fc_id, i64 x_offset,
                                   int x_layer) {
    i64 size = channel_out * len;
    initLayer(circuit, size, layerType::FCONN);
    circuit.need_phase2 = true;
    val[layer_id].resize(circuit.size);

    zkgpt_witgen::Scope _fc(zkgpt_witgen::acc().fc_s);

    // Opt-IN (the kernel is new)
    const bool witgen_gpu = gkr_config.witgen_gpu;
    const bool witgen_check = gkr_config.witgen_check;

    if (!witgen_gpu && !witgen_check) {
        fullyConnLayer_cpu(val[layer_id].data(), first_fc_id, x_offset, x_layer);
        layer_id++;
        return;
    }

    // C = A * B^T with A = val[x_layer]+x_offset
    const char* fuse_env = getenv("GKR_FC_ROUND_FUSE");
    const bool fuse = !(fuse_env && fuse_env[0] == '0');

    releaseFcResident();  // stale guard: nothing should be pending here

    if (!fuse) {
        zkgpt_gpu::gpu_fully_conn_layer(&val[x_layer][x_offset], &val[0][first_fc_id], val[layer_id].data(),
                                        (int)len, (int)channel_in, (int)channel_out);
    } else {
        gpu_fe* d_C = zkgpt_gpu::gpu_fully_conn_layer_resident(&val[x_layer][x_offset], &val[0][first_fc_id],
                                                               (int)len, (int)channel_in, (int)channel_out);

        {
            zkgpt_gpu::GkrXferTimer _x(&zkgpt_gpu::gkr_xfer_d2h_s_ref());
            HIP_CHECK(zkgpt_gpu::gkr_d2h(reinterpret_cast<gpu_fe*>(val[layer_id].data()), d_C,
                                         (size_t)circuit.size * sizeof(gpu_fe)));
        }

        fc_resident_dev = d_C;
        fc_resident_elems = circuit.size;
        fc_resident_layer = layer_id;
    }

    if (witgen_check) {
        // Field equality is exact and addition is reordering-invariant
        std::vector<F> ref(circuit.size);
        fullyConnLayer_cpu(ref.data(), first_fc_id, x_offset, x_layer);
        for (i64 g = 0; g < (i64)circuit.size; ++g) {
            if (ref[g] != val[layer_id][g]) {
                fprintf(stderr,
                        "[witgen-check] MISMATCH layer=%lld idx=%lld "
                        "(len=%lld ci=%lld co=%lld)\n",
                        (long long)layer_id, (long long)g, (long long)len, (long long)channel_in,
                        (long long)channel_out);
                abort();
            }
        }
    }

    layer_id++;
}

void neuralNetwork::refreshFCParam(const fconKernel& fc) {
    channel_in = fc.channel_in;
    channel_out = fc.channel_out;
}

i64 neuralNetwork::getFFTLen() const {
    return 1L << getFFTBitLen();
}

i8 neuralNetwork::getFFTBitLen() const {
    return 0;
}

void neuralNetwork::calcSizeAfterPool(const poolKernel& p) {}

void neuralNetwork::calcInputLayer(layer& circuit) {
    val[0].resize(circuit.size);

    assert(val[0].size() == total_in_size);
    if (getenv("GKR_DBG_LASSO_SIZES"))
        fprintf(stderr,
                "[input] circuit.size=%lld bit_length=%d total_in_size=%lld "
                "2^bl=%llu\n",
                (long long)circuit.size, (int)circuit.bit_length, (long long)total_in_size,
                1ULL << circuit.bit_length);
    auto val_0 = val[0].begin();

    i64 feature_dim = input_feature_dim ? input_feature_dim : attn_dim;
    if (feature_dim <= 0)
        feature_dim = attn_dim ? attn_dim : 1;
    i64 padded_dim = input_slot_size ? input_slot_size : (1LL << ceilPow2BitLength(feature_dim));

    if (custom_input && custom_input_data) {
        auto& src = *custom_input_data;
        if (custom_input_offset < 0 || custom_input_offset + len * feature_dim > (i64)src.size()) {
            throw std::runtime_error("custom input source is shorter than len * feature_dim");
        }
        for (i64 i = 0; i < len; ++i) {
            for (i64 j = 0; j < feature_dim; ++j) {
                i64 idx = custom_input_offset + i * feature_dim + j;
                val[0][i * padded_dim + j] = idx < (i64)src.size() ? src[idx] : F_ZERO;
            }
            for (i64 j = feature_dim; j < padded_dim; ++j)
                val[0][i * padded_dim + j] = F_ZERO;
        }
    } else {
        double num, mx = -10000, mn = 10000;
        vector<double> input_dat;
        for (i64 i = 0; i < len; i++) {
            for (i64 j = 0; j < feature_dim; j++) {
                in >> num;
                input_dat.push_back(num);
                mx = max(mx, num);
                mn = min(mn, num);
            }
        }
        pair<int, int> pm = search(0.01);
        input_e = pm.first;
        input_c = pm.second;

        double sc = input_c * pow(2, input_e);
        int k = 0;
        // GKR_TEST_INPUT=<seed>
        const char* test_in = getenv("GKR_TEST_INPUT");
        unsigned test_seed = test_in ? (unsigned)atoi(test_in) : 0;
        for (i64 i = 0; i < len; i++) {
            for (i64 j = 0; j < feature_dim; j++) {
                ll s = input_dat[k++] / sc;
                if (test_in)
                    s = (ll)(rand_r(&test_seed) % 17) - 8;
                val[0][i * padded_dim + j] = F(s);
            }
            for (i64 j = feature_dim; j < padded_dim; j++)
                val[0][i * padded_dim + j] = 0;
        }
    }

    val_0 = val[0].begin() + len * padded_dim;
    for (; val_0 < val[0].begin() + circuit.size; ++val_0)
        val_0->clear();
}

void neuralNetwork::readBias(i64 first_bias_id) {
    auto val_0 = val[0].begin() + first_bias_id;

    double num, mx = -10000, mn = 10000;
    vector<double> input_dat;
    for (i64 co = 0; co < channel_out; ++co) {
        in >> num;
        input_dat.push_back(num);
        mx = max(mx, num);
        mn = min(mn, num);
    }

    for (double i : input_dat)
        *val_0++ = F((i64)(i * exp2(w_bit + x_bit)));
}

void neuralNetwork::readFconWeight(i64 first_fc_id, int real_r, int real_c, int id) {
    double num, mx = -10000, mn = 10000;
    auto val_0 = val[0].begin() + first_fc_id;
    mat_values[id] = new int[static_cast<size_t>(channel_out) * static_cast<size_t>(channel_in)];
    for (i64 co = 0; co < channel_out; ++co)
        for (i64 ci = 0; ci < channel_in; ++ci) {
            if (co < real_c && ci < real_r) {
                mat_values[id][co * channel_in + ci] = rand() % 1024;
                val_0[co * channel_in + ci] = mat_values[id][co * channel_in + ci];
            } else {
                mat_values[id][co * channel_in + ci] = 0;
                val_0[co * channel_in + ci] = 0;
            }
        }
}

void neuralNetwork::prepareDecmpBit(i64 layer_id, i64 idx, i64 dcmp_id, i64 bit_shift) {
    auto data = abs(val[layer_id].at(idx).getInt64());
    val[0].at(dcmp_id) = (data >> bit_shift) & 1;
}

void neuralNetwork::prepareFieldBit(const F& data, i64 dcmp_id, i64 bit_shift) {
    auto tmp = abs(data.getInt64());
    val[0].at(dcmp_id) = (tmp >> bit_shift) & 1;
}

void neuralNetwork::prepareSignBit(i64 layer_id, i64 idx, i64 dcmp_id) {
    val[0].at(dcmp_id) = val[layer_id].at(idx).isNegative() ? F_ONE : F_ZERO;
}

void neuralNetwork::prepareMax(i64 layer_id, i64 idx, i64 max_id) {
    auto data = val[layer_id].at(idx).isNegative() ? F_ZERO : val[layer_id].at(idx);
    if (data > val[0].at(max_id))
        val[0].at(max_id) = data;
}

// Host reference path for the gate replay, kept as the A/B target for the GPU evaluator
void neuralNetwork::calcNormalLayer_cpu(const layer& circuit, i64 layer_id, F* out) {
#pragma omp parallel for schedule(static) num_threads(cpu_threads)
    for (u32 i = 0; i < circuit.size; ++i)
        out[i].clear();

    // Different gates can share an output index g
    const u32 nuni = (u32)circuit.uni_gates.size();
    const u32 nbin = (u32)circuit.bin_gates.size();

    std::vector<u32> offset(circuit.size + 1, 0);
    for (auto& gate : circuit.uni_gates)
        offset[gate.g + 1]++;
    for (auto& gate : circuit.bin_gates)
        offset[gate.g + 1]++;
    for (u32 g = 0; g < circuit.size; ++g)
        offset[g + 1] += offset[g];

    std::vector<u32> order(nuni + nbin);
    std::vector<u32> cursor(offset.begin(), offset.end() - 1);
    for (u32 k = 0; k < nuni; ++k)
        order[cursor[circuit.uni_gates[k].g]++] = k;  // uni: index < nuni
    for (u32 k = 0; k < nbin; ++k)
        order[cursor[circuit.bin_gates[k].g]++] = nuni + k;  // bin: index >= nuni

// `x * gate.sc` implicitly converts the int64 to Fr each time
#pragma omp parallel for schedule(static) num_threads(cpu_threads)
    for (u32 g = 0; g < circuit.size; ++g) {
        F acc = out[g];
        for (u32 p = offset[g]; p < offset[g + 1]; ++p) {
            u32 k = order[p];
            if (k < nuni) {
                const auto& gate = circuit.uni_gates[k];
                // Explicit check instead of .at()
                if ((u64)gate.u >= val[gate.lu].size()) {
                    fprintf(stderr,
                            "[calcNormalLayer] uni gate out of range: layer_id=%lld "
                            "g=%lld u=%lld lu=%d val[lu].size=%zu\n",
                            (long long)layer_id, (long long)gate.g, (long long)gate.u, gate.lu,
                            val[gate.lu].size());
                    abort();
                }
                acc = acc + val[gate.lu][gate.u] * gate.sc;
            } else {
                const auto& gate = circuit.bin_gates[k - nuni];
                int bin_lu = gate.getLayerIdU(static_cast<int>(layer_id));
                int bin_lv = gate.getLayerIdV(static_cast<int>(layer_id));
                if ((u64)gate.u >= val[bin_lu].size() || (u64)gate.v >= val[bin_lv].size()) {
                    fprintf(stderr,
                            "[calcNormalLayer] bin gate out of range: layer_id=%lld "
                            "g=%lld u=%lld v=%lld l=%d lu=%d lv=%d val[lu].size=%zu val[lv].size=%zu\n",
                            (long long)layer_id, (long long)gate.g, (long long)gate.u, (long long)gate.v,
                            (int)gate.l, bin_lu, bin_lv, val[bin_lu].size(), val[bin_lv].size());
                    abort();
                }
                acc = acc + val[bin_lu][gate.u] * val[bin_lv][gate.v] * gate.sc;
            }
        }
        out[g] = acc;
    }
}

void neuralNetwork::calcNormalLayer(const layer& circuit, i64 layer_id, bool output) {
    zkgpt_witgen::Scope _cn(zkgpt_witgen::acc().calcnorm_s);
    val[layer_id].resize(circuit.size);

    const bool gates_gpu = gkr_config.gates_gpu;
    const bool gates_check = gkr_config.gates_check;

    if (gates_gpu || gates_check) {
        // layer_id-1 == 0 hands the evaluator val[0] twice
        const F* prev =
            (layer_id > 0 && !val[layer_id - 1].empty()) ? val[layer_id - 1].data() : val[0].data();
        const size_t nprev =
            (layer_id > 0 && !val[layer_id - 1].empty()) ? val[layer_id - 1].size() : val[0].size();
        zkgpt_gpu::gpu_calc_normal_layer(circuit.uni_gates, circuit.bin_gates, static_cast<int>(layer_id),
                                         val[0].data(), val[0].size(), prev, nprev, val[layer_id].data(),
                                         circuit.size);
        if (gates_check) {
            // Field equality is exact and addition is reordering-invariant
            std::vector<F> ref(circuit.size);
            calcNormalLayer_cpu(circuit, layer_id, ref.data());
            for (u32 g = 0; g < circuit.size; ++g) {
                if (ref[g] != val[layer_id][g]) {
                    fprintf(stderr,
                            "[gates-check] MISMATCH layer=%lld idx=%u "
                            "(size=%u uni=%zu bin=%zu)\n",
                            (long long)layer_id, g, circuit.size, circuit.uni_gates.size(),
                            circuit.bin_gates.size());
                    abort();
                }
            }
        }
        return;
    }

    calcNormalLayer_cpu(circuit, layer_id, val[layer_id].data());
}

void neuralNetwork::checkNormalLayer(const layer& circuit, i64 layer_id, const vector<vector<F> >& val) {
    vector<F> valp;

    valp.resize(val[layer_id].size());

    for (auto& x : valp)
        x.clear();
    for (auto& gate : circuit.uni_gates) {
        assert(gate.g >= 0 && gate.g < valp.size());
        assert(gate.u >= 0 && gate.u < val[gate.lu].size());
        valp.at(gate.g) += val[gate.lu].at(gate.u) * gate.sc;
    }
    for (auto& gate : circuit.bin_gates) {
        int bin_lu = gate.getLayerIdU(static_cast<int>(layer_id));
        int bin_lv = gate.getLayerIdV(static_cast<int>(layer_id));
        assert(gate.g >= 0 && gate.g < valp.size());
        valp.at(gate.g) += val[bin_lu].at(gate.u) * val[bin_lv][gate.v] * gate.sc;
    }
    for (int i = 0; i < circuit.size; i++)
        assert(valp[i] == val[layer_id][i]);
}

void neuralNetwork::calcDotProdLayer(const layer& circuit, i64 layer_id) {
    val[layer_id].resize(circuit.size);
    for (int i = 0; i < circuit.size; ++i)
        val[layer_id][i].clear();

    char fft_bit = circuit.fft_bit_length;
    u32 fft_len = 1 << fft_bit;
    u8 l = layer_id - 1;
    for (auto& gate : circuit.bin_gates)
        for (int s = 0; s < fft_len; ++s)
            val[layer_id][gate.g << fft_bit | s] =
                val[layer_id][gate.g << fft_bit | s] +
                val[l][gate.u << fft_bit | s] * val[l][gate.v << fft_bit | s];
}

int neuralNetwork::getNextBit(int layer_id) {
    F mx = F_ZERO, mn = F_ZERO;
    for (const auto& x : val[layer_id]) {
        if (!x.isNegative())
            mx = max(mx, x);
        else
            mn = max(mn, -x);
    }
    i64 x = (mx + mn).getInt64();
    double real_scale = x / exp2(x_bit + w_bit);
    int res = (int)log2(((1 << (Q - 1)) - 1) / real_scale);
    return res;
}

void neuralNetwork::printLayerValues(prover& pr) {
    for (i64 i = 0; i < SIZE; ++i) {
        for (i64 j = 0; j < std::min((i64)200, pr.C.circuit[i].size); ++j)
            if (!pr.val[i][j].isZero())
                cerr << pr.val[i][j] << ' ';
        cerr << endl;
        for (i64 j = pr.C.circuit[i].zero_start_id; j < pr.C.circuit[i].size; ++j)
            if (pr.val[i].at(j) != F_ZERO) {
                exit(EXIT_FAILURE);
            }
    }
}

void neuralNetwork::printInfer(prover& pr) {
    // output the inference result with the size of (pic_parallel x n_class)
    if (out.is_open()) {
        int n_class = full_conn.back().channel_out;
        for (int p = 0; p < pic_parallel; ++p) {
            int k = -1;
            F v;
            for (int c = 0; c < n_class; ++c) {
                auto tmp = val[SIZE - 1].at(matIdx(p, c, n_class));
                if (!tmp.isNegative() && (k == -1 || v < tmp)) {
                    k = c;
                    v = tmp;
                }
            }
            out << k << endl;
        }
    }
    out.close();
}
