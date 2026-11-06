#pragma once

#include <vector>
#include <unordered_map>
#include <utility>

#include <unordered_set>
#include <iostream>
#include "global_var.hpp"
#include "hyrax.hpp"
using std::cerr;
using std::endl;
using std::vector;

struct uniGate {
    // i64 because for a gate reading val[0]
    i64 g, u;
    // Real layer id, not a flag: softmax/gelu checkers store layer_id-1 here
    int lu;
    ll sc;
    uniGate(i64 _g, i64 _u, int _lu, ll _sc) : g(_g), u(_u), lu(_lu), sc(_sc) {}
};

struct binGate {
    // See uniGate's g/u comment. u/v can be absolute val[0] offsets (getLayerIdU/V == 0)
    i64 g, u, v;
    u8 l;
    ll sc;
    binGate(i64 _g, i64 _u, i64 _v, ll _sc, u8 _l) : g(_g), u(_u), v(_v), sc(_sc), l(_l) {}
    [[nodiscard]] int getLayerIdU(int layer_id) const { return !l ? 0 : layer_id - 1; }
    [[nodiscard]] int getLayerIdV(int layer_id) const { return !(l & 1) ? 0 : layer_id - 1; }
};

enum class layerType {
    INPUT,
    FFT,
    IFFT,
    ADD_BIAS,
    RELU,
    Sqr,
    OPT_AVG_POOL,
    MAX_POOL,
    AVG_POOL,
    DOT_PROD,
    PADDING,
    FCONN,
    LAYER_NORM_1,
    LAYER_NORM_2,
    LAYER_NORM_3,
    GELU_1,
    GELU_2,
    GELU_3,
    MHA_QK,
    SOFTMAX_1,
    SOFTMAX_2,
    SOFTMAX_3,
    PLACE_HOLDER
};

class layer {
public:
    layerType ty;
    std::vector<pair<int, int> > uni_interval, bin_interval;
    i64 size{}, size_u[2]{}, size_v[2]{};
    i8 bit_length_u[2]{}, bit_length_v[2]{}, bit_length{};
    i8 max_bl_u{}, max_bl_v{};

    bool need_phase2;

    // bit decomp related
    i64 zero_start_id;

    std::vector<uniGate> uni_gates;
    std::vector<binGate> bin_gates;

    vector<i64> ori_id_u, ori_id_v;
    i8 fft_bit_length;

    layer() {
        bit_length_u[0] = bit_length_v[0] = -1;
        size_u[0] = size_v[0] = 0;
        bit_length_u[1] = bit_length_v[1] = -1;
        size_u[1] = size_v[1] = 0;
        need_phase2 = false;
        zero_start_id = 0;
        fft_bit_length = -1;
    }

    void updateSize() {
        max_bl_u = std::max(bit_length_u[0], bit_length_u[1]);
        max_bl_v = 0;
        if (!need_phase2)
            return;

        max_bl_v = std::max(bit_length_v[0], bit_length_v[1]);
    }
};

class layeredCircuit {
public:
    vector<layer> circuit;
    int size;
    void init(u8 q_bit_size, int _layer_sz);
    void initSubset();
};
