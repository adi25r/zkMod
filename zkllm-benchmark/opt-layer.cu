#include "witness-store.cuh"
#include "opt-layer.cuh"
#include "squash.cuh"
#include <algorithm>
#include <cstdlib>

namespace {

class ScopedTimer {
    Timer tm;
    double& acc;
public:
    explicit ScopedTimer(double& a) : acc(a) { tm.start(); }
    ~ScopedTimer() { tm.stop(); acc += tm.getTotalTime(); }
};

template <typename F>
auto timed(double& acc, F&& f) -> decltype(f())
{
    ScopedTimer st(acc);
    return f();
}

Weight load_weight(const string& workdir, uint layer_idx, const string& name, uint in_dim, uint out_dim)
{
    const string prefix = "layer-" + std::to_string(layer_idx);
    return create_weight(
        workdir + "/" + name + "-pp.bin",
        workdir + "/" + prefix + "-" + name + "-int.bin",
        workdir + "/" + prefix + "-" + name + "-commitment.bin",
        in_dim, out_dim);
}

LinearWeight load_linear(const string& workdir, uint layer_idx, const string& name, uint in_dim, uint out_dim)
{
    const string prefix = "layer-" + std::to_string(layer_idx);
    const string pp = workdir + "/" + name + "-pp.bin";
    const string w = workdir + "/" + prefix + "-" + name + "-int.bin";
    const string com = workdir + "/" + prefix + "-" + name + "-commitment.bin";
    LinearWeight out;
    if (weight_streaming_enabled()) out.streamed = std::make_unique<StreamedWeight>(load_streamed_weight(pp, w, com, in_dim, out_dim));
    else out.resident = std::make_unique<Weight>(create_weight(pp, w, com, in_dim, out_dim));
    return out;
}

KERNEL void row_mean_kernel(const Fr_t* X, Fr_t* out, uint rows, uint cols)
{
    const uint row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row >= rows) return;
    long acc = 0;
    for (uint j = 0; j < cols; ++j) acc += scalar_to_int(X[row * cols + j]);
    out[row] = long_to_scalar(acc / static_cast<long>(cols));
}

KERNEL void broadcast_rows_kernel(const Fr_t* v, Fr_t* out, uint rows, uint cols)
{
    const uint i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < rows * cols) out[i] = v[i / cols];
}

KERNEL void broadcast_cols_kernel(const Fr_t* v, Fr_t* out, uint rows, uint cols, long scale)
{
    const uint i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < rows * cols) out[i] = long_to_scalar(static_cast<long>(scalar_to_int(v[i % cols])) * scale);
}

FrTensor broadcast_rows(const FrTensor& v, uint rows, uint cols)
{
    FrTensor out(rows * cols);
    broadcast_rows_kernel<<<(out.size + FrNumThread - 1) / FrNumThread, FrNumThread>>>(v.gpu_data, out.gpu_data, rows, cols);
    cudaDeviceSynchronize();
    return out;
}

FrTensor broadcast_cols(const FrTensor& v, uint rows, uint cols, long scale)
{
    FrTensor out(rows * cols);
    broadcast_cols_kernel<<<(out.size + FrNumThread - 1) / FrNumThread, FrNumThread>>>(v.gpu_data, out.gpu_data, rows, cols, scale);
    cudaDeviceSynchronize();
    return out;
}

// The ReLU lookup of in -> out (multiplicities m), or its share of the partition's combined lookup.
void prove_relu(tLookupRangeMapping& relu, const FrTensor& in, const FrTensor& out, const FrTensor& m)
{
    if (squash::active()) { squash::add_mapping(relu, in); return; }
    auto temp_rand = random_vec(3);
    auto u = random_vec(ceilLog2(in.size)), v = random_vec(ceilLog2(in.size));
    vector<Polynomial> proof;
    relu.prove(in, out, m, temp_rand[0], temp_rand[1], temp_rand[2], u, v, proof);
}

// Binds a committed 1-D parameter to the tensor the prover used.
void open_vector(const Weight& w)
{
    auto padded = w.weight.pad({w.out_dim});
    auto u = random_vec(ceilLog2(padded.size));
    w.generator.open_check(padded, w.com, u, padded(u), "open_vector: opening != evaluation");
}

// Verifier check for Z = Y + B at a random point.
void check_add(const FrTensor& Z, const FrTensor& Y, const FrTensor& B)
{
    auto u = random_vec(ceilLog2(Z.size));
    if (Z(u) != Y(u) + B(u)) throw std::runtime_error("check_add: Z != Y + B");
}

// y = LN(x) = g * (x - mean) / sqrt(var + eps) + b, reusing the proven RMSNorm path on the centered input.
FrTensor opt_layernorm(const FrTensor& X, const Weight& g, const Weight& b, const LayerDims& d, bool prove, LayerTimers& t, BoundaryClaim* input_claim_out)
{
    const uint rows = d.seq_len, cols = d.embed_dim;
    FrTensor mu(rows);
    FrTensor MU = wstore::witgen(t.compute_s, [&] {
        row_mean_kernel<<<(rows + FrNumThread - 1) / FrNumThread, FrNumThread>>>(X.gpu_data, mu.gpu_data, rows, cols);
        cudaDeviceSynchronize();
        return broadcast_rows(mu, rows, cols);
    });
    FrTensor Xc = wstore::witgen(t.compute_s, [&] { return X - MU; });

    BoundaryClaim centered;
    FrTensor N = llama_rmsnorm(Xc, g, d, prove, t, input_claim_out ? &centered : nullptr);
    FrTensor B = wstore::witgen(t.compute_s, [&] { return broadcast_cols(b.weight, rows, cols, 1); });
    FrTensor Y = wstore::witgen(t.compute_s, [&] { return N + B; });

    if (prove) {
        timed(t.prove_s, [&] {
            check_add(Y, N, B);
            open_vector(b);
            if (input_claim_out) {
                // The RMSNorm sumcheck left a claim on Xc at v; lift it to X = Xc + MU.
                input_claim_out->v = centered.v;
                input_claim_out->value = centered.value + MU(centered.v);
                input_claim_out->valid = true;
            }
        });
    }
    return Y;
}

// Y = rescale(X W + b), with the bias lifted to the product's 2^32 scale.
FrTensor opt_linear(const FrTensor& X, const LinearWeight& w, const Weight& b, uint in_dim, uint out_dim, uint seq_len, uint rescale_bits,
    bool prove, LayerTimers& t)
{
    std::unique_ptr<zkFC> layer;
    if (w.resident) layer = std::make_unique<zkFC>(in_dim, out_dim, w.resident->weight);
    Rescaling rs(1u << rescale_bits);
    FrTensor Y0 = wstore::witgen(t.compute_s, [&] { return layer ? (*layer)(X) : streamed_matmul(X, *w.streamed); });
    FrTensor B = wstore::witgen(t.compute_s, [&] { return broadcast_cols(b.weight, seq_len, out_dim, 1L << 16); });
    FrTensor Y = wstore::witgen(t.compute_s, [&] { return Y0 + B; });
    FrTensor Y_ = wstore::rescale(t.compute_s, rs, Y);
    if (prove) {
        timed(t.prove_s, [&] {
            rs.prove(Y, Y_);
            check_add(Y, Y0, B);
            open_vector(b);
            if (layer) verifyWeightClaim(*w.resident, layer->prove(X, Y0)[0]);
            else streamed_prove_linear(X, Y0, *w.streamed);
        });
    }
    return Y_;
}

FrTensor opt_attention(const FrTensor& X, const OPTLayerWeights& w, const LayerDims& d, zkSoftmax& softmax, bool prove, LayerTimers& t)
{
    const uint seq_len = d.seq_len, dim = d.embed_dim;
    FrTensor Q_ = opt_linear(X, w.q, w.q_b, dim, dim, seq_len, 16, prove, t);
    FrTensor K_ = opt_linear(X, w.k, w.k_b, dim, dim, seq_len, 16, prove, t);
    FrTensor V_ = opt_linear(X, w.v, w.v_b, dim, dim, seq_len, 16, prove, t);

    Rescaling rs1(1 << 20), rs2(1 << 20);
    FrTensor shift(seq_len), S_shifted(seq_len * seq_len);
    vector<FrTensor> S_segments, Y_segments, m_segments;

    FrTensor S = wstore::witgen(t.compute_s, [&] { return FrTensor::matmul(Q_, K_.transpose(seq_len, dim), seq_len, dim, seq_len); });
    FrTensor Y = wstore::witgen(t.compute_s, [&] { return softmax.compute(S, shift, S_shifted, S_segments, Y_segments, m_segments); });
    wstore::side(shift); wstore::side(S_shifted);   // softmax out-parameters, read by softmax.prove
    wstore::side(S_segments); wstore::side(Y_segments); wstore::side(m_segments);
    FrTensor out = wstore::witgen(t.compute_s, [&] { return FrTensor::matmul(Y, V_, seq_len, seq_len, dim); });
    FrTensor out_ = wstore::rescale(t.compute_s, rs2, out);
    FrTensor out__ = wstore::rescale(t.compute_s, rs1, out_);

    if (prove) {
        timed(t.prove_s, [&] {
            rs1.prove(out_, out__);
            rs2.prove(out, out_);
            auto temp_rand = random_vec(3);
            vector<Polynomial> proof;
            auto u1 = random_vec(ceilLog2(seq_len));
            auto u2 = random_vec(ceilLog2(dim));
            auto ud = random_vec(ceilLog2(seq_len));
            auto claim = out.multi_dim_me({u1, u2}, {seq_len, dim});
            zkip(claim, Y.partial_me(u1, seq_len, seq_len), V_.partial_me(u2, dim, 1), ud, proof);

            softmax.prove(Y, S, shift, S_shifted, S_segments, Y_segments, m_segments,
                random_vec(ceilLog2(Y.size)), random_vec(ceilLog2(Y.size)), temp_rand[0], temp_rand[1], temp_rand[2], proof);

            auto u1_ = random_vec(ceilLog2(seq_len));
            auto u2_ = random_vec(ceilLog2(seq_len));
            auto ud_ = random_vec(ceilLog2(dim));
            auto claim_ = S.multi_dim_me({u1_, u2_}, {seq_len, seq_len});
            zkip(claim_, Q_.partial_me(u1_, seq_len, dim), K_.partial_me(u2_, seq_len, dim), ud_, proof);
        });
    }
    return out__;
}

// fc2(relu(fc1(x))): fc1 output rescaled to the 2^12-scale table domain, ReLU by lookup, fc2 rescaled back to 2^16.
FrTensor opt_ffn(const FrTensor& X, const OPTLayerWeights& w, const LayerDims& d, tLookupRangeMapping& relu, bool prove, LayerTimers& t)
{
    const uint seq_len = d.seq_len, embed_dim = d.embed_dim, hidden_dim = d.hidden_dim;
    FrTensor h_ = opt_linear(X, w.fc1, w.fc1_b, embed_dim, hidden_dim, seq_len, 20, prove, t);
    auto p = wstore::witgen(t.compute_s, [&] { return relu(h_); });
    auto &relu_out = p.first, &relu_m = p.second;
    FrTensor y_ = opt_linear(relu_out, w.fc2, w.fc2_b, hidden_dim, embed_dim, seq_len, 16, prove, t);
    if (prove) {
        timed(t.prove_s, [&] { prove_relu(relu, h_, relu_out, relu_m); });
    }
    return y_;
}

} // namespace

OPTLayerWeights::OPTLayerWeights(const string& workdir, uint layer_idx, const LayerDims& d):
    ln1_g(load_weight(workdir, layer_idx, "self_attn_layer_norm.weight", 1, d.embed_dim)),
    ln1_b(load_weight(workdir, layer_idx, "self_attn_layer_norm.bias", 1, d.embed_dim)),
    q(load_linear(workdir, layer_idx, "self_attn.q_proj.weight", d.embed_dim, d.embed_dim)),
    q_b(load_weight(workdir, layer_idx, "self_attn.q_proj.bias", 1, d.embed_dim)),
    k(load_linear(workdir, layer_idx, "self_attn.k_proj.weight", d.embed_dim, d.embed_dim)),
    k_b(load_weight(workdir, layer_idx, "self_attn.k_proj.bias", 1, d.embed_dim)),
    v(load_linear(workdir, layer_idx, "self_attn.v_proj.weight", d.embed_dim, d.embed_dim)),
    v_b(load_weight(workdir, layer_idx, "self_attn.v_proj.bias", 1, d.embed_dim)),
    ln2_g(load_weight(workdir, layer_idx, "final_layer_norm.weight", 1, d.embed_dim)),
    ln2_b(load_weight(workdir, layer_idx, "final_layer_norm.bias", 1, d.embed_dim)),
    fc1(load_linear(workdir, layer_idx, "fc1.weight", d.embed_dim, d.hidden_dim)),
    fc1_b(load_weight(workdir, layer_idx, "fc1.bias", 1, d.hidden_dim)),
    fc2(load_linear(workdir, layer_idx, "fc2.weight", d.hidden_dim, d.embed_dim)),
    fc2_b(load_weight(workdir, layer_idx, "fc2.bias", 1, d.embed_dim))
{}

FrTensor run_opt_layer(const FrTensor& x_in, const LayerDims& d, const OPTLayerWeights& w, tLookupRangeMapping& relu, zkSoftmax& softmax,
    bool prove, LayerTimers& t, BoundaryClaim* input_claim_out)
{
    FrTensor n1 = opt_layernorm(x_in, w.ln1_g, w.ln1_b, d, prove, t, input_claim_out);
    FrTensor a = opt_attention(n1, w, d, softmax, prove, t);
    FrTensor h = wstore::witgen(t.compute_s, [&] { return x_in + a; });
    FrTensor n2 = opt_layernorm(h, w.ln2_g, w.ln2_b, d, prove, t, nullptr);
    FrTensor f = opt_ffn(n2, w, d, relu, prove, t);
    return wstore::witgen(t.compute_s, [&] { return h + f; });
}

// ---------------------------------------------------------------------------------------------
// Op-level partitioning (see opt-layer.cuh).

std::string opt_op_name(uint j)
{
    static const char* names[OPT_OPS_PER_LAYER] = {
        "ln1_scale", "ln1_rescale1", "ln1_hadamard", "ln1_rescale2", "ln1_bias",
        "q_matmul", "q_bias", "q_rescale", "k_matmul", "k_bias", "k_rescale", "v_matmul", "v_bias", "v_rescale",
        "qk_matmul", "softmax", "av_matmul", "attn_rescale2", "attn_rescale1_residual",
        "ln2_scale", "ln2_rescale1", "ln2_hadamard", "ln2_rescale2", "ln2_bias",
        "fc1_matmul", "fc1_bias", "fc1_rescale", "relu", "fc2_matmul", "fc2_bias", "fc2_rescale_residual"};
    return names[j];
}

const std::vector<std::string>& opt_live_at(uint c)
{
    static const std::vector<std::vector<std::string>> live = {
        {"x"}, {"Xc", "gir", "x"}, {"Xc", "gir_", "x"}, {"x", "yln"}, {"x", "yln_"}, {"n", "x"},
        {"n", "q0", "x"}, {"n", "q1", "x"}, {"n", "q", "x"},
        {"k0", "n", "q", "x"}, {"k1", "n", "q", "x"}, {"k", "n", "q", "x"},
        {"k", "q", "v0", "x"}, {"k", "q", "v1", "x"}, {"k", "q", "v", "x"},
        {"s", "v", "x"}, {"v", "x", "ysm"}, {"o", "x"}, {"o_", "x"},
        {"h"}, {"Xc", "gir", "h"}, {"Xc", "gir_", "h"}, {"h", "yln"}, {"h", "yln_"}, {"h", "n"},
        {"f0", "h"}, {"f1", "h"}, {"fp", "h"}, {"h", "r"}, {"g0", "h"}, {"g1", "h"}};
    return live.at(c);
}

bool activation_streaming_enabled()
{
    const char* s = std::getenv("ZKLLM_ASTREAM");
    return !(s && std::string(s) == "0");
}

namespace {
struct PinnedPool {
    std::map<size_t, std::vector<void*>> free_;
    size_t cached = 0;
    static constexpr size_t MAX_CACHED = size_t(2) << 30;
    void* get(size_t bytes)
    {
        auto& v = free_[bytes];
        if (!v.empty()) { void* p = v.back(); v.pop_back(); cached -= bytes; return p; }
        void* p = nullptr;
        if (cudaHostAlloc(&p, bytes, cudaHostAllocDefault) != cudaSuccess) throw std::runtime_error("cudaHostAlloc failed");
        return p;
    }
    void put(void* p, size_t bytes)
    {
        if (cached + bytes > MAX_CACHED) { cudaFreeHost(p); return; }
        free_[bytes].push_back(p);
        cached += bytes;
    }
};
PinnedPool& pinned_pool() { static PinnedPool* p = new PinnedPool; return *p; }   // never destroyed: freed at exit
} // namespace

PinnedBuffer::PinnedBuffer(size_t count) : p(static_cast<Fr_t*>(pinned_pool().get(sizeof(Fr_t) * count))), n(count) {}
PinnedBuffer::~PinnedBuffer() { if (p) pinned_pool().put(p, sizeof(Fr_t) * n); }
PinnedBuffer& PinnedBuffer::operator=(PinnedBuffer&& o) noexcept
{
    if (this != &o) {
        if (p) pinned_pool().put(p, sizeof(Fr_t) * n);
        p = o.p; n = o.n; o.p = nullptr; o.n = 0;
    }
    return *this;
}
void PinnedBuffer::write(size_t first, const FrTensor& t)
{
    cudaMemcpy(p + first, t.gpu_data, sizeof(Fr_t) * t.size, cudaMemcpyDeviceToHost);
}

Activation::Activation(FrTensor&& t) : n(t.size), dev(std::make_unique<FrTensor>(std::move(t))) {}

FrTensor& Activation::gpu()
{
    if (!dev) dev = std::make_unique<FrTensor>(n, host.data());
    return *dev;
}

Activation::Activation(PinnedBuffer&& host_data) : n(static_cast<uint>(host_data.size())), host(std::move(host_data)) {}

FrTensor Activation::slice(size_t first, uint count) const
{
    FrTensor out(count);
    const size_t valid = first >= n ? 0 : std::min<size_t>(count, n - first);
    if (valid < count) cudaMemset(out.gpu_data + valid, 0, sizeof(Fr_t) * (count - valid));
    if (!valid) return out;
    if (dev) cudaMemcpy(out.gpu_data, dev->gpu_data + first, sizeof(Fr_t) * valid, cudaMemcpyDeviceToDevice);
    else cudaMemcpy(out.gpu_data, host.data() + first, sizeof(Fr_t) * valid, cudaMemcpyHostToDevice);
    return out;
}

void Activation::offload()
{
    if (!dev || !activation_streaming_enabled()) return;
    if (host.empty()) {
        host = PinnedBuffer(n);
        host.write(0, *dev);
    }
    dev.reset();
}

// Rows per chunk (a power of two) when a rows x cols activation is streamed through an op
// (ZKLLM_ACT_CHUNK_MB, default 256), or 0 when it is not: streaming is off, the tensor fits one
// chunk, or rows is not a power of two.
static uint act_chunk_rows(uint rows, uint cols)
{
    if (!activation_streaming_enabled() || rows != 1u << ceilLog2(rows)) return 0;
    const char* s = std::getenv("ZKLLM_ACT_CHUNK_MB");
    const long mb = s ? std::atol(s) : 256;
    const size_t bytes = static_cast<size_t>(mb > 0 ? mb : 256) << 20;
    uint r = rows;
    while (r > 1 && sizeof(Fr_t) * static_cast<size_t>(r) * cols > bytes) r >>= 1;
    return r == rows ? 0 : r;
}

// Op-state tensors each op reads (by op index within the layer); the others are offloaded first.
static const std::vector<std::vector<std::string>> OP_INPUTS = {
    {"x"}, {"gir"}, {"Xc", "gir_"}, {"yln"}, {"yln_"},
    {"n"}, {"q0"}, {"q1"}, {"n"}, {"k0"}, {"k1"}, {"n"}, {"v0"}, {"v1"},
    {"k", "q"}, {"s"}, {"v", "ysm"}, {"o"}, {"o_", "x"},
    {"h"}, {"gir"}, {"Xc", "gir_"}, {"yln"}, {"yln_"},
    {"n"}, {"f0"}, {"f1"}, {"fp"}, {"r"}, {"g0"}, {"g1", "h"}};

struct OptOpRunner::Impl {
    string workdir, relu_table;
    LayerDims d;
    bool prove;
    LayerTimers& t;
    double& load_s;
    // Weights of the current layer, loaded on first use.
    int layer = -1;
    std::map<string, std::unique_ptr<Weight>> small;
    std::map<string, std::unique_ptr<LinearWeight>> linear;
    std::unique_ptr<FrTensor> relu_values;
    std::unique_ptr<tLookupRangeMapping> relu;
    std::unique_ptr<zkSoftmax> softmax;

    Impl(const string& w, const LayerDims& d_, const string& rt, bool p, LayerTimers& t_, double& ls)
        : workdir(w), relu_table(rt), d(d_), prove(p), t(t_), load_s(ls) {}

    void set_layer(int l) { if (l != layer) { layer = l; small.clear(); linear.clear(); } }
    const Weight& vec(const string& name, uint dim)
    {
        auto& p = small[name];
        if (!p) p = timed(load_s, [&] { return std::make_unique<Weight>(load_weight(workdir, layer, name, 1, dim)); });
        return *p;
    }
    const LinearWeight& mat(const string& name, uint in, uint out)
    {
        auto& p = linear[name];
        if (!p) p = timed(load_s, [&] { return std::make_unique<LinearWeight>(load_linear(workdir, layer, name, in, out)); });
        return *p;
    }
    tLookupRangeMapping& relu_lookup()
    {
        if (!relu) timed(load_s, [&] {
            relu_values = std::make_unique<FrTensor>(FrTensor::from_int_bin(relu_table));
            relu = std::make_unique<tLookupRangeMapping>(-(1 << 21), 1 << 22, *relu_values);
        });
        return *relu;
    }
    zkSoftmax& softmax_lookup()
    {
        // Same parameters as partition.cu's layer path.
        if (!softmax) timed(load_s, [&] {
            softmax = std::make_unique<zkSoftmax>(vector<uint>{1 << 8, 1 << 20, 1 << 20, 1 << 15}, 1, 0, 1UL << 32,
                vector<double>{1 << 18, 1 << 22, 1.0}, d.seq_len, d.seq_len, d.embed_dim, 1);
        });
        return *softmax;
    }

    static FrTensor& get(OpState& st, const string& k)
    {
        auto it = st.find(k);
        if (it == st.end()) throw std::runtime_error("op state has no tensor " + k);
        return it->second->gpu();
    }
    static void put(OpState& st, const string& k, FrTensor&& v) { st[k] = std::make_shared<Activation>(std::move(v)); }

    // LayerNorm, split like opt_layernorm + llama_rmsnorm.
    void ln_scale(OpState& st, const string& resid, const string& gname)
    {
        const uint rows = d.seq_len, cols = d.embed_dim;
        const FrTensor& X = get(st, resid);
        const Weight& g = vec(gname, cols);
        FrTensor mu(rows);
        FrTensor MU = timed(t.compute_s, [&] {
            row_mean_kernel<<<(rows + FrNumThread - 1) / FrNumThread, FrNumThread>>>(X.gpu_data, mu.gpu_data, rows, cols);
            cudaDeviceSynchronize();
            return broadcast_rows(mu, rows, cols);
        });
        put(st, "Xc", timed(t.compute_s, [&] { return X - MU; }));
        zkFC gfc(1, cols, g.weight);
        FrTensor rms_inv = timed(t.compute_s, [&] { return llama_rms_inv(get(st, "Xc"), rows, cols, d.eps); });
        put(st, "gir", timed(t.compute_s, [&] { return gfc(rms_inv); }));
        if (prove) timed(t.prove_s, [&] { verifyWeightClaim(g, gfc.prove(rms_inv, get(st, "gir"))[0]); });
    }
    void rescale(OpState& st, const string& in, const string& out, uint bits)
    {
        Rescaling rs(1u << bits);
        put(st, out, timed(t.compute_s, [&] { return rs(get(st, in)); }));
        if (prove) timed(t.prove_s, [&] { rs.prove(get(st, in), get(st, out)); });
        st.erase(in);
    }
    void ln_hadamard(OpState& st)
    {
        put(st, "yln", timed(t.compute_s, [&] { return get(st, "gir_") * get(st, "Xc"); }));
        if (prove) timed(t.prove_s, [&] {
            auto u = random_vec(ceilLog2(get(st, "yln").size));
            auto v = random_vec(ceilLog2(get(st, "yln").size));
            hadamard_product_sumcheck(get(st, "gir_"), get(st, "Xc"), u, v);
        });
        st.erase("gir_");
        st.erase("Xc");
    }
    void bias(OpState& st, const string& in, const string& out, const string& bname, uint dim, long scale)
    {
        const Weight& b = vec(bname, dim);
        FrTensor B = timed(t.compute_s, [&] { return broadcast_cols(b.weight, d.seq_len, dim, scale); });
        put(st, out, timed(t.compute_s, [&] { return get(st, in) + B; }));
        if (prove) timed(t.prove_s, [&] { check_add(get(st, out), get(st, in), B); open_vector(b); });
        st.erase(in);
    }
    void matmul(OpState& st, const string& in, const string& out, const string& wname, uint in_dim, uint out_dim, bool erase_in)
    {
        const LinearWeight& w = mat(wname, in_dim, out_dim);
        const FrTensor& X = get(st, in);
        std::unique_ptr<zkFC> fc;
        if (w.resident) fc = std::make_unique<zkFC>(in_dim, out_dim, w.resident->weight);
        put(st, out, timed(t.compute_s, [&] { return fc ? (*fc)(X) : streamed_matmul(X, *w.streamed); }));
        if (prove) timed(t.prove_s, [&] {
            if (fc) verifyWeightClaim(*w.resident, fc->prove(X, get(st, out))[0]);
            else streamed_prove_linear(X, get(st, out), *w.streamed);
        });
        if (erase_in) st.erase(in);
    }
    void residual(OpState& st, const string& a, const string& b, const string& out)
    {
        FrTensor sum = timed(t.compute_s, [&] { return get(st, a) + get(st, b); });
        st.erase(a);
        st.erase(b);
        put(st, out, std::move(sum));
    }

    // ---- seq x hidden activations streamed through the FFN ops in row chunks ----
    using HostOut = PinnedBuffer;
    static Activation& act(OpState& st, const string& k)
    {
        auto it = st.find(k);
        if (it == st.end()) throw std::runtime_error("op state has no tensor " + k);
        return *it->second;
    }

    // Y = X W for X (seq x in_dim) and a streamed W, R rows of X at a time; the proof is the same as
    // the unchunked one (claim on Y, X folded over the batch and X's claim are evaluated by chunks).
    void matmul_rows(OpState& st, const string& in, const string& out, const string& wname, uint in_dim, uint out_dim, uint R)
    {
        const LinearWeight& w = mat(wname, in_dim, out_dim);
        const uint seq = d.seq_len, nc = seq / R, rb = ceilLog2(R);
        Activation& X = act(st, in);
        HostOut Y(static_cast<size_t>(seq) * out_dim);
        // The proof's challenges (drawn up front, as zkFC::prove does) and its reductions of X and Y,
        // accumulated while each chunk is on the GPU: Y.multi_dim_me({u_batch, u_output}) folds every row
        // by u_output and then the rows; X.partial_me(u_batch) folds each chunk's rows by the low batch
        // challenges and then the chunks by the rest. Same folds as the unchunked proof.
        vector<Fr_t> u_batch, u_input, u_output;
        std::unique_ptr<FrTensor> rowvals, parts;
        if (prove) {
            u_batch = random_vec(ceilLog2(seq));
            u_input = random_vec(ceilLog2(in_dim));
            u_output = random_vec(ceilLog2(out_dim));
            rowvals = std::make_unique<FrTensor>(seq);
            parts = std::make_unique<FrTensor>(nc * in_dim);
        }
        const vector<Fr_t> ub_lo(u_batch.begin(), u_batch.begin() + (prove ? rb : 0)), ub_hi(u_batch.begin() + (prove ? rb : 0), u_batch.end());
        for (uint c = 0; c < nc; ++ c) {
            FrTensor Xc = X.slice(static_cast<size_t>(c) * R * in_dim, R * in_dim);
            FrTensor Yc = timed(t.compute_s, [&] { return streamed_matmul(Xc, *w.streamed); });
            if (prove) timed(t.prove_s, [&] {
                FrTensor rc = Yc.partial_me(u_output, out_dim, 1);
                cudaMemcpy(rowvals->gpu_data + static_cast<size_t>(c) * R, rc.gpu_data, sizeof(Fr_t) * R, cudaMemcpyDeviceToDevice);
                FrTensor pc = rb ? Xc.partial_me(ub_lo, R, in_dim) : FrTensor(Xc);
                cudaMemcpy(parts->gpu_data + static_cast<size_t>(c) * in_dim, pc.gpu_data, sizeof(Fr_t) * in_dim, cudaMemcpyDeviceToDevice);
            });
            Y.write(static_cast<size_t>(c) * R * out_dim, Yc);
        }
        st[out] = std::make_shared<Activation>(std::move(Y));
        if (prove) timed(t.prove_s, [&] {
            Fr_t claim = (*rowvals)(u_batch);
            FrTensor X_reduced = ub_hi.empty() ? FrTensor(*parts) : parts->partial_me(ub_hi, nc, in_dim);
            Fr_t claim_X = X_reduced(u_input);
            streamed_prove_linear_reduced(claim, X_reduced, claim_X, *w.streamed, u_input, u_output);
        });
        if (in != "n") st.erase(in);
    }

    void bias_rows(OpState& st, const string& in, const string& out, const string& bname, uint dim, long scale, uint R)
    {
        const Weight& b = vec(bname, dim);
        const uint seq = d.seq_len, nc = seq / R;
        const size_t n = static_cast<size_t>(seq) * dim, cn = static_cast<size_t>(R) * dim;
        FrTensor Bc = timed(t.compute_s, [&] { return broadcast_cols(b.weight, R, dim, scale); });   // the same for every chunk
        Activation& Y = act(st, in);
        HostOut Z(n);
        // check_add(Z, Y, B). For a power-of-two width the chunks are aligned pieces of the flat
        // tensor and the evaluations are folded like Fr_me (the same check); otherwise each chunk is
        // checked on its own.
        const bool aligned = dim == 1u << ceilLog2(dim);
        const uint lc = ceilLog2(cn);
        vector<Fr_t> u, lo, hi, zv(nc), yv(nc), bv(nc);
        if (prove && aligned) {
            u = random_vec(ceilLog2(n));
            lo.assign(u.begin(), u.begin() + std::min<size_t>(lc, u.size()));
            hi.assign(u.begin() + lo.size(), u.end());
        }
        for (uint c = 0; c < nc; ++ c) {
            FrTensor Yc = Y.slice(c * cn, cn);
            FrTensor Zc = timed(t.compute_s, [&] { return Yc + Bc; });
            if (prove) timed(t.prove_s, [&] {
                if (aligned) { zv[c] = Zc(lo); yv[c] = Yc(lo); bv[c] = Bc(lo); }
                else check_add(Zc, Yc, Bc);
            });
            Z.write(c * cn, Zc);
        }
        st[out] = std::make_shared<Activation>(std::move(Z));
        if (prove) timed(t.prove_s, [&] {
            if (aligned) {
                auto fold = [&](vector<Fr_t>& v) { return hi.empty() ? v[0] : FrTensor(nc, v.data())(hi); };
                if (fold(zv) != fold(yv) + fold(bv)) throw std::runtime_error("check_add: Z != Y + B");
            }
            open_vector(b);
        });
        st.erase(in);
    }

    // Elementwise lookups (rescaling, ReLU) proved chunk by chunk: one lookup proof per R-row chunk.
    void rescale_rows(OpState& st, const string& in, const string& out, uint bits, uint cols, uint R)
    {
        Rescaling rs(1u << bits);
        const uint nc = d.seq_len / R;
        const size_t cn = static_cast<size_t>(R) * cols;
        Activation& X = act(st, in);
        HostOut Y(static_cast<size_t>(d.seq_len) * cols);
        for (uint c = 0; c < nc; ++ c) {
            FrTensor Xc = X.slice(c * cn, cn);
            FrTensor Yc = timed(t.compute_s, [&] { return rs(Xc); });
            if (prove) timed(t.prove_s, [&] { rs.prove(Xc, Yc); });
            Y.write(c * cn, Yc);
        }
        st[out] = std::make_shared<Activation>(std::move(Y));
        st.erase(in);
    }
    void relu_rows(OpState& st, const string& in, const string& out, uint cols, uint R)
    {
        auto& relu = relu_lookup();
        const uint nc = d.seq_len / R;
        const size_t cn = static_cast<size_t>(R) * cols;
        Activation& X = act(st, in);
        HostOut Y(static_cast<size_t>(d.seq_len) * cols);
        for (uint c = 0; c < nc; ++ c) {
            FrTensor Xc = X.slice(c * cn, cn);
            auto p = timed(t.compute_s, [&] { return relu(Xc); });
            if (prove) timed(t.prove_s, [&] { prove_relu(relu, Xc, p.first, p.second); });
            Y.write(c * cn, p.first);
        }
        st[out] = std::make_shared<Activation>(std::move(Y));
        st.erase(in);
    }

    void run(uint g, OpState& st)
    {
        set_layer(static_cast<int>(g / OPT_OPS_PER_LAYER));
        const uint j = g % OPT_OPS_PER_LAYER;
        for (auto& kv : st)
            if (std::find(OP_INPUTS[j].begin(), OP_INPUTS[j].end(), kv.first) == OP_INPUTS[j].end()) kv.second->offload();
        const uint seq = d.seq_len, E = d.embed_dim, H = d.hidden_dim;
        const uint RH = act_chunk_rows(seq, H);   // rows per chunk for the seq x hidden activations, 0 = unchunked
        switch (j) {
        case 0: ln_scale(st, "x", "self_attn_layer_norm.weight"); break;
        case 19: ln_scale(st, "h", "final_layer_norm.weight"); break;
        case 1: case 20: rescale(st, "gir", "gir_", 16); break;
        case 2: case 21: ln_hadamard(st); break;
        case 3: case 22: rescale(st, "yln", "yln_", 16); break;
        case 4: bias(st, "yln_", "n", "self_attn_layer_norm.bias", E, 1); break;
        case 23: bias(st, "yln_", "n", "final_layer_norm.bias", E, 1); break;
        case 5: matmul(st, "n", "q0", "self_attn.q_proj.weight", E, E, false); break;
        case 6: bias(st, "q0", "q1", "self_attn.q_proj.bias", E, 1L << 16); break;
        case 7: rescale(st, "q1", "q", 16); break;
        case 8: matmul(st, "n", "k0", "self_attn.k_proj.weight", E, E, false); break;
        case 9: bias(st, "k0", "k1", "self_attn.k_proj.bias", E, 1L << 16); break;
        case 10: rescale(st, "k1", "k", 16); break;
        case 11: matmul(st, "n", "v0", "self_attn.v_proj.weight", E, E, true); break;
        case 12: bias(st, "v0", "v1", "self_attn.v_proj.bias", E, 1L << 16); break;
        case 13: rescale(st, "v1", "v", 16); break;
        case 14: {
            FrTensor& Q = get(st, "q");
            FrTensor& K = get(st, "k");
            put(st, "s", timed(t.compute_s, [&] { return FrTensor::matmul(Q, K.transpose(seq, E), seq, E, seq); }));
            if (prove) timed(t.prove_s, [&] {
                vector<Polynomial> proof;
                auto u1 = random_vec(ceilLog2(seq)), u2 = random_vec(ceilLog2(seq)), ud = random_vec(ceilLog2(E));
                auto claim = get(st, "s").multi_dim_me({u1, u2}, {seq, seq});
                zkip(claim, Q.partial_me(u1, seq, E), K.partial_me(u2, seq, E), ud, proof);
            });
            st.erase("q");
            st.erase("k");
            break;
        }
        case 15: {
            zkSoftmax& sm = softmax_lookup();
            FrTensor& S = get(st, "s");
            FrTensor shift(seq), S_shifted(seq * seq);
            vector<FrTensor> S_segments, Y_segments, m_segments;
            put(st, "ysm", timed(t.compute_s, [&] { return sm.compute(S, shift, S_shifted, S_segments, Y_segments, m_segments); }));
            if (prove) timed(t.prove_s, [&] {
                auto temp_rand = random_vec(3);
                vector<Polynomial> proof;
                const FrTensor& Y = get(st, "ysm");
                sm.prove(Y, S, shift, S_shifted, S_segments, Y_segments, m_segments,
                    random_vec(ceilLog2(Y.size)), random_vec(ceilLog2(Y.size)), temp_rand[0], temp_rand[1], temp_rand[2], proof);
            });
            st.erase("s");
            break;
        }
        case 16: {
            FrTensor& Y = get(st, "ysm");
            FrTensor& V = get(st, "v");
            put(st, "o", timed(t.compute_s, [&] { return FrTensor::matmul(Y, V, seq, seq, E); }));
            if (prove) timed(t.prove_s, [&] {
                vector<Polynomial> proof;
                auto u1 = random_vec(ceilLog2(seq)), u2 = random_vec(ceilLog2(E)), ud = random_vec(ceilLog2(seq));
                auto claim = get(st, "o").multi_dim_me({u1, u2}, {seq, E});
                zkip(claim, Y.partial_me(u1, seq, seq), V.partial_me(u2, E, 1), ud, proof);
            });
            st.erase("ysm");
            st.erase("v");
            break;
        }
        case 17: rescale(st, "o", "o_", 20); break;
        case 18: rescale(st, "o_", "a", 20); residual(st, "x", "a", "h"); break;
        case 24: if (RH && mat("fc1.weight", E, H).streamed) matmul_rows(st, "n", "f0", "fc1.weight", E, H, RH), st.erase("n");
                 else matmul(st, "n", "f0", "fc1.weight", E, H, true); break;
        case 25: if (RH) bias_rows(st, "f0", "f1", "fc1.bias", H, 1L << 16, RH); else bias(st, "f0", "f1", "fc1.bias", H, 1L << 16); break;
        case 26: if (RH) rescale_rows(st, "f1", "fp", 20, H, RH); else rescale(st, "f1", "fp", 20); break;
        case 27: if (RH) { relu_rows(st, "fp", "r", H, RH); break; } else {
            auto& relu = relu_lookup();
            FrTensor& h_ = get(st, "fp");
            auto p = timed(t.compute_s, [&] { return relu(h_); });
            if (prove) timed(t.prove_s, [&] { prove_relu(relu, h_, p.first, p.second); });
            put(st, "r", std::move(p.first));
            st.erase("fp");
            break;
        }
        case 28: if (RH && mat("fc2.weight", H, E).streamed) matmul_rows(st, "r", "g0", "fc2.weight", H, E, RH);
                 else matmul(st, "r", "g0", "fc2.weight", H, E, true); break;
        case 29: bias(st, "g0", "g1", "fc2.bias", E, 1L << 16); break;
        case 30: rescale(st, "g1", "y", 16); residual(st, "h", "y", "x"); break;
        default: throw std::runtime_error("bad op index");
        }
        const auto& want = opt_live_at((j + 1) % OPT_OPS_PER_LAYER);
        std::vector<std::string> have;
        for (auto& kv : st) have.push_back(kv.first);
        if (have != want) throw std::runtime_error("op " + opt_op_name(j) + ": state does not match the live set of the next cut");
    }
};

OptOpRunner::OptOpRunner(const string& workdir, const LayerDims& d, const string& relu_table, bool prove, LayerTimers& t)
    : impl(std::make_unique<Impl>(workdir, d, relu_table, prove, t, load_s)) {}
OptOpRunner::~OptOpRunner() = default;
void OptOpRunner::run(uint global_op, OpState& st) { impl->run(global_op, st); }
