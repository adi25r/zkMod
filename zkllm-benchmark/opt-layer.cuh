#ifndef OPT_LAYER_CUH
#define OPT_LAYER_CUH

#include "llama-layer.cuh"
#include "weight-stream.cuh"
#include <map>
#include <memory>

// One pre-LN OPT decoder layer as committed by opt-commit.py. Like the LLaMa statement, attention is
// single-head over embed_dim and out_proj is not proved.
// A linear layer's weight matrix: streamed from host memory (default), or resident on the GPU
// as field elements for the whole layer (ZKLLM_WSTREAM=0). Exactly one pointer is set.
struct LinearWeight {
    std::unique_ptr<Weight> resident;
    std::unique_ptr<StreamedWeight> streamed;
};

struct OPTLayerWeights {
    Weight ln1_g, ln1_b;
    LinearWeight q;
    Weight q_b;
    LinearWeight k;
    Weight k_b;
    LinearWeight v;
    Weight v_b, ln2_g, ln2_b;
    LinearWeight fc1;
    Weight fc1_b;
    LinearWeight fc2;
    Weight fc2_b;
    OPTLayerWeights(const string& workdir, uint layer_idx, const LayerDims& d);
};

FrTensor run_opt_layer(const FrTensor& x_in, const LayerDims& d, const OPTLayerWeights& w, tLookupRangeMapping& relu, zkSoftmax& softmax,
    bool prove, LayerTimers& t, BoundaryClaim* input_claim_out);

// Op-level partitioning. One OPT layer is OPT_OPS_PER_LAYER tensor operations, each with its own
// proof (matmul + weight opening, bias add, rescaling, Hadamard, softmax, ReLU); steps without a
// proof (row mean, rms_inv, residual adds) are folded into an adjacent operation. The state at a
// cut is the set of named tensors still needed later (opt_live_at). Op g of the model is op
// g % OPT_OPS_PER_LAYER of layer g / OPT_OPS_PER_LAYER.
constexpr uint OPT_OPS_PER_LAYER = 31;

// A tensor of the op state. It is on the GPU while an op reads it; between uses it is kept in host
// memory and its GPU copy freed (activation streaming, ZKLLM_ASTREAM=0 keeps everything on the GPU).
// Op-state tensors are never modified in place, so the host copy stays valid after a reload.
bool activation_streaming_enabled();

// Page-locked host buffer from a per-process pool (buffers are reused by size), so activation
// transfers run at full PCIe speed without paying cudaHostAlloc on every op.
class PinnedBuffer {
public:
    PinnedBuffer() = default;
    explicit PinnedBuffer(size_t count);
    ~PinnedBuffer();
    PinnedBuffer(PinnedBuffer&& o) noexcept : p(o.p), n(o.n) { o.p = nullptr; o.n = 0; }
    PinnedBuffer& operator=(PinnedBuffer&& o) noexcept;
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    Fr_t* data() const { return p; }
    size_t size() const { return n; }
    bool empty() const { return !p; }
    void write(size_t first, const FrTensor& t);   // device -> host
private:
    Fr_t* p = nullptr;
    size_t n = 0;
};

class Activation {
public:
    explicit Activation(FrTensor&& t);
    explicit Activation(PinnedBuffer&& host_data);   // host resident
    FrTensor& gpu();
    void offload();
    // Elements [first, first + count) as a GPU tensor, without loading the whole tensor; positions
    // at or past size() read as zero.
    FrTensor slice(size_t first, uint count) const;
    // Host storage filled chunk by chunk (e.g. while streaming a file in).
    PinnedBuffer& host_buffer() { return host; }
    uint size() const { return n; }
private:
    uint n;
    std::unique_ptr<FrTensor> dev;
    PinnedBuffer host;
};
using OpState = std::map<std::string, std::shared_ptr<Activation>>;

// MLE at u (u[0] for bit 0) of a flat tensor of `total` elements given in aligned chunks of
// `chunk` = 2^c elements (chunk(i) returns elements [i*chunk, ...)). Each chunk is folded by the
// first c challenges, then the chunk values by the rest: the same folds as Fr_me, so the same value.
template <typename Chunk>
inline Fr_t chunked_me(size_t total, uint chunk, const vector<Fr_t>& u, Chunk chunk_at)
{
    const uint c = ceilLog2(chunk), nc = static_cast<uint>((total + chunk - 1) / chunk);
    if (u.size() <= c || nc == 1) return chunk_at(0)(u);
    const vector<Fr_t> lo(u.begin(), u.begin() + c), hi(u.begin() + c, u.end());
    vector<Fr_t> vals(nc);
    for (uint i = 0; i < nc; ++ i) vals[i] = chunk_at(i)(lo);
    return FrTensor(nc, vals.data())(hi);
}


std::string opt_op_name(uint op_in_layer);
// Tensors live before op `cut_in_layer` (0 = layer input {x}).
const std::vector<std::string>& opt_live_at(uint cut_in_layer);

class OptOpRunner {
public:
    OptOpRunner(const string& workdir, const LayerDims& d, const string& relu_table, bool prove, LayerTimers& t);
    ~OptOpRunner();
    // Runs model op `global_op` on st; afterwards st holds exactly opt_live_at(next cut).
    void run(uint global_op, OpState& st);
    double load_s = 0;   // weights and lookup tables, loaded on first use
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

#endif // OPT_LAYER_CUH
