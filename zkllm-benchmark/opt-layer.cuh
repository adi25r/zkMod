#ifndef OPT_LAYER_CUH
#define OPT_LAYER_CUH

#include "llama-layer.cuh"
#include "weight-stream.cuh"
#include <map>
#include <memory>

// One pre-LN OPT decoder layer as committed by opt-commit.py
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

FrTensor run_opt_layer(const FrTensor& x_in, const LayerDims& d, const OPTLayerWeights& w,
                       tLookupRangeMapping& relu, zkSoftmax& softmax, bool prove, LayerTimers& t,
                       BoundaryClaim* input_claim_out);

// Op-level partitioning
constexpr uint OPT_OPS_PER_LAYER = 31;

// A tensor of the op state
bool activation_streaming_enabled();

// Page-locked host buffer from a per-process pool
class PinnedBuffer {
public:
    PinnedBuffer() = default;
    explicit PinnedBuffer(size_t count);
    ~PinnedBuffer();
    PinnedBuffer(PinnedBuffer&& o) noexcept : p(o.p), n(o.n) {
        o.p = nullptr;
        o.n = 0;
    }
    PinnedBuffer& operator=(PinnedBuffer&& o) noexcept;
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    Fr_t* data() const { return p; }
    size_t size() const { return n; }
    bool empty() const { return !p; }
    void write(size_t first, const FrTensor& t);  // device -> host
private:
    Fr_t* p = nullptr;
    size_t n = 0;
};

class Activation {
public:
    explicit Activation(FrTensor&& t);
    explicit Activation(PinnedBuffer&& host_data);  // host resident
    FrTensor& gpu();
    void offload();
    // Elements [first
    FrTensor slice(size_t first, uint count) const;
    // Host storage filled chunk by chunk (e.g. while streaming a file in)
    PinnedBuffer& host_buffer() { return host; }
    uint size() const { return n; }

private:
    uint n;
    std::unique_ptr<FrTensor> dev;
    PinnedBuffer host;
};
using OpState = std::map<std::string, std::shared_ptr<Activation>>;

template <typename Chunk>
inline Fr_t chunked_me(size_t total, uint chunk, const vector<Fr_t>& u, Chunk chunk_at) {
    const uint c = ceilLog2(chunk), nc = static_cast<uint>((total + chunk - 1) / chunk);
    if (u.size() <= c || nc == 1)
        return chunk_at(0)(u);
    const vector<Fr_t> lo(u.begin(), u.begin() + c), hi(u.begin() + c, u.end());
    vector<Fr_t> vals(nc);
    for (uint i = 0; i < nc; ++i)
        vals[i] = chunk_at(i)(lo);
    return FrTensor(nc, vals.data())(hi);
}

std::string opt_op_name(uint op_in_layer);
// Tensors live before op `cut_in_layer` (0 = layer input {x})
const std::vector<std::string>& opt_live_at(uint cut_in_layer);

class OptOpRunner {
public:
    OptOpRunner(const string& workdir, const LayerDims& d, const string& relu_table, bool prove,
                LayerTimers& t);
    ~OptOpRunner();
    // Runs model op `global_op` on st
    void run(uint global_op, OpState& st);
    double load_s = 0;  // weights and lookup tables, loaded on first use
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

#endif  // OPT_LAYER_CUH
