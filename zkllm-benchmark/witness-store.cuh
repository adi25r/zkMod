#ifndef WITNESS_STORE_CUH
#define WITNESS_STORE_CUH

// Witness-first proving (partition prove --witgen-first), zkLLM's own method: store every tensor a proof needs, then
// prove from the stored copies. The record pass runs the forward pass only and appends every compute result of the
// layer path (and the state the proofs read as a side effect: rescaling remainders, softmax segments) to that layer's
// record in host memory. The replay pass walks the same code with proving on, and every compute step returns the
// stored tensor (host->device) instead of recomputing; a layer's record is freed once it has been proven.
//
// Each tensor is stored as int64 (as zkLLM's save_long), or as raw field elements when a value does not round-trip.

#include "fr-tensor.cuh"
#include "rescaling.cuh"
#include "timer.hpp"
#include <type_traits>
#include <utility>
#include <vector>

namespace wstore {

enum Mode { OFF, RECORD, REPLAY };

struct Stats {
    double store_s = 0;   // record: int64 conversion + device->host
    double load_s = 0;    // replay: host->device + conversion back
    size_t bytes = 0, peak_bytes = 0, tensors = 0, raw_tensors = 0, pageable_tensors = 0;
};

void set_mode(Mode m);
Mode mode();
void begin_layer(uint layer);   // record: start the layer's record; replay: rewind to it
void end_layer();               // replay: check the record was consumed whole, then free it
void put(const FrTensor& t);
FrTensor get();
void put_count(size_t n);
size_t get_count();
Stats& stats();

// Compute step of the layer path: in replay its result comes from the store and f is not run.
template <typename F>
auto witgen(double& compute_s, F&& f) -> std::decay_t<decltype(f())>
{
    using R = std::decay_t<decltype(f())>;
    if constexpr (std::is_same_v<R, FrTensor>) {
        if (mode() == REPLAY) return get();
        Timer tm; tm.start();
        R r = f();
        tm.stop(); compute_s += tm.getTotalTime();
        if (mode() == RECORD) put(r);
        return r;
    } else {   // pair<FrTensor, FrTensor> (lookup output + multiplicities)
        if (mode() == REPLAY) { FrTensor a = get(); FrTensor b = get(); return R(std::move(a), std::move(b)); }
        Timer tm; tm.start();
        R r = f();
        tm.stop(); compute_s += tm.getTotalTime();
        if (mode() == RECORD) { put(r.first); put(r.second); }
        return r;
    }
}

// State a later proof reads that the compute step produced through an out-parameter.
inline void side(FrTensor& x)
{
    if (mode() == RECORD) put(x);
    else if (mode() == REPLAY) x = get();
}

inline void side(std::vector<FrTensor>& v)
{
    if (mode() == RECORD) { put_count(v.size()); for (auto& x : v) put(x); }
    else if (mode() == REPLAY) { v.clear(); const size_t n = get_count(); for (size_t i = 0; i < n; ++ i) v.push_back(get()); }
}

// rs(X), plus the remainder rs keeps for rs.prove.
inline FrTensor rescale(double& compute_s, Rescaling& rs, const FrTensor& X)
{
    FrTensor out = witgen(compute_s, [&] { return rs(X); });
    if (mode() == RECORD) put(*rs.rem_tensor_ptr);
    else if (mode() == REPLAY) { delete rs.rem_tensor_ptr; rs.rem_tensor_ptr = new FrTensor(get()); }
    return out;
}

} // namespace wstore

#endif // WITNESS_STORE_CUH
