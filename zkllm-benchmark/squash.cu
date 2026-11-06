#include "squash.cuh"

#include <cstdlib>
#include <algorithm>
#include <map>
#include <string>

KERNEL void scalar_to_int_kernel(const Fr_t* scalar_ptr, int* int_ptr, uint n);
KERNEL void int_to_scalar_kernel(int* int_ptr, Fr_t* scalar_ptr, uint n);

namespace squash {

namespace {

// A table's queued inputs as int32 on the GPU (a quarter of the field elements' size).
struct Queue {
    int* d = nullptr;
    size_t n = 0, cap = 0;
    size_t entries = 0;
    void release() { if (d) cudaFree(d); *this = Queue(); }
};

struct State {
    bool active = false;
    std::map<uint, Queue> ranges;                            // by scaling factor
    std::map<tLookupRangeMapping*, Queue> mappings;
    size_t proofs = 0;
};

State& state() { static State s; return s; }

size_t max_elems()
{
    const char* s = std::getenv("ZKLLM_SQUASH_MAX_LOG");
    const long v = s ? std::atol(s) : 24;
    return size_t(1) << (v > 0 && v < 31 ? v : 24);
}

void append_ints(Queue& q, const FrTensor& t)
{
    if (q.n + t.size > q.cap) {
        const size_t cap = std::max(q.n + t.size, std::min(2 * q.cap, max_elems()));
        int* d;
        cudaMalloc((void**)&d, sizeof(int) * cap);
        if (q.n) cudaMemcpy(d, q.d, sizeof(int) * q.n, cudaMemcpyDeviceToDevice);
        if (q.d) cudaFree(q.d);
        q.d = d;
        q.cap = cap;
    }
    scalar_to_int_kernel<<<(t.size + FrNumThread - 1) / FrNumThread, FrNumThread>>>(t.gpu_data, q.d + q.n, t.size);
    cudaDeviceSynchronize();
    q.n += t.size;
}

KERNEL void squash_range_check(const int* v, size_t n, long low, long len, int* bad)
{
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x)
        if (v[i] < low || v[i] >= low + len) { atomicOr(bad, 1); return; }
}

// tLookup::prep indexes the table by value without bounds checks, so a value outside the table
// would write out of bounds instead of failing the argument; reject it here.
void check_in_table(const Queue& q, long low, long len, const char* what)
{
    int* bad;
    cudaMalloc((void**)&bad, sizeof(int));
    cudaMemset(bad, 0, sizeof(int));
    squash_range_check<<<1024, 256>>>(q.d, q.n, low, len, bad);
    int h;
    cudaMemcpy(&h, bad, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(bad);
    if (h) throw std::runtime_error(std::string("squash: ") + what + " value outside the lookup table");
}

FrTensor to_field(const Queue& q)
{
    FrTensor S(q.n);
    int_to_scalar_kernel<<<(q.n + FrNumThread - 1) / FrNumThread, FrNumThread>>>(q.d, S.gpu_data, q.n);
    cudaDeviceSynchronize();
    return S;
}

void prove_range(uint sf, Queue& q)
{
    if (!q.n) return;
    check_in_table(q, -static_cast<long>(sf >> 1), sf, "remainder");
    FrTensor S = to_field(q);
    q.release();
    tLookupRange tl(-static_cast<int>(sf >> 1), sf);
    FrTensor m = tl.prep(S);
    auto u = random_vec(ceilLog2(S.size)), v = random_vec(ceilLog2(S.size));
    auto ab = random_vec(2);
    std::vector<Polynomial> proof;
    tl.prove(S, m, ab[0], ab[1], u, v, proof);
    ++ state().proofs;
}

void prove_mapping(tLookupRangeMapping* lookup, Queue& q)
{
    if (!q.n) return;
    check_in_table(q, lookup->low, lookup->table.size, "mapping");
    FrTensor S = to_field(q);
    q.release();
    auto p = (*lookup)(S);
    auto u = random_vec(ceilLog2(S.size)), v = random_vec(ceilLog2(S.size));
    auto r = random_vec(3);
    std::vector<Polynomial> proof;
    lookup->prove(S, p.first, p.second, r[0], r[1], r[2], u, v, proof);
    ++ state().proofs;
}

template <class Key, class Prove>
void add(std::map<Key, Queue>& qs, Key k, const FrTensor& t, Prove prove)
{
    Queue& q = qs[k];
    if (q.n && q.n + t.size > max_elems()) prove(k, q);
    append_ints(q, t);
    ++ q.entries;
    if (q.n >= max_elems()) prove(k, q);
}

} // namespace

bool enabled()
{
    const char* s = std::getenv("ZKLLM_SQUASH");
    return s && std::string(s) == "1";
}

void begin()
{
    auto& s = state();
    if (!s.ranges.empty() || !s.mappings.empty()) throw std::runtime_error("squash::begin: lookups still queued");
    s.active = enabled();
    s.proofs = 0;
}

bool active() { return state().active; }

void add_range(uint scaling_factor, const FrTensor& rem)
{
    add(state().ranges, scaling_factor, rem, prove_range);
}

void add_mapping(tLookupRangeMapping& lookup, const FrTensor& in)
{
    add(state().mappings, &lookup, in, prove_mapping);
}

size_t flush()
{
    auto& s = state();
    auto ranges = std::move(s.ranges);
    auto mappings = std::move(s.mappings);
    s.ranges.clear();
    s.mappings.clear();
    s.active = false;
    try {
        for (auto& kv : ranges) prove_range(kv.first, kv.second);
        for (auto& kv : mappings) prove_mapping(kv.first, kv.second);
    } catch (...) {
        for (auto& kv : ranges) kv.second.release();
        for (auto& kv : mappings) kv.second.release();
        throw;
    }
    return s.proofs;
}

} // namespace squash
