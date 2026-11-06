#include "witness-store.cuh"
#include <cstdlib>
#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>

namespace wstore {
namespace {

enum Format : uint32_t { LONG = 0, RAW = 1, COUNT = 2 };

// One stored tensor (or a vector count)
struct Entry {
    uint64_t n = 0;
    Format fmt = LONG;
    void* host = nullptr;
    bool pinned = false;
};

Mode g_mode = OFF;
std::map<uint, std::vector<Entry>> g_layers;
std::vector<Entry>* g_cur = nullptr;
size_t g_pos = 0;
uint g_layer = 0;
Stats g_stats;

// Reused device staging: an int64 buffer, grown on demand, and the round-trip flag
long* g_dev_long = nullptr;
size_t g_dev_long_n = 0;
int* g_dev_flag = nullptr;

long* dev_long(size_t n) {
    if (n > g_dev_long_n) {
        if (g_dev_long)
            cudaFree(g_dev_long);
        cudaMalloc((void**)&g_dev_long, sizeof(long) * n);
        g_dev_long_n = n;
    }
    if (!g_dev_flag)
        cudaMalloc((void**)&g_dev_flag, sizeof(int));
    return g_dev_long;
}

size_t entry_bytes(const Entry& e) {
    return e.fmt == COUNT ? 0 : static_cast<size_t>(e.n) * (e.fmt == RAW ? sizeof(Fr_t) : sizeof(long));
}

void host_alloc(Entry& e, size_t bytes) {
    if (cudaMallocHost(&e.host, bytes) == cudaSuccess) {
        e.pinned = true;
        return;
    }
    cudaGetLastError();  // clear the failed pinned allocation
    e.host = std::malloc(bytes);
    if (!e.host)
        throw std::runtime_error("wstore: out of host memory (" + std::to_string(bytes) + " bytes)");
    e.pinned = false;
    g_stats.pageable_tensors++;
}

void host_free(Entry& e) {
    if (!e.host)
        return;
    if (e.pinned)
        cudaFreeHost(e.host);
    else
        std::free(e.host);
    g_stats.bytes -= entry_bytes(e);
    e.host = nullptr;
}

const Entry& next(const char* want) {
    if (!g_cur || g_pos >= g_cur->size())
        throw std::runtime_error("wstore: replay of layer " + std::to_string(g_layer) +
                                 " read past its record (" + want + ")");
    return (*g_cur)[g_pos++];
}

// flag = 1 if some element does not survive scalar -> int64 -> scalar
KERNEL void to_long_checked_kernel(const Fr_t* x, long* out, uint n, int* flag) {
    const uint i = GET_GLOBAL_ID();
    if (i >= n)
        return;
    const long l = scalar_to_long(x[i]);
    out[i] = l;
    const Fr_t back = long_to_scalar(l);
    for (int k = 0; k < 8; ++k)
        if (back.val[k] != x[i].val[k]) {
            *flag = 1;
            break;
        }
}

}  // namespace

void set_mode(Mode m) {
    g_mode = m;
}
Mode mode() {
    return g_mode;
}
Stats& stats() {
    return g_stats;
}

void begin_layer(uint layer) {
    if (g_mode == OFF)
        return;
    g_layer = layer;
    g_pos = 0;
    if (g_mode == RECORD) {
        auto& rec = g_layers[layer];
        for (auto& e : rec)
            host_free(e);
        rec.clear();
        g_cur = &rec;
    } else {
        auto it = g_layers.find(layer);
        if (it == g_layers.end())
            throw std::runtime_error("wstore: no record for layer " + std::to_string(layer));
        g_cur = &it->second;
    }
}

void end_layer() {
    if (g_mode == REPLAY && g_cur) {
        if (g_pos != g_cur->size())
            throw std::runtime_error("wstore: replay did not consume the whole record of layer " +
                                     std::to_string(g_layer));
        for (auto& e : *g_cur)
            host_free(e);
        g_layers.erase(g_layer);
    }
    g_cur = nullptr;
}

void put(const FrTensor& t) {
    const uint n = t.size;
    Timer tm;
    tm.start();
    long* dl = dev_long(n);
    cudaMemset(g_dev_flag, 0, sizeof(int));
    to_long_checked_kernel<<<(n + FrNumThread - 1) / FrNumThread, FrNumThread>>>(t.gpu_data, dl, n,
                                                                                 g_dev_flag);
    int lossy = 0;
    cudaMemcpy(&lossy, g_dev_flag, sizeof(int), cudaMemcpyDeviceToHost);
    Entry e;
    e.n = n;
    e.fmt = lossy ? RAW : LONG;
    const size_t bytes = entry_bytes(e);
    host_alloc(e, bytes);
    cudaMemcpy(e.host, lossy ? static_cast<const void*>(t.gpu_data) : static_cast<const void*>(dl), bytes,
               cudaMemcpyDeviceToHost);
    tm.stop();
    g_stats.store_s += tm.getTotalTime();
    g_cur->push_back(e);
    g_stats.bytes += bytes;
    g_stats.peak_bytes = std::max(g_stats.peak_bytes, g_stats.bytes);
    g_stats.tensors++;
    if (lossy)
        g_stats.raw_tensors++;
}

FrTensor get() {
    const Entry& e = next("tensor");
    if (e.fmt == COUNT)
        throw std::runtime_error("wstore: replay out of step (count where a tensor was recorded)");
    Timer tm;
    tm.start();
    const uint n = static_cast<uint>(e.n);
    FrTensor out(n);
    if (e.fmt == RAW)
        cudaMemcpy(out.gpu_data, e.host, entry_bytes(e), cudaMemcpyHostToDevice);
    else {
        long* dl = dev_long(n);
        cudaMemcpy(dl, e.host, entry_bytes(e), cudaMemcpyHostToDevice);
        long_to_scalar_kernel<<<(n + FrNumThread - 1) / FrNumThread, FrNumThread>>>(dl, out.gpu_data, n);
        cudaDeviceSynchronize();
    }
    tm.stop();
    g_stats.load_s += tm.getTotalTime();
    return out;
}

void put_count(size_t n) {
    Entry e;
    e.n = n;
    e.fmt = COUNT;
    g_cur->push_back(e);
}

size_t get_count() {
    const Entry& e = next("count");
    if (e.fmt != COUNT)
        throw std::runtime_error("wstore: replay out of step (tensor where a count was recorded)");
    return e.n;
}

}  // namespace wstore
