#ifndef ZKGPT_GPU_UVM_ALLOC_HPP
#define ZKGPT_GPU_UVM_ALLOC_HPP
// Shared device-buffer allocator
#include <hip/hip_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <thread>
#include <chrono>
#include <dirent.h>
#include <unistd.h>
#include "config.hpp"

namespace zkgpt_gpu {

// Per-partition VRAM high-water tracking
namespace vram_detail {
inline std::atomic<size_t>& peak_used() {
    static std::atomic<size_t> v{0};
    return v;
}
inline std::atomic<bool>& sampler_on() {
    static std::atomic<bool> r{false};
    return r;
}
inline std::atomic<size_t>& fallback_base() {
    static std::atomic<size_t> v{0};
    return v;
}

// Sum this process's per-GPU VRAM from the KFD. Returns false if unavailable
inline bool kfd_proc_vram(size_t* out) {
    char dir[128];
    snprintf(dir, sizeof dir, "/sys/class/kfd/kfd/proc/%d", (int)getpid());
    DIR* d = opendir(dir);
    if (!d)
        return false;
    size_t total = 0;
    bool any = false;
    while (struct dirent* e = readdir(d)) {
        if (strncmp(e->d_name, "vram_", 5) != 0)
            continue;
        char path[256];
        snprintf(path, sizeof path, "%s/%s", dir, e->d_name);
        if (FILE* f = fopen(path, "r")) {
            unsigned long long v = 0;
            if (fscanf(f, "%llu", &v) == 1) {
                total += (size_t)v;
                any = true;
            }
            fclose(f);
        }
    }
    closedir(d);
    if (any)
        *out = total;
    return any;
}

inline void sampler_loop() {
    while (sampler_on().load(std::memory_order_relaxed)) {
        size_t used = 0;
        bool have = kfd_proc_vram(&used);
        if (!have) {
            size_t fr = 0, tot = 0;
            if (hipMemGetInfo(&fr, &tot) == hipSuccess && tot >= fr) {
                size_t raw = tot - fr, base = fallback_base().load(std::memory_order_relaxed);
                used = raw > base ? raw - base : 0;
                have = true;
            }
        }
        if (have) {
            size_t cur = peak_used().load(std::memory_order_relaxed);
            while (used > cur && !peak_used().compare_exchange_weak(cur, used)) {
            }
        }
        // 1 ms interval
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
inline std::thread& sampler_thread() {
    static std::thread t;
    return t;
}
}  // namespace vram_detail

// Stop the sampler and JOIN it
inline void gkr_vram_peak_stop() {
    if (vram_detail::sampler_on().exchange(false)) {
        if (vram_detail::sampler_thread().joinable())
            vram_detail::sampler_thread().join();
    }
}

// Start the sampler once (idempotent)
inline void gkr_vram_peak_start() {
    if (!vram_detail::sampler_on().exchange(true)) {
        size_t fr = 0, tot = 0;
        if (hipMemGetInfo(&fr, &tot) == hipSuccess && tot >= fr)
            vram_detail::fallback_base().store(tot - fr, std::memory_order_relaxed);
        vram_detail::sampler_thread() = std::thread(vram_detail::sampler_loop);
        std::atexit(gkr_vram_peak_stop);
    }
}
inline void gkr_vram_peak_reset() {
    vram_detail::peak_used().store(0, std::memory_order_relaxed);
}
inline double gkr_vram_peak_mib() {
    return vram_detail::peak_used().load(std::memory_order_relaxed) / (1024.0 * 1024.0);
}

namespace xfer_detail {
inline std::atomic<unsigned long long>& h2d_bytes() {
    static std::atomic<unsigned long long> v{0};
    return v;
}
inline std::atomic<unsigned long long>& d2h_bytes() {
    static std::atomic<unsigned long long> v{0};
    return v;
}
}  // namespace xfer_detail

// Add sz bytes to the process-wide H2D / D2H counters
inline double& gkr_xfer_h2d_s_ref() {
    static double v = 0;
    return v;
}
inline double& gkr_xfer_d2h_s_ref() {
    static double v = 0;
    return v;
}
inline void gkr_xfer_time_reset() {
    gkr_xfer_h2d_s_ref() = 0;
    gkr_xfer_d2h_s_ref() = 0;
}
struct GkrXferTimer {
    double* acc;
    std::chrono::steady_clock::time_point t0;
    explicit GkrXferTimer(double* a) : acc(a), t0(std::chrono::steady_clock::now()) {}
    ~GkrXferTimer() { *acc += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }
};

inline void gkr_xfer_add_h2d(size_t sz) {
    xfer_detail::h2d_bytes().fetch_add(sz, std::memory_order_relaxed);
}
inline void gkr_xfer_add_d2h(size_t sz) {
    xfer_detail::d2h_bytes().fetch_add(sz, std::memory_order_relaxed);
}

// Post: both counters == 0
inline void gkr_xfer_reset() {
    xfer_detail::h2d_bytes().store(0, std::memory_order_relaxed);
    xfer_detail::d2h_bytes().store(0, std::memory_order_relaxed);
}
// Returns: bytes counted since the last gkr_xfer_reset(), in GiB
inline double gkr_xfer_h2d_gib() {
    return xfer_detail::h2d_bytes().load(std::memory_order_relaxed) / (1024.0 * 1024.0 * 1024.0);
}
inline double gkr_xfer_d2h_gib() {
    return xfer_detail::d2h_bytes().load(std::memory_order_relaxed) / (1024.0 * 1024.0 * 1024.0);
}

// hipMemcpy(dst, src, sz, hipMemcpy{Host,Device}To{Device,Host}), plus gkr_xfer_add_*(sz)
inline double& gkr_h2d_time_s() {
    static double v = 0;
    return v;
}

inline double& gkr_d2h_time_s() {
    static double v = 0;
    return v;
}

inline void gkr_xfer_wall_reset() {
    gkr_h2d_time_s() = 0;
    gkr_d2h_time_s() = 0;
}

inline hipError_t gkr_h2d(void* dst, const void* src, size_t sz) {
    gkr_xfer_add_h2d(sz);
    const auto t0 = std::chrono::steady_clock::now();
    hipError_t r = hipMemcpy(dst, src, sz, hipMemcpyHostToDevice);
    gkr_h2d_time_s() += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return r;
}

inline hipError_t gkr_d2h(void* dst, const void* src, size_t sz) {
    gkr_xfer_add_d2h(sz);
    const auto t0 = std::chrono::steady_clock::now();
    hipError_t r = hipMemcpy(dst, src, sz, hipMemcpyDeviceToHost);
    gkr_d2h_time_s() += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return r;
}

// Debug memory tracking
inline std::atomic<long>& gkr_mem_outstanding() {
    static std::atomic<long> v{0};
    return v;
}
inline bool gkr_mem_dbg() {
    static int on = -1;
    if (on < 0)
        on = getenv("GKR_DBG_MEM") ? 1 : 0;
    return on;
}
inline void gkr_vram_log(const char* tag, long delta_bytes = 0) {
    if (!gkr_mem_dbg())
        return;
    long out = gkr_mem_outstanding().load();
    size_t fr = 0, tot = 0;
    hipMemGetInfo(&fr, &tot);
    fprintf(stderr, "[mem] %-28s  delta=%+8.1f MB  tracked_outstanding=%8.1f MB  free=%7.1f / %7.1f MB\n",
            tag, delta_bytes / 1e6, out / 1e6, fr / 1e6, tot / 1e6);
    fflush(stderr);
}
inline hipError_t gkr_tracked_malloc(void** p, size_t sz, const char* tag) {
    hipError_t e =
#ifdef GKR_UVM
        hipMallocManaged(p, sz);
#else
        hipMalloc(p, sz);
#endif
    if (gkr_mem_dbg()) {
        if (e == hipSuccess)
            gkr_mem_outstanding() += (long)sz;
        size_t fr = 0, tot = 0;
        hipMemGetInfo(&fr, &tot);
        fprintf(stderr,
                "[mem] alloc %-22s  req=%8.1f MB  %s  tracked_outstanding=%8.1f MB  free=%7.1f / %7.1f MB\n",
                tag, sz / 1e6, e == hipSuccess ? "OK  " : "FAIL", gkr_mem_outstanding().load() / 1e6,
                fr / 1e6, tot / 1e6);
        fflush(stderr);
    }
    return e;
}
#define GKR_STR2(x) #x
#define GKR_STR(x) GKR_STR2(x)

#ifdef GKR_UVM

static inline size_t gkr_uvm_zerocopy_threshold() {
    if (const char* e = getenv("GKR_UVM_ZEROCOPY_GB"))
        return (size_t)(atof(e) * (1ull << 30));
    return (size_t)4 << 30;
}

// function definition for the UVM case, which uses hipMallocManaged
static inline hipError_t gkr_dmalloc(void** p, size_t sz) {
    hipError_t e = hipMallocManaged(p, sz);
    if (e != hipSuccess)
        return e;
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess)
        return hipSuccess;

    // Zero-copy for oversubscribing buffers
    if (gkr_config.uvm_zerocopy && sz >= gkr_uvm_zerocopy_threshold()) {
        hipMemAdvise(*p, sz, hipMemAdviseSetPreferredLocation, hipCpuDeviceId);
        hipMemAdvise(*p, sz, hipMemAdviseSetAccessedBy, dev);
    } else if (gkr_config.uvm_prefetch) {
        // Best-effort: make the device the home and pull pages in up front
        hipMemAdvise(*p, sz, hipMemAdviseSetPreferredLocation, dev);
        hipMemPrefetchAsync(*p, sz, dev);
    }
    return hipSuccess;
}
#define GKR_DMALLOC(p, sz) ::zkgpt_gpu::gkr_dmalloc((void**)(p), (sz))
#else
#define GKR_DMALLOC(p, sz) ::zkgpt_gpu::gkr_tracked_malloc((void**)(p), (sz), __FILE__ ":" GKR_STR(__LINE__))
#endif

// Data-movement helpers
static inline hipStream_t gkr_copy_stream() {
    static hipStream_t s = [] {
        hipStream_t t = nullptr;
        (void)hipStreamCreateWithFlags(&t, hipStreamNonBlocking);
        return t;
    }();
    return s;
}

// Persistent non-blocking compute stream
static inline hipStream_t gkr_compute_stream() {
    static hipStream_t s = [] {
        hipStream_t t = nullptr;
        (void)hipStreamCreateWithFlags(&t, hipStreamNonBlocking);
        return t;
    }();
    return s;
}

static inline hipError_t gkr_stage_h2d(void* dst, const void* src, size_t sz, hipStream_t stream) {
    hipError_t e = hipMemcpyAsync(dst, src, sz, hipMemcpyHostToDevice, stream);
    if (e == hipSuccess)
        gkr_xfer_add_h2d(sz);
#ifdef GKR_UVM
    if (e == hipSuccess) {
        int dev = 0;
        if (hipGetDevice(&dev) == hipSuccess)
            hipMemPrefetchAsync(dst, sz, dev, stream);  // make device the home
    }
#endif
    return e;
}

static inline void gkr_prefetch(const void* ptr, size_t sz, hipStream_t stream) {
#ifdef GKR_UVM
    int dev = 0;
    if (hipGetDevice(&dev) == hipSuccess)
        hipMemPrefetchAsync(const_cast<void*>(ptr), sz, dev, stream);
#else
    (void)ptr;
    (void)sz;
    (void)stream;  // device memory is already resident
#endif
}

}  // namespace zkgpt_gpu

#endif  // ZKGPT_GPU_UVM_ALLOC_HPP
