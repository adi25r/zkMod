#include <hip/hip_runtime.h>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

#include "gpu_field.hpp"
#include "gkr_pred_gpu.hpp"
#include "gpu_uvm_alloc.hpp"
#include "gkr_gpu.hpp"

typedef gpu_fe Fr_t;

// Dynamic shared memory for the block reductions
extern __shared__ Fr_t pred_smem[];

namespace {

constexpr int PRED_BLOCK = 256;
// Cap the grid so the host-side cross-block reduction stays trivial
constexpr int PRED_MAX_BLOCKS = 4096;

// Profiling
double g_t_beta = 0, g_t_stage = 0, g_t_h2d = 0, g_t_kernel = 0;
long g_n_beta = 0, g_n_p1 = 0, g_n_p2 = 0;
size_t g_bytes_gates = 0, g_bytes_beta = 0;
inline double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// Persistent device state
struct PredState {
    Fr_t* d_beta[3] = {nullptr, nullptr, nullptr};
    size_t cap_beta[3] = {0, 0, 0};

    u32 *d_g = nullptr, *d_u = nullptr, *d_v = nullptr;
    u8* d_l = nullptr;
    long long* d_sc = nullptr;
    size_t cap_gates = 0;

    Fr_t* d_out = nullptr;  // 3 * PRED_MAX_BLOCKS
    size_t cap_out = 0;
} g_pred;

hipError_t dmalloc_or_evict(void** p, size_t need) {
    if (GKR_DMALLOC(p, need) == hipSuccess)
        return hipSuccess;
    (void)hipGetLastError();
    if (!gkr_sc_release_idle())
        return hipErrorOutOfMemory;
    return GKR_DMALLOC(p, need);
}

void ensure_bytes(void** p, size_t* cap, size_t need, const char* tag) {
    if (*cap >= need)
        return;
    if (*p) {
        HIP_CHECK(hipFree(*p));
        *p = nullptr;
    }
    HIP_CHECK(dmalloc_or_evict(p, need));
    *cap = need;
    zkgpt_gpu::gkr_vram_log(tag, (long)need);
}

// Kernels

// sc travels as int64
__device__ inline Fr_t sc_to_fr_dev(long long v) {
    const bool neg = v < 0;
    // (v+1)+1 rather than -v: well-defined at INT64_MIN
    unsigned long long u = neg ? (unsigned long long)(-(v + 1)) + 1ULL : (unsigned long long)v;
    Fr_t raw = blstrs__scalar__Scalar_ZERO;
    raw.val[0] = (uint)(u & 0xffffffffULL);
    raw.val[1] = (uint)(u >> 32);
    Fr_t m = blstrs__scalar__Scalar_mont(raw);
    return neg ? device_sub(blstrs__scalar__Scalar_ZERO, m) : m;
}

__device__ inline Fr_t block_reduce(Fr_t v) {
    Fr_t* sm = pred_smem;
    const unsigned tid = threadIdx.x;
    sm[tid] = v;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sm[tid] = device_add(sm[tid], sm[tid + s]);
        __syncthreads();
    }
    return sm[0];
}

__global__ void k_pred1(const u32* __restrict__ gg, const u32* __restrict__ gu, const u8* __restrict__ glu,
                        const long long* __restrict__ gsc, u32 n, const Fr_t* __restrict__ beta_g,
                        const Fr_t* __restrict__ beta_u, Fr_t* __restrict__ out) {
    Fr_t a0 = blstrs__scalar__Scalar_ZERO, a1 = blstrs__scalar__Scalar_ZERO;
    for (u32 i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        Fr_t t = device_mul(device_mul(beta_g[gg[i]], beta_u[gu[i]]), sc_to_fr_dev(gsc[i]));
        if (glu[i])
            a1 = device_add(a1, t);
        else
            a0 = device_add(a0, t);
    }
    Fr_t r0 = block_reduce(a0);
    __syncthreads();
    Fr_t r1 = block_reduce(a1);
    if (threadIdx.x == 0) {
        out[blockIdx.x * 2 + 0] = r0;
        out[blockIdx.x * 2 + 1] = r1;
    }
}

__global__ void k_pred2(const u32* __restrict__ gg, const u32* __restrict__ gu, const u32* __restrict__ gv,
                        const u8* __restrict__ gl, const long long* __restrict__ gsc, u32 n,
                        const Fr_t* __restrict__ beta_g, const Fr_t* __restrict__ beta_u,
                        const Fr_t* __restrict__ beta_v, Fr_t* __restrict__ out) {
    Fr_t a0 = blstrs__scalar__Scalar_ZERO;
    Fr_t a1 = blstrs__scalar__Scalar_ZERO;
    Fr_t a2 = blstrs__scalar__Scalar_ZERO;
    for (u32 i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        Fr_t t = device_mul(device_mul(beta_g[gg[i]], beta_u[gu[i]]),
                            device_mul(beta_v[gv[i]], sc_to_fr_dev(gsc[i])));
        const u8 li = gl[i];
        if (li == 0)
            a0 = device_add(a0, t);
        else if (li == 1)
            a1 = device_add(a1, t);
        else
            a2 = device_add(a2, t);
    }
    Fr_t r0 = block_reduce(a0);
    __syncthreads();
    Fr_t r1 = block_reduce(a1);
    __syncthreads();
    Fr_t r2 = block_reduce(a2);
    if (threadIdx.x == 0) {
        out[blockIdx.x * 3 + 0] = r0;
        out[blockIdx.x * 3 + 1] = r1;
        out[blockIdx.x * 3 + 2] = r2;
    }
}

// Host staging
std::vector<u32> h_g, h_u, h_v;
std::vector<u8> h_l;
std::vector<long long> h_sc;
std::vector<Fr_t> h_out;

int grid_for(u32 n) {
    int b = (int)((n + PRED_BLOCK - 1) / PRED_BLOCK);
    return std::max(1, std::min(b, PRED_MAX_BLOCKS));
}

}  // namespace

// ON by default
bool gkr_pred_gpu_enabled() {
    static int on = getenv("GKR_PRED_CPU") ? 0 : 1;
    return on != 0;
}

void gkr_pred_upload_beta(int which, const std::vector<F>& tbl) {
    if (!gkr_pred_gpu_enabled() || tbl.empty())
        return;
    double t0 = now_s();
    const size_t bytes = tbl.size() * sizeof(Fr_t);
    ensure_bytes((void**)&g_pred.d_beta[which], &g_pred.cap_beta[which], bytes, "gkr_pred beta");
    // fe_to_gpu is a memcpy
    static_assert(sizeof(F) == sizeof(Fr_t), "host/device scalar layout must match");
    HIP_CHECK(zkgpt_gpu::gkr_h2d(g_pred.d_beta[which], tbl.data(), bytes));
    g_bytes_beta += bytes;
    g_t_beta += now_s() - t0;
    g_n_beta++;
}

// Stage the index/scalar SoA for a gate list and push it to the device
template <class Gate>
static void stage_gates(const std::vector<Gate>& gates, bool bin) {
    const size_t n = gates.size();
    double t0 = now_s();
    h_g.resize(n);
    h_u.resize(n);
    h_sc.resize(n);
    h_l.resize(n);
    if (bin)
        h_v.resize(n);
#pragma omp parallel for schedule(static)
    for (long i = 0; i < (long)n; ++i) {
        const auto& gt = gates[i];
        h_g[i] = gt.g;
        h_u[i] = gt.u;
        h_sc[i] = (int64_t)gt.sc;
        if constexpr (std::is_same_v<Gate, binGate>) {
            h_v[i] = gt.v;
            h_l[i] = gt.l;
        } else {
            h_l[i] = (u8)(gt.lu != 0);
        }
    }
    g_t_stage += now_s() - t0;
    t0 = now_s();

    // The five arrays grow together, so one capacity counter covers all of them
    if (g_pred.cap_gates < n) {
        for (void** p : {(void**)&g_pred.d_g, (void**)&g_pred.d_u, (void**)&g_pred.d_v, (void**)&g_pred.d_l,
                         (void**)&g_pred.d_sc}) {
            if (*p) {
                HIP_CHECK(hipFree(*p));
                *p = nullptr;
            }
        }
        HIP_CHECK(dmalloc_or_evict((void**)&g_pred.d_g, n * sizeof(u32)));
        HIP_CHECK(dmalloc_or_evict((void**)&g_pred.d_u, n * sizeof(u32)));
        HIP_CHECK(dmalloc_or_evict((void**)&g_pred.d_v, n * sizeof(u32)));
        HIP_CHECK(dmalloc_or_evict((void**)&g_pred.d_l, n * sizeof(u8)));
        HIP_CHECK(dmalloc_or_evict((void**)&g_pred.d_sc, n * sizeof(long long)));
        g_pred.cap_gates = n;
        zkgpt_gpu::gkr_vram_log("gkr_pred gates", (long)(n * (12 + 1 + sizeof(long long))));
    }
    HIP_CHECK(zkgpt_gpu::gkr_h2d(g_pred.d_g, h_g.data(), n * sizeof(u32)));
    HIP_CHECK(zkgpt_gpu::gkr_h2d(g_pred.d_u, h_u.data(), n * sizeof(u32)));
    HIP_CHECK(zkgpt_gpu::gkr_h2d(g_pred.d_l, h_l.data(), n * sizeof(u8)));
    HIP_CHECK(zkgpt_gpu::gkr_h2d(g_pred.d_sc, h_sc.data(), n * sizeof(long long)));
    if (bin)
        HIP_CHECK(zkgpt_gpu::gkr_h2d(g_pred.d_v, h_v.data(), n * sizeof(u32)));
    g_bytes_gates += n * (bin ? 13 + sizeof(long long) : 9 + sizeof(long long));
    g_t_h2d += now_s() - t0;
}

void gkr_predicate1_gpu(const std::vector<uniGate>& gates, F out[2]) {
    out[0].clear();
    out[1].clear();
    const size_t n = gates.size();
    if (!n)
        return;

    stage_gates(gates, /*bin=*/false);

    const int blocks = grid_for((u32)n);
    ensure_bytes((void**)&g_pred.d_out, &g_pred.cap_out, 3 * PRED_MAX_BLOCKS * sizeof(Fr_t), "gkr_pred out");
    double t0 = now_s();
    hipLaunchKernelGGL(k_pred1, dim3(blocks), dim3(PRED_BLOCK), PRED_BLOCK * sizeof(Fr_t), 0, g_pred.d_g,
                       g_pred.d_u, g_pred.d_l, g_pred.d_sc, (u32)n, g_pred.d_beta[0], g_pred.d_beta[1],
                       g_pred.d_out);
    HIP_CHECK(hipDeviceSynchronize());
    g_t_kernel += now_s() - t0;

    h_out.resize(2 * blocks);
    HIP_CHECK(hipMemcpy(h_out.data(), g_pred.d_out, 2 * blocks * sizeof(Fr_t), hipMemcpyDeviceToHost));
    for (int b = 0; b < blocks; ++b) {
        out[0] += fe_from_gpu(h_out[b * 2 + 0]);
        out[1] += fe_from_gpu(h_out[b * 2 + 1]);
    }
    g_n_p1++;
}

void gkr_predicate2_gpu(const std::vector<binGate>& gates, F out[3]) {
    out[0].clear();
    out[1].clear();
    out[2].clear();
    const size_t n = gates.size();
    if (!n)
        return;

    stage_gates(gates, /*bin=*/true);

    const int blocks = grid_for((u32)n);
    ensure_bytes((void**)&g_pred.d_out, &g_pred.cap_out, 3 * PRED_MAX_BLOCKS * sizeof(Fr_t), "gkr_pred out");
    double t0 = now_s();
    hipLaunchKernelGGL(k_pred2, dim3(blocks), dim3(PRED_BLOCK), PRED_BLOCK * sizeof(Fr_t), 0, g_pred.d_g,
                       g_pred.d_u, g_pred.d_v, g_pred.d_l, g_pred.d_sc, (u32)n, g_pred.d_beta[0],
                       g_pred.d_beta[1], g_pred.d_beta[2], g_pred.d_out);
    HIP_CHECK(hipDeviceSynchronize());
    g_t_kernel += now_s() - t0;

    h_out.resize(3 * blocks);
    HIP_CHECK(hipMemcpy(h_out.data(), g_pred.d_out, 3 * blocks * sizeof(Fr_t), hipMemcpyDeviceToHost));
    for (int b = 0; b < blocks; ++b) {
        out[0] += fe_from_gpu(h_out[b * 3 + 0]);
        out[1] += fe_from_gpu(h_out[b * 3 + 1]);
        out[2] += fe_from_gpu(h_out[b * 3 + 2]);
    }
    g_n_p2++;
}

// The gate arrays are re-staged on every predicate call
void gkr_pred_release_gates() {
    for (void** p : {(void**)&g_pred.d_g, (void**)&g_pred.d_u, (void**)&g_pred.d_v, (void**)&g_pred.d_l,
                     (void**)&g_pred.d_sc})
        if (*p) {
            HIP_CHECK(hipFree(*p));
            *p = nullptr;
        }
    g_pred.cap_gates = 0;
}

void gkr_pred_teardown() {
    for (int i = 0; i < 3; ++i)
        if (g_pred.d_beta[i]) {
            HIP_CHECK(hipFree(g_pred.d_beta[i]));
            g_pred.d_beta[i] = nullptr;
            g_pred.cap_beta[i] = 0;
        }
    for (void** p : {(void**)&g_pred.d_g, (void**)&g_pred.d_u, (void**)&g_pred.d_v, (void**)&g_pred.d_l,
                     (void**)&g_pred.d_sc, (void**)&g_pred.d_out}) {
        if (*p) {
            HIP_CHECK(hipFree(*p));
            *p = nullptr;
        }
    }
    g_pred.cap_gates = g_pred.cap_out = 0;
}

void gkr_pred_report() {
    if (!getenv("GKR_GEN_PROF"))
        return;
    fprintf(stderr,
            "[gkr_pred_prof] beta_h2d=%.3fs(n=%ld, %.2fGB)  stage=%.3fs  "
            "gate_h2d=%.3fs(%.2fGB)  kernel=%.3fs  (pred1 n=%ld, pred2 n=%ld)\n",
            g_t_beta, g_n_beta, g_bytes_beta / 1073741824.0, g_t_stage, g_t_h2d, g_bytes_gates / 1073741824.0,
            g_t_kernel, g_n_p1, g_n_p2);
}
