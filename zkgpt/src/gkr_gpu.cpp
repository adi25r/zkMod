#include <hip/hip_runtime.h>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "gpu_field.hpp"
#include "polynomial.h"
#include "global_var.hpp"
#include "gpu_uvm_alloc.hpp"
#include "gkr_pred_gpu.hpp"

typedef gpu_fe Fr_t;

// Lightweight bucket profiler
static double g_t_begin = 0, g_t_update = 0;
static long   g_n_begin = 0, g_n_update = 0;
static inline double _prof_now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}
void gkr_sc_report() {
    fprintf(stderr, "[gkr_sc_prof] sc_begin=%.3fs(n=%ld)  sc_update=%.3fs(n=%ld)\n",
            g_t_begin, g_n_begin, g_t_update, g_n_update);
    g_t_begin = g_t_update = 0;
    g_n_begin = g_n_update = 0;
}

// Kernel: sumcheck polynomial fold (double-buffer, race-free)
//
// Reads from in_va/in_vb/in_ma/in_mb 
// Writes folded values to out_va/out_vb/out_ma/out_mb.
// Accumulates per-block quadratic contribution into out_qa/out_qb/out_qc.
//
// Each thread i in [0, half_n) handles pair (g0=2i, g1=2i+1):
//   ev0_v = in_va[g0]*r + in_vb[g0]
//   ev1_v = in_va[g1]*r + in_vb[g1]
//   out_va[i] = ev1_v - ev0_v;  out_vb[i] = ev0_v  (interpolate)
//   same for ma/mb
//   contribution: (nma*x + ev0m) * (nva*x + ev0v) added to per-block output
//
// `base` is the global pair index of in[0]/out[0]: 0 for a resident round, the
// chunk offset for a streamed one. Only the total_sz test needs the global index.

#define SC_BLOCK 256

__global__ void k_sc_update(
    const Fr_t* __restrict__ in_va, const Fr_t* __restrict__ in_vb,
    const Fr_t* __restrict__ in_ma, const Fr_t* __restrict__ in_mb,
    Fr_t* __restrict__ out_va, Fr_t* __restrict__ out_vb,
    Fr_t* __restrict__ out_ma, Fr_t* __restrict__ out_mb,
    Fr_t r,
    Fr_t* __restrict__ out_qa,
    Fr_t* __restrict__ out_qb,
    Fr_t* __restrict__ out_qc,
    uint half_n, uint total_sz, uint base)
{
    __shared__ Fr_t s_qa[SC_BLOCK], s_qb[SC_BLOCK], s_qc[SC_BLOCK];

    const uint tid = threadIdx.x;
    const uint i   = blockIdx.x * blockDim.x + tid;

    Fr_t qa  = gpu_fe{};
    Fr_t qb_ = gpu_fe{};
    Fr_t qc  = gpu_fe{};

    if (i < half_n) {
        const uint g0 = i << 1, g1 = i << 1 | 1;

        Fr_t ev0v = device_add(device_mul(in_va[g0], r), in_vb[g0]);
        Fr_t ev1v = device_add(device_mul(in_va[g1], r), in_vb[g1]);
        Fr_t ev0m = device_add(device_mul(in_ma[g0], r), in_mb[g0]);
        Fr_t ev1m = device_add(device_mul(in_ma[g1], r), in_mb[g1]);

        if (((base + i) << 1) < total_sz) {
            Fr_t nva = device_sub(ev1v, ev0v);
            Fr_t nma = device_sub(ev1m, ev0m);

            out_va[i] = nva;  out_vb[i] = ev0v;
            out_ma[i] = nma;  out_mb[i] = ev0m;

            qa  = device_mul(nma, nva);
            qb_ = device_add(
                device_mul(nma, ev0v),
                device_mul(ev0m, nva));
            qc  = device_mul(ev0m, ev0v);
        } else {
            out_va[i] = gpu_fe{};
            out_vb[i] = gpu_fe{};
            out_ma[i] = gpu_fe{};
            out_mb[i] = gpu_fe{};
        }
    }

    s_qa[tid] = qa;  s_qb[tid] = qb_;  s_qc[tid] = qc;
    __syncthreads();

    for (uint s = SC_BLOCK >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            s_qa[tid] = device_add(s_qa[tid], s_qa[tid + s]);
            s_qb[tid] = device_add(s_qb[tid], s_qb[tid + s]);
            s_qc[tid] = device_add(s_qc[tid], s_qc[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_qa[blockIdx.x] = s_qa[0];
        out_qb[blockIdx.x] = s_qb[0];
        out_qc[blockIdx.x] = s_qc[0];
    }
}

// GPU sumcheck state
//
// Two ping-pong buffer sets per group, each holding va/vb/ma/mb. Every round
// halves the data, so set 1 only ever needs half of set 0:
//   resident: set0 = n   (the upload), set1 = n/2  -> 6n elements
//   streamed: set0 = n/2, set1 = n/4               -> 3n elements; round 0 reads
//             the host arrays in chunks and writes its n/2 output to set0.
// Streamed is for a monolith whose input layer leaves too little VRAM (opt350m
// len128/256 splits=1, layers 1-2); the fold itself stays on the GPU either way.
// There is no host-fold fallback: if neither layout fits, the run aborts.

enum { VA, VB, MA, MB };
// Pairs per streamed chunk (1 GiB of staging). GKR_SC_STREAM_PAIRS overrides it and
// GKR_SC_FORCE_STREAM=1 streams every group of >= 2 chunks -- test hooks only, to
// check a streamed proof ACCEPTs on a model small enough to run in minutes.
static const uint SC_STREAM_PAIRS = [] {
    const char *e = getenv("GKR_SC_STREAM_PAIRS");
    return e ? (uint)atoi(e) : 1u << 22;
}();
static const bool SC_FORCE_STREAM = getenv("GKR_SC_FORCE_STREAM") != nullptr;

static struct GkrScState {
    Fr_t *buf[2][2][4];       // [group][set][VA..MB]
    size_t cap[2][2];         // elements per array in each set (grow-only)
    int   cur[2];             // set holding the current data; -1 = round 0 still on the host
    const linear_poly *host_v[2], *host_m[2];
    Fr_t *out_q[3];
    uint  max_blocks;
    bool  active[2];
    Fr_t *stage_d[4];         // streamed round 0: device + pinned host chunk
    Fr_t *stage_h[4];
} g_sc;

bool gkr_sc_active_b(int b) { return g_sc.active[b]; }

// Hoisted out of gkr_sc_begin: gkr_sc_active_b is false until sc_begin, which
// runs at the END of init -- too late for the init phases to branch on.
bool gkr_sc_gpu_enabled() {
    static const int v = getenv("GKR_SC_CPU") ? 0 : 1;
    return v != 0;
}

// Deinterleave AoS linear_poly{a,b} -> two SoA arrays (pure data movement, under
// OpenMP since it runs over up to 2^27 elements).
static void deinterleave(const linear_poly* lp, size_t n, Fr_t* ha, Fr_t* hb) {
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < (long)n; ++i) {
        ha[i] = fe_to_gpu(lp[i].a);
        hb[i] = fe_to_gpu(lp[i].b);
    }
}

static void upload_lp(const linear_poly* lp, uint n, Fr_t* d_a, Fr_t* d_b) {
    // Per-call locals. Not `static`: modular-zkp's demo driver proves every partition
    // on its own thread in one process, so a shared static races. Not `thread_local`
    // either: the OpenMP workers would each see their own empty vector.
    std::vector<Fr_t> ha(n), hb(n);
    deinterleave(lp, n, ha.data(), hb.data());
    HIP_CHECK(zkgpt_gpu::gkr_h2d(d_a, ha.data(), n * sizeof(Fr_t)));
    HIP_CHECK(zkgpt_gpu::gkr_h2d(d_b, hb.data(), n * sizeof(Fr_t)));
}

static void free_set(int b, int s) {
    for (int k = 0; k < 4; ++k)
        if (g_sc.buf[b][s][k]) { HIP_CHECK(hipFree(g_sc.buf[b][s][k])); g_sc.buf[b][s][k] = nullptr; }
    g_sc.cap[b][s] = 0;
}

static bool alloc_set(int b, int s, size_t n) {
    if (g_sc.cap[b][s] >= n) return true;
    free_set(b, s);
    for (int k = 0; k < 4; ++k)
        if (hipMalloc(&g_sc.buf[b][s][k], n * sizeof(Fr_t)) != hipSuccess) {
            (void)hipGetLastError();          // clear the sticky OOM
            g_sc.buf[b][s][k] = nullptr;
            free_set(b, s);
            return false;
        }
    g_sc.cap[b][s] = n;
    return true;
}

static void free_stage() {
    for (int k = 0; k < 4; ++k) {
        if (g_sc.stage_d[k]) { HIP_CHECK(hipFree(g_sc.stage_d[k])); g_sc.stage_d[k] = nullptr; }
        if (g_sc.stage_h[k]) { HIP_CHECK(hipHostFree(g_sc.stage_h[k])); g_sc.stage_h[k] = nullptr; }
    }
}

static void free_sc_state() {
    for (int b = 0; b < 2; ++b) {
        free_set(b, 0); free_set(b, 1);
        g_sc.active[b] = false;
    }
    for (Fr_t *&q : g_sc.out_q)
        if (q) { HIP_CHECK(hipFree(q)); q = nullptr; }
    g_sc.max_blocks = 0;
    free_stage();
}

[[noreturn]] static void sc_oom(int b, uint n) {
    size_t fr = 0, tot = 0; (void)hipMemGetInfo(&fr, &tot);
    fprintf(stderr, "[gkr_sc] FATAL: group %d n=%u fits neither resident (%.1f GB) nor streamed "
            "(%.1f GB); free=%.1f GB. No host fallback by design.\n",
            b, n, 6.0 * n * sizeof(Fr_t) / 1e9, 3.0 * n * sizeof(Fr_t) / 1e9, fr / 1e9);
    abort();
}

void gkr_sc_begin(
    linear_poly* v0, linear_poly* m0, uint n0,
    linear_poly* v1, linear_poly* m1, uint n1)
{
    // Default on. GKR_SC_CPU=1 forces the host fold back on.
    if (!gkr_sc_gpu_enabled()) { free_sc_state(); return; }
    double _t0 = _prof_now();
    g_sc.active[0] = g_sc.active[1] = false;

    uint ns[2] = {n0, n1};
    linear_poly* vs[2] = {v0, v1};
    linear_poly* ms[2] = {m0, m1};

    uint max_n = std::max(n0, n1);
    if (max_n / 2 == 0) return;
    {
        char _tag[96];
        snprintf(_tag, sizeof _tag, "gkr_sc_begin n0=%u n1=%u (g_sc buffers)", n0, n1);
        zkgpt_gpu::gkr_vram_log(_tag);
    }

    // The two groups trade sizes between layers (layer 2: n1 = 2^27; layer 1:
    // n0 = 2^27, n1 = 0). With grow-only buffers per group that kept 34 GB idle
    // while group 0 asked for 34 GB more, so swap the sets when that fits better.
    auto fits = [](int g, uint n) { return !n || (g_sc.cap[g][0] >= n && g_sc.cap[g][1] >= n / 2); };
    if (!(fits(0, n0) && fits(1, n1)) && fits(1, n0) && fits(0, n1)) {
        std::swap(g_sc.buf[0], g_sc.buf[1]);
        std::swap(g_sc.cap[0], g_sc.cap[1]);
    }

    // Larger group first; on pressure it may evict the other group's (not yet
    // set up) buffers and the predicate's per-call gate arrays.
    const int order[2] = {n0 >= n1 ? 0 : 1, n0 >= n1 ? 1 : 0};
    bool done[2] = {false, false};
    for (int b : order) {
        const uint n = ns[b];
        if (n < 2) {
            if (n) {                                    // a single element: resident, trivially
                if (!alloc_set(b, 0, 1) || !alloc_set(b, 1, 1)) sc_oom(b, n);
                upload_lp(vs[b], 1, g_sc.buf[b][0][VA], g_sc.buf[b][0][VB]);
                upload_lp(ms[b], 1, g_sc.buf[b][0][MA], g_sc.buf[b][0][MB]);
                g_sc.cur[b] = 0; g_sc.active[b] = true;
            }
            done[b] = true;
            continue;
        }
        auto try_resident = [&] { return alloc_set(b, 0, n) && alloc_set(b, 1, n / 2); };
        auto try_streamed = [&] { return alloc_set(b, 0, n / 2) && alloc_set(b, 1, n / 4); };
        const bool force = SC_FORCE_STREAM && n / 2 >= 2 * SC_STREAM_PAIRS;
        bool resident = !force && try_resident();
        if (!resident) {
            free_set(b, 0); free_set(b, 1);
            if (!done[1 - b]) { free_set(1 - b, 0); free_set(1 - b, 1); }
            gkr_pred_release_gates();
            resident = !force && try_resident();
            if (!resident) {
                free_set(b, 0); free_set(b, 1);
                if (!try_streamed()) sc_oom(b, n);
            }
        }
        if (resident) {
            upload_lp(vs[b], n, g_sc.buf[b][0][VA], g_sc.buf[b][0][VB]);
            upload_lp(ms[b], n, g_sc.buf[b][0][MA], g_sc.buf[b][0][MB]);
            g_sc.cur[b] = 0;
        } else {
            g_sc.host_v[b] = vs[b]; g_sc.host_m[b] = ms[b];
            g_sc.cur[b] = -1;
            fprintf(stderr, "[gkr_sc] group %d n=%u streams round 0 from the host "
                    "(resident %.1f GB does not fit; device holds %.1f GB)\n",
                    b, n, 6.0 * n * sizeof(Fr_t) / 1e9, 3.0 * n * sizeof(Fr_t) / 1e9);
        }
        g_sc.active[b] = true;
        done[b] = true;
    }

    const uint max_blocks_needed = std::max((max_n / 2 + SC_BLOCK - 1) / SC_BLOCK,
                                            g_sc.cur[0] < 0 || g_sc.cur[1] < 0
                                                ? SC_STREAM_PAIRS / SC_BLOCK : 0u);
    if (g_sc.max_blocks < max_blocks_needed) {          // grow-only, as above
        for (Fr_t *&q : g_sc.out_q) {
            if (q) { HIP_CHECK(hipFree(q)); q = nullptr; }
            HIP_CHECK(hipMalloc(&q, max_blocks_needed * sizeof(Fr_t)));
        }
        g_sc.max_blocks = max_blocks_needed;
    }
    if ((g_sc.cur[0] < 0 || g_sc.cur[1] < 0) && !g_sc.stage_d[0])
        for (int k = 0; k < 4; ++k) {
            HIP_CHECK(hipMalloc(&g_sc.stage_d[k], 2 * (size_t)SC_STREAM_PAIRS * sizeof(Fr_t)));
            HIP_CHECK(hipHostMalloc(&g_sc.stage_h[k], 2 * (size_t)SC_STREAM_PAIRS * sizeof(Fr_t)));
        }
    g_t_begin += _prof_now() - _t0; g_n_begin++;
}

// Launch one fold over `pairs` pairs and return its reduced quadratic.
static quadratic_poly fold(Fr_t *const in[4], Fr_t *const out[4], const Fr_t &r_t,
                           uint pairs, uint total_sz, uint base) {
    const uint blocks = (pairs + SC_BLOCK - 1) / SC_BLOCK;
    k_sc_update<<<blocks, SC_BLOCK>>>(
        in[VA], in[VB], in[MA], in[MB], out[VA], out[VB], out[MA], out[MB],
        r_t, g_sc.out_q[0], g_sc.out_q[1], g_sc.out_q[2], pairs, total_sz, base);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Download per-block partial sums and reduce on CPU
    std::vector<Fr_t> ha(blocks), hb(blocks), hc(blocks);
    HIP_CHECK(hipMemcpy(ha.data(), g_sc.out_q[0], blocks * sizeof(Fr_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hb.data(), g_sc.out_q[1], blocks * sizeof(Fr_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hc.data(), g_sc.out_q[2], blocks * sizeof(Fr_t), hipMemcpyDeviceToHost));

    quadratic_poly ret;
    for (uint bl = 0; bl < blocks; ++bl)
        ret = ret + quadratic_poly(fe_from_gpu(ha[bl]), fe_from_gpu(hb[bl]), fe_from_gpu(hc[bl]));
    return ret;
}

quadratic_poly gkr_sc_update_b(int b, const F& r, uint n, uint total_sz)
{
    if (!n || !g_sc.active[b]) return quadratic_poly();
    double _t0 = _prof_now();

    Fr_t r_t    = fe_to_gpu(r);
    uint half_n = n >> 1;
    if (!half_n) return quadratic_poly();

    quadratic_poly ret;
    if (g_sc.cur[b] < 0) {
        // Streamed round 0: chunks of host pairs -> pinned stage -> device stage,
        // folded straight into set 0 at the chunk's offset.
        for (uint p0 = 0; p0 < half_n; p0 += SC_STREAM_PAIRS) {
            const uint pairs = std::min(SC_STREAM_PAIRS, half_n - p0);
            const size_t e0 = 2 * (size_t)p0, ne = 2 * (size_t)pairs;
            deinterleave(g_sc.host_v[b] + e0, ne, g_sc.stage_h[VA], g_sc.stage_h[VB]);
            deinterleave(g_sc.host_m[b] + e0, ne, g_sc.stage_h[MA], g_sc.stage_h[MB]);
            for (int k = 0; k < 4; ++k)
                HIP_CHECK(hipMemcpy(g_sc.stage_d[k], g_sc.stage_h[k], ne * sizeof(Fr_t),
                                    hipMemcpyHostToDevice));
            Fr_t *out[4];
            for (int k = 0; k < 4; ++k) out[k] = g_sc.buf[b][0][k] + p0;
            ret = ret + fold(g_sc.stage_d, out, r_t, pairs, total_sz, p0);
        }
        g_sc.cur[b] = 0;
    } else {
        const int cur = g_sc.cur[b], nxt = 1 - cur;
        ret = fold(g_sc.buf[b][cur], g_sc.buf[b][nxt], r_t, half_n, total_sz, 0);
        g_sc.cur[b] = nxt;
    }
    g_t_update += _prof_now() - _t0; g_n_update++;
    return ret;
}

void gkr_sc_handle_single(int b, F& add_term, const F& r)
{
    if (!g_sc.active[b]) return;

    int cur = g_sc.cur[b];
    Fr_t ha, hb, hma, hmb;
    HIP_CHECK(hipMemcpy(&ha,  g_sc.buf[b][cur][VA], sizeof(Fr_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&hb,  g_sc.buf[b][cur][VB], sizeof(Fr_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&hma, g_sc.buf[b][cur][MA], sizeof(Fr_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&hmb, g_sc.buf[b][cur][MB], sizeof(Fr_t), hipMemcpyDeviceToHost));

    F v_a = fe_from_gpu(ha), v_b = fe_from_gpu(hb);
    F m_a = fe_from_gpu(hma), m_b = fe_from_gpu(hmb);

    F ev = v_a * r + v_b;
    F em = m_a * r + m_b;
    add_term = add_term + ev * em;

    // Write back as constant polynomial: a=0, b=eval
    int nxt = 1 - cur;
    Fr_t ev_t = fe_to_gpu(ev), em_t = fe_to_gpu(em);
    F zero_f; zero_f.clear();
    Fr_t z_t = fe_to_gpu(zero_f);
    HIP_CHECK(hipMemcpy(g_sc.buf[b][nxt][VA], &z_t,  sizeof(Fr_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(g_sc.buf[b][nxt][VB], &ev_t, sizeof(Fr_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(g_sc.buf[b][nxt][MA], &z_t,  sizeof(Fr_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(g_sc.buf[b][nxt][MB], &em_t, sizeof(Fr_t), hipMemcpyHostToDevice));
    g_sc.cur[b] = nxt;
}

void gkr_sc_end(
    linear_poly& v0, linear_poly& m0,
    linear_poly& v1, linear_poly& m1)
{
    auto dl_first = [](int b, linear_poly& ov, linear_poly& om) {
        if (!g_sc.active[b]) { ov.clear(); om.clear(); return; }
        int cur = g_sc.cur[b];
        Fr_t ha, hb, hma, hmb;
        HIP_CHECK(hipMemcpy(&ha,  g_sc.buf[b][cur][VA], sizeof(Fr_t), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(&hb,  g_sc.buf[b][cur][VB], sizeof(Fr_t), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(&hma, g_sc.buf[b][cur][MA], sizeof(Fr_t), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(&hmb, g_sc.buf[b][cur][MB], sizeof(Fr_t), hipMemcpyDeviceToHost));
        ov.a = fe_from_gpu(ha); ov.b = fe_from_gpu(hb);
        om.a = fe_from_gpu(hma); om.b = fe_from_gpu(hmb);
    };
    dl_first(0, v0, m0);
    dl_first(1, v1, m1);

    // Deactivate only -- do NOT free: the buffers are grow-only across layers and
    // live until gkr_sc_teardown(). active[] must still clear -- it gates the GPU
    // branch, so a layer that never ran begin must not read stale device state.
    g_sc.active[0] = g_sc.active[1] = false;
}

// Release the sumcheck device buffers. Call once when the proof is done; gkr_sc_end
// deliberately does not, so the grow-only allocation survives across layers.
void gkr_sc_teardown() { free_sc_state(); }

// The predicate runs between sumcheck phases, when no group is active; it calls
// this when its own allocation does not fit. The next gkr_sc_begin re-allocates.
bool gkr_sc_release_idle() {
    if (g_sc.active[0] || g_sc.active[1]) return false;
    free_sc_state();
    return true;
}
