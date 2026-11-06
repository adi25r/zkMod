#ifndef ZKGPT_GPU_MATMUL_PROVER_HPP
#define ZKGPT_GPU_MATMUL_PROVER_HPP
/*
 * GPU-accelerated matmul product sumcheck prover
 * In zkGPT, this product exists in sum_check_product, which proves sum_x f(x) * g(x) = ans over m+1 bools
 * Uses BlendySC, using product tables to change how efficient the work is
 * Challenges are provided by zkGPT, 
*/
#include <hip/hip_runtime.h>
#include <utility>
#include "config.hpp"
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <unordered_map>
#include <thread>
#include <sched.h>
#include <omp.h>
#include <algorithm>
#include <cstdint>
#include <algorithm>

#include "gpu_field.hpp"
#include "blendysc.hpp"
#include "gpu_uvm_alloc.hpp"
#include <mcl/bn.hpp>

namespace zkgpt_gpu {

// 4 components of blendy are table expansion, ptable building, the round time, and then the time to fold
struct MatmulPhaseTimers {
    double expand = 0;   // extrapolating for x = 2
    double ptable = 0;   // build product table
    double round = 0;   // per-round p0/p1/p2 eval + D2H reduce
    double fold = 0;   // fold ptable + fold bookkeeping arrays
    // Closing counters. The four buckets above cover only the kernel brackets, so
    // they cannot say whether the residual is launch overhead or host work. These
    // three make the accounting add up:
    //   host_wall - sc_wall            = host prep outside the sumcheck (staging,
    //                                    allocation, the dense-path conversion)
    //   sc_wall - (buckets + bring)    = launch/sync/alloc overhead INSIDE
    //                                    product_sumcheck
    // None of them adds a sync: each brackets a region that already ends in one.
    double sc_wall = 0;    // whole product_sumcheck() body
    double host_wall = 0;  // whole gpu_product_sumcheck_host() body
    double bring = 0;      // sc_src_stream::bring (host convert + H2D), both call sites
    // The stage-boundary fold's bring() sits INSIDE the `fold` bracket, while the
    // expand-side one sits outside every bracket. Without this split, `fold+bring`
    // double-counts the former and understates the residual.
    double bring_fold = 0;
    bool   on = false;
    void reset() { expand = ptable = round = fold = sc_wall = host_wall = bring = bring_fold = 0; }
};
inline MatmulPhaseTimers g_matmul_timers;
using mm_clk = std::chrono::high_resolution_clock;
inline double mm_secs(mm_clk::time_point t0) {
    return std::chrono::duration<double>(mm_clk::now() - t0).count();
}

// The hipStreamSynchronize calls inside product_sumcheck constrain only the HOST --
// one stream already orders the kernels. They are kept deliberately; removing them
// measured as no win.

// Accumulates a whole-function wall time into one MatmulPhaseTimers field on scope
// exit, so the early `std::exit` on REJECT and the multiple returns cannot skip it.
struct mm_scope_timer {
    double* dst; mm_clk::time_point t0;
    explicit mm_scope_timer(double* d) : dst(d), t0(mm_clk::now()) {}
    ~mm_scope_timer() { if (g_matmul_timers.on) *dst += mm_secs(t0); }
};

inline auto reduce_local(const gpu_fe* d_arr, int nb) {
    std::vector<gpu_fe> h(nb);
    HIP_CHECK(hipMemcpy(h.data(), d_arr, nb*sizeof(gpu_fe), hipMemcpyDeviceToHost));
    FieldElement acc; acc.clear();
    for (auto& v : h) acc += fe_from_gpu(v);
    return acc;
};

// ---------------------------------------------------------------------------
// Stage-1 source streaming (GKR_SC_STREAM_BITS).
//
// The stage-1 source is read twice per stage and never written, so a bounded
// window re-uploaded per tile suffices: O(window) device memory instead of
// O(2^n), at the price of streaming it twice (~2x PCIe bytes + fe_to_gpu).
// ONLY STAGE 1 STREAMS -- from stage 2 the source is the previous fold's output,
// already device-resident.
//
// Sparse upload (default on; GKR_SC_SPARSE=0 restores the dense copy). The Lasso
// sources are mostly zero, yet the dense path moves all of them over PCIe per
// partition, pageable, which degrades badly when many partitions share the GPU (the
// CPU path skips zero pairs and never pays it). alloc() scans both arrays ONCE
// (threaded) into a per-4096-block CSR of (global index, raw 32-byte value); bring()
// then uploads only the window's nonzeros and scatters them into a zeroed device
// window, so the kernels see exactly the bytes the dense copy would have produced.
static __global__ void sc_sparse_scatter_kernel(gpu_fe* dst, const uint32_t* idx, const gpu_fe* val,
                                         long m, uint32_t off) {
    const long j = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (j < m) dst[idx[j] - off] = val[j];
}

struct sc_sparse_src {
    static constexpr int BLK_BITS = 12;            // index granularity, elements
    std::vector<long> ptr;                         // nblk+1 offsets into idx/val
    std::vector<uint32_t> idx;                     // global element index (N <= 2^32)
    std::vector<gpu_fe> val;                       // raw value bytes, same layout as the source
    long max_win_nnz = 0;                          // most nonzeros in any win-aligned window

    // Threads: the caller's affinity mask (a pinned child sees its own block of cores),
    // capped by its OpenMP budget (a split child's share of the thread pool).
    static int host_threads() {
        cpu_set_t cs; CPU_ZERO(&cs);
        int t = sched_getaffinity(0, sizeof cs, &cs) == 0 ? CPU_COUNT(&cs) : 16;
        return std::max(1, std::min({t, 64, omp_get_max_threads()}));
    }
    void build(const FieldElement* h, long N, long win) {
        const long nblk = N >> BLK_BITS;
        ptr.assign(nblk + 1, 0);
        const int T = (int)std::min<long>(host_threads(), std::max(1L, nblk));
        auto par = [&](auto&& body) {
            std::vector<std::thread> th;
            for (int t = 0; t < T; ++t)
                th.emplace_back([&, t] { for (long b = nblk * t / T; b < nblk * (t + 1) / T; ++b) body(b); });
            for (auto& x : th) x.join();
        };
        par([&](long b) {
            long c = 0;
            for (long k = b << BLK_BITS, e = (b + 1) << BLK_BITS; k < e; ++k) c += !h[k].isZero();
            ptr[b + 1] = c;
        });
        for (long b = 0; b < nblk; ++b) ptr[b + 1] += ptr[b];
        idx.resize(ptr[nblk]); val.resize(ptr[nblk]);
        par([&](long b) {
            long o = ptr[b];
            for (long k = b << BLK_BITS, e = (b + 1) << BLK_BITS; k < e; ++k)
                if (!h[k].isZero()) {
                    idx[o] = (uint32_t)k;
                    std::memcpy(&val[o], &h[k], sizeof(gpu_fe));
                    ++o;
                }
        });
        const long bpw = std::max(1L, win >> BLK_BITS);
        for (long b = 0; b < nblk; b += bpw)
            max_win_nnz = std::max(max_win_nnz, ptr[std::min(nblk, b + bpw)] - ptr[b]);
    }
    void free() { std::vector<long>().swap(ptr); std::vector<uint32_t>().swap(idx); std::vector<gpu_fe>().swap(val); }
};

struct sc_src_stream {
    const FieldElement* hA = nullptr;   // host source A (lasso_mult_v)
    const FieldElement* hB = nullptr;   // host source B (val[0], the witness)
    gpu_fe* dA = nullptr;               // device window A, `win` elements
    gpu_fe* dB = nullptr;               // device window B
    long win = 0;                       // window length in elements (power of two)
    double t_conv = 0.0, t_h2d = 0.0;   // accumulated, reported under GKR_DUMP_SIZES
    long uploads = 0, elems = 0;

    // NOT pinned, deliberately. Uploading in place means the DMA reads PAGEABLE
    // memory, so the driver bounces it (slower than pinned DMA). A pinned staging
    // path is worse overall: the host memcpy into pinned memory costs more than the
    // DMA speedup it buys, so the slow-looking DMA is the right trade.
    bool active() const { return hA != nullptr && win > 0; }

    int win_bits = 0;                   // log2(win)

    // Sparse upload state (see sc_sparse_src): host CSR + device staging per source.
    bool sparse = false;
    sc_sparse_src spA, spB;
    uint32_t *dIdxA = nullptr, *dIdxB = nullptr;
    gpu_fe *dValA = nullptr, *dValB = nullptr;
    double t_sparse_build = 0.0;
    long sparse_bytes = 0;

    void alloc(const FieldElement* a, const FieldElement* b, long window, long N = 0) {
        hA = a; hB = b; win = window;
        win_bits = 0; while (((long)1 << win_bits) < win) ++win_bits;
        HIP_CHECK(GKR_DMALLOC(&dA, win * sizeof(gpu_fe)));
        HIP_CHECK(GKR_DMALLOC(&dB, win * sizeof(gpu_fe)));
        const char* e = getenv("GKR_SC_SPARSE");
        const long blk = 1L << sc_sparse_src::BLK_BITS;
        sparse = !(e && e[0] == '0') && N > 0 && N <= (1L << 32) && N % blk == 0 && win % blk == 0;
        if (sparse) {
            const auto t0 = mm_clk::now();
            spA.build(a, N, win);
            spB.build(b, N, win);
            t_sparse_build = mm_secs(t0);
            if (spA.max_win_nnz) {
                HIP_CHECK(GKR_DMALLOC(&dIdxA, spA.max_win_nnz * sizeof(uint32_t)));
                HIP_CHECK(GKR_DMALLOC(&dValA, spA.max_win_nnz * sizeof(gpu_fe)));
            }
            if (spB.max_win_nnz) {
                HIP_CHECK(GKR_DMALLOC(&dIdxB, spB.max_win_nnz * sizeof(uint32_t)));
                HIP_CHECK(GKR_DMALLOC(&dValB, spB.max_win_nnz * sizeof(gpu_fe)));
            }
        }
    }
    void free() {
        if (dA) { HIP_CHECK(hipFree(dA)); dA = nullptr; }
        if (dB) { HIP_CHECK(hipFree(dB)); dB = nullptr; }
        for (void** p : {(void**)&dIdxA, (void**)&dIdxB, (void**)&dValA, (void**)&dValB})
            if (*p) { HIP_CHECK(hipFree(*p)); *p = nullptr; }
        spA.free(); spB.free();
    }

    // Sparse bring of one source: zero the window, upload [off, off+cnt)'s nonzeros,
    // scatter. Range must be block-aligned (checked by the caller).
    void bring_sparse(const sc_sparse_src& sp, gpu_fe* d, uint32_t* dIdx, gpu_fe* dVal,
                      long off, long cnt, hipStream_t s) {
        const long b0 = off >> sc_sparse_src::BLK_BITS, b1 = (off + cnt) >> sc_sparse_src::BLK_BITS;
        const long lo = sp.ptr[b0], m = sp.ptr[b1] - lo;
        HIP_CHECK(hipMemsetAsync(d, 0, cnt * sizeof(gpu_fe), s));
        if (m == 0) return;
        HIP_CHECK(hipMemcpyAsync(dIdx, sp.idx.data() + lo, m * sizeof(uint32_t), hipMemcpyHostToDevice, s));
        HIP_CHECK(hipMemcpyAsync(dVal, sp.val.data() + lo, m * sizeof(gpu_fe), hipMemcpyHostToDevice, s));
        hipLaunchKernelGGL(sc_sparse_scatter_kernel, dim3((unsigned)((m + 255) / 256)), dim3(256), 0, s,
                           d, dIdx, dVal, m, (uint32_t)off);
        sparse_bytes += m * (long)(sizeof(uint32_t) + sizeof(gpu_fe));
        gkr_xfer_add_h2d(m * (sizeof(uint32_t) + sizeof(gpu_fe)));
    }

    // Stage source[off .. off+cnt) of BOTH buffers into dA/dB. cnt <= win.
    //
    // No host conversion and no staging buffer: FieldElement and gpu_fe are the same
    // size and layout, so the mcl arrays are already the device representation and the
    // copy is issued straight from them. The static_assert is what makes the
    // reinterpret_cast safe -- if the sizes ever diverge this fails to compile rather
    // than uploading garbage.
    void bring(long off, long cnt, hipStream_t s) {
        static_assert(sizeof(FieldElement) == sizeof(gpu_fe),
                      "sc_src_stream uploads the mcl array directly; a size mismatch "
                      "would make the reinterpret_cast below a stride error");
        const auto t0 = mm_clk::now();
        const long blk = 1L << sc_sparse_src::BLK_BITS;
        if (sparse && off % blk == 0 && cnt % blk == 0) {
            bring_sparse(spA, dA, dIdxA, dValA, off, cnt, s);
            bring_sparse(spB, dB, dIdxB, dValB, off, cnt, s);
            HIP_CHECK(hipStreamSynchronize(s));
            t_h2d += mm_secs(t0);
            if (g_matmul_timers.on) g_matmul_timers.bring += mm_secs(t0);
            ++uploads; elems += cnt;
            return;
        }
        HIP_CHECK(hipMemcpyAsync(dA, reinterpret_cast<const gpu_fe*>(hA + off),
                                 cnt * sizeof(gpu_fe), hipMemcpyHostToDevice, s));
        HIP_CHECK(hipMemcpyAsync(dB, reinterpret_cast<const gpu_fe*>(hB + off),
                                 cnt * sizeof(gpu_fe), hipMemcpyHostToDevice, s));
        // The source is the caller's mcl array, which product_sumcheck never writes,
        // so nothing races; this sync is plain stream ordering.
        HIP_CHECK(hipStreamSynchronize(s));
        gkr_xfer_add_h2d(2 * (size_t)cnt * sizeof(gpu_fe));   // dA + dB
        t_h2d += mm_secs(t0);
        if (g_matmul_timers.on) g_matmul_timers.bring += mm_secs(t0);
        ++uploads; elems += cnt;
    }
};

// Staged produt sumcheck, external challenges come from regular zkGPT workflow
// l_stage is n / k, where l = n is a single stage and l = 1 is n stages
static std::pair<FieldElement, FieldElement> product_sumcheck(gpu_fe* bkA_d, gpu_fe* bkB_d, long N, int n, FieldElement gamma, const FieldElement* rho, int l_stage, int threads, hipStream_t stream = 0, sc_src_stream* src = nullptr) {
    mm_scope_timer _scw(&g_matmul_timers.sc_wall);
    if (l_stage <= 0 || l_stage > n) l_stage = n;

    // When streaming, bkA_d/bkB_d are null for stage 1: the source lives on the
    // host and arrives a window at a time. The caller decides feasibility (see
    // gpu_product_sumcheck_host) and only passes `src` when it holds.
    const bool streaming = (src && src->active());

    // Ext-table tiling (Fix B): cap the materialized future-variable (b3) window
    // at 2^tile_bits combos and build the ptable in tiles. -1 disables (full
    // window = original behavior). Env GKR_SC_TILE_BITS overrides the config.
    int tile_bits = gkr_config.sc_tile_bits;
    if (const char* e = getenv("GKR_SC_TILE_BITS")) tile_bits = atoi(e);

    // Buffers sized for the largest stage (first stage: l=l_stage, future_vars=fv0).
    const int ptable_cap = pow3(l_stage);
    const int fv0 = (n - l_stage > 0) ? (n - l_stage) : 0;   // num future variables
    // Materialized window per tile: min(fv0, tile_bits) when tiling, else fv0.
    const int fv_cap = (tile_bits >= 0 && tile_bits < fv0) ? tile_bits : fv0;
    const long ext_cap = (long)ptable_cap << fv_cap; // 3^l * 2^min(n-l, tile_bits)

    gpu_fe *d_bkA_dst, *d_bkB_dst;
    gpu_fe *d_bkA_ext, *d_bkB_ext;
    gpu_fe *d_ptable_src, *d_ptable_dst;
    gpu_fe *d_p0, *d_p1, *d_p2;
    gpu_fe *d_partials;

    const int PTABLE_TARGET_BLOCKS = ptable_target_blocks();
    const int partials_cap = ptable_cap + PTABLE_TARGET_BLOCKS;

    if (getenv("GKR_DUMP_SIZES")) {
        size_t fr = 0, tot = 0; hipMemGetInfo(&fr, &tot);
        // 2*ext + 2N (caller bkA/bkB sources) + 2*dst + 2*ptable + eq (1<<l_stage).
        const double dcap = (double)std::max(1L, (long)N >> (streaming ? l_stage : 1));
        // Streaming replaces the 2N resident source with 2*window, and needs a
        // SECOND fold destination (N>>2*l_stage) because the ping-pong can no
        // longer bounce off the caller's N-sized buffer -- there isn't one.
        const double dcap2 = (double)std::max(1L, (long)N >> (2 * l_stage));
        const double srcterm = streaming ? (2.0*(double)src->win + 2.0*dcap2) : (2.0*(double)N);
        double need = (2.0*ext_cap + srcterm + 2.0*dcap + 2.0*ptable_cap
                       + (double)((long)1 << l_stage)) * sizeof(gpu_fe) / 1073741824.0;
        fprintf(stderr, "[mm] n=%d l_stage=%d N=%ld ptable_cap=%d ext_cap=%ld tile_bits=%d fv_cap=%d | "
                        "stream=%s | need=%.2f GB | free=%.2f/%.2f GB\n",
                n, l_stage, N, ptable_cap, ext_cap, tile_bits, fv_cap,
                streaming ? "on" : "off", need, fr / 1073741824.0, tot / 1073741824.0);
    }
    // Fold destinations. Sized at a FLAT N/2, independent of l_stage/k.
    //
    // A combined eq-weighted stage fold (stage_fold_bk_kernel) would let dst shrink to
    // N>>l_stage, but that term grows with k while the extension-table term shrinks
    // with k, so total memory would be non-monotone in k (measured on opt2b7, t=18:
    // k=8/16/32 -> 38.0/42.9/50.7 GiB). Pinning dst at N/2 leaves total memory governed
    // by the ext-table term alone, monotone in k. The stage-boundary fold below uses l
    // SEQUENTIAL single-variable folds (k=0 LSB-first each round, as
    // gpu_bookkeeping_fold does) through one flat (N/2)-sized pair.
    // Streaming: stage 1's combined fold writes N>>l_stage here and later folds
    // ping-pong with dst2, so N>>l_stage is the high-water mark.
    const long dst_cap = std::max(1L, (long)N >> (streaming ? l_stage : 1));
    HIP_CHECK(GKR_DMALLOC(&d_bkA_dst, dst_cap*sizeof(gpu_fe)));
    HIP_CHECK(GKR_DMALLOC(&d_bkB_dst, dst_cap*sizeof(gpu_fe)));
    // Second destination pair, streaming only. In the dense path the ping-pong
    // runs between the caller's N-capacity source buffer and d_bk*_dst; when the
    // source is streamed that buffer does not exist, so stage 2 needs somewhere
    // to write.
    //
    // Sized N>>(l_stage+1), NOT N>>(2*l_stage). Stages 2+ are l SEQUENTIAL
    // single-variable folds (the `else` branch below) that ping-pong through this
    // pair: stage 2 enters with cur = N>>l_stage and its FIRST fold writes
    // cur>>1 = N>>(l_stage+1) elements here. N>>(2*l_stage) is smaller for any
    // l_stage > 1, so sizing by it overruns the buffer (GPU memory access fault).
    // Every later write is strictly smaller, so this is the high-water mark.
    gpu_fe *d_bkA_dst2 = nullptr, *d_bkB_dst2 = nullptr;
    if (streaming) {
        const long dst_cap2 = std::max(1L, (long)N >> (l_stage + 1));
        HIP_CHECK(GKR_DMALLOC(&d_bkA_dst2, dst_cap2*sizeof(gpu_fe)));
        HIP_CHECK(GKR_DMALLOC(&d_bkB_dst2, dst_cap2*sizeof(gpu_fe)));
    }
    // eq table for the combined stage fold: 1<<l_stage entries (sized for the
    // largest stage; later stages use a 1<<l prefix). Rebuilt on host per stage.
    gpu_fe* d_eq;
    HIP_CHECK(GKR_DMALLOC(&d_eq, ((long)1 << l_stage) * sizeof(gpu_fe)));
    std::vector<gpu_fe> h_eq((size_t)1 << l_stage);
    gpu_fe* orig_d_bkA_dst = d_bkA_dst;
    gpu_fe* orig_d_bkB_dst = d_bkB_dst;
    HIP_CHECK(GKR_DMALLOC(&d_bkA_ext, ext_cap*sizeof(gpu_fe)));
    HIP_CHECK(GKR_DMALLOC(&d_bkB_ext, ext_cap*sizeof(gpu_fe)));
    HIP_CHECK(GKR_DMALLOC(&d_ptable_src, ptable_cap*sizeof(gpu_fe)));
    HIP_CHECK(GKR_DMALLOC(&d_ptable_dst, ptable_cap*sizeof(gpu_fe)));
    HIP_CHECK(GKR_DMALLOC(&d_partials, partials_cap*sizeof(gpu_fe)));
    int round_max_blocks = std::max(1, ((1 << (l_stage - 1)) + threads - 1) / threads);
    HIP_CHECK(GKR_DMALLOC(&d_p0, round_max_blocks*sizeof(gpu_fe)));
    HIP_CHECK(GKR_DMALLOC(&d_p1, round_max_blocks*sizeof(gpu_fe)));
    HIP_CHECK(GKR_DMALLOC(&d_p2, round_max_blocks*sizeof(gpu_fe)));

    gpu_fe mont_TWO = fe_to_gpu(to_FE(2));

    FieldElement claim = gamma;
    bool ok = true;
    int processed = 0;       // variables already bound
    int global_round = 0;    // 0 .. n-1

    while (processed < n) {
        const int l  { std::min(l_stage, n - processed) };
        const int pow3_l  { pow3(l) };
        const int future_vars { n - processed - l };             // bits after this stage

        // EXPAND + PTABLE, tiled over the future-variable (b3) axis.
        // Split future_vars into (num_tiles_bits | tile_lo): each tile materializes
        // 2^tile_lo of the 2^future_vars combos, so peak ext = 2*3^l*2^tile_lo. The
        // expand/extrapolate kernels are per-b3 independent and build_ptable is a
        // plain sum over b3, so tiling is exact (zero recompute). Tiling disabled
        // (tile_bits<0) => tile_lo=future_vars, a single tile == the original path.
        mm_clk::time_point tp;
        int tile_lo_v = (tile_bits >= 0 && tile_bits < future_vars) ? tile_bits : future_vars;
        // Stage 1 is the only stage whose source is 2^n; later stages read the
        // previous stage's fold output, already resident and 2^l_stage smaller.
        const bool stream1 = streaming && processed == 0;
        if (stream1) {
            // A tile's expand slab is staged as one streaming window, so the tile
            // cannot outgrow it. Exact: more, smaller tiles only -- the ptable is a
            // plain sum over b3. Reported, since the effective t is then not the
            // requested t.
            const int cap = src->win_bits - l;
            if (tile_lo_v > cap) {
                static bool warned = false;
                if (!warned) {
                    fprintf(stderr, "[mm] tile_bits %d -> %d (streaming window %d, l=%d)\n",
                            tile_lo_v, std::max(0, cap), src->win_bits, l);
                    warned = true;
                }
                tile_lo_v = std::max(0, cap);
            }
        }
        const int tile_lo = tile_lo_v;
        const int num_tiles_bits = future_vars - tile_lo;          // outer loop = 2^this
        const long tile_b3 = (long)1 << tile_lo;                   // combos per tile
        const long tile_cells = (long)pow3_l << tile_lo;           // ext cells per tile

        // Zero the ptable accumulator once; each tile adds its partial sum into it.
        if (g_matmul_timers.on) tp = mm_clk::now();
        HIP_CHECK(hipMemsetAsync(d_ptable_src, 0, pow3_l*sizeof(gpu_fe), stream));
        if (g_matmul_timers.on) { HIP_CHECK(hipStreamSynchronize(stream)); g_matmul_timers.ptable += mm_secs(tp); }

        const int b_exp = (int)((((long)1 << l) << tile_lo) + threads - 1) / threads;
        const int b_ext = (int)((tile_cells + threads - 1) / threads);
        const int b_pt  = (pow3_l + threads - 1) / threads;

        for (long ti = 0; ti < ((long)1 << num_tiles_bits); ++ti) {
            // This tile covers global b3 in [ti<<tile_lo, (ti+1)<<tile_lo). In the
            // binary bk the index is (b3<<l)|b2_bin, so the input slab starts at
            // (ti<<tile_lo)<<l; expand then uses tile-local b3 (future_vars=tile_lo).
            const long in_off = (ti << tile_lo) << l;

            // READ #1 of the source. Streaming just changes where the slab comes
            // from: the window holds exactly source[in_off, in_off+slab), so the
            // kernel sees the same contiguous layout it would at bkA_d + in_off.
            const gpu_fe* srcA = bkA_d + in_off;
            const gpu_fe* srcB = bkB_d + in_off;
            if (stream1) {
                src->bring(in_off, (long)tile_b3 << l, stream);
                srcA = src->dA; srcB = src->dB;
            }

            // EXPAND (into this tile's ext slab)
            if (g_matmul_timers.on) tp = mm_clk::now();
            HIP_CHECK(hipMemsetAsync(d_bkA_ext, 0, tile_cells*sizeof(gpu_fe), stream));
            HIP_CHECK(hipMemsetAsync(d_bkB_ext, 0, tile_cells*sizeof(gpu_fe), stream));
            hipLaunchKernelGGL(expand_bk_kernel, dim3(b_exp), dim3(threads), 0, stream, srcA, d_bkA_ext, l, tile_lo, pow3_l);
            hipLaunchKernelGGL(expand_bk_kernel, dim3(b_exp), dim3(threads), 0, stream, srcB, d_bkB_ext, l, tile_lo, pow3_l);
            HIP_CHECK(hipStreamSynchronize(stream));

            // extrapolate the x=2 cells, one variable at a time
            for (int var = 0; var < l; ++var) {
                hipLaunchKernelGGL(extrapolate_bk_var_kernel, dim3(b_ext), dim3(threads), 0, stream, d_bkA_ext, var, l, tile_lo, mont_TWO, pow3_l);
                hipLaunchKernelGGL(extrapolate_bk_var_kernel, dim3(b_ext), dim3(threads), 0, stream, d_bkB_ext, var, l, tile_lo, mont_TWO, pow3_l);
                HIP_CHECK(hipStreamSynchronize(stream));
            }
            if (g_matmul_timers.on) g_matmul_timers.expand += mm_secs(tp);

            // PTABLE: two-level segmented reduction over b3 into d_ptable_src.
            if (g_matmul_timers.on) tp = mm_clk::now();
            int nb3_blocks = (pow3_l >= PTABLE_TARGET_BLOCKS)
                                 ? 1 : (PTABLE_TARGET_BLOCKS + pow3_l - 1) / pow3_l;
            long max_b3_blocks = (tile_b3 + threads - 1) / threads;
            if ((long)nb3_blocks > max_b3_blocks) nb3_blocks = (int)std::max(1L, max_b3_blocks);
            size_t pt_smem = (size_t)threads * sizeof(gpu_fe);
            hipLaunchKernelGGL(ptable_partial_kernel, dim3(pow3_l, nb3_blocks), dim3(threads), pt_smem, stream,
                d_bkA_ext, d_bkB_ext, d_partials, pow3_l, tile_b3, nb3_blocks);
            hipLaunchKernelGGL(ptable_finalize_kernel, dim3(b_pt), dim3(threads), 0, stream,
                d_partials, d_ptable_src, pow3_l, nb3_blocks);
            HIP_CHECK(hipStreamSynchronize(stream));
            if (g_matmul_timers.on) g_matmul_timers.ptable += mm_secs(tp);
        }

        int cur_ptable_size = pow3_l;
        int remaining_free  = l;
        for (int r_in = 0; r_in < l; ++r_in) {
            // ROUND COMPUTATION
            if (g_matmul_timers.on) tp = mm_clk::now();
            int future_combos = 1 << (remaining_free - 1);
            int rb = std::max(1, (future_combos + threads - 1) / threads);
            size_t smem = 3 * threads * sizeof(gpu_fe);
            hipLaunchKernelGGL(blendy_round_kernel, dim3(rb), dim3(threads), smem, stream,
                d_ptable_src, d_p0, d_p1, d_p2, remaining_free, future_combos);
            HIP_CHECK(hipStreamSynchronize(stream));

            FieldElement p0 = reduce_local(d_p0, rb);
            FieldElement p1 = reduce_local(d_p1, rb);
            FieldElement p2 = reduce_local(d_p2, rb);

            if (p0 + p1 != claim) ok = false;

            FieldElement r_i = rho[global_round];
            FieldElement lag0, lag1, lag2;
            compute_lagrange_at(r_i, lag0, lag1, lag2);
            claim = lag0 * p0 + lag1 * p1 + lag2 * p2;

            gpu_fe gL0 = fe_to_gpu(lag0), gL1 = fe_to_gpu(lag1), gL2 = fe_to_gpu(lag2);
            if (g_matmul_timers.on) g_matmul_timers.round += mm_secs(tp);

            // FOLDING
            if (g_matmul_timers.on) tp = mm_clk::now();
            int ns = cur_ptable_size / 3;
            if (ns > 0) {
                int fb = std::max(1, (ns + threads - 1) / threads);
                hipLaunchKernelGGL(fold_ptable_kernel, dim3(fb), dim3(threads), 0, stream, d_ptable_src, d_ptable_dst, gL0, gL1, gL2, ns, ns);
                std::swap(d_ptable_src, d_ptable_dst);
            }
            cur_ptable_size = ns;

            // The bk fold is deferred to the stage boundary (below): within a stage the
            // round computation reads only the ptable (blendy_round_kernel), never bkA/bkB.
            if (g_matmul_timers.on) g_matmul_timers.fold += mm_secs(tp);
            --remaining_free;
            ++global_round;
        }

        // STAGE-BOUNDARY bk fold: bind this stage's l variables (rho[processed ..
        // processed+l-1]). Non-streaming: l SEQUENTIAL single-variable folds through
        // a flat (N/2)-sized ping-pong pair -- see dst_cap's comment above for why
        // (it keeps the fold's memory out of the k-dependence). Streaming keeps the one-shot
        // combined eq-weighted fold (stage_fold_bk_kernel), since re-streaming the
        // host source l times would multiply PCIe traffic instead of saving memory.
        {
            if (g_matmul_timers.on) tp = mm_clk::now();
            if (stream1) {
                const int nb = 1 << l;
                for (int b = 0; b < nb; ++b) {
                    FieldElement e = to_FE(1);
                    for (int i = 0; i < l; ++i) {
                        FieldElement ri = rho[processed + i];
                        e = e * (((b >> i) & 1) ? ri : (to_FE(1) - ri));
                    }
                    h_eq[b] = fe_to_gpu(e);
                }
                HIP_CHECK(hipMemcpyAsync(d_eq, h_eq.data(), nb*sizeof(gpu_fe),
                                         hipMemcpyHostToDevice, stream));
                gkr_xfer_add_h2d((size_t)nb * sizeof(gpu_fe));
                // READ #2 of the source, and the reason it must be uploaded twice:
                // this fold needs rho[processed .. processed+l-1], which only
                // exists after the round loop above, so it cannot be fused into
                // the expand pass. stage_fold maps source[c, c+C) to exactly
                // output[c>>l, (c+C)>>l) with no accumulation, so chunking it is
                // a pure partition -- C need only be a multiple of 2^l, which a
                // power-of-two window >= 2^l_stage always is.
                for (long c = 0; c < (long)N; c += src->win) {
                    const long cnt = std::min<long>(src->win, (long)N - c);
                    const double _b0 = g_matmul_timers.bring;
                    src->bring(c, cnt, stream);
                    if (g_matmul_timers.on)
                        g_matmul_timers.bring_fold += g_matmul_timers.bring - _b0;
                    const long oe = cnt >> l;
                    const int cfb = (int)std::max(1L, (oe + threads - 1) / threads);
                    hipLaunchKernelGGL(stage_fold_bk_kernel, dim3(cfb), dim3(threads), 0, stream,
                        src->dA, d_eq, d_bkA_dst + (c >> l), l, oe);
                    hipLaunchKernelGGL(stage_fold_bk_kernel, dim3(cfb), dim3(threads), 0, stream,
                        src->dB, d_eq, d_bkB_dst + (c >> l), l, oe);
                    HIP_CHECK(hipStreamSynchronize(stream));
                }
                // No swap: there is no N-sized buffer to bounce off. Stage 1's
                // output becomes the source, and the second (smaller) pair takes
                // over as the destination from here on.
                bkA_d = d_bkA_dst; bkB_d = d_bkB_dst;
                d_bkA_dst = d_bkA_dst2; d_bkB_dst = d_bkB_dst2;
            } else {
                // l sequential single-variable folds, LSB-first (k=0 always: after
                // each fold the next original bit shifts into position 0 -- the
                // same convention blendysc.hpp documents for this idiom, and the
                // one gpu_bookkeeping_fold already uses in this file). Exactly
                // equal to the combined eq-weighted fold by associativity (each
                // round binds one more of rho[processed..processed+l-1], in
                // order), but never needs more than the flat (N/2)-sized dst_cap
                // buffer, regardless of l/k.
                long cur = (long)1 << (n - processed);
                for (int i = 0; i < l; ++i) {
                    const FieldElement ri = rho[processed + i];
                    const gpu_fe g_r   = fe_to_gpu(ri);
                    const gpu_fe g_omr = fe_to_gpu(to_FE(1) - ri);
                    const long oe = cur >> 1;
                    const int fbk = (int)std::max(1L, (oe + threads - 1) / threads);
                    hipLaunchKernelGGL(fold_bk_kernel, dim3(fbk), dim3(threads), 0, stream,
                        bkA_d, d_bkA_dst, g_omr, g_r, (int)oe, 0);
                    hipLaunchKernelGGL(fold_bk_kernel, dim3(fbk), dim3(threads), 0, stream,
                        bkB_d, d_bkB_dst, g_omr, g_r, (int)oe, 0);
                    HIP_CHECK(hipStreamSynchronize(stream));
                    std::swap(bkA_d, d_bkA_dst);
                    std::swap(bkB_d, d_bkB_dst);
                    cur = oe;
                }
            }
            if (g_matmul_timers.on) g_matmul_timers.fold += mm_secs(tp);
        }
        processed += l;
    }

    HIP_CHECK(hipStreamSynchronize(stream));
    gpu_fe h_A, h_B;
    HIP_CHECK(hipMemcpy(&h_A, bkA_d, sizeof(gpu_fe), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&h_B, bkB_d, sizeof(gpu_fe), hipMemcpyDeviceToHost));
    FieldElement eval_A = fe_from_gpu(h_A);
    FieldElement eval_B = fe_from_gpu(h_B);
    if (eval_A * eval_B != claim) ok = false;
    if (!ok) {
        std::cerr << "[gkr] VERIFICATION REJECTED (stage=product-sumcheck, GPU Blendy)\n";
        std::cout << "RESULT: REJECT\n";
        std::exit(2);
    }

    HIP_CHECK(hipFree(orig_d_bkA_dst)); HIP_CHECK(hipFree(orig_d_bkB_dst));
    if (d_bkA_dst2) { HIP_CHECK(hipFree(d_bkA_dst2)); HIP_CHECK(hipFree(d_bkB_dst2)); }
    HIP_CHECK(hipFree(d_bkA_ext)); HIP_CHECK(hipFree(d_bkB_ext));
    HIP_CHECK(hipFree(d_ptable_src)); HIP_CHECK(hipFree(d_ptable_dst));
    HIP_CHECK(hipFree(d_partials)); HIP_CHECK(hipFree(d_eq));
    HIP_CHECK(hipFree(d_p0)); HIP_CHECK(hipFree(d_p1)); HIP_CHECK(hipFree(d_p2));
    return {eval_A, eval_B};
}

// ---------------------------------------------------------------------------
// Cross-layer witness prefetch registry.
//
// The matmul `pa` table is folded from a slice of p->val[0]. That raw witness is
// CHALLENGE-INDEPENDENT, so while layer i's sumcheck runs on the compute stream
// we can already stream layer (i-1)'s (next-in-proving-order) witness onto the
// device on the copy stream. gpu_prefetch_matmul_witness() issues that async
// H2D (no sync); gpu_bookkeeping_fold() then consumes the resident buffer
// instead of staging again. Keyed by fc_id (unique per matmul layer).
// ---------------------------------------------------------------------------
struct PrefetchedWitness {
    gpu_fe* d_raw = nullptr;        // device buffer length 1<<totlog, staged on copy_s
    std::vector<gpu_fe> h_keep;     // host source, kept alive until the copy drains
    int totlog = 0;
};
inline std::unordered_map<int, PrefetchedWitness>& gpu_witness_cache() {
    static std::unordered_map<int, PrefetchedWitness> c;
    return c;
}

// Convert+upload val0[off .. off+2^totlog) for matmul layer `fc_id` onto the
// device on the copy stream, ASYNC (no sync) so it overlaps whatever is running
// on the compute stream. Idempotent: a second call for the same fc_id is a no-op.
inline void gpu_prefetch_matmul_witness(const std::vector<mcl::bn::Fr>& val0, long off,
                                        int totlog, int fc_id) {
    auto& cache = gpu_witness_cache();
    if (cache.count(fc_id)) return;
    const long totN = 1L << totlog;
    PrefetchedWitness pw;
    pw.totlog = totlog;
    pw.h_keep.resize(totN);
    long avail = std::max(0L, std::min(totN, (long)val0.size() - off));
    for (long i = 0; i < avail; ++i) pw.h_keep[i] = fe_to_gpu(val0[off + i]);
    if (avail < totN) std::memset(pw.h_keep.data() + avail, 0,
                                  (totN - avail) * sizeof(gpu_fe));
    HIP_CHECK(GKR_DMALLOC(&pw.d_raw, totN * sizeof(gpu_fe)));
    HIP_CHECK(zkgpt_gpu::gkr_stage_h2d(pw.d_raw, pw.h_keep.data(),
                                       totN * sizeof(gpu_fe),
                                       zkgpt_gpu::gkr_copy_stream()));
    cache[fc_id] = std::move(pw);   // copy drains on copy_s; consumer syncs before use
}

// GPU equivalent of verifier.cpp::init_bookkeeping. Folds the top `nfold` bits of
// `hsrc` (host, length 1<<totlog) MSB-first, returning a device buffer of length
// 1<<(totlog-nfold). `d_a_pre` non-null => adopt that prefetched buffer, skip H2D.
inline gpu_fe* gpu_bookkeeping_fold(const std::vector<gpu_fe>& hsrc, int totlog, int nfold,
                                 const mcl::bn::Fr* ra, hipStream_t copy_s, hipStream_t comp_s,
                                 gpu_fe* d_a_pre = nullptr) {
    long N = 1L << totlog;
    gpu_fe *d_a, *d_b;
    if (d_a_pre) {
        d_a = d_a_pre;                                // prefetched, already on copy_s
    } else {
        HIP_CHECK(GKR_DMALLOC(&d_a, N * sizeof(gpu_fe)));
        HIP_CHECK(zkgpt_gpu::gkr_stage_h2d(d_a, hsrc.data(), N * sizeof(gpu_fe), copy_s));
    }
    HIP_CHECK(GKR_DMALLOC(&d_b, (N >> 1 ? N >> 1 : 1) * sizeof(gpu_fe)));
    HIP_CHECK(hipStreamSynchronize(copy_s));
    const int threads = 256;
    long cur = N;
    for (int t = 0; t < nfold; ++t) {
        int bit = totlog - 1 - t;             // current top bit (== ra index)
        long out = cur >> 1;
        mcl::bn::Fr r = ra[bit];
        gpu_fe rho = fe_to_gpu(r);
        gpu_fe omr = fe_to_gpu(to_FE(1) - r);
        int fb = (int)std::max(1L, (out + threads - 1) / threads);
        hipLaunchKernelGGL(fold_bk_kernel, dim3(fb), dim3(threads), 0, comp_s,
                           d_a, d_b, omr, rho, (int)out, bit);
        HIP_CHECK(hipStreamSynchronize(comp_s));
        std::swap(d_a, d_b);
        cur = out;
    }
    HIP_CHECK(hipFree(d_b));
    return d_a;                                 // length 1<<(totlog-nfold)
}

// Descriptor for the next matmul layer in proving order, so this layer can
// prefetch that layer's raw witness while its own sumcheck computes. A default
// (val0==nullptr) means "no next layer" (last matmul) -> nothing to prefetch.
struct NextMatmulWitness {
    const std::vector<mcl::bn::Fr>* val0 = nullptr;
    long off = 0;
    int  totlog = 0;     // m+n of the next layer
    int  fc_id = -1;
};

// Builds pa on the GPU from the raw witness slice, takes pb from the host, and runs
// the product sumcheck with no pa round-trip. `fc_id` keys this layer in the prefetch
// cache; `nxt` describes the next matmul layer, whose H2D is issued here to overlap
// this layer's sumcheck. Returns (pa(r), pb(r)).
inline std::pair<mcl::bn::Fr, mcl::bn::Fr>
gpu_matmul_bookkeeping_sumcheck(const std::vector<mcl::bn::Fr>& val0, long off, int m, int n,
                             const mcl::bn::Fr* r_u, const mcl::bn::Fr* pb_host, mcl::bn::Fr ans,
                             int fc_id = -1, const NextMatmulWitness& nxt = {}) {
    hipStream_t copy_s = zkgpt_gpu::gkr_copy_stream();
    hipStream_t comp_s = zkgpt_gpu::gkr_compute_stream();
    const long totN = 1L << (m + n);
    const int Nm = 1 << m;

    // Adopt this layer's prefetched raw witness if one was staged for us; else
    // build the host gpu_fe slice of val0 (zero-padded past the witness) now.
    auto& cache = gpu_witness_cache();
    gpu_fe* d_pre = nullptr;
    std::vector<gpu_fe> hpa;
    auto it = cache.find(fc_id);
    const bool trace = gkr_config.dump_prefetch;
    if (fc_id >= 0 && it != cache.end()) {
        d_pre = it->second.d_raw;          // already on the device (copy_s)
        if (trace) fprintf(stderr, "[prefetch] fc_id=%d HIT (witness already resident)\n", fc_id);
    } else {
        if (trace) fprintf(stderr, "[prefetch] fc_id=%d MISS (staging inline)\n", fc_id);
        hpa.resize(totN);
        long avail = std::max(0L, std::min(totN, (long)val0.size() - off));
        for (long i = 0; i < avail; ++i) hpa[i] = fe_to_gpu(val0[off + i]);
        if (avail < totN) std::memset(hpa.data() + avail, 0, (totN - avail) * sizeof(gpu_fe));
    }

    gpu_fe* dPA = gpu_bookkeeping_fold(hpa, m + n, n, r_u, copy_s, comp_s, d_pre);  // -> 2^m
    if (d_pre) cache.erase(it);            // fold adopted+freed the buffer; drop host copy

    gpu_fe* dPB;
    HIP_CHECK(GKR_DMALLOC(&dPB, Nm * sizeof(gpu_fe)));
    std::vector<gpu_fe> hpb(Nm);
    for (int i = 0; i < Nm; ++i) hpb[i] = fe_to_gpu(pb_host[i]);
    HIP_CHECK(zkgpt_gpu::gkr_stage_h2d(dPB, hpb.data(), Nm * sizeof(gpu_fe), copy_s));
    HIP_CHECK(hipStreamSynchronize(copy_s));   // this layer's inputs resident

    // Kick off the NEXT layer's raw-witness transfer NOW: it streams on copy_s
    // while the product sumcheck below runs on comp_s (the overlap we want).
    if (nxt.val0 && nxt.fc_id >= 0) {
        if (trace) fprintf(stderr, "[prefetch] fc_id=%d -> staging next fc_id=%d (%ld elems) on copy stream\n",
                           fc_id, nxt.fc_id, 1L << nxt.totlog);
        gpu_prefetch_matmul_witness(*nxt.val0, nxt.off, nxt.totlog, nxt.fc_id);
    }

    int l_stage = m;
    if (gkr_config.matmul_k > 0) l_stage = (m + gkr_config.matmul_k - 1) / gkr_config.matmul_k;
    else if (gkr_config.matmul_l > 0) l_stage = gkr_config.matmul_l;
    std::vector<FieldElement> rr {r_u, r_u + m};
    auto res = product_sumcheck(dPA, dPB, Nm, m, ans, rr.data(), l_stage, 256, comp_s);
    HIP_CHECK(hipFree(dPA));
    HIP_CHECK(hipFree(dPB));
    return res;
}

// Drop-in replacement for verifier.cpp::sum_check_product. pa/pb: host, length
// 2^(m+1); n = m+1 variables; ans: claimed Σ pa·pb. Returns (pa(r), pb(r)).

// computes, based on the bookkeeping tables pa and pb, the claimed evaluation at the random point r
inline std::pair<mcl::bn::Fr, mcl::bn::Fr>
gpu_matmul_product_sumcheck(const mcl::bn::Fr* pa, const mcl::bn::Fr* pb, int m, const mcl::bn::Fr* r, mcl::bn::Fr ans) {
    const int n = m + 1;
    const int N { 1 << n };
    const int threads = 256;

    std::vector<gpu_fe> hA(N), hB(N);
    for (int i = 0; i < N; ++i) { 
        hA[i] = fe_to_gpu(pa[i]); 
        hB[i] = fe_to_gpu(pb[i]); 
    }

    gpu_fe *dA, *dB;
    HIP_CHECK(GKR_DMALLOC(&dA, N * sizeof(gpu_fe)));
    HIP_CHECK(GKR_DMALLOC(&dB, N * sizeof(gpu_fe)));
    // Stage on the copy stream, sumcheck on the compute stream (see gpu_uvm_alloc.hpp).
    hipStream_t copy_s = zkgpt_gpu::gkr_copy_stream();
    hipStream_t comp_s = zkgpt_gpu::gkr_compute_stream();
    HIP_CHECK(zkgpt_gpu::gkr_stage_h2d(dA, hA.data(), N * sizeof(gpu_fe), copy_s));
    HIP_CHECK(zkgpt_gpu::gkr_stage_h2d(dB, hB.data(), N * sizeof(gpu_fe), copy_s));
    HIP_CHECK(hipStreamSynchronize(copy_s));   // inputs resident before compute

    std::vector<FieldElement> rr {r, r + n};

    int l_stage = n;
    if (gkr_config.matmul_k > 0) l_stage = (n + gkr_config.matmul_k - 1) / gkr_config.matmul_k;
    else if (gkr_config.matmul_l > 0) l_stage = gkr_config.matmul_l;

    auto res = product_sumcheck(dA, dB, N, n, ans, rr.data(), l_stage, threads, comp_s);

    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dB));
    return res;
}

// Shared staged product sumcheck for host-array callers (e.g. the Lasso bridge).
// f, g: host Fr, length 2^n. gamma: claimed Σ f·g. Returns (f(r), g(r)).
inline std::pair<mcl::bn::Fr, mcl::bn::Fr>
gpu_product_sumcheck_host(const mcl::bn::Fr* f, const mcl::bn::Fr* g, int n, const mcl::bn::Fr* r, mcl::bn::Fr gamma, int l_stage) {
    mm_scope_timer _hw(&g_matmul_timers.host_wall);
    // 64-bit: n reaches 31 at opt2b7 (the whole-circuit Lasso domain), and
    // `1 << 31` in a 32-bit int is signed overflow -- it would break the streaming
    // decision below (win < (long)N against a negative N) and feed a garbage byte
    // count into the dense path's hipMalloc, surfacing as a spurious "out of memory".
    const long N = 1L << n;
    const int threads = 256;

    // GKR_DUMP_SIZES reports what this call costs: 2*N*32B of DEVICE buffers, the
    // dominant per-partition device term (it caps the concurrent wave width).
    // f = lasso_mult_v (scatter-accumulated reference counts), g = val[0] (witness).
    const bool dump = getenv("GKR_DUMP_SIZES") != nullptr;
    double t_conv = 0.0, t_h2d = 0.0, host_stage_gb = 0.0;
    auto now_s = [] {
        return std::chrono::duration<double>(
                   std::chrono::steady_clock::now().time_since_epoch()).count();
    };

    // ---- streaming decision (GKR_SC_STREAM_BITS = log2 of the window, elements).
    // Unset is the dense path, bit-for-bit. Feasibility is decided HERE rather
    // than inside product_sumcheck because it determines whether the caller
    // allocates a 2^n device source at all.
    int stream_bits = -1;
    if (const char* e = getenv("GKR_SC_STREAM_BITS")) stream_bits = atoi(e);
    const int ls = (l_stage <= 0 || l_stage > n) ? n : l_stage;
    long win = (stream_bits >= 0) ? (1L << std::min(stream_bits, 40)) : 0;
    if (win < (1L << ls)) win = 1L << ls;          // must hold at least one stage block
    // Two cases fall back to dense: a window covering the whole source saves
    // nothing, and l_stage == n is a single stage -- its expand slab IS the whole
    // source, and there is no second stage to give the smaller fold destination.
    const bool do_stream = (stream_bits >= 0) && (ls < n) && (win < (long)N);

    std::vector<FieldElement> rr(r, r + n);

    if (do_stream) {
        sc_src_stream src;
        src.alloc(f, g, win, N);
        auto res = product_sumcheck(nullptr, nullptr, N, n, gamma, rr.data(),
                                    l_stage, threads, 0, &src);
        if (dump) {
            const double dev_gb = 2.0 * (double)win * sizeof(gpu_fe) / 1073741824.0;
            const double moved  = 2.0 * (double)src.elems * sizeof(gpu_fe) / 1073741824.0;
            fprintf(stderr,
                    // Host staging is always 0.000 -- the H2D reads the mcl array in
                    // place. Kept in the line so the log format stays stable.
                    "[src] n=%d N=%ld l_stage=%d | STREAMED win=%ld (%.3f GB device, "
                    "%.3f GB host) | uploads=%ld moved=%.2f GB (%.2fx source) | "
                    "conv=%.3fs h2d=%.3fs (%.1f GB/s)\n",
                    n, N, ls, win, dev_gb, 0.0, src.uploads, moved,
                    moved / (2.0 * (double)N * sizeof(gpu_fe) / 1073741824.0),
                    src.t_conv, src.t_h2d, src.t_h2d > 0 ? moved / src.t_h2d : 0.0);
            if (src.sparse)
                fprintf(stderr, "[src] SPARSE nnzA=%zu nnzB=%zu max_win_nnz=%ld/%ld build=%.3fs "
                        "uploaded=%.3f GB (dense would be %.2f GB)\n",
                        src.spA.idx.size(), src.spB.idx.size(), src.spA.max_win_nnz,
                        src.spB.max_win_nnz, src.t_sparse_build, src.sparse_bytes / 1073741824.0, moved);
        }
        src.free();
        return res;
    }

    gpu_fe *dF, *dG;
    HIP_CHECK(GKR_DMALLOC(&dF, N * sizeof(gpu_fe)));
    HIP_CHECK(GKR_DMALLOC(&dG, N * sizeof(gpu_fe)));

    // Host->device upload of the dense source, issued straight from the mcl arrays
    // (no staging buffer -- see bring() and its static_assert). This is the path
    // every NON-streamed cell takes.
    {
        const double t0 = now_s();
        HIP_CHECK(gkr_h2d(dF, reinterpret_cast<const gpu_fe*>(f),
                          (size_t)N * sizeof(gpu_fe)));
        HIP_CHECK(gkr_h2d(dG, reinterpret_cast<const gpu_fe*>(g),
                          (size_t)N * sizeof(gpu_fe)));
        t_h2d = now_s() - t0;
        t_conv = 0.0;
        host_stage_gb = 0.0;   // no staging buffer
    }
    if (dump) {
        const double gb = 2.0 * (double)N * sizeof(gpu_fe) / 1073741824.0;
        fprintf(stderr,
                "[src] n=%d N=%ld l_stage=%d | host_staging=%.3f GB device_src=%.2f GB | "
                "conv=%.3fs h2d=%.3fs (%.1f GB/s)\n",
                n, N, l_stage, host_stage_gb, gb, t_conv, t_h2d, t_h2d > 0 ? gb / t_h2d : 0.0);
    }

    auto res = product_sumcheck(dF, dG, N, n, gamma, rr.data(), l_stage, threads);

    HIP_CHECK(hipFree(dF));
    HIP_CHECK(hipFree(dG));
    return res;
}


} // namespace zkgpt_gpu

#endif // ZKGPT_GPU_MATMUL_PROVER_HPP
