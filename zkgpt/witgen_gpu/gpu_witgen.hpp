#pragma once
#ifndef ZKGPT_GPU_WITGEN_HPP
#define ZKGPT_GPU_WITGEN_HPP

// GPU witness generation for neuralNetwork::create().
//
// Everything witgen lives in this one file, in this order:
//
//   1. shared device helpers   -- Fr <-> integer mirrors of the host conversions
//   2. FC GEMM                 -- fullyConnLayer
//   3. roundLayer              -- the per-element requantization value loop
//   4. val[0] device mirror    -- the witness arena the kernels read from
//   5. generic gate evaluator  -- calcNormalLayer on the device

#include <hip/hip_runtime.h>
#include <mcl/bn.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <type_traits>
#include <vector>

#include "circuit.h"
#include "gpu_field.hpp"
#include "gpu_uvm_alloc.hpp"

namespace zkgpt_gpu {

static_assert(sizeof(FieldElement) == sizeof(gpu_fe),
              "witgen DMA reinterprets mcl Fr as gpu_fe; sizes must match");

// ===========================================================================
// 1. Shared device helpers
// ===========================================================================
//
// Every primitive here mirrors its host counterpart bit-for-bit. The one
// non-obvious substitution: HIP's device pow(double,int) is NOT exact at every
// integer exponent (wrong at m=-4 and m=-9 on gfx90a); ldexp(1.0, m) is exact for
// every m in [-10,10] and is what this file uses. Do not swap it for pow().

// mcl::bn::Fr's isNegative() threshold: true iff half <= x < r, where
// half = (r+1)/2. Value is r's normal-form representation, NOT Montgomery --
// compare against blstrs__scalar__Scalar_unmont(x), matching prover.cpp's
// convert() calling x.isNegative() on the raw (non-Montgomery) value.
__constant__ static const gpu_fe kRoundHalf = {
    { 4160749569u, 2716924617u, 1021098056u, 2484728868u,
      1086368814u, 3693617883u, 1889062932u, 405940025u } };

// Fr -> signed __int128, mirroring prover.cpp::convert(Fr) exactly:
// isNegative() (compare normal-form value against half) -> abs (P - x if
// negative) -> take the low 128 bits -> reattach sign. Pre: |value as a
// signed residue| fits in 128 bits (true for every witness value this
// codebase produces; convert() makes the same assumption via
// getLittleEndian(bf,16)).
__device__ inline __int128 gpu_fe_to_i128(gpu_fe x) {
    gpu_fe n = blstrs__scalar__Scalar_unmont(x);
    const bool neg = blstrs__scalar__Scalar_gte(n, kRoundHalf);
    if (neg) n = blstrs__scalar__Scalar_sub(blstrs__scalar__Scalar_P, n);
    unsigned long long lo = (unsigned long long)n.val[0] | ((unsigned long long)n.val[1] << 32);
    unsigned long long hi = (unsigned long long)n.val[2] | ((unsigned long long)n.val[3] << 32);
    unsigned __int128 mag = ((unsigned __int128)hi << 64) | (unsigned __int128)lo;
    return neg ? -(__int128)mag : (__int128)mag;
}

// int64_t -> Fr (Montgomery), mirroring mcl::fp::FpT::operator=(int64_t)
// exactly: abs (UB-free at INT64_MIN) -> low 64 bits into raw limbs 0/1 ->
// negate via P-sub if negative -> Montgomery-ify. This is also the implicit
// conversion mcl performs for `Fr x = some_int128_expr;` in this codebase's
// build (FpT has no __int128 overload, so the expression narrows to
// int64_t first, then hits this same path).
__device__ inline gpu_fe gpu_i64_to_fr(long long x) {
    if (x == 0) return blstrs__scalar__Scalar_ZERO;
    const bool neg = x < 0;
    unsigned long long u = neg ? (unsigned long long)(-(x + 1)) + 1ULL
                               : (unsigned long long)x;
    gpu_fe raw = blstrs__scalar__Scalar_ZERO;
    raw.val[0] = (unsigned)(u & 0xffffffffULL);
    raw.val[1] = (unsigned)(u >> 32);
    if (neg) raw = blstrs__scalar__Scalar_sub(blstrs__scalar__Scalar_P, raw);
    return blstrs__scalar__Scalar_mont(raw);
}

// ===========================================================================
// 2. FC GEMM -- fullyConnLayer
// ===========================================================================
//
// The FC loop is a plain C = A * B^T over Fr:
//     C[i][co] = sum_ci A[i][ci] * B[co][ci]
// Each output is written by exactly one thread with a private register
// accumulator over ci. No atomics, no scatter, no inter-thread communication.

// out[i*co_n + co] = sum_ci A[i*ci_n + ci] * B[co*ci_n + ci]
static __global__ void fc_gemm_kernel(const gpu_fe* __restrict__ A,
                                      const gpu_fe* __restrict__ B,
                                      gpu_fe* __restrict__ C,
                                      int len, int ci_n, int co_n) {
    const long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)len * co_n;
    if (tid >= total) return;

    const int i  = (int)(tid / co_n);
    const int co = (int)(tid % co_n);

    gpu_fe acc = {0};

    const gpu_fe* arow = A + (long)i  * ci_n;
    const gpu_fe* brow = B + (long)co * ci_n;

    for (int ci = 0; ci < ci_n; ++ci)
        acc = device_add(acc, device_mul(arow[ci], brow[ci]));

    C[tid] = acc;
}

inline void gpu_fully_conn_layer(const FieldElement* hA,
                                 const FieldElement* hB,
                                 FieldElement* hC,
                                 int len, int channel_in, int channel_out) {
    const size_t a_elems = (size_t)len * channel_in;
    const size_t b_elems = (size_t)channel_out * channel_in;
    const size_t c_elems = (size_t)len * channel_out;

    hipStream_t copy_s = gkr_copy_stream();
    hipStream_t comp_s = gkr_compute_stream();

    gpu_fe *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    HIP_CHECK(GKR_DMALLOC(&d_A, a_elems * sizeof(gpu_fe)));
    HIP_CHECK(GKR_DMALLOC(&d_B, b_elems * sizeof(gpu_fe)));
    HIP_CHECK(GKR_DMALLOC(&d_C, c_elems * sizeof(gpu_fe)));

    HIP_CHECK(gkr_stage_h2d(d_A, reinterpret_cast<const gpu_fe*>(hA),
                            a_elems * sizeof(gpu_fe), copy_s));
    HIP_CHECK(gkr_stage_h2d(d_B, reinterpret_cast<const gpu_fe*>(hB),
                            b_elems * sizeof(gpu_fe), copy_s));

    HIP_CHECK(hipStreamSynchronize(copy_s));   // inputs resident before compute

    const int threads = 256;
    const long total  = (long)len * channel_out;
    const int nblocks = (int)((total + threads - 1) / threads);
    hipLaunchKernelGGL(fc_gemm_kernel, dim3(nblocks), dim3(threads), 0, comp_s,
                       d_A, d_B, d_C, len, channel_in, channel_out);

    HIP_CHECK(hipMemcpyAsync(reinterpret_cast<gpu_fe*>(hC), d_C,
                             c_elems * sizeof(gpu_fe),
                             hipMemcpyDeviceToHost, comp_s));
    HIP_CHECK(hipStreamSynchronize(comp_s));
    gkr_xfer_add_d2h(c_elems * sizeof(gpu_fe));

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
}

// Same as gpu_fully_conn_layer, but returns the device output buffer instead
// of D2H'ing + freeing it. Pre: len,channel_in,channel_out > 0. Post: return
// value is a live GKR_DMALLOC'd buffer of len*channel_out gpu_fe, caller owns
// it (hipFree when done).
inline gpu_fe* gpu_fully_conn_layer_resident(const FieldElement* hA,
                                             const FieldElement* hB,
                                             int len, int channel_in, int channel_out) {
    const size_t a_elems = (size_t)len * channel_in;
    const size_t b_elems = (size_t)channel_out * channel_in;
    const size_t c_elems = (size_t)len * channel_out;

    hipStream_t copy_s = gkr_copy_stream();
    hipStream_t comp_s = gkr_compute_stream();

    gpu_fe *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    HIP_CHECK(GKR_DMALLOC(&d_A, a_elems * sizeof(gpu_fe)));
    HIP_CHECK(GKR_DMALLOC(&d_B, b_elems * sizeof(gpu_fe)));
    HIP_CHECK(GKR_DMALLOC(&d_C, c_elems * sizeof(gpu_fe)));

    { GkrXferTimer _x(&gkr_xfer_h2d_s_ref());
      HIP_CHECK(gkr_stage_h2d(d_A, reinterpret_cast<const gpu_fe*>(hA),
                              a_elems * sizeof(gpu_fe), copy_s));
      HIP_CHECK(gkr_stage_h2d(d_B, reinterpret_cast<const gpu_fe*>(hB),
                              b_elems * sizeof(gpu_fe), copy_s));
      HIP_CHECK(hipStreamSynchronize(copy_s)); }

    const int threads = 256;
    const long total  = (long)len * channel_out;
    const int nblocks = (int)((total + threads - 1) / threads);
    hipLaunchKernelGGL(fc_gemm_kernel, dim3(nblocks), dim3(threads), 0, comp_s,
                       d_A, d_B, d_C, len, channel_in, channel_out);
    HIP_CHECK(hipStreamSynchronize(comp_s));

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    return d_C;
}

// ===========================================================================
// 3. roundLayer
// ===========================================================================

// out[g]            = Fr(q)                    -- the "qq" half, first block_len
// out[block_len + g] = Fr(left) * Fr(right)      -- the "s" half, next block_len
// where left/right are the exact __int128 expressions from roundLayer,
// truncated to int64_t before becoming an Fr (matching the host's implicit
// narrowing).
static __global__ void round_layer_kernel(const gpu_fe* __restrict__ prev,
                                          gpu_fe* __restrict__ out,
                                          long long block_len, int c, int m, int M) {
    const long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= block_len) return;

    const __int128 p = gpu_fe_to_i128(prev[g]);
    const double y = (double)p * c * ldexp(1.0, m);
    const __int128 q = (__int128)round(y);

    out[g] = gpu_i64_to_fr((long long)q);

    const __int128 left  = p * c * ((__int128)1 << (m + M + 1))
                          + ((__int128)1 << M)
                          - q * ((__int128)1 << (M + 1));
    const __int128 right = q * ((__int128)1 << (M + 1))
                          + ((__int128)1 << M)
                          - (__int128)c * ((__int128)1 << (m + M + 1)) * p;

    out[block_len + g] = device_mul(gpu_i64_to_fr((long long)left),
                                    gpu_i64_to_fr((long long)right));
}

// Pre: d_prev holds block_len resident gpu_fe (fullyConnLayer's device
// output). h_out points at block_len*2 contiguous FieldElement -- the exact
// [qq, qq+block_len) + [s, s+block_len) layout roundLayer lays out in
// val[0]. Post: h_out is filled; d_prev is NOT freed (caller's buffer,
// caller's call to free it) unless free_prev is true.
inline void gpu_round_layer_from_device(gpu_fe* d_prev, FieldElement* h_out,
                                        long long block_len, int c, int m, int M,
                                        bool free_prev = true) {
    hipStream_t comp_s = gkr_compute_stream();

    gpu_fe* d_out = nullptr;
    HIP_CHECK(GKR_DMALLOC(&d_out, (size_t)(2 * block_len) * sizeof(gpu_fe)));

    const int threads = 256;
    const int nblocks = (int)((block_len + threads - 1) / threads);
    hipLaunchKernelGGL(round_layer_kernel, dim3(nblocks), dim3(threads), 0, comp_s,
                       d_prev, d_out, block_len, c, m, M);

    HIP_CHECK(hipMemcpyAsync(reinterpret_cast<gpu_fe*>(h_out), d_out,
                             (size_t)(2 * block_len) * sizeof(gpu_fe),
                             hipMemcpyDeviceToHost, comp_s));
    HIP_CHECK(hipStreamSynchronize(comp_s));
    gkr_xfer_add_d2h((size_t)(2 * block_len) * sizeof(gpu_fe));

    HIP_CHECK(hipFree(d_out));
    if (free_prev) HIP_CHECK(hipFree(d_prev));
}

// ===========================================================================
// 4. val[0] device mirror (the witness arena)
// ===========================================================================
//
// val[0] is the input layer: prompt, the whole model weight region, the KV
// segments, and every builder's appended advice block. It is ~1.1 GiB on gpt2
// and it is what every gate on the device reads through, so it lives in one
// grow-only device buffer for the life of create() instead of being staged
// per call.
//
// Keeping it coherent cheaply is the only subtlety. Two things dirty it:
//
//   (a) APPEND -- a builder resize()s val[0] and fills the new tail. Detected
//       automatically: anything past `synced_` is dirty by definition.
//   (b) BACK-WRITE -- a builder writes into a block an EARLIER builder
//       appended. softmax_layer_2 does this (it fills s_ij / d3_off inside
//       softmax_layer_1's block), and so does the KV capture. There is no way
//       to detect that from sizes, so those call sites must announce it with
//       witgen_val0_dirty(from).
//
// Miss a (b) site and the device reads a stale value: the layer output is
// silently wrong, the honest proof still has a consistent-looking witness for
// the gates that were evaluated, and it surfaces far downstream as a REJECT.
// GKR_GATES_CHECK=1 catches it immediately -- use it whenever a builder is
// touched.

// ---- phase accounting for the evaluator -------------------------------------
// Always on: these are per-LAYER (21 per partition), not per-element, so the
// clock reads cost nothing measurable.
struct GateEvalTimers {
    double val0_s  = 0;    // val[0] mirror sync (H2D)
    double prev_s  = 0;    // val[layer_id-1] H2D
    double gates_s = 0;    // gate SoA H2D
    double csr_s   = 0;    // count + scan + scatter
    double redu_s  = 0;    // the reduce kernel
    double d2h_s   = 0;    // val[layer_id] readback
    size_t val0_bytes = 0, prev_bytes = 0, gate_bytes = 0, d2h_bytes = 0;
    long   layers = 0;
    void clear() { *this = GateEvalTimers{}; }
};
inline GateEvalTimers& gate_timers() { static GateEvalTimers t; return t; }

inline double wg_now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// sc is `ll` (__int128), but mcl's only integer conversion is FpT(int64_t), so
// the host expression `val[..] * gate.sc` narrows to int64 before it ever
// becomes a field element. Replicate that exactly -- an arithmetically better
// 128-bit conversion would DISAGREE with the prover and surface as a REJECT.
__device__ inline gpu_fe wg_sc_to_fr(long long v) { return gpu_i64_to_fr(v); }

struct Val0Mirror {
    gpu_fe* d    = nullptr;
    size_t  cap  = 0;      // elements allocated
    size_t  synced = 0;    // elements known to match the host
    size_t  dirty_lo = ~(size_t)0;  // lowest back-written index since last sync
    bool    active = false;
};

inline Val0Mirror& val0_mirror() { static Val0Mirror m; return m; }

// Announce a back-write into already-synced territory. Cheap and idempotent;
// call it whenever a builder writes val[0] below its own append point.
//
// This is the coarse form: it only records a LOW-WATER mark, so the next sync
// re-uploads everything from there to the end of val[0]. That is correct but
// can be very expensive for a write near the START of val[0] (e.g. the KV
// region). Prefer witgen_val0_push() below, which sends exactly the range that
// changed; this is for a back-write whose extent is not known.
inline void witgen_val0_dirty(size_t from) {
    Val0Mirror& m = val0_mirror();
    if (from < m.dirty_lo) m.dirty_lo = from;
}

// Push exactly [lo, hi) of the host's val[0] to the mirror. No-op when the
// mirror is cold (the first sync will carry the range anyway) and clamped to
// what is already synced (anything past that is picked up by the next sync's
// growth path). Use this instead of witgen_val0_dirty wherever the written
// range is known.
inline void witgen_val0_push(const FieldElement* h_val0, size_t lo, size_t hi) {
    Val0Mirror& m = val0_mirror();
    if (!m.active || !m.d) return;
    const size_t end = std::min(hi, m.synced);
    if (lo >= end) return;
    gate_timers().val0_bytes += (end - lo) * sizeof(gpu_fe);
    GkrXferTimer _x(&gkr_xfer_h2d_s_ref());
    HIP_CHECK(gkr_stage_h2d(m.d + lo, reinterpret_cast<const gpu_fe*>(h_val0) + lo,
                            (end - lo) * sizeof(gpu_fe), gkr_copy_stream()));
    HIP_CHECK(hipStreamSynchronize(gkr_copy_stream()));
}

inline void witgen_val0_reset() {
    Val0Mirror& m = val0_mirror();
    m.synced = 0;
    m.dirty_lo = ~(size_t)0;
    m.active = false;
}

// Bring the device copy up to date with h_val0[0..n). Uploads only
// [min(dirty_lo, synced), n) -- in steady state that is the one or two advice
// blocks written since the previous layer, not the whole 1.1 GiB.
inline gpu_fe* witgen_val0_sync(const FieldElement* h_val0, size_t n) {
    Val0Mirror& m = val0_mirror();

    if (m.cap < n) {
        // Grow with headroom: val[0] is appended to ~20 times per partition and
        // reallocating a 1 GiB device buffer at each is pure copy cost.
        size_t want = n + n / 4 + 1024;
        gpu_fe* nd = nullptr;
        HIP_CHECK(GKR_DMALLOC(&nd, want * sizeof(gpu_fe)));
        if (m.d) {
            if (m.synced)
                HIP_CHECK(hipMemcpy(nd, m.d, m.synced * sizeof(gpu_fe),
                                    hipMemcpyDeviceToDevice));
            HIP_CHECK(hipFree(m.d));
        }
        m.d = nd;
        m.cap = want;
        gkr_vram_log("witgen val0", (long)(want * sizeof(gpu_fe)));
    }

    const size_t lo = std::min(m.dirty_lo, m.synced);
    if (lo < n) {
        gate_timers().val0_bytes += (n - lo) * sizeof(gpu_fe);
        GkrXferTimer _x(&gkr_xfer_h2d_s_ref());
        HIP_CHECK(gkr_stage_h2d(m.d + lo,
                                reinterpret_cast<const gpu_fe*>(h_val0) + lo,
                                (n - lo) * sizeof(gpu_fe), gkr_copy_stream()));
        HIP_CHECK(hipStreamSynchronize(gkr_copy_stream()));
    }
    m.synced = n;
    m.dirty_lo = ~(size_t)0;
    m.active = true;
    return m.d;
}

// Grow the mirror to n elements WITHOUT uploading the tail, zero [synced, n)
// on the device, and declare the whole range coherent. This is the entry point
// for a builder that computes its own advice block on the device: the host's
// copy of that block is produced by a D2H afterwards, so uploading the host's
// zeros first would be pure waste. Pre: [0, synced) is already coherent.
// `keep` is the prefix the host owns and the kernel will READ (the builder's
// orgsize); [keep, n) is the block the device is about to write. Syncing
// exactly [.., keep) matters when the mirror is cold: syncing min(n, m.synced)
// is 0 on a cold mirror, so a builder run without the gate evaluator would read
// an empty mirror and produce garbage.
inline gpu_fe* witgen_val0_grow(const FieldElement* h_val0, size_t keep, size_t n) {
    Val0Mirror& m = val0_mirror();
    witgen_val0_sync(h_val0, keep);

    if (m.cap < n) {
        size_t want = n + n / 4 + 1024;
        gpu_fe* nd = nullptr;
        HIP_CHECK(GKR_DMALLOC(&nd, want * sizeof(gpu_fe)));
        if (m.d) {
            if (m.synced)
                HIP_CHECK(hipMemcpy(nd, m.d, m.synced * sizeof(gpu_fe),
                                    hipMemcpyDeviceToDevice));
            HIP_CHECK(hipFree(m.d));
        }
        m.d = nd;
        m.cap = want;
        gkr_vram_log("witgen val0", (long)(want * sizeof(gpu_fe)));
    }
    if (n > m.synced)
        HIP_CHECK(hipMemsetAsync(m.d + m.synced, 0,
                                 (n - m.synced) * sizeof(gpu_fe),
                                 gkr_compute_stream()));
    m.synced = n;
    m.dirty_lo = ~(size_t)0;
    m.active = true;
    return m.d;
}

// Copy a device-computed advice block back into the host val[0]. The host copy
// is still what the prover, the boundary exports and the remaining CPU builders
// read; only the computation moved.
inline void witgen_val0_readback(FieldElement* h_val0, size_t lo, size_t hi) {
    if (hi <= lo) return;
    Val0Mirror& m = val0_mirror();
    HIP_CHECK(hipStreamSynchronize(gkr_compute_stream()));
    HIP_CHECK(gkr_d2h(reinterpret_cast<gpu_fe*>(h_val0) + lo, m.d + lo,
                      (hi - lo) * sizeof(gpu_fe)));
}

// ===========================================================================
// 5. Generic gate evaluator -- calcNormalLayer on the device
// ===========================================================================
//
// The host loop (neuralNetwork.cpp) is
//
//   for uni: val[L][g] += val[lu][u] * sc
//   for bin: val[L][g] += val[lu][u] * val[lv][v] * sc
//
// i.e. a scatter-accumulate with heavy collision on g (fan-in 5 in roundLayer,
// up to channel_in in ln_checker_layer1). There is no atomic add for a 32-byte
// field element, so the port inverts it: counting-sort the gates into a
// per-g CSR, then reduce each g with one thread. Field addition is commutative
// and associative, so gate order within a g is irrelevant and no stable sort
// is needed.
//
// lu/lv only ever resolve to 0 or L-1 (uniGate::lu is set to layer_id-1 by the
// softmax/gelu checkers and 0 everywhere else; binGate::getLayerIdU/V are
// defined that way in circuit.h), so exactly two source buffers are needed.
//
// Gate indices/scalars travel as SoA u32/u8/int64, never as field elements --
// the same reasoning as gkr_pred_gpu.cpp, where staging gates as Fr cost far more
// host time and PCIe traffic than the kernel itself.

constexpr int WG_BLOCK = 256;

// ---- exclusive scan over u32 -------------------------------------------------
// Two-level Blelloch would need a second pass anyway; this is Hillis-Steele per
// block plus a recursion on the block sums. Depth is <=3 for any layer this
// codebase builds (max ~8M outputs).

static __global__ void wg_scan_block(const unsigned* __restrict__ in,
                                     unsigned* __restrict__ out,
                                     unsigned* __restrict__ bsum, unsigned n) {
    __shared__ unsigned s[WG_BLOCK];
    const unsigned gid = blockIdx.x * WG_BLOCK + threadIdx.x;
    const unsigned t   = threadIdx.x;
    const unsigned mine = (gid < n) ? in[gid] : 0u;
    s[t] = mine;
    __syncthreads();
    for (unsigned off = 1; off < WG_BLOCK; off <<= 1) {
        unsigned v = (t >= off) ? s[t - off] : 0u;
        __syncthreads();
        s[t] += v;
        __syncthreads();
    }
    if (gid < n) out[gid] = s[t] - mine;          // inclusive -> exclusive
    if (t == WG_BLOCK - 1 && bsum) bsum[blockIdx.x] = s[t];
}

static __global__ void wg_scan_add(unsigned* __restrict__ out,
                                   const unsigned* __restrict__ bofs, unsigned n) {
    const unsigned gid = blockIdx.x * WG_BLOCK + threadIdx.x;
    if (gid < n) out[gid] += bofs[blockIdx.x];
}

// In-place exclusive scan of d[0..n). `scratch` must hold at least
// wg_scan_scratch(n) unsigned; the recursion carves its levels out of it.
inline unsigned wg_scan_scratch(unsigned n) {
    unsigned total = 0;
    while (n > 1) { n = (n + WG_BLOCK - 1) / WG_BLOCK; total += n; }
    return total + WG_BLOCK;
}

inline void wg_exclusive_scan(unsigned* d, unsigned n, unsigned* scratch,
                              hipStream_t s) {
    if (n == 0) return;
    const unsigned nb = (n + WG_BLOCK - 1) / WG_BLOCK;
    hipLaunchKernelGGL(wg_scan_block, dim3(nb), dim3(WG_BLOCK), 0, s,
                       d, d, nb > 1 ? scratch : nullptr, n);
    if (nb > 1) {
        wg_exclusive_scan(scratch, nb, scratch + nb, s);
        hipLaunchKernelGGL(wg_scan_add, dim3(nb), dim3(WG_BLOCK), 0, s, d, scratch, n);
    }
}

// ---- CSR build + reduce ------------------------------------------------------
//
// Gates share one index space: [0, n_uni) are uni gates, [n_uni, n_uni+n_bin)
// are bin gates. That keeps the CSR to a single index array and lets one
// reduce kernel walk both lists.
//
// uniGate/binGate go to the device as-is (AoS): this runs 21 times per partition on
// freshly built lists, so a host-side SoA transpose costs more than the kernel.
// Both structs are 32-byte PODs, so the upload is a straight memcpy of
// circuit.uni_gates.data(). The one field that wants to be coalesced is `g` (the
// count and scatter passes touch nothing else), so a device kernel extracts it
// into a dense u32 array.

static __global__ void wg_extract_g(const uniGate* __restrict__ u, unsigned nu,
                                    const binGate* __restrict__ b, unsigned nb,
                                    unsigned* __restrict__ out) {
    const unsigned stride = gridDim.x * blockDim.x;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < nu; i += stride)
        out[i] = u[i].g;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < nb; i += stride)
        out[nu + i] = b[i].g;
}

static __global__ void wg_count(const unsigned* __restrict__ gg, unsigned n,
                                unsigned* __restrict__ cnt) {
    const unsigned stride = gridDim.x * blockDim.x;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        atomicAdd(&cnt[gg[i]], 1u);
}

static __global__ void wg_scatter(const unsigned* __restrict__ gg, unsigned n,
                                  unsigned* __restrict__ pos,
                                  unsigned* __restrict__ idx) {
    const unsigned stride = gridDim.x * blockDim.x;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        idx[atomicAdd(&pos[gg[i]], 1u)] = i;
}

static __global__ void wg_reduce(const unsigned* __restrict__ off,
                                 const unsigned* __restrict__ idx,
                                 unsigned sz, unsigned n_uni,
                                 const uniGate* __restrict__ ug,
                                 const binGate* __restrict__ bg,
                                 const gpu_fe* __restrict__ val0,
                                 const gpu_fe* __restrict__ prev,
                                 gpu_fe* __restrict__ out) {
    const unsigned stride = gridDim.x * blockDim.x;
    for (unsigned g = blockIdx.x * blockDim.x + threadIdx.x; g < sz; g += stride) {
        gpu_fe acc = blstrs__scalar__Scalar_ZERO;
        const unsigned e = off[g + 1];
        for (unsigned k = off[g]; k < e; ++k) {
            const unsigned i = idx[k];
            if (i < n_uni) {
                const uniGate gt = ug[i];
                // lu is a real layer id; calcNormalLayer resolves it to val[0]
                // when zero and to val[layer_id-1] otherwise (circuit.h).
                const gpu_fe* su = gt.lu ? prev : val0;
                acc = device_add(acc,
                        device_mul(su[gt.u], wg_sc_to_fr((long long)gt.sc)));
            } else {
                const binGate gt = bg[i - n_uni];
                // getLayerIdU(l) = !l ? 0 : L-1 ; getLayerIdV(l) = !(l&1) ? 0 : L-1
                const gpu_fe* su = (gt.l != 0)  ? prev : val0;
                const gpu_fe* sv = (gt.l & 1)   ? prev : val0;
                acc = device_add(acc,
                        device_mul(device_mul(su[gt.u], sv[gt.v]),
                                   wg_sc_to_fr((long long)gt.sc)));
            }
        }
        out[g] = acc;
    }
}

// ---- persistent device state -------------------------------------------------
// Grow-only, like GkrScState (gkr_gpu.cpp) and PredState (gkr_pred_gpu.cpp):
// the arrays change size per layer but the peak is bounded, so per-layer
// hipMalloc/hipFree would be pure overhead.

struct GateEvalState {
    uniGate  *d_uni = nullptr;   size_t cap_uni = 0;
    binGate  *d_bin = nullptr;   size_t cap_bin = 0;

    unsigned *d_gg  = nullptr;   size_t cap_gg  = 0;   // extracted g, both lists
    unsigned *d_idx = nullptr;   size_t cap_idx = 0;
    unsigned *d_off = nullptr;   size_t cap_off = 0;   // sz+1
    unsigned *d_pos = nullptr;   size_t cap_pos = 0;
    unsigned *d_scan = nullptr;  size_t cap_scan = 0;

    gpu_fe *d_prev = nullptr;    size_t cap_prev = 0;
    gpu_fe *d_out  = nullptr;    size_t cap_out  = 0;
};

inline GateEvalState& gate_state() { static GateEvalState s; return s; }

inline void wg_ensure(void** p, size_t* cap, size_t need, const char* tag) {
    if (*cap >= need) return;
    if (*p) { HIP_CHECK(hipFree(*p)); *p = nullptr; }
    HIP_CHECK(GKR_DMALLOC(p, need));
    *cap = need;
    gkr_vram_log(tag, (long)need);
}

inline int wg_grid(size_t n) {
    const int b = (int)((n + WG_BLOCK - 1) / WG_BLOCK);
    return std::max(1, std::min(b, 4096));
}

inline bool& gate_d2h_pending() { static bool p = false; return p; }

inline void witgen_sync_gates() {
    if (!gate_d2h_pending()) return;
    HIP_CHECK(hipStreamSynchronize(gkr_copy_stream()));
    gate_d2h_pending() = false;
}

// Free everything. Call at the end of create() -- val[0] alone is ~1.1 GiB and
// must not stay resident through proving (12 children x 1.1 GiB does not fit).
inline void witgen_teardown() {
    witgen_sync_gates();
    GateEvalState& s = gate_state();
    for (void** p : {(void**)&s.d_uni, (void**)&s.d_bin, (void**)&s.d_gg,
                     (void**)&s.d_idx, (void**)&s.d_off, (void**)&s.d_pos,
                     (void**)&s.d_scan, (void**)&s.d_prev, (void**)&s.d_out})
        if (*p) { HIP_CHECK(hipFree(*p)); *p = nullptr; }
    s = GateEvalState{};

    Val0Mirror& m = val0_mirror();
    if (m.d) { HIP_CHECK(hipFree(m.d)); m.d = nullptr; }
    m.cap = 0;
    witgen_val0_reset();
}

// uniGate/binGate declare a user-provided constructor, so they have no default
// ctor and cannot be constructed inside a kernel (the ctor is host-only).
// These POD mirrors have identical layout and are what the emission kernels
// write through; the buffers are then reinterpreted as the real types. The
// offsetof asserts are the guard -- if circuit.h ever reorders a field, this
// fails to compile rather than silently emitting a scrambled gate list.
// g/u/v are long long to match circuit.h's uniGate/binGate (opt2b7-scale val[0]
// offsets exceed u32). wg_put_uni/wg_put_bin below still take `unsigned` device-side
// arguments deliberately: GPU witgen only builds partitioned (small) circuits, so
// the values stay well under u32 range and widen safely on assignment; only the
// STORAGE needed to widen, to match the host struct for the static_asserts below.
struct DevUniGate { long long g, u; int lu; __int128 sc; };
struct DevBinGate { long long g, u, v; unsigned char l; __int128 sc; };

static_assert(sizeof(DevUniGate) == sizeof(uniGate), "uniGate mirror size");
static_assert(sizeof(DevBinGate) == sizeof(binGate), "binGate mirror size");
static_assert(offsetof(DevUniGate, g)  == offsetof(uniGate, g),  "uniGate.g");
static_assert(offsetof(DevUniGate, u)  == offsetof(uniGate, u),  "uniGate.u");
static_assert(offsetof(DevUniGate, lu) == offsetof(uniGate, lu), "uniGate.lu");
static_assert(offsetof(DevUniGate, sc) == offsetof(uniGate, sc), "uniGate.sc");
static_assert(offsetof(DevBinGate, g)  == offsetof(binGate, g),  "binGate.g");
static_assert(offsetof(DevBinGate, u)  == offsetof(binGate, u),  "binGate.u");
static_assert(offsetof(DevBinGate, v)  == offsetof(binGate, v),  "binGate.v");
static_assert(offsetof(DevBinGate, l)  == offsetof(binGate, l),  "binGate.l");
static_assert(offsetof(DevBinGate, sc) == offsetof(binGate, sc), "binGate.sc");

__device__ inline void wg_put_uni(DevUniGate* dst, unsigned g, unsigned u,
                                  int lu, __int128 sc) {
    dst->g = g; dst->u = u; dst->lu = lu; dst->sc = sc;
}
__device__ inline void wg_put_bin(DevBinGate* dst, unsigned g, unsigned u,
                                  unsigned v, __int128 sc, unsigned char l) {
    dst->g = g; dst->u = u; dst->v = v; dst->sc = sc; dst->l = l;
}

struct GateEmitBuf {
    uniGate* d_uni = nullptr;  size_t cap_uni = 0;
    binGate* d_bin = nullptr;  size_t cap_bin = 0;
};

inline GateEmitBuf& gate_emit_buf() { static GateEmitBuf b; return b; }

inline void gate_emit_reserve(size_t nu, size_t nb) {
    // The previous layer's readback reads out of these same buffers, so it has
    // to land before this layer's kernels start writing them.
    witgen_sync_gates();
    GateEmitBuf& b = gate_emit_buf();
    if (nu > b.cap_uni) {
        if (b.d_uni) HIP_CHECK(hipFree(b.d_uni));
        HIP_CHECK(GKR_DMALLOC(&b.d_uni, nu * sizeof(uniGate)));
        b.cap_uni = nu;
        gkr_vram_log("witgen emit uni", (long)(nu * sizeof(uniGate)));
    }
    if (nb > b.cap_bin) {
        if (b.d_bin) HIP_CHECK(hipFree(b.d_bin));
        HIP_CHECK(GKR_DMALLOC(&b.d_bin, nb * sizeof(binGate)));
        b.cap_bin = nb;
        gkr_vram_log("witgen emit bin", (long)(nb * sizeof(binGate)));
    }
}

// Copy the emitted lists into the host vectors the prover reads.
//
// ASYNC ON PURPOSE. Nothing reads a layer's gate list until initSubset at the
// end of create() -- the builders' own assert sweeps read val[layer_id], not the
// gates -- so these copies are issued on the copy stream and drained once.
// Blocking here is costly: D2H into pageable host memory runs much slower than
// H2D, so the gates coming back are more expensive than the H2D they replaced.
//
// The buffer must not be reallocated before the drain; each layer's lists are
// sized once, here, and never touched again until initSubset.
inline void gate_emit_readback(std::vector<uniGate>& uni, size_t nu,
                               std::vector<binGate>& bin, size_t nb) {
    GateEmitBuf& b = gate_emit_buf();
    GateEvalTimers& T = gate_timers();
    const double t0 = wg_now_s();
    // resize() would need a default ctor these types do not have.
    if (uni.size() != nu) uni.assign(nu, uniGate(0, 0, 0, 0));
    if (bin.size() != nb) bin.assign(nb, binGate(0, 0, 0, 0, 0));
    if (nu) HIP_CHECK(hipMemcpyAsync(uni.data(), b.d_uni, nu * sizeof(uniGate),
                                     hipMemcpyDeviceToHost, gkr_copy_stream()));
    if (nb) HIP_CHECK(hipMemcpyAsync(bin.data(), b.d_bin, nb * sizeof(binGate),
                                     hipMemcpyDeviceToHost, gkr_copy_stream()));
    if (nu || nb) gate_d2h_pending() = true;
    T.d2h_s += wg_now_s() - t0;
    T.d2h_bytes += nu * sizeof(uniGate) + nb * sizeof(binGate);
}

// Core: evaluate from gate arrays that are ALREADY on the device, into a device
// output buffer. This is what a builder that emits its own gates calls -- its
// gates never cross PCIe on the way in, only on the way out (the host vectors
// are still what the prover reads).
inline void gpu_gate_eval_device(const uniGate* d_uni, size_t nu,
                                 const binGate* d_bin, size_t nb,
                                 const gpu_fe* d_val0, const gpu_fe* d_prev,
                                 gpu_fe* d_out, size_t sz) {
    GateEvalState& s = gate_state();
    GateEvalTimers& T = gate_timers();
    hipStream_t st = gkr_compute_stream();

    if (nu + nb == 0) {
        HIP_CHECK(hipMemsetAsync(d_out, 0, sz * sizeof(gpu_fe), st));
        HIP_CHECK(hipStreamSynchronize(st));
        return;
    }

    double t0 = wg_now_s();
    const size_t n_gate = nu + nb;
    wg_ensure((void**)&s.d_gg,   &s.cap_gg,   n_gate * sizeof(unsigned), "witgen csr g");
    wg_ensure((void**)&s.d_idx,  &s.cap_idx,  n_gate * sizeof(unsigned), "witgen csr idx");
    wg_ensure((void**)&s.d_off,  &s.cap_off,  (sz + 1) * sizeof(unsigned), "witgen csr off");
    wg_ensure((void**)&s.d_pos,  &s.cap_pos,  (sz + 1) * sizeof(unsigned), "witgen csr pos");
    const unsigned scratch_n = wg_scan_scratch((unsigned)(sz + 1));
    wg_ensure((void**)&s.d_scan, &s.cap_scan, scratch_n * sizeof(unsigned), "witgen scan");

    hipLaunchKernelGGL(wg_extract_g, dim3(wg_grid(n_gate)), dim3(WG_BLOCK), 0, st,
                       d_uni, (unsigned)nu, d_bin, (unsigned)nb, s.d_gg);
    HIP_CHECK(hipMemsetAsync(s.d_off, 0, (sz + 1) * sizeof(unsigned), st));
    hipLaunchKernelGGL(wg_count, dim3(wg_grid(n_gate)), dim3(WG_BLOCK), 0, st,
                       s.d_gg, (unsigned)n_gate, s.d_off);
    wg_exclusive_scan(s.d_off, (unsigned)(sz + 1), s.d_scan, st);
    HIP_CHECK(hipMemcpyAsync(s.d_pos, s.d_off, (sz + 1) * sizeof(unsigned),
                             hipMemcpyDeviceToDevice, st));
    hipLaunchKernelGGL(wg_scatter, dim3(wg_grid(n_gate)), dim3(WG_BLOCK), 0, st,
                       s.d_gg, (unsigned)n_gate, s.d_pos, s.d_idx);
    HIP_CHECK(hipStreamSynchronize(st));
    T.csr_s += wg_now_s() - t0;

    t0 = wg_now_s();
    hipLaunchKernelGGL(wg_reduce, dim3(wg_grid(sz)), dim3(WG_BLOCK), 0, st,
                       s.d_off, s.d_idx, (unsigned)sz, (unsigned)nu,
                       d_uni, d_bin, d_val0, d_prev, d_out);
    HIP_CHECK(hipStreamSynchronize(st));
    T.redu_s += wg_now_s() - t0;
}

// ===========================================================================
// 6. GELU checker layer 1 -- advice values AND gate emission, fused
// ===========================================================================
//
// Port of gelu_checker_layer1 (neuralNetwork.cpp). It is the widest builder in
// the build (len * linear_dim, 196608 elements on gpt2 len64), fully
// elementwise, with no reduction and no output collision -- so one thread owns
// one element and does BOTH its advice values and its 14 gates. Fusing them
// means qx is converted from the field once instead of twice and the element's
// val[0] offsets are computed once.
//
// FLOATING POINT IS THE WHOLE DIFFICULTY HERE. The host computes `inner`,
// `middle` and `final` in double and then rounds; the rounded integer becomes a
// witness value that a later positivity assert depends on, so a single ulp of
// disagreement is a wrong witness, not a rounding detail. Two rules:
//
//   1. `#pragma clang fp contract(off)` below. hipcc defaults to
//      contract(fast), and `a*b - c*d + e` contracted into FMAs changes the
//      last bit and can flip a tie in round(). The host is built by gcc at
//      -O2, which does NOT contract by default, so off is what matches.
//   2. Every mixed __int128/double expression keeps the host's exact
//      associativity and conversion points. `cb*cx*abs*pow(...)` computes
//      cb*cx*abs in __int128 and converts ONCE, at the multiply by pow -- it is
//      not the same as doing it in double, and abs can exceed 2^53.
//
// The powers of two arrive as precomputed doubles rather than being recomputed
// on device: ldexp(1.0, n) on the host is exactly the host's pow(2, n) for
// these exponents, and it sidesteps HIP's inexact device pow entirely.
//
// GATE SLOTS: 8 uni and 6 bin per surviving element, and `gp` is exactly the
// element's rank among survivors, so the slots are gp*8 and gp*6. That
// reproduces the host's emission order within each list, which initSubset
// depends on.

static __global__ void gelu1_fused_kernel(
        gpu_fe* __restrict__ val0,
        DevUniGate* __restrict__ ug, DevBinGate* __restrict__ bg,
        long long q_offset, long long orgsize,
        long long block_len, int len, int channel_out, int real_cn_out,
        __int128 C1, __int128 C2, __int128 C3, __int128 C4,
        __int128 C5, __int128 C6, __int128 C7,
        int ca, int cb, int cc, int cx, int cy,
        double P_A, double P_B, double P_C, double P_F) {
#pragma clang fp contract(off)
    const long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= block_len) return;
    if (g % channel_out >= real_cn_out) return;   // padding column: stays zero

    const long long y_off   = orgsize + 10;
    const long long abs_off = orgsize + 10 + block_len;
    const long long t_off   = orgsize + 10 + block_len * 2;
    const long long d1      = orgsize + 10 + block_len * 3;
    const long long d2      = d1 + (long long)len * real_cn_out;
    const long long d3      = d1 + (long long)len * real_cn_out * 2;
    const long long gp      = (g / channel_out) * real_cn_out + (g % channel_out);

    const __int128 qx = gpu_fe_to_i128(val0[g + q_offset]);

    const __int128 av = qx < 0 ? -qx : qx;
    val0[abs_off + g] = gpu_i64_to_fr((long long)av);

    const __int128 t = (C6 >= C7 * av) ? (__int128)1 : (__int128)0;
    val0[t_off + g] = gpu_i64_to_fr((long long)t);

    val0[d1 + gp] = gpu_i64_to_fr((long long)(av + 1));
    val0[d2 + gp] = gpu_i64_to_fr((long long)(t + (1 - 2 * t) * (C7 * av - C6)));

    // inner = (double)ca*cx*cx*qx*qx*2^(ea+2ex) - cb*cx*abs*2^(eb+ex) + cc*2^ec
    const double inner = ((double)ca * (double)cx * (double)cx
                          * (double)qx * (double)qx) * P_A
                       - (double)((__int128)(cb * cx) * av) * P_B
                       + (double)cc * P_C;
    // middle = (double)qx + abs - abs*t*inner
    const double middle = ((double)qx + (double)av) - (double)(av * t) * inner;
    // final  = (double)cx*2^(ex-1-ey)*middle/cy
    const double fin = ((double)cx * P_F * middle) / (double)cy;
    const __int128 y = (__int128)round(fin);
    val0[y_off + g] = gpu_i64_to_fr((long long)y);

    const __int128 term1 = (2 * y + 1) * C5 + 1 - C1 * qx - C1 * av
                         + C2 * t * av * av * av - C3 * t * av * av + C4 * t * av;
    const __int128 term2 = C1 * qx + C1 * av
                         - C2 * t * av * av * av + C3 * t * av * av - C4 * t * av
                         - (2 * y - 1) * C5 + 1;
    val0[d3 + gp] = device_mul(gpu_i64_to_fr((long long)term1),
                               gpu_i64_to_fr((long long)term2));

    if (!ug) return;   // values-only mode (gate emission still on the host)

    // ---- gates, in the host's emission order ----
    const unsigned c1_u    = (unsigned)orgsize;
    const unsigned q_off   = (unsigned)(g + q_offset);
    const unsigned abs_u   = (unsigned)(abs_off + g);
    const unsigned t_u     = (unsigned)(t_off + g);
    const unsigned d1_u    = (unsigned)(d1 + gp);
    const unsigned d2_u    = (unsigned)(d2 + gp);

    const unsigned bl      = (unsigned)block_len;
    const unsigned o_abst  = (unsigned)g;              // abs_mult_t_off
    const unsigned o_qsq   = (unsigned)g + bl;         // q_square_off
    const unsigned o_dt1   = (unsigned)g + bl * 2;
    const unsigned o_dt2   = (unsigned)g + bl * 3;
    const unsigned o_t2    = (unsigned)g + bl * 4;
    const unsigned o_absc  = (unsigned)g + bl * 5;

    DevUniGate* u = ug + gp * 8;
    DevBinGate* b = bg + gp * 6;

    wg_put_bin(b + 0, o_abst, abs_u, t_u,   1, 0);
    wg_put_bin(b + 1, o_qsq,  q_off, q_off, 1, 0);

    wg_put_uni(u + 0, o_dt1, abs_u, 0,  1);
    wg_put_uni(u + 1, o_dt1, c1_u,  0,  1);
    wg_put_uni(u + 2, o_dt1, d1_u,  0, -1);

    wg_put_uni(u + 3, o_dt2, c1_u, 0, -C6);
    wg_put_uni(u + 4, o_dt2, t_u,  0, 2 * C6 + 1);
    wg_put_uni(u + 5, o_dt2, abs_u, 0, C7);
    wg_put_bin(b + 2, o_dt2, abs_u, t_u, -2 * C7, 0);
    wg_put_uni(u + 6, o_dt2, d2_u, 0, -1);

    wg_put_bin(b + 3, o_t2, t_u, t_u, 1, 0);
    wg_put_uni(u + 7, o_t2, t_u, 0, -1);

    wg_put_bin(b + 4, o_absc, q_off, q_off,  1, 0);
    wg_put_bin(b + 5, o_absc, abs_u, abs_u, -1, 0);
}

// Fills val[0][orgsize .. orgsize+alloc) with gelu_checker_layer1's advice, and
// (when emit is true) this layer's gate lists.
// Pre: the mirror is coherent through orgsize and h_val0 has already been
// resized to orgsize+alloc. Post: BOTH the device mirror and h_val0 hold the
// block; the host's own zero-fill of the block is unnecessary and must be
// dropped (this writes the zeros too, via the grow memset).
inline void gpu_gelu1(FieldElement* h_val0, size_t orgsize, size_t alloc,
                      long long q_offset,
                      long long block_len, int len, int channel_out, int real_cn_out,
                      __int128 C1, __int128 C2, __int128 C3, __int128 C4,
                      __int128 C5, __int128 C6, __int128 C7,
                      int ca, int cb, int cc, int cx, int cy,
                      int ea, int eb, int ec, int ex, int ey,
                      bool emit,
                      std::vector<uniGate>* uni, std::vector<binGate>* bin) {
    const size_t n_after = orgsize + alloc;
    gpu_fe* d = witgen_val0_grow(h_val0, orgsize, n_after);

    // val[0][orgsize] = 1 (the layer's public constant). Written here rather
    // than on the host so the whole block travels back in one D2H.
    const FieldElement one = to_FE(1);
    HIP_CHECK(hipMemcpyAsync(d + orgsize, reinterpret_cast<const gpu_fe*>(&one),
                             sizeof(gpu_fe), hipMemcpyHostToDevice,
                             gkr_compute_stream()));

    const size_t survivors = (size_t)len * real_cn_out;
    const size_t nu = emit ? survivors * 8 : 0;
    const size_t nb = emit ? survivors * 6 : 0;
    if (emit) gate_emit_reserve(nu, nb);
    GateEmitBuf& emitbuf = gate_emit_buf();   // `eb` is taken: it is a GELU exponent

    const int threads = 256;
    const int nblocks = (int)((block_len + threads - 1) / threads);
    hipLaunchKernelGGL(gelu1_fused_kernel, dim3(nblocks), dim3(threads), 0,
                       gkr_compute_stream(),
                       d,
                       emit ? reinterpret_cast<DevUniGate*>(emitbuf.d_uni) : nullptr,
                       emit ? reinterpret_cast<DevBinGate*>(emitbuf.d_bin) : nullptr,
                       q_offset, (long long)orgsize, block_len,
                       len, channel_out, real_cn_out,
                       C1, C2, C3, C4, C5, C6, C7,
                       ca, cb, cc, cx, cy,
                       ldexp(1.0, ea + 2 * ex), ldexp(1.0, eb + ex),
                       ldexp(1.0, ec), ldexp(1.0, ex - 1 - ey));

    witgen_val0_readback(h_val0, orgsize, n_after);
    if (emit) gate_emit_readback(*uni, nu, *bin, nb);
}

// ===========================================================================
// 7. Gate-only builders
// ===========================================================================
//
// These layers emit gates and get their values from the evaluator; they have no
// advice loop of their own. Each kernel is one thread per surviving element
// writing that element's fixed number of gates at closed-form slots, in the
// host's emission order (see the note above §6 on why order is load-bearing).
//
// Where a builder emits both per-element and per-ROW gates, the host interleaves
// them (row r's elements, then row r's row-gates, then row r+1's), so the slot
// arithmetic is per-row strided rather than a flat rank. gelu2/gelu3/ln3/sm3
// are flat; ln2 and ln1 are strided.

// gelu_checker_layer2: 10 uni + 4 bin per surviving element, flat rank gp.
static __global__ void gelu2_gates_kernel(
        DevUniGate* __restrict__ ug, DevBinGate* __restrict__ bg,
        long long q_offset, long long orgsize, long long block_len,
        int channel_out, int real_cn_out, int layer_id,
        __int128 C1, __int128 C2, __int128 C3, __int128 C4, __int128 C5) {
    const long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= block_len) return;
    if (g % channel_out >= real_cn_out) return;
    const long long gp = (g / channel_out) * real_cn_out + (g % channel_out);

    const unsigned y_off   = (unsigned)(orgsize + 10 + g);
    const unsigned abs_off = (unsigned)(orgsize + 10 + block_len + g);
    const unsigned t_off   = (unsigned)(orgsize + 10 + block_len * 2 + g);
    const unsigned c1      = (unsigned)orgsize;
    const unsigned q_off   = (unsigned)(g + q_offset);
    const unsigned bl      = (unsigned)block_len;
    const unsigned term1   = (unsigned)g;
    const unsigned term2   = (unsigned)g + bl;
    const unsigned abs_t   = (unsigned)g;          // abs_mult_t_off, prev layer
    const unsigned q_sq    = (unsigned)g + bl;     // q_square_off, prev layer

    DevUniGate* u = ug + gp * 10;
    DevBinGate* b = bg + gp * 4;

    wg_put_uni(u + 0, term1, c1,    0, C5 + 1);
    wg_put_uni(u + 1, term1, y_off, 0, C5 * 2);
    wg_put_uni(u + 2, term1, abs_off, 0, -C1);
    wg_put_uni(u + 3, term1, q_off,   0, -C1);
    wg_put_uni(u + 4, term1, abs_t, layer_id - 1, C4);
    wg_put_bin(b + 0, term1, abs_t, q_sq, C2, 1);
    wg_put_bin(b + 1, term1, q_sq,  t_off, -C3, 2);

    wg_put_uni(u + 5, term2, c1,    0, C5 + 1);
    wg_put_uni(u + 6, term2, y_off, 0, -C5 * 2);
    wg_put_uni(u + 7, term2, abs_off, 0, C1);
    wg_put_uni(u + 8, term2, q_off,   0, C1);
    wg_put_uni(u + 9, term2, abs_t, layer_id - 1, -C4);
    wg_put_bin(b + 2, term2, abs_t, q_sq, -C2, 1);
    wg_put_bin(b + 3, term2, q_sq,  t_off, C3, 2);
}

// gelu_checker_layer3: 1 uni + 1 bin per surviving element.
static __global__ void gelu3_gates_kernel(
        DevUniGate* __restrict__ ug, DevBinGate* __restrict__ bg,
        long long orgsize, long long block_len, int len,
        int channel_out, int real_cn_out) {
    const long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= block_len) return;
    if (g % channel_out >= real_cn_out) return;
    const long long gp = (g / channel_out) * real_cn_out + (g % channel_out);
    const unsigned d3 = (unsigned)(orgsize + 10 + block_len * 3
                                   + 2 * (long long)real_cn_out * len + gp);
    wg_put_uni(ug + gp, (unsigned)gp, d3, 0, -1);
    wg_put_bin(bg + gp, (unsigned)gp, (unsigned)g, (unsigned)(g + block_len), 1, 1);
}

// ln_checker_layer3: 1 uni + 1 bin per (i, co) with co < real_cn_in.
static __global__ void ln3_gates_kernel(
        DevUniGate* __restrict__ ug, DevBinGate* __restrict__ bg,
        long long orgsize, long long block_len, int len,
        int channel_in, int real_cn_in) {
    const long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= (long long)len * real_cn_in) return;
    const long long i  = p / real_cn_in;
    const long long co = p % real_cn_in;
    const long long g  = i * channel_in + co;
    const unsigned d2 = (unsigned)(orgsize + 10 + block_len + (long long)len * 4 + p);
    wg_put_uni(ug + p, (unsigned)p, d2, 0, -1);
    wg_put_bin(bg + p, (unsigned)p, (unsigned)g, (unsigned)(g + block_len), 1, 1);
}

// softmax_layer_3: 1 uni + 1 bin per (head, i, j).
static __global__ void sm3_gates_kernel(
        DevUniGate* __restrict__ ug, DevBinGate* __restrict__ bg,
        long long orgsize, int HEAD, int len, int HSIZE,
        long long score_count) {
    const long long n = (long long)HEAD * len * HSIZE;
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const long long head = idx / ((long long)len * HSIZE);
    const long long rem  = idx % ((long long)len * HSIZE);
    const long long i    = rem / HSIZE;
    const long long j    = rem % HSIZE;

    const long long now = head * len * HSIZE + i * HSIZE + j;   // == idx
    const unsigned d3 = (unsigned)(orgsize + 10 + (long long)HEAD * 2 * len
                                   + 4 * (long long)HEAD * score_count
                                   + head * len * HSIZE + i * HSIZE + j);
    wg_put_bin(bg + idx, (unsigned)now, (unsigned)now,
               (unsigned)(now + (long long)HEAD * len * HSIZE), 1, 1);
    wg_put_uni(ug + idx, (unsigned)now, d3, 0, -1);
}

// multi_head_matrix_QK: HSIZE bin gates per score. This is the QK^T contraction
// -- the gate list IS the contraction, so the kernel only lays out its indices
// and the evaluator's CSR reduce performs the dot products.
//
// Both the sequence and non-sequence paths compute gate_j as
// (something that depends only on the key) + q_feature -- sequenceKVValueOffset
// is affine in feature_id, and the non-sequence form is
// q_offset + token*channel_out + attn_dim + q_feature. So the host precomputes
// one base per key and one per query row, and the kernel never has to know
// which path it is on. Same trick carries softmax 1's E.V block.
static __global__ void qk_gates_kernel(
        DevBinGate* __restrict__ bg,
        const long long* __restrict__ qbase,     // [len]      query row base
        const long long* __restrict__ kbase,     // [n_keys]   key base
        const int* __restrict__ row_of,          // [score_count] query row of an offset
        const long long* __restrict__ row_start, // [len]
        int HEAD, int HSIZE, long long score_count) {
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)HEAD * score_count * HSIZE;
    if (tid >= total) return;
    const long long k    = tid % HSIZE;
    const long long targ = tid / HSIZE;            // head*score_count + row_start + key
    const long long head = targ / score_count;
    const long long off  = targ % score_count;     // row_start + key

    const long long i   = row_of[off];
    const long long key = off - row_start[i];
    const long long q_feature = head * HSIZE + k;

    wg_put_bin(bg + tid, (unsigned)targ,
               (unsigned)(qbase[i] + q_feature),
               (unsigned)(kbase[key] + q_feature), 1, 0);
}

// ---- small index tables the sequence-aware kernels need ---------------------
// Rebuilt per layer (they are a few KB) and kept in one grow-only buffer.
struct SeqTables {
    long long *d_qbase = nullptr, *d_kbase = nullptr, *d_rowstart = nullptr;
    int       *d_rowof = nullptr;
    size_t cap_q = 0, cap_k = 0, cap_rs = 0, cap_ro = 0;
};
inline SeqTables& seq_tables() { static SeqTables t; return t; }

inline void seq_upload(const std::vector<long long>& qbase,
                       const std::vector<long long>& kbase,
                       const std::vector<long long>& row_start,
                       const std::vector<int>& row_of) {
    SeqTables& t = seq_tables();
    wg_ensure((void**)&t.d_qbase,    &t.cap_q,  qbase.size()     * sizeof(long long), "witgen qbase");
    wg_ensure((void**)&t.d_kbase,    &t.cap_k,  kbase.size()     * sizeof(long long), "witgen kbase");
    wg_ensure((void**)&t.d_rowstart, &t.cap_rs, row_start.size() * sizeof(long long), "witgen rowstart");
    wg_ensure((void**)&t.d_rowof,    &t.cap_ro, row_of.size()    * sizeof(int),       "witgen rowof");
    HIP_CHECK(gkr_h2d(t.d_qbase,    qbase.data(),     qbase.size()     * sizeof(long long)));
    HIP_CHECK(gkr_h2d(t.d_kbase,    kbase.data(),     kbase.size()     * sizeof(long long)));
    HIP_CHECK(gkr_h2d(t.d_rowstart, row_start.data(), row_start.size() * sizeof(long long)));
    HIP_CHECK(gkr_h2d(t.d_rowof,    row_of.data(),    row_of.size()    * sizeof(int)));
}

// ---- host wrappers for the gate-only builders -------------------------------
//
// Each emits into the device buffer, evaluates from it, D2Hs the layer output
// and the gate lists. `d_prev` is val[layer_id-1], already on the device.

inline void gpu_emit_and_eval(size_t nu, size_t nb, size_t sz,
                              FieldElement* h_out,
                              std::vector<uniGate>& uni, std::vector<binGate>& bin,
                              const gpu_fe* d_val0, const gpu_fe* d_prev) {
    GateEvalState& s = gate_state();
    GateEmitBuf& b = gate_emit_buf();
    GateEvalTimers& T = gate_timers();
    wg_ensure((void**)&s.d_out, &s.cap_out, sz * sizeof(gpu_fe), "witgen out");
    gpu_gate_eval_device(reinterpret_cast<const uniGate*>(b.d_uni), nu,
                         reinterpret_cast<const binGate*>(b.d_bin), nb,
                         d_val0, d_prev, s.d_out, sz);
    const double t0 = wg_now_s();
    HIP_CHECK(gkr_d2h(reinterpret_cast<gpu_fe*>(h_out), s.d_out, sz * sizeof(gpu_fe)));
    T.d2h_s += wg_now_s() - t0;
    T.d2h_bytes += sz * sizeof(gpu_fe);
    gate_emit_readback(uni, nu, bin, nb);
    T.layers++;
}

// roundLayer: 3 bin + 2 uni per element, flat (create() never passes a sparsity
// map to this builder; the host arm keeps the skip for the general case).
static __global__ void round_gates_kernel(
        DevUniGate* __restrict__ ug, DevBinGate* __restrict__ bg,
        long long orgsize, long long block_len,
        __int128 sc_pp, __int128 sc_pq, __int128 sc_qq, __int128 sc_c1) {
    const long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= block_len) return;
    const unsigned p  = (unsigned)g;
    const unsigned q  = (unsigned)(g + orgsize + 20);
    const unsigned c1 = (unsigned)orgsize;
    const unsigned sv = (unsigned)(q + block_len);

    DevBinGate* b = bg + g * 3;
    DevUniGate* u = ug + g * 2;
    wg_put_bin(b + 0, (unsigned)g, p, p, sc_pp, 1);
    wg_put_bin(b + 1, (unsigned)g, p, q, sc_pq, 2);
    wg_put_bin(b + 2, (unsigned)g, q, q, sc_qq, 0);
    wg_put_uni(u + 0, (unsigned)g, c1, 0, sc_c1);
    wg_put_uni(u + 1, (unsigned)g, sv, 0, -1);
}

// ln_checker_layer2: 7 bin + 4 uni per surviving element, then 1 bin + 3 uni per
// surviving ROW -- and the host interleaves them (row r's elements, then row r's
// row-gates), so the slots are row-strided rather than a flat rank.
// One thread per (row, col) plus one per row.
static __global__ void ln2_gates_kernel(
        DevUniGate* __restrict__ ug, DevBinGate* __restrict__ bg,
        long long orgsize, long long block_len, int channel_in,
        int real_row, int real_col, int len,
        long long qw_off, long long qb_off,
        __int128 sc_a, __int128 sc_b, __int128 sc_y, __int128 sc_sig) {
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long n_elem = (long long)real_row * real_col;
    if (tid >= n_elem + real_row) return;

    const long long bin_stride = (long long)real_col * 7 + 1;
    const long long uni_stride = (long long)real_col * 4 + 3;
    const long long bl = block_len;

    if (tid < n_elem) {
        const long long i  = tid / real_col;
        const long long co = tid % real_col;
        const long long g  = i * channel_in + co;
        const unsigned sig = (unsigned)(orgsize + 10 + bl + (long long)len * 2 + i);
        const unsigned y   = (unsigned)(orgsize + 10 + g);

        DevBinGate* b = bg + i * bin_stride + co * 7;
        DevUniGate* u = ug + i * uni_stride + co * 4;

        wg_put_bin(b + 0, (unsigned)g, (unsigned)g, (unsigned)(qw_off + co), -sc_a, 2);
        wg_put_bin(b + 1, (unsigned)g, sig, (unsigned)(qb_off + co), -sc_b, 0);
        wg_put_bin(b + 2, (unsigned)g, sig, y, sc_y, 0);
        wg_put_uni(u + 0, (unsigned)g, (unsigned)orgsize, 0, 1);
        wg_put_uni(u + 1, (unsigned)g, sig, 0, sc_sig);

        const unsigned g2 = (unsigned)(g + bl);
        wg_put_bin(b + 3, g2, (unsigned)g, (unsigned)(qw_off + co), sc_a, 2);
        wg_put_bin(b + 4, g2, sig, (unsigned)(qb_off + co), sc_b, 0);
        wg_put_bin(b + 5, g2, sig, y, -sc_y, 0);
        wg_put_uni(u + 2, g2, (unsigned)orgsize, 0, 1);
        wg_put_uni(u + 3, g2, sig, 0, sc_sig);

        wg_put_bin(b + 6, (unsigned)(2 * bl + len + i), (unsigned)g, (unsigned)g, 1, 1);
    } else {
        const long long i = tid - n_elem;
        DevBinGate* b = bg + i * bin_stride + (long long)real_col * 7;
        DevUniGate* u = ug + i * uni_stride + (long long)real_col * 4;
        const unsigned b_off  = (unsigned)(orgsize + 10 + bl + len + i);
        const unsigned d1_off = (unsigned)(orgsize + 10 + bl + (long long)len * 3 + i);

        wg_put_bin(b + 0, (unsigned)(2 * bl + i),
                   (unsigned)(bl + i), (unsigned)(bl + len + i), 1, 1);
        wg_put_uni(u + 0, (unsigned)(2 * bl + i), d1_off, 0, -1);
        wg_put_uni(u + 1, (unsigned)(2 * bl + len + i), (unsigned)orgsize, 0, 1);
        wg_put_uni(u + 2, (unsigned)(2 * bl + len + i), b_off, 0, -1);
    }
}

// ===========================================================================
// 8. Softmax 1 and 2
// ===========================================================================
//
// softmax_layer_1's advice is the one genuinely reduction-shaped loop besides
// LayerNorm: per (head, query row) a max over the row's keys, then per key a
// rounded scale, a lookup in the 655360-entry exp table, and a running row sum.
// One thread per row -- HEAD*len is only 768 rows on gpt2 len64 and each walks
// at most `len` keys, so a block-per-row reduction would cost more in launch
// overhead than it saves.
//
// The exp table is built on the HOST (compute_e_table uses libm exp) and
// uploaded once. Computing it on device risks a different last bit, and a
// wrong table is a wrong witness.

struct ExpTable { int* d = nullptr; size_t cap = 0; };
inline ExpTable& exp_table() { static ExpTable t; return t; }

inline int* exp_table_upload(const int* host, size_t n) {
    ExpTable& t = exp_table();
    if (t.cap < n) {
        if (t.d) HIP_CHECK(hipFree(t.d));
        HIP_CHECK(GKR_DMALLOC(&t.d, n * sizeof(int)));
        t.cap = n;
        gkr_vram_log("witgen exp table", (long)(n * sizeof(int)));
        HIP_CHECK(gkr_h2d(t.d, host, n * sizeof(int)));
    }
    return t.d;
}

static __global__ void sm1_values_kernel(
        gpu_fe* __restrict__ val0, const gpu_fe* __restrict__ prev,
        const int* __restrict__ table,
        const long long* __restrict__ row_start,
        const long long* __restrict__ key_count,
        long long orgsize, int HEAD, int len, long long score_count,
        int c1, int e1, int eprime, double P1, double P2) {
#pragma clang fp contract(off)
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)HEAD * len) return;
    const long long head = tid / len;
    const long long i    = tid % len;

    const long long sum_E = orgsize + 10 + head * len + i;
    const long long pmax  = orgsize + 10 + (long long)HEAD * len + head * len + i;
    const long long base  = orgsize + 10 + (long long)HEAD * 2 * len;
    const long long rs = row_start[i], kc = key_count[i];

    __int128 mx = 0;
    for (long long key = 0; key < kc; ++key) {
        const __int128 S = gpu_fe_to_i128(prev[head * score_count + rs + key]);
        if (S > mx) mx = S;
    }
    const gpu_fe fr_mx = gpu_i64_to_fr((long long)mx);
    val0[pmax] = fr_mx;

    __int128 sumE = 0;
    for (long long key = 0; key < kc; ++key) {
        const long long offset = head * score_count + rs + key;
        const gpu_fe dt1 = device_sub(fr_mx, prev[offset]);
        val0[base + offset] = dt1;
        const __int128 pj = gpu_fe_to_i128(dt1);
        // tj = round(c1 * 2^(e1+eprime+1) * pj / 2^(eprime+1))
        const __int128 tj = (__int128)round(((double)c1 * P1 * (double)pj) / P2);
        val0[base + 2 * (long long)HEAD * score_count + offset] =
            gpu_i64_to_fr((long long)tj);
        const int ev = (tj < 655360 && tj >= 0) ? table[(long long)tj] : 1;
        val0[base + 3 * (long long)HEAD * score_count + offset] = gpu_i64_to_fr(ev);

        const __int128 sh_e = (__int128)1 << eprime;
        const __int128 sh_e1 = (__int128)1 << (eprime + 1);
        const __int128 mul = (__int128)c1 * ((__int128)1 << (e1 + eprime + 1));
        val0[base + (long long)HEAD * score_count + offset] =
            device_mul(gpu_i64_to_fr((long long)(mul * pj + sh_e - tj * sh_e1)),
                       gpu_i64_to_fr((long long)(-mul * pj + sh_e + tj * sh_e1)));
        sumE += ev;
    }
    // The host accumulates sumE as a field element from zero, one non-negative
    // int at a time, so the total converted once is the same element.
    val0[sum_E] = gpu_i64_to_fr((long long)sumE);
}

// softmax_layer_2: elementwise over (head, i, j) with sumE broadcast per row,
// and its gates in the same pass. Writes land INSIDE softmax_layer_1's advice
// block, which the mirror already considers synced -- the caller reads that
// range back rather than pushing the host's copy forward.
static __global__ void sm2_fused_kernel(
        gpu_fe* __restrict__ val0, const gpu_fe* __restrict__ prev,
        DevUniGate* __restrict__ ug, DevBinGate* __restrict__ bg,
        long long orgsize, int HEAD, int len, int HSIZE, int channel_in,
        long long score_count, int layer_id,
        int c1, int e1, int eprime, double P_e1) {
#pragma clang fp contract(off)
    const long long n = (long long)HEAD * len * HSIZE;
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    const long long head = tid / ((long long)len * HSIZE);
    const long long rem  = tid % ((long long)len * HSIZE);
    const long long i    = rem / HSIZE;
    const long long j    = rem % HSIZE;

    const long long sum_E  = orgsize + 10 + head * len + i;
    const long long aux    = orgsize + 10 + (long long)HEAD * 2 * len
                           + 4 * (long long)HEAD * score_count;
    const long long out_ij = head * (long long)len * HSIZE + i * HSIZE + j;
    const long long s_ij   = aux + (long long)HEAD * len * HSIZE
                           + i * channel_in + head * HSIZE + j;
    const long long d3     = aux + head * (long long)len * HSIZE + i * HSIZE + j;

    const __int128 sumE = gpu_fe_to_i128(val0[sum_E]);
    const __int128 Qij  = gpu_fe_to_i128(prev[out_ij]);
    const __int128 S = (__int128)round(((double)(Qij * c1) * P_e1) / (double)sumE);
    val0[s_ij] = gpu_i64_to_fr((long long)S);

    const __int128 he  = (__int128)1 << (eprime - 1);
    const __int128 hee = (__int128)1 << eprime;
    const __int128 mul = (__int128)c1 * ((__int128)1 << (eprime + e1));
    val0[d3] = device_mul(
        gpu_i64_to_fr((long long)(sumE * he + Qij * mul - S * sumE * hee)),
        gpu_i64_to_fr((long long)(sumE * he - Qij * mul + S * sumE * hee)));

    if (!ug) return;
    const unsigned t1 = (unsigned)out_ij;
    const unsigned t2 = (unsigned)(out_ij + (long long)HEAD * len * HSIZE);
    const unsigned sE = (unsigned)sum_E;
    DevUniGate* u = ug + tid * 4;
    DevBinGate* b = bg + tid * 2;
    const __int128 sc1 = (__int128)c1 * ((__int128)1 << (e1 + eprime));
    wg_put_uni(u + 0, t1, (unsigned)out_ij, layer_id - 1, sc1);
    wg_put_uni(u + 1, t1, sE, 0, (__int128)1 << (eprime - 1));
    wg_put_bin(b + 0, t1, (unsigned)s_ij, sE, -((__int128)1 << eprime), 0);
    wg_put_uni(u + 2, t2, (unsigned)out_ij, layer_id - 1, -sc1);
    wg_put_uni(u + 3, t2, sE, 0, (__int128)1 << (eprime - 1));
    wg_put_bin(b + 1, t2, (unsigned)s_ij, sE, (__int128)1 << eprime, 0);
}

// softmax_layer_2's second gate loop: one per (head, i, key), appended after the
// whole first loop in both lists.
static __global__ void sm2_tail_gates_kernel(
        DevUniGate* __restrict__ ug, DevBinGate* __restrict__ bg,
        long long uni_base, long long bin_base,
        long long orgsize, int HEAD, int len, int HSIZE,
        long long score_count) {
    const long long n = (long long)HEAD * score_count;
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    const long long offset = tid;                       // head*score_count + rs + key
    const long long hlh = (long long)HEAD * len * HSIZE;
    const unsigned now = (unsigned)(2 * hlh + offset);
    const unsigned dt2 = (unsigned)(orgsize + 10 + (long long)HEAD * 2 * len
                                    + (long long)HEAD * score_count + offset);
    wg_put_bin(bg + bin_base + tid, now,
               (unsigned)(hlh + offset),
               (unsigned)((long long)HEAD * score_count + hlh + offset), 1, 1);
    wg_put_uni(ug + uni_base + tid, now, dt2, 0, -1);
}

// softmax_layer_1's gates come in two blocks per head: first the whole E.V
// product (bin gates only), then the per-key checks (uni gates only). Within a
// head the E.V loop is i-major then j then key, so a gate's slot is
// HSIZE*row_start(i) + j*key_count(i) + key. Like QK, this block is a
// contraction expressed as a gate list.
static __global__ void sm1_ev_gates_kernel(
        DevBinGate* __restrict__ bg,
        const long long* __restrict__ vbase,
        const long long* __restrict__ row_start,
        const long long* __restrict__ key_count,
        const int* __restrict__ row_of,
        long long orgsize, int HEAD, int len, int HSIZE, long long score_count) {
    const long long per_head = (long long)HSIZE * score_count;
    const long long total = (long long)HEAD * per_head;
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const long long head = tid / per_head;
    long long r = tid % per_head;               // HSIZE*row_start(i) + j*kc + key

    // Recover i from r: r / HSIZE lands inside row i's score range.
    const long long i  = row_of[r / HSIZE];
    const long long rs = row_start[i], kc = key_count[i];
    const long long within = r - (long long)HSIZE * rs;
    const long long j   = within / kc;
    const long long key = within % kc;

    const long long out_ij = head * (long long)len * HSIZE + i * HSIZE + j;
    const long long E_off  = orgsize + 10 + (long long)HEAD * 2 * len
                           + 3 * (long long)HEAD * score_count
                           + head * score_count + rs + key;
    wg_put_bin(bg + tid, (unsigned)out_ij,
               (unsigned)(vbase[key] + head * HSIZE + j), (unsigned)E_off, 1, 0);
}

// The check block: 9 uni per (head, i, key), then 1 + key_count(i) per row.
static __global__ void sm1_check_gates_kernel(
        DevUniGate* __restrict__ ug,
        const long long* __restrict__ row_start,
        const long long* __restrict__ key_count,
        const int* __restrict__ row_of,
        long long orgsize, int HEAD, int len, int HSIZE,
        long long score_count, int layer_id,
        int c1, int e1, int eprime) {
    const long long per_head = 10 * score_count + len;
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    // one thread per (head, score) for the 9-gate group, plus one per (head,row)
    const long long n9 = (long long)HEAD * score_count;
    const long long nr = (long long)HEAD * len;
    if (tid >= n9 + nr) return;

    const long long base = orgsize + 10 + (long long)HEAD * 2 * len;
    const long long hlh  = (long long)HEAD * len * HSIZE;
    const __int128 sc_t1 = (__int128)c1 * ((__int128)1 << (e1 + 1 + eprime));
    const __int128 sh_e  = (__int128)1 << eprime;
    const __int128 sh_e1 = (__int128)1 << (eprime + 1);

    if (tid < n9) {
        const long long head = tid / score_count;
        const long long off  = tid % score_count;         // rs + key
        const long long i    = row_of[off];
        const long long rs   = row_start[i];
        const long long key  = off - rs;
        const long long offset = head * score_count + off;

        const unsigned dt1  = (unsigned)(base + offset);
        const unsigned t_of = (unsigned)(base + 2 * (long long)HEAD * score_count + offset);
        const unsigned pmax = (unsigned)(orgsize + 10 + (long long)HEAD * len
                                         + head * len + i);
        const unsigned chk  = (unsigned)(2 * (long long)HEAD * score_count + hlh + offset);
        const unsigned t1   = (unsigned)(hlh + offset);
        const unsigned t2   = (unsigned)((long long)HEAD * score_count + hlh + offset);

        DevUniGate* u = ug + head * per_head + 10 * rs + i + key * 9;
        wg_put_uni(u + 0, chk, dt1, 0, -1);
        wg_put_uni(u + 1, chk, (unsigned)offset, layer_id - 1, -1);
        wg_put_uni(u + 2, chk, pmax, 0, 1);
        wg_put_uni(u + 3, t1, dt1, 0, sc_t1);
        wg_put_uni(u + 4, t1, (unsigned)orgsize, 0, sh_e);
        wg_put_uni(u + 5, t1, t_of, 0, -sh_e1);
        wg_put_uni(u + 6, t2, dt1, 0, -sc_t1);
        wg_put_uni(u + 7, t2, (unsigned)orgsize, 0, sh_e);
        wg_put_uni(u + 8, t2, t_of, 0, sh_e1);
    } else {
        const long long k    = tid - n9;
        const long long head = k / len;
        const long long i    = k % len;
        const long long rs = row_start[i], kc = key_count[i];
        const unsigned sumE = (unsigned)(orgsize + 10 + head * len + i);
        const unsigned sec  = (unsigned)(3 * (long long)HEAD * score_count + hlh
                                         + head * len + i);
        DevUniGate* u = ug + head * per_head + 10 * rs + i + 9 * kc;
        wg_put_uni(u + 0, sec, sumE, 0, -1);
        for (long long key = 0; key < kc; ++key) {
            const unsigned E_off = (unsigned)(base + 3 * (long long)HEAD * score_count
                                              + head * score_count + rs + key);
            wg_put_uni(u + 1 + key, sec, E_off, 0, 1);
        }
    }
}

// ===========================================================================
// 9. LayerNorm checker layer 1
// ===========================================================================
//
// Per token row: a sum over ALL channel_in inputs, then B = 1 + sum of squares
// of the centred survivors, then sigma = round(sqrt(B)), then a per-element
// pass. One BLOCK per row with two LDS reductions -- one thread per row would
// be 64 threads on gpt2 len64, a single wavefront for the whole layer.
//
// qx is recomputed in each phase rather than held: channel_in is up to 3072
// __int128, which does not fit in LDS, and gpu_fe_to_i128 is cheap next to the
// global load it already needs.
//
// sqrt(__int128) on the host converts to double and calls sqrt(double); above
// 2^53 that conversion loses bits, but BOTH arms lose the same bits, so they
// agree. Same reasoning as GELU 1: reproduce the host's conversion points, do
// not "improve" them.

__device__ inline __int128 ln1_block_reduce(__int128 v, __int128* smem) {
    const unsigned t = threadIdx.x;
    smem[t] = v;
    __syncthreads();
    for (unsigned s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if (t < s2) smem[t] += smem[t + s2];
        __syncthreads();
    }
    return smem[0];
}

static __global__ void ln1_values_kernel(
        gpu_fe* __restrict__ val0,
        long long q_offset, long long orgsize, long long block_len,
        int len, int channel_in, int real_row, int real_col, int real_cn_in,
        long long qw_off, long long qb_off,
        int c1, int e1, int c2, int e2, int m,
        double P1, double P2) {
#pragma clang fp contract(off)
    __shared__ __int128 smem[256];
    const int i = blockIdx.x;
    if (i >= real_row) return;
    const unsigned t = threadIdx.x, nt = blockDim.x;

    // sum over ALL channel_in (the host does not skip here)
    __int128 part = 0;
    for (int co = t; co < channel_in; co += nt)
        part += gpu_fe_to_i128(val0[q_offset + (long long)i * channel_in + co]);
    const __int128 sum = ln1_block_reduce(part, smem);
    __syncthreads();

    // B = 1 + sum over survivors of a[co]^2
    part = 0;
    for (int co = t; co < real_col; co += nt) {
        const __int128 qx = gpu_fe_to_i128(val0[q_offset + (long long)i * channel_in + co]);
        const __int128 a = qx * real_cn_in - sum;
        part += a * a;
    }
    __int128 B = ln1_block_reduce(part, smem) + 1;
    __syncthreads();

    const __int128 sigma = (__int128)round(sqrt((double)B));

    const long long sum_off = orgsize + 10 + block_len + i;
    const long long b_off   = orgsize + 10 + block_len + len + i;
    const long long sig_off = orgsize + 10 + block_len + (long long)len * 2 + i;
    const long long d1_off  = orgsize + 10 + block_len + (long long)len * 3 + i;

    for (int co = t; co < real_col; co += nt) {
        const long long g = (long long)i * channel_in + co;
        const __int128 qx = gpu_fe_to_i128(val0[q_offset + g]);
        const __int128 a  = qx * real_cn_in - sum;
        const __int128 qw = gpu_fe_to_i128(val0[qw_off + co]);
        const __int128 qb = gpu_fe_to_i128(val0[qb_off + co]);

        const __int128 qy = (__int128)round(
            ((((P1 * c1) * (double)qw) * (double)a) / (double)sigma)
            + ((P2 * c2) * (double)qb));
        val0[orgsize + 10 + g] = gpu_i64_to_fr((long long)qy);

        const __int128 s1 = (__int128)1 << (m - 1);
        const __int128 t1 = (__int128)c1 * ((__int128)1 << (e1 + m)) * qw * a;
        const __int128 t2 = (__int128)c2 * ((__int128)1 << (e2 + m)) * qb * sigma;
        const __int128 term1 = (2 * qy + 1) * s1 * sigma + 1 - t1 - t2;
        const __int128 term2 = t1 + t2 - (2 * qy - 1) * s1 * sigma + 1;
        const long long d2 = orgsize + 10 + block_len + (long long)len * 4
                           + (long long)i * real_cn_in + co;
        val0[d2] = device_mul(gpu_i64_to_fr((long long)term1),
                              gpu_i64_to_fr((long long)term2));
    }

    if (t == 0) {
        val0[sum_off] = gpu_i64_to_fr((long long)sum);
        val0[b_off]   = gpu_i64_to_fr((long long)B);
        val0[sig_off] = gpu_i64_to_fr((long long)sigma);
        val0[d1_off]  = device_mul(
            gpu_i64_to_fr((long long)(sigma * sigma + sigma + 1 - B)),
            gpu_i64_to_fr((long long)(B - sigma * sigma + sigma)));
    }
}

// ln1's gates: 3 uni per surviving element, then 6 uni + 2 bin per row, with the
// host interleaving them per row -- so uni stride real_col*3+6, bin stride 2.
static __global__ void ln1_gates_kernel(
        DevUniGate* __restrict__ ug, DevBinGate* __restrict__ bg,
        long long q_offset, long long orgsize, long long block_len,
        int len, int channel_in, int real_row, int real_col, int real_cn_in) {
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long n_elem = (long long)real_row * real_col;
    if (tid >= n_elem + real_row) return;

    const long long us = (long long)real_col * 3 + 6;
    const long long bl = block_len;

    if (tid < n_elem) {
        const long long i  = tid / real_col;
        const long long co = tid % real_col;
        const long long g  = i * channel_in + co;
        const unsigned sum_off = (unsigned)(orgsize + 10 + bl + i);
        DevUniGate* u = ug + i * us + co * 3;
        wg_put_uni(u + 0, (unsigned)g, (unsigned)(q_offset + g), 0, real_cn_in);
        wg_put_uni(u + 1, (unsigned)g, sum_off, 0, -1);
        wg_put_uni(u + 2, (unsigned)(bl + (long long)len * 2 + i),
                   (unsigned)(q_offset + g), 0, 1);
    } else {
        const long long i = tid - n_elem;
        const unsigned sum_off = (unsigned)(orgsize + 10 + bl + i);
        const unsigned b_off   = (unsigned)(orgsize + 10 + bl + len + i);
        const unsigned sig_off = (unsigned)(orgsize + 10 + bl + (long long)len * 2 + i);
        DevUniGate* u = ug + i * us + (long long)real_col * 3;
        DevBinGate* b = bg + i * 2;
        wg_put_bin(b + 0, (unsigned)(bl + i), sig_off, sig_off, 1, 0);
        wg_put_uni(u + 0, (unsigned)(bl + i), sig_off, 0, 1);
        wg_put_uni(u + 1, (unsigned)(bl + i), (unsigned)orgsize, 0, 1);
        wg_put_uni(u + 2, (unsigned)(bl + i), b_off, 0, -1);
        wg_put_bin(b + 1, (unsigned)(bl + len + i), sig_off, sig_off, -1, 0);
        wg_put_uni(u + 3, (unsigned)(bl + len + i), sig_off, 0, 1);
        wg_put_uni(u + 4, (unsigned)(bl + len + i), b_off, 0, 1);
        wg_put_uni(u + 5, (unsigned)(bl + (long long)len * 2 + i), sum_off, 0, -1);
    }
}

// Resolve this layer's two source buffers: the val[0] mirror, and
// val[layer_id-1]. Mirrors gpu_calc_normal_layer's handling, including the
// aliased case where layer_id-1 == 0.
inline void gpu_layer_sources(const FieldElement* h_val0, size_t n0,
                              const FieldElement* h_prev, size_t n_prev,
                              gpu_fe** d_val0, gpu_fe** d_prev) {
    GateEvalState& s = gate_state();
    GateEvalTimers& T = gate_timers();
    double t0 = wg_now_s();
    *d_val0 = witgen_val0_sync(h_val0, n0);
    T.val0_s += wg_now_s() - t0;

    t0 = wg_now_s();
    *d_prev = *d_val0;
    if (h_prev && n_prev && h_prev != h_val0) {
        wg_ensure((void**)&s.d_prev, &s.cap_prev, n_prev * sizeof(gpu_fe), "witgen prev");
        GkrXferTimer _x(&gkr_xfer_h2d_s_ref());
        HIP_CHECK(gkr_stage_h2d(s.d_prev, reinterpret_cast<const gpu_fe*>(h_prev),
                                n_prev * sizeof(gpu_fe), gkr_copy_stream()));
        HIP_CHECK(hipStreamSynchronize(gkr_copy_stream()));
        *d_prev = s.d_prev;
        T.prev_bytes += n_prev * sizeof(gpu_fe);
    }
    T.prev_s += wg_now_s() - t0;
}

inline void gpu_gelu2(const FieldElement* h_val0, size_t n0,
                      const FieldElement* h_prev, size_t n_prev,
                      FieldElement* h_out, size_t sz,
                      std::vector<uniGate>& uni, std::vector<binGate>& bin,
                      long long q_offset, long long orgsize, long long block_len,
                      int len, int channel_out, int real_cn_out, int layer_id,
                      __int128 C1, __int128 C2, __int128 C3, __int128 C4, __int128 C5) {
    gpu_fe *d_val0, *d_prev;
    gpu_layer_sources(h_val0, n0, h_prev, n_prev, &d_val0, &d_prev);
    const size_t surv = (size_t)len * real_cn_out;
    const size_t nu = surv * 10, nb = surv * 4;
    gate_emit_reserve(nu, nb);
    GateEmitBuf& b = gate_emit_buf();
    const int nblocks = (int)((block_len + WG_BLOCK - 1) / WG_BLOCK);
    hipLaunchKernelGGL(gelu2_gates_kernel, dim3(nblocks), dim3(WG_BLOCK), 0,
                       gkr_compute_stream(),
                       reinterpret_cast<DevUniGate*>(b.d_uni),
                       reinterpret_cast<DevBinGate*>(b.d_bin),
                       q_offset, orgsize, block_len, channel_out, real_cn_out,
                       layer_id, C1, C2, C3, C4, C5);
    gpu_emit_and_eval(nu, nb, sz, h_out, uni, bin, d_val0, d_prev);
}

inline void gpu_gelu3(const FieldElement* h_val0, size_t n0,
                      const FieldElement* h_prev, size_t n_prev,
                      FieldElement* h_out, size_t sz,
                      std::vector<uniGate>& uni, std::vector<binGate>& bin,
                      long long orgsize, long long block_len,
                      int len, int channel_out, int real_cn_out) {
    gpu_fe *d_val0, *d_prev;
    gpu_layer_sources(h_val0, n0, h_prev, n_prev, &d_val0, &d_prev);
    const size_t surv = (size_t)len * real_cn_out;
    gate_emit_reserve(surv, surv);
    GateEmitBuf& b = gate_emit_buf();
    const int nblocks = (int)((block_len + WG_BLOCK - 1) / WG_BLOCK);
    hipLaunchKernelGGL(gelu3_gates_kernel, dim3(nblocks), dim3(WG_BLOCK), 0,
                       gkr_compute_stream(),
                       reinterpret_cast<DevUniGate*>(b.d_uni),
                       reinterpret_cast<DevBinGate*>(b.d_bin),
                       orgsize, block_len, len, channel_out, real_cn_out);
    gpu_emit_and_eval(surv, surv, sz, h_out, uni, bin, d_val0, d_prev);
}

inline void gpu_ln3(const FieldElement* h_val0, size_t n0,
                    const FieldElement* h_prev, size_t n_prev,
                    FieldElement* h_out, size_t sz,
                    std::vector<uniGate>& uni, std::vector<binGate>& bin,
                    long long orgsize, long long block_len,
                    int len, int channel_in, int real_cn_in) {
    gpu_fe *d_val0, *d_prev;
    gpu_layer_sources(h_val0, n0, h_prev, n_prev, &d_val0, &d_prev);
    const size_t n = (size_t)len * real_cn_in;
    gate_emit_reserve(n, n);
    GateEmitBuf& b = gate_emit_buf();
    const int nblocks = (int)((n + WG_BLOCK - 1) / WG_BLOCK);
    hipLaunchKernelGGL(ln3_gates_kernel, dim3(nblocks), dim3(WG_BLOCK), 0,
                       gkr_compute_stream(),
                       reinterpret_cast<DevUniGate*>(b.d_uni),
                       reinterpret_cast<DevBinGate*>(b.d_bin),
                       orgsize, block_len, len, channel_in, real_cn_in);
    gpu_emit_and_eval(n, n, sz, h_out, uni, bin, d_val0, d_prev);
}

inline void gpu_sm3(const FieldElement* h_val0, size_t n0,
                    const FieldElement* h_prev, size_t n_prev,
                    FieldElement* h_out, size_t sz,
                    std::vector<uniGate>& uni, std::vector<binGate>& bin,
                    long long orgsize, int HEAD, int len, int HSIZE,
                    long long score_count) {
    gpu_fe *d_val0, *d_prev;
    gpu_layer_sources(h_val0, n0, h_prev, n_prev, &d_val0, &d_prev);
    const size_t n = (size_t)HEAD * len * HSIZE;
    gate_emit_reserve(n, n);
    GateEmitBuf& b = gate_emit_buf();
    const int nblocks = (int)((n + WG_BLOCK - 1) / WG_BLOCK);
    hipLaunchKernelGGL(sm3_gates_kernel, dim3(nblocks), dim3(WG_BLOCK), 0,
                       gkr_compute_stream(),
                       reinterpret_cast<DevUniGate*>(b.d_uni),
                       reinterpret_cast<DevBinGate*>(b.d_bin),
                       orgsize, HEAD, len, HSIZE, score_count);
    gpu_emit_and_eval(n, n, sz, h_out, uni, bin, d_val0, d_prev);
}

inline void gpu_qk(const FieldElement* h_val0, size_t n0,
                   const FieldElement* h_prev, size_t n_prev,
                   FieldElement* h_out, size_t sz,
                   std::vector<uniGate>& uni, std::vector<binGate>& bin,
                   const std::vector<long long>& qbase,
                   const std::vector<long long>& kbase,
                   const std::vector<long long>& row_start,
                   const std::vector<int>& row_of,
                   int HEAD, int HSIZE, long long score_count) {
    gpu_fe *d_val0, *d_prev;
    gpu_layer_sources(h_val0, n0, h_prev, n_prev, &d_val0, &d_prev);
    seq_upload(qbase, kbase, row_start, row_of);
    SeqTables& t = seq_tables();
    const size_t nb = (size_t)HEAD * score_count * HSIZE;
    gate_emit_reserve(0, nb);
    GateEmitBuf& b = gate_emit_buf();
    const int nblocks = (int)((nb + WG_BLOCK - 1) / WG_BLOCK);
    hipLaunchKernelGGL(qk_gates_kernel, dim3(nblocks), dim3(WG_BLOCK), 0,
                       gkr_compute_stream(),
                       reinterpret_cast<DevBinGate*>(b.d_bin),
                       t.d_qbase, t.d_kbase, t.d_rowof, t.d_rowstart,
                       HEAD, HSIZE, score_count);
    gpu_emit_and_eval(0, nb, sz, h_out, uni, bin, d_val0, d_prev);
}

inline void gpu_round_gates(const FieldElement* h_val0, size_t n0,
                            const FieldElement* h_prev, size_t n_prev,
                            FieldElement* h_out, size_t sz,
                            std::vector<uniGate>& uni, std::vector<binGate>& bin,
                            long long orgsize, long long block_len,
                            __int128 sc_pp, __int128 sc_pq, __int128 sc_qq,
                            __int128 sc_c1) {
    gpu_fe *d_val0, *d_prev;
    gpu_layer_sources(h_val0, n0, h_prev, n_prev, &d_val0, &d_prev);
    const size_t nu = (size_t)block_len * 2, nb = (size_t)block_len * 3;
    gate_emit_reserve(nu, nb);
    GateEmitBuf& b = gate_emit_buf();
    const int nblocks = (int)((block_len + WG_BLOCK - 1) / WG_BLOCK);
    hipLaunchKernelGGL(round_gates_kernel, dim3(nblocks), dim3(WG_BLOCK), 0,
                       gkr_compute_stream(),
                       reinterpret_cast<DevUniGate*>(b.d_uni),
                       reinterpret_cast<DevBinGate*>(b.d_bin),
                       orgsize, block_len, sc_pp, sc_pq, sc_qq, sc_c1);
    gpu_emit_and_eval(nu, nb, sz, h_out, uni, bin, d_val0, d_prev);
}

inline void gpu_ln2_gates(const FieldElement* h_val0, size_t n0,
                          const FieldElement* h_prev, size_t n_prev,
                          FieldElement* h_out, size_t sz,
                          std::vector<uniGate>& uni, std::vector<binGate>& bin,
                          long long orgsize, long long block_len, int channel_in,
                          int real_row, int real_col, int len,
                          long long qw_off, long long qb_off,
                          __int128 sc_a, __int128 sc_b, __int128 sc_y,
                          __int128 sc_sig) {
    gpu_fe *d_val0, *d_prev;
    gpu_layer_sources(h_val0, n0, h_prev, n_prev, &d_val0, &d_prev);
    const size_t nu = (size_t)real_row * ((size_t)real_col * 4 + 3);
    const size_t nb = (size_t)real_row * ((size_t)real_col * 7 + 1);
    gate_emit_reserve(nu, nb);
    GateEmitBuf& b = gate_emit_buf();
    const size_t nthr = (size_t)real_row * real_col + real_row;
    const int nblocks = (int)((nthr + WG_BLOCK - 1) / WG_BLOCK);
    hipLaunchKernelGGL(ln2_gates_kernel, dim3(nblocks), dim3(WG_BLOCK), 0,
                       gkr_compute_stream(),
                       reinterpret_cast<DevUniGate*>(b.d_uni),
                       reinterpret_cast<DevBinGate*>(b.d_bin),
                       orgsize, block_len, channel_in, real_row, real_col, len,
                       qw_off, qb_off, sc_a, sc_b, sc_y, sc_sig);
    gpu_emit_and_eval(nu, nb, sz, h_out, uni, bin, d_val0, d_prev);
}

// softmax_layer_1: advice values, then both gate blocks. The advice block is
// appended to val[0], so the mirror is grown (not uploaded) and read back once.
inline void gpu_sm1(FieldElement* h_val0, size_t orgsize, size_t alloc,
                    const FieldElement* h_prev, size_t n_prev,
                    FieldElement* h_out, size_t sz,
                    std::vector<uniGate>& uni, std::vector<binGate>& bin,
                    const int* h_table,
                    const std::vector<long long>& vbase,
                    const std::vector<long long>& row_start,
                    const std::vector<long long>& key_count,
                    const std::vector<int>& row_of,
                    int HEAD, int len, int HSIZE, long long score_count,
                    int layer_id, int c1, int e1, int eprime) {
    const size_t n_after = orgsize + alloc;
    gpu_fe* d_val0 = witgen_val0_grow(h_val0, orgsize, n_after);

    GateEvalState& st = gate_state();
    gpu_fe* d_prev = d_val0;
    if (h_prev && n_prev && h_prev != h_val0) {
        wg_ensure((void**)&st.d_prev, &st.cap_prev, n_prev * sizeof(gpu_fe), "witgen prev");
        HIP_CHECK(gkr_h2d(st.d_prev, h_prev, n_prev * sizeof(gpu_fe)));
        d_prev = st.d_prev;
    }

    const FieldElement one = to_FE(1);
    HIP_CHECK(hipMemcpyAsync(d_val0 + orgsize, reinterpret_cast<const gpu_fe*>(&one),
                             sizeof(gpu_fe), hipMemcpyHostToDevice, gkr_compute_stream()));

    seq_upload(vbase, row_start, row_start, row_of);   // vbase in the qbase slot
    SeqTables& t = seq_tables();
    long long* d_kc = nullptr;
    static long long* s_kc = nullptr; static size_t s_kc_cap = 0;
    wg_ensure((void**)&s_kc, &s_kc_cap, key_count.size() * sizeof(long long), "witgen keycount");
    HIP_CHECK(gkr_h2d(s_kc, key_count.data(), key_count.size() * sizeof(long long)));
    d_kc = s_kc;

    int* d_tab = exp_table_upload(h_table, 655360);

    hipLaunchKernelGGL(sm1_values_kernel,
                       dim3((int)(((long long)HEAD * len + WG_BLOCK - 1) / WG_BLOCK)),
                       dim3(WG_BLOCK), 0, gkr_compute_stream(),
                       d_val0, d_prev, d_tab, t.d_rowstart, d_kc,
                       (long long)orgsize, HEAD, len, score_count,
                       c1, e1, eprime,
                       ldexp(1.0, e1 + eprime + 1), ldexp(1.0, eprime + 1));

    const size_t nb = (size_t)HEAD * HSIZE * score_count;
    const size_t nu = (size_t)HEAD * (10 * score_count + len);
    gate_emit_reserve(nu, nb);
    GateEmitBuf& b = gate_emit_buf();
    hipLaunchKernelGGL(sm1_ev_gates_kernel,
                       dim3((int)((nb + WG_BLOCK - 1) / WG_BLOCK)), dim3(WG_BLOCK), 0,
                       gkr_compute_stream(),
                       reinterpret_cast<DevBinGate*>(b.d_bin),
                       t.d_qbase, t.d_rowstart, d_kc, t.d_rowof,
                       (long long)orgsize, HEAD, len, HSIZE, score_count);
    const size_t nthr = (size_t)HEAD * score_count + (size_t)HEAD * len;
    hipLaunchKernelGGL(sm1_check_gates_kernel,
                       dim3((int)((nthr + WG_BLOCK - 1) / WG_BLOCK)), dim3(WG_BLOCK), 0,
                       gkr_compute_stream(),
                       reinterpret_cast<DevUniGate*>(b.d_uni),
                       t.d_rowstart, d_kc, t.d_rowof,
                       (long long)orgsize, HEAD, len, HSIZE, score_count,
                       layer_id, c1, e1, eprime);

    witgen_val0_readback(h_val0, orgsize, n_after);
    gpu_emit_and_eval(nu, nb, sz, h_out, uni, bin, d_val0, d_prev);
}

// softmax_layer_2: its advice lands INSIDE softmax_layer_1's block, which the
// mirror already holds, so this reads the range back rather than pushing.
inline void gpu_sm2(FieldElement* h_val0, size_t n0,
                    size_t aux_lo, size_t aux_hi,
                    const FieldElement* h_prev, size_t n_prev,
                    FieldElement* h_out, size_t sz,
                    std::vector<uniGate>& uni, std::vector<binGate>& bin,
                    long long orgsize, int HEAD, int len, int HSIZE, int channel_in,
                    long long score_count, int layer_id,
                    int c1, int e1, int eprime) {
    gpu_fe *d_val0, *d_prev;
    gpu_layer_sources(h_val0, n0, h_prev, n_prev, &d_val0, &d_prev);

    const size_t n1 = (size_t)HEAD * len * HSIZE;
    const size_t n2 = (size_t)HEAD * score_count;
    const size_t nu = n1 * 4 + n2, nb = n1 * 2 + n2;
    gate_emit_reserve(nu, nb);
    GateEmitBuf& b = gate_emit_buf();

    hipLaunchKernelGGL(sm2_fused_kernel,
                       dim3((int)((n1 + WG_BLOCK - 1) / WG_BLOCK)), dim3(WG_BLOCK), 0,
                       gkr_compute_stream(),
                       d_val0, d_prev,
                       reinterpret_cast<DevUniGate*>(b.d_uni),
                       reinterpret_cast<DevBinGate*>(b.d_bin),
                       orgsize, HEAD, len, HSIZE, channel_in, score_count, layer_id,
                       c1, e1, eprime, ldexp(1.0, e1));
    hipLaunchKernelGGL(sm2_tail_gates_kernel,
                       dim3((int)((n2 + WG_BLOCK - 1) / WG_BLOCK)), dim3(WG_BLOCK), 0,
                       gkr_compute_stream(),
                       reinterpret_cast<DevUniGate*>(b.d_uni),
                       reinterpret_cast<DevBinGate*>(b.d_bin),
                       (long long)(n1 * 4), (long long)(n1 * 2),
                       orgsize, HEAD, len, HSIZE, score_count);

    witgen_val0_readback(h_val0, aux_lo, aux_hi);
    gpu_emit_and_eval(nu, nb, sz, h_out, uni, bin, d_val0, d_prev);
}

inline void gpu_ln1(FieldElement* h_val0, size_t orgsize, size_t alloc,
                    const FieldElement* h_prev, size_t n_prev,
                    FieldElement* h_out, size_t sz,
                    std::vector<uniGate>& uni, std::vector<binGate>& bin,
                    long long q_offset, long long block_len,
                    int len, int channel_in, int real_row, int real_col,
                    int real_cn_in, long long qw_off, long long qb_off,
                    int c1, int e1, int c2, int e2, int m) {
    const size_t n_after = orgsize + alloc;
    gpu_fe* d_val0 = witgen_val0_grow(h_val0, orgsize, n_after);

    GateEvalState& st = gate_state();
    gpu_fe* d_prev = d_val0;
    if (h_prev && n_prev && h_prev != h_val0) {
        wg_ensure((void**)&st.d_prev, &st.cap_prev, n_prev * sizeof(gpu_fe), "witgen prev");
        HIP_CHECK(gkr_h2d(st.d_prev, h_prev, n_prev * sizeof(gpu_fe)));
        d_prev = st.d_prev;
    }
    const FieldElement one = to_FE(1);
    HIP_CHECK(hipMemcpyAsync(d_val0 + orgsize, reinterpret_cast<const gpu_fe*>(&one),
                             sizeof(gpu_fe), hipMemcpyHostToDevice, gkr_compute_stream()));

    hipLaunchKernelGGL(ln1_values_kernel, dim3(std::max(1, real_row)), dim3(256), 0,
                       gkr_compute_stream(),
                       d_val0, q_offset, (long long)orgsize, block_len,
                       len, channel_in, real_row, real_col, real_cn_in,
                       qw_off, qb_off, c1, e1, c2, e2, m,
                       ldexp(1.0, e1), ldexp(1.0, e2));

    const size_t nu = (size_t)real_row * ((size_t)real_col * 3 + 6);
    const size_t nb = (size_t)real_row * 2;
    gate_emit_reserve(nu, nb);
    GateEmitBuf& b = gate_emit_buf();
    const size_t nthr = (size_t)real_row * real_col + real_row;
    hipLaunchKernelGGL(ln1_gates_kernel,
                       dim3((int)((nthr + WG_BLOCK - 1) / WG_BLOCK)), dim3(WG_BLOCK), 0,
                       gkr_compute_stream(),
                       reinterpret_cast<DevUniGate*>(b.d_uni),
                       reinterpret_cast<DevBinGate*>(b.d_bin),
                       q_offset, (long long)orgsize, block_len,
                       len, channel_in, real_row, real_col, real_cn_in);

    witgen_val0_readback(h_val0, orgsize, n_after);
    gpu_emit_and_eval(nu, nb, sz, h_out, uni, bin, d_val0, d_prev);
}

// Device port of calcNormalLayer.
//
//   uni/bin  -- this layer's gate lists (host side; still needed by the prover)
//   h_val0   -- val[0].data(), n0 = val[0].size()
//   h_prev   -- val[layer_id-1].data(), n_prev its size. Pass val[0] when
//               layer_id-1 == 0; may be null only when the layer has no gates.
//   h_out    -- val[layer_id].data(), sz = circuit.size
//
// Post: h_out[0..sz) holds the same field elements the host loop would produce.
inline void gpu_calc_normal_layer(const std::vector<uniGate>& uni,
                                  const std::vector<binGate>& bin,
                                  int layer_id,
                                  const FieldElement* h_val0, size_t n0,
                                  const FieldElement* h_prev, size_t n_prev,
                                  FieldElement* h_out, size_t sz) {
    if (sz == 0) return;

    GateEvalState& s = gate_state();
    GateEvalTimers& T = gate_timers();
    hipStream_t st = gkr_compute_stream();

    const size_t nu = uni.size(), nb = bin.size();
    T.layers++;

    // ---- val[0] mirror ----
    double t0 = wg_now_s();
    gpu_fe* d_val0 = witgen_val0_sync(h_val0, n0);
    T.val0_s += wg_now_s() - t0;

    // ---- val[layer_id-1] ----
    // Uploaded unconditionally rather than only when a gate references it:
    // deciding that needs a scan of the gate lists on the host, which costs
    // more than the copy (prev_s is a few ms for the whole build).
    t0 = wg_now_s();
    gpu_fe* d_prev = d_val0;
    if (h_prev && n_prev && h_prev != h_val0) {
        wg_ensure((void**)&s.d_prev, &s.cap_prev, n_prev * sizeof(gpu_fe),
                  "witgen prev");
        GkrXferTimer _x(&gkr_xfer_h2d_s_ref());
        HIP_CHECK(gkr_stage_h2d(s.d_prev, reinterpret_cast<const gpu_fe*>(h_prev),
                                n_prev * sizeof(gpu_fe), gkr_copy_stream()));
        HIP_CHECK(hipStreamSynchronize(gkr_copy_stream()));
        d_prev = s.d_prev;
        T.prev_bytes += n_prev * sizeof(gpu_fe);
    }
    T.prev_s += wg_now_s() - t0;

    wg_ensure((void**)&s.d_out, &s.cap_out, sz * sizeof(gpu_fe), "witgen out");

    if (nu + nb == 0) {   // no gates: the layer is all zeros
        HIP_CHECK(hipMemsetAsync(s.d_out, 0, sz * sizeof(gpu_fe), st));
        HIP_CHECK(hipMemcpyAsync(reinterpret_cast<gpu_fe*>(h_out), s.d_out,
                                 sz * sizeof(gpu_fe), hipMemcpyDeviceToHost, st));
        HIP_CHECK(hipStreamSynchronize(st));
        gkr_xfer_add_d2h(sz * sizeof(gpu_fe));
        return;
    }

    // ---- gate lists, straight across ----
    t0 = wg_now_s();
    if (nu) {
        wg_ensure((void**)&s.d_uni, &s.cap_uni, nu * sizeof(uniGate), "witgen uni gates");
        HIP_CHECK(gkr_h2d(s.d_uni, uni.data(), nu * sizeof(uniGate)));
    }
    if (nb) {
        wg_ensure((void**)&s.d_bin, &s.cap_bin, nb * sizeof(binGate), "witgen bin gates");
        HIP_CHECK(gkr_h2d(s.d_bin, bin.data(), nb * sizeof(binGate)));
    }
    T.gates_s += wg_now_s() - t0;
    T.gate_bytes += nu * sizeof(uniGate) + nb * sizeof(binGate);

    gpu_gate_eval_device(s.d_uni, nu, s.d_bin, nb, d_val0, d_prev, s.d_out, sz);

    t0 = wg_now_s();
    HIP_CHECK(hipMemcpyAsync(reinterpret_cast<gpu_fe*>(h_out), s.d_out,
                             sz * sizeof(gpu_fe), hipMemcpyDeviceToHost, st));
    HIP_CHECK(hipStreamSynchronize(st));
    gkr_xfer_add_d2h(sz * sizeof(gpu_fe));
    T.d2h_s += wg_now_s() - t0;
    T.d2h_bytes += sz * sizeof(gpu_fe);
}

// One line per partition, stderr. Sizes are cumulative over the whole build.
inline void witgen_report() {
    const GateEvalTimers& T = gate_timers();
    if (!T.layers) return;
    fprintf(stderr,
            "WITGEN_GATES layers=%ld val0_s=%.4f prev_s=%.4f "
            "gates_s=%.4f csr_s=%.4f reduce_s=%.4f d2h_s=%.4f total_s=%.4f "
            "val0_gib=%.3f prev_gib=%.3f gate_gib=%.3f d2h_gib=%.3f\n",
            T.layers, T.val0_s, T.prev_s, T.gates_s, T.csr_s,
            T.redu_s, T.d2h_s,
            T.val0_s + T.prev_s + T.gates_s + T.csr_s + T.redu_s + T.d2h_s,
            T.val0_bytes / 1073741824.0, T.prev_bytes / 1073741824.0,
            T.gate_bytes / 1073741824.0, T.d2h_bytes / 1073741824.0);
}

}  // namespace zkgpt_gpu

#endif  // ZKGPT_GPU_WITGEN_HPP
