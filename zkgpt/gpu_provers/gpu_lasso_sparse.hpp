#pragma once
// Lasso multiplicity vector held on the device as (index, beta) pairs, one segment per layer table.
// Indices are distinct within a segment, so each segment is scattered by one kernel without races.
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <hip/hip_runtime.h>
#include "gpu_field.hpp"
#include "gpu_uvm_alloc.hpp"

namespace zkgpt_gpu {

static __global__ void lasso_sparse_scatter_kernel(const long long* __restrict__ idx,
                                                   const gpu_fe* __restrict__ val, long n,
                                                   gpu_fe* __restrict__ dst, long off, long cnt) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long k = (long)idx[i] - off;
    if (k < 0 || k >= cnt) return;
    dst[k] = device_add(dst[k], val[i]);
}

struct LassoSparseA {
    long long* d_idx = nullptr;
    gpu_fe*    d_val = nullptr;
    long cap = 0, nnz = 0;
    std::vector<long> seg_lo;          // segment k is [seg_lo[k], seg_lo[k+1]); seg_lo.back() == nnz
    bool active = false;

    void reset(long total) {
        if (total > cap) {
            release();
            HIP_CHECK(GKR_DMALLOC(&d_idx, std::max(1L, total) * sizeof(long long)));
            HIP_CHECK(GKR_DMALLOC(&d_val, std::max(1L, total) * sizeof(gpu_fe)));
            cap = total;
        }
        nnz = 0;
        seg_lo.assign(1, 0);
        active = true;
    }

    // One segment; idx[0..cnt) distinct. Synchronous: the caller reuses `val` afterwards.
    void append(const long long* idx, const FieldElement* val, long cnt) {
        if (cnt <= 0) return;
        if (nnz + cnt > cap) {
            fprintf(stderr, "[lasso-sparse] capacity %ld exceeded (nnz=%ld + %ld)\n", cap, nnz, cnt);
            std::exit(1);
        }
        static_assert(sizeof(FieldElement) == sizeof(gpu_fe), "mcl Fr must be the device layout");
        HIP_CHECK(gkr_h2d(d_idx + nnz, idx, (size_t)cnt * sizeof(long long)));
        HIP_CHECK(gkr_h2d(d_val + nnz, reinterpret_cast<const gpu_fe*>(val), (size_t)cnt * sizeof(gpu_fe)));
        nnz += cnt;
        seg_lo.push_back(nnz);
    }

    // dst[0..cnt) = lasso_mult_v[off .. off+cnt), on `s`.
    void materialize(gpu_fe* dst, long off, long cnt, hipStream_t s) const {
        HIP_CHECK(hipMemsetAsync(dst, 0, (size_t)cnt * sizeof(gpu_fe), s));
        const int threads = 256;
        for (size_t k = 0; k + 1 < seg_lo.size(); ++k) {
            const long lo = seg_lo[k], n = seg_lo[k + 1] - lo;
            if (n <= 0) continue;
            hipLaunchKernelGGL(lasso_sparse_scatter_kernel, dim3((unsigned)((n + threads - 1) / threads)),
                               dim3(threads), 0, s, d_idx + lo, d_val + lo, n, dst, off, cnt);
        }
        HIP_CHECK(hipGetLastError());
    }

    void release() {
        if (d_idx) HIP_CHECK(hipFree(d_idx));
        if (d_val) HIP_CHECK(hipFree(d_val));
        d_idx = nullptr; d_val = nullptr; cap = 0; nnz = 0;
        seg_lo.clear();
        active = false;
    }
};

inline LassoSparseA& lasso_sparse_a() { static LassoSparseA s; return s; }

// GKR_LASSO_SPARSE_CHECK: compare against the host-built vector.
inline void lasso_sparse_check(const FieldElement* host, long N) {
    const LassoSparseA& A = lasso_sparse_a();
    const long W = std::min(N, 1L << 22);
    gpu_fe* d = nullptr;
    HIP_CHECK(hipMalloc(&d, W * sizeof(gpu_fe)));
    std::vector<gpu_fe> h(W);
    long bad = 0, first = -1;
    for (long off = 0; off < N; off += W) {
        const long cnt = std::min(W, N - off);
        A.materialize(d, off, cnt, 0);
        HIP_CHECK(hipMemcpy(h.data(), d, cnt * sizeof(gpu_fe), hipMemcpyDeviceToHost));
        for (long i = 0; i < cnt; ++i)
            if (fe_from_gpu(h[i]) != host[off + i]) { if (first < 0) first = off + i; ++bad; }
    }
    HIP_CHECK(hipFree(d));
    fprintf(stderr, "[lasso-sparse-check] N=%ld nnz=%ld segments=%zu mismatches=%ld first=%ld\n",
            N, A.nnz, A.seg_lo.size() - 1, bad, first);
    if (bad) std::exit(3);
}

}  // namespace zkgpt_gpu
