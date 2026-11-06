#pragma once
#ifndef GPU_BFEVAL_HPP
#define GPU_BFEVAL_HPP

#include "gpu_field.hpp"
#include "gpu_uvm_alloc.hpp"

namespace zkgpt_gpu {

static __global__ void k_bfeval(const gpu_fe* __restrict__ w,
                                const gpu_fe* __restrict__ L,
                                const gpu_fe* __restrict__ R,
                                gpu_fe* __restrict__ out,
                                long N, int c, long maskc)
{
    const long idx    = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long stride = (long)gridDim.x * blockDim.x;
    gpu_fe acc = {0};                       // Montgomery zero is all-zero bits
    for (long k = idx; k < N; k += stride) {
        const gpu_fe lag = device_mul(L[k & maskc], R[k >> c]);
        acc = device_add(acc, device_mul(lag, w[k]));
    }
#ifdef __HIP_CPU_RT__
    HIP_DYNAMIC_SHARED(gpu_fe, smem)
#else
    extern __shared__ gpu_fe smem[];
#endif
    const int tid = threadIdx.x;
    smem[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = device_add(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = smem[0];
}

struct BfEvalState {
    gpu_fe *d_w = nullptr, *d_L = nullptr, *d_R = nullptr, *d_out = nullptr;
    size_t cap_w = 0, cap_L = 0, cap_R = 0, cap_out = 0;
};
inline BfEvalState& bfeval_state() { static BfEvalState s; return s; }

inline void bfeval_ensure(gpu_fe** p, size_t* cap, size_t need) {
    if (*cap >= need) return;
    if (*p) HIP_CHECK(hipFree(*p));
    HIP_CHECK(GKR_DMALLOC(p, need * sizeof(gpu_fe)));
    *cap = need;
}

inline void bfeval_teardown() {
    BfEvalState& s = bfeval_state();
    if (s.d_w)   hipFree(s.d_w);
    if (s.d_L)   hipFree(s.d_L);
    if (s.d_R)   hipFree(s.d_R);
    if (s.d_out) hipFree(s.d_out);
    s = BfEvalState();
}

// L has 2^c entries, R has 2^halfl, w has N = 2^l. Returns sum_k L*R*w.
inline FieldElement gpu_bfeval(const FieldElement* w, const FieldElement* L,
                               const FieldElement* R, long N, int c, int halfl)
{
    BfEvalState& s = bfeval_state();
    const int threads = 256;
    long blocks = (N + threads - 1) / threads;
    if (blocks > 4096) blocks = 4096;          // grid-stride caps the partials
    if (blocks < 1) blocks = 1;

    bfeval_ensure(&s.d_w,   &s.cap_w,   (size_t)N);
    bfeval_ensure(&s.d_L,   &s.cap_L,   (size_t)1 << c);
    bfeval_ensure(&s.d_R,   &s.cap_R,   (size_t)1 << halfl);
    bfeval_ensure(&s.d_out, &s.cap_out, (size_t)blocks);

    gkr_h2d(s.d_w, reinterpret_cast<const gpu_fe*>(w), (size_t)N * sizeof(gpu_fe));
    gkr_h2d(s.d_L, reinterpret_cast<const gpu_fe*>(L),
            ((size_t)1 << c) * sizeof(gpu_fe));
    gkr_h2d(s.d_R, reinterpret_cast<const gpu_fe*>(R),
            ((size_t)1 << halfl) * sizeof(gpu_fe));

    hipLaunchKernelGGL(k_bfeval, dim3(blocks), dim3(threads),
                       threads * sizeof(gpu_fe), 0,
                       s.d_w, s.d_L, s.d_R, s.d_out, N, c, ((long)1 << c) - 1);
    HIP_CHECK(hipGetLastError());

    std::vector<FieldElement> part(blocks);
    HIP_CHECK(hipMemcpy(part.data(), s.d_out, blocks * sizeof(gpu_fe),
                        hipMemcpyDeviceToHost));
    FieldElement ret = 0;
    for (long i = 0; i < blocks; ++i) ret += part[i];   // exact, order-free
    return ret;
}

}  // namespace zkgpt_gpu
#endif
