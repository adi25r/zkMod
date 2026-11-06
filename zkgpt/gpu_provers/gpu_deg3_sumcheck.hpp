#pragma once
#ifndef ZKGPT_GPU_DEG3_SUMCHECK_HPP
#define ZKGPT_GPU_DEG3_SUMCHECK_HPP

#include <hip/hip_runtime.h>
#include <mcl/bn.hpp>
#include <vector>
#include <cassert>
#include <iostream>
#include <cstdlib>
#include <chrono>

#include "gpu_field.hpp"
#include "gpu_uvm_alloc.hpp"

typedef mcl::bn::Fr FieldElement;

namespace zkgpt_gpu {

// GPU-region timer for the deg3 sumcheck times
struct Deg3Timer {
    double gpu = 0;
    bool on = false;
    void reset() { gpu = 0; }
};
inline Deg3Timer g_deg3_timer;

// Kernels

// evaluates deg 3 poly at 4 points, where the pionts are extrapolated from inputs 0 and 1
static __global__ void deg3_eval_kernel(const gpu_fe* __restrict__ lag, const gpu_fe* __restrict__ f,
                                        const gpu_fe* __restrict__ g, gpu_fe* p0_out, gpu_fe* p1_out,
                                        gpu_fe* p2_out, gpu_fe* p3_out, int half_size) {
#ifdef __HIP_CPU_RT__
    HIP_DYNAMIC_SHARED(gpu_fe, smem)
#else
    extern __shared__ gpu_fe smem[];
#endif
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    gpu_fe lp0 = {0}, lp1 = {0}, lp2 = {0}, lp3 = {0};
    if (idx < half_size) {
        int j = 2 * idx;
        gpu_fe l0 = lag[j], l1 = lag[j + 1];
        gpu_fe f0 = f[j], f1 = f[j + 1];
        gpu_fe g0 = g[j], g1 = g[j + 1];

        lp0 = device_mul(device_mul(l0, f0), g0);
        lp1 = device_mul(device_mul(l1, f1), g1);

        // at x=2: (2*v1 - v0) for each of lag, f, g
        gpu_fe L2 = device_sub(device_add(l1, l1), l0);
        gpu_fe F2 = device_sub(device_add(f1, f1), f0);
        gpu_fe G2 = device_sub(device_add(g1, g1), g0);
        lp2 = device_mul(device_mul(L2, F2), G2);

        // at x=3: (3*v1 - 2*v0) = v1+v1+v1-v0-v0
        gpu_fe L3 = device_sub(device_add(device_add(l1, l1), l1), device_add(l0, l0));
        gpu_fe F3 = device_sub(device_add(device_add(f1, f1), f1), device_add(f0, f0));
        gpu_fe G3 = device_sub(device_add(device_add(g1, g1), g1), device_add(g0, g0));
        lp3 = device_mul(device_mul(L3, F3), G3);
    }

    gpu_fe* s0 = smem;
    gpu_fe* s1 = smem + blockDim.x;
    gpu_fe* s2 = smem + 2 * blockDim.x;
    gpu_fe* s3 = smem + 3 * blockDim.x;
    s0[tid] = lp0;
    s1[tid] = lp1;
    s2[tid] = lp2;
    s3[tid] = lp3;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s0[tid] = device_add(s0[tid], s0[tid + s]);
            s1[tid] = device_add(s1[tid], s1[tid + s]);
            s2[tid] = device_add(s2[tid], s2[tid + s]);
            s3[tid] = device_add(s3[tid], s3[tid + s]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        p0_out[blockIdx.x] = s0[0];
        p1_out[blockIdx.x] = s1[0];
        p2_out[blockIdx.x] = s2[0];
        p3_out[blockIdx.x] = s3[0];
    }
}

// Linear fold: out[i] = (1-r)*in[2i] + r*in[2i+1]
static __global__ void deg3_fold_kernel(const gpu_fe* __restrict__ d_in, gpu_fe* d_out, gpu_fe omr, gpu_fe r,
                                        int out_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < out_size)
        d_out[i] = device_add(device_mul(omr, d_in[2 * i]), device_mul(r, d_in[2 * i + 1]));
}

// host code

// evaluate the deg 3 poly defined at x=0,1
static FieldElement deg3_poly_eval(FieldElement x0, FieldElement x1, FieldElement x2, FieldElement x3,
                                   FieldElement u) {
    FieldElement inv6;
    FieldElement::inv(inv6, FieldElement(6));
    FieldElement u1 = u - FieldElement(1), u2 = u - FieldElement(2), u3 = u - FieldElement(3);
    return inv6 * ((-x0) * u1 * u2 * u3 + FieldElement(3) * x1 * u * u2 * u3 -
                   FieldElement(3) * x2 * u * u1 * u3 + x3 * u * u1 * u2);
}

// performs the degree 3 sumcheck of the lagrange basis
inline void gpu_sumcheck_deg3(int l, const mcl::bn::Fr* r_pt, const mcl::bn::Fr* f_host,
                              const mcl::bn::Fr* g_host, mcl::bn::Fr S, std::vector<mcl::bn::Fr>& r_out,
                              mcl::bn::Fr& claim_f, mcl::bn::Fr& claim_g) {
    using mcl::bn::Fr;
    const int N = 1 << l;
    const int threads = 256;

    // Build eq polynomial on CPU from r_pt, matching range_proof_get_eq
    std::vector<Fr> eq(N);
    eq[0] = Fr(1) - r_pt[0];
    eq[1] = r_pt[0];
    for (int i = 2; i <= l; ++i) {
        for (int j = (1 << i) - 1; j >= 0; --j) {
            if (j & (1 << (i - 1)))
                eq[j] = eq[j - (1 << (i - 1))] * r_pt[i - 1];
            else
                eq[j] = eq[j] * (Fr(1) - r_pt[i - 1]);
        }
    }

    // Upload lag, f, g to device
    std::vector<gpu_fe> h_lag(N), h_f(N), h_g(N);
    for (int i = 0; i < N; ++i) {
        h_lag[i] = fe_to_gpu(eq[i]);
        h_f[i] = fe_to_gpu(f_host[i]);
        h_g[i] = fe_to_gpu(g_host[i]);
    }

    auto _deg3_t0 = std::chrono::high_resolution_clock::now();  // GPU region start
    gpu_fe *d_lag, *d_f, *d_g, *d_lag2, *d_f2, *d_g2;
    HIP_CHECK(GKR_DMALLOC(&d_lag, N * sizeof(gpu_fe)));
    HIP_CHECK(GKR_DMALLOC(&d_f, N * sizeof(gpu_fe)));
    HIP_CHECK(GKR_DMALLOC(&d_g, N * sizeof(gpu_fe)));
    HIP_CHECK(GKR_DMALLOC(&d_lag2, N * sizeof(gpu_fe)));
    HIP_CHECK(GKR_DMALLOC(&d_f2, N * sizeof(gpu_fe)));
    HIP_CHECK(GKR_DMALLOC(&d_g2, N * sizeof(gpu_fe)));

    HIP_CHECK(gkr_h2d(d_lag, h_lag.data(), N * sizeof(gpu_fe)));
    HIP_CHECK(gkr_h2d(d_f, h_f.data(), N * sizeof(gpu_fe)));
    HIP_CHECK(gkr_h2d(d_g, h_g.data(), N * sizeof(gpu_fe)));

    // Reduction output buffers (worst case = first round, N/2 pairs, ceil/threads blocks)
    int max_blocks = (N / 2 + threads - 1) / threads;
    gpu_fe *d_p0, *d_p1, *d_p2, *d_p3;
    HIP_CHECK(GKR_DMALLOC(&d_p0, max_blocks * sizeof(gpu_fe)));
    HIP_CHECK(GKR_DMALLOC(&d_p1, max_blocks * sizeof(gpu_fe)));
    HIP_CHECK(GKR_DMALLOC(&d_p2, max_blocks * sizeof(gpu_fe)));
    HIP_CHECK(GKR_DMALLOC(&d_p3, max_blocks * sizeof(gpu_fe)));

    r_out.resize(l);
    int cur_size = N;

    for (int i = l; i >= 1; --i) {
        int half = cur_size / 2;
        int nblocks = (half + threads - 1) / threads;
        size_t smem = 4 * (size_t)threads * sizeof(gpu_fe);

        HIP_CHECK(hipDeviceSynchronize());
        hipLaunchKernelGGL(deg3_eval_kernel, dim3(nblocks), dim3(threads), smem, 0, d_lag, d_f, d_g, d_p0,
                           d_p1, d_p2, d_p3, half);
        HIP_CHECK(hipDeviceSynchronize());

        // Reduce block partial sums on host
        std::vector<gpu_fe> hp0(nblocks), hp1(nblocks), hp2(nblocks), hp3(nblocks);
        HIP_CHECK(hipMemcpy(hp0.data(), d_p0, nblocks * sizeof(gpu_fe), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(hp1.data(), d_p1, nblocks * sizeof(gpu_fe), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(hp2.data(), d_p2, nblocks * sizeof(gpu_fe), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(hp3.data(), d_p3, nblocks * sizeof(gpu_fe), hipMemcpyDeviceToHost));

        Fr sum0, sum1, sum2, sum3;
        sum0.clear();
        sum1.clear();
        sum2.clear();
        sum3.clear();
        for (int b = 0; b < nblocks; ++b) {
            sum0 += fe_from_gpu(hp0[b]);
            sum1 += fe_from_gpu(hp1[b]);
            sum2 += fe_from_gpu(hp2[b]);
            sum3 += fe_from_gpu(hp3[b]);
        }
        assert(sum0 + sum1 == S);

        Fr rho;
        rho.setByCSPRNG();
        r_out[l - i] = rho;
        S = deg3_poly_eval(sum0, sum1, sum2, sum3, rho);

        // Fold lag, f, g on GPU
        gpu_fe g_r = fe_to_gpu(rho);
        gpu_fe g_omr = fe_to_gpu(Fr(1) - rho);
        int fb = (half + threads - 1) / threads;
        hipLaunchKernelGGL(deg3_fold_kernel, dim3(fb), dim3(threads), 0, 0, d_lag, d_lag2, g_omr, g_r, half);
        hipLaunchKernelGGL(deg3_fold_kernel, dim3(fb), dim3(threads), 0, 0, d_f, d_f2, g_omr, g_r, half);
        hipLaunchKernelGGL(deg3_fold_kernel, dim3(fb), dim3(threads), 0, 0, d_g, d_g2, g_omr, g_r, half);
        HIP_CHECK(hipDeviceSynchronize());

        std::swap(d_lag, d_lag2);
        std::swap(d_f, d_f2);
        std::swap(d_g, d_g2);
        cur_size = half;
    }

    // Read back final scalars
    gpu_fe hf, hg;
    HIP_CHECK(hipMemcpy(&hf, d_f, sizeof(gpu_fe), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&hg, d_g, sizeof(gpu_fe), hipMemcpyDeviceToHost));
    claim_f = fe_from_gpu(hf);
    claim_g = fe_from_gpu(hg);

    HIP_CHECK(hipFree(d_lag));
    HIP_CHECK(hipFree(d_f));
    HIP_CHECK(hipFree(d_g));
    HIP_CHECK(hipFree(d_lag2));
    HIP_CHECK(hipFree(d_f2));
    HIP_CHECK(hipFree(d_g2));
    HIP_CHECK(hipFree(d_p0));
    HIP_CHECK(hipFree(d_p1));
    HIP_CHECK(hipFree(d_p2));
    HIP_CHECK(hipFree(d_p3));
    if (g_deg3_timer.on)
        g_deg3_timer.gpu +=
            std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - _deg3_t0).count();
}

}  // namespace zkgpt_gpu

#endif  // ZKGPT_GPU_DEG3_SUMCHECK_HPP
