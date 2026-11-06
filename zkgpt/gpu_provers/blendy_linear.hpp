#ifndef BLENDY_LINEAR_HPP
#define BLENDY_LINEAR_HPP

#include "gpu_field.hpp"
#include "common.hpp"

#include <vector>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <chrono>

// Sub-phase timings
struct BlendyTimings {
    double t_aux_ms = 0;    // AUX reduction kernel (skipped when future_size <= 1)
    double t_ps_ms = 0;     // PS scan: D2H AUX, host prefix sum, H2D PS
    double t_round_ms = 0;  // per-round p0/p1 kernel + readback + host reduce
    double t_lag_ms = 0;    // LagVector expansion kernel
    double t_fold_ms = 0;   // f_fold MSB-fold kernel + buffer swap
};

// Kernels

// AUX computation for stage s > 1 (or s == 1 when k > 1 and future_size > 1)
static __global__ void blendy_reduce_kernel(const gpu_fe* d_in, gpu_fe* d_out, int out_size, int block_size) {
    int b2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (b2 >= out_size)
        return;
    gpu_fe acc = {0};
    int base = b2 * block_size;
    for (int j = 0; j < block_size; ++j)
        acc = device_add(acc, d_in[base + j]);
    d_out[b2] = acc;
}

static __global__ void blendy_round_linear_kernel(const gpu_fe* __restrict__ d_lag,
                                                  const gpu_fe* __restrict__ d_ps, gpu_fe* d_p0_out,
                                                  gpu_fe* d_p1_out, int lag_count, int block_size) {
    int b2s = blockIdx.x * blockDim.x + threadIdx.x;
    gpu_fe lp0 = {0}, lp1 = {0};
    if (b2s < lag_count) {
        int lo = b2s * 2 * block_size;
        int mid = lo + block_size;
        int hi = mid + block_size;
        gpu_fe lag = d_lag[b2s];
        lp0 = device_mul(lag, device_sub(d_ps[mid], d_ps[lo]));
        lp1 = device_mul(lag, device_sub(d_ps[hi], d_ps[mid]));
    }
#ifdef __HIP_CPU_RT__
    HIP_DYNAMIC_SHARED(gpu_fe, smem)
#else
    extern __shared__ gpu_fe smem[];
#endif
    gpu_fe* s0 = smem;
    gpu_fe* s1 = smem + blockDim.x;
    int tid = threadIdx.x;
    s0[tid] = lp0;
    s1[tid] = lp1;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s0[tid] = device_add(s0[tid], s0[tid + s]);
            s1[tid] = device_add(s1[tid], s1[tid + s]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        d_p0_out[blockIdx.x] = s0[0];
        d_p1_out[blockIdx.x] = s1[0];
    }
}

// Expand LagVector with a new challenge r
static __global__ void blendy_lagvec_update_kernel(const gpu_fe* d_in, gpu_fe* d_out, gpu_fe omr, gpu_fe r,
                                                   int in_size) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= in_size)
        return;
    gpu_fe v = d_in[b];
    d_out[2 * b + 0] = device_mul(v, omr);
    d_out[2 * b + 1] = device_mul(v, r);
}

// MSB-first binary fold of f_folded
static __global__ void blendy_fold_msb_kernel(const gpu_fe* d_in, gpu_fe* d_out, gpu_fe omr, gpu_fe r,
                                              int out_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < out_size)
        d_out[i] = device_add(device_mul(omr, d_in[i]), device_mul(r, d_in[i + out_size]));
}

// Verifier

struct BlendyLinearVerifier {
    FieldElement claim;
    bool ok = true;
    BlendyLinearVerifier(FieldElement gamma) : claim(gamma) {}

    // Verify p(0)+p(1)==claim, sample challenge, update claim to p(r)
    FieldElement process_round(FieldElement p0, FieldElement p1) {
        if (p0 + p1 != claim)
            ok = false;
        FieldElement r = get_random_element();
        claim = (to_FE(1) - r) * p0 + r * p1;
        return r;
    }
};

// Host driver

// Prove sum_{i in {0,1}^n} f[i] = gamma via degree-1 BlendySC with k stages. d_f
static FieldElement blendy_linear_sumcheck(const gpu_fe* d_f, int N, int n, int k, FieldElement gamma,
                                           int threads, std::vector<FieldElement>& r_out,
                                           BlendyTimings* timings = nullptr) {
    auto bl_clk = [] { return std::chrono::high_resolution_clock::now(); };
    auto bl_ms_since = [](auto t0) {
        return std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0)
            .count();
    };

    assert(n >= 0 && k >= 1 && "blendy: need n >= 0 and k >= 1");
    const int l_base = n / k;
    const int l_extra = n - k * l_base;  // 0 iff k divides n
    const int num_stages = (l_extra > 0) ? (k + 1) : k;
    const int max_l = std::max(l_base, l_extra);
    const int pow2_max = 1 << max_l;

    // max_round_blocks
    int max_lag = pow2_max / 2;
    int max_rnd_blks = std::max(1, (max_lag + threads - 1) / threads);

    gpu_fe *d_f_fold, *d_fold_tmp;
    gpu_fe *d_aux, *d_ps;
    gpu_fe *d_lag_a, *d_lag_b;
    gpu_fe *d_p0_out, *d_p1_out;

    HIP_CHECK(hipMalloc(&d_f_fold, N * sizeof(gpu_fe)));
    HIP_CHECK(hipMalloc(&d_fold_tmp, N * sizeof(gpu_fe)));
    HIP_CHECK(hipMalloc(&d_aux, pow2_max * sizeof(gpu_fe)));
    HIP_CHECK(hipMalloc(&d_ps, (pow2_max + 1) * sizeof(gpu_fe)));
    HIP_CHECK(hipMalloc(&d_lag_a, pow2_max * sizeof(gpu_fe)));
    HIP_CHECK(hipMalloc(&d_lag_b, pow2_max * sizeof(gpu_fe)));
    HIP_CHECK(hipMalloc(&d_p0_out, max_rnd_blks * sizeof(gpu_fe)));
    HIP_CHECK(hipMalloc(&d_p1_out, max_rnd_blks * sizeof(gpu_fe)));

    // Working copy of f (will be folded stage by stage)
    HIP_CHECK(hipMemcpy(d_f_fold, d_f, N * sizeof(gpu_fe), hipMemcpyDeviceToDevice));

    // LagVector starts as [1]
    gpu_fe mont_one = fe_to_gpu(to_FE(1));
    HIP_CHECK(hipMemcpy(d_lag_a, &mont_one, sizeof(gpu_fe), hipMemcpyHostToDevice));

    BlendyLinearVerifier verifier(gamma);
    r_out.clear();
    r_out.reserve(n);

    int cur_fold_size = N;

    for (int s = 1; s <= num_stages; ++s) {
        // Stage length: first k stages use l_base
        const int l = (s <= k) ? l_base : l_extra;
        if (l == 0)
            continue;  // no-op stage (only when k > n)
        const int pow2_l = 1 << l;

        // future_size = number of "b3" entries per AUX slot for this stage
        int future_size = cur_fold_size >> l;

        const gpu_fe* d_aux_ptr;
        {
            auto t_aux = bl_clk();
            if (future_size <= 1) {
                // AUX == f_fold (no reduction needed)
                d_aux_ptr = d_f_fold;
            } else {
                int ab = (pow2_l + threads - 1) / threads;
                hipLaunchKernelGGL(blendy_reduce_kernel, dim3(ab), dim3(threads), 0, 0, d_f_fold, d_aux,
                                   pow2_l, future_size);
                HIP_CHECK(hipDeviceSynchronize());
                d_aux_ptr = d_aux;
            }
            if (timings)
                timings->t_aux_ms += bl_ms_since(t_aux);
        }

        // Compute PS(s) on host ───────────────────────────────────────── Exclusive prefix sum
        {
            auto t_ps = bl_clk();
            std::vector<gpu_fe> h_aux(pow2_l);
            HIP_CHECK(hipMemcpy(h_aux.data(), d_aux_ptr, pow2_l * sizeof(gpu_fe), hipMemcpyDeviceToHost));
            std::vector<gpu_fe> h_ps(pow2_l + 1);
            h_ps[0] = fe_to_gpu(to_FE(0));
            for (int i = 0; i < pow2_l; ++i) {
                FieldElement s_val = fe_from_gpu(h_ps[i]) + fe_from_gpu(h_aux[i]);
                h_ps[i + 1] = fe_to_gpu(s_val);
            }
            HIP_CHECK(hipMemcpy(d_ps, h_ps.data(), (pow2_l + 1) * sizeof(gpu_fe), hipMemcpyHostToDevice));
            if (timings)
                timings->t_ps_ms += bl_ms_since(t_ps);
        }

        // Reset LagVector to [1] for this stage (s>1 only)
        if (s > 1) {
            HIP_CHECK(hipMemcpy(d_lag_a, &mont_one, sizeof(gpu_fe), hipMemcpyHostToDevice));
        }

        int lag_count = 1;  // grows as 2^{j'-1} each round
        std::vector<FieldElement> stage_challenges;
        stage_challenges.reserve(l);

        // l rounds for this stage
        for (int jp = 1; jp <= l; ++jp) {
            // block_size = 2^{l-jp}: number of AUX entries per half-block
            int block_size = pow2_l >> jp;
            int rb = std::max(1, (lag_count + threads - 1) / threads);
            size_t smem_sz = 2 * (size_t)threads * sizeof(gpu_fe);

            FieldElement p0, p1;
            {
                auto t_rd = bl_clk();
                hipLaunchKernelGGL(blendy_round_linear_kernel, dim3(rb), dim3(threads), smem_sz, 0, d_lag_a,
                                   d_ps, d_p0_out, d_p1_out, lag_count, block_size);
                HIP_CHECK(hipDeviceSynchronize());

                // Reduce per-block partial sums on host
                std::vector<gpu_fe> h_p0(rb), h_p1(rb);
                HIP_CHECK(hipMemcpy(h_p0.data(), d_p0_out, rb * sizeof(gpu_fe), hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(h_p1.data(), d_p1_out, rb * sizeof(gpu_fe), hipMemcpyDeviceToHost));
                p0.clear();
                p1.clear();
                for (int b = 0; b < rb; ++b) {
                    p0 += fe_from_gpu(h_p0[b]);
                    p1 += fe_from_gpu(h_p1[b]);
                }
                if (timings)
                    timings->t_round_ms += bl_ms_since(t_rd);
            }

            FieldElement r_j = verifier.process_round(p0, p1);
            r_out.push_back(r_j);  // collected MSB-first per stage
            stage_challenges.push_back(r_j);

            {
                auto t_lg = bl_clk();
                // Expand LagVector: new_lag[2b+0]=(1-r)*lag[b], new_lag[2b+1]=r*lag[b]
                FieldElement omr_fe = to_FE(1) - r_j;
                gpu_fe g_r = fe_to_gpu(r_j);
                gpu_fe g_omr = fe_to_gpu(omr_fe);
                int ub = std::max(1, (lag_count + threads - 1) / threads);
                hipLaunchKernelGGL(blendy_lagvec_update_kernel, dim3(ub), dim3(threads), 0, 0, d_lag_a,
                                   d_lag_b, g_omr, g_r, lag_count);
                HIP_CHECK(hipDeviceSynchronize());
                std::swap(d_lag_a, d_lag_b);
                lag_count *= 2;
                if (timings)
                    timings->t_lag_ms += bl_ms_since(t_lg);
            }
        }

        {
            auto t_fd = bl_clk();
            for (int jp = 0; jp < l; ++jp) {
                FieldElement r_j = stage_challenges[jp];
                FieldElement omr_fe = to_FE(1) - r_j;
                gpu_fe g_r = fe_to_gpu(r_j);
                gpu_fe g_omr = fe_to_gpu(omr_fe);
                int out_sz = cur_fold_size / 2;
                int fb = std::max(1, (out_sz + threads - 1) / threads);
                hipLaunchKernelGGL(blendy_fold_msb_kernel, dim3(fb), dim3(threads), 0, 0, d_f_fold,
                                   d_fold_tmp, g_omr, g_r, out_sz);
                HIP_CHECK(hipDeviceSynchronize());
                std::swap(d_f_fold, d_fold_tmp);
                cur_fold_size = out_sz;
            }
            if (timings)
                timings->t_fold_ms += bl_ms_since(t_fd);
        }
    }

    if (!verifier.ok) {
        std::cerr << "[BlendyLinear] sumcheck consistency FAILED\n";
        std::exit(1);
    }

    // r_out is currently in MSB-first order
    std::reverse(r_out.begin(), r_out.end());

    HIP_CHECK(hipFree(d_f_fold));
    HIP_CHECK(hipFree(d_fold_tmp));
    HIP_CHECK(hipFree(d_aux));
    HIP_CHECK(hipFree(d_ps));
    HIP_CHECK(hipFree(d_lag_a));
    HIP_CHECK(hipFree(d_lag_b));
    HIP_CHECK(hipFree(d_p0_out));
    HIP_CHECK(hipFree(d_p1_out));

    return verifier.claim;
}

#endif  // BLENDY_LINEAR_HPP
