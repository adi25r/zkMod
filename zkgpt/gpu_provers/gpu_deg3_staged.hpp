#pragma once
#ifndef ZKGPT_GPU_DEG3_STAGED_HPP
#define ZKGPT_GPU_DEG3_STAGED_HPP

#include <hip/hip_runtime.h>
#include <mcl/bn.hpp>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <chrono>

#include "gpu_field.hpp"
#include "gpu_uvm_alloc.hpp"
#include "blendysc.hpp"
#include "gpu_deg3_sumcheck.hpp"

namespace zkgpt_gpu {

inline __host__ __device__ long pow4(int n) { return 1L << (2 * n); }

static __global__ void expand_bk_kernel4(const gpu_fe* d_bk, gpu_fe* d_bk_ext,
                                         int l, int future_vars, long pow4_l)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = ((long)1 << l) << future_vars;
    if (idx >= total) return;
    int  b2_bin = (int)(idx & (((long)1 << l) - 1));
    long b3 = idx >> l;
    long b2_base4 = 0;
    for (int i = 0; i < l; ++i)
        b2_base4 += (long)((b2_bin >> i) & 1) * pow4(l - 1 - i);
    d_bk_ext[b3 * pow4_l + b2_base4] = d_bk[idx];
}

static __global__ void extrapolate_bk_var_kernel4(gpu_fe* d_ext, int var, int l, int future_vars, long pow4_l)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = pow4_l << future_vars;
    if (idx >= total) return;
    long b2 = idx % pow4_l;
    long pv = pow4(var);
    int  digit = (int)((b2 / pv) % 4);
    if (digit < 2) return;
    long base0 = idx - (long)digit * pv;
    gpu_fe v0 = d_ext[base0];
    gpu_fe v1 = d_ext[base0 + pv];
    if (digit == 2) // f(2) = 2*f(1) - f(0)
        d_ext[idx] = device_sub(device_add(v1, v1), v0);
    else // f(3) = 3*f(1) - 2*f(0) = f(1)+f(1)+f(1)-f(0)-f(0)
        d_ext[idx] = device_sub(device_add(device_add(v1, v1), v1), device_add(v0, v0));
}

// Cumulative wall time of the ptable reduction (partial + finalize, synced), all calls.
inline double g_deg3_ptable_s = 0;

static __global__ void ptable3_partial_kernel(
    const gpu_fe* __restrict__ d_A_ext, const gpu_fe* __restrict__ d_B_ext,
    const gpu_fe* __restrict__ d_C_ext, gpu_fe* d_partials,
    long pow4_l, long fut, int nb3_blocks)
{
    long b2  = blockIdx.x;
    int  blk = blockIdx.y;
    int  tid = threadIdx.x;
    const long stride = (long)nb3_blocks * blockDim.x;
    gpu_fe acc = {0};
    for (long b3 = (long)blk * blockDim.x + tid; b3 < fut; b3 += stride) {
        long o = b3 * pow4_l + b2;
        acc = device_add(acc, device_mul(device_mul(d_A_ext[o], d_B_ext[o]), d_C_ext[o]));
    }
#ifdef __HIP_CPU_RT__
    HIP_DYNAMIC_SHARED(gpu_fe, smem)
#else
    extern __shared__ gpu_fe smem[];
#endif
    smem[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = device_add(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    if (tid == 0) d_partials[b2 * nb3_blocks + blk] = smem[0];
}

static __global__ void ptable3_finalize_kernel(const gpu_fe* __restrict__ d_partials,
    gpu_fe* d_ptable, long pow4_l, int nb3_blocks)
{
    long b2 = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (b2 >= pow4_l) return;
    gpu_fe acc = d_ptable[b2];
    const long base = b2 * nb3_blocks;
    for (int i = 0; i < nb3_blocks; ++i)
        acc = device_add(acc, d_partials[base + i]);
    d_ptable[b2] = acc;
}

static __global__ void blendy_round_kernel4(const gpu_fe* __restrict__ d_ptable,
    gpu_fe* d_p0, gpu_fe* d_p1, gpu_fe* d_p2, gpu_fe* d_p3,
    int remaining_free, int total_threads)
{
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    gpu_fe lp0 = {0}, lp1 = {0}, lp2 = {0}, lp3 = {0};
    if (idx < total_threads) {
        long b2_tail = 0;
        for (int i = 0; i < remaining_free - 1; ++i)
            b2_tail += (long)((idx >> i) & 1) * pow4(i);
        long stride = pow4(remaining_free - 1);
        lp0 = d_ptable[0 * stride + b2_tail];
        lp1 = d_ptable[1 * stride + b2_tail];
        lp2 = d_ptable[2 * stride + b2_tail];
        lp3 = d_ptable[3 * stride + b2_tail];
    }
#ifdef __HIP_CPU_RT__
    HIP_DYNAMIC_SHARED(gpu_fe, smem)
#else
    extern __shared__ gpu_fe smem[];
#endif
    gpu_fe* s0 = smem;
    gpu_fe* s1 = smem + blockDim.x;
    gpu_fe* s2 = smem + 2 * blockDim.x;
    gpu_fe* s3 = smem + 3 * blockDim.x;
    s0[tid] = lp0; s1[tid] = lp1; s2[tid] = lp2; s3[tid] = lp3;
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
        d_p0[blockIdx.x] = s0[0]; d_p1[blockIdx.x] = s1[0];
        d_p2[blockIdx.x] = s2[0]; d_p3[blockIdx.x] = s3[0];
    }
}

static __global__ void fold_ptable_kernel4(const gpu_fe* __restrict__ d_src, gpu_fe* d_dst,
    gpu_fe lag0, gpu_fe lag1, gpu_fe lag2, gpu_fe lag3, long new_size, long stride) {
    long tail = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tail >= new_size) return;
    gpu_fe p0 = d_src[0 * stride + tail];
    gpu_fe p1 = d_src[1 * stride + tail];
    gpu_fe p2 = d_src[2 * stride + tail];
    gpu_fe p3 = d_src[3 * stride + tail];
    d_dst[tail] = device_add(device_add(device_mul(lag0, p0), device_mul(lag1, p1)),
                             device_add(device_mul(lag2, p2), device_mul(lag3, p3)));
}

inline void compute_lagrange_deg3_at(const FieldElement& rho,
    FieldElement& lag0, FieldElement& lag1, FieldElement& lag2, FieldElement& lag3) {
    FieldElement inv2, inv6;
    FieldElement::inv(inv2, FieldElement(2));
    FieldElement::inv(inv6, FieldElement(6));
    FieldElement u1 = rho - FieldElement(1), u2 = rho - FieldElement(2), u3 = rho - FieldElement(3);
    lag0 = FieldElement(0) - inv6 * u1 * u2 * u3;
    lag1 = inv2 * rho * u2 * u3;
    lag2 = (FieldElement(0) - inv2) * rho * u1 * u3;
    lag3 = inv6 * rho * u1 * u2;
}

static inline FieldElement deg3_reduce_local(const gpu_fe* d_arr, int nb) {
    std::vector<gpu_fe> h(nb);
    HIP_CHECK(hipMemcpy(h.data(), d_arr, nb * sizeof(gpu_fe), hipMemcpyDeviceToHost));
    FieldElement acc; acc.clear();
    for (auto& v : h) acc += fe_from_gpu(v);
    return acc;
}

inline void gpu_sumcheck_deg3_staged(int l, const FieldElement* r_pt,
    const FieldElement* f_host, const FieldElement* g_host, FieldElement S, int k,
    std::vector<FieldElement>& r_out, FieldElement& claim_f, FieldElement& claim_g,
    const FieldElement* test_challenges = nullptr) {

    const int N = 1 << l;
    const int threads = 256;
    r_out.assign(l, FieldElement(0));

    std::vector<FieldElement> eq(N);
    if (l == 0) { eq[0] = FieldElement(1); }
    else {
        eq[0] = FieldElement(1) - r_pt[0]; eq[1] = r_pt[0];
        for (int i = 2; i <= l; ++i)
            for (int j = (1 << i) - 1; j >= 0; --j) {
                if (j & (1 << (i - 1))) eq[j] = eq[j - (1 << (i - 1))] * r_pt[i - 1];
                else                    eq[j] = eq[j] * (FieldElement(1) - r_pt[i - 1]);
            }
    }

    if (k < 1) k = 1;
    int l_stage = (l + k - 1) / k;
    if (l_stage < 1) l_stage = 1;
    if (l_stage > l) l_stage = (l > 0) ? l : 1;

    int tile_bits = gkr_config.sc_tile_bits;
    if (const char* e = getenv("GKR_SC_TILE_BITS")) tile_bits = atoi(e);

    {
        const int  fv_pre  = (l - l_stage > 0) ? (l - l_stage) : 0;
        const int  fvc_pre = (tile_bits >= 0 && tile_bits < fv_pre) ? tile_bits : fv_pre;
        const double need_gib =
            3.0 * (double)(pow4(l_stage)) * (double)((size_t)1 << fvc_pre)
            * sizeof(gpu_fe) / (1024.0*1024.0*1024.0);
        size_t fr = 0, tot = 0;
        if (hipMemGetInfo(&fr, &tot) == hipSuccess && tot > 0) {
            const double tot_gib = (double)tot / (1024.0*1024.0*1024.0);
            if (need_gib > tot_gib)
                throw std::invalid_argument(
                    "deg3 extension tables need " + std::to_string(need_gib) +
                    " GiB, device has " + std::to_string(tot_gib) + " GiB (l=" +
                    std::to_string(l) + " k=" + std::to_string(k) + " l_stage=" +
                    std::to_string(l_stage) + " tile_bits=" + std::to_string(tile_bits) +
                    "). Raise k (narrower stages) or lower --sc-tile-bits.");
        }
    }

    gpu_fe *dE, *dF, *dG, *dE2, *dF2, *dG2;
    HIP_CHECK(GKR_DMALLOC(&dE, (size_t)N * sizeof(gpu_fe)));
    HIP_CHECK(GKR_DMALLOC(&dF, (size_t)N * sizeof(gpu_fe)));
    HIP_CHECK(GKR_DMALLOC(&dG, (size_t)N * sizeof(gpu_fe)));
    HIP_CHECK(GKR_DMALLOC(&dE2, (size_t)N * sizeof(gpu_fe)));
    HIP_CHECK(GKR_DMALLOC(&dF2, (size_t)N * sizeof(gpu_fe)));
    HIP_CHECK(GKR_DMALLOC(&dG2, (size_t)N * sizeof(gpu_fe)));
    gpu_fe *origE = dE, *origF = dF, *origG = dG, *origE2 = dE2, *origF2 = dF2, *origG2 = dG2;
    {
        std::vector<gpu_fe> hE(N), hF(N), hG(N);
        for (int i = 0; i < N; ++i) { hE[i] = fe_to_gpu(eq[i]); hF[i] = fe_to_gpu(f_host[i]); hG[i] = fe_to_gpu(g_host[i]); }
        HIP_CHECK(gkr_h2d(dE, hE.data(), (size_t)N * sizeof(gpu_fe)));
        HIP_CHECK(gkr_h2d(dF, hF.data(), (size_t)N * sizeof(gpu_fe)));
        HIP_CHECK(gkr_h2d(dG, hG.data(), (size_t)N * sizeof(gpu_fe)));
    }

    const long ptable_cap = pow4(l_stage);
    const int  fv0 = (l - l_stage > 0) ? (l - l_stage) : 0;
    const int  fv_cap = (tile_bits >= 0 && tile_bits < fv0) ? tile_bits : fv0;
    const long ext_cap = ptable_cap << fv_cap;
    if (getenv("GKR_DEG3_DBG"))
        fprintf(stderr, "[deg3] l=%d k=%d l_stage=%d t=%d fv0=%d fv_cap=%d ext=%.1f MiB\n",
                l, k, l_stage, tile_bits, fv0, fv_cap,
                3.0 * ext_cap * sizeof(gpu_fe) / (1024.0*1024.0));
    gpu_fe *dEe, *dFe, *dGe, *dPs, *dPd, *dp0, *dp1, *dp2, *dp3;
    HIP_CHECK(GKR_DMALLOC(&dEe, (size_t)ext_cap * sizeof(gpu_fe)));
    HIP_CHECK(GKR_DMALLOC(&dFe, (size_t)ext_cap * sizeof(gpu_fe)));
    HIP_CHECK(GKR_DMALLOC(&dGe, (size_t)ext_cap * sizeof(gpu_fe)));
    HIP_CHECK(GKR_DMALLOC(&dPs, (size_t)ptable_cap * sizeof(gpu_fe)));
    HIP_CHECK(GKR_DMALLOC(&dPd, (size_t)ptable_cap * sizeof(gpu_fe)));
    const int target_blocks = ptable_target_blocks();
    gpu_fe* dPart;
    HIP_CHECK(GKR_DMALLOC(&dPart, (size_t)(ptable_cap + target_blocks) * sizeof(gpu_fe)));
    int round_max_blocks = std::max(1, (int)(((1L << (l_stage - 1)) + threads - 1) / threads));
    HIP_CHECK(GKR_DMALLOC(&dp0, round_max_blocks * sizeof(gpu_fe)));
    HIP_CHECK(GKR_DMALLOC(&dp1, round_max_blocks * sizeof(gpu_fe)));
    HIP_CHECK(GKR_DMALLOC(&dp2, round_max_blocks * sizeof(gpu_fe)));
    HIP_CHECK(GKR_DMALLOC(&dp3, round_max_blocks * sizeof(gpu_fe)));

    FieldElement claim = S;
    bool ok = true;
    int processed = 0, global_round = 0;

    while (processed < l) {
        const int stage_l    = std::min(l_stage, l - processed);
        const long pow4_l    = pow4(stage_l);
        const int future_vars= l - processed - stage_l;
        const int  tile_lo       = (tile_bits >= 0 && tile_bits < future_vars) ? tile_bits : future_vars;
        const int  num_tiles_bits= future_vars - tile_lo;
        const long tile_b3       = (long)1 << tile_lo;
        const long tile_cells    = pow4_l << tile_lo;

        HIP_CHECK(hipMemset(dPs, 0, (size_t)pow4_l * sizeof(gpu_fe)));

        const int b_exp = (int)(((((long)1 << stage_l) << tile_lo)) + threads - 1) / threads;
        const int b_ext = (int)((tile_cells + threads - 1) / threads);
        const int b_pt  = (int)((pow4_l + threads - 1) / threads);

        for (long ti = 0; ti < ((long)1 << num_tiles_bits); ++ti) {
            const long in_off = (ti << tile_lo) << stage_l;

            HIP_CHECK(hipMemset(dEe, 0, (size_t)tile_cells * sizeof(gpu_fe)));
            HIP_CHECK(hipMemset(dFe, 0, (size_t)tile_cells * sizeof(gpu_fe)));
            HIP_CHECK(hipMemset(dGe, 0, (size_t)tile_cells * sizeof(gpu_fe)));
            hipLaunchKernelGGL(expand_bk_kernel4, dim3(b_exp), dim3(threads), 0, 0, dE + in_off, dEe, stage_l, tile_lo, pow4_l);
            hipLaunchKernelGGL(expand_bk_kernel4, dim3(b_exp), dim3(threads), 0, 0, dF + in_off, dFe, stage_l, tile_lo, pow4_l);
            hipLaunchKernelGGL(expand_bk_kernel4, dim3(b_exp), dim3(threads), 0, 0, dG + in_off, dGe, stage_l, tile_lo, pow4_l);
            HIP_CHECK(hipDeviceSynchronize());

            for (int var = 0; var < stage_l; ++var) {
                hipLaunchKernelGGL(extrapolate_bk_var_kernel4, dim3(b_ext), dim3(threads), 0, 0, dEe, var, stage_l, tile_lo, pow4_l);
                hipLaunchKernelGGL(extrapolate_bk_var_kernel4, dim3(b_ext), dim3(threads), 0, 0, dFe, var, stage_l, tile_lo, pow4_l);
                hipLaunchKernelGGL(extrapolate_bk_var_kernel4, dim3(b_ext), dim3(threads), 0, 0, dGe, var, stage_l, tile_lo, pow4_l);
                HIP_CHECK(hipDeviceSynchronize());
            }

            const auto t_pt = std::chrono::steady_clock::now();
            {
                int nb3_blocks = (pow4_l >= target_blocks)
                                     ? 1 : (int)((target_blocks + pow4_l - 1) / pow4_l);
                const long max_b3_blocks = (tile_b3 + threads - 1) / threads;
                if ((long)nb3_blocks > max_b3_blocks) nb3_blocks = (int)std::max(1L, max_b3_blocks);
                const size_t pt_smem = (size_t)threads * sizeof(gpu_fe);
                hipLaunchKernelGGL(ptable3_partial_kernel, dim3((unsigned)pow4_l, (unsigned)nb3_blocks),
                    dim3(threads), pt_smem, 0, dEe, dFe, dGe, dPart, pow4_l, tile_b3, nb3_blocks);
                hipLaunchKernelGGL(ptable3_finalize_kernel, dim3(b_pt), dim3(threads), 0, 0,
                    dPart, dPs, pow4_l, nb3_blocks);
            }
            HIP_CHECK(hipDeviceSynchronize());
            g_deg3_ptable_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_pt).count();
        }

        long cur_ptable_size = pow4_l;
        int  remaining_free  = stage_l;
        for (int r_in = 0; r_in < stage_l; ++r_in) {
            int future_combos = 1 << (remaining_free - 1);
            int rb = std::max(1, (future_combos + threads - 1) / threads);
            size_t smem = 4 * threads * sizeof(gpu_fe);
            hipLaunchKernelGGL(blendy_round_kernel4, dim3(rb), dim3(threads), smem, 0,
                dPs, dp0, dp1, dp2, dp3, remaining_free, future_combos);
            HIP_CHECK(hipDeviceSynchronize());

            FieldElement p0 = deg3_reduce_local(dp0, rb);
            FieldElement p1 = deg3_reduce_local(dp1, rb);
            FieldElement p2 = deg3_reduce_local(dp2, rb);
            FieldElement p3 = deg3_reduce_local(dp3, rb);
            if (p0 + p1 != claim) ok = false;

            FieldElement rho;
            if (test_challenges) rho = test_challenges[global_round];
            else                 rho.setByCSPRNG();
            r_out[global_round] = rho;

            FieldElement lag0, lag1, lag2, lag3;
            compute_lagrange_deg3_at(rho, lag0, lag1, lag2, lag3);
            claim = lag0 * p0 + lag1 * p1 + lag2 * p2 + lag3 * p3;

            long ns = cur_ptable_size / 4;
            if (ns > 0) {
                int fb = std::max(1, (int)((ns + threads - 1) / threads));
                hipLaunchKernelGGL(fold_ptable_kernel4, dim3(fb), dim3(threads), 0, 0,
                    dPs, dPd, fe_to_gpu(lag0), fe_to_gpu(lag1), fe_to_gpu(lag2), fe_to_gpu(lag3), ns, ns);
                std::swap(dPs, dPd);
            }
            cur_ptable_size = ns;

            int oe = 1 << (l - global_round - 1);
            if (oe > 0) {
                int fb = std::max(1, (oe + threads - 1) / threads);
                gpu_fe g_rho = fe_to_gpu(rho), omr = fe_to_gpu(FieldElement(1) - rho);
                hipLaunchKernelGGL(fold_bk_kernel, dim3(fb), dim3(threads), 0, 0, dE, dE2, omr, g_rho, oe, 0);
                hipLaunchKernelGGL(fold_bk_kernel, dim3(fb), dim3(threads), 0, 0, dF, dF2, omr, g_rho, oe, 0);
                hipLaunchKernelGGL(fold_bk_kernel, dim3(fb), dim3(threads), 0, 0, dG, dG2, omr, g_rho, oe, 0);
                HIP_CHECK(hipDeviceSynchronize());
                std::swap(dE, dE2); std::swap(dF, dF2); std::swap(dG, dG2);
            }
            --remaining_free;
            ++global_round;
        }
        processed += stage_l;
    }

    gpu_fe hE0, hF0, hG0;
    HIP_CHECK(hipMemcpy(&hE0, dE, sizeof(gpu_fe), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&hF0, dF, sizeof(gpu_fe), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&hG0, dG, sizeof(gpu_fe), hipMemcpyDeviceToHost));
    FieldElement eq_final = fe_from_gpu(hE0);
    claim_f = fe_from_gpu(hF0);
    claim_g = fe_from_gpu(hG0);
    if (eq_final * claim_f * claim_g != claim) ok = false;
    if (!ok) std::cerr << "[deg3_staged] completeness check failed (l=" << l << " k=" << k << ")\n";

    HIP_CHECK(hipFree(origE)); HIP_CHECK(hipFree(origF)); HIP_CHECK(hipFree(origG));
    HIP_CHECK(hipFree(origE2)); HIP_CHECK(hipFree(origF2)); HIP_CHECK(hipFree(origG2));
    HIP_CHECK(hipFree(dEe)); HIP_CHECK(hipFree(dFe)); HIP_CHECK(hipFree(dGe));
    HIP_CHECK(hipFree(dPs)); HIP_CHECK(hipFree(dPd)); HIP_CHECK(hipFree(dPart));
    HIP_CHECK(hipFree(dp0)); HIP_CHECK(hipFree(dp1)); HIP_CHECK(hipFree(dp2)); HIP_CHECK(hipFree(dp3));
}

}

#endif
