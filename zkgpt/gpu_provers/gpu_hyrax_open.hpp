#pragma once
// GPU bullet-fold Hyrax open. Requires -fgpu-rdc so G1Jacobian_mul /
// blstrs__g1__G1Affine_add link here.


#ifndef ZKGPT_GPU_HYRAX_OPEN_HPP
#define ZKGPT_GPU_HYRAX_OPEN_HPP

#include <hip/hip_runtime.h>
#include <mcl/bn.hpp>
#include <vector>
#include <cassert>
#include <iostream>
#include <cstdlib>
#include <cstdint>

#include "gpu_field.hpp"        // gpu_fe, HIP_CHECK
#include "gpu_uvm_alloc.hpp"    // gkr_h2d/gkr_d2h byte-counted memcpy wrappers
#include "g1_tensor.hpp"        // G1TensorJacobian, G1Jacobian_t, G1_ONE, Fr_t
#include "fr_tensor.hpp"        // FrTensor
#include "gpu_hyrax_commit.hpp" // mcl_g1_to_affine, mcl_fr_to_normal, jac_to_mcl

namespace zkgpt_gpu {

// ── Kernels ───────────────────────────────────────────────────────────────────

// Block-parallel MSM: out[blockIdx.x] = Σ_{idx in block} d_x[x_off+idx] * d_g[g_off+idx]
// G1Jacobian_mul and blstrs__g1__G1Affine_add are device fns accessible via -fgpu-rdc.
static __global__ void g1_msm_partial_kernel(
    const G1Jacobian_t* __restrict__ d_g, int g_off,
    const Fr_t*         __restrict__ d_x, int x_off,
    int n,
    G1Jacobian_t* d_out)
{
    extern __shared__ G1Jacobian_t smem_g1[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Identity: Z=0 in blstrs Jacobian convention.
    G1Jacobian_t local;
    local.x = blstrs__fp__Fp_ZERO;
    local.y = blstrs__fp__Fp_ONE;
    local.z = blstrs__fp__Fp_ZERO;

    if (idx < n)
        local = G1Jacobian_mul(d_g[g_off + idx], d_x[x_off + idx]);

    smem_g1[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            smem_g1[tid] = blstrs__g1__G1Affine_add(smem_g1[tid], smem_g1[tid + s]);
        __syncthreads();
    }
    if (tid == 0) d_out[blockIdx.x] = smem_g1[0];
}

// G1 fold: d_out[i] = d_g[i]*coeff_lo + d_g[i+half]*coeff_hi
static __global__ void g1_bullet_fold_kernel(
    const G1Jacobian_t* __restrict__ d_g,
    G1Jacobian_t* d_out,
    Fr_t coeff_lo, Fr_t coeff_hi,
    int half)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < half) {
        G1Jacobian_t lo = G1Jacobian_mul(d_g[i],        coeff_lo);
        G1Jacobian_t hi = G1Jacobian_mul(d_g[i + half], coeff_hi);
        d_out[i] = blstrs__g1__G1Affine_add(lo, hi);
    }
}

// Convert Montgomery-form gpu_fe[] → normal-form Fr_t[] for G1Jacobian_mul.
// MonMul(x_mont, 1) = x_mont * R^{-1} = x_normal.
static __global__ void fr_fe_to_normal_kernel(
    const gpu_fe* __restrict__ d_in, Fr_t* d_out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        gpu_fe one = {};
        one.val[0] = 1;
        d_out[i] = device_mul(d_in[i], one);
    }
}

// Fr inner product block reduction: out[blockIdx.x] = Σ a[a_off+i] * b[b_off+i]
static __global__ void fr_ip_kernel(
    const gpu_fe* __restrict__ a, int a_off,
    const gpu_fe* __restrict__ b, int b_off,
    int n, gpu_fe* out)
{
    extern __shared__ gpu_fe smem_fe[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    gpu_fe local = {};
    if (idx < n)
        local = device_mul(a[a_off + idx], b[b_off + idx]);
    smem_fe[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem_fe[tid] = device_add(smem_fe[tid], smem_fe[tid + s]);
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = smem_fe[0];
}

// Fr linear fold (Montgomery form): out[i] = c0*in[i] + c1*in[i+half]
static __global__ void fr_fold_kernel(
    const gpu_fe* __restrict__ in, gpu_fe* out,
    gpu_fe c0, gpu_fe c1, int half)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < half)
        out[i] = device_add(device_mul(c0, in[i]), device_mul(c1, in[i + half]));
}

// ── Helpers ───────────────────────────────────────────────────────────────────



// mcl G1 -> blstrs Jacobian. Identity MUST map to z=0: the kernels test z==0 only,
// so (0,0,1) would silently corrupt any MSM containing a zero commitment.
static inline G1Jacobian_t mcl_g1_to_jac(const mcl::bn::G1& P) {
    if (P.isZero()) return blstrs__g1__G1Affine_ZERO;
    G1Affine_t a = mcl_g1_to_affine(P);
    G1Jacobian_t j; j.x = a.x; j.y = a.y; j.z = G1_ONE;
    return j;
}

static std::vector<G1Jacobian_t> g1_to_jac_vec(const mcl::bn::G1* g, int n) {
    std::vector<G1Jacobian_t> out(n);
    for (int i = 0; i < n; ++i)
        out[i] = mcl_g1_to_jac(g[i]);
    return out;
}

// Reduce Fr partial sums (device) → single MCL Fr (host).
static mcl::bn::Fr fr_reduce_partials(gpu_fe* d_partial, int nblocks) {
    std::vector<gpu_fe> h(nblocks);
    HIP_CHECK(hipMemcpy(h.data(), d_partial, nblocks * sizeof(gpu_fe), hipMemcpyDeviceToHost));
    mcl::bn::Fr acc; acc.clear();
    for (int i = 0; i < nblocks; ++i) acc += fe_from_gpu(h[i]);
    return acc;
}

// Reduce block partial sums (device) → single MCL G1 (host).
static mcl::bn::G1 gpu_reduce_partials(G1Jacobian_t* d_partial, int nblocks) {
    std::vector<G1Jacobian_t> h(nblocks);
    HIP_CHECK(hipMemcpy(h.data(), d_partial,
                        nblocks * sizeof(G1Jacobian_t), hipMemcpyDeviceToHost));
    mcl::bn::G1 acc = jac_to_mcl(h[0]);
    for (int i = 1; i < nblocks; ++i)
        acc += jac_to_mcl(h[i]);
    return acc;
}

// ── GPU bullet_reduce ─────────────────────────────────────────────────────────
// 7 kernels queued per round, then one sync. CPU only samples c and accumulates
// gamma_w / y_w. x is kept both Montgomery (d_x_fe) and normal (d_x_msm).
static void gpu_bullet_reduce(
    mcl::bn::G1 gamma,
    const mcl::bn::Fr* a, const mcl::bn::Fr* x, mcl::bn::Fr y,
    const mcl::bn::G1* g_cpu, mcl::bn::G1& G, int n)
{
    using mcl::bn::Fr; using mcl::bn::G1;

    Fr y_w = y;
    G1 gamma_w = gamma;

    const int threads = 64;
    int max_blocks = (n / 2 + threads - 1) / threads;

    // Convert inputs to GPU formats.
    std::vector<gpu_fe>       a_fe(n), x_fe(n);
    std::vector<G1Jacobian_t> g_jac = g1_to_jac_vec(g_cpu, n);
    for (int i = 0; i < n; ++i) {
        a_fe[i] = fe_to_gpu(a[i]);
        x_fe[i] = fe_to_gpu(x[i]);
    }

    gpu_fe       *d_a,       *d_a2;
    gpu_fe       *d_x_fe,    *d_x_fe2;
    Fr_t         *d_x_msm;
    G1Jacobian_t *d_g,       *d_g2;
    G1Jacobian_t *d_g_part,  *d_g_part2;
    gpu_fe       *d_ip_part, *d_ip_part2;

    HIP_CHECK(hipMalloc(&d_a,        n          * sizeof(gpu_fe)));
    HIP_CHECK(hipMalloc(&d_a2,       n          * sizeof(gpu_fe)));
    HIP_CHECK(hipMalloc(&d_x_fe,     n          * sizeof(gpu_fe)));
    HIP_CHECK(hipMalloc(&d_x_fe2,    n          * sizeof(gpu_fe)));
    HIP_CHECK(hipMalloc(&d_x_msm,    n          * sizeof(Fr_t)));
    HIP_CHECK(hipMalloc(&d_g,        n          * sizeof(G1Jacobian_t)));
    HIP_CHECK(hipMalloc(&d_g2,       n          * sizeof(G1Jacobian_t)));
    HIP_CHECK(hipMalloc(&d_g_part,   max_blocks * sizeof(G1Jacobian_t)));
    HIP_CHECK(hipMalloc(&d_g_part2,  max_blocks * sizeof(G1Jacobian_t)));
    HIP_CHECK(hipMalloc(&d_ip_part,  max_blocks * sizeof(gpu_fe)));
    HIP_CHECK(hipMalloc(&d_ip_part2, max_blocks * sizeof(gpu_fe)));

    HIP_CHECK(gkr_h2d(d_a,    a_fe.data(),  n * sizeof(gpu_fe)));
    HIP_CHECK(gkr_h2d(d_x_fe, x_fe.data(),  n * sizeof(gpu_fe)));
    HIP_CHECK(gkr_h2d(d_g,    g_jac.data(), n * sizeof(G1Jacobian_t)));

    int cur_n = n;
    while (cur_n > 1) {
        int half        = cur_n / 2;
        int nblocks     = (half   + threads - 1) / threads;
        int full_blocks = (cur_n  + threads - 1) / threads;
        size_t g1_smem  = threads * sizeof(G1Jacobian_t);
        size_t fe_smem  = threads * sizeof(gpu_fe);

        // Challenge is random — sample on CPU before queueing GPU work.
        Fr c; c.setByCSPRNG();
        Fr invc; Fr::inv(invc, c);
        Fr c2 = c * c, invc2 = invc * invc;
        gpu_fe c_fe    = fe_to_gpu(c);
        gpu_fe invc_fe = fe_to_gpu(invc);
        Fr_t   c_frt   = mcl_fr_to_normal(c);
        Fr_t invc_frt  = mcl_fr_to_normal(invc);

        // (1) x_fe (Montgomery) → x_msm (normal) for G1Jacobian_mul
        hipLaunchKernelGGL(fr_fe_to_normal_kernel,
            dim3(full_blocks), dim3(threads), 0, 0,
            d_x_fe, d_x_msm, cur_n);

        // (2) Inner products on GPU
        // x1a2 = x[0..half) · a[half..cur_n)
        hipLaunchKernelGGL(fr_ip_kernel,
            dim3(nblocks), dim3(threads), fe_smem, 0,
            d_x_fe, 0, d_a, half, half, d_ip_part);
        // x2a1 = x[half..cur_n) · a[0..half)
        hipLaunchKernelGGL(fr_ip_kernel,
            dim3(nblocks), dim3(threads), fe_smem, 0,
            d_x_fe, half, d_a, 0, half, d_ip_part2);

        // (3) MSMs on GPU
        // msm_hi = Σ x[0..half) * g[half..cur_n)
        hipLaunchKernelGGL(g1_msm_partial_kernel,
            dim3(nblocks), dim3(threads), g1_smem, 0,
            d_g, half, d_x_msm, 0, half, d_g_part);
        // msm_lo = Σ x[half..cur_n) * g[0..half)
        hipLaunchKernelGGL(g1_msm_partial_kernel,
            dim3(nblocks), dim3(threads), g1_smem, 0,
            d_g, 0, d_x_msm, half, half, d_g_part2);

        // (4) Fr folds on GPU
        // a[i]   = a[i]*invc   + a[i+half]*c
        hipLaunchKernelGGL(fr_fold_kernel,
            dim3(nblocks), dim3(threads), 0, 0,
            d_a, d_a2, invc_fe, c_fe, half);
        // x[i]   = x[i]*c      + x[i+half]*invc
        hipLaunchKernelGGL(fr_fold_kernel,
            dim3(nblocks), dim3(threads), 0, 0,
            d_x_fe, d_x_fe2, c_fe, invc_fe, half);

        // (5) G1 fold on GPU
        // g[i] = g[i]*invc + g[i+half]*c
        hipLaunchKernelGGL(g1_bullet_fold_kernel,
            dim3(nblocks), dim3(threads), 0, 0,
            d_g, d_g2, invc_frt, c_frt, half);

        HIP_CHECK(hipDeviceSynchronize());

        Fr x1a2   = fr_reduce_partials(d_ip_part,  nblocks);
        Fr x2a1   = fr_reduce_partials(d_ip_part2, nblocks);
        G1 msm_hi = gpu_reduce_partials(d_g_part,  nblocks);
        G1 msm_lo = gpu_reduce_partials(d_g_part2, nblocks);

        G1 gamma_minus1 = G * x1a2 + msm_hi;
        G1 gamma_1      = G * x2a1 + msm_lo;
        gamma_w = gamma_minus1 * c2 + gamma_1 * invc2 + gamma_w;
        y_w     = c2 * x1a2 + invc2 * x2a1 + y_w;

        // IPA_DEBUG: verify invariant after each round by downloading folded arrays
#ifdef IPA_DEBUG
        {
            std::vector<gpu_fe>       h_x(half), h_a(half);
            std::vector<G1Jacobian_t> h_g(half);
            // d_x_fe2 and d_a2 hold the just-written folded values (before swap)
            HIP_CHECK(hipMemcpy(h_x.data(), d_x_fe2, half*sizeof(gpu_fe),       hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(h_a.data(), d_a2,    half*sizeof(gpu_fe),       hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(h_g.data(), d_g2,    half*sizeof(G1Jacobian_t), hipMemcpyDeviceToHost));
            G1 cpu_gamma = G1(); cpu_gamma.clear();
            Fr cpu_y;    cpu_y.clear();
            for (int i = 0; i < half; ++i) {
                Fr xi = fe_from_gpu(h_x[i]);
                Fr ai = fe_from_gpu(h_a[i]);
                G1 gi = jac_to_mcl(h_g[i]);
                cpu_gamma += gi * xi;
                cpu_y     += ai * xi;
            }
            cpu_gamma += G * cpu_y;
            bool g_ok = (cpu_gamma == gamma_w);
            bool y_ok = (cpu_y    == y_w);
            std::cerr << "[IPA round cur_n=" << cur_n << "->half=" << half
                      << "] gamma_ok=" << g_ok << " y_ok=" << y_ok;
            if (!g_ok) {
                std::string s1, s2;
                gamma_w.getStr(s1); cpu_gamma.getStr(s2);
                std::cerr << "\n  gamma_w   =" << s1
                          << "\n  cpu_gamma =" << s2;
            }
            std::cerr << "\n";
        }
#endif

        std::swap(d_a,    d_a2);
        std::swap(d_x_fe, d_x_fe2);
        std::swap(d_g,    d_g2);
        cur_n = half;
    }

    // Download final a[0], x[0] for base-case assertions.
    gpu_fe h_a0, h_x0;
    HIP_CHECK(hipMemcpy(&h_a0, d_a,    sizeof(gpu_fe), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&h_x0, d_x_fe, sizeof(gpu_fe), hipMemcpyDeviceToHost));
    Fr a0 = fe_from_gpu(h_a0);
    Fr x0 = fe_from_gpu(h_x0);

    if (!(a0 * x0 == y_w)) {
        std::string sa, sx, sy;
        a0.getStr(sa); x0.getStr(sx); y_w.getStr(sy);
        std::cerr << "[IPA] FAIL scalar: a0=" << sa << " x0=" << sx << " y_w=" << sy << "\n";
        assert(false);
    }
    G1Jacobian_t g0_jac;
    HIP_CHECK(hipMemcpy(&g0_jac, d_g, sizeof(G1Jacobian_t), hipMemcpyDeviceToHost));
    G1 g0 = jac_to_mcl(g0_jac);
    G1 expected = g0 * x0 + G * y_w;
    if (!(gamma_w == expected)) {
        std::string sg, se, sx, sy;
        gamma_w.getStr(sg); expected.getStr(se); x0.getStr(sx); y_w.getStr(sy);
        std::cerr << "[IPA] FAIL G1: gamma_w=" << sg
                  << "\n  expected (g0*x0+G*y_w)=" << se
                  << "\n  x0=" << sx << " y_w=" << sy << "\n";
        // Check g0 validity
        if (g0.isValid()) std::cerr << "  g0 is on curve\n";
        else std::cerr << "  g0 is NOT on curve!\n";
        assert(false);
    }

    HIP_CHECK(hipFree(d_a));       HIP_CHECK(hipFree(d_a2));
    HIP_CHECK(hipFree(d_x_fe));    HIP_CHECK(hipFree(d_x_fe2));
    HIP_CHECK(hipFree(d_x_msm));
    HIP_CHECK(hipFree(d_g));       HIP_CHECK(hipFree(d_g2));
    HIP_CHECK(hipFree(d_g_part));  HIP_CHECK(hipFree(d_g_part2));
    HIP_CHECK(hipFree(d_ip_part)); HIP_CHECK(hipFree(d_ip_part2));
}


// ── Shared open logic ─────────────────────────────────────────────────────────

// GPU MSM for tprime: Σ L[i] * comm[i] using G1TensorJacobian * FrTensor.
static mcl::bn::G1 gpu_g1_msm_from_host(
    const mcl::bn::G1* g_cpu, int g_off,
    const mcl::bn::Fr* scalars, int n)
{
    if (n == 0) { mcl::bn::G1 p; p.clear(); return p; }
    std::vector<G1Jacobian_t> gj(n);
    for (int i = 0; i < n; ++i)
        gj[i] = mcl_g1_to_jac(g_cpu[g_off + i]);
    std::vector<Fr_t> sc(n);
    for (int i = 0; i < n; ++i) sc[i] = mcl_fr_to_normal(scalars[i]);
    G1TensorJacobian B((uint)n, gj.data());
    FrTensor S((uint)n, sc.data());
    return jac_to_mcl((B * S).sum());
}

static void open_inner(
    const mcl::bn::Fr* L, const mcl::bn::Fr* R,
    const mcl::bn::Fr* LT,
    int rownum, int colnum,
    mcl::bn::G1* comm, mcl::bn::G1* g,
    mcl::bn::G1& G, mcl::bn::Fr eval)
{
    mcl::bn::G1 tprime = gpu_g1_msm_from_host(comm, 0, L, rownum);
    mcl::bn::G1 gamma  = tprime + G * eval;
    gpu_bullet_reduce(gamma, R, LT, eval, g, G, colnum);
}

static std::vector<mcl::bn::Fr> make_eq(mcl::bn::Fr* r, int l) {
    using mcl::bn::Fr;
    std::vector<Fr> eq(1 << l);
    eq[0] = Fr(1) - r[0]; eq[1] = r[0];
    for (int i = 2; i <= l; ++i)
        for (int j = (1 << i) - 1; j >= 0; --j)
            eq[j] = (j & (1 << (i-1)))
                   ? eq[j - (1 << (i-1))] * r[i-1]
                   : eq[j] * (Fr(1) - r[i-1]);
    return eq;
}

// ── Public API (drop-in for range_proof_open) ─────────────────────────────────

// __int128 (ll) variant: used for c[], t[], f[] integer arrays.
inline void gpu_range_proof_open(
    const __int128* w,
    mcl::bn::Fr* r, mcl::bn::Fr eval,
    mcl::bn::G1& G, mcl::bn::G1* g,
    mcl::bn::G1* comm, int l)
{
    using mcl::bn::Fr;
    int halfl  = l / 2;
    int rownum = 1 << halfl;
    int colnum = 1 << (l - halfl);

    auto L = make_eq(r,         halfl);
    auto R = make_eq(r + halfl, l - halfl);

    std::vector<Fr> LT(colnum, Fr(0));
    #pragma omp parallel for
    for (int i = 0; i < colnum; ++i)
        for (int j = 0; j < rownum; ++j)
            LT[i] += L[j] * Fr((int64_t)w[j + (long)i * rownum]);

    open_inner(L.data(), R.data(), LT.data(), rownum, colnum, comm, g, G, eval);
}

// Fr variant: used for G[], H[] reciprocal arrays.
inline void gpu_range_proof_open(
    const mcl::bn::Fr* w,
    mcl::bn::Fr* r, mcl::bn::Fr eval,
    mcl::bn::G1& G, mcl::bn::G1* g,
    mcl::bn::G1* comm, int l)
{
    using mcl::bn::Fr;
    int halfl  = l / 2;
    int rownum = 1 << halfl;
    int colnum = 1 << (l - halfl);

    auto L = make_eq(r,         halfl);
    auto R = make_eq(r + halfl, l - halfl);

    std::vector<Fr> LT(colnum, Fr(0));
    #pragma omp parallel for
    for (int i = 0; i < colnum; ++i)
        for (int j = 0; j < rownum; ++j)
            LT[i] += L[j] * w[j + (long)i * rownum];

    open_inner(L.data(), R.data(), LT.data(), rownum, colnum, comm, g, G, eval);
}

} // namespace zkgpt_gpu

#endif // ZKGPT_GPU_HYRAX_OPEN_HPP
