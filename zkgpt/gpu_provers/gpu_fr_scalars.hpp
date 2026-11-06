#pragma once
#include <hip/hip_runtime.h>
#include <mcl/bn.hpp>
#include <vector>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include "g1_tensor.hpp"
#include "fr_tensor.hpp"
#include "g1_msm.hpp"
#include "hip_utils.hpp"
// commit_bridge(), mcl_g1_to_affine(), jac_to_mcl(), mcl_fr_to_normal()
#include "gpu_hyrax_commit.hpp"

namespace zkgpt_gpu {

__device__ __forceinline__ bool zkgpt_ge256(const uint* a, const uint* b) {
    for (int i = 7; i >= 0; --i)
        if (a[i] != b[i])
            return a[i] > b[i];
    return true;  // equal counts as >=, matching cmpGeN
}

__device__ __forceinline__ void zkgpt_sub256(uint* r, const uint* a, const uint* b) {
    unsigned long long borrow = 0;
    for (int i = 0; i < 8; ++i) {
        // A borrow makes the 64-bit result wrap, which shows up as a non-zero high half
        unsigned long long cur = (unsigned long long)a[i] - (unsigned long long)b[i] - borrow;
        r[i] = (uint)(cur & 0xFFFFFFFFull);
        borrow = (cur >> 32) ? 1ull : 0ull;
    }
}

// Montgomery Fr -> signed 128-bit, matching prover.cpp's convert()
__global__ void zkgpt_fr_to_i128_kernel(const Fr_t* __restrict__ in, ll128* __restrict__ out, unsigned n,
                                        Fr_t half, Fr_t modulus) {
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n)
        return;

    Fr_t c = blstrs__scalar__Scalar_unmont(in[gid]);  // canonical limbs
    const bool neg = zkgpt_ge256(c.val, half.val);

    uint mag[8];
    if (neg)
        zkgpt_sub256(mag, modulus.val, c.val);  // |x| = r - canonical(x)
    else
        for (int i = 0; i < 8; ++i)
            mag[i] = c.val[i];

    unsigned long long lo = (unsigned long long)mag[0] | ((unsigned long long)mag[1] << 32);
    unsigned long long hi = (unsigned long long)mag[2] | ((unsigned long long)mag[3] << 32);
    if (neg) {  // two's complement negate
        lo = ~lo;
        hi = ~hi;
        lo += 1ull;
        if (lo == 0ull)
            hi += 1ull;
    }
    out[gid].lo = lo;
    out[gid].hi = hi;
}

__global__ void zkgpt_i128_maxbits_kernel(const ll128* __restrict__ in, unsigned n,
                                          unsigned* __restrict__ out) {
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n)
        return;
    unsigned long long lo = in[gid].lo, hi = in[gid].hi;
    if (hi >> 63) {  // negative: take the magnitude
        lo = ~lo;
        hi = ~hi;
        lo += 1ull;
        if (lo == 0ull)
            hi += 1ull;
    }
    unsigned bits;
    if (hi)
        bits = 64u + (64u - (unsigned)__clzll((long long)hi));
    else if (lo)
        bits = 64u - (unsigned)__clzll((long long)lo);
    else
        return;  // zero contributes nothing
    atomicMax(out, bits);
}

// Host side

// (r+1)/2 and r, as canonical little-endian limbs
struct FrHalfMod {
    Fr_t half{}, mod{};
    FrHalfMod() {
        mcl::bn::Fr h, two;
        two = 2;
        mcl::bn::Fr::inv(h, two);
        half = mcl_fr_to_normal(h);
        unsigned long long carry = 0;  // mod = (half << 1) - 1
        for (int i = 0; i < 8; ++i) {
            unsigned long long v = ((unsigned long long)half.val[i] << 1) | carry;
            mod.val[i] = (uint)(v & 0xFFFFFFFFull);
            carry = v >> 32;
        }
        unsigned long long borrow = 1;
        for (int i = 0; i < 8 && borrow; ++i) {
            unsigned long long cur = (unsigned long long)mod.val[i] - borrow;
            mod.val[i] = (uint)(cur & 0xFFFFFFFFull);
            borrow = (cur >> 32) ? 1ull : 0ull;
        }
    }
};
inline const FrHalfMod& fr_half_mod() {
    static FrHalfMod m;
    return m;
}

// Hyrax row-commitment straight from the Fr witness
inline mcl::bn::G1* gpu_hyrax_commit_fr_rowmajor(const mcl::bn::Fr* w, size_t real_len, const mcl::bn::G1* g,
                                                 int l) {
    commit_bridge();
    const int halfl = l / 2;
    const int rownum = 1 << halfl;
    const int colnum = 1 << (l - halfl);
    const size_t N = (size_t)rownum * colnum;
    if (real_len > N)
        real_len = N;

    const bool prof = getenv("GKR_MSM_PROF") != nullptr;
    auto now = [] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    const double t0 = prof ? now() : 0.0;

    std::vector<G1Affine_t> g_aff(colnum);
    for (int i = 0; i < colnum; ++i)
        g_aff[i] = mcl_g1_to_affine(g[i]);

    ll128* d_scalars = nullptr;
    Fr_t* d_fr = nullptr;
    unsigned* d_bits = nullptr;
    G1Affine_t* d_basis = nullptr;
    G1Jacobian_t* d_out = nullptr;

    HIP_CHECK_THROW(hipMalloc((void**)&d_scalars, N * sizeof(ll128)));
    HIP_CHECK_THROW(hipMemset(d_scalars, 0, N * sizeof(ll128)));

    int scalar_bits = 1;
    if (real_len) {
        // mcl::bn::Fr and Fr_t are both 32 bytes of the same Montgomery limbs
        static_assert(sizeof(Fr_t) == 32, "device scalar must be 32 bytes");
        HIP_CHECK_THROW(hipMalloc((void**)&d_fr, real_len * sizeof(Fr_t)));
        HIP_CHECK_THROW(hipMemcpy(d_fr, w, real_len * sizeof(Fr_t), hipMemcpyHostToDevice));
        const unsigned n = (unsigned)real_len;
        const unsigned blk = 256, grid = (n + blk - 1) / blk;
        hipLaunchKernelGGL(zkgpt_fr_to_i128_kernel, dim3(grid), dim3(blk), 0, 0, d_fr, d_scalars, n,
                           fr_half_mod().half, fr_half_mod().mod);
        HIP_CHECK_THROW(hipGetLastError());

        HIP_CHECK_THROW(hipMalloc((void**)&d_bits, sizeof(unsigned)));
        HIP_CHECK_THROW(hipMemset(d_bits, 0, sizeof(unsigned)));
        hipLaunchKernelGGL(zkgpt_i128_maxbits_kernel, dim3(grid), dim3(blk), 0, 0, d_scalars, n, d_bits);
        HIP_CHECK_THROW(hipGetLastError());
        unsigned bits = 0;
        HIP_CHECK_THROW(hipMemcpy(&bits, d_bits, sizeof(unsigned), hipMemcpyDeviceToHost));
        HIP_CHECK_THROW(hipFree(d_bits));
        HIP_CHECK_THROW(hipFree(d_fr));
        d_bits = nullptr;
        d_fr = nullptr;
        if (bits > 0)
            scalar_bits = (int)bits;
    }
    const double t1 = prof ? now() : 0.0;

    HIP_CHECK_THROW(hipMalloc((void**)&d_basis, (size_t)colnum * sizeof(G1Affine_t)));
    HIP_CHECK_THROW(
        hipMemcpy(d_basis, g_aff.data(), (size_t)colnum * sizeof(G1Affine_t), hipMemcpyHostToDevice));
    HIP_CHECK_THROW(hipMalloc((void**)&d_out, (size_t)rownum * sizeof(G1Jacobian_t)));

    gpu_msm_batched_small(d_basis, d_scalars, d_out, (uint)rownum, (uint)colnum, scalar_bits);
    const double t2 = prof ? now() : 0.0;

    std::vector<G1Jacobian_t> rows(rownum);
    HIP_CHECK_THROW(
        hipMemcpy(rows.data(), d_out, (size_t)rownum * sizeof(G1Jacobian_t), hipMemcpyDeviceToHost));
    HIP_CHECK_THROW(hipFree(d_scalars));
    HIP_CHECK_THROW(hipFree(d_basis));
    HIP_CHECK_THROW(hipFree(d_out));

    mcl::bn::G1* Tk = new mcl::bn::G1[rownum];
    for (int idx = 0; idx < rownum; ++idx)
        Tk[idx] = jac_to_mcl(rows[idx]);
    if (prof)
        fprintf(stderr,
                "[commit_fr] l=%d real=%zu of %zu (%.2f%%) bits=%d  "
                "prep=%.3fs msm=%.3fs jac_to_mcl=%.3fs  (uploaded %.1f MB, "
                "not %.1f MB)\n",
                l, real_len, N, 100.0 * (double)real_len / (double)N, scalar_bits, t1 - t0, t2 - t1,
                now() - t2, (double)real_len * sizeof(Fr_t) / (1024.0 * 1024.0),
                (double)N * sizeof(ll128) / (1024.0 * 1024.0));
    return Tk;
}

}  // namespace zkgpt_gpu
