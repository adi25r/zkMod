#ifndef BLENDYSC_HPP
#define BLENDYSC_HPP

#include "gpu_field.hpp"
#include "common.hpp"

#include <vector>
#include <iostream>
#include <cstdlib>
#include <algorithm>

// Minimum grid size for the two-level ptable reductions (ptable_partial_kernel,
// ptable3_partial_kernel): when the ptable has fewer cells than this, each cell's b3
// sum is split over ceil(target / cells) blocks. Default: 2 blocks per CU of the
// current device. Env GKR_PTABLE_BLOCKS overrides.
//
// 2 blocks/CU measured fastest at every production shape that splits; 1/CU, >=4/CU and
// 1 block per cell are slower, and results are identical at every count. The occupancy
// API cannot derive this: it is blind to the register use of the -fgpu-rdc field calls
// (device_mul lives in libbls12_381_core).
inline int ptable_target_blocks() {
    if (const char* e = getenv("GKR_PTABLE_BLOCKS")) return std::max(1, atoi(e));
    static int cus = 0;
    if (cus <= 0) {
        int dev = 0;
        HIP_CHECK(hipGetDevice(&dev));
        HIP_CHECK(hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, dev));
    }
    return 2 * cus;
}

// ── Kernels ───────────────────────────────────────────────────────────────

// Copy {0,1}^l_stage evaluations of bk into the {0,1} slots of a base-3
// array indexed by (b3 || b2_base3), leaving the "2" digits at zero.
static __global__ void expand_bk_kernel(const gpu_fe* d_bk, gpu_fe* d_bk_ext, int l_stage, int future_vars, int pow3_l) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = (1 << l_stage) * (1 << future_vars);
    if (idx >= total) return;

    int b2_bin = idx & ((1 << l_stage) - 1);
    int b3     = idx >> l_stage;

    int b2_base3 = 0;
    for (int i = 0; i < l_stage; ++i)
        b2_base3 += ((b2_bin >> i) & 1) * pow3(l_stage - 1 - i);

    d_bk_ext[b3 * pow3_l + b2_base3] = d_bk[idx];
}

// Extrapolate f(2) = 2*f(1) - f(0) for each entry where digit `var` == 2.
// Valid because each bk is multilinear (degree-1 in every variable).
static __global__ void extrapolate_bk_var_kernel(gpu_fe* d_bk_ext, int var, int l_stage, int future_vars, gpu_fe mont_TWO, int pow3_l) {
    int total = pow3_l * (1 << future_vars);
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int b3 = idx / pow3_l;
    int b2_base3 = idx % pow3_l;
    int pow3_var = pow3(var);
    int digit = (b2_base3 / pow3_var) % 3;

    if (digit == 2) {
        gpu_fe v0 = d_bk_ext[b3 * pow3_l + (b2_base3 - 2 * pow3_var)];
        gpu_fe v1 = d_bk_ext[b3 * pow3_l + (b2_base3 - 1 * pow3_var)];
        d_bk_ext[idx] = device_sub(device_mul(mont_TWO, v1), v0);
    }
}

// Two-level segmented reduction for the tiled ptable build: parallelize the b3 axis
// rather than walking it serially with pow3_l (3..81) threads. _partial sums b3 per
// (b2, blk) into d_partials; _finalize ADDS them into d_ptable, so tiles accumulate
// into a pre-zeroed table. A pure reassociation of the same sum -- base-3 layout,
// digit conventions and round loop are unchanged; only the grouping differs.
static __global__ void ptable_partial_kernel(const gpu_fe* __restrict__ d_bkA_ext,
    const gpu_fe* __restrict__ d_bkB_ext, gpu_fe* d_partials,
    int pow3_l, long tile_b3, int nb3_blocks)
{
    int b2  = blockIdx.x;                 // which base-3 cell (large dim on x)
    int blk = blockIdx.y;                 // which b3-chunk
    int tid = threadIdx.x;
    const long stride = (long)nb3_blocks * blockDim.x;
    gpu_fe acc = {0};
    for (long b3 = (long)blk * blockDim.x + tid; b3 < tile_b3; b3 += stride) {
        long idx = b3 * (long)pow3_l + b2;
        acc = device_add(acc, device_mul(d_bkA_ext[idx], d_bkB_ext[idx]));
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
    if (tid == 0) d_partials[(long)b2 * nb3_blocks + blk] = smem[0];
}

static __global__ void ptable_finalize_kernel(const gpu_fe* __restrict__ d_partials,
    gpu_fe* d_ptable, int pow3_l, int nb3_blocks)
{
    int b2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (b2 >= pow3_l) return;
    gpu_fe acc = d_ptable[b2];            // add into the (pre-zeroed) accumulator
    const long base = (long)b2 * nb3_blocks;
    for (int i = 0; i < nb3_blocks; ++i)
        acc = device_add(acc, d_partials[base + i]);
    d_ptable[b2] = acc;
}

// Compute p(0), p(1), p(2) by summing the three top-digit slices of ptable
// over all binary tails. Uses shared-memory block reduction.
static __global__ void blendy_round_kernel(const gpu_fe* __restrict__ d_ptable,
    gpu_fe* d_out_p0, gpu_fe* d_out_p1, gpu_fe* d_out_p2,
    int remaining_free, int total_threads)
{
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    gpu_fe lp0 = {0}, lp1 = {0}, lp2 = {0};
    if (idx < total_threads) {
        int b2_tail = 0;
        for (int i = 0; i < remaining_free - 1; ++i)
            b2_tail += ((idx >> i) & 1) * pow3(i);
        int stride = pow3(remaining_free - 1);
        lp0 = d_ptable[0 * stride + b2_tail];
        lp1 = d_ptable[1 * stride + b2_tail];
        lp2 = d_ptable[2 * stride + b2_tail];
    }

#ifdef __HIP_CPU_RT__
    HIP_DYNAMIC_SHARED(gpu_fe, smem)
#else
    extern __shared__ gpu_fe smem[];
#endif
    gpu_fe* s0 = smem;
    gpu_fe* s1 = smem + blockDim.x;
    gpu_fe* s2 = smem + 2 * blockDim.x;
    s0[tid] = lp0; s1[tid] = lp1; s2[tid] = lp2;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s0[tid] = device_add(s0[tid], s0[tid + s]);
            s1[tid] = device_add(s1[tid], s1[tid + s]);
            s2[tid] = device_add(s2[tid], s2[tid + s]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        d_out_p0[blockIdx.x] = s0[0];
        d_out_p1[blockIdx.x] = s1[0];
        d_out_p2[blockIdx.x] = s2[0];
    }
}

// Fold ptable: collapse top base-3 digit with Lagrange coefficients.
// ptable shrinks from 3^remaining_free to 3^(remaining_free-1).
static __global__ void fold_ptable_kernel(const gpu_fe* __restrict__ d_src, gpu_fe* d_dst,
    gpu_fe lag0, gpu_fe lag1, gpu_fe lag2, int new_size, int stride)
{
    int tail = blockIdx.x * blockDim.x + threadIdx.x;
    if (tail >= new_size) return;

    gpu_fe p0 = d_src[0 * stride + tail];
    gpu_fe p1 = d_src[1 * stride + tail];
    gpu_fe p2 = d_src[2 * stride + tail];
    d_dst[tail] = device_add(device_add(device_mul(lag0, p0),
                                         device_mul(lag1, p1)),
                              device_mul(lag2, p2));
}

// Multilinear fold of a bk array by rho:
//   bk_new[i] = (1-rho)*bk[i|bit_k=0] + rho*bk[i|bit_k=1]
static __global__ void fold_bk_kernel(const gpu_fe* d_src, gpu_fe* d_dst,
    gpu_fe one_minus_rho, gpu_fe rho, int total_elements_out, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements_out) return;

    int low_mask = (1 << k) - 1;
    int low  = idx & low_mask;
    int high = (idx >> k) << (k + 1);
    gpu_fe v0 = d_src[high | low];
    gpu_fe v1 = d_src[high | (1 << k) | low];
    d_dst[idx] = device_add(device_mul(one_minus_rho, v0), device_mul(rho, v1));
}

// Combined l-variable fold at a STAGE BOUNDARY. Binds the l lowest index bits of
// d_src in one pass, using a precomputed eq table d_eq (length 1<<l) where
//   d_eq[b] = prod_{i=0}^{l-1} ( bit_i(b) ? rho[i] : 1-rho[i] ).
// Output d_dst[j] = sum_{b=0}^{(1<<l)-1} d_eq[b] * d_src[(j<<l) | b], out_elems = in>>l.
//
// This is exactly equal to l sequential fold_bk_kernel(...,k=0) folds with
// rho[0..l-1] (verified: round r binds current bit 0 with rho[r], so old bit i
// pairs with rho[i]), but it NEVER materializes the N/2, N/4, ... intermediates --
// the dst only ever needs in>>l elements, not in/2. That is what turns the 3N peak
// (= 2N sources + 2*(N/2) dsts) into 2N + 2*(N>>l_stage), i.e. the fold-destination
// term becomes k-dependent. With a precomputed eq table each src element is read
// once and multiplied once, so this is also FEWER field mults than the sequential
// chain (which re-reads/re-multiplies ~2N times).
static __global__ void stage_fold_bk_kernel(const gpu_fe* __restrict__ d_src,
    const gpu_fe* __restrict__ d_eq, gpu_fe* d_dst, int l, long out_elems)
{
    long j = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= out_elems) return;
    const long base = j << l;
    const int  nb   = 1 << l;
    gpu_fe acc = {0};
    for (int b = 0; b < nb; ++b)
        acc = device_add(acc, device_mul(d_eq[b], d_src[base | b]));
    d_dst[j] = acc;
}

// ── Host helper ───────────────────────────────────────────────────────────

// Degree-2 Lagrange basis at points {0, 1, 2}, evaluated at rho.
inline void compute_lagrange_at(const FieldElement& rho,
    FieldElement& lag0, FieldElement& lag1, FieldElement& lag2)
{
    FieldElement one = 1, two = 2;
    FieldElement two_inv;
    FieldElement::inv(two_inv, two);
    lag0 = (rho - one) * (rho - two) * two_inv;
    lag1 = rho * (two - rho);
    lag2 = rho * (rho - one) * two_inv;
}

#endif // BLENDYSC_HPP
