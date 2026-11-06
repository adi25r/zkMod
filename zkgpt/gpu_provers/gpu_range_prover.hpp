#ifndef ZKGPT_GPU_RANGE_PROVER_HPP
#define ZKGPT_GPU_RANGE_PROVER_HPP
#include <cstdlib>
#include <vector>
#include <utility>
#include "config.hpp"
#include <mcl/bn.hpp>

#include <algorithm>
#include "gpu_field.hpp"
#include "blendy_linear.hpp"
#include "gpu_matmul_prover.hpp" 

namespace zkgpt_gpu {

// calls gpu product sumcheck with relevant parameters
inline std::pair<mcl::bn::Fr, mcl::bn::Fr> gpu_lasso_product_sumcheck(const mcl::bn::Fr* lasso_mult_v, const mcl::bn::Fr* val0, int n, const mcl::bn::Fr* r, mcl::bn::Fr combined_claim) {
    int l_stage = 4;
    // k (stage count) is the primary knob: l_stage = ceil(n/k). Recomputed here
    // from this sumcheck's own n. lasso_l is a manual l_stage override (lower prio).
    if (gkr_config.lasso_k > 0)      l_stage = (n + gkr_config.lasso_k - 1) / gkr_config.lasso_k;
    else if (gkr_config.lasso_l > 0) l_stage = gkr_config.lasso_l;
    if (l_stage < 1) l_stage = 1;
    if (l_stage > n) l_stage = n;

    // At large n the untiled ext table dominates VRAM. `t` (tile bits) is a swept
    // parameter, so nothing is auto-selected here.
    return gpu_product_sumcheck_host(lasso_mult_v, val0, n, r, combined_claim, l_stage);
}

// performs a degree 1 sumceck, where f is the host array, and S is theclaimed sum
// r_out is the l challenges used 
inline mcl::bn::Fr
gpu_sumcheck_deg1(int l, const mcl::bn::Fr* f, mcl::bn::Fr S, int k, std::vector<mcl::bn::Fr>& r_out) {
    using mcl::bn::Fr;
    int N = 1 << l;
    if (k < 1) k = 1;
    if (l > 0 && k > l) k = l;

    std::vector<gpu_fe> h_f(N);
    for (int i = 0; i < N; ++i) h_f[i] = fe_to_gpu(f[i]);
    gpu_fe* d_f = nullptr;
    GKR_DMALLOC(&d_f, (size_t)N * sizeof(gpu_fe));
    gkr_h2d(d_f, h_f.data(), (size_t)N * sizeof(gpu_fe));

    int threads = 256;
    Fr claim = blendy_linear_sumcheck(d_f, N, l, k, S, threads, r_out);

    hipFree(d_f);
    return claim;
}

} // namespace zkgpt_gpu

#endif // ZKGPT_GPU_RANGE_PROVER_HPP
