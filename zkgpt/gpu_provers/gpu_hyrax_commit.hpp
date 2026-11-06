// gpu_hyrax_commit.hpp — GPU Hyrax row-commitments via the BN254-reparameterized
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

namespace zkgpt_gpu {

// sets up the parameters for the BN254 setup
struct CommitBridge {
    mcl::bn::Fp Rmodp, Rinv;  // R mod p, R^-1 mod p (R = 2^256, BN254 Nogami Fq)
    bool inited = false;
    void init() {
        if (inited)
            return;
        Rmodp.setStr("15003436851221201713926160155297504393934861937006144945855291451476606451598", 10);
        Rinv.setStr("11963795830096653542573556069761433511859298666393959442070332436494060174150", 10);
        inited = true;
    }
};

inline CommitBridge& commit_bridge() {
    static CommitBridge b;
    b.init();
    return b;
}

// given an FR, convert to little-endian form for the GPU
inline Fr_t mcl_fr_to_normal(const mcl::bn::Fr& s) {
    uint8_t buf[32] = {0};
    s.getLittleEndian(buf, 32);
    Fr_t out{};
    for (int i = 0; i < 8; ++i) {
        unsigned v = 0;
        for (int b = 0; b < 4; ++b)
            v |= (unsigned)buf[i * 4 + b] << (8 * b);
        out.val[i] = v;
    }
    return out;
}

// Exact __int128 -> Fr (mod r), full range and sign
inline mcl::bn::Fr ll_to_fr(__int128 v) {
    const bool neg = v < 0;
    unsigned __int128 u = neg ? (unsigned __int128)0 - (unsigned __int128)v : (unsigned __int128)v;
    uint8_t buf[16];
    for (int i = 0; i < 16; ++i)
        buf[i] = (uint8_t)(u >> (8 * i));
    mcl::bn::Fr r;
    r.setLittleEndianMod(buf, 16);
    if (neg)
        mcl::bn::Fr::neg(r, r);
    return r;
}

// convert an Fp to Montgomery form for the GPU
inline void mcl_fp_to_mont(const mcl::bn::Fp& v, unsigned out[8]) {
    mcl::bn::Fp m;
    mcl::bn::Fp::mul(m, v, commit_bridge().Rmodp);
    uint8_t buf[32] = {0};
    m.getLittleEndian(buf, 32);
    for (int i = 0; i < 8; ++i) {
        unsigned x = 0;
        for (int b = 0; b < 4; ++b)
            x |= (unsigned)buf[i * 4 + b] << (8 * b);
        out[i] = x;
    }
}

inline G1Affine_t mcl_g1_to_affine(const mcl::bn::G1& P_) {
    mcl::bn::G1 P = P_;
    P.normalize();
    G1Affine_t a;
    mcl_fp_to_mont(P.x, a.x.val);
    mcl_fp_to_mont(P.y, a.y.val);
    return a;
}

inline mcl::bn::Fp mont_to_mcl_fp(const Fp_t& f) {
    uint8_t buf[32] = {0};
    for (int i = 0; i < 8; ++i)
        for (int b = 0; b < 4; ++b)
            buf[i * 4 + b] = (f.val[i] >> (8 * b)) & 0xff;
    mcl::bn::Fp m;
    m.setLittleEndianMod(buf, 32);
    mcl::bn::Fp r;
    mcl::bn::Fp::mul(r, m, commit_bridge().Rinv);
    return r;
}

inline mcl::bn::G1 jac_to_mcl(const G1Jacobian_t& j) {
    using namespace mcl::bn;
    Fp X = mont_to_mcl_fp(j.x), Y = mont_to_mcl_fp(j.y), Z = mont_to_mcl_fp(j.z);
    G1 P;
    if (Z.isZero()) {
        P.clear();
        return P;
    }
    Fp z2, z3, iz2, iz3, ax, ay;
    Fp::sqr(z2, Z);
    Fp::mul(z3, z2, Z);
    Fp::inv(iz2, z2);
    Fp::inv(iz3, z3);
    Fp::mul(ax, X, iz2);
    Fp::mul(ay, Y, iz3);
    P.x = ax;
    P.y = ay;
    P.z = 1;
    return P;
}

// Returns Tk[rownum] (new[]) = row commitments of w (length 2^l) against bases g[colnum]
inline mcl::bn::G1* gpu_hyrax_commit(const mcl::bn::Fr* w, const mcl::bn::G1* g, int l) {
    commit_bridge();
    const int halfl = l / 2;
    const int rownum = 1 << halfl;
    const int colnum = 1 << (l - halfl);
    const long N = (long)rownum * colnum;

    // Batched shared-basis Pippenger
    const bool prof = getenv("GKR_MSM_PROF") != nullptr;
    auto now = [] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    const double t0 = prof ? now() : 0.0;

    std::vector<Fr_t> scal(N);
    std::vector<G1Affine_t> g_aff(colnum);
    for (int i = 0; i < colnum; ++i)
        g_aff[i] = mcl_g1_to_affine(g[i]);
// N serial mcl_fr_to_normal calls
#pragma omp parallel for schedule(static)
    for (int idx = 0; idx < rownum; ++idx)
        for (int i = 0; i < colnum; ++i)
            scal[(long)idx * colnum + i] = mcl_fr_to_normal(w[idx + (long)i * rownum]);
    const double t1 = prof ? now() : 0.0;

    std::vector<G1Jacobian_t> rows(rownum);
    gpu_msm_batched_from_host(g_aff.data(), scal.data(), rows.data(), (uint)rownum, (uint)colnum);
    const double t2 = prof ? now() : 0.0;

    mcl::bn::G1* Tk = new mcl::bn::G1[rownum];
    for (int idx = 0; idx < rownum; ++idx)
        Tk[idx] = jac_to_mcl(rows[idx]);
    if (prof)
        fprintf(stderr, "[commit] l=%d scal_prep=%.3fs msm=%.3fs jac_to_mcl=%.3fs\n", l, t1 - t0, t2 - t1,
                now() - t2);
    return Tk;
}

// Integer-scalar variant: Tk[idx] = Sum_i w[idx + i*rownum] * g[i], w in [signed] ll
inline mcl::bn::G1* gpu_hyrax_commit_ll(const __int128* w, const mcl::bn::G1* g, int l) {
    commit_bridge();
    const int halfl = l / 2;
    const int rownum = 1 << halfl;
    const int colnum = 1 << (l - halfl);
    const long N = (long)rownum * colnum;

    const bool prof = getenv("GKR_MSM_PROF") != nullptr;
    auto now = [] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    const double t0 = prof ? now() : 0.0;

    // SMALL-SCALAR path
    std::vector<__int128> scal(N);
    std::vector<G1Affine_t> g_aff(colnum);
    for (int i = 0; i < colnum; ++i)
        g_aff[i] = mcl_g1_to_affine(g[i]);
#pragma omp parallel for schedule(static)
    for (int idx = 0; idx < rownum; ++idx)
        for (int i = 0; i < colnum; ++i)
            scal[(long)idx * colnum + i] = w[idx + (long)i * rownum];
    const double t1 = prof ? now() : 0.0;

    std::vector<G1Jacobian_t> rows(rownum);
    gpu_msm_batched_small_from_host(g_aff.data(), scal.data(), rows.data(), (uint)rownum, (uint)colnum);
    const double t2 = prof ? now() : 0.0;

    mcl::bn::G1* Tk = new mcl::bn::G1[rownum];
    for (int idx = 0; idx < rownum; ++idx)
        Tk[idx] = jac_to_mcl(rows[idx]);
    if (prof)
        fprintf(stderr, "[commit_ll] l=%d scal_prep=%.3fs msm=%.3fs jac_to_mcl=%.3fs\n", l, t1 - t0, t2 - t1,
                now() - t2);
    return Tk;
}

// ROW-MAJOR integer variant: Tk[idx] = Sum_i w[idx*colnum + i] * g[i]
inline mcl::bn::G1* gpu_hyrax_commit_ll_rowmajor(const __int128* w, const mcl::bn::G1* g, int l) {
    commit_bridge();
    const int halfl = l / 2;
    const int rownum = 1 << halfl;
    const int colnum = 1 << (l - halfl);
    const long N = (long)rownum * colnum;

    const bool prof = getenv("GKR_MSM_PROF") != nullptr;
    auto now = [] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    const double t0 = prof ? now() : 0.0;

    // Already row-major
    std::vector<G1Affine_t> g_aff(colnum);
    for (int i = 0; i < colnum; ++i)
        g_aff[i] = mcl_g1_to_affine(g[i]);
    const double t1 = prof ? now() : 0.0;

    std::vector<G1Jacobian_t> rows(rownum);
    gpu_msm_batched_small_from_host(g_aff.data(), w, rows.data(), (uint)rownum, (uint)colnum);
    const double t2 = prof ? now() : 0.0;

    mcl::bn::G1* Tk = new mcl::bn::G1[rownum];
    for (int idx = 0; idx < rownum; ++idx)
        Tk[idx] = jac_to_mcl(rows[idx]);
    if (prof)
        fprintf(stderr, "[commit_rm] l=%d scal_prep=%.3fs msm=%.3fs jac_to_mcl=%.3fs\n", l, t1 - t0, t2 - t1,
                now() - t2);
    return Tk;
}

}  // namespace zkgpt_gpu
