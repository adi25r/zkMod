#pragma once

#include <algorithm>
#include <iostream>
#include <vector>

#include "prover.hpp"
#include "hyrax.hpp"
#include "utils.hpp"

// RAII wrapper around Commit_return — owns the underlying arrays.
struct OwnedCommitment {
    int l = 0;
    std::vector<Fr> w;
    std::vector<ll> ww;
    std::vector<G1> gens;
    std::vector<G1> comm;
    G1 G;

    void clear() {
        l = 0;
        w.clear(); ww.clear(); gens.clear(); comm.clear(); G.clear();
    }

    bool empty() const { return l == 0 || comm.empty(); }

    void capture(const Commit_return &src) {
        clear();
        if (src.l <= 0 || !src.w || !src.ww || !src.g || !src.comm) return;
        l = src.l;
        std::size_t witness_count = std::size_t(1) << l;
        std::size_t gen_count    = (std::size_t(1) << (l - (l >> 1))) + 1;
        std::size_t comm_count   = std::size_t(1) << (l >> 1);
        w.assign(src.w, src.w + witness_count);
        ww.assign(src.ww, src.ww + witness_count);
        gens.assign(src.g, src.g + gen_count);
        comm.assign(src.comm, src.comm + comm_count);
        G = src.G;
    }

    Commit_return view() {
        Commit_return ret{};
        if (empty()) return ret;
        ret.l    = l;
        ret.w    = w.data();
        ret.ww   = ww.data();
        ret.g    = gens.data();
        ret.G    = G;
        ret.comm = comm.data();
        return ret;
    }
};

inline u64 next_pow2_size(u64 n) {
    return 1ULL << ceilPow2BitLength(static_cast<u32>(n));
}

inline u64 shared_commitment_size(u64 lhs_size, u64 rhs_size) {
    return next_pow2_size(std::max(lhs_size, rhs_size));
}

inline void pad_output_layer(prover &p, u64 target_size) {
    auto &out_layer = p.C.circuit[p.C.size - 1];
    if (static_cast<u64>(out_layer.size) >= target_size) return;
    const u64 padded_size = next_pow2_size(target_size);
    std::cout << "Padding output layer from " << out_layer.size << " to " << padded_size << std::endl;
    auto &out_vals = p.val[p.C.size - 1];
    auto original_size = out_vals.size();
    // Record what was real before the pad, so the boundary commit's scalar conversion
    // can skip the tail. The commitment still covers the full padded domain -- that is
    // what the successor opens against -- only the conversion work shrinks.
    p.output_real_len = static_cast<i64>(original_size);
    out_vals.resize(padded_size);
    {
        // Zero the 2^25 tail on several threads: one thread first-touching it was
        // a serial ~1 GiB page-fault walk on every non-last partition.
        F *ov = out_vals.data();
        #pragma omp parallel for schedule(static) num_threads(std::max(1, p.max_threads))
        for (long long i = (long long)original_size; i < (long long)padded_size; ++i) ov[i].clear();
    }
    out_layer.size = static_cast<int>(padded_size);
    out_layer.bit_length = ceilPow2BitLength(static_cast<u32>(padded_size));
    out_layer.updateSize();
}

inline void commit_private_output_for_next(prover &p, u64 shared_commit_size, int max_threads) {
    int logn = ceilPow2BitLength(static_cast<u32>(shared_commit_size));
    u64 n_sqrt = 1ULL << (logn - (logn >> 1));
    p.gens_output.clear();
    p.gens_output.resize(n_sqrt);
    G1 base = gen_gi(p.gens_output.data(), static_cast<int>(n_sqrt));
    p.gens_output.push_back(base);
    // keep_ww=false: main_gkr_split never calls OwnedCommitment::capture, so the host
    // __int128 mirror is write-only here and the commit can run entirely on the device.
    p.commitOutputPrivate(p.gens_output, max_threads, /*keep_ww=*/false);
}

// Same commit under an externally supplied basis (trailing element = the base G
// aliases). --concurrent needs it: gen_gi is per-process CSPRNG, so a child picking
// its own basis would commit under a different one than its successor opens against —
// silently, since both proofs still verify alone. Still routed through
// commitOutputPrivate so output_commit_time stays charged to the child, which is what
// keeps prover_core_s comparable with the sequential runner.
inline void commit_private_output_with_gens(prover &p, const std::vector<G1> &gens,
                                            int max_threads) {
    p.gens_output = gens;
    p.commitOutputPrivate(p.gens_output, max_threads);
}
