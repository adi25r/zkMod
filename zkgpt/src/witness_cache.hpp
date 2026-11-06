#pragma once
// witness_cache — persist the built circuit + witness to disk (SSD) so repeated runs
// of the same model config don't re-pay circuit construction (185 s for GPT-2 L=12).
//
// WHAT IS CACHED: exactly the output of neuralNetwork::create()'s BUILD phase — the
// layered circuit (gate lists, subset indices from initSubset) plus the witness
// (prover::val) and the FCONN weight matrices (prover::mat_val). This is the part the
// CPU zkGPT baseline treats as precomputable and excludes from prover_time
// (circuit_initialization_timer, and the per-layer setup before each
// prove_timer.start() in sumcheckInitPhase1/2).
//
// WHAT IS NOT CACHED: the Hyrax input commitment. That is real prover work and is
// re-run on every load (see neuralNetwork::create).
//
// Cache is keyed on the model config; a stale/mismatched file is ignored, never
// silently reused. Format is raw little-endian POD dumps -- valid only for the build
// that wrote it, so the header pins sizeof(F) and a format version.
//
//   ZKGPT_WITNESS_CACHE=<dir>   override cache dir (default <repo>/.witness_cache)
//   ZKGPT_WITNESS_CACHE=0       disable entirely
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>
#include "circuit.h"
#include "prover.hpp"

namespace zkgpt_wc {

static constexpr unsigned kMagic   = 0x5A574332u;   // "ZWC2"
// Bump kVersion whenever any dumped struct's raw POD layout changes (layer, uniGate,
// binGate, prover's fc_* arrays, ...): a file from an older layout has the same key
// and would otherwise be "hit" and reinterpreted, reading garbage sizes.
static constexpr unsigned kVersion = 12u;

// Scalars that buildCircuit() derives and later phases read. A cache hit skips
// buildCircuit, so without persisting these the network is left holding its
// pre-build values: len un-padded (30 vs 32), q_offset/inp_end_id zero. That would
// feed exportSplitBlockCarry a wrong carry and flip executeCommit's separate-input
// branch, so a warm-cache split run would silently prove a different circuit.
struct Derived {
    long long q_offset = 0, len = 0, channel_out = 0, input_feature_dim = 0;
    long long input_slot_size = 0, weight_start_offset = 0, inp_end_id = 0;
    int input_c = 1, input_e = 0;
};

// Keyed on what actually determines the circuit: depth, sequence length, the
// privacy/merge flags, and a hash of every FCONN's (channel_in, channel_out).
// Deriving it from full_conn rather than ctor args means a config that changes the
// circuit can never silently reuse a stale file.
//
// part_idx: the fields above pin the circuit SHAPE, not the witness VALUES. Middle
// split partitions share a shape but carry different inputs, so without this they all
// loaded partition 1's witness and still ACCEPTed. -1 = untagged (single-block runs,
// and the shape-only probe pass).
struct Key {
    int depth, len, output_private, input_private, input_separate, merge;
    int part_idx;
    unsigned long long fc_hash;
};

inline unsigned long long mix64(unsigned long long h, unsigned long long v) {
    h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h;
}

inline bool enabled() {
    const char *e = getenv("ZKGPT_WITNESS_CACHE");
    return !(e && std::string(e) == "0");
}

inline std::string cache_dir() {
    const char *e = getenv("ZKGPT_WITNESS_CACHE");
    if (e && std::string(e) != "0" && std::string(e) != "1") return e;
    return std::string(ZKGPT_SOURCE_DIR) + "/.witness_cache";
}

inline std::string cache_path(const Key &k) {
    char buf[512];
    snprintf(buf, sizeof(buf), "%s/wc_v%u_d%d_len%d_o%d_i%d_s%d_m%d_p%d_%016llx.bin",
             cache_dir().c_str(), kVersion, k.depth, k.len,
             k.output_private, k.input_private, k.input_separate, k.merge,
             k.part_idx, k.fc_hash);
    return buf;
}

// Field-by-field, not memcmp: Key has tail padding, and comparing padding bytes
// makes a hit depend on whether the writing and reading builds happened to zero them.
inline bool same_key(const Key &a, const Key &b) {
    return a.depth == b.depth && a.len == b.len
        && a.output_private == b.output_private && a.input_private == b.input_private
        && a.input_separate == b.input_separate && a.merge == b.merge
        && a.part_idx == b.part_idx && a.fc_hash == b.fc_hash;
}

// ---- raw POD helpers -------------------------------------------------------
template <class T> inline bool wr(FILE *f, const T &v) { return fwrite(&v, sizeof(T), 1, f) == 1; }
template <class T> inline bool rd(FILE *f, T &v)       { return fread(&v, sizeof(T), 1, f) == 1; }
template <class T> inline bool wrvec(FILE *f, const std::vector<T> &v) {
    u64 n = v.size();
    if (!wr(f, n)) return false;
    return n == 0 || fwrite(v.data(), sizeof(T), n, f) == n;
}
// Not resize(n)+fread: resize() value-initializes each element, which needs T
// default-constructible -- uniGate/binGate (layer.uni_gates/bin_gates) only have a
// parameterized constructor. Both are trivially copyable (fixed-size POD members,
// no user destructor), so read the raw bytes into a buffer and copy-construct the
// vector from it instead -- same "read a byte-exact struct dump" contract the rest
// of this file uses via fwrite(&pod, ...), just needs a temporary buffer here since
// the destination is being SIZED, not already sized.
template <class T> inline bool rdvec(FILE *f, std::vector<T> &v) {
    u64 n = 0;
    if (!rd(f, n)) return false;
    if (n == 0) { v.clear(); return true; }
    std::vector<char> buf(n * sizeof(T));
    if (fread(buf.data(), sizeof(T), n, f) != n) return false;
    const T *p = reinterpret_cast<const T *>(buf.data());
    v.assign(p, p + n);
    return true;
}

// Set by a successful load(). GKR_WITGEN_DEVICE's proving children clear it before
// build_partition and require it after: a miss there would silently rebuild the
// partition with no carry-in -- a self-consistent proof of the wrong input.
inline bool &last_load_hit() { static bool hit = false; return hit; }

// ---- save ------------------------------------------------------------------
// nfc = number of FCONN weight matrices in pr.mat_val; matrix i has fc_col[i]*fc_row[i] ints.
inline bool save(const Key &k, const prover &pr, int nfc, const Derived &d) {
    if (!enabled()) return false;
    ::mkdir(cache_dir().c_str(), 0755);
    const std::string path = cache_path(k);
    // Unique tmp name: concurrent partitions write different entries at once, and a
    // shared ".tmp" let them scribble over each other.
    const std::string tmp  = path + ".tmp" + std::to_string((long)getpid());
    FILE *f = fopen(tmp.c_str(), "wb");
    if (!f) return false;
    bool ok = true;
    unsigned magic = kMagic, ver = kVersion, fsz = (unsigned)sizeof(F);
    ok &= wr(f, magic) && wr(f, ver) && wr(f, fsz) && wr(f, k) && wr(f, d);

    // prover POD arrays
    ok &= fwrite(pr.fc_row, sizeof(pr.fc_row), 1, f) == 1;
    ok &= fwrite(pr.fc_col, sizeof(pr.fc_col), 1, f) == 1;
    ok &= fwrite(pr.fc_input_row, sizeof(pr.fc_input_row), 1, f) == 1;
    ok &= fwrite(pr.fc_input_col, sizeof(pr.fc_input_col), 1, f) == 1;
    ok &= fwrite(pr.fc_input_id, sizeof(pr.fc_input_id), 1, f) == 1;
    ok &= fwrite(pr.fc_start_id, sizeof(pr.fc_start_id), 1, f) == 1;
    ok &= fwrite(pr.fc_real_row, sizeof(pr.fc_real_row), 1, f) == 1;
    ok &= fwrite(pr.fc_real_col, sizeof(pr.fc_real_col), 1, f) == 1;
    ok &= fwrite(pr.fc_real_input_row, sizeof(pr.fc_real_input_row), 1, f) == 1;
    ok &= fwrite(pr.fc_real_input_col, sizeof(pr.fc_real_input_col), 1, f) == 1;

    // circuit (post-initSubset: ori_id_*/size_*/bit_length_* are part of `layer`)
    u8 csz = pr.C.size;
    ok &= wr(f, csz);
    u64 nlayers = pr.C.circuit.size();
    ok &= wr(f, nlayers);
    for (const auto &L : pr.C.circuit) {
        ok &= wr(f, L.ty) && wr(f, L.size) && wr(f, L.need_phase2)
           && wr(f, L.zero_start_id) && wr(f, L.fft_bit_length)
           && wr(f, L.bit_length) && wr(f, L.max_bl_u) && wr(f, L.max_bl_v);
        ok &= fwrite(L.size_u, sizeof(L.size_u), 1, f) == 1;
        ok &= fwrite(L.size_v, sizeof(L.size_v), 1, f) == 1;
        ok &= fwrite(L.bit_length_u, sizeof(L.bit_length_u), 1, f) == 1;
        ok &= fwrite(L.bit_length_v, sizeof(L.bit_length_v), 1, f) == 1;
        ok &= wrvec(f, L.uni_interval) && wrvec(f, L.bin_interval)
           && wrvec(f, L.uni_gates)    && wrvec(f, L.bin_gates)
           && wrvec(f, L.ori_id_u)     && wrvec(f, L.ori_id_v);
    }

    // witness
    u64 nval = pr.val.size();
    ok &= wr(f, nval);
    for (const auto &v : pr.val) ok &= wrvec(f, v);

    // FCONN weights
    int nfc_w = nfc;
    ok &= wr(f, nfc_w);
    for (int i = 0; i < nfc && ok; ++i) {
        u64 n = (u64)pr.fc_col[i] * (u64)pr.fc_row[i];
        ok &= wr(f, n);
        ok &= (n == 0) || (fwrite(pr.mat_val[i], sizeof(int), n, f) == n);
    }

    fclose(f);
    if (!ok) { remove(tmp.c_str()); return false; }
    return rename(tmp.c_str(), path.c_str()) == 0;   // atomic publish
}

// ---- load ------------------------------------------------------------------
inline bool load(const Key &k, prover &pr, int &nfc_out, Derived &d_out) {
    if (!enabled()) return false;
    FILE *f = fopen(cache_path(k).c_str(), "rb");
    if (!f) return false;
    bool ok = true;
    unsigned magic = 0, ver = 0, fsz = 0; Key kk{};
    ok &= rd(f, magic) && rd(f, ver) && rd(f, fsz) && rd(f, kk) && rd(f, d_out);
    if (!ok || magic != kMagic || ver != kVersion || fsz != sizeof(F) ||
        !same_key(kk, k)) { fclose(f); return false; }

    ok &= fread(pr.fc_row, sizeof(pr.fc_row), 1, f) == 1;
    ok &= fread(pr.fc_col, sizeof(pr.fc_col), 1, f) == 1;
    ok &= fread(pr.fc_input_row, sizeof(pr.fc_input_row), 1, f) == 1;
    ok &= fread(pr.fc_input_col, sizeof(pr.fc_input_col), 1, f) == 1;
    ok &= fread(pr.fc_input_id, sizeof(pr.fc_input_id), 1, f) == 1;
    ok &= fread(pr.fc_start_id, sizeof(pr.fc_start_id), 1, f) == 1;
    ok &= fread(pr.fc_real_row, sizeof(pr.fc_real_row), 1, f) == 1;
    ok &= fread(pr.fc_real_col, sizeof(pr.fc_real_col), 1, f) == 1;
    ok &= fread(pr.fc_real_input_row, sizeof(pr.fc_real_input_row), 1, f) == 1;
    ok &= fread(pr.fc_real_input_col, sizeof(pr.fc_real_input_col), 1, f) == 1;

    u8 csz = 0; u64 nlayers = 0;
    ok &= rd(f, csz) && rd(f, nlayers);
    if (!ok) { fclose(f); return false; }
    pr.C.size = csz;
    pr.C.circuit.assign(nlayers, layer());
    for (auto &L : pr.C.circuit) {
        ok &= rd(f, L.ty) && rd(f, L.size) && rd(f, L.need_phase2)
           && rd(f, L.zero_start_id) && rd(f, L.fft_bit_length)
           && rd(f, L.bit_length) && rd(f, L.max_bl_u) && rd(f, L.max_bl_v);
        ok &= fread(L.size_u, sizeof(L.size_u), 1, f) == 1;
        ok &= fread(L.size_v, sizeof(L.size_v), 1, f) == 1;
        ok &= fread(L.bit_length_u, sizeof(L.bit_length_u), 1, f) == 1;
        ok &= fread(L.bit_length_v, sizeof(L.bit_length_v), 1, f) == 1;
        ok &= rdvec(f, L.uni_interval) && rdvec(f, L.bin_interval)
           && rdvec(f, L.uni_gates)    && rdvec(f, L.bin_gates)
           && rdvec(f, L.ori_id_u)     && rdvec(f, L.ori_id_v);
        if (!ok) break;
    }

    u64 nval = 0;
    ok &= rd(f, nval);
    if (ok) {
        pr.val.assign(nval, std::vector<F>());
        for (auto &v : pr.val) { ok &= rdvec(f, v); if (!ok) break; }
    }

    int nfc = 0;
    ok &= rd(f, nfc);
    if (ok) {
        int **mv = new int *[nfc > 0 ? nfc : 1];
        for (int i = 0; i < nfc && ok; ++i) {
            u64 n = 0;
            ok &= rd(f, n);
            mv[i] = new int[n ? n : 1];
            ok &= (n == 0) || (fread(mv[i], sizeof(int), n, f) == n);
        }
        if (ok) { pr.mat_val = mv; nfc_out = nfc; }
    }

    fclose(f);
    if (ok) last_load_hit() = true;
    return ok;
}

// ---- partial load: one witness layer ---------------------------------------
// Reads val[layer_idx] without materialising the rest of the file. Used by the
// --concurrent runner: partition i needs partition i-1's output layer as the `w` of
// its input boundary commitment, and that array is already on disk here (8.6 GB at
// l=28, so not worth a second copy). Padding is not stored; the caller re-pads.
// False means missing/stale/short — fatal, never "commit to zeros".
// Path-based rather than Key-based: the consuming child knows the producer's cache
// path (it comes over in the boundary-commit file) but would have to reconstruct the
// producer's full Key, including its FCONN hash, to derive it.
inline bool load_layer_path(const std::string &path, int layer_idx, std::vector<F> &out) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    auto fail = [&] { fclose(f); return false; };

    unsigned magic = 0, ver = 0, fsz = 0; Key kk{}; Derived dd{};
    if (!(rd(f, magic) && rd(f, ver) && rd(f, fsz) && rd(f, kk) && rd(f, dd))) return fail();
    if (magic != kMagic || ver != kVersion || fsz != sizeof(F)) return fail();

    // Fixed-size prover POD block, ten arrays, in save()'s order.
    const long pod = (long)(sizeof(prover::fc_row) + sizeof(prover::fc_col)
                          + sizeof(prover::fc_input_row) + sizeof(prover::fc_input_col)
                          + sizeof(prover::fc_input_id) + sizeof(prover::fc_start_id)
                          + sizeof(prover::fc_real_row) + sizeof(prover::fc_real_col)
                          + sizeof(prover::fc_real_input_row) + sizeof(prover::fc_real_input_col));
    if (fseek(f, pod, SEEK_CUR) != 0) return fail();

    u8 csz = 0; u64 nlayers = 0;
    if (!(rd(f, csz) && rd(f, nlayers))) return fail();
    auto skip_vec = [&](std::size_t elem) {
        u64 n = 0;
        if (!rd(f, n)) return false;
        return fseek(f, (long)(n * elem), SEEK_CUR) == 0;
    };
    for (u64 i = 0; i < nlayers; ++i) {
        layer L;
        if (!(rd(f, L.ty) && rd(f, L.size) && rd(f, L.need_phase2)
           && rd(f, L.zero_start_id) && rd(f, L.fft_bit_length)
           && rd(f, L.bit_length) && rd(f, L.max_bl_u) && rd(f, L.max_bl_v))) return fail();
        if (fseek(f, (long)(sizeof(L.size_u) + sizeof(L.size_v)
                          + sizeof(L.bit_length_u) + sizeof(L.bit_length_v)), SEEK_CUR) != 0)
            return fail();
        if (!skip_vec(sizeof(L.uni_interval[0]))  || !skip_vec(sizeof(L.bin_interval[0]))
         || !skip_vec(sizeof(L.uni_gates[0]))     || !skip_vec(sizeof(L.bin_gates[0]))
         || !skip_vec(sizeof(L.ori_id_u[0]))      || !skip_vec(sizeof(L.ori_id_v[0]))) return fail();
    }

    u64 nval = 0;
    if (!rd(f, nval)) return fail();
    if (layer_idx < 0 || (u64)layer_idx >= nval) return fail();
    for (int i = 0; i < layer_idx; ++i)
        if (!skip_vec(sizeof(F))) return fail();
    if (!rdvec(f, out)) return fail();
    fclose(f);
    return true;
}

inline bool load_layer(const Key &k, int layer_idx, std::vector<F> &out) {
    return enabled() && load_layer_path(cache_path(k), layer_idx, out);
}

} // namespace zkgpt_wc
