#pragma once
// witness_cache — persist the built circuit + witness to disk
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

static constexpr unsigned kMagic = 0x5A574332u;  // "ZWC2"
// Bump kVersion whenever any dumped struct's raw POD layout changes
static constexpr unsigned kVersion = 12u;

// Scalars that buildCircuit() derives and later phases read
struct Derived {
    long long q_offset = 0, len = 0, channel_out = 0, input_feature_dim = 0;
    long long input_slot_size = 0, weight_start_offset = 0, inp_end_id = 0;
    int input_c = 1, input_e = 0;
};

// Keyed on what determines the circuit
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
    const char* e = getenv("ZKGPT_WITNESS_CACHE");
    return !(e && std::string(e) == "0");
}

inline std::string cache_dir() {
    const char* e = getenv("ZKGPT_WITNESS_CACHE");
    if (e && std::string(e) != "0" && std::string(e) != "1")
        return e;
    return std::string(ZKGPT_SOURCE_DIR) + "/.witness_cache";
}

inline std::string cache_path(const Key& k) {
    char buf[512];
    snprintf(buf, sizeof(buf), "%s/wc_v%u_d%d_len%d_o%d_i%d_s%d_m%d_p%d_%016llx.bin", cache_dir().c_str(),
             kVersion, k.depth, k.len, k.output_private, k.input_private, k.input_separate, k.merge,
             k.part_idx, k.fc_hash);
    return buf;
}

// Field-by-field
inline bool same_key(const Key& a, const Key& b) {
    return a.depth == b.depth && a.len == b.len && a.output_private == b.output_private &&
           a.input_private == b.input_private && a.input_separate == b.input_separate && a.merge == b.merge &&
           a.part_idx == b.part_idx && a.fc_hash == b.fc_hash;
}

// raw POD helpers
template <class T>
inline bool wr(FILE* f, const T& v) {
    return fwrite(&v, sizeof(T), 1, f) == 1;
}
template <class T>
inline bool rd(FILE* f, T& v) {
    return fread(&v, sizeof(T), 1, f) == 1;
}
template <class T>
inline bool wrvec(FILE* f, const std::vector<T>& v) {
    u64 n = v.size();
    if (!wr(f, n))
        return false;
    return n == 0 || fwrite(v.data(), sizeof(T), n, f) == n;
}
// Not resize(n)+fread
template <class T>
inline bool rdvec(FILE* f, std::vector<T>& v) {
    u64 n = 0;
    if (!rd(f, n))
        return false;
    if (n == 0) {
        v.clear();
        return true;
    }
    std::vector<char> buf(n * sizeof(T));
    if (fread(buf.data(), sizeof(T), n, f) != n)
        return false;
    const T* p = reinterpret_cast<const T*>(buf.data());
    v.assign(p, p + n);
    return true;
}

// Set by a successful load()
inline bool& last_load_hit() {
    static bool hit = false;
    return hit;
}

inline bool save(const Key& k, const prover& pr, int nfc, const Derived& d) {
    if (!enabled())
        return false;
    ::mkdir(cache_dir().c_str(), 0755);
    const std::string path = cache_path(k);
    // Unique tmp name
    const std::string tmp = path + ".tmp" + std::to_string((long)getpid());
    FILE* f = fopen(tmp.c_str(), "wb");
    if (!f)
        return false;
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
    for (const auto& L : pr.C.circuit) {
        ok &= wr(f, L.ty) && wr(f, L.size) && wr(f, L.need_phase2) && wr(f, L.zero_start_id) &&
              wr(f, L.fft_bit_length) && wr(f, L.bit_length) && wr(f, L.max_bl_u) && wr(f, L.max_bl_v);
        ok &= fwrite(L.size_u, sizeof(L.size_u), 1, f) == 1;
        ok &= fwrite(L.size_v, sizeof(L.size_v), 1, f) == 1;
        ok &= fwrite(L.bit_length_u, sizeof(L.bit_length_u), 1, f) == 1;
        ok &= fwrite(L.bit_length_v, sizeof(L.bit_length_v), 1, f) == 1;
        ok &= wrvec(f, L.uni_interval) && wrvec(f, L.bin_interval) && wrvec(f, L.uni_gates) &&
              wrvec(f, L.bin_gates) && wrvec(f, L.ori_id_u) && wrvec(f, L.ori_id_v);
    }

    // witness
    u64 nval = pr.val.size();
    ok &= wr(f, nval);
    for (const auto& v : pr.val)
        ok &= wrvec(f, v);

    // FCONN weights
    int nfc_w = nfc;
    ok &= wr(f, nfc_w);
    for (int i = 0; i < nfc && ok; ++i) {
        u64 n = (u64)pr.fc_col[i] * (u64)pr.fc_row[i];
        ok &= wr(f, n);
        ok &= (n == 0) || (fwrite(pr.mat_val[i], sizeof(int), n, f) == n);
    }

    fclose(f);
    if (!ok) {
        remove(tmp.c_str());
        return false;
    }
    return rename(tmp.c_str(), path.c_str()) == 0;  // atomic publish
}

// load
inline bool load(const Key& k, prover& pr, int& nfc_out, Derived& d_out) {
    if (!enabled())
        return false;
    FILE* f = fopen(cache_path(k).c_str(), "rb");
    if (!f)
        return false;
    bool ok = true;
    unsigned magic = 0, ver = 0, fsz = 0;
    Key kk{};
    ok &= rd(f, magic) && rd(f, ver) && rd(f, fsz) && rd(f, kk) && rd(f, d_out);
    if (!ok || magic != kMagic || ver != kVersion || fsz != sizeof(F) || !same_key(kk, k)) {
        fclose(f);
        return false;
    }

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

    u8 csz = 0;
    u64 nlayers = 0;
    ok &= rd(f, csz) && rd(f, nlayers);
    if (!ok) {
        fclose(f);
        return false;
    }
    pr.C.size = csz;
    pr.C.circuit.assign(nlayers, layer());
    for (auto& L : pr.C.circuit) {
        ok &= rd(f, L.ty) && rd(f, L.size) && rd(f, L.need_phase2) && rd(f, L.zero_start_id) &&
              rd(f, L.fft_bit_length) && rd(f, L.bit_length) && rd(f, L.max_bl_u) && rd(f, L.max_bl_v);
        ok &= fread(L.size_u, sizeof(L.size_u), 1, f) == 1;
        ok &= fread(L.size_v, sizeof(L.size_v), 1, f) == 1;
        ok &= fread(L.bit_length_u, sizeof(L.bit_length_u), 1, f) == 1;
        ok &= fread(L.bit_length_v, sizeof(L.bit_length_v), 1, f) == 1;
        ok &= rdvec(f, L.uni_interval) && rdvec(f, L.bin_interval) && rdvec(f, L.uni_gates) &&
              rdvec(f, L.bin_gates) && rdvec(f, L.ori_id_u) && rdvec(f, L.ori_id_v);
        if (!ok)
            break;
    }

    u64 nval = 0;
    ok &= rd(f, nval);
    if (ok) {
        pr.val.assign(nval, std::vector<F>());
        for (auto& v : pr.val) {
            ok &= rdvec(f, v);
            if (!ok)
                break;
        }
    }

    int nfc = 0;
    ok &= rd(f, nfc);
    if (ok) {
        int** mv = new int*[nfc > 0 ? nfc : 1];
        for (int i = 0; i < nfc && ok; ++i) {
            u64 n = 0;
            ok &= rd(f, n);
            mv[i] = new int[n ? n : 1];
            ok &= (n == 0) || (fread(mv[i], sizeof(int), n, f) == n);
        }
        if (ok) {
            pr.mat_val = mv;
            nfc_out = nfc;
        }
    }

    fclose(f);
    if (ok)
        last_load_hit() = true;
    return ok;
}

// partial load
inline bool load_layer_path(const std::string& path, int layer_idx, std::vector<F>& out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f)
        return false;
    auto fail = [&] {
        fclose(f);
        return false;
    };

    unsigned magic = 0, ver = 0, fsz = 0;
    Key kk{};
    Derived dd{};
    if (!(rd(f, magic) && rd(f, ver) && rd(f, fsz) && rd(f, kk) && rd(f, dd)))
        return fail();
    if (magic != kMagic || ver != kVersion || fsz != sizeof(F))
        return fail();

    const long pod =
        (long)(sizeof(prover::fc_row) + sizeof(prover::fc_col) + sizeof(prover::fc_input_row) +
               sizeof(prover::fc_input_col) + sizeof(prover::fc_input_id) + sizeof(prover::fc_start_id) +
               sizeof(prover::fc_real_row) + sizeof(prover::fc_real_col) + sizeof(prover::fc_real_input_row) +
               sizeof(prover::fc_real_input_col));
    if (fseek(f, pod, SEEK_CUR) != 0)
        return fail();

    u8 csz = 0;
    u64 nlayers = 0;
    if (!(rd(f, csz) && rd(f, nlayers)))
        return fail();
    auto skip_vec = [&](std::size_t elem) {
        u64 n = 0;
        if (!rd(f, n))
            return false;
        return fseek(f, (long)(n * elem), SEEK_CUR) == 0;
    };
    for (u64 i = 0; i < nlayers; ++i) {
        layer L;
        if (!(rd(f, L.ty) && rd(f, L.size) && rd(f, L.need_phase2) && rd(f, L.zero_start_id) &&
              rd(f, L.fft_bit_length) && rd(f, L.bit_length) && rd(f, L.max_bl_u) && rd(f, L.max_bl_v)))
            return fail();
        if (fseek(
                f,
                (long)(sizeof(L.size_u) + sizeof(L.size_v) + sizeof(L.bit_length_u) + sizeof(L.bit_length_v)),
                SEEK_CUR) != 0)
            return fail();
        if (!skip_vec(sizeof(L.uni_interval[0])) || !skip_vec(sizeof(L.bin_interval[0])) ||
            !skip_vec(sizeof(L.uni_gates[0])) || !skip_vec(sizeof(L.bin_gates[0])) ||
            !skip_vec(sizeof(L.ori_id_u[0])) || !skip_vec(sizeof(L.ori_id_v[0])))
            return fail();
    }

    u64 nval = 0;
    if (!rd(f, nval))
        return fail();
    if (layer_idx < 0 || (u64)layer_idx >= nval)
        return fail();
    for (int i = 0; i < layer_idx; ++i)
        if (!skip_vec(sizeof(F)))
            return fail();
    if (!rdvec(f, out))
        return fail();
    fclose(f);
    return true;
}

inline bool load_layer(const Key& k, int layer_idx, std::vector<F>& out) {
    return enabled() && load_layer_path(cache_path(k), layer_idx, out);
}

}  // namespace zkgpt_wc
