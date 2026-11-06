#pragma once
// Partition-boundary linking for the split runner (zkMod only; never used at splits=1).
//
// Partition i's output layer is all zeros (every layer of this arithmetization is a
// checker whose outputs must vanish); the activations that become partition i+1's
// input live in i's INPUT witness, as the contiguous window val[0][q_off, q_off + N)
// that exportSplitBlockCarry ships. i commits that window as part of its local
// input segments (executeCommit), and i+1 commits the same N values, in the same
// order, as the prefix of its own input (cc_input). Each proof opens only its own
// commitment, so before this link nothing tied the two: a prover could hand i+1 any
// input and both proofs still verified alone.
//
// The link: once BOTH commitments exist, draw z in F^log2(N) and check
//     sum_k W_i[q_off + k] eq(z, k)  ==  sum_k P_{i+1}[k] eq(z, k)
// with both sides proven against their Hyrax row commitments. The window sits at an
// arbitrary offset inside i's aligned segments, so the coefficient matrix of each
// segment is not one tensor; it is at most TWO (columns before / after the shift
// wraps a row), each opened with one dot-product proof. Cost: a few small IPAs.
//
// Processes: partition i and i+1 are separate forked children. Each publishes its
// side into a MAP_SHARED slot and bumps an arrival counter; whichever arrives second
// draws z and runs the check, so nobody ever blocks waiting for the other (a child
// holding a proving slot while it waits could deadlock the memory gate). The parent
// then requires every slot to read result==1 before it reports status=OK.
#include <atomic>
#include <chrono>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>

#include "prover.hpp"
#include "hyrax.hpp"
#include "utils.hpp"
#include "timer.hpp"

namespace zkmod_link {

struct SegHdr { long long start; int l; int C; long long row0; long long nrows; };

struct Slot {
    int out_ready, in_ready, arrived, result;   // result: 0 unchecked, 1 ok, -1 mismatch
    int checked_by, out_nseg;
    long long q_off, N;
    double check_s;
    // out area at data, in area at data + area_bytes (see Arena)
};

struct Arena {
    char *base = nullptr;
    size_t slot_bytes = 0, area_bytes = 0;
    int nslots = 0;
    Slot *slot(int b) const { return reinterpret_cast<Slot *>(base + (size_t)b * slot_bytes); }
    char *out_area(int b) const { return base + (size_t)b * slot_bytes + sizeof(Slot); }
    char *in_area(int b) const { return out_area(b) + area_bytes; }
};

inline Arena *&arena() { static Arena *a = nullptr; return a; }

// One MAP_SHARED arena for nslots boundaries, created by the runner BEFORE it forks.
// Lazily faulted: only bytes a publisher writes become resident (a few MiB/slot).
inline bool create(int nslots, long long N_max, int l_max) {
    if (nslots <= 0) return true;
    const size_t C_max = size_t(1) << (l_max - (l_max >> 1));
    const size_t per_seg = sizeof(SegHdr) + sizeof(G1) * (C_max + (size_t)N_max + 3)
                         + sizeof(Fr) * ((size_t)N_max + 2 * C_max);
    auto *a = new Arena;
    a->area_bytes = 16 * per_seg;
    a->slot_bytes = ((sizeof(Slot) + 2 * a->area_bytes) + 4095) & ~size_t(4095);
    a->nslots = nslots;
    void *raw = mmap(nullptr, a->slot_bytes * nslots, PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (raw == MAP_FAILED) { std::perror("[link] mmap"); delete a; return false; }
    a->base = static_cast<char *>(raw);
    for (int b = 0; b < nslots; ++b) std::memset(a->slot(b), 0, sizeof(Slot));
    arena() = a;
    return true;
}

// Rows [row0, row0 + nrows) of a segment starting at `seg_start` (2^l entries,
// C columns) that intersect the window [win_off, win_off + N). False if none do.
inline bool touched_rows(long long seg_start, int l, long long win_off, long long N,
                         long long &row0, long long &nrows, long long &covered) {
    const long long C = 1LL << (l - (l >> 1));
    const long long lo = std::max(seg_start, win_off);
    const long long hi = std::min(seg_start + (1LL << l), win_off + N);
    if (lo >= hi) return false;
    row0 = (lo - seg_start) / C;
    nrows = (hi - 1 - seg_start) / C - row0 + 1;
    covered = hi - lo;
    return true;
}

// Append one segment's touched rows (data + row commitments + its generators).
inline bool put_seg(char *&cur, char *end, const Commit_return &c, long long seg_start,
                    long long win_off, long long N, long long &covered) {
    long long row0, nrows;
    if (!touched_rows(seg_start, c.l, win_off, N, row0, nrows, covered)) return false;
    const long long C = 1LL << (c.l - (c.l >> 1));
    const size_t need = sizeof(SegHdr) + sizeof(G1) * (C + 1 + nrows) + sizeof(Fr) * nrows * C;
    if (cur + need > end) {
        std::fprintf(stderr, "[link] slot overflow (seg l=%d nrows=%lld)\n", c.l, nrows);
        _exit(1);
    }
    SegHdr h{seg_start, c.l, (int)C, row0, nrows};
    std::memcpy(cur, &h, sizeof h);                          cur += sizeof h;
    std::memcpy(cur, &c.G, sizeof(G1));                      cur += sizeof(G1);
    std::memcpy(cur, c.g, sizeof(G1) * C);                   cur += sizeof(G1) * C;
    std::memcpy(cur, c.comm + row0, sizeof(G1) * nrows);     cur += sizeof(G1) * nrows;
    std::memcpy(cur, c.w + row0 * C, sizeof(Fr) * nrows * C); cur += sizeof(Fr) * nrows * C;
    return true;
}

// eq(z[bit0 .. bit0+m), k) for k < 2^m, LSB-first (bit j of k <-> z[bit0 + j]).
inline std::vector<Fr> eq_table(const std::vector<Fr> &z, int bit0, int m) {
    std::vector<Fr> t(size_t(1) << m);
    t[0] = 1;
    for (int j = 0; j < m; ++j) {
        const size_t h = size_t(1) << j;
        const Fr zj = z[bit0 + j], one_m = Fr(1) - zj;
        for (size_t q = 0; q < h; ++q) { t[q + h] = t[q] * zj; t[q] = t[q] * one_m; }
    }
    return t;
}

// sum over the window entries this segment holds of w[p] * eq(z, p - win_off),
// PROVEN against the segment's row commitments (exit(2) REJECT on a bad opening).
// The verifier side recomputes which rows are needed rather than trusting the header.
inline bool eval_seg(const char *&cur, long long win_off, long long N, int logN,
                     const std::vector<Fr> &z, Fr &acc, long long &covered) {
    SegHdr h;
    std::memcpy(&h, cur, sizeof h); cur += sizeof h;
    const long long C = h.C;
    long long row0, nrows;
    if (C != (1LL << (h.l - (h.l >> 1))) ||
        !touched_rows(h.start, h.l, win_off, N, row0, nrows, covered) ||
        row0 != h.row0 || nrows != h.nrows || N % C != 0) {
        std::fprintf(stderr, "[link] malformed segment start=%lld l=%d C=%lld\n", h.start, h.l, C);
        return false;
    }
    G1 Gb; std::memcpy(&Gb, cur, sizeof(G1)); cur += sizeof(G1);
    const G1 *g = reinterpret_cast<const G1 *>(cur); cur += sizeof(G1) * C;
    const G1 *tk = reinterpret_cast<const G1 *>(cur); cur += sizeof(G1) * nrows;
    const Fr *rows = reinterpret_cast<const Fr *>(cur); cur += sizeof(Fr) * nrows * C;

    int logC = 0; while ((1LL << logC) < C) ++logC;
    const std::vector<Fr> lo = eq_table(z, 0, logC), hi = eq_table(z, logC, logN - logC);
    // seg position p = rho*C + c (rho segment-local); k = p + D, D = start - win_off.
    const long long D = h.start - win_off;
    long long a = D / C, b = D % C;
    if (b < 0) { b += C; --a; }
    for (int part = 0; part < (b ? 2 : 1); ++part) {
        std::vector<Fr> L(C), R(nrows), RT(C);
        for (long long c = 0; c < C; ++c) {
            const bool inA = c < C - b;
            L[c] = (part == 0) == inA ? lo[part == 0 ? c + b : c + b - C] : Fr(0);
        }
        bool any = false;
        for (long long r = 0; r < nrows; ++r) {
            const long long kh = row0 + r + a + part;
            R[r] = (kh >= 0 && kh < N / C) ? hi[kh] : Fr(0);
            any |= !R[r].isZero();
        }
        if (!any) continue;
        for (long long c = 0; c < C; ++c) RT[c] = 0;
        G1 tprime; tprime.clear();
        for (long long r = 0; r < nrows; ++r) {
            if (R[r].isZero()) continue;
            const Fr *row = rows + r * C;
            for (long long c = 0; c < C; ++c) RT[c] += R[r] * row[c];
            tprime = tprime + tk[r] * R[r];
        }
        Fr eval = 0;
        for (long long c = 0; c < C; ++c) eval += L[c] * RT[c];
        std::vector<G1> gc(g, g + C);
        timer blt;
        prove_dot_product(tprime, Gb * eval, L.data(), gc.data(), Gb, RT.data(), eval, (int)C, &blt);
        acc += eval;
    }
    return true;
}

// Publish one side of boundary b; if the other side is already there, run the check.
// side_out: partition b's window (its local input segments); else partition b+1's
// input prefix (cc_input). Returns false only on a failed check.
inline bool publish(int b, bool side_out, const std::vector<const Commit_return *> &segs,
                    const std::vector<long long> &seg_starts, long long win_off, long long N,
                    int self_idx) {
    Arena *A = arena();
    if (!A || b < 0 || b >= A->nslots) return true;
    Slot *s = A->slot(b);
    char *area = side_out ? A->out_area(b) : A->in_area(b), *cur = area, *end = area + A->area_bytes;
    long long covered_total = 0;
    int nseg = 0;
    for (size_t j = 0; j < segs.size(); ++j) {
        long long cov = 0;
        if (put_seg(cur, end, *segs[j], seg_starts[j], win_off, N, cov)) { ++nseg; covered_total += cov; }
    }
    if (covered_total != N) {
        std::fprintf(stderr, "[link] boundary %d %s side covers %lld of %lld window entries\n",
                     b, side_out ? "out" : "in", covered_total, N);
        return false;
    }
    if (side_out) { s->q_off = win_off; s->N = N; s->out_nseg = nseg; }
    __atomic_store_n(side_out ? &s->out_ready : &s->in_ready, 1, __ATOMIC_RELEASE);
    if (__atomic_fetch_add(&s->arrived, 1, __ATOMIC_ACQ_REL) != 1) return true;   // first: done

    // Second to arrive: both commitments now exist, so the challenge may be drawn.
    const auto t0 = std::chrono::steady_clock::now();
    const long long n = s->N;
    int logN = 0; while ((1LL << logN) < n) ++logN;
    if ((1LL << logN) != n) { std::fprintf(stderr, "[link] N=%lld not a power of two\n", n); return false; }
    std::vector<Fr> z(logN);
    for (auto &x : z) x.setByCSPRNG();
    Fr lhs = 0, rhs = 0;
    long long cov_out = 0, cov_in = 0, cov;
    const char *p = A->out_area(b);
    long long prev_end = -1;
    for (int j = 0; j < s->out_nseg; ++j) {
        SegHdr h; std::memcpy(&h, p, sizeof h);
        if (h.start < prev_end) { std::fprintf(stderr, "[link] overlapping segments\n"); return false; }
        prev_end = h.start + (1LL << h.l);
        if (!eval_seg(p, s->q_off, n, logN, z, lhs, cov)) return false;
        cov_out += cov;
    }
    p = A->in_area(b);
    if (!eval_seg(p, 0, n, logN, z, rhs, cov)) return false;
    cov_in = cov;
    const bool ok = cov_out == n && cov_in == n && lhs == rhs;
    s->check_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    s->checked_by = self_idx;
    __atomic_store_n(&s->result, ok ? 1 : -1, __ATOMIC_RELEASE);
    std::fprintf(stderr, "BOUNDARY_LINK boundary=%d checked_by=%d N=%lld out_segs=%d ok=%d check_s=%.3f\n",
                 b, self_idx, n, s->out_nseg, ok ? 1 : 0, s->check_s);
    if (!ok) {
        std::cout << "RESULT: REJECT" << std::endl;
        std::cerr << "[gkr] VERIFICATION REJECTED (stage=boundary_link, boundary " << b << ")\n";
    }
    return ok;
}

// Parent, after every child is reaped: every boundary must have been checked OK.
inline bool all_checked(const char *who) {
    Arena *A = arena();
    if (!A) return true;
    int ok = 0;
    for (int b = 0; b < A->nslots; ++b) ok += A->slot(b)->result == 1;
    std::cout << "BOUNDARY_LINK_SUMMARY mode=" << who << " checked_ok=" << ok
              << " of=" << A->nslots << std::endl;
    return ok == A->nslots;
}

inline void destroy() {
    Arena *A = arena();
    if (!A) return;
    munmap(A->base, A->slot_bytes * A->nslots);
    delete A;
    arena() = nullptr;
}

}  // namespace zkmod_link
