// Standalone check: staged (blendy) multi_hadamard_sumchecks vs the legacy path on random tensors.
// Usage: ./test-blendy [max_log_n]. Env knobs: ZKLLM_BLENDY_L, ZKLLM_SC_TILE_BITS.
#include "fr-tensor.cuh"
#include "polynomial.cuh"
#include "proof.cuh"
#include <iostream>
#include <string>

using namespace std;

Fr_t multi_hadamard_sumchecks_legacy(const Fr_t& claim, const vector<FrTensor>& Xs, const vector<Fr_t>& u, const vector<Fr_t>& v, vector<Polynomial>& proof);

static bool same(const Fr_t& a, const Fr_t& b) { return !(a != b); }

int main(int argc, char** argv)
{
    const uint max_n = argc > 1 ? atoi(argv[1]) : 22;
    int bad = 0;
    for (uint nf = 1; nf <= 3; ++nf)
    for (uint n = 1; n <= max_n; ++n) {
        const uint N = 1u << n;
        vector<FrTensor> Xs;
        for (uint i = 0; i < nf; ++i) Xs.push_back(FrTensor::random(N));
        auto u = random_vec(n), v = random_vec(n);
        FrTensor prod = Xs[0];
        for (uint i = 1; i < nf; ++i) prod *= Xs[i];
        Fr_t claim = prod(u);

        vector<Polynomial> pl, ps;
        string el, es;
        Fr_t rl{}, rs{};
        try { rl = multi_hadamard_sumchecks_legacy(claim, Xs, u, v, pl); } catch (exception& e) { el = e.what(); }
        try { rs = multi_hadamard_sumchecks(claim, Xs, u, v, ps); } catch (exception& e) { es = e.what(); }

        int first = -1;
        const size_t m = min(pl.size(), ps.size());
        const Fr_t pts[4] = {{0,0,0,0,0,0,0,0}, {1,0,0,0,0,0,0,0}, {2,0,0,0,0,0,0,0}, {7,0,0,0,0,0,0,0}};
        for (size_t r = 0; r < m && first < 0; ++r)
            for (auto& x : pts) if (!same(pl[r](x), ps[r](x))) { first = r; break; }
        bool ok = el.empty() && es.empty() && pl.size() == ps.size() && first < 0 && same(rl, rs);
        if (!ok) ++bad;
        cout << (ok ? "ok   " : "FAIL ") << "nf=" << nf << " n=" << n
             << " legacy_err=[" << el << "] staged_err=[" << es << "]"
             << " rounds(l/s)=" << pl.size() << "/" << ps.size() << " first_diff_round=" << first << endl;
    }
    cout << "TEST_BLENDY bad=" << bad << endl;
    return bad != 0;
}
