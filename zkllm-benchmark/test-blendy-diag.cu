// n=2, one 2-round stage (L=2), nf factors, no eq weight. Prints every intermediate of the staged
// sumcheck next to a host reference and flags the ones that differ:
//   1. extension cells ext[j][cell]   (expected: bilinear extension of the 4 input values)
//   2. ptable after begin_stage       (expected: prod_j ext_j, as raw Montgomery product: true * R^-nf)
//   3. ptable after bind(v)           (expected: sum_k Lagrange_k(v) * P[t + B*k])
//   4. round-0 / round-1 polynomials at X = 0..3
#include "fr-tensor.cuh"
#include "polynomial.cuh"
#include "proof.cuh"
#include <iostream>
#include <cstdio>
#include <string>
#define private public
#include "blendy.cuh"
#undef private
using namespace std;

__global__ void mont_k(Fr_t* x, int k)
{
    Fr_t a = x[0];
    for (int i = 0; i < k; ++i) a = blstrs__scalar__Scalar_mont(a);
    x[0] = a;
}

// raw product carrying k stray 1/R factors -> true value
static Fr_t to_true(Fr_t raw, int k)
{
    Fr_t* d;
    cudaMalloc(&d, sizeof(Fr_t));
    cudaMemcpy(d, &raw, sizeof(Fr_t), cudaMemcpyHostToDevice);
    mont_k<<<1, 1>>>(d, k);
    cudaMemcpy(&raw, d, sizeof(Fr_t), cudaMemcpyDeviceToHost);
    cudaFree(d);
    return raw;
}

// verbatim copy of the B == 4 Lagrange code in blendy.cu ptable_fold_kernel, with its intermediates dumped
static __device__ __forceinline__ Fr_t fadd(Fr_t a, Fr_t b) { return blstrs__scalar__Scalar_add(a, b); }
static __device__ __forceinline__ Fr_t fsub(Fr_t a, Fr_t b) { return blstrs__scalar__Scalar_sub(a, b); }
static __device__ __forceinline__ Fr_t fmul(Fr_t a, Fr_t b) { return blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(a, b)); }
#define ONE_FR Fr_t{1, 0, 0, 0, 0, 0, 0, 0}
#define ZERO_FR Fr_t{0, 0, 0, 0, 0, 0, 0, 0}
__global__ void lag4_dbg(Fr_t v, Fr_t inv2, Fr_t inv6, Fr_t* out)
{
    const Fr_t one = ONE_FR, two = Fr_t{2, 0, 0, 0, 0, 0, 0, 0}, three = Fr_t{3, 0, 0, 0, 0, 0, 0, 0};
    const Fr_t vm1 = fsub(v, one), vm2 = fsub(v, two), vm3 = fsub(v, three);
    out[0] = vm1; out[1] = vm2; out[2] = vm3;
    out[3] = fmul(vm1, vm2);
    out[4] = fmul(fmul(vm1, vm2), vm3);
    out[5] = fmul(fmul(fmul(vm1, vm2), vm3), inv6);
    Fr_t L[4];
    L[0] = fsub(ZERO_FR, fmul(fmul(fmul(vm1, vm2), vm3), inv6));
    L[1] = fmul(fmul(fmul(v, vm2), vm3), inv2);
    L[2] = fsub(ZERO_FR, fmul(fmul(fmul(v, vm1), vm3), inv2));
    L[3] = fmul(fmul(fmul(v, vm1), vm2), inv6);
    for (int k = 0; k < 4; ++k) out[6 + k] = L[k];
}

static __device__ __noinline__ Fr_t nfmul(Fr_t a, Fr_t b) { return blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(a, b)); }
// The L3 chain ((v*vm1)*vm2)*inv6 stage by stage, three ways:
//   [0..2] on its own, inlined   [3..5] same with __noinline__ multiplies   [6..8] fused with L2 as in the real kernel
__global__ void l3_stages(Fr_t v, Fr_t inv2, Fr_t inv6, Fr_t* out)
{
    const Fr_t one = ONE_FR, two = Fr_t{2, 0, 0, 0, 0, 0, 0, 0}, three = Fr_t{3, 0, 0, 0, 0, 0, 0, 0};
    const Fr_t vm1 = fsub(v, one), vm2 = fsub(v, two), vm3 = fsub(v, three);
    Fr_t a1 = fmul(v, vm1);
    Fr_t a2 = fmul(a1, vm2);
    out[0] = a1; out[1] = a2; out[2] = fmul(a2, inv6);
    Fr_t b1 = nfmul(v, vm1);
    Fr_t b2 = nfmul(b1, vm2);
    out[3] = b1; out[4] = b2; out[5] = nfmul(b2, inv6);
    Fr_t L2 = fsub(ZERO_FR, fmul(fmul(fmul(v, vm1), vm3), inv2));
    Fr_t c1 = fmul(v, vm1);
    Fr_t c2 = fmul(c1, vm2);
    out[6] = c1; out[7] = c2; out[8] = fmul(c2, inv6);
    out[9] = L2;
}

static bool same(const Fr_t& a, const Fr_t& b) { return !(a != b); }
static string hx(const Fr_t& a)
{
    char b[80];
    snprintf(b, sizeof b, "%08x%08x%08x%08x%08x%08x%08x%08x", a.val[7], a.val[6], a.val[5], a.val[4], a.val[3], a.val[2], a.val[1], a.val[0]);
    return b;
}
static Fr_t F(uint x) { return Fr_t{x, 0, 0, 0, 0, 0, 0, 0}; }

static int bad = 0;
static void cmp(const string& what, const Fr_t& exp, const Fr_t& got)
{
    bool ok = same(exp, got);
    if (!ok) ++bad;
    cout << "    " << what << (ok ? "  ok  " : "  MISMATCH  ") << (ok ? hx(got).substr(48) : "exp=" + hx(exp) + "\n" + string(40 + what.size(), ' ') + "got=" + hx(got)) << endl;
}

int main()
{
    const uint n = 2;
    for (int nf = 1; nf <= 3; ++nf) {
        const int B = nf + 1, cells = B * B;
        cout << "===== nf=" << nf << " (B=" << B << ", " << cells << " cells)" << endl;
        vector<FrTensor> Xs;
        for (int i = 0; i < nf; ++i) Xs.push_back(FrTensor::random(1u << n));
        auto u = random_vec(n), v = random_vec(n);
        vector<pair<const Fr_t*, uint>> tensors;
        blendy::Term term{{}, {}, false, 0};
        for (int i = 0; i < nf; ++i) { tensors.push_back({Xs[i].gpu_data, 1u << n}); term.factors.push_back(i); }
        blendy::Sumcheck sc(tensors, {term}, u, n, blendy::Order::MSB_FIRST, -1, 2, 16);

        auto r0 = sc.round()[0];   // runs begin_stage

        // host copies of the inputs: x_j[d0 + 2*d1]
        vector<vector<Fr_t>> x(nf, vector<Fr_t>(4));
        for (int j = 0; j < nf; ++j) for (int i = 0; i < 4; ++i) x[j][i] = Xs[j](i);
        auto ext_exp = [&](int j, int d0, int d1) {
            Fr_t t0 = F(d0), t1 = F(d1), one = F(1);
            Fr_t l0[2] = {one - t0, t0}, l1[2] = {one - t1, t1};
            Fr_t s = F(0);
            for (int a = 0; a < 2; ++a) for (int b = 0; b < 2; ++b) s = s + l0[a] * l1[b] * x[j][a + 2 * b];
            return s;
        };

        // 1. extension cells
        cout << "  [1] extension cells (cell = d0 + B*d1)" << endl;
        vector<Fr_t> P_true(cells);
        for (int j = 0; j < nf; ++j) {
            vector<Fr_t> h(cells);
            cudaMemcpy(h.data(), sc.ext[j], cells * sizeof(Fr_t), cudaMemcpyDeviceToHost);
            for (int c = 0; c < cells; ++c)
                cmp("ext[" + to_string(j) + "] d0=" + to_string(c % B) + " d1=" + to_string(c / B), ext_exp(j, c % B, c / B), h[c]);
        }
        // 2. ptable
        cout << "  [2] ptable after begin_stage (raw, scale R^-" << nf << ")" << endl;
        vector<Fr_t> hp(cells);
        cudaMemcpy(hp.data(), sc.ptable, cells * sizeof(Fr_t), cudaMemcpyDeviceToHost);
        for (int c = 0; c < cells; ++c) {
            Fr_t e = F(1);
            for (int j = 0; j < nf; ++j) e = e * ext_exp(j, c % B, c / B);
            P_true[c] = e;
            cmp("P d0=" + to_string(c % B) + " d1=" + to_string(c / B), e, to_true(hp[c], nf));
        }
        // round 0 polynomial
        cout << "  [4a] round-0 polynomial" << endl;
        Polynomial Q0(r0);
        for (uint X = 0; X < (uint)B; ++X) {
            Fr_t e = F(0);
            for (uint b0 = 0; b0 < 2; ++b0) {
                Fr_t t = F(1);
                for (int j = 0; j < nf; ++j) t = t * ((F(1) - F(X)) * x[j][b0] + F(X) * x[j][2 + b0]);
                e = e + t;
            }
            cmp("R0(" + to_string(X) + ")", e, Q0(F(X)));
        }

        sc.bind(v[1]);   // folds the top digit (d1) at v[1]

        // 3. folded ptable
        cout << "  [3] ptable after bind(v) (index = d0)" << endl;
        vector<Fr_t> L(B);
        for (int k = 0; k < B; ++k) {
            Fr_t num = F(1), den = F(1);
            for (int m = 0; m < B; ++m) if (m != k) { num = num * (v[1] - F(m)); den = den * (F(k) - F(m)); }
            L[k] = num / den;
        }
        vector<Fr_t> hf(B);
        cudaMemcpy(hf.data(), sc.ptable, B * sizeof(Fr_t), cudaMemcpyDeviceToHost);
        for (int t = 0; t < B; ++t) {
            Fr_t e = F(0);
            for (int k = 0; k < B; ++k) e = e + L[k] * P_true[t + B * k];
            cmp("fold[d0=" + to_string(t) + "]", e, to_true(hf[t], nf));
        }

        auto r1 = sc.round()[0];
        cout << "  [4b] round-1 polynomial" << endl;
        Polynomial Q1(r1);
        for (uint X = 0; X < (uint)B; ++X) {
            Fr_t e = F(1);
            for (int j = 0; j < nf; ++j) {
                Fr_t a0 = (F(1) - v[1]) * x[j][0] + v[1] * x[j][2];
                Fr_t a1 = (F(1) - v[1]) * x[j][1] + v[1] * x[j][3];
                e = e * ((F(1) - F(X)) * a0 + F(X) * a1);
            }
            cmp("R1(" + to_string(X) + ")", e, Q1(F(X)));
        }
    }
    // B = 4 Lagrange weights: device intermediates vs host, for several random v
    cout << "===== lag4 intermediates (device copy of the B=4 fold weights)" << endl;
    for (int rep = 0; rep < 3; ++rep) {
        Fr_t v = random_vec(1)[0];
        Fr_t inv2 = F(1) / F(2), inv6 = F(1) / F(6);
        Fr_t* d;
        cudaMalloc(&d, 10 * sizeof(Fr_t));
        lag4_dbg<<<1, 1>>>(v, inv2, inv6, d);
        Fr_t h[10];
        cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
        cudaFree(d);
        Fr_t vm1 = v - F(1), vm2 = v - F(2), vm3 = v - F(3);
        cout << "  v=" << hx(v).substr(32) << endl;
        cmp("v-1", vm1, h[0]);
        cmp("v-2", vm2, h[1]);
        cmp("v-3", vm3, h[2]);
        cmp("(v-1)(v-2)", vm1 * vm2, h[3]);
        cmp("(v-1)(v-2)(v-3)", vm1 * vm2 * vm3, h[4]);
        cmp("(v-1)(v-2)(v-3)/6", vm1 * vm2 * vm3 * inv6, h[5]);
        cmp("L0", F(0) - vm1 * vm2 * vm3 * inv6, h[6]);
        cmp("L1", v * vm2 * vm3 * inv2, h[7]);
        cmp("L2", F(0) - v * vm1 * vm3 * inv2, h[8]);
        cmp("L3", v * vm1 * vm2 * inv6, h[9]);
    }
    cout << "===== L3 chain, stage by stage" << endl;
    for (int rep = 0; rep < 3; ++rep) {
        Fr_t v = random_vec(1)[0];
        Fr_t inv2 = F(1) / F(2), inv6 = F(1) / F(6);
        Fr_t* d;
        cudaMalloc(&d, 10 * sizeof(Fr_t));
        l3_stages<<<1, 1>>>(v, inv2, inv6, d);
        Fr_t h[10];
        cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
        cudaFree(d);
        Fr_t vm1 = v - F(1), vm2 = v - F(2);
        Fr_t e1 = v * vm1, e2 = e1 * vm2, e3 = e2 * inv6;
        cout << "  v=" << hx(v) << "\n  vm2=" << hx(vm2) << "\n  inv6=" << hx(inv6) << endl;
        const char* nm[3] = {"inlined alone  ", "noinline mul   ", "fused with L2  "};
        for (int s = 0; s < 3; ++s) {
            cmp(string(nm[s]) + " v*(v-1)", e1, h[3 * s]);
            cmp(string(nm[s]) + " *(v-2)", e2, h[3 * s + 1]);
            cmp(string(nm[s]) + " *inv6", e3, h[3 * s + 2]);
        }
    }
    cout << "DIAG mismatches=" << bad << endl;
    return 0;
}
