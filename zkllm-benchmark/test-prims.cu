// GPU field primitives vs Python bignum reference (check_prims.py). All inputs are valid field elements (< p).
//   prims_fr.txt: a b | add sub rmul fmul mont unmont ex2 ex3 (inlined) | add sub rmul fmul mont unmont ex2 ex3 (__noinline__)
//   prims_fp.txt: a b | add sub mul mul(mul(a,b),a) mul(a,a)
// ex2/ex3 replay the extrapolation pass of blendy.cu (val = v1; val += (v1 - v0) once / twice).
#include "fr-tensor.cuh"
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>
using namespace std;

static __device__ __forceinline__ Fr_t fadd(Fr_t a, Fr_t b) { return blstrs__scalar__Scalar_add(a, b); }
static __device__ __forceinline__ Fr_t fsub(Fr_t a, Fr_t b) { return blstrs__scalar__Scalar_sub(a, b); }
static __device__ __forceinline__ Fr_t rmul(Fr_t a, Fr_t b) { return blstrs__scalar__Scalar_mul(a, b); }
static __device__ __forceinline__ Fr_t fmul(Fr_t a, Fr_t b) { return blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(a, b)); }

static __device__ __noinline__ Fr_t nadd(Fr_t a, Fr_t b) { return blstrs__scalar__Scalar_add(a, b); }
static __device__ __noinline__ Fr_t nsub(Fr_t a, Fr_t b) { return blstrs__scalar__Scalar_sub(a, b); }
static __device__ __noinline__ Fr_t nmul(Fr_t a, Fr_t b) { return blstrs__scalar__Scalar_mul(a, b); }
static __device__ __noinline__ Fr_t nmont(Fr_t a) { return blstrs__scalar__Scalar_mont(a); }

__global__ void k_fr(const Fr_t* a, const Fr_t* b, Fr_t* out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Fr_t x = a[i], y = b[i];
    Fr_t* o = out + (long)i * 16;
    o[0] = fadd(x, y);
    o[1] = fsub(x, y);
    o[2] = rmul(x, y);
    o[3] = fmul(x, y);
    o[4] = blstrs__scalar__Scalar_mont(x);
    o[5] = blstrs__scalar__Scalar_unmont(x);
    Fr_t diff = fsub(y, x), val = fadd(y, diff);
    o[6] = val;
    val = fadd(val, diff);
    o[7] = val;
    o[8] = nadd(x, y);
    o[9] = nsub(x, y);
    o[10] = nmul(x, y);
    o[11] = nmont(nmul(x, y));
    o[12] = nmont(x);
    o[13] = blstrs__scalar__Scalar_unmont(x);
    Fr_t d2 = nsub(y, x), v2 = nadd(y, d2);
    o[14] = v2;
    o[15] = nadd(v2, d2);
}

// One shared, non-inlined copy of the 12-limb multiply: inlining five of them into one kernel made the
// LTO build use >32 GB of host RAM.
static __device__ __noinline__ blstrs__fp__Fp fpmul(blstrs__fp__Fp a, blstrs__fp__Fp b) { return blstrs__fp__Fp_mul(a, b); }

__global__ void k_fp(const blstrs__fp__Fp* a, const blstrs__fp__Fp* b, blstrs__fp__Fp* out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    blstrs__fp__Fp x = a[i], y = b[i];
    blstrs__fp__Fp* o = out + (long)i * 5;
    o[0] = blstrs__fp__Fp_add(x, y);
    o[1] = blstrs__fp__Fp_sub(x, y);
    o[2] = fpmul(x, y);
    o[3] = fpmul(o[2], x);
    o[4] = fpmul(x, x);
}

// little-endian 32-bit limbs from a big-endian hex string
template <class T, int L>
static T from_hex(const char* hex)
{
    T t;
    memset(&t, 0, sizeof t);
    const int len = strlen(hex);
    for (int l = 0; l < L; ++l) {
        int end = len - 8 * l, start = end - 8;
        if (end <= 0) break;
        if (start < 0) start = 0;
        t.val[l] = (uint32_t)strtoul(string(hex + start, end - start).c_str(), nullptr, 16);
    }
    return t;
}

template <class T, int L>
static void hx(FILE* f, const T& a)
{
    for (int i = L - 1; i >= 0; --i) fprintf(f, "%08x", a.val[i]);
}

// random value strictly below p: random limbs, top limb below p's top limb
template <class T, int L>
static vector<T> random_below(mt19937& mt, uint32_t top_limb, int n)
{
    vector<T> v(n);
    for (auto& e : v) {
        for (int l = 0; l < L - 1; ++l) e.val[l] = mt();
        e.val[L - 1] = mt() % top_limb;
    }
    return v;
}

template <class T, int L, class K>
static void run_field(const char* name, const char* hex_p, const char* file, int outs, K kernel)
{
    T p = from_hex<T, L>(hex_p), pm1 = p, zero, one, two;
    memset(&zero, 0, sizeof zero);
    one = two = zero;
    one.val[0] = 1;
    two.val[0] = 2;
    pm1.val[0] -= 1;
    T pm2 = pm1;
    pm2.val[0] -= 1;
    vector<T> edge = {zero, one, two, pm1, pm2};
    for (int k = 1; k < L; ++k) {   // carry-heavy patterns: low k limbs all ones (< p because p's limbs above are larger)
        T e = zero;
        for (int l = 0; l < k; ++l) e.val[l] = 0xffffffffu;
        edge.push_back(e);
        T f = zero;                  // 2^(32k) - 1 shifted up by one limb
        for (int l = 1; l <= k && l < L - 1; ++l) f.val[l] = 0xffffffffu;
        edge.push_back(f);
    }
    T hi = zero;                     // large, below p: all ones up to the top limb, top limb = p_top - 1
    for (int l = 0; l < L - 1; ++l) hi.val[l] = 0xffffffffu;
    hi.val[L - 1] = p.val[L - 1] - 1;
    edge.push_back(hi);
    vector<T> A, B;
    for (auto& e : edge) for (auto& f : edge) { A.push_back(e); B.push_back(f); }
    mt19937 mt(12345);
    auto ra = random_below<T, L>(mt, p.val[L - 1], 200000), rb = random_below<T, L>(mt, p.val[L - 1], 200000);
    A.insert(A.end(), ra.begin(), ra.end());
    B.insert(B.end(), rb.begin(), rb.end());
    const int n = A.size();
    T *da, *db, *dout;
    cudaMalloc(&da, n * sizeof(T));
    cudaMalloc(&db, n * sizeof(T));
    cudaMalloc(&dout, (long)n * outs * sizeof(T));
    cudaMemcpy(da, A.data(), n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(db, B.data(), n * sizeof(T), cudaMemcpyHostToDevice);
    kernel<<<(n + 255) / 256, 256>>>(da, db, dout, n);
    cudaDeviceSynchronize();
    vector<T> out((long)n * outs);
    cudaMemcpy(out.data(), dout, out.size() * sizeof(T), cudaMemcpyDeviceToHost);
    FILE* f = fopen(file, "w");
    for (int i = 0; i < n; ++i) {
        hx<T, L>(f, A[i]); fprintf(f, " "); hx<T, L>(f, B[i]);
        for (int k = 0; k < outs; ++k) { fprintf(f, " "); hx<T, L>(f, out[(long)i * outs + k]); }
        fprintf(f, "\n");
    }
    fclose(f);
    printf("%s: wrote %s with %d rows (%zu edge values)\n", name, file, n, edge.size());
}

int main()
{
    run_field<Fr_t, 8>("Fr", "73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001", "prims_fr.txt", 16, k_fr);
    run_field<blstrs__fp__Fp, 12>("Fp",
        "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab", "prims_fp.txt", 5, k_fp);
    return 0;
}
