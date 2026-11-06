#include "blendy.cuh"

#include <cstdlib>
#include <stdexcept>
#include <algorithm>

namespace blendy {

bool enabled() {
    const char* e = getenv("ZKLLM_BLENDY");
    return !(e && atoi(e) == 0);
}

int default_stage_len() {
    const char* e = getenv("ZKLLM_BLENDY_L");
    return e ? atoi(e) : 2;
}

int default_tile_bits() {
    const char* e = getenv("ZKLLM_SC_TILE_BITS");
    return e ? atoi(e) : 16;
}

// Fr_t is in normal form and Scalar_mul is a Montgomery product

#define ONE_FR Fr_t{1, 0, 0, 0, 0, 0, 0, 0}
#define ZERO_FR Fr_t{0, 0, 0, 0, 0, 0, 0, 0}

static __device__ __forceinline__ Fr_t fadd(Fr_t a, Fr_t b) {
    return blstrs__scalar__Scalar_add(a, b);
}
static __device__ __forceinline__ Fr_t fsub(Fr_t a, Fr_t b) {
    return blstrs__scalar__Scalar_sub(a, b);
}
static __device__ __forceinline__ Fr_t rmul(Fr_t a, Fr_t b) {
    return blstrs__scalar__Scalar_mul(a, b);
}
static __device__ __forceinline__ Fr_t fmul(Fr_t a, Fr_t b) {
    return blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(a, b));
}
static __device__ __forceinline__ Fr_t eq1(Fr_t u, int bit) {
    return bit ? u : fsub(ONE_FR, u);
}

struct ExtPtrs {
    const Fr_t* p[MAX_TENSORS];
};

static __global__ void eq_table_kernel(const Fr_t* w, int nbits, Fr_t* out, long count, int to_mont) {
    long b = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= count)
        return;
    Fr_t acc = ONE_FR;
    for (int i = 0; i < nbits; ++i)
        acc = fmul(acc, eq1(w[i], (b >> i) & 1));
    out[b] = to_mont ? blstrs__scalar__Scalar_mont(acc) : acc;
}

// Expand one tile of one tensor into the binary cells of its extension table
static __global__ void expand_kernel(const Fr_t* __restrict__ src, unsigned long src_size, Fr_t* ext, int l,
                                     int B, int fv, int tile_lo, long ti, int msb) {
    const long tb3 = 1L << tile_lo;
    const long id = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (tb3 << l))
        return;
    const int b2 = id >> tile_lo;
    const long b3lo = id & (tb3 - 1);
    const unsigned long b3 = ((unsigned long)ti << tile_lo) | b3lo;
    const unsigned long idx = msb ? (((unsigned long)b2) << fv) | b3 : (b3 << l) | (unsigned long)b2;
    int cell = 0, p = 1;
    for (int i = 0; i < l; ++i) {
        int bit = msb ? (b2 >> i) & 1 : (b2 >> (l - 1 - i)) & 1;
        cell += bit * p;
        p *= B;
    }
    ext[(long)cell * tb3 + b3lo] = idx < src_size ? src[idx] : ZERO_FR;
}

// Extrapolation pass for digit position `var`
static __global__ void extrapolate_kernel(Fr_t* ext, int var, int B, long cells, int tile_lo) {
    const long tb3 = 1L << tile_lo;
    const long id = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= cells * tb3)
        return;
    const long cell = id >> tile_lo, b3lo = id & (tb3 - 1);
    long pw = 1;
    for (int i = 0; i < var; ++i)
        pw *= B;
    const int d = (cell / pw) % B;
    if (d < 2)
        return;
    Fr_t v0 = ext[(cell - d * pw) * tb3 + b3lo];
    Fr_t v1 = ext[(cell - (d - 1) * pw) * tb3 + b3lo];
    Fr_t diff = fsub(v1, v0), val = v1;
    for (int k = 2; k <= d; ++k)
        val = fadd(val, diff);
    ext[id] = val;
}

// First level of the two-level segmented ptable reduction
static __global__ void ptable_partial_kernel(Desc desc, ExtPtrs ext, const Fr_t* __restrict__ eqlow,
                                             Fr_t* partials, int cells, long tb3, int nb3) {
    const int cell = blockIdx.x, blk = blockIdx.y, g = blockIdx.z, tid = threadIdx.x;
    const TermDesc& T = desc.t[g];
    const long stride = (long)nb3 * blockDim.x;
    Fr_t acc = ZERO_FR;
    for (long b3 = (long)blk * blockDim.x + tid; b3 < tb3; b3 += stride) {
        const long o = (long)cell * tb3 + b3;
        Fr_t P = fadd(ext.p[T.f[0]][o], T.shift[0]);
        for (int j = 1; j < T.nf; ++j)
            P = rmul(P, fadd(ext.p[T.f[j]][o], T.shift[j]));
        if (T.eq)
            P = rmul(P, eqlow[b3]);
        acc = fadd(acc, P);
    }
    extern __shared__ Fr_t smem[];
    smem[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            smem[tid] = fadd(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    if (tid == 0)
        partials[((long)g * cells + cell) * nb3 + blk] = smem[0];
}

// Second level
static __global__ void ptable_finalize_kernel(Desc desc, const Fr_t* __restrict__ partials, Fr_t* ptable,
                                              int cells, int nb3, const Fr_t* __restrict__ ufut, int tile_lo,
                                              int fv, long ti) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= desc.n_terms * cells)
        return;
    const int g = idx / cells;
    Fr_t s = ZERO_FR;
    for (int i = 0; i < nb3; ++i)
        s = fadd(s, partials[(long)idx * nb3 + i]);
    Fr_t w = ONE_FR;
    if (desc.t[g].eq)
        for (int i = tile_lo; i < fv; ++i)
            w = fmul(w, eq1(ufut[i], (ti >> (i - tile_lo)) & 1));
    ptable[idx] = fadd(ptable[idx], rmul(s, w));
}

// One sumcheck round (blendysc.hpp blendy_round_kernel), for all terms in a single block
static __global__ void round_kernel(Desc desc, const Fr_t* __restrict__ ptable, long stride, int rem, int B,
                                    const Fr_t* __restrict__ ustage, Scalars coefs, Fr_t inv2, Fr_t inv3,
                                    Fr_t inv6, Fr_t* out) {
    const int tid = threadIdx.x;
    const int ntail = 1 << (rem - 1);
    int sub = 1;
    for (int i = 0; i < rem - 1; ++i)
        sub *= B;

    Fr_t acc[MAX_TERMS][4];
    for (int g = 0; g < MAX_TERMS; ++g)
        for (int t = 0; t < 4; ++t)
            acc[g][t] = ZERO_FR;

    for (int tail = tid; tail < ntail; tail += blockDim.x) {
        int tb = 0, p = 1;
        Fr_t w = ONE_FR;
        for (int i = 0; i < rem - 1; ++i) {
            int bit = (tail >> i) & 1;
            tb += bit * p;
            p *= B;
            w = fmul(w, eq1(ustage[i], bit));
        }
        for (int g = 0; g < desc.n_terms; ++g) {
            const Fr_t wg = desc.t[g].eq ? w : ONE_FR;
            for (int t = 0; t <= desc.t[g].nf; ++t)
                acc[g][t] = fadd(acc[g][t], rmul(ptable[g * stride + (long)t * sub + tb], wg));
        }
    }

    __shared__ Fr_t smem[64];
    __shared__ Fr_t res[MAX_TERMS][4];
    for (int g = 0; g < desc.n_terms; ++g)
        for (int t = 0; t <= desc.t[g].nf; ++t) {
            smem[tid] = acc[g][t];
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s)
                    smem[tid] = fadd(smem[tid], smem[tid + s]);
                __syncthreads();
            }
            if (tid == 0)
                res[g][t] = smem[0];
            __syncthreads();
        }
    if (tid != 0)
        return;

    Fr_t e[MAX_GROUPS][4];
    for (int k = 0; k < MAX_GROUPS; ++k)
        for (int t = 0; t < 4; ++t)
            e[k][t] = ZERO_FR;
    for (int g = 0; g < desc.n_terms; ++g) {
        const TermDesc& T = desc.t[g];
        for (int t = 0; t <= T.nf; ++t) {
            Fr_t x = res[g][t];
            for (int s = 0; s < T.scale; ++s)
                x = blstrs__scalar__Scalar_mont(x);
            e[T.group][t] = fadd(e[T.group][t], fmul(x, coefs.v[g]));
        }
    }
    for (int k = 0; k < desc.n_groups; ++k) {
        const int d = desc.group_deg[k];
        Fr_t c[4];
        const Fr_t* v = e[k];
        c[0] = v[0];
        if (d == 1) {
            c[1] = fsub(v[1], v[0]);
        } else if (d == 2) {
            Fr_t d2 = fadd(fsub(v[2], fadd(v[1], v[1])), v[0]);  // e2 - 2e1 + e0
            c[2] = fmul(d2, inv2);
            c[1] = fsub(fsub(v[1], v[0]), c[2]);
        } else if (d == 3) {
            Fr_t d1 = fsub(v[1], v[0]);
            Fr_t d2 = fadd(fsub(v[2], fadd(v[1], v[1])), v[0]);
            Fr_t t3 = fadd(fadd(v[2], v[2]), v[2]), t1 = fadd(fadd(v[1], v[1]), v[1]);
            Fr_t d3 = fsub(fadd(fsub(v[3], t3), t1), v[0]);  // e3 - 3e2 + 3e1 - e0
            c[3] = fmul(d3, inv6);
            c[2] = fsub(fmul(d2, inv2), fmul(d3, inv2));
            c[1] = fadd(fsub(d1, fmul(d2, inv2)), fmul(d3, inv3));
        }
        for (int t = 0; t <= d; ++t) {
            Fr_t x = c[t];
            for (int a = 0; a < desc.group_adjust[k]; ++a)
                x = blstrs__scalar__Scalar_mont(x);
            for (int a = 0; a < -desc.group_adjust[k]; ++a)
                x = rmul(x, ONE_FR);
            out[k * 4 + t] = x;
        }
    }
}

static __global__ void ptable_fold_kernel(int n_terms, Fr_t* ptable, long stride, int B, int ns, Fr_t v,
                                          Fr_t inv2, Fr_t inv6) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_terms * ns)
        return;
    const int g = idx / ns, tail = idx % ns;
    Fr_t L[4];
    const Fr_t one = ONE_FR, two = Fr_t{2, 0, 0, 0, 0, 0, 0, 0}, three = Fr_t{3, 0, 0, 0, 0, 0, 0, 0};
    const Fr_t vm1 = fsub(v, one), vm2 = fsub(v, two), vm3 = fsub(v, three);
    if (B == 2) {
        L[0] = fsub(one, v);
        L[1] = v;
    } else if (B == 3) {
        L[0] = fmul(fmul(vm1, vm2), inv2);
        L[1] = fmul(v, fsub(two, v));
        L[2] = fmul(fmul(v, vm1), inv2);
    } else {
        L[0] = fsub(ZERO_FR, fmul(fmul(fmul(vm1, vm2), vm3), inv6));
        L[1] = fmul(fmul(fmul(v, vm2), vm3), inv2);
        L[2] = fsub(ZERO_FR, fmul(fmul(fmul(v, vm1), vm3), inv2));
        L[3] = fmul(fmul(fmul(v, vm1), vm2), inv6);
    }
    Fr_t* P = ptable + g * stride;
    Fr_t acc = ZERO_FR;
    for (int k = 0; k < B; ++k)
        acc = fadd(acc, rmul(blstrs__scalar__Scalar_mont(L[k]), P[(long)k * ns + tail]));
    P[tail] = acc;
}

// Stage-boundary fold of a big array by all l stage challenges at once
static __global__ void stage_fold_kernel(const Fr_t* __restrict__ src, unsigned long src_size,
                                         const Fr_t* __restrict__ eqv, Fr_t* dst, int l, int fv, int msb,
                                         unsigned long out_elems) {
    const unsigned long j = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= out_elems)
        return;
    Fr_t acc = ZERO_FR;
    for (int b = 0; b < (1 << l); ++b) {
        unsigned long idx = msb ? (((unsigned long)b) << fv) | j : (j << l) | (unsigned long)b;
        if (idx < src_size)
            acc = fadd(acc, rmul(eqv[b], src[idx]));
    }
    dst[j] = acc;
}

static __global__ void inverse_consts_kernel(Fr_t* out) {
    const Fr_t xs[3] = {Fr_t{2, 0, 0, 0, 0, 0, 0, 0}, Fr_t{3, 0, 0, 0, 0, 0, 0, 0},
                        Fr_t{6, 0, 0, 0, 0, 0, 0, 0}};
    for (int i = 0; i < 3; ++i)
        out[i] =
            blstrs__scalar__Scalar_unmont(blstrs__scalar__Scalar_inverse(blstrs__scalar__Scalar_mont(xs[i])));
}

static void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("blendy: ") + what + ": " + cudaGetErrorString(e));
}

static const Fr_t* inverse_consts() {
    static Fr_t c[3];
    static bool init = false;
    if (!init) {
        Fr_t* d;
        cudaMalloc(&d, 3 * sizeof(Fr_t));
        inverse_consts_kernel<<<1, 1>>>(d);
        check(cudaMemcpy(c, d, 3 * sizeof(Fr_t), cudaMemcpyDeviceToHost), "inverse constants");
        cudaFree(d);
        init = true;
    }
    return c;
}

static long ipow(long b, int e) {
    long r = 1;
    while (e-- > 0)
        r *= b;
    return r;
}
static inline unsigned blocks(unsigned long n, unsigned t) {
    return (unsigned)((n + t - 1) / t);
}

constexpr int PTABLE_TARGET_BLOCKS = 2048;
constexpr int NT = 256;

Sumcheck::Sumcheck(const std::vector<std::pair<const Fr_t*, uint>>& tensors, const std::vector<Term>& terms,
                   const std::vector<Fr_t>& u_, uint num_vars, Order order_, int rounds, int l_stage,
                   int tile_bits)
    : n(num_vars), nr(rounds < 0 ? num_vars : rounds), order(order_), u(u_) {
    if (nr > n)
        throw std::runtime_error("blendy: more rounds than variables");
    if (tensors.empty() || (int)tensors.size() > MAX_TENSORS)
        throw std::runtime_error("blendy: bad tensor count");
    if (terms.empty() || (int)terms.size() > MAX_TERMS)
        throw std::runtime_error("blendy: bad term count");
    n_tensors = tensors.size();
    for (auto& t : tensors) {
        if ((unsigned long)t.second > (1UL << n))
            throw std::runtime_error("blendy: tensor longer than 2^num_vars");
        src.push_back(t.first);
        src_size.push_back(t.second);
    }

    desc = Desc{};
    desc.n_terms = terms.size();
    int max_nf = 1;
    for (int g = 0; g < desc.n_terms; ++g) {
        const Term& t = terms[g];
        TermDesc& T = desc.t[g];
        if (t.factors.empty() || (int)t.factors.size() > MAX_FACTORS)
            throw std::runtime_error("blendy: bad factor count");
        if (t.group < 0 || t.group >= MAX_GROUPS)
            throw std::runtime_error("blendy: bad group");
        T.nf = t.factors.size();
        for (int j = 0; j < T.nf; ++j) {
            if (t.factors[j] < 0 || t.factors[j] >= n_tensors)
                throw std::runtime_error("blendy: bad factor index");
            T.f[j] = t.factors[j];
            T.shift[j] = t.shifts.empty() ? ZERO_FR : t.shifts[j];
        }
        T.eq = t.eq_weighted;
        if (T.eq && u.size() != n)
            throw std::runtime_error("blendy: eq-weighted term needs |u| == num_vars");
        T.group = t.group;
        T.scale = (T.nf - 1) + (T.eq ? 1 : 0) + 2;
        desc.n_groups = std::max(desc.n_groups, t.group + 1);
        if (desc.group_deg[t.group] && desc.group_deg[t.group] != T.nf)
            throw std::runtime_error("blendy: terms of one group must have equal degree");
        desc.group_deg[t.group] = T.nf;
        max_nf = std::max(max_nf, T.nf);
    }
    B = max_nf + 1;
    L = std::max(1, std::min<int>(l_stage, std::max<int>(nr, 1)));
    T = tile_bits;
    if (nr == 0)
        return;

    const int l1 = std::min<int>(L, nr), l2 = std::min<int>(L, nr - l1);
    const int fv0 = n - l1;
    const int tile0 = (T >= 0 && T < fv0) ? T : fv0;
    const long cells0 = ipow(B, L);
    ext_cap = cells0 << tile0;
    for (int i = 0; i < n_tensors; ++i)
        cudaMalloc(&ext[i], ext_cap * sizeof(Fr_t));
    cudaMalloc(&ptable, desc.n_terms * cells0 * sizeof(Fr_t));
    partial_cap = (long)desc.n_terms * (cells0 + 2 * PTABLE_TARGET_BLOCKS);
    cudaMalloc(&partials, partial_cap * sizeof(Fr_t));
    eq_cap = std::max(1L << tile0, 1L << L);
    cudaMalloc(&eq_tab, eq_cap * sizeof(Fr_t));
    cudaMalloc(&u_dev, (n + 2 * L + 2) * sizeof(Fr_t));
    cudaMalloc(&coef_out, MAX_GROUPS * 4 * sizeof(Fr_t));
    // Stage outputs alternate between the two buffers
    const unsigned long cap0 = 1UL << (n - l1);
    const unsigned long cap1 = 1UL << (n - l1 - l2);
    for (int i = 0; i < n_tensors; ++i) {
        Fr_t *a, *b;
        cudaMalloc(&a, cap0 * sizeof(Fr_t));
        cudaMalloc(&b, cap1 * sizeof(Fr_t));
        buf[0].push_back(a);
        buf[1].push_back(b);
    }
}

Sumcheck::~Sumcheck() {
    for (int i = 0; i < MAX_TENSORS; ++i)
        if (ext[i])
            cudaFree(ext[i]);
    for (auto p : buf[0])
        cudaFree(p);
    for (auto p : buf[1])
        cudaFree(p);
    for (Fr_t* p : {ptable, partials, eq_tab, u_dev, coef_out})
        if (p)
            cudaFree(p);
}

void Sumcheck::set_group_adjust(int group, int adjust) {
    desc.group_adjust[group] = adjust;
}

void Sumcheck::begin_stage() {
    const int nb = n - done;
    l_cur = std::min<int>(L, nr - done);
    const int fv = nb - l_cur;
    const int tile_lo = (T >= 0 && T < fv) ? T : fv;
    const long tb3 = 1L << tile_lo;
    const long ntiles = 1L << (fv - tile_lo);
    const int cells = ipow(B, l_cur);
    const int msb = order == Order::MSB_FIRST;
    stage_first = done;

    bool any_eq = false;
    for (int g = 0; g < desc.n_terms; ++g)
        any_eq |= desc.t[g].eq;
    if (any_eq) {
        std::vector<Fr_t> ufut(fv), ustage(l_cur);
        for (int i = 0; i < fv; ++i)
            ufut[i] = u[msb ? i : done + l_cur + i];
        for (int i = 0; i < l_cur; ++i)
            ustage[i] = u[bit_of_round(done + l_cur - 1 - i)];
        if (fv)
            check(cudaMemcpy(u_dev, ufut.data(), fv * sizeof(Fr_t), cudaMemcpyHostToDevice), "ufut");
        check(cudaMemcpy(u_dev + n, ustage.data(), l_cur * sizeof(Fr_t), cudaMemcpyHostToDevice), "ustage");
        eq_table_kernel<<<blocks(tb3, NT), NT>>>(u_dev, tile_lo, eq_tab, tb3, 0);
    }

    cudaMemset(ptable, 0, (long)desc.n_terms * cells * sizeof(Fr_t));
    int nb3 = cells >= PTABLE_TARGET_BLOCKS ? 1 : (PTABLE_TARGET_BLOCKS + cells - 1) / cells;
    nb3 = std::min<long>(nb3, std::max<long>(1, (tb3 + NT - 1) / NT));
    ExtPtrs ep{};
    for (int i = 0; i < n_tensors; ++i)
        ep.p[i] = ext[i];

    for (long ti = 0; ti < ntiles; ++ti) {
        for (int i = 0; i < n_tensors; ++i) {
            expand_kernel<<<blocks(tb3 << l_cur, NT), NT>>>(src[i], src_size[i], ext[i], l_cur, B, fv,
                                                            tile_lo, ti, msb);
            if (B > 2)
                for (int var = 0; var < l_cur; ++var)
                    extrapolate_kernel<<<blocks((long)cells * tb3, NT), NT>>>(ext[i], var, B, cells, tile_lo);
        }
        ptable_partial_kernel<<<dim3(cells, nb3, desc.n_terms), NT, NT * sizeof(Fr_t)>>>(
            desc, ep, eq_tab, partials, cells, tb3, nb3);
        ptable_finalize_kernel<<<blocks(desc.n_terms * cells, NT), NT>>>(desc, partials, ptable, cells, nb3,
                                                                         u_dev, tile_lo, fv, ti);
    }
    check(cudaGetLastError(), "stage build");
    rem = l_cur;
    in_stage = true;
}

std::vector<std::vector<Fr_t>> Sumcheck::round(const std::vector<Fr_t>& term_coefs) {
    if (done >= nr)
        throw std::runtime_error("blendy: no rounds left");
    if (!in_stage)
        begin_stage();
    Scalars sc;
    for (int g = 0; g < MAX_TERMS; ++g)
        sc.v[g] = g < (int)term_coefs.size() ? term_coefs[g] : ONE_FR;
    const Fr_t* ic = inverse_consts();
    round_kernel<<<1, 64>>>(desc, ptable, ipow(B, l_cur), rem, B, u_dev + n, sc, ic[0], ic[1], ic[2],
                            coef_out);
    Fr_t h[MAX_GROUPS * 4];
    check(cudaMemcpy(h, coef_out, sizeof(h), cudaMemcpyDeviceToHost), "round");
    std::vector<std::vector<Fr_t>> out(desc.n_groups);
    for (int k = 0; k < desc.n_groups; ++k)
        out[k].assign(h + 4 * k, h + 4 * k + desc.group_deg[k] + 1);
    return out;
}

void Sumcheck::bind(const Fr_t& v) {
    if (!in_stage)
        throw std::runtime_error("blendy: bind before round");
    stage_v.push_back(v);
    if (rem > 1) {
        const int ns = ipow(B, rem - 1);
        const Fr_t* ic = inverse_consts();
        ptable_fold_kernel<<<blocks(desc.n_terms * ns, NT), NT>>>(desc.n_terms, ptable, ipow(B, l_cur), B, ns,
                                                                  v, ic[0], ic[2]);
    }
    --rem;
    ++done;
    if (rem == 0)
        end_stage();
}

void Sumcheck::end_stage() {
    const int msb = order == Order::MSB_FIRST;
    const int nb = n - stage_first;
    const int fv = nb - l_cur;
    // eqv index bit i <-> stage variable bound
    std::vector<Fr_t> w(l_cur);
    for (int i = 0; i < l_cur; ++i)
        w[i] = stage_v[msb ? l_cur - 1 - i : i];
    check(cudaMemcpy(u_dev + n + L, w.data(), l_cur * sizeof(Fr_t), cudaMemcpyHostToDevice), "stage v");
    eq_table_kernel<<<blocks(1L << l_cur, NT), NT>>>(u_dev + n + L, l_cur, eq_tab, 1L << l_cur, 1);
    const unsigned long out = 1UL << fv;
    for (int i = 0; i < n_tensors; ++i) {
        Fr_t* dst = buf[next_buf][i];
        stage_fold_kernel<<<blocks(out, NT), NT>>>(src[i], src_size[i], eq_tab, dst, l_cur, fv, msb, out);
        src[i] = dst;
        src_size[i] = out;
    }
    check(cudaGetLastError(), "stage fold");
    next_buf ^= 1;
    stage_v.clear();
    in_stage = false;
}

const Fr_t* Sumcheck::data(int tensor) const {
    if (in_stage)
        throw std::runtime_error("blendy: data() inside a stage");
    return src[tensor];
}

Fr_t Sumcheck::value(int tensor) const {
    if (done != nr || in_stage)
        throw std::runtime_error("blendy: value() before the last round");
    Fr_t x = ZERO_FR;
    if (src_size[tensor] > 0)
        check(cudaMemcpy(&x, src[tensor], sizeof(Fr_t), cudaMemcpyDeviceToHost), "value");
    return x;
}

}  // namespace blendy
