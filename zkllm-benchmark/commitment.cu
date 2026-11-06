#include "commitment.cuh"
#include <algorithm>
#include <cstdlib>
#include <memory>

Commitment Commitment::random(uint size) {
    Commitment out(size, G1Jacobian_generator);
    out *= FrTensor::random(size);
    return out;
}

// The commit kernels write one G1 point per tensor element before summing each row
namespace {
uint commit_rows_per_chunk(uint row_len) {
    const char* s = std::getenv("ZKLLM_COMMIT_CHUNK_MB");
    const long mb = s ? std::atol(s) : 256;
    const size_t bytes = static_cast<size_t>(mb > 0 ? mb : 256) << 20;
    return std::max<size_t>(1, bytes / (sizeof(G1Jacobian_t) * static_cast<size_t>(row_len)));
}

// launch(scalars of the chunk, G1 temporary, rows in the chunk) fills the temporary
template <typename Launch>
G1TensorJacobian commit_by_rows(uint row_len, const FrTensor& t, Launch launch) {
    const uint m = t.size / row_len;
    G1TensorJacobian out(m);
    const uint rc = std::min(m, commit_rows_per_chunk(row_len));
    for (uint r0 = 0; r0 < m; r0 += rc) {
        const uint n = std::min(rc, m - r0);
        G1TensorJacobian temp(n * row_len);
        launch(t.gpu_data + static_cast<size_t>(r0) * row_len, temp.gpu_data, n);
        cudaDeviceSynchronize();
        G1TensorJacobian rows = temp.rowwise_sum(n, row_len);
        cudaMemcpy(out.gpu_data + r0, rows.gpu_data, sizeof(G1Jacobian_t) * n, cudaMemcpyDeviceToDevice);
    }
    return out;
}
}  // namespace

G1TensorJacobian Commitment::commit(const FrTensor& t) const {
    if (t.size % size != 0)
        throw std::runtime_error("Commitment::commit - Incompatible dimensions");
    return commit_by_rows(size, t, [&](Fr_t* sc, G1Jacobian_t* tmp, uint n) {
        G1_jacobian_elementwise_mul_broadcast<<<(n * size + G1NumThread - 1) / G1NumThread, G1NumThread>>>(
            gpu_data, sc, tmp, size, n);
    });
}

DEVICE G1Jacobian_t commit_int_dev_func(G1Jacobian_t a, Fr_t s) {
    const int x = scalar_to_int(s);
    G1Jacobian_t out = blstrs__g1__G1Affine_ZERO;
#pragma unroll
    for (uint i = 0; i < 31; ++i) {
        if ((x >> i) & 1)
            out = blstrs__g1__G1Affine_add(out, a);
        a = blstrs__g1__G1Affine_double(a);
    }

    if (x < 0)
        out = blstrs__g1__G1Affine_add(out, G1Jacobian_minus(a));
    return out;
}

KERNEL void commit_int_kernel(const G1Jacobian_t* generators, const Fr_t* scalars, G1Jacobian_t* out, uint n,
                              uint m) {
    const uint gid = GET_GLOBAL_ID();
    if (gid >= m * n)
        return;
    out[gid] = commit_int_dev_func(generators[gid % n], scalars[gid]);
}

G1TensorJacobian Commitment::commit_int(const FrTensor& t) const {
    if (t.size % size != 0)
        throw std::runtime_error("Commitment::commit_int - Incompatible dimensions");
    return commit_by_rows(size, t, [&](Fr_t* sc, G1Jacobian_t* tmp, uint n) {
        commit_int_kernel<<<(n * size + G1NumThread - 1) / G1NumThread, G1NumThread>>>(gpu_data, sc, tmp,
                                                                                       size, n);
    });
}

DEVICE G1Jacobian_t commit_long_dev_func(G1Jacobian_t a, Fr_t s) {
    const long x = scalar_to_long(s);
    G1Jacobian_t out = blstrs__g1__G1Affine_ZERO;
    for (uint i = 0; i < 63; ++i) {
        if ((x >> i) & 1)
            out = blstrs__g1__G1Affine_add(out, a);
        a = blstrs__g1__G1Affine_double(a);
    }
    if (x < 0)
        out = blstrs__g1__G1Affine_add(out, G1Jacobian_minus(a));
    return out;
}

KERNEL void commit_long_kernel(const G1Jacobian_t* generators, const Fr_t* scalars, G1Jacobian_t* out, uint n,
                               uint m) {
    const uint gid = GET_GLOBAL_ID();
    if (gid >= m * n)
        return;
    out[gid] = commit_long_dev_func(generators[gid % n], scalars[gid]);
}

G1TensorJacobian Commitment::commit_long(const FrTensor& t) const {
    if (t.size % size != 0)
        throw std::runtime_error("Commitment::commit_long - Incompatible dimensions");
    return commit_by_rows(size, t, [&](Fr_t* sc, G1Jacobian_t* tmp, uint n) {
        commit_long_kernel<<<(n * size + G1NumThread - 1) / G1NumThread, G1NumThread>>>(gpu_data, sc, tmp,
                                                                                        size, n);
    });
}

// flags[0] |= 1 if some value is outside int32
KERNEL void scalar_range_kernel(const Fr_t* t, uint n, int* flags) {
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n)
        return;
    Fr_t x = t[gid];
    const bool pos =
        !x.val[7] && !x.val[6] && !x.val[5] && !x.val[4] && !x.val[3] && !x.val[2] && !(x.val[1] >> 31);
    if (!pos)
        x = blstrs__scalar__Scalar_sub({0, 0, 0, 0, 0, 0, 0, 0}, x);
    const bool fits_long =
        !x.val[7] && !x.val[6] && !x.val[5] && !x.val[4] && !x.val[3] && !x.val[2] && !(x.val[1] >> 31);
    const bool fits_int = fits_long && !x.val[1] && !(x.val[0] >> 31);
    if (!fits_int)
        atomicOr(&flags[0], 1);
    if (!fits_long)
        atomicOr(&flags[1], 1);
}

int Commitment::value_class(const FrTensor& t) {
    int* flags;
    cudaMalloc((void**)&flags, 2 * sizeof(int));
    cudaMemset(flags, 0, 2 * sizeof(int));
    scalar_range_kernel<<<(t.size + FrNumThread - 1) / FrNumThread, FrNumThread>>>(t.gpu_data, t.size, flags);
    int h[2];
    cudaMemcpy(h, flags, 2 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(flags);
    return !h[0] ? 0 : !h[1] ? 1 : 2;
}

G1TensorJacobian Commitment::commit_as(const FrTensor& t, int cls) const {
    return cls == 0 ? commit_int(t) : cls == 1 ? commit_long(t) : commit(t);
}

G1TensorJacobian Commitment::commit_auto(const FrTensor& t) const {
    return commit_as(t, value_class(t));
}

G1TensorJacobian Commitment::commit_int_multi(const vector<FrTensor>& ts) const {
    uint num_row = 0;
    for (auto& t : ts) {
        if (t.size % size != 0)
            throw std::runtime_error("Commitment::commit_int_multi - Incompatible dimensions");
        num_row += t.size / size;
    }

    G1TensorJacobian temp(num_row * size);
    auto temp_start = temp.gpu_data;
    for (auto& t : ts) {
        uint m = t.size / size;
        commit_int_kernel<<<(m * size + G1NumThread - 1) / G1NumThread, G1NumThread>>>(gpu_data, t.gpu_data,
                                                                                       temp_start, size, m);
        cudaDeviceSynchronize();
        temp_start += m * size;
    }
    return temp.rowwise_sum(temp.size / size, size);
}

DEVICE void me_open_pair(const Fr_t* scalars, const G1Jacobian_t* generators, Fr_t u, Fr_t* new_scalars,
                         G1Jacobian_t* new_generators, G1Jacobian_t* temp_out, G1Jacobian_t* temp_out0,
                         G1Jacobian_t* temp_out1, uint old_size, uint gid) {
    uint gid0 = 2 * gid;
    uint gid1 = 2 * gid + 1;

    if (gid1 >= old_size) {
        new_scalars[gid] = blstrs__scalar__Scalar_sub(
            scalars[gid0], blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(u, scalars[gid0])));
        new_generators[gid] = G1Jacobian_mul(generators[gid0], u);
        temp_out[gid] = G1Jacobian_mul(generators[gid0], scalars[gid0]);
        temp_out0[gid] = blstrs__g1__G1Affine_ZERO;
        temp_out1[gid] = blstrs__g1__G1Affine_ZERO;
        return;
    }

    new_scalars[gid] = blstrs__scalar__Scalar_add(
        scalars[gid0], blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(
                           u, blstrs__scalar__Scalar_sub(scalars[gid1], scalars[gid0]))));
    new_generators[gid] = blstrs__g1__G1Affine_add(
        generators[gid1],
        G1Jacobian_mul(blstrs__g1__G1Affine_add(generators[gid0], G1Jacobian_minus(generators[gid1])), u));
    temp_out[gid] = blstrs__g1__G1Affine_add(G1Jacobian_mul(generators[gid0], scalars[gid0]),
                                             G1Jacobian_mul(generators[gid1], scalars[gid1]));
    temp_out0[gid] = G1Jacobian_mul(generators[gid1], scalars[gid0]);
    temp_out1[gid] = G1Jacobian_mul(generators[gid0], scalars[gid1]);
}

KERNEL void me_open_step(GLOBAL Fr_t* scalars, GLOBAL G1Jacobian_t* generators,
                         Fr_t u,  // always assume that scalars and u is in mont form
                         GLOBAL Fr_t* new_scalars, GLOBAL G1Jacobian_t* new_generators,
                         GLOBAL G1Jacobian_t* temp_out, GLOBAL G1Jacobian_t* temp_out0,
                         GLOBAL G1Jacobian_t* temp_out1, uint old_size, uint new_size) {
    const uint gid = GET_GLOBAL_ID();
    if (gid >= new_size)
        return;
    me_open_pair(scalars, generators, u, new_scalars, new_generators, temp_out, temp_out0, temp_out1,
                 old_size, gid);
}

Fr_t Commitment::me_open(const FrTensor& t, const Commitment& generators, vector<Fr_t>::const_iterator begin,
                         vector<Fr_t>::const_iterator end, vector<G1Jacobian_t>& proof) {
    if (t.size != generators.size)
        throw std::runtime_error("Commitment::me_open - Incompatible dimensions " + std::to_string(t.size) +
                                 " " + std::to_string(generators.size));
    if (begin >= end) {
        proof.push_back(generators(0));
        return t(0);
    }
    uint new_size = (t.size + 1) / 2;
    FrTensor new_scalars(new_size);
    Commitment new_generators(new_size);
    G1TensorJacobian temp(new_size), temp0(new_size), temp1(new_size);
    me_open_step<<<(new_size + G1NumThread - 1) / G1NumThread, G1NumThread>>>(
        t.gpu_data, generators.gpu_data, *begin, new_scalars.gpu_data, new_generators.gpu_data, temp.gpu_data,
        temp0.gpu_data, temp1.gpu_data, t.size, new_size);
    cudaDeviceSynchronize();
    proof.push_back(temp.sum());
    proof.push_back(temp0.sum());
    proof.push_back(temp1.sum());
    return me_open(new_scalars, new_generators, begin + 1, end, proof);
}

Fr_t Commitment::open(const FrTensor& t, const G1TensorJacobian& com, const vector<Fr_t>& u) const {
    const vector<Fr_t> u_out(u.end() - ceilLog2(com.size), u.end());
    const vector<Fr_t> u_in(u.begin(), u.end() - ceilLog2(com.size));
    auto g_temp = (com.size == 1) ? com(0) : com(u_out);
    vector<G1Jacobian_t> proof;
    return me_open(t.partial_me(u_out, t.size / com.size), *this, u_in.begin(), u_in.end(), proof);
}

Weight create_weight(string generator_filename, string weight_filename, string com_filename, uint in_dim,
                     uint out_dim) {
    Commitment generator(generator_filename);
    FrTensor weight = FrTensor::from_int_bin(weight_filename);
    G1TensorJacobian com(com_filename);
    return {generator, weight, com, in_dim, out_dim};
}

// Opening batching

namespace {

struct PendingOpen {
    FrTensor scalars;       // current folded vector (owned)
    Commitment generators;  // current folded generators (owned)
    vector<Fr_t> u;         // remaining challenges
    vector<G1Jacobian_t> proof;
    Fr_t expected;
    string what;
};

struct OpenBatch {
    bool active = false;
    vector<PendingOpen> pending;
    size_t bytes = 0;
};

OpenBatch& open_batch() {
    static OpenBatch b;
    return b;
}

bool open_batching_enabled() {
    const char* s = std::getenv("ZKLLM_BATCH_OPEN");
    return s && string(s) == "1";
}

uint open_batch_min() {
    const char* s = std::getenv("ZKLLM_BATCH_OPEN_MIN");
    const long v = s ? std::atol(s) : 16384;
    return static_cast<uint>(v >= 1 ? v : 16384);
}

size_t open_batch_cap_bytes() {
    const char* s = std::getenv("ZKLLM_BATCH_OPEN_MB");
    const long mb = s ? std::atol(s) : 256;
    return static_cast<size_t>(mb > 0 ? mb : 256) << 20;
}

// One round of Commitment::me_open, the same kernel and sums
void me_open_round(const FrTensor& t, const G1TensorJacobian& g, const Fr_t& u, FrTensor& ns, Commitment& ng,
                   vector<G1Jacobian_t>& proof) {
    const uint new_size = (t.size + 1) / 2;
    G1TensorJacobian temp(new_size), temp0(new_size), temp1(new_size);
    me_open_step<<<(new_size + G1NumThread - 1) / G1NumThread, G1NumThread>>>(
        t.gpu_data, g.gpu_data, u, ns.gpu_data, ng.gpu_data, temp.gpu_data, temp0.gpu_data, temp1.gpu_data,
        t.size, new_size);
    cudaDeviceSynchronize();
    proof.push_back(temp.sum());
    proof.push_back(temp0.sum());
    proof.push_back(temp1.sum());
}

struct OpenDesc {
    const Fr_t* sc;
    const G1Jacobian_t* gen;
    Fr_t u;
    Fr_t* nsc;
    G1Jacobian_t* ngen;
    G1Jacobian_t* t;
    G1Jacobian_t* t0;
    G1Jacobian_t* t1;
    uint old_size, new_size;
    size_t offset;
};

KERNEL void me_open_step_batched(const OpenDesc* d, uint nd, size_t total) {
    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= total)
        return;
    uint lo = 0, hi = nd - 1;  // last descriptor with offset <= gid
    while (lo < hi) {
        const uint mid = (lo + hi + 1) / 2;
        if (d[mid].offset <= gid)
            lo = mid;
        else
            hi = mid - 1;
    }
    const OpenDesc& e = d[lo];
    me_open_pair(e.sc, e.gen, e.u, e.nsc, e.ngen, e.t, e.t0, e.t1, e.old_size,
                 static_cast<uint>(gid - e.offset));
}

KERNEL void g1_segment_sum(G1Jacobian_t* const* seg_ptr, const uint* seg_len, G1Jacobian_t* out) {
    extern __shared__ G1Jacobian_t seg_sdata[];
    const uint s = blockIdx.x, tid = threadIdx.x;
    const G1Jacobian_t* p = seg_ptr[s];
    G1Jacobian_t acc = blstrs__g1__G1Affine_ZERO;
    for (uint i = tid; i < seg_len[s]; i += blockDim.x)
        acc = blstrs__g1__G1Affine_add(acc, p[i]);
    seg_sdata[tid] = acc;
    __syncthreads();
    for (uint k = blockDim.x / 2; k > 0; k >>= 1) {
        if (tid < k)
            seg_sdata[tid] = blstrs__g1__G1Affine_add(seg_sdata[tid], seg_sdata[tid + k]);
        __syncthreads();
    }
    if (tid == 0)
        out[s] = seg_sdata[0];
}

template <class T>
T* device_copy(const vector<T>& h) {
    T* d;
    cudaMalloc((void**)&d, sizeof(T) * h.size());
    cudaMemcpy(d, h.data(), sizeof(T) * h.size(), cudaMemcpyHostToDevice);
    return d;
}

}  // namespace

void Commitment::begin_open_batch() {
    auto& b = open_batch();
    if (!b.pending.empty())
        throw std::runtime_error("begin_open_batch: openings still queued");
    b.active = open_batching_enabled();
}

size_t Commitment::flush_openings(vector<vector<G1Jacobian_t>>* proofs) {
    auto& b = open_batch();
    vector<PendingOpen> ps;
    ps.swap(b.pending);
    b.bytes = 0;
    const size_t n = ps.size();
    // Current state of each opening
    vector<const Fr_t*> sc(n);
    vector<const G1Jacobian_t*> gen(n);
    vector<uint> size(n), round(n, 0);
    for (size_t i = 0; i < n; ++i) {
        sc[i] = ps[i].scalars.gpu_data;
        gen[i] = ps[i].generators.gpu_data;
        size[i] = ps[i].scalars.size;
    }
    vector<void*> arenas;
    for (;;) {
        vector<size_t> act;
        for (size_t i = 0; i < n; ++i)
            if (round[i] < ps[i].u.size())
                act.push_back(i);
        if (act.empty())
            break;
        vector<OpenDesc> d(act.size());
        size_t total = 0;
        for (size_t a = 0; a < act.size(); ++a) {
            d[a].old_size = size[act[a]];
            d[a].new_size = (size[act[a]] + 1) / 2;
            d[a].offset = total;
            total += d[a].new_size;
        }
        Fr_t* nsc;
        G1Jacobian_t *ngen, *tmp;
        cudaMalloc((void**)&nsc, sizeof(Fr_t) * total);
        cudaMalloc((void**)&ngen, sizeof(G1Jacobian_t) * total);
        cudaMalloc((void**)&tmp, sizeof(G1Jacobian_t) * 3 * total);
        vector<G1Jacobian_t*> seg_ptr(3 * act.size());
        vector<uint> seg_len(3 * act.size());
        for (size_t a = 0; a < act.size(); ++a) {
            const size_t i = act[a];
            d[a].sc = sc[i];
            d[a].gen = gen[i];
            d[a].u = ps[i].u[round[i]];
            d[a].nsc = nsc + d[a].offset;
            d[a].ngen = ngen + d[a].offset;
            d[a].t = tmp + d[a].offset;
            d[a].t0 = tmp + total + d[a].offset;
            d[a].t1 = tmp + 2 * total + d[a].offset;
            seg_ptr[3 * a] = d[a].t;
            seg_ptr[3 * a + 1] = d[a].t0;
            seg_ptr[3 * a + 2] = d[a].t1;
            seg_len[3 * a] = seg_len[3 * a + 1] = seg_len[3 * a + 2] = d[a].new_size;
        }
        OpenDesc* dd = device_copy(d);
        me_open_step_batched<<<(total + G1NumThread - 1) / G1NumThread, G1NumThread>>>(
            dd, static_cast<uint>(d.size()), total);
        G1Jacobian_t** dsp = device_copy(seg_ptr);
        uint* dsl = device_copy(seg_len);
        G1Jacobian_t* sums;
        cudaMalloc((void**)&sums, sizeof(G1Jacobian_t) * seg_ptr.size());
        g1_segment_sum<<<seg_ptr.size(), G1NumThread, sizeof(G1Jacobian_t) * G1NumThread>>>(dsp, dsl, sums);
        vector<G1Jacobian_t> hs(seg_ptr.size());
        cudaMemcpy(hs.data(), sums, sizeof(G1Jacobian_t) * hs.size(), cudaMemcpyDeviceToHost);
        for (auto* p : {(void*)dd, (void*)dsp, (void*)dsl, (void*)sums, (void*)tmp})
            cudaFree(p);
        arenas.push_back(nsc);
        arenas.push_back(ngen);
        for (size_t a = 0; a < act.size(); ++a) {
            const size_t i = act[a];
            for (int k = 0; k < 3; ++k)
                ps[i].proof.push_back(hs[3 * a + k]);
            sc[i] = d[a].nsc;
            gen[i] = d[a].ngen;
            size[i] = d[a].new_size;
            ++round[i];
        }
    }
    string failed;
    for (size_t i = 0; i < n; ++i) {
        Fr_t v;
        G1Jacobian_t g;
        cudaMemcpy(&v, sc[i], sizeof(Fr_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&g, gen[i], sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost);
        ps[i].proof.push_back(g);
        if (v != ps[i].expected && failed.empty())
            failed = ps[i].what;
    }
    for (void* p : arenas)
        cudaFree(p);
    b.active = false;
    if (proofs)
        for (auto& p : ps)
            proofs->push_back(p.proof);
    if (!failed.empty())
        throw std::runtime_error(failed);
    return n;
}

void Commitment::me_open_check(const FrTensor& t, const Commitment& generators,
                               vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end,
                               const Fr_t& expected, const string& what) {
    auto& b = open_batch();
    if (!b.active) {
        vector<G1Jacobian_t> proof;
        if (me_open(t, generators, begin, end, proof) != expected)
            throw std::runtime_error(what);
        return;
    }
    if (t.size != generators.size)
        throw std::runtime_error("Commitment::me_open_check - Incompatible dimensions");
    const uint min_size = open_batch_min();
    vector<G1Jacobian_t> proof;
    std::unique_ptr<FrTensor> cur_t;
    std::unique_ptr<Commitment> cur_g;
    while (begin < end && (cur_t ? cur_t->size : t.size) > min_size) {
        const FrTensor& ti = cur_t ? *cur_t : t;
        const Commitment& gi = cur_g ? *cur_g : generators;
        const uint new_size = (ti.size + 1) / 2;
        auto ns = std::make_unique<FrTensor>(new_size);
        auto ng = std::make_unique<Commitment>(new_size);
        me_open_round(ti, gi, *begin, *ns, *ng, proof);
        cur_t = std::move(ns);
        cur_g = std::move(ng);
        ++begin;
    }
    PendingOpen p{cur_t ? std::move(*cur_t) : FrTensor(t),
                  cur_g ? Commitment(*cur_g) : Commitment(generators),
                  vector<Fr_t>(begin, end),
                  std::move(proof),
                  expected,
                  what};
    b.bytes += static_cast<size_t>(p.scalars.size) * (sizeof(Fr_t) + sizeof(G1Jacobian_t));
    b.pending.push_back(std::move(p));
    if (b.bytes > open_batch_cap_bytes()) {
        flush_openings();
        b.active = true;
    }
}

void Commitment::open_check(const FrTensor& t, const G1TensorJacobian& com, const vector<Fr_t>& u,
                            const Fr_t& expected, const string& what) const {
    const vector<Fr_t> u_out(u.end() - ceilLog2(com.size), u.end());
    const vector<Fr_t> u_in(u.begin(), u.end() - ceilLog2(com.size));
    auto g_temp = (com.size == 1) ? com(0) : com(u_out);
    (void)g_temp;
    me_open_check(t.partial_me(u_out, t.size / com.size), *this, u_in.begin(), u_in.end(), expected, what);
}
