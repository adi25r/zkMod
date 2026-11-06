#include "weight-stream.cuh"
#include "zkfc.cuh"
#include "ioutils.cuh"
#include "int-matmul.cuh"

#include <cstdlib>
#include <fstream>

bool weight_streaming_enabled() {
    const char* s = std::getenv("ZKLLM_WSTREAM");
    return !(s && std::string(s) == "0");
}

namespace {

size_t chunk_bytes() {
    const char* s = std::getenv("ZKLLM_WSTREAM_CHUNK_MB");
    const long mb = s ? std::atol(s) : 64;
    return static_cast<size_t>(mb > 0 ? mb : 64) << 20;
}

uint rows_per_chunk(const StreamedWeight& w, uint chunk_rows) {
    if (chunk_rows == 0)
        chunk_rows = static_cast<uint>(chunk_bytes() / (sizeof(int) * static_cast<size_t>(w.out_dim)));
    return std::max(1u, std::min(chunk_rows, w.in_dim));
}

// Device staging buffer for one int32 row chunk
class ChunkStager {
    const StreamedWeight& w;
    int* dev = nullptr;

public:
    const uint chunk_rows;
    ChunkStager(const StreamedWeight& w_, uint chunk_rows_)
        : w(w_), chunk_rows(rows_per_chunk(w_, chunk_rows_)) {
        cudaMalloc((void**)&dev, sizeof(int) * static_cast<size_t>(chunk_rows) * w.out_dim);
    }
    ~ChunkStager() { cudaFree(dev); }
    ChunkStager(const ChunkStager&) = delete;
    ChunkStager& operator=(const ChunkStager&) = delete;

    uint rows_at(uint i0) const { return std::min(chunk_rows, w.in_dim - i0); }
    const int* load(uint i0) {
        const size_t n = static_cast<size_t>(rows_at(i0)) * w.out_dim;
        cudaMemcpy(dev, w.host.data() + static_cast<size_t>(i0) * w.out_dim, sizeof(int) * n,
                   cudaMemcpyHostToDevice);
        return dev;
    }
};

// Doubles an eq table kept in Montgomery form
KERNEL void ws_eq_step(Fr_t* E, uint half, Fr_t u) {
    const uint gid = GET_GLOBAL_ID();
    if (gid >= half)
        return;
    const Fr_t e = E[gid];
    const Fr_t eu = blstrs__scalar__Scalar_mul(e, blstrs__scalar__Scalar_mont(u));
    E[gid + half] = eu;
    E[gid] = blstrs__scalar__Scalar_sub(e, eu);
}

KERNEL void ws_eq_init(Fr_t* E) {
    E[0] = blstrs__scalar__Scalar_mont({1, 0, 0, 0, 0, 0, 0, 0});
}

// E[x] = eq(u, x) * R, u[0] for bit 0 of x, so Scalar_mul(E[x], w) = eq(u, x) * w
FrTensor eq_table_mont(const vector<Fr_t>& u) {
    FrTensor E(1u << u.size());
    ws_eq_init<<<1, 1>>>(E.gpu_data);
    for (uint b = 0; b < u.size(); ++b) {
        const uint half = 1u << b;
        ws_eq_step<<<(half + FrNumThread - 1) / FrNumThread, FrNumThread>>>(E.gpu_data, half, u[b]);
    }
    cudaDeviceSynchronize();
    return E;
}

// C[:, :] += X[:, k0:k0+kc] * B, with B the kc x colsB int32 chunk
KERNEL void ws_matmul_accum(const Fr_t* A, const int* B, Fr_t* C, int rowsA, int lda, int k0, int kc,
                            int colsB) {
    __shared__ Fr_t A_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ Fr_t B_tile[TILE_WIDTH][TILE_WIDTH];

    const int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    const int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    Fr_t sum = blstrs__scalar__Scalar_ZERO;

    for (int t = 0; t < (kc - 1) / TILE_WIDTH + 1; ++t) {
        const int ka = t * TILE_WIDTH + threadIdx.x, kb = t * TILE_WIDTH + threadIdx.y;
        A_tile[threadIdx.y][threadIdx.x] = (row < rowsA && ka < kc)
                                               ? A[static_cast<size_t>(row) * lda + k0 + ka]
                                               : blstrs__scalar__Scalar_ZERO;
        B_tile[threadIdx.y][threadIdx.x] = (kb < kc && col < colsB)
                                               ? int_to_scalar(B[static_cast<size_t>(kb) * colsB + col])
                                               : blstrs__scalar__Scalar_ZERO;
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; ++k)
            sum = blstrs__scalar__Scalar_add(
                sum, blstrs__scalar__Scalar_mul(A_tile[threadIdx.y][k], B_tile[k][threadIdx.x]));
        __syncthreads();
    }
    if (row < rowsA && col < colsB) {
        const size_t idx = static_cast<size_t>(row) * colsB + col;
        C[idx] = blstrs__scalar__Scalar_add(C[idx], blstrs__scalar__Scalar_mont(sum));
    }
}

// dst[r] = sum_j E_out[j] * W[r, j] for the rows of one chunk (one block per row)
KERNEL void ws_row_reduce(const int* W, uint rows, uint cols, const Fr_t* E_out, Fr_t* dst) {
    __shared__ Fr_t s[FrNumThread];
    const uint r = blockIdx.x;
    if (r >= rows)
        return;
    Fr_t acc = blstrs__scalar__Scalar_ZERO;
    for (uint j = threadIdx.x; j < cols; j += blockDim.x)
        acc = blstrs__scalar__Scalar_add(
            acc, blstrs__scalar__Scalar_mul(E_out[j], int_to_scalar(W[static_cast<size_t>(r) * cols + j])));
    s[threadIdx.x] = acc;
    __syncthreads();
    for (uint stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            s[threadIdx.x] = blstrs__scalar__Scalar_add(s[threadIdx.x], s[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0)
        dst[r] = s[0];
}

KERNEL void ws_open_accum(const int* W, uint i0, uint rows, uint cols, uint log_outp, uint log_G,
                          const Fr_t* E_rows, Fr_t* acc) {
    const uint k = GET_GLOBAL_ID();
    if (k >= (1u << log_G))
        return;
    const size_t f_lo = static_cast<size_t>(i0) << log_outp, f_hi = static_cast<size_t>(i0 + rows)
                                                                    << log_outp;
    const size_t r0 = f_lo >> log_G, r1 = (f_hi + (1ul << log_G) - 1) >> log_G;
    const size_t outp_mask = (1ul << log_outp) - 1;
    Fr_t sum = blstrs__scalar__Scalar_ZERO;
    for (size_t r = r0; r < r1; ++r) {
        const size_t f = (r << log_G) + k;
        if (f < f_lo || f >= f_hi)
            continue;
        const size_t i = f >> log_outp, j = f & outp_mask;
        if (j >= cols)
            continue;
        sum = blstrs__scalar__Scalar_add(
            sum, blstrs__scalar__Scalar_mul(E_rows[r], int_to_scalar(W[(i - i0) * cols + j])));
    }
    acc[k] = blstrs__scalar__Scalar_add(acc[k], sum);
}

}  // namespace

StreamedWeight load_streamed_weight(const string& generator_filename, const string& weight_filename,
                                    const string& com_filename, uint in_dim, uint out_dim) {
    const size_t n = findsize(weight_filename) / sizeof(int);
    if (n != static_cast<size_t>(in_dim) * out_dim)
        throw std::runtime_error("load_streamed_weight: " + weight_filename + " has " + std::to_string(n) +
                                 " entries, expected in*out");
    std::vector<int> host(n);
    std::ifstream f(weight_filename, std::ios::binary);
    if (!f.read(reinterpret_cast<char*>(host.data()), sizeof(int) * n))
        throw std::runtime_error("load_streamed_weight: cannot read " + weight_filename);
    return {std::move(host), Commitment(generator_filename), G1TensorJacobian(com_filename), in_dim, out_dim};
}

FrTensor streamed_matmul(const FrTensor& X, const StreamedWeight& w, uint chunk_rows) {
    if (X.size % w.in_dim)
        throw std::runtime_error("streamed_matmul: input size does not match");
    const uint batch = X.size / w.in_dim;
    FrTensor Y(batch * w.out_dim);
    cudaMemset(Y.gpu_data, 0, sizeof(Fr_t) * Y.size);
    ChunkStager st(w, chunk_rows);
    const dim3 block(TILE_WIDTH, TILE_WIDTH),
        grid((w.out_dim + TILE_WIDTH - 1) / TILE_WIDTH, (batch + TILE_WIDTH - 1) / TILE_WIDTH);
    // Integer path
    long* Xl = nullptr;
    if (int_matmul_enabled())
        cudaMalloc((void**)&Xl, sizeof(long) * static_cast<size_t>(batch) * st.chunk_rows);
    for (uint i0 = 0; i0 < w.in_dim; i0 += st.chunk_rows) {
        const int* Wc = st.load(i0);
        const uint kc = st.rows_at(i0);
        if (Xl && int_matmul_safe(fr_slice_to_long(X.gpu_data, batch, w.in_dim, i0, kc, Xl),
                                  int_max_abs(Wc, static_cast<size_t>(kc) * w.out_dim), kc)) {
            long_matmul_to_fr(Xl, kc, Wc, w.out_dim, Y.gpu_data, w.out_dim, batch, kc, w.out_dim, true);
            continue;
        }
        ws_matmul_accum<<<grid, block>>>(X.gpu_data, Wc, Y.gpu_data, batch, w.in_dim, i0, kc, w.out_dim);
        cudaDeviceSynchronize();
    }
    if (Xl)
        cudaFree(Xl);
    return Y;
}

StreamedLinearProof streamed_prove_linear(const FrTensor& X, const FrTensor& Y, const StreamedWeight& w,
                                          const vector<Fr_t>& u_batch, const vector<Fr_t>& u_input,
                                          const vector<Fr_t>& u_output, uint chunk_rows) {
    const uint in = w.in_dim, out = w.out_dim;
    if (X.size % in)
        throw std::runtime_error("streamed_prove_linear: input size does not match");
    const uint batch = X.size / in;
    // zkFC::prove
    auto claim = Y.multi_dim_me({u_batch, u_output}, {batch, out});
    auto X_reduced = X.partial_me(u_batch, batch, in);
    auto claim_X = X.multi_dim_me({u_batch, u_input}, {batch, in});
    return streamed_prove_linear_reduced(claim, X_reduced, claim_X, w, u_input, u_output, chunk_rows);
}

StreamedLinearProof streamed_prove_linear_reduced(const Fr_t& claim, const FrTensor& X_reduced,
                                                  const Fr_t& claim_X, const StreamedWeight& w,
                                                  const vector<Fr_t>& u_input, const vector<Fr_t>& u_output,
                                                  uint chunk_rows) {
    const uint in = w.in_dim, out = w.out_dim;
    StreamedLinearProof res;
    // Commitment::open on W padded to pow2(in) x pow2(out)
    const vector<Fr_t> u_cat = concatenate(vector<vector<Fr_t>>({u_output, u_input}));
    const uint log_rows = ceilLog2(w.com.size), log_G = ceilLog2(w.generator.size), log_outp = ceilLog2(out);
    if (u_cat.size() < log_rows || u_cat.size() - log_rows != log_G || (1u << log_G) != w.generator.size)
        throw std::runtime_error("streamed_prove_linear: commitment shape does not match the weight");
    const vector<Fr_t> u_rows(u_cat.end() - log_rows, u_cat.end());
    const vector<Fr_t> u_inner(u_cat.begin(), u_cat.end() - log_rows);

    // One streamed pass
    FrTensor E_out = eq_table_mont(u_output), E_rows = eq_table_mont(u_rows);
    FrTensor W_reduced(in), v(w.generator.size);
    cudaMemset(v.gpu_data, 0, sizeof(Fr_t) * v.size);
    {
        ChunkStager st(w, chunk_rows);
        for (uint i0 = 0; i0 < in; i0 += st.chunk_rows) {
            const uint rows = st.rows_at(i0);
            const int* Wc = st.load(i0);
            ws_row_reduce<<<rows, FrNumThread>>>(Wc, rows, out, E_out.gpu_data, W_reduced.gpu_data + i0);
            ws_open_accum<<<(w.generator.size + FrNumThread - 1) / FrNumThread, FrNumThread>>>(
                Wc, i0, rows, out, log_outp, log_G, E_rows.gpu_data, v.gpu_data);
            cudaDeviceSynchronize();
        }
    }

    auto final_claim = zkip(claim, X_reduced, W_reduced, u_input, res.proof);
    res.claim_W = W_reduced(u_input);
    if (claim_X * res.claim_W != final_claim)
        throw std::runtime_error("Claim does not match");

    // verifyWeightClaim
    auto g_temp = (w.com.size == 1) ? w.com(0) : w.com(u_rows);
    (void)g_temp;
    Commitment::me_open_check(v, w.generator, u_inner.begin(), u_inner.end(), res.claim_W,
                              "verifyWeightClaim: opening != c.claim");
    return res;
}

void streamed_prove_linear(const FrTensor& X, const FrTensor& Y, const StreamedWeight& w) {
    const uint batch = X.size / w.in_dim;
    auto u_batch = random_vec(ceilLog2(batch));
    auto u_input = random_vec(ceilLog2(w.in_dim));
    auto u_output = random_vec(ceilLog2(w.out_dim));
    streamed_prove_linear(X, Y, w, u_batch, u_input, u_output);
}
