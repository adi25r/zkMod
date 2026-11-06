#include "int-matmul.cuh"

#include <climits>
#include <cstdlib>
#include <string>

bool int_matmul_enabled() {
    const char* s = std::getenv("ZKLLM_INT_MATMUL");
    return !(s && std::string(s) == "0");
}

namespace {

constexpr int IMM_BM = 64, IMM_BN = 64, IMM_BK = 16, IMM_T = 16;  // 16 x 16 threads, 4 x 4 outputs each

KERNEL void int_max_abs_kernel(const int* x, size_t n, unsigned long long* out) {
    unsigned long long m = 0;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const long v = x[i];
        m = max(m, static_cast<unsigned long long>(v < 0 ? -v : v));
    }
    for (int o = 16; o > 0; o >>= 1)
        m = max(m, __shfl_down_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0)
        atomicMax(out, m);
}

KERNEL void fr_slice_to_long_kernel(const Fr_t* A, uint rows, uint lda, uint k0, uint kc, long* out,
                                    unsigned long long* max_abs) {
    const size_t n = static_cast<size_t>(rows) * kc;
    unsigned long long m = 0;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t r = i / kc, c = i % kc;
        Fr_t x = A[r * lda + k0 + c];
        const bool pos =
            !x.val[7] && !x.val[6] && !x.val[5] && !x.val[4] && !x.val[3] && !x.val[2] && !(x.val[1] >> 31);
        if (!pos)
            x = blstrs__scalar__Scalar_sub({0, 0, 0, 0, 0, 0, 0, 0}, x);
        const bool fits =
            !x.val[7] && !x.val[6] && !x.val[5] && !x.val[4] && !x.val[3] && !x.val[2] && !(x.val[1] >> 31);
        const unsigned long long a = (static_cast<unsigned long long>(x.val[1]) << 32) | x.val[0];
        out[i] = fits ? (pos ? static_cast<long>(a) : -static_cast<long>(a)) : 0;
        m = max(m, fits ? a : ULLONG_MAX);
    }
    for (int o = 16; o > 0; o >>= 1)
        m = max(m, __shfl_down_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0)
        atomicMax(max_abs, m);
}

template <class TB>
__global__ void long_matmul_kernel(const long* A, uint lda, const TB* B, uint ldb, Fr_t* C, uint ldc, uint M,
                                   uint K, uint N, bool accumulate) {
    __shared__ long As[IMM_BK][IMM_BM];
    __shared__ TB Bs[IMM_BK][IMM_BN];
    const int tx = threadIdx.x, ty = threadIdx.y, tid = ty * IMM_T + tx;
    const size_t row0 = static_cast<size_t>(blockIdx.y) * IMM_BM,
                 col0 = static_cast<size_t>(blockIdx.x) * IMM_BN;
    long acc[4][4] = {};
    for (uint k0 = 0; k0 < K; k0 += IMM_BK) {
        for (int i = tid; i < IMM_BM * IMM_BK; i += IMM_T * IMM_T) {
            const int r = i / IMM_BK, c = i % IMM_BK;
            const size_t gr = row0 + r, gc = k0 + c;
            As[c][r] = (gr < M && gc < K) ? A[gr * lda + gc] : 0;
        }
        for (int i = tid; i < IMM_BK * IMM_BN; i += IMM_T * IMM_T) {
            const int r = i / IMM_BN, c = i % IMM_BN;
            const size_t gr = k0 + r, gc = col0 + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * ldb + gc] : TB(0);
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < IMM_BK; ++k) {
            long a[4];
            TB b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                a[i] = As[k][ty + IMM_T * i];
                b[i] = Bs[k][tx + IMM_T * i];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    acc[i][j] += a[i] * static_cast<long>(b[j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const size_t r = row0 + ty + IMM_T * i, c = col0 + tx + IMM_T * j;
            if (r >= M || c >= N)
                continue;
            const Fr_t v = long_to_scalar(acc[i][j]);
            Fr_t& dst = C[r * ldc + c];
            dst = accumulate ? blstrs__scalar__Scalar_add(dst, v) : v;
        }
}

template <class TB>
void launch(const long* A, uint lda, const TB* B, uint ldb, Fr_t* C, uint ldc, uint M, uint K, uint N,
            bool accumulate) {
    const dim3 block(IMM_T, IMM_T), grid((N + IMM_BN - 1) / IMM_BN, (M + IMM_BM - 1) / IMM_BM);
    long_matmul_kernel<TB><<<grid, block>>>(A, lda, B, ldb, C, ldc, M, K, N, accumulate);
    cudaDeviceSynchronize();
}

unsigned long read_and_free(unsigned long long* d) {
    unsigned long long h;
    cudaMemcpy(&h, d, sizeof(h), cudaMemcpyDeviceToHost);
    cudaFree(d);
    return h;
}

unsigned long long* zeroed_counter() {
    unsigned long long* d;
    cudaMalloc((void**)&d, sizeof(*d));
    cudaMemset(d, 0, sizeof(*d));
    return d;
}

}  // namespace

unsigned long int_max_abs(const int* dev, size_t n) {
    auto* d = zeroed_counter();
    int_max_abs_kernel<<<1024, 256>>>(dev, n, d);
    return read_and_free(d);
}

unsigned long fr_slice_to_long(const Fr_t* A, uint rows, uint lda, uint k0, uint kc, long* out) {
    auto* d = zeroed_counter();
    fr_slice_to_long_kernel<<<1024, 256>>>(A, rows, lda, k0, kc, out, d);
    return read_and_free(d);
}

bool int_matmul_safe(unsigned long max_a, unsigned long max_b, uint K) {
    if (max_a == ULONG_MAX || max_b == ULONG_MAX)
        return false;
    const unsigned __int128 bound = static_cast<unsigned __int128>(max_a) * max_b * K;
    return bound < (static_cast<unsigned __int128>(1) << 63);
}

void long_matmul_to_fr(const long* A, uint lda, const int* B, uint ldb, Fr_t* C, uint ldc, uint M, uint K,
                       uint N, bool accumulate) {
    launch<int>(A, lda, B, ldb, C, ldc, M, K, N, accumulate);
}

void long_matmul_to_fr(const long* A, uint lda, const long* B, uint ldb, Fr_t* C, uint ldc, uint M, uint K,
                       uint N, bool accumulate) {
    launch<long>(A, lda, B, ldb, C, ldc, M, K, N, accumulate);
}

bool try_int_matmul(const Fr_t* x, const Fr_t* y, Fr_t* out, uint M, uint N, uint P) {
    if (!int_matmul_enabled())
        return false;
    long *xl, *yl;
    cudaMalloc((void**)&xl, sizeof(long) * static_cast<size_t>(M) * N);
    cudaMalloc((void**)&yl, sizeof(long) * static_cast<size_t>(N) * P);
    const unsigned long mx = fr_slice_to_long(x, M, N, 0, N, xl), my = fr_slice_to_long(y, N, P, 0, P, yl);
    const bool ok = int_matmul_safe(mx, my, N);
    if (ok)
        launch<long>(xl, N, yl, P, out, P, M, N, P, false);
    cudaFree(xl);
    cudaFree(yl);
    return ok;
}
