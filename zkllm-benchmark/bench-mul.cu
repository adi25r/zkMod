// Throughput of the PTX Montgomery multiply vs the portable CIOS one (dependent chains, 1M threads).
#include "fr-tensor.cuh"
#include <cstdio>
#include <vector>
using namespace std;

DEVICE Fr_t blstrs__scalar__Scalar_mul_nvidia(Fr_t a, Fr_t b);
DEVICE Fr_t blstrs__scalar__Scalar_mul_default(Fr_t a, Fr_t b);

#define BENCH(NAME, FN)                                                                  \
    __global__ void NAME(const Fr_t* in, Fr_t* out, int n, int iters)                    \
    {                                                                                    \
        const int i = blockIdx.x * blockDim.x + threadIdx.x;                             \
        if (i >= n) return;                                                              \
        Fr_t x = in[i], y = in[(i + 1) % n];                                             \
        for (int k = 0; k < iters; ++k) x = FN(x, y);                                    \
        out[i] = x;                                                                      \
    }
BENCH(bench_ptx, blstrs__scalar__Scalar_mul_nvidia)
BENCH(bench_portable, blstrs__scalar__Scalar_mul_default)

int main()
{
    const int n = 1 << 20, iters = 256;
    auto h = random_vec(n);
    Fr_t *din, *dout;
    cudaMalloc(&din, n * sizeof(Fr_t));
    cudaMalloc(&dout, n * sizeof(Fr_t));
    cudaMemcpy(din, h.data(), n * sizeof(Fr_t), cudaMemcpyHostToDevice);
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    for (int rep = 0; rep < 3; ++rep) {
        float ms_ptx, ms_port;
        cudaEventRecord(a);
        bench_ptx<<<n / 256, 256>>>(din, dout, n, iters);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        cudaEventElapsedTime(&ms_ptx, a, b);
        cudaEventRecord(a);
        bench_portable<<<n / 256, 256>>>(din, dout, n, iters);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        cudaEventElapsedTime(&ms_port, a, b);
        const double muls = (double)n * iters;
        printf("BENCH_MUL rep=%d ptx=%.1f ms (%.2f Gmul/s)  portable=%.1f ms (%.2f Gmul/s)  slowdown=%.2fx\n", rep,
               ms_ptx, muls / ms_ptx / 1e6, ms_port, muls / ms_port / 1e6, ms_port / ms_ptx);
    }
    return 0;
}
