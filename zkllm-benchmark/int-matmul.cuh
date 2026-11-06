#ifndef INT_MATMUL_CUH
#define INT_MATMUL_CUH

// Exact forward matmuls on the integers the field elements encode. Activations and weights are
// small signed integers stored as field elements, so when max|A| * max|B| * K < 2^63 the product
// is computed in int64 and mapped back to the field; the result is the canonical field element
// the Montgomery path produces, bit for bit. Chunks that could overflow use the field kernel.
// ZKLLM_INT_MATMUL=0 selects the field path everywhere.

#include "fr-tensor.cuh"

bool int_matmul_enabled();

// max |x| over n ints (as unsigned, so INT_MIN is exact).
unsigned long int_max_abs(const int* dev, size_t n);

// out[r * kc + c] = A[r * lda + k0 + c] as int64 for r < rows, c < kc. Returns max |value|, or
// ULONG_MAX if some value is outside int64.
unsigned long fr_slice_to_long(const Fr_t* A, uint rows, uint lda, uint k0, uint kc, long* out);

// Whether a K-term dot product of values bounded by max_a, max_b stays inside int64.
bool int_matmul_safe(unsigned long max_a, unsigned long max_b, uint K);

// C[r * ldc + c] (+)= sum_k A[r * lda + k] * B[k * ldb + c], exact, for an M x K by K x N product.
void long_matmul_to_fr(const long* A, uint lda, const int* B, uint ldb, Fr_t* C, uint ldc, uint M, uint K, uint N, bool accumulate);
void long_matmul_to_fr(const long* A, uint lda, const long* B, uint ldb, Fr_t* C, uint ldc, uint M, uint K, uint N, bool accumulate);

// out = x (M x N) * y (N x P) through int64 if both fit and the bound allows; false otherwise
// (out untouched).
bool try_int_matmul(const Fr_t* x, const Fr_t* y, Fr_t* out, uint M, uint N, uint P);

#endif // INT_MATMUL_CUH
