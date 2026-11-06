//
// Created by 69029 on 3/9/2021.
//

#ifndef ZKCNN_UTILS_HPP
#define ZKCNN_UTILS_HPP

#include <circuit.h>
#include <omp.h>

int ceilPow2BitLengthSigned(double n);
int floorPow2BitLengthSigned(double n);

i8 ceilPow2BitLength(u32 n);
i8 floorPow2BitLength(u32 n);
// 64-bit sibling for values that can exceed u32's range -- e.g. a monolithic
// INPUT layer's total_in_size, which exceeds 2^32 at opt2b7 scale. Same algorithm
// as ceilPow2BitLength, just not width-limited.
i8 ceilPow2BitLength64(i64 n);
void field(const char* f,Fr x);



void initBetaTable(vector<F> &beta_g, u8 gLength, const vector<F>::const_iterator &r_0, const vector<F>::const_iterator &r_1,
              const F &alpha, const F &beta,int thd=32);





void initBetaTable(vector<F> &beta_g, u8 gLength, const vector<F>::const_iterator &r, const F &init,int thread=32);

// GKR_ASSERT_ZERO_BETA=1: initBetaTable zero-fills when beta is zero and early-returns
// when alpha is zero, so beta_g is identically zero exactly then. If that holds,
// skipping the gate loops is exact -- they only ever accumulate beta_g[g] * (...).
// Verify once over a full proof, then leave it off; the check is O(2^bit_length) per
// layer. No-op unless the env var is set.
void assert_zero_beta(const vector<F> &t, const char *where, int layer);

// GKR_NO_ZERO_BETA_SKIP=1 disables the all-zero-beta_g short-circuit, so the skip can
// be A/B'd back-to-back rather than across two runs minutes apart.
bool zero_beta_skip_enabled();

bool check(long x, long y, long nx, long ny);

long matIdx(long x, long y, long n);

long cubIdx(long x, long y, long z, long n, long m);

long tesIdx(long w, long x, long y, long z, long n, long m, long l);

void initLayer(layer &circuit, long size, layerType ty);

long sqr(long x);

double byte2KB(size_t x);

double byte2MB(size_t x);

double byte2GB(size_t x);

F getRootOfUnit(int n);

#endif //ZKCNN_UTILS_HPP
