// Created by 69029 on 3/9/2021

#ifndef ZKCNN_UTILS_HPP
#define ZKCNN_UTILS_HPP

#include <circuit.h>
#include <omp.h>

int ceilPow2BitLengthSigned(double n);
int floorPow2BitLengthSigned(double n);

i8 ceilPow2BitLength(u32 n);
i8 floorPow2BitLength(u32 n);
// 64-bit sibling for values that can exceed u32's range
i8 ceilPow2BitLength64(i64 n);
void field(const char* f, Fr x);

void initBetaTable(vector<F>& beta_g, u8 gLength, const vector<F>::const_iterator& r_0,
                   const vector<F>::const_iterator& r_1, const F& alpha, const F& beta, int thd = 32);

void initBetaTable(vector<F>& beta_g, u8 gLength, const vector<F>::const_iterator& r, const F& init,
                   int thread = 32);

// GKR_ASSERT_ZERO_BETA=1
void assert_zero_beta(const vector<F>& t, const char* where, int layer);

// GKR_NO_ZERO_BETA_SKIP=1 disables the all-zero-beta_g short-circuit
bool zero_beta_skip_enabled();

bool check(long x, long y, long nx, long ny);

long matIdx(long x, long y, long n);

long cubIdx(long x, long y, long z, long n, long m);

long tesIdx(long w, long x, long y, long z, long n, long m, long l);

void initLayer(layer& circuit, long size, layerType ty);

long sqr(long x);

double byte2KB(size_t x);

double byte2MB(size_t x);

double byte2GB(size_t x);

F getRootOfUnit(int n);

#endif  // ZKCNN_UTILS_HPP
