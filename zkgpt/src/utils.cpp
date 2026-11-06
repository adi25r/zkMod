// Created by 69029 on 3/9/2021

#include <cmath>
#include <iostream>
#include <thread>
#include <vector>
#include <algorithm>

#include "utils.hpp"

using std::cerr;
using std::cin;
using std::endl;
using std::string;

int ceilPow2BitLengthSigned(double n) {
    return (i8)ceil(log2(n));
}

int floorPow2BitLengthSigned(double n) {
    return (i8)floor(log2(n));
}

i8 ceilPow2BitLength(u32 n) {
    if (n == 0)
        return -1;
    // ceil(log2(n)) using integer ops avoids rounding errors from std::log
    --n;
    i8 l = 0;
    while (n) {
        n >>= 1U;
        ++l;
    }
    return l;
}

i8 ceilPow2BitLength64(i64 n) {
    if (n <= 0)
        return -1;
    i64 m = n - 1;
    i8 l = 0;
    while (m) {
        m >>= 1;
        ++l;
    }
    return l;
}

i8 floorPow2BitLength(u32 n) {
    if (n == 0)
        return -1;
    i8 l = -1;
    while (n) {
        n >>= 1U;
        ++l;
    }
    return l;
}

void initHalfTable(vector<F>& beta_f, vector<F>& beta_s, const vector<F>::const_iterator& r, const F& init,
                   u32 first_half, u32 second_half) {
    beta_f.at(0) = init;
    beta_s.at(0) = F_ONE;
    for (u32 i = 0; i < first_half; ++i) {
        for (u32 j = 0; j < (1ULL << i); ++j) {
            auto tmp = beta_f.at(j) * r[i];
            beta_f.at(j | (1ULL << i)) = tmp;
            beta_f.at(j) = beta_f[j] - tmp;
        }
    }
    for (u32 i = 0; i < second_half; ++i) {
        for (u32 j = 0; j < (1ULL << i); ++j) {
            auto tmp = beta_s[j] * r[(i + first_half)];
            beta_s[j | (1ULL << i)] = tmp;
            beta_s[j] = beta_s[j] - tmp;
        }
    }
}

void initBetaTable_worker(vector<F>& beta_g, const vector<F>& beta_f, const vector<F>& beta_s, int first_half,
                          u32 mask_fhalf, size_t start, size_t end) {
    for (size_t kp = start; kp < end; ++kp) {
        beta_g[kp] = beta_f[kp & mask_fhalf] * beta_s[kp >> first_half];
    }
}
void initBetaTable_worker2(vector<F>& beta_g, const vector<F>& beta_f, const vector<F>& beta_s,
                           int first_half, u32 mask_fhalf, size_t start, size_t end) {
    for (size_t kp = start; kp < end; ++kp) {
        beta_g[kp] += beta_f[kp & mask_fhalf] * beta_s[kp >> first_half];
    }
}
void initBetaTable(vector<F>& beta_g, u8 gLength, const vector<F>::const_iterator& r_0,
                   const vector<F>::const_iterator& r_1, const F& alpha, const F& beta, int thd) {
    u8 first_half = gLength >> 1, second_half = gLength - first_half;
    u32 mask_fhalf = (1ULL << first_half) - 1;

    vector<F> beta_f(1ULL << first_half), beta_s(1ULL << second_half);
    if (!beta.isZero()) {
        initHalfTable(beta_f, beta_s, r_1, beta, first_half, second_half);
        if (thd == 1 || gLength < 15) {
            for (size_t i = 0; i < (1ULL << gLength); ++i)
                beta_g[i] = beta_f[i & mask_fhalf] * beta_s[i >> first_half];
        } else {
            std::vector<std::thread> threads;
            threads.reserve(thd);
            const size_t total_work = size_t(1) << gLength;
            const size_t chunk = (total_work + thd - 1) / thd;
            for (int j = 0; j < thd; j++) {
                size_t start = size_t(j) * chunk;
                size_t end = std::min(total_work, start + chunk);
                if (start >= end)
                    break;
                threads.emplace_back(initBetaTable_worker, std::ref(beta_g), std::cref(beta_f),
                                     std::cref(beta_s), first_half, mask_fhalf, start, end);
            }
            for (auto& t : threads)
                t.join();
        }
    } else
        for (size_t i = 0; i < (1ULL << gLength); ++i)
            beta_g[i].clear();

    if (alpha.isZero())
        return;
    initHalfTable(beta_f, beta_s, r_0, alpha, first_half, second_half);
    if (thd == 1 || gLength < 15) {
        for (size_t i = 0; i < (1ULL << gLength); ++i)
            beta_g[i] += beta_f[i & mask_fhalf] * beta_s[i >> first_half];
    } else {
        std::vector<std::thread> threads;
        threads.reserve(thd);
        const size_t total_work = size_t(1) << gLength;
        const size_t chunk = (total_work + thd - 1) / thd;
        for (int j = 0; j < thd; j++) {
            size_t start = size_t(j) * chunk;
            size_t end = std::min(total_work, start + chunk);
            if (start >= end)
                break;
            threads.emplace_back(initBetaTable_worker2, std::ref(beta_g), std::cref(beta_f),
                                 std::cref(beta_s), first_half, mask_fhalf, start, end);
        }
        for (auto& t : threads)
            t.join();
    }
}

void initBetaTable(vector<F>& beta_g, u8 gLength, const vector<F>::const_iterator& r, const F& init,
                   int thd) {
    if (gLength == -1)
        return;
    int first_half = gLength >> 1, second_half = gLength - first_half;
    u32 mask_fhalf = (1ULL << first_half) - 1;
    vector<F> beta_f(1ULL << first_half), beta_s(1ULL << second_half);
    if (!init.isZero()) {
        initHalfTable(beta_f, beta_s, r, init, first_half, second_half);
        if (thd == 1 || gLength < 15) {
            for (size_t i = 0; i < (1ULL << gLength); ++i)
                beta_g[i] = beta_f[i & mask_fhalf] * beta_s[i >> first_half];
        } else {
            std::vector<std::thread> threads;
            threads.reserve(thd);
            const size_t total_work = size_t(1) << gLength;
            const size_t chunk = (total_work + thd - 1) / thd;
            for (int j = 0; j < thd; j++) {
                size_t start = size_t(j) * chunk;
                size_t end = std::min(total_work, start + chunk);
                if (start >= end)
                    break;
                threads.emplace_back(initBetaTable_worker, std::ref(beta_g), std::cref(beta_f),
                                     std::cref(beta_s), first_half, mask_fhalf, start, end);
            }
            for (auto& t : threads)
                t.join();
        }
    } else
        for (size_t i = 0; i < (1ULL << gLength); ++i)
            beta_g[i].clear();
}

bool zero_beta_skip_enabled() {
    static int on = getenv("GKR_NO_ZERO_BETA_SKIP") ? 0 : 1;
    return on != 0;
}

void assert_zero_beta(const vector<F>& t, const char* where, int layer) {
    static int on = getenv("GKR_ASSERT_ZERO_BETA") ? 1 : 0;
    if (!on)
        return;
    for (size_t i = 0; i < t.size(); ++i)
        if (!t[i].isZero()) {
            fprintf(stderr, "[zero-beta] VIOLATED at %s layer=%d idx=%zu of %zu\n", where, layer, i,
                    t.size());
            abort();
        }
}

bool check(long x, long y, long nx, long ny) {
    return 0 <= x && x < nx && 0 <= y && y < ny;
}

void initLayer(layer& circuit, long size, layerType ty) {
    circuit.size = circuit.zero_start_id = size;
    // 64-bit variant
    circuit.bit_length = ceilPow2BitLength64(size);
    circuit.ty = ty;
}

long sqr(long x) {
    return x * x;
}

double byte2KB(size_t x) {
    return x / 1024.0;
}

double byte2MB(size_t x) {
    return x / 1024.0 / 1024.0;
}

double byte2GB(size_t x) {
    return x / 1024.0 / 1024.0 / 1024.0;
}

long matIdx(long x, long y, long n) {  // row number , n: row len, y : column num
    assert(y < n);
    return x * n + y;
}

void field(const char* f, Fr x) {
    if (x.isNegative())
        cout << f << " -" << -x << endl;
    else
        cout << f << " " << x << endl;
}
long cubIdx(long x, long y, long z, long n, long m) {
    assert(y < n && z < m);
    return matIdx(matIdx(x, y, n), z, m);
}

long tesIdx(long w, long x, long y, long z, long n, long m, long l) {
    assert(x < n && y < m && z < l);
    return matIdx(cubIdx(w, x, y, n, m), z, l);
}

F getRootOfUnit(int n) {
    F res = -F_ONE;
    if (!n)
        return F_ONE;
    while (--n) {
        bool b = F::squareRoot(res, res);
        assert(b);
    }
    return res;
}
