#ifndef GPU_FIELD_HPP
#define GPU_FIELD_HPP

#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib>  // std::exit — HIP_CHECK aborts on allocation failure
#include <mcl/bn.hpp>
#include <mcl/bn.h>
#include "bls12_381.hpp"

typedef mcl::bn::Fr FieldElement;

#ifndef HIP_CHECK
// Exits 12 (ENOMEM) on allocation failure; other errors are non-fatal.
#define HIP_CHECK(expression)                                                                              \
    {                                                                                                      \
        const hipError_t status = expression;                                                              \
        if (status != hipSuccess) {                                                                        \
            std::cerr << "HIP error " << status << ": " << hipGetErrorString(status) << " at " << __FILE__ \
                      << ":" << __LINE__ << std::endl;                                                     \
            if (status == hipErrorOutOfMemory) {                                                           \
                std::cerr << "[fatal] GPU allocation failed — aborting (no "                               \
                             "fallback path exists)."                                                      \
                          << std::endl;                                                                    \
                std::exit(12);                                                                             \
            }                                                                                              \
        }                                                                                                  \
    }
#endif  // HIP_CHECK

#ifndef TO_FE_DEFINED
#define TO_FE_DEFINED
inline FieldElement to_FE(uint32_t val) {
    FieldElement r;
    r = val;
    return r;
}
#endif  // TO_FE_DEFINED

typedef blstrs__scalar__Scalar gpu_fe;

inline FieldElement get_R() {
    static FieldElement R;
    static bool init = false;
    if (!init) {
        static const uint32_t R_words[8] = {4294967218u, 973078527u,  4294967196u, 37945343u,
                                            4294967248u, 2730897151u, 2147483637u, 556508402u};
        mclBnFr_setLittleEndian((mclBnFr*)&R, (const uint8_t*)R_words, 32);
        init = true;
    }
    return R;
}

inline FieldElement get_R_inv() {
    static FieldElement Ri;
    static bool init = false;
    if (!init) {
        FieldElement::inv(Ri, get_R());
        init = true;
    }
    return Ri;
}

// The bridge is a REINTERPRET
static_assert(sizeof(gpu_fe) == 32, "device scalar must be 32 bytes for the memcpy bridge");

inline gpu_fe fe_to_gpu(const FieldElement& fe) {
    gpu_fe s;
    __builtin_memcpy(&s, &fe, sizeof(s));
    return s;
}

inline FieldElement fe_from_gpu(const gpu_fe& s) {
    FieldElement m;
    __builtin_memcpy(&m, &s, sizeof(s));
    return m;
}

inline __device__ gpu_fe device_add(gpu_fe a, gpu_fe b) {
    return blstrs__scalar__Scalar_add(a, b);
}

inline __device__ gpu_fe device_sub(gpu_fe a, gpu_fe b) {
    return blstrs__scalar__Scalar_sub(a, b);
}

inline __device__ gpu_fe device_mul(gpu_fe a, gpu_fe b) {
    return blstrs__scalar__Scalar_mul(a, b);
}

inline __host__ __device__ int pow3(int n) {
    int r = 1;
    for (int i = 0; i < n; ++i)
        r *= 3;
    return r;
}

#endif  // GPU_FIELD_HPP
