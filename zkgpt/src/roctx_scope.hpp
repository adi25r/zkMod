#pragma once
// roctx_scope — RAII ROCTx ranges, so rocprofv3 can attribute kernels to prover phases
#include <cstdio>

#ifdef ZKGPT_ROCTX
#include <rocprofiler-sdk-roctx/roctx.h>

namespace zkgpt_roctx {
struct Scope {
    explicit Scope(const char* name) { roctxRangePush(name); }
    ~Scope() { roctxRangePop(); }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
};
struct ScopeF {
    char buf[128];
    template <class... A>
    explicit ScopeF(const char* fmt, A... a) {
        snprintf(buf, sizeof(buf), fmt, a...);
        roctxRangePush(buf);
    }
    ~ScopeF() { roctxRangePop(); }
    ScopeF(const ScopeF&) = delete;
    ScopeF& operator=(const ScopeF&) = delete;
};
}  // namespace zkgpt_roctx

#define ZKGPT_ROCTX_CAT2(a, b) a##b
#define ZKGPT_ROCTX_CAT(a, b) ZKGPT_ROCTX_CAT2(a, b)
#define ZKGPT_ROCTX_SCOPE(name) ::zkgpt_roctx::Scope ZKGPT_ROCTX_CAT(_roctx_, __LINE__)(name)
#define ZKGPT_ROCTX_SCOPEF(...) ::zkgpt_roctx::ScopeF ZKGPT_ROCTX_CAT(_roctx_, __LINE__)(__VA_ARGS__)

#else
#define ZKGPT_ROCTX_SCOPE(name) \
    do {                        \
    } while (0)
#define ZKGPT_ROCTX_SCOPEF(...) \
    do {                        \
    } while (0)
#endif
