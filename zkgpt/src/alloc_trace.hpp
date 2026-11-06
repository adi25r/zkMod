#pragma once
// GKR_ALLOCTRACE
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <new>
#include <map>
#include <mutex>
#include <string>
#include <execinfo.h>

namespace alloctrace {

inline std::size_t threshold() {
    static std::size_t t = [] {
        const char* e = getenv("GKR_ALLOCTRACE_GIB");
        double g = e ? atof(e) : 1.0;
        return (std::size_t)(g * 1073741824.0);
    }();
    return t;
}
inline bool enabled() {
    static bool e = getenv("GKR_ALLOCTRACE") != nullptr;
    return e;
}
struct Rec {
    std::size_t size;
    std::string bt;
};
inline std::map<void*, Rec>& table() {
    static std::map<void*, Rec> t;
    return t;
}
inline std::mutex& lock() {
    static std::mutex m;
    return m;
}
// re-entrancy guard: backtrace_symbols itself allocates
inline bool& in_hook() {
    static thread_local bool b = false;
    return b;
}

inline std::string capture() {
    void* bp[24];
    int n = backtrace(bp, 24);
    char** sym = backtrace_symbols(bp, n);
    std::string s;
    if (sym) {
        for (int i = 2; i < n && i < 10; ++i) {
            s += "\n      ";
            s += sym[i];
        }
        free(sym);
    }
    return s;
}

inline void note(void* p, std::size_t sz) {
    if (!enabled() || sz < threshold() || in_hook())
        return;
    in_hook() = true;
    {
        std::lock_guard<std::mutex> g(lock());
        table()[p] = Rec{sz, capture()};
    }
    in_hook() = false;
}
inline void drop(void* p) {
    if (!enabled() || in_hook())
        return;
    in_hook() = true;
    {
        std::lock_guard<std::mutex> g(lock());
        table().erase(p);
    }
    in_hook() = false;
}
inline void report(const char* tag, long long idx) {
    if (!enabled())
        return;
    std::lock_guard<std::mutex> g(lock());
    double tot = 0;
    for (auto& kv : table())
        tot += kv.second.size;
    fprintf(stderr, "[alloctrace] %s idx=%lld outstanding=%zu total_gib=%.2f\n", tag, idx, table().size(),
            tot / 1073741824.0);
    for (auto& kv : table())
        fprintf(stderr, "  %10.2f GiB  ptr=%p%s\n", kv.second.size / 1073741824.0, kv.first,
                kv.second.bt.c_str());
    fflush(stderr);
}
}  // namespace alloctrace

void* operator new(std::size_t sz);
void* operator new[](std::size_t sz);
void operator delete(void* p) noexcept;
void operator delete[](void* p) noexcept;
void operator delete(void* p, std::size_t) noexcept;
void operator delete[](void* p, std::size_t) noexcept;
