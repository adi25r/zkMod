#include "alloc_trace.hpp"
void *operator new(std::size_t sz) {
    void *p = std::malloc(sz ? sz : 1);
    if (!p) throw std::bad_alloc();
    alloctrace::note(p, sz);
    return p;
}
void *operator new[](std::size_t sz) { return ::operator new(sz); }
void operator delete(void *p) noexcept { if (p) { alloctrace::drop(p); std::free(p); } }
void operator delete[](void *p) noexcept { ::operator delete(p); }
void operator delete(void *p, std::size_t) noexcept { ::operator delete(p); }
void operator delete[](void *p, std::size_t) noexcept { ::operator delete(p); }
