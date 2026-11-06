// LD_PRELOAD shim: logs every hipMemcpy / hipMemcpyAsync (bytes, kind, call time) from the whole process tree.
// Forked children inherit the fd; each record is one atomic 24 B O_APPEND write.
// Record (32 B): u64 t0_ns (call entry), u64 t1_ns (call return), u64 bytes, u32 kind (hipMemcpyKind; 1=H2D 2=D2H 3=D2D), u32 pid<<1|async.
// Clock = CLOCK_MONOTONIC (same as rocprofv3). A synchronous hipMemcpy returns when the copy is done, so [t0,t1] is its real duration;
// an async copy returns immediately (t1~t0), so its duration is NOT known -- consumers must assume a link rate.
// Build: gcc -O2 -shared -fPIC memcpy_shim.c -o memcpy_shim.so -ldl      Use: MEMCPY_LOG=out.bin LD_PRELOAD=./memcpy_shim.so cmd
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
static int fd = -1;
static int (*r_memcpy)(void*, const void*, size_t, int);
static int (*r_async)(void*, const void*, size_t, int, void*);
struct rec { uint64_t t0, t1, n; uint32_t kind, pid; };
__attribute__((constructor)) static void init(void) {
  const char* p = getenv("MEMCPY_LOG");
  if (p) fd = open(p, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
  r_memcpy = dlsym(RTLD_NEXT, "hipMemcpy");
  r_async = dlsym(RTLD_NEXT, "hipMemcpyAsync");
}
static uint64_t now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec; }
static void log_it(uint64_t t0, size_t n, int kind, int async) {
  if (fd < 0) return;
  struct rec r = { t0, now(), n, (uint32_t)kind, ((uint32_t)getpid() << 1) | (uint32_t)async };
  (void)!write(fd, &r, sizeof r);
}
int hipMemcpy(void* d, const void* s, size_t n, int k) { uint64_t t0 = now(); int e = r_memcpy(d, s, n, k); log_it(t0, n, k, 0); return e; }
int hipMemcpyAsync(void* d, const void* s, size_t n, int k, void* st) { uint64_t t0 = now(); int e = r_async(d, s, n, k, st); log_it(t0, n, k, 1); return e; }
