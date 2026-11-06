// uvm_probe.cpp — Experiment 3 (experiments.tex): is M_GPU a hard ceiling or a
// soft (page-migration) constraint on this GPU?
//
// Streams a read-modify-write kernel over a buffer of <gb> GiB and reports the
// effective bandwidth (GB/s, counting read+write). Sweeping <gb> from below to
// above physical VRAM exposes the behaviour past the VRAM limit:
//   * hard cliff  -> allocation fails / kernel faults  => M_GPU is a hard ceiling
//   * smooth ramp -> bandwidth degrades gracefully     => M_GPU can be a soft penalty
//   * flat-slow   -> always PCIe-bound (never in VRAM)  => UVM buys nothing here
//
// Allocation mode (argv[3]):
//   0 = hipMalloc        (device-only; baseline, OOM expected past VRAM = the cliff)
//   1 = hipMallocManaged (migration policy left to runtime / HSA_XNACK)
//   2 = hipMallocManaged + hipMemAdvise(PreferredLocation=device) + prefetch
//       (the only config that *should* pull pages resident into VRAM)
//
// Build BOTH ways (the runner does this):
//   hipcc -O2 uvm_probe.cpp -o uvm_probe                         # xnack- (default)
//   hipcc -O2 --offload-arch=gfx90a:xnack+ uvm_probe.cpp -o uvm_probe_xnack
// Mode 1/2 are only meaningful with the xnack+ binary under HSA_XNACK=1.
//
// Output line (parseable): UVM mode=<m> req_gib=<g> bw_gbps=<b> status=<s> phase_s=...
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>

using clk = std::chrono::high_resolution_clock;
static double secs(clk::time_point a){ return std::chrono::duration<double>(clk::now()-a).count(); }

__global__ void rmw(double* a, size_t n){
  size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x*blockDim.x;
  for(; i<n; i+=stride) a[i] = a[i]*1.0000001 + 1.0;
}

int main(int argc, char** argv){
  double gb   = argc>1 ? atof(argv[1]) : 32.0;
  int    iters= argc>2 ? atoi(argv[2]) : 5;
  int    mode = argc>3 ? atoi(argv[3]) : 0;
  size_t bytes = (size_t)(gb * (1ull<<30));
  size_t n = bytes / sizeof(double);
  int dev=0; (void)hipGetDevice(&dev);
  size_t freeB=0,totB=0; (void)hipMemGetInfo(&freeB,&totB);

  // Header always prints first so a fault/timeout is attributable to a size.
  printf("UVM mode=%d req_gib=%.1f vram_free_gb=%.1f ", mode, gb, freeB/1e9);
  fflush(stdout);

  double *a=nullptr; hipError_t e;
  auto t_alloc = clk::now();
  e = (mode==0) ? hipMalloc(&a, bytes) : hipMallocManaged(&a, bytes);
  if(e!=hipSuccess){ printf("status=ALLOC_FAIL(%s)\n", hipGetErrorString(e)); return 2; }
  double alloc_s = secs(t_alloc);

  double advise_s=0, prefetch_s=0;
  if(mode==2){
    auto t=clk::now();
    hipMemAdvise(a, bytes, hipMemAdviseSetPreferredLocation, dev);
    hipMemAdvise(a, bytes, hipMemAdviseSetAccessedBy, dev);
    advise_s = secs(t);
    t=clk::now();
    e = hipMemPrefetchAsync(a, bytes, dev);
    if(e!=hipSuccess){ printf("status=PREFETCH_FAIL(%s) ", hipGetErrorString(e)); }
    hipDeviceSynchronize();
    prefetch_s = secs(t);
  }

  int blk=256; size_t grid=(n+blk-1)/blk; if(grid>65535) grid=65535;
  // warm-up (forces first-touch migration); timed separately
  auto t_warm=clk::now();
  rmw<<<grid,blk>>>(a,n);
  e=hipDeviceSynchronize();
  if(e!=hipSuccess){ printf("status=WARMUP_FAULT(%s)\n", hipGetErrorString(e)); hipFree(a); return 4; }
  double warm_s = secs(t_warm);

  auto t0=clk::now();
  for(int it=0; it<iters; ++it) rmw<<<grid,blk>>>(a,n);
  e=hipDeviceSynchronize();
  if(e!=hipSuccess){ printf("status=RUN_FAULT(%s)\n", hipGetErrorString(e)); hipFree(a); return 5; }
  double dt=secs(t0);
  double bw = (double)bytes * iters * 2.0 / dt / 1e9;   // *2 = read+write
  printf("status=OK bw_gbps=%.1f run_s=%.3f warm_s=%.3f alloc_s=%.3f advise_s=%.3f prefetch_s=%.3f\n",
         bw, dt, warm_s, alloc_s, advise_s, prefetch_s);
  hipFree(a);
  return 0;
}
