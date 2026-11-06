// sample_pmc: fine-grained (default 100 us) PMC sampler over a whole process tree. perf stat -I cannot go below
// ~1 ms, so this opens the counters itself (perf_event_open, pid=child, inherit=1) and polls them on a pinned core.
//   usage: sample_pmc <interval_us> <out.csv> <event_cfg_hex[,cfg_hex...]> -- cmd args...
// Output mimics `perf stat -x, -I`:  t_seconds,delta_count,,<event-config>,... one row per event per tick, t relative to exec.
// Only the child's tree is counted (inherit), so run the sampler NOT under a taskset that excludes the workload.
#define _GNU_SOURCE
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <sys/wait.h>
#include <unistd.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
static inline uint64_t now_ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC_RAW,&t); return t.tv_sec*1000000000ull+t.tv_nsec; }
int main(int argc,char**argv){
  if(argc<6){ fprintf(stderr,"usage: %s <interval_us> <out.csv> <cfg,...> -- cmd...\n",argv[0]); return 2; }
  uint64_t step=strtoull(argv[1],0,10)*1000ull; const char*out=argv[2];
  uint64_t cfg[8]; int ne=0; for(char*s=strtok(argv[3],",");s&&ne<8;s=strtok(0,",")) cfg[ne++]=strtoull(s,0,16);
  int ci=4; if(strcmp(argv[ci],"--")) { fprintf(stderr,"missing --\n"); return 2; } ci++;
  int go[2]; if(pipe(go)) return 1;
  pid_t pid=fork();
  if(!pid){ char b; close(go[1]); if(read(go[0],&b,1)<0) _exit(127); execvp(argv[ci],argv+ci); perror("exec"); _exit(127); }
  int fd[8];
  for(int i=0;i<ne;i++){
    struct perf_event_attr a; memset(&a,0,sizeof a); a.size=sizeof a; a.type=4; /* AMD ls/l2 raw via cpu PMU: perf's 'cpu' type */
    a.config=cfg[i]; a.disabled=1; a.inherit=1; a.exclude_guest=1; a.enable_on_exec=1;
    fd[i]=syscall(SYS_perf_event_open,&a,pid,-1,-1,0);
    if(fd[i]<0){ perror("perf_event_open"); kill(pid,9); return 1; }
  }
  cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(sched_getcpu(),&cs); sched_setaffinity(0,sizeof cs,&cs);   // stay on one core
  size_t cap=1<<20, n=0; uint64_t*ts=malloc(cap*8), *v=malloc(cap*8*ne);
  if(write(go[1],"g",1)<0) return 1; uint64_t t0=now_ns(), next=t0+step; int st=0; pid_t r=0;
  for(;;){
    while(now_ns()<next){}
    uint64_t t=now_ns(); next+=step; if(next<t) next=t+step;      // never queue a burst after a slow read
    if(n==cap){ cap*=2; ts=realloc(ts,cap*8); v=realloc(v,cap*8*ne); }
    for(int i=0;i<ne;i++){ uint64_t c[3]={0}; if(read(fd[i],c,sizeof c)<0) c[0]=0; v[n*ne+i]=c[0]; }
    ts[n++]=now_ns()-t0;                                          // stamp AFTER the reads: a slow inherit read is billed to its own interval
    if((n&1023)==0 && (r=waitpid(pid,&st,WNOHANG))==pid) break;
  }
  FILE*f=fopen(out,"w"); fprintf(f,"# sample_pmc interval_us=%s events=%d samples=%zu\n",argv[1],ne,n);
  for(size_t k=1;k<n;k++) for(int i=0;i<ne;i++)
    fprintf(f,"%.7f,%llu,,%llx,%llu,100.00,,\n",ts[k]/1e9,(unsigned long long)(v[k*ne+i]-v[(k-1)*ne+i]),(unsigned long long)cfg[i],(unsigned long long)(ts[k]-ts[k-1]));
  fclose(f);
  fprintf(stderr,"sample_pmc: %zu samples, mean period %.1f us\n",n,(ts[n-1]-ts[0])/1e3/(n-1));
  return WIFEXITED(st)?WEXITSTATUS(st):1;
}
