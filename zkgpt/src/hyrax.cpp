#include "hyrax.hpp"
#include <omp.h>
#include <cstring>
#include "gpu_bfeval.hpp"
#include "phase_acct.hpp"
#include <cstdlib>
#include "timer.hpp"
#include "config.hpp"
#include "gpu_hyrax_commit.hpp"   // NOVEL: BN254 GPU MSM for Hyrax row-commitments
#include <cmath>
#include <thread>
#include <vector>
#include <mutex>
#include <iostream>
#include <chrono>
#include <algorithm>
using namespace std;
using namespace mcl::bn;

const int MAX_MSM_LEN=1e4;
const int COMM_OPT_MAX=65536; //don't optimize if larger than this
const int logmax=16;  /// max number=2^18-1
const int block_num=5;  //2^80


G1 perdersen_commit(G1* g,ll* f,int n,G1* W)
{
    G1 ret;
    ret.clear();
    
    bool *used=new bool[COMM_OPT_MAX*block_num];
    memset(used,0,sizeof(bool)*COMM_OPT_MAX*block_num);
    ll bar[10];
    ll bar_t=1;
    for(int i=0;i<8;i++)
    {
        bar[i]=bar_t;
        bar_t<<=logmax;
    }
    for(int i=0;i<n;i++)
    {
            if(f[i]==0)
                continue;
            
            if(f[i]<0)
            {
                ll tmp=-f[i];
                for(int j=0;j<block_num;j++)
                {
                    if(tmp<bar[j])
                        break;
                    ll fnow=(tmp>>(logmax*j))&65535;
                    W[fnow+(j<<logmax)]-=g[i];
                    used[fnow+(j<<logmax)]=1;
                }
            }
            else
            {
                ll tmp=f[i];
                for(int j=0;j<block_num;j++)
                {
                    if(tmp<bar[j])
                        break;
                    ll fnow=(tmp>>(logmax*j))&65535;
                    W[fnow+(j<<logmax)]+=g[i];
                    used[fnow+(j<<logmax)]=1;
                }
            }
    }
    G1 gg[logmax*block_num];
    for(int j=0;j<logmax*block_num;j++)
        gg[j].clear();
    for(int j=0;j<COMM_OPT_MAX*block_num;j++)
    {
        if(used[j])
        {
            int jj=j%COMM_OPT_MAX;
            int blk=j/COMM_OPT_MAX;
            for(int k=0;k<logmax;k++)
            {
                if(jj&(1<<k))
                    gg[k+logmax*blk]+=W[j];
            }
            W[j].clear();
            used[j]=0;            
        }
    }
    for(int j=0;j<logmax*block_num;j++)
    {
        if(j>60)
        {
            G1 gd=gg[j]*(1ll<<48);
            ret+=gd*(1ll<<(j-48)); //split
        }
        else
            ret+=gg[j]*(1ll<<j);
    }
    delete []used;
    return ret;
}


G1 perdersen_commit(G1* g,int* f,int n,G1* W)
{
    G1 ret;
    ret.clear();
    bool *used=new bool[COMM_OPT_MAX];
    memset(used,0,sizeof(bool)*COMM_OPT_MAX);
    for(int i=0;i<n;i++)
    {
            if(f[i]==0)
                continue;
            
            if(f[i]<0)
            {
                W[-f[i]]-=g[i];
                used[-f[i]]=1;
                assert(-f[i]<COMM_OPT_MAX);
            }
            else
            {
                W[f[i]]+=g[i];
                used[f[i]]=1;
                assert(f[i]<COMM_OPT_MAX);
            }
    }
    const int logn=log2(COMM_OPT_MAX)+1;
    G1 gg[40];
    for(int j=0;j<logn;j++)
        gg[j].clear();
    for(int j=1;j<COMM_OPT_MAX;j++)
    {
        if(used[j])
        {
            for(int k=0;k<logn;k++)
            {
                if(j&(1<<k))
                    gg[k]+=W[j];
            }
            W[j].clear();
            used[j]=0;            
        }
    }
    for(int j=0;j<logn;j++)
        ret+=gg[j]*(1<<j);


    delete []used;
    return ret;
}

G1 perdersen_commit(G1* g,Fr* f,int n)
{
    G1 ret;
    G1::mulVec(ret,g,f,n);
    return ret;
}

Fr lagrange(Fr *r,int l,int k)
{
    assert(k>=0 && k<(1<<l));
    Fr ret=1;
    for(int i=0;i<l;i++)
    {
        if(k&(1<<i))
            ret*=r[i];
        else
            ret*=1-r[i];
    }
    return ret;
}
void brute_force_compute_LR(Fr* L,Fr* R,Fr* r,int l)
{
    zkgpt_acct::Scope _a(zkgpt_acct::A_BF_LR, 1ULL<<l);
    int halfl=l/2,c=l-halfl;
    for(int k=0;k<(1<<c);k++)
        L[k]=lagrange(r,c,k);
    for(int k=0;k<(1<<halfl);k++)
        R[k]=lagrange(r+c,halfl,k);
}

// Reference arm for the GPU port. O(2^l * l).
// Split across nthreads with per-thread partial sums (field addition is exact, so the
// result is identical to the serial loop).
static Fr bfeval_orig(Fr* w,Fr* r,int l,int nthreads=1)
{
    const long n = 1L << l;
    if (nthreads <= 1 || l < 12) {
        Fr ret=0;
        for(long k=0;k<n;k++)
            ret+=lagrange(r,l,(int)k)*w[k];
        return ret;
    }
    std::vector<Fr> part(nthreads);
    #pragma omp parallel num_threads(nthreads)
    {
        Fr acc=0;
        #pragma omp for schedule(static)
        for(long k=0;k<n;k++)
            acc+=lagrange(r,l,(int)k)*w[k];
        part[omp_get_thread_num()]=acc;
    }
    Fr ret=0;
    for(auto &p : part) ret+=p;
    return ret;
}

Fr brute_force_compute_eval(Fr* w,Fr* r,int l,int nthreads)
{
    zkgpt_acct::Scope _a(zkgpt_acct::A_BF_EVAL, 1ULL<<l);
    static const bool gpu = getenv("GKR_BFEVAL_GPU") != nullptr;
    const bool check = getenv("GKR_BFEVAL_CHECK") != nullptr;
    if (!gpu && !check) return bfeval_orig(w, r, l, nthreads);
    if (check && !gpu) {   // multithreaded vs serial reference
        Fr got = bfeval_orig(w, r, l, nthreads), want = bfeval_orig(w, r, l);
        if (got != want) { fprintf(stderr, "[bfeval-check] MT MISMATCH l=%d\n", l); abort(); }
        return got;
    }

    // Factored form: lagrange(r,l,k) = L[k & maskc] * R[k >> c], so the device
    // loop is two multiplies per element instead of l.
    const int halfl = l / 2, c = l - halfl;
    const long N = 1L << l;
    std::vector<Fr> L(1L << c), R(1L << halfl);
    brute_force_compute_LR(L.data(), R.data(), r, l);

    Fr got = gpu ? zkgpt_gpu::gpu_bfeval(w, L.data(), R.data(), N, c, halfl)
                 : bfeval_orig(w, r, l);

    if (check && gpu) {
        Fr want = bfeval_orig(w, r, l);
        if (got != want) {
            fprintf(stderr, "[bfeval-check] MISMATCH l=%d\n", l);
            abort();
        }
    }
    return got;
}


G1 compute_Tprime(int l,Fr* R,G1* Tk) 
{
    //w has 2^l length
    int halfl=l/2;
    int rownum=(1<<halfl),colnum=(1<<(l-halfl));
    G1 ret=perdersen_commit(Tk,R,rownum);
    return ret;
}

G1 compute_RT_singlethread(Fr*w ,Fr*R,int l,G1*g,Fr*& ret) // L is row number length
{
    int halfl=l/2;
    int rownum=(1<<halfl),colnum=(1<<(l-halfl));
    Fr* res=new Fr[colnum];
    for(int i=0;i<colnum;i++)
        res[i]=0;
    timer TT;
    TT.start();
    for(int j=0;j<colnum;j++)
    for(int i=0;i<rownum;i++)
    {
        if(!w[j+i*colnum].isZero())
            res[j]+=R[i]*w[j+i*colnum];  // mat mult  (1,row)*(row,col)=(1,col)
    }
    TT.stop();
    G1 comm=perdersen_commit(g,res,colnum);
    ret=res;
    return comm;
}
G1 compute_RT(Fr *w, Fr *R, int l, G1 *g, Fr *&ret) // L is row number length
{
    int halfl = l / 2;
    int rownum = (1 << halfl), colnum = (1 << (l - halfl));
    Fr *res = new Fr[colnum];
    for (int i = 0; i < colnum; i++)
        res[i] = 0;

    auto start = std::chrono::high_resolution_clock::now();

    // Define the number of threads
    // omp_get_max_threads, not hardware_concurrency: a split child's share of the thread pool
    // (main_gkr_split child_threads_set). Unchanged in the monolith: with OMP_NUM_THREADS unset
    // both are the affinity mask's CPU count.
    const int num_threads = omp_get_max_threads();
    std::vector<std::thread> threads;
    std::mutex res_mutex;

    auto worker = [&](int start_col, int end_col) {
        Fr *local_res = new Fr[colnum]; // Thread-local result storage
        for (int i = start_col; i < end_col; i++)
            local_res[i] = 0;

        for (int j = start_col; j < end_col; j++) {
            for (int i = 0; i < rownum; i++) {
                if (!w[j + i * colnum].isZero()) {
                    local_res[j] += R[i] * w[j + i * colnum];
                }
            }
        }

        // Merge local results into global result
        std::lock_guard<std::mutex> guard(res_mutex);
        for (int j = start_col; j < end_col; j++) {
            res[j] += local_res[j];
        }

        delete[] local_res; // Free thread-local storage
    };

    // Split columns into chunks for each thread
    int chunk_size = (colnum + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; t++) {
        int start_col = t * chunk_size;
        int end_col = std::min(start_col + chunk_size, colnum);
        if (start_col < colnum) { // Avoid starting threads if no work is left
            threads.emplace_back(worker, start_col, end_col);
        }
    }

    // Join threads
    for (auto &th : threads) {
        if (th.joinable())
            th.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    // Commit the result
    G1 comm = perdersen_commit(g, res, colnum);
    ret = res;
    return comm;
}

G1 gen_gi(G1* g,int n)
{
    G1 base;
    // `base` must be initialised: it fixes every Hyrax generator and the base point G
    // the opening equation gamma == g*x + G*y is written against. The sibling
    // range_proof_gen_gi carries the same setStr.
    base.setStr("1 0x2523648240000001ba344d80000000086121000000000013a700000000000012 0x1");
    for(int i=0;i<n;i++)
    {
        Fr tmp;
        tmp.setByCSPRNG();
        g[i]=base*tmp;
    }
    return base;
}

Pack bullet_reduce(G1 gamma, Fr*a,G1*g,int n,G1& G,Fr* x,Fr y,bool need_free, timer *blt_vtime) // length n
{
    if(n==1)
    {
        Pack p(gamma,a[0],g[0],x[0],y);
        return p;
    }
    
    //step2  prover fold
    G1 gamma_minus1,gamma_1;
    Fr x1a2=0,x2a1=0;
    for(int i=0;i<n/2;i++)
    {
        x1a2+=x[i]*a[n/2+i];
        x2a1+=x[n/2+i]*a[i];
    }
    gamma_minus1=G*x1a2+perdersen_commit(g+n/2,x,n/2);
    gamma_1=G*x2a1+perdersen_commit(g,x+n/2,n/2);
    Fr c,invc;
    c.setByCSPRNG();  // step3 V choose random c
    //prover verifier both comp
    blt_vtime->start();
    Fr::inv(invc,c);
    G1 gamma_prime=gamma_minus1*c*c+gamma_1*invc*invc+gamma;
    Fr* aprime=new Fr[n/2];       
    for(int i=0;i<n/2;i++)
        aprime[i]=a[i]*invc+a[i+n/2]*c;
    G1* gprime=new G1[n/2];        
    if(n<2048)
    {   
        for(int i=0;i<n/2;i++)
            gprime[i]=g[i]*invc+g[i+n/2]*c;
    }
    else  
    {
        int num_threads=std::min(16, omp_get_max_threads());
        // ceil: with a floor chunk the last (n/2) % num_threads generators were never
        // folded whenever the thread count is not a power of two (3 or 5 threads per
        // partition), and the IPA base check then rejected an honest opening
        int chunk_size = (n/2 + num_threads - 1) / num_threads;
        std::vector<std::thread> threads1;
        auto worker1 = [&](int start_id, int end_id) 
        {
            for(int i=start_id;i<end_id;i++)
                gprime[i]=g[i]*invc+g[i+n/2]*c;
        };
        for (int t = 0; t < num_threads; t++) 
        {
            int start_col = t * chunk_size;
            int end_col = min(start_col + chunk_size, n/2);
            if (start_col >= end_col) break;
            threads1.emplace_back(worker1, start_col, end_col);
        }
        for (auto &th : threads1) 
        {
            if (th.joinable())
                th.join();
        }
    }
    blt_vtime->stop();


    //prover single compute
    Fr* xprime=new Fr[n/2];         
    Fr yprime;
    for(int i=0;i<n/2;i++)
        xprime[i]=c*x[i]+invc*x[i+n/2];
    yprime=c*c*x1a2+invc*invc*x2a1+y;

    if(need_free)
    {
        delete []a;
        delete []g;
        delete []x;
    }
    
    return bullet_reduce(gamma_prime,aprime,gprime,n/2,G,xprime,yprime,true, blt_vtime);
}   

void prove_dot_product(G1 comm_x, G1 comm_y, Fr* a, G1*g ,G1& G,Fr* x,Fr y,int n, timer *blt_vtime)  // y= <a,x> , 
{
    G1 gamma=comm_x+comm_y;
    Pack p=bullet_reduce(gamma,a,g,n,G,x,y, false, blt_vtime);
    // Bullet-fold base case: this check binds the commitment to the opened data (a
    // discarded `p` would accept forged openings). It relies on gen_gi returning an
    // initialised base point.
    if (!(p.y==p.x*p.a) || !(p.gamma==p.g*p.x+G*p.y))
    {
        std::cout<<"RESULT: REJECT\n";
        std::cerr<<"[gkr] VERIFICATION REJECTED (stage=open, IPA base check)\n";
        exit(2);
    }
}


void ll_commit_worker(G1*& Tk,G1*& g, ll*& row,int colnum,int start,int end)
{
    // Bucket scratch is per worker and allocated HERE, not by the caller: a serial
    // new+memset of 45 MiB per thread on the calling thread made commit time grow
    // linearly with threads. calloc maps zero pages lazily, so only buckets a row
    // touches are faulted in, in parallel. perdersen_commit leaves every bucket it
    // used cleared, so W stays all-zero across rows.
    G1 *W = static_cast<G1 *>(std::calloc(size_t(COMM_OPT_MAX) * block_num, sizeof(G1)));
    if (!W) throw std::bad_alloc();
    for (int idx = start; idx < end; ++idx)
    {
            Tk[idx]=perdersen_commit(g,row+idx*colnum,colnum,W);
    }
    std::free(W);
}
G1* prover_commit(ll* w, G1* g, int l,int thread_n) //compute Tk, int version with pippenger
{
    // Small-scalar MSM on GPU: it consumes the __int128 witness directly and runs only
    // ceil(bits/c) windows; the generic 256-bit MSM was about 2x slower here.
    if (!gkr_config.commit_cpu)
        return zkgpt_gpu::gpu_hyrax_commit_ll_rowmajor(w, g, l);
    int halfl=l/2;
    int rownum=(1<<halfl),colnum=(1<<(l-halfl));
    G1 *Tk=new G1[rownum];
    vector<thread> threads;
    threads.reserve(thread_n);
    int chunk = (rownum + thread_n - 1) / thread_n;
    for(int i=0;i<thread_n;i++)
    {
        int start = i * chunk;
        int end = min(rownum, start + chunk);
        if (start >= end) break;
        threads.emplace_back(ll_commit_worker,std::ref(Tk),std::ref(g),std::ref(w),colnum,start,end);
    }
    for (auto& t : threads) t.join();
    return Tk;
}



Fr prover_evaluate(Fr*w ,Fr*r,G1& G,G1* g, Fr*L,Fr*R,int l)  // nlogn brute force 
{
    int halfl=l/2;
    int rownum=(1<<halfl),colnum=(1<<(l-halfl));
    timer t(true);
    t.start();
    brute_force_compute_LR(L,R,r,l);
    Fr eval=brute_force_compute_eval(w,r,l);
    t.stop("eval total ",true,false);
    return eval;
}
namespace hyrax
{
pair<timer_range,timer_range> open(Fr*w,Fr*r,Fr eval,G1&G,G1*g,Fr*L,Fr*R,G1*tk,int l)
{
    zkgpt_acct::Scope _a(zkgpt_acct::A_HYRAX_OPEN, 1ULL<<l);
    timer prover_timer, verifier_timer, blt_vtime;
    int halfl=l/2;
    int rownum=(1<<halfl),colnum=(1<<(l-halfl));
    prover_timer.start();
    Fr* RT=new Fr[colnum];
    { zkgpt_acct::Scope _r(zkgpt_acct::A_RT, 1ULL<<l);
      compute_RT(w,R,l,g,RT); }
    prover_timer.stop();
    prover_timer.start();
    verifier_timer.start();
    G1 tprime; { zkgpt_acct::Scope _t(zkgpt_acct::A_TPRIME, 1ULL<<(l/2));
      tprime=compute_Tprime(l,R,tk); }
    prover_timer.stop();
    verifier_timer.stop();
    prover_timer.start();
    { zkgpt_acct::Scope _f(zkgpt_acct::A_FOLD, (unsigned long long)colnum);
      prove_dot_product(tprime, G*eval, L, g , G,RT,eval,colnum, &blt_vtime); }
    prover_timer.stop();
    return make_pair(prover_timer.range(),combine_ranges({verifier_timer.range(), blt_vtime.range()}));
}
}
