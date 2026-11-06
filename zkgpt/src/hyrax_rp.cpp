#undef NDEBUG
#include "hyrax_rp.hpp"
#include "config.hpp"
#include "gpu_hyrax_commit.hpp"   // NOVEL: BN254 GPU MSM for Hyrax row-commitments
#include <cmath>
#include <iostream>

#include <chrono>
#include <algorithm>
#include <thread>
#include <vector>

#include <omp.h>
using namespace std;
using namespace std::chrono;

using namespace mcl::bn;

const int MAX_MSM_LEN=1e4;
const int COMM_OPT_MAX=65536; //don't optimize if larger than this
const int logmax=16;  /// max number=2^18-1
const int block_num=4;  //2^80


G1 range_proof_perdersen_commit(G1* g,ll* f,int n,G1* W)
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



G1 range_proof_perdersen_commit_redundant(G1* g,Fr* f,ll* id,int n,int m) // compute MSM on g[i]*f[id[i]] for i=0~n-1, id[i] is in [0,m-1], n>> m
{
    G1 ret;
    ret.clear();
    G1* base2=new G1[m];
    for(int i=0;i<m;i++)
        base2[i].clear();
    for(int i=0;i<n;i++) // in [0,m-1]
    {
        base2[id[i]]+=g[i];
    }
    G1::mulVec(ret,base2,f,m);
    return ret;
}


G1 range_proof_perdersen_commit_classic(G1* g,Fr* f,int n)
{
    G1 ret;
    G1::mulVec(ret,g,f,n);
    return ret;
}

Fr range_proof_lagrange(Fr *r,int l,int k)
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


G1 range_proof_gen_gi(G1* g,int n)
{
    G1 base;
    base.setStr("1 0x2523648240000001ba344d80000000086121000000000013a700000000000012 0x1");
    
    for(int i=0;i<n;i++)
    {
        Fr tmp;
        tmp.setByCSPRNG();
        g[i]=base*tmp;
    }
    return base;
}


Pack bullet_reduce(G1 gamma, Fr*a,Fr* x,Fr y,G1*g,G1& G,int n,bool need_free) // length n
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
    gamma_minus1=G*x1a2+range_proof_perdersen_commit_classic(g+n/2,x,n/2);
    gamma_1=G*x2a1+range_proof_perdersen_commit_classic(g,x+n/2,n/2);
    Fr c,invc;
    c.setByCSPRNG();  // step3 V choose random c
    Fr::inv(invc,c);

    
    G1 gamma_prime=gamma_minus1*c*c+gamma_1*invc*invc+gamma;
    Fr* aprime=new Fr[n/2];

    for(int i=0;i<n/2;i++)
        aprime[i]=a[i]*invc+a[i+n/2]*c;
    G1* gprime=new G1[n/2];           
    
    #pragma omp parallel for
    for(int i=0;i<n/2;i++)
        gprime[i]=g[i]*invc+g[i+n/2]*c;
    

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
    
    return bullet_reduce(gamma_prime,aprime,xprime,yprime,gprime,G,n/2,true);
}   

bool range_proof_prove_dot_product(G1 comm_x, G1 comm_y, Fr* a, Fr* x,Fr y,G1*g , G1& G,int n)  // y= <a,x> , 
{
    G1 gamma=comm_x+comm_y;
    Pack p=bullet_reduce(gamma,a,x,y,g,G,n,false);
    return true;
    
}


void ll_commit_worker(G1*& Tk,G1*& g, ll*& w,int rownum,int colnum,G1*& W,int start,int end)
{
    for (int idx = start; idx < end; ++idx)
    {
            Tk[idx].clear();
            ll* f1=new ll[colnum];
            for(int i=0;i<colnum;i++)
                f1[i]=w[idx+i*rownum];
            Tk[idx]=range_proof_perdersen_commit(g,f1,colnum,W);
            delete []f1;
    }
}

static inline G1 range_proof_perdersen_commit_fast(
    G1* g,
    ll* f,
    int n,
    G1* W
)
{
    G1 ret;
    ret.clear();

    const int bucketN = COMM_OPT_MAX * block_num;
    const int ggN = logmax * block_num;

    static thread_local std::vector<unsigned char> used;
    static thread_local std::vector<int> touched;
    static thread_local std::vector<G1> gg;

    if ((int)used.size() < bucketN)
        used.assign(bucketN, 0);

    if ((int)gg.size() < ggN)
        gg.resize(ggN);

    touched.clear();
    touched.reserve(std::min<size_t>((size_t)bucketN, (size_t)n * block_num));

    std::vector<ll> bar(block_num);
    ll bar_t = 1;

    for (int i = 0; i < block_num; i++)
    {
        bar[i] = bar_t;
        bar_t <<= logmax;
    }

    const ll mask = COMM_OPT_MAX - 1;

    for (int i = 0; i < n; i++)
    {
        ll fi = f[i];

        if (fi == 0)
            continue;

        bool neg = fi < 0;
        ll tmp = neg ? -fi : fi;

        for (int j = 0; j < block_num; j++)
        {
            if (tmp < bar[j])
                break;

            ll fnow = (tmp >> (logmax * j)) & mask;
            int pos = (int)fnow + j * COMM_OPT_MAX;

            if (!used[pos])
            {
                used[pos] = 1;
                touched.push_back(pos);
            }

            if (neg)
                W[pos] -= g[i];
            else
                W[pos] += g[i];
        }
    }

    for (int j = 0; j < ggN; j++)
        gg[j].clear();

    for (int pos : touched)
    {
        int jj  = pos % COMM_OPT_MAX;
        int blk = pos / COMM_OPT_MAX;

        for (int k = 0; k < logmax; k++)
        {
            if (jj & (1LL << k))
                gg[k + logmax * blk] += W[pos];
        }

        W[pos].clear();
        used[pos] = 0;
    }

    for (int j = 0; j < ggN; j++)
    {
        if (j > 60)
        {
            G1 gd = gg[j] * (1LL << 48);
            ret += gd * (1LL << (j - 48));
        }
        else
        {
            ret += gg[j] * (1LL << j);
        }
    }

    return ret;
}
G1* range_proof_prover_commit(ll* w, G1* g, int l, int thread_n)
{
    // NOVEL: row-commitments via the BN254-reparameterized zkmod-gpu G1 MSM.
    if (!gkr_config.commit_cpu)
        return zkgpt_gpu::gpu_hyrax_commit_ll(w, g, l);
    int halfl  = l / 2;
    int rownum = 1 << halfl;
    int colnum = 1 << (l - halfl);

    G1* Tk = new G1[rownum];

    int actual_threads = std::min(thread_n, rownum);

    #pragma omp parallel num_threads(actual_threads)
    {
        std::vector<ll> f1(colnum);

        G1* W = new G1[COMM_OPT_MAX * block_num];

        for (int i = 0; i < COMM_OPT_MAX * block_num; i++)
            W[i].clear();

        #pragma omp for schedule(dynamic, 1)
        for (int idx = 0; idx < rownum; idx++)
        {
            Tk[idx].clear();

            for (int i = 0; i < colnum; i++)
                f1[i] = w[idx + i * rownum];

            Tk[idx] = range_proof_perdersen_commit_fast(
                g,
                f1.data(),
                colnum,
                W
            );
        }

        delete[] W;
    }

    return Tk;
}
void fr_commit_worker(G1*& Tk,G1*& g, ll*& w,Fr* f,int m,int rownum,int colnum,int start,int end)
{
    for (int idx = start; idx < end; ++idx)
    {
            Tk[idx].clear();
            ll* f1=new ll[colnum];
            for(int i=0;i<colnum;i++)
                f1[i]=w[idx+i*rownum];
            Tk[idx]=range_proof_perdersen_commit_redundant(g,f,f1,colnum,m);
            delete []f1;
    }
}
// Tk[idx] = Sum_i g[i] * f[w[idx + i*rownum]] -- an MSM whose scalars are GATHERED
// out of f[] by the indices in w[] (w has 2^l entries, each in [0,m-1]).
//
// Materialising the gather makes the cost independent of m and lets this reuse
// _fr_general, which has the GPU MSM short-circuit. The bucket path
// (range_proof_perdersen_commit_redundant) is unused here; it only wins when
// 2^c < colnum.
G1* range_proof_prover_commit_fr(ll* w, Fr* f,int m,G1* g, int l,int thread_n)
{
    (void)m;
    const long N = 1L << l;
    Fr* wf = new Fr[N];
    #pragma omp parallel for schedule(static) num_threads(thread_n)
    for (long i = 0; i < N; ++i)
        wf[i] = f[w[i]];
    G1* Tk = range_proof_prover_commit_fr_general(wf, g, l, thread_n);
    delete[] wf;
    return Tk;
}
void fr_commit_worker_general(G1*& Tk,G1*& g, Fr*& w,int rownum,int colnum,int start,int end)
{
    for (int idx = start; idx < end; ++idx)
    {
            Tk[idx].clear();
            Fr* f1=new Fr[colnum];
            for(int i=0;i<colnum;i++)
                f1[i]=w[idx+i*rownum];
            Tk[idx]=range_proof_perdersen_commit_classic(g,f1,colnum);
            delete []f1;
    }
}
G1* range_proof_prover_commit_fr_general(Fr* w, G1* g, int l,int thread_n) 
{
    // NOVEL: row-commitments via the BN254-reparameterized zkmod-gpu G1 MSM.
    if (!gkr_config.commit_cpu)
        return zkgpt_gpu::gpu_hyrax_commit(w, g, l);
    //w has 2^l length
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
        threads.emplace_back(fr_commit_worker_general,std::ref(Tk),std::ref(g),std::ref(w),rownum,colnum,start,end); 
    }
    for (auto& t : threads) t.join();
    return Tk;
}
Fr* range_proof_get_eq(Fr*r, int l)
{
    Fr* eq=new Fr[1<<l];
    eq[0]=1-r[0];
    eq[1]=r[0];
    for(int i=2;i<=l;i++)
    {
        auto temp = (1-r[i-1]);
        for(int j=(1<<i)-1;j>=0;j--)
        {
            if(j&(1<<(i-1)))
                eq[j]=eq[j-(1<<(i-1))]*r[i-1];
            else
                eq[j]=eq[j]*temp;
        }
    }
    return eq;
}
Fr range_proof_prover_evaluate(ll*ww ,Fr*r, int l)  
{
    Fr eval=0;
    Fr* lag=range_proof_get_eq(r,l);
    for(int k=0;k<(1<<l);k++)
        eval+=lag[k]*Fr(ww[k]);
    return eval;
}

void range_proof_open(ll*w,Fr*r,Fr eval,G1&G,G1*g,G1*comm,int l,int thread_num)
{
    int halfl=l/2;
    int rownum=(1<<halfl),colnum=(1<<(l-halfl));
    Fr*L=range_proof_get_eq(r,halfl);
    Fr*R=range_proof_get_eq(r+halfl,l-halfl);
    Fr* LT=new Fr[colnum];
    for(int i=0;i<colnum;i++)
        LT[i]=0;
    auto start3 = high_resolution_clock::now();
    #pragma omp parallel for schedule(static) num_threads(thread_num)
    for(int i=0;i<colnum;i++)
    for(int j=0;j<rownum;j++)
    {
        LT[i]+=L[j]*Fr(w[j+i*rownum]);   // mat mult  (1,row)*(row,col)=(1,col)
    }
    auto end3 = high_resolution_clock::now();
    auto duration_ms3 = duration_cast<milliseconds>(end3 - start3);

    G1 tprime=range_proof_perdersen_commit_classic(comm,L,rownum); //random combine comm
     auto start4 = high_resolution_clock::now();
    range_proof_prove_dot_product(tprime, G*eval, R, LT,eval,g , G,colnum);
    
        auto end4 = high_resolution_clock::now();
    auto duration_ms4 = duration_cast<milliseconds>(end4 - start4);
}
void range_proof_open(Fr*w,Fr*r,Fr eval,G1&G,G1*g,G1*comm,int l,int thread_num)
{
    int halfl=l/2;
    int rownum=(1<<halfl),colnum=(1<<(l-halfl));
    Fr*L=range_proof_get_eq(r,halfl);
    Fr*R=range_proof_get_eq(r+halfl,l-halfl);
    Fr* LT=new Fr[colnum];
    for(int i=0;i<colnum;i++)
        LT[i]=0;
    auto start3 = high_resolution_clock::now();
    #pragma omp parallel for schedule(static) num_threads(thread_num)
    for(int i=0;i<colnum;i++)
    {
        for(int j=0;j<rownum;j++)
        {
            LT[i]+=L[j]*w[j+i*rownum];   // mat mult  (1,row)*(row,col)=(1,col)
        }
    }
    auto end3 = high_resolution_clock::now();
    auto duration_ms3 = duration_cast<milliseconds>(end3 - start3);

    G1 tprime=range_proof_perdersen_commit_classic(comm,L,rownum); //random combine comm

    auto start4 = high_resolution_clock::now();
    range_proof_prove_dot_product(tprime, G*eval, R, LT,eval,g , G,colnum);
    auto end4 = high_resolution_clock::now();
    auto duration_ms4 = duration_cast<milliseconds>(end4 - start4);
}
