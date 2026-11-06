#include "range_prover.hpp"
#include <string>
#include "config.hpp"
#include "gpu_range_prover.hpp"
#include "gpu_deg3_sumcheck.hpp"
#include "gpu_deg3_staged.hpp"  // Blendy-staged deg3 (k = range_k stage count)
#include "gpu_hyrax_open.hpp"
#include <chrono>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>

#include <utils.hpp>
using namespace mcl::bn;
using std::unique_ptr;

namespace {
double g_t_setup = 0;   // counts + (r+f),(r+t) fraction build (CPU)
double g_t_invert = 0;  // batch inversion 1/(r+f),1/(r+t) — logup reciprocals
double g_t_commit = 0;  // Hyrax commits (f,t,c,g_comm,h_comm) — always run
double g_t_deg3 = 0;    // degree-3 well-formedness sumchecks (CPU)
double g_t_deg1 = 0;    // degree-1 reciprocal-sum sumchecks (GPU or CPU)
double g_t_open = 0;    // Hyrax opens
using clk = std::chrono::high_resolution_clock;
inline double secs_since(clk::time_point t0) {
    return std::chrono::duration<double>(clk::now() - t0).count();
}
}  // namespace

// Used for timing
static inline void gkr_phase_barrier() {
    hipDeviceSynchronize();
}

namespace {

// c must divide range_size and stay under the int shift width
constexpr int kMaxChunkBits = 30;
inline int kChunkBitsCfg() {
    const int c = gkr_config.range_c > 0 ? gkr_config.range_c : 9;
    if (c > kMaxChunkBits)
        throw std::invalid_argument("range_c (--c) = " + std::to_string(c) + " exceeds the maximum of " +
                                    std::to_string(kMaxChunkBits) +
                                    ": the lookup domain is built as "
                                    "(1 << c) in an int. Legal values that divide 54: 1,2,3,6,9,18,27.");
    return c;
}
using unsigned_ll = unsigned __int128;

int lg2(int x) {
    if (x <= 0) {
        throw std::invalid_argument("lg2 expects a positive value");
    }
    int res = 0;
    while (x >>= 1)
        ++res;
    return res;
}

int next_power_of_2(int n) {
    if (n <= 1) {
        return 1;
    }
    int power = 1;
    while (power < n) {
        power <<= 1;
    }
    return power;
}

int normalize_query_size(int query_size) {
    if (query_size <= 0) {
        throw std::invalid_argument("query_size must be positive");
    }
    return next_power_of_2(query_size);
}

unsigned_ll full_width_random(std::mt19937_64& rng) {
    return (static_cast<unsigned_ll>(rng()) << 64) | static_cast<unsigned_ll>(rng());
}

unsigned_ll range_mask(int range_size) {
    constexpr int kInputBits = static_cast<int>(sizeof(unsigned_ll) * 8);
    if (range_size < 0 || range_size > kInputBits) {
        throw std::invalid_argument("range_size exceeds supported input width");
    }
    if (range_size == 0) {
        return 0;
    }
    if (range_size == kInputBits) {
        return ~static_cast<unsigned_ll>(0);
    }
    return (static_cast<unsigned_ll>(1) << range_size) - 1;
}

}  // namespace

Fr poly_eval(Fr x0, Fr x1, Fr x2, Fr x3, Fr u)  // lagrange poly for degree 3
{
    Fr y = 1 / Fr(6) *
           ((-x0) * (u - 1) * (u - 2) * (u - 3) + 3 * x1 * u * (u - 2) * (u - 3) -
            3 * x2 * u * (u - 1) * (u - 3) + x3 * u * (u - 1) * (u - 2));
    return y;
}

range_prover::~range_prover() {
    release_allocations();
}

range_prover::SC_Return range_prover::sumcheck_deg1(int l, Fr* f, Fr S)  // sum_i f_i=S
{
    // P send V sum S
    Fr* ran = new Fr[l];
    Fr* current_f = f;
    bool owns_current_f = false;
    for (int i = l; i >= 1; i--)  // round i
    {
        Fr sum0 = 0, sum1 = 0;
#pragma omp parallel for schedule(static) num_threads(thread_num)
        for (int j = 0; j < (1 << i); j++) {
            if ((j & 1) == 0)
                sum0 += current_f[j];
            else {
                sum1 += current_f[j];
            }
        }
        // send poly: sum0,sum1
        Fr new_chlg;
        new_chlg.setByCSPRNG();
        ran[l - i] = new_chlg;
        // V send new_chlg
        S = new_chlg * (sum1 - sum0) + sum0;  // update target sum
        Fr* new_f = new Fr[1 << (i - 1)];
        Fr sum0p = 0, sum1p = 0;
#pragma omp parallel for schedule(static) num_threads(thread_num)
        for (int j = 0; j < (1 << (i - 1)); j++)
            new_f[j] = (1 - new_chlg) * current_f[j * 2] + new_chlg * current_f[j * 2 + 1];
        if (owns_current_f) {
            delete[] current_f;
        }
        current_f = new_f;
        owns_current_f = true;
    }
    SC_Return s;
    s.random = ran;
    s.claim_f = current_f[0];
    if (owns_current_f) {
        delete[] current_f;
    }
    return s;
}

range_prover::SC_Return range_prover::sumcheck_deg3(int l, Fr* r, Fr* f, Fr* g,
                                                    Fr S)  // sum_i eq(r,i) f_i g_i=S
{
    Fr* lag = range_proof_get_eq(r, l);
    // P send V sum S
    Fr* ran = new Fr[l];
    Fr *S0 = new Fr[1 << l], *S1 = new Fr[1 << l], *S2 = new Fr[1 << l], *S3 = new Fr[1 << l];
    Fr* current_f = f;
    Fr* current_g = g;
    bool owns_current_f = false;
    bool owns_current_g = false;
    for (int i = l; i >= 1; i--)  // round i
    {
        memset(S0, 0, sizeof(Fr) * (1 << i));
        memset(S1, 0, sizeof(Fr) * (1 << i));
        memset(S2, 0, sizeof(Fr) * (1 << i));
        memset(S3, 0, sizeof(Fr) * (1 << i));

        Fr sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;

#pragma omp parallel for schedule(static) num_threads(thread_num)
        for (int j = 0; j < (1 << i); j += 2) {
            S0[j >> 1] = lag[j] * current_f[j] * current_g[j];
            S1[j >> 1] = lag[j + 1] * current_f[j + 1] * current_g[j + 1];
            S2[j >> 1] = (lag[j + 1] + lag[j + 1] - lag[j]) *
                         (current_f[j + 1] + current_f[j + 1] - current_f[j]) *
                         (current_g[j + 1] + current_g[j + 1] - current_g[j]);
            S3[j >> 1] =
                (lag[j + 1] + lag[j + 1] + lag[j + 1] - lag[j] - lag[j]) *
                (current_f[j + 1] + current_f[j + 1] + current_f[j + 1] - current_f[j] - current_f[j]) *
                (current_g[j + 1] + current_g[j + 1] + current_g[j + 1] - current_g[j] - current_g[j]);
        }

        if (i < 8) {
            for (int j = 0; j < (1 << (i - 1)); j++) {
                sum0 += S0[j];
                sum1 += S1[j];
                sum2 += S2[j];
                sum3 += S3[j];
            }
        } else {
            Fr s0[8], s1[8], s2[8], s3[8];
            memset(s0, 0, sizeof(Fr) * 8);
            memset(s1, 0, sizeof(Fr) * 8);
            memset(s2, 0, sizeof(Fr) * 8);
            memset(s3, 0, sizeof(Fr) * 8);
            for (int k = 0; k < (1 << 3); k++) {
                for (int j = 0; j < (1 << (i - 1 - 3)); j++) {
                    s0[k] += S0[(k << (i - 1 - 3)) + j];
                    s1[k] += S1[(k << (i - 1 - 3)) + j];
                    s2[k] += S2[(k << (i - 1 - 3)) + j];
                    s3[k] += S3[(k << (i - 1 - 3)) + j];
                }
            }
            sum0 = s0[0] + s0[1] + s0[2] + s0[3] + s0[4] + s0[5] + s0[6] + s0[7];
            sum1 = s1[0] + s1[1] + s1[2] + s1[3] + s1[4] + s1[5] + s1[6] + s1[7];
            sum2 = s2[0] + s2[1] + s2[2] + s2[3] + s2[4] + s2[5] + s2[6] + s2[7];
            sum3 = s3[0] + s3[1] + s3[2] + s3[3] + s3[4] + s3[5] + s3[6] + s3[7];
        }
        assert(sum0 + sum1 == S);
        // send poly: sum0,sum1,sum2,sum3
        Fr new_chlg;
        new_chlg.setByCSPRNG();
        ran[l - i] = new_chlg;
        // V send new_chlg
        S = poly_eval(sum0, sum1, sum2, sum3, new_chlg);  // update target sum
        Fr* new_lag = new Fr[1 << (i - 1)];
        Fr* new_f = new Fr[1 << (i - 1)];
        Fr* new_g = new Fr[1 << (i - 1)];
        Fr sum0p = 0, sum1p = 0;
#pragma omp parallel for schedule(static) num_threads(thread_num)
        for (int j = 0; j < (1 << (i - 1)); j++) {
            new_lag[j] = lag[j * 2] + new_chlg * (lag[j * 2 + 1] - lag[j * 2]);
            new_f[j] = current_f[j * 2] + new_chlg * (current_f[j * 2 + 1] - current_f[j * 2]);
            new_g[j] = current_g[j * 2] + new_chlg * (current_g[j * 2 + 1] - current_g[j * 2]);
        }
        delete[] lag;
        lag = new_lag;
        if (owns_current_f) {
            delete[] current_f;
        }
        if (owns_current_g) {
            delete[] current_g;
        }
        current_f = new_f;
        current_g = new_g;
        owns_current_f = true;
        owns_current_g = true;
    }
    SC_Return s;
    s.random = ran;
    s.claim_f = current_f[0];
    s.claim_g = current_g[0];
    delete[] lag;
    if (owns_current_f) {
        delete[] current_f;
    }
    if (owns_current_g) {
        delete[] current_g;
    }
    delete[] S0;
    delete[] S1;
    delete[] S2;
    delete[] S3;
    return s;
}

void Constraint::prepare() {
    static std::mt19937_64 rng(std::random_device{}());
    if (inputs == nullptr) {
        throw std::runtime_error("Constraint input buffer is not initialized");
    }
    if (query_size < 0 || query_size > inputs_capacity) {
        throw std::runtime_error("Constraint query_size exceeds shared input buffer capacity");
    }

    const unsigned_ll mask = range_mask(range_size);
    for (int i = 0; i < query_size; i++) {
        inputs[i] = static_cast<ll>(full_width_random(rng) & mask);
    }
}

void range_prover::init() {
    build();
    if (initialized_) {
        return;
    }

    release_allocations();

    int max_query_size = 0;
    for (const auto& op : ops) {
        for (const auto& constraint : op.constraints) {
            max_query_size = std::max(max_query_size, constraint.query_size);
        }
    }

    inputs_capacity_ = max_query_size;
    const int max_domain = std::max(inputs_capacity_, 1 << kChunkBitsCfg());
    const int max_log = ceilPow2BitLength(static_cast<u32>(max_domain));
    generator_count_ = 1 << (max_log - max_log / 2);
    r_count_ = max_log;

    if (inputs_capacity_ > 0) {
        inputs = new ll[inputs_capacity_];
    }
    if (generator_count_ > 0) {
        g = new G1[generator_count_];
        GG = range_proof_gen_gi(g, generator_count_);
    }
    if (r_count_ > 0) {
        r = new Fr[r_count_];
    }

    for (auto& op : ops) {
        for (auto& constraint : op.constraints) {
            constraint.inputs = inputs;
            constraint.inputs_capacity = inputs_capacity_;
        }
    }

    initialized_ = true;
}

void range_prover::release_allocations() {
    delete[] g;
    delete[] r;
    delete[] inputs;
    g = nullptr;
    r = nullptr;
    inputs = nullptr;
    generator_count_ = 0;
    r_count_ = 0;
    inputs_capacity_ = 0;
    initialized_ = false;
}

void range_prover::push_back(NonlinearOpType op_type,
                             const std::vector<std::pair<int, int>>& constraint_params) {
    OP op;
    op.op_type = op_type;
    for (const auto& param : constraint_params) {
        if (param.first <= 0) {
            throw std::invalid_argument("Constraint query_size must be positive");
        }
        if (param.second <= 0 || param.second % kChunkBitsCfg() != 0) {
            throw std::invalid_argument(std::string("Constraint range_size must be a positive multiple of ") +
                                        std::to_string(kChunkBitsCfg()));
        }
        Constraint c;
        c.query_size = param.first;
        c.range_size = param.second;
        op.constraints.push_back(c);
    }
    ops.push_back(op);
}
void range_prover::range_prove(ll* x, int range, int m, int thread_num, int k_stages) {
    if (x == nullptr) {
        throw std::invalid_argument("range_prove requires a non-null input buffer");
    }
    const int kChunkBits = kChunkBitsCfg();
    const int kChunkDomainSize = 1 << kChunkBits;
    if (range <= 0 || range % kChunkBits != 0) {
        throw std::invalid_argument(
            std::string("range_prove expects range_size to be a positive multiple of ") +
            std::to_string(kChunkBits));
    }

    const int domain_size = normalize_query_size(m);
    const int chunk_count = range / kChunkBits;
    const ll chunk_mask = static_cast<ll>(kChunkDomainSize - 1);

    printf("range: %d\n", range);
    ll** data = new ll*[chunk_count];
    for (int i = 0; i < chunk_count; i++)
        data[i] = new ll[domain_size]();
    for (int i = 0; i < chunk_count; i++) {
        for (int j = 0; j < m; j++) {
            data[i][j] = (x[j] >> (kChunkBits * i)) & chunk_mask;
        }
    }
    ll* t = new ll[kChunkDomainSize];
    for (int i = 0; i < kChunkDomainSize; i++) {
        t[i] = i;
    }
    for (int i = 0; i < chunk_count; i++) {
        logup(data[i], t, domain_size, kChunkDomainSize, thread_num, k_stages);
    }

    for (int i = 0; i < chunk_count; ++i) {
        delete[] data[i];
    }
    delete[] data;
    delete[] t;
}

void range_prover::logup(ll* f, ll* t, int m, int n, int thread, int k_stages) {
    auto _t = clk::now();
    Fr* G = new Fr[m];
    Fr* F = new Fr[m];
    Fr* Hp = new Fr[n];
    Fr* H = new Fr[n];
    ll* c = new ll[n];
    memset(c, 0, sizeof(ll) * n);
    for (int i = 0; i < m; i++)
        c[f[i]]++;

    Fr r;
    r.setByCSPRNG();

    for (int i = 0; i < m; i++)
        F[i] = r + Fr(f[i]);
    for (int i = 0; i < n; i++)
        Hp[i] = r + Fr(t[i]);

    g_t_setup += secs_since(_t);
    auto _ti = clk::now();
    invVec(G, F, m);   // 1/(r+fi)
    invVec(H, Hp, n);  // 1/(r+ti)
    g_t_invert += secs_since(_ti);
    _t = clk::now();
    Fr* Hinv = new Fr[n];
    std::copy(H, H + n, Hinv);
    Fr s1 = 0, s2 = 0;
    for (int i = 0; i < m; i++)
        s1 += G[i];
    for (int i = 0; i < n; i++) {
        H[i] = H[i] * Fr(c[i]);
        s2 += H[i];
    }
    g_t_setup += secs_since(_t);

    _t = clk::now();
    const bool _mprof = getenv("GKR_MSM_PROF") != nullptr;
    auto _mnow = [] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    double _a = _mprof ? _mnow() : 0.0;

    G1* f_comm = range_proof_prover_commit(f, g, lg2(m), thread);
    G1* t_comm = range_proof_prover_commit(t, g, lg2(n), thread);
    G1* c_comm = range_proof_prover_commit(c, g, lg2(n), thread);
    double _b = _mprof ? _mnow() : 0.0;

    double _c2 = _mprof ? _mnow() : 0.0;

    G1* g_comm = range_proof_prover_commit_fr(f, Hinv, n, g, lg2(m), thread);
    double _d = _mprof ? _mnow() : 0.0;

    G1* h_comm = range_proof_prover_commit_fr_general(H, g, lg2(n), thread);
    if (_mprof) {
        const double _e = _mnow();
        fprintf(stderr,
                "[rcommit] m=%d n=%d | 3x_commit_ll=%.3fs invert_loop=%.3fs "
                "commit_fr(CPU-only)=%.3fs commit_fr_general=%.3fs\n",
                m, n, _b - _a, _c2 - _b, _d - _c2, _e - _d);
    }
    gkr_phase_barrier();
    g_t_commit += secs_since(_t);

    Fr sum = 0;
    for (int i = 0; i < m; i++)
        sum += G[i];
    Fr* rp1 = new Fr[lg2(n)];
    for (int i = 0; i < lg2(n); i++)
        rp1[i].setByCSPRNG();  // verifier challenge

    Fr c_eva = range_proof_prover_evaluate(c, rp1, lg2(n));

    // degree-3 well-formedness sumchecks
    _t = clk::now();
    Fr* rp2 = new Fr[lg2(m)];
    for (int i = 0; i < lg2(m); i++)
        rp2[i].setByCSPRNG();  // verifier challenge
    range_prover::SC_Return ret1, ret2;
    if (!gkr_config.deg3_cpu) {
        std::vector<Fr> rd1, rd2;
        Fr cf1, cg1, cf2, cg2;
        zkgpt_gpu::gpu_sumcheck_deg3_staged(lg2(n), rp1, H, Hp, c_eva, k_stages, rd1, cf1, cg1);
        zkgpt_gpu::gpu_sumcheck_deg3_staged(lg2(m), rp2, G, F, Fr(1), k_stages, rd2, cf2, cg2);
        Fr* ran1 = new Fr[lg2(n)];
        for (int i = 0; i < lg2(n); ++i)
            ran1[i] = rd1[i];
        Fr* ran2 = new Fr[lg2(m)];
        for (int i = 0; i < lg2(m); ++i)
            ran2[i] = rd2[i];
        ret1.random = ran1;
        ret1.claim_f = cf1;
        ret1.claim_g = cg1;
        ret2.random = ran2;
        ret2.claim_f = cf2;
        ret2.claim_g = cg2;
    } else {
        ret1 = sumcheck_deg3(lg2(n), rp1, H, Hp, c_eva);
        ret2 = sumcheck_deg3(lg2(m), rp2, G, F, 1);
    }
    gkr_phase_barrier();  // deg-3 sumcheck kernels finished
    g_t_deg3 += secs_since(_t);

    range_prover::SC_Return ret3, ret4;
    _t = clk::now();
    if (!gkr_config.range_cpu) {
        std::vector<Fr> r3, r4;
        Fr claim3 = zkgpt_gpu::gpu_sumcheck_deg1(lg2(m), G, sum, k_stages, r3);  // reciprocal sum, M=m
        Fr claim4 = zkgpt_gpu::gpu_sumcheck_deg1(lg2(n), H, sum, k_stages, r4);  // reciprocal sum, N=n
        Fr* ran3 = new Fr[lg2(m)];
        for (int i = 0; i < lg2(m); i++)
            ran3[i] = r3[i];
        Fr* ran4 = new Fr[lg2(n)];
        for (int i = 0; i < lg2(n); i++)
            ran4[i] = r4[i];
        ret3.random = ran3;
        ret3.claim_f = claim3;
        ret4.random = ran4;
        ret4.claim_f = claim4;
    } else {
        ret3 = sumcheck_deg1(lg2(m), G, sum);
        ret4 = sumcheck_deg1(lg2(n), H, sum);
    }
    gkr_phase_barrier();  // deg-1 reciprocal-sum sumcheck kernels finished
    g_t_deg1 += secs_since(_t);

    // Hyrax open phase
    static const bool open_gpu_env = getenv("GKR_OPEN_GPU") != nullptr;
    _t = clk::now();
    if (open_gpu_env && !gkr_config.open_cpu) {
        zkgpt_gpu::gpu_range_proof_open(c, rp1, c_eva, GG, g, c_comm, lg2(n));
        zkgpt_gpu::gpu_range_proof_open((const Fr*)H, ret1.random, ret1.claim_f, GG, g, h_comm, lg2(n));
        zkgpt_gpu::gpu_range_proof_open(t, ret1.random, ret1.claim_g - r, GG, g, t_comm, lg2(n));
        zkgpt_gpu::gpu_range_proof_open((const Fr*)G, ret2.random, ret2.claim_f, GG, g, g_comm, lg2(m));
        zkgpt_gpu::gpu_range_proof_open(f, ret2.random, ret2.claim_g - r, GG, g, f_comm, lg2(m));
        zkgpt_gpu::gpu_range_proof_open((const Fr*)G, ret3.random, ret3.claim_f, GG, g, g_comm, lg2(m));
        zkgpt_gpu::gpu_range_proof_open((const Fr*)H, ret4.random, ret4.claim_f, GG, g, h_comm, lg2(n));
    } else {
        range_proof_open(c, rp1, c_eva, GG, g, c_comm, lg2(n), thread);
        range_proof_open(H, ret1.random, ret1.claim_f, GG, g, h_comm, lg2(n), thread);
        range_proof_open(t, ret1.random, ret1.claim_g - r, GG, g, t_comm, lg2(n), thread);
        range_proof_open(G, ret2.random, ret2.claim_f, GG, g, g_comm, lg2(m), thread);
        range_proof_open(f, ret2.random, ret2.claim_g - r, GG, g, f_comm, lg2(m), thread);
        range_proof_open(G, ret3.random, ret3.claim_f, GG, g, g_comm, lg2(m), thread);
        range_proof_open(H, ret4.random, ret4.claim_f, GG, g, h_comm, lg2(n), thread);
    }
    gkr_phase_barrier();  // open MSM (bullet-fold) kernels finished
    g_t_open += secs_since(_t);

    delete[] G;
    delete[] F;
    delete[] Hp;
    delete[] H;
    delete[] c;
    delete[] Hinv;
    delete[] rp1;
    delete[] rp2;
    delete[] ret1.random;
    delete[] ret2.random;
    delete[] ret3.random;
    delete[] ret4.random;
    delete[] f_comm;
    delete[] t_comm;
    delete[] c_comm;
    delete[] g_comm;
    delete[] h_comm;
}

double range_prover::prove() {
    build();
    init();
    g_t_setup = g_t_invert = g_t_commit = g_t_deg3 = g_t_deg1 = g_t_open = 0;
    prove_timer.clear();
    prove_timer.start();
    for (auto& op : ops) {
        for (auto& constraint : op.constraints) {
            cout << "start prepare inputs" << endl;
            constraint.prepare();
            cout << "start range prove" << " query_size " << constraint.query_size << endl;
            cout << "Threads:" << thread_num << endl;
            timer constraint_timer(true);
            constraint_timer.start();
            range_prove(constraint.inputs, constraint.range_size, constraint.query_size, thread_num,
                        gkr_config.range_k);
            cout << "end range prove" << endl;
            constraint_timer.stop();
            cout << "time: " << constraint_timer.elapse_sec() << endl;
        }
    }
    prove_timer.stop();
    cout << "RANGE_BREAKDOWN"
         << " setup_s=" << g_t_setup << " invert_s=" << g_t_invert << " commit_s=" << g_t_commit
         << " deg3_s=" << g_t_deg3 << " deg1_s=" << g_t_deg1 << " open_s=" << g_t_open
         << " total_s=" << prove_timer.elapse_sec() << "\n";
    cerr << "[range] breakdown (s): setup=" << g_t_setup << " invert=" << g_t_invert
         << " commit=" << g_t_commit << " deg3=" << g_t_deg3 << " deg1(sumcheck)=" << g_t_deg1
         << " open=" << g_t_open << " | total=" << prove_timer.elapse_sec() << "\n";
    return prove_timer.elapse_sec();
}

void range_prover::build() {
    if (built_) {
        return;
    }

    ops.clear();
    int query_size = 0;

    if (!merged) {
        // layernorm
        query_size += (3 * seq_len + 2 * seq_len * AttnDim);

        // attetion
        query_size += (3 * HeadNum * seq_len * (seq_len + 1) / 2 + seq_len * AttnDim);

        // layernorms
        query_size += (3 * seq_len + 2 * seq_len * AttnDim);

        // linear (GELU)
        query_size += (2 * seq_len * LinearDim);

        query_size = normalize_query_size(query_size * LayerNum);
        push_back(NonlinearOpType::NonLinear, {{query_size, gkr_config.range_size}});
    } else {
        // layernorm
        query_size += (seq_len + seq_len * AttnDim);

        // attention
        query_size += (HeadNum * seq_len * (seq_len + 1) / 2 + seq_len * AttnDim);

        // layernorms
        query_size += (seq_len + seq_len * AttnDim);

        // linear (GELU)
        query_size += (seq_len * LinearDim);

        query_size = normalize_query_size(query_size * LayerNum);
        push_back(NonlinearOpType::NonLinear, {{query_size, gkr_config.range_size}});
    }

    built_ = true;
}
