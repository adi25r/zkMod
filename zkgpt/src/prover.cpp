#include "prover.hpp"
#include <cstring>
#include "gkr_gpu.hpp"
#include "config.hpp"          // gkr_config.commit_cpu gates the device commit path
// Unconditional, matching hyrax.cpp's include of gpu_hyrax_commit.hpp: GKR_GPU_ENABLED
// is not defined by this build, so guarding the device path on it would compile the
// path out.
#include "gpu_fr_scalars.hpp"   // NOVEL: Fr -> signed-128 scalar conversion on device
#include <hip/hip_runtime.h>
#include <chrono>
#include <iostream>
#include <utils.hpp>
#include <algorithm>
#include <stdexcept>
#include <thread>

using namespace mcl::bn;
using std::unique_ptr;

LassoInitTimers g_lasso_init_timers;
namespace {
using li_clk = std::chrono::high_resolution_clock;
inline double li_secs(li_clk::time_point t0) {
    return std::chrono::duration<double>(li_clk::now() - t0).count();
}
}

// Race-free parallel gate accumulation (u-partition).
// The accumulate does `mult_array[idx][gate.u] += ...`, so threads may run
// concurrently only over DISJOINT u. This re-partitions a gate range into T chunks
// by u (count + scatter), independently of the merge structure. Value-identical:
// only the order of additions into a slot changes.
namespace {
using UPartition = prover::UPartitionScratch;   // order: gate indices grouped by bucket; start: T+1 offsets

template <class GetU>
void build_u_partition(size_t l, size_t r, GetU get_u, u32 u_max, int T,
                       UPartition& P) {
    P.T = T;
    P.start.assign((size_t)T + 1, 0);
    P.order.resize(r - l);
    if (r <= l) return;
    const u32 span = u_max ? u_max : 1;
    auto bucket_of = [&](u32 u) {
        u64 b = (u64)u * (u64)T / (u64)span;
        return (int)(b >= (u64)T ? T - 1 : b);
    };
    std::vector<size_t> cnt((size_t)T + 1, 0);
    for (size_t i = l; i < r; ++i) cnt[bucket_of(get_u(i)) + 1]++;
    for (int t = 0; t < T; ++t) cnt[t + 1] += cnt[t];
    for (int t = 0; t <= T; ++t) P.start[t] = cnt[t];
    std::vector<size_t> cur(cnt.begin(), cnt.end() - 1);
    for (size_t i = l; i < r; ++i) P.order[cur[bucket_of(get_u(i))]++] = (u32)i;
}

// Threads to split into. More chunks than threads keeps the
// dynamic schedule balanced when u is unevenly distributed.
inline int u_part_chunks() {
    static int t = -1;
    if (t < 0) {
        const char* e = getenv("GKR_GATE_CHUNKS");
        t = e ? atoi(e) : 256;
        if (t < 1) t = 1;
    }
    return t;
}
inline bool u_part_enabled() {
    static int on = -1;
    if (on < 0) on = getenv("GKR_NO_GATE_PARALLEL") ? 0 : 1;
    return on;
}
}

// Init-phase sub-step attribution (GKR_GEN_PROF=1).
namespace {
enum { IP_FILL, IP_CLEAR, IP_VMULT, IP_BETA, IP_RELU, IP_UNI, IP_BIN, IP_UPLOAD, IP_N };
const char* ip_name[IP_N] = {"buffer assign x4","clear mult_array","V_mult gather",
                             "initBetaTable","beta_g *= relu_rou",
                             "uni_gates scatter","bin_gates scatter","gkr_sc_begin(upload)"};
double ip1_s[IP_N] = {0}, ip2_s[IP_N] = {0};
inline double ip_now() { return std::chrono::duration<double>(
    std::chrono::steady_clock::now().time_since_epoch()).count(); }
void gkr_ip_report(int idx) {
    if (!getenv("GKR_ACCT") && !getenv("GKR_GEN_PROF")) return;
    for (int ph = 0; ph < 2; ++ph) {
        double* a = ph ? ip2_s : ip1_s; double tot = 0;
        for (int i = 0; i < IP_N; ++i) tot += a[i];
        if (tot <= 0) continue;
        for (int i = 0; i < IP_N; ++i)
            fprintf(stderr, "[ip] idx=%d ph%d %-22s %7.3fs\n",
                    idx, ph + 1, ip_name[i], a[i]);
        fprintf(stderr, "[ip] idx=%d ph%d %-22s %7.3fs\n", idx, ph + 1, "TOTAL", tot);
    }
    for (int i = 0; i < IP_N; ++i) { ip1_s[i] = 0; ip2_s[i] = 0; }
}

struct IpReport { ~IpReport() {
    if (!getenv("GKR_GEN_PROF")) return;
    for (int ph = 0; ph < 2; ++ph) {
        double* a = ph ? ip2_s : ip1_s; double tot = 0;
        for (int i = 0; i < IP_N; ++i) tot += a[i];
        fprintf(stderr, "[initPhase%d_prof] total=%.3fs\n", ph + 1, tot);
        for (int i = 0; i < IP_N; ++i)
            fprintf(stderr, "    %-22s %7.3fs  %5.1f%%\n",
                    ip_name[i], a[i], tot > 0 ? 100.0 * a[i] / tot : 0.0);
    }
} } ip_report_instance;
}

void gkr_ip_report_pub(int idx) { gkr_ip_report(idx); }

// ceil(log2(n)) for size_t without the float cast/overflow pitfalls of the u32 helper.
static int ceilPow2BitLength64(size_t n) {
    if (n == 0) return -1;
    n -= 1;
    int l = 0;
    while (n) {
        n >>= 1;
        ++l;
    }
    return l;
}

linear_poly interpolate(const F &zero_v, const F &one_v) 
{
    return {one_v - zero_v, zero_v};
}

F prover::getCirValue(int layer_id, const vector<i64> &ori, u32 u) {
    return !layer_id ? val[0][ori[u]] : val[layer_id][u];
}

void prover::init() 
{
    proof_size = 0;
    cc = Commit_return();
    cc_input = Commit_return();
    cc_input_owned = false;
    cc_input_owns_gens = false;
    cc_output = Commit_return();
    input_committed_externally = false;
    release_input_commit_segments();
    r_u.resize(C.size + 1);
    r_v.resize(C.size + 1);
    prove_timer.clear();
    commit_range = {};
    for (int b = 0; b < 2; ++b) {
        V_mult[b].clear();
        mult_array[b].clear();
        tmp_V_mult[b].clear();
        tmp_mult_array[b].clear();
    }
    // lasso_mult_v is sized lazily when the lasso phase starts.
}

/**
 * This is to initialize all process.
 *
 * @param the random point to be evaluated at the output layer
 */
void prover::sumcheckInitAll(const vector<F>::const_iterator &r_0_from_v) 
{
    sumcheck_id = C.size;
    i8 last_bl = C.circuit[sumcheck_id - 1].bit_length;
    r_u[sumcheck_id].resize(last_bl);
    prove_timer.start();
    for (int i = 0; i < last_bl; ++i) 
        r_u[sumcheck_id][i] = r_0_from_v[i];
    prove_timer.stop();
}

/**
 * This is to initialize before the process of a single layer.
 *
 * @param the random combination coefficiants for multiple reduction points
 */
void prover::sumcheckInit(const F &alpha_0, const F &beta_0) 
{
    prove_timer.start();
    auto &cur = C.circuit[sumcheck_id];
    alpha = alpha_0;
    beta = beta_0;
    r_0 = r_u[sumcheck_id].begin();
    r_1 = r_v[sumcheck_id].begin();
    --sumcheck_id;
    prove_timer.stop();
}

void sc_phase1_uni_worker(vector<uniGate> &beg, std::vector<linear_poly> (&mult_array)[2],vector<F>& beta_g, vector<F>& beta_u,const int* L,const int* R,size_t start,size_t end)
{
    (void)beta_u;
    for (size_t idx = start; idx < end; ++idx)
    {
        int l=L[idx],r=R[idx];

        for (size_t i = l; i <r ; ++i) 
        {
            auto &gate = beg[i];
            bool idx = gate.lu != 0;
            mult_array[idx][gate.u] = mult_array[idx][gate.u] + beta_g[gate.g] * gate.sc;
        }
    }
}

void sc_phase1_bin_worker(layer& cur, vector<binGate> &beg, std::vector<linear_poly> (&mult_array)[2],F& V_u0,F&V_u1,vector<vector<F> >& val,vector<F>& beta_g, vector<F>& beta_u,const int* L,const int* R,int sumcheck_id,size_t start,size_t end)
{
    (void)beta_u;
    for (size_t idx = start; idx < end; ++idx)
    {
        int l=L[idx],r=R[idx];

        for (size_t i = l; i <r ; ++i) 
        {
            auto &gate = beg[i];
            bool idx = gate.getLayerIdU(sumcheck_id) != 0;
            auto val_lv =  !gate.getLayerIdV(sumcheck_id) ? val[0][cur.ori_id_v[gate.v]] : val[gate.getLayerIdV(sumcheck_id)][gate.v];
            mult_array[idx][gate.u] = mult_array[idx][gate.u] + val_lv * beta_g[gate.g] * gate.sc;  // Ahg for phase 1
        }
    }
}

static void check_lp_zero_is_memset()
{
    static const bool ok = []{
        linear_poly d; unsigned char z[sizeof(linear_poly)] = {0};
        if (memcmp(&d, z, sizeof(linear_poly)) != 0) {
            fprintf(stderr, "[prover] FATAL: linear_poly() is not all-zero bits; "
                            "the mult_array memset is invalid\n");
            abort();
        }
        return true;
    }();
    (void)ok;
}

// Bits actually addressed by the gates of the padded boundary output layer
// (pad_output_layer, non-last split partitions only), or -1 if this layer is not
// that layer or nothing is saved. output_real_len is 0 at splits=1.
int prover::zkmod_padded_eff_bits(int layer) {
    if (layer != C.size - 1 || output_real_len <= 0) return -1;
    auto &cur = C.circuit[layer];
    i64 mg = output_real_len - 1;
    for (auto &g : cur.uni_gates) mg = std::max<i64>(mg, g.g);
    for (auto &g : cur.bin_gates) mg = std::max<i64>(mg, g.g);
    int eff = ceilPow2BitLength((u32)(mg + 1));
    return eff < cur.bit_length ? eff : -1;
}

void prover::sumcheckInitPhase1(const F &relu_rou_0) 
{
    auto &cur = C.circuit[sumcheck_id];
    total[0] = ~cur.bit_length_u[0] ? 1ULL << cur.bit_length_u[0] : 0;
    total_size[0] = cur.size_u[0];
    total[1] = ~cur.bit_length_u[1] ? 1ULL << cur.bit_length_u[1] : 0;
    total_size[1] = cur.size_u[1];

    check_lp_zero_is_memset();
    // Right-size per-phase buffers for this layer's u-domain. Keep the serial assign:
    // a grow-only + OpenMP fill gave no net win.
    double _ip = ip_now();
    for (int b = 0; b < 2; ++b) {
        const size_t need = total[b];
        // assign() allocates exactly n (no geometric growth), so grow-only sizing avoids
        // per-layer reallocation churn. V_mult needs no fill (the gather writes every
        // slot); mult_array's memset covers its clearing.
        if (V_mult[b].size()    < need) V_mult[b].resize(need);
        if (mult_array[b].size() < need) mult_array[b].resize(need);
        if (need) memset(mult_array[b].data(), 0, need * sizeof(linear_poly));
        // tmp_* are read only at the CPU fold; write-only under the GPU path.
        if (!gkr_sc_gpu_enabled()) {
            if (tmp_V_mult[b].size()    < need) tmp_V_mult[b].resize(need);
            if (tmp_mult_array[b].size() < need) tmp_mult_array[b].resize(need);
        }
    }
    ip1_s[IP_FILL] += ip_now() - _ip;

    r_u[sumcheck_id].resize(cur.max_bl_u);
    beta_g.resize(1ULL << cur.bit_length);
    relu_rou = relu_rou_0;
    add_term.clear();

        _ip = ip_now();
        for (int b = 0; b < 2; ++b)
        {
            auto dep = !b ? 0 : sumcheck_id - 1;
            for (u32 u = 0; u < total[b]; ++u)
            {
                if (u >= cur.size_u[b])
                    V_mult[b][u].clear();
                else
                {
                    V_mult[b][u] = getCirValue(dep, cur.ori_id_u, u);
                }
            }
        }
        ip1_s[IP_VMULT] += ip_now() - _ip;
    prove_timer.start();
    
    _ip = ip_now();
    u64 beta_hi = 1ULL << cur.bit_length;
    {
        int eff = zkmod_padded_eff_bits(sumcheck_id);
        if (eff >= 0) {
            // Padded boundary layer: only g < 2^eff is read (every gate's g is below it);
            // eq(r, g) = eq(r_lo, g) * prod_{j>=eff} (1 - r_j) there.
            F h0 = F_ONE, h1 = F_ONE;
            for (int j = eff; j < cur.bit_length; ++j) { if (!alpha.isZero()) h0 = h0 * (F_ONE - r_0[j]); if (!beta.isZero()) h1 = h1 * (F_ONE - r_1[j]); }
            initBetaTable(beta_g, (u8)eff, r_0, r_1, alpha * h0, beta * h1);
            beta_hi = 1ULL << eff;
        } else
            initBetaTable(beta_g, cur.bit_length, r_0, r_1, alpha, beta);
    }
    ip1_s[IP_BETA] += ip_now() - _ip;
    _ip = ip_now();
    if (cur.zero_start_id < cur.size)
        for (u32 g = cur.zero_start_id; g < beta_hi; ++g)
            beta_g[g] = beta_g[g] * relu_rou;
    ip1_s[IP_RELU] += ip_now() - _ip;
    
        // u-partitioned parallel accumulate (see build_u_partition). The interval path
    // below yields only 0-4 pieces and needs >=2 to thread at all; with merge_layer
    // disabled the intervals are EMPTY, so without this the scatter runs
    // single-threaded.
    _ip = ip_now();
    if (u_part_enabled() && !cur.uni_gates.empty()) {
        const auto& GG_ = cur.uni_gates;
        UPartition &P = upart_scratch[0];   // per-prover member, see prover.hpp
        build_u_partition(0, GG_.size(), [&](size_t i){ return GG_[i].u; },
                          (u32)std::max(total[0], total[1]), u_part_chunks(), P);
        #pragma omp parallel for schedule(dynamic) num_threads(max_threads)
        for (int t = 0; t < P.T; ++t)
            for (size_t k = P.start[t]; k < P.start[t + 1]; ++k) {
                const auto &gate = GG_[P.order[k]];
                bool idx = gate.lu != 0;
                mult_array[idx][gate.u] = mult_array[idx][gate.u] + beta_g[gate.g] * gate.sc;
            }
        if (getenv("GKR_UPART_CHECK")) {   // A/B: plain loop from zero vs what u-part produced
            std::vector<linear_poly> A0(mult_array[0].size()), A1(mult_array[1].size());
            std::vector<linear_poly>* A[2] = {&A0, &A1};
            for (auto &gate: GG_) { bool idx = gate.lu != 0; (*A[idx])[gate.u] = (*A[idx])[gate.u] + beta_g[gate.g] * gate.sc; }
            size_t bad = 0; long long first_s = -1, first_u = -1;
            for (int sl = 0; sl < 2; ++sl) for (size_t u = 0; u < A[sl]->size(); ++u)
                if (memcmp(&(*A[sl])[u], &mult_array[sl][u], sizeof(linear_poly)) != 0) { if (!bad) { first_s = sl; first_u = (long long)u; } ++bad; }
            fprintf(stderr, "[upart-check] phase1 uni layer=%d gates=%zu T=%d total=%zu/%zu size_u=%lld/%lld mismatches=%zu first(slot,u)=(%lld,%lld)\n",
                    sumcheck_id, GG_.size(), P.T, (size_t)total[0], (size_t)total[1], (long long)cur.size_u[0], (long long)cur.size_u[1], bad, first_s, first_u);
            if (bad) {
                size_t lu0 = 0, lumax = 0, z_up = 0, z_pl = 0, ordered_ok = 0, umax0 = 0;
                for (auto &gate: GG_) { if (gate.lu == 0) { ++lu0; if ((size_t)gate.u > umax0) umax0 = gate.u; } if ((size_t)gate.lu > lumax) lumax = gate.lu; }
                unsigned char zb[sizeof(linear_poly)] = {0};
                for (size_t u = 0; u < A0.size(); ++u) { if (!memcmp(&mult_array[0][u], zb, sizeof(linear_poly))) ++z_up; if (!memcmp(&A0[u], zb, sizeof(linear_poly))) ++z_pl; }
                // is P.order a permutation of 0..n-1 ?
                std::vector<char> seen(GG_.size(), 0); size_t dup = 0, oor = 0;
                for (size_t k = 0; k < P.start[P.T]; ++k) { u32 i = P.order[k]; if (i >= GG_.size()) ++oor; else if (seen[i]++) ++dup; }
                for (size_t i = 0; i < GG_.size(); ++i) ordered_ok += seen[i] ? 1 : 0;
                fprintf(stderr, "[upart-check]   gates lu==0: %zu (umax %zu) lu max %zu | slot0 zero entries: upart %zu plain %zu of %zu | order: covered %zu/%zu dup %zu oor %zu span %zu\n",
                        lu0, umax0, lumax, z_up, z_pl, A0.size(), ordered_ok, GG_.size(), dup, oor, P.start[P.T]);
                // first mismatch, raw words
                const unsigned long long *wa = (const unsigned long long*)&A0[first_u], *wb = (const unsigned long long*)&mult_array[0][first_u];
                fprintf(stderr, "[upart-check]   u=%lld plain=%016llx%016llx.. upart=%016llx%016llx..\n", first_u, wa[0], wa[1], wb[0], wb[1]);
            }
        }
    }
    else if(cur.uni_interval.size()>=2)
    {
        int thd = max_threads;
        const size_t interval_cnt = cur.uni_interval.size();
        std::vector<int> L(cur.uni_interval.size()), R(cur.uni_interval.size());
        for (size_t j = 0; j <cur.uni_interval.size(); ++j) 
        {
                L[j]=cur.uni_interval[j].first;
                R[j]=cur.uni_interval[j].second;
        }
        std::vector<std::thread> threads;
        threads.reserve(thd);
        const size_t chunk = (interval_cnt + thd - 1) / thd;
        for(int i=0;i<thd;i++)
        {
            size_t start = static_cast<size_t>(i) * chunk;
            size_t end = std::min(interval_cnt, start + chunk);
            if (start >= end) break;
            threads.emplace_back(sc_phase1_uni_worker, std::ref(cur.uni_gates),std::ref(mult_array),std::ref(beta_g),std::ref(beta_u),L.data(),R.data(),start,end);
        }
        for (auto& t : threads) t.join();
    }
    else for (auto &gate: cur.uni_gates)
        {
            bool idx = gate.lu != 0;
            mult_array[idx][gate.u] = mult_array[idx][gate.u] + beta_g[gate.g] * gate.sc;
        }
        ip1_s[IP_UNI] += ip_now() - _ip;
        _ip = ip_now();
        std::vector<linear_poly> upc_snap0, upc_snap1;
        if (getenv("GKR_UPART_CHECK")) { upc_snap0 = mult_array[0]; upc_snap1 = mult_array[1]; }
        if (u_part_enabled() && !cur.bin_gates.empty()) {
        const auto& GG_ = cur.bin_gates;
        UPartition &P = upart_scratch[1];   // per-prover member, see prover.hpp
        build_u_partition(0, GG_.size(), [&](size_t i){ return GG_[i].u; },
                          (u32)std::max(total[0], total[1]), u_part_chunks(), P);
        #pragma omp parallel for schedule(dynamic) num_threads(max_threads)
        for (int t = 0; t < P.T; ++t)
            for (size_t k = P.start[t]; k < P.start[t + 1]; ++k) {
                const auto &gate = GG_[P.order[k]];
                bool idx = gate.getLayerIdU(sumcheck_id) != 0;
                auto val_lv = getCirValue(gate.getLayerIdV(sumcheck_id), cur.ori_id_v, gate.v);
                mult_array[idx][gate.u] = mult_array[idx][gate.u] + val_lv * beta_g[gate.g] * gate.sc;
            }
        if (getenv("GKR_UPART_CHECK")) {   // A/B: snapshot-before + plain loop vs u-part result
            std::vector<linear_poly>* A[2] = {&upc_snap0, &upc_snap1};
            for (auto &gate: GG_) { bool idx = gate.getLayerIdU(sumcheck_id) != 0;
                auto val_lv = getCirValue(gate.getLayerIdV(sumcheck_id), cur.ori_id_v, gate.v);
                (*A[idx])[gate.u] = (*A[idx])[gate.u] + val_lv * beta_g[gate.g] * gate.sc; }
            size_t bad = 0; long long first_s = -1, first_u = -1;
            for (int sl = 0; sl < 2; ++sl) for (size_t u = 0; u < A[sl]->size(); ++u)
                if (memcmp(&(*A[sl])[u], &mult_array[sl][u], sizeof(linear_poly)) != 0) { if (!bad) { first_s = sl; first_u = (long long)u; } ++bad; }
            fprintf(stderr, "[upart-check] phase1 bin layer=%d gates=%zu mismatches=%zu first(slot,u)=(%lld,%lld)\n",
                    sumcheck_id, GG_.size(), bad, first_s, first_u);
        }
    }
    else if(cur.bin_interval.size()>=2)
    {
        int thd = max_threads;
        const size_t interval_cnt = cur.bin_interval.size();
        std::vector<int> L(cur.bin_interval.size()), R(cur.bin_interval.size());
        for (size_t j = 0; j <cur.bin_interval.size(); ++j) 
        {
                L[j]=cur.bin_interval[j].first;
                R[j]=cur.bin_interval[j].second;
        }
        std::vector<std::thread> threads;
        threads.reserve(thd);
        const size_t chunk = (interval_cnt + thd - 1) / thd;
        for(int i=0;i<thd;i++)
        {
            size_t start = static_cast<size_t>(i) * chunk;
            size_t end = std::min(interval_cnt, start + chunk);
            if (start >= end) break;
            threads.emplace_back(sc_phase1_bin_worker, std::ref(cur),std::ref(cur.bin_gates),std::ref(mult_array),std::ref(V_u0),std::ref(V_u1),std::ref(val),std::ref(beta_g),std::ref(beta_u),L.data(),R.data(),sumcheck_id,start,end);
        }
        for (auto& t : threads) t.join();
    }
    else  for (auto &gate: cur.bin_gates) 
        {
            bool idx = gate.getLayerIdU(sumcheck_id) != 0;
            auto val_lv = getCirValue(gate.getLayerIdV(sumcheck_id), cur.ori_id_v, gate.v);
            mult_array[idx][gate.u] = mult_array[idx][gate.u] + val_lv * beta_g[gate.g] * gate.sc;  // Ahg for phase 1
        }
    ip1_s[IP_BIN] += ip_now() - _ip;
    _ip = ip_now();
    gkr_sc_begin(V_mult[0].data(), mult_array[0].data(), total[0],
                 V_mult[1].data(), mult_array[1].data(), total[1]);
    ip1_s[IP_UPLOAD] += ip_now() - _ip;
    round = 0;
    prove_timer.stop();
}


void sc_phase2_uni_worker(
    const std::vector<uniGate>& beg,
    F& sum_value,
    const F V_u0,
    const F V_u1,
    const std::vector<F>& beta_g,
    const std::vector<F>& beta_u,
    const int* L,
    const int* R,
    size_t start,
    size_t end)
{
    F local;
    local.clear();

    for (size_t idx = start; idx < end; ++idx) {
        const int l = L[idx];
        const int r = R[idx];
        F ss;
        ss.clear();
        for (int i = l; i < r; ++i) {
            const auto& gate = beg[i];
            const F V_u = gate.lu ? V_u1 : V_u0;
            ss += beta_g[gate.g] * beta_u[gate.u] * V_u * gate.sc;
        }
        local += ss;
    }

    sum_value = local;
}




void sc_phase2_bin_worker( vector<binGate> &beg, std::vector<linear_poly> (&mult_array)[2],F& V_u0,F&V_u1,vector<F>& beta_g, vector<F>& beta_u,const int* L,const int* R,int sumcheck_id,size_t start,size_t end)
{
    for (size_t idx = start; idx < end; ++idx)
    {
        int l=L[idx],r=R[idx];

        for (size_t i = l; i <r ; ++i) 
        {
            auto &gate = beg[i];
            bool idx = gate.getLayerIdV(sumcheck_id);
            auto V_u = !gate.getLayerIdU(sumcheck_id) ? V_u0 : V_u1;
            mult_array[idx][gate.v] =mult_array[idx][gate.v]+beta_g[gate.g] * beta_u[gate.u] * V_u * gate.sc;
        }
    }
}

// Guard for the memset: linear_poly() must be bit-identical to all-zero. Must
// stay function-local -- a namespace-scope static runs before initPairing().

void prover::sumcheckInitPhase2() 
{
    auto &cur = C.circuit[sumcheck_id];
    total[0] = ~cur.bit_length_v[0] ? 1ULL << cur.bit_length_v[0] : 0;
    total_size[0] = cur.size_v[0];
    total[1] = ~cur.bit_length_v[1] ? 1ULL << cur.bit_length_v[1] : 0;
    total_size[1] = cur.size_v[1];
    i8 fft_bl = cur.fft_bit_length;
    i8 cnt_bl = cur.max_bl_v;

    check_lp_zero_is_memset();
    // Right-size per-phase buffers for this layer's v-domain (see phase 1's fill).
    double _ip = ip_now();
    for (int b = 0; b < 2; ++b) {
        const size_t need = total[b];
        // assign() allocates exactly n (no geometric growth), so grow-only sizing avoids
        // per-layer reallocation churn. V_mult needs no fill (the gather writes every
        // slot); mult_array's memset covers its clearing.
        if (V_mult[b].size()    < need) V_mult[b].resize(need);
        if (mult_array[b].size() < need) mult_array[b].resize(need);
        if (need) memset(mult_array[b].data(), 0, need * sizeof(linear_poly));
        // tmp_* are read only at the CPU fold; write-only under the GPU path.
        if (!gkr_sc_gpu_enabled()) {
            if (tmp_V_mult[b].size()    < need) tmp_V_mult[b].resize(need);
            if (tmp_mult_array[b].size() < need) tmp_mult_array[b].resize(need);
        }
    }
    ip2_s[IP_FILL] += ip_now() - _ip;

    r_v[sumcheck_id].resize(cur.max_bl_v);


    beta_u.resize(1ULL << cur.max_bl_u);

    

    add_term.clear();
    prove_timer.start();
    _ip = ip_now();
    initBetaTable(beta_u, cur.max_bl_u, r_u[sumcheck_id].begin(), F_ONE,max_threads); //  beta_u is U in the code
    ip2_s[IP_BETA] += ip_now() - _ip;
    _ip = ip_now();
    for (int b = 0; b < 2; ++b)
    {
        auto dep = !b ? 0 : sumcheck_id - 1;
        for (u32 v = 0; v < total[b]; ++v)
        {
            V_mult[b][v] = v >= cur.size_v[b] ? F_ZERO : getCirValue(dep, cur.ori_id_v, v);
        }
    }
    ip2_s[IP_VMULT] += ip_now() - _ip;
    
        // Pure REDUCTION into add_term (no per-slot scatter), so any split is safe; no
    // u-partition needed. Per-thread partials, summed in a fixed order.
    _ip = ip_now();
    if (u_part_enabled() && !cur.uni_gates.empty()) {
        const auto& GG_ = cur.uni_gates;
        const int T_ = u_part_chunks();
        std::vector<F> partial_((size_t)T_);
        #pragma omp parallel for schedule(static) num_threads(max_threads)
        for (int t = 0; t < T_; ++t) {
            F acc; acc.clear();
            const size_t lo = GG_.size() * (size_t)t / T_, hi = GG_.size() * (size_t)(t + 1) / T_;
            for (size_t i = lo; i < hi; ++i) {
                const auto &gate = GG_[i];
                auto V_u = !gate.lu ? V_u0 : V_u1;
                acc = acc + beta_g[gate.g] * beta_u[gate.u] * V_u * gate.sc;
            }
            partial_[t] = acc;
        }
        for (int t = 0; t < T_; ++t) add_term += partial_[t];
    }
    else if(cur.uni_interval.size()>=2)
    {
        const int thd = max_threads;
        const size_t n = cur.uni_interval.size();

        std::vector<int> L(n), R(n);
        for (size_t j = 0; j < n; ++j) {
            L[j] = cur.uni_interval[j].first;
            R[j] = cur.uni_interval[j].second;
        }

        std::vector<F> sum(thd);
        for (int i = 0; i < thd; ++i) sum[i].clear();

        const F V_u0_snap = V_u0;
        const F V_u1_snap = V_u1;

        std::vector<std::thread> threads;
        threads.reserve(thd);
        const size_t chunk = (n + thd - 1) / thd;
        int used_threads = 0;
        for (int i = 0; i < thd; ++i) {
            size_t start = static_cast<size_t>(i) * chunk;
            size_t end = std::min(n, start + chunk);
            if (start >= end) break;
            ++used_threads;
            threads.emplace_back(
                sc_phase2_uni_worker,
                std::cref(cur.uni_gates),
                std::ref(sum[i]),
                V_u0_snap,
                V_u1_snap,
                std::cref(beta_g),
                std::cref(beta_u),
                L.data(),
                R.data(),
                start,
                end
            );
        }

        for (auto& t : threads) t.join();

        for (int i = 0; i < used_threads; ++i) add_term += sum[i];
    }
    else for (auto &gate: cur.uni_gates) 
    {
        auto V_u = !gate.lu ? V_u0 : V_u1;                  //V_u0 is claim 0, V_u1 is claim 1
        add_term = add_term + beta_g[gate.g] * beta_u[gate.u] * V_u * gate.sc;
    }
    ip2_s[IP_UNI] += ip_now() - _ip;
    _ip = ip_now();
        // NB partitioned by gate.v -- phase 2 scatters into mult_array[idx][gate.v],
    // unlike phase 1 which uses gate.u.
    if (u_part_enabled() && !cur.bin_gates.empty()) {
        const auto& GG_ = cur.bin_gates;
        UPartition &P = upart_scratch[2];   // per-prover member, see prover.hpp
        build_u_partition(0, GG_.size(), [&](size_t i){ return GG_[i].v; },
                          (u32)std::max(total[0], total[1]), u_part_chunks(), P);
        #pragma omp parallel for schedule(dynamic) num_threads(max_threads)
        for (int t = 0; t < P.T; ++t)
            for (size_t k = P.start[t]; k < P.start[t + 1]; ++k) {
                const auto &gate = GG_[P.order[k]];
                bool idx = gate.getLayerIdV(sumcheck_id);
                auto V_u = !gate.getLayerIdU(sumcheck_id) ? V_u0 : V_u1;
                mult_array[idx][gate.v] = mult_array[idx][gate.v] + beta_g[gate.g] * beta_u[gate.u] * V_u * gate.sc;
            }
    }
    else if(cur.bin_interval.size()>=2)
    {
        int thd = max_threads;
        const size_t n = cur.bin_interval.size();
        std::vector<int> L(n), R(n);
        for (size_t j = 0; j < n; ++j) {
            L[j] = cur.bin_interval[j].first;
            R[j] = cur.bin_interval[j].second;
        }
        std::vector<std::thread> threads;
        threads.reserve(thd);
        const size_t chunk = (n + thd - 1) / thd;
        for(int i=0;i<thd;i++)
        {
            size_t start = static_cast<size_t>(i) * chunk;
            size_t end = std::min(n, start + chunk);
            if (start >= end) break;
            threads.emplace_back(sc_phase2_bin_worker, std::ref(cur.bin_gates),std::ref(mult_array),std::ref(V_u0),std::ref(V_u1),std::ref(beta_g),std::ref(beta_u),L.data(),R.data(),sumcheck_id,start,end);
        }
        for (auto& t : threads) t.join();
    }
    else for (auto &gate: cur.bin_gates) 
    {
        bool idx = gate.getLayerIdV(sumcheck_id);
        auto V_u = !gate.getLayerIdU(sumcheck_id) ? V_u0 : V_u1;
        mult_array[idx][gate.v] = mult_array[idx][gate.v] + beta_g[gate.g] * beta_u[gate.u] * V_u * gate.sc;
    }
    ip2_s[IP_BIN] += ip_now() - _ip;

    _ip = ip_now();
    gkr_sc_begin(V_mult[0].data(), mult_array[0].data(), total[0],
                 V_mult[1].data(), mult_array[1].data(), total[1]);
    ip2_s[IP_UPLOAD] += ip_now() - _ip;
    round = 0;
    prove_timer.stop();
}

void prover::release_lasso_state()
{
    auto drop = [](vector<F> &v) { vector<F>().swap(v); };
    for (size_t k = 1; k < val.size(); ++k) drop(val[k]);
    drop(lasso_mult_v); drop(lasso_v_mult); drop(beta_g); drop(beta_u);
}

void prover::sumcheckLassoInit(const vector<F> &s_u, const vector<F> &s_v,const vector<vector<F>>& r_uu, const vector<vector<F>>& r_vv) 
{
    auto _ta = li_clk::now();
    // 1ULL, not 1: bit_length reaches 31 at opt2b7, and `1 << 31` in 32-bit signed
    // arithmetic is INT_MIN, which sign-extends to ~1.8e19 as a size_t and throws
    // std::length_error("vector::_M_default_append"). The conditional resize below
    // already sizes this buffer correctly; this line only pre-sizes it.
    lasso_mult_v.resize(1ULL << C.circuit[0].bit_length);
    prove_timer.start();

    sumcheck_id = 0;
    total[1] = (1ULL << C.circuit[sumcheck_id].bit_length);
    total_size[1] = C.circuit[sumcheck_id].size;
    // Threaded first-touch fill of 2^n elements. The zeros are NOT skippable -- the
    // lasso sumcheck reads the whole 2^n domain.
    if (lasso_mult_v.size() < total[1]) lasso_mult_v.resize(total[1]);
    {
        const size_t n_fill = lasso_mult_v.size();
        F* __restrict dst = lasso_mult_v.data();
        #pragma omp parallel for schedule(static) num_threads(max_threads)
        for (long long i = 0; i < (long long)n_fill; ++i) dst[i] = F_ZERO;
    }
    // Pad the base witness to the full domain so lasso reductions never read past bounds.
    if (val[0].size() < total[1]) {
        val[0].resize(total[1]);
        for (size_t i = total_size[1]; i < val[0].size(); ++i) {
            val[0][i].clear();
        }
    }

    r_u[0].resize(C.circuit[0].bit_length);
    g_lasso_init_timers.lasso_alloc += li_secs(_ta);
    timer ggg;
    ggg.start();

    i8 max_bl = 0;
    for (int i = sumcheck_id + 1; i < C.size; ++i)
        max_bl = max(max_bl, max(C.circuit[i].bit_length_u[0], C.circuit[i].bit_length_v[0]));
    if (getenv("GKR_DBG_LASSO_SIZES"))
        fprintf(stderr, "[lasso] c0.bit_length=%d total[1]=%llu val0=%zu "
                        "lasso_mult_v=%zu max_bl=%d beta_g=%llu\n",
                (int)C.circuit[0].bit_length, (unsigned long long)total[1],
                val[0].size(), lasso_mult_v.size(), (int)max_bl,
                (unsigned long long)(1ULL << max_bl));
    beta_g.resize(1ULL << max_bl);
    for (int i = sumcheck_id + 1; i < C.size; ++i) 
    {
        i32 bit_length_i = C.circuit[i].bit_length_u[0];
        i32 size_i = C.circuit[i].size_u[0];
        if (~bit_length_i) 
        {
            r_u[i].resize(C.circuit[i].max_bl_u);
            for(int j=0;j<C.circuit[i].max_bl_u;j++)
                r_u[i][j]=r_uu[i][j];
            auto _tb = li_clk::now();
            initBetaTable(beta_g, bit_length_i, r_u[i].begin(), s_u[i - 1],max_threads);
            g_lasso_init_timers.beta += li_secs(_tb);
            auto _ts = li_clk::now();
            for (u32 hu = 0; hu < size_i; ++hu)
            {
                i64 u = C.circuit[i].ori_id_u[hu];
                lasso_mult_v[u] += beta_g[hu];
            }
            g_lasso_init_timers.scatter += li_secs(_ts);
            g_lasso_init_timers.scattered += size_i;
        }
        bit_length_i = C.circuit[i].bit_length_v[0];
        size_i = C.circuit[i].size_v[0];
        if (~bit_length_i) 
        {
            r_v[i].resize(C.circuit[i].max_bl_v);
            for(int j=0;j<C.circuit[i].max_bl_v;j++)
                r_v[i][j]=r_vv[i][j];
            auto _tb = li_clk::now();
            initBetaTable( beta_g, bit_length_i, r_v[i].begin(), s_v[i - 1],max_threads);
            g_lasso_init_timers.beta += li_secs(_tb);
            auto _ts = li_clk::now();
            for (u32 hv = 0; hv < size_i; ++hv)
            {
                i64 v = C.circuit[i].ori_id_v[hv];
                lasso_mult_v[v] += beta_g[hv];
            }
            g_lasso_init_timers.scatter += li_secs(_ts);
            g_lasso_init_timers.scattered += size_i;
        }
    }
    round = 0;
    prove_timer.stop();
}

quadratic_poly prover::sumcheckUpdate1(const F &previous_random) {
    return sumcheckUpdate(previous_random, r_u[sumcheck_id]);
}

quadratic_poly prover::sumcheckUpdate2(const F &previous_random) {
    return sumcheckUpdate(previous_random, r_v[sumcheck_id]);
}

quadratic_poly prover::sumcheckUpdate(const F &previous_random, vector<F> &r_arr) 
{
    prove_timer.start();

    if (round) r_arr.at(round - 1) = previous_random;
    ++round;
    quadratic_poly ret;

    add_term = add_term * (F_ONE - previous_random);
    for (int b = 0; b < 2; ++b)
        ret = ret + sumcheckUpdateEach(previous_random, b);
    ret = ret + quadratic_poly(F_ZERO, -add_term, add_term);

    prove_timer.stop();
    proof_size += F_BYTE_SIZE * 3;
    return ret;
}



void sumcheckUpdate_worker(quadratic_poly &sum,vector<linear_poly> &tmp_v,vector<linear_poly> &tmp_mult,vector<linear_poly> &tmp_v_2,vector<linear_poly> &tmp_mult_2,Fr previous_random,int total_size,int start,int end)
{
    sum.clear();
    for(int i=start;i<end;i++)
    {
        u32 g0 = i << 1, g1 = i << 1 | 1;
        if (g0 >= total_size) 
            break;
        tmp_v_2[i] = interpolate(tmp_v[g0].eval(previous_random), tmp_v[g1].eval(previous_random));
        tmp_mult_2[i] = interpolate(tmp_mult[g0].eval(previous_random), tmp_mult[g1].eval(previous_random));
        sum = sum + tmp_mult_2[i] * tmp_v_2[i];
    }
}
quadratic_poly prover::sumcheckUpdateEach(const F &previous_random, bool idx) 
{
    // Rounds run on the device once gkr_sc_begin() uploaded the tables; the host
    // arrays below are then stale by design and only the CPU path touches them.
    if (gkr_sc_active_b(idx)) {
        quadratic_poly ret;
        if (total[idx] == 1)
            gkr_sc_handle_single(idx, add_term, previous_random);
        else if (total[idx] > 1)
            ret = gkr_sc_update_b(idx, previous_random, total[idx], total_size[idx]);
        total[idx] >>= 1;
        total_size[idx] = (total_size[idx] + 1) >> 1;
        return ret;
    }
    auto &tmp_mult = mult_array[idx];
    auto &tmp_v = V_mult[idx];
    auto &tmp_mult_2 = tmp_mult_array[idx];
    auto &tmp_v_2 = tmp_V_mult[idx];
    // The init phases size tmp_* only when the GPU path is off for the whole run; a
    // group gkr_sc_begin could not fit in VRAM lands here with them empty. Grow-only,
    // so later rounds are free.
    if (tmp_v_2.size()    < (total[idx] >> 1)) tmp_v_2.resize(total[idx] >> 1);
    if (tmp_mult_2.size() < (total[idx] >> 1)) tmp_mult_2.resize(total[idx] >> 1);

    if (total[idx] == 1) 
    {
        tmp_v[0] = tmp_v[0].eval(previous_random);
        tmp_mult[0] = tmp_mult[0].eval(previous_random);
        add_term = add_term + tmp_v[0].b * tmp_mult[0].b;
    }

    quadratic_poly ret;
    ret.clear();
    if(total[idx]<(1<<15))
    {
        for (u32 i = 0; i < (total[idx] >> 1); ++i) 
        {
            u32 g0 = i << 1, g1 = i << 1 | 1;
            if (g0 >= total_size[idx]) 
            {
                tmp_v[i].clear();
                tmp_mult[i].clear();
                continue;
            }
            tmp_v[i] = interpolate(tmp_v[g0].eval(previous_random), tmp_v[g1].eval(previous_random));
            tmp_mult[i] = interpolate(tmp_mult[g0].eval(previous_random), tmp_mult[g1].eval(previous_random));
            ret = ret + tmp_mult[i] * tmp_v[i];
        }
    }
    else
    {
        timer tt_f1,tt_f2;
        tt_f1.start();
        int total_work=(total[idx] >> 1);
        int thd = max_threads;
        std::vector<quadratic_poly> qp(thd);
        std::vector<std::thread> threads;
        threads.reserve(thd);
        int chunk = (total_work + thd - 1) / thd;
        int used_threads = 0;
        for(int j=0;j<thd;j++)
        {
            int start = j * chunk;
            int end = std::min(total_work, start + chunk);
            if (start >= end) break;
            ++used_threads;
            threads.emplace_back(sumcheckUpdate_worker,std::ref(qp[j]),std::ref(tmp_v),std::ref(tmp_mult),std::ref(tmp_v_2),std::ref(tmp_mult_2),previous_random,total_size[idx],start,end);
        }
        for (auto& t : threads) t.join();
        for(int j=0;j<used_threads;j++)
            ret=ret+qp[j];
        tt_f1.stop();
        tt_f2.start();
        for(int i=0;i<total_work;i++)
        {
            if((i<<1)<total_size[idx])
            {
                tmp_mult[i]=tmp_mult_2[i];
                tmp_v[i]=tmp_v_2[i];
            }
            else
            {
                tmp_mult[i].clear();
                tmp_v[i].clear();
            }
        }
        tt_f2.stop();
    }
    
    total[idx] >>= 1;
    total_size[idx] = (total_size[idx] + 1) >> 1;

    return ret;
}


/**
 * This is to evaluate a multi-linear extension at a random point.
 *
 * @param the value of the array & random point & the size of the array & the size of the random point
 * @return sum of `values`, or 0.0 if `values` is empty.
 */
F prover::Vres(const vector<F>::const_iterator &r, u32 output_size, u8 r_size,int layer_id,int start) 
{
    prove_timer.start();

    // The padded boundary output layer (pad_output_layer) holds
    // output_real_len real values in a 2^25 domain; the tail is zero, so folding it
    // only produces zeros. Fold the real prefix, then scale by (1 - r_i) for the
    // remaining rounds. Bit-identical result.
    if (layer_id == C.size - 1 && start == 0 &&
        output_real_len > 0 && (u64)output_real_len < (u64)output_size) {
        size_t n = (size_t)output_real_len;
        vector<F> o(val[layer_id].begin(), val[layer_id].begin() + n);
        for (u8 i = 0; i < r_size; ++i) {
            const size_t m = (n + 1) >> 1;
            const F one_m = F_ONE - r[i];
            for (size_t j = 0; j < m; ++j) {
                F a = o[j << 1] * one_m;
                if ((j << 1 | 1) < n) a = a + o[j << 1 | 1] * r[i];
                o[j] = a;
            }
            n = m;
        }
        F res = o[0];
        prove_timer.stop();
        proof_size += F_BYTE_SIZE;
        return res;
    }

    vector<F> output(output_size);
    for (u32 i = 0; i < output_size; ++i)
        output[i] = val[layer_id][i+start];
    u32 whole = 1ULL << r_size;
    for (u8 i = 0; i < r_size; ++i) {
        for (u32 j = 0; j < (whole >> 1); ++j) {
            if (j > 0)
                output[j].clear();
            if ((j << 1) < output_size)
                output[j] = output[j << 1] * (F_ONE - r[i]);
            if ((j << 1 | 1) < output_size)
                output[j] = output[j] + output[j << 1 | 1] * (r[i]);
        }
        whole >>= 1;
    }
    F res = output[0];

    prove_timer.stop();
    proof_size += F_BYTE_SIZE;
    return res;
}

void prover::sumcheckFinalize1(const F &previous_random, F &claim_0, F &claim_1) {
    prove_timer.start();
    r_u[sumcheck_id].at(round - 1) = previous_random;
    if (gkr_sc_active_b(0) || gkr_sc_active_b(1)) {
        linear_poly v0f, m0f, v1f, m1f;
        gkr_sc_end(v0f, m0f, v1f, m1f);
        V_u0 = claim_0 = total[0] ? v0f.eval(previous_random)
                       : (~C.circuit[sumcheck_id].bit_length_u[0]) ? v0f.b : F_ZERO;
        V_u1 = claim_1 = total[1] ? v1f.eval(previous_random)
                       : (~C.circuit[sumcheck_id].bit_length_u[1]) ? v1f.b : F_ZERO;
        prove_timer.stop();
        proof_size += F_BYTE_SIZE * 2;
        return;
    }
    V_u0 = claim_0 = total[0] ? V_mult[0][0].eval(previous_random) : (~C.circuit[sumcheck_id].bit_length_u[0]) ? V_mult[0][0].b : F_ZERO;
    V_u1 = claim_1 = total[1] ? V_mult[1][0].eval(previous_random) : (~C.circuit[sumcheck_id].bit_length_u[1]) ? V_mult[1][0].b : F_ZERO;
    prove_timer.stop();

    mult_array[0].clear();
    mult_array[1].clear();
    V_mult[0].clear();
    V_mult[1].clear();
    proof_size += F_BYTE_SIZE * 2;
}

void prover::sumcheckFinalize2(const F &previous_random, F &claim_0, F &claim_1) {
    prove_timer.start();
    r_v[sumcheck_id].at(round - 1) = previous_random;
    if (gkr_sc_active_b(0) || gkr_sc_active_b(1)) {
        linear_poly v0f, m0f, v1f, m1f;
        gkr_sc_end(v0f, m0f, v1f, m1f);
        claim_0 = total[0] ? v0f.eval(previous_random)
                           : (~C.circuit[sumcheck_id].bit_length_v[0]) ? v0f.b : F_ZERO;
        claim_1 = total[1] ? v1f.eval(previous_random)
                           : (~C.circuit[sumcheck_id].bit_length_v[1]) ? v1f.b : F_ZERO;
        prove_timer.stop();
        proof_size += F_BYTE_SIZE * 2;
        return;
    }
    claim_0 = total[0] ? V_mult[0][0].eval(previous_random) : (~C.circuit[sumcheck_id].bit_length_v[0]) ? V_mult[0][0].b : F_ZERO;
    claim_1 = total[1] ? V_mult[1][0].eval(previous_random) : (~C.circuit[sumcheck_id].bit_length_v[1]) ? V_mult[1][0].b : F_ZERO;
    prove_timer.stop();

    mult_array[0].clear();
    mult_array[1].clear();
    V_mult[0].clear();
    V_mult[1].clear();
    proof_size += F_BYTE_SIZE * 2;
}

void prover::sumcheck_lasso_Finalize(const F &previous_random, F &claim_1) 
{
    prove_timer.start();
    r_u[sumcheck_id].at(round - 1) = previous_random;
    claim_1 = total[1] ? V_mult[1][0].eval(previous_random) : V_mult[1][0].b;
    prove_timer.stop();
    proof_size += F_BYTE_SIZE;
}

namespace {

bool is_power_of_two_size(size_t x) {
    return x && ((x & (x - 1)) == 0);
}

size_t highest_power_of_two_leq(size_t x) {
    if (!x) return 0;
    size_t ret = 1;
    while ((ret << 1) && (ret << 1) <= x) ret <<= 1;
    return ret;
}

size_t next_aligned_block(size_t start, size_t remaining) {
    if (!remaining) return 0;
    size_t block = start ? (start & (~start + 1)) : highest_power_of_two_leq(remaining);
    if (!block) block = 1;
    while (block > remaining) block >>= 1;
    return block;
}

Commit_return commit_segment(const vector<F> &vals, size_t start, size_t len,
                             const vector<G1> &gens, int thr, prover *p, timer *T) {
    Commit_return ret{};
    if (!len || start >= vals.size()) return ret;

    int l = ceilPow2BitLength64(len);
    if (l < 0 || l >= static_cast<int>(sizeof(size_t) * 8)) {
        throw std::runtime_error("commit_segment: input too large to commit safely");
    }
    size_t padded = size_t(1) << l;
    ret.l = l;
    ret.g = (G1 *) gens.data();
    ret.G = gens.back();
    ret.w = new Fr[padded]();
    ret.ww = new ll[padded]();

    size_t lim = std::min(len, vals.size() - start);
    for (size_t i = 0; i < lim; ++i) {
        ret.w[i] = vals[start + i];
        ret.ww[i] = convert(vals[start + i]);
    }
    std::cout << "Calling external commit" << endl;
    if (p) p->prove_timer.start();
    if (T) T->start();
    ret.comm = prover_commit(ret.ww, (G1 *) gens.data(), l, thr);
    if (T) T->stop();
    if (p) p->prove_timer.stop();
    // ww only feeds prover_commit; openCommit opens through .w, and the one .ww
    // consumer (OwnedCommitment::capture) runs on cc_output alone, never on what
    // this builds. Holding it cost 2^l * sizeof(__int128).
    delete[] ret.ww; ret.ww = nullptr;
    return ret;
}

}  // namespace

Commit_return commit_values_standalone(const vector<F> &vals, int thr, timer *T) {
    Commit_return ret{};
    if (vals.empty()) return ret;
    int l = ceilPow2BitLength64(vals.size());
    size_t padded = size_t(1) << l;
    size_t gen_count = size_t(1) << (l - (l >> 1));
    G1 *gens = new G1[gen_count + 1];
    G1 base = gen_gi(gens, static_cast<int>(gen_count));
    gens[gen_count] = base;

    ret.l = l;
    ret.g = gens;
    ret.G = base;
    // Fill the WHOLE padded range, as commitInput does. `new Fr[padded]()` does NOT
    // zero (mcl's FpT has a user-provided default ctor) while `new ll[padded]()` does,
    // so a partial fill commits zeros via `ww` and opens garbage via `w` -- honest
    // proofs then REJECT non-deterministically on non-pow2 lengths.
    ret.w = new Fr[padded];
    ret.ww = new ll[padded];
    F zero; zero.clear();
    for (size_t i = 0; i < padded; ++i) {
        const F &src = (i < vals.size()) ? vals[i] : zero;
        ret.w[i] = src;
        ret.ww[i] = convert(src);
    }
    if (T) T->start();
    ret.comm = prover_commit(ret.ww, gens, l, thr);
    if (T) T->stop();
    delete[] ret.ww; ret.ww = nullptr;   // see commit_segment: write-only after the MSM
    return ret;
}

// cc.g / cc_output.g alias gens.data() -- deliberately NOT freed.
// cc_input is self-allocated in the split runner (only the demo binaries alias an
// external commitment), so it must be freed or one prefix commitment leaks per
// partition. Flags are opt-in, so unset sites never free.
// Each InputCommitmentSegment owns its `commit` by value; clear() destroys PODs
// and would leak their w/ww/comm.
// .g is NOT freed (aliases the caller's `gens`); .external_commit is NOT freed
// (points at cc_input, owned separately).
void prover::release_input_commit_segments() {
    for (auto &seg : input_commit_segments) {
        delete[] seg.commit.w;
        delete[] seg.commit.ww;
        delete[] seg.commit.comm;
        seg.commit = Commit_return();
    }
    input_commit_segments.clear();
}

prover::~prover() {
    delete[] cc.w;           delete[] cc.ww;           delete[] cc.comm;
    delete[] cc_output.w;    delete[] cc_output.ww;    delete[] cc_output.comm;
    if (cc_input_owned) {
        delete[] cc_input.w; delete[] cc_input.ww; delete[] cc_input.comm;
        if (cc_input_owns_gens) delete[] cc_input.g;
    }
    release_input_commit_segments();
}

void prover::commitInput(const vector<G1> &gens,int thr, timer *T)
{
    if (input_copy.empty()) copyInputWitness();
    const auto &input = input_copy;
    cc_input = Commit_return();
    if (separate_input && inp_end_id > 0 && inp_end_id < static_cast<i64>(input.size())) {
        cc_input = commit_segment(input, 0, inp_end_id, gens, thr, this, T);
        cc_input_owned = true;           // commit_segment new[]s w/ww/comm...
        cc_input_owns_gens = false;      // ...but ret.g aliases `gens`
        cc = commit_segment(input, static_cast<size_t>(inp_end_id), input.size() - inp_end_id, gens, thr, this, T);
        return;
    }
    separate_input = false;
    const int bitlen = C.circuit[0].bit_length >= 0
            ? C.circuit[0].bit_length
            : ceilPow2BitLength64(input.size());
    if (bitlen < 0 || bitlen >= static_cast<int>(sizeof(size_t) * 8)) {
        throw std::runtime_error("commitInput: invalid bit length for input commitment");
    }
    const size_t padded_size = size_t(1) << bitlen;

    ll* vi = new ll[padded_size];
    Fr* dat = new Fr[padded_size];
    F zero;
    zero.clear();
    for (size_t i = 0; i < padded_size; ++i) {
        const F &src = (i < input.size()) ? input[i] : zero;
        vi[i] = convert(src);
        dat[i] = src;
    }
    prove_timer.start();
    if (T) T->start();
    G1* ret = prover_commit(vi,(G1*)gens.data(),bitlen,thr);
    if (T) T->stop();
    prove_timer.stop();
    cc.comm = ret;
    cc.G = gens.back();
    cc.g = (G1*)gens.data();
    cc.l = bitlen;
    cc.w = dat;
    delete[] vi; cc.ww = nullptr;        // see commit_segment: write-only after the MSM
}

void prover::commitInputSeparate_image(const vector<G1> &gens,int thr, timer *T) {
    if (input_copy.empty()) copyInputWitness();
    separate_input = true;
    if (inp_end_id <= 0 || inp_end_id > static_cast<i64>(input_copy.size())) return;
    cc_input = commit_segment(input_copy, 0, inp_end_id, gens, thr, this, T);
    cc_input_owned = true;               // as above: arrays owned, basis aliased
    cc_input_owns_gens = false;
}

void prover::commitInputSeparate_predicate(const vector<G1> &gens,int thr, timer *T) {
    if (input_copy.empty()) copyInputWitness();
    separate_input = true;
    if (inp_end_id >= static_cast<i64>(input_copy.size())) return;
    cc = commit_segment(input_copy, static_cast<size_t>(inp_end_id), input_copy.size() - inp_end_id, gens, thr, this, T);
}

void prover::commitOutputPrivate(const vector<G1> &gens,int thr,bool keep_ww) {
    output_private = true;
    const int len = static_cast<int>(val[C.size - 1].size());
    const int l = ceilPow2BitLength(static_cast<u32>(len));
    const size_t padded = size_t(1) << l;
    const Fr *src = val[C.size - 1].data();

    // Only this prefix can be non-zero. pad_output_layer() pads the output layer out
    // to the SUCCESSOR's input-half size, and that half is set by the WEIGHT half
    // (getTotalInputSize returns 2*max(input_half, weight_half)), so at gpt2 len64
    // 65,536 real values live in a 2^25 domain -- 0.195 % of it. The commitment is
    // still over the whole padded domain; only the work of building the scalars is
    // restricted. 0 means the layer was never padded.
    size_t real_len = (output_real_len > 0 &&
                       static_cast<size_t>(output_real_len) <= static_cast<size_t>(len))
                      ? static_cast<size_t>(output_real_len)
                      : static_cast<size_t>(len);

    // `w` is what openCommit opens through, so it is always the full padded vector.
    // memset only the tail: the memcpy covers [0, len) and len is usually the whole
    // array.
    Fr *dat = new Fr[padded];
    if (output_real_len > 0) {
        // Padded boundary layer (split runs only): first-touch the 1 GiB copy on
        // all threads, not one.
        #pragma omp parallel for schedule(static) num_threads(thr)
        for (long long i = 0; i < (long long)padded; ++i) {
            if (i < len) dat[i] = src[i]; else dat[i].clear();
        }
    } else {
    memcpy(dat, src, sizeof(Fr) * len);
    if (padded > static_cast<size_t>(len))
        memset(dat + len, 0, sizeof(Fr) * (padded - static_cast<size_t>(len)));
    }

    G1 *ret = nullptr;
    ll *vo = nullptr;
    timer T;

    // Device path: never build the host __int128 array at all. Upload the real prefix
    // as Fr (2 MB here, against a 537 MB upload of converted scalars), zero the scalar
    // array on the device, and convert there. See gpu_fr_scalars.hpp for why the
    // conversion is a real computation and not a reinterpret.
    if (!gkr_config.commit_cpu && !keep_ww) {
        T.start();
        ret = zkgpt_gpu::gpu_hyrax_commit_fr_rowmajor(src, real_len,
                                                      (const G1 *) gens.data(), l);
        T.stop();
    }
    // GKR_COMMIT_FR_VERIFY=1 -- re-run the whole commit through the host convert() and
    // the CPU MSM and compare the resulting row commitments. Comparing the points is a
    // stronger check than comparing scalars: a scalar that differs but lands on the
    // same commitment cannot break an opening, and one that matches cannot hide here.
    if (ret != nullptr && getenv("GKR_COMMIT_FR_VERIFY")) {
        ll *chk = new ll[padded];
        memset(chk, 0, sizeof(ll) * padded);
        for (int i = 0; i < len; ++i)
            if (!src[i].isZero()) chk[i] = convert(src[i]);
        const bool was_cpu = gkr_config.commit_cpu;
        gkr_config.commit_cpu = true;                 // force the reference path
        G1 *ref = prover_commit(chk, (G1 *) gens.data(), l, thr);
        gkr_config.commit_cpu = was_cpu;
        const int rownum = 1 << (l / 2);
        int bad = 0;
        for (int i = 0; i < rownum; ++i) if (!(ref[i] == ret[i])) ++bad;
        fprintf(stderr, "[commit_fr_verify] l=%d real=%zu rows=%d mismatched=%d %s\n",
                l, real_len, rownum, bad, bad ? "FAIL" : "ok");
        delete[] ref;
        delete[] chk;
        if (bad) exit(EXIT_FAILURE);
    }

    if (ret == nullptr) {
        vo = new ll[padded];
        memset(vo, 0, sizeof(ll) * padded);
        // convert(0) is 0, which the memset already wrote, and convert() is not cheap
        // (isNegative, a conditional negate, getLittleEndian, a byte loop). Testing
        // isZero() rather than bounding the loop at real_len keeps this correct for
        // callers that never padded.
        for (int i = 0; i < len; ++i)
            if (!src[i].isZero()) vo[i] = convert(src[i]);
        T.start();
        ret = prover_commit(vo, (G1 *) gens.data(), l, thr);
        T.stop();
    }

    output_commit_range = T.range();
    prove_time += T.elapse_sec();
    cc_output.comm = ret;
    cc_output.G = gens.back();
    cc_output.g = (G1 *) gens.data();
    cc_output.l = l;
    cc_output.w = dat;
    cc_output.ww = vo;          // null on the device path; see the header comment

    proof_size += 1 << (cc_output.l / 2);
}

// The boundary input commitment of a non-first partition, built
// straight from val[0] -- no prefix vector copy, no serial convert() of the 2^25
// mostly-zero domain. Same commitment as commit_values_standalone(prefix): `w` is the
// full padded vector (zero tail), the scalars are convert(w[i]) with zeros skipped.
Commit_return prover::commitInputPrefixFast(size_t n, int thr, timer *T) {
    Commit_return ret{};
    if (n == 0) return ret;
    const Fr *src = val[0].data();
    int l = ceilPow2BitLength64((i64)n);
    size_t padded = size_t(1) << l;
    size_t gen_count = size_t(1) << (l - (l >> 1));
    double t0 = ip_now();
    G1 *gens = new G1[gen_count + 1];
    G1 base = gen_gi(gens, static_cast<int>(gen_count));
    gens[gen_count] = base;
    double t1 = ip_now();
    // last non-zero entry bounds the real prefix
    size_t real = n;
    while (real > 0 && src[real - 1].isZero()) --real;
    double t2 = ip_now();
    ret.l = l; ret.g = gens; ret.G = base;
    ret.w = new Fr[padded];
    #pragma omp parallel for schedule(static) num_threads(thr)
    for (long long i = 0; i < (long long)padded; ++i) {
        if ((size_t)i < n) ret.w[i] = src[i]; else ret.w[i].clear();
    }
    double t3 = ip_now();
    if (T) T->start();
    if (!gkr_config.commit_cpu) {
        ret.comm = zkgpt_gpu::gpu_hyrax_commit_fr_rowmajor(src, real, (const G1 *)gens, l);
    } else {
        ll *ww = new ll[padded]();
        for (size_t i = 0; i < real; ++i) if (!src[i].isZero()) ww[i] = convert(src[i]);
        ret.comm = prover_commit(ww, gens, l, thr);
        delete[] ww;
    }
    if (T) T->stop();
    double t4 = ip_now();
    fprintf(stderr, "[fast_prefix] n=%zu real=%zu l=%d gen_gi=%.3f scan=%.3f w=%.3f commit=%.3f\n",
            n, real, l, t1 - t0, t2 - t1, t3 - t2, t4 - t3);
    ret.ww = nullptr;
    return ret;
}

void prover::copyInputWitness() {
    std::cout << "copying input witness" << std::endl;
    input_copy = val[0];
}

void prover::addExternalInputSegment(size_t start, size_t len, Commit_return *commit, const string &label) {
    external_input_segments.push_back({start, len, commit, label});
}

void prover::clearExternalInputSegments() {
    external_input_segments.clear();
    release_input_commit_segments();
}

void prover::executeCommit(int thr) {
    std::cout<<"Start commitInput"<<std::endl;
    timer T;
    int logn = ceilPow2BitLength64(input_copy.size());
    if (logn < 0) {
        throw std::runtime_error("executeCommit: input witness is empty");
    }
    u64 n_sqrt = 1ULL << (logn - (logn >> 1));
    gens.resize(n_sqrt);
    G1 base=gen_gi(gens.data(),n_sqrt);
    gens.push_back(base);
    bool external_input_commit = input_committed_externally && cc_input.comm != nullptr;
    if (input_not_committed && external_input_segments.empty() && inp_end_id > 0 && inp_end_id < static_cast<i64>(input_copy.size())) {
        std::cout << "Commit predicate with public input prefix" << std::endl;
        release_input_commit_segments();
        auto add_public_range = [&](size_t start, size_t end) {
            while (start < end) {
                size_t len = next_aligned_block(start, end - start);
                InputCommitmentSegment seg{};
                seg.start = start;
                seg.len = len;
                seg.commit = Commit_return();
                seg.external_commit = nullptr;
                seg.label = "public";
                input_commit_segments.push_back(seg);
                start += len;
            }
        };
        auto commit_local_range = [&](size_t start, size_t end) {
            while (start < end) {
                size_t len = next_aligned_block(start, end - start);
                InputCommitmentSegment seg{};
                seg.start = start;
                seg.len = len;
                seg.external_commit = nullptr;
                seg.label = "local";
                seg.commit = commit_segment(input_copy, start, len, gens, thr, this, &T);
                input_commit_segments.push_back(seg);
                start += len;
            }
        };
        add_public_range(0, static_cast<size_t>(inp_end_id));
        commit_local_range(static_cast<size_t>(inp_end_id), input_copy.size());
    } else if ((separate_input || external_input_commit) && inp_end_id > 0 && inp_end_id < static_cast<i64>(input_copy.size())) {
        std::cout << "Commit input/predicate as aligned segments" << std::endl;
        release_input_commit_segments();
        auto commit_local_range = [&](size_t start, size_t end) {
            while (start < end) {
                size_t len = next_aligned_block(start, end - start);
                InputCommitmentSegment seg{};
                seg.start = start;
                seg.len = len;
                seg.external_commit = nullptr;
                seg.label = "local";
                seg.commit = commit_segment(input_copy, start, len, gens, thr, this, &T);
                input_commit_segments.push_back(seg);
                start += len;
            }
        };
        if (external_input_commit) {
            InputCommitmentSegment seg{};
            seg.start = 0;
            seg.len = static_cast<size_t>(inp_end_id);
            seg.commit = Commit_return();
            seg.external_commit = &cc_input;
            seg.label = "external_input";
            input_commit_segments.push_back(seg);
        } else {
            commit_local_range(0, static_cast<size_t>(inp_end_id));
        }
        commit_local_range(static_cast<size_t>(inp_end_id), input_copy.size());
    } else if (!external_input_segments.empty()) {
        std::cout << "Commit input with external segments" << std::endl;
        release_input_commit_segments();
        auto ext = external_input_segments;
        std::sort(ext.begin(), ext.end(), [](const ExternalInputSegment &a, const ExternalInputSegment &b) {
            return a.start < b.start;
        });

        auto commit_local_range = [&](size_t start, size_t end) {
            while (start < end) {
                size_t len = next_aligned_block(start, end - start);
                InputCommitmentSegment seg{};
                seg.start = start;
                seg.len = len;
                seg.external_commit = nullptr;
                seg.label = "local";
                seg.commit = commit_segment(input_copy, start, len, gens, thr, this, &T);
                input_commit_segments.push_back(seg);
                start += len;
            }
        };

        size_t cursor = 0;
        for (const auto &seg : ext) {
            if (!seg.commit) {
                throw std::runtime_error("external input segment has no commitment pointer");
            }
            if (!is_power_of_two_size(seg.len) || (seg.start % seg.len) != 0) {
                throw std::runtime_error("external input segment must be power-of-two sized and aligned");
            }
            if (seg.start < cursor || seg.start + seg.len > input_copy.size()) {
                throw std::runtime_error("external input segment range is invalid or overlapping");
            }
            commit_local_range(cursor, seg.start);
            InputCommitmentSegment out{};
            out.start = seg.start;
            out.len = seg.len;
            out.commit = Commit_return();
            out.external_commit = seg.commit;
            out.label = seg.label;
            input_commit_segments.push_back(out);
            cursor = seg.start + seg.len;
        }
        commit_local_range(cursor, input_copy.size());
    } else if ((separate_input || input_not_committed || external_input_commit) && inp_end_id > 0 && inp_end_id < static_cast<i64>(input_copy.size())) {

        if (!input_not_committed && !external_input_commit) {
            std::cout << "Commit input query." << std::endl;
            commitInputSeparate_image(gens, thr, &T);
        }
        std::cout << "Commit predicate."<<std::endl;
        commitInputSeparate_predicate(gens, thr, &T);
    } else {
        std::cout<<"Commit Input(together)"<<std::endl;
        commitInput(gens, thr, &T);
    }
    if (!input_commit_segments.empty()) {
        for (const auto &seg : input_commit_segments) {
            Commit_return *c = seg.external_commit ? seg.external_commit : const_cast<Commit_return *>(&seg.commit);
            if (c && c->l) proof_size += 1 << (c->l / 2);
        }
    } else if (separate_input && inp_end_id > 0) {
        if (include_input_commit_size && cc_input.l)
            proof_size += 1 << (cc_input.l / 2);
        if (cc.l)
            proof_size += 1 << (cc.l / 2);
    } else if (cc.l) {
        proof_size += 1 << (cc.l / 2);
    }
    // GPU commit kernels launch async; the clock must bracket DEVICE completion or
    // the MSM time leaks into the next phase's bucket.
    hipDeviceSynchronize();
    commit_range = T.range();
    prove_time += T.elapse_sec();
    input_copy.clear();
    std::cout << "Commit done" << std::endl;
}

__int128 convert(Fr x)	
{	
    int sign=0;	
    Fr abs;	
    if(x.isNegative())	
    {	
        sign=1;	
        abs=-x;	
    }	
    else	
        abs=x;	

    uint8_t bf[16]={0};	 //64 bit
    int size=abs.getLittleEndian(bf,16);	
    ll V=0;	
    for(int j=size-1;j>=0;j--)	
        V=V*256+bf[j];	
    if(sign)	
        V=-V;	
    return V;	
}
