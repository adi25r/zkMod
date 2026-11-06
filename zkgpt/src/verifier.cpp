// Created by 69029 on 3/9/2021
#include "verifier.hpp"
#include "phase_acct.hpp"
#include "phase_log.hpp"
#include "config.hpp"
#include "gkr_gpu.hpp"
#include "gpu_matmul_prover.hpp"
#include "gpu_range_prover.hpp"
#include "gkr_pred_gpu.hpp"
#include <hip/hip_runtime.h>
#include <chrono>
#include <malloc.h>  // malloc_trim (past-peak release)

namespace {
using vl_clk = std::chrono::high_resolution_clock;
inline double vl_secs(vl_clk::time_point t0) {
    return std::chrono::duration<double>(vl_clk::now() - t0).count();
}
}  // namespace

static bool pred_check_enabled() {
    static int on = getenv("GKR_PRED_CHECK") ? 1 : 0;
    return on != 0;
}

#ifdef GKR_GPU_ENABLED
static inline void gkr_prover_barrier() {
    hipDeviceSynchronize();
}
#else
static inline void gkr_prover_barrier() {}
#endif
#include "global_var.hpp"
#include <utils.hpp>
#include <circuit.h>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <initializer_list>

using namespace std;
using namespace mcl::bn;

long get_vm_peak_kb_verf() {
    FILE* f = fopen("/proc/self/status", "r");
    char line[256];
    long peak = 0;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "VmPeak: %ld", &peak) == 1)
            break;
    }
    fclose(f);
    return peak;
}

verifier::verifier(prover* pr, const layeredCircuit& cir)
    : p(pr), C(cir), max_threads(0), total_timer(true), total_slow_timer(true) {
    comm = {};
    comm_input = {};
    separate_input = false;
    input_not_committed = false;
    output_private = false;
    final_claim_u0.resize(C.size + 2);
    final_claim_v0.resize(C.size + 2);
    prover_time = verifier_time = 0;
    matrix_time = 0;
    r_u.resize(C.size + 2);
    r_v.resize(C.size + 2);
    // make the prover ready
    p->init();
}

F verifier::getFinalValue(const F& claim_u0, const F& claim_u1, const F& claim_v0, const F& claim_v1) {
    auto test_value = bin_value[0] * (claim_u0 * claim_v0) + bin_value[1] * (claim_u1 * claim_v1) +
                      bin_value[2] * (claim_u1 * claim_v0) + uni_value[0] * claim_u0 +
                      uni_value[1] * claim_u1;

    return test_value;
}

void verifier::betaInitPhase1(int depth, const F& alpha, const F& beta, const vector<F>::const_iterator& r_0,
                              const vector<F>::const_iterator& r_1, const F& relu_rou) {
    i8 bl = C.circuit[depth].bit_length;
    i8 fft_bl = C.circuit[depth].fft_bit_length;
    i8 fft_blh = C.circuit[depth].fft_bit_length - 1;
    i8 cnt_bl = bl - fft_bl, cnt_bl2 = C.circuit[depth].max_bl_u - fft_bl;

    const int zeff = p->zkmod_padded_eff_bits(depth);  // padded boundary layer (split only)
    if (zeff >= 0) {
        F h0 = F_ONE, h1 = F_ONE;
        for (int j = zeff; j < bl; ++j) {
            if (!alpha.isZero())
                h0 = h0 * (F_ONE - r_0[j]);
            if (!beta.isZero())
                h1 = h1 * (F_ONE - r_1[j]);
        }
        beta_g.assign(1ULL << zeff, F_ZERO);
        initBetaTable(beta_g, (u8)zeff, r_0, r_1, alpha * h0, beta * h1);
    } else {
        beta_g.resize(1ULL << bl);
        initBetaTable(beta_g, C.circuit[depth].bit_length, r_0, r_1, alpha, beta);
    }
    // Mirrors prover::beta_g_is_zero()
    beta_g_zero = alpha.isZero() && beta.isZero() && zero_beta_skip_enabled();
    if (beta_g_zero)
        assert_zero_beta(beta_g, "betaInitPhase1", depth);
    if (C.circuit[depth].zero_start_id < C.circuit[depth].size)
        for (u32 g = C.circuit[depth].zero_start_id;
             g < beta_g.size() && g < 1ULL << C.circuit[depth].bit_length; ++g)
            beta_g[g] = beta_g[g] * relu_rou;
    beta_u.resize(1ULL << C.circuit[depth].max_bl_u);
    initBetaTable(beta_u, C.circuit[depth].max_bl_u, r_u[depth].begin(), F_ONE);
    // Both predicate phases read these, so upload once here
    if (!beta_g_zero && gkr_pred_gpu_enabled()) {
        gkr_pred_upload_beta(0, beta_g);
        gkr_pred_upload_beta(1, beta_u);
    }
}

void verifier::betaInitPhase2(int depth) {
    beta_v.resize(1ULL << C.circuit[depth].max_bl_v);
    initBetaTable(beta_v, C.circuit[depth].max_bl_v, r_v[depth].begin(), F_ONE);
    if (!beta_g_zero && gkr_pred_gpu_enabled())
        gkr_pred_upload_beta(2, beta_v);
}

void pred1_worker(const vector<uniGate>& beg, vector<F>& uni_value, const vector<F>& beta_g,
                  const vector<F>& beta_u, const int* L, const int* R, size_t start, size_t end) {
    for (size_t idx = start; idx < end; ++idx) {
        int l = L[idx], r = R[idx];
        for (auto gate = beg.begin() + l; gate != beg.begin() + r; gate++) {
            bool idx = gate->lu;
            uni_value[idx] += beta_g[gate->g] * beta_u[gate->u] * gate->sc;
        }
    }
}
void pred2_worker(const vector<binGate>& beg, vector<F>& bin_value, const vector<F>& beta_g,
                  const vector<F>& beta_u, const vector<F>& beta_v, const int* L, const int* R, size_t start,
                  size_t end) {
    for (size_t idx = start; idx < end; ++idx) {
        int l = L[idx], r = R[idx];
        for (auto gate = beg.begin() + l; gate != beg.begin() + r; gate++) {
            bin_value[gate->l] += beta_g[gate->g] * beta_u[gate->u] * beta_v[gate->v] * gate->sc;
        }
    }
}

void verifier::predicatePhase1_cpu(int layer_id, F out[2]) {
    auto& cur_layer = C.circuit[layer_id];
    F(&uni_value)[2] = *(F(*)[2])out;
    uni_value[0].clear();
    uni_value[1].clear();
    vector<F> univ[max_threads];
    for (int i = 0; i < max_threads; i++) {
        univ[i].emplace_back(F(0));
        univ[i].emplace_back(F(0));
    }
    if (C.circuit[layer_id].uni_interval.size() >= 2) {
        int thd = max_threads;
        const size_t interval_cnt = C.circuit[layer_id].uni_interval.size();
        std::vector<int> L(interval_cnt), R(interval_cnt);
        for (size_t j = 0; j < interval_cnt; ++j) {
            L[j] = cur_layer.uni_interval[j].first;
            R[j] = cur_layer.uni_interval[j].second;
        }
        std::vector<std::thread> threads;
        threads.reserve(thd);
        const size_t chunk = (interval_cnt + thd - 1) / thd;
        int used_threads = 0;
        for (int i = 0; i < thd; i++) {
            size_t start = static_cast<size_t>(i) * chunk;
            size_t end = std::min(interval_cnt, start + chunk);
            if (start >= end)
                break;
            ++used_threads;
            threads.emplace_back(pred1_worker, std::cref(cur_layer.uni_gates), std::ref(univ[i]),
                                 std::cref(beta_g), std::cref(beta_u), L.data(), R.data(), start, end);
        }
        for (auto& t : threads)
            t.join();
        for (int i = 0; i < used_threads; i++) {
            uni_value[0] += univ[i][0];
            uni_value[1] += univ[i][1];
        }
    } else
        for (auto& gate : cur_layer.uni_gates) {
            bool idx = gate.lu;
            uni_value[idx] += beta_g[gate.g] * beta_u[gate.u] * gate.sc;
        }
}

void verifier::predicatePhase2_cpu(int layer_id, F out[3]) {
    F(&bin_value)[3] = *(F(*)[3])out;
    bin_value[0] = bin_value[1] = bin_value[2] = F_ZERO;

    auto& cur_layer = C.circuit[layer_id];
    vector<F> binv[max_threads];
    for (int i = 0; i < max_threads; i++) {
        binv[i].emplace_back(F(0));
        binv[i].emplace_back(F(0));
        binv[i].emplace_back(F(0));
    }
    if (C.circuit[layer_id].bin_interval.size() >= 2) {
        int thd = max_threads;
        const size_t interval_cnt = C.circuit[layer_id].bin_interval.size();
        std::vector<int> L(interval_cnt), R(interval_cnt);
        for (size_t j = 0; j < interval_cnt; ++j) {
            L[j] = cur_layer.bin_interval[j].first;
            R[j] = cur_layer.bin_interval[j].second;
        }
        std::vector<std::thread> threads;
        threads.reserve(thd);
        const size_t chunk = (interval_cnt + thd - 1) / thd;
        int used_threads = 0;
        for (int i = 0; i < thd; i++) {
            size_t start = static_cast<size_t>(i) * chunk;
            size_t end = std::min(interval_cnt, start + chunk);
            if (start >= end)
                break;
            ++used_threads;
            threads.emplace_back(pred2_worker, std::cref(cur_layer.bin_gates), std::ref(binv[i]),
                                 std::cref(beta_g), std::cref(beta_u), std::cref(beta_v), L.data(), R.data(),
                                 start, end);
        }
        for (auto& t : threads)
            t.join();
        for (int i = 0; i < used_threads; i++) {
            bin_value[0] += binv[i][0];
            bin_value[1] += binv[i][1];
            bin_value[2] += binv[i][2];
        }
    } else
        for (auto& gate : cur_layer.bin_gates)
            bin_value[gate.l] += beta_g[gate.g] * beta_u[gate.u] * beta_v[gate.v] * gate.sc;
}

// GPU predicate dispatch. gkr_predicate{1,2}_gpu reduce every gate in the layer into the 2
void verifier::predicatePhase1(int layer_id) {
    if (beta_g_zero) {  // every term is beta_g[g] * (...) == 0
        uni_value[0].clear();
        uni_value[1].clear();
        bin_value[0] = bin_value[1] = bin_value[2] = F_ZERO;
        return;
    }
    if (gkr_pred_gpu_enabled()) {
        gkr_predicate1_gpu(C.circuit[layer_id].uni_gates, uni_value);
        if (pred_check_enabled()) {
            F ref[2];
            predicatePhase1_cpu(layer_id, ref);
            if (ref[0] != uni_value[0] || ref[1] != uni_value[1]) {
                fprintf(stderr, "[pred-check] MISMATCH phase1 layer=%d\n", layer_id);
                abort();
            }
        }
    } else {
        predicatePhase1_cpu(layer_id, uni_value);
    }
    bin_value[0] = bin_value[1] = bin_value[2] = F_ZERO;
}

void verifier::predicatePhase2(int layer_id) {
    uni_value[0] = uni_value[0] * beta_v[0];
    uni_value[1] = uni_value[1] * beta_v[0];

    if (beta_g_zero)
        return;  // bin_value already zeroed by predicatePhase1

    if (gkr_pred_gpu_enabled()) {
        gkr_predicate2_gpu(C.circuit[layer_id].bin_gates, bin_value);
        if (pred_check_enabled()) {
            F ref[3];
            predicatePhase2_cpu(layer_id, ref);
            if (ref[0] != bin_value[0] || ref[1] != bin_value[1] || ref[2] != bin_value[2]) {
                fprintf(stderr, "[pred-check] MISMATCH phase2 layer=%d\n", layer_id);
                abort();
            }
        }
    } else {
        predicatePhase2_cpu(layer_id, bin_value);
    }
}

bool verifier::prove(int commit_thread, timer& circuit_initialization_timer) {
    // Sync commitment data generated by the prover after executeCommit()
    comm = p->cc;
    comm_input = p->cc_input;
    separate_input = p->separate_input;
    input_not_committed = p->input_not_committed;
    input_committed_externally = p->input_committed_externally;
    output_private = p->output_private;

    cout << "Circuit Initiation finished" << endl;
    circuit_initialization_timer.stop();
    cout << "Circuit Initiation time: " << circuit_initialization_timer.elapse_sec() << "s" << endl;
    cout << "Proving Service started" << endl;

    total_timer.clear();
    total_slow_timer.clear();
    total_timer.start();
    total_slow_timer.start();

    // One RESULT line per proof, at line start
    {
        zkgpt_acct::Scope _a(zkgpt_acct::A_VERIFY_GKR);
        ZKGPT_PHASE_SCOPE("verify_gkr");
        if (!verifyGKR()) {
            cout << "RESULT: REJECT\n";
            cerr << "[gkr] VERIFICATION REJECTED (stage=gkr)\n";
            return false;
        }
    }
    gkr_sc_teardown();
    gkr_pred_teardown();
    // A 2^32 Lasso domain adds three 137 GB tables
    if (C.circuit[0].bit_length >= 32) {
        auto drop = [](auto& v) { std::decay_t<decltype(v)>().swap(v); };
        for (size_t k = 1; k < p->val.size(); ++k)
            drop(p->val[k]);
        for (int b = 0; b < 2; ++b) {
            drop(p->V_mult[b]);
            drop(p->mult_array[b]);
            drop(p->tmp_V_mult[b]);
            drop(p->tmp_mult_array[b]);
        }
        malloc_trim(0);
        cout << "[lasso] dropped per-layer values and GKR scratch (domain 2^" << (int)C.circuit[0].bit_length
             << ")" << endl;
    }
    cout << "Start Lasso" << endl;
    {
        zkgpt_acct::Scope _a(zkgpt_acct::A_VERIFY_LASSO);
        ZKGPT_PHASE_SCOPE("verify_lasso");
        if (!verifyLasso()) {
            cout << "RESULT: REJECT\n";
            cerr << "[gkr] VERIFICATION REJECTED (stage=lasso)\n";
            return false;
        }
    }
    if (g_past_peak_hook) {  // past the peak: drop Lasso-only state, then release (verifier.hpp)
        ZKGPT_PHASE_SCOPE("past_peak_free");
        p->release_lasso_state();  // openCommit reads val[0] only
        auto drop = [](vector<F>& v) { vector<F>().swap(v); };
        drop(beta_g);
        drop(beta_u);
        drop(beta_v);
        malloc_trim(0);
        g_past_peak_hook();
    }
    cout << "Open Commit" << endl;
    {
        zkgpt_acct::Scope _a(zkgpt_acct::A_OPEN_COMMIT);
        ZKGPT_PHASE_SCOPE("open_commit");
        if (!openCommit()) {
            cout << "RESULT: REJECT\n";
            cerr << "[gkr] VERIFICATION REJECTED (stage=open)\n";
            return false;
        }
    }
    cout << "RESULT: ACCEPT\n";

    total_timer.stop();
    total_slow_timer.stop();

    auto prover_core_range = p->proveTimerRange();
    auto commit_range = p->commitTimerRange();
    auto total_prover_range = combine_ranges({prover_core_range, commit_range, range_prover_range});
    add_prover(PH_HYRAX_COMMIT, commit_range.duration_sec);
    // range_prover_range is only populated by the CPU demos that pass it explicitly
    prover_time += range_prover_range.duration_sec;
    return true;
}

// offset is an absolute index into val[0] (fc_input_id)
Fr* init_book_keeping(int m, int n, vector<Fr>& vec, i64 offset, vector<Fr>& ra) {
    assert(ra.size() == m + n);
    Fr* pa[30];
    for (int j = 0; j <= n; j++) {
        pa[j] = new Fr[1 << (m + n - j)];
    }
    i64 t = min((i64)(1 << (m + n)), (i64)vec.size() - offset);
    memset(pa[0] + t, 0, sizeof(Fr) * ((1 << (m + n)) - t));

    for (i64 j = 0; j < t; j++)
        pa[0][j] = vec[j + offset];
    for (int j = 1; j <= n; j++) {
        for (int kp = 0; kp < (1 << (m + n - j)); kp++) {
            pa[j][kp] =
                pa[j - 1][kp + (1 << (m + n - j))] * ra[m + n - j] + pa[j - 1][kp] * (1 - ra[m + n - j]);
        }
    }
    for (int j = 0; j < n; j++) {
        delete[] pa[j];
    }
    return pa[n];
}

Fr* init_book_keeping_fast(int m, int n, int* vec, vector<Fr>& ra, int max_threads) {
    assert(ra.size() == m + n);
    timer tt;

    Fr* ret = new Fr[1 << m];
    Fr *eqq = new Fr[1 << n], *eq1 = new Fr[1 << (n / 2)], *eq2 = new Fr[1 << (n / 2)];

    for (int j = 0; j < (1 << n); j++) {
        eqq[j] = 1;
        for (int i = 0; i < n; i++) {
            if (j & (1 << (n - 1 - i)))
                eqq[j] *= ra[m + n - 1 - i];
            else
                eqq[j] *= 1 - ra[m + n - 1 - i];
        }
    }
    for (int j = 0; j < (1 << (n / 2)); j++) {
        eq1[j] = 1;
        for (int i = 0; i < n / 2; i++) {
            if (j & (1 << (n / 2 - 1 - i)))
                eq1[j] *= ra[m + n - 1 - i];
            else
                eq1[j] *= 1 - ra[m + n - 1 - i];
        }
    }
    for (int j = 0; j < (1 << (n / 2)); j++) {
        eq2[j] = 1;
        for (int i = 0; i < n / 2; i++) {
            if (j & (1 << (n / 2 - 1 - i)))
                eq2[j] *= ra[m + n / 2 - 1 - i];
            else
                eq2[j] *= 1 - ra[m + n / 2 - 1 - i];
        }
    }
    std::vector<std::thread> threads1, threads2;
    int num_threads = max_threads;
    // ceil + skip empty
    int chunk_size = ((1 << (n / 2)) + num_threads - 1) / num_threads;
    Fr** fm = new Fr*[1 << (n / 2)];
    for (int i = 0; i < (1 << (n / 2)); i++)
        fm[i] = new Fr[65536];
    auto worker1 = [&](int start_id, int end_id) {
        for (int R = start_id; R < end_id; R++) {
            fm[R][0] = 0;
            fm[R][1] = eq2[R];
            for (int i = 2; i < (1 << 16); i++)
                fm[R][i] = fm[R][i - 1] + eq2[R];
        }
    };
    for (int t = 0; t < num_threads; t++) {
        int start_col = t * chunk_size;
        int end_col = min(start_col + chunk_size, (1 << (n / 2)));
        if (start_col >= end_col)
            break;
        threads1.emplace_back(worker1, start_col, end_col);
    }
    for (auto& th : threads1) {
        if (th.joinable())
            th.join();
    }
    auto worker2 = [&](int start_id, int end_id) {
        for (int kp = start_id; kp < end_id; kp++) {
            ret[kp] = 0;
            Fr sum = 0;
            for (int L = 0; L < (1 << (n / 2)); L++) {
                Fr sumL = 0;
                for (int R = 0; R < (1 << (n / 2)); R++) {
                    int j = ((L << (n / 2)) + R) << m;
                    if (vec[kp + j] != 0)
                        sumL += fm[R][vec[kp + j]];
                }
                sum += eq1[L] * sumL;
            }
            ret[kp] = sum;
        }
    };
    chunk_size = ((1 << m) + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; t++) {
        int start_col = t * chunk_size;
        int end_col = min(start_col + chunk_size, (1 << m));
        if (start_col >= end_col)
            break;
        threads2.emplace_back(worker2, start_col, end_col);
    }
    for (auto& th : threads2) {
        if (th.joinable())
            th.join();
    }
    for (int i = 0; i < (1 << (n / 2)); i++)
        delete[] fm[i];
    delete[] fm;
    return ret;
}

void sc_last_worker(Fr& A, Fr& B, Fr& C, Fr* read_1, Fr* read_2, Fr* write_1, Fr* write_2, size_t start,
                    size_t end, Fr ran) {
    A.clear();
    B.clear();
    C.clear();
    for (size_t kp = start; kp < end; kp++) {
        Fr &c = read_1[kp << 1 | 1], &d = read_1[kp << 1];
        if (c == d)
            write_1[kp] = d;
        else
            write_1[kp] = (c - d) * ran + d;
        Fr &a = read_2[kp << 1 | 1], &b = read_2[kp << 1];
        if (a == b)
            write_2[kp] = b;
        else
            write_2[kp] = (a - b) * ran + b;
        if (kp & 1) {
            bool az = write_2[kp - 1].isZero(), bz = write_2[kp].isZero();
            if (az && bz)
                continue;
            A += write_1[kp - 1] * write_2[kp - 1];
            B += write_1[kp] * write_2[kp];
            C +=
                (write_1[kp] + write_1[kp] - write_1[kp - 1]) * (write_2[kp] + write_2[kp] - write_2[kp - 1]);
        }
    }
}

void sc_last_worker_first(Fr& A, Fr& B, Fr& C, Fr* read_1, Fr* read_2, size_t start, size_t end) {
    A.clear();
    B.clear();
    C.clear();
    for (size_t k = start; k < end; k++) {
        if (k & 1) {
            bool a_z = read_2[k].isZero(), b_z = read_2[k - 1].isZero();
            if (a_z && b_z)
                continue;
            B += read_1[k] * read_2[k];
            C += (read_1[k] + read_1[k] - read_1[k - 1]) * (read_2[k] + read_2[k] - read_2[k - 1]);
        } else if (!read_2[k].isZero())
            A += read_1[k] * read_2[k];
    }
}

pair<Fr, Fr> sum_check_product(Fr* f, Fr* g, int m, Fr* r, Fr ans) {
    Fr A = 0, B = 0, C = 0;
    Fr *zf = new Fr[1 << m], *zg = new Fr[1 << m];
    for (int k = 0; k < (1 << m); k++) {
        zf[k] = 2 * f[k * 2 + 1] - f[k * 2];
        zg[k] = 2 * g[k * 2 + 1] - g[k * 2];
        C += zf[k] * zg[k];
    }
    for (int k = 0; k < (1 << (m + 1)); k++) {
        if (k & 1)
            B += f[k] * g[k];
        else {
            A += f[k] * g[k];
        }
    }
    Fr a = (C + A) / 2 - B, b = 2 * B - C / 2 - 3 * A / 2, c = A;
    Fr g1 = a + b + c, g0 = c;
    if (!(ans - (g1 + g0)).isZero()) {
        cout << "ans!=g(0)+g(1) " << m << endl;
        exit(0);
    }
    if (!(a + b + c - B).isZero()) {
        cout << "sum check fail! order 1 " << m << endl;
        exit(0);
    }
    if (!(4 * a + 2 * b + c - C).isZero()) {
        cout << "sum check fail! order 2 " << m << endl;
        exit(0);
    }
    Fr send_r = r[0];
    Fr *new_f = new Fr[1 << m], *new_g = new Fr[1 << m];
    for (int k = 0; k < (1 << m); k++) {
        new_f[k] = (1 - send_r) * f[k * 2] + (send_r)*f[k * 2 + 1];
        new_g[k] = (1 - send_r) * g[k * 2] + (send_r)*g[k * 2 + 1];
    }
    Fr new_ans = send_r * send_r * a + send_r * b + c;
    if (m == 0) {
        assert((new_ans - new_f[0] * new_g[0]).isZero());
        return make_pair(new_f[0], new_g[0]);
    } else
        return sum_check_product(new_f, new_g, m - 1, r + 1, new_ans);
}

bool verifier::verifyGKR() {
    F alpha = F_ONE, beta = F_ZERO, relu_rou, final_claim_u1, final_claim_v1;
    r_u[C.size].resize(C.circuit[C.size - 1].bit_length);
    for (i8 i = 0; i < C.circuit[C.size - 1].bit_length; ++i)
        r_u[C.size][i].setByCSPRNG();
    vector<F>::const_iterator r_0 = r_u[C.size].begin();
    vector<F>::const_iterator r_1;
    timer ptimer, vtimer;
    timer mat_timer;

    auto previousSum = p->Vres(r_0, C.circuit[C.size - 1].size, C.circuit[C.size - 1].bit_length, C.size - 1);
    cout << "Peak Memory after Vres: " << get_vm_peak_kb_verf() << " KB" << endl;
    if (output_private) {
        if (p->cc_output.comm == nullptr) {
            int logn = C.circuit[C.size - 1].bit_length;
            u64 n_sqrt = 1ULL << (logn - (logn >> 1));
            p->gens_output.clear();
            p->gens_output.resize(n_sqrt);
            G1 base = gen_gi(p->gens_output.data(), n_sqrt);
            p->gens_output.push_back(base);
            p->commitOutputPrivate(p->gens_output, p->max_threads);
        }
        auto& comm_out = p->cc_output;
        int half = comm_out.l / 2;
        int other = comm_out.l - half;
        Fr* Lp = new Fr[1 << other];
        Fr* Rp = new Fr[1 << half];
        brute_force_compute_LR(Lp, Rp, r_u[C.size].data(), comm_out.l);
        auto tim = hyrax::open(comm_out.w, r_u[C.size].data(), previousSum, comm_out.G, comm_out.g, Lp, Rp,
                               comm_out.comm, comm_out.l);
        p->prove_timer.merge_timer(tim.first);
        add_prover(PH_HYRAX_OPEN, tim.first.duration_sec);
        std::cout << "Open Commit (output): " << format_range(tim.first) << ": " << tim.first.duration_sec
                  << std::endl;
        verifier_time += tim.second.duration_sec;
        delete[] Lp;
        delete[] Rp;
    }
    cout << "Peak Memory after Vres: " << get_vm_peak_kb_verf() << " KB" << endl;

    ptimer.start();
    p->sumcheckInitAll(r_0);
    ptimer.stop();
    cout << "Peak Memory after sumcheckInitAll: " << get_vm_peak_kb_verf() << " KB" << endl;
    int fc_cnt = 0, now_fc_id;
    Fr prev_u1 = 0, prev_v1 = 0;
    for (int i = C.size - 1; i; --i) {
        if (C.circuit[i].ty == layerType::FCONN)
            fc_cnt++;
    }

    timer init, sc;
    double init_t = 0, sc_t = 0;
    double matrix_tt_time = 0;
    for (int i = C.size - 1; i; --i) {
        cout << "Verifying layer (size) " << i << "(" << C.circuit[i].size << ") ... "
             << get_vm_peak_kb_verf() << " KB" << endl;
        // One PHASE mark per layer so a memory trace can be placed within verify_gkr
        ZKGPT_PHASE_MARKF("gkr_layer%d%s", i, C.circuit[i].ty == layerType::FCONN ? "_fc" : "");
        timer tmp;
        tmp.start();
        auto& cur = C.circuit[i];
        ptimer.start();
        p->sumcheckInit(alpha, beta);
        ptimer.stop();
        cout << "Peak Memory after sumcheckInit: " << get_vm_peak_kb_verf() << " KB" << endl;

        if (cur.zero_start_id < cur.size) {
            relu_rou.setByCSPRNG();
        } else
            relu_rou = F_ONE;
        F previousRandom = F_ZERO;
        if (C.circuit[i].ty == layerType::FCONN)  // substitute matrix with thaler13
        {
            now_fc_id = --fc_cnt;
            int n, m, k;
            n = ceilPow2BitLength(p->fc_input_row[now_fc_id]);
            k = ceilPow2BitLength(p->fc_col[now_fc_id]);
            m = ceilPow2BitLength(p->fc_row[now_fc_id]);
            r_u[i].resize(m + n);
            r_v[i].resize(m + k);
            for (int j = m; j < m + n; ++j)
                r_u[i][j] = r_u[i + 1][j - m + k];
            for (int j = m; j < m + k; ++j)
                r_v[i][j] = r_u[i + 1][j - m];
            for (int j = 0; j < m; j++) {
                r_u[i][j].setByCSPRNG();
                r_v[i][j] = r_u[i][j];
            }

            prev_u1 = final_claim_u1;
            previousRandom = F_ZERO;
            Fr sum = 0;
            ptimer.start();
            mat_timer.start();
            auto _tpa = vl_clk::now();
            Fr* pa = init_book_keeping(m, n, p->val[0], p->fc_input_id[now_fc_id], r_u[i]);
            g_lasso_init_timers.mm_pa += vl_secs(_tpa);
            auto _tpb = vl_clk::now();
            Fr* pb = init_book_keeping_fast(m, k, p->mat_val[now_fc_id], r_v[i], max_threads);
            g_lasso_init_timers.mm_pb += vl_secs(_tpb);
            g_lasso_init_timers.mm_layers += 1;
            auto _tsc = vl_clk::now();
            static const bool matmul_cpu = getenv("GKR_MATMUL_CPU") != nullptr;
            pair<Fr, Fr> oracle =
                matmul_cpu ? sum_check_product(pa, pb, m - 1, r_u[i].data(), prev_u1)
                           : zkgpt_gpu::gpu_matmul_product_sumcheck(pa, pb, m - 1, r_u[i].data(), prev_u1);
            gkr_prover_barrier();  // matmul kernels done before the clock stops
            g_lasso_init_timers.mm_sc += vl_secs(_tsc);
            mat_timer.stop();
            ptimer.stop();
            final_claim_u0[i] = oracle.first;
            final_claim_u1 = 0;
            final_claim_v0[i] = oracle.second;
            final_claim_v1 = 0;
        } else {
            timer normal_timer;
            normal_timer.start();
            ptimer.start();
            p->sumcheckInitPhase1(relu_rou);
            ptimer.stop();
            r_u[i].resize(cur.max_bl_u);
            for (int j = 0; j < cur.max_bl_u; ++j)
                r_u[i][j].setByCSPRNG();
            sc.start();
            for (i8 j = 0; j < cur.max_bl_u; ++j) {
                F cur_claim, nxt_claim;
                ptimer.start();
                quadratic_poly poly = p->sumcheckUpdate1(previousRandom);
                ptimer.stop();

                vtimer.start();
                cur_claim = poly.eval(F_ZERO) + poly.eval(F_ONE);
                nxt_claim = poly.eval(r_u[i][j]);
                vtimer.stop();
                if (cur_claim != previousSum) {
                    cerr << cur_claim << ' ' << previousSum << endl;
                    fprintf(stderr, "Verification fail, phase1, circuit %d, current bit %d\n", i, j);
                    return false;
                }

                previousRandom = r_u[i][j];
                previousSum = nxt_claim;
            }
            sc.stop();
            sc_t += sc.elapse_sec();
            prev_u1 = final_claim_u1;
            ptimer.start();
            p->sumcheckFinalize1(previousRandom, final_claim_u0[i], final_claim_u1);
            betaInitPhase1(i, alpha, beta, r_0, r_1, relu_rou);
            predicatePhase1(i);
            ptimer.stop();

            if (cur.need_phase2) {
                timer normal_timer2;
                normal_timer2.start();
                r_v[i].resize(cur.max_bl_v);
                for (int j = 0; j < cur.max_bl_v; ++j)
                    r_v[i][j].setByCSPRNG();

                ptimer.start();
                p->sumcheckInitPhase2();
                ptimer.stop();
                previousRandom = F_ZERO;
                sc.start();
                for (u32 j = 0; j < cur.max_bl_v; ++j) {
                    ptimer.start();
                    quadratic_poly poly = p->sumcheckUpdate2(previousRandom);
                    ptimer.stop();
                    vtimer.start();
                    if (poly.eval(F_ZERO) + poly.eval(F_ONE) != previousSum) {
                        fprintf(stderr,
                                "Verification fail, phase2, circuit level %d, current bit %d, total is %d\n",
                                i, j, cur.max_bl_v);
                        return false;
                    }
                    vtimer.stop();
                    previousRandom = r_v[i][j];
                    previousSum = poly.eval(previousRandom);
                }
                sc.stop();
                sc_t += sc.elapse_sec();
                prev_v1 = final_claim_v1;
                ptimer.start();
                p->sumcheckFinalize2(previousRandom, final_claim_v0[i], final_claim_v1);
                betaInitPhase2(i);
                predicatePhase2(i);
                ptimer.stop();
            }
        }
        cout << "Peak Memory after sumcheckFinalize: " << get_vm_peak_kb_verf() << " KB" << endl;

        vtimer.start();
        if (C.circuit[i].ty != layerType::FCONN)  // for thaler13 no need to check this
        {
            F test_value =
                getFinalValue(final_claim_u0[i], final_claim_u1, final_claim_v0[i], final_claim_v1);
            if (previousSum != test_value) {
                std::cerr << "GKR layer mismatch at layer " << i << " type "
                          << static_cast<int>(C.circuit[i].ty) << ": " << test_value << ' ' << previousSum
                          << std::endl;
                return false;
            }
        }

        if (~cur.bit_length_u[1])
            alpha.setByCSPRNG();
        else
            alpha.clear();
        if ((~cur.bit_length_v[1]) || cur.ty == layerType::FFT)
            beta.setByCSPRNG();
        else
            beta.clear();
        previousSum = alpha * final_claim_u1 + beta * final_claim_v1;
        vtimer.stop();
        cout << "Peak Memory after layer sumcheck " << get_vm_peak_kb_verf() << " KB" << endl;
        r_0 = r_u[i].begin();
        r_1 = r_v[i].begin();
        beta_u.clear();
        beta_v.clear();
        tmp.stop();
    }

    // ptimer brackets every prover region here
    matrix_time = mat_timer.elapse_sec();  // total of every FCONN interval, counted once
    const double matmul_s = std::min(matrix_time, ptimer.elapse_sec());
    add_prover(PH_GKR_MATMUL, matmul_s);
    add_prover(PH_GKR_GENERIC, ptimer.elapse_sec() - matmul_s);
    verifier_time += vtimer.elapse_sec();

    return true;
}

bool verifier::verifyLasso() {
    timer ptimer, vtimer;
    auto& cur = C.circuit[0];
    cout << "Peak Memory before Lasso: " << get_vm_peak_kb_verf() << " KB" << endl;
    vector<F> sig_u(C.size - 1);
    for (int i = 0; i < C.size - 1; ++i)
        sig_u[i].setByCSPRNG();
    vector<F> sig_v(C.size - 1);
    for (int i = 0; i < C.size - 1; ++i)
        sig_v[i].setByCSPRNG();
    r_u[0].resize(cur.bit_length);
    for (int i = 0; i < cur.bit_length; ++i)
        r_u[0][i].setByCSPRNG();
    auto r_0 = r_u[0].begin();

    F previousSum = F_ZERO;
    for (int i = 1; i < C.size; ++i) {
        if (~C.circuit[i].bit_length_u[0])
            previousSum = previousSum + sig_u[i - 1] * final_claim_u0[i];
        if (~C.circuit[i].bit_length_v[0])
            previousSum = previousSum + sig_v[i - 1] * final_claim_v0[i];
    }
    ptimer.start();
    p->sumcheckLassoInit(sig_u, sig_v, r_u, r_v);
    ptimer.stop();
    cout << "Peak Memory after sumcheckLassoInit: " << get_vm_peak_kb_verf() << " KB" << endl;
    F previousRandom = F_ZERO;
    int n = C.circuit[0].bit_length;
    F gr = F_ZERO;
    // Whole Lasso grand-product sumcheck on the GPU, fed zkGPT's r_u[0]
    static const bool lasso_cpu = getenv("GKR_LASSO_CPU") != nullptr;
    if (getenv("GKR_LASSO_STATS")) {  // diagnostic: sparsity/extent of the two Lasso sources
        const size_t N = size_t(1) << n;
        const Fr *f = p->lasso_mult_v.data(), *g = p->val[0].data();
        size_t nzf = 0, nzg = 0, lastf = 0, lastg = 0;
        for (size_t k = 0; k < N; ++k) {
            if (!f[k].isZero()) {
                ++nzf;
                lastf = k;
            }
            if (!g[k].isZero()) {
                ++nzg;
                lastg = k;
            }
        }
        fprintf(stderr, "LASSO_STATS n=%d N=%zu f_nnz=%zu f_last=%zu g_nnz=%zu g_last=%zu\n", n, N, nzf,
                lastf, nzg, lastg);
    }
    if (!lasso_cpu) {
        F gamma = previousSum;
        ptimer.start();
        auto _tc = vl_clk::now();
        auto res = zkgpt_gpu::gpu_lasso_product_sumcheck(p->lasso_mult_v.data(), p->val[0].data(), n,
                                                         r_u[0].data(), gamma);
        g_lasso_init_timers.lasso_sc_call += vl_secs(_tc);
        auto _tbar = vl_clk::now();
        gkr_prover_barrier();  // sumcheck kernels done before the clock stops
        g_lasso_init_timers.lasso_barrier += vl_secs(_tbar);
        ptimer.stop();
        cout << "[timing] lasso_sumcheck_s=" << ptimer.elapse_sec() << " (n=" << n << ")" << endl;
        eval_in = res.second;                  // val[0](r), replaces read_2[0]
        previousSum = res.first * res.second;  // f(r)*g(r), final reduced claim
    } else {
        // Precompute per-level lengths using the padded domain
        if (p->total[1] == 0 || C.circuit[0].bit_length < 0) {
            throw std::runtime_error("verifyLasso: invalid input domain for lasso");
        }
        std::vector<size_t> level_len(n + 1);
        level_len[0] = static_cast<size_t>(p->total[1]);  // padded domain size (power of two)
        cout << "Lasso initial level length (padded): " << level_len[0] << " " << n << endl;
        for (int i = 1; i <= n; ++i) {
            level_len[i] = (level_len[i - 1] + 1) >> 1;
        }
        // Scratch buffers for lasso reductions (avoid one giant allocation)
        Fr* read_1 = p->lasso_mult_v.data();
        Fr* read_2 = p->val[0].data();
        std::vector<Fr> buf1, buf2, buf1_alt, buf2_alt;
        auto alloc_buffers = [&](size_t len, bool use_alt) {
            auto& w1 = use_alt ? buf1_alt : buf1;
            auto& w2 = use_alt ? buf2_alt : buf2;
            w1.resize(len);
            w2.resize(len);
            return std::pair<Fr*, Fr*>{w1.data(), w2.data()};
        };
        auto release_buffers = [&](Fr* ptr) {
            if (ptr == buf1.data()) {
                std::vector<Fr>().swap(buf1);
                std::vector<Fr>().swap(buf2);
            } else if (ptr == buf1_alt.data()) {
                std::vector<Fr>().swap(buf1_alt);
                std::vector<Fr>().swap(buf2_alt);
            }
        };
        bool use_alt = false;

        for (int i = 0; i <= n; i++) {
            ptimer.start();
            Fr A = 0, B = 0, C = 0;
            if (i == 0) {
                // Clamp in size_t, THEN narrow
                int thread_n = static_cast<int>(std::min<size_t>(static_cast<size_t>(max_threads),
                                                                 std::max<size_t>(size_t(1), level_len[0])));
                const size_t chunk = (level_len[0] + thread_n - 1) / thread_n;
                std::vector<Fr> Aa(thread_n), Bb(thread_n), Cc(thread_n);
                Aa.assign(thread_n, Fr(0));
                Bb.assign(thread_n, Fr(0));
                Cc.assign(thread_n, Fr(0));
                std::vector<std::thread> threads;
                threads.reserve(thread_n);
                int used_threads = 0;
                for (int j = 0; j < thread_n; j++) {
                    size_t start = static_cast<size_t>(j) * chunk;
                    size_t end = std::min(level_len[0], start + chunk);
                    if (start >= end)
                        break;
                    ++used_threads;
                    threads.emplace_back(sc_last_worker_first, std::ref(Aa[j]), std::ref(Bb[j]),
                                         std::ref(Cc[j]), read_1, read_2, start, end);
                }
                for (auto& t : threads)
                    t.join();
                for (int j = 0; j < used_threads; j++) {
                    A += Aa[j];
                    B += Bb[j];
                    C += Cc[j];
                }
            } else if (i <= 9) {
                timer tt;
                tt.start();
                auto write_pair = alloc_buffers(level_len[i], use_alt);
                Fr* write_1 = write_pair.first;
                Fr* write_2 = write_pair.second;
                int thread_n = static_cast<int>(std::min<size_t>(static_cast<size_t>(max_threads),
                                                                 std::max<size_t>(size_t(1), level_len[i])));
                const size_t chunk = (level_len[i] + thread_n - 1) / thread_n;
                std::vector<Fr> Aa(thread_n), Bb(thread_n), Cc(thread_n);
                Aa.assign(thread_n, Fr(0));
                Bb.assign(thread_n, Fr(0));
                Cc.assign(thread_n, Fr(0));
                std::vector<std::thread> threads;
                threads.reserve(thread_n);
                int used_threads = 0;
                for (int j = 0; j < thread_n; j++) {
                    size_t start = static_cast<size_t>(j) * chunk;
                    size_t end = std::min(level_len[i], start + chunk);
                    if (start >= end)
                        break;
                    ++used_threads;
                    threads.emplace_back(sc_last_worker, std::ref(Aa[j]), std::ref(Bb[j]), std::ref(Cc[j]),
                                         read_1, read_2, write_1, write_2, start, end, r_u[0][i - 1]);
                }
                for (auto& t : threads)
                    t.join();
                tt.stop();
                for (int j = 0; j < used_threads; j++) {
                    A += Aa[j];
                    B += Bb[j];
                    C += Cc[j];
                }
                Fr* prev_read = read_1;
                read_1 = write_1;
                read_2 = write_2;
                use_alt = !use_alt;
                release_buffers(prev_read);
            } else {
                auto write_pair = alloc_buffers(level_len[i], use_alt);
                Fr* write_1 = write_pair.first;
                Fr* write_2 = write_pair.second;
                for (size_t kp = 0; kp < level_len[i]; kp++) {
                    Fr &c = read_1[kp << 1 | 1], &d = read_1[kp << 1];
                    if (c == d)
                        write_1[kp] = d;
                    else
                        write_1[kp] = (c - d) * r_u[0][i - 1] + d;
                    Fr &a = read_2[kp << 1 | 1], &b = read_2[kp << 1];
                    if (a == b)
                        write_2[kp] = b;
                    else
                        write_2[kp] = (a - b) * r_u[0][i - 1] + b;
                    if (kp & 1) {
                        B += write_1[kp] * write_2[kp];
                        C += (write_1[kp] + write_1[kp] - write_1[kp - 1]) *
                             (write_2[kp] + write_2[kp] - write_2[kp - 1]);
                    } else
                        A += write_1[kp] * write_2[kp];
                }
                Fr* prev_read = read_1;
                read_1 = write_1;
                read_2 = write_2;
                use_alt = !use_alt;
                release_buffers(prev_read);
            }
            ptimer.stop();
            vtimer.start();
            Fr aa = (C + A) / 2 - B, bb = 2 * B - C / 2 - 3 * A / 2, cc = A;
            Fr g1 = aa + bb + cc, g0 = cc;
            if (i != n)
                previousSum = aa * r_u[0][i] * r_u[0][i] + bb * r_u[0][i] + cc;
            vtimer.stop();
        }

        eval_in = read_2[0];
    }

    beta_g.resize(1ULL << cur.bit_length);
    ptimer.start();

    {
        auto _tg = vl_clk::now();
        initBetaTable(beta_g, cur.bit_length, r_0, F_ONE);
        g_lasso_init_timers.bind_beta_g += vl_secs(_tg);
    }
    for (int i = 1; i < C.size; ++i) {
        if (~C.circuit[i].bit_length_u[0]) {
            auto _tb = vl_clk::now();
            beta_u.resize(1ULL << C.circuit[i].bit_length_u[0]);
            initBetaTable(beta_u, C.circuit[i].bit_length_u[0], r_u[i].begin(), sig_u[i - 1]);
            g_lasso_init_timers.bind_beta_uv += vl_secs(_tb);
            auto _tj = vl_clk::now();
            for (u32 j = 0; j < C.circuit[i].size_u[0]; ++j)
                gr = gr + beta_g[C.circuit[i].ori_id_u[j]] * beta_u[j];
            g_lasso_init_timers.bind_gather += vl_secs(_tj);
            g_lasso_init_timers.gathered += C.circuit[i].size_u[0];
        }

        if (~C.circuit[i].bit_length_v[0]) {
            auto _tb = vl_clk::now();
            beta_v.resize(1ULL << C.circuit[i].bit_length_v[0]);
            initBetaTable(beta_v, C.circuit[i].bit_length_v[0], r_v[i].begin(), sig_v[i - 1]);
            g_lasso_init_timers.bind_beta_uv += vl_secs(_tb);
            auto _tj = vl_clk::now();
            for (u32 j = 0; j < C.circuit[i].size_v[0]; ++j)
                gr = gr + beta_g[C.circuit[i].ori_id_v[j]] * beta_v[j];
            g_lasso_init_timers.bind_gather += vl_secs(_tj);
            g_lasso_init_timers.gathered += C.circuit[i].size_v[0];
        }
    }
    ptimer.stop();
    cout << "Peak Memory after Lasso: " << get_vm_peak_kb_verf() << " KB" << endl;
    add_prover(PH_LASSO, ptimer.elapse_sec());

    beta_u.clear();
    beta_v.clear();

    vtimer.start();
    auto result = (eval_in * gr == previousSum);
    vtimer.stop();
    verifier_time += vtimer.elapse_sec();

    if (result)
        fprintf(stderr, "All verification passed!!\n");

    beta_g.clear();
    beta_u.clear();
    beta_v.clear();
    r_u.resize(1);
    r_v.clear();

    sig_u.clear();
    sig_v.clear();
    return true;
}
__int128 convertx(Fr x) {
    int sign = 0;
    Fr abs;
    if (x.isNegative()) {
        sign = 1;
        abs = -x;
    } else
        abs = x;
    uint8_t bf[16] = {0};  // 64 bit
    int size = abs.getLittleEndian(bf, 16);
    __int128 V = 0;
    for (int j = size - 1; j >= 0; j--) {
        V = (V << 8) + bf[j];
    }
    if (sign)
        V = -V;
    return V;
}
bool verifier::openCommit() {
    cout << "Peak Memory before opening input commitment: " << get_vm_peak_kb_verf() << " KB" << endl;
    auto open_one = [&](Commit_return& c, std::vector<Fr>& r, const Fr& eval) {
        int half = c.l / 2;
        int other = c.l - half;
        Fr* Lp = new Fr[1 << other];  // L has size 2^{l - l/2}
        Fr* Rp = new Fr[1 << half];   // R has size 2^{l/2}
        brute_force_compute_LR(Lp, Rp, r.data(), c.l);
        auto _t_diag = std::chrono::steady_clock::now();
        auto tim = hyrax::open(c.w, r.data(), eval, c.G, c.g, Lp, Rp, c.comm, c.l);
        std::cerr << "OPEN_SEG l=" << c.l << " rownum=" << (1 << half) << " colnum=" << (1 << other)
                  << " dur_s="
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - _t_diag).count()
                  << std::endl;
        p->open_time += tim.first.duration_sec;
        add_prover(PH_HYRAX_OPEN, tim.first.duration_sec);
        verifier_time += tim.second.duration_sec;
        delete[] Lp;
        delete[] Rp;
    };

    auto segment_mask = [&](size_t start, int local_bits, int total_bits) {
        Fr mask = F_ONE;
        size_t high = start >> local_bits;
        for (int b = local_bits; b < total_bits; ++b) {
            bool bit = high & 1U;
            mask *= bit ? r_u[0][b] : (F_ONE - r_u[0][b]);
            high >>= 1U;
        }
        return mask;
    };

    if (!p->input_commit_segments.empty()) {
        std::cout << "Segmented input commitments opening" << std::endl;
        if (r_u.empty() || r_u[0].empty())
            return false;
        const int total_bit_len = static_cast<int>(r_u[0].size());
        Fr recombined = F_ZERO;
        for (auto& seg : p->input_commit_segments) {
            Commit_return* c = seg.external_commit ? seg.external_commit : &seg.commit;
            bool public_segment = (seg.label == "public" && !seg.external_commit && !seg.commit.comm);
            int local_bits = public_segment ? ceilPow2BitLength(static_cast<u32>(seg.len)) : c->l;
            if (local_bits < 0 || local_bits > total_bit_len) {
                std::cerr << "Invalid input segment bit length: " << seg.label << std::endl;
                return false;
            }
            size_t padded_len = size_t(1) << local_bits;
            if (seg.start % padded_len != 0) {
                std::cerr << "Unaligned input commitment segment: " << seg.label << std::endl;
                return false;
            }
            std::vector<Fr> r_seg(r_u[0].begin(), r_u[0].begin() + local_bits);
            Fr eval_seg;
            if (public_segment) {
                eval_seg = brute_force_compute_eval(p->val[0].data() + seg.start, r_seg.data(), local_bits,
                                                    max_threads);
            } else {
                if (!c || !c->comm || !c->w) {
                    std::cerr << "Invalid input commitment segment: " << seg.label << std::endl;
                    return false;
                }
                eval_seg = brute_force_compute_eval(c->w, r_seg.data(), local_bits, max_threads);
                // prove_dot_product exit(2)s on a bad opening without naming the segment
                static const bool trace_seg = getenv("GKR_TRACE_SEGMENTS") != nullptr;
                if (trace_seg) {
                    std::cerr << "[seg] opening label=" << seg.label << " start=" << seg.start
                              << " len=" << seg.len << " l=" << local_bits << std::endl;
                }
                open_one(*c, r_seg, eval_seg);
            }
            recombined += eval_seg * segment_mask(seg.start, local_bits, total_bit_len);
        }
        if (recombined != eval_in) {
            std::cerr << "Segmented input commitment recombination failed" << std::endl;
            return false;
        }
        return true;
    }

    if (input_not_committed && !comm_input.l && comm.l) {
        std::cout << "Public input plus committed predicate opening" << std::endl;
        int img_bit_len = ceilPow2BitLength(static_cast<u32>(p->inp_end_id));
        int pred_bit_len = comm.l;
        if (img_bit_len < 0 || r_u[0].size() < static_cast<size_t>(std::max(img_bit_len, pred_bit_len))) {
            fprintf(stderr, "r_u[0] is too short for public input/predicate commitments.\n");
            return false;
        }
        std::vector<Fr> r_img(r_u[0].begin(), r_u[0].begin() + img_bit_len);
        std::vector<Fr> r_pred(r_u[0].begin(), r_u[0].begin() + pred_bit_len);

        Fr eval_img = brute_force_compute_eval(p->val[0].data(), r_img.data(), img_bit_len, max_threads);
        Fr eval_pred = brute_force_compute_eval(comm.w, r_pred.data(), pred_bit_len, max_threads);

        u8 total_bit_len = r_u[0].size();
        Fr img_mask = F_ONE;
        for (u8 b = img_bit_len; b < total_bit_len; ++b)
            img_mask *= (F_ONE - r_u[0][b]);

        Fr pred_mask = F_ONE;
        size_t offset = static_cast<size_t>(p->inp_end_id) >> pred_bit_len;
        for (u8 hb = 0; hb < total_bit_len - pred_bit_len; ++hb) {
            bool bit = offset & 1U;
            pred_mask *= bit ? r_u[0][pred_bit_len + hb] : (F_ONE - r_u[0][pred_bit_len + hb]);
            offset >>= 1U;
        }
        Fr recombined = eval_img * img_mask + eval_pred * pred_mask;
        if (recombined != eval_in) {
            std::cerr << "Public input/predicate recombination failed" << std::endl;
            return false;
        }
        open_one(comm, r_pred, eval_pred);
        return true;
    }

    if ((separate_input || input_not_committed || input_committed_externally) && comm_input.l && comm.l) {
        std::cout << "Separate input commitments opening" << std::endl;
        int img_bit_len = comm_input.l;
        int pred_bit_len = comm.l;
        if (r_u[0].size() < static_cast<size_t>(std::max(img_bit_len, pred_bit_len))) {
            fprintf(stderr, "r_u[0] is too short for separate input commitments.\n");
            return false;
        }
        std::vector<Fr> r_img(r_u[0].begin(), r_u[0].begin() + img_bit_len);
        std::vector<Fr> r_pred(r_u[0].begin(), r_u[0].begin() + pred_bit_len);

        Fr eval_img = brute_force_compute_eval(comm_input.w, r_img.data(), img_bit_len, max_threads);
        Fr eval_pred = brute_force_compute_eval(comm.w, r_pred.data(), pred_bit_len, max_threads);

        u8 total_bit_len = r_u[0].size();
        Fr img_mask = F_ONE;
        for (u8 b = img_bit_len; b < total_bit_len; ++b)
            img_mask *= (F_ONE - r_u[0][b]);

        Fr pred_mask = F_ONE;
        size_t offset = static_cast<size_t>(p->inp_end_id) >> pred_bit_len;
        for (u8 hb = 0; hb < total_bit_len - pred_bit_len; ++hb) {
            bool bit = offset & 1U;
            pred_mask *= bit ? r_u[0][pred_bit_len + hb] : (F_ONE - r_u[0][pred_bit_len + hb]);
            offset >>= 1U;
        }
        Fr recombined = eval_img * img_mask + eval_pred * pred_mask;
        if (recombined != eval_in) {
            std::cerr << "Separate input commitment recombination failed" << std::endl;
            return false;
        }

        if (comm_input.comm != nullptr) {
            open_one(comm_input, r_img, eval_img);
        }
        open_one(comm, r_pred, eval_pred);
        cout << "Peak Memory after opening input commitment: " << get_vm_peak_kb_verf() << " KB" << endl;
        return true;
    }

    int half = comm.l / 2;
    int other = comm.l - half;
    Fr* Lp = new Fr[1 << other];
    Fr* Rp = new Fr[1 << half];
    timer verify_input;
    verify_input.start();
    brute_force_compute_LR(Lp, Rp, r_u[0].data(), comm.l);
    auto tim = hyrax::open(comm.w, r_u[0].data(), eval_in, comm.G, comm.g, Lp, Rp, comm.comm, comm.l);
    p->open_time += tim.first.duration_sec;
    add_prover(PH_HYRAX_OPEN, tim.first.duration_sec);
    verifier_time += tim.second.duration_sec;
    delete[] Lp;
    delete[] Rp;
    return true;
}

// Emits PROVER_BREAKDOWN / PROVER_TOTALS. residual MUST be ~0
void verifier::print_totals(double core, const double* phase, double range_s) {
    static const char* NM[PH_N] = {"gkr_generic", "gkr_matmul", "lasso", "hyrax_commit", "hyrax_open"};
    double sum = 0;
    cout << "PROVER_BREAKDOWN";
    for (int i = 0; i < PH_N; ++i) {
        cout << ' ' << NM[i] << '=' << phase[i];
        sum += phase[i];
    }
    cout << " sum=" << sum << " prover_core_s=" << core << " residual=" << (core - sum) << endl;
    // prover_core_s
    cout << "PROVER_TOTALS prover_core_s=" << core << " range_prove_s=" << range_s
         << " prover_total_s=" << (core + range_s) << endl;
}
