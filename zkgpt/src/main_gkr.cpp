#undef NDEBUG
#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <mcl/bn.hpp>
#include "config.hpp"

#include "circuit.h"
#include "neuralNetwork.hpp"
#include "verifier.hpp"
#include "models.hpp"
#include "global_var.hpp"
#include "range_prover.hpp"
#include "common.hpp"
#include "gpu_range_prover.hpp"
#include "gpu_deg3_sumcheck.hpp"   // deg3 microbench (Experiment 3 nonlinear panel)
#include "gkr_gpu.hpp"
#include "gkr_pred_gpu.hpp"        // GPU predicate reduction
#include <chrono>
#include <vector>
#include <cstdio>
#include <fstream>
#include <sys/time.h>
#ifdef GKR_UVM
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#endif

using namespace mcl::bn;
using namespace std;

#ifdef GKR_UVM
// Single-instance guard (UVM build only). Two concurrent managed/XNACK processes
// wedge the host with page-migration contention, so take
// an exclusive flock and refuse to start otherwise. The fd is leaked deliberately:
// the lock dies with the process, so a crash cannot strand it.
// Path: ZKGPT_UVM_LOCK. Bypass: ZKGPT_UVM_NO_LOCK=1.
static void uvm_single_instance_guard() {
    if (const char* off = getenv("ZKGPT_UVM_NO_LOCK"); off && off[0] == '1') {
        fprintf(stderr, "[uvm-lock] disabled via ZKGPT_UVM_NO_LOCK=1 — "
                        "concurrent instances may soft-lock the host\n");
        return;
    }
    const char* path = getenv("ZKGPT_UVM_LOCK");
    if (!path || !path[0]) path = "/tmp/zkgpt_gpu_uvm.lock";

    int fd = open(path, O_RDWR | O_CREAT, 0666);
    if (fd < 0) {
        fprintf(stderr, "[uvm-lock] cannot open lock file %s: %s — refusing to "
                        "run (set ZKGPT_UVM_LOCK to a writable path or "
                        "ZKGPT_UVM_NO_LOCK=1 to bypass)\n",
                path, strerror(errno));
        exit(1);
    }
    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        if (errno == EWOULDBLOCK)
            fprintf(stderr, "[uvm-lock] another zkgpt_gpu_uvm instance is already "
                            "running (lock: %s). Refusing to start a second — "
                            "concurrent UVM runs soft-lock the host. Wait for it "
                            "to finish, or set ZKGPT_UVM_LOCK to a different path "
                            "if you know the other run targets an isolated host.\n",
                    path);
        else
            fprintf(stderr, "[uvm-lock] flock(%s) failed: %s — refusing to run\n",
                    path, strerror(errno));
        close(fd);
        exit(1);
    }
    // Record our pid for observability; keep fd open (lock held until exit).
    if (ftruncate(fd, 0) == 0) {
        char buf[32];
        int n = snprintf(buf, sizeof buf, "%ld\n", (long)getpid());
        if (n > 0) { ssize_t w = write(fd, buf, (size_t)n); (void)w; }
    }
}
#endif  // GKR_UVM

// One PHASE line per pipeline stage for tools/run_llm_demo.sh. Host memory from
// /proc/self/status (VmRSS current, VmHWM peak); GPU memory is attributed by the
// script polling the device against each phase's [t0_us, t1_us] window.
static double phase_now_us() {
    timeval tv; gettimeofday(&tv, nullptr);
    return (double)tv.tv_sec * 1e6 + (double)tv.tv_usec;
}
static long phase_status_kb(const char *key) {
    std::ifstream f("/proc/self/status");
    std::string line;
    size_t klen = std::string(key).size();
    while (std::getline(f, line)) {
        if (line.compare(0, klen, key) == 0) {
            long v = 0;
            sscanf(line.c_str() + klen, "%ld", &v);
            return v;   // kB
        }
    }
    return -1;
}
static void emit_phase(const char *name, double t0_us, double t1_us) {
    long rss = phase_status_kb("VmRSS:");
    long hwm = phase_status_kb("VmHWM:");
    printf("PHASE name=%s t0_us=%.0f t1_us=%.0f time_s=%.6f "
           "host_rss_mib=%.1f host_peak_mib=%.1f\n",
           name, t0_us, t1_us, (t1_us - t0_us) / 1e6,
           rss / 1024.0, hwm / 1024.0);
    fflush(stdout);
}

int main(int argc, char** argv) {
#ifdef GKR_UVM
    // Enforce one UVM instance host-wide before touching the GPU or allocating
    // any managed memory (all bench modes below allocate managed too).
    uvm_single_instance_guard();
#endif
    int layers = 12, heads = 12, head_dim = 64, attn_dim = 768, lin_dim = 2304;
    int seq_len = 30;   // --len: sequence length, feeds circuit AND range prover

    // Parse flags (anywhere) and positional args (model dims then range params).
    {
        std::vector<int> pos;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto next_int = [&]{ return (i + 1 < argc) ? atoi(argv[++i]) : 0; };
            if      (a == "--only-range")    gkr_config.only_range    = true;
            else if (a == "--skip-range")    gkr_config.skip_range    = true;
            else if (a == "--cpu")           gkr_config.range_cpu     = true;
            else if (a == "--cpu-deg3")      gkr_config.deg3_cpu      = true;
            else if (a == "--cpu-open")      gkr_config.open_cpu      = true;
            else if (a == "--cpu-commit")    gkr_config.commit_cpu    = true;
            else if (a == "--uvm-prefetch")  gkr_config.uvm_prefetch  = true;
            else if (a == "--uvm-zerocopy")  gkr_config.uvm_zerocopy  = true;
            else if (a == "--dump-prefetch") gkr_config.dump_prefetch = true;
            else if (a == "--dump-callgraph")gkr_config.dump_callgraph= true;
            else if (a == "--dump-sizes")    gkr_config.dump_sizes    = true;
            else if (a == "--gpu-bookkeep")  gkr_config.gpu_bookkeep  = true;
            else if (a == "--corrupt-commit")gkr_config.corrupt_commit= true;
            else if (a == "--corrupt-layer") gkr_config.corrupt_layer = next_int();
            else if (a == "--matmul-l")      gkr_config.matmul_l      = next_int();
            else if (a == "--matmul-k")      gkr_config.matmul_k      = next_int();
            else if (a == "--lasso-l")       gkr_config.lasso_l       = next_int();
            else if (a == "--sc-tile-bits")  gkr_config.sc_tile_bits  = next_int();
            else if (a == "--len")           seq_len                  = next_int();
            else if (a == "--partitions")  gkr_config.range_partitions = next_int();
            else if (a == "--vram-gb")     gkr_config.vram_gb = (i + 1 < argc) ? atof(argv[++i]) : 64.0f;
            else if (a == "--bench-deg1")  { gkr_config.bench_deg1  = true; gkr_config.bench_n = next_int(); gkr_config.bench_k = next_int(); }
            else if (a == "--bench-deg3")  { gkr_config.bench_deg3  = true; gkr_config.bench_n = next_int(); }
            else if (a == "--bench-matmul"){ gkr_config.bench_matmul= true; gkr_config.bench_n = next_int(); gkr_config.bench_k = next_int(); }
            else if (!a.empty() && a[0] != '-') pos.push_back(atoi(a.c_str()));
        }
        if (pos.size() >= 5) { layers=pos[0]; heads=pos[1]; head_dim=pos[2]; attn_dim=pos[3]; lin_dim=pos[4]; }
        if (pos.size() >= 6) gkr_config.range_c    = pos[5];
        if (pos.size() >= 7) gkr_config.range_k    = pos[6];
        if (pos.size() >= 8) gkr_config.range_size  = pos[7];
    }
    // Witgen GPU arms. main_gkr is single-process (no fork), so unlike
    // main_gkr_split these can be read straight from the environment here --
    // there is no probe pass that has to stay HIP-free.
    gkr_config.witgen_gpu   = getenv("GKR_WITGEN_GPU")   != nullptr;
    gkr_config.witgen_check = getenv("GKR_WITGEN_CHECK") != nullptr;
    gkr_config.round_gpu    = getenv("GKR_ROUND_GPU")    != nullptr;
    gkr_config.round_check  = getenv("GKR_ROUND_CHECK")  != nullptr;
    gkr_config.gates_gpu    = getenv("GKR_GATES_GPU")    != nullptr;
    gkr_config.gates_check  = getenv("GKR_GATES_CHECK")  != nullptr;
    gkr_config.witgen_full  = getenv("GKR_WITGEN_FULL")  != nullptr
                           || getenv("GKR_GELU_GPU")     != nullptr;
    gkr_config.witgen_full_check = getenv("GKR_WITGEN_FULL_CHECK") != nullptr
                           || getenv("GKR_GELU_CHECK")   != nullptr;

    initPairing(mcl::BN254);

    if (gkr_config.bench_deg1) {
        int n = gkr_config.bench_n > 0 ? gkr_config.bench_n : 20;
        int k = gkr_config.bench_k > 0 ? gkr_config.bench_k : 1;
        bool cpu = gkr_config.range_cpu;
        size_t N = (size_t)1 << n;
        std::vector<F> f(N, to_FE(1));               // all-ones MLE → Σ = N
        F sum = to_FE((uint64_t)N);
        std::vector<F> r;
        auto t0 = std::chrono::high_resolution_clock::now();
        zkgpt_gpu::gpu_sumcheck_deg1(n, f.data(), sum, k, r);
        double dt = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t0).count();
        cout << "BENCH_DEG1 mode=" << (cpu ? "CPU" : "GPU")
             << " n=" << n << " k=" << k << " time_s=" << dt << "\n";
        return 0;
    }

    // Exp-1: isolate one matmul product sumcheck (size s=2^n, k stages,
    // l_stage=ceil(n/k)) and split the time across expand/ptable/round/fold.
    if (gkr_config.bench_matmul) {
        int n = gkr_config.bench_n > 0 ? gkr_config.bench_n : 20;
        int k = gkr_config.bench_k > 0 ? gkr_config.bench_k : 1;
        if (k < 1) k = 1; if (k > n) k = n;
        int l_stage = (n + k - 1) / k;
        size_t N = (size_t)1 << n;
        std::vector<F> a(N), b(N);
        for (size_t i = 0; i < N; ++i) { a[i].setByCSPRNG(); b[i].setByCSPRNG(); }
        F gamma; gamma.clear();
        for (size_t i = 0; i < N; ++i) gamma += a[i] * b[i];
        std::vector<F> r(n);
        for (int i = 0; i < n; ++i) r[i].setByCSPRNG();
        bool cpu = gkr_config.range_cpu;
        zkgpt_gpu::g_matmul_timers.reset();
        zkgpt_gpu::g_matmul_timers.on = true;
        auto t0 = std::chrono::high_resolution_clock::now();
        if (cpu) {
            zkgpt_gpu::gpu_product_sumcheck_host(a.data(), b.data(), n, r.data(), gamma, l_stage);
        } else {
            zkgpt_gpu::gpu_product_sumcheck_host(a.data(), b.data(), n, r.data(), gamma, l_stage);
        }
        double dt = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t0).count();
        auto& T = zkgpt_gpu::g_matmul_timers;
        cout << "BENCH_MATMUL mode=" << (cpu ? "CPU" : "GPU")
             << " n=" << n << " k=" << k << " l_stage=" << l_stage
             << " s=" << N << " time_s=" << dt
             << " expand_s=" << T.expand << " ptable_s=" << T.ptable
             << " round_s=" << T.round << " fold_s=" << T.fold << "\n";
        return 0;
    }

    // Isolate one deg3 sumcheck over l variables (N=2^l); reports GPU-region time,
    // excluding the O(N) host setup. Needs 6*N*sizeof(gpu_fe) in VRAM (OOMs past
    // ~64 GB; the GKR_UVM build oversubscribes instead). S=0 because -DNDEBUG
    // compiles out the deg3 S-assert.
    if (gkr_config.bench_deg3) {
        using mcl::bn::Fr;
        int l = gkr_config.bench_n > 0 ? gkr_config.bench_n : 24;
        if (l < 1) l = 1; if (l > 30) l = 30;        // N=2^l, int-indexed in deg3
        size_t N = (size_t)1 << l;
        std::vector<Fr> f(N), g(N), r_pt(l), r_out;
        // g≡0 ⇒ Σ eq·f·g = 0 = S, so the deg3 prover's `assert(sum0+sum1==S)`
        // holds without an O(N) eq/S host build. Buffer sizes / memory traffic /
        // kernel work are identical to random data, so timing is representative.
        for (size_t i = 0; i < N; ++i) { f[i].setByCSPRNG(); g[i].clear(); }
        for (int i = 0; i < l; ++i) r_pt[i].setByCSPRNG();
        Fr S; S.clear();
        Fr cf, cg;
        zkgpt_gpu::g_deg3_timer.reset();
        zkgpt_gpu::g_deg3_timer.on = true;
        auto t0 = std::chrono::high_resolution_clock::now();
        zkgpt_gpu::gpu_sumcheck_deg3(l, r_pt.data(), f.data(), g.data(), S, r_out, cf, cg);
        double dt = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t0).count();
        cout << "BENCH_DEG3 mode=GPU l=" << l << " N=" << N
             << " time_s=" << dt
             << " gpu_s=" << zkgpt_gpu::g_deg3_timer.gpu << "\n";
        return 0;
    }

    // --only-range skips the GKR pipeline and runs just the standalone range check
    // (self-contained: generates its own inputs and Hyrax generators), so range
    // sweeps aren't burdened by the circuit build.
    bool only_range = gkr_config.only_range;
    // Snapshot of the verifier's phase attribution: `v` is scoped to the block
    // below, but the range time it needs is only known afterwards.
    double snap_core = 0, snap_phase[verifier::PH_N] = {0};
    bool have_snap = false;
    double range_s = 0;

    if (!only_range) {
    cerr << "[gkr] building combined transformer circuit: L=" << layers
         << " h=" << heads << " d=" << head_dim
         << " attn=" << attn_dim << " lin=" << lin_dim << "\n";

    double t_cb0 = phase_now_us();
    prover p;
    // Must be nonzero: initBetaTable divides by it (SIGFPE otherwise).
    p.max_threads = 32;
    LLM nn(layers, heads, head_dim, attn_dim, lin_dim, seq_len);
    timer circuit_timer;
    nn.create(p, /*merge=*/1, circuit_timer);   // builds circuit, commits input layer
    emit_phase("circuit_build", t_cb0, phase_now_us());

    // Soundness self-test hook: GKR_CORRUPT_LAYER=<i> flips one witness value in
    // val[i] AFTER the circuit/commitment are built. An intermediate layer (i>0)
    // must make verifyGKR REJECT (boundary chaining binds adjacent ops); layer 0
    // exercises the input-commitment opening.
    if (gkr_config.corrupt_layer >= 0) {
        int li = gkr_config.corrupt_layer;
        if (li >= 0 && li < (int)p.val.size() && !p.val[li].empty()) {
            // Target the first NON-ZERO entry: corrupting a padding/zero slot
            // is not a meaningful binding test (such slots aren't constrained).
            size_t idx = 0; bool found = false;
            for (size_t j = 0; j < p.val[li].size(); ++j)
                if (!p.val[li][j].isZero()) { idx = j; found = true; break; }
            if (found) {
                p.val[li][idx] += F(1);
                cerr << "[gkr] CORRUPTED val[" << li << "][" << idx
                     << "] (first nonzero, +1) for soundness self-test\n";
            } else {
                cerr << "[gkr] val[" << li << "] is all-zero; nothing to corrupt\n";
            }
        }
    }

    double t_gkr0 = phase_now_us();
    // Per-phase accounting for the WHOLE GKR proof (not just the microbench):
    // accumulate the four phases of the Blendy product-sumcheck
    // (expand/ptable/round/fold — each spans many kernel launches per stage, and
    // "round" also includes host-side D2H reductions) plus the deg3 sumcheck GPU
    // time, summed across every matmul layer's sumcheck.
    zkgpt_gpu::g_matmul_timers.reset(); zkgpt_gpu::g_matmul_timers.on = true;
    zkgpt_gpu::g_deg3_timer.reset();    zkgpt_gpu::g_deg3_timer.on    = true;
    verifier v(&p, p.C);
    v.max_threads = 32;
    // create() does not commit; without this comm.w stays null and openCommit
    // segfaults in compute_RT. Must run after the verifier ctor -- that ctor calls
    // p->init(), which resets cc/cc_input/cc_output.
    p.executeCommit(32);

    // Must run AFTER executeCommit: before it p.cc.w is null and the injection
    // would be silently skipped.
    // Input-commitment binding self-test: tamper with the OPENED witness data
    // (what a cheating prover controls) after the commitment is fixed. With
    // the IPA base check (prove_dot_product) must REJECT, because the opened
    // data no longer matches the per-row commitment captured at commit time.
    if (gkr_config.corrupt_commit && p.cc.w) {
        long n = 1L << p.cc.l;
        long idx = 0; while (idx < n && p.cc.w[idx].isZero()) ++idx;
        if (idx < n) {
            p.cc.w[idx] += F(1);
            cerr << "[gkr] CORRUPTED commit.w[" << idx << "] (+1) for input-binding self-test\n";
        }
    }

    const bool accepted = v.prove(32, circuit_timer);   // verifyGKR + verifyLasso + openCommit
    zkgpt_gpu::g_matmul_timers.on = false;
    zkgpt_gpu::g_deg3_timer.on    = false;
    emit_phase("verify_gkr", t_gkr0, phase_now_us());
    {
        auto& MT = zkgpt_gpu::g_matmul_timers;
        cout << "MATMUL_BREAKDOWN"
             << " expand_s="  << MT.expand
             << " ptable_s="  << MT.ptable
             << " round_s="   << MT.round
             << " fold_s="    << MT.fold
             << " deg3_s="    << zkgpt_gpu::g_deg3_timer.gpu << "\n";
        fflush(stdout);
    }
    snap_core = v.prover_time;
    for (int i = 0; i < verifier::PH_N; ++i) snap_phase[i] = v.phase_s[i];
    have_snap = true;
    } // end if(!only_range)

    // ── NOVEL: GPU chunked-lookup range check (logup / Lasso) ───────────────
    // Standalone numeric range proof over the per-layer nonlinear constraints,
    // GPU-backed via blendy_linear (degree-1 staged Blendy). Skip with
    // GKR_SKIP_RANGE=1. Tune chunk bits (c) and Blendy stages (k) per layer via
    // GKR_RANGE_C / GKR_RANGE_K (see range_prover::build).
    if (!gkr_config.skip_range) {
        double t_rng0 = phase_now_us();
        range_prover rp(layers, heads, head_dim, attn_dim, lin_dim, seq_len,
                        /*threads=*/32, /*merged=*/1);
        rp.init();
        // Upstream build() takes no args; c/k/range_size are read from gkr_config.
        rp.build();
        double rt = rp.prove();
        emit_phase("range", t_rng0, phase_now_us());
        const char* mode = gkr_config.range_cpu ? "CPU" : "GPU";
        cout << "RANGE_RESULT mode=" << mode
             << " c=" << gkr_config.range_c
             << " k=" << gkr_config.range_k
             << " partitions=" << gkr_config.range_partitions
             << " time_s=" << rt << "\n";
        cerr << "[gkr] " << mode << " range-check (logup) prover time: " << rt << " s\n";
        range_s = rt;
    }
    // Totals go here, not inside prove(): the range prover deliberately stays AFTER
    // v.prove() (it is the peak-VRAM phase), so its time is only known now.
    if (have_snap) {
        verifier::print_totals(snap_core, snap_phase, range_s);
        gkr_sc_report();   // generic-sumcheck profile; mirrors main_gkr_split.cpp
    }

    // gkr_sc_end keeps its device buffers alive across layers (grow-only); release
    // them here, after every phase that could still read them has finished.
    gkr_sc_teardown();
    gkr_pred_report();
    gkr_pred_teardown();

    return 0;
}
