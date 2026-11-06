#undef NDEBUG
#include <iostream>
#include "phase_acct.hpp"
#include <memory>
#include <vector>
#include <map>  // probe memoisation, keyed on split_depth
#include <set>
#include <functional>  // unique-parts role dedup
#include <string>
#include <cctype>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mcl/bn.hpp>

#include <array>
#include <cerrno>
#include <fstream>  // GKR_PIN_CORES: /sys topology
#include <omp.h>    // omp_set_num_threads: a child's share of the thread pool
#include <sched.h>  // sched_setaffinity, cpu_set_t
#include <signal.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <poll.h>   // mem-gated release loop: wake on a child event
#include <fcntl.h>  // O_NONBLOCK on the mem-gated stat pipe
#include <unistd.h>
#include <dirent.h>  // GKR_WITGEN_DEVICE: handoff-file cleanup
#include <sys/stat.h>

#include "circuit.h"
#include "neuralNetwork.hpp"
#include "gkr_gpu.hpp"
#include "gkr_pred_gpu.hpp"
#include "verifier.hpp"
#include "models.hpp"
#include "global_var.hpp"
#include "range_prover.hpp"
#include "timer.hpp"
#include "split_utils.hpp"
#include "boundary_link.hpp"  // zkMod: bind partition i+1's input to i's committed window
#include "gpu_sampler.hpp"
#include "roctx_scope.hpp"
#include "phase_log.hpp"
#include "utils.hpp"
#include "config.hpp"
#include "profile_util.hpp"
#include "gpu_uvm_alloc.hpp"      // per-partition device VRAM high-water tracking
#include "g1_msm.hpp"             // g1_msm_xfer_h2d_bytes/reset: zkmod-gpu's own H2D counter
#include "gpu_matmul_prover.hpp"  // MatmulPhaseTimers (GKR_MM_PROF)
#include "gpu_deg3_sumcheck.hpp"  // Deg3Timer (GKR_MM_PROF)
#include "witgen_timing.hpp"
// phase_log.hpp's GKR_PHASE_XFER hook: cumulative H2D of this process, both counters
extern "C" double zkgpt_phase_h2d_gib() {
    return zkgpt_gpu::gkr_xfer_h2d_gib() + (double)g1_msm_xfer_h2d_bytes() / (1024.0 * 1024.0 * 1024.0);
}
#include "witness_cache.hpp"  // GKR_WITGEN_DEVICE: load-hit check
#include <malloc.h>           // GKR_MEMACCT: mallinfo2
#include "alloc_trace.hpp"    // GKR_ALLOCTRACE: name the leaked allocation

static void memacct(const char* tag, long long idx) {
    if (!getenv("GKR_MEMACCT"))
        return;
    struct mallinfo2 mi = mallinfo2();
    const double G = 1073741824.0;
    fprintf(stderr,
            "[memacct] %s idx=%lld live_gib=%.2f uordblks_gib=%.2f hblkhd_gib=%.2f "
            "rss_gib=%.2f hwm_gib=%.2f\n",
            tag, idx, (mi.uordblks + mi.hblkhd) / G, mi.uordblks / G, mi.hblkhd / G,
            prof_status_kb("VmRSS:") / 1048576.0, prof_status_kb("VmHWM:") / 1048576.0);
    fflush(stderr);
    alloctrace::report(tag, idx);
}

using namespace mcl::bn;
using namespace std;

static constexpr int kMerge = 1;

// Parse a comma-separated list of ints, e.g. "2,5,5" -> {2,5,5} (whitespace ok)
static std::vector<int> parse_int_csv(const std::string& s) {
    std::vector<int> out;
    std::string cur;
    for (char ch : s) {
        if (ch == ',') {
            if (!cur.empty())
                out.push_back(atoi(cur.c_str()));
            cur.clear();
        } else if (!std::isspace(static_cast<unsigned char>(ch)))
            cur.push_back(ch);
    }
    if (!cur.empty())
        out.push_back(atoi(cur.c_str()));
    return out;
}

// Per-partition knob lookup: empty -> keep default
static int pick_part(const std::vector<int>& v, int def, std::size_t i) {
    if (v.empty())
        return def;
    return v[i < v.size() ? i : v.size() - 1];
}

// Loop-invariant config plus the per-partition knob vectors
struct SplitCfg {
    std::vector<int> split_depths;
    std::vector<long long> tiws;  // total_input_witness_sizes
    std::vector<u64> scs;         // shared_commit_sizes
    int embed_size = 0, lin_dim = 0, len = 0, num_threads = 32;
    std::vector<int> part_cs, part_mks, part_rks, part_lls, part_lks;
    // Knob defaults snapshotted before any partition mutates gkr_config
    int def_c = 0, def_mk = 0, def_rk = 0, def_ll = 0, def_lk = 0;
    // `len` is the CIRCUIT sequence width
    std::string vram_csv;          // vram-csv: device-wide sampler output
    double vram_interval = 0.001;  // vram-interval seconds
    bool witgen_only = false;      // witgen-only : build (+cache) every partition, skip
                                   // proving
    std::size_t nparts() const { return split_depths.size(); }
};

// Reports the device-wide peak over [t0,t1] and the CSV to query for anything finer
static void report_vram_peak(const SplitCfg& cfg, double t0_us, double t1_us, const char* tag) {
    if (cfg.vram_csv.empty())
        return;
    auto st = zkgpt_sampler::peak_in_window(cfg.vram_csv, t0_us, t1_us);
    unsigned long long busiest = 0;
    for (const auto& kv : st)
        busiest = std::max(busiest, kv.second.vram_peak_b);
    for (const auto& kv : st) {
        if (kv.second.vram_peak_b < (1ULL << 30))
            continue;  // idle card: <1 GiB
        std::cout << "GPU_SAMPLE phase=" << tag << " pci=" << kv.first
                  << " vram_peak_mib=" << kv.second.vram_peak_b / 1048576.0
                  << " gpu_busy_max=" << kv.second.busy_max << " gpu_busy_mean=" << kv.second.busy_mean()
                  << " samples=" << kv.second.samples << std::endl;
    }
    std::cout << "GPU_SAMPLE_MAX phase=" << tag << " vram_peak_mib=" << busiest / 1048576.0
              << " csv=" << cfg.vram_csv << std::endl;
}

static void apply_part_knobs(const SplitCfg& cfg, std::size_t i) {
    gkr_config.range_c = pick_part(cfg.part_cs, cfg.def_c, i);
    gkr_config.matmul_k = pick_part(cfg.part_mks, cfg.def_mk, i);
    gkr_config.range_k = pick_part(cfg.part_rks, cfg.def_rk, i);
    gkr_config.lasso_l = pick_part(cfg.part_lls, cfg.def_ll, i);
    gkr_config.lasso_k = pick_part(cfg.part_lks, cfg.def_lk, i);
}

struct PartResult {
    double standalone_commit = 0, prover_core = 0, output_commit = 0, verifier = 0, proof_kb = 0;
};

struct PartitionState {
    std::unique_ptr<prover> p;
    std::unique_ptr<neuralNetwork> net;
    timer circuit_timer;
    // Real witness-generation wall time. circuit_timer does NOT measure this
    double build_s = 0;
    double fc_s = 0;
    double gates_s = 0;

    // Witgen-phase peaks, sampled at the build/prove seam in build_partition()
    double witgen_gpu_peak_mib = 0;
    double witgen_host_peak_mib = 0;
    double witgen_resident_mib = 0;
    double witgen_h2d_gib = 0;
    double witgen_d2h_gib = 0;

    // create()'s boundary timers
    zkgpt_witgen::Accum wg;
    double pad_s = 0;
    // Elements of the carry this partition was built from (0 for partition 0)
    long long carry_n = 0;
};

// GKR_TAMPER_CARRY=<i>
static void maybe_tamper_carry(SplitBlockCarryState& c, std::size_t i) {
    const char* e = getenv("GKR_TAMPER_CARRY");
    if (!e || (std::size_t)atoi(e) != i || c.input_data.empty())
        return;
    const std::size_t k = c.input_data.size() / 3;
    c.input_data[k] = c.input_data[k] + F(1);
    std::cerr << "[split] GKR_TAMPER_CARRY: partition " << i << " carry[" << k << "] += 1\n";
}

// GKR_DUMP_CARRY=1
static unsigned long long fp_mix(unsigned long long h, unsigned long long v) {
    h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h;
}
static unsigned long long fp_fvec(const std::vector<F>& v) {
    static_assert(sizeof(F) % sizeof(unsigned long long) == 0, "F must be u64-aligned");
    constexpr std::size_t kU = sizeof(F) / sizeof(unsigned long long);
    unsigned long long h = 1469598103934665603ULL;
    unsigned long long buf[kU];
    for (const auto& x : v) {
        std::memcpy(buf, &x, sizeof(buf));
        for (std::size_t j = 0; j < kU; ++j)
            h = fp_mix(h, buf[j]);
    }
    return fp_mix(h, v.size());
}

// A forked child's share of the run's thread pool
static int g_child_threads = 0;
static void child_threads_set(int t) {
    g_child_threads = std::max(1, t);
    omp_set_num_threads(g_child_threads);
}

static bool build_partition(const SplitCfg& cfg, std::size_t i, SplitBlockCarryState* carry_in,
                            PartitionState& st, std::function<void(const prover&)> on_values_ready = {},
                            std::function<void(int&, int&)> on_before_values = {}) {
    const bool is_first = (i == 0);
    const bool is_last = (i + 1 == cfg.nparts());
    const bool output_private = !is_last;
    const bool input_private = !is_first;
    const bool input_separate = is_first && !is_last;

    zkgpt_phase::set_part((long long)i);
    ZKGPT_ROCTX_SCOPEF("part%zu_build", i);
    ZKGPT_PHASE_SCOPE("build");
    apply_part_knobs(cfg, i);

    // readFconWeight fills weights with rand()%1024
    if (const char* s = getenv("ZKGPT_WEIGHT_SEED"))
        srand((unsigned)(atoi(s) + 7919u * (unsigned)i));

    st.p.reset(new prover());
    st.p->max_threads = g_child_threads > 0 ? g_child_threads : cfg.num_threads;
    st.net.reset(new LLMSplitBlock(cfg.split_depths[i], cfg.embed_size / 64, 64, cfg.embed_size, cfg.lin_dim,
                                   cfg.len, output_private, input_private, /*input_separate=*/is_first,
                                   nullptr, nullptr));
    if (carry_in) {
        if (!on_before_values)
            maybe_tamper_carry(*carry_in, i);  // prebuild: tampered in the hook
        st.net->initStateFromCarry(*carry_in);
        st.carry_n = carry_in->input_offset == 0 ? (long long)carry_in->input_data.size() : 0;
    }
    st.net->on_values_ready = std::move(on_values_ready);
    st.net->on_before_values = std::move(on_before_values);
    // Probed layout
    if (cfg.tiws[i] > 0)
        st.net->setTotalInputWitnessSize(cfg.tiws[i]);
    if (!is_first)
        st.p->include_input_commit_size = false;

    // create() STARTS circuit_timer and verifier::prove() stops it
    zkgpt_witgen::acc().clear();
    zkgpt_gpu::gkr_vram_peak_reset();
    zkgpt_gpu::gkr_xfer_reset();
    zkgpt_gpu::gkr_xfer_time_reset();
    g1_msm_xfer_reset();
    const double t_build0 = prof_now_us();
    {
        ZKGPT_PHASE_SCOPE("witgen");
        st.net->create(*st.p, kMerge, st.circuit_timer, (int)i);
    }
    st.build_s = (prof_now_us() - t_build0) / 1e6;
    st.wg = zkgpt_witgen::acc();
    st.fc_s = zkgpt_witgen::acc().fc_s;
    st.gates_s = st.build_s - st.fc_s;

    st.witgen_gpu_peak_mib = zkgpt_gpu::gkr_vram_peak_mib();
    st.witgen_host_peak_mib = prof_status_kb("VmHWM:") / 1024.0;
    st.witgen_resident_mib = prof_status_kb("VmRSS:") / 1024.0;
    st.witgen_h2d_gib =
        zkgpt_gpu::gkr_xfer_h2d_gib() + (double)g1_msm_xfer_h2d_bytes() / (1024.0 * 1024.0 * 1024.0);
    st.witgen_d2h_gib = zkgpt_gpu::gkr_xfer_d2h_gib();
    if (getenv("GKR_XFER_TIME"))
        fprintf(stderr, "[xfer-time] witgen h2d_s=%.3f d2h_s=%.3f of build_s=%.3f\n",
                zkgpt_gpu::gkr_xfer_h2d_s_ref(), zkgpt_gpu::gkr_xfer_d2h_s_ref(), st.build_s);
    prof_reset_peak_rss();
    zkgpt_gpu::gkr_xfer_reset();
    g1_msm_xfer_reset();

    if (!is_last) {
        const double tp0 = prof_now_us();
        pad_output_layer(*st.p, cfg.scs[i]);
        st.pad_s = (prof_now_us() - tp0) / 1e6;
    }

    if (getenv("GKR_DUMP_CARRY")) {
        unsigned long long wf = 1469598103934665603ULL;
        for (const auto& lv : st.p->val)
            wf = fp_mix(wf, fp_fvec(lv));
        std::cout
            << "CARRY_FP idx=" << i << " carry_in=" << (carry_in ? fp_fvec(carry_in->input_data) : 0ULL)
            << " carry_in_len=" << (carry_in ? carry_in->len : -1)
            << " c_off=" << (carry_in ? carry_in->input_offset : -1)
            << " c_fdim=" << (carry_in ? carry_in->feature_dim : -1)
            << " c_ce=" << (carry_in ? carry_in->input_c : -1) << "," << (carry_in ? carry_in->input_e : -1)
            << " carry_in_mib=" << (carry_in ? carry_in->input_data.size() * sizeof(F) / 1048576.0 : 0.0)
            << " val0_mib=" << (st.p->val.empty() ? 0.0 : st.p->val[0].size() * sizeof(F) / 1048576.0)
            << " witness=" << wf << " out_fp=" << fp_fvec(st.p->val.back())
            << " out_n=" << st.p->val.back().size() << " out_nz="
            << std::count_if(st.p->val.back().begin(), st.p->val.back().end(),
                             [](const F& x) { return !x.isZero(); })
            << " in_prefix_fp="
            << (is_first || st.p->val.empty()
                    ? 0ULL
                    : fp_fvec(std::vector<F>(
                          st.p->val[0].begin(),
                          st.p->val[0].begin() + std::min<std::size_t>(cfg.scs[i - 1], st.p->val[0].size()))))
            << " nlayers=" << st.p->val.size() << " first=" << (int)is_first << " last=" << (int)is_last
            << " outpriv=" << (int)output_private << " inpriv=" << (int)input_private
            << " insep=" << (int)input_separate << " tiws=" << cfg.tiws[i] << " scs=" << cfg.scs[i]
            << " pid=" << (long)getpid() << std::endl;
    }
    return true;
}

// Per-job body: runs after the instance is built.
static bool prove_built_partition(const SplitCfg& cfg, std::size_t i, PartitionState& st, PartResult& res,
                                  int threads) {
    st.p->max_threads = threads;  // concurrent/--batched build in the parent at the full budget
    const bool is_first = (i == 0);
    const bool is_last = (i + 1 == cfg.nparts());
    const bool is_split = !is_first || !is_last;

    prover* p = st.p.get();
    const double t_part0 = prof_now_us();
    zkgpt_phase::set_part((long long)i);
    // Deliberately ZKGPT_PHASE and not ZKGPT_PHASE_SCOPE
    ZKGPT_PHASE("prove");
    zkgpt_acct::reset();
    zkgpt_gpu::gkr_xfer_wall_reset();
    zkgpt_gpu::gkr_vram_peak_reset();
    zkgpt_gpu::gkr_xfer_reset();
    g1_msm_xfer_reset();
    ZKGPT_ROCTX_SCOPEF("part%zu", i);

    // MUST be constructed before ANY commit work
    double t_standalone = 0, t_output = 0, t_range_build = 0, t_range = 0, t_exec = 0, t_gkr = 0;
    verifier v(p, p->C);
    // The thread count is read off the member, not prove()'s argument
    v.max_threads = threads;

    // input_copy is never populated
    if (!is_first && p->inp_end_id > 0 && p->inp_end_id < static_cast<i64>(p->val[0].size())) {
        ZKGPT_PHASE_SCOPE("standalone_input_commit");
        zkgpt_witgen::Scope _ts(t_standalone);
        timer prefix_commit_timer;
        // Same commitment as commit_values_standalone(prefix)
        p->cc_input = p->commitInputPrefixFast((size_t)p->inp_end_id, threads, &prefix_commit_timer);
        // Self-allocated
        p->cc_input_owned = true;
        p->cc_input_owns_gens = true;
        res.standalone_commit += prefix_commit_timer.elapse_sec();
        p->input_committed_externally = true;
    }

    if (!is_last) {
        ZKGPT_PHASE_SCOPE("output_commit");
        zkgpt_witgen::Scope _ts(t_output);
        commit_private_output_for_next(*p, cfg.scs[i], threads);
    }

    range_prover rp(cfg.split_depths[i], cfg.embed_size / 64, 64, cfg.embed_size, cfg.lin_dim, cfg.len,
                    threads,
                    /*merged=*/kMerge);
    timer rp_build_timer;
    rp_build_timer.start();
    {
        ZKGPT_PHASE_SCOPE("range_build");
        zkgpt_witgen::Scope _ts(t_range_build);
        rp.init();
        rp.build();
    }
    rp_build_timer.stop();

    // Per-partition Blendy/deg3 sub-phase accounting
    const bool mm_prof = getenv("GKR_MM_PROF") != nullptr;
    if (mm_prof) {
        zkgpt_gpu::g_matmul_timers.reset();
        zkgpt_gpu::g_matmul_timers.on = true;
        zkgpt_gpu::g_deg3_timer.reset();
        zkgpt_gpu::g_deg3_timer.on = true;
    }
    // Reset unconditionally
    g_lasso_init_timers.reset();

    double rp_time;
    {
        ZKGPT_PHASE_SCOPE("range_prove");
        zkgpt_witgen::Scope _ts(t_range);
        rp_time = rp.prove();
    }

    timer exec_commit_timer;
    exec_commit_timer.start();
    fprintf(stderr, "COMMIT_SIGNAL partition=%zu phase=start t_us=%.0f\n", i, prof_now_us());
    {
        // Unconditional
        ZKGPT_PHASE_SCOPE("execute_commit");
        zkgpt_witgen::Scope _ts(t_exec);
        p->executeCommit(threads);
    }
    exec_commit_timer.stop();
    fprintf(stderr, "COMMIT_SIGNAL partition=%zu phase=end t_us=%.0f\n", i, prof_now_us());

    // Boundary link (boundary_link.hpp)
    if (zkmod_link::arena() && is_split) {
        ZKGPT_PHASE_SCOPE("boundary_link");
        bool link_ok = true;
        if (!is_last) {
            i64 off = 0, n = 0;
            if (!st.net->carryWindow(*p, off, n)) {
                std::cerr << "[link] partition " << i << ": no carry window\n";
                link_ok = false;
            } else {
                std::vector<const Commit_return*> segs;
                std::vector<long long> starts;
                for (auto& seg : p->input_commit_segments)
                    if (!seg.external_commit && seg.commit.comm) {
                        segs.push_back(&seg.commit);
                        starts.push_back((long long)seg.start);
                    }
                link_ok = zkmod_link::publish((int)i, true, segs, starts, off, n, (int)i);
            }
        }
        if (link_ok && !is_first) {
            if (!p->cc_input.comm || st.carry_n <= 0) {
                std::cerr << "[link] partition " << i << ": no input-prefix commitment\n";
                link_ok = false;
            } else {
                link_ok =
                    zkmod_link::publish((int)i - 1, false, {&p->cc_input}, {0LL}, 0, st.carry_n, (int)i);
            }
        }
        if (!link_ok)
            return false;
    }

    v.range_prove(rp_time);
    bool accepted = false;
    {
        ZKGPT_PHASE_SCOPE("gkr_prove");
        zkgpt_witgen::Scope _ts(t_gkr);
        // prove() short-circuits on the first failed stage
        accepted = v.prove(threads, st.circuit_timer);
    }
    if (mm_prof) {
        zkgpt_gpu::g_matmul_timers.on = false;
        zkgpt_gpu::g_deg3_timer.on = false;
    }

    const double output_commit_s = p->outputCommitTimerRange().duration_sec;

    res.prover_core = v.prover_time;
    res.output_commit = output_commit_s;
    res.verifier = v.verifier_time;
    res.proof_kb = static_cast<double>(p->proof_size) / 1024.0;

    // prover_core_s = standalone_commit + prover_time + output_commit
    const double core = res.standalone_commit + v.prover_time + output_commit_s;
    const double t_part1 = prof_now_us();
    std::cout << "PARTITION idx=" << i << " len=" << cfg.len << " layers=" << cfg.split_depths[i]
              << " c=" << gkr_config.range_c << " matmul_k=" << gkr_config.matmul_k
              << " range_k=" << gkr_config.range_k << " lasso_k=" << gkr_config.lasso_k
              << " lasso_l=" << gkr_config.lasso_l << " t0_us=" << (long long)t_part0
              << " t1_us=" << (long long)t_part1 << " time_s=" << (t_part1 - t_part0) / 1e6
              << " circuit_s=" << st.circuit_timer.elapse_sec() << " prover_s=" << v.prover_time
              << " commit_s=" << output_commit_s << " standalone_commit_s=" << res.standalone_commit
              << " range_s=" << rp_time << " range_build_s=" << rp_build_timer.elapse_sec()
              << " exec_commit_s=" << exec_commit_timer.elapse_sec() << " prover_zkgpt_s=" << core
              << " prover_core_s=" << core << " prover_total_s=" << (core + rp_time)
              << " verifier_s=" << res.verifier  // this partition's verifier time
              << " proof_kb=" << res.proof_kb    // this partition's proof size
              << " ph_gkr_generic=" << v.phase_s[verifier::PH_GKR_GENERIC]
              << " ph_gkr_matmul=" << v.phase_s[verifier::PH_GKR_MATMUL]
              << " ph_lasso=" << v.phase_s[verifier::PH_LASSO]
              << " ph_hyrax_commit=" << v.phase_s[verifier::PH_HYRAX_COMMIT]
              << " ph_hyrax_open=" << v.phase_s[verifier::PH_HYRAX_OPEN]
              << " gpu_peak_mib=" << zkgpt_gpu::gkr_vram_peak_mib()
              << " host_peak_mib=" << prof_status_kb("VmHWM:") / 1024.0 << " prove_h2d_gib="
              << (zkgpt_gpu::gkr_xfer_h2d_gib() +
                  (double)g1_msm_xfer_h2d_bytes() / (1024.0 * 1024.0 * 1024.0))
              << " prove_d2h_gib="
              << zkgpt_gpu::gkr_xfer_d2h_gib()
              // Real witness-gen split. witgen_s is disjoint from time_s
              << " witgen_s=" << st.build_s << " witgen_fc_s=" << st.fc_s << " witgen_gates_s=" << st.gates_s
              << " witgen_gpu_peak_mib=" << st.witgen_gpu_peak_mib
              << " witgen_host_peak_mib=" << st.witgen_host_peak_mib
              << " witgen_resident_mib=" << st.witgen_resident_mib << " witgen_h2d_gib=" << st.witgen_h2d_gib
              << " witgen_d2h_gib=" << st.witgen_d2h_gib << std::endl;
    {
        // The four-way split: witness generation / input layer / circuit construction / proof
        const zkgpt_witgen::Accum& w = st.wg;
        const double wg_witness = w.init_s + w.values_s;
        const double wg_input = w.inlayer_s + w.copyin_s;
        const double wg_circuit = w.merge_s + w.sync_s + w.subset_s + w.misc_s + w.teardown_s + st.pad_s;
        // build_s brackets create() only
        const double wg_other = st.build_s - (wg_witness + wg_input + w.carry_s + (wg_circuit - st.pad_s));
        const double pf_commit = t_standalone + t_output + t_exec + v.phase_s[verifier::PH_HYRAX_COMMIT];
        const double pf_lasso = v.phase_s[verifier::PH_LASSO];
        const double pf_sc = v.phase_s[verifier::PH_GKR_GENERIC] + v.phase_s[verifier::PH_GKR_MATMUL];
        const double pf_input = v.phase_s[verifier::PH_HYRAX_OPEN];
        const double pf_range = t_range_build + t_range;
        const double pf_wall = (t_part1 - t_part0) / 1e6;
        const double pf_other = pf_wall - (pf_commit + pf_lasso + pf_sc + pf_input + pf_range);
        std::cout << "TIMING_SPLIT idx=" << i << " wg_witness_s=" << wg_witness
                  << " wg_input_init_s=" << wg_input << " wg_circuit_s=" << wg_circuit
                  << " wg_carry_s=" << w.carry_s << " wg_other_s=" << wg_other << " pf_commit_s=" << pf_commit
                  << " pf_lasso_s=" << pf_lasso << " pf_sumcheck_s=" << pf_sc
                  << " pf_input_check_s=" << pf_input << " pf_range_s=" << pf_range
                  << " pf_other_s=" << pf_other << " pf_verify_s="
                  << res.verifier
                  // components, so a category can be re-cut without a rerun
                  << " wg_init_s=" << w.init_s << " wg_values_s=" << w.values_s
                  << " wg_inlayer_s=" << w.inlayer_s << " wg_copyin_s=" << w.copyin_s
                  << " wg_merge_s=" << w.merge_s << " wg_sync_s=" << w.sync_s << " wg_subset_s=" << w.subset_s
                  << " wg_misc_s=" << w.misc_s << " wg_teardown_s=" << w.teardown_s
                  << " wg_pad_s=" << st.pad_s << " pf_standalone_commit_s=" << t_standalone
                  << " pf_output_commit_s=" << t_output << " pf_exec_commit_s=" << t_exec
                  << " pf_hyrax_commit_s=" << v.phase_s[verifier::PH_HYRAX_COMMIT]
                  << " pf_range_build_s=" << t_range_build << " pf_range_prove_s=" << t_range
                  << " pf_gkr_wall_s=" << t_gkr << " build_s=" << st.build_s << " prove_s=" << pf_wall
                  << std::endl;
    }
    zkgpt_acct::report("part", (int)i);
    {
        extern void gkr_ip_report_pub(int);
        gkr_ip_report_pub((int)i);
    }
    if (getenv("GKR_ACCT"))
        fprintf(stderr,
                "[acct] part idx=%d %-26s h2d=%.3fs d2h=%.3fs "
                "(%.2f GiB h2d, %.2f GiB d2h)\n",
                (int)i, "PCIe blocking copies", zkgpt_gpu::gkr_h2d_time_s(), zkgpt_gpu::gkr_d2h_time_s(),
                zkgpt_gpu::gkr_xfer_h2d_gib(), zkgpt_gpu::gkr_xfer_d2h_gib());
    v.report_totals();
    gkr_sc_report();
    {
        // Splits the largest host-side item in the prover
        auto& LI = g_lasso_init_timers;
        std::cout << "LASSO_INIT idx=" << i << " beta_s=" << LI.beta << " scatter_s=" << LI.scatter
                  << " total_s=" << (LI.beta + LI.scatter) << " scattered=" << LI.scattered << " scatter_mps="
                  << (LI.scatter > 0 ? (double)LI.scattered / LI.scatter / 1e6 : 0.0)
                  // verifyLasso's binding-check tail
                  << " bind_beta_g_s=" << LI.bind_beta_g << " bind_beta_uv_s=" << LI.bind_beta_uv
                  << " bind_gather_s=" << LI.bind_gather
                  << " bind_total_s=" << (LI.bind_beta_g + LI.bind_beta_uv + LI.bind_gather)
                  << " gathered=" << LI.gathered
                  << " gather_mps=" << (LI.bind_gather > 0 ? (double)LI.gathered / LI.bind_gather / 1e6 : 0.0)
                  << " alloc_s=" << LI.lasso_alloc << " sc_call_s=" << LI.lasso_sc_call << " barrier_s="
                  << LI.lasso_barrier
                  // A non-zero residual means a PH_LASSO region is still untimed
                  << " mm_pa_s=" << LI.mm_pa << " mm_pb_s=" << LI.mm_pb << " mm_sc_s=" << LI.mm_sc
                  << " mm_layers=" << LI.mm_layers << " mm_acct_s=" << (LI.mm_pa + LI.mm_pb + LI.mm_sc)
                  << " lasso_acct_s="
                  << (LI.lasso_alloc + LI.beta + LI.scatter + LI.lasso_sc_call + LI.lasso_barrier +
                      LI.bind_beta_g + LI.bind_beta_uv + LI.bind_gather)
                  << std::endl;
    }
    if (mm_prof) {
        auto& MT = zkgpt_gpu::g_matmul_timers;
        std::cout << "MATMUL_BREAKDOWN idx=" << i << " expand_s=" << MT.expand << " ptable_s=" << MT.ptable
                  << " round_s=" << MT.round << " fold_s=" << MT.fold
                  << " deg3_s=" << zkgpt_gpu::g_deg3_timer.gpu << " bucket_sum_s="
                  << (MT.expand + MT.ptable + MT.round + MT.fold + zkgpt_gpu::g_deg3_timer.gpu)
                  << " bring_s=" << MT.bring << " bring_fold_s=" << MT.bring_fold
                  << " sc_wall_s=" << MT.sc_wall << " host_wall_s=" << MT.host_wall << " sc_residual_s="
                  << (MT.sc_wall - (MT.expand + MT.ptable + MT.round + MT.fold + (MT.bring - MT.bring_fold)))
                  << std::endl;
    }
    return accepted;
}

static const std::vector<int>& physical_cpus() {
    // Only CPUs in the inherited affinity mask
    static const std::vector<int> cpus = [] {
        std::vector<int> out;
        std::set<std::string> seen;
        cpu_set_t allowed;
        CPU_ZERO(&allowed);
        const bool have_mask = sched_getaffinity(0, sizeof allowed, &allowed) == 0;
        for (int c = 0; c < CPU_SETSIZE; ++c) {
            if (have_mask && !CPU_ISSET(c, &allowed))
                continue;
            char path[128];
            std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list",
                          c);
            std::ifstream f(path);
            if (!f)
                continue;  // no such cpu
            std::string sibs;
            std::getline(f, sibs);
            if (seen.insert(sibs).second)
                out.push_back(c);  // lowest sibling wins
        }
        return out;
    }();
    return cpus;
}

// GKR_PIN_CORES=1: pin each child to its own block of `per_child` physical cores
static int* g_core_owner = nullptr;
static int g_core_blocks = 0, g_core_tpc = 0, g_my_block = -1;
static void core_table_init(int per_child) {
    if (!getenv("GKR_PIN_CORES") || per_child <= 0 || g_core_owner)
        return;
    g_core_tpc = per_child;
    g_core_blocks = (int)(physical_cpus().size() / (size_t)per_child);
    if (g_core_blocks <= 0)
        return;
    void* raw =
        mmap(nullptr, sizeof(int) * g_core_blocks, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) {
        std::perror("[split] pin: mmap");
        return;
    }
    g_core_owner = static_cast<int*>(raw);
    std::memset(g_core_owner, 0, sizeof(int) * g_core_blocks);
}
static bool pin_child_cores(int idx) {
    if (!g_core_owner)
        return false;
    int blk = -1;
    for (int b = 0; b < g_core_blocks && blk < 0; ++b) {
        int expect = 0;
        if (__atomic_compare_exchange_n(&g_core_owner[b], &expect, idx + 1, false, __ATOMIC_ACQ_REL,
                                        __ATOMIC_ACQUIRE))
            blk = b;
    }
    const bool shared = blk < 0;
    if (shared)
        blk = idx % g_core_blocks;
    else
        g_my_block = blk;
    const std::vector<int>& cpus = physical_cpus();
    const size_t first = (size_t)blk * (size_t)g_core_tpc;
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int j = 0; j < g_core_tpc; ++j)
        CPU_SET(cpus[first + j], &set);
    if (sched_setaffinity(0, sizeof set, &set) != 0) {
        std::perror("[split] pin: sched_setaffinity");
        return false;
    }
    std::cerr << "[split] pin: child " << idx << " -> block " << blk << " cpus " << cpus[first] << "-"
              << cpus[first + g_core_tpc - 1] << " (" << g_core_tpc << " physical cores"
              << (shared ? ", SHARED: every block held" : "") << ")\n";
    return true;
}
static void unpin_child_cores() {
    if (g_core_owner && g_my_block >= 0)
        __atomic_store_n(&g_core_owner[g_my_block], 0, __ATOMIC_RELEASE);
    g_my_block = -1;
}

// Boundary-link arena for a split run (boundary_link.hpp), created before any fork
static bool link_begin(const SplitCfg& cfg) {
    const int nb = (int)cfg.nparts() - 1;
    if (nb <= 0)
        return true;
    long long fdim = 1;
    while (fdim < cfg.embed_size)
        fdim <<= 1;
    long long tmax = 1;
    for (long long t : cfg.tiws)
        tmax = std::max(tmax, t);
    int l = 0;
    while ((1LL << l) < tmax)
        ++l;
    return zkmod_link::create(nb, 2LL * cfg.len * fdim, l + 1);
}
static bool link_end(const char* mode, bool all_ok) {
    if (all_ok)
        all_ok = zkmod_link::all_checked(mode);
    zkmod_link::destroy();
    return all_ok;
}

struct RunTotals {
    double prover_sum = 0, output_commit_sum = 0, verifier_sum = 0, proof_kb = 0;
    double makespan = 0;
};

static bool run_split_partitions(const SplitCfg& cfg, bool unique_parts, RunTotals& out) {
    SplitBlockCarryState carry;
    bool has_carry = false;
    double total_prover_time = 0.0;
    double total_output_commit_time = 0.0;
    double total_verifier_time = 0.0;
    double combined_proof_size_kb = 0.0;
    // zkGPT's split baseline reports the PARALLEL makespan
    double max_prover_core = 0.0;
    double max_output_commit = 0.0;

    std::vector<std::size_t> run_idx;
    if (unique_parts && cfg.nparts() > 3) {
        run_idx.push_back(0);
        std::set<int> seen_mid;  // one middle per distinct depth
        for (std::size_t i = 1; i + 1 < cfg.nparts(); ++i)
            if (seen_mid.insert(cfg.split_depths[i]).second)
                run_idx.push_back(i);
        run_idx.push_back(cfg.nparts() - 1);
        std::cerr << "[split] unique-parts: proving " << run_idx.size() << " of " << cfg.nparts()
                  << " partitions (idx:";
        for (auto i : run_idx)
            std::cerr << ' ' << i;
        std::cerr << ") -- MEASUREMENT ONLY, proof is not end-to-end valid\n";
    } else {
        for (std::size_t i = 0; i < cfg.nparts(); ++i)
            run_idx.push_back(i);
        if (!link_begin(cfg))
            return false;
    }
    std::cout << "SPLIT_MODE unique_parts=" << (unique_parts ? 1 : 0) << " proved=" << run_idx.size()
              << " of=" << cfg.nparts() << std::endl;

    for (std::size_t j = 0; j < run_idx.size(); ++j) {
        const std::size_t i = run_idx[j];
        const bool is_last = (i + 1 == cfg.nparts());

        // Top of the iteration
        memacct("floor", (long long)i);

        PartitionState st;
        if (!build_partition(cfg, i, has_carry ? &carry : nullptr, st))
            return false;

        if (cfg.witgen_only) {
            std::cout << "WITGEN_ONLY idx=" << i << " witgen_s=" << st.build_s << " witgen_fc_s=" << st.fc_s
                      << " witgen_gates_s=" << st.gates_s
                      << " gpu_peak_mib=" << zkgpt_gpu::gkr_vram_peak_mib()
                      << " witgen_gpu_peak_mib=" << st.witgen_gpu_peak_mib
                      << " witgen_host_peak_mib=" << st.witgen_host_peak_mib
                      << " witgen_resident_mib=" << st.witgen_resident_mib
                      << " witgen_h2d_gib=" << st.witgen_h2d_gib << " witgen_d2h_gib=" << st.witgen_d2h_gib
                      << std::endl;
            if (!is_last) {
                SplitBlockCarryState next_carry = st.net->exportSplitBlockCarry(*st.p);
                carry = std::move(next_carry);
                has_carry = true;
            }
            continue;
        }

        SplitBlockCarryState next_carry;
        if (!is_last) {
            ZKGPT_PHASE_SCOPE("export_carry");
            next_carry = st.net->exportSplitBlockCarry(*st.p);
        }

        PartResult res;
        if (!prove_built_partition(cfg, i, st, res, cfg.num_threads))
            return false;

        total_prover_time += res.standalone_commit + res.prover_core + res.output_commit;
        total_output_commit_time += res.output_commit;
        total_verifier_time += res.verifier;
        combined_proof_size_kb += res.proof_kb;
        max_prover_core = std::max(max_prover_core, res.standalone_commit + res.prover_core);
        max_output_commit = std::max(max_output_commit, res.output_commit);

        if (!is_last) {
            carry = std::move(next_carry);
            has_carry = true;
        }
    }

    if (!link_end("sequential", true))
        return false;
    out.makespan = max_prover_core + max_output_commit;
    out.prover_sum = total_prover_time;
    out.output_commit_sum = total_output_commit_time;
    out.verifier_sum = total_verifier_time;
    out.proof_kb = combined_proof_size_kb;
    return true;
}

static bool run_split_sequential(SplitCfg& cfg, bool unique_parts) {
    RunTotals grand;
    if (!run_split_partitions(cfg, unique_parts, grand))
        return false;

    std::cout << "Split proof results:" << std::endl;
    std::cout << "  Output commit time: " << grand.output_commit_sum << " s" << std::endl;
    std::cout << "  Prover time (makespan/zkGPT, max partition): " << grand.makespan << " s" << std::endl;
    std::cout << "  Prover time (single-GPU sequential, sum):    " << grand.prover_sum << " s" << std::endl;
    std::cout << "  Total verifier time:" << grand.verifier_sum << " s" << std::endl;
    std::cout << "  Combined proof size:" << grand.proof_kb << " KB" << std::endl;
    std::cout << "SPLIT_SUMMARY makespan_s=" << grand.makespan << " seq_sum_s=" << grand.prover_sum
              << " output_commit_s=" << grand.output_commit_sum << " parts=" << cfg.nparts()
              << " len=" << cfg.len << " verifier_s=" << grand.verifier_sum << " proof_kb=" << grand.proof_kb
              << std::endl;
    return true;
}

// concurrent: fork-and-hold
static bool run_split_concurrent(const SplitCfg& cfg, int width) {
    const int n = static_cast<int>(cfg.nparts());
    const int w = (width > 0 && width < n) ? width : n;

    const int nwaves = (n + w - 1) / w;
    // One host thread budget split across the concurrent children
    const int tpc = std::max(1, cfg.num_threads / w);
    core_table_init(tpc);

    SplitBlockCarryState carry;
    bool has_carry = false;
    bool all_ok = true;
    double makespan_s = 0.0;
    const pid_t sampler =
        cfg.vram_csv.empty() ? -1 : zkgpt_sampler::start(cfg.vram_csv, cfg.vram_interval, prof_now_us);

    // One release gate per wave
    std::vector<std::array<int, 2>> gates(nwaves);
    for (int g = 0; g < nwaves; ++g)
        if (pipe(gates[g].data()) != 0) {
            std::perror("[split] pipe");
            return false;
        }
    if (!link_begin(cfg))
        return false;

    std::vector<std::vector<pid_t>> wave_pids(nwaves);
    const double t_prep0 = prof_now_us();
    for (int i = 0; i < n && all_ok; ++i) {
        const bool is_last = (i + 1 == n);
        const int my_wave = i / w;

        PartitionState st;
        if (!build_partition(cfg, i, has_carry ? &carry : nullptr, st)) {
            all_ok = false;
            break;
        }
        SplitBlockCarryState next_carry;
        if (!is_last) {
            ZKGPT_PHASE_SCOPE("export_carry");
            next_carry = st.net->exportSplitBlockCarry(*st.p);
        }

        std::cout.flush();
        std::cerr.flush();
        pid_t pid = fork();
        if (pid == 0) {
            // Die with the parent, or a killed run leaves orphans holding GPU memory
            prctl(PR_SET_PDEATHSIG, SIGTERM);
            if (getppid() == 1)
                _exit(1);
            child_threads_set(tpc);
            for (int g = 0; g < nwaves; ++g) {
                close(gates[g][1]);
                if (g != my_wave)
                    close(gates[g][0]);
            }
            // Drop what only the parent needs
            carry.input_data = std::vector<F>();
            next_carry.input_data = std::vector<F>();

            {
                ZKGPT_PHASE_SCOPE("gate_wait");
                char go = 0;
                while (read(gates[my_wave][0], &go, 1) < 0 && errno == EINTR) {
                }
                close(gates[my_wave][0]);
            }
            pin_child_cores(i);

            zkgpt_gpu::gkr_vram_peak_start();
            PartResult res;
            bool ok = prove_built_partition(cfg, i, st, res, tpc);
            gkr_sc_teardown();
            gkr_pred_report();
            gkr_pred_teardown();
            zkgpt_gpu::gkr_vram_peak_stop();
            unpin_child_cores();
            std::cout.flush();
            _exit(ok ? 0 : 1);
        }
        if (pid < 0) {
            std::perror("[split] fork");
            all_ok = false;
            break;
        }
        wave_pids[my_wave].push_back(pid);

        st.p.reset();
        st.net.reset();
        if (!is_last) {
            carry = std::move(next_carry);
            has_carry = true;
        }
        std::cerr << "[split] built+forked " << (i + 1) << "/" << n << " wave " << my_wave << " (pid " << pid
                  << ")\n";
    }
    const double prep_s = (prof_now_us() - t_prep0) / 1e6;
    carry.input_data = std::vector<F>();  // parent is done with the chain
    for (int g = 0; g < nwaves; ++g)
        close(gates[g][0]);  // parent never reads

    // release wave by wave
    double first_release_us = 0, last_join_us = 0;
    zkgpt_phase::set_part(-1);
    for (int g = 0; g < nwaves && all_ok; ++g) {
        const double t_rel = prof_now_us();
        if (g == 0)
            first_release_us = t_rel;
        ZKGPT_PHASE_MARKF("release_wave%d", g);
        close(gates[g][1]);  // EOF == go
        for (pid_t pid : wave_pids[g]) {
            int stt = 0;
            if (waitpid(pid, &stt, 0) < 0) {
                std::perror("[split] waitpid");
                all_ok = false;
                continue;
            }
            if (!WIFEXITED(stt) || WEXITSTATUS(stt) != 0) {
                std::cerr << "[split] child pid " << pid << " failed (status " << stt << ")\n";
                all_ok = false;
            }
        }
        last_join_us = prof_now_us();
        ZKGPT_PHASE_MARKF("joined_wave%d", g);
        makespan_s += (last_join_us - t_rel) / 1e6;
    }
    // Reap anything still outstanding after an early failure
    if (!all_ok)
        for (int g = 0; g < nwaves; ++g) {
            close(gates[g][1]);
            for (pid_t pid : wave_pids[g]) {
                int stt = 0;
                waitpid(pid, &stt, WNOHANG);
            }
        }

    zkgpt_sampler::stop(sampler);
    report_vram_peak(cfg, first_release_us, last_join_us, "concurrent");
    all_ok = link_end("concurrent", all_ok);

    // No aggregate proof_kb/verifier_s
    std::cout << "SPLIT_CONCURRENT parts=" << n << " width=" << w << " waves=" << nwaves
              << " threads_per_child=" << tpc << " threads_total=" << cfg.num_threads << " prep_s=" << prep_s
              << " concurrent_makespan_s=" << makespan_s << " status=" << (all_ok ? "OK" : "FAIL")
              << std::endl;
    return all_ok;
}

static bool run_split_batched(const SplitCfg& cfg, int batch) {
    const int n = static_cast<int>(cfg.nparts());
    const int b = (batch > 0 && batch < n) ? batch : n;
    const int nb = (n + b - 1) / b;
    // Parent's build is not given a slice of this budget
    const int tpc = std::max(1, cfg.num_threads / b);
    core_table_init(tpc);

    SplitBlockCarryState carry;
    bool has_carry = false;
    bool all_ok = true;
    const pid_t sampler =
        cfg.vram_csv.empty() ? -1 : zkgpt_sampler::start(cfg.vram_csv, cfg.vram_interval, prof_now_us);

    std::vector<std::array<int, 2>> gates(nb);
    for (int g = 0; g < nb; ++g)
        if (pipe(gates[g].data()) != 0) {
            std::perror("[split] pipe");
            return false;
        }
    if (!link_begin(cfg))
        return false;

    std::vector<std::vector<pid_t>> batch_pids(nb);
    std::vector<double> build_s(nb, 0.0), join_wait_s(nb, 0.0);

    auto build_batch = [&](int g) -> bool {
        const int lo = g * b, hi = std::min(n, (g + 1) * b);
        const double t_b0 = prof_now_us();
        for (int i = lo; i < hi; ++i) {
            const bool is_last = (i + 1 == n);

            PartitionState st;
            if (!build_partition(cfg, i, has_carry ? &carry : nullptr, st))
                return false;
            SplitBlockCarryState next_carry;
            if (!is_last) {
                ZKGPT_PHASE_SCOPE("export_carry");
                next_carry = st.net->exportSplitBlockCarry(*st.p);
            }

            std::cout.flush();
            std::cerr.flush();
            pid_t pid = fork();
            if (pid == 0) {
                prctl(PR_SET_PDEATHSIG, SIGTERM);
                if (getppid() == 1)
                    _exit(1);
                child_threads_set(tpc);
                for (int q = 0; q < nb; ++q) {
                    close(gates[q][1]);
                    if (q != g)
                        close(gates[q][0]);
                }
                carry.input_data = std::vector<F>();
                next_carry.input_data = std::vector<F>();

                {
                    ZKGPT_PHASE_SCOPE("gate_wait");
                    char go = 0;
                    while (read(gates[g][0], &go, 1) < 0 && errno == EINTR) {
                    }
                    close(gates[g][0]);
                }
                pin_child_cores(i);

                zkgpt_gpu::gkr_vram_peak_start();
                PartResult res;
                bool ok = prove_built_partition(cfg, i, st, res, tpc);
                gkr_sc_teardown();
                gkr_pred_report();
                gkr_pred_teardown();
                zkgpt_gpu::gkr_vram_peak_stop();
                unpin_child_cores();
                std::cout.flush();
                _exit(ok ? 0 : 1);
            }
            if (pid < 0) {
                std::perror("[split] fork");
                return false;
            }
            batch_pids[g].push_back(pid);

            st.p.reset();
            st.net.reset();
            if (!is_last) {
                carry = std::move(next_carry);
                has_carry = true;
            }
            std::cerr << "[split] built+forked " << (i + 1) << "/" << n << " batch " << g << " (pid " << pid
                      << ")\n";
        }
        build_s[g] = (prof_now_us() - t_b0) / 1e6;
        return true;
    };

    const double t0 = prof_now_us();

    if (!build_batch(0))
        all_ok = false;
    const double exposed_build_s = build_s[0];

    double first_release_us = 0, last_join_us = 0;
    for (int g = 0; g < nb && all_ok; ++g) {
        const double t_rel = prof_now_us();
        if (g == 0)
            first_release_us = t_rel;
        zkgpt_phase::set_part(-1);
        ZKGPT_PHASE_MARKF("release_batch%d", g);
        close(gates[g][1]);  // EOF == go, batch g starts proving

        if (g + 1 < nb && !build_batch(g + 1)) {
            all_ok = false;
            break;
        }  // the overlap

        const double t_join0 = prof_now_us();
        for (pid_t pid : batch_pids[g]) {
            int stt = 0;
            if (waitpid(pid, &stt, 0) < 0) {
                std::perror("[split] waitpid");
                all_ok = false;
                continue;
            }
            if (!WIFEXITED(stt) || WEXITSTATUS(stt) != 0) {
                std::cerr << "[split] child pid " << pid << " failed (status " << stt << ")\n";
                all_ok = false;
            }
        }
        last_join_us = prof_now_us();
        zkgpt_phase::set_part(-1);
        ZKGPT_PHASE_MARKF("joined_batch%d", g);
        join_wait_s[g] = (last_join_us - t_join0) / 1e6;
        std::cout << "BATCHED_GROUP g=" << g << " lo=" << (g * b) << " hi=" << std::min(n, (g + 1) * b)
                  << " build_s=" << build_s[g] << " next_build_s=" << (g + 1 < nb ? build_s[g + 1] : 0.0)
                  << " join_wait_s="
                  << join_wait_s[g]
                  // join_wait ~0 means the children were already done when the parent finished building
                  << " bottleneck="
                  << (g + 1 >= nb ? "exposed-prove" : (join_wait_s[g] > 1.0 ? "prove" : "build"))
                  << std::endl;
    }
    const double total_s = (prof_now_us() - t0) / 1e6;
    carry.input_data = std::vector<F>();
    for (int g = 0; g < nb; ++g)
        close(gates[g][0]);  // parent never reads

    if (!all_ok)
        for (int g = 0; g < nb; ++g) {
            close(gates[g][1]);
            for (pid_t pid : batch_pids[g]) {
                int stt = 0;
                waitpid(pid, &stt, WNOHANG);
            }
        }

    zkgpt_sampler::stop(sampler);
    report_vram_peak(cfg, first_release_us, last_join_us, "pipeline");
    all_ok = link_end("batched", all_ok);

    double build_sum_s = 0;
    for (double v : build_s)
        build_sum_s += v;

    // batched_total_s INCLUDES witgen
    std::cout << "SPLIT_BATCHED parts=" << n << " batch=" << b << " batches=" << nb
              << " threads_per_child=" << tpc << " threads_total=" << cfg.num_threads
              << " build_sum_s=" << build_sum_s << " exposed_build_s=" << exposed_build_s
              << " batched_total_s=" << total_s << " status=" << (all_ok ? "OK" : "FAIL") << std::endl;
    return all_ok;
}

// overlapped: fork before any HIP, children do their own GPU witgen
struct ChainShm {
    unsigned long long count;  // elements in data[]
    long long input_offset, len, feature_dim;
    int input_c, input_e;
    F data[1];  // [cap] elements follow
};

static long long cgroup_available_kib() {
    FILE* f = fopen("/proc/self/cgroup", "r");
    if (!f)
        return -1;
    char line[4096], rel[4096] = "";
    while (fgets(line, sizeof line, f)) {
        if (sscanf(line, "0::%4095s", rel) == 1)
            break;
    }
    fclose(f);
    if (!rel[0])
        return -1;
    std::string dir = std::string("/sys/fs/cgroup") + (std::strcmp(rel, "/") ? rel : "");
    long long best = -1;
    for (;;) {
        long long max_b = -1, cur_b = -1, inactive_b = 0;
        if (FILE* m = fopen((dir + "/memory.max").c_str(), "r")) {
            if (fscanf(m, "%lld", &max_b) != 1)
                max_b = -1;  // "max" = no limit at this level
            fclose(m);
        }
        if (max_b >= 0) {
            if (FILE* c = fopen((dir + "/memory.current").c_str(), "r")) {
                if (fscanf(c, "%lld", &cur_b) != 1)
                    cur_b = -1;
                fclose(c);
            }
            if (FILE* s = fopen((dir + "/memory.stat").c_str(), "r")) {
                while (fgets(line, sizeof line, s)) {
                    if (sscanf(line, "inactive_file %lld", &inactive_b) == 1)
                        break;
                }
                fclose(s);
            }
            if (cur_b >= 0) {
                long long kib = (max_b - std::max(0LL, cur_b - inactive_b)) / 1024;
                if (kib < 0)
                    kib = 0;
                if (best < 0 || kib < best)
                    best = kib;
            }
        }
        if (dir == "/sys/fs/cgroup")
            break;
        dir.erase(dir.rfind('/'));
    }
    return best;
}

static long long mem_available_kib() {
    FILE* f = fopen("/proc/meminfo", "r");
    if (!f)
        return 0;
    char line[256];
    long long kib = 0;
    while (fgets(line, sizeof line, f)) {
        if (sscanf(line, "MemAvailable: %lld kB", &kib) == 1)
            break;
    }
    fclose(f);
    const long long cg = cgroup_available_kib();
    return cg >= 0 && cg < kib ? cg : kib;
}

// A child's high-water RSS (VmHWM, KiB) from /proc
static long long proc_hwm_kib(pid_t pid) {
    char path[64];
    std::snprintf(path, sizeof path, "/proc/%d/status", (int)pid);
    FILE* f = fopen(path, "r");
    if (!f)
        return 0;
    char line[256];
    long long kib = 0;
    while (fgets(line, sizeof line, f)) {
        if (sscanf(line, "VmHWM: %lld kB", &kib) == 1)
            break;
    }
    fclose(f);
    return kib;
}

// A child's CURRENT resident set (VmRSS, KiB)
static long long proc_rss_kib(pid_t pid) {
    char path[64];
    std::snprintf(path, sizeof path, "/proc/%d/status", (int)pid);
    FILE* f = fopen(path, "r");
    if (!f)
        return 0;
    char line[256];
    long long kib = 0;
    while (fgets(line, sizeof line, f)) {
        if (sscanf(line, "VmRSS: %lld kB", &kib) == 1)
            break;
    }
    fclose(f);
    return kib;
}

// VRAM side of the memory gate
struct VramDev {
    long long gpu_id = 0;
    std::string drm;
};
static long long read_ll(const std::string& path) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f)
        return -1;
    long long v = -1;
    if (fscanf(f, "%lld", &v) != 1)
        v = -1;
    fclose(f);
    return v;
}
static bool vram_dev_for_hip_index(int k, VramDev& d) {
    const std::string base = "/sys/class/kfd/kfd/topology/nodes/";
    int seen = 0;
    for (int node = 0; node < 64; ++node) {
        const long long id = read_ll(base + std::to_string(node) + "/gpu_id");
        if (id < 0)
            break;
        if (id == 0)
            continue;
        if (seen++ != k)
            continue;
        FILE* f = fopen((base + std::to_string(node) + "/properties").c_str(), "r");
        if (!f)
            return false;
        char key[128];
        long long val;
        int minor = -1;
        while (fscanf(f, "%127s %lld", key, &val) == 2)
            if (!strcmp(key, "drm_render_minor"))
                minor = (int)val;
        fclose(f);
        if (minor < 0)
            return false;
        d.gpu_id = id;
        d.drm = "/sys/class/drm/renderD" + std::to_string(minor) + "/device/";
        return read_ll(d.drm + "mem_info_vram_total") > 0;
    }
    return false;
}
static long long vram_free_kib(const VramDev& d) {
    const long long t = read_ll(d.drm + "mem_info_vram_total");
    const long long u = read_ll(d.drm + "mem_info_vram_used");
    return (t < 0 || u < 0) ? -1 : (t - u) / 1024;
}
static long long proc_vram_kib(pid_t pid, const VramDev& d) {
    const long long b =
        read_ll("/sys/class/kfd/kfd/proc/" + std::to_string((int)pid) + "/vram_" + std::to_string(d.gpu_id));
    return b < 0 ? 0 : b / 1024;
}

// child -> parent event in memory-gated mode
struct ChainStat {
    char ev;
    int idx;
};

// Deletes partition idx's witness-cache file(s) ("..._p<idx>_<hash>.bin") from dir
static void remove_part_files(const std::string& dir, int idx) {
    DIR* d = opendir(dir.c_str());
    if (!d)
        return;
    const std::string tag = "_p" + std::to_string(idx) + "_";
    while (dirent* e = readdir(d))
        if (std::strstr(e->d_name, tag.c_str()))
            ::unlink((dir + "/" + e->d_name).c_str());
    closedir(d);
}

static bool run_split_overlapped(const SplitCfg& cfg, int batch, bool want_witgen_gpu, bool want_round_gpu,
                                 bool want_gates_gpu, bool want_witgen_full, bool barrier) {
    const int n = static_cast<int>(cfg.nparts());
    // batch < 0: memory-gated mode
    const bool mem_gated = (batch < 0);
    const int b = mem_gated ? n : (batch > 0 && batch < n) ? batch : n;
    // GKR_TPC: threads per child, overriding num_threads/b
    int tpc = std::max(1, cfg.num_threads / b);
    if (const char* e = getenv("GKR_TPC")) {
        int v = atoi(e);
        if (v > 0)
            tpc = v;
    }
    tpc = std::min(tpc, std::max(1, cfg.num_threads));
    core_table_init(tpc);

    long long cap = 0;
    for (long long t : cfg.tiws)
        cap = std::max(cap, t);
    if (cap <= 0) {
        // Defensive: the probe always sets tiws > 0
        const long long slice = 1LL * cfg.len * (1LL << ceilPow2BitLength(cfg.embed_size));
        cap = std::max(1LL << 26, 8 * slice);
        std::cerr << "[split] witgen-chain: no probed layout, carry cap " << cap << " elems\n";
    }
    cap = cap * 2;
    const std::size_t shm_bytes = sizeof(ChainShm) + (std::size_t)cap * sizeof(F);

    void* raw = mmap(nullptr, shm_bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) {
        std::perror("[split] mmap");
        return false;
    }
    ChainShm* shm = static_cast<ChainShm*>(raw);
    shm->count = 0;
    if (!link_begin(cfg)) {
        munmap(raw, shm_bytes);
        return false;
    }
    std::cerr << "[split] witgen-chain carry buffer " << (shm_bytes / 1048576.0) << " MiB (cap " << cap
              << " elems)\n";

    // START THE SAMPLER BEFORE ANY PIPE EXISTS. It is a fork
    const pid_t sampler =
        cfg.vram_csv.empty() ? -1 : zkgpt_sampler::start(cfg.vram_csv, cfg.vram_interval, prof_now_us);

    // gates[i]: child i signals it has published its carry
    std::vector<std::array<int, 2>> gates(n > 1 ? n - 1 : 0);
    for (auto& g : gates)
        if (pipe(g.data()) != 0) {
            std::perror("[split] pipe");
            munmap(raw, shm_bytes);
            return false;
        }
    // started[i]: child i has its build slot (or, ungated, has begun building)
    const bool prebuild_on =
        !(getenv("GKR_PREBUILD") && !strcmp(getenv("GKR_PREBUILD"), "0")) && !zkgpt_wc::enabled();
    std::vector<std::array<int, 2>> started(n > 1 ? n - 1 : 0);
    for (auto& g : started)
        if (pipe(g.data()) != 0) {
            std::perror("[split] pipe");
            munmap(raw, shm_bytes);
            return false;
        }

    int allgate[2] = {-1, -1}, notify[2] = {-1, -1};
    if (barrier && (pipe(allgate) != 0 || pipe(notify) != 0)) {
        std::perror("[split] pipe");
        munmap(raw, shm_bytes);
        return false;
    }
    int tok[2];
    if (pipe(tok) != 0) {
        std::perror("[split] pipe");
        munmap(raw, shm_bytes);
        return false;
    }
    // Memory-gated mode preloads NO tokens
    for (int i = 0; i < (mem_gated ? 0 : b); ++i)
        if (write(tok[1], "x", 1) != 1) {
            std::perror("[split] token");
            return false;
        }
    // Memory-gated mode
    std::vector<std::array<int, 2>> ptok(mem_gated ? n : 0);
    for (auto& t : ptok)
        if (pipe(t.data()) != 0) {
            std::perror("[split] pipe");
            munmap(raw, shm_bytes);
            return false;
        }
    // Memory-gated mode also gates the BUILD chain
    const bool gate_builds = mem_gated && !barrier;
    int btok[2] = {-1, -1}, stat[2] = {-1, -1};
    if (mem_gated && (pipe(btok) != 0 || pipe(stat) != 0)) {
        std::perror("[split] pipe");
        munmap(raw, shm_bytes);
        return false;
    }
    auto send_stat = [&](char ev, int idx) {
        if (!mem_gated)
            return;
        ChainStat m{ev, idx};
        if (write(stat[1], &m, sizeof m) != (ssize_t)sizeof m)
            _exit(1);
    };

    // GKR_SPLIT_DEVICES="5,6": round-robin the children across HIP devices
    std::vector<std::string> devlist;
    if (const char* dl = getenv("GKR_SPLIT_DEVICES")) {
        std::string cur;
        for (const char* q = dl;; ++q) {
            if (*q == ',' || *q == '\0') {
                if (!cur.empty())
                    devlist.push_back(cur);
                cur.clear();
                if (*q == '\0')
                    break;
            } else
                cur += *q;
        }
        std::cerr << "[split] spreading children over " << devlist.size() << " HIP devices (" << dl << ")\n";
    }

    // GKR_WITGEN_DEVICE=W
    const char* wdev = getenv("GKR_WITGEN_DEVICE");
    const bool ctrl = wdev && *wdev;
    std::string wc_dir;
    std::vector<std::array<int, 2>> ready;
    if (ctrl) {
        if (mem_gated) {
            std::cerr << "[split] GKR_WITGEN_DEVICE is not supported with --overlapped -1\n";
            munmap(raw, shm_bytes);
            return false;
        }
        wc_dir = "/dev/shm/zkgpt_wc_" + std::to_string((long)getpid());
        if (::mkdir(wc_dir.c_str(), 0700) != 0 && errno != EEXIST) {
            std::perror("[split] mkdir handoff dir");
            munmap(raw, shm_bytes);
            return false;
        }
        ready.resize(n);
        for (auto& r : ready)
            if (pipe(r.data()) != 0) {
                std::perror("[split] pipe");
                munmap(raw, shm_bytes);
                return false;
            }
        std::cerr << "[split] witgen controller on HIP device " << wdev << ", handoff via " << wc_dir << "\n";
    }

    const double t0 = prof_now_us();
    std::vector<pid_t> pids;
    bool all_ok = true;

    // Forked after every pipe exists
    pid_t ctrl_pid = -1;
    if (ctrl) {
        std::cout.flush();
        std::cerr.flush();
        ctrl_pid = fork();
        if (ctrl_pid == 0) {
            prctl(PR_SET_PDEATHSIG, SIGTERM);
            if (getppid() == 1)
                _exit(1);
            signal(SIGPIPE, SIG_IGN);  // a dead child must not kill the chain
            for (auto& g : gates) {
                close(g[0]);
                close(g[1]);
            }
            for (auto& g : started) {
                close(g[0]);
                close(g[1]);
            }
            if (barrier) {
                close(allgate[0]);
                close(allgate[1]);
                close(notify[0]);
                close(notify[1]);
            }
            close(tok[0]);
            close(tok[1]);
            for (auto& t : ptok) {
                close(t[0]);
                close(t[1]);
            }
            for (auto& r : ready)
                close(r[0]);
            setenv("HIP_VISIBLE_DEVICES", wdev, 1);
            setenv("ZKGPT_WITNESS_CACHE", wc_dir.c_str(), 1);
            // Safe here and ONLY here: this process was forked before any HIP call
            gkr_config.witgen_gpu = want_witgen_gpu;
            gkr_config.round_gpu = want_round_gpu;
            gkr_config.gates_gpu = want_gates_gpu;
            gkr_config.witgen_full = want_witgen_full;

            SplitBlockCarryState carry;
            bool has_carry = false;
            for (int i = 0; i < n; ++i) {
                const bool is_last = (i + 1 == n);
                PartitionState st;
                const double tb0 = prof_now_us();
                if (!build_partition(cfg, i, has_carry ? &carry : nullptr, st))
                    _exit(1);
                const double build_s = (prof_now_us() - tb0) / 1e6;  // includes the cache save
                if (!is_last) {
                    ZKGPT_PHASE_SCOPE("export_carry");
                    carry = st.net->exportSplitBlockCarry(*st.p);
                    has_carry = true;
                }
                (void)!write(ready[i][1], "x", 1);
                close(ready[i][1]);
                std::cout << "WITGEN_CTRL idx=" << i << " build_s=" << build_s << " fc_s=" << st.fc_s
                          << " gpu_peak_mib=" << st.witgen_gpu_peak_mib
                          << " t_us=" << (prof_now_us() - t0) / 1e6 << std::endl;
            }
            _exit(0);
        }
        if (ctrl_pid < 0) {
            std::perror("[split] fork controller");
            all_ok = false;
        }
    }

    for (int i = 0; i < n && all_ok; ++i) {
        std::cout.flush();
        std::cerr.flush();
        pid_t pid = fork();
        if (pid == 0) {
            prctl(PR_SET_PDEATHSIG, SIGTERM);
            if (getppid() == 1)
                _exit(1);
            child_threads_set(tpc);
            if (!devlist.empty())
                setenv("HIP_VISIBLE_DEVICES", devlist[i % devlist.size()].c_str(), 1);
            for (int j = 0; j < (int)gates.size(); ++j) {
                if (j != i)
                    close(gates[j][1]);  // only child i writes gate i
                if (j != i - 1)
                    close(gates[j][0]);  // only child i reads gate i-1
                if (j != i)
                    close(started[j][1]);
                if (j != i - 1)
                    close(started[j][0]);
            }
            auto signal_started = [&] {  // see started[] above
                if (i + 1 < n) {
                    if (write(started[i][1], "x", 1) != 1)
                        _exit(1);
                    close(started[i][1]);
                }
            };
            if (ctrl) {
                for (int j = 0; j < n; ++j) {
                    close(ready[j][1]);  // only the controller writes
                    if (j != i)
                        close(ready[j][0]);
                }
                setenv("ZKGPT_WITNESS_CACHE", wc_dir.c_str(), 1);
            }

            const bool is_last = (i + 1 == n);
            zkgpt_phase::set_part(i);
            SplitBlockCarryState carry;
            if (ctrl) {
                ZKGPT_PHASE_SCOPE("handoff_wait");  // controller has built partition i
                char go = 0;
                ssize_t r;
                while ((r = read(ready[i][0], &go, 1)) < 0 && errno == EINTR) {
                }
                if (r != 1) {
                    std::cerr << "[split] child " << i << ": witgen controller died\n";
                    _exit(1);
                }
                close(ready[i][0]);
            }
            const bool prebuild = prebuild_on && !ctrl && i > 0;
            auto acquire_build_slot = [&] {
                if (!gate_builds)
                    return;
                ZKGPT_PHASE_SCOPE("build_wait");  // acquire a build slot
                send_stat('w', i);
                char go = 0;
                ssize_t r;
                while ((r = read(btok[0], &go, 1)) < 0 && errno == EINTR) {
                }
                if (r != 1) {
                    std::cerr << "[split] child " << i << ": build gate closed\n";
                    _exit(1);
                }
            };
            auto wait_carry = [&] {
                {
                    ZKGPT_PHASE_SCOPE("carry_wait");
                    char go = 0;
                    ssize_t r;
                    while ((r = read(gates[i - 1][0], &go, 1)) < 0 && errno == EINTR) {
                    }
                    if (r != 1) {
                        std::cerr << "[split] child " << i << ": upstream died\n";
                        _exit(1);
                    }
                    close(gates[i - 1][0]);
                }
                ZKGPT_PHASE_SCOPE("carry_import");
                carry.input_data.assign(shm->data, shm->data + shm->count);
                carry.input_offset = shm->input_offset;
                carry.len = shm->len;
                carry.feature_dim = shm->feature_dim;
                carry.input_c = shm->input_c;
                carry.input_e = shm->input_e;
            };
            if (ctrl) {
                pin_child_cores(i);  // controller mode: the build happened in the controller
            } else if (i == 0) {
                pin_child_cores(i);
                signal_started();  // child 0 builds unconditionally
            } else if (prebuild) {
                {
                    ZKGPT_PHASE_SCOPE("chain_wait");  // predecessor has its build slot
                    char go = 0;
                    ssize_t r;
                    while ((r = read(started[i - 1][0], &go, 1)) < 0 && errno == EINTR) {
                    }
                    if (r != 1) {
                        std::cerr << "[split] child " << i << ": upstream died\n";
                        _exit(1);
                    }
                    close(started[i - 1][0]);
                }
                acquire_build_slot();
                pin_child_cores(i);
                signal_started();
                // Placeholder of the boundary's known shape
                long long f = 1;
                while (f < cfg.embed_size)
                    f <<= 1;
                carry.input_data.assign((size_t)cfg.len * f, F_ZERO);
                carry.input_offset = 0;
                carry.len = cfg.len;
                carry.feature_dim = f;
                carry.input_c = 1;
                carry.input_e = 0;
            } else {
                wait_carry();
                acquire_build_slot();
                pin_child_cores(i);
                signal_started();
            }
            // Safe here and ONLY here: this process was forked before any HIP call
            gkr_config.witgen_gpu = want_witgen_gpu;
            gkr_config.round_gpu = want_round_gpu;
            gkr_config.gates_gpu = want_gates_gpu;
            gkr_config.witgen_full = want_witgen_full;

            PartitionState st;
            // Hands the carry to child i+1 through shm and opens its gate
            auto publish_carry = [&](const SplitBlockCarryState& nx) {
                ZKGPT_PHASE_SCOPE("carry_publish");
                if ((long long)nx.input_data.size() > cap) {
                    std::cerr << "[split] carry " << nx.input_data.size() << " > cap " << cap << "\n";
                    _exit(1);
                }
                std::memcpy(shm->data, nx.input_data.data(), nx.input_data.size() * sizeof(F));
                shm->count = nx.input_data.size();
                shm->input_offset = nx.input_offset;
                shm->len = nx.len;
                shm->feature_dim = nx.feature_dim;
                shm->input_c = nx.input_c;
                shm->input_e = nx.input_e;
                if (write(gates[i][1], "x", 1) != 1)
                    _exit(1);
                close(gates[i][1]);
            };
            // GKR_EARLY_CARRY=1
            const bool early = !ctrl && !is_last && getenv("GKR_EARLY_CARRY");
            SplitBlockCarryState early_carry;
            bool published = false;
            std::function<void(const prover&)> hook;
            if (early)
                hook = [&](const prover& pr) {
                    early_carry = st.net->exportSplitBlockCarry(pr);
                    publish_carry(early_carry);
                    published = true;
                    std::cout << "EARLY_CARRY idx=" << i << " t_us=" << (prof_now_us() - t0) / 1e6
                              << std::endl;
                };
            std::function<void(int&, int&)> pre_hook;
            if (prebuild)
                pre_hook = [&](int& ic, int& ie) {
                    const long long l0 = carry.len, f0 = carry.feature_dim;
                    wait_carry();
                    if (carry.len != l0 || carry.feature_dim != f0 || carry.input_offset != 0 ||
                        (long long)carry.input_data.size() != l0 * f0) {
                        std::cerr << "[split] child " << i << ": prebuild shape mismatch len=" << carry.len
                                  << "/" << l0 << " fdim=" << carry.feature_dim << "/" << f0
                                  << " off=" << carry.input_offset << " n=" << carry.input_data.size()
                                  << "\n";
                        _exit(1);
                    }
                    maybe_tamper_carry(carry, (std::size_t)i);
                    ic = carry.input_c;
                    ie = carry.input_e;
                };
            const double tb0 = prof_now_us();
            zkgpt_wc::last_load_hit() = false;
            if (!build_partition(cfg, i, (i > 0 && !ctrl) ? &carry : nullptr, st, hook, pre_hook))
                _exit(1);
            const double build_s = (prof_now_us() - tb0) / 1e6;
            if (ctrl) {
                if (!zkgpt_wc::last_load_hit()) {
                    std::cerr << "[split] child " << i << ": handoff MISS in " << wc_dir
                              << " -- refusing to prove a rebuilt (carry-less) partition\n";
                    _exit(1);
                }
                remove_part_files(wc_dir, i);  // free the /dev/shm copy
            }

            if (!is_last && !ctrl) {
                SplitBlockCarryState nx = st.net->exportSplitBlockCarry(*st.p);
                if (!published) {
                    publish_carry(nx);
                } else if (nx.input_data != early_carry.input_data ||
                           nx.input_offset != early_carry.input_offset || nx.len != early_carry.len ||
                           nx.feature_dim != early_carry.feature_dim || nx.input_c != early_carry.input_c ||
                           nx.input_e != early_carry.input_e) {
                    // Child i+1 already built from the early copy: its proof would be of the wrong input
                    std::cerr << "[split] child " << i
                              << ": EARLY carry differs from the"
                                 " end-of-build carry -- early publish is unsafe here\n";
                    _exit(1);
                }
            }
            std::cout << "OVERLAPPED_BUILT idx=" << i << " witgen_s=" << build_s << " witgen_fc_s=" << st.fc_s
                      << " witgen_host_peak_mib=" << st.witgen_host_peak_mib
                      << " witgen_resident_mib=" << st.witgen_resident_mib
                      << " witgen_h2d_gib=" << st.witgen_h2d_gib << " witgen_d2h_gib=" << st.witgen_d2h_gib
                      << " t_us=" << (prof_now_us() - t0) / 1e6 << std::endl;

            if (barrier) {
                ZKGPT_PHASE_SCOPE("barrier_wait");
                close(allgate[1]);  // child never writes the gate
                if (is_last) {
                    if (write(notify[1], "x", 1) != 1)
                        _exit(1);
                }
                close(notify[1]);
                char go = 0;  // blocks until parent's EOF
                while (read(allgate[0], &go, 1) < 0 && errno == EINTR) {
                }
                close(allgate[0]);
            }
            send_stat('B', i);
            {
                ZKGPT_PHASE_SCOPE("token_wait");  // acquire a proving slot
                send_stat('W', i);
                char t = 0;
                ssize_t r;
                while ((r = read(mem_gated ? ptok[i][0] : tok[0], &t, 1)) < 0 && errno == EINTR) {
                }
                if (r != 1) {
                    std::cerr << "[split] child " << i << ": prove gate closed\n";
                    _exit(1);
                }
            }
            zkgpt_gpu::gkr_vram_peak_start();
            PartResult res;
            // Peak-release semaphore
            if (mem_gated)
                g_past_peak_hook = [&] { send_stat('R', i); };
            bool ok = prove_built_partition(cfg, i, st, res, tpc);
            g_past_peak_hook = nullptr;
            gkr_sc_teardown();
            gkr_pred_report();
            gkr_pred_teardown();
            zkgpt_gpu::gkr_vram_peak_stop();
            // Return the slot
            if (!mem_gated && write(tok[1], "x", 1) != 1)
                ok = false;
            unpin_child_cores();
            std::cout.flush();
            _exit(ok ? 0 : 1);
        }
        if (pid < 0) {
            std::perror("[split] fork");
            all_ok = false;
            break;
        }
        pids.push_back(pid);
    }

    for (auto& g : gates) {
        close(g[0]);
        close(g[1]);
    }
    for (auto& r : ready) {
        close(r[0]);
        close(r[1]);
    }  // controller death -> EOF in children
    double prep_s = 0, makespan_s = 0, t_release = 0;
    if (barrier) {
        close(allgate[0]);  // parent only writes the gate
        close(notify[1]);   // only the last child writes
        char done = 0;
        ssize_t r;
        while ((r = read(notify[0], &done, 1)) < 0 && errno == EINTR) {
        }
        close(notify[0]);
        if (r != 1) {
            std::cerr << "[split] barrier: last child died before witgen\n";
            all_ok = false;
        }
        t_release = prof_now_us();
        prep_s = (t_release - t0) / 1e6;
        close(allgate[1]);  // EOF == go, to every child
    }
    // Memory-gated mode
    std::vector<bool> reaped(n, false);
    std::vector<int> exit_stt(n, 0);
    if (mem_gated && all_ok) {
        const char* env = getenv("GKR_MIN_FREE_MIB");
        const long long min_free_kib = (env ? atoll(env) : 40960LL) * 1024LL;
        const int gpu_cap = getenv("GKR_GPU_PROCS") ? atoi(getenv("GKR_GPU_PROCS")) : 0;
        long long peak_kib =
            (getenv("GKR_PART_PEAK_MIB") ? atoll(getenv("GKR_PART_PEAK_MIB")) : 0LL) * 1024LL;
        long long built_kib =
            (getenv("GKR_PART_BUILT_MIB") ? atoll(getenv("GKR_PART_BUILT_MIB")) : 0LL) * 1024LL;
        VramDev vdev;
        bool vram_gate = false;
        {
            const char* g = getenv("GKR_VRAM_GATE");
            const char* hv = getenv("HIP_VISIBLE_DEVICES");
            const bool single = devlist.empty() && ctrl_pid <= 0 && (!hv || !strchr(hv, ','));
            if ((!g || strcmp(g, "0")) && single)
                vram_gate = vram_dev_for_hip_index(hv && *hv ? atoi(hv) : 0, vdev);
        }
        long long vpeak_kib =
            (getenv("GKR_PART_VRAM_MIB") ? atoll(getenv("GKR_PART_VRAM_MIB")) : 0LL) * 1024LL;
        long long vwg_kib =
            (getenv("GKR_PART_WG_VRAM_MIB") ? atoll(getenv("GKR_PART_WG_VRAM_MIB")) : 0LL) * 1024LL;
        const long long vmargin_kib =
            (getenv("GKR_VRAM_MARGIN_MIB") ? atoll(getenv("GKR_VRAM_MARGIN_MIB")) : 2048LL) * 1024LL;
        std::vector<long long> vhwm(n, 0);
        // Once partitions have FINISHED
        std::vector<long long> built_obs(n, 0);
        long long done_peak = 0, done_vpeak = 0, done_built = 0, done_vbuild = 0;
        std::vector<long long> vbuild_obs(n, 0);
        if (vram_gate)
            std::cerr << "[split] vram-gated: gpu_id=" << vdev.gpu_id
                      << " free=" << vram_free_kib(vdev) / 1048576.0
                      << " GiB prove_est=" << vpeak_kib / 1048576.0
                      << " GiB build_est=" << vwg_kib / 1048576.0 << " GiB margin=" << vmargin_kib / 1048576.0
                      << " GiB\n";
        close(stat[1]);
        fcntl(stat[0], F_SETFL, fcntl(stat[0], F_GETFL) | O_NONBLOCK);
        std::vector<char> state(n, 'F');
        std::vector<long long> hwm(n, 0), rhold(n, 0);  // rhold: max RSS seen while 'R'
        state[0] = 'b';
        int live = n, prove_tok = 0, build_tok = gate_builds ? 1 : n;
        double last_tick_us = prof_now_us();
        std::cerr << "[split] mem-gated: min_free=" << min_free_kib / 1048576.0
                  << " GiB peak_est=" << peak_kib / 1048576.0 << " GiB built_est=" << built_kib / 1048576.0
                  << " GiB tpc=" << tpc << "\n";
        while (live > 0) {
            ChainStat m;
            while (read(stat[0], &m, sizeof m) == (ssize_t)sizeof m)
                if (m.idx >= 0 && m.idx < n && state[m.idx] != 'X') {
                    state[m.idx] = m.ev;
                    if (m.ev == 'R')
                        std::cerr << "[split] mem-gated: partition " << m.idx
                                  << " past peak (rss=" << proc_rss_kib(pids[m.idx]) / 1048576.0
                                  << " GiB, peak "
                                  << std::max(hwm[m.idx], proc_hwm_kib(pids[m.idx])) / 1048576.0 << " GiB)\n";
                }
            int stt = 0;
            pid_t r;
            while ((r = waitpid(-1, &stt, WNOHANG)) > 0) {
                for (int i = 0; i < n; ++i)
                    if (pids[i] == r && !reaped[i]) {
                        reaped[i] = true;
                        exit_stt[i] = stt;
                        state[i] = 'X';
                        --live;
                        if (!WIFEXITED(stt) || WEXITSTATUS(stt) != 0) {
                            std::cerr << "[split] child pid " << r << " failed (status " << stt << ")\n";
                            all_ok = false;
                        } else {
                            done_peak = std::max(done_peak, hwm[i]);
                            done_vpeak = std::max(done_vpeak, vhwm[i]);
                            done_built = std::max(done_built, built_obs[i]);
                            done_vbuild = std::max(done_vbuild, vbuild_obs[i]);
                            const long long op = peak_kib, ob = built_kib;
                            if (done_peak > 0)
                                peak_kib = done_peak * 11 / 10;
                            if (done_built > 0)
                                built_kib = done_built * 11 / 10;
                            if (vram_gate && done_vpeak > 0)
                                vpeak_kib = done_vpeak * 11 / 10;
                            if (vram_gate && done_vbuild > 0)
                                vwg_kib = done_vbuild * 11 / 10;
                            std::cerr
                                << "[split] mem-gated: partition " << i << " done: host peak "
                                << hwm[i] / 1048576.0 << " GiB -> peak_est " << op / 1048576.0 << " -> "
                                << peak_kib / 1048576.0 << " GiB, built_est " << ob / 1048576.0 << " -> "
                                << built_kib / 1048576.0 << " GiB"
                                << (vram_gate ? ", vprove_est " + std::to_string(vpeak_kib / 1024) + " MiB"
                                              : std::string())
                                << "\n";
                        }
                    }
            }
            if (!all_ok)
                break;
            long long growth = 0, build_growth = 0, vgrowth = 0, our_rss = 0, our_vram = 0;
            int nP = 0, nb = 0, nwait = 0, headW = -1, headw = -1;
            for (int i = 0; i < n; ++i) {
                if (state[i] == 'X' || state[i] == 'F' || state[i] == 'w') {
                    if (state[i] == 'w' && headw < 0)
                        headw = i;
                    continue;
                }
                hwm[i] = std::max(hwm[i], proc_hwm_kib(pids[i]));
                if (state[i] == 'P' || state[i] == 'R') {
                    peak_kib = std::max(peak_kib, hwm[i]);
                    ++nP;
                }
                if (state[i] == 'B' || state[i] == 'W') {
                    built_kib = std::max(built_kib, hwm[i]);
                    built_obs[i] = std::max(built_obs[i], hwm[i]);
                }
                if (state[i] == 'W' && headW < 0)
                    headW = i;
                // Full charge, not "remaining growth": a granted slot reserves its
                if (state[i] == 'P' || state[i] == 'W' || state[i] == 'B')
                    growth += peak_kib;
                else if (state[i] == 'R') {  // past peak: what it holds, from its own RSS
                    rhold[i] = std::max(rhold[i], proc_rss_kib(pids[i]));
                    growth += rhold[i] * 11 / 10;
                } else if (state[i] == 'b') {
                    growth += built_kib;
                    ++nb;
                }
                if (state[i] == 'b' || state[i] == 'B' || state[i] == 'W')
                    ++nwait;
                our_rss += proc_rss_kib(pids[i]);
                if (vram_gate) {
                    const long long cur = proc_vram_kib(pids[i], vdev);
                    our_vram += cur;
                    vhwm[i] = std::max(vhwm[i], cur);
                    if (state[i] == 'P') {
                        vpeak_kib = std::max(vpeak_kib, vhwm[i]);
                        vgrowth += std::max(0LL, vpeak_kib - cur);
                    } else if (state[i] == 'b') {
                        vwg_kib = std::max(vwg_kib, vhwm[i]);
                        vbuild_obs[i] = std::max(vbuild_obs[i], vhwm[i]);
                        vgrowth += std::max(0LL, vwg_kib - cur);
                    }
                }
            }
            build_growth = growth + (long long)nb * (peak_kib - built_kib);
            const long long avail = mem_available_kib();
            const long long room = avail - growth - min_free_kib;
            const long long build_room = avail - build_growth - min_free_kib;
            const long long vfree = vram_gate ? vram_free_kib(vdev) : 0;
            // Strict on VRAM as on host
            const long long vroom = vram_gate ? vfree - vgrowth - vmargin_kib : 0;
            const bool v_prove_ok = !vram_gate || vroom - vpeak_kib >= 0;
            long long kcap = peak_kib > 0 ? (avail + our_rss - min_free_kib) / peak_kib : n;
            if (vram_gate && vpeak_kib > 0)
                kcap = std::min(kcap, (vfree + our_vram - vmargin_kib) / vpeak_kib);
            kcap = std::max(1LL, std::min<long long>(kcap, n));
            const bool cap_ok = nwait < kcap;
            // GKR_GPU_PROCS=K
            const bool gpu_ok = gpu_cap <= 0 || nP + nwait < gpu_cap;
            const bool thr_ok = (long long)(nP + nwait + 1) * tpc <= cfg.num_threads;
            const bool v_build_ok = !vram_gate || vroom - vwg_kib >= 0;
            bool granted = false;
            if (headW >= 0 && prove_tok < n && ((room >= 0 && v_prove_ok) || nP == 0)) {
                granted = true;
                if (room < 0)
                    std::cerr << "[split] mem-gated: nothing proving -- releasing partition " << headW
                              << " to prove despite room=" << room / 1048576.0 << " GiB\n";
                if (write(ptok[headW][1], "x", 1) != 1) {
                    std::perror("[split] token");
                    all_ok = false;
                    break;
                }
                ++prove_tok;
                state[headW] = 'P';
                std::cerr << "[split] mem-gated: prove slot -> partition " << headW
                          << " (avail=" << avail / 1048576.0 << " GiB owed=" << growth / 1048576.0
                          << " GiB proving=" << nP + 1 << ")\n";
            } else if (headw >= 0 && build_tok < n &&
                       ((build_room - built_kib >= 0 && v_build_ok && cap_ok && gpu_ok && thr_ok) ||
                        (nP == 0 && headW < 0 && nb == 0))) {
                granted = true;
                if (write(btok[1], "x", 1) != 1) {
                    std::perror("[split] btoken");
                    all_ok = false;
                    break;
                }
                ++build_tok;
                state[headw] = 'b';
                std::cerr << "[split] mem-gated: build slot -> partition " << headw
                          << " (avail=" << avail / 1048576.0 << " GiB owed=" << growth / 1048576.0
                          << " GiB proving=" << nP << " witgen_cap=" << kcap << " pending=" << nwait + 1
                          << ")\n";
            }
            if (prof_now_us() - last_tick_us >= 60e6) {  // progress line once a minute
                last_tick_us = prof_now_us();
                std::cerr << "[split] mem-gated: t+" << (int)((prof_now_us() - t0) / 1e6)
                          << "s avail=" << avail / 1048576.0 << " GiB owed=" << growth / 1048576.0
                          << " GiB proving=" << nP << " pending=" << nwait << " witgen_cap=" << kcap
                          << " peak_est=" << peak_kib / 1048576.0
                          << " GiB built_est=" << built_kib / 1048576.0 << " GiB"
                          << (vram_gate ? " vram_free=" + std::to_string(vfree / 1024) +
                                              "MiB vram_owed=" + std::to_string(vgrowth / 1024) +
                                              "MiB vprove_est=" + std::to_string(vpeak_kib / 1024) +
                                              "MiB vbuild_est=" + std::to_string(vwg_kib / 1024) + "MiB"
                                        : std::string())
                          << "\n";
            }
            // Event-driven
            if (live > 0 && !granted) {
                struct pollfd pf{stat[0], POLLIN, 0};
                poll(&pf, 1, 1000);
            }
        }
        if (!all_ok)  // one child died: don't strand the rest in a gate
            for (int i = 0; i < n; ++i)
                if (!reaped[i])
                    kill(pids[i], SIGTERM);
        close(stat[0]);
    }
    if (mem_gated) {
        close(btok[0]);
        close(btok[1]);
    }
    for (auto& t : ptok) {
        close(t[0]);
        close(t[1]);
    }
    close(tok[0]);
    close(tok[1]);

    for (size_t k = 0; k < pids.size(); ++k) {
        if (k < reaped.size() && reaped[k])
            continue;  // already collected by the release loop
        const pid_t pid = pids[k];
        int stt = 0;
        if (waitpid(pid, &stt, 0) < 0) {
            std::perror("[split] waitpid");
            all_ok = false;
            continue;
        }
        if (!WIFEXITED(stt) || WEXITSTATUS(stt) != 0) {
            std::cerr << "[split] child pid " << pid << " failed (status " << stt << ")\n";
            all_ok = false;
        }
    }
    if (ctrl_pid > 0) {
        int stt = 0;
        if (!all_ok)
            kill(ctrl_pid, SIGTERM);
        if (waitpid(ctrl_pid, &stt, 0) < 0 || !WIFEXITED(stt) || WEXITSTATUS(stt) != 0) {
            std::cerr << "[split] witgen controller failed (status " << stt << ")\n";
            all_ok = false;
        }
    }
    const double total_s = (prof_now_us() - t0) / 1e6;
    if (ctrl) {  // leftovers from a failed run
        for (int i = 0; i < n; ++i)
            remove_part_files(wc_dir, i);
        ::rmdir(wc_dir.c_str());
    }

    if (barrier)
        makespan_s = (prof_now_us() - t_release) / 1e6;
    zkgpt_sampler::stop(sampler);
    report_vram_peak(cfg, t0, prof_now_us(), "witgen_chain");
    munmap(raw, shm_bytes);
    all_ok = link_end("overlapped", all_ok);

    std::cout << "SPLIT_OVERLAPPED parts=" << n << " batch=" << (mem_gated ? -1 : b)
              << " mem_gated=" << (mem_gated ? 1 : 0) << " threads_per_child=" << tpc
              << " threads_total=" << cfg.num_threads << " witgen_gpu=" << (want_witgen_gpu ? 1 : 0)
              << " round_gpu=" << (want_round_gpu ? 1 : 0) << " gates_gpu=" << (want_gates_gpu ? 1 : 0)
              << " witgen_full=" << (want_witgen_full ? 1 : 0) << " witgen_device=" << (ctrl ? wdev : "-1")
              << " carry_mib=" << (shm_bytes / 1048576.0) << " barrier=" << (barrier ? 1 : 0)
              << " prep_s=" << (barrier ? prep_s : 0.0)
              << " barrier_makespan_s=" << (barrier ? makespan_s : 0.0) << " overlapped_total_s=" << total_s
              << " status=" << (all_ok ? "OK" : "FAIL") << std::endl;
    return all_ok;
}

int main(int argc, char** argv) {
    initPairing(mcl::BN254);
    // The VRAM sampler is started AFTER arg parsing

    int depth = 12;
    int embed_size = 768;
    int lin_dim = -1;  // FFN width
    int num_splits = 2;
    int len = 30;
    int num_threads = 32;

    // Positional: depth embed_size num_splits [len] [threads]
    std::vector<int> explicit_splits, part_cs, part_mks, part_rks, part_lls, part_lks;
    bool unique_parts = false;  // unique-parts: prove one block per structural role
    // concurrent [W]: prove the partitions in W concurrent child processes
    bool concurrent = false;
    bool witgen_barrier = false;  // witgen-barrier: all witgen, THEN all proving
    // batched [B]: B = partitions proved concurrently per batch
    bool pipeline = false;
    int pipe_batch = 0;  // 0 => one batch, degenerates to --concurrent
    // overlapped [B]: fork before any HIP so children can do GPU witgen
    bool witgen_chain = false;
    int chain_batch = 0;
    const bool want_witgen_gpu = getenv("GKR_WITGEN_GPU") != nullptr;
    const bool want_round_gpu = getenv("GKR_ROUND_GPU") != nullptr;
    const bool want_gates_gpu = getenv("GKR_GATES_GPU") != nullptr;
    const bool want_witgen_full = getenv("GKR_WITGEN_FULL") != nullptr || getenv("GKR_GELU_GPU") != nullptr;
    gkr_config.witgen_gpu = false;
    gkr_config.round_gpu = false;
    gkr_config.gates_gpu = false;
    gkr_config.witgen_full = false;
    gkr_config.witgen_check = getenv("GKR_WITGEN_CHECK") != nullptr;
    gkr_config.round_check = getenv("GKR_ROUND_CHECK") != nullptr;
    gkr_config.gates_check = getenv("GKR_GATES_CHECK") != nullptr;
    gkr_config.witgen_full_check =
        getenv("GKR_WITGEN_FULL_CHECK") != nullptr || getenv("GKR_GELU_CHECK") != nullptr;
    bool witgen_only = false;  // witgen-only : build+cache every partition, skip proving
    int conc_width = 0;        // 0 => all partitions at once
    // threads-total: host thread budget
    int threads_total = 0;
    std::string vram_csv;
    double vram_interval = 0.001;
    {
        std::vector<int> pos;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto next = [&] { return (i + 1 < argc) ? atoi(argv[++i]) : 0; };
            auto next_str = [&]() -> std::string {
                return (i + 1 < argc) ? std::string(argv[++i]) : std::string();
            };
            if (a == "--c")
                gkr_config.range_c = next();
            else if (a == "--matmul-k")
                gkr_config.matmul_k = next();
            else if (a == "--range-k")
                gkr_config.range_k = next();
            else if (a == "--lasso-l")
                gkr_config.lasso_l = next();
            else if (a == "--lasso-k")
                gkr_config.lasso_k = next();
            else if (a == "--sc-tile-bits")
                gkr_config.sc_tile_bits = next();
            // Per-phase CPU reference paths
            else if (a == "--cpu")
                gkr_config.range_cpu = true;
            else if (a == "--cpu-deg3")
                gkr_config.deg3_cpu = true;
            else if (a == "--cpu-open")
                gkr_config.open_cpu = true;
            else if (a == "--cpu-commit")
                gkr_config.commit_cpu = true;
            else if (a == "--partitions")
                gkr_config.range_partitions = next();
            else if (a == "--vram-gb")
                gkr_config.vram_gb = (i + 1 < argc) ? atof(argv[++i]) : 64.0f;
            else if (a == "--splits")
                explicit_splits = parse_int_csv(next_str());
            else if (a == "--cs")
                part_cs = parse_int_csv(next_str());
            else if (a == "--matmul-ks")
                part_mks = parse_int_csv(next_str());
            else if (a == "--range-ks")
                part_rks = parse_int_csv(next_str());
            else if (a == "--lasso-ls")
                part_lls = parse_int_csv(next_str());
            else if (a == "--lasso-ks")
                part_lks = parse_int_csv(next_str());
            else if (a == "--unique-parts")
                unique_parts = true;
            else if (a == "--witgen-only")
                witgen_only = true;
            else if (a == "--uvm-prefetch")
                gkr_config.uvm_prefetch = true;
            else if (a == "--concurrent") {
                concurrent = true;
                if (i + 1 < argc && argv[i + 1][0] != '-')
                    conc_width = atoi(argv[++i]);
            }
            // pipeline / --witgen-chain are the original spellings
            else if (a == "--batched" || a == "--pipeline") {
                pipeline = true;
                if (i + 1 < argc && argv[i + 1][0] != '-')
                    pipe_batch = atoi(argv[++i]);
            } else if (a == "--witgen-barrier") {
                witgen_barrier = true;
            } else if (a == "--overlapped" || a == "--witgen-chain") {
                witgen_chain = true;
                // Accept a bare number OR a negative one
                if (i + 1 < argc &&
                    (argv[i + 1][0] != '-' || (argv[i + 1][1] && isdigit((unsigned char)argv[i + 1][1]))))
                    chain_batch = atoi(argv[++i]);
            } else if (a == "--threads-total")
                threads_total = next();
            else if (a == "--vram-csv")
                vram_csv = next_str();
            else if (a == "--vram-interval")
                vram_interval = (i + 1 < argc) ? atof(argv[++i]) : 0.001;
            else if (!a.empty() && a[0] != '-')
                pos.push_back(atoi(a.c_str()));
            else {
                std::cerr << "unknown arg: " << a << "\n";
                return EXIT_FAILURE;
            }
        }
        if (pos.size() >= 1)
            depth = pos[0];
        if (pos.size() >= 2)
            embed_size = pos[1];
        if (pos.size() >= 3)
            num_splits = pos[2];
        if (pos.size() >= 4)
            len = pos[3];
        if (pos.size() >= 5)
            num_threads = pos[4];
        if (pos.size() >= 6)
            lin_dim = pos[5];  // FFN width (like the non-split baseline's lin_dim arg)
    }
    if (lin_dim <= 0)
        lin_dim = embed_size * 3;  // back-compat default when omitted

    if (depth <= 0 || num_splits < 1) {
        std::cerr << "Usage: main_gkr_split <depth> <embed_size> <num_splits> [len] [threads] [lin_dim]"
                     " [--splits a,b,c] [--c N|--cs a,b,c] [--matmul-k N|--matmul-ks a,b,c]"
                     " [--range-k N|--range-ks a,b,c] [--range-size N] [--vram-gb N]\n";
        return EXIT_FAILURE;
    }

    // Boundary composition
    std::vector<int> split_depths;
    if (!explicit_splits.empty()) {
        int sum = 0;
        bool bad = false;
        for (int d : explicit_splits) {
            if (d < 1)
                bad = true;
            sum += d;
        }
        if (bad || sum != depth) {
            std::cerr << "--splits must be positive ints summing to depth=" << depth << " (got sum=" << sum
                      << ")\n";
            return EXIT_FAILURE;
        }
        split_depths = explicit_splits;
        num_splits = static_cast<int>(split_depths.size());
    } else {
        num_splits = std::max(1, std::min(depth, num_splits));
        split_depths.assign(num_splits, depth / num_splits);
        for (int i = 0; i < depth % num_splits; ++i)
            ++split_depths[i];
    }

    for (const auto* v : {&part_cs, &part_mks, &part_rks})
        if (!v->empty() && v->size() != 1 && (int)v->size() != num_splits)
            std::cerr << "[split] warning: per-partition knob vector size " << v->size() << " != #partitions "
                      << num_splits << " (clamping to last)\n";

    std::cerr << "[split] depth=" << depth << " embed=" << embed_size << " lin=" << lin_dim
              << " splits=" << num_splits << " len=" << len << " threads=" << num_threads << " split_depths:";
    for (auto d : split_depths)
        std::cerr << ' ' << d;
    std::cerr << std::endl;

    {
        int w = 1;
        while (w < len)
            w <<= 1;
        len = w;
    }
    std::cerr << "[split] len=" << len << " merge=" << kMerge << std::endl;

    std::map<int, long long> probe_cache;
    // The probe's one output depends only on the block geometry
    const std::string probe_dir = [] {
        const char* e = getenv("GKR_PROBE_CACHE");
        if (e)
            return std::string(strcmp(e, "0") ? e : "");
        const char* h = getenv("HOME");
        return h ? std::string(h) + "/.cache/zkml_probe" : std::string();
    }();
    auto probe_file = [&](int d) {
        return probe_dir + "/d" + std::to_string(d) + "_e" + std::to_string(embed_size) + "_l" +
               std::to_string(lin_dim) + "_len" + std::to_string(len) + "_m" + std::to_string(kMerge);
    };
    auto probe_load = [&](int d, long long& v) {
        if (probe_dir.empty())
            return false;
        std::ifstream f(probe_file(d));
        return bool(f >> v) && v > 0;
    };
    auto probe_store = [&](int d, long long v) {
        if (probe_dir.empty())
            return;
        mkdir(probe_dir.c_str(), 0750);
        const std::string p = probe_file(d), tmp = p + ".tmp" + std::to_string(getpid());
        {
            std::ofstream f(tmp);
            f << v << "\n";
            if (!f)
                return;
        }
        rename(tmp.c_str(), p.c_str());
    };
    auto probe_boundary_sizes = [&](std::vector<long long>& tiws, std::vector<u64>& scs) {
        tiws.assign(num_splits, 0);
        scs.assign(num_splits > 1 ? num_splits - 1 : 0, 0);
        for (int i = 0; i < num_splits; ++i) {
            const int key = split_depths[i];
            auto it = probe_cache.find(key);
            long long cached = 0;
            if (it == probe_cache.end() && probe_load(key, cached)) {
                std::cerr << "[split] probe d=" << key << " from " << probe_file(key) << std::endl;
                it = probe_cache.emplace(key, cached).first;
            }
            if (it == probe_cache.end()) {
                // The probe pass is a full witgen per partition
                zkgpt_phase::set_part(i);
                ZKGPT_PHASE_SCOPE("probe_witgen");
                // Run in a forked child with the full thread budget
                int fds[2];
                if (pipe(fds) != 0) {
                    perror("probe pipe");
                    exit(EXIT_FAILURE);
                }
                std::cout.flush();
                std::cerr.flush();
                fflush(nullptr);
                const pid_t pid = fork();
                if (pid < 0) {
                    perror("probe fork");
                    exit(EXIT_FAILURE);
                }
                if (pid == 0) {
                    close(fds[0]);
                    // all-false flags
                    prover probe_p;
                    probe_p.max_threads = num_threads;
                    std::unique_ptr<neuralNetwork> probe_nn(
                        new LLMSplitBlock(split_depths[i], embed_size / 64, 64, embed_size, lin_dim, len,
                                          false, false, false, nullptr, nullptr));
                    timer probe_circuit_timer;
                    probe_nn->create(probe_p, kMerge, probe_circuit_timer);
                    const long long v = probe_nn->getTotalInputSize();
                    std::cout.flush();
                    std::cerr.flush();
                    fflush(nullptr);
                    _exit(write(fds[1], &v, sizeof v) == (ssize_t)sizeof v ? 0 : 1);
                }
                close(fds[1]);
                long long v = 0;
                const bool got = read(fds[0], &v, sizeof v) == (ssize_t)sizeof v;
                close(fds[0]);
                int st = 0;
                waitpid(pid, &st, 0);
                if (!got || !WIFEXITED(st) || WEXITSTATUS(st) != 0 || v <= 0) {
                    std::cerr << "[split] probe d=" << key << " failed (status " << st << ")" << std::endl;
                    exit(EXIT_FAILURE);
                }
                it = probe_cache.emplace(key, v).first;
                probe_store(key, it->second);
            }
            tiws[i] = it->second;
            std::cout << "Block " << (i + 1) << " total_input_witness_size (probe): " << tiws[i] << std::endl;

            if (i > 0) {
                u64 shared = next_pow2_size(static_cast<u64>(tiws[i] >> 1));
                scs[i - 1] = shared;
                tiws[i] = static_cast<long long>(shared << 1);
                std::cout << "Shared commitment size blocks " << i << "-" << (i + 1) << ": " << shared
                          << std::endl;
            }
        }
        zkgpt_phase::set_part(-1);
        std::cerr << "[split] probe done" << std::endl;
    };

    std::vector<long long> total_input_witness_sizes;
    std::vector<u64> shared_commit_sizes;
    probe_boundary_sizes(total_input_witness_sizes, shared_commit_sizes);

    SplitCfg cfg;
    cfg.split_depths = split_depths;
    cfg.tiws = total_input_witness_sizes;
    cfg.scs = shared_commit_sizes;
    cfg.scs.resize(num_splits, 0);  // scs[last] is never read
    cfg.embed_size = embed_size;
    cfg.lin_dim = lin_dim;
    cfg.len = len;
    cfg.num_threads = num_threads;
    cfg.part_cs = part_cs;
    cfg.part_mks = part_mks;
    cfg.part_rks = part_rks;
    cfg.part_lls = part_lls;
    cfg.part_lks = part_lks;
    // Snapshot before any partition mutates gkr_config
    cfg.def_c = gkr_config.range_c;
    cfg.def_mk = gkr_config.matmul_k;
    cfg.def_rk = gkr_config.range_k;
    cfg.def_ll = gkr_config.lasso_l;
    cfg.def_lk = gkr_config.lasso_k;
    if (threads_total > 0)
        cfg.num_threads = threads_total;
    // The whole process
    omp_set_num_threads(std::max(1, cfg.num_threads));
    cfg.witgen_only = witgen_only;
    cfg.vram_csv = vram_csv;
    cfg.vram_interval = vram_interval;

    bool ok;
    if (witgen_chain) {
        if (concurrent || pipeline || unique_parts || witgen_only) {
            std::cerr << "--overlapped is exclusive with the other run modes\n";
            return EXIT_FAILURE;
        }
        if (num_splits < 2) {
            std::cerr << "--overlapped needs at least 2 partitions\n";
            return EXIT_FAILURE;
        }
        // gkr_config.witgen_gpu stays false here
        return run_split_overlapped(cfg, chain_batch, want_witgen_gpu, want_round_gpu, want_gates_gpu,
                                    want_witgen_full, witgen_barrier)
                   ? EXIT_SUCCESS
                   : EXIT_FAILURE;
    }
    if (pipeline) {
        if (concurrent || unique_parts || witgen_only) {
            std::cerr << "--batched is exclusive with --concurrent/--unique-parts/--witgen-only\n";
            return EXIT_FAILURE;
        }
        if (num_splits < 2) {
            std::cerr << "--batched needs at least 2 partitions\n";
            return EXIT_FAILURE;
        }
        // The parent builds, so GPU witgen would give it a HIP context before fork()
        if (want_witgen_gpu || want_gates_gpu || want_witgen_full) {
            std::cerr << "--batched is incompatible with GKR_WITGEN_GPU/GKR_GATES_GPU: the parent "
                         "builds, so it would hold a HIP context at fork() and every "
                         "child segfaults. Use --overlapped for GPU witgen.\n";
            return EXIT_FAILURE;
        }
        return run_split_batched(cfg, pipe_batch) ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (concurrent) {
        if (witgen_only) {
            std::cerr << "--witgen-only is not supported with --concurrent (it builds "
                         "every partition in the sequential parent already; run "
                         "single-partition --witgen-only processes concurrently instead)\n";
            return EXIT_FAILURE;
        }
        if (num_splits < 2) {
            std::cerr << "--concurrent needs at least 2 partitions\n";
            return EXIT_FAILURE;
        }
        if (unique_parts) {
            std::cerr << "--concurrent and --unique-parts are mutually exclusive: "
                         "--unique-parts deliberately emits an invalid proof\n";
            return EXIT_FAILURE;
        }
        // Teardown happens per-child
        return run_split_concurrent(cfg, conc_width) ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    gkr_config.witgen_gpu = want_witgen_gpu;
    gkr_config.round_gpu = want_round_gpu;
    gkr_config.gates_gpu = want_gates_gpu;
    gkr_config.witgen_full = want_witgen_full;
    zkgpt_gpu::gkr_vram_peak_start();  // background device-VRAM high-water sampler
    // Same device-wide sampler as the concurrent path
    const pid_t seq_sampler =
        cfg.vram_csv.empty() ? -1 : zkgpt_sampler::start(cfg.vram_csv, cfg.vram_interval, prof_now_us);
    const double t_seq0 = prof_now_us();
    ok = run_split_sequential(cfg, unique_parts);
    const double t_seq1 = prof_now_us();
    zkgpt_sampler::stop(seq_sampler);
    report_vram_peak(cfg, t_seq0, t_seq1, "sequential");
    // gkr_sc_end keeps its device buffers alive across layers AND across partitions
    gkr_sc_teardown();
    gkr_pred_report();
    gkr_pred_teardown();
    zkgpt_gpu::gkr_vram_peak_stop();  // join sampler while the HIP runtime is still alive
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
