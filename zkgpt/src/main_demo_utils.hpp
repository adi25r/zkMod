#include <cstdlib>
#include <unistd.h>
#include <sys/wait.h>
#pragma once

#include <algorithm>
#include <chrono>
#include <iostream>
#include <thread>
#include <utility>
#include <vector>
#include <sys/resource.h>
#ifdef __GLIBC__
#include <malloc.h>
#endif

#include "verifier.hpp"
#include "range_prover.hpp"
#include "utils.hpp"
#include "timer.hpp"

struct VerifierJob {
    verifier *instance;
    prover *p;
    range_prover *rp;
    timer *circuit_timer;
    u64 shared_commit_size;
    int commit_threads;
    const char *label;
    double standalone_commit_time = 0.0;
};


struct OwnedCommitment {
    int l = 0;
    std::vector<Fr> w;
    std::vector<ll> ww;
    std::vector<G1> gens;
    std::vector<G1> comm;
    G1 G;

    void clear() {
        l = 0;
        w.clear();
        ww.clear();
        gens.clear();
        comm.clear();
        G.clear();
    }

    bool empty() const { return l == 0 || comm.empty(); }

    void capture(const Commit_return &src) {
        clear();
        if (src.l <= 0 || src.w == nullptr || src.ww == nullptr ||
            src.g == nullptr || src.comm == nullptr) {
            return;
        }
        l = src.l;
        const std::size_t witness_count = std::size_t(1) << l;
        const std::size_t gen_count = (std::size_t(1) << (l - (l >> 1))) + 1;
        const std::size_t comm_count = std::size_t(1) << (l >> 1);
        w.assign(src.w, src.w + witness_count);
        ww.assign(src.ww, src.ww + witness_count);
        gens.assign(src.g, src.g + gen_count);
        comm.assign(src.comm, src.comm + comm_count);
        G = src.G;
    }

    Commit_return view() {
        Commit_return ret{};
        if (empty()) return ret;
        ret.l = l;
        ret.w = w.data();
        ret.ww = ww.data();
        ret.g = gens.data();
        ret.G = G;
        ret.comm = comm.data();
        return ret;
    }
};

inline void commit_private_output_for_next(prover &p, u64 shared_commit_size, int max_threads) {
    int logn = ceilPow2BitLength(static_cast<u32>(shared_commit_size));
    u64 n_sqrt = 1ULL << (logn - (logn >> 1));
    p.gens_output.clear();
    p.gens_output.resize(n_sqrt);
    G1 base = gen_gi(p.gens_output.data(), n_sqrt);
    p.gens_output.push_back(base);
    p.commitOutputPrivate(p.gens_output, max_threads);
}

inline void release_commitment(Commit_return &commitment) {
    delete[] commitment.w;
    delete[] commitment.ww;
    delete[] commitment.g;
    delete[] commitment.comm;
    commitment = {};
}

inline void release_partition_commitment(Commit_return &commitment) {
    delete[] commitment.w;
    delete[] commitment.ww;
    delete[] commitment.comm;
    commitment = {};
}

inline void release_partition_local_commitments(prover &p, bool release_external_input = false) {
    release_partition_commitment(p.cc);
    if (!p.input_committed_externally) {
        release_partition_commitment(p.cc_input);
    } else if (release_external_input) {
        release_commitment(p.cc_input);
    } else {
        p.cc_input = {};
    }
    release_partition_commitment(p.cc_output);
    for (auto &segment : p.input_commit_segments) {
        if (segment.external_commit == nullptr) {
            release_partition_commitment(segment.commit);
        } else {
            segment.commit = {};
        }
    }
    p.input_commit_segments.clear();
}

inline void trim_allocator_to_os() {
#ifdef __GLIBC__
    malloc_trim(0);
#endif
}

long get_vm_peak_kb() {
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256];
    long peak = 0;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "VmPeak: %ld", &peak) == 1) break;
    }
    fclose(f);
    return peak;
}

long get_current_rss_kb() {
    FILE* f = fopen("/proc/self/status", "r");
    char line[256];
    long rss = 0;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "VmRSS: %ld", &rss) == 1) break;
    }
    fclose(f);
    return rss;  // in KB
}

inline u64 next_pow2_size(u64 n) {
    return 1ULL << ceilPow2BitLength(n);
}

inline u64 shared_commitment_size(u64 lhs_size, u64 rhs_size) {
    return next_pow2_size(std::max(lhs_size, rhs_size));
}

inline void pad_output_layer(prover &p, u64 target_size) {
    auto &out_layer = p.C.circuit[p.C.size - 1];
    if (out_layer.size >= target_size) return;

    const auto padded_size = next_pow2_size(target_size);
    std::cout << "Padding output layer from " << out_layer.size << " to " << padded_size << std::endl;
    auto &out_vals = p.val[p.C.size - 1];
    auto original_size = out_vals.size();
    out_vals.resize(padded_size);
    for (std::size_t i = original_size; i < padded_size; ++i) out_vals[i].clear();

    out_layer.size = padded_size;
    out_layer.bit_length = ceilPow2BitLength(padded_size);
    out_layer.updateSize();
}

inline bool run_verifiers_parallel(const std::vector<VerifierJob> &jobs, int max_threads, int mode=0) {
    long peak_1, peak_2;
    long cur_1;
    peak_1 = get_vm_peak_kb();
    cur_1 = get_current_rss_kb();
    if (jobs.empty()) return true;

    std::cout << "Initial Memory utilization (kB):" << cur_1 << "  Peak: " << peak_1 << std::endl;

    auto combine_ranges_list = [&](std::initializer_list<timer_range> list) {
    return combine_ranges(std::vector<timer_range>(list));};

    std::vector<bool> results(jobs.size(), true);
    std::vector<double> standalone_commit_times(jobs.size(), 0.0);
    for (std::size_t i = 0; i < jobs.size(); ++i) standalone_commit_times[i] = jobs[i].standalone_commit_time;

    if (mode == 1) {
        double max_prover_time = 0.0;
        double max_commit_time = 0.0;
        double max_output_commit_time = 0.0;
        double max_verifier_time = 0.0;
        double combined_proof_size_kb = 0.0;
        std::cout << "Running verifiers sequentially" << std::endl;
        for (std::size_t i = 0; i < jobs.size(); ++i) {
            if (i + 1 < jobs.size()) {
                int logn = ceilPow2BitLength(static_cast<u32>(jobs[i].shared_commit_size));
                u64 n_sqrt = 1ULL << (logn - (logn >> 1));
                jobs[i].instance->p->gens_output.clear();
                jobs[i].instance->p->gens_output.resize(n_sqrt);
                G1 base = gen_gi(jobs[i].instance->p->gens_output.data(), n_sqrt);
                jobs[i].instance->p->gens_output.push_back(base);
                jobs[i].instance->p->commitOutputPrivate(jobs[i].instance->p->gens_output, max_threads);
                auto *next_p = jobs[i + 1].instance->p;
                std::vector<F> next_input_prefix(next_p->input_copy.begin(),
                                                 next_p->input_copy.begin() + next_p->inp_end_id);
                timer prefix_commit_timer;
                next_p->cc_input = commit_values_standalone(next_input_prefix, max_threads, &prefix_commit_timer);
                standalone_commit_times[i + 1] += prefix_commit_timer.elapse_sec();
                next_p->input_committed_externally = true;
            }

            const auto &job = jobs[i];
            double rp_time = job.rp->prove();
            auto rp_range = job.rp->prove_timer.range();
            job.instance->p->executeCommit(max_threads);
            job.instance->range_prove(rp_time, rp_range);
            results[i] = job.instance->prove(job.commit_threads, *job.circuit_timer);
            auto output_commit_range = job.instance->p->outputCommitTimerRange();
            auto prover_time = standalone_commit_times[i] + job.instance->prover_time + output_commit_range.duration_sec;
            delete job.rp;
            max_prover_time += prover_time;
            max_output_commit_time += output_commit_range.duration_sec;
            max_verifier_time += job.instance->verifier_time;
            combined_proof_size_kb += static_cast<double>(job.instance->p->proof_size) / 1024.0;
            release_partition_local_commitments(*job.instance->p, true);
            trim_allocator_to_os();
            if (i >= 1) {
                delete jobs[i-1].instance->p;
                delete jobs[i-1].instance;
            }
        }

        auto prove_time = max_prover_time;
        std::cout << "Results: ";
        std::cout << max_output_commit_time << "," << prove_time << "," << max_verifier_time << "," << combined_proof_size_kb << std::endl;
        peak_2 = get_vm_peak_kb();
        std::cout << "Peak Memory utilization (kB):" << peak_2 << "  " << peak_2-cur_1 << " " << peak_1 << std::endl;
        return true;
    }
    else {
    // GKR_DEMO_FORK=1: prove the partitions in forked PROCESSES instead of threads.
    // Same wall-clock semantics as the thread version (all partitions concurrent),
    // but our GPU prover keeps process-wide device state (grow-only kernel State
    // singletons, streams), which is correct for one prover per process -- what
    // main_gkr_split does -- and silently wrong for 12 prover threads sharing it.
    // The boundary commits below then run on the CPU so the parent reaches fork()
    // without a HIP context (a context does not survive fork).
    const bool demo_fork = getenv("GKR_DEMO_FORK") && atoi(getenv("GKR_DEMO_FORK")) > 0;
    const bool commit_cpu_saved = gkr_config.commit_cpu;
    if (demo_fork) gkr_config.commit_cpu = true;
    std::vector<std::thread> threads;
    threads.reserve(jobs.size());
    if (jobs.size() > 1) {
        for (std::size_t i = 0; i + 1 < jobs.size(); ++i) {
            threads.emplace_back([&, i]() {
                int logn = ceilPow2BitLength(static_cast<u32>(jobs[i].shared_commit_size));
                u64 n_sqrt = 1ULL << (logn - (logn >> 1));
                jobs[i].instance->p->gens_output.clear();
                jobs[i].instance->p->gens_output.resize(n_sqrt);
                G1 base = gen_gi(jobs[i].instance->p->gens_output.data(), n_sqrt);
                jobs[i].instance->p->gens_output.push_back(base);
                jobs[i].instance->p->commitOutputPrivate(jobs[i].instance->p->gens_output, max_threads);
                auto *next_p = jobs[i + 1].instance->p;
                std::vector<F> next_input_prefix(next_p->input_copy.begin(),
                                                 next_p->input_copy.begin() + next_p->inp_end_id);
                timer prefix_commit_timer;
                next_p->cc_input = commit_values_standalone(next_input_prefix, max_threads, &prefix_commit_timer);
                standalone_commit_times[i + 1] += prefix_commit_timer.elapse_sec();
                next_p->input_committed_externally = true;
            });
        }
    }
    for (auto &t : threads) t.join();
    threads.clear();
    if (demo_fork) gkr_config.commit_cpu = commit_cpu_saved;
    std::cout << "Parallel verification start" << std::endl;
    // Per-partition results. Threads fill the job objects in place; forked children
    // ship this struct back through a pipe and the parent fills the same fields.
    struct DemoPartResult { int idx; int ok; double prover_time, out_commit, verifier_time; int out_ready; long long proof_size; };
    std::vector<DemoPartResult> fork_res(jobs.size());
    if (demo_fork) {
        int pfd[2], tok[2];
        if (pipe(pfd) != 0 || pipe(tok) != 0) { std::perror("[demo-fork] pipe"); return false; }
        // GKR_DEMO_FORK_MAX=n: at most n partitions prove at once (one MI210 does not
        // hold 12 provers of a 2^27-domain partition). Token pipe: parent pre-fills n
        // slots, every child takes one before it starts, the parent refills one per
        // reaped child. Default: no cap.
        int cap = getenv("GKR_DEMO_FORK_MAX") ? atoi(getenv("GKR_DEMO_FORK_MAX")) : 0;
        if (cap <= 0 || cap > (int)jobs.size()) cap = (int)jobs.size();
        for (int k = 0; k < cap; ++k) if (write(tok[1], "t", 1) != 1) { std::perror("[demo-fork] token"); return false; }
        std::cout << "[demo-fork] partitions=" << jobs.size() << " concurrency cap=" << cap << std::endl;
        std::vector<pid_t> pids(jobs.size(), -1);
        std::cout.flush(); fflush(stdout);
        for (std::size_t i = 0; i < jobs.size(); ++i) {
            pid_t pid = fork();
            if (pid < 0) { std::perror("[demo-fork] fork"); return false; }
            if (pid == 0) {
                close(pfd[0]); close(tok[1]);
                char t; if (read(tok[0], &t, 1) != 1) _exit(4);   // wait for a slot
                close(tok[0]);
                double rp_time = jobs[i].rp->prove();
                auto rp_range = jobs[i].rp->prove_timer.range();
                jobs[i].instance->p->executeCommit(max_threads);
                jobs[i].instance->range_prove(rp_time, rp_range);
                bool ok = jobs[i].instance->prove(jobs[i].commit_threads, *jobs[i].circuit_timer);
                auto ocr = jobs[i].instance->p->outputCommitTimerRange();
                DemoPartResult r{(int)i, ok ? 1 : 0, jobs[i].instance->prover_time, ocr.duration_sec,
                                 jobs[i].instance->verifier_time, ocr.ready ? 1 : 0,
                                 (long long)jobs[i].instance->p->proof_size};
                std::cout.flush(); fflush(stdout);
                if (write(pfd[1], &r, sizeof r) != (ssize_t)sizeof r) _exit(2);
                _exit(ok ? 0 : 1);
            }
            pids[i] = pid;
        }
        close(pfd[1]);   // tok[0] stays open in the parent: a refill after the last child
                         // exited would otherwise SIGPIPE us before Results: is printed
        std::vector<char> got(jobs.size(), 0);
        std::vector<int> stat(jobs.size(), 0);
        // Reap in completion order, refilling one slot per exit so the next waiting
        // child can start; results are drained afterwards (12 x 48 B fits the pipe).
        for (std::size_t done = 0; done < jobs.size(); ++done) {
            int st = 0; pid_t w = waitpid(-1, &st, 0);
            if (w < 0) { std::perror("[demo-fork] waitpid"); break; }
            for (std::size_t i = 0; i < jobs.size(); ++i) if (pids[i] == w) stat[i] = st;
            if (write(tok[1], "t", 1) != 1) std::perror("[demo-fork] token refill");
        }
        close(tok[1]); close(tok[0]);
        DemoPartResult r;
        while (read(pfd[0], &r, sizeof r) == (ssize_t)sizeof r) { fork_res[r.idx] = r; got[r.idx] = 1; }
        close(pfd[0]);
        for (std::size_t i = 0; i < jobs.size(); ++i) {
            const int st = stat[i];
            results[i] = got[i] && fork_res[i].ok && WIFEXITED(st) && WEXITSTATUS(st) == 0;
            if (!got[i]) std::cerr << "[demo-fork] partition " << i << " returned no result (status " << st << ")\n";
        }
    } else {
    for (std::size_t i = 0; i < jobs.size(); ++i) {
        threads.emplace_back([&, i]() {
            double rp_time = jobs[i].rp->prove();
            auto rp_range = jobs[i].rp->prove_timer.range();
            jobs[i].instance->p->executeCommit(max_threads);
            jobs[i].instance->range_prove(rp_time, rp_range);
            results[i] = jobs[i].instance->prove(jobs[i].commit_threads, *jobs[i].circuit_timer);
        });
    }

    for (auto &t : threads) t.join();
    }
        bool all_passed = std::all_of(results.begin(), results.end(), [](bool r) { return r; });
        if (!all_passed) {
            std::cerr << "Verification failed for";
            for (std::size_t i = 0; i < jobs.size(); ++i) {
                if (!results[i]) {
                    if (jobs[i].label && jobs[i].label[0]) std::cerr << ' ' << jobs[i].label;
                    else std::cerr << " verifier[" << i << ']';
                }
            }
            std::cerr << std::endl;
            return false;
        }



        double max_prover_time = 0.0;
        double max_commit_time = 0.0;
        double max_output_commit_time = 0.0;
        double max_verifier_time = 0.0;
        double combined_proof_size_kb = 0.0;
        for (std::size_t i = 0; i < jobs.size(); ++i) {
            const auto &job = jobs[i];
            auto output_commit_range = job.instance->p->outputCommitTimerRange();
            double pt = job.instance->prover_time, oc = output_commit_range.duration_sec,
                   vt = job.instance->verifier_time, ps = (double)job.instance->p->proof_size;
            bool oc_ready = output_commit_range.ready;
            if (demo_fork) { pt = fork_res[i].prover_time; oc = fork_res[i].out_commit; vt = fork_res[i].verifier_time;
                             ps = (double)fork_res[i].proof_size; oc_ready = fork_res[i].out_ready; }
            auto prover_time = standalone_commit_times[i] + pt + oc;
            max_prover_time = std::max(max_prover_time, prover_time);
            if (oc_ready)
                max_output_commit_time = std::max(max_output_commit_time, oc);
            max_verifier_time = std::max(max_verifier_time, vt);
            combined_proof_size_kb += ps / 1024.0;
        }

        auto prove_time = max_prover_time;
        std::cout << "Results: ";
        std::cout << max_output_commit_time << "," << prove_time << "," << max_verifier_time << "," << combined_proof_size_kb << std::endl;
        peak_2 = get_vm_peak_kb();
        std::cout << "Peak Memory utilization (kB):" << peak_2 << "  " << peak_2-cur_1 << " " << peak_1 << std::endl;
        return true;
    }
}
