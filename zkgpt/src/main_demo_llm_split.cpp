//
// A demo that commits to the input separately (image/predicate style) and runs the prover.
//

#undef NDEBUG
#include "circuit.h"
#include "neuralNetwork.hpp"
#include "verifier.hpp"
#include "models.hpp"
#include "global_var.hpp"
#include <iostream>
#include <pthread.h>
#include "timer.hpp"
#include "range_prover.hpp"
#include "config.hpp"
#include "hyrax_rp.hpp"
#include <signal.h>
#include <execinfo.h>
#include "main_demo_utils.hpp"
#include "utils.hpp"
#include <string>
#include <memory>

using namespace mcl::bn;
using namespace std;


namespace {

bool run_split_sequential_lazy(const std::vector<int> &split_depths,
                               const std::vector<long long> &total_input_witness_sizes,
                               const std::vector<u64> &shared_commit_sizes,
                               int embed_size,
                               int lin_dim,
                               int len,
                               int num_threads,
                               int merge_flag) {
    long peak_1 = get_vm_peak_kb();
    long cur_1 = get_current_rss_kb();
    std::cout << "Initial Memory utilization (kB):" << cur_1 << "  Peak: " << peak_1 << std::endl;
    std::cout << "Running verifiers sequentially" << std::endl;

    SplitBlockCarryState carry;
    bool has_carry = false;
    OwnedCommitment previous_commit;
    double total_prover_time = 0.0;
    double total_output_commit_time = 0.0;
    double total_verifier_time = 0.0;
    double combined_proof_size_kb = 0.0;

    for (std::size_t i = 0; i < split_depths.size(); ++i) {
        const bool is_first = (i == 0);
        const bool is_last = (i + 1 == split_depths.size());
        const bool output_private = !is_last;
        const bool input_private = !is_first;
        const bool input_separate = is_first;

        std::unique_ptr<prover> p(new prover());
        p->max_threads = num_threads;
        std::unique_ptr<LLMSplitBlock> network(new LLMSplitBlock(split_depths[i], embed_size / 64, 64,
                                                                 embed_size, lin_dim, len,
                                                                 output_private, input_private, input_separate,
                                                                 nullptr, nullptr));
        if (has_carry) {
            network->initStateFromCarry(carry);
        }
        network->setTotalInputWitnessSize(total_input_witness_sizes[i]);
        if (!is_first) {
            p->include_input_commit_size = false;
        }

        timer circuit_timer;
        network->create(*p, merge_flag, circuit_timer);
        if (!is_last) {
            pad_output_layer(*p, shared_commit_sizes[i]);
        }

        std::unique_ptr<verifier> v(new verifier(p.get(), p->C));
        v->max_threads = num_threads;
        if (!is_first && !previous_commit.empty()) {
            p->cc_input = previous_commit.view();
            p->input_committed_externally = true;
        }
        if (!is_last) {
            commit_private_output_for_next(*p, shared_commit_sizes[i], num_threads);
        }

        std::unique_ptr<range_prover> rp(new range_prover(split_depths[i], embed_size / 64, 64,
                                                          embed_size, lin_dim, len,
                                                          num_threads, merge_flag));
        double rp_time = rp->prove();
        auto rp_range = rp->prove_timer.range();
        p->executeCommit(num_threads);
        v->range_prove(rp_time, rp_range);
        if (!v->prove(num_threads, circuit_timer)) {
            return false;
        }

        auto output_commit_range = p->outputCommitTimerRange();
        total_prover_time += v->prover_time + output_commit_range.duration_sec;
        total_output_commit_time += output_commit_range.duration_sec;
        total_verifier_time += v->verifier_time;
        combined_proof_size_kb += static_cast<double>(p->proof_size) / 1024.0;

        if (!is_last) {
            previous_commit.clear();
            previous_commit.capture(p->cc_output);
            carry = network->exportSplitBlockCarry(*p);
            has_carry = true;
        } else {
            previous_commit.clear();
        }

        v.reset();
        rp.reset();
        release_partition_local_commitments(*p);
        network.reset();
        p.reset();
        trim_allocator_to_os();
    }

    auto prove_time = total_prover_time + total_output_commit_time;
    std::cout << "Results: ";
    std::cout << total_output_commit_time << "," << prove_time << ","
              << total_verifier_time << "," << combined_proof_size_kb << std::endl;
    long peak_2 = get_vm_peak_kb();
    std::cout << "Peak Memory utilization (kB):" << peak_2 << "  " << peak_2 - cur_1 << " " << peak_1 << std::endl;
    return true;
}

}  // namespace

int main(int argc, char **argv) {
    fprintf(stderr, "[demo] entering main\n");
#ifdef __GLIBC__
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, (long long int) 16 * 1024 * 1024);
    if (pthread_setattr_default_np(&attr) != 0) {
        std::cerr << "pthread_setattr_default_np failed\n";
    }
    pthread_attr_destroy(&attr);
#endif

    initPairing(mcl::BN254);

    
    int depth = 12;
    int embed_size = 768;
    int len = 30;
    int num_splits = 2;
    int mode=0;
    if (argc > 2) { 
        depth = atoi(argv[2]);
        embed_size = atoi(argv[3]);
    }
    if (argc > 4) {
        num_splits = atoi(argv[4]);
    }
    if (argc > 5) {
        len = atoi(argv[5]);
    }
    if (argc > 6) {
        mode = atoi(argv[6]);
    } 
    // argv[7]/argv[8] as in modular-zkp/zkGPT's main_demo_llm_split.cpp: FFN width
    // (default embed*3 = the shipped 2304; GPT-2 proper is 4*768 = 3072) and the
    // layer-merge flag, forwarded to BOTH create() and the range prover so circuit
    // and range statement agree. Same arithmetization as the paper repo by
    // construction; the GPU/CPU choice is made by the usual env gates.
    int lin_dim = 0;
    int merge_flag = 1;
    if (argc > 7) lin_dim = atoi(argv[7]);
    if (argc > 8) merge_flag = atoi(argv[8]);
    if (lin_dim <= 0) lin_dim = embed_size * 3;
    cout << "lin_dim: " << lin_dim << endl;
    cout << "merge: " << merge_flag << endl;
    // Prover knobs by env (this driver keeps modular-zkp's positional CLI). Same
    // fields main_gkr_split sets from --c/--matmul-k/--range-k/--lasso-k/
    // --sc-tile-bits and --cpu/--cpu-deg3/--cpu-open/--cpu-commit.
    auto envi = [](const char *k, int &dst) { if (const char *e = getenv(k)) dst = atoi(e); };
    envi("GKR_C",            gkr_config.range_c);
    envi("GKR_MATMUL_K",     gkr_config.matmul_k);
    envi("GKR_RANGE_K",      gkr_config.range_k);
    envi("GKR_LASSO_K",      gkr_config.lasso_k);
    envi("GKR_SC_TILE_BITS", gkr_config.sc_tile_bits);
    if (getenv("GKR_RANGE_CPU_ALL")) {   // the CPU arm: every range-prover stage on host
        gkr_config.range_cpu = gkr_config.deg3_cpu = true;
        gkr_config.open_cpu  = gkr_config.commit_cpu = true;
    }
    cout << "knobs: c=" << gkr_config.range_c << " matmul_k=" << gkr_config.matmul_k
         << " range_k=" << gkr_config.range_k << " lasso_k=" << gkr_config.lasso_k
         << " sc_tile_bits=" << gkr_config.sc_tile_bits
         << " range_cpu=" << gkr_config.range_cpu << endl;
    cout << ((mode==0) ? "Parallel run" : "Sequential run") << endl;
    cout << "len: " << len << endl;
    if (num_splits < 1) num_splits = 1;
    if (depth <= 0) {
        std::cerr << "Invalid depth provided." << std::endl;
        return EXIT_FAILURE;
    }
    num_splits = std::max(1, std::min(depth, std::max(2, num_splits)));
    int num_threads = (int) (atoi(argv[1])/num_splits);

    std::vector<int> split_depths(num_splits, depth / num_splits);
    for (int i = 0; i < depth % num_splits; ++i) ++split_depths[i];
    std::cerr << "[demo] depth=" << depth << " splits=" << num_splits << " split depths:" << " threads per split=" << num_threads;
    for (auto d : split_depths) std::cerr << ' ' << d;
    std::cerr << std::endl;

    std::vector<long long> total_input_witness_sizes(num_splits, 0);
    std::vector<u64> shared_commit_sizes(num_splits > 1 ? num_splits - 1 : 0);
    std::vector<prover *> probe_provers(num_splits, nullptr);
    std::vector<LLMSplitBlock *> probe_networks(num_splits, nullptr);

    for (int i = 0; i < num_splits; ++i) {
        probe_provers[i] = new prover();
        probe_provers[i]->max_threads = num_threads;
        prover *prev_prover = (i == 0) ? nullptr : probe_provers[i - 1];
        neuralNetwork *prev_block = (i == 0) ? nullptr : probe_networks[i - 1];
        probe_networks[i] = new LLMSplitBlock(split_depths[i], embed_size/64, 64, embed_size, lin_dim, len,
                                              false, false, false, prev_prover, prev_block);
        timer probe_timer;
        probe_networks[i]->create(*probe_provers[i], merge_flag, probe_timer);
        total_input_witness_sizes[i] = probe_networks[i]->getTotalInputSize();
        std::cout << "Total Input Witness Size block" << (i + 1) << " (probe): " << total_input_witness_sizes[i] << std::endl;
        if (i > 0) {
            auto shared_commit_size = next_pow2_size(static_cast<u64>(total_input_witness_sizes[i] >> 1));
            shared_commit_sizes[i - 1] = shared_commit_size;
            total_input_witness_sizes[i] = static_cast<long long>(shared_commit_size << 1);
            std::cout << "Shared commitment size between blocks " << i << " and " << (i + 1) << ": " << shared_commit_size << std::endl;
        }
    }
    std::cerr << "[demo] probe sizing done" << std::endl;

    for (auto *nn : probe_networks) delete nn;
    for (auto *p : probe_provers) delete p;

    if (mode == 1) {
        bool ok = run_split_sequential_lazy(split_depths, total_input_witness_sizes,
                                            shared_commit_sizes, embed_size, lin_dim, len, num_threads, merge_flag);
        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    std::vector<range_prover *> rps(num_splits, nullptr);
    for (int i = 0; i < num_splits; ++i) {
        rps[i] = new range_prover(split_depths[i], embed_size/64, 64, embed_size, lin_dim, len, num_threads, merge_flag);
    }
    std::cerr << "[demo] range provers ready" << std::endl;

    std::vector<prover *> provers(num_splits, nullptr);
    std::vector<LLMSplitBlock *> networks(num_splits, nullptr);
    std::vector<verifier *> verifiers(num_splits, nullptr);
    std::vector<timer> circuit_timers(num_splits);

    for (int i = 0; i < num_splits; ++i) {
        provers[i] = new prover();
        provers[i]->max_threads = num_threads;
        bool is_first = (i == 0);
        bool is_last = (i == num_splits - 1);
        // Privacy: first block hides output, middle hide both, last hides input.
        bool output_private = !is_last;
        bool input_private = !is_first;
        bool input_separate = is_first;
        prover *prev_prover = is_first ? nullptr : provers[i - 1];
        neuralNetwork *prev_block = is_first ? nullptr : networks[i - 1];
        networks[i] = new LLMSplitBlock(split_depths[i], embed_size/64, 64, embed_size, lin_dim, len,
                                        output_private, input_private, input_separate, prev_prover, prev_block);
        networks[i]->setTotalInputWitnessSize(total_input_witness_sizes[i]);
        if (!is_first) {
            provers[i]->include_input_commit_size = false;
        }
        networks[i]->create(*provers[i], 1, circuit_timers[i]);
        if (!is_last) {
            pad_output_layer(*provers[i], shared_commit_sizes[i]);
        }
    }

    std::cout << "Memory Usage: " << get_current_rss_kb() << std::endl;

    for (int i = 0; i < num_splits; ++i) {
        verifiers[i] = new verifier(provers[i], provers[i]->C);
        verifiers[i]->max_threads = num_threads;
    }



    std::vector<std::string> labels;
    labels.reserve(num_splits);
    std::vector<VerifierJob> verifier_jobs;
    verifier_jobs.reserve(num_splits);
    for (int i = 0; i < num_splits; ++i) {
        labels.emplace_back("block" + std::to_string(i + 1));
        u64 shared_commit_size = (i < static_cast<int>(shared_commit_sizes.size())) ? shared_commit_sizes[i] : 0;
        verifier_jobs.push_back({verifiers[i], provers[i], rps[i], &circuit_timers[i], shared_commit_size, num_threads, labels.back().c_str(), 0.0});
    }

    if (!run_verifiers_parallel(verifier_jobs, num_threads, mode)) {
        return EXIT_FAILURE;
    }
    std::cerr << "[demo] all verifiers done" << std::endl;

    for (auto *nn : networks) delete nn;
    for (auto *p : provers) {
        if (p) release_partition_local_commitments(*p, true);
        delete p;
    }
    for (auto *v : verifiers) delete v;
    for (auto *rp : rps) delete rp;
    trim_allocator_to_os();
}
