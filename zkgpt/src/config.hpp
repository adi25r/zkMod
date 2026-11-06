#pragma once

struct GkrConfig {
    // Range prover parameters
    int range_c = 9;
    int range_k = 1;
    int range_size = 54;

    // Run-mode flags
    bool only_range = false;  // only-range : skip GKR, run only range prover
    bool skip_range = false;  // skip-range : skip range prover

    // CPU/GPU path selectors
    bool range_cpu = false;   // cpu : deg1 sumcheck on CPU
    bool deg3_cpu = false;    // cpu-deg3 : deg3 sumcheck on CPU
    bool open_cpu = false;    // cpu-open : IPA open on CPU
    bool commit_cpu = false;  // cpu-commit : Hyrax commit (MSM) on CPU

    // UVM
    bool uvm_prefetch = false;  // uvm-prefetch: hipMemAdvise+prefetch (UVM binary only)
    bool uvm_zerocopy = false;  // uvm-zerocopy: keep oversubscribing buffers host
    // resident + device-accessed (stream over PCIe) instead of on-demand page migration

    // Debug
    bool dump_prefetch = false;   // dump-prefetch : trace matmul prefetch hits/misses
    bool dump_callgraph = false;  // dump-callgraph : print GKR call graph
    bool dump_sizes = false;      // dump-sizes : print layer sizes during traversal
    bool gpu_bookkeep = false;    // gpu-bookkeep : GPU bookkeeping path in verifier

    // Soundness / fault injection (testing only)
    int corrupt_layer = -1;       // corrupt-layer <i>
    bool corrupt_commit = false;  // corrupt-commit

    // Microbench modes (exit after bench)
    bool bench_deg1 = false;    // bench-deg1 <n> <k>
    bool bench_deg3 = false;    // bench-deg3 <n>
    bool bench_matmul = false;  // bench-matmul <n> <k>
    int bench_n = -1;
    int bench_k = -1;

    // Range prover partitioning
    int range_partitions = 1;  // partitions N
    float vram_gb = 64.0f;     // vram-gb N (MI210 default)

    // Internal stage-count overrides
    int matmul_l = -1;  // matmul-l <l>
    int matmul_k = -1;
    // calls
    int lasso_l = -1;  // lasso-l <l> : override l_stage directly in the lasso sumcheck
    int lasso_k = -1;  // lasso-k <k> : lasso Blendy stage COUNT
                       // recomputed from the sumcheck's own n

    // Blendy ext-table tiling
    int sc_tile_bits = -1;  // sc-tile-bits <b> : cap the materialized future-variable (b3) window
                            // of the staged product_sumcheck ext tables at 2^b combos

    // GPU fully-connected witness generation
    bool witgen_gpu = false;    // env GKR_WITGEN_GPU sets the initial value
    bool witgen_check = false;  // env GKR_WITGEN_CHECK: run both paths, assert equal

    // GPU roundLayer
    bool round_gpu = false;    // env GKR_ROUND_GPU
    bool round_check = false;  // env GKR_ROUND_CHECK: run both paths, assert equal

    // GPU calcNormalLayer
    bool gates_gpu = false;    // env GKR_GATES_GPU
    bool gates_check = false;  // env GKR_GATES_CHECK: run both paths, assert equal

    // GPU builders
    bool witgen_full = false;        // env GKR_WITGEN_FULL / GKR_GELU_GPU
    bool witgen_full_check = false;  // env GKR_WITGEN_FULL_CHECK / GKR_GELU_CHECK
};

extern GkrConfig gkr_config;
