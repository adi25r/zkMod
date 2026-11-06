#pragma once

struct GkrConfig {
    // Range prover parameters (set from CLI positional args 6/7/8)
    // c MUST divide range_size: the chunked lookup decomposes a range_size-bit value
    // into range_size/c chunks of c bits, and the CPU baseline enforces
    // `range % chunk_bits == 0` (range_prover.cpp). Legal c for 54: 1,2,3,6,9,18,27,54.
    // range_size 54 matches the CPU arm's statement; smaller values prove a weaker one.
    int  range_c    = 9;
    int  range_k    = 1;
    int  range_size = 54;

    // Run-mode flags
    bool only_range    = false;   // --only-range  : skip GKR, run only range prover
    bool skip_range    = false;   // --skip-range  : skip range prover

    // CPU/GPU path selectors
    bool range_cpu     = false;   // --cpu         : deg1 sumcheck on CPU
    bool deg3_cpu      = false;   // --cpu-deg3    : deg3 sumcheck on CPU
    bool open_cpu      = false;   // --cpu-open    : IPA open on CPU
    bool commit_cpu    = false;   // --cpu-commit  : Hyrax commit (MSM) on CPU

    // UVM
    bool uvm_prefetch  = false;   // --uvm-prefetch: hipMemAdvise+prefetch (UVM binary only)
    bool uvm_zerocopy  = false;   // --uvm-zerocopy: keep oversubscribing buffers host-
                                  // resident + device-accessed (stream over PCIe) instead
                                  // of on-demand page migration. UVM binary only.

    // Debug
    bool dump_prefetch  = false;  // --dump-prefetch  : trace matmul prefetch hits/misses
    bool dump_callgraph = false;  // --dump-callgraph : print GKR call graph
    bool dump_sizes     = false;  // --dump-sizes     : print layer sizes during traversal
    bool gpu_bookkeep   = false;  // --gpu-bookkeep   : GPU bookkeeping path in verifier

    // Soundness / fault injection (testing only)
    int  corrupt_layer  = -1;    // --corrupt-layer <i>
    bool corrupt_commit = false;  // --corrupt-commit

    // Microbench modes (exit after bench)
    bool bench_deg1    = false;   // --bench-deg1 <n> <k>
    bool bench_deg3    = false;   // --bench-deg3 <n>
    bool bench_matmul  = false;   // --bench-matmul <n> <k>
    int  bench_n       = -1;
    int  bench_k       = -1;

    // Range prover partitioning
    // P partitions = ceil(LayerNum/P) layers per wave, proven sequentially.
    // 0 = auto-compute from vram_gb using the Hyrax commit memory model.
    // 1, NOT auto. Auto-splitting LayerNum into P partitions registers P constraints of
    // next_pow2(per_layer * l_p) instead of one of next_pow2(per_layer * LayerNum), and
    // sum(next_pow2) != next_pow2(sum) -- so a VRAM-derived P silently changes the total
    // padded slot count and breaks parity with the CPU arm's single constraint.
    int   range_partitions = 1;   // --partitions N
    float vram_gb          = 64.0f; // --vram-gb N  (MI210 default)

    // Internal stage-count overrides
    int  matmul_l      = -1;     // --matmul-l <l> : override l_stage directly (rarely what you want:
                                  //                  l_stage is per-call, so a fixed l implies a
                                  //                  different k = ceil(n/l) at every matmul call site
                                  //                  since n varies by op). Ignored if matmul_k is set.
    int  matmul_k      = -1;     // --matmul-k <k> : hold k (Blendy stage COUNT) fixed across all matmul
                                  //                  calls; l_stage = ceil(n/k) is recomputed per call
                                  //                  site from that call's own n. Takes priority over matmul_l.
    int  lasso_l       = -1;     // --lasso-l <l>  : override l_stage directly in the lasso sumcheck
    int  lasso_k       = -1;     // --lasso-k <k>  : lasso Blendy stage COUNT; l_stage = ceil(n/k) is
                                  //                  recomputed from the sumcheck's own n. Priority over lasso_l.

    // Blendy ext-table tiling.
    int  sc_tile_bits  = -1;     // --sc-tile-bits <b> : cap the materialized future-variable (b3) window
                                  //                  of the staged product_sumcheck ext tables at 2^b combos,
                                  //                  building the 3^l ptable in tiles. -1 = disabled (materialize
                                  //                  the full 2^(n-l) window = current behavior). Makes the ext
                                  //                  footprint 2*3^l*2^b instead of 2*3^l*2^(n-l), so k still
                                  //                  scales memory as 3^l (space-time knob preserved) but with a
                                  //                  base you choose to fit VRAM. Env GKR_SC_TILE_BITS overrides.

    // GPU fully-connected witness generation. Runtime-settable, NOT a first-call
    // static read of GKR_WITGEN_GPU: --witgen-chain must keep it OFF in the parent
    // (whose probe pass calls fullyConnLayer) so the parent reaches fork() with no
    // HIP context, then turn it ON inside each child. See main_gkr_split.cpp.
    bool witgen_gpu    = false;  // env GKR_WITGEN_GPU sets the initial value
    bool witgen_check  = false;  // env GKR_WITGEN_CHECK: run both arms, assert equal

    // GPU roundLayer, fused onto a resident fullyConnLayer output (no D2H/H2D at
    // the fc->round boundary). Independent opt-in from witgen_gpu: the GPU path
    // only activates when BOTH are set AND fullyConnLayer's output for that call
    // is actually resident (see neuralNetwork.cpp); otherwise roundLayer falls
    // back to its CPU body unchanged.
    bool round_gpu     = false;  // env GKR_ROUND_GPU
    bool round_check   = false;  // env GKR_ROUND_CHECK: run both arms, assert equal

    // GPU calcNormalLayer: the gate-driven scatter-accumulate that computes EVERY
    // layer's output (neuralNetwork.cpp). Independent of witgen_gpu/round_gpu --
    // it needs no other kernel, only val[0] and val[layer_id-1] staged to the
    // device. Same runtime-settable discipline as the two above: --witgen-chain
    // must keep it OFF in the parent so the parent reaches fork() HIP-free.
    bool gates_gpu     = false;  // env GKR_GATES_GPU
    bool gates_check   = false;  // env GKR_GATES_CHECK: run both arms, assert equal

    // GPU builders: every layer builder's advice values AND gate emission run in
    // kernels, writing straight into the val[0] device mirror. One flag rather
    // than one per builder -- they share the mirror, so half-enabling them just
    // costs readbacks. GKR_GELU_GPU/GKR_GELU_CHECK remain as aliases because the
    // tooling already sets them.
    bool witgen_full   = false;  // env GKR_WITGEN_FULL / GKR_GELU_GPU
    bool witgen_full_check = false;  // env GKR_WITGEN_FULL_CHECK / GKR_GELU_CHECK
};

extern GkrConfig gkr_config;
