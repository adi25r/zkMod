#!/bin/bash
# Threads-per-partition sweep: fixed 192-thread budget, 1 GPU,
# GKR_TPC = 1..32 for every arm. Arms 2-4: pinned + GPU_MAX_HW_QUEUES=1 + GKR_GPU_PROCS=12 (the
# lassosparse settings). Arm 1: unpinned and no GPU cap, as in the cpuhostthreadscaling gated sweep
# (GKR_GPU_PROCS counts every child, so it would throttle the CPU arm too).
# Logs: tools/results/speedup_analysis_models/paper_arith/logs_tpc_sweep_2026-10-24/tpc<N>/
set -u
cd "$(dirname "$(readlink -f "$0")")/../../../.."
R=tools/results/speedup_analysis_models/paper_arith
BIN=${BIN:-./bin/main_gkr_split_pool_2026-10-24}   # 192-thread pool enforced (child omp/max_threads = tpc, gate caps alive*tpc)
for t in ${TPCS:-1 2 4 8 16 32}; do
  L=$R/logs_tpc_sweep_pool_2026-10-24/tpc$t; mkdir -p $L
  TPC=$t LOGS=$L BIN=$BIN GPU_MAX_HW_QUEUES=1 GKR_GPU_PROCS=12 GKR_PIN_CORES=1 ARMS="2 3 4" \
    $R/_queue/run_paper_arith.sh ${MODELS:-gpt2 opt350m} 2>&1 | grep -E "rc=" | sed "s/^/tpc=$t /"
  ( unset GKR_PIN_CORES GKR_GPU_PROCS GPU_MAX_HW_QUEUES
    TPC=$t LOGS=$L BIN=$BIN ARMS="1" $R/_queue/run_paper_arith.sh ${MODELS:-gpt2 opt350m} 2>&1 | grep -E "rc=" | sed "s/^/tpc=$t /" )
done
echo TPC_SWEEP_DONE
