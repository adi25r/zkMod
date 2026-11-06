#!/bin/bash
# Fig 5c, memory-gated: zkMod CPU witgen + CPU prove vs TOTAL host-thread budget T,
# seq len 64. Unlike run_fig5_threads.sh (fixed W=4 concurrent provers), this runs
# --overlapped -1: as many partitions prove at once as the strict memory gate admits, and T is
# split evenly across all partitions (tpc = max(1, T / splits), no GKR_TPC). Gate seeds = the
# CPU-arm seeds of run_paper_arith.sh.
# Children are NOT pinned (see the unset below).
# Holds $QUIET while a point runs so other experiments on the box can wait for it.
set -u
cd "$(dirname "$(readlink -f "$0")")/../../../.."
LOGS=tools/results/figures/logs
QUIET=${QUIET:-${SLURM_TMPDIR:-/tmp}/zkml_QUIET.lock}
export LD_LIBRARY_PATH="/opt/rocm-scale/lib:/opt/rocm/lib:/opt/rocm/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset GKR_PIN_CORES   # NOT pinned: on the CPU arm pinning slows the witgen chain (gpt2 T=192: 118.2 s pinned vs 101.2 s unpinned)
export ZKGPT_WITNESS_CACHE=0 GKR_LASSO_CPU=1 GKR_MATMUL_CPU=1 GKR_PRED_CPU=1 GKR_SC_CPU=1 GKR_EARLY_CARRY=1
unset OMP_NUM_THREADS GKR_BFEVAL_GPU GKR_SC_STREAM_BITS GKR_TPC HIP_VISIBLE_DEVICES GKR_WITGEN_GPU GKR_ROUND_GPU
BIN=${BIN:-./bin/main_gkr_split_bfmt_2026-10-23}
#                 L   attn lin  splits min_free peak_cpu built
declare -A SPEC=( [gpt2]="12 768 3072 12 32768 28000 7000" [opt350m]="24 1024 4096 24 65536 30000 7000" )
MODELS=${MODELS:-"gpt2 opt350m"}
THREADS=${THREADS:-"192 128 64 32 16"}
note(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }
trap 'rm -f "$QUIET"' EXIT
for m in $MODELS; do
  read -r L attn lin splits minfree peak built <<< "${SPEC[$m]}"
  export GKR_MIN_FREE_MIB=$minfree GKR_PART_PEAK_MIB=$peak GKR_PART_BUILT_MIB=$built
  for t in $THREADS; do
    log="$LOGS/${m}_t${t}_arm1_cpuwg_cpuprove.log"
    if grep -q "status=OK" "$log" 2>/dev/null; then note "$m t=$t already done -- skip"; continue; fi
    touch "$QUIET"   # others stop launching; wait for their in-flight runs to drain
    while [[ -z ${SLURM_JOB_ID:-} ]] && pgrep '^main_gkr_split' >/dev/null; do note "waiting for other main_gkr_split runs to finish"; sleep 30; done
    note "=== $m T=$t gated (tpc=$(( t / splits > 0 ? t / splits : 1 ))) cpu+cpu ==="
    ( $BIN "$L" "$attn" "$splits" 64 "$t" "$lin" --overlapped -1 --c 9 --cpu --cpu-deg3 --cpu-open --cpu-commit ) > "$log" 2>&1
    note "$m t=$t rc=$? ok=$(grep -c 'status=OK' "$log") pass=$(grep -c 'All verification passed' "$log")/$splits $(grep -o 'overlapped_total_s=[0-9.]*' "$log")"
    rm -f "$QUIET"; sleep 60   # window for other experiments to start between points
  done
done
note FIG5C_GATED_DONE
