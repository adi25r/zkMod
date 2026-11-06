#!/bin/bash
# Figure 3, baseline only: latency breakdown (witgen vs prove) of the splits=1
# monolith vs host thread count, plus a GPU-witgen arm (GPU used ONLY for witness
# generation; proving stays on the CPU at 192 threads). x = model, series = threads.
#
# Runs strictly one at a time: the paper-layout monolith needs ~171 GiB (gpt2) /
# ~356 GiB (opt350m) host, and a thread-scaling curve is meaningless if two runs
# share the cores. opt1b3 is NOT attempted: its monolith needs a 2^33 Lasso domain
# (~512 GiB of tables) on top of a 2^32-element witness (128 GiB) -- more than the
# box has.
#
# Idempotent: a log with "All verification passed" is skipped, so a killed queue
# can be relaunched. Timing per arm = PHASE witgen / PHASE prove wall (dur_s).
set -u
cd "$(dirname "$(readlink -f "$0")")/../../../.."
LOGS=tools/results/figures/logs; mkdir -p "$LOGS"
export LD_LIBRARY_PATH="/opt/rocm-scale/lib:/opt/rocm/lib:/opt/rocm/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ZKGPT_WITNESS_CACHE=0 GKR_LASSO_CPU=1 GKR_MATMUL_CPU=1 GKR_PRED_CPU=1 GKR_SC_CPU=1
unset OMP_NUM_THREADS GKR_BFEVAL_GPU GKR_SC_STREAM_BITS
declare -A SPEC=( [gpt2]="12 768 3072" [opt350m]="24 1024 4096" )
MODELS=${MODELS:-"gpt2 opt350m"}
THREADS=${THREADS:-"192 128 64 32 16"}
BIN=${BIN:-./bin/main_gkr_split_witgen}
note(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }
summ(){ grep -E '^PHASE .*ev=end .*phase=(witgen|prove) ' "$1" | grep -oE 'phase=\w+|dur_s=[0-9.]+' | paste -sd' '; }
done_log(){ grep -q "All verification passed" "$1" 2>/dev/null; }

# Wait for the box to be quiet enough: MemAvailable >= need GiB and no other
# main_gkr_split of ours running (shared box -- other users' load is out of our hands).
wait_mem(){ local need=$1
  while :; do
    local avail; avail=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
    if (( avail >= need )) && ! pgrep -f 'bin/main_gkr_split' >/dev/null; then return; fi
    note "waiting: MemAvailable=${avail}GiB need=${need}GiB"; sleep 60
  done; }
declare -A NEED=( [gpt2]=200 [opt350m]=400 )

for m in $MODELS; do
  read -r L attn lin <<< "${SPEC[$m]}"
  # CPU witgen + CPU prove at each thread count
  for t in $THREADS; do
    log="$LOGS/${m}_t${t}_cpu.log"
    if done_log "$log"; then note "$m t=$t cpu already done -- skip"; continue; fi
    wait_mem "${NEED[$m]}"
    note "=== $m threads=$t cpu witgen ==="
    ( unset HIP_VISIBLE_DEVICES GKR_WITGEN_GPU GKR_ROUND_GPU GKR_GATES_GPU GKR_WITGEN_FULL
      $BIN "$L" "$attn" 1 64 "$t" "$lin" --c 9 --cpu --cpu-deg3 --cpu-open --cpu-commit ) > "$log" 2>&1
    note "$m t=$t cpu rc=$? pass=$(grep -c 'All verification passed' "$log") $(summ "$log")"
  done
  # GPU witgen (entire witgen on the device: FC, round, gate evaluator, all builders),
  # CPU prove at 192 threads. No fallback: if the full port does not fit, the row fails.
  log="$LOGS/${m}_gpuwg_full.log"
  if done_log "$log"; then note "$m gpu witgen already done -- skip"; else
    wait_mem "${NEED[$m]}"
    note "=== $m GPU witgen (full port) + cpu prove ${ZK_THREADS:-192} ==="
    ( export HIP_VISIBLE_DEVICES="${FIG_GPU:-0}" GKR_WITGEN_GPU=1 GKR_ROUND_GPU=1 GKR_GATES_GPU=1 GKR_WITGEN_FULL=1
      $BIN "$L" "$attn" 1 64 "${ZK_THREADS:-192}" "$lin" --c 9 --cpu --cpu-deg3 --cpu-open --cpu-commit ) > "$log" 2>&1
    note "$m gpuwg_full rc=$? pass=$(grep -c 'All verification passed' "$log") $(summ "$log")"
  fi
done
note FIG3_DONE
