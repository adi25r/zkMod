#!/usr/bin/env bash
# run_exp4_llama8b.sh — Experiment 4: do the GPT-2 conclusions hold at Llama-8B?
#
# GPT-2 (monolithic, L=12) established three claims. This script re-tests each of
# them at Llama-8B, which can ONLY be run split (see WHY SPLIT below):
#
#   C1  t is the memory knob; matmul/lasso k is NOT.
#       -> compare peak across (k=8,t=20) vs (k=16,t=20)  [k varies, expect flat]
#          and across (k=8,t=20) vs (k=8,t=-1)/(k=8,t=18) [t varies, expect move]
#   C2  Tuning is a FEASIBILITY gate, not a latency win.
#       -> does untiled (t=-1) OOM at 8B the way untiled L=12 OOMed on GPT-2?
#   C3  UVM is a bad fallback.
#       -> device vs UVM at an IDENTICAL config (t=20,k=8) isolates the backend.
#
# WHY SPLIT: monolithic 8B is impossible, for two independent reasons.
#   (a) total_in_size at depth 32 = 6,979,452,928 > 2^32, and layer::size /
#       uniGate.g / binGate.{g,u,v} are all u32 -> neuralNetwork.cpp asserts.
#       Max monolithic depth is L=19.
#   (b) Even with u64 indices, n = C.circuit[0].bit_length = 33, so the 3N GPU
#       floor (= 3*2^n*32B, independent of BOTH k and t) is 768 GiB. At L=19 it
#       is still 384 GiB. Neither fits a 64 GiB MI210 or the 723 GB host.
#   Splitting to 1 layer/block gives n=28 -> a 24 GiB floor, which fits. So
#   depth partitioning is what makes 8B representable at all, not an optimization.
#
# METRIC: makespan = max_i(prover_zkgpt_s) over partitions, per the split
# convention (zkGPT's parallel path reports max, not sum). Partitions are NOT
# homogeneous: p0 carries the input commit and runs ~2x the time and ~2x the peak
# of p1..pn. p0 USUALLY carries the max, but not always. A truncated run therefore gives a LOWER BOUND on the true
# 32-partition makespan. Runs default to UNCAPPED (PART_TIMEOUT=0) so headline
# numbers come from all 32 partitions; set PART_TIMEOUT=<sec> for a quick probe.
#
# SERIAL BY NECESSITY: each 8B config peaks at ~455 GB host, so two concurrent
# runs OOM a 723 GB box. Do not parallelise these across GPUs.
#
# Usage:  tools/run_exp4_llama8b.sh [--gpu N] [--wait-for-pid PID]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

GPU="${GPU:-2}"
WAIT_PID="${WAIT_PID:-}"
# 0 = no wall-clock cap; every config runs all 32 partitions. At ~400 s/partition
# that is ~3.5 h per device config and ~8 h for the UVM arm. Set to a positive
# number of seconds only for a quick ranking probe -- never for headline numbers.
PART_TIMEOUT="${PART_TIMEOUT:-0}"
# Prove one partition per structural role instead of all 32 (see sweep_partitions.sh).
# ON by default: makespan and peak VRAM are role-determined, so this is the same
# measurement ~10x cheaper. Set 0 for a real end-to-end proof.
export UNIQUE_PARTS="${UNIQUE_PARTS:-1}"
UVM_TIMEOUT="${UVM_TIMEOUT:-0}"
OUTDIR="${OUTDIR:-tools/results/e2e_v3}"

while [[ $# -gt 0 ]]; do case "$1" in
  --gpu) GPU="$2"; shift 2;;
  --wait-for-pid) WAIT_PID="$2"; shift 2;;
  *) echo "unknown arg: $1" >&2; exit 1;; esac; done

mkdir -p "$OUTDIR/data"
DEV="$HERE/../bin/main_gkr_split"
UVM="$HERE/../bin/main_gkr_split_uvm"
for b in "$DEV" "$UVM"; do
  [[ -x "$b" ]] || { echo "missing $b — build: cmake --build build --target main_gkr_split main_gkr_split_uvm -j" >&2; exit 1; }
done

# Optionally hold until an earlier run releases its host memory.
if [[ -n "$WAIT_PID" ]]; then
  echo "[exp4] waiting for pid $WAIT_PID to exit before starting (host-memory serialisation)"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 120; done
  echo "[exp4] pid $WAIT_PID gone; proceeding"
fi

# one config: <tag> <binary> <tile_bits> <lasso_k> <c> <timeout> [xnack]
run_cfg() {
  local tag="$1" bin="$2" tile="$3" k="$4" c="$5" to="$6" xnack="${7:-0}"
  local out="$OUTDIR/data/exp4_llama8b_${tag}.csv"
  echo ""
  echo "[exp4] ===== $tag =====  tile_bits=$tile lasso_k=$k c=$c timeout=${to}s xnack=$xnack bin=$(basename "$bin")"
  echo "[exp4] $(date '+%F %T')  free_host=$(free -g | awk 'NR==2{print $7}')GB"
  HSA_XNACK="$xnack" GKR_SC_TILE_BITS="$tile" HIP_VISIBLE_DEVICES="$GPU" BIN="$bin" \
    tools/sweep_partitions.sh --model llama8b \
      --splits 32 --ks "$k" --cs "$c" --mks 1 --rks 1 --retries 0 \
      --timeout "$to" --out "$out" --out-dir "$OUTDIR"
  # makespan = max prover_zkgpt_s over whatever partitions completed.
  # NB sweep_partitions.sh writes logs to $(dirname "$out")/logs_<TS>, i.e. under
  # OUTDIR/data -- not OUTDIR. Globbing the wrong level silently reports "no log".
  local log; log="$(ls -t "$OUTDIR"/data/logs_*/s32_c${c}_mk1_rk1_k${k}.log 2>/dev/null | head -1)"
  if [[ -n "$log" ]]; then
    # offsets: "prover_zkgpt_s=" is 15 chars, "gpu_peak_mib=" is 13. Getting the
    # first wrong lops a digit off every value and silently understates makespan.
    awk -v tag="$tag" '
      match($0,/prover_zkgpt_s=[0-9.]+/){v=substr($0,RSTART+15,RLENGTH-15)+0; if(v>mx)mx=v; n++}
      match($0,/prover_total_s=[0-9.]+/){w=substr($0,RSTART+15,RLENGTH-15)+0; if(w>mt)mt=w}
      match($0,/gpu_peak_mib=[0-9.]+/){p=substr($0,RSTART+13,RLENGTH-13)+0; if(p>pk)pk=p}
      END{if(n)printf "[exp4] RESULT %s  parts=%d  makespan_core_s=%.2f  makespan_total_s=%.2f  peak_mib=%.1f  W=%d\n",tag,n,mx,mt,pk,int(65536/pk)
          else printf "[exp4] RESULT %s  NO PARTITIONS COMPLETED (OOM/assert — see log)\n",tag}' "$log"
  else
    echo "[exp4] RESULT $tag  no log found"
  fi
}

echo "[exp4] start $(date '+%F %T')  gpu=$GPU  outdir=$OUTDIR"

# ---------------------------------------------------------------------------
# WHY BOTH k AND t MATTER HERE, AND WHY THEY ARE COUPLED.
# Tiling is NOT matmul-only. The lasso funnels through the same routine:
#   gpu_lasso_product_sumcheck (gpu_range_prover.hpp)
#     -> gpu_product_sumcheck_host (gpu_matmul_prover.hpp)
#       -> product_sumcheck (gpu_matmul_prover.hpp), which reads GKR_SC_TILE_BITS.
# product_sumcheck applies Blendy TWICE: l_stage stages the current variables
# (ptable_cap = 3^l_stage) and tile_bits tiles the future-variable axis
# (fv_cap = min(n-l, t)), giving
#       ext_cap = 3^l_stage * 2^min(n - l_stage, t)
# with l_stage = ceil(n / lasso_k) on the lasso side. So k and t are NOT
# independent knobs: raising k lowers l_stage, which shrinks 3^l_stage but
# GROWS the future-variable count (n - l_stage) that t has to cap. The min()
# is what couples them. A grid -- not two separate line searches -- is required
# to see the interaction, which is the point of sweeping both cross-sections.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# GRID. Two axes are swept jointly because k and t are coupled through the
# min(n-l, t) above; c is swept as a line because it acts on the range prover,
# not the sumcheck, so it is separable from (t,k).
#
# CHOOSING k VALUES: only l_stage = ceil(n/k) matters, and at 8B n=28 per
# partition, so MANY k collapse to the same run --
#     k  = 2   3   4   5   6   7/8  10  14  15..28  29+
#     l  = 14  10  7   6   5   4    3   2   2       1
# Sweeping k=8,16,32 therefore samples only l = 4,2,1 -- three distinct points
# dressed up as a k axis. KS below hits each distinct l instead. (k=1 -> l=28 is
# excluded: ptable_cap = 3^28 overflows int and the config is not runnable.)
#
# CHOOSING t VALUES: ext_cap = 3^l * 2^min(n-l, t), so each +1 of t DOUBLES the
# ext table until it saturates at t >= n-l. The feasible window measured so far
# is t in [14,20] at k=8; TS brackets it on both sides to locate the two cliffs.
# ---------------------------------------------------------------------------
TS="${TS:--1,12,14,15,16,17,18,19,20,21,22,24}"
KS="${KS:-2,3,4,5,6,7,10,14,28}"
CS="${CS:-4,6,7,10,13,16,20}"
TBASE="${TBASE:-20}"       # (t,k) held here while sweeping c
KBASE="${KBASE:-8}"
CBASE="${CBASE:-7}"        # c held here while sweeping the (t,k) grid
GRID="${GRID:-1}"          # 1 = full t x k grid; 0 = cross-sections only
RUN_UVM="${RUN_UVM:-0}"    # UVM arm is OFF by default: managed runs have wedged
                           # the driver before.

IFS=',' read -ra TARR <<< "$TS"
IFS=',' read -ra KARR <<< "$KS"
IFS=',' read -ra CARR <<< "$CS"

if [[ "$GRID" == 1 ]]; then
  for t in "${TARR[@]}"; do for k in "${KARR[@]}"; do
    run_cfg "t${t}_k${k}_c${CBASE}" "$DEV" "$t" "$k" "$CBASE" "$PART_TIMEOUT"
  done; done
else
  for t in "${TARR[@]}"; do
    run_cfg "t${t}_k${KBASE}_c${CBASE}" "$DEV" "$t" "$KBASE" "$CBASE" "$PART_TIMEOUT"
  done
  for k in "${KARR[@]}"; do
    run_cfg "t${TBASE}_k${k}_c${CBASE}" "$DEV" "$TBASE" "$k" "$CBASE" "$PART_TIMEOUT"
  done
fi

# c line at the base (t,k)
for c in "${CARR[@]}"; do
  run_cfg "t${TBASE}_k${KBASE}_c${c}" "$DEV" "$TBASE" "$KBASE" "$c" "$PART_TIMEOUT"
done

[[ "$RUN_UVM" == 1 ]] && run_cfg "uvm_t${TBASE}_k${KBASE}_c${CBASE}" "$UVM" "$TBASE" "$KBASE" "$CBASE" "$UVM_TIMEOUT" 1

echo ""
echo "[exp4] DONE $(date '+%F %T') — grep '^\[exp4\] RESULT' on this log for the summary"
