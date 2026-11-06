#!/bin/bash
# Paper-arithmetization MONOLITH baseline: our binary, splits=1, CPU
# witgen + CPU prove. Differs from the arms in the split count only (same layout rule:
# the probe pads the monolith to 2*pow2(input+aux) exactly as modular-zkp's demo does).
# Runs after run_paper_arith.sh so the arms' timings are not perturbed.
set -u
cd "$(dirname "$(readlink -f "$0")")/../../../../.."
LOGS=${LOGS:-tools/results/speedup_analysis_models/paper_arith/logs}; mkdir -p "$LOGS"; Q=tools/results/speedup_analysis_models/paper_arith/_queue/run_paper_arith.out
# ROCm 7 runtime: libamdhip64.so.7 comes from /opt/rocm-scale/lib, libomp from /opt/rocm/llvm/lib.
export LD_LIBRARY_PATH="/opt/rocm-scale/lib:/opt/rocm/lib:/opt/rocm/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ZKGPT_WITNESS_CACHE=0 GKR_LASSO_CPU=1 GKR_MATMUL_CPU=1 GKR_PRED_CPU=1 GKR_SC_CPU=1
unset HIP_VISIBLE_DEVICES GKR_BFEVAL_GPU GKR_WITGEN_GPU GKR_ROUND_GPU GKR_SC_STREAM_BITS
declare -A SPEC=( [gpt2]="12 768 3072" [opt350m]="24 1024 4096" )
note(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }
LEN=${LEN:-64}; SUF=""; [[ $LEN -ne 64 ]] && SUF="_len${LEN}"
for m in "$@"; do
  read -r L attn lin <<< "${SPEC[$m]}"
  if grep -q "All verification passed" "$LOGS/${m}${SUF}_arm0_mono.log" 2>/dev/null; then note "$m arm0 already done -- skip"; continue; fi
  note "=== $m len$LEN arm0 (paper mono) ==="
  ${BIN:-./bin/main_gkr_split_witgen} "$L" "$attn" 1 "$LEN" "${ZK_THREADS:-192}" "$lin" --c 9 --cpu --cpu-deg3 --cpu-open --cpu-commit > "$LOGS/${m}${SUF}_arm0_mono.log" 2>&1
  note "$m len$LEN arm0 rc=$? pass=$(grep -c 'All verification passed' $LOGS/${m}${SUF}_arm0_mono.log) $(grep -E '^PHASE .*ev=end .*phase=(witgen|prove) ' $LOGS/${m}${SUF}_arm0_mono.log | grep -oE 'phase=\w+|dur_s=[0-9.]+' | paste -sd' ')"
done
note MONO_DONE
