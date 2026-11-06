#!/bin/bash

# zkLLM baseline (monolithic, --splits 1 --sequential, layer prover) on one H100 with the SMs it may use capped
# by CUDA MPS (CUDA_MPS_ACTIVE_THREAD_PERCENTAGE), swept over SM_COUNTS, to find where it stops scaling.
# Each cap is checked with a probe that prints the SM count the device reports to an MPS client at that cap.
#
#   ./run_sm_scaling.sh
#   MODELS="125m" SEQ_LENS="2048" SM_COUNTS="16 32 64 132" REPS=2 ./run_sm_scaling.sh

set -euo pipefail
cd "$(dirname "$0")"
TMP_DIR="${TMPDIR:-/tmp}/zkllm-${USER:-user}"
JOBS="${JOBS:-$(nproc)}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"

MODELS="${MODELS:-125m 1.3b}"
SEQ_LENS="${SEQ_LENS:-64 2048}"
SM_COUNTS="${SM_COUNTS:-132 64 32 16 8}"
REPS="${REPS:-1}"


# CUDA 12.9 is required: ptxas from 12.6 and 12.8 miscompiles the carry-chain field multiply (see Makefile).
: "${CUDA_HOME:?set CUDA_HOME to a CUDA 12.9 toolkit}"
test -x "$CUDA_HOME/bin/nvcc" || { echo "CUDA 12.9 toolkit not found"; exit 1; }
export ARCH="sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 | head -1 | tr -d .)"
TOTAL_SMS=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | grep -q "H100" && echo 132 || echo 0)
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | head -1) ($ARCH) models=$MODELS seq=$SEQ_LENS sms=$SM_COUNTS"
if [ "$(cat .build-arch 2>/dev/null)" != "$ARCH" ]; then make clean; echo "$ARCH" > .build-arch; fi
make -j"$JOBS" partition ppgen commit-param

cat > sm_probe.cu <<'EOF'
#include <cstdio>
#include <cuda_runtime.h>
int main() {
  int n = 0;
  if (cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess) return 1;
  printf("%d\n", n);
  return 0;
}
EOF
"$CUDA_HOME/bin/nvcc" -arch="$ARCH" -o sm_probe sm_probe.cu
FULL_SMS=$(./sm_probe)
[ "$TOTAL_SMS" = 0 ] && TOTAL_SMS=$FULL_SMS
echo "device SMs without MPS: $FULL_SMS"

RESULTS="results-sm-scaling-${RUN_ID}"
OUT="$PWD/$RESULTS"
mkdir -p "$OUT"

export CUDA_MPS_PIPE_DIRECTORY="$TMP_DIR/mps" CUDA_MPS_LOG_DIRECTORY="$TMP_DIR/mps-log"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
trap 'jobs -p | xargs -r kill 2>/dev/null; echo quit | nvidia-cuda-mps-control' EXIT
trap 'echo "== TERM, interrupted $(date)"; exit 143' TERM

prepare() {
  local model="$1"
  if [ ! -d "model-storage/models--facebook--opt-$model" ]; then python download-models.py "facebook/opt-$model" "${HF_TOKEN:-x}" >&2; fi
  if [ ! -f "zkllm-workdir/opt-$model/.complete" ]; then
    rm -rf "zkllm-workdir/opt-$model"
    python opt-commit.py "$model" >&2
    touch "zkllm-workdir/opt-$model/.complete"
  fi
  local sq="$(realpath zkllm-workdir)-sqrt/opt-$model"
  if [ ! -f "$sq/.complete" ]; then
    rm -rf "$sq"
    python opt-recommit.py "$model" --src "$(realpath "zkllm-workdir/opt-$model")" --dst "$sq" >&2
    touch "$sq/.complete"
  fi
  cat "$sq"/* > /dev/null
  echo "$sq"
}

# model seq workdir sms tag
run() {
  local model="$1" seq="$2" wd="$3" sms="$4" tag="$5" rc=0
  local pct=$(( (sms * 100 + TOTAL_SMS - 1) / TOTAL_SMS ))
  (( pct > 100 )) && pct=100
  export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$pct
  local got; got=$(./sm_probe)
  local log="$OUT/${tag}.log" t0 t1
  t0=$(date +%s.%N)
  python opt-partition.py "$model" "$seq" --input_file "opt-${model}_input_${seq}.bin" --devices 0 --workdir "$wd" \
    --splits 1 --sequential --prover-mode layer --out-json "$OUT/${tag}.json" > "$log" 2>&1 || rc=$?
  t1=$(date +%s.%N)
  unset CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
  echo "RESULT model=opt-$model seq=$seq sms_req=$sms mps_pct=$pct sms_seen=$got tag=$tag rc=$rc" \
       "wall_s=$(awk -v a="$t1" -v b="$t0" 'BEGIN{printf "%.2f", a-b}') $(grep -E "SPLIT_SUMMARY" "$log" || true)" \
       | tee -a "$OUT/summary.log"
}

set +e
for model in $MODELS; do
  WD=$(prepare "$model") || { echo "prepare opt-$model failed"; continue; }
  for seq in $SEQ_LENS; do
    for r in $(seq 1 "$REPS"); do
      for sms in $SM_COUNTS; do
        echo "== opt-$model seq=$seq sms=$sms rep=$r $(date)"
        run "$model" "$seq" "$WD" "$sms" "${model}_s${seq}_sm${sms}_r${r}"
      done
    done
  done
done
echo "== done $(date): $OUT"
