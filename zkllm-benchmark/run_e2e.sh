#!/bin/bash

# zkLLM vs zkMod e2e (run_experiments.sh) on one GPU: CUDA 12.9, MPS, square-root commitments.
#
#   SEQ_LEN=64 ./run_e2e.sh                    # all sizes
#   SEQ_LEN=64 PIPE_SPLITS="4 8" ./run_e2e.sh 125m 1.3b

set -euo pipefail
cd "$(dirname "$0")"
TMP_DIR="${TMPDIR:-/tmp}/zkllm-${USER:-user}"
JOBS="${JOBS:-$(nproc)}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"

MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(125m 1.3b 2.7b 6.7b 13b)
SEQ_LEN="${SEQ_LEN:-2048}"
export ZKMOD_PROVER_MODE="${ZKMOD_PROVER_MODE:-op}"
export PIPE_SPLITS="${PIPE_SPLITS:-4 8 12 16}"


# CUDA 12.9 is required: ptxas from 12.6 and 12.8 miscompiles the carry-chain field multiply (see Makefile).
: "${CUDA_HOME:?set CUDA_HOME to a CUDA 12.9 toolkit}"
test -x "$CUDA_HOME/bin/nvcc" || { echo "CUDA 12.9 toolkit not found"; exit 1; }
export ARCH="sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 | head -1 | tr -d .)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | head -1) ($ARCH) seq_len=$SEQ_LEN models=${MODELS[*]}"
make clean
make -j"$JOBS" partition ppgen commit-param

SQRT_ROOT="$(realpath zkllm-workdir)-sqrt"
for M in "${MODELS[@]}"; do
  if [ ! -d "model-storage/models--facebook--opt-$M" ]; then python download-models.py "facebook/opt-$M" "${HF_TOKEN:-x}"; fi
  if [ ! -f "zkllm-workdir/opt-$M/.complete" ]; then
    rm -rf "zkllm-workdir/opt-$M"
    python opt-commit.py "$M"
    touch "zkllm-workdir/opt-$M/.complete"
  fi
  if [ ! -f "$SQRT_ROOT/opt-$M/.complete" ]; then
    rm -rf "$SQRT_ROOT/opt-$M"
    python opt-recommit.py "$M" --src "$(realpath zkllm-workdir/opt-$M)" --dst "$SQRT_ROOT/opt-$M"
    touch "$SQRT_ROOT/opt-$M/.complete"
  fi
done
export WORKDIR_ROOT="$SQRT_ROOT"

export CUDA_MPS_PIPE_DIRECTORY="$TMP_DIR/mps" CUDA_MPS_LOG_DIRECTORY="$TMP_DIR/mps-log"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d

RESULTS="results-h100-e2e-${SEQ_LEN}-${RUN_ID}"
trap 'echo quit | nvidia-cuda-mps-control' EXIT
trap 'echo "== TERM, interrupted $(date)"; kill $HPID 2>/dev/null; exit 143' TERM

set +e
./run_experiments.sh "$SEQ_LEN" "$RESULTS" "${MODELS[@]}" &
HPID=$!
wait $HPID
rc=$?
set -e
echo "== harness exit $rc $(date)"
python plots/parse_e2e.py "$RESULTS/summary.log" > "$RESULTS/h100_e2e.csv" || true
exit $rc
