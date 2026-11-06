#!/bin/bash
# zkLLM vs zkMod experiments across OPT model sizes, single GPU (device 0).
# Usage: ./run_experiments.sh [seq_len] [out_dir] [models...]
#   ./run_experiments.sh                              # defaults: seq_len=2048, all pre-LN OPT sizes
#   ./run_experiments.sh 2048 results 125m 1.3b        # subset of sizes
#   WORKDIR_ROOT=zkllm-workdir-sqrt ./run_experiments.sh ...   # weights committed elsewhere (opt-recommit.py)
#   ZKMOD_PROVER_MODE=layer ./run_experiments.sh ...            # zkMod partitions on the layer path (default: op
#                                                               # path with streamed activations, ~1/4 the memory)
set -euo pipefail
cd "$(dirname "$0")"

SEQ_LEN="${1:-2048}"
OUT="${2:-results}"
shift $(( $# >= 1 ? 1 : 0 )) || true
shift $(( $# >= 1 ? 1 : 0 )) || true
MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=(125m 1.3b 2.7b 6.7b 13b)   # 350m excluded: post-LN, not implemented (see opt-layer.cu)
fi

mkdir -p "$OUT"
HFTOK="${HF_TOKEN:-x}"
# Kill the nvidia-smi memory monitor if the script exits early (set -e), or it keeps
# appending samples to the last .mem file forever and corrupts its recorded peak.
trap 'jobs -p | xargs -r kill 2>/dev/null' EXIT

echo "== build =="
make partition ppgen commit-param

for M in "${MODELS[@]}"; do
  echo "== opt-$M: weights =="
  if [ ! -d "model-storage/models--facebook--opt-$M" ]; then
    python download-models.py "facebook/opt-$M" "$HFTOK"
  fi

  echo "== opt-$M: commit =="
  COMMIT_LOG="$OUT/opt-${M}_commit.log"
  WD="${WORKDIR_ROOT:-zkllm-workdir}/opt-$M"
  if [ -n "${WORKDIR_ROOT:-}" ] && [ ! -d "$WD" ]; then
    echo "missing $WD (create it with opt-recommit.py)"; exit 1
  fi
  if [ ! -d "zkllm-workdir/opt-$M" ] || [ -z "$(ls -A "zkllm-workdir/opt-$M" 2>/dev/null)" ]; then
    python opt-commit.py "$M" 2>&1 | tee "$COMMIT_LOG"
  fi

  IN="opt-${M}_input_${SEQ_LEN}.bin"

  run() {
    local tag="$1"; shift
    local memlog="$OUT/opt-${M}_${tag}.mem"
    : > "$memlog"
    ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 >> "$memlog"; sleep 0.5; done ) & local mon=$!
    local t0 t1 rc=0
    t0=$(date +%s.%N)
    python opt-partition.py "$M" "$SEQ_LEN" --input_file "$IN" --devices 0 --workdir "$WD" \
      --out-json "$OUT/opt-${M}_${tag}.json" "$@" > "$OUT/opt-${M}_${tag}.log" 2>&1 || rc=$?
    t1=$(date +%s.%N)
    kill "$mon" 2>/dev/null || true
    local peak
    peak=$(sort -n "$memlog" | tail -1)
    local summary
    summary=$(grep -E "SPLIT_SUMMARY|PIPELINE_SUMMARY" "$OUT/opt-${M}_${tag}.log" || true)
    echo "RESULT model=opt-$M tag=$tag rc=$rc wall_s=$(awk -v a="$t1" -v b="$t0" 'BEGIN{printf "%.2f", a-b}') peak_mem_MiB=$peak $summary" | tee -a "$OUT/summary.log"
  }

  echo "== opt-$M: warm page cache =="
  cat "$WD"/* > /dev/null

  echo "== opt-$M: baseline + zkMod =="
  # The monolithic baseline runs on the layer path (its fastest); zkMod partitions on $ZKMOD_MODE.
  ZKMOD_MODE="${ZKMOD_PROVER_MODE:-op}"
  # SKIP_SEQ=1 skips these three and expects $OUT/opt-${M}_zkmod2_seq.{mem,log} from an earlier run (copied in),
  # which the pipeline memory gate below reads: for sweeping PIPE_SPLITS without redoing the baselines.
  if [ -z "${SKIP_SEQ:-}" ]; then
    run zkllm       --splits 1 --sequential
    run zkmod2_seq  --splits 2 --sequential --prover-mode "$ZKMOD_MODE"
    run zkmod4_seq  --splits 4 --sequential --prover-mode "$ZKMOD_MODE"
  fi
  # The pipeline memory gate needs the real per-prover footprint, which is set by the per-layer
  # working set (grows with hidden/ffn dim) and the prover path: take it from this model's own
  # sequential zkMod run, whose partitions use the same path one at a time. cudaMalloc is unchecked
  # in the prover, so an OOM shows up as corrupted proofs rather than a clean error.
  PEAK=$(sort -n "$OUT/opt-${M}_zkmod2_seq.mem" | tail -1)
  # Witgen footprint: the first prep_s seconds of the zkmod2_seq run are its witgen chain (0.5s samples).
  PREP=$(grep -o 'prep_s=[0-9.]*' "$OUT/opt-${M}_zkmod2_seq.log" | tail -1 | cut -d= -f2)
  WITGEN=$(head -n "$(printf '%.0f' "$(awk -v p="${PREP:-0}" 'BEGIN{print p*2}')")" "$OUT/opt-${M}_zkmod2_seq.mem" | sort -n | tail -1)
  MEM="--mem-per-proc-mb $(( PEAK + 1024 )) --witgen-margin-mb $(( ${WITGEN:-0} + 1024 )) --prover-mode $ZKMOD_MODE"
  # PIPE_SPLITS: partition counts of the pipelined runs (default 2 4 8). A count above the model's layer
  # count is rejected by opt-partition.py; uneven splits give the remainder layers to the first partitions.
  for P in ${PIPE_SPLITS:-2 4 8}; do
    run "zkmod${P}_pipe" --splits "$P" --pipeline $MEM
  done
done

echo "== done: see $OUT/summary.log =="
