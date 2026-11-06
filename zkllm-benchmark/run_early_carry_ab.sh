#!/bin/bash

# Early witgen carry (opt-partition.py --early-carry / ZKLLM_EARLY_CARRY=1) against the per-partition witgen chain,
# and both against the monolithic zkLLM proof (--splits 1, layer path), on the same binary, weights and input.
# Off and on alternate within each repetition so drift on the node hits both arms alike. Every run must report
# status=OK; with the carry on, carry_ok also checks each published carry byte-for-byte against the producing
# prover's own output.
#
#   ./run_early_carry_ab.sh                               # opt-125m, seq_len 2048
#   MODEL=2.7b CONFIGS="mono pipe8 pipe16 pipe32" ./run_early_carry_ab.sh
#
# CONFIGS: mono = --splits 1 --sequential on the layer path (the zkLLM baseline, run once per repetition);
# wf = mono with --witgen-first (zkLLM's method: witgen of every layer into host memory, then proving from it);
# seqP = --sequential --splits P; pipeP = --pipeline --splits P (both on $ZKMOD_PROVER_MODE, off and on).
# COMMITMENTS=sqrt (default) re-commits with square-root shaped generators ; =plain
# uses opt-commit.py's shape.
#
# Needs model-storage: python download-models.py facebook/opt-$MODEL x

set -euo pipefail

cd "$(dirname "$0")"
TMP_DIR="${TMPDIR:-/tmp}/zkllm-${USER:-user}"
JOBS="${JOBS:-$(nproc)}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"

MODEL="${MODEL:-125m}"
SEQ_LEN="${SEQ_LEN:-2048}"
REPS="${REPS:-3}"
CONFIGS="${CONFIGS:-mono pipe4 pipe8 pipe12}"
COMMITMENTS="${COMMITMENTS:-sqrt}"
export ZKMOD_PROVER_MODE="${ZKMOD_PROVER_MODE:-op}"


# CUDA 12.9 is required: ptxas from 12.6 and 12.8 miscompiles the carry-chain field multiply (see Makefile).
: "${CUDA_HOME:?set CUDA_HOME to a CUDA 12.9 toolkit}"
test -x "$CUDA_HOME/bin/nvcc" || { echo "CUDA 12.9 toolkit not found (CUDA_HOME='$CUDA_HOME')"; exit 1; }
echo "building with $("$CUDA_HOME/bin/nvcc" --version | tail -1)"
# Build for this GPU (sm_90 = H100, sm_89 = L40S); objects left for another arch are rebuilt.
export ARCH="${ARCH:-sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 | head -1 | tr -d .)}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | head -1) ($ARCH)"
if [ "$(cat .build-arch 2>/dev/null)" != "$ARCH" ]; then make clean; echo "$ARCH" > .build-arch; fi
make -j"$JOBS" partition ppgen commit-param

test -d "model-storage/models--facebook--opt-$MODEL" || { echo "missing model-storage for opt-$MODEL (download on the login node)"; exit 1; }
WD="zkllm-workdir/opt-$MODEL"
if [ ! -f "$WD/.complete" ]; then
  echo "== committing opt-$MODEL $(date)"
  rm -rf "$WD"
  python opt-commit.py "$MODEL"
  touch "$WD/.complete"
fi
if [ "$COMMITMENTS" = sqrt ]; then
  # Next to the integer weights, so it persists when those are on /scratch; its symlinks need the resolved path.
  SQ="$(realpath zkllm-workdir)-sqrt/opt-$MODEL"
  if [ ! -f "$SQ/.complete" ]; then
    echo "== square-root re-commit opt-$MODEL $(date)"
    rm -rf "$SQ"
    python opt-recommit.py "$MODEL" --src "$(realpath "$WD")" --dst "$SQ"
    touch "$SQ/.complete"
  fi
  WD="$SQ"
fi
echo "weights: $WD"
cat "$WD"/* > /dev/null   # warm the page cache

# Without MPS the pipelined provers time-slice instead of co-executing.
export CUDA_MPS_PIPE_DIRECTORY="$TMP_DIR/mps" CUDA_MPS_LOG_DIRECTORY="$TMP_DIR/mps-log"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d

RESULTS="results-early-carry-ab-opt-${MODEL}-${SEQ_LEN}-${COMMITMENTS}-${RUN_ID}"
OUT="$PWD/$RESULTS"
mkdir -p "$OUT"
trap 'echo quit | nvidia-cuda-mps-control; jobs -p | xargs -r kill 2>/dev/null' EXIT
trap 'echo "== TERM $(date)"; exit 143' TERM

IN="opt-${MODEL}_input_${SEQ_LEN}.bin"
# Each run: tag, carry (0|1), opt-partition.py args. Records wall time and peak GPU memory (0.5 s samples).
run() {
  local tag="$1" carry="$2"; shift 2
  local memlog="$OUT/${tag}.mem" log="$OUT/${tag}.log" rc=0
  : > "$memlog"
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 >> "$memlog"; sleep 0.5; done ) &
  local mon=$!
  local t0 t1
  t0=$(date +%s.%N)
  ZKLLM_EARLY_CARRY="$carry" python opt-partition.py "$MODEL" "$SEQ_LEN" --input_file "$IN" --devices 0 --workdir "$WD" \
    --prover-mode "$ZKMOD_PROVER_MODE" --out-json "$OUT/${tag}.json" "$@" > "$log" 2>&1 || rc=$?
  t1=$(date +%s.%N)
  kill "$mon" 2>/dev/null || true
  local summary
  summary=$(grep -E "SPLIT_SUMMARY|PIPELINE_SUMMARY" "$log" || true)
  echo "RESULT model=opt-$MODEL tag=$tag carry=$carry rc=$rc wall_s=$(awk -v a="$t1" -v b="$t0" 'BEGIN{printf "%.2f", a-b}')" \
       "peak_mem_MiB=$(sort -n "$memlog" | tail -1) $summary" | tee -a "$OUT/summary.log"
}

# Pipeline memory gate: per-prover peak and witgen footprint from a sequential 2-split run, as run_experiments.sh.
echo "== reference run for the memory gate $(date)"
run ref_seq2 0 --splits 2 --sequential
PEAK=$(sort -n "$OUT/ref_seq2.mem" | tail -1)
PREP=$(grep -o 'prep_s=[0-9.]*' "$OUT/ref_seq2.log" | tail -1 | cut -d= -f2)
WITGEN=$(head -n "$(printf '%.0f' "$(awk -v p="${PREP:-0}" 'BEGIN{print p*2}')")" "$OUT/ref_seq2.mem" | sort -n | tail -1)
MEM=(--mem-per-proc-mb $(( PEAK + 1024 )) --witgen-margin-mb $(( ${WITGEN:-0} + 1024 )))
echo "memory gate: ${MEM[*]}"

set +e
for r in $(seq 1 "$REPS"); do
  for cfg in $CONFIGS; do
    P="${cfg//[!0-9]/}"
    case "$cfg" in
      mono)  run "mono_r${r}" 0 --splits 1 --sequential --prover-mode layer; continue ;;
      wf)    run "wf_r${r}" 0 --splits 1 --sequential --prover-mode layer --witgen-first; continue ;;
      seq*)  args=(--splits "$P" --sequential) ;;
      pipe*) args=(--splits "$P" --pipeline "${MEM[@]}") ;;
      *) echo "unknown config $cfg"; continue ;;
    esac
    # alternate which arm goes first
    if (( r % 2 )); then order="0 1"; else order="1 0"; fi
    for c in $order; do
      run "${cfg}_carry${c}_r${r}" "$c" "${args[@]}"
    done
  done
done
set -e

# Per config: mean e2e and witgen chain time (prep_s for seq, witgen_chain_s for pipe), carry off and on, and the
# speedup of each over the monolithic proof's mean e2e.
python - "$OUT/summary.log" <<'EOF' | tee "$OUT/ab.txt"
import re, sys, collections
rows = [dict(re.findall(r'(\w+)=(\S+)', l)) for l in open(sys.argv[1]) if l.startswith('RESULT')]
acc = collections.defaultdict(lambda: collections.defaultdict(list))
mono, wf, bad = [], [], []
for r in rows:
    if r.get('status') != 'OK' or r.get('rc') != '0':
        bad.append(r['tag'])
        continue
    if r['tag'].startswith('mono_'):
        mono.append(float(r['e2e_s']))
        continue
    if r['tag'].startswith('wf_'):
        wf.append(r)
        continue
    m = re.match(r'(\w+?)_carry(\d)_r\d+$', r['tag'])
    if not m:
        continue
    cfg, c = m.groups()
    acc[cfg][c + 'e2e'].append(float(r['e2e_s']))
    acc[cfg][c + 'wg'].append(float(r.get('prep_s') or r.get('witgen_chain_s') or 0))
mean = lambda v: sum(v) / len(v) if v else float('nan')
base = mean(mono)
print(f'monolithic zkLLM: e2e {base:.2f} s (n={len(mono)})')
if wf:
    f = lambda k: mean([float(r.get(k, 'nan')) for r in wf])
    print(f'witness-first (host memory): e2e {f("e2e_s"):.2f} s, record {f("wf_record_s"):.2f} s '
          f'(witgen {f("wf_witgen_s"):.2f}, store {f("wf_store_s"):.2f}), load {f("wf_load_s"):.2f} s, '
          f'proving {f("seq_sum_s"):.2f} s, peak host {f("wf_peak_gb"):.2f} GB (n={len(wf)})')
print(f'{"config":8s} {"n":>3s} {"witgen_off":>10s} {"witgen_on":>9s} {"e2e_off":>8s} {"e2e_on":>7s} '
      f'{"vs_mono_off":>11s} {"vs_mono_on":>10s}')
key = lambda c: (c.rstrip('0123456789'), int(re.sub(r'\D', '', c) or 0))
for cfg in sorted(acc, key=key):
    d = acc[cfg]
    off, on = mean(d['0e2e']), mean(d['1e2e'])
    print(f'{cfg:8s} {min(len(d["0e2e"]), len(d["1e2e"])):3d} {mean(d["0wg"]):10.2f} {mean(d["1wg"]):9.2f} '
          f'{off:8.2f} {on:7.2f} {base / off:10.2f}x {base / on:9.2f}x')
print('failed runs:', ' '.join(bad) if bad else 'none')
EOF
echo "== done $(date): $OUT"
