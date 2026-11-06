#!/bin/bash

# zkLLM's witness-first method (opt-partition.py --witgen-first: witgen of every layer into host memory, then proving
# from the stored tensors) against the monolithic proof that recomputes as it proves (--splits 1, layer path).
# First a smoke test on opt-$SMOKE_MODEL; the full run on $MODELS starts only if it passes. Each pair must report
# status=OK and the witness-first proof's output must equal the monolithic one byte for byte.
#
#   ./run_witgen_first.sh                       # smoke opt-125m, then opt-2.7b and opt-13b, seq_len 2048
#   MODELS=13b REPS=3 ./run_witgen_first.sh
#
# The store needs host memory: about ws_peak_gb in the log (opt-13b at 2048 is a few hundred GB).
# Needs model-storage: python download-models.py facebook/opt-$MODEL x

set -euo pipefail

cd "$(dirname "$0")"
TMP_DIR="${TMPDIR:-/tmp}/zkllm-${USER:-user}"
JOBS="${JOBS:-$(nproc)}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"

SMOKE_MODEL="${SMOKE_MODEL:-125m}"
MODELS="${MODELS:-2.7b 13b}"
SEQ_LEN="${SEQ_LEN:-2048}"
REPS="${REPS:-2}"
COMMITMENTS="${COMMITMENTS:-sqrt}"


# CUDA 12.9 is required: ptxas from 12.6 and 12.8 miscompiles the carry-chain field multiply (see Makefile).
: "${CUDA_HOME:?set CUDA_HOME to a CUDA 12.9 toolkit}"
test -x "$CUDA_HOME/bin/nvcc" || { echo "CUDA 12.9 toolkit not found (CUDA_HOME='$CUDA_HOME')"; exit 1; }
echo "building with $("$CUDA_HOME/bin/nvcc" --version | tail -1)"
export ARCH="${ARCH:-sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 | head -1 | tr -d .)}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | head -1) ($ARCH)"
if [ "$(cat .build-arch 2>/dev/null)" != "$ARCH" ]; then make clean; echo "$ARCH" > .build-arch; fi
make -j"$JOBS" partition ppgen commit-param

RESULTS="results-witgen-first-${SEQ_LEN}-${RUN_ID}"
OUT="$PWD/$RESULTS"
mkdir -p "$OUT"
trap 'jobs -p | xargs -r kill 2>/dev/null' EXIT
trap 'echo "== TERM $(date)"; exit 143' TERM

# Commits opt-$1's weights if needed (and re-commits them square-root shaped with $2=sqrt); prints the workdir.
prepare() {
  local model="$1" com="$2"
  test -d "model-storage/models--facebook--opt-$model" || { echo "missing model-storage for opt-$model" >&2; return 1; }
  local wd="zkllm-workdir/opt-$model"
  if [ ! -f "$wd/.complete" ]; then
    echo "== committing opt-$model $(date)" >&2
    rm -rf "$wd"
    python opt-commit.py "$model" >&2
    touch "$wd/.complete"
  fi
  if [ "$com" = sqrt ]; then
    local sq="$(realpath zkllm-workdir)-sqrt/opt-$model"
    if [ ! -f "$sq/.complete" ]; then
      echo "== square-root re-commit opt-$model $(date)" >&2
      rm -rf "$sq"
      python opt-recommit.py "$model" --src "$(realpath "$wd")" --dst "$sq" >&2
      touch "$sq/.complete"
    fi
    wd="$sq"
  fi
  cat "$wd"/* > /dev/null   # warm the page cache
  echo "$wd"
}

# One monolithic proof: model, workdir, tag, extra opt-partition.py args. Records wall time, peak GPU memory
# (0.5 s samples) and the proof's output tensor (for the byte comparison).
run() {
  local model="$1" wd="$2" tag="$3"; shift 3
  local memlog="$OUT/${tag}.mem" log="$OUT/${tag}.log" rc=0
  : > "$memlog"
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 >> "$memlog"; sleep 0.5; done ) &
  local mon=$!
  local t0 t1
  t0=$(date +%s.%N)
  rm -f part0_prove_out.bin
  python opt-partition.py "$model" "$SEQ_LEN" --input_file "opt-${model}_input_${SEQ_LEN}.bin" --devices 0 --workdir "$wd" \
    --splits 1 --sequential --prover-mode layer --out-json "$OUT/${tag}.json" "$@" > "$log" 2>&1 || rc=$?
  t1=$(date +%s.%N)
  kill "$mon" 2>/dev/null || true
  cp part0_prove_out.bin "$OUT/${tag}.out.bin" 2>/dev/null || true
  local summary
  summary=$(grep -E "SPLIT_SUMMARY" "$log" || true)
  echo "RESULT model=opt-$model tag=$tag rc=$rc wall_s=$(awk -v a="$t1" -v b="$t0" 'BEGIN{printf "%.2f", a-b}')" \
       "peak_mem_MiB=$(sort -n "$memlog" | tail -1) $summary" | tee -a "$OUT/summary.log"
  [ "$rc" = 0 ] && grep -q "status=OK" <<< "$summary"
}

same_output() {
  if cmp -s "$OUT/$1.out.bin" "$OUT/$2.out.bin"; then echo "OUTPUT_MATCH $1 $2 ok" | tee -a "$OUT/summary.log"
  else echo "OUTPUT_MATCH $1 $2 DIFFER" | tee -a "$OUT/summary.log"; return 1; fi
}

echo "== smoke test opt-$SMOKE_MODEL $(date)"
WD=$(prepare "$SMOKE_MODEL" "$COMMITMENTS")
run "$SMOKE_MODEL" "$WD" "smoke_mono" || { echo "SMOKE FAILED (mono)"; exit 1; }
run "$SMOKE_MODEL" "$WD" "smoke_wf" --witgen-first || { echo "SMOKE FAILED (witgen-first)"; exit 1; }
same_output smoke_mono smoke_wf || { echo "SMOKE FAILED (outputs differ)"; exit 1; }
echo "== smoke test passed $(date)"

set +e
for model in $MODELS; do
  echo "== full run opt-$model $(date)"
  WD=$(prepare "$model" "$COMMITMENTS") || { echo "prepare opt-$model failed"; continue; }
  echo "weights: $WD"
  for r in $(seq 1 "$REPS"); do
    # alternate which arm goes first so drift on the node hits both alike
    if (( r % 2 )); then order="mono wf"; else order="wf mono"; fi
    for arm in $order; do
      if [ "$arm" = wf ]; then run "$model" "$WD" "${model}_wf_r${r}" --witgen-first
      else run "$model" "$WD" "${model}_mono_r${r}"; fi
    done
    same_output "${model}_mono_r${r}" "${model}_wf_r${r}"
  done
done
set -e

python - "$OUT/summary.log" <<'EOF' | tee "$OUT/witgen-first.txt"
import re, sys, collections
lines = open(sys.argv[1]).read().splitlines()
rows = [dict(re.findall(r'(\w+)=(\S+)', l)) for l in lines if l.startswith('RESULT')]
acc = collections.defaultdict(lambda: collections.defaultdict(list))
bad = [r['tag'] for r in rows if r.get('status') != 'OK' or r.get('rc') != '0']
for r in rows:
    if r['tag'] in bad or r['tag'].startswith('smoke_'):
        continue
    m = re.match(r'(.+)_(mono|wf)_r\d+$', r['tag'])
    if m:
        acc[m.group(1)][m.group(2)].append(r)
mean = lambda v: sum(v) / len(v) if v else float('nan')
f = lambda rs, k: mean([float(r[k]) for r in rs if k in r])
print(f'{"model":6s} {"n":>2s} {"mono_e2e":>9s} {"wf_e2e":>8s} {"wf_record":>9s} {"wf_witgen":>9s} {"wf_store":>8s} '
      f'{"wf_load":>7s} {"mono_prove":>10s} {"wf_prove":>8s} {"host_gb":>7s}')
for model, d in acc.items():
    mo, wf = d['mono'], d['wf']
    print(f'{model:6s} {min(len(mo), len(wf)):2d} {f(mo, "e2e_s"):9.2f} {f(wf, "e2e_s"):8.2f} {f(wf, "wf_record_s"):9.2f} '
          f'{f(wf, "wf_witgen_s"):9.2f} {f(wf, "wf_store_s"):8.2f} {f(wf, "wf_load_s"):7.2f} {f(mo, "seq_sum_s"):10.2f} '
          f'{f(wf, "seq_sum_s"):8.2f} {f(wf, "wf_peak_gb"):7.2f}')
mism = [l for l in lines if l.startswith('OUTPUT_MATCH') and l.endswith('DIFFER')]
print('failed runs:', ' '.join(bad) if bad else 'none')
print('output mismatches:', len(mism))
EOF
echo "== done $(date): $OUT"
