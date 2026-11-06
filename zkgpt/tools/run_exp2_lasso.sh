#!/usr/bin/env bash
# run_exp2_lasso.sh — Experiment 2 (experiments.tex): nonlinear (logup) cost model.
#
# Sweeps the chunked-logup range prover over chunk bits c and Blendy stages k,
# across several model sizes (layer counts), capturing the full phase breakdown
# (setup / invert / commit / deg3 / deg1 / open) AND peak VRAM. Used to (a) show
# cost is convex in c with an interior c*, (b) fit alpha/beta and the dominant
# commit constant, and (c) confirm the batch-inversion (logup reciprocals) term
# is negligible vs the Hyrax commit.
#
# SLOW experiment: each (config,c,k) point runs the prover twice (timing + a
# rocprofv3 memory pass) and the commit dominates (~10-60 s/point), so the grid
# is kept deliberately small but spanning. Designed to finish overnight.
#
# Sweep rationale:
#   * range_size=48 -> c divisors {2,3,4,6,8,12,16}; we skip c=2 (huge M, slow,
#     table side trivial) and sweep c in {3,4,6,8,12,16} to bracket the c* minimum.
#   * k in {1,2,4}: the deg1 space-time knob (minor here since commit dominates,
#     but needed to confirm that).
#   * configs at L in {1,2,4} GPT-2 blocks: varies |constraints| so a regression
#     can separate the per-constraint terms from the fixed cost.
#   * mode=gpu only by default (set MODES=cpu,gpu to also get the CPU deg1 path;
#     doubles runtime).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${BIN:-$HERE/../bin/zkgpt_gpu}"
TS="$(date +%Y%m%d_%H%M%S)"
RES="$HERE/results/lasso"; mkdir -p "$RES/data" "$RES/images"
OUT="${OUT:-$RES/data/exp2_lasso_${TS}.csv}"
LOG="${LOG:-$RES/data/exp2_lasso_${TS}.log}"

# NOTE: HIP's device order differs from rocm-smi's; pick the index by PCI address.
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-3}"
RANGE_SIZE="${RANGE_SIZE:-48}"
CS="${CS:-3,4,6,8,12,16}"
KS="${KS:-1,2,4}"
# MODES=cpu,gpu gives the deg1 sumcheck CPU-vs-GPU comparison; only deg1 switches
# between modes. Peak GPU mem is profiled once (on the gpu rows) since the commit
# governs it identically.
MODES="${MODES:-gpu}"
# GPT-2 block dims; vary layer count L to scan the constraint-count axis.
CONFIGS="${CONFIGS:-1 12 64 768 3072|2 12 64 768 3072|4 12 64 768 3072}"

[[ -x "$BIN" ]] || { echo "build zkgpt_gpu first ($BIN missing)" >&2; exit 1; }
echo "[exp2] lasso sweep  range_size=$RANGE_SIZE cs=$CS ks=$KS modes=$MODES" | tee "$LOG"
echo "[exp2] configs=$CONFIGS" | tee -a "$LOG"
echo "[exp2] csv=$OUT log=$LOG" | tee -a "$LOG"

TMPDIR_E2="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_E2"' EXIT
i=0
IFS='|' read -ra CFG_ARR <<< "$CONFIGS"
for cfg in "${CFG_ARR[@]}"; do
    L="${cfg%% *}"
    echo "[exp2] === config L=$L  ($cfg) ===" | tee -a "$LOG"
    part="$TMPDIR_E2/part_${i}.csv"
    # add a 'config' column by post-processing: compare_range writes its own csv;
    # we tag rows with the layer count via awk after.
    python3 "$HERE/compare_range.py" full \
        --bin "$BIN" --config "$cfg" --range-size "$RANGE_SIZE" \
        --cs "$CS" --ks "$KS" --modes "$MODES" --out "$part" 2>&1 | tee -a "$LOG"
    # prepend a layers column
    awk -v L="$L" 'NR==1{print "layers,"$0; next}{print L","$0}' "$part" \
        > "$TMPDIR_E2/tagged_${i}.csv"
    i=$((i+1))
done

# Merge all tagged CSVs (keep one header).
{ head -1 "$TMPDIR_E2/tagged_0.csv"
  for f in "$TMPDIR_E2"/tagged_*.csv; do tail -n +2 "$f"; done
} > "$OUT"

python3 "$HERE/plot_regular.py" cost "$OUT" --outdir "$RES/images" 2>&1 | tee -a "$LOG" || \
    echo "[exp2] plotting skipped (matplotlib?)" | tee -a "$LOG"

echo "[exp2] DONE -> $OUT" | tee -a "$LOG"
