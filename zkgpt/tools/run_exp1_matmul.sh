#!/usr/bin/env bash
# run_exp1_matmul.sh — Experiment 1 (experiments.tex): matmul layer cost model.
#
# Sweeps the Blendy product-sumcheck over contraction size s=2^n and stage count
# k, capturing per-phase GPU time (expand / ptable / round / fold) AND peak VRAM
# (rocprofv3 allocation trace). Output is regression-ready to fit alpha and to
# confirm cost ~ alpha * k * s with the compute->table shift as k grows.
#
# Fast experiment (~seconds/point). Safe to run alongside Exp 2.
#
# Usage:  tools/run_exp1_matmul.sh            # default sweep
#         NS=14,16,18,20,22 KS=1,2,4,8 tools/run_exp1_matmul.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${BIN:-$HERE/../bin/zkgpt_gpu}"
TS="$(date +%Y%m%d_%H%M%S)"
RES="$HERE/results/matmul"; mkdir -p "$RES/data" "$RES/images"
OUT="${OUT:-$RES/data/exp1_matmul_${TS}.csv}"
LOG="${LOG:-$RES/data/exp1_matmul_${TS}.log}"

# Three axes:
#   NS     contraction size n (layer is s = 2^n)
#   KS     Blendy stage count k — the claimed space-time knob
#   TILES  ext-table tile width t (GKR_SC_TILE_BITS), -1 = untiled
#
# TILES is what makes this the post-tiling sweep. Since tiling caps the
# extension tables at 3^l·2^min(n-l,t), peak memory is expected to be FLAT in k
# and to move only with t; the sweep exists to confirm that and to measure
# whether t costs anything in time. Infeasible (n,k,t) points are skipped
# against the real allocator model (2*ext + 3N + 2*ptable), not a proxy.
NS="${NS:-20,22,24,26,27,29}"
KS="${KS:-1,2,4,8,16}"
TILES="${TILES:--1,14,16,18,20,22}"
MODES="${MODES:-gpu}"
VRAM_GB="${VRAM_GB:-60}"

[[ -x "$BIN" ]] || { echo "build zkgpt_gpu first ($BIN missing)" >&2; exit 1; }
echo "[exp1] matmul sweep  ns=$NS ks=$KS tiles=$TILES modes=$MODES budget=${VRAM_GB}GiB" | tee "$LOG"
echo "[exp1] csv=$OUT log=$LOG" | tee -a "$LOG"

python3 "$HERE/compare_range.py" matmul \
    --bin "$BIN" --ns "$NS" --ks "$KS" --tiles="$TILES" --modes "$MODES" \
    --vram-budget-gb "$VRAM_GB" --out "$OUT" 2>&1 | tee -a "$LOG"

python3 "$HERE/plot_regular.py" cost "$OUT" --outdir "$RES/images" 2>&1 | tee -a "$LOG" || \
    echo "[exp1] plotting skipped (matplotlib?)" | tee -a "$LOG"

echo "[exp1] DONE -> $OUT" | tee -a "$LOG"
