#!/usr/bin/env bash
# sweep_partitions.sh — THE partition/staging sweep for the zkMod split prover:
# k (matmul_k/range_k/lasso_k), c (range chunk bits), t (tile bits), and
# partitions (even split counts OR explicit uneven compositions), all in one
# driver, for bin/main_gkr_split.
#
# Covers the k/c sweep, the (t, lasso_k) axis, multi-GPU fan-out for Llama, and
# explicit-composition sweeps with boundary analysis.
#
# Emits ONE CSV ROW PER PARTITION in the schema tools/wave_pack.py consumes:
#
#   config_id,num_splits,composition,c,matmul_k,range_k,lasso_k,tile_bits,\
#   stream_bits,status,part_idx,layers,part_time_s,part_gpu_peak_mib,\
#   part_host_peak_mib,part_prover_zkgpt_s,part_circuit_s,part_range_s
#
# (`composition` is blank for --splits points; `num_splits` is the composition's
# length for --configs points.) It parses the binary's own PARTITION line, which
# self-reports per-partition wall cost (time_s), device-VRAM high-water
# (gpu_peak_mib) AND host-RSS high-water (host_peak_mib) via internal samplers —
# correct on ROCm/AMD, no external polling:
#
#   PARTITION idx=I layers=L ... lasso_k=KK ... time_s=T ... gpu_peak_mib=V \
#       host_peak_mib=H
#
# A run is ACCEPT iff it prints "Split proof results"; OOM / REJECT / TIMEOUT
# points get a status row with no partition data so wave_pack skips them.
#
# The per-partition (cost, VRAM, host RAM) points feed wave_pack.py, which
# computes the wave-scheduled makespan under a VRAM (and, since stage-1 source
# streaming, optionally a HOST-RAM) cap — plans.tex eq. total_time / wave_vram.
#
# The knobs (all BlendySC / Lasso / logup space-time parameters from plan.tex):
#   splits/configs — partition count (even) or explicit composition (uneven)
#   c        — range chunk bits (see the c-loop comment)
#   matmul_k — matmul GKR Blendy stage count (l_stage = ceil(n/k)); swept axis
#   range_k  — Lasso degree-1 sumcheck stages
#   lasso_k  — Blendy stage COUNT of the (dominant) Lasso product-sumcheck; THE
#              VRAM lever pre-streaming. HIGHER k => smaller 3^l ext buffer.
#   t        — extension-table tile bits (GKR_SC_TILE_BITS / --sc-tile-bits);
#              the memory knob POST-tiling
#              Phase 2. Empty (default) = don't touch it at all (inherit the
#              caller's env / the binary's own default), for callers like
#              run_exp4_llama8b.sh that already export GKR_SC_TILE_BITS.
#   stream-bits — stage-1 Lasso source streaming window (GKR_SC_STREAM_BITS);
#              collapses the device peak so HOST RAM caps the wave width
#              instead (project_source_streaming_host_bound). Same empty-means-
#              inherit convention as t.
#   lin      — FFN width (6th positional argv); <=0 => embed*3.
#
# --model presets the (depth, embed, lin) shape (main_gkr_split forces
# head_dim=64, so num_heads = embed/64):
#   gpt2     depth=12 embed=768  lin=3072    -> results/kc
#   llama2b  depth=32 embed=2048 lin=8192    -> results/llama2b
#   llama8b  depth=32 embed=4096 lin=14336   -> results/llama
# Any of --depth/--embed/--lin/--splits override the preset.
#
# --gpus with more than one device fans the SPLITS values round-robin across GPUs
# (one background worker each, own CSV) and auto-merges.
# A single device (default) runs the grid sequentially.
# (Multi-GPU fan-out applies to --splits only, not --configs or --tune.)
#
# Usage:
#   # single-GPU GPT-2 sweep:
#   HIP_VISIBLE_DEVICES=6 tools/sweep_partitions.sh --model gpt2 \
#       --splits 3,4,6,12 --mks 1,2,4 --ks 8,16,32 --cs 9 --rks 1
#
#   # multi-GPU Llama-8B sweep, splits fanned across GPUs 0-3:
#   tools/sweep_partitions.sh --model llama8b --gpus 0,1,2,3 --splits 8,16,32 \
#       --ks 16,32 --mks 1,2
#
#   # add the tile-bits axis (post-tiling memory knob):
#   tools/sweep_partitions.sh --model gpt2 --splits 12 --ks 8,16 --ts -1,14,18,20
#
#   # explicit uneven compositions instead of even splits:
#   tools/sweep_partitions.sh --model gpt2 --configs "6,6;3,9;2,4,6" --cs 9
#   python3 tools/wave_pack.py <that csv> --analyze     # per-cut activation-width analysis
#
#   # coordinate-descent tune (c, then lasso_k, then t) at one splits value,
#   # minimizing wave-packed time instead of running the full grid:
#   tools/sweep_partitions.sh --model gpt2 --splits 12 --tune \
#       --cs 10,12,13,14,16 --ks 8,16,32 --ts -1,16,18,20
#
# Then (grid mode):  python3 tools/wave_pack.py <that csv> --vram-gb 64 [--host-gb 426]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${BIN:-$HERE/../bin/main_gkr_split}"
# UNIQUE_PARTS=1 -> pass --unique-parts: prove one block per structural role
# (first / middle / last) instead of all N. Partitions sharing (role, split_depth)
# do identical work, so makespan and peak VRAM are unchanged, but a 32-way llama8b
# split costs 3 partitions instead of 32 (~190 min -> ~20 min). The proof emitted
# in this mode is NOT end-to-end valid -- measurement only.
UNIQUE_PARTS="${UNIQUE_PARTS:-0}"
UP_ARG=(); [[ "$UNIQUE_PARTS" == 1 ]] && UP_ARG=(--unique-parts)

# ---- model presets (overridable) --------------------------------------------
MODEL="${MODEL:-gpt2}"
DEPTH="${DEPTH:-}"; EMBED="${EMBED:-}"; LIN="${LIN:-}"; SPLITS="${SPLITS:-}"
OUTDIR="${OUTDIR:-}"
LEN="${LEN:-30}"; THREADS="${THREADS:-32}"
# c must divide range_size=54 (legal: 1,2,3,6,9,18,27,54); the binary rejects others.
CS="${CS:-9}"; MKS="${MKS:-1,2,4}"; RKS="${RKS:-1}"; KS="${KS:-8,16,32}"
# TS / STREAM_BITS default EMPTY = do not pass --sc-tile-bits / touch
# GKR_SC_STREAM_BITS at all (inherit the caller's exported env, or the binary's
# own default). This is what keeps run_exp4_llama8b.sh's external
# `GKR_SC_TILE_BITS=$tile tools/sweep_partitions.sh ...` working unchanged.
TS="${TS:-}"; STREAM_BITS="${STREAM_BITS:-}"
CONFIGS="${CONFIGS:-}"          # ';'-separated uneven compositions, e.g. "6,6;3,9"
GPUS="${GPUS:-}"
TIMEOUT="${TIMEOUT:-3600}"; OUT="${OUT:-}"
RETRIES="${RETRIES:-1}"          # re-run non-deterministic failures (crash/timeout) this many times
TUNE=0
VRAM_GB="${VRAM_GB:-64}"; HOST_GB="${HOST_GB:-}"       # passed to wave_pack.py in --tune mode
TUNE_STAGES="${TUNE_STAGES:-c k t}"                    # coordinate-descent order

while [[ $# -gt 0 ]]; do case "$1" in
  --model) MODEL="$2"; shift 2;;
  --depth) DEPTH="$2"; shift 2;; --embed) EMBED="$2"; shift 2;; --lin) LIN="$2"; shift 2;;
  --splits) SPLITS="$2"; shift 2;; --cs) CS="$2"; shift 2;; --mks) MKS="$2"; shift 2;;
  --rks) RKS="$2"; shift 2;; --ks) KS="$2"; shift 2;; --gpus) GPUS="$2"; shift 2;;
  --ts) TS="$2"; shift 2;; --stream-bits) STREAM_BITS="$2"; shift 2;;
  --configs) CONFIGS="$2"; shift 2;;
  --len) LEN="$2"; shift 2;; --threads) THREADS="$2"; shift 2;;
  --timeout) TIMEOUT="$2"; shift 2;; --out) OUT="$2"; shift 2;; --out-dir) OUTDIR="$2"; shift 2;;
  --retries) RETRIES="$2"; shift 2;;
  --tune) TUNE=1; shift;;
  --tune-stages) TUNE_STAGES="$2"; shift 2;;
  --vram-gb) VRAM_GB="$2"; shift 2;; --host-gb) HOST_GB="$2"; shift 2;;
  *) echo "unknown arg: $1" >&2; exit 1;; esac; done

# Resolve model preset for any dimension/dir the caller did not pin.
case "$MODEL" in
  gpt2)    : "${DEPTH:=12}" "${EMBED:=768}"  "${LIN:=3072}"  "${SPLITS:=3,4,6,12}"; DEFDIR="kc";      TIMEOUT="${TIMEOUT:-3600}";;
  llama2b) : "${DEPTH:=32}" "${EMBED:=2048}" "${LIN:=8192}"  "${SPLITS:=8,16,32}"; DEFDIR="llama2b"; TIMEOUT="${TIMEOUT:-5400}";;
  llama8b) : "${DEPTH:=32}" "${EMBED:=4096}" "${LIN:=14336}" "${SPLITS:=8,16,32}"; DEFDIR="llama";   TIMEOUT="${TIMEOUT:-5400}";;
  *) echo "unknown --model '$MODEL' (want gpt2|llama2b|llama8b)" >&2; exit 1;; esac

[[ -x "$BIN" ]] || { echo "missing $BIN — build: cmake --build build --target main_gkr_split -j" >&2; exit 1; }
TS_STAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${OUTDIR:-$HERE/results/$DEFDIR}"; mkdir -p "$OUTDIR"

CSV_HEADER="config_id,num_splits,composition,c,matmul_k,range_k,lasso_k,tile_bits,stream_bits,status,part_idx,layers,part_time_s,part_gpu_peak_mib,part_host_peak_mib,part_prover_zkgpt_s,part_circuit_s,part_range_s"

# Run ONE (P-or-comp, C, MK, RK, KK, T, STREAM) point: invoke the binary, parse
# its PARTITION lines (or emit a single non-ACCEPT status row), append to $out.
# For --configs points also append boundary rows (BND_RE-style) to $out.boundaries.csv.
#   run_point <out> <cid> <num_splits_field> <composition_field> \
#             <positional_splits_arg> <extra_bin_args...> -- <C> <MK> <RK> <KK> <T> <STREAM>
run_point() {
  local out="$1" cid="$2" nsplits_field="$3" comp_field="$4" pos_splits="$5"; shift 5
  local -a extra=()
  while [[ "$1" != "--" ]]; do extra+=("$1"); shift; done
  shift
  local C="$1" MK="$2" RK="$3" KK="$4" T="$5" STREAM="$6"

  local logdir; logdir="$(dirname "$out")/logs_${TS_STAMP}"; mkdir -p "$logdir"
  local LOG="$logdir/${cid}.log" STATUS="" rc attempt
  local -a tenv=()
  [[ -n "$T" ]]      && tenv+=(GKR_SC_TILE_BITS="$T")
  [[ -n "$STREAM" ]] && tenv+=(GKR_SC_STREAM_BITS="$STREAM")
  for attempt in $(seq 0 "$RETRIES"); do
    [[ $attempt -gt 0 ]] && echo "[sweep]   retry $attempt/$RETRIES ($cid)" >&2
    echo "[sweep] $cid lin=$LIN (gpu=${HIP_VISIBLE_DEVICES:-?}) ..." >&2
    # --timeout 0 => no wall-clock cap at all. A truncated run yields a
    # makespan over only the partitions that finished, which is a LOWER
    # BOUND (p0 usually but not always carries the max), so any headline
    # number should come from an uncapped run.
    local to=(); [[ "$TIMEOUT" != 0 ]] && to=(timeout "$TIMEOUT")
    env "${tenv[@]}" "${to[@]}" "$BIN" "$DEPTH" "$EMBED" "$pos_splits" "$LEN" "$THREADS" "$LIN" \
        --c "$C" --matmul-k "$MK" --range-k "$RK" --lasso-k "$KK" \
        "${extra[@]}" "${UP_ARG[@]}" > "$LOG" 2>&1
    rc=$?
    if   grep -q "Split proof results" "$LOG"; then STATUS=ACCEPT
    elif grep -q "RESULT: REJECT\|REJECTED" "$LOG"; then STATUS=REJECT
    elif grep -qi "out of memory" "$LOG"; then STATUS=OOM
    elif [[ $rc == 124 ]]; then STATUS=TIMEOUT
    else STATUS=FAIL; fi                      # crash / GPU fault / other
    [[ "$STATUS" == ACCEPT || "$STATUS" == REJECT || "$STATUS" == OOM ]] && break
  done
  echo "[sweep]   -> $STATUS" >&2
  if [[ "$STATUS" == ACCEPT ]]; then
    grep '^PARTITION' "$LOG" | while read -r line; do
      local idx lay t v h pz cs rs
      idx=$(sed -nE 's/.*idx=([0-9]+).*/\1/p'                       <<< "$line")
      lay=$(sed -nE 's/.*layers=([0-9]+).*/\1/p'                    <<< "$line")
      t=$(sed -nE  's/.*[[:space:]]time_s=([0-9.eE+-]+).*/\1/p'     <<< "$line")
      v=$(sed -nE  's/.*[[:space:]]gpu_peak_mib=([0-9.eE+-]+).*/\1/p' <<< "$line")
      h=$(sed -nE  's/.*[[:space:]]host_peak_mib=([0-9.eE+-]+).*/\1/p' <<< "$line")
      pz=$(sed -nE 's/.*prover_zkgpt_s=([0-9.eE+-]+).*/\1/p'        <<< "$line")
      cs=$(sed -nE 's/.*[[:space:]]circuit_s=([0-9.eE+-]+).*/\1/p'  <<< "$line")
      rs=$(sed -nE 's/.*[[:space:]]range_s=([0-9.eE+-]+).*/\1/p'    <<< "$line")
      echo "${cid},${nsplits_field},${comp_field},${C},${MK},${RK},${KK},${T},${STREAM},ACCEPT,${idx},${lay},${t},${v},${h},${pz},${cs},${rs}" >> "$out"
    done
    if [[ -n "$comp_field" ]]; then
      local bnd_out="${out}.boundaries.csv"
      [[ -f "$bnd_out" ]] || echo "config,n_parts,boundary_idx,cut_after_block,block_lo,block_hi,shared_commit_size,status" > "$bnd_out"
      local cum=0 bi=0
      local -a comp_arr; IFS=',' read -ra comp_arr <<< "$comp_field"
      local -a cuts=()
      local x
      for ((x=0; x<${#comp_arr[@]}-1; x++)); do cum=$((cum + comp_arr[x])); cuts+=("$cum"); done
      grep -oE 'Shared commitment size blocks [0-9]+-[0-9]+: [0-9]+' "$LOG" | while read -r bline; do
        local lo hi w cut
        lo=$(sed -nE 's/.*blocks ([0-9]+)-.*/\1/p' <<< "$bline")
        hi=$(sed -nE 's/.*-([0-9]+): .*/\1/p' <<< "$bline")
        w=$(sed -nE 's/.*: ([0-9]+)$/\1/p' <<< "$bline")
        cut="${cuts[$bi]:-}"
        echo "${comp_field},${#comp_arr[@]},${bi},${cut},${lo},${hi},${w},ACCEPT" >> "$bnd_out"
        bi=$((bi+1))
      done
    fi
  else
    echo "${cid},${nsplits_field},${comp_field},${C},${MK},${RK},${KK},${T},${STREAM},${STATUS},,,,,,,," >> "$out"
  fi
}

# Full (splits x c x mk x rk x k x t) grid for one device, appending
# per-partition rows to $1. Reads DEPTH/EMBED/LIN/... from the enclosing scope.
sweep_one() {  # <out_csv> <splits_csv>
  local out="$1" splits="$2"
  echo "$CSV_HEADER" > "$out"
  local -a A_SPLITS A_CS A_MKS A_RKS A_KS A_TS A_STREAM
  IFS=',' read -ra A_SPLITS <<< "$splits"
  IFS=',' read -ra A_CS     <<< "$CS"
  IFS=',' read -ra A_MKS    <<< "$MKS"
  IFS=',' read -ra A_RKS    <<< "$RKS"
  IFS=',' read -ra A_KS     <<< "$KS"
  IFS=',' read -ra A_TS     <<< "${TS:-__unset__}"
  IFS=',' read -ra A_STREAM <<< "${STREAM_BITS:-__unset__}"

  local P C MK RK KK T STREAM
  for P in "${A_SPLITS[@]}"; do
    (( P <= DEPTH )) || { echo "[sweep]   [skip] splits=$P > depth=$DEPTH" >&2; continue; }
    for C in "${A_CS[@]}"; do
      # No RANGE_MOD % C == 0 guard: chunk count is l = ceil(range/c), and the argument
      # proves x < 2^(c*ceil(range/c)), i.e. up to c-1 bits of slack.
      for MK in "${A_MKS[@]}"; do for RK in "${A_RKS[@]}"; do for KK in "${A_KS[@]}"; do
        for T in "${A_TS[@]}"; do for STREAM in "${A_STREAM[@]}"; do
          [[ "$T" == "__unset__" ]] && T=""
          [[ "$STREAM" == "__unset__" ]] && STREAM=""
          local CID="s${P}_c${C}_mk${MK}_rk${RK}_k${KK}"
          [[ -n "$T" ]] && CID="${CID}_t${T}"
          [[ -n "$STREAM" ]] && CID="${CID}_sb${STREAM}"
          run_point "$out" "$CID" "$P" "" "$P" -- "$C" "$MK" "$RK" "$KK" "$T" "$STREAM"
        done; done
      done; done; done
    done
  done
}

# Explicit uneven-composition sweep. Each composition
# is invoked with positional splits=1 and --splits <comp>, which main_gkr_split
# treats as authoritative (overrides + derives num_splits from the composition).
sweep_configs() {  # <out_csv> <configs_semicolon_string>
  local out="$1" configs="$2"
  echo "$CSV_HEADER" > "$out"
  local -a A_CS A_MKS A_RKS A_KS A_TS A_STREAM
  IFS=',' read -ra A_CS     <<< "$CS"
  IFS=',' read -ra A_MKS    <<< "$MKS"
  IFS=',' read -ra A_RKS    <<< "$RKS"
  IFS=',' read -ra A_KS     <<< "$KS"
  IFS=',' read -ra A_TS     <<< "${TS:-__unset__}"
  IFS=',' read -ra A_STREAM <<< "${STREAM_BITS:-__unset__}"

  local chunk comp C MK RK KK T STREAM
  IFS=';' read -ra A_CFG <<< "$configs"
  for chunk in "${A_CFG[@]}"; do
    chunk="$(echo "$chunk" | xargs)"; [[ -z "$chunk" ]] && continue
    local sum=0 bad=0 v
    IFS=',' read -ra comp_arr <<< "$chunk"
    for v in "${comp_arr[@]}"; do sum=$((sum + v)); (( v < 1 )) && bad=1; done
    if [[ $bad == 1 || $sum != "$DEPTH" ]]; then
      echo "[sweep]   [skip] invalid composition '$chunk' (must be >=1 and sum to depth=$DEPTH)" >&2
      continue
    fi
    comp="$chunk"
    for C in "${A_CS[@]}"; do for MK in "${A_MKS[@]}"; do for RK in "${A_RKS[@]}"; do for KK in "${A_KS[@]}"; do
      for T in "${A_TS[@]}"; do for STREAM in "${A_STREAM[@]}"; do
        [[ "$T" == "__unset__" ]] && T=""
        [[ "$STREAM" == "__unset__" ]] && STREAM=""
        local label; label="cfg$(tr ',' '-' <<< "$comp")"
        local CID="${label}_c${C}_mk${MK}_rk${RK}_k${KK}"
        [[ -n "$T" ]] && CID="${CID}_t${T}"
        [[ -n "$STREAM" ]] && CID="${CID}_sb${STREAM}"
        run_point "$out" "$CID" "${#comp_arr[@]}" "$comp" 1 --splits "$comp" -- "$C" "$MK" "$RK" "$KK" "$T" "$STREAM"
      done; done
    done; done; done; done
  done
}

# Coordinate descent over c, then lasso_k, then t (order = TUNE_STAGES), each
# stage minimizing wave_pack.py's wave_time_s at a FIXED splits value (the
# first of --splits), keeping the previous stage's winner.
tune() {
  local splits_first; IFS=',' read -ra _sp <<< "$SPLITS"; splits_first="${_sp[0]}"
  local cur_c cur_k cur_t
  IFS=',' read -ra _cs <<< "$CS"; cur_c="${_cs[0]}"
  IFS=',' read -ra _ks <<< "$KS"; cur_k="${_ks[0]}"
  IFS=',' read -ra _ts <<< "${TS:--1}"; cur_t="${_ts[0]}"
  echo "[sweep] --tune splits=$splits_first stages='$TUNE_STAGES' start c=$cur_c k=$cur_k t=$cur_t" >&2

  local scratch="$OUTDIR/tune_scratch_${TS_STAMP}.csv"
  local orig_cs="$CS" orig_ks="$KS" orig_ts="${TS:--1}"
  local stage
  for stage in $TUNE_STAGES; do
    # Pin all three to their current best, then restore ONLY this stage's full
    # grid, so this pass sweeps just one axis with the other two held fixed.
    CS="$cur_c"; KS="$cur_k"; TS="$cur_t"
    case "$stage" in
      c) CS="$orig_cs";;
      k) KS="$orig_ks";;
      t) TS="$orig_ts";;
    esac
    echo "[sweep]   tune stage=$stage grid: c={$CS} k={$KS} t={$TS}  (pinned others: c=$cur_c k=$cur_k t=$cur_t)" >&2
    sweep_one "$scratch" "$splits_first"
    local best
    best="$(python3 "$HERE/wave_pack.py" "$scratch" --vram-gb "$VRAM_GB" \
            ${HOST_GB:+--host-gb "$HOST_GB"} --best-only 2>/dev/null)" || true
    if [[ -z "$best" ]]; then
      echo "[sweep]   WARNING: no feasible point for stage=$stage; keeping current values" >&2
      continue
    fi
    local bc bk bt
    bc="$(grep '^c=' <<< "$best" | cut -d= -f2)"
    bk="$(grep '^lasso_k=' <<< "$best" | cut -d= -f2)"
    bt="$(grep '^tile_bits=' <<< "$best" | cut -d= -f2)"
    case "$stage" in
      c) cur_c="$bc"; echo "[sweep]   best c=$cur_c" ;;
      k) cur_k="$bk"; echo "[sweep]   best lasso_k=$cur_k" ;;
      t) cur_t="$bt"; echo "[sweep]   best t=$cur_t" ;;
    esac
  done
  echo "[sweep] --tune DONE splits=$splits_first -> c=$cur_c lasso_k=$cur_k t=$cur_t" >&2
  echo "c=$cur_c"
  echo "lasso_k=$cur_k"
  echo "tile_bits=$cur_t"
  rm -f "$scratch"
}

echo "[sweep] model=$MODEL depth=$DEPTH embed=$EMBED lin=$LIN len=$LEN | grid: splits={$SPLITS} mks={$MKS} rks={$RKS} ks={$KS} cs={$CS} ts={${TS:-inherit}} stream={${STREAM_BITS:-inherit}}" >&2

if [[ "$TUNE" == 1 ]]; then
  tune
elif [[ -n "$CONFIGS" ]]; then
  OUT="${OUT:-$OUTDIR/${MODEL}_configs_${TS_STAMP}.csv}"
  sweep_configs "$OUT" "$CONFIGS"
  echo "[sweep] done -> $OUT" >&2
  echo "[sweep] analyze: python3 $HERE/wave_pack.py $OUT --analyze" >&2
else
  IFS=',' read -ra A_GPUS <<< "${GPUS:-}"
  if [[ -n "${GPUS:-}" && ${#A_GPUS[@]} -gt 1 ]]; then
    # ---- multi-GPU fan-out: one worker per split value, round-robin over GPUs ----
    IFS=',' read -ra A_SPLITS <<< "$SPLITS"
    echo "[sweep] fanning ${#A_SPLITS[@]} split value(s) across ${#A_GPUS[@]} GPU(s): $GPUS" >&2
    pids=()
    for idx in "${!A_SPLITS[@]}"; do
      P="${A_SPLITS[$idx]}"; G="${A_GPUS[$((idx % ${#A_GPUS[@]}))]}"
      OUTW="$OUTDIR/${MODEL}_s${P}_${TS_STAMP}.csv"
      echo "[sweep] splits=$P -> HIP GPU $G -> $(basename "$OUTW")" >&2
      ( export HIP_VISIBLE_DEVICES="$G"; sweep_one "$OUTW" "$P" ) \
          > "$OUTDIR/launch_s${P}_${TS_STAMP}.log" 2>&1 &
      pids+=($!)
    done
    fail=0
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    MERGED="${OUT:-$OUTDIR/${MODEL}_all_${TS_STAMP}.csv}"; first=1
    for f in "$OUTDIR/${MODEL}"_s*_"${TS_STAMP}".csv; do
      [[ -f "$f" ]] || continue
      if [[ $first == 1 ]]; then head -1 "$f" > "$MERGED"; first=0; fi
      tail -n +2 "$f" >> "$MERGED"
    done
    echo "[sweep] done -> merged: $MERGED  (workers fail flag: $fail)" >&2
    echo "[sweep] pack:  python3 $HERE/wave_pack.py $MERGED --vram-gb 64" >&2
  else
    # ---- single-GPU sequential sweep ----
    [[ -n "${GPUS:-}" ]] && export HIP_VISIBLE_DEVICES="${A_GPUS[0]}"
    OUT="${OUT:-$OUTDIR/${MODEL}_${TS_STAMP}.csv}"
    sweep_one "$OUT" "$SPLITS"
    echo "[sweep] done -> $OUT" >&2
    echo "[sweep] pack:  python3 $HERE/wave_pack.py $OUT --vram-gb 64" >&2
  fi
fi
