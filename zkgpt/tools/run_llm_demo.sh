#!/usr/bin/env bash
# run_llm_demo.sh — device-vs-UVM baseline profiling sweep over (L, c, k) with
# per-section AND per-phase breakdowns.
#
# Sweeps four axes: binary  x  chunk-bits c  x  Blendy stages k  x  depth L, and
# for each combination runs the full proof and records a timing + memory
# breakdown. Two binaries are compared by default:
#
#   device : bin/zkgpt_gpu      — hipMalloc working buffers; hard VRAM ceiling
#            (OOMs when a layer's working set exceeds physical VRAM). The baseline.
#   uvm    : bin/zkgpt_gpu_uvm  — managed (UVM) working buffers; oversubscribes
#            physical VRAM by migrating pages to/from host memory.
#
# Both binaries build from the same main_gkr.cpp, so their PHASE / MATMUL /
# RANGE markers line up 1:1. Per (binary, c, k, L) we capture, PHASE by PHASE:
#   circuit_build : nn.create()  (build circuit + commit input; host-bound)
#   verify_gkr    : verifier.prove() (verifyGKR + verifyLasso + openCommit)
#   range         : range_prover build()/prove() (GPU logup range check)
# with, for each PHASE: wall time, host RSS peak (VmRSS/VmHWM), and GPU-memory
# peak (this script polls the device and matches samples to the phase window).
#
# It ALSO records per-component timings for the WHOLE proof:
#   matmul sumcheck phases : expand / ptable / round / fold  (the four phases of
#            the Blendy staged product-sumcheck; each spans many kernel launches
#            over all stages and matmul layers)
#   deg3 sumcheck  : total GPU time
#   range prover phases : setup / invert / commit / deg3 / deg1 / open
#
# NOTE on c/k: chunk-bits c and Blendy stages k are RANGE-PROVER knobs
# (range_c / range_k). They have NO effect on circuit_build or verify_gkr, so a
# c/k sweep only makes sense with the range prover ON (do not pass --skip-range).
# Any c is legal (chunk count = ceil(range_size/c)); e.g.
# --range-size 48 --cs 4,6,8,12,16,24.
#
# Build first:
#   cmake --build build-cuda --target zkgpt_gpu zkgpt_gpu_uvm -j
#
# Usage:
#   tools/run_llm_demo.sh [OPTIONS]
#
# Options / env:
#   --binaries device,uvm  which binaries to run   (env BINARIES; default device,uvm)
#   --uvm / --device-only  shorthands: device-vs-UVM (default) / skip the UVM arm
#   --depths L1,L2,...   depths to sweep            (env DEPTHS; default 1,2,4,6,8,12)
#   --cs C1,C2,...       chunk-bits c to sweep      (env CS; default 7)
#   --ks K1,K2,...       Blendy stages k to sweep   (env KS; default 1)
#   --c N / --k N        single-value shorthand for --cs / --ks
#   --range-size N       range bit-width            (env RANGE_SIZE; default: binary's 49)
#   --model NAME|"H d attn lin"                     (env MODEL; default gpt2 width)
#                        presets: gpt2 llama8b llama405b  (width only; L is swept)
#   --skip-range         GKR+Lasso+commit only (c/k become no-ops)
#   --gpu N              CUDA device index           (env GPU; default 0)
#   --vram-gb N          VRAM for range auto-partition (env VRAM_GB; default 64)
#   --timeout N          per-run seconds, 0=none     (env TIMEOUT; default 1800)
#   --poll N             GPU poll interval seconds   (env POLL; default 0.2)
#   --out FILE           CSV output path             (env OUT)
#
# Examples:
#   tools/run_llm_demo.sh                                          # L sweep, device vs uvm
#   tools/run_llm_demo.sh --range-size 48 --cs 4,6,8,12,16,24 --ks 1,2,4 --depths 1,2,4
#   tools/run_llm_demo.sh --binaries uvm --cs 7,49 --ks 1,2 --depths 2

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── defaults ─────────────────────────────────────────────────────────────────
BINARIES="${BINARIES:-device,uvm}"
DEPTHS="${DEPTHS:-1,2,4,6,8,12}"
CS="${CS:-7}"
KS="${KS:-1}"
RANGE_SIZE="${RANGE_SIZE:-}"
MODEL="${MODEL:-gpt2}"
SKIP_RANGE=0
ONLY_RANGE=0
PARTITIONS="${PARTITIONS:-}"     # range-prover partition count (--partitions); "" = auto
MATMUL_K="${MATMUL_K:-}"         # GKR matmul Blendy stages (--matmul-k); "" = off (unstaged 3^n)
TILE_BITS="${TILE_BITS:-}"       # ext-table tile width t (GKR_SC_TILE_BITS); "" = binary default
GPU="${GPU:-0}"
VRAM_GB="${VRAM_GB:-64}"
TIMEOUT="${TIMEOUT:-1800}"
POLL="${POLL:-0.2}"
OUT="${OUT:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --binaries)   BINARIES="$2";   shift 2 ;;
    --uvm)         BINARIES="device,uvm"; shift ;;   # device-vs-UVM (the default); explicit alias
    --device-only) BINARIES="device";     shift ;;   # skip the UVM arm
    --depths)     DEPTHS="$2";     shift 2 ;;
    --cs)         CS="$2";         shift 2 ;;
    --ks)         KS="$2";         shift 2 ;;
    --c)          CS="$2";         shift 2 ;;
    --k)          KS="$2";         shift 2 ;;
    --range-size) RANGE_SIZE="$2"; shift 2 ;;
    --model)      MODEL="$2";      shift 2 ;;
    --skip-range) SKIP_RANGE=1;    shift   ;;
    --only-range) ONLY_RANGE=1;    shift   ;;
    --partitions) PARTITIONS="$2"; shift 2 ;;
    --matmul-k)   MATMUL_K="$2";   shift 2 ;;
    --tile-bits)  TILE_BITS="$2";  shift 2 ;;
    --gpu)        GPU="$2";        shift 2 ;;
    --vram-gb)    VRAM_GB="$2";    shift 2 ;;
    --timeout)    TIMEOUT="$2";    shift 2 ;;
    --poll)       POLL="$2";       shift 2 ;;
    --out)        OUT="$2";        shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

case "$MODEL" in
  gpt2)      WIDTH="12 64 768 3072" ;;
  llama8b)   WIDTH="32 128 4096 14336" ;;
  llama405b) WIDTH="128 128 16384 53248" ;;
  *)         WIDTH="$MODEL" ;;   # raw "H d attn lin"
esac
read -r HEADS HDIM ATTN LIN <<< "$WIDTH"

bin_path() {
  case "$1" in
    device) echo "$HERE/../bin/zkgpt_gpu" ;;
    uvm)    echo "$HERE/../bin/zkgpt_gpu_uvm" ;;
    *)      echo "" ;;
  esac
}

IFS=',' read -ra BINS <<< "$BINARIES"
IFS=',' read -ra LS   <<< "$DEPTHS"
IFS=',' read -ra CLIST <<< "$CS"
IFS=',' read -ra KLIST <<< "$KS"

for b in "${BINS[@]}"; do
  p="$(bin_path "$b")"
  [[ -n "$p" ]] || { echo "unknown binary tag: $b (use device|uvm)" >&2; exit 1; }
  [[ -x "$p" ]] || { echo "missing $p — build: cmake --build build-cuda --target zkgpt_gpu zkgpt_gpu_uvm -j" >&2; exit 1; }
done

# c/k only affect the range prover.
EFF_RS="${RANGE_SIZE:-49}"   # binary default range_size when unset
if [[ $SKIP_RANGE -eq 1 && ( ${#CLIST[@]} -gt 1 || ${#KLIST[@]} -gt 1 ) ]]; then
  echo "[demo] WARNING: --skip-range with a c/k sweep — c/k are range-prover" >&2
  echo "[demo]          knobs, so every (c,k) run will be identical here." >&2
fi

# GPU-memory backend: nvidia-smi poll (CUDA) > rocprofv3 alloc-trace (ROCm) > none.
# nvidia-smi gives per-phase peaks via time-series polling; rocprofv3 gives only a
# whole-run peak (an allocation trace can't be bucketed into the binary's PHASE
# windows), so on the ROCm path only the overall gpu_peak_mib column is filled.
SMI_MODE=none; HAVE_SMI=0
if command -v nvidia-smi >/dev/null 2>&1; then
  SMI_MODE=nvidia; HAVE_SMI=1
  echo "[demo] GPU-memory backend: nvidia-smi (per-phase polling)" >&2
elif command -v rocprofv3 >/dev/null 2>&1; then
  SMI_MODE=rocprof; HAVE_SMI=1
  echo "[demo] GPU-memory backend: rocprofv3 alloc-trace (whole-run peak only)" >&2
else
  echo "[demo] no nvidia-smi or rocprofv3 — GPU-memory columns will be empty" >&2
fi

RES="$HERE/results/uvm"; mkdir -p "$RES/data"
TS="$(date +%Y%m%d_%H%M%S)"
MODE_TAG="full"; [[ $SKIP_RANGE -eq 1 ]] && MODE_TAG="no_range"; [[ $ONLY_RANGE -eq 1 ]] && MODE_TAG="only_range"
[[ -z "$OUT" ]] && OUT="$RES/data/uvm_profile_${MODE_TAG}_${TS}.csv"

echo "[demo] binaries=$BINARIES  model=$MODEL width='$WIDTH' mode=$MODE_TAG"
echo "[demo] depths=$DEPTHS  cs=$CS  ks=$KS  range_size=${RANGE_SIZE:-def($EFF_RS)}"
echo "[demo] gpu=$GPU vram_gb=$VRAM_GB timeout=${TIMEOUT}s -> $OUT"

echo "model,binary,dims,L,c,k,mode,status,wall_s,\
circuit_build_s,circuit_build_host_peak_mib,circuit_build_gpu_peak_mib,\
verify_gkr_s,verify_gkr_host_peak_mib,verify_gkr_gpu_peak_mib,\
range_s,range_host_peak_mib,range_gpu_peak_mib,\
mm_expand_s,mm_ptable_s,mm_round_s,mm_fold_s,mm_deg3_s,\
matmul_prover_s,total_prover_s,verifier_s,proof_kb,\
range_result_s,range_partitions,rc_setup_s,rc_invert_s,rc_commit_s,rc_deg3_s,rc_deg1_s,rc_open_s,\
host_peak_mib,gpu_peak_mib,rc,\
prover_core_s,range_prove_s,prover_total_s,prover_residual_s,\
ph_gkr_generic_s,ph_gkr_matmul_s,ph_lasso_s,ph_hyrax_commit_s,ph_hyrax_open_s" > "$OUT"

run_point() {  # <binary_tag> <c> <k> <L>
  local btag="$1" c="$2" k="$3" L="$4"
  local UVM; UVM="$(bin_path "$btag")"
  local DIMS="$L $HEADS $HDIM $ATTN $LIN"
  # positional: L H d attn lin c k [range_size]
  local args=($DIMS "$c" "$k")
  [[ -n "$RANGE_SIZE" ]] && args+=("$RANGE_SIZE")
  args+=(--vram-gb "$VRAM_GB")
  [[ $SKIP_RANGE -eq 1 ]] && args+=(--skip-range)
  # device-side experiment knobs: run only the range prover (the full L=12
  # pipeline OOMs on device) and split the logup into P sequential sub-proofs.
  [[ $ONLY_RANGE -eq 1 ]] && args+=(--only-range)
  [[ -n "$PARTITIONS" ]] && args+=(--partitions "$PARTITIONS")
  # stage the GKR matmul too, else the full L=12 pipeline builds an unstaged 3^n
  # ptable (n~18-22) on top of resident weights and OOMs on device.
  [[ -n "$MATMUL_K" ]] && args+=(--matmul-k "$MATMUL_K")
  # Extension-table tile width t (GKR_SC_TILE_BITS; -1 = untiled). Post-tiling
  # this -- not k -- is the memory knob: peak is 2*3^l*2^min(n-l,t) + 3N + 2*3^l.
  [[ -n "$TILE_BITS" ]] && export GKR_SC_TILE_BITS="$TILE_BITS"

  local tag="${btag}.c${c}.k${k}.L${L}"
  local log="$RES/data/$(basename "${OUT%.csv}").${tag}.log"
  local vram="$RES/data/$(basename "${OUT%.csv}").${tag}.vram"
  : > "$vram"

  # rocprofv3 --memory-allocation-trace is used ONLY for the device binary:
  # it cannot see hipMallocManaged (so it yields no useful UVM gpu_peak) AND it
  # segfaults when wrapping the managed UVM binary — the crash kills the run
  # mid-range, dropping the RANGE_BREAKDOWN (open_s) line. So UVM always runs
  # clean (time only; gpu_peak_mib stays empty for uvm rows).
  local vrampid="" rocdir="" gpu_peak_mib="" use_rocprof=0
  if [[ "$SMI_MODE" == nvidia ]]; then
    ( while true; do
        m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | head -1)
        printf '%s,%s\n' "$(date +%s.%N)" "${m:-}" >> "$vram"
        sleep "$POLL"
      done ) &
    vrampid=$!
  elif [[ "$SMI_MODE" == rocprof && "$btag" != uvm ]]; then
    use_rocprof=1
    rocdir="$RES/data/$(basename "${OUT%.csv}").${tag}.rocprof"
    rm -rf "$rocdir"; mkdir -p "$rocdir"
  fi

  local to=(); [[ "$TIMEOUT" != 0 ]] && to=(timeout "$TIMEOUT")
  local t0 t1 wall rc
  # ROCm/HIP: pin with HIP_VISIBLE_DEVICES (CUDA_VISIBLE_DEVICES is a no-op on HIP).
  t0=$(date +%s.%N)
  if [[ $use_rocprof -eq 1 ]]; then
    # device only: rocprofv3 traces every hipMalloc; peak live bytes -> whole-run
    # gpu_peak_mib (per-phase columns stay empty).
    HIP_VISIBLE_DEVICES="$GPU" CUDA_VISIBLE_DEVICES="$GPU" "${to[@]}" \
      rocprofv3 --memory-allocation-trace --output-format csv -d "$rocdir" -o run \
      -- "$UVM" "${args[@]}" > "$log" 2>&1
    rc=$?
  else
    HIP_VISIBLE_DEVICES="$GPU" CUDA_VISIBLE_DEVICES="$GPU" "${to[@]}" "$UVM" "${args[@]}" > "$log" 2>&1
    rc=$?
  fi
  t1=$(date +%s.%N)
  wall=$(awk "BEGIN{printf \"%.2f\", $t1-$t0}")

  [[ -n "$vrampid" ]] && { kill "$vrampid" 2>/dev/null; wait "$vrampid" 2>/dev/null; }
  if [[ $use_rocprof -eq 1 && -n "$rocdir" ]]; then
    gpu_peak_mib=$(python3 "$HERE/.rocprof_peak.py" "$rocdir" -o run 2>/dev/null)
    rm -rf "$rocdir"
  fi

  # Status. NB: a device-side GPU OOM does NOT abort — HIP_CHECK only logs and
  # continues, so hipMalloc failure leaves a null/garbage device buffer, later
  # kernels fault ("illegal memory access"), and the verifier rejects the
  # resulting garbage. That surfaces as "RESULT: REJECT" but is really OOM, not a
  # soundness failure. So OOM markers take precedence over REJECT.
  local status
  if   grep -q "RESULT: ACCEPT" "$log"; then status=ACCEPT
  elif grep -qiE "HIP error 2: out of memory|out of memory at|illegal memory access" "$log"; then status=GPU_OOM
  elif grep -q "RESULT: REJECT" "$log"; then status=REJECT
  elif grep -qi "Assertion .* failed" "$log"; then status=ASSERT
  elif [[ $rc == 124 || $rc == 143 ]];  then status=TIMEOUT
  elif [[ $rc == 137 ]];                then status=OOM_KILLED
  else status="ERR($rc)"; fi

  local row
  row=$(MODEL="$MODEL" BIN="$btag" DIMS="$DIMS" L="$L" C="$c" K="$k" \
        MODE="$MODE_TAG" STATUS="$status" WALL="$wall" RC="$rc" \
        LOG="$log" VRAM="$vram" GPU_PEAK_MIB="$gpu_peak_mib" \
        python3 "$HERE/.run_llm_demo_parse.py")
  echo "$row" >> "$OUT"

  echo "$row" | awk -F, '{
    printf "  %-6s c=%-3s k=%-3s L=%-3s %-11s wall=%-7ss | gkr=%-6ss range=%-6ss | mm(exp/ptab/rnd/fold)=%s/%s/%s/%s | rc_open=%s | gpu_peak=%sMiB\n",
      $2,$5,$6,$4,$8,$9,$13,$16,$19,$20,$21,$22,$35,$37 }'

  [[ "$status" == ACCEPT ]] && rm -f "$log"
  rm -f "$vram"
}

emit_skip() {  # <binary_tag> <c> <k> <L> <reason>
  local btag="$1" c="$2" k="$3" L="$4" reason="$5"
  local DIMS="$L $HEADS $HDIM $ATTN $LIN"
  echo "$MODEL,$btag,\"$DIMS\",$L,$c,$k,$MODE_TAG,SKIP($reason),,,,,,,,,,,,,,,,,,,,,,,,,,,," >> "$OUT"
  printf "  %-6s c=%-3s k=%-3s L=%-3s SKIP (%s)\n" "$btag" "$c" "$k" "$L" "$reason"
}

for btag in "${BINS[@]}"; do
  echo "==== binary: $btag ===="
  for c in "${CLIST[@]}"; do
    for k in "${KLIST[@]}"; do
      # No range_size % c == 0 guard: the chunk count is l = ceil(range/c), so any c is
      # legal; the argument proves x < 2^(c*ceil(range/c)), i.e. up to c-1 bits of
      # slack. A divisibility guard would silently skip the optimum c*=13-14.
      for L in "${LS[@]}"; do
        run_point "$btag" "$c" "$k" "$L"
      done
    done
  done
done

echo "[demo] DONE -> $OUT"
echo "[demo] plot (L on x):  python3 $HERE/plot_uvm.py profile $OUT"
echo "[demo] plot (c on x):  python3 $HERE/plot_uvm.py profile $OUT --x c"
echo "[demo] plot (k on x):  python3 $HERE/plot_uvm.py profile $OUT --x k"
