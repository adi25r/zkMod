#!/usr/bin/env bash
# run_overlay_experiment.sh — the three runs tools/overlay_concurrent.py needs, into one
# output directory, in the only order that makes them comparable.
#
#   A  concurrent, unprofiled   -> the timeline to explain (PHASE lines + device VRAM)
#   B  sequential, unprofiled   -> uncontended per-phase DURATIONS
#   C  sequential, profiled     -> per-phase kernel + memory SIZES
#
# Why three and not two:
#   * rocprofv3 cannot see A at all -- it does not follow fork-without-exec and the
#     children _exit() without flushing. This is a hard limit of rocprofiler-sdk 0.6.0
#     (no --attach), not a configuration mistake. See tools/run_profiled.sh.
#   * The profiler inflates wall time, so C's durations are not usable. Durations come
#     from B, sizes from C. Mixing them up is the one error that silently poisons the
#     dilation table.
#
# THREAD COUNTS ARE NOT A DETAIL. A's `--concurrent W` splits the host budget across
# children (tpc = threads/W, main_gkr_split.cpp), so B and C must run at that same
# per-child count -- not at the full budget -- or the measured "dilation" is really just
# a thread-count change wearing a contention costume. This script derives it for you.
# Witgen is exempt (it is thread-invariant in every arm), which is what makes a single
# 16-thread sequential run valid for both halves; overlay_concurrent.py re-checks that
# by comparing A's parent-side witgen against B's.
#
# Usage:
#   tools/run_overlay_experiment.sh -g 7
#   tools/run_overlay_experiment.sh -g 7 -w 7 -- 12 768 12 64 192 3072
#   tools/run_overlay_experiment.sh -g 7 --skip A       # reuse an earlier A
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

GPU="${GPU:-7}"
MODE="${MODE:-concurrent}"     # concurrent | overlapped | overlapped-nobarrier
WIDTH=""                       # --concurrent width; default = number of partitions
OUT=""
SKIP=""
TIMEOUT="${TIMEOUT:-0}"        # 0 = uncapped; a truncated run yields a bogus makespan

while [[ $# -gt 0 ]]; do
  case "$1" in
    -g) GPU="$2"; shift 2 ;;
    -m) MODE="$2"; shift 2 ;;
    -w) WIDTH="$2"; shift 2 ;;
    -o) OUT="$2"; shift 2 ;;
    --skip) SKIP="$SKIP $2"; shift 2 ;;
    --) shift; break ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# depth embed splits len threads lin  -- GPT-2, 12 partitions, len 64.
ARGS=("$@")
[[ ${#ARGS[@]} -eq 0 ]] && ARGS=(12 768 12 64 192 3072)
DEPTH="${ARGS[0]}"; EMBED="${ARGS[1]}"; SPLITS="${ARGS[2]}"
LEN="${ARGS[3]}";   THREADS="${ARGS[4]}"; LIN="${ARGS[5]}"
EXTRA=("${ARGS[@]:6}")

[[ -z "$WIDTH" ]] && WIDTH="$SPLITS"
TPC=$(( THREADS / WIDTH )); (( TPC < 1 )) && TPC=1

# THE BINARY'S KNOB DEFAULTS ARE CPU-ARM DEFAULTS AND ABORT A GPU-PROVING RUN.
# At range_k=1 / lasso_k=-1 the range prover's deg3 stage is one un-split block, and
# sizing its extension tables throws before any proving happens:
#   "deg3 extension tables need 98304.000000 GiB, device has 63.984375 GiB
#    (l=20 k=1 l_stage=20 tile_bits=-1)"
# It is harmless on the CPU arms (they never allocate those tables), which is why the
# defaults survive in the repo. Every recorded *gpuprove* arm in
# tools/results/gpt2_ablation uses exactly the set below -- c=18 is also the measured
# GPU optimum (project_c_knob_mistuned_both_arms, project_commit_fr_gathered_msm).
# Raising k shrinks l_stage = ceil(l/k), so 3^l_stage stops being astronomical; no
# --sc-tile-bits is needed once k is sane, and streaming is already on by default.
KNOBS=(--c 18 --matmul-k 4 --range-k 8 --lasso-k 8 --sc-tile-bits 18)
if [[ ! " ${EXTRA[*]:-} " =~ (--c|--matmul-k|--range-k|--lasso-k)[[:space:]] ]]; then
  EXTRA+=("${KNOBS[@]}")
  echo "[overlay] using default GPU knobs: ${KNOBS[*]}" >&2
fi

OUT="${OUT:-tools/results/overlay/$(date +%Y%m%d_%H%M%S)}"
skipped() { [[ " $SKIP " == *" $1 "* ]]; }

# ── preflight ─────────────────────────────────────────────────────────────────
# A wedged KFD serialises every new HIP process box-wide, on any GPU. Without this the
# runs simply hang, in D state, with no error and no output. Cheap to check, expensive to discover late.
if ps -eo stat,wchan:24 | awk '$2=="kfd_create_process"' | grep -q .; then
  echo "[overlay] ABORT: the KFD driver is wedged. These are stuck in kfd_create_process" >&2
  echo "          and they block every new HIP process on this box:" >&2
  ps -eo pid,stat,etime,wchan:24,cmd | awk '$4=="kfd_create_process"' | sed 's/^/    /' >&2
  exit 3
fi
# Other tenants share this box; a neighbour on our card changes the measured makespan
# with no error anywhere (feedback_shared_box_contaminates_sweeps).
#
# -g IS A HIP INDEX, AND rocm-smi USES A DIFFERENT NAMESPACE. Reading `rocm-smi GPU[$GPU]`
# and then passing $GPU to HIP_VISIBLE_DEVICES checks a DIFFERENT CARD than it runs on.
# The only identifier stable across the two is the PCI bus id, so resolve through that:
# hipDeviceGetPCIBusId for the HIP side, /sys/class/drm for the free-memory side (which
# is also how src/gpu_sampler.hpp keys its rows). The map is cached because building the
# probe needs a working HIP, and the check must still say something honest when HIP is
# wedged and the probe cannot run.
PCI_MAP="tools/.hip_pci_map"
if [[ ! -s "$PCI_MAP" ]]; then
  probe_src=$(mktemp --suffix=.cpp)
  cat > "$probe_src" <<'PROBE'
#include <hip/hip_runtime.h>
#include <cstdio>
int main() {
    int n = 0;
    if (hipGetDeviceCount(&n) != hipSuccess) return 1;
    for (int i = 0; i < n; ++i) {
        char bus[64] = {0};
        if (hipDeviceGetPCIBusId(bus, sizeof(bus), i) == hipSuccess)
            printf("%d %s\n", i, bus);
    }
    return 0;
}
PROBE
  if hipcc -O0 -o "${probe_src%.cpp}" "$probe_src" 2>/dev/null; then
    timeout 60 "${probe_src%.cpp}" > "$PCI_MAP" 2>/dev/null || : > "$PCI_MAP"
  fi
  rm -f "$probe_src" "${probe_src%.cpp}"
fi

BUS=$(awk -v g="$GPU" '$1==g {print tolower($2)}' "$PCI_MAP" 2>/dev/null | head -1)
if [[ -z "$BUS" ]]; then
  echo "[overlay] NOTE: cannot resolve HIP device $GPU to a PCI bus id ($PCI_MAP is" >&2
  echo "          empty or missing, and the probe needs a working HIP). Skipping the" >&2
  echo "          free-VRAM check rather than reporting another card's number --" >&2
  echo "          rocm-smi's GPU indices are NOT HIP's. Free memory, by rocm-smi index:" >&2
  rocm-smi --showmeminfo vram 2>/dev/null | grep Used | sed 's/^/            /' >&2
else
  # Strip the domain (0000:) that HIP prints but /sys/class/drm's link does not.
  SHORT="${BUS#0000:}"
  FREE_MIB=""
  for d in /sys/class/drm/card*/device; do
    tgt=$(readlink -f "$d" 2>/dev/null) || continue
    case "$tgt" in *"$SHORT") ;; *) continue ;; esac
    tot=$(cat "$d/mem_info_vram_total" 2>/dev/null || echo 0)
    use=$(cat "$d/mem_info_vram_used"  2>/dev/null || echo 0)
    [[ "$tot" -gt 0 ]] && FREE_MIB=$(( (tot - use) / 1048576 ))
    break
  done
  echo "[overlay] gpu=HIP$GPU pci=$BUS free=${FREE_MIB:-?} MiB  host_avail=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo) GiB"
  if [[ -n "$FREE_MIB" && "$FREE_MIB" -lt 55000 ]]; then
    echo "[overlay] WARNING: only ${FREE_MIB} MiB free on HIP device $GPU ($BUS) -- a" >&2
    echo "          co-tenant will contaminate both the makespan and the VRAM trace." >&2
  fi
fi

# GKR_WITGEN_GPU would make the three runs incomparable in a way nothing reports:
# --concurrent returns in main_gkr_split.cpp BEFORE the assignment that honours
# it, because the parent must reach fork() without a HIP context. So A would build on
# the CPU while B and C built on the GPU, and the witgen dilation would be measuring
# that, not contention.
if [[ -n "${GKR_WITGEN_GPU:-}" && "$MODE" == "concurrent" ]]; then
  echo "[overlay] ABORT: GKR_WITGEN_GPU with -m concurrent. The concurrent arm silently" >&2
  echo "          ignores it (parent must fork HIP-free at main_gkr_split.cpp:1683)" >&2
  echo "          while the sequential arms honour it, so A would build on CPU and" >&2
  echo "          B/C on GPU. Use -m overlapped for GPU witgen." >&2
  exit 2
fi
# Same for the witness cache: a hit returns from create() before any witness work, so
# a warm cache makes witgen durations meaningless (the witness cache is live).
export ZKGPT_WITNESS_CACHE=0

# SOURCE STREAMING IS MANDATORY, NOT A TUNING CHOICE. Unset, gpu_matmul_prover.hpp
# takes "the dense path, bit-for-bit": the dF/dG source is 2*2^n*32B = 12.4 GiB per
# partition and the untiled ext table another ~43.5 GB -- about a 56.4 GiB
# peak for a SINGLE partition against a 64 GiB card. NARROWING THE WAVE DOES NOT FIX IT:
# tools/results/gpt2_ablation/README.md records OOM at W=12 *and at W=4*, because every
# child stays alive holding its context. With streaming on the peak is 4.7 GiB/partition
# and 12 fit. --sc-tile-bits 18 in KNOBS is the separate, also-required knob (ext table
# 43.5 GB at t=-1 vs 1.36 GB at t=18).
export GKR_SC_STREAM_BITS="${GKR_SC_STREAM_BITS:-22}"
export GKR_BFEVAL_GPU="${GKR_BFEVAL_GPU:-1}"
echo "[overlay] GKR_SC_STREAM_BITS=$GKR_SC_STREAM_BITS GKR_BFEVAL_GPU=$GKR_BFEVAL_GPU" >&2

for b in bin/main_gkr_split bin/main_gkr_split_roctx; do
  [[ -x "$b" ]] || { echo "[overlay] missing $b -- cmake --build build --target $(basename "$b") -j" >&2; exit 1; }
done

TO=(); [[ "$TIMEOUT" != 0 ]] && TO=(timeout "$TIMEOUT")

# GKR_PIN_CORES: A's children pin THEMSELVES, each to a disjoint block of TPC
# physical cores (main_gkr_split.cpp::pin_child_cores, called in the child right
# after fork). B and C are single processes at that same per-child budget, so they
# get one block here -- otherwise A's 16 cores would be compared against B's 384
# and every "dilation" would carry a resource change inside it.
PIN=()
if [[ -n "${GKR_PIN_CORES:-}" ]]; then
  PIN=(taskset -c "0-$((TPC-1))")
  echo "[overlay] pinning ON: A's children -> disjoint ${TPC}-core blocks; B and C -> cpus 0-$((TPC-1))" >&2
fi

# The sampler runs at 1 ms, where the sysfs reads are the whole budget (~20 us
# each, x4 attributes x8 cards). Tell it which card is under test so it polls that one
# every tick and sweeps the others at 100 ms; without this all three runs still work,
# they just resample seven idle cards a thousand times a second.
export ZKGPT_SAMPLER_PCI="${BUS:-}"

# Only now that every precondition has passed -- an aborted run should leave no trace.
mkdir -p "$OUT"

cat > "$OUT/config.txt" <<EOF
depth=$DEPTH embed=$EMBED splits=$SPLITS len=$LEN lin=$LIN
threads_total=$THREADS concurrent_width=$WIDTH threads_per_child=$TPC
mode=$MODE witgen_gpu=${GKR_WITGEN_GPU:-0} round_gpu=${GKR_ROUND_GPU:-0}
gates_gpu=${GKR_GATES_GPU:-0} witgen_full=${GKR_WITGEN_FULL:-0} pin_cores=${GKR_PIN_CORES:-0}
pci=${BUS:-unknown}
extra=${EXTRA[*]:-}
gpu=$GPU host=$(hostname) date=$(date -Is)
rocm=$(cat /opt/rocm/.info/version 2>/dev/null)
EOF
echo "[overlay] out=$OUT"; sed 's/^/          /' "$OUT/config.txt"

# ── A: concurrent, unprofiled ─────────────────────────────────────────────────
if ! skipped A; then
  A_MODE=(--concurrent "$WIDTH")
  [[ "$MODE" == "overlapped" ]] && A_MODE=(--overlapped "$WIDTH" --witgen-barrier)
  [[ "$MODE" == "overlapped-nobarrier" ]] && A_MODE=(--overlapped "$WIDTH")
  echo "[overlay] A/3: $MODE width=$WIDTH (tpc=$TPC) witgen_gpu=${GKR_WITGEN_GPU:-0}"
  HIP_VISIBLE_DEVICES="$GPU" "${TO[@]}" ./bin/main_gkr_split \
    "$DEPTH" "$EMBED" "$SPLITS" "$LEN" "$THREADS" "$LIN" "${EXTRA[@]}" \
    "${A_MODE[@]}" --vram-csv "$OUT/A_concurrent_vram.csv" \
    > "$OUT/A_concurrent.log" 2>&1
  rc=$?
  echo "[overlay]   rc=$rc phases=$(grep -c '^PHASE' "$OUT/A_concurrent.log")"
  if [[ $rc -ne 0 ]]; then
    echo "[overlay] ABORT: run A failed -- B and C only have meaning folded onto A's" >&2
    echo "          timeline, so there is nothing to gain by running them. Cause:" >&2
    grep -m3 -E "out of memory|deg3 extension|terminate called|REJECT" "$OUT/A_concurrent.log" \
      | sed 's/^/            /' >&2
    exit 1
  fi
fi

# ── B: sequential, unprofiled, AT THE PER-CHILD THREAD COUNT ──────────────────
if ! skipped B; then
  echo "[overlay] B/3: sequential, threads=$TPC (matching A's per-child budget)"
  HIP_VISIBLE_DEVICES="$GPU" "${PIN[@]}" "${TO[@]}" ./bin/main_gkr_split \
    "$DEPTH" "$EMBED" "$SPLITS" "$LEN" "$TPC" "$LIN" "${EXTRA[@]}" \
    --vram-csv "$OUT/B_sequential_vram.csv" \
    > "$OUT/B_sequential.log" 2>&1
  rc=$?
  echo "[overlay]   rc=$rc phases=$(grep -c '^PHASE' "$OUT/B_sequential.log")"
  [[ $rc -ne 0 ]] && { echo "[overlay] ABORT: run B failed (no uncontended baseline)" >&2; exit 1; }
fi

# ── C: sequential, profiled ───────────────────────────────────────────────────
# Traces only, no --pmc: counter collection serialises kernel dispatches, which would
# distort the very kernel_busy fractions this pass is collected for.
if ! skipped C; then
  echo "[overlay] C/3: sequential under rocprofv3 (kernel + marker + memory)"
  HIP_VISIBLE_DEVICES="$GPU" "${PIN[@]}" "${TO[@]}" rocprofv3 \
    --kernel-trace --marker-trace \
    --memory-allocation-trace --memory-copy-trace --scratch-memory-trace \
    --stats --output-format csv \
    -d "$OUT/C_profile" -o run -- ./bin/main_gkr_split_roctx \
    "$DEPTH" "$EMBED" "$SPLITS" "$LEN" "$TPC" "$LIN" "${EXTRA[@]}" \
    > "$OUT/C_profile.log" 2>&1
  echo "[overlay]   rc=$? csv=$(find "$OUT/C_profile" -name '*.csv' 2>/dev/null | wc -l)"
fi

# ── merge ─────────────────────────────────────────────────────────────────────
echo "[overlay] merging"
python3 tools/overlay_concurrent.py --run-dir "$OUT" | tee "$OUT/summary.txt"

echo
echo "[overlay] self-test (B against itself: every dilation must be 1.00, overlap 1)"
python3 tools/overlay_concurrent.py \
  --concurrent-log "$OUT/B_sequential.log" \
  --concurrent-vram "$OUT/B_sequential_vram.csv" \
  --sequential-log "$OUT/B_sequential.log" \
  --profile-dir "$OUT/C_profile" \
  -o "$OUT/selftest" > "$OUT/selftest.txt" 2>&1
grep -E "peak concurrent|dilation|VRAM:" "$OUT/selftest.txt" | head -5
# The 1 ms sampler writes ~90 MB of CSV per arm. Gzip rather than downsample: the
# resolution is the point, and the readers accept either form.
gzip -f "$OUT"/*_cpu.csv "$OUT"/*_vram.csv 2>/dev/null
echo "[overlay] sampler CSVs gzipped: $(du -sh "$OUT" | cut -f1) total"

echo "[overlay] done: $OUT"
