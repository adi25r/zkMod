#!/usr/bin/env bash
# run_profiled.sh — per-phase ALU / occupancy / kernel data via rocprofv3.
#
# SEQUENTIAL RUNS ONLY, and that is a hard constraint, not a default: rocprofv3 does
# not follow fork() without exec(), and --concurrent proves each partition in a forked
# child. Wrapping the concurrent parent was measured to yield only agent_info.csv --
# the parent is deliberately HIP-free, so the profiler sees zero kernels. There is no
# --attach in this ROCm build (rocprofiler-sdk 0.6.0 / ROCm 6.4.3) to work around it.
#
# For the concurrent path use the built-in device-wide sampler instead:
#   ./bin/main_gkr_split ... --concurrent 10 --vram-csv out.csv
# which records VRAM + gpu_busy_percent + mem_busy_percent for every card.
#
# Kernel ALU behaviour is a property of the work, not of concurrency, so the phase
# breakdown collected here applies to both modes; only contention differs.
#
# CAVEAT ON TIMINGS: --pmc serialises kernel dispatches to read counters, so wall
# clocks under this script are NOT comparable to an unprofiled run. Use the counters
# and the ratios between phases, never the absolute durations.
#
# Usage:
#   tools/run_profiled.sh [-g GPU] [-o OUTDIR] [-- <main_gkr_split args>]
# Examples:
#   tools/run_profiled.sh -g 3 -- 2 768 2 64 32 3072 --c 7 --lasso-k 8
#   tools/run_profiled.sh -g 3 -o tools/results/profile/gpt2 -- 12 768 12 64 16 3072
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

GPU="${GPU:-3}"
OUT="tools/results/profile/$(date +%Y%m%d_%H%M%S)"
# VALUBusy/SALUBusy: fraction of time the vector/scalar ALUs issued.
# MemUnitBusy: memory pipeline occupancy -- the counterpart that says "ALU-idle
#   because it is waiting on memory".
# Occupancy / MeanOccupancyPerCU: waves resident, i.e. whether the kernel even has
#   enough parallelism to hide latency.
# All confirmed present on gfx90a via `rocprofv3 -L`.
PMC="${PMC:-VALUBusy SALUBusy MemUnitBusy Occupancy MeanOccupancyPerCU SQ_INSTS_VALU SQ_WAVES}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -g) GPU="$2"; shift 2 ;;
    -o) OUT="$2"; shift 2 ;;
    --) shift; break ;;
    *)  echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
ARGS=("$@")
if [[ ${#ARGS[@]} -eq 0 ]]; then
  ARGS=(2 768 2 64 32 3072 --c 7 --matmul-k 1 --range-k 1 --lasso-k 8)
fi

for a in "${ARGS[@]}"; do
  if [[ "$a" == "--concurrent" ]]; then
    echo "run_profiled.sh: --concurrent is not profilable (rocprofv3 does not follow" >&2
    echo "  fork(); use --vram-csv for the concurrent path). Refusing." >&2
    exit 2
  fi
done

BIN=bin/main_gkr_split_roctx
if [[ ! -x "$BIN" ]]; then
  echo "run_profiled.sh: $BIN missing. Build it with:" >&2
  echo "  cmake --build build --target main_gkr_split_roctx -j" >&2
  exit 1
fi

# A process stuck in the kernel's kfd_create_process serialises EVERY new HIP process
# on the box, on any GPU. Without this check the run just hangs with no error and no
# output.
if ps -eo stat,wchan:24 | awk '$2=="kfd_create_process"' | grep -q .; then
  echo "run_profiled.sh: the KFD driver is wedged -- processes are stuck in" >&2
  echo "  kfd_create_process, which blocks every new HIP process box-wide:" >&2
  ps -eo pid,stat,etime,wchan:24,cmd | awk '$4=="kfd_create_process"' | sed 's/^/    /' >&2
  echo "  Refusing to launch; the run would hang indefinitely." >&2
  exit 3
fi

mkdir -p "$OUT"
echo "[prof] gpu=$GPU out=$OUT"
echo "[prof] args: ${ARGS[*]}"

# Pass 1: traces only. Kernel durations here are trustworthy, and the ROCTx ranges
# from src/phase_log.hpp (ZKGPT_PHASE_SCOPE) name the phase each dispatch belongs to.
#
# --memory-allocation-trace is the reason this pass exists in its current form: it
# records every hipMalloc/hipFree with size and timestamp, so the device high-water
# mark can be RECONSTRUCTED EXACTLY. The in-binary gpu_peak_mib is a 1 ms poll sampler
# (gpu_provers/gpu_uvm_alloc.hpp) and is a lower bound by construction -- it misses
# any allocation shorter-lived than the interval, and it cannot see the 28 raw
# hipMalloc sites' individual lifetimes.
echo "[prof] pass 1/2: kernel + marker + memory trace"
HIP_VISIBLE_DEVICES="$GPU" rocprofv3 \
  --kernel-trace --marker-trace \
  --memory-allocation-trace --memory-copy-trace --scratch-memory-trace \
  --stats --output-format csv \
  -d "$OUT/trace" -o run -- "$BIN" "${ARGS[@]}" > "$OUT/trace.log" 2>&1
echo "[prof]   rc=$?"

# Pass 2: counters. Separate pass because --pmc serialises dispatches; keeping it
# apart leaves pass 1's timings usable.
echo "[prof] pass 2/2: counters ($PMC)"
HIP_VISIBLE_DEVICES="$GPU" rocprofv3 \
  --pmc $PMC --marker-trace --output-format csv \
  -d "$OUT/pmc" -o run -- "$BIN" "${ARGS[@]}" > "$OUT/pmc.log" 2>&1
echo "[prof]   rc=$?"

echo "[prof] outputs:"
find "$OUT" -name '*.csv' | sed 's/^/  /'
cat <<EOF

Query hints:
  # slowest kernels
  column -s, -t < $OUT/trace/run_kernel_stats.csv | head -20
  # which phase a dispatch belongs to: join kernel_trace timestamps against
  # marker_api_trace ranges (part<i>, part<i>_build, witgen, range_build,
  # range_prove, execute_commit, gkr_prove, verify_gkr, verify_lasso, open_commit)
  head -5 $OUT/trace/run_marker_api_trace.csv
  # exact device high-water mark, per phase
  head -5 $OUT/trace/run_memory_allocation_trace.csv

This directory is an input to tools/overlay_concurrent.py, which folds it onto a
concurrent run's PHASE timeline. Prefer driving the whole thing with
tools/run_overlay_experiment.sh rather than calling this script directly.
EOF
