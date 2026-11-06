#!/bin/bash
# Run any command under the memory-bandwidth collectors.   Usage:  wrap.sh <prefix> <command...>
#   CPU : perf stat -I $INTERVAL_MS over the whole process tree (forks/threads included)  -> <prefix>.cpu.csv
#   GPU : 10 ms poller (NVML or AMD sysfs)                                                -> <prefix>.gpu.csv
#   also <prefix>.start_epoch (wall clock at start) and <prefix>.log (the command's stdout+stderr).
# Env:  GPU_VENDOR=nvidia|amd|none (default nvidia)   GPU_INDEX=0   INTERVAL_MS=10
#       PERF_EVENTS   comma list of perf events whose counts are DRAM cache-line fills/transfers (default = AMD Zen4, see README)
set -u
P=$1; shift; HERE=$(cd "$(dirname "$0")" && pwd)
EVENTS=${PERF_EVENTS:-ls_any_fills_from_sys.dram_io_all,l2_pf_miss_l2_l3.all}
date +%s.%N > "$P.start_epoch"
case ${GPU_VENDOR:-nvidia} in
  nvidia) python3 "$HERE/sampler_nvidia.py" "$P.gpu.csv" 0.01 "${GPU_INDEX:-0}" & SP=$! ;;
  amd)    python3 "$HERE/sampler_amd.py"    "$P.gpu.csv" 0.01 & SP=$! ;;
  *)      SP= ;;
esac
perf stat -x, -I "${INTERVAL_MS:-10}" -o "$P.cpu.csv" -e "$EVENTS" -- "$@" > "$P.log" 2>&1
rc=$?
[ -n "${SP:-}" ] && kill $SP 2>/dev/null && wait $SP 2>/dev/null
exit $rc
