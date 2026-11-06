#!/bin/bash
# op-partition-sweep.sh <device> <out> <model>...: for each model, calibrate, then prove the whole model
# with K = all, 1, 2, 4, ... ops per partition (op-partition.py). Results: <out>/opt-<model>/summary.log.
# KS="0 64 ..." restricts the K values. Needs CUDA_MPS_PIPE_DIRECTORY pointing at a running MPS daemon.
set -u
cd "$(dirname "$0")"
DEV=$1; OUT=$2; shift 2
declare -A LAYERS=([125m]=12 [1.3b]=24 [2.7b]=32 [6.7b]=32 [13b]=40)
for M in "$@"; do
  TOTAL=$(( 31 * ${LAYERS[$M]} ))
  echo "== opt-$M: calibrate $(date)"
  python3 op-partition.py "$M" 2048 --calibrate --device "$DEV" --out "$OUT" --state-dir "/dev/shm/zkmod-ops-$DEV" || continue
  for K in ${KS:-0 $(k=1; while [ $k -lt $TOTAL ]; do echo $k; k=$((k * 2)); done)}; do
    echo "== opt-$M K=$K $(date)"
    python3 op-partition.py "$M" 2048 --ops-per-part "$K" --device "$DEV" --out "$OUT" --state-dir "/dev/shm/zkmod-ops-$DEV" --concurrent "${CONCURRENT:-16}" 2>&1 | tail -3
  done
done
echo "SWEEP_DONE $(date)"
