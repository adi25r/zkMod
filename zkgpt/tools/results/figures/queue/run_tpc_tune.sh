#!/bin/bash
# Threads-per-partition tuning, 192-thread pool binary, len 64, one GPU.
# 192/partitions (gpt2 16, opt350m/opt1b3 8) comes from the main figure queue (logs/); this adds
# the other points. Per (model, arm) the fastest becomes the headline setting in run_paper_arith.sh.
# Logs: tools/results/figures/logs_tune_2026-10-24/tpc<N>/<model>_arm*.log
set -u
cd "$(dirname "$(readlink -f "$0")")/../../../.."
Q=tools/results/speedup_analysis_models/paper_arith/_queue
export BIN=./bin/main_gkr_split_pool_2026-10-24
T=tools/results/figures/logs_tune_2026-10-24
note(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }
while kill -0 42072 2>/dev/null; do sleep 10; done      # the queue's 192/n len-64 arms (gpt2/opt350m/opt1b3)
note "tune up (pid $$)"
run(){ local t=$1 arms=$2; shift 2; mkdir -p $T/tpc$t
       TPC=$t LOGS=$T/tpc$t ARMS="$arms" $Q/run_paper_arith.sh "$@" 2>&1 | grep -E "rc=" | sed "s/^/tpc=$t /"; }
for t in 4 8 32;  do run $t "2 3 4" gpt2; done
for t in 4 16 32; do run $t "1 2 3 4" opt350m; done
for t in 4 16 32; do run $t "1 2 3 4" opt1b3; done
note SMALL_DONE
for t in 32 96;   do run $t "4" opt2b7; done          # 64 = run_paper_arith's opt2b7 default
note TUNE_DONE
