#!/bin/bash
# FULL figure rerun on the 192-thread-pool binary. Logs from binaries without the pool ran
# over their thread budget (children used the whole budget + 384-wide OpenMP teams; the
# monolith's compute_RT used 384 threads), so they are not comparable.
# Already in logs/ from the gpt2 TPC sweep (same binary, same settings): gpt2 arm1 (tpc 16 =
# 192/12, also = fig5c's T=192 point) and the gpt2 len-64 monolith (also fig3's t192).
# Threads per partition: run_paper_arith.sh tpc_for (192/partitions; opt2b7 64).
set -u
cd "$(dirname "$(readlink -f "$0")")/../../../.."
Q=tools/results/speedup_analysis_models/paper_arith/_queue
F=tools/results/figures/queue
export LOGS=tools/results/figures/logs
export BIN=./bin/main_gkr_split_pool_2026-10-24
note(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }
note "pool rerun up (pid $$)"
# Fig 5 / 5a / 5b / table: all arms, len 64 (gpt2 arm1 already here)
ARMS="1 2 3 4" $Q/run_paper_arith.sh gpt2 opt350m opt1b3
# Fig 3 (monolith t-series + GPU-witgen bar); its t192 is also the len-64 arm0 baseline
$F/run_fig3.sh
for m in opt350m; do [[ -f $LOGS/${m}_arm0_mono.log ]] || cp -p $LOGS/${m}_t192_cpu.log $LOGS/${m}_arm0_mono.log; done
note FIG3_FIG5_DONE
# Fig 5c (arm1 vs thread budget, memory-gated, tpc = T/partitions)
$F/run_fig5_threads_gated.sh
# Fig 6 / 7 / 7b: sequence length
LEN=128 ARMS=4 $Q/run_paper_arith.sh gpt2 opt350m opt1b3
LEN=256 ARMS=4 $Q/run_paper_arith.sh gpt2 opt350m
LEN=128 $Q/run_paper_mono.sh gpt2 opt350m
LEN=256 $Q/run_paper_mono.sh gpt2
note SEQLEN_DONE
# Fig 8: partition count (gpt2 s12 rerun under its own name, as before)
for n in 1 2 3 4 5 6 7 8 9 10 11 12; do SPLITS=$n ARMS=4 $Q/run_paper_arith.sh gpt2; done
for n in 6 8 12; do SPLITS=$n ARMS=4 $Q/run_paper_arith.sh opt350m; done
note FIG8_DONE
python3 tools/plot_figures.py
note PLOTTED_MAIN
# opt2b7 last: GPU arms (~1 h each), then CPU arm 1 (~2+ h); arm 3 not rerun
ARMS="2 4" $Q/run_paper_arith.sh opt2b7
ARMS="1" $Q/run_paper_arith.sh opt2b7
python3 tools/plot_figures.py
note POOL_RERUN_DONE
