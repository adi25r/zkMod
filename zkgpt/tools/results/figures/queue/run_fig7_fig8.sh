#!/bin/bash
# Figures 7 + 8 data.
#
# Fig 7 -- sequence-length scaling, len 64 / 128 / 256: the CPU monolith (splits=1,
#   192 threads) vs ours (GPU witgen + GPU prove, forked --overlapped partitions).
#   Models: gpt2 and opt350m (the only ones whose monolith fits in host memory; the
#   paper layout's input witness is weight-dominated and padded to a fixed half, so
#   the monolith's footprint stays inside the same padded half up to len 128 -- gpt2 171 -> 206 GiB --
#   and doubles at 256 (422 GiB, 2^30 witness / 2^31 Lasso); opt350m len 256 crosses 2^32 and aborts).
#   Arm rows at 64 / 128 / 256 already exist under tools/results/figures/logs;
#   this queue adds the monoliths that were missing: gpt2 len256, opt350m len128 / 256.
# Fig 8 -- partition scaling, gpt2 len 64, GPU witgen + GPU prove, splits 1..12.
#   splits=1 is the sequential path (no --overlapped); its 2^30 Lasso runs on the GPU
#   (it fits: 32-34 s in every past run). No host-Lasso fallback -- an OOM fails the
#   row. Uneven splits (5, 7, 8, 9, 10, 11) get depth/n layers with the remainder on
#   the first partitions (main_gkr_split's default division).
#
# Idempotent (done rows are skipped). One run at a time.
set -u
cd "$(dirname "$(readlink -f "$0")")/../../../.."
Q=tools/results/speedup_analysis_models/paper_arith/_queue
note(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }
note "fig7/fig8 queue up (pid $$)"

# ---- Fig 7: missing monoliths (same consolidated logs/ dir) ----
export LOGS=tools/results/figures/logs
LEN=256 $Q/run_paper_mono.sh gpt2
LEN=128 $Q/run_paper_mono.sh opt350m
# opt350m len256 mono: its 2^32 Lasso domain is invalid (verifyLasso throws after ~75 min
# at ~760 GiB host) -- don't re-attempt once the log records that abort.
grep -q "invalid input domain for lasso" "$LOGS/opt350m_len256_arm0_mono.log" 2>/dev/null || LEN=256 $Q/run_paper_mono.sh opt350m
# arm4 rows at 128/256 for opt350m come from the fig5/6 queue; make sure they exist.
LEN=128 ARMS=4 $Q/run_paper_arith.sh opt350m
LEN=256 ARMS=4 $Q/run_paper_arith.sh opt350m

# ---- Fig 8: partition sweep ----
export LOGS=tools/results/figures/logs
for n in 12 6 4 3 2 8 10 9 7 5 11 1; do
  SPLITS=$n ARMS=4 $Q/run_paper_arith.sh gpt2
done
# ---- opt350m partition sweep: with 6 partitions (4 layers each) the last partition's
# ~110 s proof is an unhidden tail, which is why opt350m's speedup trails gpt2's.
# 8 / 12 / 24 partitions = 3 / 2 / 1 layers each. Same flags otherwise.
for n in 12 8 6; do   # 24 = the default (logs/opt350m_arm4_gpuwg_gpuprove.log)
  SPLITS=$n ARMS=4 $Q/run_paper_arith.sh opt350m
done
# (opt1b3 24 / opt2b7 32 partitions are the defaults, so their max-partition
# runs are the Fig 5 rows -- logs/<m>_arm4_gpuwg_gpuprove.log.)
note FIG78_DONE
