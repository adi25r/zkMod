#!/bin/bash
# Figures 5 + 6 data. Everything is the GPU/GPU arm (arm4: GPU witgen +
# GPU prove, forked --overlapped partitions, paper arithmetization) unless noted.
#
# Fig 5 -- all models, len 64: prover time, proof size, verifier time, e2e response
#   time. Needs the per-partition verifier_s / proof_kb fields of the PARTITION
#   line; arm4 is run for gpt2 / opt350m / opt1b3 / opt2b7 into
#   tools/results/figures/logs/. Baseline rows come from paper_arith/logs/<m>_arm0_mono.log
#   (their "Split proof results" block already has all four numbers).
# Fig 6 -- memory vs model size (arm4 len64 rows above + mono) and vs input size:
#   gpt2 arm4 at len 128 / 256, opt350m arm4 at 128 / 256, opt1b3 arm4 at 128, and
#   the gpt2 monolith at len 128 (len 256 mono would be a 2^31-element witness,
#   ~684 GiB -- more than the box).
# Order: cheap Fig 5 rows first, then the ~1.8 h opt2b7 arm4, then the seq-len rows.
set -u
cd "$(dirname "$(readlink -f "$0")")/../../../.."
Q=tools/results/speedup_analysis_models/paper_arith/_queue
export LOGS=tools/results/figures/logs
note(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }
note "fig5/fig6 queue up (pid $$)"
# GPU/GPU arm only: the CPU-witgen / CPU-prove arms 1-3 are not reported.
ARMS="${FIG56_ARMS:-4}" $Q/run_paper_arith.sh gpt2 opt350m opt1b3
LEN=128 ARMS=4 $Q/run_paper_arith.sh gpt2
LEN=256 ARMS=4 $Q/run_paper_arith.sh gpt2
LEN=128 $Q/run_paper_mono.sh gpt2
LEN=128 ARMS=4 $Q/run_paper_arith.sh opt350m
LEN=256 ARMS=4 $Q/run_paper_arith.sh opt350m
LEN=128 ARMS=4 $Q/run_paper_arith.sh opt1b3

# --- Fig 5, opt2b7 (all four arms; run_paper_arith.sh goes 1,2,3,4): arm4 ~1.8 h, arm2 ~1.8 h, arm1 and
# arm3 ~5.6 h each (16 x ~1200 s CPU proves, one partition at a time).
# FIG56_OPT2B7=0: opt2b7 runs as separate per-arm jobs 
[[ ${FIG56_OPT2B7:-1} == 1 ]] && ARMS="1 2 3 4" $Q/run_paper_arith.sh opt2b7
note FIG5_ARMS_DONE
note FIG56_DONE
