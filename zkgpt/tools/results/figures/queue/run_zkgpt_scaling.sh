#!/bin/bash
# zkgptseqlenscaling and zkgptpartitionscaling data.
#
# zkgptseqlenscaling -- sequence-length scaling, len 64 / 128 / 256: the CPU monolith (splits=1,
#   192 threads) vs ours (GPU witgen + GPU prove, forked --overlapped partitions).
#   Models: gpt2 and opt350m (the only ones whose monolith fits in host memory; the
#   input witness is weight-dominated and padded to a fixed half, so
#   the monolith's footprint stays inside the same padded half up to len 128 -- gpt2 171 -> 206 GiB --
#   and doubles at 256 (422 GiB, 2^30 witness / 2^31 Lasso); opt350m len 256 crosses 2^32 and aborts).
#   GPU rows at 64 / 128 / 256 already exist under tools/results/figures/logs;
#   this queue adds the monoliths that were missing: gpt2 len256, opt350m len128 / 256.
# zkgptpartitionscaling -- partition scaling, len 64, GPU witgen + GPU prove, at 1, 2, 8, 12, 16 and 24 partitions.
#   splits=1 is the sequential path (no --overlapped); its 2^30 Lasso runs on the GPU
#   (it fits: 32-34 s in every past run). No host-Lasso fallback -- an OOM fails the
#   row. Uneven splits (5, 7, 8, 9, 10, 11) get depth/n layers with the remainder on
#   the first partitions (main_gkr_split's default division).
#
# Idempotent (done rows are skipped). One run at a time.
set -u
cd "$(dirname "$(readlink -f "$0")")/../../../.."
Q=tools/results/figures/queue
note(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }
note "scaling queue up (pid $$)"

# ---- sequence length: missing monoliths (same consolidated logs/ dir) ----
export LOGS=tools/results/figures/logs
LEN=256 $Q/run_monolith.sh gpt2
LEN=128 $Q/run_monolith.sh opt350m
# opt350m len256 mono: its 2^32 Lasso domain is invalid (verifyLasso throws after ~75 min
# at ~760 GiB host) -- don't re-attempt once the log records that abort.
grep -q "invalid input domain for lasso" "$LOGS/opt350m_len256_mono.log" 2>/dev/null || LEN=256 $Q/run_monolith.sh opt350m
# gpu rows at 128/256 for opt350m come from run_seqlen64.sh; make sure they exist.
LEN=128 MODES=gpu $Q/run_zkmod.sh opt350m
LEN=256 MODES=gpu $Q/run_zkmod.sh opt350m

# ---- partition sweep ----
export LOGS=tools/results/figures/logs
for n in 1 2 8; do   # 12 = the default (logs/gpt2_gpu.log)
  SPLITS=$n MODES=gpu $Q/run_zkmod.sh gpt2
done
# ---- opt350m partition sweep
for n in 1 2 8 12 16; do   # 24 = the default (logs/opt350m_gpu.log)
  SPLITS=$n MODES=gpu $Q/run_zkmod.sh opt350m
done
note SCALING_DONE
