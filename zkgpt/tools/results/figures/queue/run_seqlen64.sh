#!/bin/bash
# seqlen64allcpu, seqlen64allgpu: every model at len 64, proof size, verifier time and e2e response
# time, CPU witgen + CPU prove and GPU witgen + GPU prove. Also the GPU runs at len 128 and 256 that
# zkgptseqlenscaling reads, and the gpt2 len 128 monolith.
# opt2b7 runs last: about 1.8 h GPU, about 5.6 h CPU.
set -u
cd "$(dirname "$(readlink -f "$0")")/../../../.."
Q=tools/results/figures/queue
export LOGS=tools/results/figures/logs
note(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }
note "seqlen64 queue up (pid $$)"
MODES="cpu gpu" $Q/run_zkmod.sh gpt2 opt350m opt1b3
LEN=128 MODES=gpu $Q/run_zkmod.sh gpt2
LEN=256 MODES=gpu $Q/run_zkmod.sh gpt2
LEN=128 $Q/run_monolith.sh gpt2
LEN=128 MODES=gpu $Q/run_zkmod.sh opt350m
LEN=256 MODES=gpu $Q/run_zkmod.sh opt350m
[[ ${SEQLEN64_OPT2B7:-1} == 1 ]] && MODES="cpu gpu" $Q/run_zkmod.sh opt2b7
note SEQLEN64_DONE
