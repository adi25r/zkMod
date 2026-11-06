#!/bin/bash
# Run every zkGPT experiment and regenerate the figures.
# Everything lives in this directory: queue/ (the run scripts + their .out), logs/
# (every run, one file each), figs/ (png + pdf + md per figure). The plotting is tools/plot_figures.py.
#
#   tools/results/figures/run_all.sh            # run all queues (idempotent), then plot
#   tools/results/figures/run_all.sh --plot     # plot only, from whatever logs exist
#
# Every queue skips rows whose log already shows a complete, verified run, so this
# can be re-launched after a kill. Runs are strictly one at a time (the monoliths
# need 170-400 GiB of host RAM and a thread-scaling curve cannot share cores), so
# a cold start takes many hours. House convention
# is to launch it detached in the `char` screen:
#
#   screen -dmS char bash -c "tools/results/figures/run_all.sh 2>&1 | tee -a tools/results/figures/queue/run_all.out"
#
# Binary: one pinned copy of bin/main_gkr_split (built from build-scale/, see
# CMakeLists.txt) for every queue, so a rebuild never clobbers a binary a sweep is running.
set -u
cd "$(dirname "$0")/../../.."          # repo root
Q=tools/results/figures/queue
note(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

if [[ "${1:-}" != "--plot" ]]; then
  # Every queue runs ONE pinned binary (BIN), copied off bin/main_gkr_split, so a rebuild
  # mid-run cannot change what a later row measures. Override with BIN=... to reuse a copy.
  if [[ -z "${BIN:-}" ]]; then
    [[ -x bin/main_gkr_split ]] || { echo "missing bin/main_gkr_split -- cmake --build build-scale --target main_gkr_split" >&2; exit 1; }
    export BIN=bin/main_gkr_split_run_$(date +%Y-%m-%d_%H%M)
    cp bin/main_gkr_split "$BIN"
  fi
  export BIN; note "BIN=$BIN ($(md5sum < "$BIN" | cut -c1-12))"
  note "zkgptbaselineprofile";                THREADS="${BASELINE_THREADS:-192 128 64 32 16 8 4 2 1}" $Q/run_zkgptbaselineprofile.sh
  note "zkgptseqlenscaling + zkgptpartitionscaling";  $Q/run_zkgpt_scaling.sh
  # The baseline's t192 run is the len-64 monolith (same flags, same binary).
  L=tools/results/figures/logs
  for m in gpt2 opt350m; do [[ -f $L/${m}_arm0_mono.log ]] || cp -p "$L/${m}_t192_cpu.log" "$L/${m}_arm0_mono.log"; done
  note "seqlen64allcpu + seqlen64allgpu";     $Q/run_seqlen64.sh
  note "cpuhostthreadscaling";                $Q/run_cpuhostthreadscaling.sh
fi

note "plotting"
python3 tools/plot_figures.py > /dev/null
ls -1 tools/results/figures/figs
note ALL_FIGURES_DONE
