#!/bin/bash
# Run every experiment behind Figures 3, 5, 6, 7, 8 and regenerate the figures.
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
  note "Fig 3: baseline latency vs host threads";      THREADS="${FIG3_THREADS:-192 128 64 32 16 8 4 2 1}" $Q/run_fig3.sh
  note "Fig 7/8: seq-len scaling + partition scaling";  $Q/run_fig7_fig8.sh
  # Fig 3's t192 run is the len-64 monolith baseline (same flags, same binary).
  L=tools/results/figures/logs
  for m in gpt2 opt350m; do [[ -f $L/${m}_arm0_mono.log ]] || cp -p "$L/${m}_t192_cpu.log" "$L/${m}_arm0_mono.log"; done
  note "Fig 5/6: all models + memory";                  FIG56_ARMS="${FIG56_ARMS:-1 2 3 4}" $Q/run_fig5_fig6.sh
  note "Fig 5c: zkMod CPU arm vs thread budget";        $Q/run_fig5_threads_gated.sh
  P=tools/results/speedup_analysis_models/paper_arith/_queue/run_paper_arith.sh
  note "Fig 8 CPU: gpt2 CPU+CPU at 1..12 partitions";   for n in 1 2 3 4 5 6 7 8 9 10 11 12; do LOGS=$L SPLITS=$n ARMS=1 $P gpt2; done
  note "GPU monolith: opt350m splits=1";                LOGS=$L SPLITS=1 ARMS=4 $P opt350m
fi

note "plotting"
python3 tools/plot_figures.py > /dev/null
ls -1 tools/results/figures/figs
note ALL_FIGURES_DONE
