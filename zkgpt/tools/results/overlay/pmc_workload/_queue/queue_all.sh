#!/usr/bin/env bash
# queue_all.sh -- the whole outstanding characterisation queue, one screen job.
#
#   screen -S char
#   tools/results/overlay/pmc_workload/_queue/queue_all.sh 2>&1 | tee -a _queue/all.log
#   (ctrl-a d to detach;  screen -r char  to come back;  _queue/qctl status  any time)
#
# WHAT IT RUNS, in the order it runs them -- cheapest and most useful first, so a box
# that gets busy overnight still leaves whole rows finished rather than a torn matrix:
#
#   figs0   per-phase average utilisation for the sequential 12-partition workload
#           and for monolithic zkGPT on the GPU, from counters already on disk
#   cpuzk   zkGPT, len 64, CPU witgen + CPU proving, 12 partitions -- perf record,
#           then figures. Also covers the cpu+cpu/len64 matrix cell (see below).
#   gg64 gg128 gg256    zkGPT, GPU witgen + GPU proving, 12 partitions
#   cg64 cg128 cg256    zkGPT, CPU witgen + GPU proving, 12 partitions
#   cc128 cc256         zkGPT, CPU witgen + CPU proving, 12 partitions
#   figs    every figure regenerated, plus MATRIX.md
#
# EVERY CELL IS 12 PARTITIONS, SEQUENTIAL (no --concurrent/--batched/--overlapped),
# NOT monolithic (num_splits=1): wall time / speedup
# claims are not the point of this queue, per-partition GPU/CPU utilisation over
# time is, and that needs the same partitioned shape the profiling tooling
# (plot_pmc_timeline.py, plot_phase_util.py, plot_pmc_sequential.py) already reads.
# Every cell uses ALL 192 threads for whichever partition is currently running --
# sequential mode has no concurrent partitions to divide the budget across, unlike
# --concurrent/--batched/--overlapped's tpc = threads/splits.
#
# THE 3x3 IS THE POINT. Nine cells, arm x sequence length, all at 192 threads,
# all 12 partitions:
#
#            len 64          len 128           len 256
#   cpu+cpu  cc_len64        cc_len128         cc_len256
#   cpu+gpu  cg_len64        cg_len128         cg_len256
#   gpu+gpu  gg_len64        gg_len128         gg_len256
#
#   cpu+cpu/len64 is cc_len64, not a separate cc64/cpu_mono cell -- at 12
#   partitions the two would be byte-for-byte the same run (same model, same
#   KNOBS/CPU_FLAGS, same threads, same sequential mode).
#
# ── STOPPING AND RESUMING ─────────────────────────────────────────────────────
#
#   _queue/qctl status          what is running, what is done, what is left
#   _queue/qctl pause           finish the run in flight, then wait
#   _queue/qctl pause --now     kill the run in flight and wait  (it gets retried)
#   _queue/qctl resume          carry on
#   _queue/qctl stop            finish the run in flight, then exit
#   _queue/qctl stop --now      kill the run in flight and exit
#   _queue/qctl retry gg128     clear a stage's done marker so it runs again
#
# A WATCHDOG DOES THE SAME THING AUTOMATICALLY when a neighbour turns up: it samples
# the box every 20 s during every run and kills the run after three consecutive bad
# samples -- another GPU busy, load over the stage's limit, or (CPU cells only)
# another user burning CPU. It does NOT suspend and it does not throttle. Thresholds
# are env-overridable: LOAD_GPU LOAD_CPU OTHER_CPU_CC GATE_LOAD_CC, see below.
#
# That is deliberate, and it is the whole reason the watchdog exists. A measurement
# that shares the box is not a slow measurement, it is a WRONG one: a run at load
# 145-226 beside another tenant has to be thrown away. SIGSTOP would be worse than useless --
# a suspended run still holds its VRAM and its HSA queues, so the neighbour we are
# yielding to gains nothing, while the wall time we came to measure is destroyed
# either way. So a disturbed run is killed, its partial output deleted, its stage
# left UNMARKED, and the outer pass loop retries it once the box is quiet again.
#
# EVERY STAGE IS RESUMABLE. A stage that finishes touches _queue/.done/<stage>; a
# rerun skips it. Delete the marker (or `qctl retry`) to force one, or pass --force.
# --only / --skip take comma lists. --list prints the stages and exits. --passes
# (default 3) bounds how many times a stage the watchdog just killed is retried
# within this one launch; if something is still missing after that, the script
# exits and says so -- relaunch it by hand (or `qctl retry <stage>` first) to try
# again.
#
# NO CALL GRAPHS ON THE perf RECORD PASSES. `--call-graph dwarf,16384` dumps 16 KB of
# stack per sample at 199 Hz x 192 threads (tens of GB per run). Nothing reads it: both figures parse
# `perf report --no-children -g none --sort time,sym`, and perf_report.txt was
# generated with -g none too. Dropping it cuts perf.data by ~3 orders of magnitude,
# which is what makes len128 and len256 CPU runs fit on a volume that is already 90%
# full. If a call graph is ever wanted, take it from a short targeted run.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/../../../../.." || exit 1

# This repo (and its build/ and bin/) lives on shared NFS storage mounted on several
# hosts in the pool, each with its OWN /opt/rocm install at its own real path -- a
# binary's baked-in RPATH only resolves on the host that built it. main_gkr_split's
# RPATH here also only captured .../llvm/lib (compiler runtime), not the directory
# holding libamdhip64.so itself, so even a same-host rebuild needs this. Harmless
# where it's already resolvable via RPATH.
export LD_LIBRARY_PATH="/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# The repo (and .done/) is shared NFS storage reachable from several hosts in the
# pool. Two instances launched from different hosts race on the same .done markers
# and result directories with no other protection -- exactly what silently
# clobbered good data. flock this one host-visible
# lockfile so a second concurrent launch (from ANY host, since it's the same NFS
# path) refuses to start instead of racing. Must happen before the `trap ... EXIT`
# below: a refused second instance's own EXIT trap would otherwise delete the
# FIRST instance's $CURRENT out from under it.
mkdir -p tools/results/overlay/pmc_workload/_queue
exec 9>tools/results/overlay/pmc_workload/_queue/.lock
if ! flock -n 9; then
  echo "queue_all.sh: another instance already holds .lock -- refusing to start" >&2
  echo "(check 'qctl status'; if that instance is actually dead, rm the .lock file)" >&2
  exit 1
fi

ROOT=tools/results/overlay/pmc_workload
Q=$ROOT/_queue
DONE=$Q/.done
STATUS=$Q/STATUS
CONTROL=$Q/.control
ABORT=$Q/.abort
CURRENT=$Q/.current
GPU=0                            # host-specific: a wrong index silently fails every GPU
                                  # cell with "HIP error 100: no ROCm-capable device is
                                  # detected"; only check_accepts catches it, not rc/watchdog.
THREADS=192                     # one thread per physical core, never more
MIN_FREE_GB=250                 # abort a run stage rather than fill /nvmepool
# THE TWO ARMS NEED DIFFERENT DEFINITIONS OF "QUIET", and conflating them either
# poisons the CPU cells or starves the GPU ones. This box is routinely shared, so an absolute "must be alone" gate
# on everything would simply never fire.
#
#   GPU cells care about the CARD, not the host. What ruins them is another tenant on
#   a GPU; host load barely reaches them, because GPU proving is on the device and
#   CPU witgen is single-threaded in every arm (project_witgen_fully_on_gpu measured
#   1.0 core). Runs at load 80-193 with
#   the GPUs to themselves are good data, so that is the calibration.
#
#   CPU cells care about the HOST, absolutely. cc_len128/cc_len256 have to match
#   a quiet-box cpu_mono run or the column cannot be read. A
#   neighbour taking 14 cores is 7% of the budget and lands directly in the numbers.
#
# All four are env-overridable, so a run can be loosened without editing the script:
#   LOAD_GPU=200 LOAD_CPU=240 queue_all.sh
LOAD_GPU=${LOAD_GPU:-200}       # watchdog ceiling during a GPU cell
LOAD_CPU=${LOAD_CPU:-230}       # ...during a CPU cell (the run itself is ~192)
OTHER_CPU_CC=${OTHER_CPU_CC:-400}   # another user's summed %CPU that aborts a CPU cell
GATE_LOAD_CC=${GATE_LOAD_CC:-12}    # load a CPU cell must see before it will start
KNOBS=(--c 18 --matmul-k 4 --range-k 8 --lasso-k 8 --sc-tile-bits 18)
CPU_FLAGS=(--cpu --cpu-open --cpu-commit --cpu-deg3)
mkdir -p "$DONE"
trap 'rm -f "$CURRENT"' EXIT

ALL=(figs0 cpuzk gg64 gg128 gg256 cg64 cg128 cg256 cc128 cc256 figs)
ONLY=""; SKIP=""; FORCE=0; PASSES=3
while [[ $# -gt 0 ]]; do
  case "$1" in
    --only) ONLY=$2; shift 2;;
    --skip) SKIP=$2; shift 2;;
    --force) FORCE=1; shift;;
    --passes) PASSES=$2; shift 2;;
    --list) printf '%s\n' "${ALL[@]}"; exit 0;;
    *) echo "unknown option $1"; exit 2;;
  esac
done

say()  { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }
note() { say "$*"; echo "$(date +%m-%d\ %H:%M:%S) $*" >> "$STATUS"; }

free_gb() { df -BG --output=avail /nvmepool | tail -1 | tr -dc 0-9; }
load1()   { cut -d' ' -f1 /proc/loadavg; }
loadi()   { awk '{print int($1)}' /proc/loadavg; }
busy_gpus() { rocm-smi --showuse 2>/dev/null | grep -c "use (%): [1-9]"; }

# Another user's summed %CPU. ps reports a lifetime average, which is what we want
# here: a neighbour's hour-long job reads high, a momentary `ls` does not.
other_cpu() {
  ps -eo user:24,pcpu --no-headers 2>/dev/null |
    awk -v me="$(id -un)" '$1!=me {s+=$2} END {printf "%d", s+0}'
}

# A wedged KFD serialises every new HIP process box-wide, on any GPU, with no error
# and no output -- see reference_kfd_create_process_wedge. Cheap to check.
kfd_wedged() { ps -eo wchan:24 | awk '$1=="kfd_create_process"' | grep -q .; }

# ── operator control ──────────────────────────────────────────────────────────
control() { [[ -s $CONTROL ]] && tr -d ' \n' < "$CONTROL" || echo run; }

# Block here while paused; leave entirely on stop. Called between stages, and by the
# watchdog once it has cleared the run that was in flight.
gate() {
  local c first=1
  while :; do
    c=$(control)
    [[ $c == stop ]] && { note "control=stop -- exiting cleanly"; rm -f "$ABORT"; exit 0; }
    [[ $c != pause ]] && { [[ $first -eq 0 ]] && note "resumed"; rm -f "$ABORT"; return 0; }
    [[ $first -eq 1 ]] && note "PAUSED by qctl -- waiting for 'qctl resume'"
    first=0
    sleep 60
  done
}

# ── run one measurement under the watchdog ────────────────────────────────────
# The command runs in its own session (setsid) so the whole tree can be signalled as
# a process group -- rocprofv3 forks the workload, and killing only the wrapper would
# leave the run holding the card.
WATCH_REASON=""
watched() {                     # loadlimit expected_gpus othercpu cap logfile -- cmd...
  local lim=$1 mygpu=$2 ocpu=$3 cap=$4 log=$5; shift 5
  [[ ${1:-} == -- ]] && shift
  WATCH_REASON=""
  setsid "$@" > "$log" 2>&1 &
  local pid=$! t0=$SECONDS bad=0 rc=0 why now
  while :; do
    if ! kill -0 "$pid" 2>/dev/null; then wait "$pid"; rc=$?; break; fi
    sleep 20
    why=""
    now=0
    # Only .abort (the --now variants) kills a run in flight. A plain pause/stop is
    # honoured at the next stage boundary by gate(), which is what makes it the
    # graceful one -- having the watchdog react to the control file here made
    # `qctl stop` behave identically to `qctl stop --now`.
    if [[ -f $ABORT ]]; then why="operator abort"; now=1
    elif [[ $((SECONDS-t0)) -gt $cap ]]; then why="over the ${cap}s cap"; now=1
    else
      local ld ob oc
      ld=$(loadi); oc=$(other_cpu)
      # A CPU-only cell (mygpu=0, e.g. perf_cell) never touches the card, so another
      # tenant's GPU job is not its problem -- that check must not fire on a CPU
      # cell for as long as ANY GPU anywhere was busy.
      if [[ $mygpu -gt 0 ]]; then ob=$(( $(busy_gpus) - mygpu )); else ob=0; fi
      if   [[ $ld -gt $lim ]];                     then why="load $ld over the $lim limit"
      elif [[ $ob -gt 0 ]];                        then why="$ob other GPU(s) busy"
      elif [[ $ocpu -gt 0 && $oc -gt $ocpu ]];     then why="another user at ${oc}% CPU"
      fi
    fi
    if [[ -n $why ]]; then
      bad=$((bad+1))
      say "     watchdog: $why  ($bad/3)"
      if [[ $now -eq 1 || $bad -ge 3 ]]; then
        WATCH_REASON=$why
        kill -TERM -- "-$pid" 2>/dev/null; sleep 10
        kill -KILL -- "-$pid" 2>/dev/null
        wait "$pid" 2>/dev/null
        rc=99; break
      fi
    else
      bad=0
    fi
  done
  return $rc
}

# Throw away a killed run's partial output. Guarded to paths under $ROOT so a bad
# expansion cannot reach anything else.
scrub() {
  local p=$1
  [[ $p == "$ROOT"/*/* && -d $p ]] || return 0
  say "     discarding partial output $p"
  rm -rf -- "$p"
}

# `watched`'s rc only reports whether the WATCHDOG killed the run (rc=99) -- a run
# that fails on its own (wrong GPU index, missing library, an assert) exits some
# OTHER way and looks identical to success from the caller's point of view. This
# bit repeatedly (a libomp/RPATH load failure, a stale-report shortcut, a nonexistent
# GPU index) -- each got marked done with zero real data. The one signal that is actually load-bearing either
# way is the partition count: an honest run prints exactly one "RESULT: ACCEPT"
# per partition, so require that instead of trusting a clean exit code.
check_accepts() {                # logfile splits label
  local log=$1 want=$2 label=$3
  local got
  got=$(grep -c '^RESULT: ACCEPT' "$log" 2>/dev/null)
  if [[ $got -ne $want ]]; then
    note "!! $label: $got/$want ACCEPT -- treating as failed (see $log)"
    return 1
  fi
  return 0
}

# Wait for a quiet box. Four consecutive clean checks, not one: a run that starts in a lull
# can then hit load 145-226 with 3-4 GPUs busy and has to be discarded. `need_gpu` also gates on the card being idle; a CPU-only stage
# does not care what the GPUs are doing.
quiet_wait() {                  # $1 = need_gpu (0/1)
  local need_gpu=$1 streak=0 ok
  while :; do
    gate
    ok=1
    if [[ $need_gpu -eq 1 ]]; then
      # The card must be ours alone; the host may be busy, and usually is.
      kfd_wedged && ok=0
      [[ $(busy_gpus) -ne 0 ]] && ok=0
      [[ $(loadi) -ge $LOAD_GPU ]] && ok=0
    else
      # A CPU cell needs the host to itself, full stop.
      [[ $(loadi) -ge $GATE_LOAD_CC ]] && ok=0
      [[ $(other_cpu) -gt $OTHER_CPU_CC ]] && ok=0
    fi
    # pgrep -x matches against /proc's `comm`, which the kernel truncates to 15
    # chars -- "main_gkr_split_roctx" (21 chars) can NEVER match at that length,
    # pgrep says so on stderr every time, and its truncated comm ("main_gkr_split_",
    # trailing underscore) doesn't match the plain "main_gkr_split" pattern either.
    # So the roctx binary was invisible to this check. `-f "bin/main_gkr_split"` is
    # a substring match on the full command line instead -- it catches both
    # binaries (both are invoked as .../bin/main_gkr_split[_roctx]) and cannot
    # self-match this script's own argv (just "bash queue_all.sh").
    pgrep -x perf >/dev/null && ok=0
    pgrep -f "bin/main_gkr_split" >/dev/null && ok=0
    if [[ $ok -eq 1 ]]; then streak=$((streak+1)); else streak=0; fi
    say "   quiet_streak=$streak load=$(load1) busy_gpus=$(busy_gpus) other_cpu=$(other_cpu)% free=$(free_gb)G"
    [[ $streak -ge 4 ]] && return 0
    sleep 180
  done
}

# ── one rocprofv3 cell: an unprofiled timing run, then three counter passes ────
# Three counter passes, always. FETCH_SIZE/WRITE_SIZE and L2CacheHit need six TCC
# counters between them and do not fit one slot set beside VALUBusy, and rocprofv3
# DROPS what does not fit while only warning, leaving a silent 0.0. Splitting is not optional.
#
# Returns non-zero if any sub-run was killed, so the caller leaves the stage unmarked.
pmc_cell() {                    # tag len witgen(cpu|gpu) cap splits
  local tag=$1 len=$2 wit=$3 cap=$4 splits=${5:-1}
  local base="$ROOT/$tag" rc t0
  export ZKGPT_WITNESS_CACHE=0 GKR_SC_STREAM_BITS=22 GKR_BFEVAL_GPU=1
  if [[ $wit == gpu ]]; then
    export GKR_WITGEN_GPU=1 GKR_ROUND_GPU=1 GKR_GATES_GPU=1 GKR_WITGEN_FULL=1
  else
    unset GKR_WITGEN_GPU GKR_ROUND_GPU GKR_GATES_GPU GKR_WITGEN_FULL
  fi
  export HIP_VISIBLE_DEVICES=$GPU

  # An UNPROFILED timing run FIRST. --pmc serialises kernel dispatches to read the
  # counter registers and inflates wall time ~2.2x, so the profiled logs' witgen_s /
  # prover_total_s are the counters' companions, not timings. Without this pass the
  # matrix would have no comparable wall time for the GPU cells at all -- and "where
  # is the time saved" is the whole question.
  mkdir -p "$base/timing"
  say "   $tag/timing : unprofiled  (load=$(load1) busy_gpus=$(busy_gpus))"
  t0=$SECONDS
  watched "$LOAD_GPU" 1 0 "$cap" "$base/timing/run.log" -- \
    ./bin/main_gkr_split 12 768 "$splits" "$len" "$THREADS" 3072 "${KNOBS[@]}"
  rc=$?
  say "     rc=$rc elapsed=$((SECONDS-t0))s accepts=$(grep -c '^RESULT: ACCEPT' "$base/timing/run.log" 2>/dev/null)"
  if [[ $rc -eq 99 ]]; then
    note "!! $tag/timing killed: $WATCH_REASON -- will retry"
    scrub "$base/timing"; return 1
  fi
  check_accepts "$base/timing/run.log" "$splits" "$tag/timing" || { scrub "$base/timing"; return 1; }

  for spec in "bandwidth FETCH_SIZE WRITE_SIZE" \
              "cache L2CacheHit MemUnitBusy MemUnitStalled" \
              "compute VALUBusy SALUBusy MeanOccupancyPerCU SQ_WAVES SQ_INSTS_VALU"; do
    set -- $spec
    local set_name=$1; shift
    local out="$base/$set_name"
    mkdir -p "$out"
    say "   $tag/$set_name : $*  (load=$(load1) busy_gpus=$(busy_gpus) free=$(free_gb)G)"
    t0=$SECONDS
    watched "$LOAD_GPU" 1 0 "$cap" "$out/pmc.log" -- \
      rocprofv3 --pmc "$@" --marker-trace --output-format csv \
      -d "$out/pmc" -o run -- bin/main_gkr_split_roctx \
      12 768 "$splits" "$len" "$THREADS" 3072 "${KNOBS[@]}"
    rc=$?
    say "     rc=$rc elapsed=$((SECONDS-t0))s accepts=$(grep -c '^RESULT: ACCEPT' "$out/pmc.log" 2>/dev/null)"
    if [[ $rc -eq 99 ]]; then
      note "!! $tag/$set_name killed: $WATCH_REASON -- will retry"
      scrub "$out"; return 1
    fi
    check_accepts "$out/pmc.log" "$splits" "$tag/$set_name" || { scrub "$out"; return 1; }
    [[ $rc -ne 0 ]] && grep -m2 -E "out of memory|deg3 extension|terminate|Aborted|assert" "$out/pmc.log" | sed 's/^/     /'
    grep -m1 "Unable to find all counters" "$out/pmc.log" | sed 's/^/     DROPPED: /'
  done
  return 0
}

# ── one perf cell (no GPU kernels at all, so rocprofv3 has nothing to count) ───
# perf stat for the totals, perf record for the shape, then the time-bucketed symbol
# report both figures parse. -n on that report is load-bearing: perf rounds
# percentages to 2 dp and a single-threaded phase's 1 s buckets print 0.00% and
# vanish (that erases single-threaded witgen).
perf_cell() {                   # tag len splits cap do_stat
  local tag=$1 len=$2 splits=$3 cap=$4 do_stat=$5
  local out="$ROOT/$tag" rc t0
  mkdir -p "$out"
  local args=(12 768 "$splits" "$len" "$THREADS" 3072 "${KNOBS[@]}" "${CPU_FLAGS[@]}")
  export ZKGPT_WITNESS_CACHE=0 GKR_MATMUL_CPU=1 GKR_LASSO_CPU=1 GKR_PRED_CPU=1
  unset GKR_WITGEN_GPU GKR_ROUND_GPU GKR_GATES_GPU GKR_WITGEN_FULL HIP_VISIBLE_DEVICES

  if [[ $do_stat -eq 1 ]]; then
    say "   $tag perf stat  (load=$(load1))"
    t0=$SECONDS
    watched "$LOAD_CPU" 0 "$OTHER_CPU_CC" "$cap" "$out/stat_run.log" -- \
      perf stat -x, -o "$out/perf_stat.csv" \
      -e cycles,instructions,cache-references,cache-misses,branch-misses,context-switches,page-faults \
      ./bin/main_gkr_split "${args[@]}"
    rc=$?
    say "     rc=$rc elapsed=$((SECONDS-t0))s accepts=$(grep -c '^RESULT: ACCEPT' "$out/stat_run.log" 2>/dev/null)"
    [[ $rc -eq 99 ]] && { note "!! $tag perf stat killed: $WATCH_REASON -- will retry"; return 1; }
    check_accepts "$out/stat_run.log" "$splits" "$tag/stat" || return 1
  else
    say "   $tag perf stat already collected -- record pass only"
  fi

  say "   $tag perf record  (no call graph -- see the header)"
  t0=$SECONDS
  watched "$LOAD_CPU" 0 "$OTHER_CPU_CC" "$cap" "$out/record_run.log" -- \
    perf record -F 199 -o "$out/perf.data" ./bin/main_gkr_split "${args[@]}"
  rc=$?
  say "     rc=$rc elapsed=$((SECONDS-t0))s accepts=$(grep -c '^RESULT: ACCEPT' "$out/record_run.log" 2>/dev/null) perf.data=$(du -h "$out/perf.data" 2>/dev/null | cut -f1)"
  if [[ $rc -eq 99 ]]; then
    note "!! $tag perf record killed: $WATCH_REASON -- will retry"
    rm -f "$out/perf.data"; return 1
  fi
  check_accepts "$out/record_run.log" "$splits" "$tag/record" || { rm -f "$out/perf.data"; return 1; }

  say "   $tag time-bucketed symbol report"
  perf report -i "$out/perf.data" --stdio --no-children -g none -n \
    --sort time,sym --time-quantum 1s --percent-limit 0 \
    > "$out/report_by_time.txt" 2>/dev/null
  say "     rows=$(wc -l < "$out/report_by_time.txt")"
  return 0
}

gpu_figs() {                    # arm dir -> the three GPU figure families
  local d="$ROOT/$1"
  [[ -d $d ]] || return 0
  python3 tools/plot_phase_util.py "$d"                 2>&1 | sed 's/^/     /'
  python3 tools/plot_pmc_kernels.py "$d" --all          2>&1 | tail -2 | sed 's/^/     /'
  python3 tools/plot_pmc_timeline.py "$d" --all         2>&1 | tail -2 | sed 's/^/     /'
  for s in bandwidth cache compute; do
    [[ -d $d/$s ]] && python3 tools/pmc_by_phase.py "$d/$s" > "$d/$s/by_phase.txt" 2>&1
  done
}

cpu_figs() {                    # arm dir -> the two perf figures
  local d="$ROOT/$1"
  [[ -s $d/report_by_time.txt ]] || { say "   no report_by_time.txt in $1, skipping"; return 0; }
  python3 tools/plot_perf_phases.py "$d" 2>&1 | sed 's/^/     /'
  python3 tools/plot_perf_util.py   "$d" 2>&1 | sed 's/^/     /'
}

wanted() {                      # stage -> should we run it
  local s=$1
  [[ -n "$ONLY" && ",$ONLY," != *",$s,"* ]] && return 1
  [[ -n "$SKIP" && ",$SKIP," == *",$s,"* ]] && return 1
  [[ $FORCE -eq 0 && -f "$DONE/$s" ]] && return 1
  gate
  echo "$s" > "$CURRENT"
  return 0
}

all_done() {
  local s
  for s in "${ALL[@]}"; do
    [[ -n "$ONLY" && ",$ONLY," != *",$s,"* ]] && continue
    [[ -n "$SKIP" && ",$SKIP," == *",$s,"* ]] && continue
    [[ -f "$DONE/$s" ]] || return 1
  done
  return 0
}

# ── the stages, as one pass over the list ─────────────────────────────────────
run_stages() {
  local s len cap cell ok

  if wanted figs0; then
    note "=== figs0: per-phase utilisation from counters already on disk"
    # item 1: the sequential 12-partition workload.  item 2: monolithic on the GPU.
    python3 tools/plot_phase_util.py "$ROOT/zkmod" \
      --title "Per-phase average GPU utilisation -- sequential 12-partition workload" \
      2>&1 | sed 's/^/     /'
    python3 tools/plot_phase_util.py "$ROOT/mono" \
      --title "Per-phase average GPU utilisation -- zkGPT monolithic on the GPU" \
      2>&1 | sed 's/^/     /'
    for a in len128_zkmod len128_mono len256_zkmod len256_mono; do
      python3 tools/plot_phase_util.py "$ROOT/$a" 2>&1 | sed 's/^/     /'
    done
    touch "$DONE/figs0"; note "figs0 done"
  fi

  # zkMod on the CPU: 12 partitions, run_split_sequential, the SAME perf treatment
  # as cpu_mono so the two are readable side by side.
  #
  # The perf stat pass is already collected, so only the record pass runs here --
  # do_stat=0. Gated and watched like every other CPU cell.
  # There is no separate cc64 (cpu+cpu, len64) cell: at 12 splits it would be
  # byte-for-byte the same run as this one (same model, same KNOBS/CPU_FLAGS, same
  # 192 threads, same sequential 12-partition mode), so cc_len64 IS the
  # cpu+cpu/len64 cell.
  if wanted cpuzk; then
    note "=== cpuzk: zkGPT, len 64, CPU witgen + CPU proving, 12 partitions, $THREADS threads"
    ok=1
    if [[ -s $ROOT/cc_len64/report_by_time.txt ]]; then
      say "   report_by_time.txt already present -- plotting only"
    else
      quiet_wait 0
      perf_cell cc_len64 64 12 7200 1 || ok=0
    fi
    if [[ $ok -eq 1 ]]; then
      cpu_figs cc_len64
      touch "$DONE/cpuzk"; note "cpuzk done"
    fi
  fi

  # gpu+gpu row: fast, so it goes first and completes a whole row early. 12
  # partitions, sequential (no --concurrent/--batched/--overlapped) -- wall time
  # is not the point here, per-partition GPU utilisation over time is, and
  # sequential is what the existing timeline/phase-utilisation tooling
  # (plot_pmc_timeline.py, plot_phase_util.py) already expects (same shape as the
  # original zkmod/len128_zkmod/len256_zkmod dirs, just at 192 threads not 16).
  for cell in "gg64 64 3600" "gg128 128 5400" "gg256 256 7200"; do
    set -- $cell; s=$1; len=$2; cap=$3
    wanted "$s" || continue
    note "=== $s: zkGPT len $len, GPU witgen + GPU proving, 12 partitions, $THREADS threads"
    quiet_wait 1
    [[ $(free_gb) -lt $MIN_FREE_GB ]] && { note "!! only $(free_gb)G free, skipping $s"; continue; }
    if pmc_cell "gg_len$len" "$len" gpu "$cap" 12; then
      touch "$DONE/$s"; note "$s done"
    fi
  done

  # cpu+gpu row: CPU witgen is the slow half and runs four times per cell.
  for cell in "cg64 64 7200" "cg128 128 14400" "cg256 256 28800"; do
    set -- $cell; s=$1; len=$2; cap=$3
    wanted "$s" || continue
    note "=== $s: zkGPT len $len, CPU witgen + GPU proving, 12 partitions, $THREADS threads"
    quiet_wait 1
    [[ $(free_gb) -lt $MIN_FREE_GB ]] && { note "!! only $(free_gb)G free, skipping $s"; continue; }
    if pmc_cell "cg_len$len" "$len" cpu "$cap" 12; then
      touch "$DONE/$s"; note "$s done"
    fi
  done

  # cpu+cpu row, len128/256 (len64 is cpuzk/cc_len64 above). Everything here is
  # capped at $THREADS=192, never more (192 physical cores, the other
  # 192 logical CPUs are SMT siblings that hurt this workload).
  for cell in "cc128 128 21600 cc_len128" "cc256 256 43200 cc_len256"; do
    set -- $cell; s=$1; len=$2; cap=$3; tag=$4
    wanted "$s" || continue
    note "=== $s: zkGPT len $len, CPU witgen + CPU proving, 12 partitions, $THREADS threads"
    quiet_wait 0
    [[ $(free_gb) -lt $MIN_FREE_GB ]] && { note "!! only $(free_gb)G free, skipping $s"; continue; }
    if perf_cell "$tag" "$len" 12 "$cap" 1; then
      cpu_figs "$tag"
      touch "$DONE/$s"; note "$s done"
    fi
  done

  # figs is marked done only on the last pass, so a retried cell still gets plotted.
  if [[ ${LAST_PASS:-0} -eq 1 ]] && wanted figs; then
    note "=== figs: regenerating every figure and the matrix summary"
    for a in gg_len64 gg_len128 gg_len256 cg_len64 cg_len128 cg_len256; do
      [[ -d $ROOT/$a ]] && { say "   $a"; gpu_figs "$a"; }
    done
    for a in cc_len64 cc_len128 cc_len256; do
      [[ -d $ROOT/$a ]] && { say "   $a"; cpu_figs "$a"; }
    done
    python3 "$Q/matrix_summary.py" > "$Q/MATRIX.md" 2>&1
    say "   wrote $Q/MATRIX.md"
    touch "$DONE/figs"; note "figs done"
  fi
  rm -f "$CURRENT"
}

# ── main ──────────────────────────────────────────────────────────────────────
note "queue_all starting  (pid $$, free $(free_gb)G, load $(load1))"
echo run > "$CONTROL"; rm -f "$ABORT"
echo "$$" > "$Q/.pid"

# Do not even plot while another perf record is live. The figure stages are only a
# core or two of Python, but cc_len64 is a 192-thread timed measurement and every
# core taken from it lands in its numbers.
while pgrep -x perf >/dev/null; do
  say "   waiting for an in-flight perf record (load $(load1))"; sleep 120
done

# Several passes, because a cell the watchdog killed is left unmarked and has to come
# round again. Later passes only pick up what is still missing.
for p in $(seq 1 "$PASSES"); do
  [[ $p -eq $PASSES ]] && LAST_PASS=1 || LAST_PASS=0
  if all_done; then note "everything marked done"; break; fi
  [[ $p -gt 1 ]] && note "--- pass $p of $PASSES: retrying what the watchdog killed"
  run_stages
done

note "ALL STAGES COMPLETE  (free $(free_gb)G)"
for s in "${ALL[@]}"; do [[ -f $DONE/$s ]] || note "   NOT done: $s"; done
