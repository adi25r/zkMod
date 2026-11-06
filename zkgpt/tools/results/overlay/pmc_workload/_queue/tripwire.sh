#!/usr/bin/env bash
# tripwire.sh -- catches box-wide CPU contention in the act.
#
# sar's own 10-min samples can show load spiking to 100-160+ for a few minutes and
# dropping back to near-zero, but by the time anyone runs `ps` to look, the spike
# is already gone. The box has no process accounting (no lastcomm/acct) and no atop/pidstat history, so a
# spike that has already ended cannot be attributed after the fact -- the only way
# to see who is doing it is to catch one WHILE it is happening.
#
#   tripwire.sh start [threshold]   # default threshold 100, background, survives logout
#   tripwire.sh stop
#   tripwire.sh status              # running?  how many spikes caught?  latest ones
#
# ONE SNAPSHOT PER SPIKE, not per poll. It arms on crossing above `threshold`,
# writes one ps + rocm-smi snapshot, then will not fire again until load has
# dropped back under `threshold/2` (hysteresis) -- without that, a five-minute
# sustained spike would write one file every 5 seconds for no extra information.
#
# Polls every 5 s, deliberately faster than sar's 10 min and our own watchdog's
# 20 s: the point is to have a real chance of landing inside a spike, not just
# confirming afterward (via the queue's own kill log) that one happened.
set -uo pipefail
# Resolve to an absolute path BEFORE cd -- $0 is whatever the caller typed (often
# relative), and re-execing a relative $0 after changing directories fails silently.
SELF=$(readlink -f "${BASH_SOURCE[0]}")
cd "$(dirname "$(readlink -f "$0")")/../../../../.." || exit 1
OUT=tools/results/overlay/pmc_workload/_queue/contention
PIDF=tools/results/overlay/pmc_workload/_queue/.tripwire.pid
mkdir -p "$OUT"

loadi() { awk '{print int($1)}' /proc/loadavg; }

snapshot() {                    # load threshold
  local f="$OUT/$(date +%Y%m%d_%H%M%S)_load$1.txt"
  {
    echo "=== $(date) -- load crossed $2, now $(cat /proc/loadavg) ==="
    echo
    echo "--- uptime ---"; uptime
    echo
    echo "--- top 40 by %CPU, every user ---"
    ps -eo user:16,pid,ppid,pcpu,pmem,etime,cmd --no-headers --sort=-pcpu | head -40
    echo
    echo "--- process COUNT by user (a fork storm shows here even if each is cheap) ---"
    ps -eo user:16 --no-headers | sort | uniq -c | sort -rn
    echo
    echo "--- GPU use ---"
    rocm-smi --showuse 2>/dev/null | grep -E "GPU\[|use"
  } > "$f"
  echo "$f"
}

run() {                         # threshold
  local thresh=${1:-100} clear=$(( ${1:-100} / 2 )) armed=1 l f
  echo "$$" > "$PIDF"
  echo "[tripwire] $(date +%H:%M:%S) watching: fire above $thresh, rearm below $clear, pid $$"
  while :; do
    l=$(loadi)
    if   [[ $armed -eq 1 && $l -gt $thresh ]]; then
      f=$(snapshot "$l" "$thresh")
      echo "[tripwire] $(date +%H:%M:%S) SPIKE load=$l -- wrote $f"
      armed=0
    elif [[ $armed -eq 0 && $l -lt $clear ]]; then
      echo "[tripwire] $(date +%H:%M:%S) cleared, load=$l -- rearmed"
      armed=1
    fi
    sleep 5
  done
}

case "${1:-status}" in
  start)
    if [[ -s $PIDF ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
      echo "already running, pid $(cat "$PIDF")"; exit 0
    fi
    setsid nohup "$SELF" _run "${2:-100}" >> "$OUT/tripwire.log" 2>&1 < /dev/null &
    disown
    sleep 1
    echo "started, pid $(cat "$PIDF" 2>/dev/null || echo '?'), threshold=${2:-100}, log=$OUT/tripwire.log"
    ;;
  _run) run "${2:-100}" ;;
  stop)
    if [[ -s $PIDF ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
      kill "$(cat "$PIDF")"; echo "stopped"
    else
      echo "not running"
    fi
    rm -f "$PIDF"
    ;;
  status)
    if [[ -s $PIDF ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
      echo "running, pid $(cat "$PIDF")"
    else
      echo "not running"
    fi
    echo "spikes caught: $(ls "$OUT"/*_load*.txt 2>/dev/null | wc -l)"
    ls -t "$OUT"/*_load*.txt 2>/dev/null | head -5 | sed 's/^/  /'
    ;;
  *) sed -n '2,17p' "${BASH_SOURCE[0]}"; exit 2;;
esac
