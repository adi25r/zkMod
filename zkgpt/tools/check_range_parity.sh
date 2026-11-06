#!/usr/bin/env bash
# check_range_parity.sh — checks that both arms build the same range-check circuit.
#
# Both arms must print the SAME `start range prove query_size N` for the same
# (model, seq_len). A mismatch (e.g. exactly half) means one arm built a merged
# circuit and passed merged=1 to range_prover while the other did not.
#
# Only the query_size line is needed, and it is printed right after the circuit is
# built -- long before the proof finishes -- so each run is killed as soon as it
# appears rather than proving all 16 cells.
#
# The circuit width is next_pow2(len) on both arms (the binary rounds it).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

GPU="${GPU:-3}"
BIN=./bin/main_gkr_split
CPU_LOGS="${CPU_LOGS:-/nvmepool/pawks/workspace/aditya/zkgpt-gpu/tools/results/model_family/logs/cpu_parallel}"

next_pow2() { local w=1; while [ "$w" -lt "$1" ]; do w=$((w*2)); done; echo "$w"; }

printf "%-9s %5s %6s %14s %14s %s\n" model len cw CPU_query GPU_query verdict
fail=0
for m in "gpt2:768:3072" "opt350m:1024:4096" "opt1b3:2048:8192" "opt2b7:2560:10240"; do
  IFS=':' read -r name embed lin <<< "$m"
  for len in 30 64 128 256; do
    cw=$(next_pow2 "$len")   # report only; the binary rounds len itself
    cpu=$(grep -ho "start range prove query_size [0-9]*" \
            "$CPU_LOGS/${name}_len${len}_p0.log" 2>/dev/null | head -1 | awk '{print $NF}')
    [ -z "$cpu" ] && cpu="-"

    log=$(mktemp)
    HIP_VISIBLE_DEVICES=$GPU timeout 3600 $BIN 1 "$embed" 1 "$len" 32 "$lin" \
        --matmul-k 1 --range-k 1 --lasso-k 8 > "$log" 2>&1 &
    pid=$!
    gpu=""
    for _ in $(seq 1 3600); do
      gpu=$(grep -ho "start range prove query_size [0-9]*" "$log" 2>/dev/null | head -1 | awk '{print $NF}')
      [ -n "$gpu" ] && break
      kill -0 $pid 2>/dev/null || break
      sleep 1
    done
    kill -TERM $pid 2>/dev/null; wait $pid 2>/dev/null
    rm -f "$log"
    [ -z "$gpu" ] && gpu="-"

    if [ "$cpu" = "$gpu" ] && [ "$cpu" != "-" ]; then v="MATCH"; else v="** MISMATCH **"; fail=1; fi
    printf "%-9s %5s %6s %14s %14s %s\n" "$name" "$len" "$cw" "$cpu" "$gpu" "$v"
  done
done
echo
[ "$fail" -eq 0 ] && echo "ALL CELLS MATCH -- arms prove the same range statement" \
                  || echo "MISMATCHES PRESENT -- arms still differ"
exit $fail
