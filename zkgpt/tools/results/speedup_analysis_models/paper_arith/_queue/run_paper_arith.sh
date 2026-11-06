#!/bin/bash
# Paper-arithmetization sweep: CPU witgen + CPU prove (arm1), GPU witgen + GPU prove (arm4). main_gkr_split proves modular-zkp's exact statement (LLMSplitBlock, probe half/half layout, merge=1,
# merged range set) while keeping our runner: forked --overlapped partitions,
# concurrent witgen, GPU witgen / GPU prover gates, same knobs as speedup_analysis*.
# Baseline for the figures = modular-zkp's own binary (results/same_arith/logs/<m>_mz_cpu.log).
set -u
cd "$(dirname "$(readlink -f "$0")")/../../../../.."
LOGS=${LOGS:-tools/results/speedup_analysis_models/paper_arith/logs}; mkdir -p "$LOGS"   # LOGS=... to write elsewhere (figure reruns)
# ROCm 7 runtime: libamdhip64.so.7 comes from /opt/rocm-scale/lib, libomp from /opt/rocm/llvm/lib.
export LD_LIBRARY_PATH="/opt/rocm-scale/lib:/opt/rocm/lib:/opt/rocm/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GKR_EARLY_CARRY=1   # hand the carry to partition i+1 as soon as its input values are final
unset KMP_BLOCKTIME   # NOT zkMod-specific: it also speeds the gpt2 T=192 monolith (234.3 -> 215.7 s), so the arms don't get it either
export ZKGPT_WITNESS_CACHE=0
declare -A SPEC=( [gpt2]="12 768 3072 12" [opt350m]="24 1024 4096 24" [opt1b3]="24 2048 8192 24" [opt2b7]="32 2560 10240 32" )
declare -A MIN_FREE_MIB=( [gpt2]=32768  [opt350m]=65536  [opt1b3]=65536  [opt2b7]=65536 )
# Memory-gate seeds, keyed model:len, = MEASURED peaks (PARTITION host_peak_mib /
# witgen_host_peak_mib / gpu_peak_mib, early-carry runs) + ~10%. The gate
# only uses a seed until partitions finish; after that it re-derives the estimate from
# their observed peaks (x1.1), so a seed matters for the first wave only. A
# "len64 value x LEN/64" rule over-estimates by up to 3.3x (gpt2 len256: 102 vs 31 GiB),
# so it is kept only as the fallback for an unmeasured model:len.
# opt350m defaults to 24 partitions (1 layer each): the partition sweep
# put it at 206 s vs 640 s at 6 partitions under the strict gate. Its seeds are
# the measured 24-partition peaks (len64: host 22.0 GiB, built 6.2, VRAM 8.9); len128/256
# scaled by the 6-partition len ratios. The gate re-derives from finished partitions.
# opt1b3 -> 24 and opt2b7 -> 32 partitions (1 layer each) (s24: 544 vs
# 1201 s at 12; s32: 2960 vs 5349 s at 16). GPU-prove seeds = the measured s24/s32 arm4
# peaks + 10% (opt1b3 len64 host 44.6 / built 12.2 / VRAM 15.5 GiB, len128 89.4 / 24.2 /
# 28.9; opt2b7 188.5 / 38.8 / 20.4). CPU-prove seeds are still the 12/16-partition values
# (unmeasured at 1 layer; an over-seed only slows the first wave).
declare -A PEAK_GPU_MIB=( [gpt2:64]=25000 [gpt2:128]=28000 [gpt2:256]=35000
                          [opt350m:64]=25000 [opt350m:128]=28000 [opt350m:256]=35000
                          [opt1b3:64]=49000 [opt1b3:128]=98500 [opt2b7:64]=207500 )   # host, GPU-prove arms
declare -A PEAK_CPU_MIB=( [gpt2:64]=28000 [opt350m:64]=30000 [opt1b3:64]=112000 [opt2b7:64]=400000 )   # host, CPU-prove arms
declare -A BUILT_MIB=(    [gpt2:64]=7000 [gpt2:128]=9000 [gpt2:256]=14000
                          [opt350m:64]=7000 [opt350m:128]=9500 [opt350m:256]=19000
                          [opt1b3:64]=13500 [opt1b3:128]=26700 [opt2b7:64]=43000 )
# gpt2:64 / opt350m:64 = measured GPU-prove peak 5.30 GiB/partition (gpt2 + opt350m arm4) + 10%;
# a larger seed over-reserves and holds proving slots back on the VRAM gate.
declare -A PEAK_VRAM_MIB=( [gpt2:64]=5900 [gpt2:128]=11500 [gpt2:256]=14500
                           [opt350m:64]=5900 [opt350m:128]=12000 [opt350m:256]=16000
                           [opt1b3:64]=17000 [opt1b3:128]=32000 [opt2b7:64]=14500 )   # one proving partition, sumcheck on GPU
# opt2b7 VRAM 14500 = the GPU-Lasso 1-layer test with the streamed fold-dst (13.0 GiB) + 10%
# seed <table> <model>: model:LEN if measured, else the model:64 value scaled by LEN/64.
seed(){ local -n T=$1; local v=${T[$2:$LEN]:-}; [[ -n $v ]] && { echo "$v"; return; }
        v=${T[$2:64]:-}; [[ -z $v ]] && v=${PEAK_GPU_MIB[$2:64]}; echo $(( v * (LEN > 64 ? LEN / 64 : 1) )); }
# Threads per partition: every partition shares ONE pool of 192 threads, enforced by
# the binary (child omp/max_threads = GKR_TPC, gate admits
# at most 192/GKR_TPC children). Default = 192 / partitions: the gpt2 arm1 sweep (1..32 per
# partition) was fastest there (16: 66.4 s; 8: 87.9; 32: 85.5). opt2b7 at its default split is the
# exception: host RAM admits ~3 provers, so 192/32 = 6 would idle ~170 cores -> 64 (3 alive).
# TPC=<n> overrides for every arm and model.
ZK_THREADS=${ZK_THREADS:-192}   # the shared pool (192 on the MI210 box, 48 on an H100 node)
tpc_for(){ local t=$(( ZK_THREADS / $2 )); (( t < 1 )) && t=1
           [[ $1 == opt2b7 && -z "${SPLITS:-}" ]] && t=$(( ZK_THREADS / 3 ))
           [[ -n "${TPC:-}" ]] && t=$TPC; echo $t; }
note(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }
LEN=${LEN:-64}; SUF=""; [[ $LEN -ne 64 ]] && SUF="_len${LEN}"   # seq-length sweep: LEN=128 ./run_paper_arith.sh gpt2
ARMS=${ARMS:-"1 4"}                                              # ARMS="4" to run only GPU+GPU
# EXTRA="..." is appended to every main_gkr_split command (e.g. the --vram-csv profiling runs).
# PROFILE=1: every run also writes a 10 ms memory profile next to its log --
#   <log>.vram.csv     device-wide VRAM of HIP device 0
#   <log>.vram_cpu.csv per-PID rss_pages + vram_b (KFD per-process VRAM)
# on the same clock as the log's PHASE markers, to find where in a partition its peaks land.
if [[ "${PROFILE:-0}" == 1 ]]; then
  export ZKGPT_SAMPLER_PCI=0000:43:00.0 ZKGPT_SAMPLER_KFD_GPU_ID=57300   # HIP device 0 (KFD topology node 2)
fi
# BIN=<path> runs another binary (a profiling queue pins its own copy so a rebuild cannot swap it mid-queue).
pa(){ [[ "${PROFILE:-0}" == 1 ]] && echo "--vram-csv ${1%.log}.vram.csv --vram-interval 0.01"; }
MODELS=("$@"); [[ ${#MODELS[@]} -eq 0 ]] && MODELS=(gpt2 opt350m opt1b3 opt2b7)
# A run is done when every partition verified: forked arms print status=OK, the
# splits=1 sequential path prints no SPLIT_OVERLAPPED line at all.
done_ok(){ grep -q "status=OK" "$1" 2>/dev/null || { [[ "${SPLITS:-}" == 1 ]] && grep -q "All verification passed" "$1" 2>/dev/null; }; }
for m in "${MODELS[@]}"; do
  read -r L attn lin splits <<< "${SPEC[$m]}"
  # SPLITS=<n> overrides the per-model partition count (partition-scaling sweep); the
  # log name then carries _s<n>. splits=1 has no --overlapped (needs >= 2 partitions):
  # it runs the sequential path, same GPU gates, no memory gate.
  OVL="--overlapped -1"
  if [[ -n "${SPLITS:-}" ]]; then splits=$SPLITS; SUF="${SUF%_s*}_s${splits}"; [[ $splits -eq 1 ]] && OVL=""; fi
  # Partition-count sweep: seeds were measured at the default count. FEWER partitions
  # (bigger ones) scale the seed up by layers-per-partition; MORE partitions keep the
  # default's measured seed -- never guess a peak below one we measured (opt2b7's ~370 GiB
  # is mostly a Lasso domain that may not shrink with the partition, and an under-seed
  # over-admits the first wave). The gate lowers it from the first finished partition.
  dsplits=$(awk '{print $4}' <<< "${SPEC[$m]}")
  f100=$(( 100 * dsplits / splits )); (( f100 < 100 )) && f100=100
  PK_GPU=$(( $(seed PEAK_GPU_MIB $m) * f100 / 100 )); PK_CPU=$(( $(seed PEAK_CPU_MIB $m) * f100 / 100 ))
  PK_BUILT=$(( $(seed BUILT_MIB $m) * f100 / 100 )); PK_VRAM=$(( $(seed PEAK_VRAM_MIB $m) * f100 / 100 ))
  (( PK_VRAM > 60000 )) && PK_VRAM=60000
  TPCm=$(tpc_for $m $splits)
  note "$m len$LEN splits=$splits tpc=$TPCm gate seeds (MiB): host_gpuarm=$PK_GPU host_cpuarm=$PK_CPU built=$PK_BUILT vram=$PK_VRAM"
  export GKR_MIN_FREE_MIB="${MIN_FREE_MIB[$m]}" GKR_PART_BUILT_MIB="$PK_BUILT"
  # arm1: CPU witgen + CPU prove
  if [[ " $ARMS " != *" 1 "* ]]; then :; elif done_ok "$LOGS/${m}${SUF}_arm1_cpuwg_cpuprove.log"; then note "$m arm1 already done -- skip"; else
  note "=== $m len$LEN arm1 ==="
  ( unset HIP_VISIBLE_DEVICES GKR_BFEVAL_GPU GKR_WITGEN_GPU GKR_ROUND_GPU GKR_SC_STREAM_BITS
    export GKR_LASSO_CPU=1 GKR_MATMUL_CPU=1 GKR_PRED_CPU=1 GKR_SC_CPU=1 GKR_PART_PEAK_MIB="$PK_CPU" GKR_TPC="$TPCm"
    "${BIN:-./bin/main_gkr_split}" "$L" "$attn" "$splits" "$LEN" "$ZK_THREADS" "$lin" $OVL --c 9 --cpu --cpu-deg3 --cpu-open --cpu-commit ${EXTRA:-} $(pa "$LOGS/${m}${SUF}_arm1_cpuwg_cpuprove.log") ) > "$LOGS/${m}${SUF}_arm1_cpuwg_cpuprove.log" 2>&1
  note "$m len$LEN arm1 rc=$? $(grep -oE 'overlapped_total_s=[0-9.]+ status=\w+' $LOGS/${m}${SUF}_arm1_cpuwg_cpuprove.log) pass=$(grep -c 'All verification passed' $LOGS/${m}${SUF}_arm1_cpuwg_cpuprove.log)"
  fi
  # GPU prove: GPU_MAX_HW_QUEUES=1 + GKR_GPU_PROCS=12 keep the MI210's HW
  # scheduler under its 8-process / 24-queue limit (above it the runlist time-slices every per-round
  # sync: opt350m arm4 125 -> 82.5 s, Lasso sumcheck 11 -> 1.6 s); pinned (GKR_PIN_CORES=1, slot-claim).
  # The CPU monolith uses no GPU, so neither touches the baseline.
  # arm4: GPU witgen + GPU prove
  if [[ " $ARMS " != *" 4 "* ]]; then :; elif done_ok "$LOGS/${m}${SUF}_arm4_gpuwg_gpuprove.log"; then note "$m arm4 already done -- skip"; else
  note "=== $m len$LEN arm4 ==="
  ( unset GKR_LASSO_CPU GKR_MATMUL_CPU GKR_PRED_CPU GKR_SC_CPU
    export GPU_MAX_HW_QUEUES=1 GKR_GPU_PROCS="${GKR_GPU_PROCS:-12}" GKR_PIN_CORES=1 GKR_BFEVAL_GPU=1 GKR_SC_STREAM_BITS=22 GKR_WITGEN_GPU=1 GKR_ROUND_GPU=1 HIP_VISIBLE_DEVICES="${FIG_GPU:-0}" GKR_PART_VRAM_MIB="$PK_VRAM" GKR_PART_PEAK_MIB="$PK_GPU" GKR_TPC="$TPCm"
    "${BIN:-./bin/main_gkr_split}" "$L" "$attn" "$splits" "$LEN" "$ZK_THREADS" "$lin" $OVL --c 18 --matmul-k 4 --range-k 8 --lasso-k 8 --sc-tile-bits 18 ${EXTRA:-} $(pa "$LOGS/${m}${SUF}_arm4_gpuwg_gpuprove.log") ) > "$LOGS/${m}${SUF}_arm4_gpuwg_gpuprove.log" 2>&1
  note "$m len$LEN arm4 rc=$? $(grep -oE 'overlapped_total_s=[0-9.]+ status=\w+' $LOGS/${m}${SUF}_arm4_gpuwg_gpuprove.log) pass=$(grep -c 'All verification passed' $LOGS/${m}${SUF}_arm4_gpuwg_gpuprove.log)"
  fi
done
note ALL_DONE
