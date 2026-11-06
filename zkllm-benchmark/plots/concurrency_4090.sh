#!/bin/bash
# plots/concurrency_4090.sh <model> <embed> <hidden> <gpu> <wstream 0|1> <N>
# Proves layers 0..N-1 as N concurrent one-layer provers on one GPU and prints a CONC line
# (throughput in layers/min; a prover that runs out of memory fails with status=FAIL).
# Needs a running MPS daemon: set CUDA_MPS_PIPE_DIRECTORY to its pipe directory. Outputs and
# act-pp-<embed>.bin (./ppgen <embed> ...) live in $OUT (default /tmp/zkllm-conc).
S="${OUT:-/tmp/zkllm-conc}"
mkdir -p "$S"
cd "$(dirname "$0")/.."
: "${CUDA_MPS_PIPE_DIRECTORY:?set CUDA_MPS_PIPE_DIRECTORY to the pipe directory of the MPS daemon}"
[ -f "$S/act-pp-$2.bin" ] || CUDA_VISIBLE_DEVICES=$4 ./ppgen "$2" "$S/act-pp-$2.bin" > /dev/null
m=$1; E=$2; H=$3; gpu=$4; ws=$5; N=$6
export CUDA_MPS_PIPE_DIRECTORY CUDA_VISIBLE_DEVICES=$gpu ZKLLM_WSTREAM=$ws
mem=$S/conc_${m}_ws${ws}_n${N}.mem; : > $mem
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu >> $mem; sleep 0.2; done ) & mon=$!
t0=$(date +%s.%N)
for ((l=0; l<N; l++)); do
  o=$S/conc_${m}_ws${ws}_n${N}_l$l
  ./partition opt-${m}_input_2048.bin 2048 $E $H zkllm-workdir/opt-$m $l 1 $o.bin prove --arch opt --swiglu-table relu-table.bin \
    --part-idx $l --act-pp $S/act-pp-$E.bin --commit-input 1 --commit-output 1 --prefix $o --eps 1e-05 > $o.stdout 2> $o.stderr &
done
wait $(jobs -p | grep -v "^$mon$") 2>/dev/null
t1=$(date +%s.%N); kill $mon
ok=$(cat $S/conc_${m}_ws${ws}_n${N}_l*.stdout | grep -c "status=OK")
echo "CONC model=$m ws=$ws N=$N ok=$ok/$N wall_s=$(echo "$t1-$t0" | bc) layers_per_min=$(echo "scale=2; $ok*60/($t1-$t0)" | bc) smi_peak_MiB=$(sort -n $mem | tail -1)"
grep -h "partition:" $S/conc_${m}_ws${ws}_n${N}_l*.stderr | head -2
