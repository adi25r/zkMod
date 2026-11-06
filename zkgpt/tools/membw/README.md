# membw -- whole-run memory-bandwidth / residency timelines

Answers: *how hard does a run drive host DRAM and GPU memory, and when* -- for a monolithic baseline vs the partitioned
prover, at 10 ms resolution, over the whole process tree (not one partition). Written for the AMD MI210 box; the
collectors here also run on NVIDIA. **Nothing in the prover is ported** -- you run your own binary under `wrap.sh`.

## What to collect (per run)

| signal | tool | file | meaning |
|---|---|---|---|
| host DRAM read GB/s | `perf stat -I 10` (whole process tree, forks included) | `<prefix>.cpu.csv` | cache-line fills from DRAM x 64 B. **Reads only** -- write-backs and device DMA are not counted, so it is a lower bound |
| GPU memory activity % | NVML `utilization.memory` (`sampler_nvidia.py`) | `<prefix>.gpu.csv` col 2 | % of the sample period the memory was being read/written. **Activity, not bandwidth** |
| GPU busy % | NVML `utilization.gpu` | col 3 | |
| GPU residency | NVML `memory.used` | col 4 | bytes allocated on the device (whole device, so run on an otherwise idle card) |
| phase timeline (optional) | the program's own log | `<prefix>.log` | lines `PHASE part=N pid=P ev=begin\|end phase=NAME d=D t_us=EPOCH_US`; without them you just get no timeline strip |

Collect at least: **baseline run, your system's run(s)** -- same input, one at a time, nothing else on the box.

## Run

```bash
pip install nvidia-ml-py matplotlib numpy          # sampler + plots
export GPU_VENDOR=nvidia GPU_INDEX=0               # amd -> sampler_amd.py (edit its PCI address D); none -> CPU only
export PERF_EVENTS=...                             # see "CPU counters" -- default is AMD Zen4 and WILL NOT WORK on Intel
mkdir -p logs
./wrap.sh logs/baseline  ./your_baseline  <args>
./wrap.sh logs/ours      ./your_system    <args>
python3 plot_membw.py --logs logs --out figs --run baseline="Baseline" --run ours="Ours" \
        [--skip-phase probe_witgen] [--ddr-peak <host DRAM GB/s>] [--bytes-per-count 64]
```

Outputs in `figs/`: `membw_overview.{png,pdf,md}` (all runs overlaid + summary table: mean/peak GB/s, % of DRAM peak,
total TB read, GPU activity, VRAM peak) and `timeline_<prefix>.{png,pdf}` (phase strip over CPU DRAM / GPU activity / VRAM,
one shared clock). `--skip-phase X` starts every clock when phase X ends (we used it to drop a 300 s throwaway sizing pass).
`--ddr-peak` is only the dotted reference line and the "% of peak" column: set it to your host's DRAM peak.

## CPU counters -- CALIBRATE BEFORE TRUSTING ANY NUMBER

The perf event choice decides the answer. On Zen4 the obvious event (`ls_any_fills_from_sys.dram_io_all`) captured only
~25% of a streaming read: hardware L2 prefetch fills bypass it. Adding `l2_pf_miss_l2_l3.all` fixed it (~1.1x true reads).
No uncore/data-fabric PMU was exposed on our box, so this is the core-side proxy.

1. Pick events for your CPU and put them in `PERF_EVENTS` (counts must be DRAM transfers; `--bytes-per-count` = bytes per count).
   * AMD Zen4 (default): `ls_any_fills_from_sys.dram_io_all,l2_pf_miss_l2_l3.all`, 64 B.
   * Intel: look for the IMC uncore events `uncore_imc_*/cas_count_read/` (+ `cas_count_write/` for writes) -- these give real DRAM
     traffic incl. writes; note perf may report them in MiB, not counts (then use `--bytes-per-count 1048576`). Verify with step 2.
2. Calibrate: `gcc -O2 -fopenmp calibrate_stream.c -o calib && OMP_NUM_THREADS=<cores> perf stat -x, -e $PERF_EVENTS ./calib`.
   The program prints how many GB it read (~43 GB, plus ~9 GB of first-touch); the counts x 64 B must be within ~10-20% of that.
   If not, change events before collecting anything.

Needs `perf_event_paranoid <= 0` (or root) so perf can attach to the child's whole process tree.

## GPU side on NVIDIA -- what the NVML numbers are and are not

* `utilization.memory` is an activity gauge and NVML refreshes it only every ~1/6-1 s depending on driver, so 10 ms polling
  returns repeated values: treat its fine structure as unreliable. (On the MI210 the equivalent gauge changed every ~4 ms and
  topped out near 60% even in memory-heavy phases; it is **not** a fraction of peak bandwidth.)
* For real bytes/s on the device, collect one of these instead (they are not wired into `plot_membw.py`; plot them yourself
  on the same epoch clock):
  * **Nsight Systems** (best time resolution): `nsys profile --gpu-metrics-devices=all --gpu-metrics-frequency=10000 -o trace <cmd>`
    -- includes DRAM read/write bandwidth per sample on data-centre GPUs (needs admin rights / `perf_event_paranoid`; check
    your GPU is supported -- `nsys profile --gpu-metrics-devices=help`).
  * **DCGM**: `dcgmi dmon -e 1005 -d 100` (`DCGM_FI_PROF_DRAM_ACTIVE`, fraction of cycles the DRAM interface is active);
    ~100 ms minimum.
  * **Nsight Compute** per kernel: `ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`
    -- serialises kernels and, like rocprof, will not follow forked children; use for one-process / one-partition characterisation.
* If the system forks one process per partition, profilers that attach to a single process miss the children; the NVML
  poller and `perf stat` are device-/tree-wide, which is why they are the default here.

## Gotchas

* One run at a time, idle machine (a shared GPU's `memory.used` and any other user's DRAM traffic leak into the curves).
* Do not compare walls to published timings: the wrapper adds ~0 overhead at 10 ms (measured, <1 s on 46-68 s runs) but any
  `--skip-phase` window is removed from the plots and table only, not from the log.
* The sampler buffers 64 KB; the `.gpu.csv` is complete only after the wrapper exits.
* Peaks at 10 ms are noisy (+-15-20%): quote means and totals first, peaks as "up to ~".
