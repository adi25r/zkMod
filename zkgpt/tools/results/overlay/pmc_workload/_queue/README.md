# Overnight profiling queue

Two chained jobs, launched 2026-09-06 03:42 under `nohup setsid` (PPID 1, no
controlling TTY), so they outlive the session that started them. Scripts and logs
live here rather than in a session scratchpad for the same reason.

    queue_cpubase.sh  -> cpubase.log    runs FIRST
    queue_seqlen.sh   -> seqlen.log     waits for the above, then runs

## 1. CPU baseline characterisation  -> ../cpu_mono/

zkGPT as the baseline actually runs it: monolithic (`splits=1`), every GPU
offload forced onto its CPU reference arm, witgen on the host, 192 threads.
Same circuit and knobs as the GPU arms, so only the execution path differs.

    GKR_MATMUL_CPU=1 GKR_LASSO_CPU=1 GKR_PRED_CPU=1
    main_gkr_split 12 768 1 64 192 3072 --c 18 --matmul-k 4 --range-k 8
                   --lasso-k 8 --sc-tile-bits 18 --cpu --cpu-open
                   --cpu-commit --cpu-deg3

rocprofv3 cannot profile this -- it counts kernel dispatches and there are none.
perf is the equivalent, and the metrics map onto the GPU table:

| GPU table | here |
|---|---|
| per-kernel time | per-function (`perf record -g`, dwarf) -> perf_report.txt |
| cyc/instr per wave | CPI = cycles / instructions -> perf_stat.csv |
| VALU busy % | IPC |
| L2 hit % | LLC-loads vs LLC-load-misses |
| HBM GB/s | LLC misses x line size |

**The dedicated L3 PMU is NOT available on this box.** `perf list` advertises
`l3_cache_accesses` / `l3_misses` from its CPU-model event tables, but no `amd_l3`
device exists under /sys/bus/event_source/devices (only cpu, msr, ibs_*,
amd_iommu_*), so those events cannot be counted at any spelling -- queue_cpul3.sh
failed with "Bad event name" for exactly this reason and is not worth retrying.
The generic `cpu` PMU's cache-references / cache-misses give the same two numbers
and were already collected in the main pass:

    cache-references  180.0 G      cache-misses  6.6 G  -> LLC hit 96.32%
    memory traffic    423.9 GB (misses x 64 B)  ->  0.64 GB/s over 664 s
    IPC 0.767  CPI 1.304  branch-misses 147.6 G
    context-switches 242.9 M  page-faults 137.6 M

`perf record` inflates wall time; use `perf stat` for totals and `perf record`
only for the shape.

## 2. Longer GPT-2 sequence lengths  -> ../len{128,256}_{zkmod,mono}/

The same three counter passes as the len-64 characterisation, in priority order
so a box that gets busy later still leaves a complete len128 pair:

    len128_zkmod (12 partitions) -> len128_mono (1) -> len256_zkmod -> len256_mono

Each pass: bandwidth (FETCH_SIZE WRITE_SIZE), cache (L2CacheHit MemUnitBusy
MemUnitStalled), compute (VALUBusy SALUBusy MeanOccupancyPerCU SQ_WAVES
SQ_INSTS_VALU). Split into three because six TCC counters do not fit one pass and
rocprofv3 DROPS what does not fit while only warning.

`--len` sets chunk_width via next_pow2(len), and chunk_width IS the circuit width
so these are larger circuits, not just longer prompts. Each pass is
capped at 90 min so one hang cannot block the queue.

## Both wait for a quiet box

Four consecutive checks of zero busy GPUs and load < 12, polled every 3 min. The
first mono attempt on 2026-09-05 ran at load 145-226 with 3-4 GPUs busy and had
to be discarded (kept as ../mono_CONTAMINATED_2026-09-05/), which is why the gate
is this strict. Each pass logs the box state at launch so contamination is
visible after the fact.

## Reading the results

    python3 tools/pmc_by_phase.py <dir>/pmc              # per-phase
    python3 tools/plot_pmc_kernels.py <arm-dir> --all    # per-kernel + usage
    python3 tools/plot_pmc_timeline.py <arm-dir> --all   # timelines

---

# The full characterisation queue (2026-09-06)  -> queue_all.sh

Everything outstanding in one resumable screen job. Supersedes the three scripts
above, which are kept because they document how `cpu_mono`, `cpu_zkmod` and the
`len{128,256}_*` dirs were produced.

    screen -S char
    tools/results/overlay/pmc_workload/_queue/queue_all.sh 2>&1 | tee -a _queue/all.log
    # ctrl-a d to detach, `screen -r char` to return, `cat _queue/STATUS` any time
    queue_all.sh --list          # the stages
    queue_all.sh --only gg64     # one stage
    queue_all.sh --skip cc256    # everything but the long pole

Stages, in run order. Each touches `.done/<stage>` and is skipped on a rerun:

| stage | what |
|---|---|
| `figs0` | per-phase average utilisation for the sequential 12-partition workload and for monolithic zkGPT on the GPU, from counters already on disk |
| `cpuzk` | the `cc_len64` perf figures once its record pass lands (renamed from `cpu_zkmod` 2026-09-09, see below) |
| `gg64` `gg128` `gg256` | zkGPT, GPU witgen + GPU proving, 12 partitions sequential |
| `cg64` `cg128` `cg256` | zkGPT, CPU witgen + GPU proving, 12 partitions sequential |
| `cc128` `cc256` | zkGPT, CPU witgen + CPU proving, 12 partitions sequential (len 64 is `cc_len64`, produced by the `cpuzk` stage) |
| `figs` | every figure regenerated, plus `MATRIX.md` |

## The 3x3

|  | len 64 | len 128 | len 256 |
|---|---|---|---|
| cpu+cpu | `cc_len64` | `cc_len128` | `cc_len256` |
| cpu+gpu | `cg_len64` | `cg_len128` | `cg_len256` |
| gpu+gpu | `gg_len64` | `gg_len128` | `gg_len256` |

**All nine at 192 threads.** The gpu+gpu row is re-run rather than reusing `mono/`,
`len128_mono/` and `len256_mono/`: those ran at **16** threads, because they were
sized for the 12-way overlay comparison (`tpc = 192/12`). GPU proving is nearly
thread-insensitive and witgen is thread-invariant, so the old dirs are not wrong --
but a matrix whose rows differ in thread count cannot be read down a column, and the
GPU row costs ~2 h. The 16-thread dirs are untouched; the overlay figures still cite
them.

Each GPU cell is **four** runs: one unprofiled for the wall time, then the same three
counter passes as before. The unprofiled pass is not optional -- `--pmc` serialises
dispatches (~2.2x here), so a profiled log's `witgen_s` is not a wall time, and
"where is the time saved" is the whole question.

## perf.data: no call graphs, on purpose

`cpu_mono/perf.data` is **82 GB** and `cc_len64/perf.data` (renamed from `cpu_zkmod/`
2026-09-09) 37 GB, entirely from
`--call-graph dwarf,16384` -- 16 KB of stack per sample, 199 Hz, 192 threads. **None
of it is read.** Both figures parse `perf report --no-children -g none --sort
time,sym`, and `perf_report.txt` was generated with `-g none` as well. `queue_all.sh`
records without `-g`, which is what makes the len128 and len256 CPU cells fit on a
volume already at 90 %. Those two files are 114 GB that can be deleted once their
`report_by_time.txt` exists.

Each run stage refuses to start below 250 GB free and logs the box state, so a full
volume shows up as a skipped stage rather than a truncated `perf.data`.

## Stopping and resuming

    _queue/qctl status          what is running, what is done, what is left
    _queue/qctl pause           let the run in flight finish, then wait
    _queue/qctl pause --now     kill the run in flight and wait  (the stage retries)
    _queue/qctl resume
    _queue/qctl stop            finish the run in flight, then exit
    _queue/qctl stop --now      kill the run in flight and exit
    _queue/qctl retry gg128     clear a stage's marker so it runs again

A watchdog samples the box every 20 s during every run and does the same thing on
its own when a neighbour turns up, after three consecutive bad samples.

**It kills rather than suspends, on purpose.** A measurement that shares the box is
not a slow measurement, it is a wrong one, so there is nothing to salvage by pausing
it -- and a SIGSTOPped run still holds its VRAM and its HSA queues, so the neighbour
being yielded to gains nothing either. The killed run's partial output is deleted,
its stage is left unmarked, and a later pass (`--passes`, default 3) retries it.

`--passes` (default 3) bounds how many times a stage the watchdog just killed
mid-run is retried within one launch. If something is still missing after that, the
script exits and lists it -- relaunch by hand (or `qctl retry <stage>` first, to
force a stage past its own marker) to try again.

### "Quiet" means different things to the two arms

Conflating them either poisons the CPU cells or starves the GPU ones. This box is
routinely shared -- two other tenants were on it when this was written.

| | GPU cells (`gg_*`, `cg_*`) | CPU cells (`cc_*`) |
|---|---|---|
| must be free | the **card** | the **host** |
| start gate | 0 busy GPUs, no wedged KFD, load < `LOAD_GPU` (200) | load < `GATE_LOAD_CC` (12), other users < `OTHER_CPU_CC` (400 %) |
| watchdog aborts on | any other GPU busy, load > `LOAD_GPU` | load > `LOAD_CPU` (230), other users > `OTHER_CPU_CC` |

GPU cells tolerate host load because GPU proving is on the device and CPU witgen is
single-threaded in every arm (measured at 1.0 core). The `len{128,256}` passes
already in the tree ran at load 80-193 with the GPUs to themselves and are good
data -- that is the calibration. CPU cells tolerate nothing, because `cpu_mono` was
measured on a quiet box and `cc_len128`/`cc_len256` have to match it or the column
cannot be read down.

All four are env-overridable:

    LOAD_GPU=120 GATE_LOAD_CC=40 _queue/queue_all.sh

## Catching contention in the act

`tripwire.sh` -- the box has no process accounting (no `lastcomm`/`acct`, no
`atop`/`pidstat` history), and `sar`'s own 10-min samples show load spiking to
100-160+ for a few minutes and dropping back to near-zero, repeating every
20-40 min. By the time you run `ps` to look, the spike is already gone. This
polls every 5 s and snapshots who is running the moment load crosses a
threshold, so the next spike gets caught rather than just inferred after the
fact from the queue's own kill log.

    tripwire.sh start [threshold]   # default 100; backgrounds, survives logout
    tripwire.sh stop
    tripwire.sh status              # running? how many spikes caught? latest ones

One snapshot per spike (arms above `threshold`, rearms once load drops back
under `threshold/2`), written to `_queue/contention/<timestamp>_load<N>.txt`:
`ps` sorted by %CPU, a process count by user, and `rocm-smi --showuse`.
