#!/usr/bin/env python3
"""
compare_range.py — CPU vs GPU comparison for the zkgpt-gpu range check (logup),
sweeping the tunable knobs c (chunk bits) and k (Blendy stages). BOTH cpu and
gpu honor c and k (the CPU path uses a staged k-aware mirror of the GPU
blendy_linear sumcheck), so a row pair isolates exactly "host loop vs GPU
kernel" at the same algorithm and (c, k).

Two benchmark modes:

  deg1  (default) — the question "for what model sizes does launching a GPU
        kernel make a difference?". Runs ONLY the degree-1 reciprocal-sum
        sumcheck (the sole GPU-kernel part of logup) at each model's deg1 size
        n = log2(next_pow2(query_size)), sweeping k, cpu vs gpu. This isolates
        the kernel from the (huge, commit-dominated) full pipeline so it scales
        to large-model sizes. Models: gpt2, llama8b, llama405b (+ custom).

  full  — the whole range prover (logup over all chunks) for a given LLM config,
        sweeping c and k. Emits a per-phase breakdown (setup/commit/deg3/deg1/
        open). Commit dominates and scales with query_size, so this is only
        practical for small configs.

  matmul — matmul layer product-sumcheck per-phase sweep (Experiment 1).

Timing comes from a clean (un-profiled) run; peak GPU memory and compute
utilisation come from SEPARATE profiler runs so they don't perturb timing.

Profiler selection (automatic):
  Memory  : rocprofv3 (ROCm) > nvidia-smi poll (CUDA) > 0
  Compute : nsys sqlite (CUDA) > rocprofv3 --pmc (ROCm) > ncu raw counters (CUDA) > zeros

Examples:
  tools/compare_range.py deg1 --models gpt2,llama8b,llama405b --ks 1,2,4,8
  tools/compare_range.py full --config "1 12 64 768 2304" --range-size 48 \\
       --cs auto --ks 1,2,4
  tools/compare_range.py matmul --ns 18,20,22,24 --ks 2,4,8
"""

import argparse
import csv
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BIN = os.path.normpath(os.path.join(HERE, "..", "bin", "zkgpt_gpu"))

RANGE_RE = re.compile(r"RANGE_RESULT\s+mode=(\w+)\s+c=(\d+)\s+k=(\d+)\s+"
                      r"partitions=\d+\s+time_s=([\d.eE+-]+)")
BRK_RE = re.compile(r"RANGE_BREAKDOWN\s+setup_s=([\d.eE+-]+)\s+invert_s=([\d.eE+-]+)"
                    r"\s+commit_s=([\d.eE+-]+)\s+deg3_s=([\d.eE+-]+)\s+deg1_s=([\d.eE+-]+)"
                    r"\s+open_s=([\d.eE+-]+)\s+total_s=([\d.eE+-]+)")
BENCH_RE = re.compile(r"BENCH_DEG1\s+mode=(\w+)\s+n=(\d+)\s+k=(\d+)\s+time_s=([\d.eE+-]+)")
QSIZE_RE = re.compile(r"start range prove\s+query_size\s+(\d+)")
MATMUL_RE = re.compile(r"BENCH_MATMUL\s+mode=(\w+)\s+n=(\d+)\s+k=(\d+)\s+l_stage=(\d+)\s+s=(\d+)"
                       r"\s+time_s=([\d.eE+-]+)\s+expand_s=([\d.eE+-]+)\s+ptable_s=([\d.eE+-]+)"
                       r"\s+round_s=([\d.eE+-]+)\s+fold_s=([\d.eE+-]+)")

# Transformer presets: (layers, heads, head_dim, attn_dim, linear_dim).
MODELS = {
    "gpt2":      (12,  12, 64,   768,   3072),
    "llama8b":   (32,  32, 128,  4096,  14336),
    "llama405b": (126, 128, 128, 16384, 53248),
}
SEQ_LEN = 30   # matches range_prover's canonical config

COMPUTE_METRICS = ("VALUBusy", "MemUnitBusy", "OccupancyPercent")


def next_pow2(n):
    return 1 << (n - 1).bit_length() if n > 1 else 1


def query_size_n(layers, heads, head_dim, attn_dim, linear_dim, seq_len=SEQ_LEN):
    """Replicates range_prover::build() (merged=1) to get the deg1 size n."""
    q = (seq_len + seq_len * attn_dim)
    q += (heads * seq_len * (seq_len + 1) // 2 + seq_len * attn_dim)
    q += (seq_len + seq_len * attn_dim)
    q += (seq_len * linear_dim)
    q = next_pow2(q * layers)
    return int(math.log2(q))


def _profiler():
    """Return the best available AMD/NVIDIA profiler tag, or None."""
    if shutil.which("rocprofv3"):
        return "rocprofv3"
    if shutil.which("ncu"):
        return "ncu"
    return None


# ── peak GPU memory ───────────────────────────────────────────────────────────

def gpu_agent_ids(agent_info_csv):
    ids = set()
    with open(agent_info_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("Agent_Type", "").strip().strip('"') == "GPU":
                ids.add(int(row["Logical_Node_Id"]))
    return ids


def peak_gpu_bytes(trace_csv, agent_info_csv):
    gpus = gpu_agent_ids(agent_info_csv)
    live, cur, peak = {}, 0, 0
    rows = []
    with open(trace_csv, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((int(r["Start_Timestamp"]), r))
    rows.sort(key=lambda x: x[0])
    for _, r in rows:
        op = r["Operation"].strip().strip('"')
        addr = r["Address"]
        if op == "MEMORY_ALLOCATION_ALLOCATE":
            if int(r["Agent_Id"]) in gpus:
                sz = int(r["Allocation_Size"])
                live[addr] = sz
                cur += sz
                peak = max(peak, cur)
        elif op == "MEMORY_ALLOCATION_FREE":
            cur -= live.pop(addr, 0)
    return peak


def _profiled_mem_rocprofv3(binpath, args, env_extra):
    env = dict(os.environ)
    env.update(env_extra)
    outdir = tempfile.mkdtemp(prefix="rocprof_cmp_")
    try:
        cmd = ["rocprofv3", "--memory-allocation-trace", "--output-format", "csv",
               "-d", outdir, "-o", "run", "--", binpath] + args
        subprocess.run(cmd, env=env, capture_output=True, text=True)
        trace = os.path.join(outdir, "run_memory_allocation_trace.csv")
        agents = os.path.join(outdir, "run_agent_info.csv")
        if not (os.path.exists(trace) and os.path.exists(agents)):
            return 0
        return peak_gpu_bytes(trace, agents)
    finally:
        shutil.rmtree(outdir, ignore_errors=True)


def _profiled_mem_nsys(binpath, args, env_extra):
    """Peak GPU memory via nsys's CUDA_GPU_MEMORY_USAGE_EVENTS alloc/free trace.

    Far finer-grained than nvidia-smi polling (event-driven, not 10Hz sampling)
    and, unlike nvidia-smi, only counts bytes this process actually allocated
    on the device instead of whatever the whole GPU happens to be holding —
    but it still sums over the *entire* traced run, so it's only a clean
    per-phase number when binpath/args isolate that phase (e.g. --only-range).
    """
    import sqlite3
    env = dict(os.environ)
    env.update(env_extra)
    outdir = tempfile.mkdtemp(prefix="nsys_mem_")
    try:
        cmd = [
            "nsys", "profile",
            "--trace=cuda",
            "--cuda-memory-usage=true",
            "--export=sqlite",
            "-o", os.path.join(outdir, "run"),
            "--force-overwrite", "true",
            "--", binpath,
        ] + args
        subprocess.run(cmd, env=env, capture_output=True, text=True)
        db = os.path.join(outdir, "run.sqlite")
        if not os.path.exists(db):
            return 0
        conn = sqlite3.connect(db)
        try:
            rows = conn.execute(
                "SELECT address, bytes, memoryOperationType "
                "FROM CUDA_GPU_MEMORY_USAGE_EVENTS ORDER BY start"
            ).fetchall()
        except sqlite3.OperationalError:
            return 0
        live, cur, peak = {}, 0, 0
        for addr, nbytes, op in rows:
            if op == 0:      # CUDA_DEV_MEM_EVENT_OPR_ALLOCATION
                live[addr] = nbytes
                cur += nbytes
                peak = max(peak, cur)
            else:             # CUDA_DEV_MEM_EVENT_OPR_DEALLOCATION
                cur -= live.pop(addr, 0)
        return peak
    finally:
        shutil.rmtree(outdir, ignore_errors=True)


def profiled_mem_nvsmi(binpath, args, env_extra):
    """Peak process GPU memory by polling nvidia-smi while the binary runs."""
    import threading
    import time
    env = dict(os.environ)
    env.update(env_extra)
    proc = subprocess.Popen([binpath] + args, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    peak = [0]
    stop = threading.Event()

    def _poll():
        while not stop.is_set():
            try:
                r = subprocess.run(
                    ["nvidia-smi",
                     "--query-compute-apps=pid,used_gpu_memory",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2)
                for line in r.stdout.splitlines():
                    parts = [x.strip() for x in line.split(",")]
                    if len(parts) == 2 and parts[0].isdigit():
                        if int(parts[0]) == proc.pid:
                            try:
                                peak[0] = max(peak[0], int(parts[1]) * 1024 * 1024)
                            except ValueError:
                                pass
            except Exception:
                pass
            stop.wait(0.1)

    t = threading.Thread(target=_poll, daemon=True)
    t.start()
    proc.wait()
    stop.set()
    t.join(timeout=2)
    return peak[0]


def profiled_mem(binpath, args, env_extra):
    """Peak GPU memory: rocprofv3 (ROCm) > nsys event trace (CUDA) > nvidia-smi poll > 0."""
    if shutil.which("rocprofv3"):
        return _profiled_mem_rocprofv3(binpath, args, env_extra)
    if shutil.which("nsys"):
        return _profiled_mem_nsys(binpath, args, env_extra)
    if shutil.which("nvidia-smi"):
        return profiled_mem_nvsmi(binpath, args, env_extra)
    return 0


# ── compute utilisation metrics ───────────────────────────────────────────────
# VALUBusy        = % GPU-busy time the vector/shader ALUs are issuing
# MemUnitBusy     = % GPU-busy time the memory unit is active
# OccupancyPercent= wave/warp occupancy vs hardware max
#
# Collected in a SEPARATE run from timing so profiler overhead doesn't skew it.
# Preference: nsys sqlite (CUDA, no kernel replay) > rocprofv3 --pmc (ROCm) >
#             ncu raw counters (CUDA, fallback with replay overhead)

# ── nsys (Nsight Systems) path — CUDA, no kernel replay ──────────────────────
# Samples hardware counters at ~10 kHz, exports sqlite. Restrict samples to the
# kernel execution window via CUPTI_ACTIVITY_KIND_KERNEL timestamps.
# Metric names for the ad10x set (RTX 4090 / Ada AD102); nsys auto-selects the
# set so we don't need --gpu-metrics-set and the mapping stays portable.
_NSYS_METRIC_MAP = {
    "SM Issue [Throughput %]":                "VALUBusy",
    "DRAM Read Bandwidth [Throughput %]":     "_dram_read",
    "DRAM Write Bandwidth [Throughput %]":    "_dram_write",
    "Compute Warps in Flight [Throughput %]": "OccupancyPercent",
}


def _profiled_compute_nsys(binpath, args, env_extra):
    import sqlite3
    env = dict(os.environ)
    env.update(env_extra)
    outdir = tempfile.mkdtemp(prefix="nsys_cmp_")
    try:
        cmd = [
            "nsys", "profile",
            "--gpu-metrics-devices=cuda-visible",
            "--gpu-metrics-frequency=10000",
            "--export", "sqlite",
            "-o", os.path.join(outdir, "run"),
            "--force-overwrite", "true",
            "--", binpath,
        ] + args
        subprocess.run(cmd, env=env, capture_output=True, text=True)
        db = os.path.join(outdir, "run.sqlite")
        if not os.path.exists(db):
            return {m: 0.0 for m in COMPUTE_METRICS}

        conn = sqlite3.connect(db)

        # Restrict samples to the kernel execution window.
        row = conn.execute(
            "SELECT MIN(start), MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ).fetchone()
        ts_min, ts_max = (row[0], row[1]) if row and row[0] is not None else (None, None)

        name_to_id = dict(conn.execute(
            "SELECT DISTINCT metricName, metricId FROM TARGET_INFO_GPU_METRICS"
        ).fetchall())

        vals = {v: [] for v in _NSYS_METRIC_MAP.values()}
        for metric_name, friendly in _NSYS_METRIC_MAP.items():
            mid = name_to_id.get(metric_name)
            if mid is None:
                continue
            if ts_min is not None:
                samples = conn.execute(
                    "SELECT value FROM GPU_METRICS "
                    "WHERE metricId=? AND timestamp>=? AND timestamp<=?",
                    (mid, ts_min, ts_max),
                ).fetchall()
            else:
                samples = conn.execute(
                    "SELECT value FROM GPU_METRICS WHERE metricId=?", (mid,)
                ).fetchall()
            vals[friendly] = [s[0] for s in samples]

        def avg(key):
            lst = vals.get(key, [])
            return sum(lst) / len(lst) if lst else 0.0

        return {
            "VALUBusy":         avg("VALUBusy"),
            "MemUnitBusy":      max(avg("_dram_read"), avg("_dram_write")),
            "OccupancyPercent": avg("OccupancyPercent"),
        }
    finally:
        shutil.rmtree(outdir, ignore_errors=True)


# ── rocprofv3 path — ROCm hardware counters ───────────────────────────────────
# Kernel-duration-weighted average so short memset kernels don't skew results.
def _profiled_compute_rocprofv3(binpath, args, env_extra, metrics=COMPUTE_METRICS):
    env = dict(os.environ)
    env.update(env_extra)
    outdir = tempfile.mkdtemp(prefix="rocprof_pmc_")
    try:
        cmd = ["rocprofv3", "--pmc", *metrics, "--output-format", "csv",
               "-d", outdir, "-o", "run", "--", binpath] + args
        subprocess.run(cmd, env=env, capture_output=True, text=True)
        cc = os.path.join(outdir, "run_counter_collection.csv")
        if not os.path.exists(cc):
            return {m: 0.0 for m in metrics}
        num = {m: 0.0 for m in metrics}
        den = {m: 0.0 for m in metrics}
        with open(cc, newline="") as f:
            for r in csv.DictReader(f):
                name = r.get("Counter_Name", "").strip().strip('"')
                if name not in num:
                    continue
                try:
                    val = float(r["Counter_Value"])
                    dur = float(r["End_Timestamp"]) - float(r["Start_Timestamp"])
                except (ValueError, KeyError):
                    continue
                if dur <= 0:
                    dur = 1.0
                num[name] += val * dur
                den[name] += dur
        return {m: (num[m] / den[m] if den[m] else 0.0) for m in metrics}
    finally:
        shutil.rmtree(outdir, ignore_errors=True)


# ── ncu path — CUDA fallback (replays kernels, slower) ───────────────────────
# Raw counter pairs: ratio gives the friendly % without needing the deprecated
# .avg.pct_of_peak_sustained_active metric names (removed in ncu 2025.1).
_NCU_RAW_METRICS = [
    "sm__pipe_alu_cycles_active",
    "sm__cycles_elapsed",
    "l1tex__cycles_active",
    "l1tex__cycles_elapsed",
    "sm__maximum_warps_per_active_cycle_pct",
]


def profiled_compute_ncu(binpath, args, env_extra):
    env = dict(os.environ)
    env.update(env_extra)
    result = subprocess.run(
        ["ncu", "--csv", "--metrics", ",".join(_NCU_RAW_METRICS),
         "--target-processes", "all", "--", binpath] + args,
        env=env, capture_output=True, text=True)
    # ncu writes ==PROF== status lines to stdout when run via subprocess; filter them.
    csv_lines = [l for l in result.stdout.splitlines() if not l.startswith("==")]
    vals = {m: [] for m in _NCU_RAW_METRICS}
    try:
        reader = csv.DictReader(csv_lines)
        for row in reader:
            name = row.get("Metric Name", "").strip().strip('"')
            base = name[:-4] if name.endswith(".avg") else name
            if base not in vals:
                continue
            try:
                vals[base].append(float(row.get("Metric Value", "0").replace(",", ".")))
            except ValueError:
                pass
    except Exception:
        pass

    def avg(key):
        lst = vals.get(key, [])
        return sum(lst) / len(lst) if lst else 0.0

    alu     = avg("sm__pipe_alu_cycles_active")
    elapsed = avg("sm__cycles_elapsed")
    l1_act  = avg("l1tex__cycles_active")
    l1_ela  = avg("l1tex__cycles_elapsed")
    occ     = avg("sm__maximum_warps_per_active_cycle_pct")
    return {
        "VALUBusy":         100.0 * alu / elapsed if elapsed else 0.0,
        "MemUnitBusy":      100.0 * l1_act / l1_ela if l1_ela else 0.0,
        "OccupancyPercent": occ,
    }


def profiled_compute(binpath, args, env_extra, metrics=COMPUTE_METRICS):
    """Dispatch to best available compute profiler: nsys > rocprofv3 > ncu."""
    if shutil.which("nsys"):
        return _profiled_compute_nsys(binpath, args, env_extra)
    if shutil.which("rocprofv3"):
        return _profiled_compute_rocprofv3(binpath, args, env_extra, metrics)
    if shutil.which("ncu"):
        return profiled_compute_ncu(binpath, args, env_extra)
    return {m: 0.0 for m in metrics}


# ── run helpers ───────────────────────────────────────────────────────────────
def run(binpath, args, env_extra, regex, group):
    env = dict(os.environ)
    env.update(env_extra)
    out = subprocess.run([binpath] + args, env=env, capture_output=True, text=True)
    m = regex.search(out.stdout)
    if not m:
        return None, out
    return m, out


def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


# ── deg1 micro-benchmark sweep ────────────────────────────────────────────────
def bench_deg1(args):
    models = []
    for name in args.models.split(","):
        name = name.strip()
        if not name:
            continue
        if name not in MODELS:
            sys.exit(f"unknown model '{name}'; known: {list(MODELS)}")
        n = query_size_n(*MODELS[name])
        models.append((name, n))
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    points = {}
    for x in args.ns.split(","):
        x = x.strip()
        if x:
            points.setdefault(int(x), "")
    for name, n in models:
        points[n] = name
    sweep = sorted(points.items())

    print("deg1 micro-benchmark (isolated reciprocal-sum sumcheck)")
    for name, n in models:
        print(f"  model {name:10s} -> deg1 n={n}  (2^{n} = {1<<n:,} elems)")
    print()

    rows = []
    for n, label in sweep:
        if n > args.max_n:
            print(f"  [skip] n={n}{(' ('+label+')') if label else ''} exceeds --max-n {args.max_n}")
            continue
        for k in ks:
            for mode in modes:
                bench_args = ["--bench-deg1", str(n), str(k)]
                m, out = run(args.bin, bench_args, {}, BENCH_RE, 4)
                tag = label or f"n{n}"
                if not m:
                    print(f"  {tag:10s} {mode:3s} n={n} k={k}  FAILED/OOM")
                    continue
                t = float(m.group(4))
                gpu_path = True
                mem = 0 if (args.no_mem or not gpu_path) else profiled_mem(args.bin, bench_args, {})
                comp = ({} if (args.no_compute or not gpu_path)
                        else profiled_compute(args.bin, bench_args, {}))
                rows.append(dict(bench="deg1", model=label, n=n, c="", k=k,
                                 mode=mode, time_s=t, peak_gpu_bytes=mem,
                                 valu_busy=comp.get("VALUBusy", ""),
                                 mem_busy=comp.get("MemUnitBusy", ""),
                                 occupancy=comp.get("OccupancyPercent", "")))
                cstr = (f"  valu={comp['VALUBusy']:.1f}% mem={comp['MemUnitBusy']:.1f}% "
                        f"occ={comp['OccupancyPercent']:.1f}%" if comp else "")
                print(f"  {tag:10s} {mode:3s} n={n:<3d} k={k:<3d} "
                      f"time={t:8.4f}s  peak_gpu={mem/1024/1024:9.2f} MiB{cstr}")
    return rows


# ── full pipeline sweep ───────────────────────────────────────────────────────
def bench_full(args):
    cfg = args.config.split()
    rsize = args.range_size
    cs = ([d for d in divisors(rsize) if 2 <= d <= args.cap_c]
          if args.cs == "auto" else [int(x) for x in args.cs.split(",") if x.strip()])
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    print(f"full pipeline  config='{args.config}'  range_size={rsize}  "
          f"cs={cs}  ks={ks}  modes={modes}\n")

    rows = []
    for mode in modes:
        for c in cs:
            for k in ks:
                run_args = cfg + [str(c), str(k), str(rsize), "--only-range"]
                if mode == "cpu":
                    run_args.append("--cpu")
                m, out = run(args.bin, run_args, {}, RANGE_RE, 4)
                if not m:
                    print(f"  {mode:3s} c={c} k={k}  FAILED")
                    continue
                t = float(m.group(4))
                bm = BRK_RE.search(out.stdout)
                brk = dict(setup=float(bm.group(1)), invert=float(bm.group(2)),
                           commit=float(bm.group(3)), deg3=float(bm.group(4)),
                           deg1=float(bm.group(5)), open=float(bm.group(6))) if bm else {}
                # Peak GPU mem is governed by the (always-GPU) Hyrax commit, so
                # it is identical for cpu/gpu deg1 modes — only profile once (gpu).
                gpu_path = True
                mem = 0 if (args.no_mem or not gpu_path) else profiled_mem(args.bin, run_args, {})
                comp = ({} if (args.no_compute or not gpu_path)
                        else profiled_compute(args.bin, run_args, {}))
                # Lookup-argument quantities, following plan.tex §"Lasso" exactly:
                #   constraints = query_size (parsed from prover); scales with L.
                #   w = chunks per value = 2*ceil(range_bits/c).
                #   M = constraints * w, N = 2^c.
                qm = QSIZE_RE.search(out.stdout)
                constraints = int(qm.group(1)) if qm else 0
                w = 2 * -(-rsize // c)
                M_lookups = constraints * w if constraints else ""
                rows.append(dict(bench="full", c=c, k=k, mode=mode,
                                 time_s=t, peak_gpu_bytes=mem,
                                 constraints=(constraints or ""), M=M_lookups,
                                 N=(1 << c), w=w,
                                 valu_busy=comp.get("VALUBusy", ""),
                                 mem_busy=comp.get("MemUnitBusy", ""),
                                 occupancy=comp.get("OccupancyPercent", ""),
                                 **{f"{p}_s": brk.get(p, "") for p in
                                    ("setup", "invert", "commit", "deg3", "deg1", "open")}))
                extra = (f"  inv={brk.get('invert',0):.3f} commit={brk.get('commit',0):.2f} "
                         f"deg1={brk.get('deg1',0):.3f}" if brk else "")
                cstr = (f"  valu={comp['VALUBusy']:.1f}% mem={comp['MemUnitBusy']:.1f}% "
                        f"occ={comp['OccupancyPercent']:.1f}%" if comp else "")
                print(f"  {mode:3s} c={c:<3d} k={k:<3d} "
                      f"M={M_lookups} N={1<<c} "
                      f"time={t:8.3f}s  peak_gpu={mem/1024/1024:8.2f} MiB{extra}{cstr}")
    return rows


# ── matmul (Blendy product-sumcheck) per-phase sweep — Experiment 1 ──────────
FE_BYTES = 32   # sizeof(gpu_fe) — BN254 scalar


def matmul_alloc_model(n, k, tile_bits):
    """Predicted device allocation of one product_sumcheck, in field elements.

    Mirrors the `need` expression at gpu_matmul_prover.hpp:84 exactly:

        need = 2*ext + 3N + 2*ptable

    where N = 2^n, l = ceil(n/k), ptable = 3^l, and the extension tables are
    tiled over the future-variable axis so each holds
    ext = 3^l * 2^min(n-l, tile_bits)  (tile_bits < 0 disables tiling).

    The 3N term is the caller's dF/dG (2N, gpu_product_sumcheck_host) plus the
    two Fix-A right-sized fold destinations (2 * N/2). It depends on NEITHER k
    NOR tile_bits — it is the hard floor set by n alone, and it is the term
    plan.tex's eq. memory_peak understates as 2N.

    Returns (need_elems, ext_elems, ptable_elems, floor_elems, l, fv_cap).
    """
    N = 1 << n
    l = (n + k - 1) // k
    fv = n - l                                    # future variables
    fv_cap = tile_bits if (0 <= tile_bits < fv) else fv
    ptable = 3 ** l
    ext = ptable * (1 << fv_cap)
    floor = 3 * N
    return 2 * ext + floor + 2 * ptable, ext, ptable, floor, l, fv_cap


def bench_matmul(args):
    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    tiles = [int(x) for x in args.tiles.split(",") if x.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    budget_bytes = int(args.vram_budget_gb * (1 << 30))
    print(f"matmul per-phase sweep  ns={ns}  ks={ks}  tile_bits={tiles}  "
          f"modes={modes}  vram_budget={args.vram_budget_gb:.1f} GiB\n")
    rows = []
    for mode in modes:
        for n in ns:
            for k in ks:
                if k > n:
                    continue
                # Any t >= n-l materializes the full future axis, i.e. is
                # identical to untiled. Run one point per DISTINCT materialized
                # window (fv_cap) so a wide t list costs no duplicate GPU runs.
                seen_fv = set()
                for tb in tiles:
                    need, ext, ptable, floor, l, fv_cap = matmul_alloc_model(n, k, tb)
                    if fv_cap in seen_fv:
                        continue
                    seen_fv.add(fv_cap)
                    need_gb = need * FE_BYTES / 2**30
                    tag = f"n={n:<3d} k={k:<3d} t={tb:<3d} l={l:<2d}"
                    # Skip only what genuinely cannot fit. Post-tiling this is
                    # usually the 3N floor, not the extension tables.
                    if need * FE_BYTES > budget_bytes:
                        why = ("floor 3N" if floor > 2 * ext else "ext tables")
                        print(f"  {mode:3s} {tag}  SKIP (need {need_gb:.1f} GiB "
                              f"> budget, dominated by {why})")
                        continue
                    bench_args = ["--bench-matmul", str(n), str(k)]
                    env = {"GKR_SC_TILE_BITS": str(tb)}
                    m, out = run(args.bin, bench_args, env, MATMUL_RE, 6)
                    if not m:
                        print(f"  {mode:3s} {tag}  FAILED")
                        continue
                    l_stage, s = int(m.group(4)), int(m.group(5))
                    t, exp, pt, rnd, fold = (float(m.group(i)) for i in (6, 7, 8, 9, 10))
                    mem = 0 if args.no_mem else profiled_mem(args.bin, bench_args, env)
                    comp = {} if args.no_compute else profiled_compute(args.bin, bench_args, env)
                    # time_s is wall-clock around the whole bench and is dominated
                    # by the O(2^n) host CSPRNG vector prep (~70 s at n=29). Fit
                    # the cost model against gpu_s — the four device phases — not
                    # time_s, or alpha absorbs the host setup.
                    gpu_s = exp + pt + rnd + fold
                    rows.append(dict(bench="matmul", mode=mode, n=n, k=k, tile_bits=tb,
                                     l_stage=l_stage, fv_cap=fv_cap, s=s,
                                     gpu_s=gpu_s, time_s=t,
                                     peak_gpu_bytes=mem,
                                     pred_need_bytes=need * FE_BYTES,
                                     pred_ext_bytes=2 * ext * FE_BYTES,
                                     pred_floor_bytes=floor * FE_BYTES,
                                     expand_s=exp, ptable_s=pt, round_s=rnd, fold_s=fold,
                                     ks=k * s,
                                     valu_busy=comp.get("VALUBusy", ""),
                                     mem_busy=comp.get("MemUnitBusy", ""),
                                     occupancy=comp.get("OccupancyPercent", "")))
                    cstr = (f"  valu={comp['VALUBusy']:.1f}% mem={comp['MemUnitBusy']:.1f}% "
                            f"occ={comp['OccupancyPercent']:.1f}%" if comp else "")
                    print(f"  {mode:3s} {tag} gpu={gpu_s:8.4f}s (wall {t:7.2f}s)  "
                          f"peak={mem/2**30:6.2f} GiB (pred {need_gb:6.2f})  "
                          f"exp={exp:.4f} ptab={pt:.4f} round={rnd:.4f} fold={fold:.4f}{cstr}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="bench", required=False)

    d = sub.add_parser("deg1", help="isolated deg1 sumcheck sweep over models/k")
    d.add_argument("--models", default="gpt2,llama8b,llama405b")
    d.add_argument("--ns", default="16,18,20,22,24,26,28",
                   help="extra deg1 sizes to sweep (comma list; '' for models only)")
    d.add_argument("--ks", default="1,2,4,8")
    d.add_argument("--modes", default="gpu")
    d.add_argument("--max-n", type=int, default=29, help="skip sizes with n above this")

    f = sub.add_parser("full", help="full range prover sweep over c/k")
    f.add_argument("--config", default="1 12 64 768 2304")
    f.add_argument("--range-size", type=int, default=48)
    f.add_argument("--cs", default="auto")
    f.add_argument("--ks", default="1,2,4")
    f.add_argument("--modes", default="cpu,gpu")
    f.add_argument("--cap-c", type=int, default=16)

    mm = sub.add_parser("matmul", help="matmul product-sumcheck per-phase sweep (Exp 1)")
    mm.add_argument("--ns", default="18,20,22,24",
                    help="contraction sizes s=2^n to sweep (comma list of n)")
    mm.add_argument("--ks", default="1,2,4,8", help="Blendy stage counts k")
    mm.add_argument("--modes", default="gpu", help="gpu")
    mm.add_argument("--tiles", default="-1",
                    help="ext-table tile widths t to sweep (GKR_SC_TILE_BITS); "
                         "-1 = untiled. The tiled ext table holds 3^l·2^min(n-l,t) "
                         "entries, so t caps the materialized future-variable window.")
    mm.add_argument("--vram-budget-gb", type=float, default=60.0,
                    help="skip (n,k,t) whose predicted allocation 2*ext+3N+2*ptable "
                         "exceeds this (default 60 GiB, just under one MI210's 64)")

    for p in (d, f, mm):
        p.add_argument("--bin", default=DEFAULT_BIN)
        p.add_argument("--out", default=os.path.join(HERE, "results", "range", "data", "range_compare.csv"))
        p.add_argument("--no-mem", action="store_true",
                       help="skip peak GPU memory profiling")
        p.add_argument("--no-compute", action="store_true",
                       help="skip GPU compute utilisation profiling "
                            "(VALUBusy/MemUnitBusy/OccupancyPercent)")

    args = ap.parse_args()
    if args.bench is None:
        args = ap.parse_args(["deg1"])

    if not os.path.exists(args.bin):
        sys.exit(f"binary not found: {args.bin} (build the zkgpt_gpu target first)")

    needs_mem     = not args.no_mem
    needs_compute = not getattr(args, "no_compute", False)
    prof = _profiler()
    has_compute = shutil.which("nsys") or prof is not None
    has_mem     = shutil.which("rocprofv3") or shutil.which("nvidia-smi")

    if needs_compute and not has_compute:
        print("WARNING: no GPU compute profiler found (nsys / rocprofv3 / ncu); "
              "compute metrics will be 0 — rerun with --no-compute to silence",
              file=sys.stderr)
        args.no_compute = True
    if needs_mem and not has_mem:
        print("WARNING: no GPU memory profiler found (rocprofv3 / nvidia-smi); "
              "peak_gpu_bytes will be 0 — rerun with --no-mem to silence",
              file=sys.stderr)
        args.no_mem = True

    rows = {"deg1": bench_deg1, "full": bench_full,
            "matmul": bench_matmul}[args.bench](args)
    if not rows:
        sys.exit("no results")

    fields = list(rows[0].keys())
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
