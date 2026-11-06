#!/usr/bin/env python3
"""run_witgen_matrix.py — separate WHERE PROVING RUNS from WHERE WITGEN RUNS.

The five-arm ladder reports prove_only_s, which excludes circuit precompute by
design ("cache-warmth dependent and swamps the comparison"). Witness generation
lives entirely inside what that metric excludes, so the witgen question is
invisible there. This harness reports wall_s -- witgen ON the clock -- which is
the only convention in which pipelining witgen can show up at all.

Arms (a 2x2 on proving x witgen, plus the naive baseline):

                        proving:  CPU                 GPU
    witgen CPU, serial            cpu_naive           --
    witgen CPU, overlapped        cpu_orchestrated            gpu_orchestrated
    witgen GPU, chained           cpu_separate           gpu_separate

  cpu_naive   CPU proving, CPU witgen, sequential (no partition concurrency)
  cpu_orchestrated    CPU proving, CPU witgen, --batched W
  cpu_separate   CPU proving, GPU witgen, --overlapped W
  gpu_orchestrated    GPU proving, CPU witgen, --batched W
  gpu_separate   GPU proving, GPU witgen, --overlapped W

cpu_orchestrated->cpu_separate and gpu_orchestrated->gpu_separate isolate the witgen axis at fixed
proving location. cpu_*->gpu_* isolates the proving axis at fixed witgen
location. Everything else (circuit, partitioning, width, c) is held equal.

CPU-PROVING ARMS NEED FOUR GATES, NOT TWO. --cpu/--cpu-deg3/--cpu-open/
--cpu-commit plus GKR_MATMUL_CPU/GKR_LASSO_CPU still leave the predicate and
generic-GKR sumcheck on the device: a run so configured reports gpu_peak_mib=1380
and is NOT a CPU baseline. GKR_PRED_CPU=1 and GKR_SC_CPU=1 are required to reach
gpu_peak_mib=0. Verified before this harness was written.

ZKGPT_WEIGHT_SEED is mandatory for every arm: readFconWeight draws weights from
the process-global rand() stream, so the chain (which builds each partition in a
separate forked child) would otherwise regenerate partition 0's weights for every
partition and prove the WRONG model while still reporting ACCEPT.

Usage:
  tools/run_witgen_matrix.py --model gpt2 --len 30 --gpu 7
  tools/run_witgen_matrix.py --model gpt2 --len 30 --gpu 7 --arms gpu_orchestrated,gpu_separate
"""
import argparse
import csv
import glob
import json
import os
import re
import subprocess
import threading
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vram_model

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BIN = os.path.join(ROOT, "bin", "main_gkr_split")
RESULTS_DIR = os.path.join(HERE, "results", "witgen_matrix")

# model -> (L, heads, head_dim, attn/embed, lin). Exactly the ladder's specs
# so these rows are comparable to five_arm/.
MODELS = {
    "gpt2":    (12, 12, 64,  768,  3072),
    "opt350m": (24, 16, 64, 1024,  4096),
    "opt1b3":  (24, 32, 64, 2048,  8192),
    "opt2b7":  (32, 40, 64, 2560, 10240),
}

# Per-model granularity/concurrency from the ladder's tuned gpu_zkmod_dev rows at
# len=30. WIDTH IS NOT "ALL PARTITIONS": host RAM binds before VRAM on the big
# models (opt2b7 ran width=5 at a 4.82 GiB device peak), so splits-wide concurrency
# would ask for ~1.6 TB and OOM the box.
MODEL_CFG = {
    "gpt2":    {"group_layers": 1},
    "opt350m": {"group_layers": 4},
    "opt1b3":  {"group_layers": 2},
    "opt2b7":  {"group_layers": 2},
}

# MEASURED per-partition peaks of THESE end-to-end arms, (host GiB, VRAM GiB).
# Do NOT reuse the ladder's widths -- those omit the boundary commits; end-to-end
# partitions also hold the padded output layer and the carry (opt1b3: 125 GiB/part,
# not ~50, and width=12 OOM-killed).
# Width is PER ARM: CPU-proving arms peak at 0.45 GiB device, GPU-proving at 7-22.
ARM_PEAK_GIB = {
    ("gpt2",    "cpu"): (16.3,  0.5),
    ("gpt2",    "gpu"): (15.4,  3.3),
    ("opt350m", "cpu"): (64.1,  0.5),
    ("opt350m", "gpu"): (58.5,  5.7),
    ("opt1b3",  "cpu"): (125.4, 0.5),
    ("opt1b3",  "gpu"): (125.0, 9.4),
    ("opt2b7",  "cpu"): (240.0, 0.5),
    ("opt2b7",  "gpu"): (240.0, 12.0),   # VRAM still an estimate; host measured
}

# The VRAM column above is the STREAMED peak; --stream-bits off is 2-4x higher, so
# one number mis-sizes the wave when the knob moves. Measured dense peaks:
#   gpt2    7365 MiB = 7.19   opt350m 22095 MiB = 21.6
#   opt1b3 42151 MiB = 41.2   opt2b7  unmeasured
VRAM_DENSE_GIB = {
    "gpt2":    7.2,
    "opt350m": 21.6,
    "opt1b3":  41.2,
}

# Streaming trades ~4% of per-partition proving time (stage-1 source uploaded twice)
# for a smaller device peak, i.e. for wave width. It pays only when that wave is
# worth more than 4%.
# gpt2: NOT worth it (len=30, 12 partitions, same binary):
#     dense    W=8   117.92 s   <-- best      streamed W=12  121.43 s
#     dense    W=7   121.99 s               streamed W=7   122.94 s
# Streaming halves the peak (7365 -> 3349 MiB), lifting the cap 7 -> 12, but one
# MI210 saturates near 7 concurrent partitions, so W=7 and W=12 are within noise.
# Larger models are the opposite and keep it on: opt350m 21.6 -> 5.7 GiB (width
# 2 -> 6), opt2b7 331.5 s -> 192.0 s.
STREAM_DEFAULT = {"gpt2": -1, "opt350m": 20, "opt1b3": 20, "opt2b7": 20}
PHYS_CORES = 192            # 2x EPYC 9654; SMT siblings do not help here
CPU_PROVE_VRAM_GIB = 0.6    # GPU-witgen buffers only
VRAM_BUDGET_GIB = 58.0      # of 64, leaving headroom for fragmentation

HOST_RESERVE_GIB = 128.0
HOST_FRAC = 0.75


def host_avail_gib():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024.0 * 1024.0)
    return 0.0

CPU_FLAGS = ["--cpu", "--cpu-deg3", "--cpu-open", "--cpu-commit"]
CPU_ENV = {"GKR_MATMUL_CPU": "1", "GKR_LASSO_CPU": "1",
           "GKR_PRED_CPU": "1", "GKR_SC_CPU": "1"}

# schedule x proving x witgen. Orchestrated has the PARENT build, so it can never
# do GPU witgen -- the parent would hold a HIP context at fork() and every child
# segfaults (main_gkr_split.cpp). That rules out orch_*_gpu, leaving 6 of the
# 8 combinations reachable.
ARMS = {           # arm -> (proving, witgen, mode)
    # Plain zkGPT: ONE partition spanning every layer, i.e. no zkMod partitioning at
    # all. This is the honest "without zkMod" baseline. `sequential` below is still a
    # zkMod run (L/g partitions), just proved one after another -- it already pays the
    # boundary commits and the padded per-partition output layer, so it is NOT the
    # cost of not partitioning. build() forces group_layers=L for this arm.
    "monolithic":   ("cpu", "cpu", None),
    "sequential":   ("cpu", "cpu", None),           # interleaved build/prove
    "orch_cpu_cpu": ("cpu", "cpu", "--batched"),
    "orch_gpu_cpu": ("gpu", "cpu", "--batched"),
    "sep_cpu_cpu":  ("cpu", "cpu", "--overlapped"),
    "sep_gpu_cpu":  ("gpu", "cpu", "--overlapped"),
    "sep_cpu_gpu":  ("cpu", "gpu", "--overlapped"),
    "sep_gpu_gpu":  ("gpu", "gpu", "--overlapped"),
}
# older spellings accepted on --arms
ARM_ALIAS = {"cpu_naive": "sequential",
             "cpu_pipe": "orch_cpu_cpu", "gpu_pipe": "orch_gpu_cpu",
             "cpu_chain": "sep_cpu_gpu", "gpu_chain": "sep_gpu_gpu",
             "cpu_batched": "orch_cpu_cpu", "gpu_batched": "orch_gpu_cpu",
             "cpu_overlapped": "sep_cpu_gpu", "gpu_overlapped": "sep_gpu_gpu",
             "cpu_orchestrated": "orch_cpu_cpu", "gpu_orchestrated": "orch_gpu_cpu",
             "cpu_separate": "sep_cpu_gpu", "gpu_separate": "sep_gpu_gpu"}

PEAK_RE = re.compile(r"gpu_peak_mib=([0-9.eE+-]+)")
HOST_RE = re.compile(r"^PARTITION .* host_peak_mib=([0-9.eE+-]+)", re.M)
# Device-wide peak, sampled from sysfs by gpu_sampler.hpp across every process on
# the card. This is the real aggregate; summing per-process gpu_peak_mib is not,
# because children are staggered and their lifetime maxima never coincide.
GPUTOT_RE = re.compile(r"GPU_SAMPLE_MAX phase=\S+ vram_peak_mib=([0-9.eE+-]+)")


def _tree_rss_kib(root):
    """Summed RSS of a pid and all its descendants, from /proc."""
    try:
        kids = {}
        for d in os.listdir("/proc"):
            if not d.isdigit():
                continue
            try:
                with open(f"/proc/{d}/stat") as f:
                    parts = f.read().rsplit(") ", 1)[1].split()
                kids.setdefault(int(parts[1]), []).append(int(d))
            except (OSError, IndexError, ValueError):
                continue
        seen, stack, tot = set(), [root], 0
        while stack:
            pid = stack.pop()
            if pid in seen:
                continue
            seen.add(pid)
            stack.extend(kids.get(pid, ()))
            try:
                with open(f"/proc/{pid}/statm") as f:
                    tot += int(f.read().split()[1]) * (os.sysconf("SC_PAGE_SIZE") // 1024)
            except (OSError, IndexError, ValueError):
                pass
        return tot
    except OSError:
        return 0


class HostSampler(threading.Thread):
    """Peak summed RSS over the run's whole process tree, sampled at 0.5 s.

    The per-process host_peak_mib on the PARTITION line cannot answer "how much
    did this arm actually need" -- W children hold their witnesses concurrently.
    Unlike the device, host allocations persist for a partition's whole lifetime,
    so they really do coincide; this measures that rather than assuming it.
    """

    def __init__(self, pid):
        super().__init__(daemon=True)
        self.pid, self.peak_kib, self._done = pid, 0, threading.Event()

    def run(self):
        while not self._done.wait(0.5):
            self.peak_kib = max(self.peak_kib, _tree_rss_kib(self.pid))

    def stop(self):
        self._done.set()
        self.join(timeout=2)
        return self.peak_kib / 1024.0
# Anchored to the PARTITION line on purpose. --witgen-chain ALSO emits a
# OVERLAPPED_BUILT progress line carrying witgen_s= and witgen_fc_s=
# (main_gkr_split.cpp), so an unanchored pattern counts every chain
# partition twice while the pipeline arms count once -- inflating witgen on
# exactly the arms under test. wall_s is unaffected (harness clock).
WITGEN_RE = re.compile(r"^PARTITION .* witgen_s=([0-9.eE+-]+)", re.M)
FC_RE = re.compile(r"^PARTITION .* witgen_fc_s=([0-9.eE+-]+)", re.M)
REPORT_RE = re.compile(
    r"(?:batched_total_s|overlapped_total_s|pipeline_total_s|chain_total_s)"
    r"=([0-9.eE+-]+)")
SUMM_RE = re.compile(r"SPLIT_SUMMARY makespan_s=([0-9.eE+-]+) seq_sum_s=([0-9.eE+-]+)")



# Per-partition device/host peak for THIS config, measured once and cached.
#
# The wave width must be sized from a footprint the run will actually have; fixed
# per-model tables mis-size other lengths (gpt2 len=64 needs ~16 GiB/partition vs 7.2
# at len=30, so --batched at width 8 would ask 128 GiB of a 64 GiB card and OOM-kill
# children with HIP error 2 in gpu_matmul_prover.hpp).
#
# Both modes cap concurrent PROVING at `width` (the token pipe is pre-loaded with
# `width` bytes), so `width * per_partition` is the bound for --batched and
# --overlapped alike. --overlapped happens to peak lower because its serial witgen
# chain staggers arrivals, but that is luck, not a bound.
PROBE_CACHE = os.path.join(RESULTS_DIR, "vram_probe_cache.json")


def _probe_key(a, splits):
    return "|".join(str(x) for x in
                    (a.model, a.len, splits, a.c, a.matmul_k, a.range_k,
                     a.tile_bits, a.stream_bits))


def probe_partition_peaks(a, splits, L, attn, lin):
    """(vram_gib, host_gib) for one GPU-proving partition. Cached per config."""
    key = _probe_key(a, splits)
    cache = {}
    if os.path.exists(PROBE_CACHE):
        try:
            cache = json.load(open(PROBE_CACHE))
        except (ValueError, OSError):
            cache = {}
    if key in cache:
        return tuple(cache[key])

    cmd = [BIN, str(L), str(attn), str(splits), str(a.len), str(a.threads), str(lin),
           "--c", str(a.c),
           "--matmul-k", str(a.matmul_k), "--range-k", str(a.range_k),
           "--sc-tile-bits", str(a.tile_bits)]
    env = dict(os.environ, HIP_VISIBLE_DEVICES=str(a.gpu), ZKGPT_WITNESS_CACHE="0",
               ZKGPT_WEIGHT_SEED=str(a.weight_seed),
               GKR_WITGEN_GPU="1", GKR_ROUND_GPU="1",
               GKR_DUMP_SIZES="1", GKR_DEG3_DBG="1")
    if a.stream_bits >= 0:
        env["GKR_SC_STREAM_BITS"] = str(a.stream_bits)

    print(f"[matrix] probing per-partition peak ({a.model} len={a.len} "
          f"splits={splits}) -- one partition, no mode flag", flush=True)
    # No mode flag => run_split_sequential, one partition at a time. Kill as soon
    # as partition 0 reports: every staged call site has run by then, so the peak
    # and the [mm]/[deg3] shapes are both complete.
    buf = []
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in proc.stdout:
            buf.append(line)
            if line.startswith("PARTITION "):
                break
    finally:
        proc.kill()
        proc.wait()
    text = "".join(buf)

    peaks = [float(x) for x in PEAK_RE.findall(text)]
    hosts = [float(x) for x in HOST_RE.findall(text)]
    if not peaks:
        raise SystemExit("[matrix] probe produced no gpu_peak_mib; cannot size the "
                         "wave. Pass --width explicitly.")
    vram = max(peaks) / 1024.0
    host = (max(hosts) / 1024.0) if hosts else 0.0

    # Cross-check against the closed form (tools/vram_model.py). The sampled peak
    # is a 1 ms high-water mark and can miss a short-lived allocation; the model
    # floor cannot. Take the larger.
    floor_gib = 0.0
    for m in vram_model.MM_RE.finditer(text):
        n, ls, N, pc, ec, _tb, _fv, st = m.groups()
        floor_gib = max(floor_gib, vram_model.mm_bytes(
            int(n), int(ls), int(N), int(pc), int(ec), st == "on") / (1024.0 ** 3))
    for m in vram_model.DEG3_RE.finditer(text):
        l, _k, ls, _t, _fv0, fvc = m.groups()
        floor_gib = max(floor_gib, vram_model.deg3_bytes(
            int(l), int(ls), int(fvc)) / (1024.0 ** 3))
    if floor_gib > vram:
        print(f"[matrix] model floor {floor_gib:.2f} GiB exceeds sampled "
              f"{vram:.2f} GiB; using the floor", flush=True)
        vram = floor_gib

    print(f"[matrix] probed per-partition: VRAM {vram:.2f} GiB "
          f"(model floor {floor_gib:.2f}), host {host:.1f} GiB", flush=True)
    cache[key] = [vram, host]
    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        json.dump(cache, open(PROBE_CACHE, "w"), indent=1)
    except OSError:
        pass
    return vram, host


# Shared box: other users' jobs land on the same GPUs and the same 192 cores, and
# a contended run produces timings that look like a result. The whole-card
# gpu_busy_percent counts co-tenants, which is the point here -- we want to know
# if ANYONE is using the card before we trust a number.
def box_contention():
    """[(pci, busy_pct, used_gib, total_gib)] for every card, plus 1-min load."""
    out = []
    for dev in sorted(glob.glob("/sys/class/drm/card*/device")):
        try:
            pci = os.path.basename(os.path.realpath(dev))
            busy = int(open(os.path.join(dev, "gpu_busy_percent")).read())
            used = int(open(os.path.join(dev, "mem_info_vram_used")).read())
            tot = int(open(os.path.join(dev, "mem_info_vram_total")).read())
        except (OSError, ValueError):
            continue
        out.append((pci, busy, used / 2**30, tot / 2**30))
    return out, os.getloadavg()[0]


def check_box_quiet(force, target_pci=None, busy_max=20, load_warn=48.0):
    """Gate the run on box contention, and return what it saw for the record.

    Two DIFFERENT risks, so two different responses:
      * target GPU busy or holding memory -> REFUSE. A co-tenant on our card is
        an OOM risk (the wave is sized against the whole card), not just noise.
      * host load -> WARN and record. We ask for 192 threads on a 192-core box,
        so any other user contends; on a box with 18 logged-in users, waiting for
        load 0 never terminates. Record it so a row's timings carry their context.

    Other users on OTHER cards are harmless -- only the target card and the shared
    cores reach us. HIP index != rocm-smi index here (hip7 = 0000:a3:00.0), so the
    caller passes the PCI id; without one we gate on every card.
    """
    cards, load = box_contention()
    watch = [c for c in cards if target_pci is None or c[0] == target_pci]
    if target_pci and not watch:
        print(f"[matrix] --gpu-pci {target_pci} matched no card; gating on all",
              flush=True)
        watch = cards
    hot = [c for c in watch if c[1] > busy_max or c[2] > 1.0]

    if hot:
        print(f"[matrix] TARGET GPU CONTENDED (load1={load:.1f}):", flush=True)
        for pci, busy, used, tot in hot:
            print(f"[matrix]    {pci} busy={busy:3d}% vram={used:5.1f}/{tot:.0f} GiB",
                  flush=True)
        if not force:
            raise SystemExit(
                "[matrix] refusing to start: a co-tenant on the target card is an "
                "OOM risk and contaminates timings. Pick an idle card with --gpu / "
                "--gpu-pci, wait, or pass --allow-contended.")
        print("[matrix] --allow-contended set; running anyway", flush=True)
    elif load > load_warn:
        print(f"[matrix] WARNING: host load1={load:.1f} on {PHYS_CORES} physical cores -- "
              f"other users are competing for the cores. Target GPU is clean, so "
              f"GPU-proving arms are usable; CPU-arm wall times will be inflated.",
              flush=True)
    else:
        tgt = target_pci or "all cards"
        print(f"[matrix] box quiet: {tgt} idle, load1={load:.1f}", flush=True)
    return load


def build(a, arm):
    L, heads, hdim, attn, lin = MODELS[a.model]
    proving, witgen, mode = ARMS[arm]
    g = a.group_layers or MODEL_CFG[a.model]["group_layers"]
    splits = L // g
    if arm == "monolithic":
        g, splits = L, 1       # one partition = the whole model = no zkMod
    if a.width:
        width = a.width
    else:
        hpk, _tbl_vpk = ARM_PEAK_GIB[(a.model, proving)]
        vpk, probed_host = probe_partition_peaks(a, splits, L, attn, lin)
        if probed_host > hpk:
            hpk = probed_host
        if proving == "cpu":
            # Device holds only the witgen buffers here; host binds. Measured
            # 6.1 GiB total at width 12, i.e. ~0.5 GiB/partition.
            vpk = CPU_PROVE_VRAM_GIB
        hbud = max(1.0, (host_avail_gib() - HOST_RESERVE_GIB) * HOST_FRAC)
        w_host = int(hbud // hpk)
        w_vram = int(VRAM_BUDGET_GIB // vpk)
        width = max(1, min(splits, w_host, w_vram))
        print(f"[matrix] {arm}: width={width} "
              f"(vram {VRAM_BUDGET_GIB:.0f}/{vpk:.2f}={w_vram}, "
              f"host {hbud:.0f}/{hpk:.1f}={w_host}, splits={splits})", flush=True)

    cmd = [BIN, str(L), str(attn), str(splits), str(a.len), str(a.threads), str(lin),
           "--c", str(a.c),
           "--vram-csv", os.path.join(RESULTS_DIR, f"vram_{arm}_{int(time.time())}.csv")]
    if proving == "gpu":
        # matmul_k is the STAGE COUNT, not a stage width: k=1 means ONE stage over
        # all n variables, so ptable_cap = 3^n. Measured at k=1: n=12 builds a
        # 531441-entry product table for 4096 elements of data (130x), which is why
        # ph_gkr_matmul came out SLOWER on GPU than CPU. Only the largest matmul
        # escaped, via the VRAM clamp forcing l_stage=4. Keep k >= ~n/4.
        cmd += ["--matmul-k", str(a.matmul_k), "--range-k", str(a.range_k),
                "--sc-tile-bits", str(a.tile_bits)]
    else:
        cmd += CPU_FLAGS
    if mode:
        cmd += [mode, str(width)]

    env = dict(os.environ, HIP_VISIBLE_DEVICES=str(a.gpu), ZKGPT_WITNESS_CACHE="0",
               ZKGPT_WEIGHT_SEED=str(a.weight_seed))
    if proving == "gpu" and a.stream_bits >= 0:
        env["GKR_SC_STREAM_BITS"] = str(a.stream_bits)
    if proving == "cpu":
        env.update(CPU_ENV)
    if witgen == "gpu":
        env["GKR_WITGEN_GPU"] = "1"
        # roundLayer's GPU kernel (witgen_gpu/gpu_witgen.hpp). Bundled into
        # the same "witgen=gpu" arms as the FC kernel -- both are part of what
        # this harness means by GPU witgen now.
        env["GKR_ROUND_GPU"] = "1"
    return cmd, env, splits, width


def run(a, arm, log_dir):
    cmd, env, splits, width = build(a, arm)
    log = os.path.join(log_dir, f"{arm}.log")
    t0 = time.time()
    with open(log, "w") as f:
        f.write("# " + " ".join(cmd) + "\n")
        f.write("# env: " + " ".join(f"{k}={v}" for k, v in sorted(env.items())
                                    if k.startswith(("GKR_", "ZKGPT_", "HIP_"))) + "\n")
        f.flush()
        try:
            proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                                    text=True, start_new_session=True)
            hs = HostSampler(proc.pid)
            hs.start()
            try:
                rc = proc.wait(timeout=a.timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                f.write("\n*** TIMEOUT ***\n")
                rc = 124
            host_total_mib = hs.stop()
        except OSError as e:
            f.write(f"\n*** LAUNCH FAILED: {e} ***\n")
            rc, host_total_mib = 127, 0.0
    wall = time.time() - t0
    txt = open(log, errors="ignore").read()
    L = MODELS[a.model][0]
    g = a.group_layers or MODEL_CFG[a.model]["group_layers"]
    n_exp = splits            # one proof per partition

    peaks = [float(x) for x in PEAK_RE.findall(txt)]
    hosts = [float(x) for x in HOST_RE.findall(txt)]
    wg = [float(x) for x in WITGEN_RE.findall(txt)]
    fc = [float(x) for x in FC_RE.findall(txt)]
    rep = REPORT_RE.search(txt)
    proving, witgen, _ = ARMS[arm]

    return {
        "arm": arm, "proving": proving, "witgen": witgen,
        "model": a.model, "len": a.len, "L": L,
        "c": a.c, "matmul_k": a.matmul_k if proving == "gpu" else "",
        "stream_bits": a.stream_bits if proving == "gpu" else "",
        "splits": splits, "group_layers": g, "width": width,
        "wall_s": round(wall, 2),
        "reported_s": round(float(rep.group(1)), 2) if rep else "",
        "witgen_sum_s": round(sum(wg), 2) if wg else "",
        "witgen_fc_sum_s": round(sum(fc), 2) if fc else "",
        "gpu_peak_mib": round(max(peaks), 1) if peaks else "",
        # per-PROCESS peaks. Multiply by width for an upper bound on the
        # aggregate; siblings do not all peak at the same instant, which is
        # why gpt2 runs 12x7.2=86 GiB "worth" on a 64 GiB card.
        "host_peak_mib": round(max(hosts), 1) if hosts else "",
        # MEASURED aggregates, not width x per-process.
        "host_total_mib": round(host_total_mib, 1),
        "gpu_total_mib": round(max(float(x) for x in GPUTOT_RE.findall(txt)), 1)
                         if GPUTOT_RE.search(txt) else "",
        "accept": txt.count("RESULT: ACCEPT"), "reject": txt.count("RESULT: REJECT"),
        "expected": n_exp, "rc": rc,
        "ok": int(rc == 0 and txt.count("RESULT: ACCEPT") == n_exp
                  and txt.count("RESULT: REJECT") == 0),
        "log": log,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt2", choices=sorted(MODELS))
    ap.add_argument("--len", type=int, default=30)
    ap.add_argument("--group-layers", type=int, default=0, dest="group_layers",
                    help="0 = per-model default from MODEL_CFG")
    ap.add_argument("--width", type=int, default=0,
                    help="0 = per-model default from MODEL_CFG (NOT all partitions)")
    ap.add_argument("--timeout", type=int, default=21600, help="per-arm seconds")
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--c", type=int, default=9)
    # Source streaming defaults OFF in the prover (gpu_matmul_prover.hpp,
    # stream_bits=-1). Without it the resident 2N source dominates: it was 34.4 of
    # OPT-1.3B's 35.3 GB matmul footprint, and turning it on cut OPT-350M's measured
    # per-partition peak 21.6 -> 5.6 GiB. That is what decides how many partitions
    # fit, so it is armed by default here.
    ap.add_argument("--stream-bits", type=int, default=None, dest="stream_bits",
                    help="GKR_SC_STREAM_BITS for GPU arms; -1 disables streaming. "
                         "Default is per-model (STREAM_DEFAULT): off for gpt2, on above.")
    ap.add_argument("--matmul-k", type=int, default=4, dest="matmul_k",
                    help="Blendy stage COUNT for matmul; 1 = single stage = 3^n ptable")
    ap.add_argument("--range-k", type=int, default=8, dest="range_k")
    ap.add_argument("--sc-tile-bits", type=int, default=18, dest="tile_bits")
    ap.add_argument("--gpu", type=int, default=7)
    ap.add_argument("--weight-seed", type=int, default=1234, dest="weight_seed")
    ap.add_argument("--arms",
                    default="sequential,orch_cpu_cpu,orch_gpu_cpu,"
                            "sep_cpu_cpu,sep_gpu_cpu,sep_cpu_gpu,sep_gpu_gpu")
    ap.add_argument("--min-avail-gib", type=float, default=500.0, dest="min_avail_gib",
                    help="refuse to start below this much free host RAM")
    ap.add_argument("--gpu-pci", default=None,
                    help="PCI id of the card --gpu selects (hip7=0000:a3:00.0); "
                         "gate only on it instead of every card")
    ap.add_argument("--allow-contended", action="store_true",
                    help="run even if other users are on the GPUs/host")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    # Resolve the per-model streaming default (see STREAM_DEFAULT). Done here, once,
    # so build() and the width computation both see the same concrete value.
    if a.stream_bits is None:
        a.stream_bits = STREAM_DEFAULT.get(a.model, 20)
        print(f"[matrix] --stream-bits defaulted to {a.stream_bits} for {a.model} "
              f"({'dense' if a.stream_bits < 0 else 'streamed'})", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = int(time.time())
    log_dir = os.path.join(RESULTS_DIR, f"logs_{a.model}_len{a.len}_{stamp}")
    os.makedirs(log_dir)
    out = a.out or os.path.join(RESULTS_DIR, f"matrix_{a.model}_len{a.len}_{stamp}.csv")

    box_load1 = check_box_quiet(a.allow_contended, a.gpu_pci)
    avail = host_avail_gib()
    print(f"[matrix] host MemAvailable={avail:.0f} GiB "
          f"(budget {(avail-HOST_RESERVE_GIB)*HOST_FRAC:.0f} GiB)", flush=True)
    # A previous run's survivors silently shrink every width computed below: a leftover
    # opt1b3 set held 540 GiB once and collapsed every arm to width=1. Refuse rather
    # than quietly produce a no-concurrency run that looks like a real measurement.
    if avail < a.min_avail_gib:
        raise SystemExit(f"[matrix] only {avail:.0f} GiB available (< --min-avail-gib "
                         f"{a.min_avail_gib}); check for leftover main_gkr_split "
                         f"processes before starting")
    rows = []
    for arm in (ARM_ALIAS.get(x, x) for x in a.arms.split(",")):
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm}")
        proving, witgen, mode = ARMS[arm]
        _, _, _sp, _w = build(a, arm)
        print(f"=== {arm}  (proving={proving} witgen={witgen} "
              f"{mode or 'sequential'}) splits={_sp} width={_w} ===", flush=True)
        r = run(a, arm, log_dir)
        rows.append(r)
        print(f"    wall={r['wall_s']}s  accept={r['accept']}/{r['expected']}  "
              f"gpu_peak={r['gpu_peak_mib']}  witgen_sum={r['witgen_sum_s']}  "
              f"{'OK' if r['ok'] else '*** BAD ***'}", flush=True)

    fields = ["arm", "proving", "witgen", "model", "len", "L", "splits",
              "group_layers", "width", "c", "matmul_k", "stream_bits", "wall_s", "reported_s",
              "witgen_sum_s",
              "witgen_fc_sum_s", "gpu_peak_mib", "host_peak_mib", "gpu_total_mib", "host_total_mib", "accept", "reject", "expected",
              "rc", "ok", "log"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nCSV:  {out}\nLogs: {log_dir}")

    base = next((r for r in rows if r["arm"] == "sequential"), None)
    print("\n=== wall_s, witgen INCLUDED (the ladder's prove_only_s excludes it) ===")
    for r in rows:
        sp = f"{base['wall_s']/r['wall_s']:.2f}x" if base and r["wall_s"] else "-"
        flag = "" if r["ok"] else "   *** FAILED ***"
        print(f"  {r['arm']:<11} prove={r['proving']:<3} witgen={r['witgen']:<3} "
              f"{r['wall_s']:>9.1f} s  {sp:>7} vs cpu_naive{flag}")


if __name__ == "__main__":
    main()
