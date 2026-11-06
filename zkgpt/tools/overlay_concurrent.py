#!/usr/bin/env python3
"""Reconstruct what a --concurrent run was doing, by folding a PROFILED SEQUENTIAL run's
per-phase costs onto the concurrent run's PHASE timeline.

Why this exists: rocprofv3 cannot profile the concurrent arm at all (it does not follow
fork-without-exec, and the children _exit() without flushing), and the in-binary
gpu_peak_mib is a 1 ms poll sampler that is a lower bound by construction. So the costs
have to come from a run the profiler CAN see -- a sequential one -- and be projected
onto the concurrent wall clock.

Three inputs, and each is load-bearing for a different reason:

  A  concurrent run   -- WHEN each partition is in each phase (PHASE lines, 12
                         processes) plus the device-wide VRAM the sampler measured.
  B  sequential run   -- how long each phase takes UNCONTENDED and UNPROFILED. Durations
                         must come from here: the profiler inflates wall time.
  C  profiled seq run -- kernel busy, transfers, and EXACT device high-water per phase.
                         Sizes come from here; never take a duration from it.

The join between C and A/B is BY PHASE NAME, not by clock: kernels and allocations are
attributed to phases inside C's own profiler clock by containment in the ROCTx ranges
that ZKGPT_PHASE_SCOPE emits alongside every PHASE line. A's PHASE lines and A's
--vram-csv are already the same clock (both gettimeofday), so that join is exact.

Usage:
  tools/overlay_concurrent.py --run-dir tools/results/overlay/<stamp>
  tools/overlay_concurrent.py --concurrent-log A.log --concurrent-vram A_vram.csv \\
      --sequential-log B.log --profile-dir C_profile -o out/

Self-test (validates the merge before it is trusted on A): pass the SAME sequential run
as both --concurrent-log and --sequential-log. Every dilation must come out 1.0 and
every overlap count 1.
"""
import argparse
import bisect
import collections
import csv
import os
import re
import sys

# ── parsing ────────────────────────────────────────────────────────────────────

# PHASE part=3 pid=48211 ev=begin phase=range_prove d=1 t_us=... [dur_s=...]
PHASE_RE = re.compile(
    r"^PHASE\s+part=(-?\d+)\s+pid=(\d+)\s+ev=(\w+)\s+phase=(\S+)\s+d=(\d+)\s+"
    r"t_us=(\d+)(?:\s+dur_s=([\d.]+))?")

# PARTITION idx=0 ... time_s=... -- used only to cross-check that phases account for it.
PARTITION_RE = re.compile(r"^PARTITION\s+idx=(\d+)\s+(.*)$")

Interval = collections.namedtuple("Interval", "part pid phase depth t0 t1 dur")


def parse_phase_log(path):
    """PHASE lines -> completed intervals. Stack per (pid, part) so nesting is honoured
    and a phase appearing twice in one partition (e.g. export_carry) stays distinct."""
    intervals, stacks, marks = [], collections.defaultdict(list), []
    unclosed = 0
    with open(path, errors="replace") as f:
        for line in f:
            m = PHASE_RE.match(line)
            if not m:
                continue
            part, pid, ev, phase, depth, t_us, dur = m.groups()
            part, pid, depth, t_us = int(part), int(pid), int(depth), int(t_us)
            key = (pid, part)
            if ev == "begin":
                stacks[key].append((phase, depth, t_us))
            elif ev == "end":
                stk = stacks[key]
                # Pop the matching open frame; tolerate interleaving from a phase whose
                # scope crossed a set_part() (the parent's release marks do this).
                for j in range(len(stk) - 1, -1, -1):
                    if stk[j][0] == phase:
                        _, d0, t0 = stk.pop(j)
                        intervals.append(Interval(part, pid, phase, d0, t0, t_us,
                                                  (t_us - t0) / 1e6))
                        break
                else:
                    unclosed += 1
            elif ev == "mark":
                marks.append((part, pid, phase, t_us))
    for stk in stacks.values():
        unclosed += len(stk)
    return intervals, marks, unclosed


def read_csv_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", errors="replace") as f:
        return list(csv.DictReader(f))


def col(row, *names):
    """rocprofv3 column names have moved between SDK versions; accept any spelling."""
    for n in names:
        if n in row and row[n] != "":
            return row[n]
    low = {k.lower(): v for k, v in row.items()}
    for n in names:
        if n.lower() in low and low[n.lower()] != "":
            return low[n.lower()]
    return None


def parse_profile(prof_dir):
    """Profiler CSVs -> ROCTx ranges, kernel dispatches, allocation events. All
    timestamps stay in the profiler's own clock; we never convert them."""
    def find(suffix):
        hits = []
        for root, _, files in os.walk(prof_dir):
            for fn in files:
                if fn.endswith(suffix):
                    hits.append(os.path.join(root, fn))
        return sorted(hits)

    ranges = []
    for path in find("marker_api_trace.csv"):
        for r in read_csv_rows(path):
            name = col(r, "Name", "Function", "Kind_Name", "Operation")
            t0 = col(r, "Start_Timestamp", "Start_Timestamp_Ns")
            t1 = col(r, "End_Timestamp", "End_Timestamp_Ns")
            if not name or t0 is None or t1 is None:
                continue
            t0, t1 = int(t0), int(t1)
            if t1 <= t0:          # roctxMark and zero-length pushes
                continue
            ranges.append((name.strip(), t0, t1))

    kernels = []
    for path in find("kernel_trace.csv"):
        for r in read_csv_rows(path):
            t0 = col(r, "Start_Timestamp"); t1 = col(r, "End_Timestamp")
            if t0 is None or t1 is None:
                continue
            kernels.append((col(r, "Kernel_Name", "Name") or "?", int(t0), int(t1)))

    allocs = []
    for path in find("memory_allocation_trace.csv"):
        for r in read_csv_rows(path):
            t0 = col(r, "Start_Timestamp")
            size = col(r, "Allocation_Size", "Size")
            kind = (col(r, "Operation", "Kind_Name", "Kind") or "").upper()
            addr = col(r, "Address", "Starting_Address")
            if t0 is None:
                continue
            allocs.append((int(t0), kind, int(size or 0), addr))

    copies = []
    for path in find("memory_copy_trace.csv"):
        for r in read_csv_rows(path):
            t0 = col(r, "Start_Timestamp")
            if t0 is None:
                continue
            copies.append((int(t0), (col(r, "Operation", "Kind_Name") or "").upper(),
                           int(col(r, "Bytes", "Size") or 0)))
    return ranges, kernels, allocs, copies


# ── C: per-phase cost table ────────────────────────────────────────────────────

def is_free(kind):
    return kind.endswith("FREE")


def is_alloc(kind):
    return kind.endswith("ALLOCATE") or kind.endswith("MALLOC")


def resident_series(allocs):
    """Running device-resident bytes over time, from the alloc/free event stream.
    Free records carry no size, so the size is remembered per address at allocation."""
    live, cur, series = {}, 0, []
    for t, kind, size, addr in sorted(allocs):
        if is_free(kind):
            cur -= live.pop(addr, 0)
        elif is_alloc(kind):
            if addr is not None:
                live[addr] = size
            cur += size
        else:
            continue
        series.append((t, cur))
    return series


def innermost_phase_ranges(ranges, phase_names):
    """A partition range (part<i>) encloses its phase ranges. Return, per phase range,
    the enclosing partition index so costs land on (partition, phase)."""
    parts = []
    for name, t0, t1 in ranges:
        m = re.fullmatch(r"part(\d+)(?:_build)?", name)
        if m:
            parts.append((t0, t1, int(m.group(1))))
    parts.sort()

    def owner(t0):
        best = None
        for p0, p1, idx in parts:
            if p0 <= t0 <= p1 and (best is None or p0 >= best[0]):
                best = (p0, idx)
        return best[1] if best else -1

    out = []
    for name, t0, t1 in ranges:
        if re.fullmatch(r"part(\d+)(?:_build)?", name):
            continue
        if phase_names and name not in phase_names:
            continue
        out.append((owner(t0), name, t0, t1))
    return out


def overlap(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))


def resident_stats(series, starts, t0, t1):
    """(max, TIME-WEIGHTED mean) device-resident bytes over [t0, t1] of the step
    function `series`.

    Time-weighted is not a refinement, it is the only correct statistic here: the mean
    over event POINTS over-weights the instants around allocations. On a phase that
    allocates 8 GiB 1 ms in and holds it for 10 s, the point mean reads 6 GiB and the
    time-weighted mean reads ~8 GiB -- and 8 GiB is what a sibling partition actually
    has to share the card with."""
    if not series:
        return 0, 0.0
    i = bisect.bisect_right(starts, t0) - 1
    cur = series[i][1] if i >= 0 else 0
    mx, area, t = cur, 0.0, t0
    j = bisect.bisect_left(starts, t0)
    while j < len(series) and series[j][0] < t1:
        tt, v = series[j]
        area += cur * (tt - t)
        t, cur = tt, v
        mx = max(mx, cur)
        j += 1
    area += cur * (t1 - t)
    span = t1 - t0
    return mx, (area / span if span > 0 else float(cur))


TRAJ_SAMPLES = 33          # points per phase; 32 intervals is finer than any phase's
                           # allocation count and keeps the tables small.


def child_resident_series(series, prs):
    """What ONE partition's process holds on the device, over time.

    Modelled from C's PROCESS-TOTAL resident curve, not from per-partition allocation
    attribution. Attribution looks more precise and is worse: it captures only what it
    can charge to a phase range and silently drops everything else a real child also
    holds -- the HIP context, long-lived buffers, anything allocated outside a named
    phase. Measured, that lost ~26 GiB of a 50 GiB device peak, and the B-against-itself
    self-test caught it (7.7 predicted vs 30.6 measured with a single partition live).

    C runs the partitions sequentially in one address space, so its total also carries
    whatever partitions before p never gave back. That is subtracted: for partition p,

        one_child(t) = C_total(t) - C_total(entry of p) + baseline

    leaving the process floor plus exactly what p has allocated so far. `baseline` is
    C's floor before any partition ran, which every child pays for independently.

    Returns (series, {partition: entry_offset_bytes}, baseline_bytes).
    """
    if not series:
        return [], {}, 0
    starts = [t for t, _ in series]
    baseline = min(v for _, v in series[:max(1, len(series) // 50)])

    # Partition p's entry = the earliest start among its own ranges.
    first = {}
    for part, _name, t0, _t1 in prs:
        if part >= 0 and (part not in first or t0 < first[part]):
            first[part] = t0

    entry = {}
    for part, t0 in first.items():
        i = bisect.bisect_right(starts, t0) - 1
        entry[part] = max(0, (series[i][1] if i >= 0 else 0) - baseline)
    return series, entry, baseline


def sample_trajectory(series_pt, t0, t1, k=TRAJ_SAMPLES, offset=0):
    """The live-bytes SHAPE of one phase instance: k samples of that partition's own
    resident bytes across [t0, t1], as a fraction of the phase.

    This is what replaces a per-phase scalar. A phase that allocates at its start and
    frees at its end overlaps a sibling completely differently depending on where each
    one is inside it, and a single mean or peak cannot express that.

    Fractional (not absolute-offset) time is the right parameterisation: a dilated phase
    is doing the same work more slowly, so an allocation that happened 30% of the way
    through uncontended still happens ~30% of the way through when contended.
    """
    if not series_pt:
        return [0.0] * k
    ts = [x[0] for x in series_pt]
    out = []
    for i in range(k):
        t = t0 + (t1 - t0) * (i / (k - 1.0)) if k > 1 else t0
        j = bisect.bisect_right(ts, t) - 1
        v = float(series_pt[j][1]) if j >= 0 else 0.0
        out.append(max(0.0, v - offset))
    return out


def traj_lookup(traj, by_phase_traj, part, phase, u):
    """Live bytes for `part` at fraction u through `phase`; falls back to the average
    shape of that phase across partitions when this one was not resolved."""
    v = traj.get((part, phase)) or by_phase_traj.get(phase)
    if not v:
        return 0.0
    u = 0.0 if u < 0 else (1.0 if u > 1 else u)
    x = u * (len(v) - 1)
    i = int(x)
    if i >= len(v) - 1:
        return v[-1]
    frac = x - i
    return v[i] * (1 - frac) + v[i + 1] * frac


def phase_costs(prof_dir, phase_names):
    """Attribute kernels, allocations and copies to the ROCTx range that contains them.

    Everything here is bisect-driven on pre-sorted arrays. The naive nested scan is
    O(ranges x events), and a real GPT-2 profile has ~250 ranges against millions of
    dispatches -- that version does not finish."""
    ranges, kernels, allocs, copies = parse_profile(prof_dir)
    if not ranges:
        return {}, {}, "no ROCTx ranges in %s -- was the binary built with -DZKGPT_ROCTX?" % prof_dir

    series = resident_series(allocs)
    ser_starts = [t for t, _ in series]

    kernels.sort(key=lambda k: k[1])
    k_starts = [k[1] for k in kernels]
    # A kernel can start before a range and end inside it, so the scan has to begin
    # this far back. One pass to find it beats widening the window by guesswork.
    k_maxdur = max((k[2] - k[1] for k in kernels), default=0)

    allocs.sort(key=lambda a: a[0])
    a_starts = [a[0] for a in allocs]
    copies.sort(key=lambda c: c[0])
    c_starts = [c[0] for c in copies]

    prs = innermost_phase_ranges(ranges, phase_names)
    # One child's resident curve = C's process total minus what earlier partitions
    # retained; the SHAPE of each phase is sampled off that.
    tot_series, entry_off, baseline = child_resident_series(series, prs)
    traj = {}
    costs = {}
    for part, name, t0, t1 in prs:
        c = costs.setdefault((part, name), {
            "n": 0, "span_ns": 0, "kernel_ns": 0, "dispatches": 0,
            "resident_max_b": 0, "resident_mean_b": 0.0, "alloc_b": 0, "free_b": 0,
            "copy_b": 0})
        c["n"] += 1
        c["span_ns"] += t1 - t0

        j = bisect.bisect_left(k_starts, t0 - k_maxdur)
        while j < len(kernels) and kernels[j][1] < t1:
            ov = overlap(kernels[j][1], kernels[j][2], t0, t1)
            if ov > 0:
                c["kernel_ns"] += ov
                c["dispatches"] += 1
            j += 1

        mx, mean = resident_stats(series, ser_starts, t0, t1)
        c["resident_max_b"] = max(c["resident_max_b"], mx)
        c["resident_mean_b"] = max(c["resident_mean_b"], mean)
        # The trajectory is the load-bearing one; max/mean stay only for the report.
        shape = sample_trajectory(tot_series, t0, t1,
                                  offset=entry_off.get(part, 0))
        prev = traj.get((part, name))
        # A phase can recur (export_carry); keep the instance with the largest area,
        # which is the one that actually shapes the overlap.
        if prev is None or sum(shape) > sum(prev):
            traj[(part, name)] = shape

        for j in range(bisect.bisect_left(a_starts, t0), bisect.bisect_right(a_starts, t1)):
            _t, kind, size, _a = allocs[j]
            if is_free(kind):
                c["free_b"] += size
            elif is_alloc(kind):
                c["alloc_b"] += size
        for j in range(bisect.bisect_left(c_starts, t0), bisect.bisect_right(c_starts, t1)):
            c["copy_b"] += copies[j][2]
    return costs, traj, None


# ── A: overlap timeline and predictions ────────────────────────────────────────

def build_segments(intervals):
    """Split the wall clock at every phase boundary. Each segment carries the INTERVALS
    live in it (not just their names): the predictor needs each partition's phase start
    and end to know how far into that phase it is at this instant."""
    edges = sorted({t for iv in intervals for t in (iv.t0, iv.t1)})
    segs = []
    for a, b in zip(edges, edges[1:]):
        if b <= a:
            continue
        mid = (a + b) / 2.0
        active = [iv for iv in intervals if iv.t0 <= mid < iv.t1]
        segs.append((a, b, active))
    return segs


GPU_PHASES = {"witgen", "range_build", "range_prove", "execute_commit",
              "gkr_prove", "verify_gkr", "verify_lasso", "open_commit",
              "sequence_kv_commit", "standalone_input_commit", "output_commit"}
# gate_wait/carry_wait/token_wait/barrier_wait are blocked, not working -- excluded so
# "how many partitions are actually competing" is not inflated by parked ones.


def leaf_active(active):
    """Keep only the innermost phase per partition: 'prove' encloses 'gkr_prove'
    encloses 'verify_lasso', and counting all three triples that partition's weight.
    Returns {partition: Interval}."""
    NEST = {"prove": 0, "build": 0, "gkr_prove": 1}
    out = {}
    for iv in active:
        rank = 2 if iv.phase not in NEST else NEST[iv.phase]
        if iv.part not in out or rank > out[iv.part][0]:
            out[iv.part] = (rank, iv)
    return {p: v[1] for p, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", help="directory written by run_overlay_experiment.sh")
    ap.add_argument("--concurrent-log")
    ap.add_argument("--concurrent-vram")
    ap.add_argument("--sequential-log")
    ap.add_argument("--profile-dir")
    ap.add_argument("-o", "--out")
    ap.add_argument("--proc-overhead-gib", type=float, default=0.0,
                    help="Device memory each PROCESS holds that no allocation trace can "
                         "see -- HIP context, code objects, queues. It never passes "
                         "through hipMalloc, so rocprofv3 cannot record it, yet every "
                         "concurrent child pays it again. Estimate it as (measured - "
                         "predicted) from a single-partition run, then it is added once "
                         "per live partition here.")
    ap.add_argument("--pci", help="PCI bus id of the card the run used, e.g. 0000:a3:00.0. "
                                  "Read from <run-dir>/config.txt when --run-dir is given.")
    a = ap.parse_args()

    if a.run_dir:
        a.concurrent_log = a.concurrent_log or os.path.join(a.run_dir, "A_concurrent.log")
        a.concurrent_vram = a.concurrent_vram or os.path.join(a.run_dir, "A_concurrent_vram.csv")
        a.sequential_log = a.sequential_log or os.path.join(a.run_dir, "B_sequential.log")
        a.profile_dir = a.profile_dir or os.path.join(a.run_dir, "C_profile")
        a.out = a.out or os.path.join(a.run_dir, "overlay")
        cfg = os.path.join(a.run_dir, "config.txt")
        if not a.pci and os.path.exists(cfg):
            mm = re.search(r"pci=(\S+)", open(cfg).read())
            if mm:
                a.pci = mm.group(1)
    for req in ("concurrent_log", "sequential_log", "profile_dir"):
        if not getattr(a, req):
            ap.error("--%s is required (or use --run-dir)" % req.replace("_", "-"))
    a.out = a.out or "overlay_out"
    os.makedirs(a.out, exist_ok=True)

    A, A_marks, A_unclosed = parse_phase_log(a.concurrent_log)
    B, _, B_unclosed = parse_phase_log(a.sequential_log)
    # Drop the boundary-size probe from both sides. It is a throwaway witgen that only
    # reads getTotalInputSize(), so it has no dilation to measure and no counterpart in
    # the profiled run; keeping it would put a ~22 s single-partition prologue in front
    # of every co-activity and occupancy figure.
    probe_s = sum(iv.dur for iv in A if iv.phase == "probe_witgen")
    A = [iv for iv in A if iv.phase != "probe_witgen"]
    B = [iv for iv in B if iv.phase != "probe_witgen"]
    if probe_s:
        print("excluded probe_witgen: %.1f s from the concurrent timeline" % probe_s)
    if not A:
        sys.exit("no PHASE lines in %s" % a.concurrent_log)
    if not B:
        sys.exit("no PHASE lines in %s" % a.sequential_log)

    names = {iv.phase for iv in A} | {iv.phase for iv in B}
    costs, traj, err = phase_costs(a.profile_dir, names)
    if err:
        print("WARNING: %s" % err, file=sys.stderr)

    # ── 1. per-phase cost table (from C) ───────────────────────────────────────
    with open(os.path.join(a.out, "phase_costs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["partition", "phase", "n", "span_s", "kernel_busy_s", "dispatches",
                    "kernel_busy_frac", "resident_max_gib", "resident_mean_gib",
                    "alloc_gib", "free_gib", "copy_gib"])
        for (part, phase), c in sorted(costs.items()):
            span = c["span_ns"] / 1e9
            w.writerow([part, phase, c["n"], "%.4f" % span,
                        "%.4f" % (c["kernel_ns"] / 1e9), c["dispatches"],
                        "%.4f" % (c["kernel_ns"] / c["span_ns"] if c["span_ns"] else 0),
                        "%.3f" % (c["resident_max_b"] / 2**30),
                        "%.3f" % (c["resident_mean_b"] / 2**30),
                        "%.3f" % (c["alloc_b"] / 2**30), "%.3f" % (c["free_b"] / 2**30),
                        "%.3f" % (c["copy_b"] / 2**30)])

    # Phase-level fallbacks, for partitions the profile did not resolve individually.
    by_phase = collections.defaultdict(lambda: {"res": [], "busy": []})
    for (part, phase), c in costs.items():
        by_phase[phase]["res"].append(c["resident_mean_b"])
        if c["span_ns"]:
            by_phase[phase]["busy"].append(c["kernel_ns"] / c["span_ns"])

    by_phase_traj = {}
    for (part, phase), v in traj.items():
        acc = by_phase_traj.setdefault(phase, [0.0] * len(v))
        for i, x in enumerate(v):
            acc[i] += x
    for phase in by_phase_traj:
        n = sum(1 for (pp, ph) in traj if ph == phase) or 1
        by_phase_traj[phase] = [x / n for x in by_phase_traj[phase]]

    def resident_at(iv, t):
        """Live bytes for the partition in interval `iv` at wall time `t`.

        The phase's own SHAPE evaluated at this partition's progress through it -- not a
        per-phase constant. Two partitions in the same phase at different offsets
        contribute different amounts, which is the entire point: they were released
        together but drift apart, so at any instant they sit at different points on the
        same allocate/free curve."""
        span = iv.t1 - iv.t0
        u = 0.0 if span <= 0 else (t - iv.t0) / float(span)
        return traj_lookup(traj, by_phase_traj, iv.part, iv.phase, u)

    def busy_for(part, phase):
        c = costs.get((part, phase))
        if c and c["span_ns"]:
            return c["kernel_ns"] / c["span_ns"]
        r = by_phase.get(phase, {}).get("busy") or []
        return sum(r) / len(r) if r else 0.0

    # ── 2+3+5. overlap timeline, predicted VRAM, predicted GPU demand ──────────
    segs = build_segments(A)
    t_origin = min(iv.t0 for iv in A)
    with open(os.path.join(a.out, "overlay_timeline.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t0_us", "t1_us", "rel_s", "dur_s", "n_active", "n_gpu_active",
                    "pred_vram_gib", "pred_gpu_demand", "active"])
        for t0, t1, active in segs:
            leaf = leaf_active(active)
            gpu = {p: iv for p, iv in leaf.items() if iv.phase in GPU_PHASES}
            mid = (t0 + t1) / 2.0
            pred_v = (sum(resident_at(iv, mid) for iv in gpu.values())
                      + len(gpu) * a.proc_overhead_gib * 2**30)
            pred_d = sum(busy_for(p, iv.phase) for p, iv in gpu.items())
            w.writerow(["%d" % t0, "%d" % t1, "%.3f" % ((t0 - t_origin) / 1e6),
                        "%.3f" % ((t1 - t0) / 1e6), len(leaf), len(gpu),
                        "%.3f" % (pred_v / 2**30), "%.3f" % pred_d,
                        "|".join("%d:%s@%.2f" % (p, iv.phase,
                                 (mid - iv.t0) / max(1.0, float(iv.t1 - iv.t0)))
                                 for p, iv in sorted(leaf.items()))])

    # ── predicted vs measured VRAM (same clock, exact join) ────────────────────
    vram_rows = []
    if a.concurrent_vram and os.path.exists(a.concurrent_vram):
        meas = collections.defaultdict(list)
        for r in read_csv_rows(a.concurrent_vram):
            try:
                meas[r["pci"]].append((float(r["t_us"]), int(r["vram_used_b"])))
            except (KeyError, ValueError):
                continue
        # Pin the card EXPLICITLY. The sampler records every GPU on the box, and
        # "busiest wins" silently compares against a co-tenant's card, making a correct prediction
        # look too low. run_overlay_experiment.sh records the bus id in config.txt.
        if meas:
            pci = a.pci
            if pci and pci not in meas:
                print("WARNING: pci %s not in the vram csv; cards present: %s"
                      % (pci, ", ".join(sorted(meas))), file=sys.stderr)
                pci = None
            if not pci:
                pci = max(meas, key=lambda k: max(v for _, v in meas[k]))
                print("WARNING: no --pci given, falling back to the busiest card (%s). "
                      "If another tenant is on the box this is the WRONG card."
                      % pci, file=sys.stderr)
            base = min(v for _, v in meas[pci])       # co-tenant baseline
            si = 0
            with open(os.path.join(a.out, "vram_pred_vs_meas.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t_us", "rel_s", "meas_gib", "meas_minus_base_gib",
                            "pred_gib", "residual_gib", "n_gpu_active"])
                for t, used in sorted(meas[pci]):
                    while si + 1 < len(segs) and segs[si][1] <= t:
                        si += 1
                    if not (segs[si][0] <= t < segs[si][1]):
                        continue
                    leaf = leaf_active(segs[si][2])
                    gpu = {p: iv for p, iv in leaf.items() if iv.phase in GPU_PHASES}
                    pred = (sum(resident_at(iv, t) for iv in gpu.values())
                            + len(gpu) * a.proc_overhead_gib * 2**30)
                    mb = (used - base) / 2**30
                    pg = pred / 2**30
                    vram_rows.append((mb, pg))
                    w.writerow(["%.0f" % t, "%.3f" % ((t - t_origin) / 1e6),
                                "%.3f" % (used / 2**30), "%.3f" % mb, "%.3f" % pg,
                                "%.3f" % (pg - mb), len(gpu)])

    # ── 4. dilation: concurrent duration / uncontended duration ────────────────
    def total_by(intervals):
        d = collections.defaultdict(float)
        for iv in intervals:
            d[(iv.part, iv.phase)] += iv.dur
        return d

    Ad, Bd = total_by(A), total_by(B)
    # Mean co-active GPU partitions, time-weighted, over each phase's OWN window.
    # Keyed off the segment's full active set, not its leaf set: an enclosing phase
    # (gkr_prove wraps verify_lasso) is shadowed in the leaf view and would otherwise
    # report a co-active count of zero for a window in which it was plainly running.
    coactive = collections.defaultdict(lambda: [0.0, 0.0])
    for t0, t1, active in segs:
        n = len({p for p, iv in leaf_active(active).items() if iv.phase in GPU_PHASES})
        for key in {(iv.part, iv.phase) for iv in active}:
            acc = coactive[key]
            acc[0] += n * (t1 - t0)
            acc[1] += (t1 - t0)

    dil_rows = []
    with open(os.path.join(a.out, "dilation.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["partition", "phase", "concurrent_s", "sequential_s", "dilation",
                    "mean_coactive_gpu", "kernel_busy_frac_seq"])
        for key in sorted(set(Ad) | set(Bd)):
            part, phase = key
            ca, sb = Ad.get(key, 0.0), Bd.get(key, 0.0)
            acc = coactive.get(key, [0.0, 0.0])
            mc = acc[0] / acc[1] if acc[1] else 0.0
            dil = ca / sb if sb > 0 else float("nan")
            w.writerow([part, phase, "%.4f" % ca, "%.4f" % sb,
                        "%.4f" % dil if sb > 0 else "",
                        "%.3f" % mc, "%.4f" % busy_for(part, phase)])
            if sb > 0 and phase in GPU_PHASES:
                dil_rows.append((phase, part, dil, mc, ca, sb))

    # ── text Gantt ─────────────────────────────────────────────────────────────
    span_us = max(iv.t1 for iv in A) - t_origin
    W = 100
    gantt = os.path.join(a.out, "gantt.txt")
    with open(gantt, "w") as f:
        f.write("concurrent timeline, %.1f s wide, one row per partition\n" % (span_us / 1e6))
        f.write("  legend: W witgen  B build  g gate/blocked  R range_prove  "
                "r range_build  C commit  K gkr_prove  . idle\n\n")
        SYM = {"witgen": "W", "build": "B", "gate_wait": "g",
               "carry_wait": "g", "token_wait": "g", "barrier_wait": "g",
               "range_prove": "R", "range_build": "r", "execute_commit": "C",
               "sequence_kv_commit": "C", "standalone_input_commit": "C",
               "output_commit": "C", "gkr_prove": "K", "verify_gkr": "G",
               "verify_lasso": "L", "open_commit": "O", "export_carry": "e",
               "carry_publish": "e", "carry_import": "e", "prove": "-"}
        for part in sorted({iv.part for iv in A}):
            row = ["."] * W
            for iv in A:
                if iv.part != part:
                    continue
                c0 = int((iv.t0 - t_origin) / span_us * (W - 1))
                c1 = int((iv.t1 - t_origin) / span_us * (W - 1))
                for c in range(max(0, c0), min(W, c1 + 1)):
                    sym = SYM.get(iv.phase, "?")
                    if row[c] == "." or sym not in "-":
                        row[c] = sym
            f.write("p%-3d |%s|\n" % (part, "".join(row)))

    # ── summary ────────────────────────────────────────────────────────────────
    print("=" * 78)
    print("OVERLAY  A=%s" % os.path.basename(a.concurrent_log))
    print("=" * 78)
    print("intervals: concurrent=%d (unclosed %d)  sequential=%d (unclosed %d)"
          % (len(A), A_unclosed, len(B), B_unclosed))
    print("phase cost entries from profile: %d over %d partitions"
          % (len(costs), len({p for p, _ in costs})))
    if not costs:
        print("  !! no per-phase costs -- VRAM and demand predictions are all zero")

    maxn = max((len({p for p, iv in leaf_active(s[2]).items() if iv.phase in GPU_PHASES})
                for s in segs), default=0)
    print("\npeak concurrent GPU-active partitions: %d" % maxn)
    hist = collections.defaultdict(float)
    for t0, t1, active in segs:
        gpu = {p for p, iv in leaf_active(active).items() if iv.phase in GPU_PHASES}
        hist[len(gpu)] += (t1 - t0) / 1e6
    print("time at each concurrency level (s):")
    for k in sorted(hist):
        print("   %2d active: %8.2f s  %s" % (k, hist[k], "#" * int(hist[k] / max(hist.values()) * 40)))

    if vram_rows:
        res = [p - m for m, p in vram_rows]
        mm = max(m for m, _ in vram_rows)
        mp = max(p for _, p in vram_rows)
        print("\nVRAM: measured peak %.1f GiB (over co-tenant baseline), predicted peak %.1f GiB"
              % (mm, mp))
        print("      mean residual %+.2f GiB, max |residual| %.2f GiB over %d samples"
              % (sum(res) / len(res), max(abs(r) for r in res), len(res)))
        if mm > 0:
            print("      sum-of-concurrent-phase / measured = %.2fx" % (mp / mm))

    if dil_rows:
        print("\nmost-dilated phases (concurrent / uncontended):")
        agg = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0])
        for phase, part, dil, mc, ca, sb in dil_rows:
            g = agg[phase]
            g[0] += ca; g[1] += sb; g[2] += mc; g[3] += 1
        rows = [(ph, g[0] / g[1], g[2] / g[3], g[0], g[1])
                for ph, g in agg.items() if g[1] > 0]
        rows.sort(key=lambda r: -r[1])
        print("  %-24s %8s %10s %10s %8s" % ("phase", "dilation", "conc_s", "seq_s", "coactive"))
        for ph, dil, mc, ca, sb in rows:
            print("  %-24s %8.2fx %10.1f %10.1f %8.1f" % (ph, dil, ca, sb, mc))

    print("\nwrote: %s" % a.out)
    for fn in ("phase_costs.csv", "overlay_timeline.csv", "dilation.csv",
               "vram_pred_vs_meas.csv", "gantt.txt"):
        fp = os.path.join(a.out, fn)
        if os.path.exists(fp):
            print("   %s" % fp)


if __name__ == "__main__":
    main()
