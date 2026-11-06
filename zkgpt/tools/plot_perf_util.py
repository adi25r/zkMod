#!/usr/bin/env python3
"""plot_perf_util.py — CPU utilisation over time, the host counterpart to the
"GPU busy" panel in plot_pmc_timeline.py.

perf samples at a fixed per-thread frequency, so samples-per-second divided by
that frequency IS cores busy -- no extra instrumentation needed, and it comes from
the same report_by_time.txt the phase figure reads.

The split is the point. Samples are classified into OpenMP runtime (barrier
spin-wait, the timestamp helper, the flag checks) and everything else, so the
chart separates cores that are RUNNING from cores that are merely occupied. On the
monolithic CPU baseline ~79% of all samples are the former, which is invisible in
any aggregate "CPU utilisation" number.

  panel   cores busy over time, stacked: real work + OpenMP spin
  strip   the run's leaf phases, coloured as in plot_overlay_trace

Usage:
  tools/plot_perf_util.py <run dir>            # e.g. .../cpu_mono
"""
import argparse
import collections
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plotstyle as ps  # noqa: E402  house style (paper scripts)
from plotstyle import COLOR, SPEC  # noqa: E402  phase colours, one source of truth
from plot_perf_phases import phases, clock_offset, seq_len  # noqa: E402

# dataviz reference palette, categorical slots 1-2 (two series, fixed order).
WORK, SPIN = ps.STACK_2[0], ps.STACK_2[1]
BG, INK, INK2, GRID = "none", "black", "black", "gray"
CORES = 192.0            # 2x EPYC 9654, one thread per physical core
FREQ = 199.0             # perf record -F 199, per thread

ROW = re.compile(r"^\s*[0-9.]+%\s+([0-9]+)\s+([0-9]+\.[0-9]+)\s+\[.\]\s+(.*?)\s*$")
# The OpenMP runtime's spin path. These are threads waiting at a barrier, not
# doing the prover's work.
SPIN_RE = re.compile(r"kmp_flag|__kmp_hardware_timestamp|__kmp_wait|notdone_check"
                     r"|__kmp_yield|kmp_barrier|__kmp_join_barrier")


def buckets(path):
    """{t_monotonic: [spin_samples, work_samples]} at the report's own quantum."""
    out = collections.defaultdict(lambda: [0.0, 0.0])
    for line in open(path, errors="replace"):
        m = ROW.match(line)
        if not m:
            continue
        n, tb, sym = float(m.group(1)), float(m.group(2)), m.group(3)
        out[tb][0 if SPIN_RE.search(sym) else 1] += n
    return out


BUILD_PROVE_RE = re.compile(
    r"^PHASE\s+part=(-?\d+)\s+pid=\d+\s+ev=(begin|end)\s+phase=(build|prove)\s+"
    r"d=0\s+t_us=(\d+)")


def partition_windows(log_path):
    """{idx: (build_begin_s, prove_end_s)} in CLOCK_REALTIME seconds -- the
    partition's TRUE span, not the PARTITION summary line's own t0_us/t1_us.

    Those fields are the 'prove' phase only (t0_us here is bit-for-bit the
    'PHASE ... phase=prove ev=begin' timestamp) -- the same 'circuit_s is NOT
    witness-generation time' trap, known from a different
    field, and it silently drops witgen/build/export_carry from any figure
    sliced to that window. probe_witgen is deliberately NOT included: it is a
    one-time whole-run pass before partition 0's build even starts, not part of
    any individual partition ("the whole leading segment of a run")."""
    begin, end = {}, {}
    for line in open(log_path, errors="replace"):
        m = BUILD_PROVE_RE.match(line)
        if not m:
            continue
        part, ev, phase, t_us = m.groups()
        part = int(part)
        if part < 0:
            continue  # parent-process markers (--concurrent/--batched/--overlapped only)
        t = int(t_us) / 1e6
        if phase == "build" and ev == "begin":
            begin[part] = t
        elif phase == "prove" and ev == "end":
            end[part] = t
    return {i: (begin[i], end[i]) for i in begin if i in end}


def draw(run_dir, out_path, partition=None):
    rpt = os.path.join(run_dir, "report_by_time.txt")
    if not os.path.exists(rpt):
        sys.exit("no report_by_time.txt under " + run_dir)
    log = next((os.path.join(run_dir, c) for c in ("record_run.log", "stat_run.log")
                if os.path.exists(os.path.join(run_dir, c))), None)
    ph = phases(log) if log else []
    off = clock_offset()

    b = buckets(rpt)
    if not b:
        sys.exit("no sample rows parsed from " + rpt)
    ts = sorted(b)
    t0 = ts[0]
    xs = [t - t0 for t in ts]
    # samples in a 1 s bucket / 199 Hz per thread = cores busy that second
    spin = [b[t][0] / FREQ for t in ts]
    work = [b[t][1] / FREQ for t in ts]
    tot = [s + w for s, w in zip(spin, work)]

    # Leaf phases only; parents nest and would double-draw the strip.
    parent = set()
    for nm, a0, a1 in ph:
        for nm2, c0, c1 in ph:
            if nm2 != nm and a0 <= c0 and c1 <= a1 and (c1 - c0) < (a1 - a0):
                parent.add(nm)
                break
    bands = [(a - off - t0, c - off - t0, nm) for nm, a, c in ph if nm not in parent]

    n_parts = None
    if partition is not None and log:
        wins = partition_windows(log)
        n_parts = len(wins)
        win = wins.get(partition)
        if win is None:
            sys.exit("no PARTITION idx=%d in %s (have %s)"
                     % (partition, log, sorted(wins)))
        p0, p1 = win[0] - off - t0, win[1] - off - t0
        # Rebase everything to the partition's own start, same as the whole-run
        # plot rebases to the run's start -- so this reads exactly like that
        # figure, just for one partition instead of all twelve back to back.
        keep = [i for i, x in enumerate(xs) if p0 <= x <= p1]
        xs = [xs[i] - p0 for i in keep]
        spin = [spin[i] for i in keep]
        work = [work[i] for i in keep]
        tot = [tot[i] for i in keep]
        bands = [(max(a, p0) - p0, min(c, p1) - p0, nm) for a, c, nm in bands
                 if a < p1 and c > p0]
        if not xs:
            sys.exit("no perf samples fall inside partition %d's own window "
                     "(%.1f-%.1fs) -- check the run's clock_offset" % (partition, p0, p1))

    ps.apply()
    # Two y-scales of the same series, not a dual axis: the full range is dominated
    # by a 350-core spin burst, which renders witgen -- the longest phase in the run
    # at 275 s -- as an invisible ~1-core line. The zoom panel is the same data with
    # a 25-core ceiling so the single-threaded phases are legible at all.
    fig = plt.figure(figsize=(13.5, 8.8), dpi=150)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.6, 1.5, 0.34], hspace=0.20,
                          left=0.075, right=0.985, top=0.885, bottom=0.155)
    ax, axz, axp = (fig.add_subplot(gs[i]) for i in range(3))

    ax.fill_between(xs, work, 0, color=WORK, alpha=0.9, linewidth=0,
                    label="doing the prover's work")
    ax.fill_between(xs, tot, work, color=SPIN, alpha=0.9, linewidth=0,
                    label="OpenMP barrier spin")
    ax.axhline(CORES, color="#C2453B", linewidth=0.9, linestyle="--")
    ax.text(max(xs) * 0.004, CORES, " 192 physical cores", va="bottom", ha="left",
            fontsize=7.8, color="#C2453B")
    ax.set_ylim(0, max(max(tot) * 1.18, CORES * 1.10))
    ax.set_xlim(0, max(xs))
    ax.set_ylabel("cores busy", fontsize=9.5, color=INK2)
    ax.grid(axis="y", color="gray", linestyle="dotted", linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(True)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.6)
    ax.set_xticklabels([])
    mw, ms = sum(work) / len(work), sum(spin) / len(spin)
    ax.text(0.998, 0.93,
            "mean %.0f of 192 cores busy  ·  %.0f working, %.0f spinning (%.0f%%)"
            % (mw + ms, mw, ms, 100 * ms / (mw + ms) if (mw + ms) else 0),
            transform=ax.transAxes, ha="right", fontsize=8.8, color=INK)
    ax.legend(frameon=True, facecolor="white", edgecolor="black", framealpha=0.95, prop={"weight": "bold", "size": 12}, fontsize=8.4, ncol=2, loc="upper left",
              bbox_to_anchor=(0.0, 1.0))

    # zoom: identical series, clipped ceiling
    axz.fill_between(xs, work, 0, color=WORK, alpha=0.9, linewidth=0)
    axz.fill_between(xs, tot, work, color=SPIN, alpha=0.9, linewidth=0)
    axz.set_ylim(0, 25)
    axz.set_xlim(0, max(xs))
    axz.set_ylabel("cores busy\n(zoom to 25)", fontsize=9.5, color=INK2)
    axz.grid(axis="y", color="gray", linestyle="dotted", linewidth=0.7)
    axz.set_axisbelow(True)
    for sp in ("top", "right"):
        axz.spines[sp].set_visible(True)
    axz.spines["left"].set_color(GRID)
    axz.spines["bottom"].set_color(GRID)
    axz.tick_params(colors=INK2, labelsize=8.6)
    axz.set_xticklabels([])
    wit = [w for x, w in zip(xs, work) if any(a <= x <= c and nm == "witgen"
                                              for a, c, nm in bands)]
    if wit:
        axz.text(0.998, 0.88, "witgen runs at %.1f cores — single-threaded, and the "
                 "longest phase in the run" % (sum(wit) / len(wit)),
                 transform=axz.transAxes, ha="right", fontsize=8.6, color=INK)

    for a, c, nm in bands:
        axp.barh(0, c - a, 0.8, left=a, color=COLOR.get(nm, "#9AA3A8"), linewidth=0)
        if c - a > max(xs) * 0.045:
            axp.text((a + c) / 2, 0, nm, ha="center", va="center", fontsize=7.4,
                     color="white", fontweight="bold")
    axp.set_xlim(0, max(xs))
    axp.set_ylim(-0.5, 0.5)
    axp.set_yticks([])
    axp.set_xlabel("seconds into partition %d" % partition if partition is not None
                    else "seconds into the run", fontsize=9.5, color=INK2)
    axp.tick_params(colors=INK2, labelsize=8.6)
    for s in ("top", "right", "left"):
        axp.spines[s].set_visible(True)
    axp.spines["bottom"].set_color(GRID)

    # A color-only legend for the strip: segments too narrow for their inline
    # text label (the >4.5%-of-width gate above) are otherwise unidentifiable.
    # Fixed SPEC order, not appearance order, so a phase is the same color/legend
    # position in every figure this script draws, whether or not it happens to
    # show up here.
    present = {nm for _, _, nm in bands}
    handles = [Patch(facecolor=COLOR[nm], label=nm) for nm, *_ in SPEC if nm in present]
    if handles:
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.0),
                  frameon=True, facecolor="white", edgecolor="black", framealpha=0.95, prop={"weight": "bold", "size": 12}, fontsize=8.4, ncol=min(len(handles), 6),
                  labelcolor=INK, handlelength=1.4, handleheight=1.4,
                  columnspacing=1.3)

    name = os.path.basename(run_dir.rstrip("/"))
    fig.suptitle("zkGPT on CPU: cores busy, and how many are only waiting",
                 x=0.075, y=0.965, ha="left", fontsize=13.5, color=INK,
                 weight="semibold")
    scope = ("partition %d of %d -- every partition follows the same pattern"
              % (partition, n_parts) if partition is not None else "whole run")
    fig.text(0.075, 0.905,
             "%s  ·  GPT-2 len %s, 192 threads, all offloads on their CPU reference "
             "arm  ·  %s, %.0f s  ·  perf -F %d Hz per thread, %g s buckets"
             % (name, seq_len(log) if log else "?", scope, max(xs), FREQ,
                (ts[1] - ts[0]) if len(ts) > 1 else 1),
             ha="left", fontsize=8.6, color=INK2)
    fig.savefig(out_path, transparent=True, dpi=300)
    plt.close(fig)
    return out_path, mw, ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("-o", "--out")
    # Default to partition 0, not the whole run: every partition follows the same
    # pattern, so one is shown at a legible scale. --all-partitions
    # draws the whole-run figure instead, for when the run-length total itself
    # (not the per-partition shape) is what's wanted.
    ap.add_argument("--partition", type=int, default=0)
    ap.add_argument("--all-partitions", action="store_true")
    a = ap.parse_args()
    partition = None if a.all_partitions else a.partition
    out = a.out or os.path.join(a.run_dir, "figs", "utilisation.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    p, mw, ms = draw(a.run_dir, out, partition=partition)
    print("wrote %s  (mean %.0f working, %.0f spinning of 192 cores)" % (p, mw, ms))


if __name__ == "__main__":
    main()
