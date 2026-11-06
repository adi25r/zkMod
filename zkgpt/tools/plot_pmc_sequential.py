#!/usr/bin/env python3
"""plot_pmc_sequential.py -- the whole sequential run, GPU behaviour second by
second, all partitions concatenated on ONE real wall-clock axis.

plot_pmc_timeline.py answers "how does partition 3 behave over its own
lifetime" -- every figure resets to t=0 at that partition's start, so partition
3's figure cannot be lined up against partition 4's. This answers a different
question: "what does the device do across the ENTIRE run", with every
partition's window kept at its own place on a single timeline, so a reader can
see the run breathe -- a fast partition followed by a slow one, or the gap
between one partition's proving ending and the next's build finishing (there
is none here: `run_split_sequential` builds partition i+1 only after
partition i's PROOF completes, not concurrently with it, so back-to-back
partitions abut with no visible gap).

  panel 1  GPU busy, % of bin
  panel 2  HBM bandwidth, GB/s
  panel 3  occupancy, waves per CU (gfx90a holds 32)
  panel 4  VALU busy / MemUnit busy / L2 hit, %
  strip    every partition's phases, back to back, with a boundary tick and
           "part N" label at each partition's own build start

SAME THREE COUNTER PASSES AS plot_pmc_timeline.py, same caveat: bandwidth,
occupancy/VALU and MemUnit/L2 come from three separate profiled runs of the
same workload (rocprofv3 drops counters that do not fit one pass), aligned by
wall-clock position, not literally simultaneous.

BIN WIDTH DEFAULTS COARSER THAN THE PER-PARTITION FIGURE (10 ms, not 1 ms).
A full sequential run is ~12x one partition's span; at 1 ms that is hundreds of
thousands of bins for no visible gain -- the panels render fewer pixels wide
than that. Pass --bin-ms to change it.

Usage:
  tools/plot_pmc_sequential.py <pmc_workload/arm>
  tools/plot_pmc_sequential.py <arm> --bin-ms 5 -o out.png
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plotstyle as ps  # noqa: E402  house style (paper scripts)
from plotstyle import COLOR  # noqa: E402  phase colours, one source of truth
from plot_perf_phases import seq_len                              # noqa: E402
from plot_pmc_timeline import (ranges, counters, binned, busy_frac,  # noqa: E402
                               style_axis, S1, S2, S3, CU_WAVES, HBM_PEAK)

BG, INK, INK2, GRID = "none", "black", "black", "gray"


def draw(arm_dir, out_path, bin_ms=10.0):
    sets = {n: os.path.join(arm_dir, n, "pmc")
            for n in ("bandwidth", "cache", "compute")}
    missing = [n for n, p in sets.items() if not os.path.isdir(p)]
    if missing:
        sys.exit("missing counter pass(es) under %s: %s"
                 % (arm_dir, ", ".join(missing)))

    series, rgs = {}, {}
    for name, d in sets.items():
        rgs[name] = ranges(d)
        series[name] = counters(d)

    parts = sorted(int(k[4:]) for k in rgs["compute"]
                   if k.startswith("part") and k[4:].isdigit())
    if not parts:
        sys.exit("no ROCTx part<i> ranges under " + sets["compute"])

    # The global window: earliest build start to latest prove end, over every
    # partition and every pass (a pass's own profiling overhead shifts its
    # absolute timestamps slightly, so take the widest span any pass reports).
    def span_of(rg):
        los, his = [], []
        for i in parts:
            key = "part%d" % i
            if key not in rg:
                continue
            a0 = min(a for a, _b in rg[key])
            b0 = max(b for _a, b in rg[key])
            bkey = key + "_build"
            if bkey in rg:
                a0 = min(a0, min(a for a, _b in rg[bkey]))
            los.append(a0); his.append(b0)
        return (min(los), max(his)) if los else (None, None)

    windows = {name: span_of(rg) for name, rg in rgs.items()}
    if any(w[0] is None for w in windows.values()):
        sys.exit("a counter pass is missing some partitions' ranges")

    # NOT a cross-pass min/max: the three counter passes are three SEPARATE
    # invocations of the binary, run minutes apart in real wall-clock time
    # (bandwidth, then cache, then compute). Taking
    # min(every start)/max(every end) across them measures the GAP between
    # passes, not the run. The span is the LONGEST of the three passes' own individual
    # durations, exactly as plot_pmc_timeline.py does for one partition.
    span_ns = max(w[1] - w[0] for w in windows.values())
    nbins = max(64, int(span_ns / (bin_ms * 1e6)))

    def bin_series(name, key, how):
        w0, w1 = windows[name]
        vals = binned(series[name], key, w0, w1, nbins, how)
        # windows differ slightly per pass; index by fraction of ITS OWN span so
        # every series still lines up on the shared 0..span x-axis below.
        return vals

    t0b, t1b = windows["bandwidth"]
    binw = (t1b - t0b) / nbins / 1e9
    rd = bin_series("bandwidth", "FETCH_SIZE", "sum")
    wr = bin_series("bandwidth", "WRITE_SIZE", "sum")
    bw = [(rd[i] + wr[i]) * 1024.0 / 1e9 / binw if binw > 0 else 0.0
          for i in range(nbins)]

    occ = bin_series("compute", "MeanOccupancyPerCU", "wmean")
    valu = bin_series("compute", "VALUBusy", "wmean")
    memu = bin_series("cache", "MemUnitBusy", "wmean")
    l2 = bin_series("cache", "L2CacheHit", "wmean")
    t0c, t1c = windows["compute"]
    busy = busy_frac(series["compute"], t0c, t1c, nbins)

    span = span_ns / 1e9
    xs = [(i + 0.5) * span / nbins for i in range(nbins)]

    # Partition boundaries + every phase, all in the SAME real-time frame (t0c).
    boundaries = []
    for i in parts:
        key, bkey = "part%d" % i, "part%d_build" % i
        rg = rgs["compute"]
        a0 = min(a for a, _b in rg.get(bkey, rg[key]))
        boundaries.append((i, (a0 - t0c) / 1e9))
    phases = sorted((a - t0c, b - t0c, nm) for nm, ivs in rgs["compute"].items()
                    for a, b in ivs if nm in COLOR)

    ps.apply()
    fig = plt.figure(figsize=(15.5, 10.4), dpi=150)
    gs = fig.add_gridspec(5, 1, height_ratios=[0.85, 1.0, 1.0, 1.15, 0.42],
                          hspace=0.30, left=0.068, right=0.988, top=0.888,
                          bottom=0.085)
    ax0, ax1, ax2, ax3, axp = (fig.add_subplot(gs[i]) for i in range(5))

    ax0.fill_between(xs, busy, 0, color="#D97B22", alpha=0.85, linewidth=0)
    style_axis(ax0, 108)
    ax0.set_ylabel("GPU busy\n(% of bin)", fontsize=9, color=INK2)
    ax0.text(0.998, 0.86, "mean %.0f%% of the run has a kernel resident"
             % (sum(busy) / len(busy)), transform=ax0.transAxes, ha="right",
             fontsize=8.4, color=INK)

    ax1.fill_between(xs, bw, 0, color="#3AA0B0", alpha=0.85, linewidth=0)
    style_axis(ax1, max(bw) * 1.28 if any(bw) else 1)
    ax1.set_ylabel("HBM bandwidth\n(GB/s)", fontsize=9, color=INK2)
    ax1.text(0.998, 0.86, "peak %.0f GB/s  ·  %.0f%% of the MI210's %d"
             % (max(bw), 100 * max(bw) / HBM_PEAK, HBM_PEAK),
             transform=ax1.transAxes, ha="right", fontsize=8.4, color=INK)

    ax2.fill_between(xs, occ, 0, color="#4E8F55", alpha=0.85, linewidth=0)
    style_axis(ax2, CU_WAVES * 1.1)
    ax2.axhline(CU_WAVES, color="#C2453B", linewidth=0.9, linestyle="--")
    ax2.text(span * 0.002, CU_WAVES, " 32 waves/CU, the gfx90a maximum",
             va="top", ha="left", fontsize=7.6, color="#C2453B")
    ax2.set_ylabel("occupancy\n(waves per CU)", fontsize=9, color=INK2)

    for ys, c, lab in ((valu, S1, "VALU busy"), (memu, S2, "MemUnit busy"),
                       (l2, S3, "L2 hit rate")):
        ax3.plot(xs, ys, color=c, linewidth=1.4, label=lab)
        j = max(range(nbins), key=lambda i: ys[i])
        ax3.text(xs[j], min(ys[j] + 3, 101), lab, color=c, fontsize=8.2,
                 ha="center", fontweight="bold")
    style_axis(ax3, 112)
    ax3.set_ylabel("utilisation\n(%)", fontsize=9, color=INK2)
    ax3.legend(frameon=True, facecolor="white", edgecolor="black", framealpha=0.95, prop={"weight": "bold", "size": 12}, fontsize=8.2, ncol=3, loc="upper left",
               bbox_to_anchor=(0.0, -0.14))

    for a, b, nm in phases:
        x0, x1 = a / 1e9, b / 1e9
        axp.barh(0, x1 - x0, 0.8, left=x0, color=COLOR[nm], linewidth=0)
        if x1 - x0 > span * 0.02:
            axp.text((x0 + x1) / 2, 0, nm, ha="center", va="center",
                     fontsize=6.6, color="white", fontweight="bold", rotation=0,
                     clip_on=True)
    for i, bx in boundaries:
        axp.axvline(bx, color=BG, linewidth=1.2, zorder=3)
        axp.text(bx, 0.62, "part%d" % i, ha="left", va="bottom", fontsize=6.8,
                 color=INK2, rotation=0)
    axp.set_xlim(0, span)
    axp.set_ylim(-0.5, 0.62)
    axp.set_yticks([])
    axp.set_xlabel("seconds into the run", fontsize=9, color=INK2)
    axp.tick_params(colors=INK2, labelsize=8.5)
    for s in ("top", "right", "left"):
        axp.spines[s].set_visible(True)
    axp.spines["bottom"].set_color(GRID)
    for ax in (ax0, ax1, ax2, ax3):
        ax.set_xlim(0, span)
        ax.set_xticklabels([])

    name = os.path.basename(arm_dir.rstrip("/"))
    log = os.path.join(arm_dir, "compute", "pmc.log")
    fig.suptitle("The whole sequential run, second by second -- %s" % name,
                 x=0.068, y=0.972, ha="left", fontsize=14.5, color=INK,
                 weight="semibold")
    fig.text(0.068, 0.925,
             "GPT-2 len %s, %d partitions run one after another, one MI210   ·   "
             "%.1f s   ·   %d bins of %.1f ms\n"
             "Three separate counter passes over the same workload, aligned by "
             "wall-clock position -- not literally simultaneous."
             % (seq_len(log) if os.path.exists(log) else "?", len(parts), span,
                nbins, span / nbins * 1000),
             ha="left", fontsize=8.6, color=INK2, linespacing=1.5)
    fig.savefig(out_path, transparent=True, dpi=300)
    plt.close(fig)
    return out_path, span, len(parts), max(bw), max(occ)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm_dir",
                    help="pmc_workload/<arm>, holding bandwidth/ cache/ compute/")
    ap.add_argument("--bin-ms", type=float, default=10.0)
    ap.add_argument("-o", "--out")
    a = ap.parse_args()
    out = a.out or os.path.join(a.arm_dir, "figs", "sequential.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    p, span, n, pk, occ = draw(a.arm_dir, out, a.bin_ms)
    print("wrote %s  (%d partitions, %.1f s, peak %.0f GB/s, peak %.1f waves/CU)"
          % (p, n, span, pk, occ))


if __name__ == "__main__":
    main()
