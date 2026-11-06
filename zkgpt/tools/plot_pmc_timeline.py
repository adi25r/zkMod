#!/usr/bin/env python3
"""plot_pmc_timeline.py — one partition's GPU behaviour over its own lifetime.

pmc_by_phase.py answers "how does range_prove behave"; this answers "what does the
device do, second by second, while ONE partition runs". Same counters, resolved on
time instead of folded into a phase average -- which is the only way to see that a
phase's mean is the average of two different regimes, or that a spike sits at a
phase boundary rather than inside it.

  panel 1  HBM bandwidth, GB/s          (FETCH_SIZE + WRITE_SIZE per bin)
  panel 2  occupancy, waves per CU      (MeanOccupancyPerCU; gfx90a holds 32)
  panel 3  utilisation, %               (VALUBusy, MemUnitBusy, L2CacheHit)
  strip    the partition's phases, coloured as in plot_overlay_trace

THE THREE PANELS COME FROM THREE SEPARATE RUNS, and that is forced, not sloppy:
FETCH_SIZE, WRITE_SIZE and L2CacheHit need six TCC counters between them and do
not fit the hardware slots beside VALUBusy, and rocprofv3 silently DROPS what does
not fit rather than failing. So each counter set was collected in its own pass over
the same workload, and the panels are aligned on time-since-this-partition-started
within their own pass. The workload is deterministic and the phase sequence
matches, but a feature in panel 1 and a feature in panel 3 at the same x are NOT
the same instant -- they are the same point in the same program. Read them as
aligned, not simultaneous.

Usage:
  tools/plot_pmc_timeline.py <pmc_workload/arm> --partition 0
  tools/plot_pmc_timeline.py <pmc_workload/arm> --all      # one figure per partition
"""
import argparse
import collections
import csv
import os
import sys

import matplotlib
import matplotlib.colors
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plotstyle as ps  # noqa: E402  house style (paper scripts)
from plotstyle import COLOR  # noqa: E402  phase colours, one source of truth
from plot_perf_phases import seq_len  # noqa: E402  reads chunk_width off a PARTITION line

# dataviz reference palette, light mode, categorical slots 1-3 in fixed order.
# Validated with scripts/validate_palette.js: worst adjacent CVD dE 9.2 (deutan),
# normal-vision 27.6, both PASS. Slot 3 (#1baf7a) measures 2.74:1 on this surface
# -> contrast WARN, so the relief rule applies and every series in panel 3 carries
# a direct label rather than relying on the legend alone.
S1, S2, S3 = [matplotlib.colors.to_hex(c) for c in ps.blues(3)]
BG, INK, INK2, GRID = "none", "black", "black", "gray"
CU_WAVES = 32.0          # gfx90a: 8 waves/SIMD x 4 SIMDs
HBM_PEAK = 1638.0        # MI210 HBM2e, GB/s


def ranges(pmc_dir):
    """ROCTx ranges from the marker trace: {name: [(t0, t1), ...]}."""
    out = collections.defaultdict(list)
    for root, _d, files in os.walk(pmc_dir):
        for fn in files:
            if not fn.endswith("marker_api_trace.csv"):
                continue
            with open(os.path.join(root, fn), newline="", errors="replace") as f:
                for r in csv.DictReader(f):
                    nm = r.get("Function") or r.get("Name")
                    try:
                        out[nm].append((int(r["Start_Timestamp"]),
                                        int(r["End_Timestamp"])))
                    except (KeyError, ValueError, TypeError):
                        continue
    return out


def counters(pmc_dir):
    """[(t0, t1, name, value)] from every counter_collection.csv under pmc_dir."""
    rows = []
    for root, _d, files in os.walk(pmc_dir):
        for fn in files:
            if not fn.endswith("counter_collection.csv"):
                continue
            with open(os.path.join(root, fn), newline="", errors="replace") as f:
                for r in csv.DictReader(f):
                    try:
                        rows.append((int(r["Start_Timestamp"]),
                                     int(r["End_Timestamp"]),
                                     r["Counter_Name"], float(r["Counter_Value"])))
                    except (KeyError, ValueError, TypeError):
                        continue
    return rows


def binned(rows, want, t0, t1, nbins, how):
    """Bin one counter into `nbins` over [t0,t1]. how='sum' or 'wmean'.

    A dispatch is SPREAD across every bin it overlaps, in proportion to the time it
    spends there -- it is not charged to the bin containing its start. That
    distinction is invisible at coarse bins and wrong at fine ones: a 20 ms kernel
    dumped whole into one 1 ms bin read 13,688 GB/s on a card whose ceiling is 1638.

    Rates are weighted by the overlap duration, so a 40 ms kernel at 80% says more
    about a bin than a 4 us kernel at 10%.
    """
    tot = [0.0] * nbins
    wgt = [0.0] * nbins
    span = t1 - t0
    if span <= 0:
        return tot
    w = span / nbins
    for a, b, cn, cv in rows:
        if cn != want:
            continue
        lo, hi = max(a, t0), min(max(b, a + 1), t1)
        if hi <= lo:
            continue
        i0 = int((lo - t0) / w)
        i1 = min(nbins - 1, int((hi - t0) / w))
        dur = hi - lo
        for i in range(i0, i1 + 1):
            bs, be = t0 + i * w, t0 + (i + 1) * w
            ov = min(hi, be) - max(lo, bs)
            if ov <= 0:
                continue
            frac = ov / dur
            if how == "sum":
                tot[i] += cv * frac        # bytes land where they were moved
            else:
                tot[i] += cv * ov
                wgt[i] += ov
    if how == "wmean":
        return [tot[i] / wgt[i] if wgt[i] else 0.0 for i in range(nbins)]
    return tot


def busy_frac(rows, t0, t1, nbins):
    """% of each bin with a kernel resident on the device.

    Derived from dispatch start/end stamps rather than a counter: every row of the
    counter CSV carries the dispatch it belongs to, so the union of those intervals
    IS device-busy time. Under --pmc dispatches are serialised, so this is "was the
    GPU running a kernel", not "were several kernels overlapping".
    """
    span = t1 - t0
    out = [0.0] * nbins
    if span <= 0:
        return out
    w = span / nbins
    for a_, b_ in {(x[0], x[1]) for x in rows}:
        lo, hi = max(a_, t0), min(b_, t1)
        if hi <= lo:
            continue
        i0, i1 = int((lo - t0) / w), min(nbins - 1, int((hi - t0) / w))
        for i in range(i0, i1 + 1):
            bs, be = t0 + i * w, t0 + (i + 1) * w
            out[i] += max(0.0, min(hi, be) - max(lo, bs))
    return [min(100.0, 100.0 * v / w) for v in out]


def style_axis(ax, ymax=None):
    if ymax:
        ax.set_ylim(0, ymax)
    ax.grid(axis="y", color="gray", linestyle="dotted", linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(True)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5)


def draw(arm_dir, part, out_path, bin_ms=1.0, n_parts=1):
    sets = {n: os.path.join(arm_dir, n, "pmc")
            for n in ("bandwidth", "cache", "compute")}
    missing = [n for n, p in sets.items() if not os.path.isdir(p)]
    if missing:
        sys.exit("missing counter pass(es) under %s: %s"
                 % (arm_dir, ", ".join(missing)))

    series, windows, phases = {}, {}, None
    for name, d in sets.items():
        rg = ranges(d)
        key = "part%d" % part
        if key not in rg:
            have = sorted(k for k in rg
                          if k.startswith("part") and k[4:].isdigit())
            sys.exit("%s: no ROCTx range %r (have %s)" % (d, key, ", ".join(have)))
        # part<i> covers PROVING only; witgen sits inside part<i>_build, which ends
        # where part<i> begins. A window of just part<i> silently drops witness
        # generation -- the phase the GPU port was about. Widen to the build.
        w0, w1 = rg[key][0]
        bkey = key + "_build"
        if bkey in rg:
            w0 = min(w0, min(a for a, _b in rg[bkey]))
        windows[name] = (w0, w1)
        series[name] = counters(d)
        if name == "compute":
            t0, t1 = windows[name]
            phases = sorted((a, b, nm) for nm, ivs in rg.items() for a, b in ivs
                            if nm in COLOR and t0 <= a and b <= t1)

    # Bin width comes from the data's own resolution: dispatches are timestamped
    # individually, so 1 ms bins are real measurement, not interpolation.
    span_ns = max(w[1] - w[0] for w in windows.values())
    nbins = max(16, int(span_ns / (bin_ms * 1e6)))

    # KiB per bin -> GB/s. The bin is a real time window, so this is bandwidth
    # achieved over that window -- not the per-kernel figure pmc_by_phase reports.
    t0b, t1b = windows["bandwidth"]
    binw = (t1b - t0b) / nbins / 1e9
    rd = binned(series["bandwidth"], "FETCH_SIZE", t0b, t1b, nbins, "sum")
    wr = binned(series["bandwidth"], "WRITE_SIZE", t0b, t1b, nbins, "sum")
    bw = [(rd[i] + wr[i]) * 1024.0 / 1e9 / binw if binw > 0 else 0.0
          for i in range(nbins)]

    t0c, t1c = windows["compute"]
    occ = binned(series["compute"], "MeanOccupancyPerCU", t0c, t1c, nbins, "wmean")
    valu = binned(series["compute"], "VALUBusy", t0c, t1c, nbins, "wmean")
    t0k, t1k = windows["cache"]
    memu = binned(series["cache"], "MemUnitBusy", t0k, t1k, nbins, "wmean")
    l2 = binned(series["cache"], "L2CacheHit", t0k, t1k, nbins, "wmean")
    busy = busy_frac(series["compute"], t0c, t1c, nbins)

    span = max((w[1] - w[0]) / 1e9 for w in windows.values())
    xs = [(i + 0.5) * span / nbins for i in range(nbins)]

    ps.apply()
    fig = plt.figure(figsize=(13.5, 10.0), dpi=150)
    gs = fig.add_gridspec(5, 1, height_ratios=[0.85, 1.0, 1.0, 1.15, 0.34],
                          hspace=0.30, left=0.078, right=0.985, top=0.878,
                          bottom=0.10)
    ax0, ax1, ax2, ax3, axp = (fig.add_subplot(gs[i]) for i in range(5))

    ax0.fill_between(xs, busy, 0, color="#D97B22", alpha=0.85, linewidth=0)
    style_axis(ax0, 108)
    ax0.set_ylabel("GPU busy\n(% of bin)", fontsize=9, color=INK2)
    ax0.text(0.998, 0.86, "mean %.0f%% of this partition has a kernel resident"
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
    ax2.text(span * 0.004, CU_WAVES, " 32 waves/CU, the gfx90a maximum",
             va="top", ha="left", fontsize=7.6, color="#C2453B")
    ax2.set_ylabel("occupancy\n(waves per CU)", fontsize=9, color=INK2)

    for ys, c, lab in ((valu, S1, "VALU busy"), (memu, S2, "MemUnit busy"),
                       (l2, S3, "L2 hit rate")):
        ax3.plot(xs, ys, color=c, linewidth=2.0, label=lab)
        # Relief rule: slot 3 is under 3:1 on this surface, so each series is
        # direct-labelled at its own maximum rather than left to the legend.
        j = max(range(nbins), key=lambda i: ys[i])
        ax3.text(xs[j], min(ys[j] + 3, 101), lab, color=c, fontsize=8.2,
                 ha="center", fontweight="bold")
    style_axis(ax3, 112)
    ax3.set_ylabel("utilisation\n(%)", fontsize=9, color=INK2)
    # Legend BELOW the axes: the direct labels sit at each series' own maximum,
    # which for these kernels is up against the top of the panel, exactly where an
    # inside legend goes. Both are required (>=2 series need a legend; the contrast
    # WARN on slot 3 needs the direct labels), so the legend moves instead.
    ax3.legend(frameon=True, facecolor="white", edgecolor="black", framealpha=0.95, prop={"weight": "bold", "size": 12}, fontsize=8.2, ncol=3, loc="upper left",
               bbox_to_anchor=(0.0, -0.14))

    for a, b, nm in phases or []:
        x0, x1 = (a - t0c) / 1e9, (b - t0c) / 1e9
        axp.barh(0, x1 - x0, 0.8, left=x0, color=COLOR[nm], linewidth=0)
        if x1 - x0 > span * 0.045:
            axp.text((x0 + x1) / 2, 0, nm, ha="center", va="center",
                     fontsize=7.4, color="white", fontweight="bold")
    axp.set_xlim(0, span)
    axp.set_ylim(-0.5, 0.5)
    axp.set_yticks([])
    axp.set_xlabel("seconds since this partition started", fontsize=9, color=INK2)
    axp.tick_params(colors=INK2, labelsize=8.5)
    for s in ("top", "right", "left"):
        axp.spines[s].set_visible(True)
    axp.spines["bottom"].set_color(GRID)
    for ax in (ax0, ax1, ax2, ax3):
        ax.set_xlim(0, span)
        ax.set_xticklabels([])

    fig.suptitle("Partition %d on the GPU, second by second" % part, x=0.078,
                 y=0.968, ha="left", fontsize=14, color=INK, weight="semibold")
    log = os.path.join(arm_dir, "compute", "pmc.log")
    fig.text(0.078, 0.918,
             "GPT-2 len %s, %d partitions run sequentially, one MI210   ·   "
             "%.1f s   ·   %d bins of %.2f ms\n"
             "Three separate counter passes over the same workload (the counters "
             "do not fit one pass), aligned on time since this partition started."
             % (seq_len(log) if os.path.exists(log) else "?", n_parts, span,
                nbins, span / nbins * 1000),
             ha="left", fontsize=8.6, color=INK2, linespacing=1.5)

    fig.savefig(out_path, transparent=True, dpi=300)
    plt.close(fig)
    return out_path, span, max(bw), max(occ)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm_dir",
                    help="pmc_workload/<arm>, holding bandwidth/ cache/ compute/")
    ap.add_argument("--partition", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--bin-ms", type=float, default=1.0,
                    help="bin width in ms (default 1; the dispatch data "
                         "is timestamped per kernel, so 1 ms is real)")
    ap.add_argument("-o", "--out")
    a = ap.parse_args()

    rg = ranges(os.path.join(a.arm_dir, "compute", "pmc"))
    all_parts = sorted(int(k[4:]) for k in rg
                       if k.startswith("part") and k[4:].isdigit())
    parts = all_parts if a.all else [a.partition]
    outdir = a.out or os.path.join(a.arm_dir, "figs")
    os.makedirs(outdir, exist_ok=True)
    for p in parts:
        path, span, pk, occ = draw(a.arm_dir, p,
                                   os.path.join(outdir, "part%02d.png" % p), a.bin_ms,
                                   n_parts=len(all_parts))
        print("wrote %s  (%.1f s, peak %.0f GB/s, peak %.1f waves/CU)"
              % (path, span, pk, occ))


if __name__ == "__main__":
    main()
