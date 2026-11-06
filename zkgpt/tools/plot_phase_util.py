#!/usr/bin/env python3
"""plot_phase_util.py -- per-phase AVERAGE device utilisation, one bar group per phase.

plot_pmc_timeline.py resolves a single partition's counters on time; pmc_by_phase.py
prints the same numbers as a text table. This is the figure between them: every phase
of the run, averaged over every partition, as four bars you can read at a glance --
which is the form the question "where is the device idle?" actually wants.

  bars    GPU busy (kernel residency), VALU busy, MemUnit busy, occupancy
  table   the counters that do not belong on a 0-100 axis: L2 hit, MemUnit stalled,
          HBM GB/s, kernel seconds, and the phase's share of the run

FOUR THINGS ABOUT THE NUMBERS, none of them cosmetic:

1. THE BARS COME FROM DIFFERENT PASSES. VALUBusy and MeanOccupancyPerCU are the
   compute pass; MemUnitBusy, MemUnitStalled and L2CacheHit the cache pass; FETCH/
   WRITE the bandwidth pass. Six TCC counters do not fit one hardware slot set and
   rocprofv3 silently DROPS what does not fit. Same workload, same phase sequence,
   three runs. A phase with no bar for a metric did not collect it -- that is not a
   measured zero, and it is never padded to one.

2. "GPU busy" is kernel RESIDENCY -- the union of dispatch intervals over the phase's
   wall time -- and under --pmc dispatches are serialised, which stretches the gaps
   between them and not the kernels. So it is a LOWER BOUND on the unprofiled run's
   busy fraction. Use it to rank phases against each other, not as an absolute.

3. Occupancy is MeanOccupancyPerCU, which counts WAVES PER CU, not a percentage.
   gfx90a holds 32, so it is plotted as a percentage OF 32 and labelled with the raw
   wave count. Reading it as a percent directly is a 32x error.

4. Rates are averaged weighted by kernel duration; counts are summed. A 40 ms kernel
   at 80% says more about the phase than a 4 us kernel at 10%.

Usage:
  tools/plot_phase_util.py <pmc_workload/arm>              # e.g. .../zkmod
  tools/plot_phase_util.py <arm> -o out.png --title "..."
"""
import argparse
import collections
import os
import sys

import matplotlib
import matplotlib.colors
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plotstyle as ps  # noqa: E402  house style (paper scripts)
from pmc_by_phase import load_ranges, find, col, RATE               # noqa: E402

# dataviz reference palette, light mode, categorical slots 1-4 in fixed order.
S1, S2, S3, S4 = [matplotlib.colors.to_hex(c) for c in ps.blues(4)]
BG, INK, INK2, GRID = "none", "black", "black", "gray"
CU_WAVES = 32.0          # gfx90a: 8 waves/SIMD x 4 SIMDs
HBM_PEAK = 1638.0        # MI210 HBM2e, GB/s

BARS = [("busy", "GPU busy (kernel residency)", S1),
        ("VALUBusy", "VALU busy", S2),
        ("MemUnitBusy", "memory unit busy", S3),
        ("occ_pct", "occupancy (of 32 waves/CU)", S4)]


def counters(pass_dir):
    """[(t0, t1, counter, value)] for one counter pass."""
    rows = []
    for path in find(pass_dir, "counter_collection.csv"):
        import csv
        with open(path, newline="", errors="replace") as f:
            for r in csv.DictReader(f):
                t0, t1 = col(r, "Start_Timestamp"), col(r, "End_Timestamp")
                cn, cv = col(r, "Counter_Name"), col(r, "Counter_Value")
                if t0 is None or cn is None or cv is None:
                    continue
                try:
                    rows.append((int(t0), int(t1) if t1 else int(t0), cn, float(cv)))
                except ValueError:
                    continue
    return rows


def collect(arm_dir):
    """{phase: {metric: value}} folded over every pass and every partition."""
    agg = collections.defaultdict(lambda: collections.defaultdict(float))
    wall = collections.defaultdict(float)     # phase wall time, ns, one pass only
    kern = collections.defaultdict(float)     # union of dispatch intervals, ns
    have_wall = False

    for sub in sorted(os.listdir(arm_dir)):
        pd = os.path.join(arm_dir, sub)
        if not os.path.isdir(pd) or sub in ("figs",):
            continue
        rows = counters(pd)
        if not rows:
            continue
        ranges, _parts = load_ranges(pd)
        if not ranges:
            continue
        ranges.sort(key=lambda r: (r[2], -(r[3] - r[2])))

        def phase_of(t):
            best = None
            for _p, nm, a, b in ranges:
                if a <= t <= b and (best is None or (b - a) < best[1]):
                    best = (nm, b - a)
            return best[0] if best else None

        for t0, t1, cn, cv in rows:
            ph = phase_of(t0)
            if ph is None:
                continue
            w = max(1, t1 - t0)
            if cn in RATE:
                agg[ph][cn] += cv * w
                agg[ph]["_w_" + cn] += w
            else:
                agg[ph][cn] += cv

        # Wall and residency are pass-invariant in kind but not in value (the
        # profiler dilates each pass differently), so take them from ONE pass --
        # the first that has dispatches -- rather than mixing dilations.
        if not have_wall:
            have_wall = True
            for _p, nm, a, b in ranges:
                wall[nm] += b - a
            # union of dispatch intervals per phase; dispatches are serialised
            # under --pmc, but merge anyway so an overlap can never exceed wall
            byph = collections.defaultdict(list)
            for t0, t1, _cn, _cv in rows:
                ph = phase_of(t0)
                if ph is not None:
                    byph[ph].append((t0, max(t1, t0)))
            for ph, iv in byph.items():
                iv.sort()
                cur0, cur1 = iv[0]
                for a, b in iv[1:]:
                    if a > cur1:
                        kern[ph] += cur1 - cur0
                        cur0, cur1 = a, b
                    else:
                        cur1 = max(cur1, b)
                kern[ph] += cur1 - cur0

    out = {}
    for ph, a in agg.items():
        m = {}
        for cn in RATE:
            w = a.get("_w_" + cn, 0.0)
            if w:
                m[cn] = a[cn] / w
        if "MeanOccupancyPerCU" in m:
            m["occ_pct"] = 100.0 * m["MeanOccupancyPerCU"] / CU_WAVES
        m["kern_s"] = kern.get(ph, 0.0) / 1e9
        m["wall_s"] = wall.get(ph, 0.0) / 1e9
        if m["wall_s"] > 0:
            m["busy"] = min(100.0, 100.0 * m["kern_s"] / m["wall_s"])
        rd, wr = a.get("FETCH_SIZE", 0.0), a.get("WRITE_SIZE", 0.0)
        if (rd or wr) and m["kern_s"] > 0:
            m["GBs"] = (rd + wr) * 1024.0 / 1e9 / m["kern_s"]
        out[ph] = m
    return out


def draw(arm_dir, out_path, title=None):
    ph = collect(arm_dir)
    if not ph:
        sys.exit("no counter rows with a containing ROCTx range under " + arm_dir)
    order = sorted(ph, key=lambda p: -ph[p].get("wall_s", 0.0))
    run_s = sum(ph[p].get("wall_s", 0.0) for p in order)

    ps.apply()
    fig = plt.figure(figsize=(14.6, 1.05 * len(order) + 3.0), dpi=150)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.72, 1.0], wspace=0.035,
                          left=0.135, right=0.985, top=1 - 1.75 / (1.05 * len(order) + 3.0),
                          bottom=1.05 / (1.05 * len(order) + 3.0))
    ax, axt = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    h = 0.20
    for i, (key, lab, c) in enumerate(BARS):
        ys, xs = [], []
        for j, p in enumerate(order):
            if key in ph[p]:
                ys.append(j + (1.5 - i) * h)
                xs.append(ph[p][key])
        ax.barh(ys, xs, h * 0.92, color=c, linewidth=0, label=lab, zorder=3)
        for y, x in zip(ys, xs):
            ax.text(x + 0.9, y, "%.0f" % x if x >= 1 else "%.1f" % x,
                    va="center", ha="left", fontsize=7.4, color=INK2, zorder=4)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9.4)
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlabel("percent", fontsize=9.5, color=INK2)
    ax.grid(axis="x", color="gray", linestyle="dotted", linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(True)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.6)
    for j in range(len(order)):
        ax.axhline(j + 0.5, color=GRID, linewidth=0.7, zorder=1)
    ax.legend(frameon=True, facecolor="white", edgecolor="black", framealpha=0.95, prop={"weight": "bold", "size": 12}, fontsize=8.6, ncol=2, loc="lower left",
              bbox_to_anchor=(0.0, 1.005), handlelength=1.5, columnspacing=1.5)

    # right: the counters that do not share a 0-100 axis
    cols = [("waves/CU\nof 32", lambda m: m.get("MeanOccupancyPerCU")),
            ("L2 hit %", lambda m: m.get("L2CacheHit")),
            ("MemUnit\nstalled %", lambda m: m.get("MemUnitStalled")),
            ("HBM GB/s\nof 1638", lambda m: m.get("GBs")),
            ("kernel s", lambda m: m.get("kern_s")),
            ("% of run", lambda m: 100.0 * m.get("wall_s", 0) / run_s if run_s else None)]
    axt.set_xlim(0, len(cols))
    axt.set_ylim(len(order) - 0.5, -0.5)
    axt.axis("off")
    for k, (hd, _f) in enumerate(cols):
        axt.text(k + 0.5, -0.62, hd, ha="center", va="bottom", fontsize=8.0,
                 color=INK2, linespacing=1.35)
    for j, p in enumerate(order):
        if j % 2 == 0:
            axt.axhspan(j - 0.5, j + 0.5, color="#F2EFE9", zorder=0)
        for k, (_hd, f) in enumerate(cols):
            v = f(ph[p])
            axt.text(k + 0.5, j, "--" if v is None else
                     ("%.2f" % v if v < 10 else "%.0f" % v),
                     ha="center", va="center", fontsize=8.6, color=INK, zorder=2)

    name = os.path.basename(arm_dir.rstrip("/"))
    fig.suptitle(title or "Per-phase average GPU utilisation -- %s" % name,
                 x=0.135, y=1 - 0.42 / (1.05 * len(order) + 3.0), ha="left",
                 fontsize=13.6, color=INK, weight="semibold")
    fig.text(0.135, 1 - 0.92 / (1.05 * len(order) + 3.0),
             "averaged over every partition, weighted by kernel duration  ·  %.0f s of "
             "phase time  ·  counters from three separate rocprofv3 passes  ·  GPU busy "
             "is a lower bound (--pmc serialises dispatches)" % run_s,
             ha="left", fontsize=8.6, color=INK2)
    fig.savefig(out_path, bbox_inches="tight", transparent=True, dpi=300)
    plt.close(fig)
    return out_path, order, ph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm_dir")
    ap.add_argument("-o", "--out")
    ap.add_argument("--title")
    a = ap.parse_args()
    out = a.out or os.path.join(a.arm_dir, "figs", "phase_utilisation.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    p, order, ph = draw(a.arm_dir, out, a.title)
    print("wrote %s" % p)
    for q in order:
        m = ph[q]
        print("  %-22s busy %5.1f%%  VALU %5.1f%%  memU %5.1f%%  occ %4.1f wav/CU  "
              "%7.1f s" % (q, m.get("busy", 0), m.get("VALUBusy", 0),
                           m.get("MemUnitBusy", 0), m.get("MeanOccupancyPerCU", 0),
                           m.get("wall_s", 0)))


if __name__ == "__main__":
    main()
