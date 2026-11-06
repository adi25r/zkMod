#!/usr/bin/env python3
"""plot_pmc_kernels.py — which KERNEL each phase actually spends its time in.

The phase tables say range_prove is 12% VALU at 1.1 waves/CU. They cannot say why,
because a phase is not a unit of work -- it is a bag of dispatches, and in this
workload one kernel usually owns almost all of a phase's time. This resolves each
phase into its kernels, per partition, and puts the launch geometry beside them,
because that is what explains the occupancy: several of the dominant kernels launch
a SINGLE workgroup and therefore use one of the MI210's 104 CUs.

  bars     one row per phase, segments = kernel time, widest kernel first
  table    per kernel: time, VALU busy % AND VALU instructions/s, MemUnit busy %,
           occupancy, HBM bandwidth, L2 hit rate

VALUBusy and the instruction rate are both there because they answer different
questions and disagree in an informative way. VALUBusy divides by CU_NUM, so a
kernel that saturates one CU's ALUs reads ~1% and looks idle; the rate does not,
so it separates "barely uses the device" from "barely does arithmetic".
blendy_reduce_kernel is the second: 0.1 G instr/s against extrapolate_bk_var's
124.4, a thousandfold gap, for 25% of all kernel time.

Kernel time, not wall time: --pmc serialises dispatches, so the gaps between them
are profiler artefact and only the dispatch durations mean anything. A phase's bar
is therefore shorter than the phase's wall clock, deliberately.

Usage:
  tools/plot_pmc_kernels.py <pmc_workload/arm> --partition 0
  tools/plot_pmc_kernels.py <pmc_workload/arm> --all
"""
import argparse
import collections
import csv
import os
import re
import sys

import matplotlib
import matplotlib.colors
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plotstyle as ps  # noqa: E402  house style (paper scripts)
from plot_pmc_timeline import ranges  # noqa: E402

# dataviz reference palette, light mode, categorical slots 1-7 in fixed order.
# Validated with scripts/validate_palette.js on the adjacent pairlist (this is a
# stack, so adjacent is the right list): worst CVD dE 9.1 protan, worst
# normal-vision 19.6, both PASS. Three slots are under 3:1 on the surface, so the
# relief rule applies -- every segment wide enough to hold one is direct-labelled,
# and the legend names all of them.
SLOTS = [matplotlib.colors.to_hex(c) for c in ps.blues(7)]
OTHER = "#9AA3A8"
BG, INK, INK2, GRID = "none", "black", "black", "gray"
CU_NUM = 104          # MI210
TOPN = 7


def short(name):
    """Kernel names carry their whole C++ signature; keep the identifier."""
    n = name.split("(")[0].strip()
    n = re.sub(r"^_ZN[0-9_]*", "", n)
    n = n.split("::")[-1]
    return n[:38]


# Which counters are rates (duration-weighted) and which are volumes (summed).
RATE = {"VALUBusy", "SALUBusy", "MemUnitBusy", "MemUnitStalled", "L2CacheHit",
        "MeanOccupancyPerCU"}
SIZE = {"FETCH_SIZE", "WRITE_SIZE", "SQ_INSTS_VALU", "SQ_WAVES"}
CLOCK = 1.7e9         # MI210 peak engine clock, for the cycles/instruction column
# MI210: 104 CUs x 4 SIMDs, one VALU instruction per SIMD per clock at ~1.7 GHz.
VALU_PEAK = 104 * 4 * 1.7e9


def usage(arm_dir, part):
    """Per-kernel counters for one partition, merged across the three passes.

    The passes are separate runs, so this joins them BY KERNEL NAME. That is
    legitimate here in a way joining on time would not be: a kernel's arithmetic
    intensity and launch shape do not change between runs of the same
    deterministic workload, only its wall placement does.

    Rates are duration-weighted, sizes summed. Returns {kernel: {counter: value}}
    plus "_dur" (ns of dispatch time seen for that kernel, per pass).
    """
    out = collections.defaultdict(dict)
    for name in ("bandwidth", "cache", "compute"):
        d = os.path.join(arm_dir, name, "pmc")
        rg = ranges(d)
        key = "part%d" % part
        if key not in rg:
            continue
        p0, p1 = rg[key][0]
        # part<i> covers PROVING only. witgen sits inside part<i>_build, which ends
        # where part<i> begins -- so a window of just part<i> silently drops witness
        # generation, the phase the whole GPU port was about. Widen to the build.
        bkey = key + "_build"
        if bkey in rg:
            p0 = min(p0, min(a for a, _b in rg[bkey]))
        acc = collections.defaultdict(lambda: collections.defaultdict(float))
        seen = set()
        with open(os.path.join(d, "run_counter_collection.csv"),
                  newline="", errors="replace") as f:
            for r in csv.DictReader(f):
                try:
                    t0 = int(r["Start_Timestamp"])
                    if not (p0 <= t0 <= p1):
                        continue
                    k, cn = r["Kernel_Name"], r["Counter_Name"]
                    dur = int(r["End_Timestamp"]) - t0
                    cv = float(r["Counter_Value"])
                except (KeyError, ValueError):
                    continue
                if (t0, k) not in seen:       # one dispatch, several counter rows
                    seen.add((t0, k))
                    acc[k]["_dur"] += dur
                if cn in SIZE:
                    acc[k][cn] += cv
                elif cn in RATE:
                    acc[k][cn] += cv * dur
                    acc[k]["_w_" + cn] += dur
        for k, m in acc.items():
            out[k]["_dur"] = out[k].get("_dur", 0.0) + m.get("_dur", 0.0)
            for cn, v in m.items():
                if cn.startswith("_"):
                    continue
                if cn in SIZE:
                    out[k][cn] = out[k].get(cn, 0.0) + v
                else:
                    w = m.get("_w_" + cn, 0.0)
                    out[k][cn] = v / w if w else 0.0
            # GB/s needs this pass's own dispatch time, not the merged total.
            if "FETCH_SIZE" in m or "WRITE_SIZE" in m:
                secs = m.get("_dur", 0.0) / 1e9
                moved = (m.get("FETCH_SIZE", 0.0) + m.get("WRITE_SIZE", 0.0)) * 1024.0
                out[k]["GBps"] = moved / 1e9 / secs if secs > 0 else 0.0
                out[k]["GiB"] = moved / 2**30
            # VALU instructions per second. This is the denominator-free companion
            # to VALUBusy: that metric divides by CU_NUM, so a kernel saturating one
            # CU's ALUs reads ~1% device-wide and looks idle. The rate says how much
            # arithmetic is actually issued, whatever fraction of the device issues it.
            if "SQ_INSTS_VALU" in m:
                secs = m.get("_dur", 0.0) / 1e9
                out[k]["GVALUps"] = (m["SQ_INSTS_VALU"] / 1e9 / secs
                                     if secs > 0 else 0.0)
                # Cycles one wave spends per VALU instruction it issues, over
                # THAT WAVE'S lifetime -- not the kernel's. The distinction is the
                # whole metric: a kernel launching 230k waves runs them in batches
                # of however many are resident, so dividing the kernel's duration
                # by one wave's instruction count charges a wave for every other
                # wave's time. Scaling by resident/total recovers one wave's share.
                #
                # Floor is 4 on CDNA (a 64-lane wave issues on a 16-wide SIMD over
                # 4 cycles). Read it beside waves/CU: a high value only costs
                # throughput when occupancy is too low to hide it.
                waves = m.get("SQ_WAVES", 0.0)
                occ = m.get("MeanOccupancyPerCU", 0.0)
                wocc = m.get("_w_MeanOccupancyPerCU", 0.0)
                occ = occ / wocc if wocc else 0.0
                resident = occ * CU_NUM
                ipt = m["SQ_INSTS_VALU"] / (waves * 64.0) if waves else 0.0
                share = min(1.0, resident / waves) if waves else 1.0
                out[k]["CPI"] = (secs * CLOCK * share / ipt) if ipt > 0 else 0.0
    return out


def collect(pmc_dir, part):
    """{phase: {kernel: ns}}, plus {kernel: (grid, wg)} for the same partition."""
    rg = ranges(pmc_dir)
    key = "part%d" % part
    if key not in rg:
        return None, None, None
    p0, p1 = rg[key][0]
    # part<i> covers PROVING only. witgen sits inside part<i>_build, which ends
    # where part<i> begins, so a window of just part<i> drops witness generation
    # entirely -- the phase the GPU port was about. Widen to the build.
    bkey = key + "_build"
    if bkey in rg:
        p0 = min(p0, min(a for a, _b in rg[bkey]))
    phases = sorted((a, b, nm) for nm, ivs in rg.items() for a, b in ivs
                    if p0 <= a and b <= p1 and not nm.startswith("part"))

    def phase_of(t):
        best = None
        for a, b, nm in phases:
            if a <= t <= b and (best is None or (b - a) < best[0]):
                best = (b - a, nm)          # innermost range wins
        return best[1] if best else None

    agg = collections.defaultdict(lambda: collections.defaultdict(float))
    geom, seen = {}, set()
    with open(os.path.join(pmc_dir, "run_counter_collection.csv"),
              newline="", errors="replace") as f:
        for r in csv.DictReader(f):
            try:
                t0 = int(r["Start_Timestamp"])
                k = r["Kernel_Name"]
            except (KeyError, ValueError):
                continue
            if (t0, k) in seen:
                continue                    # one row per counter; count once
            seen.add((t0, k))
            ph = phase_of(t0)
            if ph is None:
                continue
            agg[ph][k] += int(r["End_Timestamp"]) - t0
            try:
                g, w = int(r["Grid_Size"]), int(r["Workgroup_Size"])
                geom[k] = (g, w)
            except (KeyError, ValueError):
                pass
    return agg, geom, (p1 - p0) / 1e9


def draw(arm_dir, part, out_path):
    agg, geom, wall = collect(os.path.join(arm_dir, "compute", "pmc"), part)
    if agg is None:
        sys.exit("no ROCTx range part%d under %s" % (part, arm_dir))

    total = collections.Counter()
    for ph, ks in agg.items():
        for k, v in ks.items():
            total[k] += v
    top = [k for k, _ in total.most_common(TOPN)]
    color = {k: SLOTS[i] for i, k in enumerate(top)}
    rows = [ph for ph in sorted(agg, key=lambda p: -sum(agg[p].values()))
            if sum(agg[ph].values()) / 1e9 >= 0.002]
    span = max(sum(agg[ph].values()) for ph in rows) / 1e9

    ps.apply()
    nk = min(TOPN, len(total)) + (1 if len(total) > TOPN else 0)
    fig = plt.figure(figsize=(14.4, 0.62 * len(rows) + 0.30 * nk + 2.6), dpi=150)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.62 * len(rows) + 1.0,
                                               0.30 * nk + 1.0],
                          hspace=0.34, left=0.118, right=0.985, top=0.90,
                          bottom=0.06)
    ax = fig.add_subplot(gs[0])
    axt = fig.add_subplot(gs[1])
    gap = span * 0.0018                     # 2px surface gap between segments

    for i, ph in enumerate(rows):
        left = 0.0
        for k, v in sorted(agg[ph].items(), key=lambda kv: -kv[1]):
            s = v / 1e9
            ax.barh(i, max(s - gap, 0), 0.62, left=left,
                    color=color.get(k, OTHER), linewidth=0)
            if s > span * 0.085:
                ax.text(left + s / 2, i, short(k), ha="center", va="center",
                        color="white", fontsize=7.8, fontweight="bold")
            left += s
        ax.text(left + span * 0.008, i, "%.2f s" % left, va="center",
                fontsize=8.6, color=INK, fontweight="bold")

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=9.2)
    ax.invert_yaxis()
    ax.set_xlim(0, span * 1.14)
    ax.set_xlabel("kernel time (s)  —  summed dispatch durations, not wall clock",
                  fontsize=9.2, color=INK2)
    ax.grid(axis="x", color="gray", linestyle="dotted", linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(True)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.6)

    # The legend is replaced by a usage table on its own axes: for these kernels
    # the question is never "which colour" but "what is it doing to the device",
    # and a colour key that omitted the counters would send the reader to a second
    # figure. It doubles as the table view the contrast WARN on three slots obliges.
    use = usage(arm_dir, part)
    axt.set_xlim(0, 1)
    axt.set_ylim(-(nk + 0.9), 1.1)
    axt.axis("off")
    colx = [0.026, 0.320, 0.383, 0.452, 0.537, 0.617, 0.694, 0.770, 0.845, 0.918]
    # Two-line headers: the short forms ("MemU%", "L2%") were ambiguous enough to
    # need asking about, and every one of these is a different KIND of quantity --
    # a count, a duration, two shares of time, a residency, a rate and a hit ratio.
    hdr = (("", "kernel"), ("kernel", "sec"), ("% of", "time"),
           ("VALU", "busy %"), ("G VALU", "instr/s"), ("cyc/instr", "per wave"),
           ("MemUnit", "busy %"), ("waves/CU", "of 32"), ("HBM", "GB/s"),
           ("L2 hit", "%"))
    for cx, (h1, h2) in zip(colx, hdr):
        ha = "left" if h2 == "kernel" else "right"
        axt.text(cx, 0.62, h1, fontsize=8.0, color=INK2, fontweight="bold", ha=ha)
        axt.text(cx, 0.14, h2, fontsize=8.0, color=INK2, fontweight="bold", ha=ha)
    axt.plot([0.008, 0.905], [-0.28, -0.28], color=GRID, linewidth=1.0)
    tot_s = sum(total.values()) / 1e9
    for n, k in enumerate(top):
        yy = -(n + 0.85)
        g, w = geom.get(k, (0, 0))
        wg = g // w if w else 0
        u = use.get(k, {})
        axt.add_patch(plt.Rectangle((0.008, yy - 0.22), 0.017, 0.44,
                                    facecolor=color[k], linewidth=0))
        axt.text(colx[0], yy, short(k), fontsize=8.2, color=INK, ha="left",
                 va="center")
        for cx, v in zip(colx[1:], [
                "%.2f" % (total[k] / 1e9),
                "%.0f" % (100 * total[k] / 1e9 / tot_s) if tot_s else "-",
                "%.0f" % u.get("VALUBusy", 0.0),
                "%.1f" % u.get("GVALUps", 0.0),
                "%.0f" % u.get("CPI", 0.0),
                "%.0f" % u.get("MemUnitBusy", 0.0),
                "%.1f" % u.get("MeanOccupancyPerCU", 0.0),
                "%.0f" % u.get("GBps", 0.0),
                "%.0f" % u.get("L2CacheHit", 0.0)]):
            axt.text(cx, yy, v, fontsize=8.2, color=INK, ha="right", va="center")
    if len(total) > TOPN:
        yy = -(len(top) + 0.85)
        axt.add_patch(plt.Rectangle((0.008, yy - 0.22), 0.017, 0.44,
                                    facecolor=OTHER, linewidth=0))
        axt.text(colx[0], yy, "%d other kernels" % (len(total) - TOPN),
                 fontsize=8.2, color=INK2, ha="left", va="center")

    fig.suptitle("Partition %d kernel time breakdown" % part,
                 x=0.012, y=0.978, ha="left", fontsize=13.5, color=INK,
                 weight="semibold")
    fig.text(0.012, 0.941,
             "GPT-2 len 64, 12 partitions sequential, one MI210   ·   partition "
             "wall %.1f s, kernel time %.2f s   ·   %d distinct kernels"
             % (wall, sum(total.values()) / 1e9, len(total)),
             ha="left", fontsize=8.6, color=INK2)
    fig.text(0.012, 0.935, "", ha="left")
    fig.savefig(out_path, transparent=True, dpi=300)
    plt.close(fig)
    return out_path, sum(total.values()) / 1e9, len(total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm_dir")
    ap.add_argument("--partition", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("-o", "--out")
    a = ap.parse_args()
    rg = ranges(os.path.join(a.arm_dir, "compute", "pmc"))
    parts = (sorted(int(k[4:]) for k in rg
                    if k.startswith("part") and k[4:].isdigit())
             if a.all else [a.partition])
    outdir = a.out or os.path.join(a.arm_dir, "figs")
    os.makedirs(outdir, exist_ok=True)
    for p in parts:
        path, ks, n = draw(a.arm_dir, p,
                           os.path.join(outdir, "kernels_part%02d.png" % p))
        print("wrote %s  (%.2f s kernel time, %d kernels)" % (path, ks, n))


if __name__ == "__main__":
    main()
