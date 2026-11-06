#!/usr/bin/env python3
"""plot_perf_phases.py — the CPU arm's answer to plot_pmc_kernels.py.

Same picture, other side of the boundary: one bar per prover phase, segmented by
the FUNCTION the CPU spent that phase in, with a usage table underneath. The GPU
version resolves phases into kernels via ROCTx containment; this resolves them
into symbols via perf sample timestamps, joined to the binary's own PHASE markers.

THE TWO CLOCKS DIFFER AND ARE RECONCILED HERE. PHASE lines are stamped with
gettimeofday (CLOCK_REALTIME); perf samples carry CLOCK_MONOTONIC. The offset is
measured once and applied, which is sound because the offset is stable absent an
NTP step. The check that it worked: the first PHASE marker must land a few seconds
AFTER perf's first sample, since perf record starts before the program does.

Falls back to phase durations plus a global function table when per-sample data is
unavailable -- an 82 GB perf.data with dwarf stacks does not always survive
`perf script` in reasonable time, and a coarser figure beats none.

Usage:
  tools/plot_perf_phases.py <cpu_mono dir> [-o out.png]
"""
import argparse
import bisect
import collections
import os
import re
import sys
import time

import matplotlib
import matplotlib.colors
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plotstyle as ps  # noqa: E402  house style (paper scripts)
from plotstyle import COLOR  # noqa: E402  phase colours, one source of truth

SLOTS = [matplotlib.colors.to_hex(c) for c in ps.blues(7)]
OTHER = "#9AA3A8"
BG, INK, INK2, GRID = "none", "black", "black", "gray"
TOPN = 7
PHASE_RE = re.compile(
    r"^PHASE\s+part=(-?\d+)\s+pid=(\d+)\s+ev=(\w+)\s+phase=(\S+)\s+d=(\d+)\s+t_us=(\d+)")


def phases(log_path):
    """[(name, t0_s, t1_s)] in CLOCK_REALTIME seconds, from the run's PHASE lines."""
    open_at, out = {}, []
    for line in open(log_path, errors="replace"):
        m = PHASE_RE.match(line)
        if not m:
            continue
        _part, _pid, ev, name, _d, t_us = m.groups()
        t = int(t_us) / 1e6
        if ev == "begin":
            open_at.setdefault(name, []).append(t)
        elif ev == "end" and open_at.get(name):
            out.append((name, open_at[name].pop(), t))
    return sorted(out, key=lambda x: x[1])


def clock_offset():
    """REALTIME - MONOTONIC, to map PHASE stamps onto perf sample stamps."""
    return time.time() - time.clock_gettime(time.CLOCK_MONOTONIC)


def seq_len(log_path):
    """GPT-2 sequence length, from the PARTITION line's own len= (circuit width;
    logs before 2026-09-14 spell it chunk_width=), so this is the authoritative
    source. Was hardcoded to 64 (the first run this script ever plotted); once
    cc_len128/cc_len256 reused it, the figure kept reporting "len 64" for every
    length. Falls back to "?" rather than guessing from the directory name,
    which is a convention (cc_len128), not a measurement."""
    for line in open(log_path, errors="replace"):
        if line.startswith("PARTITION "):
            m = re.search(r"(?:chunk_width|len)=(\d+)", line)
            return m.group(1) if m else "?"
    return "?"


def short(sym):
    s = sym.split("(")[0].strip()
    for pre in ("_ZN", "_Z"):
        if s.startswith(pre):
            s = re.sub(r"^_ZN?[0-9]*", "", s)
    s = s.replace("::", ".")
    return s[:44] if s else "[unknown]"


def per_phase_functions(run_dir, ph, off):
    """{phase: {symbol: weight}} from perf report's own time bucketing.

    `perf script -F sym` returns EMPTY symbols on this perf.data even though
    perf report resolves them fine, so attribution goes through
    `perf report --sort time,sym --time-quantum 1s` instead: each row is
    (percent-of-run, monotonic 1 s bucket, symbol). Buckets are mapped onto phases
    with the REALTIME-MONOTONIC offset, so a bucket lands in whichever phase
    contains its midpoint.

    1 s buckets are coarse against phases of 10-270 s, which is fine here, but it
    does mean a bucket straddling a phase boundary is charged wholly to one side.
    """
    path = os.path.join(run_dir, "report_by_time.txt")
    if not os.path.exists(path):
        return None
    # -n gives raw sample COUNTS. Percentages are rounded to 2 dp, so with
    # --percent-limit 0 a single-threaded phase's buckets print 0.00% and sum to
    # nothing (one thread at 199 Hz puts ~199 samples in a 1 s bucket, under the
    # 0.005% cutoff that the 192-thread spin phases clear easily).
    rx = re.compile(r"^\s*[0-9.]+%\s+([0-9]+)\s+([0-9]+\.[0-9]+)\s+\[.\]\s+(.*?)\s*$")
    iv = sorted((t0 - off, t1 - off, nm) for nm, t0, t1 in ph)
    agg = collections.defaultdict(lambda: collections.defaultdict(float))
    tot = 0.0
    for line in open(path, errors="replace"):
        m = rx.match(line)
        if not m:
            continue
        pct, tb, sym = float(m.group(1)), float(m.group(2)), m.group(3)
        mid = tb + 0.5
        best = None
        for a, b, nm in iv:
            if a <= mid <= b and (best is None or (b - a) < best[0]):
                best = (b - a, nm)
        if best:
            agg[best[1]][sym] += pct
            tot += pct
    return (agg, tot) if tot else None


def draw(run_dir, out_path):
    log = None
    for cand in ("record_run.log", "stat_run.log"):
        p = os.path.join(run_dir, cand)
        if os.path.exists(p):
            log = p
            break
    if not log:
        sys.exit("no record_run.log / stat_run.log under " + run_dir)
    ph = phases(log)
    if not ph:
        sys.exit("no PHASE markers in " + log)

    res = per_phase_functions(run_dir, ph, clock_offset())
    attributed = res is not None
    if attributed:
        agg, nsamp = res
    else:
        agg, nsamp = {}, 0

    # Phase wall seconds are exact from the markers either way.
    wall = collections.defaultdict(float)
    for nm, t0, t1 in ph:
        wall[nm] += t1 - t0
    # LEAF phases only. prove contains gkr_prove contains verify_gkr/verify_lasso/
    # open_commit, and execute_commit/range_prove/sequence_kv_commit sit inside prove
    # too, so charting every phase double-counts. A phase is a parent if some other
    # phase's interval nests strictly inside one of its own -- same rule the GPU figure gets for free from ROCTx
    # innermost-wins attribution.
    parent = set()
    for nm, a0, a1 in ph:
        for nm2, b0, b1 in ph:
            if nm2 != nm and a0 <= b0 and b1 <= a1 and (b1 - b0) < (a1 - a0):
                parent.add(nm)
                break
    inner = {nm: w for nm, w in wall.items() if nm not in parent}
    rows = [nm for nm in sorted(inner, key=lambda n: -inner[n]) if inner[nm] >= 0.05]

    if attributed:
        total = collections.Counter()
        for nm, syms in agg.items():
            for s, c in syms.items():
                total[s] += c
        top = [s for s, _ in total.most_common(TOPN)]
        color = {s: SLOTS[i] for i, s in enumerate(top)}
        span = max(inner[nm] for nm in rows)
    else:
        top, color, total = [], {}, collections.Counter()
        span = max(inner[nm] for nm in rows)

    nk = (min(TOPN, len(total)) + (1 if len(total) > TOPN else 0)) if attributed else 0
    ps.apply()
    fig = plt.figure(figsize=(14.4, 0.62 * len(rows) + 0.30 * nk + 3.0), dpi=150)
    gs = fig.add_gridspec(2, 1,
                          height_ratios=[0.62 * len(rows) + 1.0, 0.30 * nk + 1.0],
                          hspace=0.34, left=0.135, right=0.985, top=0.90, bottom=0.06)
    ax = fig.add_subplot(gs[0])
    axt = fig.add_subplot(gs[1])
    gap = span * 0.0018

    for i, nm in enumerate(rows):
        if attributed and agg.get(nm):
            tot_c = sum(agg[nm].values()) or 1.0
            left = 0.0
            for s, c in sorted(agg[nm].items(), key=lambda kv: -kv[1]):
                w = inner[nm] * c / tot_c          # samples -> seconds of this phase
                # Subtract the surface gap only from segments wide enough to show
                # one. witgen spreads over 17k symbols; charging every sliver a gap
                # would visibly shorten its bar.
                ax.barh(i, w - gap if w > 3 * gap else w, 0.62, left=left,
                        color=color.get(s, OTHER), linewidth=0)
                if w > span * 0.085:
                    ax.text(left + w / 2, i, short(s), ha="center", va="center",
                            color="white", fontsize=7.6, fontweight="bold")
                left += w
        else:
            ax.barh(i, inner[nm] - gap, 0.62, left=0,
                    color=COLOR.get(nm, OTHER), linewidth=0)
        ax.text(inner[nm] + span * 0.008, i, "%.1f s" % inner[nm], va="center",
                fontsize=8.6, color=INK, fontweight="bold")

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=9.2)
    ax.invert_yaxis()
    ax.set_xlim(0, span * 1.14)
    ax.set_xlabel("phase wall seconds" + ("  —  segmented by function, from perf samples"
                                          if attributed else ""),
                  fontsize=9.2, color=INK2)
    ax.grid(axis="x", color="gray", linestyle="dotted", linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(True)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.6)

    axt.set_xlim(0, 1)
    axt.axis("off")
    if attributed:
        axt.set_ylim(-(nk + 1.0), 1.1)
        colx = [0.026, 0.560, 0.660, 0.760, 0.870]
        for cx, (h1, h2) in zip(colx, (("", "function"), ("CPU", "sec"),
                                       ("% of", "samples"), ("samples", "(k)"),
                                       ("phases", "seen in"))):
            axt.text(cx, 0.62, h1, fontsize=8.0, color=INK2, fontweight="bold",
                     ha="left" if h2 == "function" else "right")
            axt.text(cx, 0.14, h2, fontsize=8.0, color=INK2, fontweight="bold",
                     ha="left" if h2 == "function" else "right")
        axt.plot([0.008, 0.92], [-0.28, -0.28], color=GRID, linewidth=1.0)
        tot_wall = sum(inner[nm] for nm in rows)
        for n, s in enumerate(top):
            yy = -(n + 0.85)
            axt.add_patch(plt.Rectangle((0.008, yy - 0.22), 0.017, 0.44,
                                        facecolor=color[s], linewidth=0))
            axt.text(colx[0], yy, short(s), fontsize=8.2, color=INK, va="center")
            seen = sum(1 for nm in rows if agg.get(nm, {}).get(s))
            for cx, v in zip(colx[1:], [
                    "%.1f" % (tot_wall * total[s] / nsamp),
                    "%.1f" % (100.0 * total[s] / nsamp),
                    "%.0f" % (total[s] / 1000.0),
                    "%d" % seen]):
                axt.text(cx, yy, v, fontsize=8.2, color=INK, ha="right", va="center")
        if len(total) > TOPN:
            yy = -(len(top) + 0.85)
            axt.add_patch(plt.Rectangle((0.008, yy - 0.22), 0.017, 0.44,
                                        facecolor=OTHER, linewidth=0))
            axt.text(colx[0], yy, "%d other symbols" % (len(total) - TOPN),
                     fontsize=8.2, color=INK2, va="center")
    else:
        axt.set_ylim(-1, 1)
        axt.text(0.008, 0.2, "Per-function attribution unavailable: perf.data could "
                 "not be converted to per-sample records.\nBars are phase wall time "
                 "from the run's own PHASE markers, which are exact.",
                 fontsize=8.4, color=INK2, va="center", linespacing=1.6)

    fig.suptitle("zkGPT on CPU: where each phase's time goes", x=0.012, y=0.978,
                 ha="left", fontsize=13.5, color=INK, weight="semibold")
    fig.text(0.012, 0.941,
             "GPT-2 len %s, monolithic (splits=1), 192 threads, all offloads on "
             "their CPU reference arm   ·   %.0f s of phase time%s"
             % (seq_len(log), sum(inner[nm] for nm in rows),
                ("   ·   %.1fM samples in these phases" % (nsamp / 1e6))
                if attributed else ""),
             ha="left", fontsize=8.6, color=INK2)
    fig.savefig(out_path, transparent=True, dpi=300)
    plt.close(fig)
    return out_path, attributed, nsamp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("-o", "--out")
    a = ap.parse_args()
    out = a.out or os.path.join(a.run_dir, "figs", "phases.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    p, attributed, n = draw(a.run_dir, out)
    print("wrote %s  (%s%s)" % (p, "per-function" if attributed else "phase-only",
                                ", %.1fM samples" % (n / 1e6) if attributed else ""))


if __name__ == "__main__":
    main()
