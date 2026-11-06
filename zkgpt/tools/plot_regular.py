#!/usr/bin/env python3
"""plot_regular.py — device / zkMod figures. One tool, four views.

Subcommands (former plot_range / plot_llama2b / plot_hetero / plot_ck_sweep):

  cost     the Exp-1/Exp-2 cost-model "hero" figures from a compare_range.py CSV.
           Dispatches on the CSV's `bench` column:
             matmul -> matmul_k_story.png      (effect of k: time, peak, binding
                                                resource, alpha collapse)
                       matmul_tile_story.png   (effect of tile width t: time,
                                                peak, memory-model validation,
                                                (memory,time) frontier) — only
                                                when the CSV has >1 tile_bits
                       phases/n*.png           (per-n phase breakdown)
             full   -> lasso_c_story.png  + phases/c*.png    (effect of c, L)
             deg1   -> deg1_time_vs_n.png, deg1_peak_mem.png
  llama2b  the llama2b (depth=32, 2048/8192) partition/staging sweep: a
           (splits x lasso_k) feasibility grid + a peak-VRAM/time tradeoff scatter.
  hetero   the heterogeneous-partition probe vs the uniform-k grid optimum:
           per-partition peak-VRAM profile (p0 is the lone binding partition) +
           total prover time per arm.
  ck       per-depth knob sweep (matmul_k / range_c / range_k) vs time & GPU
           memory, one figure per input CSV (one file per depth L).

Usage:
  python3 tools/plot_regular.py cost    [compare_range.csv] [--outdir DIR]
  python3 tools/plot_regular.py llama2b  [--src CSV] [--out DIR]
  python3 tools/plot_regular.py hetero   [--dir results/llama2b/hetero]
  python3 tools/plot_regular.py ck       <ck_sweep_L*_*.csv> ... [--out fig.png]
"""
import argparse
import collections
import csv
import glob
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Stable phase colors so the same phase is the same color across every figure.
PHASE_COLOR = {
    "expand": "#4C72B0", "ptable": "#DD8452", "round": "#55A868", "fold": "#C44E52",
    "setup": "#8172B3", "invert": "#937860", "commit": "#DA8BC3",
    "deg3": "#8C8C8C", "deg1": "#CCB974", "open": "#64B5CD",
}


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _mirror_out(path):
    """Mirror a results/<x>/data/foo.csv input into results/<x>/images/foo.png."""
    d, fname = os.path.split(path)
    png = os.path.splitext(fname)[0] + ".png"
    if os.path.basename(d) == "data":
        return os.path.join(os.path.dirname(d), "images", png)
    return os.path.join(d, png)


# ══════════════════════════════════════════════════════════════════════════════
# cost — the Exp-1/Exp-2 cost-model figures (former plot_range.py)
# ══════════════════════════════════════════════════════════════════════════════

def mode_of(r):
    return r.get("mode", "gpu")


def _cmap_ramp(cmap_name, n_items, lo=0.45, hi=0.95):
    """Sample a named colormap so an ordered key reads light->dark within one hue."""
    cmap = plt.get_cmap(cmap_name)
    if n_items <= 1:
        return [cmap(hi)]
    return [cmap(lo + (hi - lo) * i / (n_items - 1)) for i in range(n_items)]


GPU_CMAP, CPU_CMAP = "Blues", "Oranges"


def _mode_colors(keys):
    """Return {('gpu',key):color, ('cpu',key):color} blue/orange ramps over keys."""
    keys = list(keys)
    gpu = _cmap_ramp(GPU_CMAP, len(keys))
    cpu = _cmap_ramp(CPU_CMAP, len(keys))
    out = {}
    for i, k in enumerate(keys):
        out[("gpu", k)] = gpu[i]
        out[("cpu", k)] = cpu[i]
    return out


def _tile_bits(r):
    """Extension-table tile width t for a row; -1 (untiled) for pre-tiling CSVs."""
    v = fnum(r.get("tile_bits"))
    return -1 if v is None else int(v)


def _gpu_s(r):
    """Device-region time: the four sumcheck phases.

    NOT time_s — that is wall-clock around the whole microbench and is dominated
    by the O(2^n) host CSPRNG vector prep (~70 s at n=29), which would swamp any
    cost model fitted against it.
    """
    v = fnum(r.get("gpu_s"))
    if v is not None:
        return v
    return sum(fnum(r.get(p + "_s")) or 0.0 for p in ("expand", "ptable", "round", "fold"))


def _untiled_ref(rows_nk):
    """The reference row for a (n,k) group: the untiled point if present,
    else the one materializing the widest future window."""
    for r in rows_nk:
        if _tile_bits(r) < 0:
            return r
    return max(rows_nk, key=lambda r: fnum(r.get("fv_cap")) or 0)


def plot_matmul(rows, outdir):
    rows = [r for r in rows if mode_of(r) == "gpu"]   # GPU data only
    plot_matmul_k(rows, outdir)
    if len({_tile_bits(r) for r in rows}) > 1:
        plot_matmul_tile(rows, outdir)
    plot_matmul_phases(rows, outdir)


def plot_matmul_k(rows, outdir):
    """Figure 1 — the k axis: is k a time knob, a memory knob, or both?"""
    ns = sorted({int(r["n"]) for r in rows})
    mcolor = _mode_colors(ns)
    gcolor = {n: mcolor[("gpu", n)] for n in ns}
    have_compute = any(fnum(r.get("valu_busy")) is not None for r in rows)

    # One reference row per (n,k) so the k panels are single-valued when the CSV
    # carries a tile axis.
    by_nk = defaultdict(list)
    for r in rows:
        by_nk[(int(r["n"]), int(r["k"]))].append(r)
    ref = {nk: _untiled_ref(rs) for nk, rs in by_nk.items()}

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 11))

    # (a) time vs k — the claimed cost ~ alpha*k*s
    for n in ns:
        pts = sorted((k, _gpu_s(r)) for (nn, k), r in ref.items() if nn == n)
        if pts:
            ax1.plot([p[0] for p in pts], [p[1] for p in pts], "-o",
                     color=gcolor[n], lw=2, ms=5, label=f"n={n}")
    ax1.set_yscale("log")
    ax1.set_xlabel("k   (Blendy stage count)")
    ax1.set_ylabel("GPU sumcheck time (s, log)")
    ax1.set_title("(a) Time vs k  —  the runtime knob")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(title="size", ncol=2, fontsize=8)

    # (b) THE headline: peak memory vs k, one line per tile width.
    # Pre-tiling this slopes down steeply (k was the memory lever); post-tiling
    # the tiled lines are flat and only the untiled line still slopes.
    tbs = sorted({_tile_bits(r) for r in rows})
    n_big = ns[-1] if ns else None
    tcolors = _cmap_ramp("viridis", len(tbs))
    any_mem = False
    for i, tb in enumerate(tbs):
        pts = sorted((int(r["k"]), fnum(r["peak_gpu_bytes"]) or 0)
                     for r in rows
                     if int(r["n"]) == n_big and _tile_bits(r) == tb
                     and (fnum(r["peak_gpu_bytes"]) or 0) > 0)
        if not pts:
            continue
        any_mem = True
        lbl = "untiled" if tb < 0 else f"t={tb}"
        ax2.plot([p[0] for p in pts], [p[1] / 2**30 for p in pts], "-o",
                 color=tcolors[i], lw=2, ms=5, label=lbl)
    if n_big is not None:
        floor = 3 * (1 << n_big) * 32 / 2**30
        ax2.axhline(floor, color="crimson", ls=":", lw=2)
        ax2.text(0.02, floor, f" 3N floor = {floor:.1f} GiB (independent of k and t)",
                 color="crimson", fontsize=8, va="bottom", transform=ax2.get_yaxis_transform())
    ax2.set_yscale("log")
    ax2.set_xlabel("k   (Blendy stage count)")
    ax2.set_ylabel("peak GPU memory (GiB, log)")
    ax2.set_title(f"(b) Peak memory vs k, per tile width   (n={n_big})")
    ax2.grid(True, which="both", alpha=0.3)
    if any_mem:
        ax2.legend(title="tile width", ncol=2, fontsize=8)

    # (c) where the kernel is bound, as k grows
    if have_compute:
        for n in ns:
            cpts = sorted((k, fnum(r.get("valu_busy"))) for (nn, k), r in ref.items()
                          if nn == n and fnum(r.get("valu_busy")) is not None)
            mpts = sorted((k, fnum(r.get("mem_busy"))) for (nn, k), r in ref.items()
                          if nn == n and fnum(r.get("mem_busy")) is not None)
            if cpts:
                ax3.plot([p[0] for p in cpts], [p[1] for p in cpts], "-o",
                         color=gcolor[n], lw=2, ms=5)
            if mpts:
                ax3.plot([p[0] for p in mpts], [p[1] for p in mpts], "--s",
                         color=gcolor[n], lw=1.5, ms=4, alpha=0.7)
        ax3.set_ylim(0, 100)
        ax3.set_xlabel("k   (Blendy stage count)")
        ax3.set_ylabel("GPU utilization (% of busy time)")
        ax3.set_title("(c) Binding resource vs k")
        ax3.grid(True, alpha=0.3)
        leg3 = ax3.legend(handles=[Line2D([], [], color=gcolor[n], lw=3, label=f"n={n}")
                                   for n in ns], title="size", loc="upper right", ncol=2, fontsize=8)
        ax3.add_artist(leg3)
        ax3.legend(handles=[Line2D([], [], color="0.3", lw=2, marker="o", label="VALU busy (compute)"),
                            Line2D([], [], color="0.3", lw=1.5, ls="--", marker="s", label="Mem-unit busy")],
                   loc="lower left", fontsize=8)
    else:
        ax3.text(0.5, 0.5, "no compute-metric data\n(run without --no-compute)",
                 ha="center", va="center", transform=ax3.transAxes, color="0.4")
        ax3.set_axis_off()

    # (d) which predictor does cost actually follow? plan.tex says alpha*k*s.
    # The competing hypothesis is the extension-table work the expand phase
    # streams: ext = 3^l * 2^min(n-l, t), i.e. 2^n*(3/2)^l untiled. These
    # disagree in SIGN — k*s rises with k, ext falls with k — so the sweep
    # discriminates them cleanly.
    def _mape_through_origin(pairs):
        sxy = sum(x * y for x, y in pairs)
        sxx = sum(x * x for x, _ in pairs)
        if not sxx:
            return float("nan"), 0.0
        a = sxy / sxx
        errs = [abs(y - a * x) / y for x, y in pairs if y > 0]
        return (100 * sum(errs) / len(errs) if errs else float("nan")), a

    # Normalise each n's curve to its smallest measured k so all sizes overlay,
    # then compare against what alpha*k*s predicts (cost rising linearly in k).
    # The disagreement is in SIGN, which no choice of alpha can repair.
    ks_pairs = [(int(r["k"]) * (1 << int(r["n"])), _gpu_s(r))
                for r in rows if _gpu_s(r) > 0]
    mape_ks, _ = _mape_through_origin(ks_pairs)

    kmin_all = []
    for n in ns:
        pts = sorted((k, _gpu_s(r)) for (nn, k), r in ref.items() if nn == n)
        pts = [(k, v) for k, v in pts if v > 0]
        if len(pts) < 2:
            continue
        k0, v0 = pts[0]
        kmin_all.append(k0)
        ax4.plot([k for k, _ in pts], [v / v0 for _, v in pts], "-o",
                 color=gcolor[n], lw=2, ms=5, label=f"n={n} (measured)")
    if kmin_all:
        k0 = min(kmin_all)
        kk = sorted({int(r["k"]) for r in rows if int(r["k"]) >= k0})
        ax4.plot(kk, [k / k0 for k in kk], "k--", lw=2,
                 label="plan.tex  $\\alpha\\,k\\,s$  (linear in $k$)")
    ax4.axhline(1.0, color="0.6", lw=0.8, ls=":")
    ax4.set_yscale("log")
    ax4.set_xlabel("k   (Blendy stage count)")
    ax4.set_ylabel("cost relative to smallest k  (log)")
    ax4.set_title("(d) plan.tex eq. matmul_cost has the wrong sign\n"
                  f"measured cost FALLS with k; $\\alpha ks$ rises  "
                  f"({mape_ks:.0f}% dev as a fit)")
    ax4.grid(True, which="both", alpha=0.3)
    ax4.legend(fontsize=8, loc="upper right")

    fig.suptitle("Matmul (Blendy product sumcheck) — effect of the stage count k",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(outdir, "matmul_k_story.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    print("wrote", p)


def plot_matmul_tile(rows, outdir):
    """Figure 2 — the t axis: t is the memory knob, and is it free in time?"""
    ns = sorted({int(r["n"]) for r in rows})
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 11))

    # Group by (n,k); within a group t varies. Only groups with >1 tile point
    # say anything about the t axis.
    by_nk = defaultdict(list)
    for r in rows:
        by_nk[(int(r["n"]), int(r["k"]))].append(r)
    groups = {nk: rs for nk, rs in by_nk.items()
              if len({_tile_bits(r) for r in rs}) > 1}
    gkeys = sorted(groups)
    gcolors = _cmap_ramp("plasma", max(1, len(gkeys)))

    def _tx(r):
        """x position for a tile width: untiled plotted at the widest window."""
        tb = _tile_bits(r)
        fv = fnum(r.get("fv_cap"))
        return fv if tb < 0 and fv is not None else tb

    # (a) time vs t — plan.tex asserts tiling is free; this measures it.
    for i, nk in enumerate(gkeys):
        pts = sorted((_tx(r), _gpu_s(r)) for r in groups[nk])
        ax1.plot([p[0] for p in pts], [p[1] for p in pts], "-o",
                 color=gcolors[i], lw=1.8, ms=4, label=f"n={nk[0]} k={nk[1]}")
    ax1.set_xlabel("t   (tile width; materialized future window = $2^t$)")
    ax1.set_ylabel("GPU sumcheck time (s)")
    ax1.set_yscale("log")
    ax1.set_title("(a) Time vs t  —  is tiling free?")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=7, ncol=2)

    # (b) peak memory vs t — the memory knob post-tiling
    for i, nk in enumerate(gkeys):
        pts = sorted((_tx(r), (fnum(r["peak_gpu_bytes"]) or 0) / 2**30)
                     for r in groups[nk] if (fnum(r["peak_gpu_bytes"]) or 0) > 0)
        if pts:
            ax2.plot([p[0] for p in pts], [p[1] for p in pts], "-o",
                     color=gcolors[i], lw=1.8, ms=4, label=f"n={nk[0]} k={nk[1]}")
    ax2.set_xlabel("t   (tile width)")
    ax2.set_ylabel("peak GPU memory (GiB)")
    ax2.set_yscale("log")
    ax2.set_title("(b) Peak memory vs t  —  the memory knob")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=7, ncol=2)

    # (c) model validation: measured peak vs predicted 2*ext + 3N + 2*ptable
    mx, my = [], []
    for r in rows:
        meas = fnum(r.get("peak_gpu_bytes")) or 0
        pred = fnum(r.get("pred_need_bytes")) or 0
        if meas > 0 and pred > 0:
            mx.append(pred / 2**30); my.append(meas / 2**30)
    if mx:
        ax3.scatter(mx, my, s=26, color="#4C72B0", zorder=3, alpha=0.8)
        lim = [min(mx + my) * 0.8, max(mx + my) * 1.2]
        ax3.plot(lim, lim, "k--", lw=1.5, label="y = x (exact)")
        err = [100 * abs(m - p) / p for p, m in zip(mx, my)]
        mape = sum(err) / len(err)
        ax3.set_xlim(lim); ax3.set_ylim(lim)
        ax3.set_title(f"(c) Memory model validation   (MAPE {mape:.2f}%, n={len(mx)} pts)")
        ax3.legend(fontsize=8)
    else:
        ax3.text(0.5, 0.5, "no predicted-memory column", ha="center",
                 va="center", transform=ax3.transAxes, color="0.4")
    ax3.set_xscale("log"); ax3.set_yscale("log")
    ax3.set_xlabel("predicted peak  $2\\,ext + 3N + 2\\,ptable$  (GiB, log)")
    ax3.set_ylabel("measured peak (GiB, log)")
    ax3.grid(True, which="both", alpha=0.3)

    # (d) the actual question for the solver: is there a (memory, time) frontier?
    ncolors = _cmap_ramp("viridis", max(1, len(ns)))
    for i, n in enumerate(ns):
        pts = [((fnum(r["peak_gpu_bytes"]) or 0) / 2**30, _gpu_s(r), int(r["k"]), _tile_bits(r))
               for r in rows if int(r["n"]) == n and (fnum(r["peak_gpu_bytes"]) or 0) > 0]
        if not pts:
            continue
        tiled = [p for p in pts if p[3] >= 0]
        until = [p for p in pts if p[3] < 0]
        ax4.scatter([p[0] for p in tiled], [p[1] for p in tiled],
                    color=ncolors[i], s=30, marker="o", alpha=0.85, zorder=3)
        ax4.scatter([p[0] for p in until], [p[1] for p in until],
                    color=ncolors[i], s=52, marker="x", lw=1.6, zorder=4)
    ax4.set_xscale("log"); ax4.set_yscale("log")
    ax4.set_xlabel("peak GPU memory (GiB, log)  →  more space")
    ax4.set_ylabel("GPU sumcheck time (s, log)  →  slower")
    ax4.set_title("(d) Is there a (memory, time) Pareto frontier?")
    ax4.grid(True, which="both", alpha=0.3)
    ax4.legend(handles=([Line2D([], [], color=ncolors[i], lw=0, marker="o", label=f"n={n}")
                         for i, n in enumerate(ns)]
                        + [Line2D([], [], color="0.3", lw=0, marker="o", label="tiled"),
                           Line2D([], [], color="0.3", lw=0, marker="x", label="untiled")]),
               fontsize=7, ncol=2)

    fig.suptitle("Matmul — effect of the extension-table tile width t",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(outdir, "matmul_tile_story.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    print("wrote", p)


def plot_matmul_phases(rows, outdir):
    phases = ["expand", "ptable", "round", "fold"]
    phases_dir = os.path.join(outdir, "phases")
    os.makedirs(phases_dir, exist_ok=True)
    ns = sorted({int(r["n"]) for r in rows})
    for n in ns:
        n_rows = sorted([r for r in rows if int(r["n"]) == n],
                        key=lambda r: (int(r["k"]), _tile_bits(r)))
        if not n_rows:
            continue
        multi_t = len({_tile_bits(r) for r in n_rows}) > 1
        labels = [(f"k{r['k']}/t{_tile_bits(r)}" if multi_t else f"k={r['k']}")
                  for r in n_rows]
        x = list(range(len(labels)))
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.5), 5))
        bottom = [0.0] * len(labels)
        for ph in phases:
            vals = [fnum(r.get(ph + "_s")) or 0.0 for r in n_rows]
            ax.bar(x, vals, bottom=bottom, label=ph, color=PHASE_COLOR[ph])
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=90 if multi_t else 0)
        ax.set_ylabel("GPU time (s)")
        ax.set_title(f"Matmul phase breakdown  n={n}  (s=2^{n}={2**n:,})")
        ax.legend(); ax.grid(True, axis="y", alpha=0.3)
        pf = os.path.join(phases_dir, f"n{n}.png")
        fig.tight_layout(); fig.savefig(pf, dpi=120); plt.close(fig)
        print("wrote", pf)


def plot_full(rows, outdir):
    phases = ["setup", "invert", "commit", "deg3", "deg1", "open"]
    phases_dir = os.path.join(outdir, "phases")
    os.makedirs(phases_dir, exist_ok=True)

    rows = [r for r in rows if r.get("c") and mode_of(r) == "gpu"]   # GPU data only
    layers = sorted({int(r.get("layers", 1)) for r in rows})
    cs = sorted({int(r["c"]) for r in rows})
    ks = sorted({int(r["k"]) for r in rows})
    mcolor = _mode_colors(layers)        # ('gpu', L) -> color (blue)
    k_ref = ks[0]
    L_ref = layers[0]

    c_ref_M = 12 if 12 in cs else cs[len(cs) // 2]

    def fmt_M(x):
        x = int(x)
        if x > 0 and (x & (x - 1)) == 0:          # power of two -> 2^e
            return f"2$^{{{x.bit_length() - 1}}}$"
        for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
            if x >= div:
                return f"{x / div:.1f}{suf}"
        return str(x)

    LM = {}
    for L in layers:
        ms = [fnum(r.get("M")) for r in rows if int(r.get("layers", 1)) == L
              and int(r["c"]) == c_ref_M and fnum(r.get("M"))]
        if ms:
            LM[L] = int(ms[0])

    def llabel(L):
        return f"M={fmt_M(LM[L])}" if L in LM else f"L={L}"

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))

    for L in layers:
        pts = sorted([(int(r["c"]), float(r["time_s"])) for r in rows
                      if int(r.get("layers", 1)) == L and int(r["k"]) == k_ref])
        if not pts:
            continue
        ax1.plot([p[0] for p in pts], [p[1] for p in pts], "-o",
                 color=mcolor[("gpu", L)], lw=2, ms=5)
    cstars = []
    for L in layers:
        ref = sorted([(int(r["c"]), float(r["time_s"])) for r in rows
                      if int(r.get("layers", 1)) == L and int(r["k"]) == k_ref])
        if ref:
            cstar, tstar = min(ref, key=lambda p: p[1])
            cstars.append((L, cstar))
            ax1.plot([cstar], [tstar], marker="*", color="black", ms=14,
                     zorder=5, mec="white", mew=0.8)
    ax1.set_yscale("log")
    ax1.set_xlabel("c   (chunk bits)")
    ax1.set_ylabel(f"total range-prove time (s, log),  k={k_ref}")
    ax1.set_title("Time vs c   (* = optimum c*)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(handles=[Line2D([], [], color=mcolor[("gpu", L)], lw=2, ls="-", marker="o",
                               label=f"M={fmt_M(LM[L])}" if L in LM else f"L={L}")
                        for L in layers],
               title="#lookups", ncol=2, fontsize=8, loc="best")

    L_big = layers[-1]

    for L in layers:
        pts = []
        for c in cs:
            rr = [r for r in rows if int(r.get("layers", 1)) == L
                  and int(r["c"]) == c and int(r["k"]) == k_ref]
            if not rr:
                continue
            tot = float(rr[0]["time_s"])
            com = float(rr[0].get("commit_s", 0) or 0)
            if tot > 0:
                pts.append((c, 100 * com / tot))
        if not pts:
            continue
        pts.sort()
        ax2.plot([p[0] for p in pts], [p[1] for p in pts], "-o",
                 color=mcolor[("gpu", L)], lw=2, ms=4)
    ax2.set_xlabel("c   (chunk bits)")
    ax2.set_ylabel("commit share of total (%)")
    ax2.set_ylim(0, 100)
    ax2.set_title("Commit share vs c")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(50, color="0.5", ls=":", lw=1)
    leg2 = ax2.legend(handles=[Line2D([], [], color=mcolor[("gpu", L)], lw=3,
                                      label=llabel(L)) for L in layers],
                      title=f"#lookups (@ c={c_ref_M})", loc="best", fontsize=8)
    ax2.add_artist(leg2)

    cb = sorted(cs)
    width = 0.6
    x = list(range(len(cb)))
    bottom = [0.0] * len(cb)
    for ph in phases:
        vals = []
        for c in cb:
            rr = [r for r in rows if int(r.get("layers", 1)) == L_big
                  and int(r["c"]) == c and int(r["k"]) == k_ref]
            vals.append(float(rr[0].get(ph + "_s", 0) or 0) if rr else 0.0)
        ax3.bar(x, vals, width, bottom=bottom, color=PHASE_COLOR[ph],
                edgecolor="white", lw=0.3, label=ph)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax3.set_xticks(x); ax3.set_xticklabels([f"c={c}" for c in cb])
    ax3.set_ylabel("time (s)")
    ax3.set_title(f"Phase breakdown vs c   (M={fmt_M(LM[L_big])} @ c={c_ref_M})"
                  if L_big in LM else f"Phase breakdown vs c")
    ax3.grid(True, axis="y", alpha=0.3)
    ax3.legend(handles=[mpatches.Patch(color=PHASE_COLOR[ph], label=ph) for ph in phases],
               ncol=2, fontsize=8, loc="upper right")

    fig.suptitle("Range check (chunked logup) — effect of c and L",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(outdir, "lasso_c_story.png")
    fig.savefig(p); plt.close(fig)
    print("wrote", p)

    for L in layers:
        for c in cs:
            c_rows = sorted([r for r in rows if int(r["c"]) == c
                             and int(r.get("layers", 1)) == L],
                            key=lambda r: int(r["k"]))
            if not c_rows:
                continue
            labels = [f"k={r['k']}" for r in c_rows]
            x = list(range(len(labels)))
            fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.9), 5))
            bottom = [0.0] * len(labels)
            for ph in phases:
                vals = [float(r.get(ph + "_s", 0) or 0) for r in c_rows]
                ax.bar(x, vals, bottom=bottom, label=ph, color=PHASE_COLOR[ph])
                bottom = [b + v for b, v in zip(bottom, vals)]
            ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylabel("time (s)")
            ax.set_title(f"Range-prover phases  c={c}  L={L}  "
                         f"(N={c_rows[0].get('N', '?')})")
            ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
            pf = os.path.join(phases_dir, f"c{c}_L{L}.png")
            fig.tight_layout(); fig.savefig(pf); plt.close(fig)
            print("wrote", pf)


def plot_deg1(rows, outdir):
    series = defaultdict(list)
    model_ns = {}
    for r in rows:
        key = (mode_of(r), int(r["k"]))
        series[key].append((int(r["n"]), float(r["time_s"]),
                            fnum(r["peak_gpu_bytes"]) or 0))
        if r.get("model"):
            model_ns[r["model"]] = int(r["n"])
    for k in series:
        series[k].sort()

    plt.figure(figsize=(9, 6))
    for (mode, k), pts in sorted(series.items()):
        style = "-o" if mode == "gpu" else "--s"
        plt.plot([p[0] for p in pts], [p[1] for p in pts], style, label=f"{mode} k={k}")
    for name, n in sorted(model_ns.items(), key=lambda x: x[1]):
        plt.axvline(n, color="gray", ls=":", alpha=0.6)
        plt.text(n, plt.ylim()[1], name, rotation=90, va="top", ha="right",
                 fontsize=8, color="gray")
    plt.yscale("log")
    plt.xlabel("deg1 size n  (2^n field elements)")
    plt.ylabel("sumcheck time (s, log scale)")
    plt.title("deg1 reciprocal-sum sumcheck: CPU loop vs GPU kernel")
    plt.legend(); plt.grid(True, which="both", alpha=0.3)
    p = os.path.join(outdir, "deg1_time_vs_n.png")
    plt.tight_layout(); plt.savefig(p); plt.close()
    print("wrote", p)

    plt.figure(figsize=(9, 6))
    any_mem = False
    for (mode, k), pts in sorted(series.items()):
        if mode != "gpu":
            continue
        ms = [p[2] / 1024 / 1024 for p in pts]
        if any(ms):
            any_mem = True
            plt.plot([p[0] for p in pts], ms, "-o", label=f"gpu k={k}")
    if any_mem:
        plt.xlabel("deg1 size n"); plt.ylabel("peak GPU memory (MiB)")
        plt.title("GPU peak device memory vs deg1 size")
        plt.legend(); plt.grid(True, alpha=0.3)
        p = os.path.join(outdir, "deg1_peak_mem.png")
        plt.tight_layout(); plt.savefig(p); plt.close()
        print("wrote", p)
    else:
        plt.close()


def cmd_cost(args):
    # Presentation styling: readable from across a room (scoped to the cost view).
    plt.rcParams.update({
        "font.size": 12, "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.labelsize": 12, "legend.fontsize": 9, "figure.dpi": 130,
    })
    if not os.path.exists(args.csv):
        sys.exit(f"csv not found: {args.csv}")
    rows = load(args.csv)
    if not rows:
        sys.exit("empty csv")
    os.makedirs(args.outdir, exist_ok=True)
    bench = rows[0].get("bench", "deg1")
    if bench == "deg1":
        plot_deg1(rows, args.outdir)
    elif bench == "matmul":
        plot_matmul(rows, args.outdir)
    else:
        plot_full(rows, args.outdir)


# ══════════════════════════════════════════════════════════════════════════════
# llama2b — partition/staging sweep (former plot_llama2b.py)
# ══════════════════════════════════════════════════════════════════════════════

def cmd_llama2b(args):
    SRC = args.src
    OUT = args.out
    DEVICE_GB = 64.0
    os.makedirs(OUT, exist_ok=True)

    agg = collections.OrderedDict()
    for r in csv.DictReader(open(SRC)):
        a = agg.setdefault(r["config_id"], dict(
            st=r["status"], splits=int(r["num_splits"]), mk=int(r["matmul_k"]),
            lk=int(r["lasso_k"]), prov=0.0, peak=0.0))
        a["st"] = r["status"]

        def f(c):
            try: return float(r[c] or 0)
            except ValueError: return 0.0
        a["prov"] += f("part_prover_zkgpt_s")
        a["peak"] = max(a["peak"], f("part_gpu_peak_mib"))

    SPLITS = [8, 16, 32]
    LKS = [8, 16, 32]

    # matmul_k is inert (verified below): collapse it by mean over feasible mk.
    cell_t, cell_m, cell_ok = {}, {}, {}
    for s in SPLITS:
        for lk in LKS:
            v = [a for a in agg.values() if a["splits"] == s and a["lk"] == lk]
            ok = [a for a in v if a["st"] == "ACCEPT"]
            cell_ok[(s, lk)] = bool(ok)
            if ok:
                cell_t[(s, lk)] = np.mean([a["prov"] for a in ok])
                cell_m[(s, lk)] = np.mean([a["peak"] for a in ok]) / 1024.0

    # Figure 1: feasibility / performance grid
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    T = np.full((len(SPLITS), len(LKS)), np.nan)
    for i, s in enumerate(SPLITS):
        for j, lk in enumerate(LKS):
            if cell_ok[(s, lk)]:
                T[i, j] = cell_t[(s, lk)] / 1000.0

    im = ax.imshow(T, cmap="viridis_r", aspect="auto", origin="lower")
    for i, s in enumerate(SPLITS):
        for j, lk in enumerate(LKS):
            if cell_ok[(s, lk)]:
                ax.text(j, i + 0.13, f"{cell_t[(s,lk)]/1000:.2f} ks",
                        ha="center", va="center", color="white", fontsize=12, fontweight="bold")
                ax.text(j, i - 0.20, f"{cell_m[(s,lk)]:.1f} GB",
                        ha="center", va="center", color="white", fontsize=9.5)
            else:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor="0.86",
                                           edgecolor="white", lw=2, zorder=2))
                ax.text(j, i, "OOM", ha="center", va="center", color="0.30",
                        fontsize=12, fontweight="bold", zorder=3)

    ax.set_xticks(range(len(LKS)));    ax.set_xticklabels(LKS)
    ax.set_yticks(range(len(SPLITS))); ax.set_yticklabels([f"{s}\n({32//s} layer{'s' if 32//s>1 else ''}/part)" for s in SPLITS])
    ax.set_xlabel("staging factor  $k$  (lasso_k)")
    ax.set_ylabel("partitions (zkMod)")
    cb = fig.colorbar(im, ax=ax); cb.set_label("prover time (ks)")
    ax.set_xticks(np.arange(-.5, len(LKS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(SPLITS), 1), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_llama2b_feasibility.png", dpi=180)

    # Figure 2: memory / time tradeoff
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    style = {8: ("s", "#c44e52"), 16: ("^", "#dd8452"), 32: ("o", "#4c72b0")}
    for s in SPLITS:
        pts = [(a["peak"] / 1024, a["prov"]) for a in agg.values()
               if a["splits"] == s and a["st"] == "ACCEPT"]
        m, c = style[s]
        if pts:
            x, y = zip(*pts)
            ax.scatter(x, y, marker=m, s=110, color=c, edgecolor="white", lw=1.2, zorder=3,
                       label=f"{s} partitions ({32//s} layer{'s' if 32//s>1 else ''}/part)")
        else:
            ax.scatter([], [], marker=m, s=110, color=c, label=f"{s} partitions — all OOM")

    ax.axvline(DEVICE_GB, color="0.35", ls="--", lw=1.4)
    ax.text(DEVICE_GB - 0.8, ax.get_ylim()[1] * 0.97, "MI210  64 GB", rotation=90,
            va="top", ha="right", color="0.35", fontsize=9)
    for a in agg.values():
        if a["st"] == "ACCEPT" and a["mk"] == 8:
            ax.annotate(f"$k$={a['lk']}", (a["peak"] / 1024, a["prov"]),
                        textcoords="offset points", xytext=(9, 5), fontsize=9, color="0.25")
    ax.set_xlabel("peak GPU memory (GB)")
    ax.set_ylabel("prover time (s)")
    ax.set_xlim(20, 68)
    ax.legend(frameon=False, loc="center left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_llama2b_tradeoff.png", dpi=180)

    # sanity: is matmul_k inert?
    print("matmul_k sensitivity (fixed splits, lasso_k):")
    for s in SPLITS:
        for lk in LKS:
            v = sorted([a for a in agg.values() if a["splits"] == s and a["lk"] == lk
                        and a["st"] == "ACCEPT"], key=lambda a: a["mk"])
            if len(v) > 1:
                t = [a["prov"] for a in v]; p = [a["peak"] / 1024 for a in v]
                print(f"  splits={s:2} k={lk:2}: time {min(t):.0f}-{max(t):.0f}s "
                      f"(spread {100*(max(t)-min(t))/min(t):.1f}%)  peak {min(p):.1f}-{max(p):.1f} GB")
    print(f"\nwrote {OUT}/fig_llama2b_feasibility.png\nwrote {OUT}/fig_llama2b_tradeoff.png")


# ══════════════════════════════════════════════════════════════════════════════
# hetero — heterogeneous-partition probe (former plot_hetero.py)
# ══════════════════════════════════════════════════════════════════════════════

def cmd_hetero(args):
    D = args.dir
    CAP = 64.0

    LABEL = {
        "A_ctrl_k8":     ("uniform $k$=8  (grid optimum)", "#4c72b0"),
        "B_unif_k4":     ("uniform $k$=4", "#c44e52"),
        "C_het_p0-8_r4": ("hetero: p0 $k$=8, rest $k$=4", "#55a868"),
        "D_het_p0-8_r6": ("hetero: p0 $k$=8, rest $k$=6", "#8172b2"),
        "E_hetsplit":    ("hetero splits: p0=1 layer, rest=2", "#ccb974"),
        "F_unif_k6":     ("uniform $k$=6", "#937860"),
    }
    ORDER = ["A_ctrl_k8", "B_unif_k4", "F_unif_k6", "C_het_p0-8_r4", "D_het_p0-8_r6", "E_hetsplit"]

    arms = {}
    for f in sorted(glob.glob(f"{D}/*.log")):
        name = re.sub(r"_\d{8}_\d{6}$", "", os.path.basename(f)[:-4])
        txt = open(f, errors="ignore").read()
        peaks, times = [], []
        for m in re.finditer(r"PARTITION .*?prover_zkgpt_s=([0-9.]+) gpu_peak_mib=([0-9.]+)", txt):
            times.append(float(m.group(1)))
            peaks.append(float(m.group(2)) / 1024.0)
        oom = bool(re.search(r"out of memory|hipErrorOutOfMemory", txt, re.I))
        arms[name] = dict(peaks=peaks, times=times, oom=oom,
                          total=sum(times), peak=max(peaks) if peaks else float("nan"))

    present = [a for a in ORDER if a in arms]
    if not present:
        sys.exit("no arms found")

    # Fig 1: per-partition peak profile
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for name in present:
        a = arms[name]
        if not a["peaks"]:
            continue
        lab, col = LABEL[name]
        ax.plot(range(len(a["peaks"])), a["peaks"], marker="o", ms=4.5, lw=1.6,
                color=col, label=lab)
    oomers = [n for n in present if arms[n]["oom"]]
    for j, name in enumerate(oomers):
        a = arms[name]
        lab, col = LABEL[name]
        n = len(a["peaks"])                       # partition index where it died
        y = CAP + 2.0 + 2.6 * j                   # stagger: B and F both die at p0
        ax.plot([n], [y], marker="X", ms=13, color=col, clip_on=False, zorder=5)
        ax.annotate(f"OOM at p{n} — {lab}", (n, y), textcoords="offset points",
                    xytext=(13, -3), color=col, fontsize=8.5, fontweight="bold")
    ax.axhline(CAP, color="0.35", ls="--", lw=1.4)
    ax.text(31.6, CAP + 1.0, "MI210  64 GB", color="0.35", fontsize=9, ha="right")
    ax.set_xlabel("partition index")
    ax.set_ylabel("peak GPU memory (GB)")
    ax.set_ylim(0, CAP + 10)
    ax.set_xlim(-1.5, 33)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{D}/fig_hetero_profile.png", dpi=180)

    # Fig 2: total prover time
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ok = [n for n in present if not arms[n]["oom"] and arms[n]["times"]]
    y = np.arange(len(ok))
    vals = [arms[n]["total"] for n in ok]
    cols = [LABEL[n][1] for n in ok]
    ax.barh(y, vals, color=cols, height=0.6)
    for i, n in enumerate(ok):
        a = arms[n]
        ax.text(a["total"] + max(vals) * 0.015, i,
                f"{a['total']:.0f} s   ({a['peak']:.1f} GB)", va="center", fontsize=9.5)
    ax.set_yticks(y); ax.set_yticklabels([LABEL[n][0] for n in ok], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("total prover time (s)")
    ax.set_xlim(0, max(vals) * 1.32)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{D}/fig_hetero_time.png", dpi=180)

    # table
    print(f"{'arm':34} {'status':7} {'parts':>5} {'peak GB':>8} {'prover s':>9} {'vs ctrl':>8}")
    base = arms.get("A_ctrl_k8", {}).get("total")
    for n in present:
        a = arms[n]
        st = "OOM" if a["oom"] else "ACCEPT"
        sp = f"{base/a['total']:.2f}x" if (base and not a["oom"] and a["total"]) else "-"
        pk = f"{a['peak']:.1f}" if a["peaks"] else "-"
        tt = f"{a['total']:.0f}" if not a["oom"] else "-"
        print(f"{LABEL[n][0]:34} {st:7} {len(a['peaks']):5} {pk:>8} {tt:>9} {sp:>8}")
    print(f"\nwrote {D}/fig_hetero_profile.png\nwrote {D}/fig_hetero_time.png")


# ══════════════════════════════════════════════════════════════════════════════
# ck — per-depth knob sweep (former plot_ck_sweep.py)
# ══════════════════════════════════════════════════════════════════════════════

# (axis name in the CSV's `axis` column, CSV column to use as the x value, x-axis label)
_CK_AXES = [("matmul_k", "matmul_k", "matmul_k (Blendy stage count)"),
            ("range_c", "range_c", "range_c (chunk bits)"),
            ("range_k", "range_k", "range_k (Blendy stages)")]


def _ck_plot_one(path, out):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        print(f"skip {path}: empty"); return
    L = rows[0]["L"]

    fig, axes = plt.subplots(3, 2, figsize=(13, 14))

    for i, (filter_axis, x_col, axis_label) in enumerate(_CK_AXES):
        ax_t, ax_m = axes[i]
        arows = sorted((r for r in rows if r["axis"] == filter_axis),
                       key=lambda r: fnum(r[x_col]) or 0)
        xs, tot, rng, fail_x, fail_st = [], [], [], [], []
        mem = []
        for r in arows:
            x = fnum(r[x_col])
            if x is None: continue
            if r["status"] == "ACCEPT":
                xs.append(x)
                tot.append(fnum(r["total_prover_s"]))
                rng.append(fnum(r["range_prover_s"]))
            else:
                fail_x.append(x); fail_st.append(r["status"])
            mem_mib = fnum(r["peak_gpu_mem_mib"])
            if mem_mib is not None:
                mem.append((x, mem_mib))

        if xs:
            ax_t.plot(xs, tot, marker="o", color="#1f77b4", label="total prover time")
            ax_t.plot(xs, rng, marker="s", color="#d62728", label="range/logup prover time")
        if fail_x:
            ytop = max([v for v in tot if v is not None], default=1) * 1.1 or 1
            ax_t.scatter(fail_x, [ytop] * len(fail_x), marker="x", s=80, color="crimson", zorder=6)
            for x, st in zip(fail_x, fail_st):
                ax_t.annotate(st, (x, ytop), fontsize=7, rotation=90, ha="center", va="bottom", color="crimson")
        ax_t.set_xlabel(axis_label)
        ax_t.set_ylabel("seconds")
        ax_t.set_title(f"L={L}: time vs {filter_axis}")
        ax_t.grid(True, ls=":", alpha=0.4)
        ax_t.legend(fontsize=8)

        if mem:
            mxs = [p[0] for p in mem]
            ax_m.plot(mxs, [p[1] for p in mem], marker="^", color="#9467bd", label="peak GPU memory (MiB)")
        ax_m.set_xlabel(axis_label)
        ax_m.set_ylabel("MiB")
        ax_m.set_title(f"L={L}: GPU memory vs {filter_axis}")
        ax_m.grid(True, ls=":", alpha=0.4)
        ax_m.legend(fontsize=8)

    fig.suptitle(f"Depth L={L} — knob sweep (full pipeline incl. range/logup)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# e2e — end-to-end model figures. Data-driven: everything is read from the
# --data-dir sweep directory, nothing is hardcoded, so a re-sweep just redraws.
# ══════════════════════════════════════════════════════════════════════════════

# Okabe-Ito, validated with scripts/validate_palette.js (pairs="all", light).
# Arms keep one hue each across every panel; colour follows the entity, not rank.
E2E = {"tuned": "#0072B2", "base": "#D55E00", "uvm": "#CC79A7"}
# 5-phase ramp: passes lightness/chroma, CVD separation at the *floor* (dE 7.6),
# so segments are ALWAYS direct-labelled — colour never carries identity alone.
PHASE_C = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
PHASE_K = ["ph_gkr_generic_s", "ph_gkr_matmul_s", "ph_lasso_s",
           "ph_hyrax_commit_s", "ph_hyrax_open_s"]
PHASE_N = ["GKR generic (CPU)", "GKR matmul", "GKR wiring (Lasso)",
           "Hyrax commit", "Hyrax open"]
# Sequential ramp for the (t,k) grid: ONE hue, light->dark (magnitude encoding).
# Light end stops well short of white so the palest cell still clears ~2:1 on the
# figure surface; every cell is also direct-labelled, so colour never carries the
# value alone -- required here because the feasible makespans span only ~6%.
SEQ_BLUE = ["#DCE9F5", "#B4CFE8", "#7FADD6", "#4A8CC4", "#0072B2", "#04527F"]
VRAM_MIB = 65536.0          # one MI210
FLOOR_8B_MIB = 49152.0      # 3N at n=29 = 3*2^29*32B — independent of k AND t
E2E_DIR = os.path.join(HERE, "results", "e2e_v2", "data")


def _e2e_rows(fname):
    p = os.path.join(E2E_DIR, fname)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def _f(r, k):
    try:
        return float(r.get(k, "") or "")
    except ValueError:
        return None


def _exp4_summary(fname="exp4_summary.csv"):
    """Rows from the exp4 summary CSV (built by the extraction step, which pairs
    each driver-log config line with its partition log). status is OK / OOM /
    NOT_RUN -- NOT_RUN means the stale range_size%c guard skipped it, which is a
    different fact from an OOM and must not be drawn as one."""
    p = os.path.join(E2E_DIR, fname)
    if not os.path.exists(p):
        return []
    out = []
    for r in csv.DictReader(open(p)):
        tot = fnum(r.get("makespan_total_s"))
        out.append(dict(tag=r["tag"], t=int(r["t"]), k=int(r["k"]), c=int(r["c"]),
                        parts=int(r["parts"]), total=tot,
                        peak=fnum(r.get("peak_mib")), status=r["status"],
                        ok=(r["status"] == "OK" and tot is not None),
                        src=r.get("src", ""),
                        uvm=r["tag"].startswith("uvm")))
    return out


def _grid_cell(S, t, k, c=7):
    for r in S:
        if r["t"] == t and r["k"] == k and r["c"] == c and not r["uvm"]:
            return r
    return None


def _heat(ax, S, ts, ks, value, fmt, title, cbar_label, n=28, ylab=True):
    """Grid heatmap over (t, k). OOM cells are hatched, never coloured -- an
    infeasible point must not read as a fast (pale) one. Every feasible cell is
    direct-laballed because the value spread is far too narrow for colour alone."""
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    vals = [value(r) for t in ts for k in ks
            if (r := _grid_cell(S, t, k)) and r["ok"] and value(r) is not None]
    if not vals:
        return None
    norm = Normalize(vmin=min(vals), vmax=max(vals))

    for xi, t in enumerate(ts):
        for yi, k in enumerate(ks):
            r = _grid_cell(S, t, k)
            if r is None:
                ax.add_patch(plt.Rectangle((xi - .5, yi - .5), 1, 1, facecolor="0.93",
                                           edgecolor="white", linewidth=2))
                continue
            if not r["ok"] or value(r) is None:
                ax.add_patch(plt.Rectangle((xi - .5, yi - .5), 1, 1, facecolor="none",
                                           edgecolor=E2E["base"], linewidth=1.8,
                                           hatch="xxx"))
                ax.text(xi, yi, "OOM", ha="center", va="center", fontsize=8,
                        color=E2E["base"], fontweight="bold")
                continue
            v = value(r)
            ax.add_patch(plt.Rectangle((xi - .5, yi - .5), 1, 1, facecolor=cmap(norm(v)),
                                       edgecolor="white", linewidth=2))
            ax.text(xi, yi, fmt(v), ha="center", va="center", fontsize=8.5,
                    color="white" if norm(v) > 0.55 else "0.15", fontweight="bold")

    ax.set_xlim(-.5, len(ts) - .5); ax.set_ylim(-.5, len(ks) - .5)
    ax.set_xticks(range(len(ts)))
    ax.set_xticklabels(["untiled" if t < 0 else str(t) for t in ts])
    ax.set_yticks(range(len(ks)))
    # l_stage = ceil(n/k) is what actually sizes the buffers; many k collapse onto
    # the same l, so label both or the axis looks like it has more resolution.
    # Panels share this axis -- repeating it collides with the neighbour's colorbar.
    ax.set_yticklabels([f"{k}  ($\\ell$={-(-n // k)})" for k in ks] if ylab else [])
    ax.set_xlabel("ext-table tile bits  $t$")
    if ylab:
        ax.set_ylabel("lasso stage count  $k$")
    ax.set_title(title)
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(cbar_label, fontsize=9)
    cb.outline.set_visible(False)
    return True


def _pick(rows, t, k, c, uvm):
    for r in rows:
        if r["t"] == t and r["k"] == k and r["c"] == c and r["uvm"] == uvm and r["ok"]:
            return r
    return None


def _exp4_results(log="exp4_driver.log"):
    """Parse the [exp4] RESULT lines — the authoritative 8B summary."""
    p = os.path.join(E2E_DIR, log)
    if not os.path.exists(p):
        return {}
    out = {}
    for line in open(p):
        m = re.search(r"RESULT (\S+)\s+(.*)", line)
        if not m:
            continue
        tag, rest = m.group(1), m.group(2)
        if "NO PARTITIONS" in rest:
            out[tag] = {"oom": True}
        else:
            g = dict(re.findall(r"(\w+)=([\d.]+)", rest))
            if g:
                out[tag] = {"oom": False, **{k: float(v) for k, v in g.items()}}
    return out


def cmd_e2e(args):
    global E2E_DIR
    if getattr(args, "data_dir", None):
        E2E_DIR = args.data_dir
    print(f"reading {E2E_DIR}")
    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.labelsize": 11, "legend.fontsize": 9, "figure.dpi": 130,
    })
    out = args.out
    os.makedirs(out, exist_ok=True)

    base = _e2e_rows("gpt2_baseline_c7_mk8_untiled.csv")
    tune = _e2e_rows("gpt2_tuned_c13_mk16_t18.csv")

    # ── Figure 1: GPT-2 — where the time goes, and the feasibility cliff ─────
    if base and tune:
        fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(17, 5))

        Ls = [int(r["L"]) for r in tune]
        bot = [0.0] * len(tune)
        for ci, (key, nm) in enumerate(zip(PHASE_K, PHASE_N)):
            vals = [(_f(r, key) or 0.0) for r in tune]
            a1.bar(range(len(Ls)), vals, bottom=bot, color=PHASE_C[ci], label=nm,
                   width=0.62, edgecolor="white", linewidth=1.2)
            for i, v in enumerate(vals):      # direct labels: required, see PHASE_C
                if v > 2.0:
                    a1.text(i, bot[i] + v / 2, f"{v:.0f}", ha="center", va="center",
                            fontsize=8, color="white", fontweight="bold")
            bot = [b + v for b, v in zip(bot, vals)]
        rng = [(_f(r, "range_prove_s") or 0.0) for r in tune]
        a1.bar(range(len(Ls)), rng, bottom=bot, color="0.45", label="range check",
               width=0.62, edgecolor="white", linewidth=1.2, hatch="//")
        for i, v in enumerate(rng):
            a1.text(i, bot[i] + v / 2, f"{v:.0f}", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
        a1.set_xticks(range(len(Ls))); a1.set_xticklabels(Ls)
        a1.set_xlabel("transformer depth L")
        a1.set_ylabel("prover time (s)")
        a1.set_title("(a) prover time by phase (tuned arm)\nlabels = seconds")
        a1.legend(fontsize=8, loc="upper left")
        a1.grid(True, axis="y", ls=":", alpha=0.4)

        def series(rows, key):
            xs, ys = [], []
            for r in rows:
                v = _f(r, key)
                if v is not None:
                    xs.append(int(r["L"])); ys.append(v)
            return xs, ys

        bx, by = series(base, "prover_total_s")
        tx, ty = series(tune, "prover_total_s")
        a2.plot(tx, ty, "o-", color=E2E["tuned"], lw=2, ms=7, label="tuned (c=13, k=16, t=18)")
        a2.plot(bx, by, "s-", color=E2E["base"], lw=2, ms=7, label="default (c=7, k=8, untiled)")
        oom_L = [int(r["L"]) for r in base if r["status"] != "ACCEPT"]
        for L in oom_L:
            a2.scatter([L], [max(by) * 1.12], marker="o", s=110, facecolors="none",
                       edgecolors=E2E["base"], linewidths=2.2, zorder=5)
            a2.annotate("OOM", (L, max(by) * 1.12), textcoords="offset points",
                        xytext=(0, -16), ha="center", fontsize=8.5,
                        color=E2E["base"], fontweight="bold")
        a2.set_xlabel("transformer depth L"); a2.set_ylabel("prover_total_s (s)")
        a2.set_title("(b) prover_total_s vs depth")
        a2.set_xticks(Ls); a2.grid(True, ls=":", alpha=0.4); a2.legend(loc="upper left")

        bmx, bmy = series(base, "gpu_peak_mib")
        tmx, tmy = series(tune, "gpu_peak_mib")
        a3.plot(tmx, [v / 1024 for v in tmy], "o-", color=E2E["tuned"], lw=2, ms=7, label="tuned")
        a3.plot(bmx, [v / 1024 for v in bmy], "s-", color=E2E["base"], lw=2, ms=7, label="default")
        a3.axhline(VRAM_MIB / 1024, color="0.35", ls="--", lw=1.5)
        a3.text(1.1, VRAM_MIB / 1024 * 1.03, "MI210 VRAM = 64 GiB", fontsize=9, color="0.3")
        if bmy and tmy:
            a3.annotate(f"{bmy[-1]/1024:.0f} GiB",
                        xy=(bmx[-1], bmy[-1] / 1024), xytext=(4.5, 88), fontsize=9,
                        color=E2E["base"], fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color=E2E["base"], lw=1.5))
        a3.set_xlabel("transformer depth L"); a3.set_ylabel("peak GPU memory (GiB)")
        a3.set_title("(c) peak GPU memory vs depth")
        a3.set_xticks(Ls); a3.grid(True, ls=":", alpha=0.4); a3.legend(loc="upper left")

        fig.suptitle("GPT-2, monolithic prover (device, 1x MI210)",
                     fontweight="bold")
        fig.tight_layout()
        p = os.path.join(out, "e2e_gpt2.png")
        fig.savefig(p, dpi=150); plt.close(fig)
        print(f"wrote {p}")

    # ── Figure 2: Llama-8B — the (t,k) grid + matched device/UVM pair ────────
    S = _exp4_summary()
    if S:
        ts = sorted({r["t"] for r in S if r["c"] == 7 and not r["uvm"]})
        # Grid rows only. The c-line is run at a base (t,k) that need not be a grid
        # point (k=8 in the v3 sweep); admitting it would draw a row of one cell
        # and five blanks, which reads as missing data rather than "not swept".
        kcount = collections.Counter(r["k"] for r in S if r["c"] == 7 and not r["uvm"])
        ks = sorted(k for k, n in kcount.items() if n > 1)

        fig = plt.figure(figsize=(17.5, 5.4))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 0.7], wspace=0.32)
        b1, b2, b3 = (fig.add_subplot(gs[0]), fig.add_subplot(gs[1]),
                      fig.add_subplot(gs[2]))

        _heat(b1, S, ts, ks, lambda r: r["total"], lambda v: f"{v:.0f}",
              "(a) makespan (s)   [c=7]", "makespan (s)")
        _heat(b2, S, ts, ks, lambda r: (r["peak"] or 0) / 1024.0,
              lambda v: f"{v:.0f}", "(b) peak GPU memory (GiB)   [c=7]",
              "peak GPU memory (GiB)", ylab=False)

        # (c) device vs UVM at the SAME (t,k,c) — the matched pair. Comparing the
        # best device config against the only UVM config would overstate the gap.
        dev = _pick(S, t=20, k=8, c=7, uvm=False)
        uvm = _pick(S, t=20, k=8, c=7, uvm=True)
        if dev and uvm:
            b3.bar([0, 1], [dev["total"], uvm["total"]],
                   color=[E2E["tuned"], E2E["uvm"]], width=0.5)
            for x, r in zip([0, 1], [dev, uvm]):
                b3.text(x, r["total"] * 1.02, f"{r['total']:.1f} s", ha="center",
                        va="bottom", fontsize=13, fontweight="bold")
                b3.text(x, r["total"] * 0.5, f"{r['parts']}/32 parts", ha="center",
                        fontsize=8.5, color="white", fontweight="bold")
            b3.text(0.5, uvm["total"] * 0.72, f"{uvm['total']/dev['total']:.2f}x",
                    ha="center", fontsize=17, fontweight="bold", color="0.3")
            b3.set_xticks([0, 1])
            b3.set_xticklabels(["device\n(hipMalloc)", "UVM\n(managed)"])
            b3.set_ylim(0, uvm["total"] * 1.25)
        b3.set_ylabel("makespan (s)")
        b3.set_title("(c) device vs UVM\nat t=20, k=8, c=7")
        b3.grid(True, axis="y", ls=":", alpha=0.4)

        fig.suptitle("Llama-8B, zkMod split (32 partitions, 1x MI210)", fontweight="bold")
        p = os.path.join(out, "e2e_llama8b.png")
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"wrote {p}")

        # ── Figure 3: one knob per panel, the other two held fixed ───────────
        # k is held at the finest value present at EVERY feasible t, so the t panel
        # is a clean cross-section rather than a mix of feasible/infeasible rows.
        kfix = next((k for k in ks if all(
            (r := _grid_cell(S, t, k)) and r["ok"] for t in ts if t >= 0)), 14)
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
        panels = [
            (ax[0], "t", [r for r in S if r["k"] == kfix and r["c"] == 7 and r["t"] >= 0],
             "ext-table tile bits  t", f"(a) vary t   (k={kfix}, c=7)"),
            (ax[1], "k", [r for r in S if r["t"] == 20 and r["c"] == 7],
             "lasso stage count  k", "(b) vary k   (t=20, c=7)"),
            (ax[2], "c", [r for r in S if r["t"] == 20 and r["k"] == 8],
             "chunk bits  c", "(c) vary c   (t=20, k=8)"),
        ]
        for a, key, rows, xlab, ttl in panels:
            rows = sorted([r for r in rows if not r["uvm"]], key=lambda r: r[key])
            okr = [r for r in rows if r["ok"]]
            if len(okr) > 1:
                a.plot([r[key] for r in okr], [r["total"] for r in okr], "o-",
                       color=E2E["tuned"], lw=2, ms=8)
            elif okr:
                a.plot([okr[0][key]], [okr[0]["total"]], "o", color=E2E["tuned"], ms=8)
            for r in okr:
                a.annotate(f"{r['total']:.0f}\n({r['parts']}/32)", (r[key], r["total"]),
                           textcoords="offset points", xytext=(0, 9), ha="center", fontsize=7.5)
            for r in rows:
                if not r["ok"]:
                    a.axvline(r[key], color=E2E["base"], ls=":", lw=1.6)
                    a.text(r[key], 0.06, r["status"], rotation=90, ha="right", va="bottom",
                           transform=a.get_xaxis_transform(), fontsize=8,
                           color=E2E["base"], fontweight="bold")
            a.set_xlabel(xlab); a.set_title(ttl)
            a.grid(True, ls=":", alpha=0.4)
            if okr:
                lo = min(r["total"] for r in okr); hi = max(r["total"] for r in okr)
                a.set_ylim(lo - (hi - lo) * 0.6 - 3, hi + (hi - lo) * 0.9 + 5)
        ax[0].set_ylabel("makespan (s) = max$_i$ prover_total_s")
        fig.suptitle("Llama-8B knob cross-sections (one knob varied, other two fixed)",
                     fontweight="bold")
        fig.tight_layout()
        p = os.path.join(out, "e2e_llama8b_knobs.png")
        fig.savefig(p, dpi=150); plt.close(fig)
        print(f"wrote {p}")
def cmd_ck(args):
    paths = []
    for pat in args.csv:
        paths.extend(sorted(glob.glob(pat)) or [pat])

    if args.out and len(paths) > 1:
        raise SystemExit("--out only valid with a single input CSV")

    for p in paths:
        out = args.out or _mirror_out(p)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        _ck_plot_one(p, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cost", help="Exp-1/Exp-2 cost-model figures (compare_range CSV)")
    c.add_argument("csv", nargs="?", default=os.path.join(HERE, "range_compare.csv"))
    c.add_argument("--outdir", default=HERE)
    c.set_defaults(func=cmd_cost)

    l = sub.add_parser("llama2b", help="llama2b partition/staging sweep figures")
    l.add_argument("--src", default=os.path.join(HERE, "results/llama2b/llama_all_20260713_150819.csv"))
    l.add_argument("--out", default=os.path.join(HERE, "results/llama2b"))
    l.set_defaults(func=cmd_llama2b)

    h = sub.add_parser("hetero", help="heterogeneous-partition probe figures")
    h.add_argument("--dir", default=os.path.join(HERE, "results/llama2b/hetero"))
    h.set_defaults(func=cmd_hetero)

    e = sub.add_parser("e2e", help="Exp-4 end-to-end figures (GPT-2 monolith + Llama-8B split)")
    e.add_argument("--out", default=os.path.join(HERE, "results/e2e/images"))
    e.add_argument("--data-dir", default=E2E_DIR,
                   help="sweep data dir (default: %(default)s)")
    e.set_defaults(func=cmd_e2e)

    k = sub.add_parser("ck", help="per-depth knob sweep figures (ck_sweep CSVs)")
    k.add_argument("csv", nargs="+", help="one or more ck_sweep_L<L>_*.csv (globs OK)")
    k.add_argument("--out", default="", help="only valid with a single input file")
    k.set_defaults(func=cmd_ck)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    main()
