"""Shared plot style for the zkLLM figures: blue gradient (light = baseline, dark = zkMod), baseline hatched."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 8, "font.weight": "bold", "axes.labelweight": "bold", "axes.titleweight": "bold",
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6, "axes.linewidth": 0.6, "lines.linewidth": 1.0, "lines.markersize": 4,
    "hatch.linewidth": 0.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    "legend.frameon": True, "legend.framealpha": 1.0, "legend.edgecolor": "black",
    "legend.fancybox": False, "legend.borderpad": 0.2, "legend.handlelength": 1.2, "legend.handleheight": 0.6,
    "legend.handletextpad": 0.4, "legend.labelspacing": 0.15, "legend.columnspacing": 0.6, "legend.borderaxespad": 0.3,
})
BAR = dict(edgecolor="black", linewidth=0.5)
BLUES = ["#E6ECF5", "#B3C6E0", "#7FA0CB", "#4F7CAC", "#2B5180"]   # light -> dark, as zkML's SHADE
BASE = (BLUES[0], "////")                   # zkLLM baseline
ZK = (BLUES[4], "")                         # zkMod
WG_FILL, PV_FILL = ("#B0C4DE", ""), (BLUES[3], "")
MARKS = ["o", "s", "^", "D", "v"]


def ramp(n, light=BLUES[1], dark=BLUES[4]):
    """n solid blues, light -> dark (an ordered family, e.g. split counts)."""
    if n == 1:
        return [(dark, "")]
    lo, hi = (np.array([int(c[i:i + 2], 16) for i in (1, 3, 5)]) for c in (light, dark))
    return [("#%02X%02X%02X" % tuple(np.round(lo + (hi - lo) * t).astype(int)), "") for t in np.linspace(0, 1, n)]


def name(model, newline=False):
    """'opt-1.3b' / '1.3b' -> 'OPT-1.3B' (or 'OPT\\n1.3B')."""
    return "OPT" + ("\n" if newline else "-") + model.removeprefix("opt-").upper()


def bar(ax, x, h, w, shade, bottom=None, label=None):
    fc, hatch = shade
    ax.bar(x, h, w, bottom=bottom, color=fc, hatch=hatch, label=label, **BAR)


def grid(ax, axis="y"):
    ax.grid(axis=axis, color="0.85", lw=0.4, which="major"); ax.set_axisbelow(True)


def speedup(ax, x, y, v, rotation=0, size=6.5, dy=2):
    """'3.65x' just above (x, y); v = baseline / zkMod."""
    if v != v:
        return
    ax.annotate(f"{v:.2f}\u00d7", (x, y), xytext=(0, dy), textcoords="offset points", ha="center", va="bottom",
                fontsize=size, rotation=rotation)


def pair_bars(ax, groups, base, zk, base_label, zk_label, ylabel):
    """One zkLLM / zkMod bar pair per group; zkMod's bar carries its speedup."""
    x = np.arange(len(groups)); w, gap = 0.28, 0.2      # gap keeps the speedup clear of the baseline bar
    d = (w + gap) / 2
    bar(ax, x - d, base, w, BASE, label=base_label)
    bar(ax, x + d, zk, w, ZK, label=zk_label)
    for xx, b, z in zip(x, base, zk):
        speedup(ax, xx + d, z, b / z)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel); grid(ax)


def group_bars(ax, groups, series, vals, label_fmt, ylabel, size=5.5, annotate=None, legend_cols=2,
               legend_title=None, legend_loc="upper left"):
    """Grouped speedup bars: one group per model, one bar per series (partition count / seq length),
    shaded light -> dark in series order. vals[g][s] = speedup or None (no run). Bars are labelled
    with their speedup only when there are few enough to keep the labels apart (annotate=None:
    at most 10 bars). legend_loc "upper ..." reserves headroom above the bars for the legend;
    "lower ..." lays it over the bars' lower part (it must stay below the shortest bar it covers)."""
    if annotate is None:
        annotate = len(groups) * len(series) <= 10
    x = np.arange(len(groups)); w = 0.84 / len(series)
    for j, (sv, shade) in enumerate(zip(series, ramp(len(series)))):
        xs = x - 0.42 + w * (j + 0.5)
        hs = [vals[g].get(sv) for g in groups]
        bar(ax, [a for a, h in zip(xs, hs) if h], [h for h in hs if h], w * 0.92, shade, label=label_fmt(sv))
        for a, h in zip(xs, hs):
            if h and annotate:
                speedup(ax, a, h, h, rotation=90, size=size)
    ax.set_xticks(x); ax.set_xticklabels([name(g, True) if not g.startswith(("GPT", "OPT")) else g for g in groups])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel(ylabel); grid(ax)
    rows = -(-len(series) // legend_cols) if legend_loc.startswith("upper") else 0
    top = max(h for g in groups for h in vals[g].values() if h)
    ax.set_ylim(0, top * ((1.45 if annotate else 1.15) + 0.2 * rows))
    leg = ax.legend(loc=legend_loc, ncol=legend_cols, fontsize=6, columnspacing=1.0, handlelength=1.2,
                    borderaxespad=0.3, frameon=True, framealpha=1, title=legend_title, title_fontsize=6)
    leg._legend_box.align = "left"
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}\u00d7"))


def save(fig, path_noext):
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_noext}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path_noext}.png/.pdf")
