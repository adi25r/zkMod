"""plotstyle.py -- the one house style for every figure under tools/.

A transcription of /home/pawks/workspace/aditya/scripts/p1.py (+ p2/p3 for the
legend/stacked variants), the paper's plotting scripts. Do not restyle here;
change p1.py first.

  rcParams   font 14, bold axis labels/titles, tick labels 14
  bars       cm.Blues ramp 0.3..0.9 per series, edgecolor black, group centres
             at index*(bwmax+0.1), bar_width = 0.8*bwmax/n
  ticks      x tick labels 16 pt bold, rotation 0
  grid       y major, dotted, 0.7, gray
  labels     value above each bar, 12 pt black, (0,3) pt offset
  legend     INSIDE the axes: framed, white face, black edge, framealpha 0.95,
             bold, ncol = number of series; headroom comes from ylim
  geomean    grey axvspan alpha 0.3 behind the last group
  canvas     fig.set_size_inches(15, 3); savefig bbox tight, transparent, dpi 300
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

EDGE = "black"
LABEL_C = "black"
LABEL_SIZE = 12
STACK_2 = ["#4682B4", "#B0C4DE"]          # p3.py: base (bottom), overhead (top)
BWMAX, GAP, BW_FACTOR = 0.9, 0.1, 0.8
CANVAS = (15, 3)


def apply():
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })


def blues(n, lo=0.3, hi=0.9):
    return cm.Blues(np.linspace(lo, hi, max(n, 1)))


def figure(**kw):
    return plt.subplots(figsize=CANVAS, **kw)


def grid(ax):
    ax.grid(True, which="major", axis="y", linestyle="dotted", linewidth=0.7, color="gray")


def xticks(ax, positions, labels, size=16):
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=size, fontweight="bold")


def legend(ax, loc="upper left", ncol=None, handles=None, labels=None):
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    return ax.legend(handles, labels, loc=loc, frameon=True, ncol=ncol or len(labels),
                     prop={"weight": "bold", "size": 12}, facecolor="white",
                     edgecolor="black", framealpha=0.95)


def grouped_bars(ax, groups, series, values, colors=None, bwmax=BWMAX, gap=GAP):
    """values[series_i][group_j] (None = no bar). p1.py geometry.
    Returns ({series: BarContainer}, group_centres)."""
    n = len(series)
    colors = blues(n) if colors is None else colors
    bar_width = BW_FACTOR * bwmax / n
    idx = np.arange(len(groups)) * (bwmax + gap)
    out = {}
    for i, name in enumerate(series):
        xs, ys = [], []
        for j in range(len(groups)):
            v = values[i][j]
            if v is None:
                continue
            xs.append(idx[j] + (i - (n - 1) / 2) * bar_width); ys.append(v)
        out[name] = ax.bar(xs, ys, bar_width, label=name, color=colors[i], edgecolor=EDGE)
    xticks(ax, idx, groups)
    half = (bwmax + gap) / 2
    ax.set_xlim(idx[0] - half, idx[-1] + half)
    return out, idx


def stacked_bars(ax, groups, parts, values, colors=None, width=0.6):
    """values[part_i][group_j] stacked bottom-up; p3.py stacked variant
    (black edges, linewidth 1.2). Returns (containers, tops)."""
    colors = (STACK_2 if len(parts) == 2 else blues(len(parts))) if colors is None else colors
    x = np.arange(len(groups))
    bottom = np.zeros(len(groups))
    conts = []
    for i, name in enumerate(parts):
        v = np.array(values[i], dtype=float)
        conts.append(ax.bar(x, v, width, bottom=bottom, label=name, color=colors[i],
                            edgecolor=EDGE, linewidth=1.2))
        bottom += v
    xticks(ax, x, groups, size=14)
    return conts, bottom


def value_labels(ax, rects, fstr="{:.2f}", prefix=None, rotation=0):
    for k, r in enumerate(rects):
        h = r.get_height()
        if h is None or h <= 0:
            continue
        ax.annotate((prefix[k] if prefix else "") + fstr.format(h),
                    xy=(r.get_x() + r.get_width() / 2, r.get_y() + h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=LABEL_SIZE, color=LABEL_C, rotation=rotation)


def top_labels(ax, xs, tops, fstr="{:.2f}", prefix=None):
    for k, (x, t) in enumerate(zip(xs, tops)):
        ax.annotate((prefix[k] if prefix else "") + fstr.format(t), xy=(x, t), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=LABEL_SIZE, color=LABEL_C)


def shade_group(ax, idx, j, bwmax=BWMAX, gap=GAP):
    half = (bwmax + gap) / 2
    ax.axvspan(idx[j] - half, idx[j] + half, color="gray", alpha=0.3)


def trim_xlim(ax, padding=0.35):
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo + padding, hi - padding)


def headroom(ax, factor=1.6, log_factor=12.0, log=False):
    """Explicit ylim (p1.py's `ylim`) leaving room for the inside legend + labels."""
    lo, hi = ax.get_ylim()
    if log:      # start at the decade below the shortest bar so every bar has a visible base
        ymin = min((r.get_height() for r in ax.patches if r.get_height() > 0), default=lo)
        lo = 10 ** (np.floor(np.log10(ymin)) )
        if ymin / lo < 2: lo /= 10
    ax.set_ylim(lo if log else 0, hi * (log_factor if log else factor))


def save(fig, path_noext, size=CANVAS, top=None):
    """Transparent 300 dpi PDF (+PNG twin at the same path). Returns the PDF path."""
    fig.tight_layout()
    fig.set_size_inches(*size)
    if top is not None:
        fig.subplots_adjust(top=top)
    os.makedirs(os.path.dirname(path_noext) or ".", exist_ok=True)
    pdf = path_noext + ".pdf"
    for p in (pdf, path_noext + ".png"):
        fig.savefig(p, bbox_inches="tight", transparent=True, dpi=300)
    plt.close(fig)
    return pdf

# ---- prover phase table (shared by the overlay / PMC figures) -----------------
# One row per PHASE marker (src/phase_log.hpp). Grouped hues, not a Blues ramp:
# fifteen phases need a family structure to be readable -- witgen greens, waits
# greys, commits purples, range oranges, verify teals. This is the only place
# the house style keeps a categorical palette.
# (phase, colour, what it does, where)
SPEC = [
 ("witgen","#4E8F55","build the circuit and witness for real",
  "LLMSplitBlock::create"),
 ("carry_publish","#79B97F","export boundary state into the shared mmap",
  "exportSplitBlockCarry"),
 ("carry_import","#A6D3A9","adopt the predecessor's boundary state",
  "initStateFromCarry"),

 ("carry_wait","#B9C3CA","BLOCKED on the predecessor's witgen",
  "read(carry gate)"),
 ("barrier_wait","#8C99A3","BLOCKED until every partition finishes witgen",
  "read(allgate)"),
 ("token_wait","#DCE3E7","BLOCKED waiting for a proving slot",
  "read(token pipe)"),

 ("sequence_kv_commit","#4B3F9E","commit this chunk's K/V, one per layer",
  "commit_values_standalone"),
 ("standalone_input_commit","#6A5FBE","commit the input-witness prefix",
  "commit_values_standalone"),
 ("output_commit","#9086D8","commit private output for the next partition",
  "commit_private_output_for_next"),
 ("execute_commit","#B3ABE8","commit all remaining witness segments",
  "prover::executeCommit"),

 ("range_build","#F0B878","build the lookup tables and logup constraints",
  "range_prover::init + build"),
 ("range_prove","#D97B22","logup deg1 + deg3 sumchecks   [GPU-BOUND]",
  "range_prover::prove"),

 ("verify_gkr","#1F6B78","GKR layer sumchecks + matmul product-sumcheck",
  "verifier::verifyGKR"),
 ("verify_lasso","#3AA0B0","Lasso lookup argument",
  "verifier::verifyLasso"),
 ("open_commit","#7CC7D2","Hyrax opening / bullet-fold of every commitment",
  "verifier::openCommit"),
]
LEAF = {x[0] for x in SPEC}
COLOR = {x[0]: x[1] for x in SPEC}
