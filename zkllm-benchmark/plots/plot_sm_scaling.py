#!/usr/bin/env python3
"""zkLLM baseline (monolithic, layer prover) on one H100 with its SMs capped by CUDA MPS (run_sm_scaling.sh).

  python3 plots/plot_sm_scaling.py [--seq N] [--bars] [results_dir ...]      # default: every results-sm-scaling-*

Writes (with --seq N: only that sequence length, to sm_scaling_N.png/.pdf / sm_scaling_N.csv)
  plots/sm_scaling.png/.pdf      end-to-end time vs SMs
  plots/data/sm_scaling.csv      every run, plus the fit and the SM count where doubling stops paying 10%

--bars draws one row per model/seq instead, an end-to-end time bar per SM count, to sm_scaling[_N]_bars.png/.pdf.

Times are the mean over reps with rc=0 and status=OK; error bars (--bars) are +/-1 sample std dev of e2e. SMs = the count the device reported to the capped process.
"""
import csv
import glob
import os
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

import paperstyle as ps

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

MODELS = ["125m", "1.3b", "2.7b", "6.7b", "13b"]
DOUBLING_GAIN = 0.10


def load(dirs):
    acc = defaultdict(list)
    for d in dirs:
        for line in open(os.path.join(d, "summary.log")):
            r = dict(re.findall(r"(\w+)=(\S+)", line))
            if line.startswith("RESULT") and r.get("rc") == "0" and r.get("status") == "OK":
                acc[(r["model"].removeprefix("opt-"), int(r["seq"]), int(r["sms_seen"]))].append(
                    (float(r["e2e_s"]), float(r["makespan_s"])))
    out = defaultdict(dict)
    for (m, seq, sms), v in acc.items():
        es = [e for e, _ in v]
        mu = sum(es) / len(es)
        sd = (sum((e - mu) ** 2 for e in es) / (len(es) - 1)) ** 0.5 if len(es) > 1 else 0.0
        out[(m, seq)][sms] = (mu, sum(p for _, p in v) / len(v), len(v), sd)
    return out


def amdahl(pts):
    """Least squares t = a + b/s."""
    xs = [1.0 / s for s, _ in pts]
    ys = [t for _, t in pts]
    n, mx, my = len(xs), sum(xs) / len(xs), sum(ys) / len(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


def knee(a, b):
    """Smallest s where going s -> 2s cuts t by less than DOUBLING_GAIN: (b/2s) / (a + b/s) < g."""
    return b * (1 - 2 * DOUBLING_GAIN) / (2 * DOUBLING_GAIN * a) if a > 0 else float("inf")


def plot_bars(rows, seq_only, suffix):
    """One row per series: an end-to-end time bar per SM count."""
    sms_all = sorted({s for r in rows for s, _ in r["pts"]})
    n = len(sms_all)
    fig, axes = plt.subplots(len(rows), 1, figsize=(7.0, 0.3 + 1.0 * len(rows)), squeeze=False, sharex=True)
    for ax, r in zip(axes[:, 0], rows):
        raw = dict(r["raw"])
        x = np.arange(n)
        for j, s in enumerate(sms_all):
            if s not in raw:
                continue
            e2e, pv, nrep, sd = raw[s]
            ps.bar(ax, j, e2e, 0.8, ps.PV_FILL)
            if nrep > 1:
                ax.errorbar(j, e2e, yerr=sd, color="black", lw=0.5, capsize=1.2, capthick=0.5)
        ax.set_xticks(x); ax.set_xticklabels([str(s) for s in sms_all], fontsize=7)
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.6, n - 0.4)
        ax.text(0.995, 0.95, ps.name(r["model"]) + ("" if seq_only else f", seq {r['seq']}"), transform=ax.transAxes,
                ha="right", va="top", fontsize=7, bbox=dict(facecolor="white", edgecolor="black", lw=0.5, pad=1.5))
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
        ps.grid(ax)
    axes[-1, 0].set_xlabel("SMs")
    fig.supylabel("End-to-end time (s)", fontsize=8, fontweight="bold")
    fig.tight_layout(h_pad=0.3)
    ps.save(fig, os.path.join(HERE, f"sm_scaling{suffix}_bars"))


def main():
    args = sys.argv[1:]
    seq_only = None
    if "--seq" in args:
        i = args.index("--seq")
        seq_only = int(args[i + 1])
        del args[i:i + 2]
    bars = "--bars" in args
    if bars:
        args.remove("--bars")
    suffix = f"_{seq_only}" if seq_only else ""
    dirs = args or sorted(glob.glob(os.path.join(ROOT, "results-sm-scaling-*")))
    data = load(dirs)
    if seq_only:
        data = {k: v for k, v in data.items() if k[1] == seq_only}
    series = sorted(data, key=lambda k: (MODELS.index(k[0]) if k[0] in MODELS else 99, k[1]))

    rows = []
    for m, seq in series:
        pts = sorted((s, v[0]) for s, v in data[(m, seq)].items())
        a, b = amdahl(pts)
        rows.append(dict(model=m, seq=seq, pts=pts, a=a, b=b, knee=knee(a, b),
                         raw=sorted(data[(m, seq)].items())))

    with open(os.path.join(HERE, "data", f"sm_scaling{suffix}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "seq_len", "sms", "n", "e2e_s", "e2e_std_s", "prove_s", "speedup_vs_min_sms",
                    "fit_sm_independent_s", "fit_sm_work_s_at_1sm", "knee_sms_10pct"])
        for r in rows:
            t0 = r["pts"][0][1]
            for s, (e, p, n, sd) in r["raw"]:
                w.writerow([f"opt-{r['model']}", r["seq"], s, n, f"{e:.1f}", f"{sd:.2f}", f"{p:.1f}", f"{t0 / e:.2f}",
                            f"{r['a']:.1f}", f"{r['b']:.0f}", f"{r['knee']:.0f}"])

    if bars:
        plot_bars(rows, seq_only, suffix)
        return

    fig, ax1 = plt.subplots(figsize=(3.4, 1.6))
    sms_all = sorted({s for r in rows for s, _ in r["pts"]})
    for i, r in enumerate(rows):
        lab = ps.name(r["model"]) + ("" if seq_only else f", seq {r['seq']}")
        xs, ys = zip(*r["pts"])
        ax1.plot(xs, ys, ps.MARKS[i % len(ps.MARKS)] + "-", color="black", mfc="white", label=lab)
    ax1.set_xscale("log", base=2)
    ticks = [t for t in sms_all if t in (sms_all[0], 16, 32, 64, sms_all[-1])]
    ax1.set_xticks(ticks)
    ax1.set_xticklabels([str(t) for t in ticks])
    ax1.minorticks_off()
    ax1.set_xlabel("SMs")
    ps.grid(ax1, "both")
    ax1.set_ylim(0, ax1.get_ylim()[1] * 1.05)
    ax1.set_ylabel("End-to-end time (s)")
    ax1.legend(loc="upper right")
    fig.tight_layout()
    ps.save(fig, os.path.join(HERE, f"sm_scaling{suffix}"))
    for r in rows:
        print(f"opt-{r['model']} seq {r['seq']}: t = {r['a']:.1f} + {r['b']:.0f}/SMs  "
              f"(SM-bound share at 132: {r['b'] / 132 / (r['a'] + r['b'] / 132):.0%}, 10%-per-doubling knee ~{r['knee']:.0f} SMs)")


if __name__ == "__main__":
    main()
