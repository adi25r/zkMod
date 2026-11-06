#!/usr/bin/env python3
"""Early witgen carry: end-to-end time of pipelined zkMod vs the monolithic zkLLM proof, carry off vs on, split into
witness generation and proof generation.

  python3 plots/plot_early_carry.py [results_dir ...]

Defaults are the run_early_carry_ab.sh batch runs on one H100 (opt-2.7b, opt-13b). Writes
  plots/early_carry_e2e.png/.pdf   one panel per model: end-to-end time of the monolithic proof and, per partition
                                   count, the per-partition witgen chain (carry off) and one early-carry witgen
                                   process (carry on), stacked as witness generation + proof generation
  plots/data/early_carry_e2e.csv   the numbers behind it (table view)

speedup = mean e2e_s of the monolithic runs (mono_r*, --splits 1 on the layer path) / mean e2e_s of the pipelined runs
at that partition count. Only runs with rc=0 and status=OK count.

The split follows the critical path, so the two segments of every bar add up to its end-to-end time:
  monolithic  witness generation = the prover's own forward pass (its PARTITION witgen_s); proof generation = the rest
              of e2e_s (proving, plus the per-layer weight loading the prover interleaves with it)
  pipelined   witness generation = the witgen chain (witgen_chain_s), which runs while early provers already prove;
              proof generation = the proving still left once the chain has finished (e2e_s - witgen_chain_s)
"""
import json
import csv
import os
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt

import paperstyle as ps

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_DIRS = ["results-early-carry-ab-opt-2.7b-2048-sqrt-6057959", "results-early-carry-ab-opt-13b-2048-sqrt-6057960"]

# Arm = blue level of the proof-generation segment; witness generation is the light segment under it in every arm.
MONO, OFF, ON = (ps.BLUES[2], ""), (ps.BLUES[3], ""), (ps.BLUES[4], "")


def load(results_dir):
    rows = [dict(re.findall(r"(\w+)=(\S+)", l)) for l in open(os.path.join(results_dir, "summary.log"))
            if l.startswith("RESULT")]
    mono, mono_wg, e2e, wg, model = [], [], defaultdict(list), defaultdict(list), None
    for r in rows:
        if r.get("rc") != "0" or r.get("status") != "OK":
            continue
        model = r["model"]
        if r["tag"].startswith("mono_"):
            mono.append(float(r["e2e_s"]))
            run = json.load(open(os.path.join(results_dir, r["tag"] + ".json")))
            mono_wg.append(float(run["partitions"][0]["witgen_s"]))
            continue
        m = re.match(r"pipe(\d+)_carry(\d)_r\d+$", r["tag"])
        if m:
            key = (int(m.group(1)), int(m.group(2)))
            e2e[key].append(float(r["e2e_s"]))
            wg[key].append(float(r["witgen_chain_s"]))
    mean = lambda v: sum(v) / len(v)
    base, base_wg = mean(mono), mean(mono_wg)
    parts = sorted({p for p, _ in e2e})
    out = []
    for p in parts:
        row = {"model": model, "partitions": p, "mono_e2e_s": round(base, 2), "mono_witgen_s": round(base_wg, 2),
               "n_mono": len(mono)}
        for c, name in ((0, "off"), (1, "on")):
            row[f"e2e_{name}_s"] = round(mean(e2e[(p, c)]), 2)
            row[f"witgen_{name}_s"] = round(mean(wg[(p, c)]), 2)
            row[f"speedup_{name}"] = round(base / mean(e2e[(p, c)]), 3)
            row[f"n_{name}"] = len(e2e[(p, c)])
        out.append(row)
    return model, base, base_wg, out


def main():
    dirs = sys.argv[1:] or [os.path.join(ROOT, d) for d in DEFAULT_DIRS]
    data = [load(d) for d in dirs]

    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    with open(os.path.join(HERE, "data", "early_carry_e2e.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0][3][0]))
        w.writeheader()
        for *_, rows in data:
            w.writerows(rows)

    fig, axes = plt.subplots(1, len(data), figsize=(3.5 * len(data), 1.7),
                             gridspec_kw={"width_ratios": [len(rows) + 1 for *_, rows in data]})
    axes = axes if len(data) > 1 else [axes]
    W, GAP = 0.36, 0.04   # bar width; gap between the off/on pair
    for ax, (model, base, base_wg, rows) in zip(axes, data):
        def bar(x, total, wg, shade, sp=None):
            ps.bar(ax, x, wg, W, ps.WG_FILL)
            ps.bar(ax, x, total - wg, W, shade, bottom=wg)
            if sp:
                ps.speedup(ax, x, total, sp, rotation=90)

        bar(0, base, base_wg, MONO)
        ticks, names = [0], ["zkLLM"]
        for i, r in enumerate(rows, start=1):
            bar(i - W / 2 - GAP / 2, r["e2e_off_s"], r["witgen_off_s"], OFF, r["speedup_off"])
            bar(i + W / 2 + GAP / 2, r["e2e_on_s"], r["witgen_on_s"], ON, r["speedup_on"])
            ticks.append(i)
            names.append(str(r["partitions"]))
        ax.set_xticks(ticks, names)
        ax.set_xlim(-0.6, len(rows) + 0.6)
        # rotated speedups need ~30% of the axis above the tallest zkMod bar; the legend panel also ~30% above those
        zk = max(max(r["e2e_off_s"], r["e2e_on_s"]) for r in rows)
        ax.set_ylim(0, max(base * 1.08, zk / (0.40 if ax is axes[0] else 0.66)))
        ax.set_xlabel(f"Partitions ({ps.name(model)})")
        ps.grid(ax)
    axes[0].set_ylabel("End-to-end time (s)")
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, hatch=h, label=l, **ps.BAR) for (c, h), l in
               ((ps.WG_FILL, "Witgen"), (MONO, "zkLLM prove"), (OFF, "zkMod prove"), (ON, "zkMod + early carry prove"))]
    axes[0].legend(handles=handles, loc="upper right", ncol=2)
    fig.tight_layout(w_pad=0.8)
    ps.save(fig, os.path.join(HERE, "early_carry_e2e"))

if __name__ == "__main__":
    main()
