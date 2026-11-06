#!/usr/bin/env python3
"""Witness-first zkLLM (witgen of every layer into host memory, then proving from it) against the monolithic proof
that recomputes as it proves, and the best pipelined zkMod run (early carry), split into where the time goes.

  python3 plots/plot_witgen_first.py [witgen_first_results_dir]

Defaults are the run_witgen_first.sh run (job 6097775) and, for zkMod, the run_early_carry_ab.sh runs
(6057959, 6057960), all on one H100. Writes
  plots/witgen_first_e2e.png/.pdf   one panel per model, three bars: zkLLM normal, zkLLM witness-first, zkMod
  plots/data/witgen_first_e2e.csv   the numbers behind it (table view)

Segments add up to each bar's end-to-end time (mean over runs with rc=0 and status=OK):
  normal         witness generation = the prover's own forward pass (witgen_s); proof generation = the rest of e2e_s
  witness-first  witness generation = the record pass's compute (wf_witgen_s); copy to host RAM = wf_store_s;
                 load to GPU = wf_load_s; proof generation = the rest (proving, both passes' weight loading, commits)
  zkMod          two side-by-side lanes from t=0, since they run concurrently: the witgen chain (0 to witgen_chain_s)
                 and proving (first prover launch, ~0.1 s, to e2e_s); the bar's height is e2e_s
"""
import csv
import json
import os
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt

import paperstyle as ps
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WF_DIR = "results-witgen-first-2048-6097775"
ZKMOD_DIRS = {"opt-2.7b": "results-early-carry-ab-opt-2.7b-2048-sqrt-6057959",
              "opt-13b": "results-early-carry-ab-opt-13b-2048-sqrt-6057960"}

# Blues light -> dark: transfer to host, witgen, load to GPU, zkLLM prove, zkMod prove.
STORE, LOAD = (ps.BLUES[0], ""), (ps.BLUES[2], "")
ZKMOD = (ps.BLUES[4], "")

mean = lambda v: sum(v) / len(v)


def rows_of(d):
    return [dict(re.findall(r"(\w+)=(\S+)", l)) for l in open(os.path.join(d, "summary.log"))
            if l.startswith("RESULT") and "rc=0" in l and "status=OK" in l]


def load_wf(d):
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows_of(d):
        m = re.match(r"(.+)_(mono|wf)_r\d+$", r["tag"])
        if not m or r["tag"].startswith("smoke_"):
            continue
        a = acc[r["model"]][m.group(2)]
        a.append(r)
        if m.group(2) == "mono":
            r["witgen_s"] = json.load(open(os.path.join(d, r["tag"] + ".json")))["partitions"][0]["witgen_s"]
    out = {}
    for model, arms in acc.items():
        f = lambda arm, k: mean([float(r[k]) for r in arms[arm]])
        out[model] = {"mono_e2e_s": f("mono", "e2e_s"), "mono_witgen_s": f("mono", "witgen_s"),
                      "wf_e2e_s": f("wf", "e2e_s"), "wf_witgen_s": f("wf", "wf_witgen_s"),
                      "wf_store_s": f("wf", "wf_store_s"), "wf_load_s": f("wf", "wf_load_s"),
                      "wf_peak_gb": f("wf", "wf_peak_gb"), "n": min(len(arms["mono"]), len(arms["wf"]))}
    return out


def load_zkmod(d):
    # Best partition count with the early carry on, by mean e2e.
    e2e, wg = defaultdict(list), defaultdict(list)
    for r in rows_of(d):
        m = re.match(r"pipe(\d+)_carry1_r\d+$", r["tag"])
        if m:
            e2e[int(m.group(1))].append(float(r["e2e_s"]))
            wg[int(m.group(1))].append(float(r["witgen_chain_s"]))
    p = min(e2e, key=lambda k: mean(e2e[k]))
    return {"zkmod_partitions": p, "zkmod_e2e_s": mean(e2e[p]), "zkmod_witgen_s": mean(wg[p]), "zkmod_n": len(e2e[p])}


def main():
    wf_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, WF_DIR)
    data = load_wf(wf_dir)
    models = [m for m in ZKMOD_DIRS if m in data]
    for m in models:
        data[m].update(load_zkmod(os.path.join(ROOT, ZKMOD_DIRS[m])))
        d = data[m]
        d["speedup_vs_mono"] = d["mono_e2e_s"] / d["zkmod_e2e_s"]
        d["speedup_vs_wf"] = d["wf_e2e_s"] / d["zkmod_e2e_s"]

    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    with open(os.path.join(HERE, "data", "witgen_first_e2e.csv"), "w", newline="") as f:
        fields = ["model"] + list(data[models[0]])
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for m in models:
            w.writerow({"model": m, **{k: round(v, 3) if isinstance(v, float) else v for k, v in data[m].items()}})

    fig, axes = plt.subplots(1, len(models), figsize=(3.5 * len(models), 1.7),
                             gridspec_kw={"width_ratios": [4.15] + [3.45] * (len(models) - 1)})
    axes = axes if len(models) > 1 else [axes]
    W = 0.6
    for ax, m in zip(axes, models):
        d = data[m]

        def stack(x, segs):
            bottom = 0
            for h, shade in segs:
                ps.bar(ax, x, h, W, shade, bottom=bottom)
                bottom += h

        stack(0, [(d["mono_witgen_s"], ps.WG_FILL), (d["mono_e2e_s"] - d["mono_witgen_s"], ps.PV_FILL)])
        wf_rest = d["wf_e2e_s"] - d["wf_witgen_s"] - d["wf_store_s"] - d["wf_load_s"]
        stack(1, [(d["wf_witgen_s"], ps.WG_FILL), (d["wf_store_s"], STORE), (d["wf_load_s"], LOAD), (wf_rest, ps.PV_FILL)])
        # zkMod: witgen chain and provers run at the same time, so they are two lanes from 0, not a stack.
        lane = W / 2 - 0.02
        ps.bar(ax, 2 - W / 4 - 0.01, d["zkmod_witgen_s"], lane, ps.WG_FILL)
        ps.bar(ax, 2 + W / 4 + 0.01, d["zkmod_e2e_s"], lane, ZKMOD)
        ax.annotate(f"{d['speedup_vs_mono']:.2f}\u00d7 vs zkLLM\n{d['speedup_vs_wf']:.2f}\u00d7 vs witness-first", (2, d["zkmod_e2e_s"]),
                    xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=6)
        ax.set_xticks([0, 1, 2], ["zkLLM", "zkLLM\nwitness-first", "zkMod"])
        ax.set_xlim(-0.6, 3.55 if ax is axes[0] else 2.85)   # left panel: an empty slot for the legend
        ax.set_ylim(0, d["wf_e2e_s"] * 1.08)
        ax.set_xlabel(ps.name(m))
        ps.grid(ax)
    axes[0].set_ylabel("End-to-end time (s)")
    handles = [Patch(facecolor=c, hatch=h, label=l, **ps.BAR) for (c, h), l in
               ((ps.WG_FILL, "Witgen"), (STORE, "Copy to host"), (LOAD, "Load to GPU"),
                (ps.PV_FILL, "zkLLM prove"), (ZKMOD, "zkMod prove"))]
    axes[0].legend(handles=handles, loc="upper right", ncol=1)
    fig.tight_layout(w_pad=0.8)
    ps.save(fig, os.path.join(HERE, "witgen_first_e2e"))

if __name__ == "__main__":
    main()
