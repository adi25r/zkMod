#!/usr/bin/env python3
"""zkLLM (witness-first: all witgen into host memory, then all proving) vs zkMod on one H100, end-to-end and prover time.

  python3 plots/plot_h100_e2e_wf.py [seq_len]      # default 2048

Same layout as plot_h100_e2e.py, with the zkLLM baseline replaced by witness-first zkLLM (opt-partition.py
--witgen-first, run_witgen_first.sh). Writes
  plots/h100_e2e_wf.png/.pdf    end-to-end time (left) and prover time (right), zkLLM vs zkMod
  plots/data/h100_e2e_wf.csv   the numbers behind it
(other sequence lengths write h100_e2e_wf_<seq>.png/.pdf / .csv)

  zkLLM  mean over the witness-first runs (rc=0, status=OK) in every results-witgen-first-*/summary.log, per model:
         e2e = e2e_s (record pass + proving), prover = prover_total_s (seq_sum_s; the record pass excluded)
  zkMod  seq 2048: plots/data/h100_e2e.csv (written by plot_h100_e2e.py); other lengths: results-h100-e2e-<seq>-*/summary.log
         (run_e2e.sh). Per model, the pipelined split count with the fastest end-to-end time, and the slowest
         partition's prover time at that split count
"""
import csv
import glob
import os
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt

import paperstyle as ps

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

MODELS = ["125m", "1.3b", "2.7b", "6.7b", "13b"]
mean = lambda v: sum(v) / len(v)
SEQ = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
SUF = "" if SEQ == 2048 else f"_{SEQ}"


def summary_rows(pattern):
    for log in sorted(glob.glob(os.path.join(ROOT, pattern, "summary.log"))):
        for line in open(log):
            r = dict(re.findall(r"(\w+)=(\S+)", line))
            if line.startswith("RESULT") and r.get("rc") == "0" and r.get("status") == "OK":
                yield r


def load_wf():
    acc, smoke = defaultdict(list), defaultdict(list)
    for r in summary_rows(f"results-witgen-first-{SEQ}-*"):
        if r.get("witgen_first") == "True":
            (smoke if r["tag"].startswith("smoke_") else acc)[r["model"].removeprefix("opt-")].append(r)
    for m, rs in smoke.items():
        acc.setdefault(m, rs)
    return {m: dict(e2e=mean([float(r["e2e_s"]) for r in rs]), prover=mean([float(r["seq_sum_s"]) for r in rs]),
                    store=mean([float(r["wf_store_s"]) for r in rs]), peak_gb=mean([float(r["wf_peak_gb"]) for r in rs]), n=len(rs))
            for m, rs in acc.items()}


def load_zkmod():
    if SEQ != 2048:
        out = {}
        for r in summary_rows(f"results-h100-e2e-{SEQ}-*"):
            m, tag = r["model"].removeprefix("opt-"), r["tag"]
            d = out.setdefault(m, dict(splits=0, e2e=float("inf"), prover=0.0, old_e2e=0.0))
            if tag == "zkllm":
                d["old_e2e"] = float(r["e2e_s"])
            elif tag.endswith("_pipe") and float(r["e2e_s"]) < d["e2e"]:
                d.update(splits=int(r["parts"]), e2e=float(r["e2e_s"]), prover=float(r["prover_max_s"]))
        return {m: d for m, d in out.items() if d["splits"]}
    out = {}
    for r in csv.DictReader(open(os.path.join(HERE, "data", "h100_e2e.csv"))):
        out[r["model"].removeprefix("opt-")] = dict(splits=int(r["e2e_best_splits"]), e2e=float(r["e2e_best_s"]),
                                                    prover=float(r["prover_best_s"]), old_e2e=float(r["e2e_zkllm_s"]))
    return out


def main():
    wf, zk = load_wf(), load_zkmod()
    rows = [dict(model=m, wf=wf[m], zk=zk[m]) for m in MODELS if m in wf and m in zk]
    missing = [m for m in MODELS if m not in wf]
    if missing:
        print("no witness-first runs yet for:", " ".join(missing))

    with open(os.path.join(HERE, "data", f"h100_e2e_wf{SUF}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "n_wf", "e2e_zkllm_wf_s", "e2e_zkllm_normal_s", "wf_store_s", "wf_peak_host_gb", "zkmod_splits",
                    "e2e_zkmod_s", "e2e_speedup", "prover_zkllm_wf_s", "prover_zkmod_s", "prover_speedup"])
        for r in rows:
            a, z = r["wf"], r["zk"]
            w.writerow([f"opt-{r['model']}", a["n"], f"{a['e2e']:.1f}", f"{z['old_e2e']:.1f}", f"{a['store']:.1f}",
                        f"{a['peak_gb']:.1f}", z["splits"], f"{z['e2e']:.1f}", f"{a['e2e'] / z['e2e']:.2f}",
                        f"{a['prover']:.1f}", f"{z['prover']:.1f}", f"{a['prover'] / z['prover']:.2f}"])

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
    for ax, (key, ylabel) in zip(axes, (("e2e", "End-to-end time (s)"), ("prover", "Prover time (s)"))):
        ps.pair_bars(ax, [ps.name(r["model"], True) for r in rows], [r["wf"][key] for r in rows],
                     [r["zk"][key] for r in rows], "zkLLM (witness-first)", "zkMod", ylabel)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    axes[0].legend(loc="upper left")
    fig.tight_layout(w_pad=0.8)
    ps.save(fig, os.path.join(HERE, f"h100_e2e_wf{SUF}"))
    for r in rows:
        print(r["model"], round(r["wf"]["e2e"], 1), r["zk"]["splits"], r["zk"]["e2e"], round(r["wf"]["e2e"] / r["zk"]["e2e"], 2))


if __name__ == "__main__":
    main()
