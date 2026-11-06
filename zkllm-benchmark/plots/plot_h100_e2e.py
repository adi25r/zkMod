#!/usr/bin/env python3
"""zkLLM vs zkMod on one H100: end-to-end and prover-time improvement, from the latest sweeps.

  python3 plots/plot_h100_e2e.py [full_sweep_summary.log] [splits_sweep_summary.log]

Defaults are the two latest H100 result directories; without them it replots data/h100_e2e.csv. Writes
  plots/h100_e2e.png/.pdf   end-to-end time (left) and prover time (right), zkLLM vs zkMod
  plots/data/h100_e2e.csv   the numbers behind it, every measured configuration (table view)

Definitions (the same ones used in the sweep write-ups):
  e2e     the driver's own e2e_s: baseline = its single partition; zkMod = witness-generation chain + provers
          overlapped (pipelined), measured from pipeline start to the last prover finishing
  prover  each partition's prover_total_s (input/output commit, proving, openings; the prover's own recomputed
          forward pass and process start-up excluded). Baseline = its single partition; zkMod = the SLOWEST
          partition, i.e. what you wait for if every partition input were ready at once
zkMod uses, per model, the pipelined split count with the fastest end-to-end time among those measured; the prover row
shows the prover time at that same split count (not each metric's own best: prover time alone keeps falling up to 32 splits).
"""
import csv
import glob
import os
import re
import sys

import matplotlib.pyplot as plt

import paperstyle as ps

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

MODELS = ["125m", "1.3b", "2.7b", "6.7b", "13b"]

full_log = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results-h100-sqrt-op-23112589", "summary.log")
splits_log = sys.argv[2] if len(sys.argv) > 2 else (sorted(glob.glob(os.path.join(ROOT, "results-h100-pipesplits-*", "summary.log")))[-1:] or [None])[0]


def parse(path):
    """summary.log -> {(model, tag): dict(ok, e2e, makespan, pmax)}"""
    out = {}
    if not path or not os.path.exists(path):
        return out
    for line in open(path):
        m = re.match(r"RESULT model=opt-(\S+) tag=(\S+) rc=(\d+) ", line)
        if not m:
            continue
        f = {k: float(v) for k, v in re.findall(r"(?<![A-Za-z_])(makespan_s|prover_max_s|e2e_s)=([\d.]+)", line)}
        st = re.search(r"status=(\w+)", line)
        out[(m.group(1), m.group(2))] = dict(ok=(m.group(3) == "0" and bool(st) and st.group(1) == "OK"),
                                             e2e=f.get("e2e_s"), makespan=f.get("makespan_s"), pmax=f.get("prover_max_s"))
    return out


first = parse(full_log)
data = dict(first)
data.update({k: v for k, v in parse(splits_log).items() if v["ok"]})     # the splits sweep adds 12/16/24/32
for k, v in first.items():                                              # keep the first sweep's 8-split numbers (same session as the baseline)
    if k[1] == "zkmod8_pipe" and v["ok"]:
        data[k] = v

PIPE_TAGS = ["zkmod2_pipe", "zkmod4_pipe", "zkmod8_pipe", "zkmod12_pipe", "zkmod16_pipe", "zkmod24_pipe", "zkmod32_pipe"]
SEQ_TAGS = ["zkmod2_seq", "zkmod4_seq"]
splits = lambda t: int(re.match(r"zkmod(\d+)_", t).group(1))


def table(metric, tag_by_model=None):
    """metric 'e2e': zkMod = the pipelined split count with the fastest end-to-end time, per model.
    metric 'prover': zkMod = the slowest partition's prover time at tag_by_model."""
    rows = []
    for m in MODELS:
        base = data.get((m, "zkllm"))
        if not base or not base["ok"]:
            continue
        b = base["e2e"] if metric == "e2e" else base["makespan"]
        cfg = {}
        for t in PIPE_TAGS + (SEQ_TAGS if metric == "e2e" else []):
            r = data.get((m, t))
            if r and r["ok"]:
                cfg[t] = r["e2e"] if metric == "e2e" else r["pmax"]
        pipe = {t: v for t, v in cfg.items() if t in PIPE_TAGS and v is not None}
        best_tag = min(pipe, key=pipe.get) if tag_by_model is None else tag_by_model[m]
        rows.append(dict(model=m, base=b, cfg=cfg, best_tag=best_tag, best_splits=splits(best_tag), best=pipe[best_tag],
                         saved=b - pipe[best_tag], speedup=b / pipe[best_tag]))
    return rows


def write_csv(tables, path):
    e2e, prov = tables["e2e"], tables["prover"]
    cols_e = SEQ_TAGS + PIPE_TAGS
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "e2e_zkllm_s"] + [f"e2e_{c}_s" for c in cols_e]
                   + ["e2e_best_splits", "e2e_best_s", "e2e_improvement_s", "e2e_speedup",
                      "prover_zkllm_s"] + [f"prover_{c}_s" for c in PIPE_TAGS]
                   + ["prover_best_splits", "prover_best_s", "prover_improvement_s", "prover_speedup"])
        for re_, rp in zip(e2e, prov):
            w.writerow([f"opt-{re_['model']}", f"{re_['base']:.1f}"] + [f"{re_['cfg'][c]:.1f}" if c in re_["cfg"] else "" for c in cols_e]
                       + [re_["best_splits"], f"{re_['best']:.1f}", f"{re_['saved']:.1f}", f"{re_['speedup']:.2f}",
                          f"{rp['base']:.1f}"] + [f"{rp['cfg'][c]:.1f}" if c in rp["cfg"] else "" for c in PIPE_TAGS]
                       + [rp["best_splits"], f"{rp['best']:.1f}", f"{rp['saved']:.1f}", f"{rp['speedup']:.2f}"])


def plot(tables, path):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
    for ax, (metric, ylabel) in zip(axes, (("e2e", "End-to-end time (s)"), ("prover", "Prover time (s)"))):
        rows = tables[metric]
        ps.pair_bars(ax, [ps.name(r["model"], True) for r in rows], [r["base"] for r in rows], [r["best"] for r in rows],
                     "zkLLM", "zkMod", ylabel)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    axes[0].legend(loc="upper left")
    fig.tight_layout(w_pad=0.8)
    ps.save(fig, path)


def from_csv(path):
    """The tables back from data/h100_e2e.csv, for when the raw sweep logs are not on this machine."""
    tables = {"e2e": [], "prover": []}
    for r in csv.DictReader(open(path)):
        for metric in tables:
            b, best = float(r[f"{metric}_zkllm_s"]), float(r[f"{metric}_best_s"])
            tables[metric].append(dict(model=r["model"].removeprefix("opt-"), base=b, best=best, best_splits=int(r[f"{metric}_best_splits"]),
                                       saved=b - best, speedup=b / best))
    return tables


CSV = os.path.join(HERE, "data", "h100_e2e.csv")
if data:
    e2e_rows = table("e2e")
    tables = {"e2e": e2e_rows, "prover": table("prover", {r["model"]: r["best_tag"] for r in e2e_rows})}
    write_csv(tables, CSV)
else:
    print(f"no sweep logs found; plotting {CSV}")
    tables = from_csv(CSV)
plot(tables, os.path.join(HERE, "h100_e2e"))
for metric, rows in tables.items():
    print(metric, [(r["model"], round(r["base"], 1), r["best_splits"], round(r["best"], 1), round(r["saved"], 1), round(r["speedup"], 2)) for r in rows])
