#!/usr/bin/env python3
"""zkLLM vs zkMod on one H100: peak GPU memory and end-to-end time per model (layout of plots/blendy_tiling.png).

  python3 plots/plot_h100_memory.py [full_sweep_summary.log] [splits_sweep_summary.log]

Inputs: the latest full H100 sweep (all five models: zkLLM and zkMod pipelined at 2, 4 and 8 splits) and the splits sweep
(opt-2.7b, 6.7b and 13b at 12, 16, 24 and 32 splits). Writes plots/h100_memory.png/.pdf and plots/data/h100_memory.csv;
without those logs it replots plots/data/h100_memory.csv. Legend 'zkMod N' = zkMod pipelined, N splits.

  peak memory   the highest nvidia-smi `memory.used` sample (whole GPU, MiB / 1024) over the run's proving command;
                for the pipelined configs that is every concurrently running prover plus the witness-generation chain
  time          the driver's own end-to-end e2e_s (same quantity as plots/h100_e2e.png's top chart)
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

# zkLLM hatched; the zkMod split counts are an ordered family, so one grey ramp light -> dark
SPLITS = [2, 4, 8, 12, 16, 24, 32]
RAMP = ps.ramp(len(SPLITS))
CONFIGS = [("zkllm", "zkLLM", ps.BASE)] + [(f"zkmod{p}_pipe", f"zkMod {p}", RAMP[i]) for i, p in enumerate(SPLITS)]
MODELS = ["125m", "1.3b", "2.7b", "6.7b", "13b"]
H100_GIB = 80

full_log = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results-h100-sqrt-op-23112589", "summary.log")
splits_log = sys.argv[2] if len(sys.argv) > 2 else (sorted(glob.glob(os.path.join(ROOT, "results-h100-pipesplits-*", "summary.log")))[-1:] or [None])[0]


def parse(path):
    out = {}
    if not path or not os.path.exists(path):
        return out
    for line in open(path):
        m = re.match(r"RESULT model=opt-(\S+) tag=(\S+) rc=(\d+) .*?peak_mem_MiB=(\d+) .*?e2e_s=([\d.]+).*?status=(\w+)", line)
        if m and m.group(3) == "0" and m.group(6) == "OK":
            out[(m.group(1), m.group(2))] = (int(m.group(4)) / 1024.0, float(m.group(5)))
    return out


CSV = os.path.join(HERE, "data", "h100_memory.csv")
first = parse(full_log)
res = dict(first)
res.update(parse(splits_log))
res.update({k: v for k, v in first.items() if k[1] == "zkmod8_pipe"})   # 8 splits: keep the full sweep's (same session as zkLLM)

if res:
    with open(CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "config", "peak_gpu_gib", "e2e_s"])
        for m in MODELS:
            for tag, _, _ in CONFIGS:
                if (m, tag) in res:
                    w.writerow([f"opt-{m}", tag, f"{res[(m, tag)][0]:.1f}", f"{res[(m, tag)][1]:.1f}"])
else:                                                                     # raw sweep logs not on this machine
    print(f"no sweep logs found; plotting {CSV}")
    res = {(r["model"].removeprefix("opt-"), r["config"]): (float(r["peak_gpu_gib"]), float(r["e2e_s"]))
           for r in csv.DictReader(open(CSV))}

fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.0, 3.2), sharex=True)
w = 0.105
for xi, m in enumerate(MODELS):
    have = [(tag, shade) for tag, _, shade in CONFIGS if (m, tag) in res]
    for j, (tag, shade) in enumerate(have):
        x = xi + (j - (len(have) - 1) / 2) * w                         # bars of a model are centred; 125m and 1.3b have four
        mem, tim = res[(m, tag)]
        ps.bar(a1, x, mem, w, shade)
        ps.bar(a2, x, tim, w, shade)
        if tag != "zkllm" and (m, "zkllm") in res:
            ps.speedup(a2, x, tim, res[(m, "zkllm")][1] / tim, rotation=90, size=6)
a1.axhline(H100_GIB, color="black", ls="--", lw=0.8)
a1.set_ylim(0, 100)
a1.set_ylabel("GPU mem. (GiB)")
a2.set_ylabel("E2E time (s)")
a2.set_ylim(0, a2.get_ylim()[1] * 1.3)
for a in (a1, a2):
    a.set_xticks(range(len(MODELS)))
    a.set_xticklabels([ps.name(m, True) for m in MODELS])
    a.set_xlim(-0.55, len(MODELS) - 0.45)
    ps.grid(a)
from matplotlib.lines import Line2D
a1.legend(handles=[Line2D([], [], color="black", ls="--", lw=0.8, label="H100 memory (80 GiB)")], loc="upper left")
handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, hatch=h, **ps.BAR) for _, _, (c, h) in CONFIGS]
a2.legend(handles, [l for _, l, _ in CONFIGS], loc="upper left", ncol=len(CONFIGS))
fig.tight_layout(h_pad=0.6)
ps.save(fig, os.path.join(HERE, "h100_memory"))
print({f"{m}/{t.replace('zkmod','').replace('_pipe','')}": (round(res[(m, t)][0], 1), round(res[(m, t)][1], 1)) for m in MODELS for t, _, _ in CONFIGS if (m, t) in res})
