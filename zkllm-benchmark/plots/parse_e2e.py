#!/usr/bin/env python3
"""run_experiments.sh summary.log -> the e2e CSV plot_memory.py plots (same columns as h100_e2e.csv).

  python3 plots/parse_e2e.py results-4090-wstream/summary.log > plots/data/4090_e2e.csv

Times are the harness's wall_s. A config that failed (rc != 0) is left empty. peak_gb is the
highest nvidia-smi sample (MiB / 1024) over the model's successful configs.
"""
import csv, re, sys

TAGS = ["zkllm", "zkmod2_seq", "zkmod4_seq", "zkmod2_pipe", "zkmod4_pipe", "zkmod8_pipe"]
res, peak, order = {}, {}, []
for line in open(sys.argv[1]):
    m = re.match(r"RESULT model=opt-(\S+) tag=(\S+) rc=(\d+) wall_s=(\S+) peak_mem_MiB=(\d+)", line)
    if not m:
        continue
    model, tag, rc, wall, mib = m[1], m[2], int(m[3]), float(m[4]), int(m[5])
    if model not in order:
        order.append(model)
    if rc == 0:
        res[(model, tag)] = wall
        peak[model] = max(peak.get(model, 0), mib)
w = csv.writer(sys.stdout)
w.writerow(["model"] + ["zkllm"] + TAGS[1:] + ["best_speedup", "peak_gb"])
for model in order:
    base = res.get((model, "zkllm"))
    zk = [res[(model, t)] for t in TAGS[1:] if (model, t) in res]
    best = f"{base / min(zk):.2f}" if base and zk else ""
    w.writerow([model] + [f"{res[(model, t)]:.1f}" if (model, t) in res else "" for t in TAGS] + [best, f"{peak.get(model, 0) / 1024:.1f}"])
