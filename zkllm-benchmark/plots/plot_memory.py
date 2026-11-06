#!/usr/bin/env python3
"""Figures for the zkLLM memory / Blendy analysis and the H100 zkMod runs.

  python3 plots/plot_memory.py [--data plots/data] [--out plots]

Inputs (plots/data):
  mem_breakdown.csv            categorize_memtrace.py output: bytes live at the traced peak, by category
  mem_<model>_<cfg>_timeline.csv  live device bytes after every cudaMalloc/cudaFree (traced build)
  prove_4090.csv               per-layer prove_s, RTX 4090, normal (untraced) build
  concurrency_4090.csv         plots/concurrency_4090.sh results: N provers at once, layers/min, how many finished
  op_partition_4090.csv        OPS_SUMMARY lines of op-partition-sweep.sh (results-ops-4090/opt-*/summary.log)
Outputs (plots):
  mem_peak_breakdown.png       what is live at the peak, legacy vs Blendy, per model
  mem_timeline.png             live device memory over one layer, legacy vs Blendy
  blendy_tiling.png            peak memory and prove time: legacy / Blendy t=16 / Blendy untiled
  4090_e2e.png, 4090_e2e_before.png  the same for RTX 4090 runs (plots/parse_e2e.py output), if present
  concurrency_4090.png         concurrent one-layer provers on one RTX 4090 (MPS): resident vs streamed weights
  op_partition_4090.png        zkMod with K tensor operations per partition (op-partition.py), K = 1, 2, 4, ..., all
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GIB = 2 ** 30
MODELS = ["opt-125m", "opt-1.3b", "opt-6.7b"]
CONFIG_LABEL = {"legacy": "legacy sumcheck", "blendy": "Blendy (t=16)", "blendy_untiled": "Blendy, untiled",
                "wstream": "Blendy + streamed weights"}
CONFIG_COLOR = {"legacy": "#8C8C8C", "blendy": "#4C72B0", "blendy_untiled": "#DD8452", "wstream": "#2CA02C"}
CONFIG_SHORT = {"legacy": "legacy", "blendy": "Blendy", "wstream": "+stream"}
CATEGORIES = [  # bottom to top
    ("resident layer weights", "#4C72B0"),
    ("weight copies (zkFC, opening pads)", "#8172B3"),
    ("activations", "#55A868"),
    ("lookup proof tensors", "#CCB974"),
    ("lookup tables", "#937860"),
    ("sumcheck working set", "#C44E52"),
    ("streamed-weight chunks and folds", "#17BECF"),
    ("input / other", "#BBBBBB"),
]
GPU_4090_GIB = 24564 / 1024


def savefig(fig, out, name):
    p = os.path.join(out, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def load_breakdown(data):
    d = defaultdict(lambda: defaultdict(float))
    for r in csv.DictReader(open(os.path.join(data, "mem_breakdown.csv"))):
        d[(r["model"], r["config"])][r["category"]] += int(r["bytes"]) / GIB
    return d


def plot_peak_breakdown(data, out):
    d = load_breakdown(data)
    configs = ["legacy", "blendy", "wstream"]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    width, gap = 0.26, 0.04
    for mi, m in enumerate(MODELS):
        for ci, c in enumerate(configs):
            x = mi + (ci - 1) * (width + gap)
            bottom = 0.0
            for cat, color in CATEGORIES:
                v = d[(m, c)].get(cat, 0.0)
                ax.bar(x, v, width, bottom=bottom, color=color, edgecolor="white", linewidth=0.5,
                       label=cat if (mi == 0 and ci == 0 and any(d[k].get(cat, 0) > 0 for k in d)) else None)
                if v >= 1.0:
                    ax.text(x, bottom + v / 2, f"{v:.1f}", ha="center", va="center", fontsize=7, color="white")
                bottom += v
            ax.text(x, bottom + 0.25, f"{bottom:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
            ax.text(x, -0.35, CONFIG_SHORT[c], ha="center", va="top", fontsize=7)
    ax.axhline(GPU_4090_GIB, color="black", ls="--", lw=0.8)
    ax.text(-0.5, GPU_4090_GIB + 0.3, "RTX 4090 (24 GiB)", ha="left", va="bottom", fontsize=8)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODELS)
    ax.tick_params(axis="x", pad=14)
    ax.set_ylabel("device memory live at peak (GiB)")
    ax.set_ylim(0, GPU_4090_GIB + 2.5)
    ax.set_title("zkLLM: what is live at the prover's peak (one OPT layer, seq 2048)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, -0.02, "Exact: every cudaMalloc/cudaFree traced (plots/memtrace). Excludes the ~0.5 GiB CUDA context.",
             fontsize=7, color="#555555")
    savefig(fig, out, "mem_peak_breakdown.png")


def load_timeline(data, model, config):
    t, b = [], []
    with open(os.path.join(data, f"mem_{model}_{config}_timeline.csv")) as f:
        next(f)
        for line in f:
            x, y = line.split(",")
            t.append(float(x) / 1000)
            b.append(int(y) / GIB)
    return np.array(t), np.array(b)


def plot_timeline(data, out):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(13, 3.8))
    for ax, m in zip(axes, MODELS):
        for c in ["legacy", "blendy", "wstream"]:
            t, b = load_timeline(data, m, c)
            i = int(np.argmax(b))
            ax.step(t / t[-1], b, where="post", color=CONFIG_COLOR[c], lw=0.9,
                    label=f"{CONFIG_LABEL[c]} (peak {b[i]:.2f} GiB)", alpha=0.9 if c == "wstream" else 0.7)
            ax.plot(t[i] / t[-1], b[i], "v", color=CONFIG_COLOR[c], ms=6)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
        ax.legend(fontsize=7.5, frameon=False, loc="upper left")
        ax.set_title(m)
        ax.set_xlabel("fraction of layer run (load → witgen → prove)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("live device memory (GiB)")
    fig.suptitle("zkLLM: live device memory over one OPT layer (▼ = peak)", y=1.02)
    fig.text(0.01, -0.04, "x is normalised per run: the traced build's wall time is inflated by the tracer, "
             "so only the shape and the memory levels are meaningful.", fontsize=7, color="#555555")
    fig.tight_layout()
    savefig(fig, out, "mem_timeline.png")


def plot_blendy_tiling(data, out):
    d = load_breakdown(data)
    prove = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(data, "prove_4090.csv"))):
        prove[(r["model"], r["config"])].append(float(r["prove_s"]))
    configs = ["legacy", "blendy", "blendy_untiled", "wstream"]
    x = np.arange(len(MODELS))
    w = 0.2
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.2))
    for i, c in enumerate(configs):
        peak = [sum(d[(m, c)].values()) for m in MODELS]
        ps = [np.mean(prove[(m, c)]) for m in MODELS]
        b1 = a1.bar(x + (i - 1.5) * w, peak, w, color=CONFIG_COLOR[c], label=CONFIG_LABEL[c])
        b2 = a2.bar(x + (i - 1.5) * w, ps, w, color=CONFIG_COLOR[c], label=CONFIG_LABEL[c])
        a1.bar_label(b1, fmt="%.1f", fontsize=7, padding=2)
        a2.bar_label(b2, fmt="%.2f", fontsize=6.5, padding=2)
    a1.axhline(GPU_4090_GIB, color="black", ls="--", lw=0.8)
    a1.text(-0.5, GPU_4090_GIB + 0.3, "RTX 4090 (24 GiB)", ha="left", va="bottom", fontsize=8)
    a1.set_ylim(0, 28)
    a1.set_ylabel("peak device memory (GiB, traced)")
    a1.set_title("Peak memory")
    a2.set_ylabel("prove time per layer (s)")
    a2.set_title("Prove time")
    for a in (a1, a2):
        a.set_xticks(x)
        a.set_xticklabels(MODELS)
        a.spines[["top", "right"]].set_visible(False)
    a2.legend(fontsize=8, frameon=False, loc="upper left")
    a2.set_ylim(0, 13)
    fig.suptitle("Blendy sumcheck with and without t-tiling, and with streamed weights (one OPT layer, seq 2048, RTX 4090)", y=1.02)
    fig.tight_layout()
    savefig(fig, out, "blendy_tiling.png")


def plot_e2e(csv_path, out, name, title, gpu_label, gpu_gb):
    rows = list(csv.DictReader(open(csv_path)))
    models = [r["model"] for r in rows]
    cfgs = [("zkllm", "zkLLM (monolithic)", "#333333"),
            ("zkmod2_seq", "zkMod 2, sequential", "#C6DBEF"),
            ("zkmod4_seq", "zkMod 4, sequential", "#6BAED6"),
            ("zkmod2_pipe", "zkMod 2, pipelined", "#FDD0A2"),
            ("zkmod4_pipe", "zkMod 4, pipelined", "#FD8D3C"),
            ("zkmod8_pipe", "zkMod 8, pipelined", "#D94801")]
    val = lambda r, k: float(r[k]) if r.get(k) else np.nan
    x = np.arange(len(models))
    w = 0.13
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw={"width_ratios": [2.2, 2.2, 1]})
    a1, a2, a3 = axes
    base = np.array([val(r, "zkllm") for r in rows])
    tmax = np.nanmax([val(r, k) for r in rows for k, _, _ in cfgs])
    smax = 1.0
    for i, (k, label, color) in enumerate(cfgs):
        t = np.array([val(r, k) for r in rows])
        xs = x + (i - 2.5) * w
        a1.bar(xs, np.nan_to_num(t), w, color=color, label=label, edgecolor="white", linewidth=0.4)
        for xi, ti in zip(xs, t):
            if np.isnan(ti):
                a1.text(xi, tmax * 0.01, "fail", rotation=90, ha="center", va="bottom", fontsize=6.5, color="#B22222")
        if k != "zkllm":
            sp = base / t
            smax = max(smax, np.nanmax(np.append(sp, 0)))
            a2.bar(x + (i - 3) * w, np.nan_to_num(sp), w, color=color, label=label, edgecolor="white", linewidth=0.4)
    a1.set_ylabel("end-to-end time (s)")
    a1.set_title("End-to-end proving time")
    a1.legend(fontsize=7.5, frameon=False, loc="upper left")
    a2.axhline(1.0, color="black", lw=0.8)
    a2.set_ylabel("speedup over monolithic zkLLM")
    a2.set_title("Speedup (zkLLM time / config time)")
    for r, xi, b in zip(rows, x, base):
        sps = [b / val(r, k) for k, _, _ in cfgs[1:] if r.get(k)]
        if np.isnan(b) or not sps:
            a2.text(xi, 0.1, "zkLLM baseline\nfailed", ha="center", va="bottom", fontsize=7, color="#B22222")
            continue
        a2.text(xi, max(sps) + 0.06 * smax, f"best {max(sps):.2f}x", ha="center", fontsize=7.5, fontweight="bold")
    a2.set_ylim(0, smax * 1.25)
    peak = [val(r, "peak_gb") for r in rows]
    b = a3.bar(x, peak, 0.6, color="#4C72B0")
    a3.bar_label(b, labels=[f"{p:.0f}" if p > 0 else "fail" for p in peak], fontsize=8, padding=2)
    a3.axhline(gpu_gb, color="black", ls="--", lw=0.8)
    a3.text(len(models) - 0.6, gpu_gb * 1.01, gpu_label, ha="right", va="bottom", fontsize=8)
    a3.set_ylim(0, gpu_gb * 1.13)
    a3.set_ylabel("peak GPU memory (GB)")
    a3.set_title("Peak memory")
    for a in axes:
        a.set_xticks(x)
        a.set_xticklabels(models, fontsize=8.5)
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    savefig(fig, out, name)


def plot_4090(data, out):
    for f, name, what in [("4090_e2e_sqrt_op.csv", "4090_e2e_sqrt_op.png", "square-root commitments, op-path partitions, decoupled witgen chain"),
                          ("4090_e2e_sqrt.csv", "4090_e2e_sqrt.png", "+ integer matmul, streamed activations, square-root weight commitments"),
                          ("4090_e2e.csv", "4090_e2e.png", "+ integer matmul, streamed activations"),
                          ("4090_e2e_bopen.csv", "4090_e2e_bopen.png", "+ integer matmul, streamed activations, batched openings"),
                          ("4090_e2e_wstream.csv", "4090_e2e_wstream.png", "Blendy + streamed weights"),
                          ("4090_e2e_before.csv", "4090_e2e_before.png", "before: legacy sumcheck, resident weights")]:
        if os.path.exists(os.path.join(data, f)):
            plot_e2e(os.path.join(data, f), out, name,
                     f"zkLLM vs zkMod partitions on one RTX 4090, MPS (OPT, seq 2048) - {what}", "RTX 4090 (24 GB)", 24)


def plot_concurrency(data, out):
    rows = list(csv.DictReader(open(os.path.join(data, "concurrency_4090.csv"))))
    models = [m for m in ["opt-1.3b", "opt-2.7b", "opt-6.7b", "opt-13b"] if any(r["model"] == m for r in rows)]
    colors = {"resident": CONFIG_COLOR["blendy"], "streamed": CONFIG_COLOR["wstream"]}
    labels = {"resident": "weights resident on GPU", "streamed": "weights streamed from host"}
    fig, axes = plt.subplots(1, len(models), figsize=(3.4 * len(models), 3.6), sharey=False)
    for ax, m in zip(axes, models):
        best = {}
        for wmode in ["resident", "streamed"]:
            rs = sorted((r for r in rows if r["model"] == m and r["weights"] == wmode), key=lambda r: int(r["n_provers"]))
            ok = [r for r in rs if r["ok"] == r["n_provers"]]
            bad = [r for r in rs if r["ok"] != r["n_provers"]]
            if ok:
                ax.plot([int(r["n_provers"]) for r in ok], [float(r["layers_per_min"]) for r in ok], "o-", color=colors[wmode],
                        label=labels[wmode])
                best[wmode] = max(float(r["layers_per_min"]) for r in ok)
            for r in bad:
                ax.plot(int(r["n_provers"]), 0, "x", color=colors[wmode], ms=9, mew=2)
                ax.annotate("OOM", (int(r["n_provers"]), 0), textcoords="offset points", xytext=(0, 7 if wmode == "resident" else 16),
                            ha="center", fontsize=7, color=colors[wmode])
        if "streamed" in best:
            note = (f"best: {best['streamed'] / best['resident']:.2f}x" if "resident" in best
                    else "resident: does not fit")
            ax.text(0.03, 0.97, note, transform=ax.transAxes, ha="left", va="top", fontsize=8, fontweight="bold")
        ax.set_title(m)
        ax.set_xlabel("concurrent one-layer provers")
        ax.set_xticks(sorted({int(r["n_provers"]) for r in rows if r["model"] == m}))
        ax.set_ylim(-0.6, 1.25 * max(best.values(), default=1.0))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("throughput (layers / min)")
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, fontsize=8, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Concurrent provers on one RTX 4090 (24 GiB, MPS), one OPT layer each, seq 2048 (x = out of memory)", y=1.03)
    fig.tight_layout()
    savefig(fig, out, "concurrency_4090.png")


OP_VARIANTS = [("field", "field-arithmetic matmul, resident activations", "#BBBBBB"),
               ("astream", "+ streamed activations", "#9ECAE1"),
               ("intmm", "+ integer forward matmul", "#FD8D3C"),
               ("bopen", "+ batched openings", "#D94801")]


def plot_op_partition(data, out):
    path = os.path.join(data, "op_partition_4090.csv")
    if not os.path.exists(path):
        return
    rows = [r for r in csv.DictReader(open(path)) if r["status"] == "OK"]
    for r in rows:
        r.setdefault("variant", "field")
    latest = OP_VARIANTS[-1][0]
    models = [m for m in ["125m", "1.3b", "2.7b", "6.7b", "13b"] if any(r["model"] == m and r["variant"] == latest for r in rows)]
    layer = {}
    e2e_path = os.path.join(data, "4090_e2e.csv")
    if os.path.exists(e2e_path):
        for r in csv.DictReader(open(e2e_path)):
            ts = [float(r[k]) for k in ["zkllm", "zkmod2_seq", "zkmod4_seq", "zkmod2_pipe", "zkmod4_pipe", "zkmod8_pipe"] if r.get(k)]
            if ts:
                layer[r["model"]] = (float(r["zkllm"]) if r.get("zkllm") else None, min(ts))
    fig, axes = plt.subplots(2, len(models), figsize=(5.4 * len(models), 7.6), squeeze=False)
    for ci, m in enumerate(models):
        a = axes[0][ci]
        total, ymax = None, 0
        for v, label, color in OP_VARIANTS:
            rs = sorted((r for r in rows if r["model"] == m and r["variant"] == v), key=lambda r: int(r["ops_per_part"]))
            if not rs:
                continue
            total = int(rs[0]["total_ops"])
            ks = [int(r["ops_per_part"]) for r in rs]
            e2e = [float(r["e2e_s"]) for r in rs]
            ymax = max(ymax, max(e2e))
            a.plot(ks, e2e, "o-", color=color, ms=4, lw=1.6 if v == latest else 1.2, label=label)
            i = int(np.argmin(e2e))
            if v in (OP_VARIANTS[0][0], latest):
                a.annotate(f"best {e2e[i]:.0f} s", (ks[i], e2e[i]), textcoords="offset points", xytext=(0, 7 if v != latest else -13), ha="center",
                           fontsize=7.5, fontweight="bold", color=color if v != latest else "black")
        if m in layer:
            zk, best = layer[m]
            if zk:
                a.axhline(zk, color="#333333", ls="--", lw=1, label=f"zkLLM monolithic, current code ({zk:.0f} s)")
            a.axhline(best, color="#4C72B0", ls=":", lw=1.2, label=f"best layer partitions, current code ({best:.0f} s)")
        a.axvline(31, color="#999999", lw=0.8)
        a.text(31, ymax * 0.02, " 1 layer", fontsize=7, color="#777777", va="bottom")
        a.set_xscale("log", base=2)
        a.set_ylim(0, ymax * 1.12)
        a.set_title(f"opt-{m} ({total} ops)")
        a.set_xlabel("tensor operations per partition (K; rightmost = all)")
        a.set_ylabel("end-to-end time (s)" if ci == 0 else "")
        a.legend(fontsize=6.8, frameon=False, loc="upper right")
        a.spines[["top", "right"]].set_visible(False)
        b = axes[1][ci]
        rs = sorted((r for r in rows if r["model"] == m and r["variant"] == latest), key=lambda r: int(r["ops_per_part"]))
        ks = [int(r["ops_per_part"]) for r in rs]
        prove = np.array([float(r["prove_sum_s"]) for r in rs])
        commit = np.array([float(r["commit_sum_s"]) for r in rs])
        other = np.array([float(r["proc_wall_sum_s"]) for r in rs]) - prove - commit
        wit = np.array([float(r["witgen_s"]) for r in rs])
        xs = np.arange(len(ks))
        b.bar(xs, prove, color="#4C72B0", label="proving")
        b.bar(xs, commit, bottom=prove, color="#C44E52", label="boundary commit + open")
        b.bar(xs, other, bottom=prove + commit, color="#BBBBBB", label="forward pass, state I/O, startup")
        b.plot(xs, [float(r["e2e_s"]) for r in rs], "k_", ms=14, mew=2, label="end-to-end time")
        b.plot(xs[wit > 0], wit[wit > 0], "_", color="#D94801", ms=14, mew=2, label="forward-pass (witgen) chain")
        b.set_xticks(xs)
        b.set_xticklabels([str(k) if k != int(rs[0]["total_ops"]) else "all" for k in ks], fontsize=7.5)
        b.set_xlabel("tensor operations per partition (K)")
        b.set_ylabel("seconds, summed over prover processes" if ci == 0 else "")
        b.set_title(f"opt-{m}, current code: where the time goes", fontsize=9)
        b.spines[["top", "right"]].set_visible(False)
        if ci == 0:
            b.legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle("zkMod with tensor-operation partitions on one RTX 4090 (MPS, <=16 provers at once, OPT, seq 2048)", y=1.0)
    fig.tight_layout()
    savefig(fig, out, "op_partition_4090.png")


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--data", default=os.path.join(here, "data"))
    ap.add_argument("--out", default=here)
    a = ap.parse_args()
    plt.rcParams.update({"font.size": 9})
    plot_peak_breakdown(a.data, a.out)
    plot_timeline(a.data, a.out)
    plot_blendy_tiling(a.data, a.out)
    # the H100 figure (h100_e2e.png) is drawn by plot_h100_e2e.py
    plot_4090(a.data, a.out)
    plot_concurrency(a.data, a.out)
    plot_op_partition(a.data, a.out)


if __name__ == "__main__":
    main()
