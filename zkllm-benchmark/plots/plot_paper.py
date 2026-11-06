#!/usr/bin/env python3
"""zkLLM vs zkMod figures for the evaluation section (experiments 3-5), one H100, OPT models.

  python3 plots/plot_paper.py

Writes plots/<name>.png/.pdf and plots/data/<name>.csv:
  zkllm_seq64        experiment 3: response time at seq 64, zkLLM vs zkMod (pipelined, best split count per model)
  zkllm_partitions   experiment 4: speedup over zkLLM at seq 64 vs partition count, all models (zkLLM = 1 partition)
  zkllm_seqlen       experiment 5: response time at seq 64 and 2048, each at its best split count
  zkllm_partitions_scaling  seq 64: response time and per-partition peak GPU memory vs partition count (zkLLM = 1 partition)
  zkllm_seqlen_scaling      SEQLEN_SPLITS (8) partitions: zkMod speedup over zkLLM vs sequence length
  zkllm_proof_overheads     seq 64 proof size (proof_size.py, analytic) and 100 Mbps transmit time, zkLLM vs zkMod
                            at its best split count; also plots/zkllm_proof_overheads.tex (verify left [TBD])

Response time = the driver's e2e_s;
per-partition GPU memory = max over partitions of alloc_peak_mb in results-h100-e2e-<L>-*/opt-*.json (as zkGPT's Fig 8). zkLLM records no proof size, so
unlike zkGPT no transmission time is added to response time; proof_size.py counts it from the layer dimensions.
  seq 64    results-h100-e2e-64-*/summary.log (run_e2e.sh), runs with rc=0 and status=OK
  seq L     results-h100-e2e-<L>-*/summary.log likewise (SEQ_LEN=<L> PIPE_SPLITS=8), 2048 falling back to data/h100_e2e.csv
  seq 2048  data/h100_e2e.csv (plot_h100_e2e.py)
"""
import csv
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

import paperstyle as ps
import proof_size

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS = ["125m", "1.3b", "2.7b", "6.7b", "13b"]


SEQLEN_SPLITS = 8
SEQLEN_MAX = 256    # zkllm_seqlen_scaling sweeps seq 64..SEQLEN_MAX


def load64(seq=64, field="e2e_s"):
    """{model: {"zkllm": v, <splits>: v}} over the pipelined zkMod runs; v = the RESULT line's `field`."""
    out = {}
    for log in sorted(glob.glob(os.path.join(ROOT, f"results-h100-e2e-{seq}-*", "summary.log"))):
        for line in open(log):
            r = dict(re.findall(r"(\w+)=(\S+)", line))
            if not line.startswith("RESULT") or r.get("rc") != "0" or r.get("status") != "OK":
                continue
            m, tag = r["model"].removeprefix("opt-"), r["tag"]
            k = "zkllm" if tag == "zkllm" else (int(tag[5:-5]) if tag.endswith("_pipe") else None)
            if k is not None:
                out.setdefault(m, {})[k] = float(r[field])
    return out


def load_part_mem(seq=64):
    """{model: {"zkllm": v, <splits>: v}}: max over partitions of each process's allocator peak (alloc_peak_mb), MiB."""
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, f"results-h100-e2e-{seq}-*", "opt-*.json"))):
        hit = re.match(r"opt-(.+?)_(zkllm|zkmod\d+_pipe)\.json$", os.path.basename(f))
        if not hit:
            continue
        m, tag = hit.groups()
        d = json.load(open(f))
        if d.get("status") != "OK":
            continue
        k = "zkllm" if tag == "zkllm" else int(tag[5:-5])
        out.setdefault(m, {})[k] = max(q["alloc_peak_mb"] for q in d["partitions"])
    return out


def load2048():
    out = {}
    for r in csv.DictReader(open(os.path.join(HERE, "data", "h100_e2e.csv"))):
        out[r["model"].removeprefix("opt-")] = dict(zkllm=float(r["e2e_zkllm_s"]), best=float(r["e2e_best_s"]),
                                                    splits=int(r["e2e_best_splits"]))
    return out


def best(d):
    k = min((k for k in d if k != "zkllm"), key=d.get)
    return k, d[k]


def write(name, header, rows):
    with open(os.path.join(HERE, "data", f"{name}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def seq64(d64):
    ms = [m for m in MODELS if m in d64 and "zkllm" in d64[m]]
    fig, ax = plt.subplots(figsize=(3.4, 1.6))
    ps.pair_bars(ax, [ps.name(m, True) for m in ms], [d64[m]["zkllm"] for m in ms], [best(d64[m])[1] for m in ms],
                 "zkLLM", "zkMod", "Response time (s)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.05)
    ax.legend(loc="upper left")
    fig.tight_layout()
    ps.save(fig, os.path.join(HERE, "zkllm_seq64"))
    write("zkllm_seq64", ["model", "zkllm_s", "zkmod_splits", "zkmod_s", "speedup"],
          [[f"opt-{m}", f"{d64[m]['zkllm']:.1f}", best(d64[m])[0], f"{best(d64[m])[1]:.1f}",
            f"{d64[m]['zkllm'] / best(d64[m])[1]:.2f}"] for m in ms])


def partitions(d64):
    """seq 64: speedup over zkLLM vs partition count, every model on one axis, zkLLM = 1 partition (1x)."""
    ms = [m for m in MODELS if m in d64 and "zkllm" in d64[m]]
    pts = {m: sorted([(1, d64[m]["zkllm"])] + [(k, v) for k, v in d64[m].items() if k != "zkllm"]) for m in ms}
    xs = {m: [p for p, _ in pts[m]] for m in ms}
    fig, ax = plt.subplots(figsize=(3.4, 1.6))
    _lines(ax, xs, {m: [d64[m]["zkllm"] / v for _, v in pts[m]] for m in ms}, ms)
    ax.set_xticks(sorted({p for m in ms for p in xs[m]}))
    ax.set_xlabel("Partitions"); ax.set_ylabel("Speedup over zkLLM"); ps.grid(ax, "both")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}\u00d7"))
    ax.legend(loc="upper left", ncol=3)
    fig.tight_layout()
    ps.save(fig, os.path.join(HERE, "zkllm_partitions"))
    write("zkllm_partitions", ["model", "partitions", "response_s", "speedup_vs_zkllm"],
          [[f"opt-{m}", p, f"{v:.1f}", f"{d64[m]['zkllm'] / v:.2f}"] for m in ms for p, v in pts[m]])


def seqlen(d64, d2k):
    ms = [m for m in MODELS if m in d64 and "zkllm" in d64[m] and m in d2k]
    series = [("zkLLM, 64", ps.BASE, lambda m: d64[m]["zkllm"]), ("zkMod, 64", (ps.BLUES[2], ""), lambda m: best(d64[m])[1]),
              ("zkLLM, 2048", (ps.BLUES[1], "xxxx"), lambda m: d2k[m]["zkllm"]), ("zkMod, 2048", ps.ZK, lambda m: d2k[m]["best"])]
    fig, ax = plt.subplots(figsize=(3.4, 1.6))
    x = np.arange(len(ms)); w, gap = 0.16, 0.06
    for j, (lab, shade, f) in enumerate(series):
        xs = x + (j - 1.5) * (w + gap)
        ps.bar(ax, xs, [f(m) for m in ms], w, shade, label=lab)
        if j % 2:                                      # zkMod bar: speedup over zkLLM at the same length
            for xx, m in zip(xs, ms):
                ps.speedup(ax, xx, f(m), series[j - 1][2](m) / f(m), rotation=90, size=6)
    ax.set_xticks(x); ax.set_xticklabels([ps.name(m, True) for m in ms])
    ax.set_yscale("log"); ax.set_ylabel("Response time (s)"); ps.grid(ax)
    lo, hi = ax.get_ylim(); ax.set_ylim(min(f(m) for _, _, f in series for m in ms) / 2, hi * 6)
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    ps.save(fig, os.path.join(HERE, "zkllm_seqlen"))
    rows = []
    for m in ms:
        k, v = best(d64[m])
        rows.append([f"opt-{m}", 64, f"{d64[m]['zkllm']:.1f}", k, f"{v:.1f}", f"{d64[m]['zkllm'] / v:.2f}"])
        rows.append([f"opt-{m}", 2048, f"{d2k[m]['zkllm']:.1f}", d2k[m]["splits"], f"{d2k[m]['best']:.1f}",
                     f"{d2k[m]['zkllm'] / d2k[m]['best']:.2f}"])
    write("zkllm_seqlen", ["model", "seq_len", "zkllm_s", "zkmod_splits", "zkmod_s", "speedup"], rows)


def _lines(ax, xs_by_model, ys_by_model, ms, **kw):
    for i, m in enumerate(ms):
        if m in ys_by_model:
            ax.plot(xs_by_model[m], ys_by_model[m], ps.MARKS[i % len(ps.MARKS)] + kw.get("ls", "-"), color="black",
                    mfc=kw.get("mfc", "black"), label=kw.get("label", "{}").format(ps.name(m)))


def partitions_scaling(d64):
    """seq 64: zkMod speedup over zkLLM per model, one bar per partition count."""
    ms = [m for m in MODELS if m in d64 and "zkllm" in d64[m]]
    mem = load_part_mem(64)
    pts = {m: sorted([(1, "zkllm")] + [(k, k) for k in d64[m] if k != "zkllm"]) for m in ms}
    xs = {m: [p for p, _ in pts[m]] for m in ms}
    ks = sorted({k for m in ms for _, k in pts[m] if k != "zkllm"})
    fig, a = plt.subplots(figsize=(3.4, 1.7))
    ps.group_bars(a, ms, ks, {m: {k: d64[m]["zkllm"] / d64[m][k] for k in ks if k in d64[m]} for m in ms},
                  lambda k: f"{k} partitions", "Speedup over zkLLM")
    fig.tight_layout()
    ps.save(fig, os.path.join(HERE, "zkllm_partitions_scaling"))
    write("zkllm_partitions_scaling", ["model", "partitions", "response_s", "speedup_vs_zkllm", "peak_gpu_gib_per_partition"],
          [[f"opt-{m}", p, f"{d64[m][k]:.1f}", f"{d64[m]['zkllm'] / d64[m][k]:.2f}", f"{mem[m][k] / 1024:.1f}"]
           for m in ms for p, k in pts[m]])


def seqlen_scaling(d2k):
    """SEQLEN_SPLITS partitions at every sequence length up to SEQLEN_MAX that has runs: zkMod speedup over zkLLM."""
    P = SEQLEN_SPLITS
    data = {}                                                   # (model, seq) -> (zkllm, zkmod)
    for L in sorted({int(re.search(r"results-h100-e2e-(\d+)-", d).group(1))
                     for d in glob.glob(os.path.join(ROOT, "results-h100-e2e-*-*"))}):
        if L > SEQLEN_MAX:
            continue
        for m, d in load64(L).items():
            if "zkllm" in d and P in d:
                data[(m, L)] = (d["zkllm"], d[P])
    for r in csv.DictReader(open(os.path.join(HERE, "data", "h100_e2e.csv"))):
        m, v = r["model"].removeprefix("opt-"), r.get(f"e2e_zkmod{P}_pipe_s")
        if v and 2048 <= SEQLEN_MAX and (m, 2048) not in data:
            data[(m, 2048)] = (float(r["e2e_zkllm_s"]), float(v))
    ms = [m for m in MODELS if any(k[0] == m for k in data)]
    Ls = {m: sorted(L for mm, L in data if mm == m) for m in ms}
    ticks = sorted({L for m in ms for L in Ls[m]})
    fig, b = plt.subplots(figsize=(3.4, 1.7))
    ps.group_bars(b, ms, ticks, {m: {L: data[(m, L)][0] / data[(m, L)][1] for L in Ls[m]} for m in ms},
                  lambda L: f"Length {L}", "Speedup over zkLLM", legend_cols=3)
    fig.tight_layout()
    ps.save(fig, os.path.join(HERE, "zkllm_seqlen_scaling"))
    write("zkllm_seqlen_scaling", ["model", "seq_len", "partitions", "zkllm_s", "zkmod_s", "speedup"],
          [[f"opt-{m}", L, P, f"{data[(m, L)][0]:.1f}", f"{data[(m, L)][1]:.1f}", f"{data[(m, L)][0] / data[(m, L)][1]:.2f}"]
           for m in ms for L in Ls[m]])


LINK_BPS = 100e6    # proof transmission: one 100 Mbps link, as zkGPT


def proof_overheads(d64):
    """seq 64: analytic proof size and its transmit time, zkLLM vs zkMod at its zkllm_seq64 split count."""
    ms = [m for m in MODELS if m in d64 and "zkllm" in d64[m]]
    rows, L = [], [r"\begin{table}[t]", r"  \centering", r"  \small", r"  \setlength{\tabcolsep}{4pt}",
                   r"  \renewcommand{\arraystretch}{0.9}", r"  \begin{tabular}{llccc}", r"    \toprule",
                   r"    Model & Impl. & Proof (KB) & Transmit (s) & Verify (s) \\", r"    \midrule"]
    for i, m in enumerate(ms):
        P = best(d64[m])[0]
        for impl, name, parts in (("zkllm", "Baseline", 1), ("zkmod", r"\X", P)):
            b = proof_size.proof_bytes(m, 64, parts)
            rows.append([f"opt-{m}", impl, parts, f"{b / 1024:.1f}", f"{b * 8 / LINK_BPS:.3f}"])
            lead = rf"\multirow{{2}}{{*}}{{{ps.name(m)}}}" if parts == 1 else ""
            L.append(f"    {lead} & {name} & {b / 1024:.0f} & {b * 8 / LINK_BPS:.3f} & [TBD] \\\\")
        if i < len(ms) - 1:
            L.append(r"    \midrule")
    L += [r"    \bottomrule", r"  \end{tabular}",
          r"  \caption{Proof and verification overheads at sequence length 64 for zkLLM. Proof sizes are counted from the "
          r"layer dimensions; transmission assumes a 100\,Mbps network.}",
          r"  \label{tab:proof-overheads-zkllm}", r"\end{table}"]
    open(os.path.join(HERE, "zkllm_proof_overheads.tex"), "w").write("\n".join(L) + "\n")
    write("zkllm_proof_overheads", ["model", "impl", "partitions", "proof_kb", "transmit_s"], rows)


if __name__ == "__main__":
    d64, d2k = load64(), load2048()
    seq64(d64)
    partitions(d64)
    seqlen(d64, d2k)
    partitions_scaling(d64)
    seqlen_scaling(d2k)
    proof_overheads(d64)
    for n in ("zkllm_seq64", "zkllm_partitions", "zkllm_seqlen", "zkllm_partitions_scaling", "zkllm_seqlen_scaling",
              "zkllm_proof_overheads"):
        print(f"-- {n}"); print(open(os.path.join(HERE, "data", f"{n}.csv")).read())
