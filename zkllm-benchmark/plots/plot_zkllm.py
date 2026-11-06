#!/usr/bin/env python3
"""zkLLM vs zkMod figures, one H100, OPT models. Reads results-h100-e2e-<L>-*/ from run_e2e.sh and writes plots/figs/:
  zkllm_seq64                response time at seq 64
  zkllm_partitions_scaling   response time and per-partition peak GPU memory vs partition count
  zkllm_seqlen_scaling       zkMod speedup over zkLLM vs sequence length
  zkllm_proof_overheads      seq 64 proof size and 100 Mbps transmit time (.csv and .tex)
"""
import csv
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

import plotstyle as ps
import proof_size

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "figs")
os.makedirs(OUT, exist_ok=True)
MODELS = ["125m", "1.3b", "2.7b", "6.7b", "13b"]


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




def top(d):
    k = max(k for k in d if k != "zkllm")
    return k, d[k]


def write(name, header, rows):
    with open(os.path.join(OUT, f"{name}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def seq64(d64):
    ms = [m for m in MODELS if m in d64 and "zkllm" in d64[m]]
    fig, ax = plt.subplots(figsize=(3.4, 1.6))
    ps.pair_bars(ax, [ps.name(m, True) for m in ms], [d64[m]["zkllm"] for m in ms], [top(d64[m])[1] for m in ms],
                 "zkLLM", "zkMod", "Response time (s)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.05)
    ax.legend(loc="upper left")
    fig.tight_layout()
    ps.save(fig, os.path.join(OUT, "zkllm_seq64"))
    write("zkllm_seq64", ["model", "zkllm_s", "zkmod_splits", "zkmod_s", "speedup"],
          [[f"opt-{m}", f"{d64[m]['zkllm']:.1f}", top(d64[m])[0], f"{top(d64[m])[1]:.1f}",
            f"{d64[m]['zkllm'] / top(d64[m])[1]:.2f}"] for m in ms])






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
    fig, a = plt.subplots(figsize=(3.4, 1.3))
    ps.group_bars(a, ms, ks, {m: {k: d64[m]["zkllm"] / d64[m][k] for k in ks if k in d64[m]} for m in ms},
                  str, "Speedup over\nzkLLM", legend_cols=len(ks),
                  legend_loc="lower right")
    fig.tight_layout()
    ps.save(fig, os.path.join(OUT, "zkllm_partitions_scaling"))
    write("zkllm_partitions_scaling", ["model", "partitions", "response_s", "speedup_vs_zkllm", "peak_gpu_gib_per_partition"],
          [[f"opt-{m}", p, f"{d64[m][k]:.1f}", f"{d64[m]['zkllm'] / d64[m][k]:.2f}", f"{mem[m][k] / 1024:.1f}"]
           for m in ms for p, k in pts[m]])


def seqlen_scaling(d2k):
    """Each model at its max partition count at seq 64, at every sequence length up to SEQLEN_MAX that has runs: zkMod speedup over zkLLM."""
    P = {m: max(k for k in d if k != "zkllm") for m, d in load64(64).items()}
    data = {}                                                   # (model, seq) -> (zkllm, zkmod)
    for L in sorted({int(re.search(r"results-h100-e2e-(\d+)-", d).group(1))
                     for d in glob.glob(os.path.join(ROOT, "results-h100-e2e-*-*"))}):
        if L > SEQLEN_MAX:
            continue
        for m, d in load64(L).items():
            if "zkllm" in d and P.get(m) in d:
                data[(m, L)] = (d["zkllm"], d[P[m]])
    for r in []:
        m, v = r["model"].removeprefix("opt-"), r.get(f"e2e_zkmod{P.get(m)}_pipe_s")
        if v and 2048 <= SEQLEN_MAX and (m, 2048) not in data:
            data[(m, 2048)] = (float(r["e2e_zkllm_s"]), float(v))
    ms = [m for m in MODELS if any(k[0] == m for k in data)]
    Ls = {m: sorted(L for mm, L in data if mm == m) for m in ms}
    ticks = sorted({L for m in ms for L in Ls[m]})
    fig, b = plt.subplots(figsize=(3.4, 1.3))
    ps.group_bars(b, ms, ticks, {m: {L: data[(m, L)][0] / data[(m, L)][1] for L in Ls[m]} for m in ms},
                  str, "Speedup over\nzkLLM", legend_cols=len(ticks),
                  legend_loc="lower right")
    fig.tight_layout()
    ps.save(fig, os.path.join(OUT, "zkllm_seqlen_scaling"))
    write("zkllm_seqlen_scaling", ["model", "seq_len", "partitions", "zkllm_s", "zkmod_s", "speedup"],
          [[f"opt-{m}", L, P[m], f"{data[(m, L)][0]:.1f}", f"{data[(m, L)][1]:.1f}", f"{data[(m, L)][0] / data[(m, L)][1]:.2f}"]
           for m in ms for L in Ls[m]])


LINK_BPS = 100e6    # proof transmission: one 100 Mbps link, as zkGPT


def proof_overheads(d64):
    """seq 64: analytic proof size and its transmit time, zkLLM vs zkMod at its zkllm_seq64 split count."""
    ms = [m for m in MODELS if m in d64 and "zkllm" in d64[m]]
    rows, L = [], [r"\begin{table}[t]", r"  \centering", r"  \small", r"  \setlength{\tabcolsep}{4pt}",
                   r"  \renewcommand{\arraystretch}{0.9}", r"  \begin{tabular}{llcccc}", r"    \toprule",
                   r"    Model & Impl. & Commit. (MB) & Proof (KB) & Transmit (s) & Verify (s) \\", r"    \midrule"]
    for i, m in enumerate(ms):
        P = top(d64[m])[0]
        for impl, name, parts in (("zkllm", "Baseline", 1), ("zkmod", r"\X", P)):
            b = proof_size.proof_bytes(m, 64, parts, batched=True)
            c = proof_size.commitment_bytes(m, 64, parts)
            rows.append([f"opt-{m}", impl, parts, f"{c / 1e6:.3f}", f"{b / 1024:.1f}", f"{b * 8 / LINK_BPS:.3f}"])
            lead = rf"\multirow{{2}}{{*}}{{{ps.name(m)}}}" if parts == 1 else ""
            L.append(f"    {lead} & {name} & {c / 1e6:.{3 if c < 1e6 else 2}f} & {b / 1024:.0f} & {b * 8 / LINK_BPS:.3f} & [TBD] \\\\")
        if i < len(ms) - 1:
            L.append(r"    \midrule")
    L += [r"    \bottomrule", r"  \end{tabular}",
          r"  \caption{Proof and verification overheads at sequence length 64 for zkLLM. Proof sizes are counted from the "
          r"layer dimensions, with each operation batched across the layers of a proof (all layers for the baseline, "
          r"each partition's layers for \X). Commitments are to the model weights, published once and the same for both, "
          r"plus for \X the intermediate value at each partition boundary; "
          r"transmission assumes a 100\,Mbps network.}",
          r"  \label{tab:proof-overheads-zkllm}", r"\end{table}"]
    open(os.path.join(OUT, "zkllm_proof_overheads.tex"), "w").write("\n".join(L) + "\n")
    write("zkllm_proof_overheads", ["model", "impl", "partitions", "commitment_mb", "proof_kb", "transmit_s"], rows)


if __name__ == "__main__":
    d64 = load64()
    seq64(d64)
    partitions_scaling(d64)
    seqlen_scaling({})
    proof_overheads(d64)
