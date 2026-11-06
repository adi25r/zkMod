#!/usr/bin/env python3
"""Same-arithmetization comparison: modular-zkp's split driver (LLMSplitBlock, paper
half/half layout, merge=1) run as (a) modular-zkp's own binary on the CPU, (b) our
binary with the CPU prover, (c) our binary with the GPU prover. Witgen is on the CPU
in all three, so (b)->(c) isolates the prover.

Logs are the drivers' stdout with every line timestamped by _queue/tsrun.sh:
  results/speedup_analysis_models/same_arith/logs/<model>_{mz_cpu,ours_cpu,ours_gpu}.log
witgen wall = first line -> 'Parallel verification start' (probe pass + all create()s),
prove wall  = that line  -> 'Results:' (falls back to the last RESULT: ACCEPT).

Figures (house style, results/speedup_analysis_models/figs):
  witgen_prove_<model>  stacked witgen + prove per arm
  times_all             total wall per arm per model (log axis)
  speedup_all           prove-wall speedup of the GPU prover vs each CPU arm
"""
import argparse, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plotstyle as ps

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
LOGS = os.path.join(RES, "speedup_analysis_models", "same_arith", "logs")
FIGS = os.path.join(RES, "speedup_analysis_models", "figs")
MODELS = ["gpt2", "opt350m", "opt1b3"]
PRETTY = {"gpt2": "GPT-2", "opt350m": "OPT-350M", "opt1b3": "OPT-1.3B"}
ARMS = [("mz_cpu", "modular-zkp, CPU prover"), ("ours_cpu", "ours, CPU prover"), ("ours_gpu", "ours, GPU prover")]
ARM_LABEL = dict(ARMS)


def walls(path):
    """(witgen_s, prove_s, passes, results_line) or None."""
    if not os.path.exists(path):
        return None
    L = []
    for line in open(path, errors="replace"):
        m = re.match(r"(\d+\.\d+) (.*)", line.rstrip("\n"))
        if m:
            L.append((float(m.group(1)), m.group(2)))
    if not L:
        return None
    t0 = L[0][0]
    tp = next((t for t, s in L if s.startswith("Parallel verification start")), None)
    tr = next((t for t, s in L if s.startswith("Results:")), None)
    if tr is None:
        acc = [t for t, s in L if s.startswith("RESULT: ACCEPT") or "All verification passed" in s]
        tr = max(acc) if acc else None
    passes = sum(1 for _, s in L if "All verification passed" in s)
    nparts = next((int(m.group(1)) for _, s in L for m in [re.search(r"splits=(\d+)", s)] if m), None)
    if tp is None or tr is None or (nparts and passes < nparts):
        return None                              # unfinished or not fully verified
    res = next((s for _, s in L if s.startswith("Results:")), "")
    return (tp - t0, tr - tp, passes, res)


def load():
    per = {}
    for g in MODELS:
        d = {a: walls(os.path.join(LOGS, f"{g}_{a}.log")) for a, _ in ARMS}
        if any(d.values()):
            per[g] = d
    return per


def fig_witgen_prove(g, d):
    rows = [(ARM_LABEL[a], d[a][0], d[a][1]) for a, _ in ARMS if d.get(a)]
    fig, ax = ps.figure()
    _c, tops = ps.stacked_bars(ax, [r[0] for r in rows], ["witness generation", "proving"],
                               [[r[1] for r in rows], [r[2] for r in rows]])
    ps.top_labels(ax, range(len(rows)), tops, fstr="{:,.0f}")
    ax.set_ylabel("Time (s)")
    ps.grid(ax); ps.headroom(ax); ps.legend(ax)
    return ps.save(fig, os.path.join(FIGS, f"witgen_prove_{g}"))


def fig_times_all(per):
    groups = [g for g in MODELS if g in per]
    labels = [l for _, l in ARMS]
    vals = [[(per[g][a][0] + per[g][a][1]) if per[g].get(a) else None for g in groups] for a, _ in ARMS]
    fig, ax = ps.figure()
    rects, _idx = ps.grouped_bars(ax, [PRETTY[g] for g in groups], labels, vals)
    ax.set_yscale("log")
    for l in labels:
        ps.value_labels(ax, rects[l], fstr="{:,.0f}", rotation=90)
    ax.set_ylabel("Time (s)")
    ps.grid(ax); ps.headroom(ax, log=True, log_factor=40.0); ps.legend(ax)
    return ps.save(fig, os.path.join(FIGS, "times_all"))


def fig_speedup_all(per):
    groups = [g for g in MODELS if g in per and per[g].get("ours_gpu")]
    series = ["GPU prover vs modular-zkp CPU prover", "GPU prover vs our CPU prover"]
    vals = [[(per[g]["mz_cpu"][1] / per[g]["ours_gpu"][1]) if per[g].get("mz_cpu") else None for g in groups],
            [(per[g]["ours_cpu"][1] / per[g]["ours_gpu"][1]) if per[g].get("ours_cpu") else None for g in groups]]
    fig, ax = ps.figure()
    rects, _idx = ps.grouped_bars(ax, [PRETTY[g] for g in groups], series, vals)
    for l in series:
        ps.value_labels(ax, rects[l], fstr="{:.2f}")
    ax.axhline(1.0, color="black", linestyle="dashed", linewidth=1)
    ax.set_ylabel("Proving speedup")
    ps.grid(ax); ps.headroom(ax); ps.legend(ax)
    return ps.save(fig, os.path.join(FIGS, "speedup_all"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figs", action="store_true")
    args = ap.parse_args()
    ps.apply()
    per = load()
    print(f"{'model':8s} {'arm':10s} {'witgen_s':>9s} {'prove_s':>8s} {'total_s':>8s} pass   (whole-run wall clock)")
    for g in MODELS:
        for a, _ in ARMS:
            v = per.get(g, {}).get(a)
            if v:
                print(f"{g:8s} {a:10s} {v[0]:9.1f} {v[1]:8.1f} {v[0]+v[1]:8.1f} {v[2]:4d}")
    for g in MODELS:
        if g in per and per[g].get("ours_gpu"):
            gp = per[g]["ours_gpu"][1]
            s = [f"vs {a}: {per[g][a][1]/gp:.2f}x" for a in ("mz_cpu", "ours_cpu") if per[g].get(a)]
            print(f"{g:8s} GPU prove-wall speedup  " + "  ".join(s))
    if args.no_figs:
        return
    for g, d in per.items():
        print("wrote", fig_witgen_prove(g, d))
    print("wrote", fig_times_all(per))
    print("wrote", fig_speedup_all(per))


if __name__ == "__main__":
    main()
