#!/usr/bin/env python3
"""plot_speedup.py -- tables + figures for the 5-arm ablation.

Suites (--suite), all under results/speedup_analysis_models/:
  paper         gpt2 / opt350m / opt1b3 / opt2b7 at len 64 (default), paper arithmetization
                on every arm and the baseline; logs paper_arith/logs/<model>_<arm>.log.
  paper_seqlen  the same at len 64 / 128 / 256 (logs from results/figures/logs).

Arms: arm0 = this repo's monolithic path (main_gkr_split splits=1, all-CPU, same
flags as arm1): witgen_s from the PARTITION line, prove from SPLIT_SUMMARY
makespan_s. If only zkGPT-og's log exists (arm0_zkgpt_mono.log) its WITGEN: line
is used instead. A monolith that got through witgen + range + commit + GKR but
died in Lasso (opt2b7: 2^33 domain) is a PARTIAL lower bound: it is tabulated
with ">=", excluded from every speedup figure, and not drawn as a time.
arm1-4 = --overlapped runs: total from SPLIT_OVERLAPPED overlapped_total_s;
witgen/prove split is a proxy (witgen = wall offset of the last OVERLAPPED_BUILT,
which overlaps proving by design), the total is exact.

Figures (PDF + PNG, tools/plotstyle.py house style) in the suite's figs/:
  witgen_prove_<group>   stacked witgen | prove per arm
  phases_<group>         per-partition prover phases per split arm
  times_all              end-to-end seconds, groups x arms, log axis
  speedup_all            speedup over the monolith, groups x arms

Usage: tools/plot_speedup.py [--suite models|seqlen] [--only <group>] [--no-figs]
"""
import argparse
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plotstyle as ps

RES = os.path.join(HERE, "results")

ARM_FILES = [
    ("arm0_mono.log",           "Monolithic (splits=1, CPU)"),
    ("arm1_cpuwg_cpuprove.log", "CPU witgen + CPU prove"),
    ("arm2_cpuwg_gpuprove.log", "CPU witgen + GPU prove"),
    ("arm3_gpuwg_cpuprove.log", "GPU witgen + CPU prove"),
    ("arm4_gpuwg_gpuprove.log", "GPU witgen + GPU prove"),
]
ARM_LABELS = [l for _f, l in ARM_FILES]
ARM0_FALLBACK = "arm0_zkgpt_mono.log"

PHASES = [
    ("witness generation", ["witgen_s"]),
    ("GKR sumcheck",       ["ph_gkr_generic", "ph_gkr_matmul"]),
    ("Lasso",              ["ph_lasso"]),
    ("commitments",        ["ph_hyrax_commit", "ph_hyrax_open", "standalone_commit_s"]),
    ("range prover",       ["range_s"]),
]

SPEED = os.path.join(RES, "speedup_analysis_models")   # figs/, logs/, paper_arith/, same_arith/
SUITES = {
    # The paper arithmetization on every arm AND the baseline (the only circuit
    # main_gkr_split builds; produced by paper_arith/_queue/run_paper_arith.sh
    # + run_paper_mono.sh): LLMSplitBlock, probe half/half layout, merge=1, merged range set.
    # Baseline = OUR binary at splits=1 under the same layout, wall clock from PHASE markers
    # (the same quantity as the arms' overlapped_total_s). modular-zkp's own binary is the
    # separate cross-check in same_arith/ (plot_same_arith.py).
    "paper": dict(
        groups=["gpt2", "opt350m", "opt1b3", "opt2b7"],
        pretty={"gpt2": "GPT-2", "opt350m": "OPT-350M", "opt1b3": "OPT-1.3B", "opt2b7": "OPT-2.7B"},
        figs=os.path.join(SPEED, "figs"),
        path=lambda g, log: os.path.join(SPEED, "paper_arith", "logs", f"{g}_{log}"),
        base_wall=True,
    ),
    # Sequence-length sweep under the same arithmetization; len64 rows are the paper
    # suite's own logs, other lengths carry a _len<L> infix (produced by the figures
    # queues -- tools/results/figures/logs holds the complete set).
    "paper_seqlen": dict(
        groups=["gpt2_len64", "gpt2_len128", "gpt2_len256",
                "opt350m_len64", "opt350m_len128", "opt350m_len256",
                "opt1b3_len64", "opt1b3_len128"],
        pretty={"gpt2_len64": "GPT-2\nlen 64", "gpt2_len128": "GPT-2\nlen 128", "gpt2_len256": "GPT-2\nlen 256",
                "opt350m_len64": "OPT-350M\nlen 64", "opt350m_len128": "OPT-350M\nlen 128", "opt350m_len256": "OPT-350M\nlen 256",
                "opt1b3_len64": "OPT-1.3B\nlen 64", "opt1b3_len128": "OPT-1.3B\nlen 128"},
        figs=os.path.join(SPEED, "figs", "seqlen"),
        path=lambda g, log: os.path.join(RES, "figures", "logs",
                                         (g.replace("_len64", "") if g.endswith("_len64") else g) + "_" + log),
        base_wall=True,
    ),
}

PART = re.compile(r"^PARTITION idx=.*$", re.M)
PHASE_END = r"^PHASE .*ev=end\s+phase=%s\s.*\bdur_s=([\d.]+)"


# ---------------------------------------------------------------- parsing ----
def read(path):
    return open(path, errors="replace").read() if os.path.exists(path) else None


def mz_walls(text):
    """(witgen_wall, prove_wall) from a per-line-timestamped modular-zkp split-demo log,
    or None. Whole-run wall clock: witgen = process start -> 'Parallel verification
    start' (probe pass + every create()), prove = that line -> 'Results:'. Requires
    every partition's 'All verification passed'."""
    L = [(float(m.group(1)), m.group(2)) for m in
         (re.match(r"(\d+\.\d+) (.*)", ln) for ln in text.splitlines()) if m]
    if not L:
        return None
    t0 = L[0][0]
    tp = next((t for t, s in L if s.startswith("Parallel verification start")), None)
    tr = next((t for t, s in L if s.startswith("Results:")), None)
    n = next((int(m.group(1)) for _, s in L for m in [re.search(r"splits=(\d+)", s)] if m), 0)
    passes = sum(1 for _, s in L if "All verification passed" in s)
    if tp is None or tr is None or passes < n:
        return None
    return (tp - t0, tr - tp)


def arm0_times(path, wall=False):
    """(witgen_s, prove_s, partial) or None. wall=True: PHASE witgen/prove durations
    (whole-run wall clock) instead of witgen_s + makespan_s (prover timers)."""
    text = read(path)
    if text is None:
        return None
    if wall and "All verification passed" in text:
        w = re.search(PHASE_END % "witgen", text, re.M); p = re.search(PHASE_END % "prove", text, re.M)
        if w and p:
            return (float(w.group(1)), float(p.group(1)), False)
    mz = mz_walls(text)                      # modular-zkp's driver, timestamped stdout
    if mz:
        return (mz[0], mz[1], False)
    og = re.search(r"^WITGEN: .*\bwitgen_total_s=([\d.]+) prove_wall_s=([\d.]+)", text, re.M)
    if og and "All verification passed" in text:
        return (float(og.group(1)), float(og.group(2)), False)
    wm = re.search(r"^PARTITION .*\bwitgen_s=([\d.]+)", text, re.M)
    sm = re.search(r"^SPLIT_SUMMARY makespan_s=([\d.]+)", text, re.M)
    if wm and sm:
        return (float(wm.group(1)), float(sm.group(1)), False)
    ph = {}
    for name in ("witgen", "sequence_kv_commit", "range_prove", "execute_commit", "verify_gkr"):
        m = re.search(PHASE_END % name, text, re.M)
        if m:
            ph[name] = float(m.group(1))
    if "witgen" in ph and "verify_gkr" in ph:
        return (ph["witgen"], sum(v for k, v in ph.items() if k != "witgen"), True)
    return None


def split_arm_times(path):
    """(witgen_s, prove_s) or None; total exact, split a proxy (see module doc)."""
    text = read(path)
    if text is None:
        return None
    m = re.search(r"^SPLIT_OVERLAPPED .*\boverlapped_total_s=([\d.]+) \bstatus=(\w+)", text, re.M)
    if not m or m.group(2) != "OK":
        return None
    total = float(m.group(1))
    built = [float(x) for x in re.findall(r"^OVERLAPPED_BUILT .*\bt_us=([\d.]+)", text, re.M)]
    wg = min(max(built) if built else 0.0, total)
    return (wg, total - wg)


def phases_of(path):
    """Per-partition phase averages [s] in PHASES order, or None."""
    text = read(path)
    if text is None:
        return None
    rows = []
    for line in PART.findall(text):
        d = {}
        for tok in line.split():
            k, _, v = tok.partition("=")
            try:
                d[k] = float(v)
            except ValueError:
                pass
        rows.append(d)
    if not rows:
        return None
    return [sum(sum(r.get(k, 0.0) for k in ks) for r in rows) / len(rows) for _n, ks in PHASES]


def base_label(suite):
    return suite.get("base_label", ARM_LABELS[0])


def load_group(suite, g):
    """{'base': (wg, pv, partial)|None, 'arms': [(label, wg, pv)]}"""
    wall = suite.get("base_wall", False)
    base = arm0_times(suite["path"](g, ARM_FILES[0][0]), wall) or arm0_times(suite["path"](g, ARM0_FALLBACK), wall)
    arms = []
    for log, label in ARM_FILES[1:]:
        t = split_arm_times(suite["path"](g, log))
        if t is None:
            print(f"  [{g}] skip {log} -- no valid data", file=sys.stderr)
            continue
        arms.append((label, t[0], t[1]))
    return {"base": base, "arms": arms}


# ---------------------------------------------------------------- figures ----
def short(label):
    return label.replace(" + ", "\n+ ").replace(" (", "\n(")


def fig_witgen_prove(suite, g, d):
    rows = []
    if d["base"] and not d["base"][2]:          # a partial monolith is not a time
        rows.append((base_label(suite), d["base"][0], d["base"][1]))
    rows += list(d["arms"])
    if not rows:
        return None
    fig, ax = ps.figure()
    _c, tops = ps.stacked_bars(ax, [short(r[0]) for r in rows], ["witness generation", "proving"],
                               [[r[1] for r in rows], [r[2] for r in rows]])
    ps.top_labels(ax, range(len(rows)), tops, fstr="{:,.0f}")
    ax.set_ylabel("Time (s)")
    ps.grid(ax); ps.headroom(ax); ps.legend(ax)
    return ps.save(fig, os.path.join(suite["figs"], f"witgen_prove_{g}"))


def fig_phases(suite, g):
    labels, mats = [], []
    for log, label in ARM_FILES[1:]:
        v = phases_of(suite["path"](g, log))
        if v is not None:
            labels.append(label); mats.append(v)
    if not mats:
        return None
    fig, ax = ps.figure()
    _c, tops = ps.stacked_bars(ax, [short(l) for l in labels], [n for n, _k in PHASES],
                               [[m[k] for m in mats] for k in range(len(PHASES))])
    ps.top_labels(ax, range(len(labels)), tops, fstr="{:,.0f}")
    ax.set_ylabel("Time per partition (s)")
    ps.grid(ax); ps.headroom(ax); ps.legend(ax)
    return ps.save(fig, os.path.join(suite["figs"], f"phases_{g}"))


def fig_times_all(suite, per):
    groups = [g for g in suite["groups"] if g in per and per[g]["arms"]]
    if not groups:
        return None
    vals = []
    for label in ARM_LABELS:
        row = []
        for g in groups:
            d = per[g]
            if label == ARM_LABELS[0]:
                row.append(d["base"][0] + d["base"][1] if d["base"] and not d["base"][2] else None)
            else:
                row.append({l: wg + pv for l, wg, pv in d["arms"]}.get(label))
        vals.append(row)
    fig, ax = ps.figure()
    labels = [base_label(suite)] + ARM_LABELS[1:]
    rects, _idx = ps.grouped_bars(ax, [suite["pretty"][g] for g in groups], labels, vals)
    ax.set_yscale("log")
    for label in labels:
        ps.value_labels(ax, rects[label], fstr="{:,.0f}", rotation=90)
    ax.set_ylabel("Time (s)")
    # five entries do not fit one row at 12 pt bold on a 15 in canvas -> two rows
    ps.grid(ax); ps.headroom(ax, log=True, log_factor=40.0); ps.legend(ax, ncol=3)
    return ps.save(fig, os.path.join(suite["figs"], "times_all"))


def fig_speedup_all(suite, per):
    groups = [g for g in suite["groups"]
              if g in per and per[g]["base"] and not per[g]["base"][2] and per[g]["arms"]]
    if not groups:
        return None
    series = ARM_LABELS[1:]
    vals = []
    for label in series:
        row = []
        for g in groups:
            b = per[g]["base"][0] + per[g]["base"][1]
            m = {l: wg + pv for l, wg, pv in per[g]["arms"]}
            row.append(b / m[label] if label in m else None)
        vals.append(row)
    names = [suite["pretty"][g] for g in groups]
    fig, ax = ps.figure()
    rects, _idx = ps.grouped_bars(ax, names, series, vals)
    for label in series:
        ps.value_labels(ax, rects[label], fstr="{:.2f}")
    ax.axhline(1.0, color="black", linestyle="dashed", linewidth=1)
    ax.set_ylabel("Speedup")
    ps.grid(ax); ps.headroom(ax); ps.legend(ax)
    return ps.save(fig, os.path.join(suite["figs"], "speedup_all"))


# ------------------------------------------------------------------ tables ----
def print_table(suite, g, d):
    base = d["base"]
    base_t = base[0] + base[1] if base else None
    partial = bool(base and base[2])
    print(f"\n=== {g} ===")
    if base is None:
        print("(no monolithic baseline; absolute times only)")
    elif partial:
        print("(monolith is PARTIAL -- died in Lasso; lower bound, speedups are >=)")
    print(f"{'arm':40s} {'witgen':>8} {'prove':>8} {'total':>8} {'speedup':>9}")
    if base:
        print(f"{base_label(suite) + (' (partial)' if partial else ''):40s} {base[0]:8.1f} {base[1]:8.1f} {base_t:8.1f} {'':>3}{1.0:5.2f}x")
    for label, wg, pv in d["arms"]:
        t = wg + pv
        sp = f"{'>=' if partial else '':>3}{base_t / t:5.2f}x" if base_t else f"{'-':>9}"
        print(f"{label:40s} {wg:8.1f} {pv:8.1f} {t:8.1f} {sp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=SUITES, default="paper")
    ap.add_argument("--only", help="one group (e.g. opt1b3, len128)")
    ap.add_argument("--no-figs", action="store_true")
    a = ap.parse_args()
    suite = SUITES[a.suite]
    groups = [a.only] if a.only else suite["groups"]
    if a.only and a.only not in suite["groups"]:
        sys.exit(f"unknown group {a.only!r} for suite {a.suite}: {suite['groups']}")
    if not a.no_figs:
        ps.apply(); os.makedirs(suite["figs"], exist_ok=True)
    per = {}
    for g in groups:
        d = load_group(suite, g)
        if not d["arms"] and not d["base"]:
            print(f"  [{g}] no data", file=sys.stderr)
            continue
        per[g] = d
        print_table(suite, g, d)
        if not a.no_figs:
            for p in (fig_witgen_prove(suite, g, d), fig_phases(suite, g)):
                if p:
                    print(p)
    if not a.no_figs and not a.only:
        for p in (fig_times_all(suite, per), fig_speedup_all(suite, per)):
            if p:
                print(p)


if __name__ == "__main__":
    main()
