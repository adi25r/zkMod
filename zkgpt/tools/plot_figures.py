#!/usr/bin/env python3
"""All paper figures (3, 5, 6, 7, 8) from one place.

    python3 tools/plot_figures.py            # every figure
    python3 tools/plot_figures.py 5 7        # a subset

Reads only tools/results/figures/logs/ (the consolidated log directory written by the
queues in tools/results/figures/queue/) and writes tools/results/figures/figs/
(<name>.png, .pdf and a .md table with the exact values). tools/results/figures/

Log naming (all produced by bin/main_gkr_split / _witgen; one run per file):
    <model>_t<T>_cpu.log / <model>_gpuwg_full.log      Fig 3 monolith thread series
    <model>[_len<L>][_s<n>]_arm0_mono.log                 CPU monolith (splits=1)
    <model>[_len<L>][_s<n>]_arm4_gpuwg_gpuprove.log       ours: GPU witgen + GPU prove

Metric conventions (shared by every figure):
    witgen / prove       PHASE ev=end phase=witgen|prove dur_s (wall clock)
    e2e response time    ours = overlapped_total_s (first fork .. last verification),
                         monolith = witgen + prove; BOTH + transmission of every proof
                         over one 100 Mbps link (LINK_BPS)
    proof size           sum of per-partition proof_kb (one proof per partition)
    verifier time        max over partitions of verifier_s: each sub-proof + its boundary
                         commitment verifies independently, so a verifier runs them in parallel
                         (verifier_serial_s = the sum, for a single-threaded verifier)
    memory               host_peak_mib (VmHWM of the partition process) and
                         gpu_peak_mib (device high-water), max over partitions
A run counts only when it finished (status=OK / SPLIT_SUMMARY) and every partition
printed "All verification passed".
"""
import os, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "tools/results/figures")
LOGS = os.environ.get("FIG_LOGS", os.path.join(BASE, "logs"))     # FIG_LOGS / FIG_OUT: e.g. the 48-core rerun
FIGS = os.environ.get("FIG_OUT", os.path.join(BASE, "figs"))

LINK_BPS = 100e6          # proof transmission: one 100 Mbps link, all proofs serialised over it
HOST_GIB, DEV_GIB = 723, 64
MODELS = [("gpt2", "GPT-2 (124M)"), ("opt350m", "OPT-350M"), ("opt1b3", "OPT-1.3B"), ("opt2b7", "OPT-2.7B")]
LBL = dict(MODELS)
LBL2 = {"gpt2": "GPT-2\n124M", "opt350m": "OPT\n350M", "opt1b3": "OPT\n1.3B", "opt2b7": "OPT\n2.7B"}
NAN = float("nan")

KV = re.compile(r"(\w+)=([-0-9.eE+]+)")
PH = re.compile(r"^PHASE part=(-?\d+) .*ev=end .*phase=(\w+) .*dur_s=([0-9.]+)")


# ---------------------------------------------------------------- log parsing
def parse(name):
    """One run. None unless it finished and every partition verified."""
    path = os.path.join(LOGS, name)
    if not os.path.exists(path):
        return None
    parts, phases, summ, passed, split = {}, {}, {}, 0, {}
    for line in open(path, errors="replace"):
        if line.startswith("TIMING_SPLIT "):
            d = {k: float(v) for k, v in KV.findall(line)}
            split[int(d["idx"])] = d
        if line.startswith("PARTITION "):
            d = {k: float(v) for k, v in KV.findall(line)}
            parts[int(d["idx"])] = d
        m = PH.match(line)
        if m:
            k = (int(m.group(1)), m.group(2))
            phases[k] = phases.get(k, 0.0) + float(m.group(3))
        if line.startswith("SPLIT_OVERLAPPED "):
            summ.update({k: float(v) for k, v in KV.findall(line)})
            summ["ok"] = "status=OK" in line
        if line.startswith("SPLIT_SUMMARY "):
            summ["ok"] = True                      # sequential path ran to the end
        if "Total verifier time:" in line:
            summ["verifier_sum"] = float(line.split(":")[1].split()[0])
        if "Combined proof size:" in line:
            summ["proof_kb"] = float(line.split(":")[1].split()[0])
        if "All verification passed" in line:
            passed += 1
    if not parts or passed < len(parts) or not summ.get("ok"):
        return None
    return dict(parts=parts, phases=phases, summ=summ, n=len(parts), split=split)


def suffix(length=64, splits=None):
    return ("" if length == 64 else f"_len{length}") + ("" if splits is None else f"_s{splits}")


def mono(model, length=64):
    return parse(f"{model}{suffix(length)}_arm0_mono.log")


ARMS = {1: ("arm1_cpuwg_cpuprove", "CPU witgen + CPU prove"),
        2: ("arm2_cpuwg_gpuprove", "CPU witgen + GPU prove"),
        3: ("arm3_gpuwg_cpuprove", "GPU witgen + CPU prove"),
        4: ("arm4_gpuwg_gpuprove", "GPU witgen + GPU prove")}


def arm(model, k, length=64, splits=None):
    return parse(f"{model}{suffix(length, splits)}_{ARMS[k][0]}.log")


def ours(model, length=64, splits=None):
    return arm(model, 4, length, splits)


def tx_s(proof_kb):
    return proof_kb * 1024 * 8 / LINK_BPS if proof_kb == proof_kb else NAN


def proof_kb(r, model=None):
    """Total proof size. Per-partition proof_kb summed; monolith's summary line."""
    P = r["parts"]
    if all("proof_kb" in p for p in P.values()):
        return sum(p["proof_kb"] for p in P.values())
    if "proof_kb" in r["summ"]:
        return r["summ"]["proof_kb"]
    # Proof size is fixed by the circuit + partitioning, not by which device proved
    # it: an arm1-3 log from before the field existed takes arm4's value.
    o = ours(model) if model else None
    if o and o is not r and o["n"] == r["n"]:
        return proof_kb(o, model)
    return NAN


def verifier_s(r, model=None, serial=False):
    """Verifier wall time. Partitions verify independently (sub-proof + boundary commitment),
    so the parallel verifier time is the slowest partition; serial=True gives the sum."""
    P = r["parts"]
    agg = sum if serial else max
    if all("verifier_s" in p for p in P.values()):
        return agg(p["verifier_s"] for p in P.values())
    if len(P) == 1 and "verifier_sum" in r["summ"]:
        return r["summ"]["verifier_sum"]
    # The verifier is the same host code in every arm; same fallback as proof_kb.
    o = ours(model) if model else None
    if o and o is not r and o["n"] == r["n"]:
        return verifier_s(o, model, serial)
    return NAN


def metrics(r, model=None):
    """witgen, prove, e2e (incl. proof transfer), proof_kb, verifier_s, host GiB, dev GiB."""
    P = r["parts"]
    wg = r["phases"].get((0, "witgen"), NAN)
    prove = max(r["phases"].get((i, "prove"), NAN) for i in P)
    pk = proof_kb(r, model)
    if "overlapped_total_s" in r["summ"]:
        e2e = r["summ"]["overlapped_total_s"]
    else:                                          # monolith / splits=1 sequential path
        e2e = r["phases"].get((0, "witgen"), 0) + r["phases"].get((0, "prove"), 0)
    e2e += tx_s(pk)
    host = max(p["host_peak_mib"] for p in P.values()) / 1024
    dev = max(p["gpu_peak_mib"] for p in P.values()) / 1024
    return dict(witgen=wg, prove=prove, e2e=e2e, proof_kb=pk, tx=tx_s(pk),
                verifier=verifier_s(r, model), verifier_serial=verifier_s(r, model, True), host=host, dev=dev, n=r["n"])


def fmt(v, d=1):
    return "n/a" if v is None or v != v else f"{v:.{d}f}"


def save(fig, name, table):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    open(os.path.join(FIGS, f"{name}.md"), "w").write("\n".join(table) + "\n")
    plt.close(fig)
    print(f"-- {name}"); print("\n".join(table))


# ---------------------------------------------------------------- paper style
# Bars: blue gradient (light = monolith/host, dark = zkMod/device), monolith hatched.
# Lines: black, told apart by marker + line style. No titles or value labels except zkMod's speedup over the
# monolith (_sp) -- the .md table carries the rest. Legends stay inside the axes.
plt.rcParams.update({
    "font.size": 8, "font.weight": "bold", "axes.labelweight": "bold", "axes.titleweight": "bold",
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6, "axes.linewidth": 0.6, "lines.linewidth": 1.0, "lines.markersize": 4,
    "hatch.linewidth": 0.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    "legend.frameon": True, "legend.framealpha": 1.0, "legend.edgecolor": "black",
    "legend.fancybox": False, "legend.borderpad": 0.2, "legend.handlelength": 1.2, "legend.handleheight": 0.6,
    "legend.handletextpad": 0.4, "legend.labelspacing": 0.15, "legend.columnspacing": 0.6, "legend.borderaxespad": 0.3,
})
BAR = dict(edgecolor="black", linewidth=0.5)
# monolith + the four arms, light -> dark
SHADE = {"base": ("#E6ECF5", "////"), 1: ("#B3C6E0", ""), 2: ("#7FA0CB", ""), 3: ("#4F7CAC", ""), 4: ("#2B5180", "")}
WG_FILL, PV_FILL = ("#B0C4DE", ""), ("#4F7CAC", "")
C_BASE, C_OURS = "black", "black"         # line plots stay black (hollow vs filled markers); only bars use the blues
SHORT = {1: "CPU+CPU", 2: "CPU+GPU", 3: "GPU+CPU", 4: "GPU+GPU"}
MARK = {"gpt2": "o", "opt350m": "s", "opt1b3": "^", "opt2b7": "D"}
NAME = {"gpt2": "GPT-2", "opt350m": "OPT-350M", "opt1b3": "OPT-1.3B", "opt2b7": "OPT-2.7B"}
NAME2 = {"gpt2": "GPT-2", "opt350m": "OPT\n350M", "opt1b3": "OPT\n1.3B", "opt2b7": "OPT\n2.7B"}


def _bar(ax, x, h, w, shade, bottom=None, label=None):
    fc, hatch = shade
    ax.bar(x, h, w, bottom=bottom, color=fc, hatch=hatch, label=label, **BAR)


def _grid(ax, axis="y"):
    ax.grid(axis=axis, color="0.85", lw=0.4, which="major"); ax.set_axisbelow(True)


def _sp(ax, x, y, v, rotation=0, size=6.5, dy=2, va="bottom"):
    """'3.65x' next to (x, y); v = monolith / zkMod."""
    if v != v:
        return
    ax.annotate(f"{v:.2f}\u00d7", (x, y), xytext=(0, dy), textcoords="offset points", ha="center", va=va,
                fontsize=size, rotation=rotation)


def _oom(ax, x):
    """No bar (the run does not fit in memory): a hyphen at the baseline."""
    ax.text(x, 0.01, "-", ha="center", va="bottom", fontsize=9,
            transform=ax.get_xaxis_transform())


# ---------------------------------------------------------------- Figure 3
def fig3():
    """Baseline thread scaling: CPU monolith (splits=1) end-to-end speedup (witgen + prove) over its
    own run at the smallest thread count, logs/<m>_t<T>_cpu.log (run_fig3.sh, T = 8, 16, ... 192)."""
    models = [m for m, _ in MODELS[:2]]
    data = {}
    for m in models:
        for f in os.listdir(LOGS):
            t = re.fullmatch(rf"{m}_t(\d+)_cpu\.log", f)
            r = parse(f) if t and int(t.group(1)) % 8 == 0 else None
            if r:
                data[(m, int(t.group(1)))] = (r["phases"][(0, "witgen")], r["phases"][(0, "prove")])
    if not data:
        print("fig3: no data"); return
    fig, ax = plt.subplots(figsize=(3.4, 1.4))
    tbl = ["| model | threads | witgen s | prove s | total s | speedup vs fewest threads |", "|---|---|---|---|---|---|"]
    t0s = {}
    for m, c in zip(models, (C_BASE, C_OURS)):
        ts = sorted(t for mm, t in data if mm == m)
        if not ts:
            continue
        t0 = t0s[m] = ts[0]; base = sum(data[(m, t0)])
        ax.plot(ts, [base / sum(data[(m, t)]) for t in ts], MARK[m] + "-", color=c, mfc=c, ms=3,
                label=NAME[m])
        for t in ts:
            a_, b_ = data[(m, t)]
            tbl.append(f"| {LBL[m]} | {t} | {a_:.1f} | {b_:.1f} | {a_ + b_:.1f} | {base / (a_ + b_):.2f}x |")
    ticks = [t for t in (8, 16, 32, 64, 128, 192) if any(tt == t for _, tt in data)]
    ax.set_xscale("log", base=2); ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Host threads")
    ax.set_ylabel(f"Speedup (vs {min(t0s.values())}T)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}×"))
    _grid(ax, "both"); ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "fig3_baseline_breakdown", tbl)


# ---------------------------------------------------------------- Figure 5
def fig5():
    """All models at len 64: proof size, verifier time, e2e -- the CPU
    monolith and all four partitioned arms (CPU/GPU witgen x CPU/GPU prove)."""
    series = [("base", "Monolith")] + [(k, SHORT[k]) for k in ARMS]
    rows = {}
    for m, _ in MODELS:
        b = mono(m)
        if b: rows[(m, "base")] = metrics(b, m)
        for k in ARMS:
            r = arm(m, k)
            if r: rows[(m, k)] = metrics(r, m)
    if not rows:
        print("fig5: no data"); return
    panels = [("proof_kb", "Proof size (KB)"), ("verifier", "Verifier time (s)"),
              ("e2e", "End-to-end time (s)")]
    fig = plt.figure(figsize=(7.0, 3.4))
    gs = fig.add_gridspec(2, 2)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
    x = np.arange(len(MODELS)); w = 0.8 / len(series)
    for ax, (key, ylabel) in zip(axes, panels):
        for j, (ser, lab) in enumerate(series):
            vals = [rows[(m, ser)][key] if (m, ser) in rows else NAN for m, _ in MODELS]
            xs = x - 0.4 + w * (j + 0.5)
            _bar(ax, xs, vals, w * 0.95, SHADE[ser], label=lab)
            if key == "e2e" and ser != "base":
                for xx, v, (m, _) in zip(xs, vals, MODELS):
                    if (m, "base") in rows:
                        _sp(ax, xx, v, rows[(m, "base")]["e2e"] / v, rotation=90, size=6.5)
            if ser == "base":
                for xx, v in zip(xs, vals):
                    if v != v:
                        _oom(ax, xx)
        ax.set_xticks(x); ax.set_xticklabels([NAME2[m] for m, _ in MODELS])
        if key != "proof_kb":
            ax.set_yscale("log")
        ax.set_ylabel(ylabel); _grid(ax)
    # headroom on the e2e panel for the legend
    lo, hi = axes[2].get_ylim(); axes[2].set_ylim(lo, hi * 15)
    axes[2].legend(loc="upper right", ncol=5)
    fig.tight_layout(w_pad=0.8, h_pad=0.8)
    tbl = ["| model | series | proof KB | verifier s (parallel: max) | verifier s (serial: sum) | proof tx s @100Mbps | e2e s (incl. tx) |",
           "|---|---|---|---|---|---|---|"]
    for m, _ in MODELS:
        for ser, lab in series:
            if (m, ser) in rows:
                r = rows[(m, ser)]
                tbl.append(f"| {LBL[m]} | {lab} | {fmt(r['proof_kb'], 2)} | {fmt(r['verifier'], 2)} | {fmt(r['verifier_serial'], 2)} | {fmt(r['tx'], 3)} | {fmt(r['e2e'], 2)} |")
    save(fig, "fig5_e2e", tbl)
    # LaTeX proof-overhead table (paper layout, tab:proof-overheads-zkgpt); \X rows = zkMod GPU+GPU
    # (proof size is device-independent), verify = parallel (max over partitions), "-" = no run
    def cell(r, k, d):
        return "-" if r is None or r[k] != r[k] else f"{r[k]:.{d}f}"
    L = [r"\begin{table}[t]", r"  \centering", r"  \small", r"  \setlength{\tabcolsep}{4pt}",
         r"  \renewcommand{\arraystretch}{0.9}", r"  \begin{tabular}{llccc}", r"    \toprule",
         r"    Model & Impl. & Proof (KB) & Transmit (s) & Verify (s) \\", r"    \midrule"]
    for i, (m, _) in enumerate(MODELS):
        bm, zm = rows.get((m, "base")), rows.get((m, 4))
        row = lambda r: f"{cell(r, 'proof_kb', 0)} & {cell(r, 'tx', 3)} & {cell(r, 'verifier', 2)}"
        L.append(rf"    \multirow{{2}}{{*}}{{{NAME[m]}}} & Baseline & {row(bm)} \\ & \X & {row(zm)} \\")
        L.append(r"    \midrule" if i < len(MODELS) - 1 else r"    \bottomrule")
    L += [r"  \end{tabular}",
          r"  \caption{Proof and verification overheads at sequence length 64 for zkGPT. Proof sizes also apply to the GPU implementation. Transmission assumes a 100\,Mbps network.}",
          r"  \label{tab:proof-overheads-zkgpt}", r"\end{table}"]
    open(os.path.join(FIGS, "fig5_table.md"), "w").write("\n".join(L) + "\n")


# ---------------------------------------------------------------- Figure 5a / 5b / 5c
def _fig5_panel(rows, k, name):
    """Fig 5a/5b: monolith vs one zkMod configuration, all models at len 64. Each bar is
    stacked compute + proof transfer (100 Mbps); speedup over the monolith above the zkMod bar."""
    fig, ax = plt.subplots(figsize=(3.4, 1.6))
    x = np.arange(len(MODELS)); w = 0.38
    for j, (ser, lab, shade) in enumerate((("base", "Monolith", SHADE["base"]),
                                           (k, f"zkMod {SHORT[k]}", SHADE[k]))):
        xs = x + (j - 0.5) * w
        comp = [rows[(m, ser)]["e2e"] - rows[(m, ser)]["tx"] if (m, ser) in rows else NAN for m, _ in MODELS]
        tx = [rows[(m, ser)]["tx"] if (m, ser) in rows else NAN for m, _ in MODELS]
        _bar(ax, xs, comp, w * 0.95, shade, label=lab)
        ax.bar(xs, tx, w * 0.95, bottom=comp, color="#1A2F4A", edgecolor="black", linewidth=0.5,
               label="Transfer" if j == 0 else None)
        for (m, _), xx, c, t in zip(MODELS, xs, comp, tx):
            if c != c:
                _oom(ax, xx)
            elif ser != "base" and (m, "base") in rows:
                ax.text(xx, (c + t) * 1.15, f"{rows[(m, 'base')]['e2e'] / (c + t):.1f}x",
                        ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels([NAME[m] for m, _ in MODELS], rotation=40, ha="right",
                                         rotation_mode="anchor", fontsize=6.5)
    ax.set_yscale("log"); ax.set_ylabel("End-to-end time (s)"); _grid(ax)
    # log autoscale on bars from 0 can cut off the shortest bar; floor below the smallest compute time
    lo = min(r["e2e"] - r["tx"] for r in rows.values())
    ax.set_ylim(lo / 3, ax.get_ylim()[1] * 4)
    ax.legend(loc="upper left", ncol=3)
    fig.tight_layout()
    tbl = ["| model | series | compute s | transfer s @100Mbps | e2e s | speedup vs monolith |",
           "|---|---|---|---|---|---|"]
    for m, _ in MODELS:
        for ser, lab in (("base", "Monolith"), (k, f"zkMod {SHORT[k]}")):
            if (m, ser) in rows:
                r = rows[(m, ser)]
                sp = (f"{rows[(m, 'base')]['e2e'] / r['e2e']:.2f}x"
                      if ser != "base" and (m, "base") in rows else "")
                tbl.append(f"| {LBL[m]} | {lab} | {r['e2e'] - r['tx']:.2f} | {r['tx']:.3f} | {r['e2e']:.2f} | {sp} |")
    save(fig, name, tbl)


def fig5ab():
    """Fig 5a: monolith vs zkMod CPU+CPU; Fig 5b: monolith vs zkMod GPU+GPU. All models, len 64."""
    rows = {}
    for m, _ in MODELS:
        b = mono(m)
        if b: rows[(m, "base")] = metrics(b, m)
        for k in ARMS:
            r = arm(m, k)
            if r: rows[(m, k)] = metrics(r, m)
    if not rows:
        print("fig5ab: no data"); return
    _fig5_panel(rows, 1, "fig5a_e2e_cpu")
    _fig5_panel(rows, 4, "fig5b_e2e_gpu")


def fig5c():
    """Fig 5c: end-to-end time vs total host-thread budget (16..192), len 64. Dashed = CPU
    monolith (logs/<m>_t<T>_cpu.log, run_fig3.sh); solid = zkMod CPU+CPU
    (logs/<m>_t<T>_arm1_cpuwg_cpuprove.log, run_fig5_threads.sh)."""
    budgets = sorted({int(t) for f in os.listdir(LOGS)
                      for t in re.findall(r"^\w+?_t(\d+)_arm1_cpuwg_cpuprove\.log$", f)})   # Fig 3's step-8 monolith series has no zkMod pair
    budgets = [t for t in budgets if t >= 16]          # the 1..8 tail belongs to Fig 3
    rows = {}
    for m, _ in MODELS:
        for t in budgets:
            for ser, f in (("base", f"{m}_t{t}_cpu.log"), ("ours", f"{m}_t{t}_{ARMS[1][0]}.log")):
                r = parse(f)
                if r: rows[(m, ser, t)] = metrics(r, m)["e2e"]
    if not rows:
        print("fig5c: no data"); return
    fig, ax = plt.subplots(figsize=(3.4, 1.8))
    for m, _ in MODELS:
        for ser, ls, lab in (("base", "--", "monolith"), ("ours", "-", "zkMod CPU+CPU")):
            pts = [(t, rows[(m, ser, t)]) for t in budgets if (m, ser, t) in rows]
            if pts:
                ax.plot([t for t, _ in pts], [v for _, v in pts], MARK[m] + ls, color=C_BASE if ser == "base" else C_OURS,
                        mfc="white" if ser == "base" else C_OURS, label=f"{NAME[m]} {lab}")
    ax.set_xscale("log", base=2); ax.set_xticks(budgets); ax.set_xticklabels([str(t) for t in budgets])
    ax.minorticks_off()
    ax.set_yscale("log"); ax.set_xlabel("Host threads"); ax.set_ylabel("End-to-end time (s)"); _grid(ax)
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi * 6)
    ax.legend(loc="upper left", ncol=2, fontsize=5.5)
    fig.tight_layout()
    tbl = ["| model | series | threads | e2e s (incl. tx) |", "|---|---|---|---|"]
    for m, _ in MODELS:
        for ser, lab in (("base", "monolith"), ("ours", "zkMod CPU+CPU")):
            for t in budgets:
                if (m, ser, t) in rows:
                    tbl.append(f"| {LBL[m]} | {lab} | {t} | {rows[(m, ser, t)]:.1f} |")
    save(fig, "fig5c_e2e_threads", tbl)

# ---------------------------------------------------------------- Figure 6
def fig6():
    """Memory: (a) vs model size at len 64, (b) vs input size; host and device peaks per partition."""
    from matplotlib.lines import Line2D
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 1.8))
    x = np.arange(len(MODELS)); w = 0.26
    tbl = ["| model | len | zkMod host peak/partition GiB | zkMod device-memory peak/partition GiB | parts | mono host peak GiB |",
           "|---|---|---|---|---|---|"]
    for i, (m, _) in enumerate(MODELS):
        o, bm = ours(m), mono(m)
        bh = metrics(bm)["host"] if bm else NAN
        if not o:
            continue
        mt = metrics(o)
        for xx, val, shade, lab in ((x[i] - w, bh, SHADE["base"], "Monolith host"),
                                    (x[i], mt["host"], SHADE[2], "zkMod host"),
                                    (x[i] + w, mt["dev"], SHADE[4], "zkMod GPU")):
            _bar(a, xx, val, w, shade, label=lab if i == 0 else None)
            if val != val:
                _oom(a, xx)
        tbl.append(f"| {LBL[m]} | 64 | {mt['host']:.1f} | {mt['dev']:.1f} | {mt['n']} | {fmt(bh)} |")
    a.axhline(HOST_GIB, color="black", ls=":", lw=0.8, label="Host RAM")
    a.axhline(DEV_GIB, color="black", ls="--", lw=0.8, label="GPU memory")
    a.set_xticks(x); a.set_xticklabels([NAME[m] for m, _ in MODELS])
    a.set_yscale("log"); a.set_ylabel("Peak memory (GiB)"); _grid(a)
    a.set_ylim(top=a.get_ylim()[1] * 25)
    a.legend(loc="upper left", ncol=3)
    lens = [64, 128, 256]
    models = []
    for m in ("gpt2", "opt350m", "opt1b3"):
        pts = [(L, metrics(ours(m, L))) for L in lens if ours(m, L)]
        if pts:
            models.append(m)
            b.plot([L for L, _ in pts], [t["host"] for _, t in pts], MARK[m] + "-", color=C_BASE, mfc="white")
            b.plot([L for L, _ in pts], [t["dev"] for _, t in pts], MARK[m] + "--", color=C_OURS, mfc=C_OURS)
            for L, t in pts:
                if L != 64:
                    bm = mono(m, L)
                    tbl.append(f"| {LBL[m]} | {L} | {t['host']:.1f} | {t['dev']:.1f} | {t['n']} | {fmt(metrics(bm)['host']) if bm else 'n/a'} |")
    mh = [(L, metrics(mono("gpt2", L))["host"]) for L in lens if mono("gpt2", L)]
    if mh:
        b.plot([t[0] for t in mh], [t[1] for t in mh], MARK["gpt2"] + ":", color="0.4", mfc="0.4")
    b.set_xscale("log", base=2); b.set_xticks(lens); b.set_xticklabels([str(L) for L in lens])
    b.minorticks_off()
    b.set_yscale("log"); b.set_xlabel("Sequence length"); b.set_ylabel("Peak memory (GiB)"); _grid(b, "both")
    b.set_ylim(top=b.get_ylim()[1] * 5)
    h = [Line2D([], [], color="black", ls="", marker=MARK[m], mfc="white", label=NAME[m]) for m in models]
    h += [Line2D([], [], color=C_BASE, ls="-", label="zkMod host"),
          Line2D([], [], color=C_OURS, ls="--", label="zkMod GPU")]
    if mh:
        h.append(Line2D([], [], color="0.4", ls=":", label="Monolith host"))
    b.legend(handles=h, loc="upper left", ncol=3)
    fig.tight_layout()
    save(fig, "fig6_memory", tbl)


# ---------------------------------------------------------------- Figure 7
def fig7():
    """Sequence-length scaling (64/128/256): zkMod speedup over the monolith, grouped bars per model."""
    lens = [64, 128, 256]
    tbl = ["| model | len | mono e2e s | zkMod e2e s | e2e speedup |",
           "|---|---|---|---|---|"]
    models = ("gpt2", "opt350m")
    sp = {}
    for m in models:
        bm = {L: metrics(mono(m, L)) for L in lens if mono(m, L)}
        om = {L: metrics(ours(m, L)) for L in lens if ours(m, L)}
        sp[m] = {L: bm[L]["e2e"] / om[L]["e2e"] for L in lens if L in bm and L in om}
        for L in lens:
            if L in bm or L in om:
                be = bm[L]["e2e"] if L in bm else NAN
                oe = om[L]["e2e"] if L in om else NAN
                f = lambda p, q: "n/a" if (p != p or q != q) else f"{p / q:.2f}x"
                tbl.append(f"| {LBL[m]} | {L} | {fmt(be)} | {fmt(oe)} | {f(be, oe)} |")
    if not any(sp.values()):
        print("fig7: no data"); return
    # grouped bars: one group per model, one bar per sequence length (light -> dark)
    fig, a = plt.subplots(1, 1, figsize=(3.4, 1.7))
    x = np.arange(len(models)); w = 0.8 / len(lens)
    for j, (L, shade) in enumerate(zip(lens, (SHADE[2], SHADE[3], SHADE[4]))):
        xs = x - 0.4 + w * (j + 0.5)
        hs = [sp[m].get(L) for m in models]
        _bar(a, [p for p, h in zip(xs, hs) if h], [h for h in hs if h], w * 0.92, shade, label=f"Length {L}")
        for p, h in zip(xs, hs):
            if h: _sp(a, p, h, h, size=6)
    a.set_xticks(x); a.set_xticklabels([NAME[m] for m in models]); a.tick_params(axis="x", length=0)
    a.set_ylabel("Speedup over zkGPT"); _grid(a)
    a.set_ylim(0, max(v for d in sp.values() for v in d.values()) * 1.4)
    a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}\u00d7"))
    a.legend(loc="upper left", ncol=3)
    fig.tight_layout()
    save(fig, "fig7_seqlen", tbl)


# ---------------------------------------------------------------- Figure 7b
def fig7b():
    """Sequence-length scaling: witgen and prove time vs length, monolith vs ours, one axes."""
    lens = [64, 128, 256]
    models = [m for m, _ in MODELS[:2]]
    series = ["mono", "ours"]
    data = {}
    for m in models:
        for L in lens:
            bm = mono(m, L)
            if bm:
                data[(m, L, "mono")] = (bm["phases"].get((0, "witgen"), NAN), bm["phases"].get((0, "prove"), NAN))
            om = ours(m, L)
            if om:
                # wall clock, not summed: every partition's witgen starts at fork and its dur_s
                # includes waiting on the chain, so the last one ends the witgen phase; the rest
                # of overlapped_total_s is the proving that outlasts it.
                wg = max(om["phases"].get((i, "witgen"), 0) for i in om["parts"])
                pv = om["summ"]["overlapped_total_s"] - wg
                data[(m, L, "ours")] = (wg, pv)
    if not data:
        print("fig7b: no data"); return
    # one time series per (model, system, phase): marker = model, dashed/hollow = monolith,
    # solid/filled = zkMod, colour = phase
    fig, a = plt.subplots(1, 1, figsize=(3.4, 1.9))
    PH_COL = {0: "#8FA9CC", 1: "#2B5180"}
    for m in models:
        for s in series:
            ls, mono_s = ("--", True) if s == "mono" else ("-", False)
            for k in (0, 1):
                pts = [(L, data[(m, L, s)][k]) for L in lens if (m, L, s) in data]
                if pts:
                    c = PH_COL[k]
                    a.plot(*zip(*pts), MARK[m] + ls, color=c, mfc="white" if mono_s else c, ms=3.5, lw=1)
    a.set_xscale("log", base=2); a.set_xticks(lens); a.set_xticklabels([str(L) for L in lens]); a.minorticks_off()
    a.set_xlim(lens[0] / 1.25, lens[-1] * 1.25)
    a.set_yscale("log"); a.set_xlabel("Sequence length"); a.set_ylabel("Time (s)")
    _grid(a, "both")
    from matplotlib.lines import Line2D
    h = ([Line2D([], [], color=PH_COL[0], lw=2, label="Witgen"), Line2D([], [], color=PH_COL[1], lw=2, label="Prove")]
         + [Line2D([], [], color="k", ls=ls, lw=1, label=l) for ls, l in (("--", "Monolith"), ("-", "zkMod"))]
         + [Line2D([], [], color="k", marker=MARK[m], ls="", ms=3.5, mfc="white", label=NAME[m]) for m in models])
    lo, hi = a.get_ylim(); a.set_ylim(lo, hi * 4)
    a.legend(handles=h, loc="upper left", ncol=3, fontsize=6, columnspacing=0.8, handlelength=1.8)
    fig.tight_layout()
    tbl = ["| model | len | series | witgen s | prove s | total s |", "|---|---|---|---|---|---|"]
    for m in models:
        for L in lens:
            for s in series:
                if (m, L, s) in data:
                    wg, pv = data[(m, L, s)]
                    tbl.append(f"| {LBL[m]} | {L} | {'zkMod' if s == 'ours' else s} | {fmt(wg)} | {fmt(pv)} | {fmt(wg + pv)} |")
    save(fig, "fig7b_seqlen_breakdown", tbl)


# ---------------------------------------------------------------- Figure 8
def fig8():
    """Partition scaling, len 64, GPU+GPU: speedup over the monolith, grouped bars per model
    (gpt2 1..12 partitions, opt350m 6/8/12/24)."""
    pts = [(n, metrics(ours("gpt2", 64, n))) for n in range(1, 13) if ours("gpt2", 64, n)]
    if not pts:
        print("fig8: no data"); return
    ns = [n for n, _ in pts]; e2e = [t["e2e"] for _, t in pts]
    host = [t["host"] for _, t in pts]; dev = [t["dev"] for _, t in pts]
    bm = mono("gpt2")
    if not bm:
        print("fig8: no monolith"); return
    be = metrics(bm)["e2e"]
    # grouped bars, one group per model and one bar per partition count it was run at; the bar's
    # count is its tick label (the two models use different counts), the model sits below its group
    groups = [("gpt2", be, {n: be / t["e2e"] for n, t in pts})]
    b350 = mono("opt350m")
    if b350:
        e350 = metrics(b350)["e2e"]
        r350 = {n: ours("opt350m", 64, n) for n in (6, 8, 12)}
        r350[24] = ours("opt350m")
        groups.append(("opt350m", e350, {n: e350 / metrics(r)["e2e"] for n, r in sorted(r350.items()) if r}))
    fig, a = plt.subplots(1, 1, figsize=(7.0, 1.7))
    x0, ticks, labs = 0.0, [], []
    for m, _, sp in groups:
        xs = x0 + np.arange(len(sp))
        _bar(a, xs, list(sp.values()), 0.75, SHADE[4])
        for xx, v in zip(xs, sp.values()):
            _sp(a, xx, v, v, rotation=90, size=5.5)
        ticks += list(xs); labs += [str(n) for n in sp]
        a.text(xs.mean(), -0.22, NAME[m], transform=a.get_xaxis_transform(), ha="center", va="top")
        x0 = xs[-1] + 2
    a.set_xticks(ticks); a.set_xticklabels(labs, fontsize=6.5); a.tick_params(axis="x", length=0)
    a.set_ylabel("Speedup over zkGPT"); _grid(a)
    a.set_ylim(0, max(v for _, _, sp in groups for v in sp.values()) * 1.35)
    a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}\u00d7"))
    fig.tight_layout()
    be = metrics(bm)["e2e"] if bm else NAN
    tbl = ["| partitions | e2e s | speedup vs 1 | speedup vs monolith | host peak/partition GiB | device-memory peak/partition GiB |",
           "|---|---|---|---|---|---|"]
    for n, t in pts:
        tbl.append(f"| {n} | {t['e2e']:.1f} | {(e2e[0] / t['e2e']):.2f}x | {fmt(be / t['e2e'], 2)}x | {t['host']:.1f} | {t['dev']:.1f} |")
    save(fig, "fig8_partitions", tbl)


def fig8k():
    """Partition scaling at the best BlendySc stage count k per partition count: gpt2 len 64, GPU+GPU.
    Candidates: logs_ksweep/k<k>/ (--matmul-k = --lasso-k = k) and the default run (matmul 4 / lasso 8)."""
    global LOGS
    bm = mono("gpt2")
    if not bm:
        print("fig8k: no monolith"); return
    be = metrics(bm)["e2e"]
    kdir = os.path.join(BASE, "logs_ksweep")
    ks = sorted(int(d[1:]) for d in os.listdir(kdir) if d.startswith("k")) if os.path.isdir(kdir) else []
    best, allk, home = {}, {}, LOGS
    for n in range(1, 13):
        cand = []
        r = ours("gpt2", 64, n)
        if r: cand.append(("4/8", metrics(r)["e2e"]))
        for k in ks:
            LOGS = os.path.join(kdir, f"k{k}")
            try:
                r = ours("gpt2", 64, n)
            finally:
                LOGS = home
            if r: cand.append((str(k), metrics(r)["e2e"]))
        if cand:
            allk[n] = cand
            best[n] = min(cand, key=lambda c: c[1])
    if not best:
        print("fig8k: no data"); return
    ns = sorted(best); sp = [be / best[n][1] for n in ns]
    fig, a = plt.subplots(1, 1, figsize=(3.4, 1.6))
    a.plot(ns, sp, "o-", color=C_OURS, mfc=C_OURS, ms=3.5)
    for n, v in zip(ns, sp):
        a.annotate(f"k={best[n][0]}\n{v:.2f}\u00d7", (n, v), xytext=(0, 3), textcoords="offset points",
                   ha="center", va="bottom", fontsize=5)
    a.set_xlabel("Partitions (GPT-2)"); a.set_ylabel("Speedup over zkGPT")
    a.set_xticks(range(1, 13)); a.set_ylim(0, max(sp) * 1.45); _grid(a, "both")
    a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}\u00d7"))
    fig.tight_layout()
    tbl = ["| partitions | best k | e2e s | speedup vs monolith | all k: e2e s |", "|---|---|---|---|---|"]
    for n in ns:
        tbl.append(f"| {n} | {best[n][0]} | {best[n][1]:.1f} | {be / best[n][1]:.2f}x | "
                   + ", ".join(f"{k}: {e:.1f}" for k, e in allk[n]) + " |")
    save(fig, "fig8k_partitions_bestk", tbl)


# ---------------------------------------------------------------- paper: all models, one arm
def _vs_mono(k, name):
    """Monolith vs one partitioned arm at len 64, every model: response time incl. proof transmission."""
    rows = {}
    for m, _ in MODELS:
        b, r = mono(m), arm(m, k)
        if b: rows[(m, "base")] = metrics(b, m)
        if r: rows[(m, "ours")] = metrics(r, m)
    if not rows:
        print(f"{name}: no data"); return
    # proof transmission (<= 0.2 s) is included in each bar but is invisible at this scale; the .md table splits it out
    fig, ax = plt.subplots(figsize=(3.4, 1.6))
    x = np.arange(len(MODELS)); w, gap = 0.4, 0.04   # narrow: bars nearly touch within a model
    for j, (ser, shade, lab) in enumerate((("base", SHADE["base"], "Monolith"), ("ours", SHADE[k], f"zkMod {SHORT[k]}"))):
        xs = x + (j - 0.5) * (w + gap)
        vals = [rows[(m, ser)]["e2e"] if (m, ser) in rows else NAN for m, _ in MODELS]
        _bar(ax, xs, vals, w, shade, label=lab)
        for xx, v in zip(xs, vals):
            if v != v:
                _oom(ax, xx)
    # on zkMod's bar: speedup with a monolith, its time without (OOM)
    for xi, (m, _) in enumerate(MODELS):
        if (m, "ours") not in rows:
            continue
        o, xo = rows[(m, "ours")]["e2e"], x[xi] + (w + gap) / 2
        if (m, "base") in rows:   # centred over the pair, above the (taller) monolith bar
            b = rows[(m, "base")]["e2e"]
            _sp(ax, x[xi], max(o, b), b / o)
        else:
            ax.annotate(f"{o:.0f}" if o >= 100 else f"{o:.1f}", (xo, o), xytext=(0, 2),
                        textcoords="offset points", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels([NAME2[m] for m, _ in MODELS])
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    ax.set_yscale("log"); ax.set_ylabel("Response time (s)", labelpad=1); _grid(ax)
    lo = min(r["e2e"] for r in rows.values()) / 3
    ax.set_ylim(lo, max(r['e2e'] for r in rows.values()) * 2.2)   # labels only; the legend sits over the bars
    # top, over OPT-1.3B's empty monolith slot: clear of every bar and label
    ax.legend(loc="upper center", bbox_to_anchor=((2 + 0.5) / len(MODELS), 1.0), fontsize=5.5, borderaxespad=0.2,
              borderpad=0.25, handlelength=1.0, handletextpad=0.3, labelspacing=0.25)
    fig.tight_layout()
    tbl = ["| model | series | compute s | proof tx s | response s | speedup |", "|---|---|---|---|---|---|"]
    for m, _ in MODELS:
        for ser in ("base", "ours"):
            if (m, ser) in rows:
                r = rows[(m, ser)]
                sp = (f"{rows[(m, 'base')]['e2e'] / r['e2e']:.2f}x" if ser == "ours" and (m, "base") in rows else "")
                tbl.append(f"| {LBL[m]} | {'Monolith' if ser == 'base' else 'zkMod ' + SHORT[k]} | {fmt(r['e2e'] - r['tx'], 2)} | {fmt(r['tx'], 3)} | {fmt(r['e2e'], 2)} | {sp} |")
    save(fig, name, tbl)


def fig_cpu64():
    """Experiment 1: all models at len 64, monolith vs zkMod CPU witgen + CPU prove."""
    _vs_mono(1, "seqlen64allcpu")


def fig_gpu64():
    """Experiment 3: all models at len 64, monolith vs zkMod GPU witgen + GPU prove."""
    _vs_mono(4, "seqlen64allgpu")


# ---------------------------------------------------------------- paper: thread scaling
def fig_threads(models=("gpt2", "opt350m"), name="threads_cpu"):
    """Experiment 2: zkMod CPU+CPU speedup over the CPU monolith per host-thread budget (len 64).
    zkMod at t threads: logs/threads/t<t>/, except the budget fig5's arm1 ran at (its own log)."""
    tdir = os.path.join(LOGS, "threads")
    budgets = sorted({int(d[1:]) for d in os.listdir(tdir) if re.fullmatch(r"t\d+", d)} if os.path.isdir(tdir) else set())
    # the fig5 threads queue writes flat logs/<model>_t<T>_arm1_cpuwg_cpuprove.log
    budgets = sorted(set(budgets) | {int(t) for f in os.listdir(LOGS)
                                      for t in re.findall(rf"^(?:gpt2|opt350m)_t(\d+)_{ARMS[1][0]}\.log$", f)})
    rows = {}
    for m in models:
        a1 = arm(m, 1)
        if a1:
            t0 = int(re.search(r"threads=(\d+)", open(os.path.join(LOGS, f"{m}_{ARMS[1][0]}.log")).read()).group(1))
            rows[(m, t0, "ours")] = metrics(a1, m)
            budgets = sorted(set(budgets) | {t0})
        for t in budgets:
            r = parse(os.path.join("threads", f"t{t}", f"{m}_{ARMS[1][0]}.log")) or parse(f"{m}_t{t}_{ARMS[1][0]}.log")
            if r: rows[(m, t, "ours")] = metrics(r, m)
            b = parse(f"{m}_t{t}_cpu.log")
            if b: rows[(m, t, "base")] = metrics(b, m)
    if not rows:
        print("threads: no data"); return
    # speedup = monolith / zkMod CPU+CPU at the same thread budget (only budgets that have both)
    sp = {(m, t): rows[(m, t, "base")]["e2e"] / rows[(m, t, "ours")]["e2e"]
          for m in models for t in budgets if (m, t, "base") in rows and (m, t, "ours") in rows}
    budgets = sorted({t for _, t in sp})
    fig, ax = plt.subplots(figsize=(3.4, 1.5))
    for m, ls in zip(models, ("-", "--")):
        pts = [(t, sp[(m, t)]) for t in budgets if (m, t) in sp]
        ax.plot([t for t, _ in pts], [v for _, v in pts], MARK[m] + ls, color="black", mfc="black", label=NAME[m])
        below = m == "gpt2" and len(models) > 1          # the lower line labels below, the upper above
        for t, v in pts:
            _sp(ax, t, v, v, size=5.5, dy=-4 if below else 3, va="top" if below else "bottom")
    ax.set_xscale("log", base=2); ax.set_xticks(budgets); ax.set_xticklabels([str(t) for t in budgets])
    ax.minorticks_off()
    ax.set_xlabel("Host threads"); ax.set_ylabel("Speedup over monolith"); _grid(ax)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}\u00d7"))
    ax.set_ylim(min(1, min(sp.values()) * 0.85), max(sp.values()) * 1.12)
    ax.legend(loc="upper left")
    fig.tight_layout()
    tbl = ["| model | threads | mono response s | zkMod CPU+CPU response s | speedup |", "|---|---|---|---|---|"]
    for m in models:
        for t in budgets:
            b = rows.get((m, t, "base")); o = rows.get((m, t, "ours"))
            be, oe = (b["e2e"] if b else NAN), (o["e2e"] if o else NAN)
            tbl.append(f"| {LBL[m]} | {t} | {fmt(be)} | {fmt(oe)} | {'n/a' if be != be or oe != oe else f'{be / oe:.2f}x'} |")
    save(fig, name, tbl)


# ---------------------------------------------------------------- paper: partition scaling, both models
def fig_partitions_all():
    """Experiment 4: GPU+GPU response time vs partition count at len 64, one panel per model; the monolith is the
    first bar and every zkMod bar carries its speedup over it."""
    data = {}
    for m in ("gpt2", "opt350m"):
        pts = {n: metrics(r, m)["e2e"] for n in range(1, 33) for r in [ours(m, 64, n)] if r}
        d = ours(m)
        if d: pts.setdefault(d["n"], metrics(d, m)["e2e"])
        if pts:
            b = mono(m)
            data[m] = (metrics(b, m)["e2e"] if b else NAN, pts)
    if not data:
        print("partitions_all: no data"); return
    fig, axes = plt.subplots(1, len(data), figsize=(7.0, 1.7), squeeze=False,
                             gridspec_kw={"width_ratios": [len(p) + 1 for _, p in data.values()]})
    tbl = ["| model | partitions | response s | speedup vs monolith |", "|---|---|---|---|"]
    for ax, (m, (be, pts)) in zip(axes[0], data.items()):
        ns = sorted(pts)
        _bar(ax, 0, be, 0.7, SHADE["base"], label="Monolith")
        for i, n in enumerate(ns, start=1):
            _bar(ax, i, pts[n], 0.7, SHADE[4], label="zkMod GPU+GPU" if i == 1 else None)
            _sp(ax, i, pts[n], be / pts[n], rotation=90, size=6)
            tbl.append(f"| {LBL[m]} | {n} | {pts[n]:.1f} | {'n/a' if be != be else f'{be / pts[n]:.2f}x'} |")
        ax.set_xticks(range(len(ns) + 1)); ax.set_xticklabels(["Mono"] + [str(n) for n in ns])
        ax.set_xlabel(f"Partitions ({NAME[m]})"); _grid(ax)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
    axes[0, 0].set_ylabel("Response time (s)")
    axes[0, 0].legend(loc="upper right")
    fig.tight_layout(w_pad=0.8)
    save(fig, "partitions_gpu", tbl)


# ---------------------------------------------------------------- flame chart
# The TIMING_SPLIT hierarchy, summed over a run's partitions: witness-generation phase
# -> witness generation / input layer init / circuit construction, proof -> commit /
# Lasso / sumcheck / input layer check / range check, each split into its components.
# Partitions overlap in time, so the root is total work, not end-to-end latency.
def _flame_tree(sp, pa):
    S = lambda k: sum(d.get(k, 0.0) for d in sp.values())
    P = lambda k: sum(d.get(k, 0.0) for d in pa.values())
    return ("all partitions", [
        ("witgen phase", [
            ("witness generation", [("array setup", S("wg_init_s")),
                                    ("values + gate lists", S("wg_values_s"))]),
            ("input layer init", [("input layer + weights", S("wg_inlayer_s")),
                                  ("input copy", S("wg_copyin_s"))]),
            ("circuit construction", [("merge", S("wg_merge_s")), ("initSubset", S("wg_subset_s")),
                                      ("output padding", S("wg_pad_s")),
                                      ("sync, count, teardown", S("wg_sync_s") + S("wg_misc_s") + S("wg_teardown_s"))]),
            ("other", [("carry + rest", S("wg_carry_s") + S("wg_other_s"))]),
        ]),
        ("proof", [
            ("commit", [("input prefix", S("pf_standalone_commit_s")), ("output", S("pf_output_commit_s")),
                        ("input (executeCommit)", S("pf_exec_commit_s")), ("in-GKR Hyrax", S("pf_hyrax_commit_s"))]),
            ("Lasso", [("Lasso", S("pf_lasso_s"))]),
            ("sumcheck", [("GKR generic", P("ph_gkr_generic")), ("GKR matmul", P("ph_gkr_matmul"))]),
            ("input layer check", [("Hyrax open", S("pf_input_check_s"))]),
            ("range check", [("build", S("pf_range_build_s")), ("prove", S("pf_range_prove_s"))]),
            ("other", [("verification", S("pf_verify_s")),
                       ("unattributed", max(0.0, S("pf_other_s") - S("pf_verify_s")))]),
        ]),
    ])


_FLAME_SHORT = {"witness generation": "witgen", "input layer init": "input init",
                "circuit construction": "circuit", "input layer check": "input chk",
                "values + gate lists": "values", "input layer + weights": "inp+wts",
                "input copy": "copy", "initSubset": "subset", "output padding": "pad",
                "input prefix": "prefix", "input (executeCommit)": "input",
                "in-GKR Hyrax": "Hyrax", "range check": "range", "GKR generic": "generic",
                "GKR matmul": "matmul", "Hyrax open": "open", "verification": "verify",
                "unattributed": "rest", "output": "out"}


def _flame_val(node):
    name, kids = node
    return kids if isinstance(kids, float) else sum(_flame_val(k) for k in kids)


def fig_flame():
    runs = [(m, ours(m)) for m, _ in MODELS]
    runs = [(m, r) for m, r in runs if r and len(r["split"]) == r["n"]]
    if not runs:
        print("flame: no data"); return
    # witgen family light greys, proof family darker greys; siblings alternate
    fam = {"witgen phase": 0.97, "proof": 0.78}
    fig, axes = plt.subplots(len(runs), 1, figsize=(13, 2.5 * len(runs)), squeeze=False)
    tbl = ["| model | level | node | seconds (summed) | % of total |", "|---|---|---|---|---|"]
    for ax, (m, r) in zip(axes[:, 0], runs):
        tree = _flame_tree(r["split"], r["parts"])
        total = _flame_val(tree)

        def draw(node, x0, depth, family, k):
            name, kids = node
            v = _flame_val(node)
            if family is None:
                color = "white"
            else:
                color = str(fam[family] - 0.05 * (depth - 1) - (0.06 if k % 2 else 0))
            ax.barh(depth, v, left=x0, height=0.92, color=color, edgecolor="black", linewidth=0.4)
            frac = v / total if total else 0
            # ~0.0053 of the axis width per bold 7.5 pt character on this 13 in canvas;
            # a label that would spill shrinks to its value, then to nothing.
            per_char = 0.0053 * 13 / fig.get_size_inches()[0]
            val = f"{v:.1f} s"
            short = _FLAME_SHORT.get(name, name)
            if frac > per_char * (max(len(name), len(val)) + 2):
                lab = f"{name}\n{val}"
            elif frac > per_char * (max(len(short), len(val)) + 1):
                lab = f"{short}\n{val}"
            elif frac > per_char * (len(val) + 1):
                lab = val
            else:
                lab = None
            if lab:
                ax.text(x0 + v / 2, depth, lab, ha="center", va="center", fontsize=7.5, color="black")
            tbl.append(f"| {LBL[m]} | {depth} | {name} | {v:.2f} | {100 * frac:.1f} |")
            if isinstance(kids, list):
                x = x0
                for j, c in enumerate(kids):
                    draw(c, x, depth + 1, family or c[0], j)
                    x += _flame_val(c)

        draw(tree, 0.0, 0, None, 0)
        ax.set_xlim(0, total); ax.set_ylim(-0.5, 3.5)
        ax.set_yticks(range(4)); ax.set_yticklabels(["Run", "Phase", "Category", "Component"], fontsize=8)
        ax.set_xlabel("Time summed over partitions (s)", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
        for sp_ in ("top", "right", "left"):
            ax.spines[sp_].set_visible(False)
        ax.set_title(NAME[m], fontsize=8, loc="left")
    fig.tight_layout()
    save(fig, "fig_flame_timing_split", tbl)


FIGURES = {"3": fig3, "5": fig5, "5ab": fig5ab, "5c": fig5c, "6": fig6, "7": fig7, "7b": fig7b, "8": fig8, "8k": fig8k, "flame": fig_flame,
           "cpu64": fig_cpu64, "gpu64": fig_gpu64, "threads": fig_threads,
           "threads_gpt2": lambda: fig_threads(("gpt2",), "threads_cpu_gpt2"), "partitions": fig_partitions_all}

if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    want = sys.argv[1:] or list(FIGURES)
    for k in want:
        FIGURES[k]()
    print("wrote", FIGS)
