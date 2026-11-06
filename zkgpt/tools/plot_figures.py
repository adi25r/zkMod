#!/usr/bin/env python3
"""All zkGPT figures from one place.

    python3 tools/plot_figures.py                                  # every figure
    python3 tools/plot_figures.py seqlen64allcpu seqlen64allgpu    # a subset

Reads tools/results/figures/logs/ (written by the queues in tools/results/figures/queue/) and writes
tools/results/figures/figs/ (<name>.png, .pdf and a .md table with the exact values).

Log naming (all produced by bin/main_gkr_split / _witgen; one run per file):
    <model>_t<T>_cpu.log                               monolith thread series
    <model>[_len<L>][_s<n>]_arm0_mono.log                 CPU monolith (splits=1)
    <model>[_len<L>][_s<n>]_arm1_cpuwg_cpuprove.log       zkMod CPU witgen + CPU prove
    <model>[_len<L>][_s<n>]_arm4_gpuwg_gpuprove.log       zkMod GPU witgen + GPU prove

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
from matplotlib.ticker import ScalarFormatter

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


def save(fig, name, table, tight=True):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"), dpi=200,
                    bbox_inches="tight" if tight else None)
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
# monolith, CPU+CPU, GPU+GPU
SHADE = {"base": ("#E6ECF5", "////"), 1: ("#B3C6E0", ""), 4: ("#2B5180", "")}
WG_FILL, PV_FILL = ("#B0C4DE", ""), ("#4F7CAC", "")
C_BASE, C_OURS = "black", "black"         # line plots stay black (hollow vs filled markers); only bars use the blues
SHORT = {1: "CPU+CPU", 4: "GPU+GPU"}
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


def zkgptbaselineprofile():
    """CPU monolith response time (witgen + prove) versus host thread count."""
    models = [m for m, _ in MODELS[:2]]
    data = {}
    for m in models:
        for f in os.listdir(LOGS):
            t = re.fullmatch(rf"{m}_t(\d+)_cpu\.log", f)
            r = parse(f) if t and (int(t.group(1)) in (1, 2, 4) or int(t.group(1)) % 8 == 0) else None
            if r:
                data[(m, int(t.group(1)))] = (r["phases"][(0, "witgen")], r["phases"][(0, "prove")])
    if not data:
        print("zkgptbaselineprofile: no data"); return
    fig, ax = plt.subplots(figsize=(623.001 / 200, 204.001 / 200), dpi=200)
    fig.subplots_adjust(left=0.18, right=0.988, bottom=0.29, top=0.97)
    tbl = ["| model | threads | witgen s | prove s | total s | speedup vs fewest threads |", "|---|---|---|---|---|---|"]
    t0s = {}
    for m, c in zip(models, ("black", "#2B5180")):
        ts = sorted(t for mm, t in data if mm == m)
        if not ts:
            continue
        t0 = t0s[m] = ts[0]; base = sum(data[(m, t0)])
        # Connect the 32/64/128 endpoints directly; keep all measurements in the table.
        plot_ts = [t for t in ts if not (32 < t < 64 or 64 < t < 128)]
        markers = [j for j, t in enumerate(plot_ts) if t in (1, 2, 4, 8, 16, 32, 64, 128, ts[-1])]
        ax.plot(plot_ts, [sum(data[(m, t)]) for t in plot_ts], MARK[m] + "-", color=c,
                mfc="white", ms=2.7, lw=0.85, markevery=markers,
                label=NAME[m])
        for t in ts:
            a_, b_ = data[(m, t)]
            tbl.append(f"| {LBL[m]} | {t} | {a_:.1f} | {b_:.1f} | {a_ + b_:.1f} | {base / (a_ + b_):.2f}x |")
    ticks = [t for t in (1, 2, 4, 8, 16, 32, 64, 128) if any(tt == t for _, tt in data)]
    ax.set_xscale("log", base=2); ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlim(1, 128)
    ax.set_yscale("log")
    ax.set_yticks([200, 500, 1000, 2000])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    for tick, label in zip(ticks, ax.get_xticklabels()):
        if tick == 128:
            label.set_horizontalalignment("right")
    ax.minorticks_off()
    ax.set_xlabel("Host threads", fontsize=7, labelpad=2)
    ax.set_ylabel("Response time (s)", fontsize=6.5, labelpad=3)
    ax.yaxis.label.set_y(0.35)
    ax.tick_params(axis="both", labelsize=6.5, pad=2)
    totals = [sum(times) for (_, t), times in data.items() if t <= 128]
    ax.set_ylim(min(totals) / 1.15, max(totals) * 1.22)
    _grid(ax, "both"); ax.legend(loc="upper right", fontsize=6)
    save(fig, "zkgptbaselineprofile", tbl, tight=False)


def _seqlen64(k, name, series):
    """All models, len 64: end-to-end time of the CPU monolith vs one zkMod mode. The zkMod bar carries the
    speedup over the monolith; where there is no monolith (no speedup) it carries its time in seconds."""
    rows = {}
    for m, _ in MODELS:
        b = mono(m)
        if b: rows[(m, "base")] = metrics(b, m)
        r = arm(m, k)
        if r: rows[(m, k)] = metrics(r, m)
    if not rows:
        print(f"{name}: no data"); return
    fig, ax = plt.subplots(figsize=(623.001 / 200, 204.001 / 200), dpi=200)
    fig.subplots_adjust(left=0.18, right=0.988, bottom=0.25, top=0.97)
    x = np.arange(len(MODELS)); w = 0.8 / len(series)
    for j, (ser, lab) in enumerate(series):
        vals = [rows[(m, ser)]["e2e"] if (m, ser) in rows else NAN for m, _ in MODELS]
        xs = x - 0.4 + w * (j + 0.5)
        _bar(ax, xs, vals, w * 0.85, SHADE[ser], label=lab)
        if ser == k:
            for xx, v, (m, _) in zip(xs, vals, MODELS):
                if v != v:
                    continue
                if (m, "base") in rows:
                    ax.annotate(f"{rows[(m, 'base')]['e2e'] / v:.2f}×",
                                (xx - 0.2, max(v, rows[(m, "base")]["e2e"])),
                                xytext=(0, 3), textcoords="offset points",
                                ha="center", va="bottom", fontsize=5.5,
                                bbox=dict(facecolor="white", edgecolor="none", pad=0.3), zorder=10)
                else:
                    ax.annotate(f"{v:.0f}s", (xx, v), xytext=(0, 2), textcoords="offset points",
                                ha="center", va="bottom", fontsize=5.5)
        else:
            for xx, v in zip(xs, vals):
                if v != v:
                    ax.text(xx, 40, "-", ha="center", va="bottom", fontsize=7, zorder=6)
    ax.set_xticks(x); ax.set_xticklabels([NAME2[m] for m, _ in MODELS], fontsize=6)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.tick_params(axis="y", labelsize=6.5, pad=2)
    ax.set_yscale("log"); ax.set_ylabel("Response time (s)", fontsize=6.5, labelpad=3); _grid(ax)
    ax.yaxis.label.set_y(0.35)
    ax.set_ylim(5, max(r["e2e"] for r in rows.values()) * 3)
    ax.set_xlim(-0.6, len(MODELS) - 0.4)
    ax.legend(loc="lower right", ncol=2, fontsize=6, borderpad=0.3, handlelength=1.2, columnspacing=0.8)
    tbl = ["| model | series | e2e s (incl. tx) |", "|---|---|---|"]
    for m, _ in MODELS:
        for ser, lab in series:
            if (m, ser) in rows:
                tbl.append(f"| {LBL[m]} | {lab} | {fmt(rows[(m, ser)]['e2e'], 2)} |")
    save(fig, name, tbl, tight=False)


def zkgpt_proof_overheads():
    """LaTeX proof-overhead table: monolith vs zkMod GPU+GPU, len 64. Proof size is device-independent; verify = parallel
    (max over partitions); "-" = no run."""
    rows = {}
    for m, _ in MODELS:
        b, r = mono(m), arm(m, 4)
        if b: rows[(m, "base")] = metrics(b, m)
        if r: rows[(m, 4)] = metrics(r, m)
    if not rows:
        print("zkgpt_proof_overheads: no data"); return
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
    open(os.path.join(FIGS, "zkgpt_proof_overheads.md"), "w").write("\n".join(L) + "\n")









def zkgptseqlenscaling():
    """Sequence-length scaling (64/128/256) at each model's best partition count (its default log): zkMod speedup over the monolith, grouped bars per model."""
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
        print("zkgptseqlenscaling: no data"); return
    # grouped bars: one group per model, one bar per sequence length (light -> dark)
    fig, a = plt.subplots(1, 1, figsize=(3.4, 1.3))
    x = np.arange(len(models)); w = 0.8 / len(lens)
    for j, (L, shade) in enumerate(zip(lens, (("#7FA0CB", ""), ("#4F7CAC", ""), SHADE[4]))):
        xs = x - 0.4 + w * (j + 0.5)
        hs = [sp[m].get(L) for m in models]
        _bar(a, [p for p, h in zip(xs, hs) if h], [h for h in hs if h], w * 0.92, shade, label=str(L))
    a.set_xticks(x); a.set_xticklabels([NAME[m] for m in models]); a.tick_params(axis="x", length=0)
    a.set_ylabel("Speedup over\nzkGPT"); _grid(a)
    a.set_ylim(0, max(v for d in sp.values() for v in d.values()) * 1.15)
    a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}\u00d7"))
    # over the bars' lower part, clear of every bar top and label
    a.legend(loc="lower right", ncol=3, framealpha=1)
    fig.tight_layout()
    save(fig, "zkgptseqlenscaling", tbl)




PARTITION_COUNTS = (1, 2, 8, 12, 16, 24)      # bars per model (a model without a run at a count leaves a gap)


def at(model, length, n):
    """zkMod GPU+GPU run at exactly n partitions: the default log (best config) when it ran at n, else the _s<n> log."""
    d = ours(model, length)
    return d if d and d.get("n") == n else ours(model, length, n)


def _ramp(n):
    lo, hi = (np.array([int(c[i:i + 2], 16) for i in (1, 3, 5)]) for c in (SHADE[1][0], SHADE[4][0]))
    return [("#%02X%02X%02X" % tuple(np.round(lo + (hi - lo) * t).astype(int)), "") for t in np.linspace(0, 1, n)]


def zkgptpartitionscaling():
    """Partition scaling, len 64, GPU+GPU: speedup over the CPU monolith, one group per model and one bar per
    partition count in PARTITION_COUNTS (1 = the GPU monolith), shaded light -> dark; the legend is the counts."""
    models = [m for m in ("gpt2", "opt350m") if mono(m)]
    if not models:
        print("zkgptpartitionscaling: no monolith"); return
    tbl = ["| model | partitions | e2e s | speedup vs monolith | host peak/partition GiB | device-memory peak/partition GiB |",
           "|---|---|---|---|---|---|"]
    sp = {}
    for m in models:
        be = metrics(mono(m))["e2e"]; sp[m] = {}
        for n in range(1, 33):
            r = at(m, 64, n)
            if not r:
                continue
            t = metrics(r)
            if n in PARTITION_COUNTS:
                sp[m][n] = be / t["e2e"]
            tbl.append(f"| {LBL[m]} | {n} | {t['e2e']:.1f} | {be / t['e2e']:.2f}x | {t['host']:.1f} | {t['dev']:.1f} |")
    counts = [n for n in PARTITION_COUNTS if any(n in sp[m] for m in models)]
    if not counts:
        print("zkgptpartitionscaling: no data"); return
    fig, a = plt.subplots(1, 1, figsize=(3.4, 1.3))
    x = np.arange(len(models)); w = 0.84 / len(counts)
    for j, (n, shade) in enumerate(zip(counts, _ramp(len(counts)))):
        xs = x - 0.42 + w * (j + 0.5)
        hs = [sp[m].get(n) for m in models]
        _bar(a, [p for p, h in zip(xs, hs) if h], [h for h in hs if h], w * 0.92, shade, label=str(n))
    a.set_xticks(x); a.set_xticklabels([NAME[m] for m in models]); a.tick_params(axis="x", length=0)
    a.set_ylabel("Speedup over\nzkGPT"); _grid(a)
    a.set_ylim(0, max(v for d in sp.values() for v in d.values()) * 1.15)
    a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}\u00d7"))
    a.legend(loc="lower right", ncol=len(counts), fontsize=6, columnspacing=1.0, handlelength=1.2,
             borderaxespad=0.3, framealpha=1)
    fig.tight_layout()
    save(fig, "zkgptpartitionscaling", tbl)




# ---------------------------------------------------------------- paper: all models, one arm






def cpuhostthreadscaling(models=("gpt2", "opt350m"), name="cpuhostthreadscaling"):
    """zkMod CPU+CPU speedup over the CPU monolith per host-thread budget (len 64).
    zkMod at t threads: logs/threads/t<t>/, except the default budget (its own log)."""
    tdir = os.path.join(LOGS, "threads")
    budgets = sorted({int(d[1:]) for d in os.listdir(tdir) if re.fullmatch(r"t\d+", d)} if os.path.isdir(tdir) else set())
    # run_cpuhostthreadscaling.sh writes flat logs/<model>_t<T>_arm1_cpuwg_cpuprove.log
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
        print("cpuhostthreadscaling: no data"); return
    # speedup = monolith / zkMod CPU+CPU at the same thread budget (only budgets that have both)
    sp = {(m, t): rows[(m, t, "base")]["e2e"] / rows[(m, t, "ours")]["e2e"]
          for m in models for t in budgets if (m, t, "base") in rows and (m, t, "ours") in rows}
    budgets = sorted({t for _, t in sp})
    fig, ax = plt.subplots(figsize=(623.001 / 200, 204.001 / 200), dpi=200)
    fig.subplots_adjust(left=0.18, right=0.988, bottom=0.29, top=0.97)
    for m, ls, color in zip(models, ("-", "--"), ("black", "#2B5180")):
        pts = [(t, sp[(m, t)]) for t in budgets if (m, t) in sp]
        ax.plot([t for t, _ in pts], [v for _, v in pts], MARK[m] + ls,
                color=color, mfc="white", ms=2.7, lw=0.85, label=NAME[m])
        below = m == "gpt2" and len(models) > 1          # the lower line labels below, the upper above
        for t, v in pts:
            ax.annotate(f"{v:.2f}×", (t, v), xytext=(0, -4 if below else 3),
                        textcoords="offset points", ha="center", va="top" if below else "bottom",
                        fontsize=5, bbox=dict(facecolor="white", edgecolor="none", pad=0.3), zorder=10)
    ax.set_xscale("log", base=2); ax.set_xticks(budgets); ax.set_xticklabels([str(t) for t in budgets])
    ax.minorticks_off()
    ax.set_xlabel("Host threads", fontsize=7, labelpad=2)
    ax.set_ylabel("Speedup", fontsize=7, labelpad=3)
    ax.tick_params(axis="both", labelsize=6.5, pad=2)
    ax.set_xlim(min(budgets) / 1.12, max(budgets) * 1.18)
    _grid(ax)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}\u00d7"))
    ax.set_ylim(0.8, max(sp.values()) * 1.2)
    ax.set_yticks([1, 2, 3, 4])
    ax.axhline(1, color="black", linestyle=":", linewidth=0.7)
    ax.legend(loc="upper left", fontsize=6)
    tbl = ["| model | threads | mono response s | zkMod CPU+CPU response s | speedup |", "|---|---|---|---|---|"]
    for m in models:
        for t in budgets:
            b = rows.get((m, t, "base")); o = rows.get((m, t, "ours"))
            be, oe = (b["e2e"] if b else NAN), (o["e2e"] if o else NAN)
            tbl.append(f"| {LBL[m]} | {t} | {fmt(be)} | {fmt(oe)} | {'n/a' if be != be or oe != oe else f'{be / oe:.2f}x'} |")
    save(fig, name, tbl, tight=False)


# ---------------------------------------------------------------- paper: partition scaling, both models


# The TIMING_SPLIT hierarchy, summed over a run's partitions: witness-generation phase
# -> witness generation / input layer init / circuit construction, proof -> commit /
# Lasso / sumcheck / input layer check / range check, each split into its components.
# Partitions overlap in time, so the root is total work, not end-to-end latency.








FIGURES = {"zkgptbaselineprofile": zkgptbaselineprofile,
           "seqlen64allcpu": lambda: _seqlen64(1, "seqlen64allcpu", [("base", "Baseline"), (1, "zkMod")]),
           "seqlen64allgpu": lambda: _seqlen64(4, "seqlen64allgpu", [("base", "Monolith"), (4, SHORT[4])]),
           "cpuhostthreadscaling": cpuhostthreadscaling,
           "zkgptseqlenscaling": zkgptseqlenscaling,
           "zkgptpartitionscaling": zkgptpartitionscaling,
           "zkgpt_proof_overheads": zkgpt_proof_overheads}

if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    want = sys.argv[1:] or list(FIGURES)
    for k in want:
        FIGURES[k]()
    print("wrote", FIGS)
