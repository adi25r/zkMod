#!/usr/bin/env python3
"""plot_membw.py -- whole-run memory-system utilisation over time, from wrap.sh output.

  plot_membw.py --logs DIR --out DIR --run base="Baseline" --run ours="Ours" [--skip-phase probe_witgen] [--ddr-peak 921.6]

Per run <prefix> it reads <prefix>.cpu.csv (perf -I), <prefix>.gpu.csv (sampler), <prefix>.start_epoch and, if the
program prints  `PHASE part=N pid=P ev=begin|end phase=NAME d=D t_us=EPOCH_US`  lines, <prefix>.log.
Writes membw_overview.{png,pdf,md} (all runs overlaid + summary table) and timeline_<prefix>.{png,pdf} per run
(partition phase timeline on top, CPU DRAM GB/s, GPU memory activity %, VRAM resident GiB underneath, one clock).
--skip-phase NAME: drop that phase and start every clock when it ends (e.g. a throwaway sizing pass).
CPU GB/s = sum(perf counts) * --bytes-per-count / interval (default 64 B = one cache line per count).
"""
import argparse, csv, os, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--logs", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--run", action="append", required=True, metavar="PREFIX=LABEL")
ap.add_argument("--skip-phase"); ap.add_argument("--ddr-peak", type=float, default=921.6, help="host DRAM peak GB/s (default 2x EPYC 9654 DDR5-4800)")
ap.add_argument("--bytes-per-count", type=float, default=64.0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
PH = re.compile(r"^PHASE\s+part=(-?\d+)\s+pid=\d+\s+ev=(begin|end)\s+phase=(\S+)\s+d=\d+\s+t_us=(\d+)")
COLOR = {"witgen": "#4E8F55", "carry_wait": "#B9C3CA", "token_wait": "#DCE3E7", "standalone_input_commit": "#6A5FBE",
         "output_commit": "#9086D8", "range_build": "#F0B878", "range_prove": "#D97B22", "execute_commit": "#B3ABE8",
         "verify_gkr": "#1F6B78", "verify_lasso": "#3AA0B0", "open_commit": "#7CC7D2", "boundary_link": "#B9C3CA"}
DRAW = ["witgen", "carry_wait", "token_wait", "standalone_input_commit", "output_commit", "range_build", "range_prove",
        "execute_commit", "boundary_link", "verify_gkr", "verify_lasso", "open_commit"]
LABEL = {"execute_commit": "exec", "standalone_input_commit": "", "output_commit": "", "range_build": "", "boundary_link": "", "token_wait": ""}
P = lambda pre, ext: os.path.join(a.logs, pre + ext)

def cpu(pre):
    acc = {}
    for r in csv.reader(l for l in open(P(pre, ".cpu.csv")) if not l.startswith("#")):
        if len(r) >= 4 and r[1].replace(".", "").isdigit(): acc[float(r[0])] = acc.get(float(r[0]), 0) + float(r[1])
    t = np.array(sorted(acc)); return t, np.array([acc[x] for x in t]) * a.bytes_per_count / 1e9 / np.diff(np.r_[0, t])

def gpu(pre):
    """-> t, mem_busy %, gpu_busy %, vram GiB (None if absent) -- or None when the run has no GPU sampler output"""
    if not os.path.exists(P(pre, ".gpu.csv")): return None
    s0 = float(open(P(pre, ".start_epoch")).read())
    rows = [l.split(",") for l in open(P(pre, ".gpu.csv")) if l.count(",") >= 2 and l.endswith("\n")]
    if not rows: return None
    m = np.array([[float(x) for x in r] for r in rows if len(r) == len(rows[0])])
    return m[:, 0] - s0, m[:, 1], m[:, 2], (m[:, 3] / 2**30 if m.shape[1] > 3 else None)

def spans(pre):
    if not os.path.exists(P(pre, ".log")): return {}, 0.0
    op, out, t0 = {}, {}, 0.0
    for l in open(P(pre, ".log"), errors="replace"):
        m = PH.match(l)
        if not m: continue
        p, ev, ph, t = int(m.group(1)), m.group(2), m.group(3), int(m.group(4)) / 1e6
        if a.skip_phase and ph == a.skip_phase and ev == "end": t0 = max(t0, t)
        if p < 0 or ph not in DRAW: continue
        if ev == "begin": op[(p, ph)] = t
        elif (p, ph) in op: out.setdefault(p, []).append((ph, op.pop((p, ph)), t))
    return out, t0

runs = [r.split("=", 1) for r in a.run]
fig, ax = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True)
rows = ["| run | wall s | CPU DRAM read GB/s mean | peak | % of DRAM peak (mean) | total read TB | mem activity % mean | peak | gpu busy % mean | VRAM peak GiB |", "|---|---|---|---|---|---|---|---|---|---|"]
for pre, name in runs:
    sp, t0e = spans(pre); s0 = float(open(P(pre, ".start_epoch")).read()); T0 = max(0.0, t0e - s0) if a.skip_phase else 0.0
    t, v = cpu(pre); m = t >= T0; t, v = t[m] - T0, v[m]
    g = gpu(pre)
    if g is not None: gm = g[0] >= T0; g = (g[0][gm] - T0, g[1][gm], g[2][gm], None if g[3] is None else g[3][gm])
    ax[0].plot(t, v, lw=0.6, label=name)
    if g is not None: ax[1].plot(g[0], g[1], lw=0.6, label=name)
    mean, tot = v.mean(), (v * np.diff(np.r_[0, t])).sum() / 1e3
    gcol = ("%.1f | %.0f | %.1f" % (g[1].mean(), g[1].max(), g[2].mean())) if g is not None else "- | - | -"
    vcol = ("%.1f" % g[3].max()) if g is not None and g[3] is not None else "-"
    rows.append(f"| {name} | {t[-1]:.0f} | {mean:.1f} | {v.max():.1f} | {100*mean/a.ddr_peak:.1f} | {tot:.2f} | {gcol} | {vcol} |")
    # per-run timeline
    gpu_on = g is not None and g[1].max() > 0; vram_on = gpu_on and g[3] is not None
    n = len(sp); npan = 1 + (n > 0) + gpu_on + vram_on
    if npan < 2: continue
    h = ([max(1.2, 0.28 * n)] if n else []) + [1] * (1 + gpu_on + vram_on)
    f2, ax2 = plt.subplots(len(h), 1, figsize=(9, 1.6 * (len(h) - 1) + (2.2 + 0.28 * n if n else 0) + 0.6), sharex=True, gridspec_kw={"height_ratios": h}, squeeze=False)
    ax2 = ax2[:, 0]; i = 0
    if n:
        for row, p in enumerate(sorted(sp)):
            cw = [(x, y) for ph, x, y in sp[p] if ph == "carry_wait"]
            for ph, x, y in sorted(sp[p], key=lambda q: DRAW.index(q[0])):
                if y < s0 + T0: continue
                ax2[0].barh(row, y - x, left=x - s0 - T0, height=0.8, color=COLOR.get(ph, "#B9C3CA"), lw=0)
                la = max([x] + [yy for _, yy in cw if x < yy <= y]) if ph == "witgen" else x
                if y - la > 0.045 * t[-1] and LABEL.get(ph, ph): ax2[0].text(la - s0 - T0 + (y - la) / 2, row, LABEL.get(ph, ph), ha="center", va="center", fontsize=6.5)
        ax2[0].set_ylim(n - 0.4, -0.6); ax2[0].set_yticks(range(n)); ax2[0].set_yticklabels([f"p{p}" for p in sorted(sp)], fontsize=7); ax2[0].set_title(name, fontsize=10); i = 1
    ax2[i].plot(t, v, color="#2B5180", lw=0.5); ax2[i].set_ylabel("CPU DRAM\nread (GB/s)", fontsize=8); ax2[i].grid(alpha=.3); i += 1
    if gpu_on:
        ax2[i].plot(g[0], g[1], color="#D97B22", lw=0.5); ax2[i].set_ylim(0, 100); ax2[i].set_ylabel("GPU memory\nactivity (%)", fontsize=8); ax2[i].grid(alpha=.3); i += 1
    if vram_on:
        ax2[i].plot(g[0], g[3], color="#4B3F9E", lw=0.8); ax2[i].set_ylim(0, None); ax2[i].set_ylabel("VRAM resident\n(GiB)", fontsize=8); ax2[i].grid(alpha=.3)
    ax2[-1].set_xlabel("seconds since start" + (f" ({a.skip_phase} excluded)" if a.skip_phase else "")); ax2[-1].set_xlim(0, t[-1]); f2.tight_layout()
    for ext in ("png", "pdf"): f2.savefig(os.path.join(a.out, f"timeline_{pre}.{ext}"), dpi=200)
    plt.close(f2)
ax[0].set_ylabel("CPU DRAM read (GB/s)"); ax[0].axhline(a.ddr_peak, color="k", ls=":", lw=0.8); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
ax[1].set_ylabel("GPU memory activity (%)"); ax[1].set_ylim(0, 100); ax[1].set_xlabel("seconds since start"); ax[1].grid(alpha=.3); fig.tight_layout()
for ext in ("png", "pdf"): fig.savefig(os.path.join(a.out, f"membw_overview.{ext}"), dpi=200)
open(os.path.join(a.out, "membw_overview.md"), "w").write("\n".join(rows) + "\n"); print("\n".join(rows))
