#!/usr/bin/env python3
"""plot_membw.py -- memory-system utilisation over time for the whole run (not per partition).
Reads results/membw/logs/<arm>.{cpu,gpu}.csv written by results/membw/wrap.sh.
  CPU: perf (ls_any_fills_from_sys.dram_io_all + l2_pf_miss_l2_l3.all) x 64 B / interval = DRAM->core READ GB/s
       (calibrated on a 64-thread read stream: counts ~1.1x true reads; excludes write-backs, so a lower bound on traffic).
  GPU: sysfs mem_busy_percent of the MI210 (UMC/HBM activity, a % not GB/s) polled at 10 ms.
"""
import csv, os, re, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); D = f"{HERE}/results/membw"; L = f"{D}/logs"
ARMS = [("gpt2_arm0_mono", "Monolith baseline (CPU)", "#7f7f7f"),
        ("gpt2_arm1", "zkMod CPU+CPU", "#6baed6"),
        ("gpt2_arm4", "zkMod GPU+GPU", "#2B5180")]
DDR_PEAK = 2 * 460.8   # 2x EPYC 9654, 12ch DDR5-4800 each (spec, theoretical)

def cpu(p):
    acc = {}
    for r in csv.reader(l for l in open(p) if not l.startswith("#")):
        if len(r) < 4 or not r[1].replace(".", "").isdigit(): continue
        acc[float(r[0])] = acc.get(float(r[0]), 0) + float(r[1])
    t = np.array(sorted(acc)); v = np.array([acc[x] for x in t]) * 64 / 1e9 / np.diff(np.r_[0, t])
    return t, v


LOGMAP = {"gpt2_arm0_mono": f"{L}/gpt2_arm0_mono.log", "gpt2_arm1": f"{L}/gpt2_arm1_cpuwg_cpuprove.log", "gpt2_arm4": f"{L}/gpt2_arm4_gpuwg_gpuprove.log"}

def t0_of(k):
    """seconds after start_epoch at which the throwaway probe_witgen pass ends (0 if none)."""
    s0 = float(open(f"{L}/{k}.start_epoch").read())
    for l in open(LOGMAP[k], errors="replace"):
        m = re.match(r"^PHASE\s+part=0\s+pid=\d+\s+ev=end\s+phase=probe_witgen\s+d=\d+\s+t_us=(\d+)", l)
        if m: return int(m.group(1)) / 1e6 - s0
    return 0.0

def cropc(t, v, T0):
    m = t >= T0; return t[m] - T0, v[m]

def gpu(prefix):
    """-> (t, mem_busy %, gpu_busy %, vram_used GiB or None for runs sampled before the 4th column existed)"""
    s0 = float(open(prefix + ".start_epoch").read())
    rows = [l.split(",") for l in open(prefix + ".gpu.csv") if l.count(",") >= 2 and l.endswith("\n")]
    a = np.array([[float(x) for x in r] for r in rows if len(r) == len(rows[0])])
    return a[:, 0] - s0, a[:, 1], a[:, 2], (a[:, 3] / 2**30 if a.shape[1] > 3 else None)

def gcrop(g, T0):
    m = g[0] >= T0
    return (g[0][m] - T0, g[1][m], g[2][m], None if g[3] is None else g[3][m])

fig, ax = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True)
rows = ["| run | wall s | CPU DRAM read GB/s mean | peak (10 ms) | % of 921.6 GB/s peak (mean) | total read TB | HBM busy % mean | peak | gpu_busy % mean | VRAM used GiB (peak, of 64) |", "|---|---|---|---|---|---|---|---|---|---|"]
for k, name, c in ARMS:
    if not os.path.exists(f"{L}/{k}.cpu.csv"): continue
    T0 = t0_of(k); t, v = cropc(*cpu(f"{L}/{k}.cpu.csv"), T0); g = gcrop(gpu(f"{L}/{k}"), T0)
    ax[0].plot(t, v, color=c, lw=0.6, label=name)
    ax[1].plot(g[0], g[1], color=c, lw=0.5, label=name)
    mean = v.mean(); tot = (v * np.diff(np.r_[0, t])).sum() / 1e3
    rows.append(f"| {name} | {t[-1]:.0f} | {mean:.1f} | {v.max():.1f} | {100*mean/DDR_PEAK:.1f} | {tot:.2f} | {g[1].mean():.1f} | {g[1].max():.0f} | {g[2].mean():.1f} | {('%.1f' % g[3].max()) if g[3] is not None and g[3].max() > 1 else '-'} |")
ax[0].set_ylabel("CPU DRAM read fills (GB/s)"); ax[0].axhline(DDR_PEAK, color="k", ls=":", lw=0.8)
ax[0].text(0.99, DDR_PEAK, "2-socket DDR5 peak 922", ha="right", va="bottom", transform=ax[0].get_yaxis_transform(), fontsize=8)
ax[0].legend(loc="center right", fontsize=8); ax[0].grid(alpha=.3)
ax[1].set_ylabel("MI210 HBM busy (%)"); ax[1].set_xlabel("seconds since proving pipeline start (probe excluded)"); ax[1].set_ylim(0, 100); ax[1].grid(alpha=.3)
fig.tight_layout()
os.makedirs(f"{D}/figs", exist_ok=True)
for ext in ("png", "pdf"): fig.savefig(f"{D}/figs/membw_gpt2.{ext}", dpi=200)
open(f"{D}/figs/membw_gpt2.md", "w").write("\n".join(rows) + "\n"); print("\n".join(rows))


# ---- per-arm figure: partition phase timeline on top, memory curves on the SAME clock below ----
import re
sys.path.insert(0, HERE)
from plotstyle import COLOR  # noqa: E402
PH = re.compile(r"^PHASE\s+part=(-?\d+)\s+pid=\d+\s+ev=(begin|end)\s+phase=(\S+)\s+d=\d+\s+t_us=(\d+)")
DRAW = ["witgen", "carry_wait", "token_wait", "standalone_input_commit", "output_commit", "range_build", "range_prove",
        "execute_commit", "boundary_link", "verify_gkr", "verify_lasso", "open_commit"]
LABEL = {"token_wait": "", "probe_witgen": "probe_witgen (sizing pass)", "execute_commit": "exec", "standalone_input_commit": "", "output_commit": "", "range_build": "", "boundary_link": ""}

LOG = {"gpt2_arm0_mono": f"{L}/gpt2_arm0_mono.log", "gpt2_arm1": f"{L}/gpt2_arm1_cpuwg_cpuprove.log", "gpt2_arm4": f"{L}/gpt2_arm4_gpuwg_gpuprove.log"}

def spans(log):
    open_, out = {}, {}
    for l in open(log, errors="replace"):
        m = PH.match(l)
        if not m or m.group(3) not in DRAW: continue
        p, ev, ph, t = int(m.group(1)), m.group(2), m.group(3), int(m.group(4)) / 1e6
        if p < 0: continue
        if ev == "begin": open_[(p, ph)] = t
        elif (p, ph) in open_: out.setdefault(p, []).append((ph, open_.pop((p, ph)), t))
    return out

def arm_figure(k, name):
    sp = spans(LOG[k])
    if not sp: return
    T0 = t0_of(k); s0 = float(open(f"{L}/{k}.start_epoch").read()) + T0   # clock starts when the probe pass ends
    t, v = cropc(*cpu(f"{L}/{k}.cpu.csv"), T0); g = gcrop(gpu(f"{L}/{k}"), T0)
    n = len(sp)
    has_gpu = g[1].max() > 0          # CPU-only runs: no HBM panel
    has_v = has_gpu and g[3] is not None
    np_ = 2 + has_gpu + has_v
    fig, ax = plt.subplots(np_, 1, figsize=(9, 2.2 + 0.28 * n + 1.6 * (np_ - 1)), sharex=True,
                           gridspec_kw={"height_ratios": [max(1.2, 0.28 * n)] + [1] * (np_ - 1)})
    for row, p in enumerate(sorted(sp)):
        cw = [(a, b) for ph, a, b in sp[p] if ph == "carry_wait"]
        for ph, a, b in sorted(sp[p], key=lambda x: DRAW.index(x[0])):   # carry_wait drawn after witgen -> on top
            ax[0].barh(row, b - a, left=a - s0, height=0.8, color=COLOR.get(ph, "#B9C3CA"), lw=0)
            la = max([a] + [y for _, y in cw if a < y <= b]) if ph == "witgen" else a   # label witgen after its wait
            if b - la > 0.045 * (t[-1] if len(t) else 1) and LABEL.get(ph, ph):
                ax[0].text(la - s0 + (b - la) / 2, row, LABEL.get(ph, ph), ha="center", va="center", fontsize=6.5)
    ax[0].set_ylim(n - 0.4, -0.6); ax[0].set_yticks(range(n)); ax[0].set_yticklabels([f"p{p}" for p in sorted(sp)], fontsize=7)
    ax[0].set_title(f"gpt2 len64 -- {name}", fontsize=10)
    ax[1].plot(t, v, color="#2B5180", lw=0.5); ax[1].set_ylabel("CPU DRAM\nread (GB/s)", fontsize=8); ax[1].grid(alpha=.3)
    if has_gpu:
        ax[2].plot(g[0], g[1], color="#D97B22", lw=0.5); ax[2].set_ylim(0, 100); ax[2].set_ylabel("MI210 HBM\nactivity (%)", fontsize=8); ax[2].grid(alpha=.3)
        if has_v:
            ax[3].plot(g[0], g[3], color="#4B3F9E", lw=0.8); ax[3].set_ylim(0, 64); ax[3].axhline(64, color="k", ls=":", lw=0.8)
            ax[3].set_ylabel("VRAM resident\n(GiB of 64)", fontsize=8); ax[3].grid(alpha=.3)
    ax[-1].set_xlabel("seconds since proving pipeline start (probe excluded)")
    ax[-1].set_xlim(0, max(t[-1], max(b for sp_ in sp.values() for _, _, b in sp_) - s0))
    fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(f"{D}/figs/timeline_{k}.{ext}", dpi=200)

for k, name, _ in ARMS:
    if os.path.exists(LOG[k]) and os.path.exists(f"{L}/{k}.cpu.csv"): arm_figure(k, name)
