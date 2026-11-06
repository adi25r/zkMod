#!/usr/bin/env python3
"""plot_membw_timeline.py -- paper figure: per-partition witness-generation / proof-generation timeline over the
whole-run memory curves, from membw/wrap.sh output (house style, plotstyle.py).

  plot_membw_timeline.py --logs results/membw_1ms/logs --out results/membw_1ms/figs/paper \
      --run gpt2_arm4,gpt2_arm4_gpuwg_gpuprove.log,"zkMod" [--run ...] [--skip-phase probe_witgen]

--run PREFIX,PHASE_LOG,LABEL  (PHASE_LOG is relative to --logs).  Top panel: one row per partition, two boxes -- witness
generation (PHASE witgen minus its carry_wait intervals -- the gap is the partition blocked on its predecessor) and proof generation (PHASE prove = every proving
kernel: input/output commit, range, execute_commit, GKR, Lasso, open). Below, on the same clock: host DRAM read GB/s
(perf, thin = per perf interval, bold = 100 ms mean), GPU memory activity % and resident VRAM when a GPU was sampled.
"""
import argparse, csv, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plotstyle as ps
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ap = argparse.ArgumentParser()
ap.add_argument("--logs", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--run", action="append", required=True); ap.add_argument("--skip-phase", default="probe_witgen")
ap.add_argument("--keep-prewait-stub", action="store_true", help="also draw the short witgen segment that runs BEFORE a partition blocks in carry_wait")
ap.add_argument("--bytes-per-count", type=float, default=64.0); ap.add_argument("--smooth-s", type=float, default=0.1)
a = ap.parse_args()
PH = re.compile(r"^PHASE\s+part=(-?\d+)\s+pid=\d+\s+ev=(begin|end)\s+phase=(\S+)\s+d=\d+\s+t_us=(\d+)")
WG, PR = ps.blues(3, 0.25, 0.95)[0], ps.blues(3, 0.25, 0.95)[2]      # light = witness generation, dark = proof generation

def phases(path):
    op, out, t0 = {}, {}, None
    for l in open(path, errors="replace"):
        m = PH.match(l)
        if not m: continue
        p, ev, ph, t = int(m.group(1)), m.group(2), m.group(3), int(m.group(4)) / 1e6
        if ph == a.skip_phase and ev == "end": t0 = t
        if p < 0 or ph not in ("witgen", "prove", "carry_wait"): continue
        if ev == "begin": op[(p, ph)] = t
        elif (p, ph) in op:
            sp_ = (op.pop((p, ph)), t)
            if ph == "carry_wait": out.setdefault(p, {}).setdefault("cw", []).append(sp_)
            else: out.setdefault(p, {})[ph] = sp_
    return out, t0

def cpu(p):
    acc = {}
    for r in csv.reader(l for l in open(p) if not l.startswith("#")):
        if len(r) >= 4 and r[1].replace(".", "").isdigit(): acc[float(r[0])] = acc.get(float(r[0]), 0) + float(r[1])
    t = np.array(sorted(acc)); c = np.array([acc[x] for x in t]) * a.bytes_per_count
    return t, c

def binned(t, c, w, T):                       # exact mean rate per w-second bin (counts attributed to interval end)
    edges = np.arange(0, T + w, w); s, _ = np.histogram(t, edges, weights=c); return edges[:-1] + w / 2, s / w / 1e9

def smooth(t, v, w, T):
    edges = np.arange(0, T + w, w); idx = np.clip(np.digitize(t, edges) - 1, 0, len(edges) - 2)
    s = np.bincount(idx, v, len(edges) - 1); n = np.maximum(np.bincount(idx, minlength=len(edges) - 1), 1); return edges[:-1] + w / 2, s / n

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
ps.apply(); plt.rcParams.update({"font.size": 12, "xtick.labelsize": 12, "ytick.labelsize": 12, "pdf.fonttype": 42})
for spec in a.run:
    pre, plog, label = spec.split(",", 2)
    sp, t0e = phases(os.path.join(a.logs, plog)); s0 = float(open(os.path.join(a.logs, pre + ".start_epoch")).read())
    off = (t0e or s0) - s0                                            # clock origin relative to start_epoch
    tc, cc = cpu(os.path.join(a.logs, pre + ".cpu.csv")); T = max(d[k][1] for d in sp.values() for k in ("witgen", "prove") if k in d) - s0 - off
    tc = tc - off; m = tc >= 0; tc, cc = tc[m], cc[m]
    rate = cc / np.maximum(np.diff(np.r_[0, tc]), 1e-6) / 1e9
    gp = os.path.join(a.logs, pre + ".gpu.csv"); g = None
    if os.path.exists(gp):
        rows = [l.split(",") for l in open(gp) if l.count(",") >= 2 and l.endswith("\n")]
        if rows:
            arr = np.array([[float(x) for x in r] for r in rows if len(r) == len(rows[0])]); arr[:, 0] -= s0 + off
            arr = arr[arr[:, 0] >= 0]; g = arr if arr[:, 1].max() > 0 else None
    n = len(sp); npan = 2 + (0 if g is None else 1 + (g.shape[1] > 3))
    hr = [max(1.0, 0.27 * n)] + [1.0] * (npan - 1)
    fig, ax = plt.subplots(npan, 1, sharex=True, gridspec_kw={"height_ratios": hr, "hspace": 0.12})
    def minus(span, holes):                    # witgen span with the carry_wait intervals removed
        segs, cur = [], span[0]
        for h0, h1 in sorted(holes):
            if h1 <= cur or h0 >= span[1]: continue
            if h0 > cur: segs.append((cur, h0))
            cur = max(cur, h1)
        if cur < span[1]: segs.append((cur, span[1]))
        return segs
    for p in sorted(sp):                       # row = partition index (first-starting on top); labels count down n..1
        d = sp[p]
        if "witgen" in d:
            segs = minus(d["witgen"], d.get("cw", []))
            for x, y in (segs if a.keep_prewait_stub else segs[-1:]):
                ax[0].barh(p, y - x, left=x - s0 - off, height=0.78, color=WG, edgecolor="black", linewidth=0.6)
        if "prove" in d:
            x, y = d["prove"]; ax[0].barh(p, y - x, left=x - s0 - off, height=0.78, color=PR, edgecolor="black", linewidth=0.6)
    ax[0].set_ylim(n - 0.4, -0.6); ax[0].set_yticks(range(n)); ax[0].set_yticklabels([str(n - p) if n > 1 else label for p in sorted(sp)])   # first-starting partition on top, labelled n
    ax[0].set_ylabel("Partition" if n > 1 else "")
    ax[0].legend(handles=[Patch(fc=WG, ec="black", label="Witness generation"), Patch(fc=PR, ec="black", label="Proof generation")],
                 loc="lower center", bbox_to_anchor=(0.5, 1.01), borderaxespad=0, ncol=2, frameon=True, facecolor="white", edgecolor="black", framealpha=0.95, prop={"weight": "bold", "size": 12})
    ax[0].grid(True, axis="x", linestyle="dotted", linewidth=0.7, color="gray"); ax[0].set_axisbelow(True)
    i = 1
    ax[i].step(tc, rate, where="pre", color="0.65", lw=0.5); tb, vb = binned(tc, cc, a.smooth_s, T); ax[i].plot(tb, vb, color="black", lw=1.1)
    ax[i].set_ylabel("DRAM (GB/s)", fontsize=11); ax[i].set_ylim(0, None); ps.grid(ax[i]); i += 1
    if g is not None:
        ts, vs = smooth(g[:, 0], g[:, 1], a.smooth_s, T)
        ax[i].plot(g[:, 0], g[:, 1], color="0.65", lw=0.5); ax[i].plot(ts, vs, color=PR, lw=1.1)
        ax[i].set_ylabel("GPU act. (%)", fontsize=11); ax[i].set_ylim(0, 100); ps.grid(ax[i]); i += 1
        if g.shape[1] > 3:
            ax[i].plot(g[:, 0], g[:, 3] / 2**30, color=PR, lw=1.1); ax[i].set_ylabel("VRAM (GiB)", fontsize=11); ax[i].set_ylim(0, None); ps.grid(ax[i])
    ax[-1].set_xlabel("Time (s)"); ax[-1].set_xlim(0, T * 1.01); fig.align_ylabels(ax)
    for x in ax: x.tick_params(length=3)
    ps.save(fig, os.path.join(a.out, f"timeline_{pre}"), size=(12, 1.4 + 0.2 * n + 1.7 * (npan - 1)))
    print("wrote", os.path.join(a.out, f"timeline_{pre}.pdf"))
