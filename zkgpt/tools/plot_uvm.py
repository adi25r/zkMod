#!/usr/bin/env python3
"""plot_uvm.py — UVM-viability figures (Experiment 3). One tool, four views.

Subcommands (former plot_uvm_exp / plot_uvm_matmul / plot_uvm_story /
plot_uvm_profile):

  exp      slowdown vs device baseline, two panels (matmul, deg3), from a
           run_uvm_exp CSV. Shows whether UVM is a cliff (hard ceiling) or a
           smooth sub-2x ramp (soft penalty).
  story    the takeaway figure: proving time vs working-set memory for device /
           UVM / UVM+prefetch, log-y, with the device-memory wall shaded — is
           UVM worth it? (same run_uvm_exp CSV as `exp`).
  matmul   proving time vs k per layer size n, device vs UVM, from a
           run_uvm_matmul CSV; device OOM / UVM TIMEOUT annotated.
  profile  device-vs-UVM 2x2 profile from a run_llm_demo (--uvm) sweep CSV;
           pick which of {L,c,k} is on the x-axis, the other two pinned.

Usage:
  python3 tools/plot_uvm.py exp     <uvm_exp.csv>    [--out fig.png] [--logy]
  python3 tools/plot_uvm.py story   <uvm_exp.csv>    [--out fig.png]
  python3 tools/plot_uvm.py matmul  <uvm_matmul.csv> [--out fig.png]
  python3 tools/plot_uvm.py profile <sweep.csv> [--x L|c|k] [--L N] [--c N] [--k N] [--out fig.png]
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ── exp: slowdown vs device baseline ──────────────────────────────────────────

def cmd_exp(args):
    rows = list(csv.DictReader(open(args.csv)))
    fams = ["matmul", "deg3"]
    dev_metric = {}          # (fam,n,knob) -> device metric_s (OK only)
    dev_best = {}            # fam -> fastest feasible device metric
    for r in rows:
        if r["arm"] == "device" and r["status"] == "OK":
            m = fnum(r["metric_s"])
            if m is None: continue
            dev_metric[(r["family"], r["n"], r["knob"])] = m
            dev_best[r["family"]] = min(dev_best.get(r["family"], 1e30), m)

    fig, axes = plt.subplots(1, len(fams), figsize=(7*len(fams), 5), squeeze=False)
    arm_style = {
        "device": dict(marker="o", color="#1f77b4", label="device memory"),
        "uvm":    dict(marker="s", color="#d62728", label="UVM (managed)"),
        "uvm_pf": dict(marker="^", color="#9467bd", label="UVM + prefetch/advise"),
    }
    # device memory capacity (GiB) = peak / rho, recovered from any row
    dev_cap = None
    for r in rows:
        peak = fnum(r["peak_gib"]); rho = fnum(r["rho"])
        if peak and rho:
            dev_cap = peak / rho; break
    for ax, fam in zip(axes[0], fams):
        frows = [r for r in rows if r["family"] == fam]
        if not frows:
            ax.set_visible(False); continue
        best = dev_best.get(fam)
        for arm in ("uvm", "uvm_pf"):
            xs, ys, fail_x = [], [], []
            for r in frows:
                if r["arm"] != arm: continue
                peak = fnum(r["peak_gib"]); m = fnum(r["metric_s"])
                if peak is None: continue
                denom = dev_metric.get((fam, r["n"], r["knob"])) or best
                if r["status"] == "OK" and m and denom:
                    xs.append(peak); ys.append((m / denom - 1.0) * 100.0)
                else:
                    fail_x.append((peak, r["status"]))
            if xs:
                order = sorted(range(len(xs)), key=lambda i: xs[i])
                ax.plot([xs[i] for i in order], [ys[i] for i in order], **arm_style[arm])
            for peak, st in fail_x:
                ax.scatter([peak], [ax.get_ylim()[1]], marker="x",
                           s=70, color=arm_style[arm]["color"], zorder=6)
                ax.annotate(st, (peak, ax.get_ylim()[1]), fontsize=7, rotation=90,
                            ha="center", va="top", color=arm_style[arm]["color"])
        ax.axhline(0.0, ls=":", color="#1f77b4", lw=1.2)
        ax.annotate("device baseline (0%)", (ax.get_xlim()[0], 0.0), fontsize=8,
                    va="bottom", ha="left", color="#1f77b4")
        if dev_cap:
            ax.axvline(dev_cap, ls="--", color="gray", lw=1)
            ax.annotate(f"device memory limit ({dev_cap:.0f} GiB)",
                        (dev_cap, ax.get_ylim()[1]), fontsize=8,
                        rotation=90, va="top", ha="right", color="gray")
        if args.logy: ax.set_yscale("log")
        ax.set_xlabel("working-set memory (GiB)")
        ax.set_ylabel("% slower than device (same config)")
        ax.set_title(f"{fam} layer")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend()
    fig.suptitle("UVM viability: slowdown vs device baseline (Exp 3)", fontweight="bold")
    fig.tight_layout()
    out = args.out or args.csv.rsplit(".", 1)[0] + ".png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


# ── story: proving time vs working-set memory ─────────────────────────────────

def cmd_story(args):
    rows = list(csv.DictReader(open(args.csv)))
    fams = ["matmul", "deg3"]

    dev_cap = None
    for r in rows:
        peak, rho = fnum(r["peak_gib"]), fnum(r["rho"])
        if peak and rho:
            dev_cap = peak / rho; break

    style = {
        "device": dict(marker="o", color="#1f77b4", label="device (hipMalloc)"),
        "uvm":    dict(marker="s", color="#d62728", label="UVM (managed)"),
        "uvm_pf": dict(marker="^", color="#9467bd", label="UVM + prefetch/advise"),
    }
    titles = {"matmul": "matmul product-sumcheck", "deg3": "deg3 sumcheck"}

    fig, axes = plt.subplots(1, len(fams), figsize=(7.5*len(fams), 5.2), squeeze=False)
    for ax, fam in zip(axes[0], fams):
        frows = [r for r in rows if r["family"] == fam]
        if not frows:
            ax.set_visible(False); continue
        fails = []   # (mem, status, color)
        for arm in ("device", "uvm", "uvm_pf"):
            pts = []
            for r in frows:
                if r["arm"] != arm: continue
                mem, t = fnum(r["peak_gib"]), fnum(r["metric_s"])
                if mem is None: continue
                if r["status"] == "OK" and t:
                    pts.append((mem, t))
                else:
                    fails.append((mem, r["status"], style[arm]["color"]))
            pts.sort()
            if pts:
                ax.plot([m for m, _ in pts], [t for _, t in pts], **style[arm])

        ax.set_yscale("log")
        if dev_cap:
            xhi = max(fnum(r["peak_gib"]) or 0 for r in frows) * 1.08
            ax.axvspan(dev_cap, xhi, color="0.93", zorder=0)
            ax.axvline(dev_cap, ls="--", color="gray", lw=1.2)
            ax.annotate(f"device memory limit ({dev_cap:.0f} GiB)",
                        (dev_cap, ax.get_ylim()[1]), fontsize=8, rotation=90,
                        va="top", ha="right", color="gray")
            ax.set_xlim(left=0, right=xhi)

        ylo, _ = ax.get_ylim()
        for mem, st, color in fails:
            ax.scatter([mem], [ylo*1.4], marker="x", s=70, color=color, zorder=6)
            ax.annotate(st, (mem, ylo*1.4), fontsize=7, rotation=90,
                        ha="center", va="bottom", color=color)

        ax.set_xlabel("working-set memory (GiB)")
        ax.set_ylabel("proving time (s)")
        ax.set_title(titles.get(fam, fam))
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(loc="upper left")

    fig.suptitle("Experiment 3: UVM", fontweight="bold")
    fig.tight_layout()
    out = args.out or args.csv.rsplit(".", 1)[0] + "_story.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


# ── matmul: time vs k, device vs UVM ──────────────────────────────────────────

def cmd_matmul(args):
    rows = list(csv.DictReader(open(args.csv)))
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        try: n, k = int(r["n"]), int(r["k"])
        except ValueError: continue
        t = float(r["time_s"]) if r["time_s"] else None
        by[n][r["mode"]].append((k, t, r["status"], r.get("workingset_gib", "")))

    ns = sorted(by)
    fig, axes = plt.subplots(1, len(ns), figsize=(6*len(ns), 4.5), squeeze=False)
    style = {"device": dict(marker="o", color="#1f77b4", label="device (hipMalloc)"),
             "uvm":    dict(marker="s", color="#d62728", label="UVM (managed, XNACK=1)")}
    for ax, n in zip(axes[0], ns):
        infeasible = []   # (k, status, color) drawn after we know the y-range
        for mode in ("device", "uvm"):
            pts = sorted(by[n].get(mode, []))
            ks  = [k for k, t, s, w in pts if t is not None]
            ts  = [t for k, t, s, w in pts if t is not None]
            if ks:
                ax.plot(ks, ts, **style[mode])
            for k, t, s, w in pts:
                if t is None and s in ("OOM", "TIMEOUT", "ERR"):
                    infeasible.append((k, s, style[mode]["color"], mode))
        ax.set_ylim(bottom=0)
        ylo, yhi = ax.get_ylim()
        ymark = yhi * 0.03
        for k, s, color, mode in infeasible:
            ax.scatter([k], [ymark], marker="x", s=80, color=color, zorder=5)
            ax.annotate(f"{mode}\n{s}", (k, ymark), fontsize=7,
                        ha="center", va="bottom", color=color)
        ax.set_xlabel("k (Blendy stages)")
        ax.set_ylabel("proving time (s)")
        ax.set_title(f"n={n} (s=2^{n})  matmul product-sumcheck")
        ax.grid(True, ls=":", alpha=0.4)
        ax.legend()
    fig.suptitle("Small-k + UVM vs larger-k in device memory", fontweight="bold")
    fig.tight_layout()
    out = args.out or args.csv.rsplit(".", 1)[0] + ".png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


# ── profile: device-vs-UVM 2x2 from a run_llm_demo --uvm sweep ────────────────

# Blendy product-sumcheck phases (NB: protocol phases, not single kernels).
MM_PHASES = [
    ("mm_expand_s", "expand", "#0072B2"),
    ("mm_ptable_s", "ptable", "#E69F00"),
    ("mm_round_s",  "round",  "#009E73"),
    ("mm_fold_s",   "fold",   "#CC79A7"),
]
PIPE_PHASES = [
    ("circuit_build_s", "circuit_build", "#0072B2"),
    ("verify_gkr_s",    "verify_gkr",    "#E69F00"),
    ("range_s",         "range",         "#009E73"),
]
RANGE_PHASES = [
    ("rc_setup_s",  "setup",  "#8172B3"),
    ("rc_invert_s", "invert", "#937860"),
    ("rc_commit_s", "commit", "#DA8BC3"),
    ("rc_deg3_s",   "deg3",   "#8C8C8C"),
    ("rc_deg1_s",   "deg1",   "#CCB974"),
    ("rc_open_s",   "open",   "#64B5CD"),
]
BIN_COLOR = {"device": "#4C72B0", "uvm": "#DD8452"}
AXIS_LABEL = {"L": "depth L (transformer blocks)", "c": "chunk-bits c",
              "k": "Blendy stages k"}


def _profile_default_out(path, x):
    d, fname = os.path.split(path)
    png = os.path.splitext(fname)[0] + f".x{x}.png"
    if os.path.basename(d) == "data":
        return os.path.join(os.path.dirname(d), "images", png)
    return os.path.join(d, png)


def cmd_profile(args):
    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.labelsize": 11, "legend.fontsize": 9, "figure.dpi": 130,
    })
    xk = args.x

    rows = list(csv.DictReader(open(args.csv)))
    if not rows:
        raise SystemExit("no rows in " + args.csv)
    for r in rows:
        if not r.get("binary"):
            r["binary"] = "run"

    def vals(key):
        return sorted({fnum(r[key]) for r in rows if fnum(r.get(key)) is not None})

    other = [a for a in ("L", "c", "k") if a != xk]
    pins = {}
    forced = {}
    for a in other:
        want = getattr(args, a)
        present = vals(a)
        if not present:
            pins[a] = None
        elif want is not None and want in present:
            pins[a] = want
            forced[a] = want
        else:
            pins[a] = None  # resolved below

    free = [a for a in other if a not in forced and vals(a)]
    if free:
        import itertools
        choices = [vals(a) for a in free]
        best_combo, best_score = None, None
        for combo in itertools.product(*choices):
            trial = dict(forced)
            trial.update(zip(free, combo))

            def matches(r, trial=trial):
                return all(fnum(r.get(a)) == trial[a] for a in other)

            n_accept = sum(1 for r in rows if matches(r) and r.get("status") == "ACCEPT")
            tiebreak = tuple(
                combo[free.index(a)] if a == "L" else -combo[free.index(a)]
                for a in free
            )
            score = (n_accept, tiebreak)
            if best_score is None or score > best_score:
                best_score, best_combo = score, trial
        pins.update(best_combo)

    def keep(r):
        for a in other:
            if pins[a] is not None and fnum(r.get(a)) != pins[a]:
                return False
        return True

    sub = [r for r in rows if keep(r)]
    if not sub:
        raise SystemExit(f"no rows after pinning {pins}")

    order = []
    for r in sub:
        if r["binary"] not in order:
            order.append(r["binary"])
    order.sort(key=lambda b: {"device": 0, "uvm": 1}.get(b, 2))

    xs = sorted({fnum(r[xk]) for r in sub if fnum(r.get(xk)) is not None})
    xpos = np.arange(len(xs))
    xt = [str(int(v)) if v == int(v) else str(v) for v in xs]
    idx = {v: i for i, v in enumerate(xs)}

    def series(binary):
        out = [None] * len(xs)
        for r in sub:
            if r["binary"] == binary:
                v = fnum(r[xk])
                if v in idx:
                    out[idx[v]] = r
        return out

    nb = len(order)
    bw = 0.8 / max(nb, 1)

    def offs(j):
        return (j - (nb - 1) / 2.0) * bw

    def bc(b, fb):
        return BIN_COLOR.get(b, fb)

    def annotate_fail(ax, xp, r, ytop):
        st = (r or {}).get("status", "")
        if r is not None and st and st != "ACCEPT":
            ax.annotate(st, (xp, ytop), fontsize=6.5, rotation=90,
                        ha="center", va="bottom", color="crimson")

    def panel_stacked(ax, comps, title, ylabel, tag_bins=True):
        for j, b in enumerate(order):
            rs = series(b)
            bottom = np.zeros(len(xs))
            for key, label, color in comps:
                v = np.array([fnum((rs[i] or {}).get(key)) or 0.0 for i in range(len(xs))])
                ax.bar(xpos + offs(j), v, bw * 0.92, bottom=bottom, color=color,
                       edgecolor=bc(b, "black"), linewidth=1.0,
                       label=label if j == 0 else None)
                bottom += v
            for i in range(len(xs)):
                if tag_bins and nb > 1:
                    ax.annotate(b, (xpos[i] + offs(j), 0), fontsize=6, rotation=90,
                                ha="center", va="bottom", color=bc(b, "black"), alpha=0.8)
                annotate_fail(ax, xpos[i] + offs(j), rs[i], bottom[i] * 1.01)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(ncol=2, fontsize=8)

    def panel_grouped(ax, key, title, ylabel, scale=1.0, palette=None):
        for j, b in enumerate(order):
            rs = series(b)
            v = np.array([(fnum((rs[i] or {}).get(key)) or 0.0) * scale for i in range(len(xs))])
            ax.bar(xpos + offs(j), v, bw * 0.92,
                   color=(palette or BIN_COLOR).get(b, "#777777"), label=b)
            for i in range(len(xs)):
                annotate_fail(ax, xpos[i] + offs(j), rs[i], v.max() * 1.01 if v.max() else 1)
        ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    if xk == "L":
        panel_stacked(axes[0, 0], MM_PHASES,
                      "Blendy product-sumcheck phases (whole proof), stacked", "seconds")
        panel_stacked(axes[0, 1], PIPE_PHASES,
                      "pipeline phase time, stacked", "seconds")
        panel_grouped(axes[1, 0], "gpu_peak_mib", "peak GPU memory (overall)", "GiB", 1 / 1024.0,
                      palette={"device": "#55A868", "uvm": "#DD8452"})
        panel_grouped(axes[1, 1], "host_peak_mib", "peak host memory (RSS)", "GiB", 1 / 1024.0,
                      palette={"device": "#C44E52", "uvm": "#DD8452"})
    else:
        panel_stacked(axes[0, 0], RANGE_PHASES,
                      "range prover phases, stacked", "seconds")
        panel_grouped(axes[0, 1], "range_result_s", "range total time", "seconds")
        panel_grouped(axes[1, 0], "range_gpu_peak_mib", "range-phase peak GPU memory", "GiB",
                      1 / 1024.0, palette={"device": "#55A868", "uvm": "#DD8452"})
        panel_grouped(axes[1, 1], "range_host_peak_mib", "range-phase peak host RSS", "GiB",
                      1 / 1024.0, palette={"device": "#C44E52", "uvm": "#DD8452"})

    for ax in axes.flat:
        ax.set_xticks(xpos); ax.set_xticklabels(xt)
        ax.set_xlabel(AXIS_LABEL[xk])
        ax.grid(True, axis="y", ls=":", alpha=0.4)

    pin_str = "  ".join(f"{a}={int(pins[a]) if pins[a] is not None else '*'}" for a in other)
    model = sub[0].get("model", ""); mode = sub[0].get("mode", "")
    fig.suptitle(f"device vs UVM — x={xk}  (pinned {pin_str})  model={model} mode={mode}  "
                 f"[{os.path.basename(args.csv)}]", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = args.out or _profile_default_out(args.csv, xk)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("exp", help="slowdown vs device baseline (uvm_exp CSV)")
    e.add_argument("csv"); e.add_argument("--out", default=""); e.add_argument("--logy", action="store_true")
    e.set_defaults(func=cmd_exp)

    s = sub.add_parser("story", help="time vs working-set memory (uvm_exp CSV)")
    s.add_argument("csv"); s.add_argument("--out", default="")
    s.set_defaults(func=cmd_story)

    m = sub.add_parser("matmul", help="time vs k, device vs UVM (uvm_matmul CSV)")
    m.add_argument("csv"); m.add_argument("--out", default="")
    m.set_defaults(func=cmd_matmul)

    p = sub.add_parser("profile", help="device-vs-UVM 2x2 profile (llm_demo --uvm sweep CSV)")
    p.add_argument("csv")
    p.add_argument("--x", choices=["L", "c", "k"], default="L")
    p.add_argument("--L", type=float, default=None)
    p.add_argument("--c", type=float, default=None)
    p.add_argument("--k", type=float, default=None)
    p.add_argument("--out", default="")
    p.set_defaults(func=cmd_profile)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    main()
