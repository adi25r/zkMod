#!/usr/bin/env python3
"""Plot tools/run_five_arm_ladder.sh's output: the five-arm ablation ladder vs
sequence length (for one model), or --family: CPU-zkMod vs GPU-zkMod-dev
concurrent makespan across a whole model family.

Former plot_seqlen.py (four-arm schema: cpu/uvm/zkmod_uvm/tuned, from the
now-deleted run_seqlen_ladder.sh) — renamed and rewritten for the five-arm
schema run_five_arm_ladder.sh emits (arm,model,len,L,splits,group_layers,
status,wall_s,prover_total_s,prove_only_s,width,waves,gpu_peak_mib,rc), since
the old schema's only producer is gone. Also folds in plot_model_family_all4.py
/ plot_model_family_cpu_vs_gpu.py / plot_stream_matrix.py's family-of-models
comparison (former sources: raw run_parallel_partitions.py CSVs per model) --
the five-arm CSV already carries prove_only_s/width/waves per model, so the
family view now reads directly off tools/results/five_arm/data/five_arm_*.csv
instead of a separate concurrent-engine CSV per model.

Usage:
  tools/plot_ladder.py --model gpt2 [--csv PATH] [--out DIR]     # single model, 5 arms x len
  tools/plot_ladder.py --family [--models gpt2,opt350m,opt1b3,opt2b7]
"""
import argparse
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "results/five_arm/data")

# Categorical, fixed order (validated adjacent-pairs). Each arm gets a related
# but distinct hue since colour follows the ARM, never its rank.
ARMS = ["cpu_zkgpt", "cpu_zkmod", "gpu_uvm", "gpu_zkmod_uvm", "gpu_zkmod_dev"]
ARM_C = {
    "cpu_zkgpt":     "#D55E00",
    "cpu_zkmod":     "#E69F00",
    "gpu_uvm":       "#009E73",
    "gpu_zkmod_uvm": "#CC79A7",
    "gpu_zkmod_dev": "#0072B2",
}
ARM_LABEL = {
    "cpu_zkgpt":     "CPU zkGPT (monolithic)",
    "cpu_zkmod":     "CPU + zkMod partitioning",
    "gpu_uvm":       "GPU, UVM (naive)",
    "gpu_zkmod_uvm": "GPU + zkMod (UVM)",
    "gpu_zkmod_dev": "GPU + zkMod + tiling (device, concurrent)",
}
# "uvm" (naive) is typically 30-1000x every other arm -- sharing a linear
# y-axis with it flattens the rest to near-zero. Dropped from the CHART only
# (still in the table); folded into a footnote instead.
CHART_SKIP = {"gpu_uvm"}
INFEAS = "#B00020"
TAG = {"FAIL": "SEGV", "OOM": "OOM", "TIMEOUT": "DNF", "WEDGED_PRE": "wedge",
       "WEDGED_POST": "wedge", "PARTIAL": "partial"}


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load(csv_path):
    """-> {(arm,len): row}, deduped preferring the last OK row."""
    best = {}
    for r in csv.DictReader(open(csv_path)):
        key = (r["arm"], int(r["len"]))
        cur = best.get(key)
        if cur is None or r["status"] == "OK" or cur["status"] != "OK":
            best[key] = r
    return best


def value(row):
    """The bar height: prove_only_s (concurrent arms) else prover_total_s
    (single-process arms). wall_s includes precompute and is never used as
    the plotted metric -- see run_five_arm_ladder.sh's own METRICS comment."""
    if row is None:
        return None
    v = fnum(row.get("prove_only_s"))
    return v if v is not None else fnum(row.get("prover_total_s"))


def panel(ax, rows, lens, arms, ymax, title):
    n = len(arms)
    bw = 0.82 / n
    for ai, arm in enumerate(arms):
        xs, ys, marks = [], [], []
        for li, L in enumerate(lens):
            x = li + (ai - (n - 1) / 2) * bw
            r = rows.get((arm, L))
            v = value(r)
            if r and r["status"] == "OK" and v is not None:
                xs.append(x); ys.append(v)
            elif r:
                marks.append((x, r["status"]))
        ax.bar(xs, ys, width=bw * 0.92, color=ARM_C[arm], label=ARM_LABEL[arm],
               edgecolor="white", linewidth=0.8, zorder=3)
        for x, y in zip(xs, ys):
            ax.text(x, y + ymax * 0.012, f"{y:.0f}", ha="center", va="bottom",
                    fontsize=7.5, color="0.15", zorder=4)
        for x, st in marks:
            ax.bar([x], [ymax * 0.94], width=bw * 0.92, color="none",
                   edgecolor=INFEAS, linewidth=1.6, hatch="xxx", zorder=3)
            ax.text(x, ymax * 0.47, TAG.get(st, st), ha="center", va="center",
                    rotation=90, fontsize=8, color=INFEAS, fontweight="bold", zorder=4)

    ax.set_xticks(range(len(lens)))
    ax.set_xticklabels([str(L) for L in lens])
    ax.set_xlabel("sequence length")
    ax.set_title(title, fontsize=11.5)
    ax.set_ylim(0, ymax)
    ax.grid(True, axis="y", ls=":", alpha=0.4, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def render(model, rows, lens, arms, out_path, skip=frozenset(), title_suffix=""):
    arms_plot = [a for a in arms if a not in skip]
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13,
                         "axes.titleweight": "bold", "axes.labelsize": 11,
                         "legend.fontsize": 9.5})
    fig, ax = plt.subplots(1, 1, figsize=(10, 5.8))

    feas = [value(rows[k]) for k in rows if rows[k]["status"] == "OK"
            and k[0] not in skip and value(rows[k])]
    ymax = (max(feas) if feas else 1) * 1.18

    panel(ax, rows, lens, arms_plot, ymax,
          "prove_only_s (concurrent arms)  /  prover_total_s (monolithic arms)")
    ax.set_ylabel("prover time (s)")

    fig.suptitle(f"{model}: five-arm ablation vs sequence length" + title_suffix,
                 fontsize=13.5, fontweight="bold")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 0.1), framealpha=0.95)

    stset = {rows[k]["status"] for k in rows if k[0] not in skip}
    notes = ["cpu_zkgpt/gpu_uvm are monolithic (1 partition).",
             "cpu_zkmod/gpu_zkmod_dev run partitions as concurrent OS processes "
             "(prove_only_s = circuit precompute excluded, sum over waves of the max)."]
    if "gpu_uvm" in skip:
        vals = []
        for L in lens:
            r = rows.get(("gpu_uvm", L))
            v = value(r) if r else None
            vals.append(f"len={L}: " + (f"{v:.0f}s" if r and r["status"] == "OK" and v else (r["status"] if r else "-")))
        notes.append("GPU, UVM (naive) omitted -- typically 30-1000x every other arm. "
                     "Real values: " + ", ".join(vals) + ".")
    if stset & {"WEDGED_PRE", "WEDGED_POST"}:
        notes.append("wedge: a managed (UVM) run left the GPU reporting free VRAM while "
                      "every hipMalloc fails -- needs `rocm-smi --gpureset`.")
    fig.text(0.008, 0.02, "\n".join(notes), fontsize=7.5, color="0.4", linespacing=1.5)

    fig.tight_layout(rect=(0, 0.2, 1, 0.95))
    fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"wrote {out_path}")


def cmd_single(args):
    csv_path = args.csv or os.path.join(DATA_DIR, f"five_arm_{args.model}.csv")
    if not os.path.exists(csv_path):
        raise SystemExit(f"no data at {csv_path}")
    out_dir = args.out or os.path.join(HERE, "results/five_arm/images")
    os.makedirs(out_dir, exist_ok=True)

    rows = load(csv_path)
    lens = sorted({k[1] for k in rows})
    arms = [a for a in ARMS if any(k[0] == a for k in rows)]
    if not lens:
        raise SystemExit(f"no data in {csv_path}")

    render(args.model, rows, lens, arms, os.path.join(out_dir, f"five_arm_{args.model}.png"))
    render(args.model, rows, lens, arms, os.path.join(out_dir, f"five_arm_{args.model}_zoom.png"),
           skip=CHART_SKIP, title_suffix=" (zoomed: UVM naive omitted)")

    hdr = f"\n{'arm':<24}{'len':>5}{'time_s':>12}{'status':>10}"
    print(hdr); print("-" * len(hdr.strip()))
    for ar in arms:
        for L in lens:
            r = rows.get((ar, L))
            if not r:
                continue
            v = value(r) if r["status"] == "OK" else None
            print(f"{ar:<24}{L:>5}{(f'{v:.6g}' if v else '-'):>12}{r['status']:>10}")


def rbar(ax, x, w, h, color, r=0.045):
    pad = r * w
    ax.add_patch(FancyBboxPatch(
        (x + pad, 0), w - 2 * pad, max(h - pad, 1e-9),
        boxstyle=f"round,pad={pad},rounding_size={pad}",
        linewidth=0, facecolor=color, mutation_aspect=1e-3, clip_on=False, zorder=3))


def cmd_family(args):
    models = args.models.split(",")
    out_dir = args.out or os.path.join(HERE, "results/five_arm/images")
    os.makedirs(out_dir, exist_ok=True)

    per_model = {}
    for m in models:
        p = os.path.join(DATA_DIR, f"five_arm_{m}.csv")
        if os.path.exists(p):
            per_model[m] = load(p)
    if not per_model:
        raise SystemExit(f"no five_arm_<model>.csv found under {DATA_DIR} for models={models}")

    lens = sorted({L for rows in per_model.values() for (_, L) in rows})
    fig, axes = plt.subplots(1, len(per_model), figsize=(4.6 * len(per_model), 5.4))
    if len(per_model) == 1:
        axes = [axes]
    fig.patch.set_facecolor("#fcfcfb")
    CPU_C, GPU_C = ARM_C["cpu_zkmod"], ARM_C["gpu_zkmod_dev"]

    for ax, m in zip(axes, per_model):
        rows = per_model[m]
        ax.set_facecolor("#fcfcfb")
        c = [value(rows.get(("cpu_zkmod", l))) for l in lens]
        g = [value(rows.get(("gpu_zkmod_dev", l))) for l in lens]
        top = max([v for v in c + g if v] or [1])
        for i, l in enumerate(lens):
            if c[i]:
                rbar(ax, i - 0.40, 0.36, c[i], CPU_C)
            if g[i]:
                rbar(ax, i + 0.04, 0.36, g[i], GPU_C)
            if c[i] and g[i]:
                sp = c[i] / g[i]
                ax.text(i - 0.18, max(c[i], g[i]) + top * 0.075, f"{sp:.2f}x",
                        ha="center", va="bottom", fontsize=10.5,
                        color="#0b0b0b" if sp >= 1.15 else "#b03030",
                        fontweight="bold" if sp < 1.15 else "normal")
                gw = rows.get(("gpu_zkmod_dev", l), {})
                w, wv = gw.get("width"), gw.get("waves")
                if w:
                    ax.text(i + 0.22, g[i] + top * 0.018, f"{w}x{wv}",
                            ha="center", va="bottom", fontsize=8.5, color="#8a8984")
        ax.set_xticks(range(len(lens)))
        ax.set_xticklabels(lens, fontsize=11)
        ax.set_xlim(-0.75, len(lens) - 0.25)
        ax.set_ylim(0, top * 1.24)
        ax.set_title(m, fontsize=13.5, fontweight="bold", pad=16, loc="left")
        ax.set_xlabel("sequence length", fontsize=10.5)
        ax.grid(axis="y", color="#e8e7e2", linewidth=0.9, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axes[0].set_ylabel("prove_only_s (concurrent, precompute excluded)", fontsize=11)
    h = [plt.Rectangle((0, 0), 1, 1, fc=CPU_C), plt.Rectangle((0, 0), 1, 1, fc=GPU_C)]
    fig.legend(h, ["CPU + zkMod (concurrent partitions)", "GPU + zkMod + tiling (concurrent, device)"],
               loc="lower center", ncol=2, frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Model family: CPU vs GPU zkMod concurrent makespan", fontsize=15, fontweight="bold",
                 x=0.006, ha="left")
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])
    out = os.path.join(out_dir, "family_cpu_vs_gpu.png")
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    print(f"wrote {out}")

    print(f"\n{'model':10} {'len':>5} {'CPU s':>10} {'GPU s':>10} {'speedup':>9}")
    for m in per_model:
        rows = per_model[m]
        for l in lens:
            c, g = value(rows.get(("cpu_zkmod", l))), value(rows.get(("gpu_zkmod_dev", l)))
            if c and g:
                print(f"{m:10} {l:>5} {c:>10.1f} {g:>10.1f} {c/g:>8.2f}x")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt2", help="single-model view: which five_arm_<model>.csv")
    ap.add_argument("--csv", default="", help="override the single-model CSV path")
    ap.add_argument("--out", default="")
    ap.add_argument("--family", action="store_true", help="model-family CPU-vs-GPU view instead")
    ap.add_argument("--models", default="gpt2,opt350m,opt1b3,opt2b7",
                     help="--family only: comma-separated model names")
    args = ap.parse_args()
    if args.family:
        cmd_family(args)
    else:
        cmd_single(args)


if __name__ == "__main__":
    main()
