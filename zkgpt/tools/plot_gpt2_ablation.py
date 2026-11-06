#!/usr/bin/env python3
"""plot_gpt2_len64.py — GPT-2 len=64 witgen/proving ablation (2026-08-27).

Writes witgen_story_gpt2.png, speedup_gpt2.png, phases_gpt2.png (superseding the
len=30 versions from plot_gpt2_final.py / plot_witgen_story.py).

BASELINE IS THE REAL zkGPT THIS TIME. Arm 0 is
zkGPT-og/build-aligned/src/demo_llm_run (branch
align-arithmetization-to-sequence-split), NOT the fork at parts=1. That matters:
  * upstream zkGPT-og's initParam hardcoded `i64 pos = 32*1024` for the weight
    region -- no input alignment at all. Correct only at seq_len<=32 (32 padded
    tokens x 1024 padded feature dim = 32768); at len=64 the true boundary is
    65536 and the weights overwrote the prompt. Fixed by porting modular-zkp's
    weight_start_offset = next_pow2(len*input_slot_size) + its overflow-safe
    ceilPow2BitLengthSize(size_t). Verified: weight_start_offset=65536 at len 64.
  * the fork at parts=1 does NOT do the paper's two-pass power-of-two alignment
    (it is gated on `output_private || input_private`, both false for a single
    partition), so its input layer is 165.3M vs zkGPT-og's 536.9M. Using it as
    the baseline UNDERSTATED every speedup. Arm 0 here verifies ("All
    verification passed!!") with total_input_witness_size = 2^29.

Config: GPT-2, 12 layers, seq len 64, chunk_width 64 (num_chunks=1), lin_dim
3072, 192 threads (one per PHYSICAL core; proving anti-scales
past ~16-24 threads), ZKGPT_WITNESS_CACHE=0, quiet box, single runs.
  split arms : 12 partitions, --overlapped 12 --witgen-barrier.
               CPU prove: c=9 + --cpu --cpu-deg3 --cpu-open --cpu-commit +
               GKR_{LASSO,MATMUL,PRED,SC}_CPU=1.
               GPU prove: c=18 --matmul-k 4 --range-k 8 --lasso-k 8
               --sc-tile-bits 18, GKR_BFEVAL_GPU=1, range sumchecks on CPU.
               GPU witgen: GKR_WITGEN_GPU=1 GKR_ROUND_GPU=1. One MI210.
All arms ACCEPT 12/12; arm 0 verifies.

SOURCE STREAMING IS ON (GKR_SC_STREAM_BITS=22) for the GPU-proving arms, and it
has to be. Streaming off makes the dense dF/dG source 2*2^n*32B = 12.4 GiB per
partition; 12 of those is 148.9 GiB against a 64 GiB card (2.33x oversubscribed).
It OOMed at W=12 twice AND at W=4 (gpu_matmul_prover.hpp:707-708) -- narrowing
the wave does not help enough because all 12 children stay alive holding
contexts. With streaming on, peak is 4.7 GiB/partition and W=12 fits.
Tiling is a SEPARATE knob and still load-bearing with streaming off: at
l_stage=4, ext = 2*3^l*2^min(n-l,t) is 43.5 GB untiled (t=-1) vs 1.36 GB at
t=18. Failed no-stream runs kept in the log dir for the record.

CAVEATS: single runs, makespan noise ~10-15%. Phase figure covers the four fork
arms only -- zkGPT-og emits no per-phase instrumentation (just a `Results:`
line), so the baseline cannot be decomposed the same way.
"""
import os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "results", "gpt2_ablation", "logs", "t192")
OUTD = os.path.join(HERE, "results", "gpt2_ablation", "figs")

WITGEN_C, PROVE_C = "#B25A22", "#2F6DA8"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, MUTED, GRID, FACE = "#1a1a18", "#6b7579", "#e6e4df", "#fcfcfb"

# (log, label, tick, witgen_s, proving_s). witgen/proving are the arm's own
# reported wall numbers: build_s/prove_wall_s for arm 0, prep_s/barrier_makespan_s
# for the split arms.
ARMS = [
    ("arm0_zkgpt_mono.log"    ,     "zkGPT monolithic\n(upstream, CPU)", "zkGPT\nmonolithic",
     621.463, 579.438),
    ("arm1_cpuwg_cpuprove.log",   "CPU witgen + CPU prove\n(zkMod)",   "CPU witgen\nCPU prove\n(zkMod)",
     313.177, 115.772),
    ("arm2_cpuwg_gpuprove.log",   "CPU witgen + GPU prove",            "CPU witgen\nGPU prove",
     311.556,  54.921),
    ("arm3_gpuwg_cpuprove.log",   "GPU witgen + CPU prove",            "GPU witgen\nCPU prove",
      79.575, 119.568),
    ("arm4_gpuwg_gpuprove.log",   "GPU witgen + GPU prove",            "GPU witgen\nGPU prove",
      83.573,  61.570),
]

PHASES = [
    ("witness generation", ["witgen_s"],                                 BLUE),
    ("GKR sumcheck",       ["ph_gkr_generic", "ph_gkr_matmul"],          ORANGE),
    ("Lasso",              ["ph_lasso"],                                 AQUA),
    ("commitments",        ["ph_hyrax_commit", "ph_hyrax_open",
                            "standalone_commit_s"],                      YELLOW),
    ("range prover",       ["range_s"],                                  MAGENTA),
]
PART = re.compile(r"^PARTITION idx=.*$", re.M)
SUB = ("GPT-2, 12 layers, seq len 64, 12 partitions, 192 threads, one MI210")


def phases_of(path):
    rows = []
    for line in PART.findall(open(path, errors="replace").read()):
        d = {}
        for tok in line.split():
            k, _, v = tok.partition("=")
            try: d[k] = float(v)
            except ValueError: pass
        rows.append(d)
    if not rows:
        return None
    return [sum(sum(r.get(k, 0.0) for k in ks) for r in rows) / len(rows)
            for _n, ks, _c in PHASES]


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "figure.facecolor": FACE, "axes.facecolor": FACE,
        "savefig.facecolor": FACE, "savefig.dpi": 200, "savefig.bbox": "tight",
        "axes.edgecolor": GRID, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": INK})


def fig_story():
    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    base = ARMS[0][3] + ARMS[0][4]
    xmax = base * 1.22
    gap = xmax * (2.0 / (9.8 * 200))
    for i, (_lg, label, _tk, wg, pv) in enumerate(ARMS):
        tot = wg + pv
        ax.barh(i, wg, 0.58, color=WITGEN_C, linewidth=0)
        ax.barh(i, pv, 0.58, left=wg + gap, color=PROVE_C, linewidth=0)
        for val, left in ((wg, 0.0), (pv, wg + gap)):
            if val > xmax * 0.05:
                ax.text(left + val / 2, i, f"{val:.0f}", ha="center", va="center",
                        color="white", fontsize=9.5, fontweight="bold")
        ax.text(tot + xmax * 0.012, i, f"{tot:.0f} s", va="center", fontsize=10,
                fontweight="bold", color=INK)
        ax.text(tot + xmax * 0.085, i, f"{base/tot:.2f}×", va="center",
                fontsize=9.2, color=MUTED)
    ax.set_yticks(range(len(ARMS)))
    ax.set_yticklabels([a[1] for a in ARMS], fontsize=9.5)
    ax.invert_yaxis(); ax.set_xlim(0, xmax)
    ax.set_xlabel("wall time (s)  —  witness generation, then proving (disjoint)",
                  fontsize=9.5, color=MUTED)
    ax.grid(axis="x", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.set_title("Witness generation bounds the speedup from layer partitioning",
                 fontsize=12.0, fontweight="bold", loc="left", pad=26)
    ax.text(0.0, 1.018, SUB + ";\nwitness generation and proving measured disjointly",
            transform=ax.transAxes, fontsize=8.6, color=MUTED, va="bottom",
            linespacing=1.4)
    ax.legend(handles=[Patch(facecolor=WITGEN_C, label="witness generation"),
                       Patch(facecolor=PROVE_C, label="proving")],
              ncol=2, loc="lower right", bbox_to_anchor=(1.0, -0.20),
              frameon=False, fontsize=9)
    p = os.path.join(OUTD, "witgen_story_gpt2.png"); fig.savefig(p); plt.close(fig)
    return p


def fig_speedup():
    base = ARMS[0][3] + ARMS[0][4]
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    cols = [MUTED, BLUE, AQUA, ORANGE, YELLOW]
    vals = [a[3] + a[4] for a in ARMS]
    sp = [base / v for v in vals]
    for i, (v, s) in enumerate(zip(vals, sp)):
        ax.bar(i, s, 0.62, color=cols[i], linewidth=0)
        ax.text(i, s, f"{s:.2f}×", ha="center", va="bottom", fontsize=11,
                fontweight="bold", color=INK)
        ax.text(i, s * 0.5, f"{v:.0f} s", ha="center", va="center", fontsize=9.5,
                color="white", fontweight="bold")
    ax.axhline(1.0, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)))
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([a[2] for a in ARMS], fontsize=8.4)
    ax.set_xlim(-0.6, len(ARMS) - 0.3); ax.set_ylim(0, max(sp) * 1.16)
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_ylabel("speedup over monolithic zkGPT (wall clock)", fontsize=9.5)
    ax.set_title("End-to-end speedup with witness generation included",
                 fontsize=12.0, fontweight="bold", loc="left", pad=26)
    ax.text(0.0, 1.018, SUB + ";\nbaseline is upstream zkGPT with the input/weight alignment fixed",
            transform=ax.transAxes, fontsize=8.6, color=MUTED, va="bottom",
            linespacing=1.4)
    p = os.path.join(OUTD, "speedup_gpt2.png"); fig.savefig(p); plt.close(fig)
    return p


def fig_phases():
    rows = [(a[1], phases_of(os.path.join(LOGS, a[0]))) for a in ARMS]
    rows = [(l, v) for l, v in rows if v]                 # arm 0 has no PARTITION lines
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    tot = [sum(v) for _l, v in rows]
    xmax = max(tot) * 1.16
    gap = xmax * (2.0 / (10.0 * 200))
    for i, (_l, vals) in enumerate(rows):
        left = 0.0
        for (name, _k, col), v in zip(PHASES, vals):
            ax.barh(i, v, 0.58, left=left, color=col, linewidth=0)
            if v > xmax * 0.035:
                ax.text(left + v / 2, i, f"{v:.1f}", ha="center", va="center",
                        color="white", fontsize=9, fontweight="bold")
            left += v + gap
        ax.text(tot[i] + xmax * 0.012, i, f"{tot[i]:.0f} s", va="center",
                fontsize=10, fontweight="bold", color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([l for l, _v in rows], fontsize=9.5)
    ax.invert_yaxis(); ax.set_xlim(0, xmax)
    ax.set_xlabel("prover work per partition, averaged over partitions (s)",
                  fontsize=9.5, color=MUTED)
    ax.grid(axis="x", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.set_title("Per-partition prover work by protocol phase",
                 fontsize=12.0, fontweight="bold", loc="left", pad=26)
    ax.text(0.0, 1.018, SUB + ";\nzkGPT baseline omitted — it emits no per-phase instrumentation",
            transform=ax.transAxes, fontsize=8.6, color=MUTED, va="bottom",
            linespacing=1.4)
    ax.legend(handles=[Patch(facecolor=c, label=n) for n, _k, c in PHASES],
              ncol=5, loc="lower right", bbox_to_anchor=(1.02, -0.26),
              frameon=False, fontsize=8.8)
    p = os.path.join(OUTD, "phases_gpt2.png"); fig.savefig(p); plt.close(fig)
    return p


def main():
    style(); os.makedirs(OUTD, exist_ok=True)
    for f in (fig_story, fig_speedup, fig_phases):
        print(f())
    base = ARMS[0][3] + ARMS[0][4]
    print(f"\n{'arm':34s} {'witgen':>8} {'prove':>8} {'total':>8} {'speedup':>8}")
    for _lg, label, _tk, wg, pv in ARMS:
        t = wg + pv
        print(f"{label.replace(chr(10),' '):34s} {wg:8.1f} {pv:8.1f} {t:8.1f} {base/t:7.2f}x")


if __name__ == "__main__":
    main()
