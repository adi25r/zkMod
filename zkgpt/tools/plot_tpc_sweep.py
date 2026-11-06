#!/usr/bin/env python3
"""Threads-per-partition sweep (2026-10-24): zkMod CPU witgen + CPU prove (arm 1), gpt2 len 64,
one 192-thread pool shared by every partition (bin/main_gkr_split_pool_2026-10-24: a child
runs GKR_TPC threads and the gate admits at most 192 / GKR_TPC children at once).
Logs: tools/results/speedup_analysis_models/paper_arith/logs_tpc_sweep_pool_2026-10-24/
tpc<N>/gpt2_arm1_cpuwg_cpuprove.log and mono/gpt2_arm0_mono.log (the monolith at 192 threads,
same binary). Writes figs/tpc_sweep_gpt2.{png,pdf,md}, paper style from plot_figures.py.

    python3 tools/plot_tpc_sweep.py
"""
import os, re
import plot_figures as pf          # rcParams, save(), BAR, SHADE, grid helpers
import matplotlib.pyplot as plt

D = os.path.join(pf.ROOT, "tools/results/speedup_analysis_models/paper_arith/logs_tpc_sweep_pool_2026-10-24")
POOL, PARTS = 192, 12


def prove_concurrency(path):
    """(max, time-weighted mean) number of partitions inside phase=prove at once."""
    ev = []
    for line in open(path, errors="replace"):
        m = re.match(r"^PHASE .*ev=(begin|end) +phase=prove +.*t_us=(\d+)", line)
        if m:
            ev.append((int(m.group(2)), 1 if m.group(1) == "begin" else -1))
    ev.sort()
    c = mx = 0; area = 0.0
    for (t, d), (t1, _) in zip(ev, ev[1:] + [ev[-1]]):
        c += d; mx = max(mx, c); area += c * (t1 - t)
    return mx, area / (ev[-1][0] - ev[0][0])


def main():
    # monolith e2e = its witgen + prove PHASE lines (as in plot_figures)
    txt = open(os.path.join(D, "mono/gpt2_arm0_mono.log"), errors="replace").read()
    assert "All verification passed" in txt
    base = sum(float(x) for x in re.findall(r"ev=end +phase=(?:witgen|prove) +.*dur_s=([0-9.]+)", txt))
    rows = []
    for t in sorted(int(x[3:]) for x in os.listdir(D) if x.startswith("tpc")):
        path = os.path.join(D, f"tpc{t}/gpt2_arm1_cpuwg_cpuprove.log")
        if not os.path.exists(path):
            continue
        txt = open(path, errors="replace").read()
        m = re.search(r"overlapped_total_s=([0-9.]+) status=OK", txt)
        if not m or txt.count("All verification passed") != PARTS:
            continue
        mx, mean = prove_concurrency(path)
        rows.append((t, float(m.group(1)), min(PARTS, POOL // t), mx, mean))

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    xs = range(len(rows))
    fc, hatch = pf.SHADE[1]
    ax.bar(xs, [r[1] for r in rows], 0.6, color=fc, hatch=hatch, label="zkMod CPU+CPU", **pf.BAR)
    ax.axhline(base, color="black", ls="--", lw=0.8, label=f"Monolith ({base:.0f} s)")
    for x, r in zip(xs, rows):
        ax.text(x, r[1] * 1.03, f"{base / r[1]:.2f}x", ha="center", va="bottom", fontsize=6.5,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    ax.set_xticks(list(xs), [str(r[0]) for r in rows])
    ax.set_xlabel(f"Threads per partition (pool = {POOL})")
    ax.set_ylabel("End-to-end time (s)")
    ax.set_ylim(0, max(max(r[1] for r in rows), base) * 1.15)
    ax.grid(axis="y", color="0.85", lw=0.4); ax.set_axisbelow(True)
    ax.legend(loc="upper right")

    tbl = ["| threads/partition | pool admits | proving at once (max / mean) | e2e s | speedup vs monolith |",
           "|---|---|---|---|---|"]
    tbl += [f"| {t} | {adm} | {mx} / {mean:.1f} | {e:.1f} | {base / e:.2f}x |" for t, e, adm, mx, mean in rows]
    tbl.append(f"\nMonolith (splits=1, 192 threads, same binary): {base:.1f} s. gpt2 len 64, CPU witgen + CPU prove.")
    pf.save(fig, "tpc_sweep_gpt2", tbl)


if __name__ == "__main__":
    main()
