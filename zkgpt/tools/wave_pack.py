#!/usr/bin/env python3
"""wave_pack.py — offline wave scheduler + analysis for tools/sweep_partitions.sh.

Reads the per-partition CSV from tools/sweep_partitions.sh and, for each
configuration (num_splits/composition, c, matmul_k, range_k, lasso_k, tile_bits,
stream_bits), computes the wave-scheduled proving time under a GPU-VRAM cap
(and, if the CSV carries part_host_peak_mib and --host-gb is given, a HOST-RAM
cap too — since source streaming, host RAM rather than VRAM often sets the wave
width, see tile_bits/stream_bits and project_source_streaming_host_bound):

    Time = sum over waves of ( max partition cost in that wave )        (plans.tex eq. total_time)
    subject to, per wave:  sum of partition VRAM <= M_GPU               (eq. wave_vram)
                           sum of partition host RAM <= HOST_BUDGET      (if --host-gb given)

using the greedy packer from plans.tex (Solution WIP / PackWaves): sort a
config's partitions by cost ascending, then sweep, closing a wave whenever the
next partition would breach either cap. Partition cost = its measured wall
time; partition VRAM/host = its measured solo peak (summed across a wave
approximates concurrent residency).

It then ranks configs by wave time — the naive brute-force optimum you can later
try to beat with the analytical solver.

Usage:
  python3 tools/wave_pack.py <partitions.csv> [--vram-gb 64] [--host-gb 426] [--out summary.csv]
  python3 tools/wave_pack.py <partitions.csv> --best-only          # machine-parseable winner, for sweep_partitions.sh --tune
  python3 tools/wave_pack.py <partitions.csv> --analyze            # boundary/composition analysis + plots
                                                                    # (former boundary.py `analyze`; needs a
                                                                    # `composition` column + a sibling
                                                                    # <csv>.boundaries.csv from --configs sweeps)
"""
import argparse
import csv
import os
import sys
from collections import defaultdict


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _close_waves(parts, cap, use_idx):
    """parts: list of (cost, vram, host); use_idx picks which field (1=vram,
    2=host) each partition's usage is measured in. Greedy pack sorted by cost
    ascending (plans.tex), closing a wave whenever the next partition would
    breach cap."""
    ordered = sorted(parts, key=lambda p: p[0])
    total, wave_used, wave_max, n_waves = 0.0, 0.0, 0.0, 0
    for p in ordered:
        cost, use = p[0], p[use_idx]
        if wave_used + use > cap and wave_used > 0:
            total += wave_max
            n_waves += 1
            wave_used, wave_max = 0.0, 0.0
        wave_used += use
        wave_max = max(wave_max, cost)
    total += wave_max
    n_waves += 1
    return total, n_waves


def pack_waves(parts, cap_mib, host_cap_mib=None):
    """parts: list of (cost_s, vram_mib, host_mib_or_None). Returns
    (wave_time, n_waves, w_vram, w_host) or None if any single partition exceeds
    a cap (infeasible). w_vram/w_host are the naive floor(budget/peak) widths
    (informational — the greedy packer does not assume uniform peaks); the
    pack itself is run against whichever resource is more binding, since
    post-streaming that is often HOST RAM, not VRAM (see
    project_source_streaming_host_bound)."""
    if any(v is None for _, v, _ in parts):
        return None
    if any(v > cap_mib for _, v, _ in parts):
        return None
    have_host = host_cap_mib is not None and all(h is not None for _, _, h in parts)
    if have_host and any(h > host_cap_mib for _, _, h in parts):
        return None

    max_vram = max(v for _, v, _ in parts)
    max_host = max((h for _, _, h in parts if h is not None), default=None)
    w_vram = max(1, int(cap_mib // max_vram)) if max_vram else None
    w_host = (max(1, int(host_cap_mib // max_host)) if (have_host and max_host) else None)

    if have_host and w_host is not None and (w_vram is None or w_host < w_vram):
        total, n_waves = _close_waves(parts, host_cap_mib, use_idx=2)
    else:
        total, n_waves = _close_waves(parts, cap_mib, use_idx=1)
    return total, n_waves, w_vram, w_host


def load_rows(path):
    return list(csv.DictReader(open(path)))


def analyze(args):
    """Former boundary.py `analyze`: summarize + plot a --configs composition
    sweep. Expects a `composition` column in the CSV and a sibling
    `<csv>.boundaries.csv` (written by sweep_partitions.sh when --configs is used)."""
    prows = load_rows(args.csv)
    bpath = args.csv + ".boundaries.csv"
    brows = load_rows(bpath) if os.path.exists(bpath) else []

    by_cfg = defaultdict(list)
    for r in prows:
        key = r.get("composition") or r.get("config_id")
        by_cfg[key].append(r)

    print(f"\nBoundary/composition exploration  ({os.path.basename(args.csv)})")
    hdr = (f"{'config':<18}{'parts':>6}{'ok':>4}{'maxPart_s':>11}{'seq_s':>9}"
           f"{'maxVRAM':>9}{'sumVRAM':>9}")
    print(hdr)
    print("-" * len(hdr))
    summary = []
    for cfg, rs in sorted(by_cfg.items(), key=lambda kv: (len(kv[1]), kv[0])):
        ok = all(r["status"] == "ACCEPT" for r in rs)
        times = [fnum(r.get("part_time_s")) for r in rs if fnum(r.get("part_time_s")) is not None]
        vrams = [fnum(r.get("part_gpu_peak_mib")) for r in rs if fnum(r.get("part_gpu_peak_mib")) is not None]
        max_t = max(times) if times else 0.0
        seq_t = sum(times) if times else 0.0
        max_v = max(vrams) if vrams else 0.0
        sum_v = sum(vrams) if vrams else 0.0
        summary.append((cfg, len(rs), ok, max_t, seq_t, max_v, sum_v))
        print(f"{cfg:<18}{len(rs):>6}{('Y' if ok else 'N'):>4}{max_t:>11.2f}"
              f"{seq_t:>9.2f}{max_v/1024.0:>8.1f}G{sum_v/1024.0:>8.1f}G")

    if brows:
        widths = defaultdict(list)
        for b in brows:
            cut = b.get("cut_after_block")
            w = fnum(b.get("shared_commit_size"))
            if cut not in ("", None) and w is not None:
                widths[int(cut)].append(w)
        if widths:
            uniq = sorted({w for ws in widths.values() for w in ws})
            print("\nBoundary widths (shared_commit_size) by cut position:")
            for cut in sorted(widths):
                ws = widths[cut]
                print(f"  cut@block {cut:>3}: {sorted(set(int(x) for x in ws))}")
            if len(uniq) == 1:
                print(f"  -> width is CONSTANT ({int(uniq[0])}) across positions: "
                      "boundary cost is position-independent (load-balancing regime).")
            else:
                print(f"  -> width VARIES ({int(min(uniq))}..{int(max(uniq))}): "
                      "activation-aware placement matters.")

    if args.no_plots:
        return 0
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"\n(matplotlib unavailable: {e}; skipping plots)")
        return 0

    base = os.path.splitext(args.csv)[0]

    if brows:
        by_cfg_b = defaultdict(list)
        for b in brows:
            by_cfg_b[b["config"]].append(b)
        finest = max(by_cfg_b.values(), key=len)
        xs, ys = [], []
        for b in sorted(finest, key=lambda r: int(r["boundary_idx"])):
            cut = fnum(b.get("cut_after_block"))
            w = fnum(b.get("shared_commit_size"))
            if cut is not None and w is not None:
                xs.append(cut); ys.append(w)
        if xs:
            plt.figure(figsize=(7, 4))
            plt.plot(xs, ys, "o-")
            plt.xlabel("cut position (after block #)")
            plt.ylabel("shared_commit_size (activation width, field elems)")
            plt.title(f"Boundary activation profile  [{finest[0]['config']}]")
            plt.grid(True, alpha=0.3)
            plt.tight_layout(); plt.savefig(base + ".activation_profile.png", dpi=120)
            plt.close()

    if summary:
        summary2 = sorted(summary, key=lambda s: (s[1], s[0]))
        labels = [s[0] for s in summary2]
        max_t = [s[3] for s in summary2]
        max_v = [s[5] / 1024.0 for s in summary2]
        fig, ax1 = plt.subplots(figsize=(max(7, len(labels) * 0.7), 4))
        x = range(len(labels))
        ax1.bar([i - 0.2 for i in x], max_t, width=0.4, label="max part time (s)")
        ax1.set_ylabel("max-partition time (s)")
        ax1.set_xticks(list(x)); ax1.set_xticklabels(labels, rotation=45, ha="right")
        ax2 = ax1.twinx()
        ax2.bar([i + 0.2 for i in x], max_v, width=0.4, color="orange",
                label="max part VRAM (GiB)")
        ax2.set_ylabel("max-partition VRAM (GiB)")
        ax1.set_title("Slowest partition & peak VRAM per boundary config")
        fig.tight_layout(); fig.savefig(base + ".maxpart_by_config.png", dpi=120)
        plt.close(fig)

    print(f"\nplots written next to {base}.*.png")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--vram-gb", type=float, default=64.0)
    ap.add_argument("--host-gb", type=float, default=None,
                    help="host-RAM budget in GiB; applied only if the CSV carries "
                         "part_host_peak_mib (post-2026-08 sweep_partitions.sh)")
    ap.add_argument("--out", default="")
    ap.add_argument("--best-only", action="store_true",
                    help="print only the winning row as key=value lines (for "
                         "sweep_partitions.sh --tune) and skip the table/CSV write")
    ap.add_argument("--analyze", action="store_true",
                    help="boundary/composition analysis mode (former boundary.py analyze) "
                         "instead of wave-packing")
    ap.add_argument("--no-plots", action="store_true", help="(--analyze only) skip plots")
    args = ap.parse_args()

    if args.analyze:
        return analyze(args)

    cap = args.vram_gb * 1024.0  # MiB
    host_cap = args.host_gb * 1024.0 if args.host_gb is not None else None

    by_cfg = defaultdict(list)
    meta = {}
    for r in csv.DictReader(open(args.csv)):
        cid = r["config_id"]
        meta[cid] = r
        if r.get("status") == "ACCEPT" and fnum(r.get("part_time_s")) is not None:
            by_cfg[cid].append((fnum(r["part_time_s"]), fnum(r.get("part_gpu_peak_mib")),
                                 fnum(r.get("part_host_peak_mib"))))

    results = []
    for cid, parts in by_cfg.items():
        m = meta[cid]
        seq_total = sum(c for c, _, _ in parts)
        max_vram = max((v for _, v, _ in parts if v is not None), default=None)
        sum_vram = sum(v for _, v, _ in parts) if all(v is not None for _, v, _ in parts) else None
        packed = pack_waves(parts, cap, host_cap)
        wave_time, n_waves, w_vram, w_host = (packed if packed else (None, None, None, None))
        results.append({
            "config_id": cid,
            "num_splits": m["num_splits"], "composition": m.get("composition", ""),
            "c": m["c"], "matmul_k": m["matmul_k"], "range_k": m["range_k"],
            "lasso_k": m.get("lasso_k", ""),
            "tile_bits": m.get("tile_bits", ""), "stream_bits": m.get("stream_bits", ""),
            "n_parts": len(parts),
            "n_waves": n_waves,
            "w_vram": w_vram, "w_host": w_host,
            "wave_time_s": wave_time,
            "seq_total_s": seq_total,
            "speedup": (seq_total / wave_time) if wave_time else None,
            "max_part_vram_gib": (max_vram / 1024.0) if max_vram is not None else None,
            "sum_vram_gib": (sum_vram / 1024.0) if sum_vram is not None else None,
        })

    skipped = sorted({cid for cid, m in meta.items()
                      if cid not in by_cfg or pack_waves(by_cfg.get(cid, [(0, None, None)]), cap, host_cap) is None})

    feasible = [r for r in results if r["wave_time_s"] is not None]
    feasible.sort(key=lambda r: r["wave_time_s"])

    if args.best_only:
        if not feasible:
            print("NONE", file=sys.stderr)
            return 1
        b = feasible[0]
        for k in ("config_id", "num_splits", "composition", "c", "matmul_k", "range_k",
                  "lasso_k", "tile_bits", "stream_bits", "n_waves", "wave_time_s", "speedup"):
            print(f"{k}={b[k]}")
        return 0

    print(f"\nWave scheduling @ M_GPU = {args.vram_gb:.0f} GiB"
          + (f", host budget = {args.host_gb:.0f} GiB" if args.host_gb is not None else "")
          + f"   ({os.path.basename(args.csv)})")
    print(f"{'config':<28}{'parts':>6}{'waves':>6}{'wave_s':>10}{'seq_s':>10}"
          f"{'speedup':>9}{'maxVRAM':>9}{'sumVRAM':>9}")
    print("-" * 95)
    for r in feasible:
        print(f"{r['config_id']:<28}{r['n_parts']:>6}{r['n_waves']:>6}"
              f"{r['wave_time_s']:>10.2f}{r['seq_total_s']:>10.2f}"
              f"{(r['speedup'] or 0):>9.2f}"
              f"{(r['max_part_vram_gib'] or 0):>9.1f}{(r['sum_vram_gib'] or 0):>9.1f}")

    if feasible:
        b = feasible[0]
        print(f"\nBEST (min wave time): {b['config_id']}  "
              f"num_splits={b['num_splits']} c={b['c']} matmul_k={b['matmul_k']} "
              f"range_k={b['range_k']} lasso_k={b['lasso_k']} tile_bits={b['tile_bits']} "
              f"stream_bits={b['stream_bits']}  ->  {b['wave_time_s']:.2f}s "
              f"in {b['n_waves']} wave(s), {b['speedup']:.2f}x vs sequential")
    if skipped:
        print(f"\nnot scheduled (non-ACCEPT or a partition exceeds a budget): "
              + ", ".join(sorted(skipped)))

    out = args.out
    if not out:
        d, fn = os.path.split(args.csv)
        out = os.path.join(d, "wave_" + os.path.splitext(fn)[0] + f"_cap{int(args.vram_gb)}.csv")
    fieldnames = list(results[0].keys()) if results else [
        "config_id", "num_splits", "composition", "c", "matmul_k", "range_k", "lasso_k",
        "tile_bits", "stream_bits", "n_parts", "n_waves", "w_vram", "w_host",
        "wave_time_s", "seq_total_s", "speedup", "max_part_vram_gib", "sum_vram_gib"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(results, key=lambda r: (r["wave_time_s"] is None, r["wave_time_s"] or 0)):
            w.writerow(r)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
