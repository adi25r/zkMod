#!/usr/bin/env python3
"""Aggregate rocprofv3 --pmc counters per PROVER PHASE, not per kernel.

rocprofv3 reports counters per dispatch. On its own that is a list of thousands of
kernel names; what the phase question needs is "how ALU-bound is range_prove", which
means folding every dispatch into the ROCTx range that contains it. The same
containment join tools/overlay_concurrent.py uses, applied to counter rows.

Occupancy-style counters are averaged weighted by kernel duration (a 40 ms kernel at
80 % occupancy says more about the phase than a 4 us kernel at 10 %); instruction and
wave counts are summed.

IMPORTANT: --pmc serialises kernel dispatches to read the counter registers, so the
wall times in this pass are NOT comparable to an unprofiled run (measured ~2.2x
slower here). Use the counters and the ratios between phases, never the durations.

Usage:
  tools/pmc_by_phase.py <pmc-dir>          # dir containing run_*.csv
"""
import collections
import csv
import os
import re
import sys

# Averaged over time (a rate or a percentage), vs summed (a count).
RATE = {"VALUBusy", "SALUBusy", "MemUnitBusy", "Occupancy", "MeanOccupancyPerCU",
        "GPUBusy", "L2CacheHit", "MemUnitStalled"}

# FETCH_SIZE / WRITE_SIZE are reported per dispatch in KiB (see rocprofv3 -L:
# FETCH_SIZE = (TCC_EA_RDREQ_32B*32 + (TCC_EA_RDREQ - TCC_EA_RDREQ_32B)*64)/1024).
# They are COUNTS, so they sum -- they are deliberately not in RATE above.
BW = ("FETCH_SIZE", "WRITE_SIZE")


def col(row, *names):
    for n in names:
        if n in row and row[n] != "":
            return row[n]
    low = {k.lower(): v for k, v in row.items()}
    for n in names:
        if n.lower() in low and low[n.lower()] != "":
            return low[n.lower()]
    return None


def find(d, suffix):
    hits = []
    for root, _dirs, files in os.walk(d):
        for fn in files:
            if fn.endswith(suffix):
                hits.append(os.path.join(root, fn))
    return sorted(hits)


def load_ranges(d):
    """ROCTx ranges -> (partition, phase, t0, t1), innermost wins for attribution."""
    out, parts = [], []
    for path in find(d, "marker_api_trace.csv"):
        with open(path, newline="", errors="replace") as f:
            for r in csv.DictReader(f):
                nm = col(r, "Name", "Function")
                a, b = col(r, "Start_Timestamp"), col(r, "End_Timestamp")
                if not nm or a is None or b is None:
                    continue
                a, b = int(a), int(b)
                if b <= a:
                    continue
                nm = nm.strip()
                m = re.fullmatch(r"part(\d+)(?:_build)?", nm)
                (parts if m else out).append(
                    (int(m.group(1)), a, b) if m else (nm, a, b))
    parts.sort(key=lambda x: x[1])

    def owner(t):
        best = None
        for idx, a, b in parts:
            if a <= t <= b and (best is None or (b - a) < best[1]):
                best = (idx, b - a)
        return best[0] if best else -1

    return [(owner(a), nm, a, b) for nm, a, b in out], parts


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    d = sys.argv[1]

    ranges, _parts = load_ranges(d)
    if not ranges:
        sys.exit("no ROCTx ranges under %s (was --marker-trace passed?)" % d)
    ranges.sort(key=lambda r: (r[2], -(r[3] - r[2])))

    # counter rows: one per (dispatch, counter)
    rows = []
    for path in find(d, "counter_collection.csv"):
        with open(path, newline="", errors="replace") as f:
            for r in csv.DictReader(f):
                t0 = col(r, "Start_Timestamp")
                t1 = col(r, "End_Timestamp")
                cn = col(r, "Counter_Name")
                cv = col(r, "Counter_Value")
                if t0 is None or cn is None or cv is None:
                    continue
                try:
                    rows.append((int(t0), int(t1) if t1 else int(t0), cn, float(cv),
                                 col(r, "Kernel_Name", "Name") or "?"))
                except ValueError:
                    continue
    if not rows:
        sys.exit("no counter rows under %s" % d)

    # innermost containing range per dispatch
    def phase_of(t):
        best = None
        for _p, nm, a, b in ranges:
            if a <= t <= b and (best is None or (b - a) < best[1]):
                best = (nm, b - a)
        return best[0] if best else None

    agg = collections.defaultdict(lambda: collections.defaultdict(float))
    dur = collections.defaultdict(float)
    ndis = collections.Counter()
    seen = collections.defaultdict(set)
    for t0, t1, cn, cv, kn in rows:
        ph = phase_of(t0)
        if ph is None:
            continue
        w = max(1, t1 - t0)
        if cn in RATE:
            agg[ph][cn] += cv * w
            agg[ph]["_w_" + cn] += w
        else:
            agg[ph][cn] += cv
        if (t0, kn) not in seen[ph]:
            seen[ph].add((t0, kn))
            ndis[ph] += 1
            dur[ph] += w

    present = {cn for _t0, _t1, cn, _cv, _kn in rows}
    cols = ["VALUBusy", "SALUBusy", "MemUnitBusy", "MemUnitStalled", "L2CacheHit",
            "Occupancy", "MeanOccupancyPerCU"]

    def rate_of(a, n):
        w = a.get("_w_" + n, 0.0)
        return a.get(n, 0.0) / w if w else 0.0

    # Print only the columns this pass actually collected. Gating the whole table
    # on VALUBusy would print nothing for a memory-only pass, and padding absent
    # counters with 0.0 would render an uncollected counter as a measurement.
    show = [c for c in cols if c in present]
    if show:
        hdr = {"VALUBusy": "VALU%", "SALUBusy": "SALU%", "MemUnitBusy": "MemU%",
               "MemUnitStalled": "MemStl%", "L2CacheHit": "L2 hit%",
               "Occupancy": "Occup%",
               "MeanOccupancyPerCU": "wav/CU/32"}
        has_iv = "SQ_INSTS_VALU" in present
        key = "VALUBusy" if "VALUBusy" in present else show[0]
        order = sorted(agg, key=lambda p: -rate_of(agg[p], key))
        print("%-24s%s %9s%s %10s" %
              ("phase", "".join(" %9s" % hdr[c] for c in show), "kern_s",
               " %11s" % "SQ_INSTS_VALU" if has_iv else "", "dispatches"))
        for ph in order:
            a = agg[ph]
            print("%-24s%s %9.2f%s %10d" %
                  (ph, "".join(" %9.1f" % rate_of(a, c) for c in show),
                   dur[ph] / 1e9,
                   " %11.3g" % a.get("SQ_INSTS_VALU", 0.0) if has_iv else "",
                   ndis[ph]))
    else:
        order = sorted(agg, key=lambda p: -dur[p])

    if any(c in present for c in BW):
        # KiB -> GiB for the volumes; KiB -> GB/s (decimal, to compare against the
        # MI210's 1638 GB/s HBM2e peak) for the rate. The denominator is the summed
        # KERNEL time in the phase, not the phase's wall time: this is the bandwidth
        # achieved while kernels were actually running, which is the number that says
        # whether a kernel is memory bound. Phase wall time includes host gaps and
        # would understate it.
        print()
        # Only show L2 when it was actually collected. It lives in a different pass
        # (six TCC counters do not fit beside FETCH/WRITE), and printing 0.0 for an
        # absent counter is indistinguishable from a measured zero.
        l2 = "L2CacheHit" in present
        print("%-24s %10s %10s %10s %9s %9s%s" %
              ("phase", "read GiB", "write GiB", "total GiB", "kern_s", "GB/s",
               "%8s" % "L2 hit%" if l2 else ""))
        bworder = sorted(agg, key=lambda p: -(agg[p].get("FETCH_SIZE", 0.0) +
                                              agg[p].get("WRITE_SIZE", 0.0)))
        for ph in bworder:
            a = agg[ph]
            rd, wr = a.get("FETCH_SIZE", 0.0), a.get("WRITE_SIZE", 0.0)
            ks = dur[ph] / 1e9
            gbs = (rd + wr) * 1024.0 / 1e9 / ks if ks > 0 else 0.0
            print("%-24s %10.2f %10.2f %10.2f %9.2f %9.1f%s" %
                  (ph, rd / 1048576.0, wr / 1048576.0, (rd + wr) / 1048576.0,
                   ks, gbs,
                   "%8.1f" % rate_of(a, "L2CacheHit") if l2 else ""))
        trd = sum(agg[p].get("FETCH_SIZE", 0.0) for p in agg)
        twr = sum(agg[p].get("WRITE_SIZE", 0.0) for p in agg)
        tks = sum(dur[p] for p in agg) / 1e9
        print("%-24s %10.2f %10.2f %10.2f %9.2f %9.1f" %
              ("TOTAL", trd / 1048576.0, twr / 1048576.0, (trd + twr) / 1048576.0,
               tks, (trd + twr) * 1024.0 / 1e9 / tks if tks > 0 else 0.0))
        print("\nMI210 HBM2e peak is 1638 GB/s; --pmc serialises dispatches, so the "
              "VOLUMES are exact\nand the rates are per-kernel achieved bandwidth, "
              "not a whole-run average.")

    out = os.path.join(d, "pmc_by_phase.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phase"] + cols + ["kernel_s", "SQ_INSTS_VALU", "SQ_WAVES",
                                       "dispatches", "read_gib", "write_gib",
                                       "gb_per_s", "l2_hit_pct"])
        for ph in order:
            a = agg[ph]
            rd, wr = a.get("FETCH_SIZE", 0.0), a.get("WRITE_SIZE", 0.0)
            ks = dur[ph] / 1e9
            w.writerow([ph] + ["%.3f" % rate_of(a, c) for c in cols] +
                       ["%.4f" % ks, "%.0f" % a.get("SQ_INSTS_VALU", 0),
                        "%.0f" % a.get("SQ_WAVES", 0), ndis[ph],
                        "%.4f" % (rd / 1048576.0), "%.4f" % (wr / 1048576.0),
                        "%.2f" % ((rd + wr) * 1024.0 / 1e9 / ks if ks > 0 else 0.0),
                        "%.2f" % rate_of(a, "L2CacheHit")])
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
