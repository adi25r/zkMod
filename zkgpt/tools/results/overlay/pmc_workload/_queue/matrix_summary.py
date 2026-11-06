#!/usr/bin/env python3
"""matrix_summary.py -- the 3x3 (arm x sequence length) as one markdown table.

Reads the PARTITION line each run prints, so witgen and proving come from the
binary's own instrumentation rather than from wall-clock arithmetic over a log.

TIMINGS COME FROM THE UNPROFILED PASS ONLY. rocprofv3 --pmc serialises kernel
dispatches to read the counter registers (~2.2x here), so a profiled run's
witgen_s/prover_total_s are not wall times. GPU cells therefore read timing/run.log
and the CPU cells stat_run.log; a cell whose unprofiled pass has not landed prints
"--" rather than borrowing the profiled number.

Utilisation columns come from the counter passes and are only defined where there
are kernels, so the cpu+cpu row leaves them blank by construction, not by omission.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "..", "..", ".."))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "..")))

# arm -> {len: (dir, timing log relative to it)}
CELLS = {
    # Same cpuzk-stage run (cpu+cpu, len 64, 12 partitions), named to match
    # cc_len128/cc_len256.
    "cpu+cpu": {64:  ("cc_len64",  "stat_run.log"),
                128: ("cc_len128", "stat_run.log"),
                256: ("cc_len256", "stat_run.log")},
    "cpu+gpu": {64:  ("cg_len64",  "timing/run.log"),
                128: ("cg_len128", "timing/run.log"),
                256: ("cg_len256", "timing/run.log")},
    "gpu+gpu": {64:  ("gg_len64",  "timing/run.log"),
                128: ("gg_len128", "timing/run.log"),
                256: ("gg_len256", "timing/run.log")},
}
FIELDS = ("witgen_s", "prover_total_s", "time_s", "gpu_peak_mib", "host_peak_mib")


# Time fields are summed and memory fields are maxed across all PARTITION lines --
# every cell is a 12-partition sequential run. Summing time
# gives the true whole-run total; maxing memory gives the true peak, since each
# partition's own peak is independent (they run one at a time, not concurrently).
SUM_FIELDS = {"witgen_s", "prover_total_s", "time_s"}
MAX_FIELDS = {"gpu_peak_mib", "host_peak_mib"}


def partition_line(path):
    if not os.path.exists(path):
        return None
    agg = {}
    seen = False
    for line in open(path, errors="replace"):
        if not line.startswith("PARTITION "):
            continue
        d = dict(re.findall(r"(\w+)=([-\d.eE+]+)", line))
        try:
            vals = {k: float(d[k]) for k in FIELDS if k in d}
        except ValueError:
            continue
        seen = True
        for k, v in vals.items():
            if k in MAX_FIELDS:
                agg[k] = max(agg.get(k, v), v)
            else:
                agg[k] = agg.get(k, 0.0) + v
    return agg if seen else None



# Every partition follows the same pattern (the per-partition
# PARTITION lines in a cell vary by a few percent, never in shape), so a single
# partition's own phase breakdown is more useful than the 12-partition sum above --
# it shows where time actually goes inside one partition instead of just how much
# there was in total. Partition 0, not an average: it is exactly what is on disk,
# no averaging-across-partitions assumption to defend.
PHASE_FIELDS = ("witgen_s", "ph_gkr_generic", "ph_gkr_matmul", "ph_lasso",
                "ph_hyrax_commit", "ph_hyrax_open", "range_s", "exec_commit_s",
                "prover_total_s", "time_s")


def single_partition_line(path, idx=0):
    if not os.path.exists(path):
        return None
    target = "PARTITION idx=%d " % idx
    for line in open(path, errors="replace"):
        if not line.startswith(target):
            continue
        d = dict(re.findall(r"(\w+)=([-\d.eE+]+)", line))
        try:
            return {k: float(d[k]) for k in PHASE_FIELDS if k in d}
        except ValueError:
            return None
    return None


def util(arm_dir):
    """Kernel-duration-weighted VALU busy and occupancy over the whole run."""
    try:
        sys.path.insert(0, os.path.join(ROOT, "..", "..", "..", "..", "tools"))
        from plot_phase_util import collect
    except Exception:
        return None
    if not os.path.isdir(arm_dir):
        return None
    try:
        ph = collect(arm_dir)
    except Exception:
        return None
    if not ph:
        return None
    out = {}
    for key in ("VALUBusy", "MemUnitBusy", "MeanOccupancyPerCU", "busy"):
        num = sum(m.get(key, 0.0) * m.get("kern_s", 0.0) for m in ph.values()
                  if key in m)
        den = sum(m.get("kern_s", 0.0) for m in ph.values() if key in m)
        if den:
            out[key] = num / den
    return out or None


def cell(v, fmt="%.1f"):
    return "--" if v is None else fmt % v


print("# zkGPT: arm x sequence length, 12-partition sequential\n")
print("All cells: 12 layers, embed 768, linear 3072, split into **12 partitions** run "
      "sequentially (one layer per partition, no --concurrent/--batched/--overlapped -- "
      "wall time is not the point, per-partition GPU/CPU utilisation over time is), "
      "**192 threads**, `--c 18 --matmul-k 4 --range-k 8 --lasso-k 8 --sc-tile-bits 18`, "
      "`ZKGPT_WITNESS_CACHE=0`.\n")
print("Arm names are witgen+proving. `cpu+gpu` is CPU witness generation feeding GPU "
      "proving; `gpu+gpu` runs both on the device. `gpu+cpu` does not appear because it "
      "is not a configuration anyone would run -- and in the forked modes it is "
      "unreachable (the parent would hold a HIP context at `fork()`).\n")

print("## Wall time, unprofiled (summed across all 12 partitions; memory is the peak)\n")
print("| arm | len | witgen s | proving s | total s | GPU peak MiB | host peak MiB |")
print("|---|---:|---:|---:|---:|---:|---:|")
for arm in ("cpu+cpu", "cpu+gpu", "gpu+gpu"):
    for ln in (64, 128, 256):
        d, log = CELLS[arm][ln]
        p = partition_line(os.path.join(ROOT, d, log)) or {}
        print("| %s | %d | %s | %s | %s | %s | %s |" % (
            arm, ln, cell(p.get("witgen_s")), cell(p.get("prover_total_s")),
            cell(p.get("time_s")), cell(p.get("gpu_peak_mib"), "%.0f"),
            cell(p.get("host_peak_mib"), "%.0f")))

print("\n## One partition's phase breakdown (partition 0 -- every partition follows "
      "the same pattern, so this is more useful than the 12-partition sum above for "
      "seeing where time actually goes)\n")
print("| arm | len | witgen s | gkr generic s | gkr matmul s | lasso s | "
      "hyrax commit s | hyrax open s | range s | exec commit s | prover total s | "
      "time s |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for arm in ("cpu+cpu", "cpu+gpu", "gpu+gpu"):
    for ln in (64, 128, 256):
        d, log = CELLS[arm][ln]
        p = single_partition_line(os.path.join(ROOT, d, log)) or {}
        print("| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            arm, ln, cell(p.get("witgen_s")), cell(p.get("ph_gkr_generic")),
            cell(p.get("ph_gkr_matmul")), cell(p.get("ph_lasso")),
            cell(p.get("ph_hyrax_commit")), cell(p.get("ph_hyrax_open")),
            cell(p.get("range_s")), cell(p.get("exec_commit_s")),
            cell(p.get("prover_total_s")), cell(p.get("time_s"))))

print("\n## Device utilisation, kernel-duration weighted over the whole run\n")
print("| arm | len | GPU busy % | VALU busy % | MemUnit busy % | occupancy waves/CU of 32 |")
print("|---|---:|---:|---:|---:|---:|")
for arm in ("cpu+gpu", "gpu+gpu"):
    for ln in (64, 128, 256):
        d, _ = CELLS[arm][ln]
        u = util(os.path.join(ROOT, d)) or {}
        print("| %s | %d | %s | %s | %s | %s |" % (
            arm, ln, cell(u.get("busy")), cell(u.get("VALUBusy")),
            cell(u.get("MemUnitBusy")), cell(u.get("MeanOccupancyPerCU"), "%.2f")))
print("\n`cpu+cpu` has no kernels, so it has no device utilisation. Its equivalent "
      "figures are `figs/phases.png` and `figs/utilisation.png` in each `cc_*` "
      "directory, from perf.\n")

print("## Figures\n")
for arm in ("cpu+cpu", "cpu+gpu", "gpu+gpu"):
    for ln in (64, 128, 256):
        d, _ = CELLS[arm][ln]
        f = os.path.join(ROOT, d, "figs")
        n = len(os.listdir(f)) if os.path.isdir(f) else 0
        print("- `%s/` %s len %d -- %d figure%s" % (d, arm, ln, n,
                                                    "" if n == 1 else "s"))
