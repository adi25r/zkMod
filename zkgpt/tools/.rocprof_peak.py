#!/usr/bin/env python3
"""Peak GPU memory (MiB) from a rocprofv3 --memory-allocation-trace output dir.

Replays the alloc/free events (running live-byte sum, tracked by address,
filtered to GPU agents) and reports the peak — the same method as
compare_range.py's peak_gpu_bytes(). Used by run_llm_demo.sh as the ROCm
fallback for the GPU-memory column when nvidia-smi is absent.

This is a WHOLE-RUN peak: an allocation trace has no clean mapping into the
binary's PHASE windows (different clock domain), so it fills the overall
gpu_peak_mib column only, not the per-phase *_gpu_peak_mib columns.

Usage: python3 .rocprof_peak.py <outdir> [-o <basename>]
Prints peak MiB (float) or nothing if the trace is missing/empty.
"""
import csv
import os
import sys

outdir = sys.argv[1] if len(sys.argv) > 1 else ""
base = "run"
if "-o" in sys.argv:
    base = sys.argv[sys.argv.index("-o") + 1]

trace = os.path.join(outdir, f"{base}_memory_allocation_trace.csv")
agents = os.path.join(outdir, f"{base}_agent_info.csv")
if not (os.path.exists(trace) and os.path.exists(agents)):
    sys.exit(0)


def gpu_agent_ids(path):
    ids = set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("Agent_Type", "").strip().strip('"') == "GPU":
                ids.add(int(row["Logical_Node_Id"]))
    return ids


gpus = gpu_agent_ids(agents)
rows = []
with open(trace, newline="") as f:
    for r in csv.DictReader(f):
        rows.append((int(r["Start_Timestamp"]), r))
rows.sort(key=lambda x: x[0])

live, cur, peak = {}, 0, 0
for _, r in rows:
    op = r["Operation"].strip().strip('"')
    addr = r["Address"]
    if op == "MEMORY_ALLOCATION_ALLOCATE":
        if int(r["Agent_Id"]) in gpus:
            sz = int(r["Allocation_Size"])
            live[addr] = sz
            cur += sz
            peak = max(peak, cur)
    elif op == "MEMORY_ALLOCATION_FREE":
        cur -= live.pop(addr, 0)

if peak > 0:
    print(f"{peak / (1024 * 1024):.1f}")
