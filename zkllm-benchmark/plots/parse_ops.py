#!/usr/bin/env python3
"""OPS_SUMMARY lines (op-partition.py) -> CSV, one variant column per group of summary logs.

  python3 plots/parse_ops.py field=results-ops-4090 astream=results-ops-4090-astream \
      intmm=results-ops-4090-intmm bopen=results-ops-4090-bopen > plots/data/op_partition_4090.csv

A bare path (no "name=") is read as variant "field". A directory means its opt-*/summary.log.
"""
import csv, glob, os, sys

rows = []
for arg in sys.argv[1:]:
    variant, _, path = arg.rpartition("=")
    variant = variant or "field"
    files = sorted(glob.glob(os.path.join(path, "opt-*", "summary.log"))) if os.path.isdir(path) else [path]
    for f in files:
        for line in open(f):
            if line.startswith("OPS_SUMMARY "):
                rows.append({"variant": variant, **dict(kv.split("=", 1) for kv in line.split()[1:])})
if rows:
    w = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
