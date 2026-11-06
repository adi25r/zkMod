#!/usr/bin/env python3
"""Build exp4_summary.csv from a run_exp4_llama8b.sh driver log.

The driver prints one `[exp4] RESULT <tag> ...` line per config. This turns that
into a tidy CSV the plotter reads. Kept as a script (not an ad-hoc shell one-liner)
so a re-sweep is one command and the status vocabulary is defined in exactly one
place:

  OK       at least one partition completed; makespan is max over those partitions
  OOM      the config could not allocate — an INFEASIBLE point, not a slow one
  NOT_RUN  the driver skipped it (e.g. an unrunnable c); must never be drawn as OOM

Usage:
  tools/exp4_summarize.py tools/results/e2e_v3/data/exp4_map.log \
      [more_logs...] -o tools/results/e2e_v3/data/exp4_summary.csv
"""
import argparse
import csv
import os
import re
import sys

# tags look like  t20_k8_c13  or  uvm_t20_k8_c7
TAG_RE = re.compile(r"^(?:(uvm)_)?t(-?\d+)_k(\d+)_c(\d+)$")
RESULT_RE = re.compile(r"\[exp4\] RESULT (\S+)\s+(.*)")


def parse_log(path, only=None):
    rows = {}
    for line in open(path):
        m = RESULT_RE.search(line)
        if not m:
            continue
        tag, rest = m.group(1), m.group(2).strip()
        if only and not any(tag.startswith(p) for p in only):
            continue
        tm = TAG_RE.match(tag)
        if not tm:
            print(f"warn: unparseable tag {tag!r}, skipping", file=sys.stderr)
            continue
        uvm, t, k, c = tm.groups()
        row = dict(tag=tag, t=int(t), k=int(k), c=int(c), uvm=int(bool(uvm)),
                   parts=0, makespan_core_s="", makespan_total_s="", peak_mib="",
                   src=os.path.basename(path))
        if "NO PARTITIONS" in rest:
            row["status"] = "OOM"
        elif "no log found" in rest:
            row["status"] = "NOT_RUN"
        else:
            g = dict(re.findall(r"(\w+)=([\d.]+)", rest))
            row.update(parts=int(float(g.get("parts", 0))),
                       makespan_core_s=g.get("makespan_core_s", ""),
                       makespan_total_s=g.get("makespan_total_s", ""),
                       peak_mib=g.get("peak_mib", ""),
                       status="OK" if g.get("parts") else "NOT_RUN")
        # later logs win, so a re-run of one config supersedes the earlier probe
        rows[tag] = row
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", help="driver log(s); later files win on tag collision")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--only", action="append", metavar="PREFIX",
                    help="keep only tags with this prefix. Applies to EVERY log, so "
                         "use it when borrowing one arm (e.g. --only uvm_) from an "
                         "older sweep without importing its device rows -- those were "
                         "taken under a different timeout and are not comparable.")
    a = ap.parse_args()

    rows = {}
    for p in a.logs:
        if not os.path.exists(p):
            print(f"warn: missing {p}", file=sys.stderr)
            continue
        rows.update(parse_log(p, only=a.only))

    if not rows:
        raise SystemExit("no RESULT lines found")

    cols = ["tag", "t", "k", "c", "uvm", "parts",
            "makespan_core_s", "makespan_total_s", "peak_mib", "status", "src"]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for tag in sorted(rows, key=lambda x: (rows[x]["uvm"], rows[x]["c"],
                                               rows[x]["t"], rows[x]["k"])):
            w.writerow(rows[tag])

    n_ok = sum(1 for r in rows.values() if r["status"] == "OK")
    n_oom = sum(1 for r in rows.values() if r["status"] == "OOM")
    print(f"wrote {a.out}  ({len(rows)} configs: {n_ok} OK, {n_oom} OOM, "
          f"{len(rows)-n_ok-n_oom} NOT_RUN)")


if __name__ == "__main__":
    main()
