#!/usr/bin/env python3
"""Classify the allocations live at the traced peak into memory categories.

  python3 plots/categorize_memtrace.py <traced partition binary> plots/data/mem_*_peak.csv > plots/data/mem_breakdown.csv

Frames in *_peak.csv are raw backtrace_symbols strings; static functions only carry a
binary offset, so they are resolved with addr2line against the traced binary that wrote them.
"""
import csv, re, subprocess, sys, os

CATEGORIES = [   # first match wins, checked against every resolved frame of an allocation
    ("sumcheck working set", r"blendy::Sumcheck|_legacy|Fr_hp_sc|Fr_ip_sc|step_poly|zkip\b|multi_hadamard"),
    ("weight copies (zkFC, opening pads)", r"zkFC::zkFC|verifyWeightClaim|open_vector|Commitment::open|me_open"),
    ("streamed-weight chunks and folds", r"ChunkStager|eq_table_mont|streamed_prove_linear"),
    ("resident layer weights", r"create_weight|OPTLayerWeights"),
    ("lookup proof tensors", r"tLookup::prove|tLookup::prep|Rescaling::prove|tLookupRangeMapping::prove"),
    ("lookup tables", r"tLookup::tLookup|tLookupRange::tLookupRange|tLookupRangeMapping::tLookupRangeMapping|Rescaling::Rescaling"),
    ("activations", r"run_opt_layer|opt-layer\.cu"),
]
OTHER = "input / other"


def name_only(sym):
    """Function name without its parameter list, so parameter types never match a category."""
    return sym.split("(")[0] if not sym.startswith("operator()") else sym


def resolver(binary):
    cache = {}
    def res(frame):
        m = re.search(r"\(\+(0x[0-9a-f]+)\)", frame)
        if m:
            off = m.group(1)
            if off not in cache:
                o = subprocess.run(["addr2line", "-f", "-C", "-e", binary, off], capture_output=True, text=True).stdout.split("\n")
                cache[off] = name_only(o[0]) + " @" + (o[1].split("/")[-1] if len(o) > 1 else "")
            return cache[off]
        m = re.search(r"\(([^+)]+)\+", frame)
        if m:
            name = m.group(1)
            if name not in cache:
                cache[name] = name_only(subprocess.run(["c++filt", name], capture_output=True, text=True).stdout.strip())
            return cache[name]
        return frame
    return res


def main():
    binary, files = sys.argv[1], sys.argv[2:]
    res = resolver(binary)
    w = csv.writer(sys.stdout)
    w.writerow(["model", "config", "category", "bytes"])
    for f in files:
        m = re.match(r"mem_(.+?)_(legacy|blendy_untiled|blendy|wstream)_peak\.csv", os.path.basename(f))
        model, config = m.group(1), m.group(2)
        tot = {}
        for r in csv.DictReader(open(f)):
            frames = " | ".join(res(x) for x in r["frames"].split(";"))
            cat = next((c for c, pat in CATEGORIES if re.search(pat, frames)), OTHER)
            tot[cat] = tot.get(cat, 0) + int(r["bytes"])
        for cat, b in tot.items():
            w.writerow([model, config, cat, b])


if __name__ == "__main__":
    main()
