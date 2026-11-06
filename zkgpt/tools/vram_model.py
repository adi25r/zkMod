#!/usr/bin/env python3
"""vram_model.py — per-partition device VRAM as arithmetic, not as a sampled peak.

Every device buffer in the staged provers is <count> * sizeof(gpu_fe), and every
count is a closed form in the circuit shape and the tuning knobs:

  l_stage    = ceil(l / k)                      clamped to [1, l]
  ptable_cap = 4^l_stage (deg3) | 3^l_stage (matmul)
  fv_cap     = min(l - l_stage, sc_tile_bits)
  ext_cap    = ptable_cap << fv_cap
  dst_cap    = max(1, N >> l_stage)
  rmb        = ceil(2^(l_stage-1) / threads)

so a partition's footprint is decidable before any GPU work. gpu_peak_mib is a
1 ms-sampled per-process high-water mark (gpu_uvm_alloc.hpp:63): it reports a
LOWER bound on one process and, by construction, cannot see a sibling. Sizing a
concurrent width from it either over-counts (peaks that never coincide) or
under-counts (peaks missed between samples). This computes the bound instead.

Inputs are the [mm]/[deg3] lines the provers already emit under GKR_DUMP_SIZES=1
and GKR_DEG3_DBG=1, which carry the shape (n, N, l) each sumcheck actually ran.

  tools/vram_model.py --log run.log --width-budget 58
"""
import argparse
import re
import sys

FE = 32                      # sizeof(gpu_fe): blstrs__scalar__Scalar, 8 x uint32
PTABLE_TARGET_BLOCKS = 2048  # both provers
THREADS = 256

MM_RE = re.compile(r"\[mm\] n=(\d+) l_stage=(\d+) N=(\d+) ptable_cap=(\d+) "
                   r"ext_cap=(\d+) tile_bits=(-?\d+) fv_cap=(\d+) \| stream=(\w+)")
DEG3_RE = re.compile(r"\[deg3\] l=(\d+) k=(\d+) l_stage=(\d+) t=(-?\d+) "
                     r"fv0=(\d+) fv_cap=(\d+)")
PEAK_RE = re.compile(r"gpu_peak_mib=([0-9.eE+-]+)")


def rmb(l_stage):
    return max(1, ((1 << (l_stage - 1)) + THREADS - 1) // THREADS)


def mm_bytes(n, l_stage, N, ptable_cap, ext_cap, streaming, win=0):
    """gpu_matmul_prover.hpp:203-232. Live set for one product_sumcheck call."""
    dst_cap = max(1, N >> l_stage)
    if streaming:
        src = 2 * win + 2 * max(1, N >> (2 * l_stage))
    else:
        src = 2 * N                       # caller's bkA/bkB, resident
    elems = (2 * ext_cap + src + 2 * dst_cap + 2 * ptable_cap
             + (1 << l_stage)             # d_eq
             + (ptable_cap + PTABLE_TARGET_BLOCKS)   # d_partials
             + 3 * rmb(l_stage))          # d_p0..d_p2
    return elems * FE


def deg3_bytes(l, l_stage, fv_cap):
    """gpu_deg3_staged.hpp:273-314. Live set for one gpu_sumcheck_deg3_staged call."""
    N = 1 << l
    ptable_cap = 4 ** l_stage
    ext_cap = ptable_cap << fv_cap
    elems = (6 * N                        # dE,dF,dG,dE2,dF2,dG2
             + 3 * ext_cap                # dEe,dFe,dGe
             + 2 * ptable_cap             # dPs,dPd
             + (ptable_cap + PTABLE_TARGET_BLOCKS)   # dPart
             + 4 * rmb(l_stage))          # dp0..dp3
    return elems * FE


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True,
                    help="run log produced with GKR_DUMP_SIZES=1 GKR_DEG3_DBG=1")
    ap.add_argument("--width-budget", type=float, default=58.0,
                    help="usable VRAM in GiB (of 64, leaving fragmentation headroom)")
    a = ap.parse_args()

    text = open(a.log, errors="replace").read()
    calls = []
    for m in MM_RE.finditer(text):
        n, ls, N, pc, ec, _tb, _fv, st = m.groups()
        b = mm_bytes(int(n), int(ls), int(N), int(pc), int(ec), st == "on")
        calls.append((b, f"matmul n={n} l_stage={ls} N={N}"))
    for m in DEG3_RE.finditer(text):
        l, _k, ls, _t, _fv0, fvc = m.groups()
        b = deg3_bytes(int(l), int(ls), int(fvc))
        calls.append((b, f"deg3   l={l} l_stage={ls}"))

    if not calls:
        sys.exit("no [mm]/[deg3] lines: re-run with GKR_DUMP_SIZES=1 GKR_DEG3_DBG=1")

    calls.sort(reverse=True)
    gib = lambda b: b / (1024.0 ** 3)
    peak_b, peak_tag = calls[0]

    print(f"{len(calls)} staged sumcheck calls; largest live sets:")
    seen = set()
    for b, tag in calls:
        key = tag.split(" N=")[0]
        if key in seen:
            continue
        seen.add(key)
        print(f"   {gib(b):7.3f} GiB   {tag}")
        if len(seen) >= 8:
            break

    print(f"\nfloor (largest single call)  {gib(peak_b):.3f} GiB   [{peak_tag}]")
    obs = [float(x) for x in PEAK_RE.findall(text)]
    if obs:
        o = max(obs) / 1024.0
        print(f"sampled gpu_peak_mib (max)   {o:.3f} GiB")
        print(f"persistent residue           {o - gib(peak_b):+.3f} GiB "
              f"(commit/predicate buffers live across calls)")
        per = o
    else:
        per = gib(peak_b)

    print(f"\nwidth at {a.width_budget:.0f} GiB budget: {max(1, int(a.width_budget // per))}"
          f"  (per-partition {per:.2f} GiB)")


if __name__ == "__main__":
    main()
