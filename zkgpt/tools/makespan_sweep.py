#!/usr/bin/env python3
"""makespan_sweep.py — Measured makespan vs the Blendy stage parameter k, and
validation of the cost-equation predictions against ground-truth wall-clock.

Backs experiments.tex Exp 5/6: prove that choosing k by the cost/VRAM equations
lands at (or near) the makespan optimum found by exhaustive grid search.

A "wave" = W subproofs launched concurrently on ONE GPU, W = floor(M_GPU/peak(k)).
makespan(k) = wall-clock to clear `batch` subproofs in ceil(batch/W) waves.
Each subproof = one matmul Blendy product-sumcheck (s=2^n, k stages) via the
zkgpt_gpu micro-bench. Concurrency is multi-process (low-occupancy high-k kernels
co-execute on idle CUs).

Two predictors are scored against the measured grid-search argmin:
  Model A (naive, as in experiments.tex):  makespan_A(k) = ceil(B/W(k)) * iso(k)
  Model B (resource-area throughput):       makespan_B(k) = B * iso(k) * max(util(k)/100, peak(k)/M)

Usage:
  tools/makespan_sweep.py --n 24 --batches 24 --ks 6,8,12,16 --trials 3
  tools/makespan_sweep.py --n 18,20,22,24,26 --batches 8,16,24 --trials 3 \
       --out tools/makespan_broad.csv
"""
import argparse, csv, os, subprocess, sys, time, math, re, glob, statistics
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
TIME_RE = re.compile(r"time_s=([0-9.eE+-]+)")

def load_exp1(csv_glob):
    """(n,k) -> dict(iso, valu, mem, peak) from most recent exp1 matmul GPU rows."""
    files = sorted(glob.glob(os.path.join(HERE, "**", csv_glob), recursive=True))
    tab = {}
    if not files:
        return tab
    for r in csv.DictReader(open(files[-1])):
        if r.get("mode") != "gpu":
            continue
        try:
            tab[(int(r["n"]), int(r["k"]))] = dict(
                iso=float(r["time_s"]), peak=float(r["peak_gpu_bytes"]),
                valu=float(r["valu_busy"] or 0), mem=float(r["mem_busy"] or 0))
        except (ValueError, KeyError):
            pass
    return tab

def run_one(binpath, n, k, gpu):
    env = dict(os.environ, HIP_VISIBLE_DEVICES=str(gpu))
    p = subprocess.run([binpath, "--bench-matmul", str(n), str(k)], env=env,
                       capture_output=True, text=True)
    m = TIME_RE.search(p.stdout)
    return (float(m.group(1)) if m else float("nan")), p.returncode

def run_batch(binpath, n, k, gpu, batch, W):
    """Run `batch` subproofs in waves of W; return (makespan, mean_region, fails)."""
    t0 = time.perf_counter()
    regions, fails, remaining = [], 0, batch
    while remaining > 0:
        w = min(W, remaining)
        with ThreadPoolExecutor(max_workers=w) as ex:
            res = list(ex.map(lambda _: run_one(binpath, n, k, gpu), range(w)))
        regions += [r[0] for r in res]
        fails += sum(1 for r in res if r[1] != 0)
        remaining -= w
    return time.perf_counter() - t0, statistics.mean(regions), fails

def wave_width(info, budget, batch):
    if info and info["peak"] > 0:
        return max(1, min(batch, int(budget // info["peak"])))
    return batch

def predict(ks, exp1, n, batch, budget):
    """Model A and B predicted-best k (None if data missing)."""
    A, B = {}, {}
    for k in ks:
        info = exp1.get((n, k))
        if not info:
            continue
        W = wave_width(info, budget, batch)
        waves = math.ceil(batch / W)
        A[k] = waves * info["iso"]
        util = max(info["valu"], info["mem"]) / 100.0
        B[k] = batch * info["iso"] * max(util, info["peak"] / budget)
    pa = min(A, key=A.get) if A else None
    pb = min(B, key=B.get) if B else None
    return pa, pb, A, B

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", default=os.path.join(HERE, "..", "bin", "zkgpt_gpu"))
    ap.add_argument("--n", default="24")
    ap.add_argument("--ks", default="2,4,6,8,12,16")
    ap.add_argument("--batches", default="8,16,24")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--mem-gb", type=float, default=60.0)
    ap.add_argument("--ext-cap-log2", type=int, default=27,
                    help="skip (n,k) whose base-3 working set exceeds this (OOM guard)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    binpath = os.path.abspath(args.bin)
    if not os.path.exists(binpath):
        sys.exit(f"missing binary {binpath}")
    ns = [int(x) for x in args.n.split(",") if x.strip()]
    ks_all = [int(x) for x in args.ks.split(",") if x.strip()]
    batches = [int(x) for x in args.batches.split(",") if x.strip()]
    exp1 = load_exp1("exp1_matmul_*.csv")
    budget = args.mem_gb * 1e9
    ext_cap = 1 << args.ext_cap_log2
    rows = []
    print(f"makespan sweep  ns={ns} ks={ks_all} batches={batches} "
          f"trials={args.trials} gpu={args.gpu} M_GPU={args.mem_gb}GB\n")

    for n in ns:
        # feasible k: base-3 working set 3^ceil(n/k)*2^(n-l) within ext_cap
        ks = []
        for k in ks_all:
            if k > n:
                continue
            l = (n + k - 1) // k
            if (3 ** l) * (1 << (n - l)) <= ext_cap:
                ks.append(k)
        for batch in batches:
            measured = {}
            for k in ks:
                info = exp1.get((n, k))
                W = wave_width(info, budget, batch)
                waves = math.ceil(batch / W)
                trials = []
                fails_tot = 0
                for _ in range(args.trials):
                    ms, mean_r, fails = run_batch(binpath, n, k, args.gpu, batch, W)
                    trials.append(ms); fails_tot += fails
                med = statistics.median(trials)
                measured[k] = med
                pk = (info["peak"] / 1e9) if info else float("nan")
                rows.append(dict(n=n, batch=batch, k=k, W=W, num_waves=waves,
                                 peak_gb=round(pk, 3),
                                 makespan_med=round(med, 4),
                                 makespan_min=round(min(trials), 4),
                                 makespan_max=round(max(trials), 4),
                                 trials=args.trials, fails=fails_tot))
                print(f"  n={n} B={batch:<3d} k={k:<2d} W={W:<3d} waves={waves:<2d} "
                      f"peak={pk:6.3f}GB  makespan_med={med:8.3f}s "
                      f"[{min(trials):.2f},{max(trials):.2f}]"
                      f"{'  FAILS='+str(fails_tot) if fails_tot else ''}")
            if not measured:
                continue
            best_k = min(measured, key=measured.get)
            pa, pb, A, B = predict(ks, exp1, n, batch, budget)
            def gap(pk):
                if pk is None or pk not in measured:
                    return float("nan")
                return 100.0 * (measured[pk] - measured[best_k]) / measured[best_k]
            print(f"  => n={n} B={batch}: MEASURED argmin k={best_k} "
                  f"({measured[best_k]:.2f}s) | ModelA picks k={pa} (gap {gap(pa):+.1f}%) "
                  f"| ModelB picks k={pb} (gap {gap(pb):+.1f}%)\n")
            rows.append(dict(n=n, batch=batch, k="SUMMARY", W="", num_waves="",
                             peak_gb="", makespan_med="",
                             makespan_min=f"measured_argmin={best_k}",
                             makespan_max=f"A={pa}(gap{gap(pa):+.1f}%);B={pb}(gap{gap(pb):+.1f}%)",
                             trials="", fails=""))
    if args.out:
        with open(args.out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader(); wr.writerows(rows)
        print(f"wrote {args.out}")

if __name__ == "__main__":
    main()
