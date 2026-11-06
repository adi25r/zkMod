#!/usr/bin/env python3
"""run_parallel_partitions.py — Exp 5D: can the zkMod makespan be reached on ONE GPU?

The seq-len ladder reports two numbers for the split arms (see run_seqlen_ladder.sh):
the single-GPU sequential SUM that main_gkr_split actually costs, and the MAX over
partitions -- a makespan that assumes one GPU per partition. This driver tests the
third, previously unmeasured execution model: run the partitions CONCURRENTLY on a
single GPU and measure the wall clock. That number must land between the two.

A "wave" = W single-partition proofs launched concurrently on one GPU (the same
pattern as makespan_sweep.py's run_batch, with the matmul micro-bench swapped for a
full one-layer proof). Each subproof is `main_gkr_split <depth=1> <attn> <splits=1>`,
i.e. one transformer layer proven standalone. MEASUREMENT MODE: a standalone block
omits the cross-block boundary commits, so this is valid for time and memory but is
not an end-to-end proof -- same caveat as --unique-parts.

Why the configs differ sharply. Device-only concurrency width is W = floor(VRAM/peak):
  untiled (t=-1) peak ~25 GB at len=64  -> W = 2   (needs UVM to go wider)
  tuned  (t=18)  peak ~5.6 GB at len=64 -> W = 11  (nearly all 12 fit natively)
So tiling is what makes single-GPU wave-packing practical; that is the claim under
test.

SAFETY (this is a shared box):
  * Host RAM is the real limit, not VRAM: each partition holds tens of GB of host
    witness (measured host_peak_mib ~24 GB at len=30, ~50 GB at len=64), so 12
    concurrent runs want 290-600 GB. The width is capped by a host-RAM budget and
    the run ABORTS rather than swapping the machine.
  * Each subproof takes --threads CPU threads; 12 x 32 = 384 = every core on this
    box, so high widths are CPU-oversubscribed. Recorded in the CSV; interpret the
    parallel makespan with that in mind.
  * UVM configs are opt-in (--allow-uvm). Managed runs have wedged this driver
    before. They run last, one at a time, with a health probe, and are stopped with
    SIGTERM (never SIGKILL) on timeout.

Usage:
  tools/run_parallel_partitions.py --len 30 --gpu 4 --configs tuned-device,untiled-device
  tools/run_parallel_partitions.py --len 30 --gpu 4 --configs untiled-uvm --allow-uvm
"""
import argparse
import csv
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

TOT_RE = re.compile(r"prover_total_s=([0-9.eE+-]+)")
PEAK_RE = re.compile(r"gpu_peak_mib=([0-9.eE+-]+)")
HOST_RE = re.compile(r"host_peak_mib=([0-9.eE+-]+)")

VRAM_MIB = 64 * 1024          # MI210
GIB = 1024.0

HOST_RESERVE_GIB = 128.0      # absolute floor left for the OS, page cache, other tenants
HOST_FRAC = 0.75              # of what remains after the reserve
MEM_SETTLE_GIB = 64.0         # extra free RAM to wait for before a probe/run
MEM_SETTLE_WAIT_S = 300       # give up waiting after this and proceed anyway


def budget_gib(a):
    """Host budget = fraction of (available - absolute reserve).

    Both terms matter: the fraction scales with the box, the reserve keeps a fixed
    cushion that a big model's per-worker peak cannot eat into.
    """
    if a.host_budget_gib:
        return a.host_budget_gib
    return max(1.0, (host_free_gib() - HOST_RESERVE_GIB) * HOST_FRAC)


def classify_rc(rc):
    """Why a subproof died. The old code called every abnormal exit 'TIMEOUT'
    because it only looked for a missing prover_total_s, which hid the OOM kills
    behind a label that pointed at the wrong knob."""
    if rc == 0:
        return "OK"
    if rc == 124:
        return "TIMEOUT"        # driver's own SIGTERM after --timeout
    if rc in (-9, 137):
        return "OOM_KILL"       # kernel OOM killer
    if rc in (-15, 143):
        return "SIGTERM"        # killed by something outside this driver
    return "FAIL"


def wait_for_host_mem(need_gib, what):
    """Block until the box actually has the RAM, so one heavy run cannot poison the
    next one's probe. This is exactly how the opt2b7 probes died on 2026-07-31: they
    launched 8 s after an OOM storm and asked for ~182 GiB straight away."""
    want = need_gib + MEM_SETTLE_GIB
    t0 = time.time()
    while host_free_gib() < want and time.time() - t0 < MEM_SETTLE_WAIT_S:
        print(f"[D]   waiting for host RAM before {what}: "
              f"avail={host_free_gib():.0f} GiB < want={want:.0f} GiB", flush=True)
        time.sleep(30)
    return host_free_gib()

# config -> (binary, tile_bits, range_c, lasso_k, uvm)
CONFIGS = {
    "untiled-device": ("main_gkr_split",     "-1", 9,  8,  False),
    "tuned-device":   ("main_gkr_split",     "18", 9,  16, False),
    "untiled-uvm":    ("main_gkr_split_uvm", "-1", 9,  8,  True),
    "tuned-uvm":      ("main_gkr_split_uvm", "18", 9,  16, True),
    # "custom" is filled in from --custom-tile/--custom-c/--custom-lasso-k at
    # runtime (see main()), so a sweep can probe arbitrary (c,k,t) against the
    # concurrent-makespan objective instead of the four fixed presets above.
}


def host_free_gib():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024.0 * 1024.0)
    return 0.0


def gpu_ok(gpu):
    """A GPU that reports free VRAM but fails every hipMalloc is the wedge signature."""
    try:
        p = subprocess.run([os.path.join(ROOT, "bin", "zkgpt_gpu"),
                            "1", "12", "64", "768", "3072", "--bench-matmul", "16", "4"],
                           env=dict(os.environ, HIP_VISIBLE_DEVICES=str(gpu)),
                           capture_output=True, text=True, timeout=120)
        return p.returncode == 0
    except subprocess.SubprocessError:
        return False


def run_one(cfg, a, idx, threads, stream_bits=-1):
    """One standalone single-layer proof. Returns (prover_total_s, peak_mib, host_mib, rc).

    `threads` is passed explicitly rather than read off `a` so the caller can split a
    fixed HOST BUDGET across the concurrent workers -- see main(). The CPU harness
    (run_cpu_parallel_partitions.py) has always done this; this one used to give every
    worker a fixed 32, so the GPU arm's total host parallelism scaled with the width
    while the CPU arm's stayed at 320. That made the two arms incomparable exactly
    where the width collapses (wide models), which is where the GPU looked worst."""
    binname, tile, c, lk, uvm = CONFIGS[cfg]
    env = dict(os.environ,
               HIP_VISIBLE_DEVICES=str(a.gpu),
               GKR_SC_TILE_BITS=tile)
    # Stage-1 source streaming. -1 = dense. Streaming trades ~4% of per-partition
    # prover time (the source is uploaded twice per stage-1) for a much smaller
    # device peak, so it is only worth enabling when that peak is what caps the
    # wave width -- see pick_stream().
    env["GKR_SC_STREAM_BITS"] = str(stream_bits)
    if uvm:
        env["HSA_XNACK"] = "1"
    # The circuit width is next_pow2(len), rounded by the binary itself (the circuit is
    # LLMSplitBlock, modular-zkp's).
    cmd = [os.path.join(ROOT, "bin", binname),
           str(a.group_layers), str(a.attn), "1", str(a.len), str(threads), str(a.lin),
           "--c", str(c), "--matmul-k", os.environ.get("PP_MATMUL_K", "4"),
           "--range-k", os.environ.get("PP_RANGE_K", "8"), "--lasso-k", str(lk)]
    # k is the STAGE COUNT, not a stage width, so k=1 means one stage over all l
    # variables and the deg3 extension table is 3*4^l entries -- 393216 GiB at l=21.
    # The binary hard-rejects such a schedule ("a reported k must be the k that
    # ran", gpu_deg3_staged.hpp), so the point aborts with rc=-6. k=4 gives l_stage~6 => ~12 GiB
    # at tile_bits=16, feasible across this grid. Override with PP_MATMUL_K.
    # start_new_session so a timeout kills the whole process group, and so a
    # SIGTERM here never reaches this driver.
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, start_new_session=True)
    try:
        out, _ = p.communicate(timeout=a.timeout)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        # SIGTERM only: SIGKILL on a page-faulting managed run wedges the driver.
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        try:
            out, _ = p.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            out = ""
        rc = 124
    tot = TOT_RE.search(out or "")
    peak = PEAK_RE.search(out or "")
    host = HOST_RE.search(out or "")
    if a.keep_logs:
        with open(os.path.join(a.logdir, f"{cfg}_len{a.len}_p{idx}.log"), "w") as f:
            f.write(out or "")
    return (float(tot.group(1)) if tot else float("nan"),
            float(peak.group(1)) if peak else float("nan"),
            float(host.group(1)) if host else float("nan"),
            rc)


def probe(cfg, a, stream_bits=-1):
    """One isolated run -> (prover_total_s, gpu_peak_mib, host_peak_mib).

    The width has to come from a MEASURED peak, not a guess: picking W too large is
    how you either OOM the card or swap the host.
    """
    print(f"[D] probing {cfg} len={a.len} (1 isolated partition) ...", flush=True)
    # A wide model's single partition is itself a big allocation (opt2b7 ~182 GiB),
    # so the probe gets the same settle-wait the concurrent waves do.
    wait_for_host_mem(0.0, f"probe {cfg} len={a.len}")
    # Full budget for the isolated probe, as run_cpu_parallel_partitions.py does.
    tot, peak, host, rc = run_one(cfg, a, "probe", a.threads_total, stream_bits)
    print(f"[D]   probe rc={rc} ({classify_rc(rc)}) prover_total_s={tot:.4g} "
          f"gpu_peak={peak/GIB:.2f} GiB host_peak={host/GIB:.2f} GiB", flush=True)
    return tot, peak, host, rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--len", type=int, default=30)
    ap.add_argument("--parts", type=int, default=12, help="partitions to clear (GPT-2 L=12)")
    ap.add_argument("--group-layers", type=int, default=1,
                    help="transformer layers combined into ONE partition/process (>1 "
                         "trades a bigger per-partition circuit -- more device memory, "
                         "more n -- for FEWER concurrent processes, i.e. more threads/worker "
                         "at a given wave width). --parts stays the total model depth; the "
                         "number of concurrent PROCESSES is ceil(parts/group_layers).")
    ap.add_argument("--gpu", type=int, default=4)
    ap.add_argument("--attn", type=int, default=768)
    ap.add_argument("--lin", type=int, default=3072)
    ap.add_argument("--threads", type=int, default=None,
                    help="fixed threads PER WORKER (legacy). Omit to split --threads-total "
                         "across the concurrent workers, which is what the CPU harness does.")
    ap.add_argument("--threads-total", type=int, default=320,
                    help="host thread budget shared by the concurrent workers; matches "
                         "run_cpu_parallel_partitions.py --threads-total")
    ap.add_argument("--configs", default="tuned-device,untiled-device")
    ap.add_argument("--custom-tile", default=None,
                    help="GKR_SC_TILE_BITS for the 'custom' config (enables it)")
    ap.add_argument("--custom-c", type=int, default=None,
                    help="range chunk bits (--c) for the 'custom' config")
    ap.add_argument("--custom-lasso-k", type=int, default=None,
                    help="lasso-k for the 'custom' config")
    ap.add_argument("--width", type=int, default=0, help="0 = derive from measured peak")
    ap.add_argument("--stream", choices=["auto", "on", "off"], default="auto",
                    help="stage-1 source streaming. auto = enable only when the DEVICE "
                         "peak is what caps the wave width AND streaming actually "
                         "removes a wave (see main()); on/off force it.")
    ap.add_argument("--stream-bits", type=int, default=22,
                    help="log2 of the streaming window in field elements (22 = 4M "
                         "elements = 128 MiB per side)")
    ap.add_argument("--timeout", type=int, default=3600, help="per-subproof seconds")
    ap.add_argument("--host-budget-gib", type=float, default=0.0,
                    help="0 = use 80%% of currently-available host RAM")
    ap.add_argument("--allow-uvm", action="store_true")
    ap.add_argument("--keep-logs", action="store_true", default=True)
    ap.add_argument("--out", default=os.path.join(HERE, "results/seqlen/data/parallel_partitions.csv"))
    ap.add_argument("--logdir", default=os.path.join(HERE, "results/seqlen/logs/parallel"))
    a = ap.parse_args()

    os.makedirs(a.logdir, exist_ok=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    if a.custom_tile is not None or a.custom_c is not None or a.custom_lasso_k is not None:
        if a.custom_tile is None or a.custom_c is None or a.custom_lasso_k is None:
            sys.exit("--custom-tile/--custom-c/--custom-lasso-k must be given together")
        CONFIGS["custom"] = ("main_gkr_split", str(a.custom_tile), a.custom_c,
                              a.custom_lasso_k, False)

    cfgs = [c.strip() for c in a.configs.split(",") if c.strip()]
    for c in cfgs:
        if c not in CONFIGS:
            sys.exit(f"unknown config {c}; choose from {','.join(CONFIGS)}")
    # Device configs first, UVM last -- a wedge should never be able to corrupt a
    # measurement that had not run yet.
    cfgs.sort(key=lambda c: CONFIGS[c][4])
    if any(CONFIGS[c][4] for c in cfgs) and not a.allow_uvm:
        sys.exit("UVM configs requested without --allow-uvm")

    budget = budget_gib(a)
    # n_procs = number of concurrent OS PROCESSES to launch. With group_layers=1
    # (the default, and every prior use of this script) this equals a.parts exactly,
    # so nothing below changes behavior unless --group-layers is passed. >1 groups
    # that many consecutive layers into ONE combined circuit per process -- fewer,
    # bigger partitions -- which is what a.parts is NOT: a.parts stays the true model
    # depth so the CSV/-- prints keep reporting "how much model this run covers".
    n_procs = -(-a.parts // a.group_layers)
    print(f"[D] gpu={a.gpu} len={a.len} parts={a.parts} group_layers={a.group_layers} "
          f"n_procs={n_procs} threads_total={a.threads_total} "
          f"host_avail={host_free_gib():.0f} GiB reserve={HOST_RESERVE_GIB:.0f} GiB "
          f"budget={budget:.0f} GiB", flush=True)

    new = not os.path.exists(a.out)
    fh = open(a.out, "a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(["config", "len", "parts", "group_layers", "n_procs", "width", "waves",
                    "threads", "gpu", "status", "parallel_makespan_s", "iso_prover_total_s",
                    "seq_sum_est_s", "mean_prover_total_s", "gpu_peak_mib",
                    "host_peak_mib", "host_total_gib", "fails", "threads_total",
                    "stream_bits"])

    for cfg in cfgs:
        uvm = CONFIGS[cfg][4]
        if uvm and not gpu_ok(a.gpu):
            print(f"[D] ABORT: gpu {a.gpu} unhealthy before {cfg}", file=sys.stderr)
            break

        def widths(peak, host):
            """(w_vram, w_host, W, waves) for a measured per-partition pair."""
            wv = n_procs if uvm else (max(1, int(VRAM_MIB // peak)) if peak > 0 else n_procs)
            wh = max(1, int(budget // (host / GIB))) if host > 0 else n_procs
            W_ = a.width or max(1, min(n_procs, wv, wh))
            return wv, wh, W_, -(-n_procs // W_)

        stream_bits = -1
        iso_t, peak, host, rc = probe(cfg, a, -1)
        if rc != 0:
            w.writerow([cfg, a.len, a.parts, a.group_layers, n_procs, "", "", a.threads_total, a.gpu,
                        "PROBE_FAIL", "", "", "", "", "", "", "", 1, a.threads_total, -1]); fh.flush()
            continue
        w_vram, w_host, W, waves = widths(peak, host)

        # Should this cell stream its stage-1 source? Streaming shrinks the device
        # peak several-fold but costs ~4% of per-partition prover time (the source
        # is uploaded twice per stage 1). That trade only pays if the DEVICE peak is
        # what caps the wave width. When the host bound is already the tighter one,
        # freeing VRAM buys nothing and the 4% is pure loss -- so check first.
        #
        # A single prover process cannot make this call: 18.27 GB "fits" in 64 GB
        # every time. What does not fit is W of them, and only this driver knows W.
        if a.stream == "on":
            stream_bits = a.stream_bits
        elif a.stream == "auto" and not uvm:
            if w_vram >= w_host or W >= n_procs:
                print(f"[D] {cfg}: stream=off (VRAM is not the binding constraint: "
                      f"vram_allows={min(w_vram,n_procs)} >= host_allows={min(w_host,n_procs)})",
                      flush=True)
            else:
                print(f"[D] {cfg}: VRAM binds (vram_allows={min(w_vram,n_procs)} < "
                      f"host_allows={min(w_host,n_procs)}); probing streamed ...", flush=True)
                s_t, s_peak, s_host, s_rc = probe(cfg, a, a.stream_bits)
                if s_rc != 0:
                    print(f"[D] {cfg}: streamed probe rc={s_rc}, keeping dense", flush=True)
                else:
                    sv, sh, sW, s_waves = widths(s_peak, s_host)
                    # Fewer waves is the objective; on a tie keep dense, which is
                    # faster per partition.
                    if s_waves < waves:
                        print(f"[D] {cfg}: stream=ON  waves {waves}->{s_waves} "
                              f"(W {W}->{sW}, peak {peak/GIB:.2f}->{s_peak/GIB:.2f} GiB, "
                              f"iso {iso_t:.1f}->{s_t:.1f}s)", flush=True)
                        stream_bits = a.stream_bits
                        iso_t, peak, host = s_t, s_peak, s_host
                        w_vram, w_host, W, waves = sv, sh, sW, s_waves
                    else:
                        print(f"[D] {cfg}: stream=off (no wave gain: {waves} -> {s_waves})",
                              flush=True)
        # Split the host budget across the concurrent workers, exactly as the CPU
        # harness does (threads_per_worker = threads_total // W). --threads pins a
        # fixed per-worker count instead, for reproducing older runs.
        tpw = a.threads if a.threads else max(1, a.threads_total // W)
        print(f"[D] {cfg}: W={W} (vram_allows={min(w_vram,n_procs)} "
              f"host_allows={min(w_host,n_procs)}) waves={waves} "
              f"threads/worker={tpw} (total={tpw*W}) "
              f"host_need={W*host/GIB:.0f} GiB", flush=True)
        if W * host / GIB > budget * 1.05:
            print(f"[D] SKIP {cfg}: would exceed host budget", file=sys.stderr)
            w.writerow([cfg, a.len, a.parts, a.group_layers, n_procs, W, waves, tpw, a.gpu,
                        "HOST_LIMIT", "", f"{iso_t:.6g}", "", "", f"{peak:.6g}", f"{host:.6g}", "", 0,
                        a.threads_total, stream_bits])
            fh.flush(); continue

        # One full concurrent run at a FIXED width. An OOM kill invalidates the
        # makespan (the survivors ran against dying neighbours), so the whole run is
        # retried at a smaller width rather than patched up partition-by-partition --
        # a makespan has to belong to one width to mean anything.
        def run_at(width):
            tp = a.threads if a.threads else max(1, a.threads_total // width)
            wait_for_host_mem(width * host / GIB, f"{cfg} len={a.len} W={width}")
            t0 = time.perf_counter()
            tots, peaks, hosts, rcs, remaining, idx = [], [], [], [], n_procs, 0
            while remaining > 0:
                k = min(width, remaining)
                with ThreadPoolExecutor(max_workers=k) as ex:
                    res = list(ex.map(lambda j: run_one(cfg, a, idx + j, tp, stream_bits), range(k)))
                tots += [r[0] for r in res]
                peaks += [r[1] for r in res]
                hosts += [r[2] for r in res]
                rcs += [r[3] for r in res]
                remaining -= k; idx += k
            return time.perf_counter() - t0, tots, peaks, hosts, rcs, tp

        while True:
            wall, tots, peaks, hosts, rcs, tpw = run_at(W)
            kinds = [classify_rc(rc) for rc in rcs]
            fails = sum(1 for kd in kinds if kd != "OK")
            oom = sum(1 for kd in kinds if kd in ("OOM_KILL", "SIGTERM"))
            if oom == 0 or W <= 1 or a.width:
                break
            newW = max(1, min(W - 1, int(W * 0.7)))
            print(f"[D] {cfg}: {oom}/{n_procs} partitions killed under memory pressure "
                  f"at W={W} ({W*host/GIB:.0f} GiB); retrying whole run at W={newW}",
                  file=sys.stderr, flush=True)
            W = newW
            waves = -(-n_procs // W)

        good = [t for t in tots if t == t]
        if fails == 0:
            status = "OK"
        else:
            # Report what actually happened: labelling every abnormal exit
            # "TIMEOUT" would make a host-RAM problem look like a slow prover.
            worst = max(kinds, key=lambda kd: ("OK", "FAIL", "TIMEOUT", "SIGTERM",
                                               "OOM_KILL").index(kd))
            status = worst
        w.writerow([cfg, a.len, a.parts, a.group_layers, n_procs, W, waves, tpw, a.gpu, status,
                    f"{wall:.6g}", f"{iso_t:.6g}", f"{iso_t*n_procs:.6g}",
                    f"{statistics.mean(good):.6g}" if good else "",
                    f"{max(p for p in peaks if p == p):.6g}" if any(p == p for p in peaks) else "",
                    f"{max(h for h in hosts if h == h):.6g}" if any(h == h for h in hosts) else "",
                    f"{W*host/GIB:.6g}", fails, a.threads_total, stream_bits])
        fh.flush()
        print(f"[D] {cfg}: parallel_makespan={wall:.1f}s  (isolated={iso_t:.1f}s, "
              f"seq_sum_est={iso_t*n_procs:.1f}s)  W={W} status={status} fails={fails}",
              flush=True)

        if uvm and not gpu_ok(a.gpu):
            print(f"[D] WARNING: gpu {a.gpu} unhealthy AFTER {cfg} -- stopping",
                  file=sys.stderr)
            break

    fh.close()
    print(f"[D] wrote {a.out}")


if __name__ == "__main__":
    main()
