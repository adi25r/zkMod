#!/usr/bin/env python3
"""Parse one run's log + GPU-memory samples into a single CSV row for
run_llm_demo.sh. Not meant to be called directly — the sweep script sets the
inputs via environment variables.

Reads:
  LOG   path to the run's combined stdout/stderr
  VRAM  path to "epoch_seconds,mem_used_mib" samples (may be empty)
  MODEL DIMS L C K MODE STATUS WALL RC   run metadata

Per PHASE line (emitted by main_gkr.cpp) we get t0_us,t1_us,time_s and host
peak RSS; we compute that phase's GPU-memory peak as the max VRAM sample whose
timestamp falls in [t0,t1]. Missing values render as empty CSV fields.
"""
import os
import re

LOG = os.environ.get("LOG", "")
VRAM = os.environ.get("VRAM", "")


def read(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ""


text = read(LOG)

# ── VRAM samples: list of (epoch_seconds, mib) ───────────────────────────────
samples = []
for line in read(VRAM).splitlines():
    parts = line.split(",")
    if len(parts) != 2:
        continue
    try:
        samples.append((float(parts[0]), float(parts[1])))
    except ValueError:
        continue


def gpu_peak(t0_s, t1_s):
    """Max VRAM sample (MiB) in the wall-clock window [t0_s, t1_s]."""
    vals = [m for (t, m) in samples if t0_s <= t <= t1_s]
    return max(vals) if vals else None


# ── PHASE markers ────────────────────────────────────────────────────────────
# PHASE name=X t0_us=.. t1_us=.. time_s=.. host_rss_mib=.. host_peak_mib=..
phases = {}
for m in re.finditer(
    r"PHASE name=(\S+) t0_us=(\d+) t1_us=(\d+) time_s=([\d.eE+-]+) "
    r"host_rss_mib=([\d.eE+-]+) host_peak_mib=([\d.eE+-]+)",
    text,
):
    name = m.group(1)
    t0_us, t1_us = float(m.group(2)), float(m.group(3))
    phases[name] = {
        "time_s": m.group(4),
        "host_peak_mib": m.group(6),
        "gpu_peak_mib": gpu_peak(t0_us / 1e6, t1_us / 1e6),
    }


def ph(name, field):
    v = phases.get(name, {}).get(field)
    return "" if v is None else v


# ── scalar timings the binary already prints ─────────────────────────────────
def grab(pattern):
    m = re.search(pattern, text)
    return m.group(1) if m else ""


matmul_s   = grab(r"Matrix multiplication Prover time:\s*([\d.eE+-]+)s")
# NB total_s is range-EXCLUSIVE from main_gkr.cpp (verifier::range_prover_time is
# still 0 when prove() prints it); prefer prover_core_s / prover_total_s below.
total_s    = grab(r"Total Prover time:\s*([\d.eE+-]+)s")
verifier_s = grab(r"Verifier time:\s*([\d.eE+-]+)s")
proof_kb   = grab(r"Proof size:\s*([\d.eE+-]+)KB")

# ── PROVER_TOTALS / PROVER_BREAKDOWN ─────────────
def tot(field):
    return grab(r"PROVER_TOTALS[^\n]*\b" + field + r"=([\d.eE+-]+)")


def phase(field):
    return grab(r"PROVER_BREAKDOWN[^\n]*\b" + field + r"=([\d.eE+-]+)")


prover_core_s  = tot("prover_core_s")     # GKR + commit + open + lasso, no range
range_prove_s  = tot("range_prove_s")
prover_total_s = tot("prover_total_s")    # == zkGPT's "Total Prover time"
# residual must be ~0; non-zero means a prover_time site bypassed add_prover()
prover_residual_s = phase("residual")
ph_gkr_generic  = phase("gkr_generic")
ph_gkr_matmul   = phase("gkr_matmul")
ph_lasso        = phase("lasso")
ph_hyrax_commit = phase("hyrax_commit")
ph_hyrax_open   = phase("hyrax_open")

# ── whole-proof matmul kernel breakdown (MATMUL_BREAKDOWN) ────────────────────
def mm(field):
    return grab(r"MATMUL_BREAKDOWN[^\n]*\b" + field + r"=([\d.eE+-]+)")


mm_expand, mm_ptable, mm_round = mm("expand_s"), mm("ptable_s"), mm("round_s")
mm_fold, mm_deg3               = mm("fold_s"), mm("deg3_s")

# ── range prover: RANGE_RESULT + RANGE_BREAKDOWN ─────────────────────────────
range_result_s   = grab(r"RANGE_RESULT[^\n]*time_s=([\d.eE+-]+)")
range_partitions = grab(r"RANGE_RESULT[^\n]*partitions=(\d+)")


def bd(field):
    return grab(r"RANGE_BREAKDOWN[^\n]*\b" + field + r"=([\d.eE+-]+)")


rc_setup, rc_invert, rc_commit = bd("setup_s"), bd("invert_s"), bd("commit_s")
rc_deg3, rc_deg1, rc_open      = bd("deg3_s"), bd("deg1_s"), bd("open_s")

# ── overall peaks ────────────────────────────────────────────────────────────
host_peaks = [
    float(phases[n]["host_peak_mib"])
    for n in phases
    if phases[n]["host_peak_mib"] not in ("", None)
]
host_peak = max(host_peaks) if host_peaks else ""
# Overall GPU-memory peak. nvidia-smi path derives it from the polled time series;
# the rocprofv3 path (no time series) passes a precomputed whole-run peak via env.
gpu_peak_all = max((m for _, m in samples), default="")
_gp_env = os.environ.get("GPU_PEAK_MIB", "").strip()
if _gp_env:
    try:
        gpu_peak_all = float(_gp_env)
    except ValueError:
        pass

row = [
    os.environ.get("MODEL", ""),
    os.environ.get("BIN", ""),
    '"' + os.environ.get("DIMS", "") + '"',
    os.environ.get("L", ""),
    os.environ.get("C", ""),
    os.environ.get("K", ""),
    os.environ.get("MODE", ""),
    os.environ.get("STATUS", ""),
    os.environ.get("WALL", ""),
    ph("circuit_build", "time_s"),
    ph("circuit_build", "host_peak_mib"),
    ph("circuit_build", "gpu_peak_mib"),
    ph("verify_gkr", "time_s"),
    ph("verify_gkr", "host_peak_mib"),
    ph("verify_gkr", "gpu_peak_mib"),
    ph("range", "time_s"),
    ph("range", "host_peak_mib"),
    ph("range", "gpu_peak_mib"),
    mm_expand, mm_ptable, mm_round, mm_fold, mm_deg3,
    matmul_s,
    total_s,
    verifier_s,
    proof_kb,
    range_result_s,
    range_partitions,
    rc_setup, rc_invert, rc_commit, rc_deg3, rc_deg1, rc_open,
    host_peak,
    gpu_peak_all,
    os.environ.get("RC", ""),
    prover_core_s, range_prove_s, prover_total_s, prover_residual_s,
    ph_gkr_generic, ph_gkr_matmul, ph_lasso, ph_hyrax_commit, ph_hyrax_open,
]
print(",".join(str(x) for x in row))
