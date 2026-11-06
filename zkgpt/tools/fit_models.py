#!/usr/bin/env python3
"""fit_models.py — validate the plan.tex cost AND peak-memory models and derive
corrected ones, the way the offline profiler is meant to: CALIBRATE FROM A FEW
PROBE RUNS, never by fitting the whole sweep (that would defeat the purpose).

Both runtime and memory are validated identically (calibrate from probes ->
held-out error); the memory target is peak_gpu_bytes.

Backs plan.tex §"Matrix Multiplication" (Exp 1) and §"Lasso" (Exp 2). Per
experiment:

  1. PROPOSED — the plan.tex equation. Its constants are solved from the minimum
     number of probe runs (one per unknown); report the error on the configs NOT
     probed (held-out MAPE).
  2. UPDATED  — choose the model FORM by held-out generalization: greedily add
     library terms (and drop proposed terms) keeping only changes that improve
     held-out-after-calibration error. Then calibrate it from a few probes.
  3. CALIBRATION CURVE — held-out error vs how many probe runs you spend.

Probe runs are chosen UP-FRONT by D-optimality on the model's features (without
looking at any timing), so it is a valid "which runs should I profile" recipe.
Calibration over the probes is relative (weights 1/y), so a slow probe can't
dominate. No model is ever scored by an in-sample fit to the full sweep.

Terminology: "proposed" = the equation as written in plan.tex; "updated" = the
corrected equation this script derives. "held-out MAPE" = percent error on the
distinct configs that were not used as calibration probes.

Theory recap (plan.tex):
  matmul cost   : cost(k)      = alpha * k * s,            s = 2^n
  matmul memory   : (2^(n+1) + 2*3^(n/k)) * F
  lasso  cost   : cost(c,k)    = alpha * k * M(c) + beta * 2^c,
                  M(c) = |constraints| * 2 * ceil(49/c)
  lasso  memory : (3*M + 3*2^c) * F + 2^c * 4

Usage:
  tools/fit_models.py                         # latest exp1 + exp2 in tools/
  tools/fit_models.py --exp1 FILE.csv --exp2 FILE.csv
  tools/fit_models.py --only matmul --phases  # also fit each GPU phase
  tools/fit_models.py --fe-bytes 32           # BN254 Fr (default); 48 = BLS12-381
"""
import argparse, csv, glob, itertools, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------
# NNLS (Lawson-Hanson) — coefficients are physical counts/throughputs, so >= 0.
# ----------------------------------------------------------------------------
def nnls(A, b, maxiter=None):
    m, n = A.shape
    if maxiter is None:
        maxiter = 3 * n
    x = np.zeros(n)
    P = np.zeros(n, dtype=bool)         # passive (free) set
    w = A.T @ (b - A @ x)
    it = 0
    while (not P.all()) and (w[~P].max(initial=-np.inf) > 1e-12) and it < 30 * n:
        it += 1
        j = np.where(~P)[0][np.argmax(w[~P])]
        P[j] = True
        while True:
            Ap = A[:, P]
            s_p, *_ = np.linalg.lstsq(Ap, b, rcond=None)
            if (s_p > 0).all():
                x[P] = s_p
                break
            idx = np.where(P)[0]
            mask = s_p <= 0
            ratios = x[idx[mask]] / (x[idx[mask]] - s_p[mask])
            alpha = ratios.min()
            x[idx] = x[idx] + alpha * (s_p - x[idx])
            P[idx[x[idx] <= 1e-12]] = False
        w = A.T @ (b - A @ x)
    return x


# ----------------------------------------------------------------------------
# Fit / metrics in relative space (weight 1/y).
# ----------------------------------------------------------------------------
def design(rows, feats):
    return np.column_stack([[f(r) for r in rows] for _, f in feats])


def fit_rel(rows, feats, y):
    """Weighted NNLS minimizing sum((A x / y - 1)^2)."""
    A = design(rows, feats)
    W = 1.0 / np.maximum(y, 1e-12)
    coef = nnls(A * W[:, None], y * W)
    pred = A @ coef
    return coef, pred


def geomean(a):
    """Geometric mean of a non-negative array (robust to a few large outliers)."""
    a = np.asarray(a, float)
    return float(np.exp(np.mean(np.log(np.maximum(a, 1e-12)))))


def metrics(y, pred):
    ape = np.abs(pred - y) / np.maximum(np.abs(y), 1e-12)
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return dict(mape=100 * ape.mean(), geo=100 * geomean(ape),
                maxape=100 * ape.max(), r2=r2)


def distinct_idx(rows, feats, y):
    """Indices of configs that are DISTINCT in the model's feature space (e.g. the
    lasso cost is k-invariant, so its k-replicas collapse), keeping the cheapest
    real run of each -- the pool of configs you'd ever bother to profile."""
    seen = {}
    for i, r in enumerate(rows):
        key = tuple(round(f(r), 9) for _, f in feats)
        if key not in seen or y[i] < y[seen[key]]:
            seen[key] = i
    return sorted(seen.values())


def calibrate(rows, feats, y, B):
    """Derive the model's constants from B probe runs chosen up-front by
    D-optimality (no peeking at timings). Returns (coef, probe_idx, distinct_idx).
    The fit is over B points only -- this is the profiler, NOT a sweep."""
    di = distinct_idx(rows, feats, y)
    B = max(len(feats), min(B, len(di) - 1))
    sub = [rows[i] for i in di]
    chosen = dopt_subset(sub, feats, B)
    probe = [di[j] for j in chosen]
    coef, _ = fit_rel([rows[i] for i in probe], feats, y[probe])
    return coef, probe, di


def heldout_ape(rows, feats, y, coef, probe, di):
    """Per-config absolute % error on the DISTINCT configs NOT used as probes."""
    test = [i for i in di if i not in set(probe)]
    if not test:
        return np.array([])
    pred = design([rows[i] for i in test], feats) @ coef
    yv = y[test]
    return np.abs(pred - yv) / np.maximum(np.abs(yv), 1e-12)


def heldout_mape(rows, feats, y, coef, probe, di):
    """% error on the DISTINCT configs that were NOT used as probes."""
    ape = heldout_ape(rows, feats, y, coef, probe, di)
    return float("nan") if ape.size == 0 else 100 * float(ape.mean())


def calib_score(rows, feats, y):
    """Held-out error when this model form is calibrated from its minimum probes.
    This is the model-selection criterion -- generalization, not in-sample fit."""
    if not feats:
        return float("inf")
    coef, probe, di = calibrate(rows, feats, y, len(feats))
    return heldout_mape(rows, feats, y, coef, probe, di)


def coef_of(feats, coef, name):
    return next((c for (n, _), c in zip(feats, coef) if n == name), 0.0)


def eqn(feats, coef):
    parts = []
    for (name, _), c in zip(feats, coef):
        if c <= 1e-12:
            continue
        parts.append(f"{c:.4g}*[{name}]")
    return " + ".join(parts) if parts else "0"


# ----------------------------------------------------------------------------
# Model-form selection by GENERALIZATION (held-out after calibration), never by
# in-sample fit on the whole sweep.
# ----------------------------------------------------------------------------
def discover_heldout(rows, y, baseline, pool, min_gain=2.0, required=()):
    """Forward-add pool terms that improve held-out-after-calibration error, then
    backward-drop any baseline term that isn't pulling its weight. Terms named in
    `required` are kept on physical grounds even if this data can't constrain them
    (e.g. the lasso 2^c table term, negligible inside the swept c-range but the
    cost wall outside it). Returns the final feature list + a search history."""
    required = set(required)
    feats = list(baseline)
    used = {n for n, _ in feats}
    best = calib_score(rows, feats, y)
    history = [("proposed", best)]
    while True:
        cand = [(n, f) for n, f in pool if n not in used]
        scored = sorted(((calib_score(rows, feats + [(n, f)], y), n, f) for n, f in cand),
                        key=lambda t: t[0] if t[0] == t[0] else 1e9)
        if not scored or best - scored[0][0] < min_gain:
            break
        _, n, f = scored[0]
        feats.append((n, f)); used.add(n); best = scored[0][0]
        history.append((f"+{n}", best))
    # backward: drop any *proposed*, non-required term that doesn't cost >= min_gain
    for n, f in list(baseline):
        if n in required or len(feats) <= 1:
            continue
        trial = [t for t in feats if t[0] != n]
        s = calib_score(rows, trial, y)
        if s - best < min_gain:
            feats = trial; best = s
            history.append((f"-{n}", best))
    return feats, history


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def latest(pattern, exclude=()):
    # search results/ subfolders (and tools/ root, for back-compat)
    fs = [f for f in glob.glob(os.path.join(HERE, "**", pattern), recursive=True)
          if not any(x in os.path.basename(f) for x in exclude)]
    return max(fs, key=os.path.getmtime) if fs else None


# ----------------------------------------------------------------------------
# Experiment 1 — matmul
# ----------------------------------------------------------------------------
def run_matmul(path, fe, phases=False, plot=True, hw=None):
    rows = [r for r in load_csv(path) if r.get("mode") == "gpu" and float(r["time_s"]) > 0]
    for r in rows:
        r["_n"] = int(r["n"]); r["_k"] = int(r["k"])
        r["_l"] = int(r["l_stage"]); r["_s"] = float(r["s"])
        r["_keff"] = math.ceil(r["_n"] / r["_l"])   # stages actually executed
        r["_ws"] = (3.0 ** r["_l"]) * (2.0 ** (r["_n"] - r["_l"]))   # stage-0 ptable working set
    y = np.array([float(r["time_s"]) for r in rows])

    print(f"\n{'='*72}\nEXPERIMENT 1  matmul   ({len(rows)} GPU points, {path})\n{'='*72}")

    F = {
        "k*s":      lambda r: r["_k"] * r["_s"],
        "keff*s":   lambda r: r["_keff"] * r["_s"],
        "ptable_ws":lambda r: r["_ws"],
        "keff*3^l": lambda r: r["_keff"] * (3.0 ** r["_l"]),
        "3^l":      lambda r: 3.0 ** r["_l"],
        "s":        lambda r: r["_s"],
        "const":    lambda r: 1.0,   # fixed kernel-launch / setup overhead floor
    }
    theory = [("k*s", F["k*s"])]
    pool = [(n, f) for n, f in F.items() if n != "k*s"]

    stated, disc = report(rows, y, theory, pool,
           theory_str="cost = alpha * k * s   (plan.tex eq:matmul_cost)")
    if plot:
        fitdir = os.path.join(HERE, "results", "fits", "images")
        plot_proposed("Exp 1: matmul", rows, y, stated, "_k", ["_n"],
                      os.path.join(fitdir, "fit_matmul_proposed.png"))
        plot_updated("Exp 1: matmul", rows, y, disc, "_k", ["_n"],
                     os.path.join(fitdir, "fit_matmul_updated.png"))
    if hw:
        hw_matmul(rows, stated, disc, hw)

    # ---- MEMORY model (peak_gpu_bytes), same calibrate-from-probes treatment ----
    pv = np.array([float(r["peak_gpu_bytes"]) for r in rows])
    keep = [i for i, v in enumerate(pv) if v > 0]
    if len(keep) > 4:
        mrows = [rows[i] for i in keep]; mvy = pv[keep]
        print(f"\n{'- '*20}\nMEMORY MODEL (peak memory)")
        MF = {
            "bk":        lambda r: 2.0 ** (r["_n"] + 1),       # BkA/BkB
            "3^l":       lambda r: 3.0 ** r["_l"],             # proposed ptable term
            "ptable_ws": lambda r: r["_ws"],                   # corrected ptable: 3^l*2^(n-l)
            "const":     lambda r: 1.0,                        # fixed context/allocator
        }
        mtheory = [("bk", MF["bk"]), ("3^l", MF["3^l"])]
        mpool = [("ptable_ws", MF["ptable_ws"]), ("const", MF["const"])]
        ms, md = report(mrows, mvy, mtheory, mpool, quantity="memory",
                        theory_str="memory = (2^(n+1) + 2*3^(n/k))*F   (plan.tex eq:vram_peak)")
        if plot:
            fitdir = os.path.join(HERE, "results", "fits", "images")
            plot_proposed("Exp 1: matmul memory", mrows, mvy, ms, "_k", ["_n"],
                          os.path.join(fitdir, "mem_matmul_proposed.png"), ylabel="peak memory (bytes)")
            plot_updated("Exp 1: matmul memory", mrows, mvy, md, "_k", ["_n"],
                         os.path.join(fitdir, "mem_matmul_updated.png"), ylabel="peak memory (bytes)")

    if phases:
        phase_fit(rows, F,
                  [("expand_s", "expand"), ("ptable_s", "ptable"),
                   ("round_s", "round"), ("fold_s", "fold")])


# ----------------------------------------------------------------------------
# Experiment 2 — lasso
# ----------------------------------------------------------------------------
def run_lasso(path, fe, phases=False, plot=True, hw=None):
    rows = [r for r in load_csv(path) if r.get("mode") == "gpu" and float(r["time_s"]) > 0]
    for r in rows:
        r["_c"] = int(r["c"]); r["_k"] = int(r["k"])
        r["_M"] = float(r["M"]); r["_N"] = float(r["N"])
        # constraints = query_size = next_pow2(per_layer*L); c-invariant unlike M=constraints*w
        r["_constraints"] = float(r["constraints"]) if r.get("constraints") else (
            r["_M"] / (2 * math.ceil(48 / r["_c"])))
        # l = chunks per value = ceil(range/c) = M/(2*constraints); each chunk is a
        # sequential logup, so cost ~ l*(open over query + table-side commit/open).
        r["_lchunks"] = r["_M"] / (2 * r["_constraints"])
    y = np.array([float(r["time_s"]) for r in rows])

    print(f"\n{'='*72}\nEXPERIMENT 2  lasso    ({len(rows)} GPU points, {path})\n{'='*72}")

    F = {
        "k*M":        lambda r: r["_k"] * r["_M"],
        "2^c":        lambda r: r["_N"],
        "M":          lambda r: r["_M"],
        "sqrt(M)":    lambda r: math.sqrt(r["_M"]),
        # ── structural terms for the corrected PCS-dominated form (eq:lasso_cost) ──
        "l*sqrt(mq)": lambda r: r["_lchunks"] * math.sqrt(r["_constraints"]),  # Hyrax open, query side
        "l*2^c":      lambda r: r["_lchunks"] * r["_N"],                       # table-side commit/open
        "sqrt(mq)":   lambda r: math.sqrt(r["_constraints"]),
        "const":      lambda r: 1.0,   # fixed overhead floor
    }
    theory = [("k*M", F["k*M"]), ("2^c", F["2^c"])]
    pool = [(n, f) for n, f in F.items() if n not in ("k*M", "2^c")]

    stated, disc = report(rows, y, theory, pool,
           theory_str="cost = alpha*k*M(c) + beta*2^c   (plan.tex eq:lasso_cost)",
           required=["2^c"])   # keep the table term on physical grounds (option a)

    # ── CORRECTED structural model: cost = l*(gamma*sqrt(mq) + beta*2^c) ──────────
    # Validated the same calibrate-from-probes -> held-out way as everything else.
    corr = [("l*sqrt(mq)", F["l*sqrt(mq)"]), ("l*2^c", F["l*2^c"])]
    ccoef, cprobe, cdi = calibrate(rows, corr, y, len(corr))
    cape = heldout_ape(rows, corr, y, ccoef, cprobe, cdi)
    print("\n[1b] CORRECTED MODEL (plan.tex eq:lasso_cost, PCS-dominated):")
    print("     cost = ceil(49/c) * (gamma*sqrt(mq) + beta*2^c)")
    print(f"     calibrated from {len(cprobe)} probe(s): {eqn(corr, ccoef)}")
    print(f"     held-out  MAPE {100*cape.mean():.1f}%   geo-mean {100*geomean(cape):.1f}%"
          f"   max {100*cape.max():.1f}%   ({len(cape)} configs)")

    # ── does adding the BlendySC k knob (deg-1 sumcheck ~ l*k) earn its term? ──────
    corrk = corr + [("const", F["const"]), ("l*k", lambda r: r["_lchunks"] * r["_k"])]
    kcoef, kprobe, kdi = calibrate(rows, corrk, y, len(corrk))
    kape = heldout_ape(rows, corrk, y, kcoef, kprobe, kdi)
    base = corr + [("const", F["const"])]
    bcoef, bprobe, bdi = calibrate(rows, base, y, len(base))
    bape = heldout_ape(rows, base, y, bcoef, bprobe, bdi)
    print(f"     +const floor:        held-out MAPE {100*bape.mean():.1f}%   geo {100*geomean(bape):.1f}%")
    print(f"     +const +l*k (BlendySC): held-out MAPE {100*kape.mean():.1f}%   geo {100*geomean(kape):.1f}%"
          f"   -> k term changes MAPE by {100*(kape.mean()-bape.mean()):+.2f} pts")
    if plot:
        fitdir = os.path.join(HERE, "results", "fits", "images")
        plot_proposed("Exp 2: lasso", rows, y, stated, "_c", ["layers", "_k"],
                      os.path.join(fitdir, "fit_lasso_proposed.png"))
        plot_updated("Exp 2: lasso", rows, y, disc, "_c", ["layers", "_k"],
                     os.path.join(fitdir, "fit_lasso_updated.png"))
        # The corrected structural form (eq:lasso_cost) -- the model whose geo-mean
        # is quoted above. Calibrated from probes, shown against every config.
        plot_updated("Exp 2: lasso (corrected eq:lasso_cost)", rows, y,
                     (base, bcoef, 100 * bape.mean(), bprobe), "_c", ["layers", "_k"],
                     os.path.join(fitdir, "fit_lasso_corrected.png"))
    if hw:
        hw_lasso(rows, stated, disc, hw)

    # ---- MEMORY model: plan.tex eq:lasso_vram ----
    # GPU grand product Lasso buffers:
    #   d_f (M), d_lhs_tree (2M-1)  =>  3M Fr from query side
    #   d_T (N), d_rhs_tree (2N-1)  =>  3N Fr from table side
    #   d_counts (N u32)
    # => peak = (3M + 3N)*F + N*4
    print(f"\n{'- '*20}\nMEMORY MODEL (peak memory)")
    pv = np.array([float(r["peak_gpu_bytes"]) for r in rows])
    keep = [i for i, v in enumerate(pv) if v > 0]
    if len(keep) > 3:
        mrows = [rows[i] for i in keep]; mvy = pv[keep]

        c_invariant = all(
            len({round(pv[i]) for i, r in enumerate(rows) if r["layers"] == key}) == 1
            for key in {r["layers"] for r in rows})

        # Calibrate proposed model from its minimum probes and report held-out MAPE.
        # Skip the model-search step: when measurement doesn't track the model's buffers,
        # fitting alternative terms would just describe the Hyrax allocations, not eq:lasso_vram.
        MF = [("M", lambda r: r["_M"]), ("2^c", lambda r: r["_N"])]
        pcoef, pprobe, pdi = calibrate(mrows, MF, mvy, len(MF))
        pm = heldout_mape(mrows, MF, mvy, pcoef, pprobe, pdi)
        print(f"   plan.tex model:  memory = (3M+3*2^c)*F + 4*2^c   (eq:lasso_vram)")
        print(f"   calibrated from {len(pprobe)} probe(s): {eqn(MF, pcoef)}")
        print(f"   held-out MAPE {pm:.0f}%  on the other {len(pdi)-len(pprobe)} distinct configs")

        if c_invariant:
            print(f"\n   Diagnosis: peak_gpu_bytes is INVARIANT in c (flat per layer count).")
            print(f"   The {pm:.0f}% MAPE is expected: the binary's Hyrax logup allocates generator")
            print(f"   arrays sized by query_size (c-independent), not the GPU grand product")
            print(f"   Lasso buffers (d_f, d_T, d_counts, d_lhs_tree, d_rhs_tree) described")
            print(f"   in plan.tex. To validate eq:lasso_vram, instrument the GPU grand product")
            print(f"   path and capture its per-c peak allocation separately.")

        if plot:
            fitdir = os.path.join(HERE, "results", "fits", "images")
            plot_proposed("Exp 2: lasso memory", mrows, mvy, (MF, pcoef, pm, pprobe),
                          "_c", ["layers", "_k"],
                          os.path.join(fitdir, "mem_lasso_proposed.png"), ylabel="peak memory (bytes)")

    if phases:
        phase_fit(rows, F,
                  [("commit_s", "commit"), ("open_s", "open"),
                   ("deg3_s", "deg3"), ("deg1_s", "deg1"),
                   ("invert_s", "invert"), ("setup_s", "setup")])


# ----------------------------------------------------------------------------
# Shared report
# ----------------------------------------------------------------------------
def _probe_label(rows, idx):
    out = []
    for i in idx:
        r = rows[i]
        keys = ("_n", "_k") if "_n" in r else ("layers", "_c")
        out.append("(" + ",".join(f"{k.strip('_')}={r[k]}" for k in keys) + ")")
    return "; ".join(out)


def eqn_all(feats, coef):
    """Like eqn() but shows EVERY term (tiny ones as ~0), to display a model form
    that includes a required-but-unconstrained term (e.g. pinned 2^c)."""
    out = []
    for (name, _), c in zip(feats, coef):
        out.append(f"{c:.4g}*[{name}]" if abs(c) > 1e-12 else f"~0*[{name}]")
    return " + ".join(out)


def report(rows, y, theory, pool, theory_str, required=(), quantity="cost"):
    """Everything here is calibrate-from-probes -> predict held-out. No model is
    ever scored by fitting the full sweep."""
    # [1] PROPOSED model, calibrated from its minimum probes
    pcoef, pprobe, pdi = calibrate(rows, theory, y, len(theory))
    pm = heldout_mape(rows, theory, y, pcoef, pprobe, pdi)
    print(f"\n[1] PROPOSED MODEL (plan.tex):  {theory_str}")
    print(f"    calibrated from {len(pprobe)} probe(s): {eqn(theory, pcoef)}")
    print(f"    held-out MAPE {pm:.1f}%  on the other {len(pdi)-len(pprobe)} distinct configs")

    # [2] UPDATED model form, selected by held-out generalization
    feats, hist = discover_heldout(rows, y, theory, pool, required=required)
    ucoef, uprobe, udi = calibrate(rows, feats, y, len(feats))
    um = heldout_mape(rows, feats, y, ucoef, uprobe, udi)
    print("\n[2] UPDATED MODEL  (form chosen by held-out error, calibrated from probes):")
    print("    search:    " + "  ->  ".join(f"{h[0]}({h[1]:.1f}%)" for h in hist))
    print(f"    calibrated from {len(uprobe)} probe(s): {quantity} = {eqn_all(feats, ucoef)}")
    if any(abs(c) <= 1e-12 for c in ucoef):
        print("    (terms shown ~0 are pinned on physical grounds but unconstrained by "
              "this data -- add probes that exercise them to fit their constant)")
    print(f"    probes: {_probe_label(rows, uprobe)}")
    print(f"    held-out MAPE {um:.1f}%  on the other {len(udi)-len(uprobe)} distinct configs")

    # [3] calibration curve: held-out error vs how many probes you spend
    print("\n[3] CALIBRATION CURVE  (B D-optimal probes -> held-out MAPE on the rest):")
    for B in range(len(feats), min(len(feats) + 5, len(udi))):
        c, pr, di = calibrate(rows, feats, y, B)
        print(f"      {B:2d} probes -> {heldout_mape(rows, feats, y, c, pr, di):5.1f}%   "
              f"({100*B/len(di):.0f}% of {len(di)} configs)")

    return (theory, pcoef, pm, pprobe), (feats, ucoef, um, uprobe)


def _knob_panel(ax, plt, rows, y, pred, knob, groups, probe=(), line_style="-",
                ylabel="time (s)"):
    """measured points + model curve, one colored series per group; probes circled."""
    probe = set(probe)
    keys = sorted({tuple(r[g] for g in groups) for r in rows})
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(keys)))
    for col, key in zip(cmap, keys):
        idx = [i for i, r in enumerate(rows) if tuple(r[g] for g in groups) == key]
        idx.sort(key=lambda i: rows[i][knob])
        xs = [rows[i][knob] for i in idx]
        lbl = ",".join(f"{g.strip('_')}={v}" for g, v in zip(groups, key))
        ax.plot(xs, [y[i] for i in idx], "o", color=col, ms=5)
        ax.plot(xs, [pred[i] for i in idx], line_style, color=col, lw=1.6, label=lbl)
        prx = [rows[i][knob] for i in idx if i in probe]
        if prx:
            ax.scatter(prx, [pred[i] for i in idx if i in probe], s=130,
                       facecolors="none", edgecolors="k", linewidths=1.6, zorder=5)
    ax.set_yscale("log")
    ax.set_xlabel(knob.strip("_")); ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, which="both")


def plot_proposed(title, rows, y, proposed, knob, groups, outpath, ylabel="time (s)"):
    """Single panel: the PROPOSED (plan.tex) model vs measured."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    feats, coef, mape, pr_idx = proposed
    pred = design(rows, feats) @ coef
    # Geo over the points NOT used for calibration: probe rows fit ~exactly
    # (zero residual), and a geometric mean is destroyed by near-zero entries.
    held = [i for i in range(len(rows)) if i not in set(pr_idx or [])]
    ape = np.abs(pred[held] - y[held]) / np.maximum(np.abs(y[held]), 1e-12)
    geo = 100 * geomean(ape)
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 5.6))
    fig.suptitle(f"{title} — PROPOSED (plan.tex)", fontsize=12, fontweight="bold")
    _knob_panel(ax, plt, rows, y, pred, knob, groups, line_style=":", ylabel=ylabel)
    ax.plot([], [], "ko", ms=5, label="measured")
    ax.plot([], [], "k:", lw=1.6,
            label=f"proposed (held-out MAPE {mape:.0f}%; held-out geo {geo:.1f}%)")
    ax.set_title("proposed model (dotted) vs measured (points)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=130); plt.close(fig)
    print(f"\n   [plot] {outpath}")


def plot_updated(title, rows, y, updated, knob, groups, outpath, ylabel="time (s)"):
    """Two panels for the UPDATED model: (left) parity, (right) value-vs-knob.
    Calibrated from probes (circled), shown against every config."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    feats, coef, mape, probe = updated
    pred = design(rows, feats) @ coef
    pr = sorted(set(probe))
    meas_lbl = ylabel.split(" (")[0]
    # Geo over held-out points only (exclude probes: their ~0 residual would
    # corrupt the geometric mean). Matches the held-out MAPE shown.
    held = [i for i in range(len(rows)) if i not in set(pr)]
    ape = np.abs(pred[held] - y[held]) / np.maximum(np.abs(y[held]), 1e-12)
    geo = 100 * geomean(ape)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6))
    fig.suptitle(f"{title} — UPDATED (calibrated from {len(pr)} probes, not fit to the sweep)",
                 fontsize=12, fontweight="bold")

    lo = min(y.min(), pred.min()) * 0.7
    hi = max(y.max(), pred.max()) * 1.4
    ax1.plot([lo, hi], [lo, hi], "k--", lw=1, label="ideal (pred=meas)")
    ax1.scatter(y, pred, s=30, c="tab:blue", marker="^", alpha=0.7,
                label=f"updated  held-out MAPE {mape:.0f}%; held-out geo {geo:.1f}%")
    ax1.scatter(y[pr], pred[pr], s=140, facecolors="none", edgecolors="k",
                linewidths=1.6, label="probe runs (calibration)")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlim(lo, hi); ax1.set_ylim(lo, hi)
    ax1.set_xlabel(f"measured {ylabel}"); ax1.set_ylabel(f"predicted {meas_lbl}")
    ax1.set_title("calibrated prediction vs measured (parity)")
    ax1.legend(fontsize=8, loc="upper left"); ax1.grid(alpha=0.3, which="both")

    _knob_panel(ax2, plt, rows, y, pred, knob, groups, probe=pr, line_style="-", ylabel=ylabel)
    ax2.plot([], [], "ko", ms=5, label="measured")
    ax2.plot([], [], "k-", lw=1.6, label=f"updated (held-out {mape:.0f}%; geo {geo:.1f}%)")
    ax2.plot([], [], "o", mfc="none", mec="k", ms=10, label="probe runs")
    ax2.set_title("calibrated model (solid) vs measured, probes circled")
    ax2.legend(fontsize=7, ncol=2)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=130); plt.close(fig)
    print(f"   [plot] {outpath}")


def phase_fit(rows, F, phase_cols):
    """For each timed phase, report the single library term that best explains it."""
    print("\n-- per-phase scaling (best single term, relative fit) --")
    pool = list(F.items())
    for col, label in phase_cols:
        if col not in rows[0]:
            continue
        yv = np.array([float(r.get(col) or 0) for r in rows])
        if yv.max() <= 0:
            continue
        best = None
        for name, f in pool:
            coef, pred = fit_rel(rows, [(name, f)], yv)
            mm = metrics(yv, pred)
            if best is None or mm["mape"] < best[0]:
                best = (mm["mape"], name, coef[0], mm["r2"])
        print(f"   {label:8s} ~ {best[2]:.3g}*[{best[1]}]   "
              f"MAPE {best[0]:5.1f}%  R^2 {best[3]:.3f}")


# ----------------------------------------------------------------------------
# D-optimal probe selection: pick which few runs to profile, up-front.
# ----------------------------------------------------------------------------
def dopt_subset(rows, feats, k):
    """Pick k well-conditioned, well-spread calibration points up-front (greedy
    D-optimal on the model's feature columns) -- chosen WITHOUT looking at the
    timings, so it's a valid 'which runs should I profile' recommendation."""
    A = design(rows, feats)
    A = A / (np.linalg.norm(A, axis=0) + 1e-30)
    chosen, M = [], np.zeros((A.shape[1], A.shape[1]))
    
    # For lasso, force a middle _c value as first probe
    if rows and "_c" in rows[0]:
        c_values = sorted(set(r["_c"] for r in rows))
        if len(c_values) > 1:
            mid_c = c_values[len(c_values) // 2]
            # Find the middle row with middle _c value
            candidates = [i for i, r in enumerate(rows) if r["_c"] == mid_c]
            if candidates:
                mid_idx = candidates[len(candidates) // 2]
                chosen.append(mid_idx)
                M = M + np.outer(A[mid_idx], A[mid_idx])
    
    for _ in range(k):
        best_i, best_v = None, -np.inf
        for i in range(len(rows)):
            if i in chosen:
                continue
            ld = np.linalg.slogdet(M + np.outer(A[i], A[i]) + 1e-9 * np.eye(A.shape[1]))[1]
            if ld > best_v:
                best_v, best_i = ld, i
        if best_i is not None:
            chosen.append(best_i)
            M = M + np.outer(A[best_i], A[best_i])
    return chosen[:k]


# ----------------------------------------------------------------------------
# Hardware-spec -> coefficient calculator (plan.tex §Hardware Characterization)
#
#   omega_compute = f * (CU * lanes_per_cu) / I_fmul     field-muls / s
#   omega_memory  = B / (mu * F)                         field-muls / s
#   omega_eff     = gamma * min(omega_compute, omega_memory)
#   alpha = phi_blendy / omega_eff ,   beta = phi_lasso / omega_eff
#
# alpha/beta have units s per (element * stage) / s per (table entry). gamma is
# the fraction of peak the kernel sustains; here we DON'T assume it — we read it
# back from the fitted coefficient (gamma_implied = alpha_hw(gamma=1) / alpha_fit)
# and cross-check against rocprof's measured VALU/mem busy in the CSV.
# ----------------------------------------------------------------------------
def roofline(hw):
    N = hw["cus"] * hw["lanes_per_cu"]
    wc = hw["freq"] * N / hw["ifmul"]
    wm = hw["bw"] / (hw["mu"] * hw["F"])
    bound = "memory" if wm <= wc else "compute"
    return N, wc, wm, min(wc, wm), bound


def hw_header(hw):
    N, wc, wm, weff, bound = roofline(hw)
    print(f"\n{'-'*72}\nHARDWARE ROOFLINE  (MI210 defaults; override via --freq/--cus/...)\n{'-'*72}")
    print(f"   lanes      = CU*lanes_per_cu = {hw['cus']}*{hw['lanes_per_cu']} = {N}")
    print(f"   omega_compute = f*lanes/I_fmul     = "
          f"{hw['freq']:.2g}*{N}/{hw['ifmul']:.0f}  = {wc:.3e} field-mul/s")
    print(f"   omega_memory  = B/(mu*F)           = "
          f"{hw['bw']:.2g}/({hw['mu']:.0f}*{hw['F']:.0f}) = {wm:.3e} field-mul/s")
    print(f"   => {bound}-bound, omega_eff(gamma=1) = {weff:.3e} field-mul/s "
          f"  (peak {1/weff:.2e} s per field-op)")
    return weff, bound


def hw_report(name, weff, phi, phi_name, coef_fit, knob_label, util):
    """Compare hardware-predicted coefficient (at gamma=1) to the fitted one and
    back out the sustained efficiency gamma; cross-check against measured busy%."""
    a_hw = phi / weff                       # predicted at gamma=1
    gamma = a_hw / coef_fit if coef_fit > 0 else float("nan")
    print(f"\n   {name}:  predicted {knob_label} = {phi_name}/omega_eff "
          f"= {phi:.0f}/{weff:.2e} = {a_hw:.2e} s   (at gamma=1, peak)")
    print(f"            fitted    {knob_label} = {coef_fit:.2e} s")
    print(f"            => implied sustained efficiency gamma = {100*gamma:.2f}% of peak")
    if util:
        print(f"            rocprof busy% (this sweep): VALU {util[0]:.1f}%  mem {util[1]:.1f}%  "
              f"(max VALU {util[2]:.1f}%) -- consistent with low gamma")
    return a_hw, gamma


def hw_matmul(rows, stated, disc, hw):
    weff, _ = hw_header(hw)
    a_d = coef_of(disc[0], disc[1], "k*s")
    a_t = coef_of(stated[0], stated[1], "k*s")
    vb = [float(r["valu_busy"]) for r in rows if r.get("valu_busy")]
    mb = [float(r["mem_busy"]) for r in rows if r.get("mem_busy")]
    util = (float(np.mean(vb)), float(np.mean(mb)), float(np.max(vb))) if vb else None
    print("\n== matmul:  alpha = phi_blendy / omega_eff ==")
    hw_report("alpha (vs derived k*s coef)", weff, hw["phi_blendy"], "phi_blendy",
              a_d, "alpha", util)
    print(f"   note: the swept points run at <1% of roofline (the throughput regime is the\n"
          f"         large-n / small-k corner that EXT_CAP filters out), so the measured alpha\n"
          f"         is dominated by per-launch overhead (the +const term), not HBM bandwidth.")


def hw_lasso(rows, stated, disc, hw):
    weff, _ = hw_header(hw)
    b_t = coef_of(stated[0], stated[1], "2^c")
    b_d = coef_of(disc[0], disc[1], "2^c")
    print("\n== lasso:  beta = phi_lasso / omega_eff  (RHS table build, 2^c entries) ==")
    hw_report("beta (vs stated 2^c coef)", weff, hw["phi_lasso"], "phi_lasso",
              b_t, "beta", None)
    print(f"   note: the dominant lasso term is the Hyrax commit/open ~ sqrt(M) (MSM / group\n"
          f"         ops), NOT a field-mul count, so it needs an MSM-throughput constant\n"
          f"         (group-adds/s), not omega_eff. Only the 2^c table-build term maps to beta.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp1", default=None)
    ap.add_argument("--exp2", default=None)
    ap.add_argument("--only", choices=["matmul", "lasso"], default=None)
    ap.add_argument("--fe-bytes", type=int, default=32, help="field element bytes (BN254=32, BLS12-381=48)")
    ap.add_argument("--phases", action="store_true", help="also fit each GPU phase column")
    ap.add_argument("--no-plot", action="store_true", help="skip writing fit_*.png")
    ap.add_argument("--hw", action="store_true",
                    help="compute alpha/beta from hardware specs and compare to calibrated constants")
    # hardware spec (MI210 / BN254 defaults)
    ap.add_argument("--freq", type=float, default=1.7e9, help="GPU clock Hz")
    ap.add_argument("--cus", type=int, default=104, help="compute units")
    ap.add_argument("--lanes-per-cu", type=int, default=64, help="stream procs per CU")
    ap.add_argument("--bw", type=float, default=1.6e12, help="HBM bandwidth B/s")
    ap.add_argument("--ifmul", type=float, default=75, help="int ops per field multiply (~50-100)")
    ap.add_argument("--mu", type=float, default=3, help="field elements moved per op")
    ap.add_argument("--phi-blendy", type=float, default=4, help="field-muls per element per stage")
    ap.add_argument("--phi-lasso", type=float, default=2, help="field-muls per table entry")
    a = ap.parse_args()
    plot = not a.no_plot
    hw = None
    if a.hw:
        hw = dict(freq=a.freq, cus=a.cus, lanes_per_cu=a.lanes_per_cu, bw=a.bw,
                  ifmul=a.ifmul, mu=a.mu, F=a.fe_bytes,
                  phi_blendy=a.phi_blendy, phi_lasso=a.phi_lasso)

    e1 = a.exp1 or latest("exp1_matmul_*.csv", exclude=("compute",))
    e2 = a.exp2 or latest("exp2_lasso_*.csv", exclude=("Lsweep", "partial"))

    if a.only != "lasso":
        if e1:
            run_matmul(e1, a.fe_bytes, a.phases, plot, hw)
        else:
            print("no exp1 matmul CSV found", file=sys.stderr)
    if a.only != "matmul":
        if e2:
            run_lasso(e2, a.fe_bytes, a.phases, plot, hw)
        else:
            print("no exp2 lasso CSV found", file=sys.stderr)


if __name__ == "__main__":
    main()
