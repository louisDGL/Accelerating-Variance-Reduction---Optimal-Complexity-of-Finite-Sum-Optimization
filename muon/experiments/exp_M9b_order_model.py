"""M9b: the sampling-order part of M9 on a sum with identical Hessians.

Part D of M9 measures, on two memory-limited instances, the fastest cooldown that a fixed-length polar step fed by a
SAGA table can follow, and finds it next to the memory factor (1 - (b + m)/n)^(n/b) of the table (fact F3), with
values on both sides of it. F3 is a statement on the MEAN memory error; it does not bound a single run. This script
asks what a single run can follow when nothing but the table limits it, on the solvable model of hypothesis B4 of
the main study (results/findings.md, Section 9; theory/B4_sampling_order.tex):

    f_i(W) = ||W - y_i||^2 / 2,        W in R^{K x 1},        i = 1..n        (identical Hessians: the identity).

With E = W - W_* and Psi_i = phi_i - W_* (phi_i: the point at which entry i was last refreshed) the SAGA estimator is
omega = E + mean_i Psi_i - mean_{j in B} Psi_j, and the polar factor of the one-column matrix omega is
omega / ||omega||: the iteration is "normalized SAGA on a geometric cooldown" and only knows the table. On this
model linear SAGA is solved in the main study: per pass, the squared error contracts by exp(-1) in expectation under
independent sampling at large steps, by exp(-2) under reshuffling, and a single run under independent sampling is
faster than the expectation (exponent 1.55 to 1.67 nats per pass for n = 100 to 10^4, experiment B4b).

  D  The design of part D of M9 (same schemes, grids of rho, seeds, horizon, criteria; tasks built by
     exp_M9_schedule.order_task and run by the lab) on the model instance n = 1000, K = 10: a MatrixLSProblem with
     x_i = 1 and lam = 1e-6. The lab multiplies the length of a fixed-length step by max(1, K/d)^(1/2) = K^(1/2) on a
     K x 1 matrix; lr_0 is divided by that factor so that the total step length is 4 ||W_*||_op as in M9. Output:
     the thresholds rho_star next to those measured on soft-mem and ls-mem (read from M9_summary.json).
  L  Long horizons (80 passes, where part D has 23 / log(1/rho) passes), batch one, n in {100, 1000, 10000},
     independent indices and a fresh permutation per pass, on a grid of x = log(1/rho) (nats of lr^2 lost per pass).
     Own simulator in error coordinates (`model_run`, 20 runs at once; it agrees step by step with the lab on the
     model instance, muon/tests/test_order_model.py, archived locally in _archive/), so that neither a distance to W_* of the order of the rounding
     errors nor the cost of the lab limits the horizon. Output: the fraction of the runs on schedule and the level
     of dist^2 / lr^2 by x, the largest x followed, next to the exponents of linear SAGA on the same model (read
     from results/B4b_summary.json of the main study when that file exists).

Criteria (those of part D of M9): C = dist^2 / (length of the last step)^2 at the end of the run; a run is on
schedule if C <= 100 x the median C of the same cell at the reference rho = 0.61; rho_star = smallest rho of the grid
with at least 80% of the runs on schedule. Every number printed here is in muon/results/M9b_summary.json.

Outputs: muon/results/M9b_order.pkl (cache of part D), muon/results/M9b_summary.json.
Usage: python muon/experiments/exp_M9b_order_model.py [--quick] [--parts D,L] [--jobs N]
Run time: about 4 minutes with 6 processes (D: 1.5 min, L: 2.5 min).
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

import common as C
import exp_M9_schedule as M9

MODEL = dict(n=1000, K=10, lam=1e-6, seed=0)        # part D: the size of soft-mem and ls-mem (n = 1000, K = 10)
D_BATCHES = (1, 5, 20, 50)
LONG_N = (100, 1000, 10000)                          # part L
LONG_K = 10
LONG_PASSES = 80
LONG_RUNS = 20
LONG_X = (0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.8)
X_REF = math.log(1.0 / M9.RHOS[0])                   # the reference rho of part D, 0.61
CLEAN_FACTOR = 10.0                                  # second reading: level within this factor of the reference
ORDERS = (("iid", "independent indices"), ("reshuffle", "fresh permutation per pass"))


# ----------------------------------------------------------------------------------------------
# the model instance, for the lab
# ----------------------------------------------------------------------------------------------
def model_spec(n, K, lam=1e-6, seed=0):
    return {"loss": "ls", "model": "identical-hessians", "n": int(n), "K": int(K), "d": 1, "lam": float(lam),
            "seed": int(seed)}


def model_problem(s):
    """f_i(W) = ||W - y_i||^2 / 2 + (lam/2) ||W||^2 on K x 1 matrices: x_i = 1, targets y_i = w + a_i with
    sum_i a_i = 0 and ||w|| = 1, so that W_* = w / (1 + lam). The optimum is set (closed form)."""
    from lab import optimum, problems
    rng = np.random.default_rng(s["seed"])
    A = rng.standard_normal((s["n"], s["K"]))
    A -= A.mean(0)
    w = rng.standard_normal(s["K"])
    w /= np.linalg.norm(w)
    prob = problems.MatrixLSProblem(np.ones((s["n"], 1)), A + w[None, :], s["lam"],
                                    name=f"identical-hessians(n={s['n']},K={s['K']},seed={s['seed']})")
    optimum.solve_optimum(prob)
    return prob


def register(s):
    """Make common.problem(s) return the model instance (common.py only knows the two generators of the lab). The
    worker processes of common.run_tasks inherit the registration when they are forked."""
    key = json.dumps(s, sort_keys=True)
    if key not in C._PROBLEMS:
        C._PROBLEMS[key] = model_problem(s)
    return C._PROBLEMS[key]


# ----------------------------------------------------------------------------------------------
# the criteria of part D of M9
# ----------------------------------------------------------------------------------------------
def thresholds(by_rho):
    """The reading of part D of M9 for one cell and one scheme. `by_rho`: list of dicts with keys 'rho' and 'C'
    (final dist^2 / lr^2 of every run), in the order of the grid (decreasing rho, the reference first). Adds
    'on_schedule' to every entry and returns (reference level, rho_star, rho_star_monotone)."""
    ref = float(np.median(by_rho[0]["C"]))
    for x in by_rho:
        x["on_schedule"] = int(sum(c <= M9.TRACK_FACTOR * ref for c in x["C"]))
    ok = [x["on_schedule"] >= M9.TRACK_QUORUM * len(x["C"]) for x in by_rho]
    star = min((x["rho"] for x, o in zip(by_rho, ok) if o), default=None)
    mono = None
    for x, o in zip(by_rho, ok):
        if not o:
            break
        mono = x["rho"]
    return ref, star, mono


def part_D(tag, summary, jobs, quick):
    s = model_spec(**(dict(MODEL, n=200) if quick else MODEL))
    prob = register(s)
    n = s["n"]
    nm = M9.norms(s)
    scale = max(1.0, prob.K / prob.d) ** 0.5
    rhos = M9.RHOS if not quick else (0.61, 0.368, 0.18, 0.05)
    low = () if quick else M9.RHOS_LOW
    batches = D_BATCHES if not quick else (5,)
    seeds = M9.ORDER_SEEDS if not quick else M9.ORDER_SEEDS[:2]
    grid_of = lambda b, m: rhos + (low if m == 3 * b else ())                                       # noqa: E731
    cells = [(b, name, order, m) for b in batches if 4 * b <= n for (name, order, m) in M9.schemes(b)]
    tasks, index = [], []
    for b, name, order, m in cells:
        for rho in grid_of(b, m):
            for seed in seeds:
                t, info = M9.order_task(s, b, rho, order, m, seed, nm["W_star_op"] / scale)
                tasks.append(t)
                index.append((b, name, rho, info))
    if jobs and jobs > 1 and mp.get_start_method() != "fork":
        jobs = 1                                                  # the registration is inherited by forked workers only
    print(f"\n=== D on the model instance: n = {n}, K = {s['K']}, b in {list(batches)}, seeds {list(seeds)}, "
          f"{len(tasks)} runs", flush=True)
    res = C.run_tasks(tasks, f"{tag}_order", jobs)
    if any(r.get("error") for r in res):
        raise RuntimeError(f"failed runs: {[r['error'] for r in res if r.get('error')][:3]}")
    tk = f"{M9.ORDER_TOL:g}"
    e = {"cells": {}, "spec": s, "norms": nm, "step_scale": scale, "rho_grid": list(rhos), "tol": M9.ORDER_TOL,
         "rho_grid_extension_m3b": list(low), "rho_ref": rhos[0], "total_decay": M9.TOTAL_DECAY,
         "path_factor": M9.PATH_FACTOR, "track_factor": M9.TRACK_FACTOR, "track_quorum": M9.TRACK_QUORUM}
    for b, name, order, m in cells:
        by_rho = []
        for rho in grid_of(b, m):
            sel = [(info, r) for (bb, nn, rr, info), r in zip(index, res) if (bb, nn, rr) == (b, name, rho)]
            Cs = []
            for _, r in sel:
                t = M9.ratio_trace(r, nm["D0"])
                Cs.append(float(t[2][-1]) if t is not None else float("inf"))
            pt = [r["passes_to"][tk] for _, r in sel]
            by_rho.append({"rho": rho, **sel[0][0], "C": Cs, "passes_to": pt,
                           "reached": int(np.isfinite(pt).sum()), "runs": len(Cs)})
        ref, star, mono = thresholds(by_rho)
        full = [x for x in by_rho if x["reached"] == x["runs"]]
        fastest = None
        if full:                                                  # as in part D of M9: every seed reaches the target
            x = min(full, key=lambda x: float(np.median(x["passes_to"])))
            fastest = {"rho": x["rho"], **M9.stats(x["passes_to"])}
        cell = e["cells"].setdefault(str(b), {"schemes": {}, "seeds": list(seeds)})
        cell["schemes"][name] = {"order": order, "refresh_extra": m,
                                 "memory_factor": (1.0 - (b + m) / n) ** (n / b) if order == "iid" else None,
                                 "reference_C": ref, "rho_star": star, "rho_star_monotone": mono,
                                 "fastest": fastest,
                                 "level_over_reference": [float(np.median(x["C"]) / ref) for x in by_rho],
                                 "by_rho": by_rho}
    summary["D"] = e
    m9 = None if quick else C.load_json("M9_summary")
    cmp_ = {}
    print(f"  {'b':>3s} {'scheme':34s} {'memory':>7s} {'rho_*':>6s} {'(mono)':>6s} {'C at rho_ref':>12s}   "
          f"{'soft-mem':>8s} {'ls-mem':>7s}   runs on schedule by rho")
    for b in sorted(e["cells"], key=int):
        for name, c in e["cells"][b]["schemes"].items():
            meas, fast = {}, {"model": (c["fastest"] or {}).get("median")}
            for inst in M9.ORDER_ON:
                try:
                    sc = m9["D"][inst]["cells"][b]["schemes"][name]
                    meas[inst] = sc["rho_star"]
                    fast[inst] = (sc.get("fastest") or {}).get("median")
                except (KeyError, TypeError):
                    meas[inst] = fast[inst] = None
            cmp_.setdefault(b, {})[name] = {"model": c["rho_star"], **meas, "passes_at_fastest_rho": fast}
            g = lambda v: "-" if v is None else format(v, "g")                                      # noqa: E731
            print(f"  {b:>3s} {name:34s} "
                  f"{'' if c['memory_factor'] is None else format(c['memory_factor'], '.3f'):>7s} "
                  f"{g(c['rho_star']):>6s} {g(c['rho_star_monotone']):>6s} {c['reference_C']:12.3g}   "
                  f"{g(meas.get('soft-mem')):>8s} {g(meas.get('ls-mem')):>7s}   "
                  + " ".join(str(x["on_schedule"]) for x in c["by_rho"]), flush=True)
    summary["D_vs_M9"] = cmp_


# ----------------------------------------------------------------------------------------------
# part L: long horizons, simulator in error coordinates
# ----------------------------------------------------------------------------------------------
def model_run(n, K, x, order, passes, runs=1, seed=0, b=1, lam=0.0, E0=None, eta0=None, batches=None, trace=False):
    """`runs` independent runs of the fixed-length recipe on the model, in error coordinates.

        omega = (1 + lam) E + mean_i Psi_i - mean_{j in B} Psi_j ;   Psi_j <- E for j in B ;
        E <- E - eta_t omega / ||omega||,        eta_t = eta_0 exp(-x t b / (2 n))        (lr^2 loses exp(-x) per pass),

    from E_0 (default: a unit vector) with the table initialised at E_0 and eta_0 = PATH_FACTOR ||E_0|| (1 - q^(b/n)),
    q = exp(-x / 2): the convention of part D of M9 on a horizon without end. `order`: 'iid' (indices uniform and
    independent; a uniform subset for b > 1), 'reshuffle' or 'shuffle_once'; b must divide n. `batches`: an iterator
    of index arrays replacing the internal draws (runs = 1; used by the tests to feed the batches of the lab).
    Returns the array (passes, runs) of ||E||^2 / eta^2 at the end of every pass, eta being the length of the last
    step; with trace=True, also the array (iterations, runs) of ||E||^2 after every step."""
    if n % b:
        raise ValueError("b must divide n")
    if order not in ("iid", "reshuffle", "shuffle_once"):
        raise ValueError(f"unknown order {order!r}")
    if batches is not None and runs != 1:
        raise ValueError("explicit batches are for a single run")
    rng = np.random.default_rng(seed)
    step = math.exp(-x * b / (2.0 * n))
    if E0 is None:
        E = np.zeros((runs, K))
        E[:, 0] = 1.0
    else:
        E = np.tile(np.asarray(E0, dtype=float).reshape(1, K), (runs, 1))
    if eta0 is None:
        eta0 = M9.PATH_FACTOR * float(np.linalg.norm(E[0])) * (1.0 - step)
    Psi = np.repeat(E[:, None, :], n, axis=1)
    ar = np.arange(runs)[:, None]
    steps = n // b
    lev = np.empty((passes, runs))
    tr = np.empty((passes * steps, runs)) if trace else None
    perm = None
    t = 0
    eta = eta0
    for p in range(passes):
        pbar = Psi.mean(1)                                        # exact table mean at the start of every pass
        if batches is None:
            if order == "iid" and b == 1:
                draws = rng.integers(n, size=(steps, runs, 1))
            elif order == "reshuffle" or (order == "shuffle_once" and perm is None):
                perm = np.argsort(rng.random((runs, n)), axis=1)
        for k in range(steps):
            if batches is not None:
                B = np.atleast_1d(np.asarray(next(batches))).reshape(1, b)
            elif order == "iid":
                B = draws[k] if b == 1 else np.argpartition(rng.random((runs, n)), b, axis=1)[:, :b]
            else:
                B = perm[:, k * b:(k + 1) * b]
            PB = Psi[ar, B]
            om = (1.0 + lam) * E + pbar - PB.mean(1)
            pbar = pbar + (b * E - PB.sum(1)) / n
            Psi[ar, B] = E[:, None, :]
            eta = eta0 * math.exp(-x * t * b / (2.0 * n))
            nr = np.sqrt(np.einsum("ij,ij->i", om, om))
            nr[nr == 0.0] = 1.0
            E = E - eta * om / nr[:, None]
            if trace:
                tr[t] = np.einsum("ij,ij->i", E, E)
            t += 1
        lev[p] = np.einsum("ij,ij->i", E, E) / eta ** 2
    return (lev, tr) if trace else lev


def _long_task(args):
    n, K, x, order, passes, runs, seed = args
    lev = model_run(n, K, x, order, passes, runs=runs, seed=seed)
    h = passes // 2
    with np.errstate(divide="ignore"):
        med = np.median(np.log(lev), axis=1)
    return {"x": x, "rho": math.exp(-x), "level_half": lev[h - 1].tolist(), "level_end": lev[-1].tolist(),
            "rate_second_half": float((med[-1] - med[h - 1]) / (passes - h))}


def read_long(rows, key, factor):
    """Largest x of the grid followed at the horizon `key` ('level_half' or 'level_end'): at least 80% of the runs
    have a level at most `factor` times the median level at the reference x (first row). Returns (x_star, x_star
    with the condition at every smaller x of the grid too, runs on schedule by x)."""
    ref = float(np.median(rows[0][key]))
    on = [int(sum(c <= factor * ref for c in r[key])) for r in rows]
    ok = [o >= M9.TRACK_QUORUM * len(r[key]) for o, r in zip(on, rows)]
    star = max((r["x"] for r, o in zip(rows, ok) if o), default=None)
    mono = None
    for r, o in zip(rows, ok):
        if not o:
            break
        mono = r["x"]
    return star, mono, on, ref


def main_study_exponents(n):
    """Exponents of linear SAGA on the same model (nats per pass of the squared error), from the experiment B4b of
    the main study: the largest typical exponent of a run under independent sampling over the steps tried, the
    expectation at that step, and the largest exact mean-square exponent under reshuffling. None if the file is
    absent."""
    path = os.path.join(C.MUON, os.pardir, "results", "B4b_summary.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        typ = json.load(f).get("typical", {}).get(f"n={n}")
    if not typ:
        return None
    c_best = max(typ, key=lambda c: typ[c]["pooled"])
    return {"typical_iid": typ[c_best]["pooled"], "typical_iid_se": typ[c_best]["pooled_se"], "at_c": float(c_best),
            "expectation_iid_at_that_c": typ[c_best]["expectation"],
            "reshuffled_mean_square_best": max(v["rr"] for v in typ.values()),
            "source": "results/B4b_summary.json, typical/n=<n>/<c>/{pooled, expectation, rr}"}


def part_L(summary, jobs, quick):
    ns = LONG_N if not quick else (100,)
    passes = LONG_PASSES if not quick else 40
    runs = LONG_RUNS if not quick else 8
    xs = (X_REF,) + (LONG_X if not quick else (1.0, 1.4, 1.8, 2.2, 2.6))
    tasks = [(n, LONG_K, x, order, passes, runs, 1000 + 10 * i)
             for n in ns for order, _ in ORDERS for i, x in enumerate(xs)]
    print(f"\n=== L: batch one, K = {LONG_K}, n in {list(ns)}, {passes} passes, {runs} runs per point, "
          f"{len(tasks)} points", flush=True)
    t0 = time.perf_counter()
    if jobs and jobs > 1:
        with mp.Pool(jobs) as pool:
            res = pool.map(_long_task, sorted(tasks, key=lambda a: -a[0]), chunksize=1)
        res = {(a[0], a[3], a[2]): r for a, r in zip(sorted(tasks, key=lambda a: -a[0]), res)}
    else:
        res = {(a[0], a[3], a[2]): _long_task(a) for a in tasks}
    print(f"  {len(tasks)} points in {time.perf_counter() - t0:.0f}s", flush=True)
    sec = {"K": LONG_K, "passes": passes, "runs": runs, "x_grid": list(xs), "x_ref": X_REF, "cells": {}}
    for n in ns:
        cell = sec["cells"].setdefault(f"n={n}", {"orders": {}, "linear_saga": main_study_exponents(n)})
        for order, label in ORDERS:
            rows = [res[(n, order, x)] for x in xs]
            out = {"label": label, "rows": rows}
            for key in ("level_half", "level_end"):
                star, mono, on, ref = read_long(rows, key, M9.TRACK_FACTOR)
                cstar, cmono, con, _ = read_long(rows, key, CLEAN_FACTOR)
                out[key] = {"passes": passes // 2 if key == "level_half" else passes, "reference_level": ref,
                            "x_star": star, "x_star_monotone": mono, "on_schedule": on,
                            "x_star_clean": cstar, "x_star_clean_monotone": cmono, "on_schedule_clean": con,
                            "level_over_reference": [float(np.median(r[key]) / ref) for r in rows]}
            cell["orders"][order] = out
            e = out["level_end"]
            print(f"  n = {n:5d} {label:27s} x_star = {e['x_star']} (within {CLEAN_FACTOR:g} x the reference level: "
                  f"{e['x_star_clean']}); at {passes // 2} passes: {out['level_half']['x_star']} "
                  f"({out['level_half']['x_star_clean']})")
            print("      x          :", " ".join(f"{x:5.2f}" for x in xs))
            print("      on schedule:", " ".join(f"{o:5d}" for o in e["on_schedule"]))
            print("      level / ref:", " ".join(f"{v:.2g}" for v in e["level_over_reference"]), flush=True)
        ls = cell["linear_saga"]
        if ls:
            print(f"      linear SAGA on the model (B4b): typical run, independent sampling {ls['typical_iid']:.3f} "
                  f"(expectation {ls['expectation_iid_at_that_c']:.3f}); reshuffling, mean square "
                  f"{ls['reshuffled_mean_square_best']:.3f}")
    summary["L"] = sec


def definitions():
    return {
        "model": "f_i(W) = ||W - y_i||^2 / 2 + (lam/2)||W||^2 on K x 1 matrices (x_i = 1): identical Hessians; the "
                 "polar factor of the estimator is the estimator divided by its norm",
        "D": "the design of part D of M9 on the model instance (definitions/order of M9_summary.json), with lr_0 "
             "divided by step_scale = max(1, K/d)^(1/2) so that the total step length is path_factor x ||W_*||_op; "
             "level_over_reference = median C at each rho of the grid over the reference level",
        "D_vs_M9": "rho_star on the model next to rho_star on soft-mem and ls-mem (M9_summary.json, "
                   "D/<instance>/cells/<b>/schemes/<scheme>/rho_star), by batch size and scheme; "
                   "passes_at_fastest_rho = median passes to the target of part D (relative squared distance 1e-9, "
                   "extra refreshes charged) at the rho with the smallest median among those where every seed "
                   "reaches it (fastest/median of the same cells)",
        "L": f"batch one, runs of {LONG_PASSES} passes of `model_run` (20 runs per point, error coordinates, "
             f"lam = 0, E_0 a unit vector, eta_0 = {M9.PATH_FACTOR:g} ||E_0|| (1 - q^(1/n))); x = log(1/rho) = nats "
             f"of lr^2 lost per pass; level = ||E||^2 / eta^2 at the end of a pass; a run is on schedule at a horizon "
             f"if its level is at most {M9.TRACK_FACTOR:g} x the median level at x_ref = log(1/0.61) at that horizon "
             f"(x_star_clean: {CLEAN_FACTOR:g} x); x_star = largest x of the grid with at least "
             f"{M9.TRACK_QUORUM:g} of the runs on schedule (x_star_monotone: and at every smaller x of the grid); "
             f"rate_second_half = slope per pass of the median log level over the second half of the run (0: on "
             f"schedule; x: frozen iterate)",
        "linear_saga": "exponents of linear SAGA on the same model with the same n, read from the experiment B4b "
                       "of the main study (None when results/B4b_summary.json is absent): typical_iid = largest "
                       "pooled exponent of single runs under independent sampling over the steps tried there "
                       "(at_c: gamma n lambda of that step); expectation_iid_at_that_c = exact exponent of the "
                       "expected squared error at that step; reshuffled_mean_square_best = largest exact "
                       "mean-square exponent under reshuffling over the same steps",
    }


def main():
    argv = sys.argv
    quick = "--quick" in argv

    def opt(name, default=None):
        return argv[argv.index(name) + 1] if name in argv else default
    jobs = int(opt("--jobs", "6"))
    parts = opt("--parts", "D,L").split(",")
    tag = "M9b_quick" if quick else "M9b"
    summary = C.load_json(f"{tag}_summary") or {}
    t0 = time.perf_counter()
    for part in parts:
        if part == "D":
            part_D(tag, summary, jobs, quick)
        elif part == "L":
            part_L(summary, jobs, quick)
        else:
            raise ValueError(f"unknown part {part!r}")
    summary["experiment"] = "M9b: the sampling-order part of M9 on a sum with identical Hessians"
    summary["quick"] = bool(quick)
    summary["code"] = C.code_fingerprint()
    summary["definitions"] = definitions()
    C.save_json(f"{tag}_summary", summary)
    print(f"\nM9b done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
