"""Data shared by the explanatory figures about the recipe of Section 3.3 of paper/paper_saga_to_muon.tex (steps of
fixed length on a geometric cooldown of the learning rate, fed by a SAGA table):

    figT14_estimator_error, figT15_schedule_law, figT15b_schedule_band, figT16_same_schedule
                                                                        (lecture, paper/talk/figures)
    figE5_recipe_table                                                  (appendix of the paper, paper/figures)

Nothing is drawn here and nothing is written: the module reads the summary and the cache of part C of experiment M9
and recomputes one quantity along one cached run.

Sources (read only).
  muon/results/M9_summary.json   C/<instance>: spec, passes (300), seeds (10..14), norms, deep (runs with the SAGA
                                 table), controls (runs without a table) and their statistics.
  muon/results/M9_law.pkl        the 110 runs of part C, keyed by the sha1 of the task (as muon/experiments/common.py,
                                 task_key). Each run: trace = 240 records with passes (per-sample gradient evaluations
                                 / n; the initial full gradient of the table counts for one pass), rel (squared
                                 distance to W_* over ||W_*||_F^2; W_0 = 0, so rel = 1 at the start and rel is also the
                                 fraction of the initial squared distance), eta (the length of the last step taken,
                                 i.e. the learning rate lr_t, the scale max(1, K/d)^(1/2) being 1 on these instances;
                                 NaN at the first record), and final_rel, min_rel.
  muon/lab (project code)        problems.make_matrix_ls, optimum.solve_optimum, exact.collect_states: the instance
                                 and the states (iterate and table) of one cached run, recomputed (about 6 s).

Functions.
  load()                the summary and the cache.
  series(...)           one configuration of part C on one instance: its cfg and its five runs (passes, rel, lr).
                        The learning rate of every record is checked against the closed form of the schedule
                        (lr_0 q^((k - 1)/period) for the k-th step) and the statistics against the summary.
  check_paper(...)      the numbers of (M5) that the figures rely on, recomputed from the cache; SystemExit if one is
                        not reproduced.
  same_but_estimator()  the pair of cached configurations that differ only by the estimator (SAGA table / minibatch
                        gradient); SystemExit if any other key differs.
  estimator_error(...)  exact mean squared error of two estimators of grad F(W_t) at the states of one cached run with
                        the table (enumeration of the n indices, and a closed form; SystemExit if they disagree or if
                        the recomputed trajectory is not the cached one).
  law_data(...)         the five configurations of the schedule law with their floors, levels and band widths;
  print_law(...)        prints them (every number that the figures of the law show).

Status of what the figures built on this module show: measured (cached runs of experiment M9, part C; five seeds, one
data draw per instance) and measured exactly (the enumeration over the index along one of these runs). Nothing here is
proved at these cooldowns: Proposition 5(b) certifies cooldowns 1e4 to 1e6 times slower, for the exact polar factor
without momentum; Proposition 5(c) (scale invariance on least squares) is what the floors of the schedule law
illustrate (the constant eps of ns5 breaks the invariance).
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))                 # paper/scripts -> project root
RESULTS = os.path.join(ROOT, "muon", "results")

INSTANCE = "ls-cond"            # least squares, n = 200, K x d = 8 x 40, kappa = 1e4 (conditioning-limited)
OTHER = "soft-cond"             # the softmax instance of part C (used for the range "8 % to 34 %" only)

# keys of M9_summary.json, C/<instance>/deep (SAGA table) and C/<instance>/controls (no table)
POLAR = "polar, no momentum"
RECIPE = "ns5 + Nesterov (Muon's recipe)"                                  # at its tuned point
NS5 = "ns5 + Nesterov at q = 0.8, lr_0 = 0.1"                              # fixed point, eps = 1e-7 (default of ns5)
NS5_EPS = "ns5 + Nesterov, eps = 1e-30, at q = 0.8, lr_0 = 0.1"
SGD = "no table (minibatch gradient), ns5 + Nesterov, b=1"
MARS = "MARS-M, b=1"
STORM = "STORM-Muon, b=1"

DASH = (0, (4.0, 1.8))          # ns5 (Newton-Schulz) with the table; MARS-M among the runs without a table
DOT = (0, (1.0, 1.4))           # ns5 with eps = 1e-30
DIST_LABEL = r"$\|W_t - W_\ast\|_F^2\ /\ \|W_\ast\|_F^2$"


def task_key(task):
    """Key of a task in the caches of muon/experiments/common.py (common.task_key)."""
    return hashlib.sha1(json.dumps(task, sort_keys=True).encode()).hexdigest()


def load():
    with open(os.path.join(RESULTS, "M9_summary.json"), encoding="utf-8") as f:
        m9 = json.load(f)
    path = os.path.join(RESULTS, "M9_law.pkl")
    if not os.path.exists(path):
        raise SystemExit(f"{path} is missing (cache of part C of experiment M9)")
    with open(path, "rb") as f:
        cache = pickle.load(f)
    return m9, cache


def _schedule(method, cfg, n):
    """(lr_0, q, period in iterations of one factor q, gradient evaluations before the first step, per step,
    discount of the first step) of a configuration of part C."""
    if method == "vr":
        if cfg.get("schedule") != "geometric" or cfg.get("length") != "const":
            raise SystemExit(f"not a fixed-length geometric recipe: {cfg}")
        b = int(cfg["batch"])
        init = n if cfg["estimator"] == "saga" else 0
        return float(cfg["lr"]), float(cfg["lr_decay"]), n / b, init, b, 0
    kind, q, period = cfg["schedule"]                              # momentum-based estimators (muon/lab/mvr.py)
    if kind != "geometric":
        raise SystemExit(f"not a geometric cooldown: {cfg}")
    b = int(cfg["batch"])
    return float(cfg["lr"]), float(q), float(period), 0, 2 * b, b     # 2 b gradients per step, b at the first one


def series(m9, cache, kind, label, instance=INSTANCE):
    """One configuration of part C (kind: 'deep' or 'controls'): dict with cfg, method, lr0, q and `runs`, one per
    seed: seed, passes, rel, lr (the learning rate of the last step taken at each record; lr_0 at the first record,
    where no step has been taken), final, lowest."""
    e = m9["C"][instance]
    d = e[kind][label]
    method = d.get("method", "vr")
    cfg = d["cfg"]
    n = e["spec"]["n"]
    lr0, q, period, init, per_step, first = _schedule(method, cfg, n)
    runs = []
    for seed in e["seeds"]:
        task = {"spec": e["spec"], "method": method, "cfg": cfg, "passes": e["passes"], "seed": seed,
                "stop_rel": 0.0}
        r = cache.get(task_key(task))
        if r is None:
            raise SystemExit(f"run missing from M9_law.pkl: {instance}, {label}, seed {seed}")
        tr = r["trace"]
        p, rel, eta = (np.asarray(tr[k], dtype=float) for k in ("passes", "rel", "eta"))
        if not (np.isnan(eta[0]) and np.isfinite(eta[1:]).all() and p[0] == init / n and rel[0] == 1.0):
            raise SystemExit(f"unexpected trace: {instance}, {label}, seed {seed}")
        k = (p[1:] * n - init + first) / per_step                 # number of steps taken at each record
        if np.abs(k - np.round(k)).max() > 1e-6:
            raise SystemExit(f"records between two steps: {instance}, {label}, seed {seed}")
        formula = lr0 * q ** ((np.round(k) - 1.0) / period)        # the k-th step has the length lr_0 q^((k-1)/period)
        if not np.allclose(eta[1:], formula, rtol=1e-9, atol=0.0):
            raise SystemExit(f"the recorded learning rate is not lr_0 q^(steps/period): {instance}, {label}, "
                             f"seed {seed}")
        lr = eta.copy()
        lr[0] = lr0
        runs.append({"seed": seed, "passes": p, "rel": rel, "lr": lr, "final": float(r["final_rel"]),
                     "lowest": float(r["min_rel"])})
    out = {"instance": instance, "kind": kind, "label": label, "method": method, "cfg": cfg, "lr0": lr0, "q": q,
           "runs": runs, "D0": float(e["norms"]["D0"]), "n": n}
    # the statistics of the summary are those of these runs
    if kind == "deep":
        med = float(np.median([r["lowest"] for r in runs]))
        if not np.isclose(med, d["min_rel"]["median"], rtol=1e-12):
            raise SystemExit(f"summary and cache disagree on the lowest distance: {instance}, {label}")
        out["on_schedule"] = int(d["on_schedule"])
        out["level"] = d["level"]["median"] if "level" in d else float("nan")   # median of dist^2 / lr^2 (absolute)
    else:
        if not np.allclose([r["final"] for r in runs], d["values"], rtol=1e-12):
            raise SystemExit(f"summary and cache disagree on the final distance: {instance}, {label}")
    return out


def lowest_median(s):
    return float(np.median([r["lowest"] for r in s["runs"]]))


def final_median(s):
    return float(np.median([r["final"] for r in s["runs"]]))


def ratio(run):
    """rel / lr^2 at the records after the first one (the ratio of the two axes of the parametric plot)."""
    return run["passes"][1:], run["rel"][1:] / run["lr"][1:] ** 2


def level(s, window=(1e-12, 1e-6)):
    """Median over the seeds of the median of rel / lr^2 over the records with rel in the window (the definition of
    the 'level' of the schedule law in muon/experiments/exp_M9_schedule.py, law_depth, here in relative units), and
    the same level in absolute units, ||W - W_*||_F^2 / lr^2, to compare with the summary."""
    lv = []
    for r in s["runs"]:
        _, x = ratio(r)
        rel = r["rel"][1:]
        win = (rel >= window[0]) & (rel <= window[1])
        if win.any() and rel.min() <= window[0]:
            lv.append(float(np.median(x[win])))
    med = float(np.median(lv)) if lv else float("nan")
    return med, med * s["D0"], len(lv)


def band_width(s, window=(1e-12, 1e-6)):
    """Per seed, max / min of rel / lr^2 over the records with rel in the window (a factor); list over the seeds that
    cross the window."""
    out = []
    for r in s["runs"]:
        _, x = ratio(r)
        rel = r["rel"][1:]
        win = (rel >= window[0]) & (rel <= window[1])
        if win.any() and rel.min() <= window[0]:
            out.append(float(x[win].max() / x[win].min()))
    return out


def check_paper(m9, cache):
    """The numbers of (M5) (paper, Section 3.3; corrected wording of the audit) that the figures rely on, recomputed
    from the cache. Returns a dict of the recomputed values; SystemExit if one is not reproduced."""
    out = {}
    polar = series(m9, cache, "deep", POLAR)
    ns5 = series(m9, cache, "deep", NS5)
    eps = series(m9, cache, "deep", NS5_EPS)
    out["floor_polar"] = lowest_median(polar)
    out["floor_ns5"] = lowest_median(ns5)
    out["floor_eps"] = lowest_median(eps)
    if f"{out['floor_polar']:.1e}" != "2.3e-29":
        raise SystemExit(f"(M5): floor of the exact polar factor on least squares {out['floor_polar']:.3e}, paper "
                         "2.3e-29")
    if f"{out['floor_eps']:.1e}" != "4.1e-30":
        raise SystemExit(f"(M5): floor of ns5 with eps = 1e-30 on least squares {out['floor_eps']:.3e}, paper 4.1e-30")
    if not 0.5e-18 <= out["floor_ns5"] <= 2e-18:
        raise SystemExit(f"(M5): floor of ns5 at q = 0.8, lr_0 = 0.1 {out['floor_ns5']:.3e}, paper: near 1e-18")
    # without a table, batch one, the two instances: medians of the final squared distance (fraction of the initial)
    med = {}
    for inst in (INSTANCE, OTHER):
        for label in (SGD, MARS, STORM):
            med[(inst, label)] = final_median(series(m9, cache, "controls", label, inst))
    out["no_table"] = med
    lo, hi = min(med.values()), max(med.values())
    if (round(100 * lo), round(100 * hi)) != (8, 34):
        raise SystemExit(f"(M5): without a table at batch one {100 * lo:.1f} % to {100 * hi:.1f} %, paper (corrected) "
                         "8 % to 34 %")
    # the level of the law in absolute units is the one of the summary
    for s in (polar, ns5, eps):
        _, absolute, k = level(s)
        if not np.isclose(absolute, s["level"], rtol=1e-9) or k != s["on_schedule"]:
            raise SystemExit(f"level of dist^2 / lr^2 of '{s['label']}': {absolute:.6g} from the cache, "
                             f"{s['level']:.6g} in the summary")
    return out


def same_but_estimator(m9, cache):
    """The two cached configurations of the controlled comparison (least squares, batch one): Muon's map and momentum
    (ns5 + Nesterov 0.9) on the schedule lr_0 = 0.3, q = 0.75, with the SAGA table and with the minibatch gradient.
    SystemExit if the two configurations differ by anything but the estimator."""
    tb = series(m9, cache, "deep", RECIPE)
    no = series(m9, cache, "controls", SGD)
    a, b = tb["cfg"], no["cfg"]
    diff = sorted(k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>"))
    if diff != ["estimator"] or (a["estimator"], b["estimator"]) != ("saga", "sgd"):
        raise SystemExit(f"the two configurations differ by {diff}, expected ['estimator'] (saga / sgd)")
    want = dict(orth="ns5", beta=0.9, nesterov=True, lr_decay=0.75, lr=0.3, batch=1, schedule="geometric",
                length="const")
    bad = {k: a.get(k) for k, v in want.items() if a.get(k) != v}
    if bad:
        raise SystemExit(f"unexpected configuration of the recipe at its tuned point: {bad}")
    if [r["seed"] for r in tb["runs"]] != [r["seed"] for r in no["runs"]]:
        raise SystemExit("the two configurations do not have the same seeds")
    return tb, no


# ----------------------------------------------------------------------------------------------------------------------
# the error of the estimator along one cached run (exact enumeration of the index)
# ----------------------------------------------------------------------------------------------------------------------
def _lab():
    muon = os.path.join(ROOT, "muon")
    if muon not in sys.path:
        sys.path.insert(0, muon)
    from lab import exact, optimum, problems
    return exact, optimum, problems


def problem(spec):
    """The instance of a specification with its minimizer (as muon/experiments/common.py, problem)."""
    _, optimum, problems = _lab()
    kw = {k: v for k, v in spec.items() if k != "loss"}
    mk = problems.make_softmax if spec["loss"] == "softmax" else problems.make_matrix_ls
    prob = mk(**kw)
    optimum.solve_optimum(prob)
    return prob


def estimator_error(m9, cache, label=POLAR, seed=10, passes=80, every=20, instance=INSTANCE):
    """Exact mean over the index j (uniform on the n indices) of ||omega_j - grad F(W_t)||_F^2 at the states of the
    cached run (instance, label, seed) with the SAGA table, every `every` iterations up to `passes` passes, for
      'saga'   omega_j = zbar + lam W_t + (r_j(W_t) - rtab_j) x_j^T      the estimator of the run (table rtab, zbar);
      'sgd'    omega_j = lam W_t + r_j(W_t) x_j^T                        the batch-one gradient, no table, evaluated at
                                                                        the SAME states (it does not drive the run);
    with r_j(W) the residual of sample j (least squares: W x_j - y_j). Each mean is computed by enumeration of the n
    matrices and by the closed form
      saga:  (1/n) sum_j ||d_j||^2 ||x_j||^2 - ||(1/n) sum_j d_j x_j^T||_F^2 + ||zbar - (1/n) sum_j rtab_j x_j^T||_F^2,
             d_j = r_j(W_t) - rtab_j   (variance + squared bias; the bias is the rounding drift of zbar)
      sgd:   (1/n) sum_j ||r_j||^2 ||x_j||^2 - ||(1/n) sum_j r_j x_j^T||_F^2;
    SystemExit if the two disagree, or if the recomputed trajectory is not the cached one (relative squared distance
    at every cached record of the window). Also returns ||grad F(W_t)||_F^2, the relative squared distance and the
    variance of the batch-one gradient at W_*."""
    exact, _, _ = _lab()
    e = m9["C"][instance]
    s = series(m9, cache, "deep", label, instance)
    cfg = s["cfg"]
    if cfg.get("estimator") != "saga" or int(cfg.get("batch", 1)) != 1:
        raise SystemExit("estimator_error needs a batch-one run with the SAGA table")
    run = next(r for r in s["runs"] if r["seed"] == seed)
    prob = problem(e["spec"])
    n, lam, X = prob.n, prob.lam, prob.X
    D0 = float(np.vdot(prob.W_star, prob.W_star))
    if not np.isclose(D0, e["norms"]["D0"], rtol=1e-12):
        raise SystemExit("the instance rebuilt from its specification is not the one of the summary")
    at = list(range(0, passes * n + 1, every))
    states = exact.collect_states(prob, np.zeros((prob.K, prob.d)), max(at) + 1, at, seed=seed, **cfg)
    if [t for t, _, _ in states] != at:
        raise SystemExit("the run stopped before the last requested state")
    sq = prob.sqnorms
    rows = []
    for t, W, st in states:
        R = prob.loss_residuals(W)                                      # n x K
        G = prob.grad(W)
        d = R - st.rtab
        # enumeration of the n estimators
        om = (st.zbar + lam * W)[None] + d[:, :, None] * X[:, None, :]
        e_saga = float(((om - G[None]) ** 2).sum(axis=(1, 2)).mean())
        sg = (lam * W)[None] + R[:, :, None] * X[:, None, :]
        e_sgd = float(((sg - G[None]) ** 2).sum(axis=(1, 2)).mean())
        # closed forms
        drift = st.zbar - st.rtab.T @ X / n
        c_saga = float(((d ** 2).sum(axis=1) * sq).mean() - ((d.T @ X / n) ** 2).sum() + (drift ** 2).sum())
        c_sgd = float(((R ** 2).sum(axis=1) * sq).mean() - ((R.T @ X / n) ** 2).sum())
        for a, b_, name in ((e_saga, c_saga, "SAGA"), (e_sgd, c_sgd, "batch-one gradient")):
            if abs(a - b_) > 1e-8 * max(abs(a), abs(b_)) + 1e-300:
                raise SystemExit(f"error of the {name} estimator at iteration {t}: enumeration {a:.12e}, closed form "
                                 f"{b_:.12e}")
        E = W - prob.W_star
        rows.append((1.0 + t / n, e_saga, e_sgd, float(np.vdot(G, G)), float(np.vdot(E, E)) / D0))
    A = np.array(rows)
    # the trajectory is the cached one
    cp, crel = run["passes"], run["rel"]
    checked, worst = 0, 0.0
    for p, rel in zip(cp, crel):
        k = np.nonzero(np.abs(A[:, 0] - p) < 1e-9)[0]
        if p <= passes + 1 and len(k):
            checked += 1
            worst = max(worst, abs(A[k[0], 4] - rel) / rel)
    if checked < 20 or worst > 1e-6:
        raise SystemExit(f"the recomputed run is not the cached one: {checked} records compared, largest relative "
                         f"difference {worst:.2e}")
    Rs = prob.R_star
    var_star = float(((Rs ** 2).sum(axis=1) * sq).mean() - ((Rs.T @ X / n) ** 2).sum())
    return {"passes": A[:, 0], "saga": A[:, 1], "sgd": A[:, 2], "grad2": A[:, 3], "rel": A[:, 4],
            "var_star": var_star, "checked": checked, "worst": worst, "seed": seed, "label": label, "cfg": cfg,
            "n": n, "K": prob.K, "d": prob.d, "lam": lam, "spec": e["spec"], "lr0": s["lr0"], "q": s["q"]}


# ----------------------------------------------------------------------------------------------------------------------
# the schedule law (parametric plot and band) and its printed numbers, shared by figT15, figT15b and figE5
# ----------------------------------------------------------------------------------------------------------------------
LAW_KEYS = (("polar", "deep", POLAR), ("ns5", "deep", NS5), ("eps", "deep", NS5_EPS),
            ("sgd", "controls", SGD), ("mars", "controls", MARS))


def law_data(m9, cache):
    """The five configurations of the schedule law on the least-squares instance (keys polar, ns5, eps: SAGA table;
    sgd, mars: no table), with, for the table runs: floor (median over the seeds of the lowest relative squared
    distance), level_rel / level_abs (see level), width (see band_width), width_summary (the 'width_median' of the
    summary: max / min of dist^2 / lr^2 over the second half of the run up to the last record within a factor 3 of
    the level, median over the seeds) and decades (per seed, the decades of lr^2 between the first and the last
    record whose ratio is within a factor 3 of the level). Also check (the numbers of (M5), see check_paper) and
    no_range (smallest and largest final value of the ten runs without a table)."""
    D = {"check": check_paper(m9, cache)}
    for k, kind, label in LAW_KEYS:
        D[k] = series(m9, cache, kind, label)
    for k in ("polar", "ns5", "eps"):
        s = D[k]
        s["floor"] = lowest_median(s)
        s["level_rel"], s["level_abs"], s["crossing"] = level(s)
        s["width"] = band_width(s)
        s["width_summary"] = float(m9["C"][INSTANCE]["deep"][s["label"]]["band"]["width_median"])
        dec = []
        for r in s["runs"]:
            _, x = ratio(r)
            lr2 = r["lr"][1:] ** 2
            ok = np.nonzero((x <= 3.0 * s["level_rel"]) & (x >= s["level_rel"] / 3.0))[0]
            dec.append(float(np.log10(lr2[ok[0]] / lr2[ok[-1]])))
        s["decades"] = dec
    no = [r["final"] for k in ("sgd", "mars") for r in D[k]["runs"]]
    D["no_range"] = (min(no), max(no))
    return D


def print_law(D):
    chk = D["check"]
    print("checks against (M5): floors (median over the seeds of the lowest relative squared distance) "
          f"{chk['floor_polar']:.2e} (exact polar; paper 2.3e-29), {chk['floor_ns5']:.2e} (ns5; paper: near 1e-18), "
          f"{chk['floor_eps']:.2e} (eps = 1e-30; paper 4.1e-30)")
    print("   without a table, batch one, median final squared distance / initial: "
          + ", ".join(f"{inst} {lab.split(',')[0].split(' (')[0]} {100 * v:.1f} %"
                      for (inst, lab), v in chk["no_table"].items()) + "  (paper, corrected: 8 % to 34 %)")
    for k, _, _ in LAW_KEYS:
        s = D[k]
        print(f"{k:5s} '{s['label']}': lr_0 = {s['lr0']:g}, q = {s['q']:g}; cfg {s['cfg']}")
        print("      lowest per seed: " + ", ".join(f"{r['lowest']:.2e}" for r in s["runs"])
              + "; final: " + ", ".join(f"{r['final']:.3g}" for r in s["runs"]))
        if "floor" in s:
            print(f"      floor (median) {s['floor']:.2e}; level of rel / lr^2 (rel in [1e-12, 1e-6]) "
                  f"{s['level_rel']:.3g} (absolute {s['level_abs']:.5g}; summary {s['level']:.5g}); width of the band "
                  "per seed over that window, max/min: " + ", ".join(f"{w:.2f}" for w in s["width"])
                  + f"; width of the summary (second half of the run, up to 3 x level) {s['width_summary']:.2f}"
                  + "; decades of lr^2 within a factor 3 of the level, per seed: "
                  + ", ".join(f"{d:.1f}" for d in s["decades"]))
    print(f"no table, the ten runs drawn: final {100 * D['no_range'][0]:.1f} % to {100 * D['no_range'][1]:.1f} % "
          "of the initial squared distance")


def tex(v):
    """2.3e-29 -> $2.3\\cdot10^{-29}$ (mathtext)."""
    m, e = f"{v:.1e}".split("e")
    return rf"${m}\cdot10^{{{int(e)}}}$"
