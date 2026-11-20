"""Hypothesis B4, refinements of the solvable model (vrlab.order_model) and of its use on logistic regression.
 (G) fixed g = gamma lambda: the exact second-moment matrices of the i.i.d. and reshuffled orders against the limits
     2 (1 - g)/(2 - g) and min(2, log((2 - g)/g)); the best exact reshuffled exponent over c at finite n.
 (T) typical runs under i.i.d. sampling: exponent of single simulated runs for n up to 1e5, against the exact
     exponent of the expectation, the reshuffled exponent at the same (n, c) and the heuristic
     order_model.iid_typical_heuristic.
 (F) logistic regression: the effective curvature L~ of each problem of experiment B4 (order_model
     .effective_curvature_glm at the optimum) and the two-term law min(r_rr(c_min), log((2 - g)/g)), g = gamma L~,
     against the measured reshuffled exponents of results/B4_summary.json; 1 - g/(2 - g) against the i.i.d. runs.
 (O) the other extreme, components on orthogonal directions: exponents of the three orders against c = gamma R^2.
 (N) the numbers printed in the B4 note outside its paragraph "Outcome of the experiments", regenerated from
     vrlab.order_model and vrlab.bounds (no number of the note comes from a scratch computation any more).
Exponents are per pass (n iterations), for SQUARED errors, in nats. Definitions: entry "definitions" of
results/B4b_summary.json. Requires results/B4_summary.json (run exp_B4_sampling_order.py first).
Usage: python experiments/exp_B4b_refinements.py [--quick]        (VRLAB_WORKERS=k: k worker processes, default 1)
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.optimize import brentq, minimize_scalar

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import plotstyle as ps  # noqa: E402
from vrlab import algorithms, bounds, order_model  # noqa: E402

QUICK = "--quick" in sys.argv
WORKERS = max(1, int(os.environ.get("VRLAB_WORKERS", "1")))
TAG = "B4b_quick" if QUICK else "B4b"

# (G)
G_GS = [0.05, 0.1, 0.2, 0.25, 0.32, 0.5, 0.75, 0.9, 1.1, 1.25]
G_NS = [100, 1000] if QUICK else [1000, 10000, 100000]
G_BEST_NS = [100] if QUICK else [100, 1000, 10000]
# (T)
T_NS = [100, 1000] if QUICK else [100, 1000, 10000, 100000]
T_CS = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
T_EXTRA = {} if QUICK else {1000: [64.0, 125.0, 250.0]}      # g = c/n up to the textbook value 1/4 at n = 1000
T_SEEDS = 4 if QUICK else 12
T_PASSES, T_BURN = (120, 40) if QUICK else (260, 60)
# (F) the logistic problems of experiment B4 (taken from its summary)
F_N, F_D = (300, 20) if QUICK else (1000, 50)
# (O)
O_CS = [0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.65, 0.7, 1.0]
O_N = 1000
O_PASSES = 4000 if QUICK else 40000
O_SEEDS = [0, 1, 2]

CONFIG = dict(G_GS=G_GS, G_NS=G_NS, G_BEST_NS=G_BEST_NS, T_NS=T_NS, T_CS=T_CS, T_EXTRA=T_EXTRA, T_SEEDS=T_SEEDS,
              T_PASSES=T_PASSES, T_BURN=T_BURN, F_N=F_N, F_D=F_D, O_CS=O_CS, O_N=O_N, O_PASSES=O_PASSES, O_SEEDS=O_SEEDS)

DEFINITIONS = {
    "exponent": "contraction exponent per pass (n iterations) of a SQUARED error, in nats; negative: growth.",
    "c, g": "c = gamma n lambda and g = gamma lambda = c / n for a mode of eigenvalue lambda of the common Hessian.",
    "fixed_g.<g>": "iid_exact[n]: order_model.iid_exponent(n, n g), exact exponent of E e^2 for i.i.d. indices; "
                   "iid_limit: 2 (1 - g)/(2 - g); rr_exact[n]: order_model.rr_ms_exponent(n, n g), exact exponent of "
                   "E e^2 for the reshuffled order; rr_limit: min(2, log((2 - g)/g)); rr_mean[n]: exponent of the "
                   "squared mean (first-moment pass matrix).",
    "fixed_g.crossing": "g at which g/(2 - g) = exp(-2), i.e. 2/(1 + e^2).",
    "rr_best.<n>": "c maximising order_model.rr_ms_exponent(n, c) over [1, n] (bounded scalar search) and the maximum.",
    "typical.<n>.<c>": f"single runs of the scalar model with i.i.d. indices, {T_SEEDS} seeds, {T_PASSES} passes, "
                       f"state rescaled at every pass; exponent of one run = minus the least-squares slope of the "
                       f"pass-end values of log max(e^2, max_j psi_j^2) over the passes {T_BURN}..{T_PASSES}; median, "
                       f"min, max over the seeds; pooled: slope of the seed average of that logarithm between the two "
                       f"ends of the window, with its standard error over the seeds; expectation: exact exponent of "
                       f"E e^2 (order_model.iid_exponent); rr: exact exponent of E e^2 under reshuffling "
                       f"(order_model.rr_ms_exponent); heuristic: order_model.iid_typical_heuristic(n, c).",
    "typical.heuristic_error": "max and median of |heuristic - median| over the cells with c >= 8 and c/n <= 0.05, and "
                               "over the cells with c <= 4.",
    "logistic.<problem>": "L_eff: order_model.effective_curvature_glm(X, h, lambda) at the optimum, h_i the second "
                          "derivative of the logistic loss; L_eff_over_R2, L_eff_over_mean_Li (mean_i of "
                          "h_i ||x_i||^2 + lambda), L_eff_over_lmax (lambda_max(F''(eta_*))); by_step[gammaR2]: g = "
                          "gamma L_eff, cap = log((2 - g)/g) (null for g >= 2), law = r_rr(c_min) of experiment B4, "
                          "two_term = min(law, cap), measured = median reshuffled exponent of B4 (null unless every "
                          "run converged), rel_err_law and rel_err_two_term = measured / prediction - 1; "
                          "iid_measured = median i.i.d. exponent of B4 and iid_expectation_heuristic = 1 - g/(2 - g).",
    "logistic.summary": "over the (problem, step) pairs with a measured reshuffled exponent, by group of steps and of "
                        "kappa: number of pairs, max and median of |rel_err| for the law (R3) alone and for the "
                        "two-term law, and the number of pairs within 10%.",
    "logistic.best_step": "gammaR2 at which the two terms are equal (brentq on the step), the value of the two-term law "
                          "there, and the best step and exponent of the grid of experiment B4.",
    "orthogonal.<c>": f"components on orthogonal directions, c = gamma n lambda = gamma R^2, n = {O_N}: cyclic = exact "
                      f"exponent (2x2 pass map), cyclic_limit = the same for n -> infinity, rr and iid = Lyapunov "
                      f"exponents of single runs ({O_PASSES} passes, seeds {O_SEEDS}: median, min, max).",
    "note": "numbers printed in theory/B4_sampling_order.tex outside its Outcome paragraph; each key names the "
            "function of vrlab.order_model or vrlab.bounds that produces it (see the comments of the note).",
}


def fk(x):
    return f"{x:.4g}"


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    return o


def spread(values):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if not v.size:
        return {"median": None, "min": None, "max": None, "n": 0}
    return {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()), "n": int(v.size)}


# ----------------------------------------------------------------------------------------------
# (T) typical runs under i.i.d. sampling: batched simulation of the scalar model
# ----------------------------------------------------------------------------------------------
def typical_task(task):
    """R = len(cs) * seeds independent runs of e+ = e - g (e - psi_i + psibar), psi_i <- e advance together (one
    uniform index per run and per step). The state of every run is rescaled at the end of each pass (the recursion is
    linear) and psibar is recomputed there. Returns log max(e^2, max psi^2) at pass ends, array (passes + 1, R)."""
    n, cs, seeds, passes = task
    R = len(cs) * seeds
    g = np.repeat(np.asarray(cs, dtype=np.float64), seeds) / n
    rng = np.random.default_rng([20261119, n, len(cs)])
    e, psi = np.ones(R), np.ones((R, n))
    ar = np.arange(R)
    logscale = np.zeros(R)
    out = np.zeros((passes + 1, R))
    chunk = max(1, min(n, 100000))
    for p in range(passes):
        pbar = psi.mean(axis=1)
        done = 0
        while done < n:
            k = min(chunk, n - done)
            idx = rng.integers(0, n, size=(k, R))
            for s in range(k):
                i = idx[s]
                d = e - psi[ar, i]
                en = e - g * (d + pbar)
                pbar += d / n
                psi[ar, i] = e
                e = en
            done += k
        size = np.maximum(np.abs(e), np.abs(psi).max(axis=1))
        e /= size
        psi /= size[:, None]
        logscale += np.log(size)
        out[p + 1] = 2.0 * logscale
    return out


def run_typical(res, pmap):
    tasks = [(n, tuple(T_CS + T_EXTRA.get(n, [])), T_SEEDS, T_PASSES) for n in T_NS]
    todo = [t for t in tasks if t not in res["typical"]]
    if not todo:
        return
    with common.Timer(f"typical i.i.d. runs: n in {[t[0] for t in todo]}"):
        for t, out in zip(todo, pmap(typical_task, todo)):
            res["typical"][t] = out
            common.save_result(TAG, res)


def analyse_typical(res):
    out = {"window_passes": [T_BURN, T_PASSES], "n_seeds": T_SEEDS}
    x = np.arange(T_PASSES + 1, dtype=np.float64)
    errs_large, errs_small = [], []
    for (n, cs, seeds, passes), L in res["typical"].items():
        row = {}
        L = L.reshape(passes + 1, len(cs), seeds)
        for a, c in enumerate(cs):
            ex = [float(-np.polyfit(x[T_BURN:], L[T_BURN:, a, s], 1)[0]) for s in range(seeds)]
            per_seed = -(L[-1, a] - L[T_BURN, a]) / (passes - T_BURN)
            heur = order_model.iid_typical_heuristic(n, c)
            cell = dict(spread(ex), pooled=float(per_seed.mean()), pooled_se=float(per_seed.std(ddof=1) / np.sqrt(seeds)),
                        expectation=order_model.iid_exponent(n, c), rr=order_model.rr_ms_exponent(n, c),
                        heuristic=heur, g=c / n)
            cell["rr_over_typical"] = cell["rr"] / cell["median"]
            row[fk(c)] = cell
            if c >= 8 and c / n <= 0.05:
                errs_large.append(abs(heur - cell["median"]))
            elif c <= 4:
                errs_small.append(abs(heur - cell["median"]))
        out[f"n={n}"] = row
    out["heuristic_error"] = {"c_ge_8": {"max": float(np.max(errs_large)), "median": float(np.median(errs_large)), "n": len(errs_large)},
                              "c_le_4": {"max": float(np.max(errs_small)), "median": float(np.median(errs_small)), "n": len(errs_small)}}
    return out


# ----------------------------------------------------------------------------------------------
# (G) fixed g
# ----------------------------------------------------------------------------------------------
def analyse_fixed_g():
    out = {"crossing": 2.0 / (1.0 + np.e ** 2), "by_g": {}}
    for g in G_GS:
        row = {"iid_limit": order_model.iid_exponent_fixed_g(g), "rr_limit": order_model.rr_exponent_fixed_g(g),
               "fluctuation_factor": order_model.fluctuation_factor(g), "iid_exact": {}, "rr_exact": {}, "rr_mean": {}}
        for n in G_NS:
            row["iid_exact"][str(n)] = order_model.iid_exponent(n, n * g)
            row["rr_exact"][str(n)] = order_model.rr_ms_exponent(n, n * g)
            row["rr_mean"][str(n)] = order_model.rr_exponent(n * g, n=n) if n <= 10000 else None
        out["by_g"][fk(g)] = row
    out["rr_best"] = {}
    for n in G_BEST_NS:
        r = minimize_scalar(lambda c: -order_model.rr_ms_exponent(n, c), bounds=(1.0, float(n)), method="bounded",
                            options={"xatol": 1e-3})
        out["rr_best"][str(n)] = {"c": float(r.x), "g": float(r.x) / n, "exponent": float(-r.fun)}
    return out


# ----------------------------------------------------------------------------------------------
# (F) logistic regression: effective curvature and the two-term law
# ----------------------------------------------------------------------------------------------
def analyse_logistic():
    B4 = json.load(open(common.result_path("B4_quick_summary" if QUICK else "B4_summary", "json")))
    logi = B4["logistic"]
    out, rows = {}, []
    for key, P in logi["problems"].items():
        prob = common.logistic_problem(F_N, F_D, P["kappa"], P["alpha"], seed=0)
        u = prob.y * (prob.X @ prob.theta_star)
        h = 1.0 / (2.0 + np.exp(u) + np.exp(-u))                       # second derivative of log(1 + exp(-u))
        L_eff = order_model.effective_curvature_glm(prob.X, h, prob.lam)
        Li = h * np.einsum("ij,ij->i", prob.X, prob.X) + prob.lam
        e = {"L_eff": L_eff, "L_eff_over_R2": L_eff / prob.R2, "L_eff_over_mean_Li": L_eff / float(Li.mean()),
             "L_eff_over_lmax": L_eff / P["hess_star_max"], "by_step": {}}
        for ck, cfg in logi["configs"].items():
            pk, o, gk = ck.rsplit("|", 2)
            if pk != key or o != "rr":
                continue
            gR2 = float(gk.split("=")[1])
            g = gR2 * L_eff / prob.R2
            cap = float(np.log((2.0 - g) / g)) if g < 2.0 else None
            law = order_model.rr_exponent(cfg["c_min"])
            two = min(law, cap) if cap is not None else None
            meas = cfg["exponent"]["median"] if cfg["all_converged"] else None
            iid = logi["configs"][f"{pk}|iid|{gk}"]
            cell = {"g": g, "cap": cap, "law": law, "two_term": two, "measured": meas, "c_min": cfg["c_min"],
                    "rel_err_law": meas / law - 1.0 if meas is not None else None,
                    "rel_err_two_term": meas / two - 1.0 if (meas is not None and two is not None and two > 0) else None,
                    "iid_measured": iid["exponent"]["median"] if iid["all_converged"] else None,
                    "iid_expectation_heuristic": 1.0 - g / (2.0 - g) if g < 2.0 else None}
            e["by_step"][gk.split("=")[1]] = cell
            if cell["rel_err_two_term"] is not None:
                rows.append((P["kappa"], gR2, cell["rel_err_law"], cell["rel_err_two_term"]))
        # step at which the two terms are equal, against the best step of the grid
        n_lmin = P["n_lmin_over_R2"]

        def gap(gR2):
            g = gR2 * L_eff / prob.R2
            return order_model.rr_exponent(gR2 * n_lmin) - np.log((2.0 - g) / g)
        hi = 1.999 * prob.R2 / L_eff
        if gap(1e-4) < 0 < gap(hi):
            gs = float(brentq(gap, 1e-4, hi, xtol=1e-10))
            head = logi["headline"][key]["rr_best"]
            e["best_step"] = {"gammaR2_two_term": gs, "exponent_two_term": order_model.rr_exponent(gs * n_lmin),
                              "gammaR2_grid": head["gammaR2"] if head else None,
                              "exponent_grid": head["exponent"]["median"] if head else None}
        out[key] = e
    summ = {}
    for name, sel in (("gammaR2_le_half", lambda k, g: g <= 0.5), ("gammaR2_ge_1", lambda k, g: g >= 1.0),
                      ("gammaR2_ge_1_kappa_le_100", lambda k, g: g >= 1.0 and k <= 100.0),
                      ("gammaR2_ge_1_kappa_ge_1000", lambda k, g: g >= 1.0 and k >= 1000.0), ("all", lambda k, g: True)):
        r = [x for x in rows if sel(x[0], x[1])]
        if r:
            summ[name] = {"n_pairs": len(r),
                          "law": {"max_abs_rel_err": float(max(abs(x[2]) for x in r)), "median_abs_rel_err": float(np.median([abs(x[2]) for x in r])),
                                  "n_within_10pct": sum(abs(x[2]) <= 0.10 for x in r)},
                          "two_term": {"max_abs_rel_err": float(max(abs(x[3]) for x in r)), "median_abs_rel_err": float(np.median([abs(x[3]) for x in r])),
                                       "n_within_10pct": sum(abs(x[3]) <= 0.10 for x in r)}}
    out["summary"] = summ
    return out


# ----------------------------------------------------------------------------------------------
# (O) orthogonal components
# ----------------------------------------------------------------------------------------------
def orth_task(task):
    c, order, seed = task
    try:
        return order_model.orth_exponent(c, order, n=O_N, passes=O_PASSES, seed=seed)
    except FloatingPointError:
        return float("nan")


def run_orth(res, pmap):
    tasks = [(c, o, s) for c in O_CS for o in ("rr", "iid") for s in O_SEEDS]
    todo = [t for t in tasks if t not in res["orth"]]
    if todo:
        with common.Timer(f"orthogonal model: {len(todo)} runs"):
            for t, r in zip(todo, pmap(orth_task, todo)):
                res["orth"][t] = r
        common.save_result(TAG, res)


def analyse_orth(res):
    out = {"n": O_N, "passes": O_PASSES, "cyclic_threshold_limit": 2.0 / 3.0, "by_c": {}}
    for c in O_CS:
        out["by_c"][fk(c)] = {"cyclic": order_model.orth_exponent(c, "cyclic", n=O_N),
                              "cyclic_limit": float(-2.0 * np.log(np.abs(np.linalg.eigvals(order_model.orth_cyclic_matrix(c))).max())),
                              "rr": spread([res["orth"][(c, "rr", s)] for s in O_SEEDS]),
                              "iid": spread([res["orth"][(c, "iid", s)] for s in O_SEEDS])}
    best = max(O_CS, key=lambda c: out["by_c"][fk(c)]["cyclic_limit"])
    out["cyclic_best"] = {"c": best, "exponent": out["by_c"][fk(best)]["cyclic_limit"], "exact_value_at_one_half": float(np.log(2.0))}
    return out


# ----------------------------------------------------------------------------------------------
# (N) numbers of the note
# ----------------------------------------------------------------------------------------------
def note_numbers():
    om = order_model
    N = {}
    # --- (R1)
    n, c = 1000, 2.0
    Tn = np.linalg.matrix_power(om.iid_moment_matrix(n, c / n), n)
    mom, two = np.ones(4), {}
    for u in range(1, 13):
        mom = Tn @ mom
        if u in (2, 8, 12):
            approx = np.exp(-2 * c * u) + om.iid_plateau(c) / n * np.exp(-u)
            two[str(u)] = {"exact": float(mom[0]), "two_term": float(approx), "exact_over_two_term": float(mom[0] / approx)}
    N["R1_two_term_n1000_c2"] = two
    N["R1_u_cross_n1000_c2"] = float(np.log(n * (2 * c - 1) ** 2 / (2 * c ** 3)) / (2 * c - 1))
    N["R1_n_times_amplitude_n20000"] = {fk(c): {"exact": 20000 * om.iid_slow_amplitude(20000, c), "limit": om.iid_plateau(c)}
                                        for c in (1.0, 2.0, 4.0, 16.0)}
    N["R1_exponent_first_order_n2000"] = {fk(c): {"exact": om.iid_exponent(2000, c), "first_order": 1.0 - (c * c / (2 * c - 1) - 0.5) / 2000}
                                          for c in (1.0, 2.0)}
    N["R1_fixed_g"] = {fk(g): {"limit": om.iid_exponent_fixed_g(g), "exact_n1e5": om.iid_exponent(100000, 100000 * g)} for g in (0.25, 0.5, 1.1)}
    # --- (R2)
    omega, c_star = om.cyclic_critical()
    h = lambda w: w * np.sin(0.5 * w) + np.cos(0.5 * w)
    crossings = [float(brentq(h, (2 * k - 1) * np.pi, 2 * k * np.pi, xtol=1e-14)) for k in (1, 2, 3)]
    N["R2_omega_star"], N["R2_c_star"] = omega, c_star
    N["R2_crossings_c"] = [0.5 * w * w for w in crossings]
    N["R2_slope_at_c_star"] = 2 * c_star * (2 * c_star - 1) / ((2 * c_star ** 2 + 3 * c_star) ** 2 + 8 * c_star)
    N["R2_exponent_limit"] = {fk(c): om.cyclic_exponent_limit(c) for c in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 18.0, 20.0, 30.0, 50.0, 100.0, 400.0)}
    grid = np.linspace(17.8, 400.0, 400 if QUICK else 3823)
    vals = np.array([om.cyclic_exponent_limit(c) for c in grid])
    k = int(np.argmin(vals))
    N["R2_fastest_growth_on_17.8_400"] = {"c": float(grid[k]), "exponent": float(vals[k]), "grid_points": int(grid.size)}
    roots = [z for z in om.cyclic_roots(1000.0, re_min=-1.0, re_max=1.0, im_max=10.0) if 5.5 < z.imag < 7.0]
    N["R2_large_c_root_near_2pi_c1000"] = {"real_part": float(max(roots, key=lambda z: z.real).real) if roots else None,
                                           "expansion": 4 * np.pi ** 2 / (1000.0 * (1 + 4 * np.pi ** 2))}
    N["R2_K"] = c_star * (2 * c_star ** 2 + 3 * c_star + 4) / (2 * c_star - 1)
    ns_thr = [50, 100] if QUICK else [50, 100, 200, 400, 800, 1000]
    thr = {str(n): float(brentq(lambda c: om.cyclic_radius(n, c / n) - 1.0, 8.0, 30.0, xtol=1e-9, rtol=1e-12)) for n in ns_thr}
    N["R2_threshold_finite_n"] = {n: {"c": v, "n_times_gap": int(n) * (c_star - v)} for n, v in thr.items()}
    rad = {n: om.cyclic_radius(n, 0.25) for n in list(range(2, 121)) + ([] if QUICK else [200, 1000])}
    unstable = sorted(n for n, r in rad.items() if r > 1.0)
    N["R2_textbook_step_g_quarter"] = {"smallest_unstable_n": unstable[0], "all_unstable_from_it_to_120": unstable[:unstable.index(120) + 1] == list(range(unstable[0], 121)),
                                       "stable_n": [n for n, r in rad.items() if r < 1.0][-1],
                                       "growth_nats_per_pass": {str(n): float(2 * n * np.log(rad[n])) for n in (100, 120, 200, 1000) if n in rad}}
    ex = {}
    for seed in (0, 1, 2):
        prob = om.identical_hessian_problem(100, [0.02, 0.3, 1.0], seed=seed)
        row = {}
        for order, passes in (("cyclic", 300), ("so", 300), ("iid", 100), ("rr", 100)):
            r = algorithms.saga(prob, np.zeros(prob.d), passes=passes, seed=seed, record_every=100 * prob.n, record_F=False, sampling=order)
            row[order] = {"D0": float(r["D"][0]), "D_100_passes": float(r["D"][1]), "D_end": float(r["D"][-1]), "passes": passes}
        ex[str(seed)] = row
    N["R2_example_n100_eigs_0.02_0.3_1"] = ex
    N["IAG_threshold"] = {"limit": om.iag_critical(), "n=100": om.iag_critical(100), "n=1600": om.iag_critical(1600),
                          "n=100_companion": float(brentq(lambda c: om.iag_radius(100, c / 100) - 1.0, 3.0, 6.0, xtol=1e-12))}
    # --- (R3)
    N["R3_exponent"] = {fk(c): om.rr_exponent(c) for c in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)}
    N["R3_crossing_r_equals_1"] = float(brentq(lambda c: om.rr_exponent(c) - 1.0, 0.3, 2.0, xtol=1e-12))
    cg = np.exp(np.linspace(np.log(1e-3), np.log(1e5), 4001))
    rg = np.array([om.rr_exponent(c) for c in cg])
    N["R3_grid_4001"] = {"increasing": bool(np.all(np.diff(rg) > 0)), "max_ratio_to_min_2c_2": float(np.max(rg / np.minimum(2 * cg, 2.0)))}
    N["R3_large_c"] = {"1e3": 1e3 * (2 - om.rr_exponent(1e3)), "1e4": 1e4 * (2 - om.rr_exponent(1e4)), "limit": 2 * (np.e - 2)}
    n, c, p = 200, 4.0, 6
    mean6 = float((np.linalg.matrix_power(om.rr_pass_matrix(n, c / n), p) @ np.ones(2))[0])
    ms6 = float((np.linalg.matrix_power(om.rr_moment_matrix(n, c / n), p) @ np.ones(4))[0])
    N["R3_n200_c4_after_6_passes"] = {"mean": mean6, "root_mean_square": float(np.sqrt(ms6)), "rms_over_mean": float(np.sqrt(ms6) / abs(mean6))}
    N["R3_mean_square_vs_mean"] = {f"n={n}|c={fk(c)}": {"mean_square": om.rr_ms_exponent(n, c), "squared_mean": om.rr_exponent(c, n=n), "fluid": om.rr_exponent(c)}
                                   for n, c in ((100, 32.0), (1000, 1.0), (1000, 4.0), (1000, 16.0), (1000, 32.0))}
    N["R3_fixed_g"] = {fk(g): {"limit": om.rr_exponent_fixed_g(g), "exact_n1e4": om.rr_ms_exponent(10000, 10000 * g)} for g in (0.2, 0.25, 0.5, 1.25)}
    # --- lower bounds on the instance (S), n = 256
    lb = {}
    for ratio in (0.1, 1.0 / 3.0):
        Q, q = bounds.hard_instance_constants(256, ratio * 256)
        w = 1.0 - q * q
        lb[fk(ratio)] = {"q": q, "floor_iid_nats_per_pass": float(-256 * np.log(1.0 - w / 256)), "floor_any_nats_per_pass": float(2 * np.log(1.0 / q)),
                         "ratio_of_floors": float(2 * np.log(1.0 / q) / w), "half_one_minus_q2": 0.5 * w}
    N["floors_n256"] = lb
    N["log_32_times_256"] = float(np.log(32 * 256))
    # --- orthogonal extreme
    N["orth_cyclic_matrix_limit_c_two_thirds"] = [float(x) for x in sorted(np.linalg.eigvals(om.orth_cyclic_matrix(2.0 / 3.0)).real)]
    N["orth_rr_mean_map_threshold"] = float(18.0 - np.sqrt(300.0))
    return N


# ----------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------
def plot(summary):
    fig = ps.figure(ps.FIG_W, 7.4)
    gs = fig.add_gridspec(4, 2, height_ratios=[1.0, 0.02, 1.0, 0.02])
    mk = dict(markersize=5.5, markeredgecolor=ps.INK["surface"], markeredgewidth=0.8)
    BL, OR, AQ = ps.BLUE, ps.ORANGE, ps.AQUA

    # (a) fixed g
    ax = fig.add_subplot(gs[0, 0])
    gg = np.linspace(0.02, 1.3, 300)
    ax.plot(gg, [order_model.rr_exponent_fixed_g(g) for g in gg], color=OR, label="reshuffled: min(2, log((2 − g)/g))")
    ax.plot(gg, [order_model.iid_exponent_fixed_g(g) for g in gg], color=BL, label="i.i.d., expectation: 2(1 − g)/(2 − g)")
    n_ref = str(G_NS[1] if len(G_NS) > 1 else G_NS[0])
    fx = summary["fixed_g"]["by_g"]
    ax.plot(G_GS, [fx[fk(g)]["rr_exact"][n_ref] for g in G_GS], linestyle="none", marker="o", color=OR,
            label=f"reshuffled: exact second moments, n = {n_ref}", **mk)
    ax.plot(G_GS, [fx[fk(g)]["iid_exact"][n_ref] for g in G_GS], linestyle="none", marker="s", color=BL,
            label=f"i.i.d.: exact second moments, n = {n_ref}", **mk)
    ax.axhline(0.0, color=ps.INK["axis"], linewidth=0.8)
    ax.axvline(0.25, color=ps.INK["axis"], linewidth=0.8)
    ax.annotate("textbook step\non the stiffest mode", (0.25, -0.5), xytext=(4, 0), textcoords="offset points", fontsize=ps.SMALL,
                color=ps.INK["secondary"], va="center")
    ax.set_ylim(-0.75, 2.15)
    ax.set_xlabel("g = γ λ   (n → ∞ at fixed g)")
    ax.set_ylabel("exponent per pass of E e²")
    ax.set_title("model: a mode at a fixed step")
    h, l = ax.get_legend_handles_labels()
    ps.legend_cell(fig, gs[1, 0], h, l, ncol=1)

    # (b) typical i.i.d. runs against n
    ax = fig.add_subplot(gs[0, 1])
    ty = summary["typical"]
    ramp = [ps.BLUE_RAMP[k] for k in (3, 6, 9, 12)]
    for c, col in zip((1.0, 4.0, 16.0, 32.0), ramp):
        ns = [n for n in T_NS if fk(c) in ty[f"n={n}"]]
        ax.plot(ns, [ty[f"n={n}"][fk(c)]["median"] for n in ns], marker="o", color=col, label=f"i.i.d., typical run, c = {c:g}", **mk)
        ax.plot(ns, [ty[f"n={n}"][fk(c)]["heuristic"] for n in ns], color=col, linestyle=(0, (1, 2)), linewidth=1.2)
    ax.plot([], [], color=ps.INK["secondary"], linestyle=(0, (1, 2)), linewidth=1.2, label="heuristic δ log(n/c) = 2 log(2/δ)")
    ns = [n for n in T_NS if n >= 1000] or T_NS
    ax.plot(ns, [ty[f"n={n}"][fk(32.0)]["rr"] for n in ns], color=OR, marker="o", label="reshuffled, exact, c = 32", **mk)
    ax.axhline(1.0, color=BL, linestyle=(0, (4, 2)), linewidth=1.2, label="i.i.d., expectation (limit)")
    ax.set_xscale("log")
    ax.set_ylim(0.8, 2.1)
    ax.set_xlabel("n")
    ax.set_ylabel("exponent per pass (median over seeds)")
    ax.set_title("model: typical runs under i.i.d. sampling")
    h, l = ax.get_legend_handles_labels()
    ps.legend_cell(fig, gs[1, 1], h, l, ncol=1)

    # (c) logistic regression: measured reshuffled exponent against the two predictions
    ax = fig.add_subplot(gs[2, 0])
    lg = summary["logistic"]
    pts = {True: [], False: []}
    for key, e in lg.items():
        if key == "summary":
            continue
        for gk, cell in e["by_step"].items():
            if cell["measured"] is not None and cell["two_term"] is not None:
                pts[float(gk) <= 0.5].append((cell["law"], cell["two_term"], cell["measured"]))
    for small, lab in ((True, "γR² ≤ 1/2"), (False, "γR² ≥ 1")):
        v = pts[small]
        if not v:
            continue
        ax.plot([p[0] for p in v], [p[2] for p in v], linestyle="none", marker="o", markersize=5, markerfacecolor=ps.INK["surface"],
                markeredgecolor=ps.INK["muted"], markeredgewidth=1.1, label=f"law (R3) alone, {lab}" if not small else None)
        ax.plot([p[1] for p in v], [p[2] for p in v], linestyle="none", marker="o" if small else "D", color=OR if small else AQ,
                label=f"two-term law, {lab}", **mk)
    ax.plot([0, 2.05], [0, 2.05], color=ps.INK["axis"], linewidth=0.8)
    ax.set_xlim(0, 2.1)
    ax.set_ylim(0, 2.1)
    ax.set_xlabel("predicted exponent per pass")
    ax.set_ylabel("measured, reshuffled SAGA\n(median over seeds)")
    ax.set_title(f"logistic regression, n = {F_N}")
    h, l = ax.get_legend_handles_labels()
    ps.legend_cell(fig, gs[3, 0], h, l, ncol=1)

    # (d) orthogonal extreme
    ax = fig.add_subplot(gs[2, 1])
    ob = summary["orthogonal"]["by_c"]
    cc = np.linspace(0.02, 1.0, 200)
    ax.plot(cc, [-2.0 * np.log(np.abs(np.linalg.eigvals(order_model.orth_cyclic_matrix(c))).max()) for c in cc], color=AQ,
            label="fixed order (exact, n → ∞)")
    ax.plot(O_CS, [ob[fk(c)]["rr"]["median"] for c in O_CS], linestyle="none", marker="o", color=OR, label="reshuffled, typical run", **mk)
    ax.plot(O_CS, [ob[fk(c)]["iid"]["median"] for c in O_CS], linestyle="none", marker="s", color=BL, label="i.i.d., typical run", **mk)
    ax.axhline(0.0, color=ps.INK["axis"], linewidth=0.8)
    ax.axvline(2.0 / 3.0, color=ps.INK["axis"], linewidth=0.8)
    ax.annotate("2/3", (2.0 / 3.0, 0.75), xytext=(-3, 0), textcoords="offset points", ha="right", fontsize=ps.SMALL, color=ps.INK["secondary"])
    ax.set_ylim(-1.0, 0.8)
    ax.set_xlabel("c = γ n λ = γ R²")
    ax.set_ylabel("exponent per pass")
    ax.set_title("other extreme: orthogonal components")
    h, l = ax.get_legend_handles_labels()
    ps.legend_cell(fig, gs[3, 1], h, l, ncol=1)
    ps.suptitle(fig, "Figure B4b — the two regimes of the model, typical runs, the two-term law\nand the orthogonal extreme")
    ps.savefig(fig, "figB4b_refinements")


def main():
    ps.setup()
    res = common.load_result(TAG)
    if res is None or res.get("config") != CONFIG:
        res = {"config": CONFIG, "typical": {}, "orth": {}}
    else:
        print("loaded cached", TAG)
    pool = Pool(WORKERS) if WORKERS > 1 else None
    pmap = (lambda f, tasks: pool.imap(f, tasks, chunksize=1)) if pool else (lambda f, tasks: map(f, tasks))
    try:
        run_orth(res, pmap)
        run_typical(res, pmap)
    finally:
        if pool:
            pool.close()
            pool.join()
    with common.Timer("fixed g, logistic regression, numbers of the note"):
        summary = {"definitions": DEFINITIONS, "config": {k: (v if not isinstance(v, dict) else {str(a): b for a, b in v.items()}) for k, v in CONFIG.items()},
                   "fixed_g": analyse_fixed_g(), "typical": analyse_typical(res), "logistic": analyse_logistic(),
                   "orthogonal": analyse_orth(res), "note": note_numbers()}
    summary = clean(summary)
    plot(summary)
    common.save_json(TAG + "_summary", summary)
    print("done B4b")


if __name__ == "__main__":
    main()
