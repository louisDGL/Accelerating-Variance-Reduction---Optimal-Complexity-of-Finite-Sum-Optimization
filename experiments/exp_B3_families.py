"""Hypothesis B3 on other data designs: the measurements of Section 5.3 of the report (exp_B3_tightness.py), repeated
unchanged on the synthetic designs of Na, Derezinski and Mahoney (Math. Program. 201, 2023, Section 5; port
vrlab.problems.make_logistic_ndm) and on two further data sets of our own generator.

Protocol (that of exp_B3_tightness.py): l2-regularized logistic regression, n = 1000, d = 50,
kappa in {10, 1e2, 1e3, 1e4, 1e5}; eta_* by vrlab.solvers.solve_optimum (default, corrected line search); SAGA with the
textbook constants gamma = 1/(4R^2), Delta = 3 gamma^2, theta_0 = 0, 100 passes, records every n/20 iterations, exact
instrumentation every n/10 iterations, stop at ||theta_t - eta_*||^2 <= 1e-26. Summaries (function b3_summary, a
transcription of exp_B3_tightness.py checked against results/B3_summary.json): passes 2-20, records with V_t >= 1e-18,
speed-up, order-B and order-A shares, geometric means of the ratios, exponents; late window 20-100.
Designs (one data set per configuration; kappa only sets lambda):
  P  make_logistic_ndm(1000, 50, kappa, kap, high_coherence) for the low- and high-coherence families, kap in {0.5, 1}
     (hat Sigma diagonal with condition number d**(2 kap) = 50 and 2500), data seed 0, max_row_norm = 1, SAGA seeds 0-4;
  L  as P with kap = 1, but with labels as informative as ours: the labels of make_logistic_ndm(..., max_row_norm = c)
     (drawn from sigmoid(c x_i^T w), w = meta["planted"]) on the features and lambda of P, c chosen by bisection so that
     the Bayes accuracy mean_i max(p_i, 1 - p_i), p_i = sigmoid(c x_i^T w), equals 1/2 + arctan(2)/pi = 0.852, that of
     the labels of make_logistic (snr = 2, noise = 1); (c X, c^2 lambda) with the step 1/(4R^2) is the same SAGA up to
     theta -> theta / c (checked on one configuration), and this rescaling keeps the floors 1e-18 and 1e-26 of P;
  R  as P with kap = 1 and data seed 1, SAGA seeds 0-4;
  C  our generator (problems.make_logistic and solvers.solve_optimum, i.e. common.logistic_problem as in
     exp_B3_tightness.py), alpha in {0, 1, 2}, data seeds 1 and 2, SAGA seeds 0-2 (data seed 0 is the B3 experiment
     itself: results/B3_summary.json and the cache results/B3.pkl, summarized again here with the same code).
Static geometry per problem: max_i ||x_i||^2 / mean_i ||x_i||^2, normalized coherence (n/d) max_i h_i (leverage scores
at lambda = 0), cond(hat Sigma), stable rank tr(hat Sigma) / lambda_max(hat Sigma), mean_i L_i / R^2,
lambda_min(F''(eta_*)) / lambda, Bayes accuracy of the labels, the weighted smoothness of each eigenvector v_j of
hat Sigma, ell_j = sum_i U_ij^2 ||x_i||^2 / max_i ||x_i||^2 (X = U S V^T thin SVD: the value of L_eff / R^2 along v_j if
every sample had the curvature 1/4 of a zero margin and lambda were neglected), and the limits Lambda(u) / R^2 of
L_eff / R^2 at eta_* along the eigenvectors u of F''(eta_*) (Proposition B2-limits); residual of eta_* in 60-digit
arithmetic (vrlab.highprec). Derived quantities quoted in Section 5.3 (groups, comparisons) are written to the summary.
Checks: (i) b3_summary reproduces results/B3_summary.json from results/B3.pkl (skipped if the cache is absent);
(ii) the Bayes accuracy of the labels of make_logistic (replay of its random stream); (iii) the scale equivalence of L.
BLAS: the reported results were computed with OPENBLAS_NUM_THREADS=1 (with a multithreaded BLAS, make_logistic gives
features that differ in their last bits). No code of the companion implementation is used.
Usage: OPENBLAS_NUM_THREADS=1 python experiments/exp_B3_families.py [--quick] [--procs 3]
(290 SAGA runs: 1131 s of wall-clock time with --procs 3 on a 4-core laptop shared with other jobs.)
Outputs: results/B3_families.pkl (all runs; resumable: configurations already cached are not recomputed),
results/B3_families_summary.json.
"""
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from vrlab import algorithms, analysis, problems, solvers  # noqa: E402

QUICK = "--quick" in sys.argv
PROCS = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 3
N, D = (300, 20) if QUICK else (1000, 50)
KAPPAS = [10.0, 1000.0] if QUICK else [10.0, 100.0, 1000.0, 10000.0, 100000.0]
ALPHAS = [0.0, 2.0] if QUICK else [0.0, 1.0, 2.0]
KAPS = [0.5, 1.0]
SEEDS_NDM = list(range(2 if QUICK else 5))
SEEDS_C = list(range(2 if QUICK else 3))
DSEEDS_C = [1, 2]
PASSES = 30 if QUICK else 100
TAG = "B3_families_quick" if QUICK else "B3_families"
B3_TAG = "B3_quick" if QUICK else "B3"
V_MIN = 1e-18
KEYS = ["I1", "I3", "I2", "I4", "I5"]
RATIO_KEYS = ["r_sq", "r_coco", "r_var", "r_sc", "r_min", "mu_eff", "L_eff", "rho_exact", "rho_bound"]
BAYES_TARGET = 0.5 + float(np.arctan(2.0)) / np.pi      # = 1 - arctan(1/2)/pi: labels of make_logistic, snr 2, noise 1


# ----------------------------------------------------------------------------------------------------------------------
# summaries: transcription of exp_B3_tightness.py (fig3 with window (2, 20) and the per-configuration block of main)
# ----------------------------------------------------------------------------------------------------------------------
def _ratio_curves(runs, key):
    T = min(len(r["instr"]) for r in runs)
    t = np.array([q["t"] for q in runs[0]["instr"][:T]])
    X = np.array([[q[key] for q in r["instr"][:T]] for r in runs])
    return t, X


def b3_summary(runs, n, passes, window=(2, 20)):
    """The dict of results/B3_summary.json["attribution"][key], computed as in exp_B3_tightness.py."""
    sh, sp = [], []
    for r in runs:
        att = analysis.attribution(r["instr"], r["gamma"], r["Delta"], r["R2"], r["mu"], n,
                                   t_min=window[0] * n, t_max=window[1] * n, v_min=1e-18)
        sh.append(att["shares"])
        sp.append(att["speedup_total"])
    sh = np.array(sh)
    shm = np.nanmean(sh, 0)
    out = {"shares": dict(zip(KEYS, map(float, shm))), "speedup": float(np.nanmean(sp)),
           "shares_min_over_seeds": dict(zip(KEYS, map(float, np.nanmin(sh, 0)))),
           "shares_max_over_seeds": dict(zip(KEYS, map(float, np.nanmax(sh, 0))))}
    d = {}
    tV, XV = _ratio_curves(runs, "V")
    for rk in RATIO_KEYS:
        t, X = _ratio_curves(runs, rk)
        sel = (t >= 2 * n) & (t <= 20 * n)
        ok = (XV[:, sel] >= 1e-18)
        vals = np.log(np.clip(X[:, sel], 1e-300, None))[ok]
        d[rk + "_geomean"] = float(np.exp(vals.mean())) if vals.size else float("nan")
    d["r_init"] = float(np.mean([r["r_init"] for r in runs]))
    d["final_D"] = float(np.mean([r["D"][-1] for r in runs]))
    late = [analysis.attribution(r["instr"], r["gamma"], r["Delta"], r["R2"], r["mu"], n, t_min=20 * n, t_max=passes * n,
                                 v_min=1e-18) for r in runs if len(r["instr"]) > 0]
    late = [a for a in late if np.all(np.isfinite(a["shares"])) and len(a["t"]) >= 10]
    if late:
        d["shares_late_20_100"] = dict(zip(KEYS, map(float, np.mean([a["shares"] for a in late], 0))))
        d["speedup_late_20_100"] = float(np.mean([a["speedup_total"] for a in late]))
    early = [analysis.attribution(r["instr"], r["gamma"], r["Delta"], r["R2"], r["mu"], n, t_min=2 * n, t_max=20 * n,
                                  v_min=1e-18) for r in runs]
    d["loo_exponents_per_iter"] = {k: float(np.mean([a["loo_exponents"][k] for a in early])) for k in KEYS}
    orderA = [analysis.attribution_A(r["instr"], r["gamma"], r["Delta"], r["R2"], r["mu"], n, t_min=2 * n, t_max=20 * n,
                                     v_min=1e-18) for r in runs]
    d["shares_orderA"] = dict(zip(analysis.STEP_KEYS_A, map(float, np.nanmean([a["shares"] for a in orderA], 0))))
    d["c_exact_per_iter"] = float(np.mean([a["c_exact"] for a in early]))
    d["c_bound_per_iter"] = float(np.mean([a["c_bound"] for a in early]))
    out.update(d)
    return out


def summary_extras(runs, n, mu, window=(2, 20)):
    """Quantities not recorded by exp_B3_tightness.py: per-seed speed-ups and order-A shares, records kept in the
    window, mu_eff/mu, and the product of the measured factors (mu_eff/mu) (1 - r_coco (r_sq/4 + 3/8)) / (3/8) of the
    paragraph 'Interpretation' of Section 5.3 (window geometric means of the ratios)."""
    sp, shA, kept = [], [], []
    for r in runs:
        att = analysis.attribution(r["instr"], r["gamma"], r["Delta"], r["R2"], r["mu"], n,
                                   t_min=window[0] * n, t_max=window[1] * n, v_min=V_MIN)
        attA = analysis.attribution_A(r["instr"], r["gamma"], r["Delta"], r["R2"], r["mu"], n,
                                      t_min=window[0] * n, t_max=window[1] * n, v_min=V_MIN)
        sp.append(float(att["speedup_total"]))
        shA.append([float(x) for x in attA["shares"]])
        kept.append(int(len(att["t"])))
    shA = np.array(shA)
    return {"speedup_per_seed": sp, "speedup_min_over_seeds": float(np.nanmin(sp)),
            "speedup_max_over_seeds": float(np.nanmax(sp)),
            "shares_orderA_min_over_seeds": dict(zip(analysis.STEP_KEYS_A, map(float, np.nanmin(shA, 0)))),
            "shares_orderA_max_over_seeds": dict(zip(analysis.STEP_KEYS_A, map(float, np.nanmax(shA, 0)))),
            "window_records_kept_per_seed": kept, "mu": float(mu), "passes_run_per_seed": [float(r["iters"][-1]) / n for r in runs]}


def add_products(s):
    """mu_eff/mu, the I1-I2 factor (1 - r_coco (r_sq/4 + 3/8)) / (3/8) and their product over the speed-up."""
    mu_ratio = 1.0 / s["r_sc_geomean"]                    # = geometric mean of mu_eff/mu over the same records
    fac = (1.0 - s["r_coco_geomean"] * (s["r_sq_geomean"] / 4.0 + 3.0 / 8.0)) / (3.0 / 8.0)
    s["mu_eff_over_mu_geomean"] = float(mu_ratio)
    s["I1I2_factor"] = float(fac)
    s["product_over_speedup"] = float(mu_ratio * fac / s["speedup"])
    return s


# ----------------------------------------------------------------------------------------------------------------------
# problems
# ----------------------------------------------------------------------------------------------------------------------
def ours_replay(n, d, alpha, seed, snr=2.0, noise=1.0, R=1.0):
    """Replay of the random stream of problems.make_logistic (power spectrum, rotated): features, scaled logits and
    labels, to compute the Bayes accuracy of its labels (it does not store the planted vector)."""
    rng = np.random.default_rng(seed)
    ev = problems.covariance_spectrum(d, alpha, "power")
    G = rng.standard_normal((n, d))
    X = G * np.sqrt(ev)[None, :]
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    X = X @ Q.T
    X *= R / np.sqrt(np.einsum("ij,ij->i", X, X).max())
    w = rng.standard_normal(d)
    logits = X @ w
    logits *= snr / logits.std()
    y = np.sign(logits + noise * rng.standard_normal(n))
    y[y == 0] = 1.0
    return X, logits, y


def ours_bayes(n, d, alpha, seed, prob=None, noise=1.0):
    """Bayes accuracy mean_i Phi(|l_i| / noise) of the labels y_i = sign(l_i + noise eps_i) of make_logistic, with a
    check that the replay reproduces its features and labels exactly."""
    from scipy.special import ndtr
    X, logits, y = ours_replay(n, d, alpha, seed, noise=noise)
    out = {"bayes_acc": float(np.mean(ndtr(np.abs(logits) / noise))),
           "label_agreement": float(np.mean(np.sign(logits) == y))}
    if prob is not None:
        out["replay_exact"] = bool(np.array_equal(X, prob.X) and np.array_equal(y, prob.y))
    return out


def ndm_bayes(X, w):
    p = problems.sigmoid(X @ w)
    return float(np.mean(np.maximum(p, 1.0 - p)))


def label_scale(kap, high_coherence, data_seed, target=BAYES_TARGET):
    """Scale c such that the labels of make_logistic_ndm(..., max_row_norm = c), drawn from sigmoid(c x_i^T w) with
    the rows x_i of the max_row_norm = 1 design, have Bayes accuracy `target` (bisection on log c; the accuracy
    mean_i sigmoid(c |x_i^T w|) is increasing in c)."""
    p1 = problems.make_logistic_ndm(N, D, 10.0, kap=kap, high_coherence=high_coherence, seed=data_seed)
    z = np.abs(p1.X @ p1.meta["planted"])
    lo, hi = 1e-3, 1e6
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if np.mean(problems.sigmoid(mid * z)) < target:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def make_problem(cfg, solve=True):
    """Problem of a configuration (eta_* by solvers.solve_optimum unless solve=False)."""
    if cfg["design"] == "C":
        prob = problems.make_logistic(N, D, kappa=cfg["kappa"], alpha=cfg["alpha"], seed=cfg["data_seed"])
        planted = None
    else:
        hc = cfg["family"] == "hc"
        prob = problems.make_logistic_ndm(N, D, cfg["kappa"], kap=cfg["kap"], high_coherence=hc, seed=cfg["data_seed"])
        planted = prob.meta["planted"]
        if cfg.get("label_scale") is not None:
            c = cfg["label_scale"]
            pc = problems.make_logistic_ndm(N, D, cfg["kappa"], kap=cfg["kap"], high_coherence=hc, seed=cfg["data_seed"],
                                            max_row_norm=c)
            meta = dict(prob.meta, label_scale=c, planted=c * planted)
            prob = problems.LogisticProblem(prob.X, pc.y, prob.lam, name=prob.name + f"-labels(c={c:.6g})")
            prob.meta = meta
            planted = meta["planted"]
    if solve:
        solvers.solve_optimum(prob)
    return prob, planted


def static_geometry(prob):
    X, n, d = prob.X, prob.n, prob.d
    sq = prob.sqnorms
    lev = prob.leverage_scores(lam=0.0)                  # n h_i with h_i = x_i^T (X^T X)^{-1} x_i
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    ell = (U ** 2 * sq[:, None]).sum(0) / sq.max()       # direction-weighted smoothness of each right singular vector
    ev = prob.cov_eigs
    return {"max_over_mean_sqnorm": float(sq.max() / sq.mean()), "coherence": float(lev.max() / d),
            "cov_cond": float(ev[-1] / ev[0]), "stable_rank": float(ev.sum() / ev[-1]),
            "mean_L_over_R2": float(prob.L_i.mean() / prob.R2),
            "dir_smoothness_median": float(np.median(ell)), "dir_smoothness_min": float(ell.min()),
            "dir_smoothness_max": float(ell.max()), "dir_smoothness_at_smallest_sv": float(ell[np.argmin(s)])}


def optimum_geometry(prob):
    """lambda_min(F''(eta_*))/lambda and the limits Lambda(u) = u^T M_* u / u^T H_* u of L_eff at eta_* (Proposition
    B2-limits, M_* = (1/n) sum_i f_i''(eta_*)^2), divided by R^2: along the bottom eigenvector of H_* = F''(eta_*), the
    median over its eigenvectors, and the extreme values over all directions (generalized eigenvalues)."""
    from scipy.linalg import eigh
    th = prob.theta_star
    w = prob.sigma_prime(th)
    lam = prob.lam
    H = prob.hessian(th)
    h, V = np.linalg.eigh(H)
    coef = w ** 2 * prob.sqnorms + 2.0 * lam * w
    Ms = (prob.X.T * coef) @ prob.X / prob.n + lam ** 2 * np.eye(prob.d)
    Lam = np.einsum("ij,ik,kj->j", V, Ms, V) / h
    gen = eigh(Ms, H, eigvals_only=True)
    return {"hess_min_over_lam": float(h[0] / lam), "Lambda_bottom_over_R2": float(Lam[0] / prob.R2),
            "Lambda_median_over_R2": float(np.median(Lam) / prob.R2), "Lambda_min_over_R2": float(gen[0] / prob.R2),
            "Lambda_max_over_R2": float(gen[-1] / prob.R2), "sigma_prime_mean": float(w.mean())}


def optimum_precision(prob, dps=60):
    """Residual ||F'(eta_*)|| re-evaluated in `dps` digits on the stored float64 data, the rigorous bound
    (r/mu)^2 on the squared error of eta_* and the first-order estimate ||F''^{-1} F'||^2."""
    from vrlab import highprec
    e1, r = highprec.newton_correction(prob, prob.theta_star, dps=dps)
    return {"grad_norm_float64": float(prob.opt_info["grad_norm"]), "newton_iters": int(prob.opt_info["iters"]),
            "grad_norm_mp": float(r), "err_sq_bound": float((r / prob.mu) ** 2), "err_sq_estimate": float(e1 @ e1)}


# ----------------------------------------------------------------------------------------------------------------------
# configurations and worker
# ----------------------------------------------------------------------------------------------------------------------
def config_key(c):
    if c["design"] == "C":
        return f"C,alpha={c['alpha']:g},dseed={c['data_seed']},kappa={c['kappa']:g}"
    return f"{c['design']},{c['family']},kap={c['kap']:g},dseed={c['data_seed']},kappa={c['kappa']:g}"


def group_key(c):
    return config_key(c).rsplit(",kappa=", 1)[0]


def configs(scales):
    out = []
    for fam in ("lc", "hc"):
        for kap in KAPS:
            out += [dict(design="P", family=fam, kap=kap, data_seed=0, kappa=k, seeds=SEEDS_NDM) for k in KAPPAS]
        out += [dict(design="L", family=fam, kap=1.0, data_seed=0, kappa=k, seeds=SEEDS_NDM, label_scale=scales[fam])
                for k in KAPPAS]
        out += [dict(design="R", family=fam, kap=1.0, data_seed=1, kappa=k, seeds=SEEDS_NDM) for k in KAPPAS]
    for ds in DSEEDS_C:
        for a in ALPHAS:
            out += [dict(design="C", alpha=a, data_seed=ds, kappa=k, seeds=SEEDS_C) for k in KAPPAS]
    out.sort(key=lambda c: -c["kappa"])                  # the runs of large kappa do not stop early: start with them
    return out


def run_config(cfg):
    t0 = time.time()
    prob, planted = make_problem(cfg)
    geo = static_geometry(prob)
    geo.update(optimum_geometry(prob))
    if cfg["design"] == "C":
        geo.update(ours_bayes(N, D, cfg["alpha"], cfg["data_seed"], prob))
    else:
        geo["bayes_acc"] = ndm_bayes(prob.X, planted)
        geo["label_agreement"] = float(np.mean(np.sign(prob.X @ planted) == prob.y))
    geo.update(optimum_precision(prob))
    runs = []
    for s in cfg["seeds"]:
        r = algorithms.saga(prob, np.zeros(D), passes=PASSES, seed=s, record_every=max(1, N // 20),
                            instrument_every=max(1, N // 10), stop_D=1e-26)
        runs.append(common.strip_run(r))
    summ = b3_summary(runs, N, PASSES)
    summ.update(summary_extras(runs, N, prob.mu))
    add_products(summ)
    msg = (f"[{config_key(cfg)}] {time.time() - t0:.1f}s  speed-up {summ['speedup']:.3g}  final D "
           + " ".join(f"{r['D'][-1]:.1e}" for r in runs))
    return config_key(cfg), {"cfg": cfg, "problem": prob.summary(), "geometry": geo, "summary": summ, "runs": runs}, msg


# ----------------------------------------------------------------------------------------------------------------------
# checks
# ----------------------------------------------------------------------------------------------------------------------
def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, prefix + k + "."))
        else:
            out[prefix + k] = v
    return out


def self_check_b3():
    """b3_summary on the cached runs of exp_B3_tightness.py against results/B3_summary.json (every field). Returns the
    check and the summaries of the cached runs with summary_extras (per-seed speed-ups of data seed 0), used for the
    data-seed comparison (identical to B3_summary.json when the check passes)."""
    res = common.load_result(B3_TAG)
    path = common.result_path(B3_TAG + "_summary", "json")
    if res is None or not os.path.exists(path):
        print("self-check skipped: no cache", B3_TAG)
        return {"done": False}, {}
    with open(path, encoding="utf-8") as f:
        ref = json.load(f)["attribution"]
    n, passes = res["config"]["n"], res["config"]["passes"]
    worst, nfields, ndiff, mismatch, summ = 0.0, 0, 0, [], {}
    for (kappa, alpha), runs in res["runs"].items():
        key = f"kappa={kappa:g},alpha={alpha:g}"
        s = b3_summary(runs, n, passes)
        mine, theirs = _flat(s), _flat(ref[key])
        if set(mine) != set(theirs):
            mismatch.append(key)
        for k, v in theirs.items():
            a, b = float(mine.get(k, np.nan)), float(v)
            nfields += 1
            if not (a == b or (np.isnan(a) and np.isnan(b))):
                ndiff += 1
                worst = max(worst, abs(a - b) / max(abs(a), abs(b), 1e-300))
        s.update(summary_extras(runs, n, runs[0]["mu"]))
        summ[key] = s
    out = {"done": True, "configurations": len(res["runs"]), "fields_compared": nfields, "fields_differing": ndiff,
           "max_rel_diff": worst, "key_sets_differ": mismatch, "reproduced_exactly": bool(ndiff == 0 and not mismatch)}
    print(f"self-check: b3_summary reproduces {B3_TAG}_summary.json: {out['reproduced_exactly']} "
          f"({nfields} fields, {ndiff} differ, max rel. diff {worst:.1e}, key sets differ: {mismatch})", flush=True)
    return out, summ


def scale_check(scale, kappa=1000.0, family="lc", passes=20, seed=0):
    """Design L: SAGA on make_logistic_ndm(..., max_row_norm = c) itself against SAGA on the design used (features and
    lambda of max_row_norm = 1, same labels): theta -> theta / c, D, M -> D / c^2, c^2 M, Delta -> Delta / c^4."""
    hc = family == "hc"
    lit = problems.make_logistic_ndm(N, D, kappa, kap=1.0, high_coherence=hc, seed=0, max_row_norm=scale)
    solvers.solve_optimum(lit)
    used, _ = make_problem(dict(design="L", family=family, kap=1.0, data_seed=0, kappa=kappa, label_scale=scale))
    kw = dict(passes=passes, seed=seed, record_every=max(1, N // 20), instrument_every=max(1, N // 10), stop_D=1e-26)
    ra = algorithms.saga(lit, np.zeros(D), **kw)
    rb = algorithms.saga(used, np.zeros(D), **kw)
    c2 = scale ** 2

    def rel(a, b):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))) if a.size else float("nan")

    keep = np.asarray(rb["V"]) >= V_MIN
    out = {"family": family, "kappa": kappa, "passes": passes, "label_scale": scale,
           "labels_equal": bool(np.array_equal(lit.y, used.y)),
           "R2_rel": rel(lit.R2, c2 * used.R2), "lam_rel": rel(lit.lam, c2 * used.lam), "kappa_rel": rel(lit.kappa, used.kappa),
           "eta_rel": rel(scale * lit.theta_star, used.theta_star),
           "V_rel": rel(c2 * np.asarray(ra["V"])[keep], np.asarray(rb["V"])[keep]),
           "D_rel": rel(c2 * np.asarray(ra["D"])[keep], np.asarray(rb["D"])[keep])}
    qa = [q for q in ra["instr"] if c2 * q["V"] >= V_MIN]
    qb = [q for q in rb["instr"] if q["V"] >= V_MIN]
    m = min(len(qa), len(qb))
    for k in ["rho_exact", "r_sq", "r_coco", "r_var", "r_sc", "r_min"]:
        out[k + "_rel"] = rel([q[k] for q in qa[:m]], [q[k] for q in qb[:m]])
    aa = analysis.attribution(ra["instr"], ra["gamma"], ra["Delta"], ra["R2"], ra["mu"], N, t_min=2 * N, t_max=20 * N,
                              v_min=V_MIN / c2)
    ab = analysis.attribution(rb["instr"], rb["gamma"], rb["Delta"], rb["R2"], rb["mu"], N, t_min=2 * N, t_max=20 * N,
                              v_min=V_MIN)
    out["shares_absdiff"] = float(np.max(np.abs(aa["shares"] - ab["shares"])))
    out["speedup_rel"] = rel(aa["speedup_total"], ab["speedup_total"])
    out["records_compared"] = m
    print("scale check (design L):", {k: (f"{v:.1e}" if isinstance(v, float) else v) for k, v in out.items()}, flush=True)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# derived quantities quoted in the report
# ----------------------------------------------------------------------------------------------------------------------
def _rng(vals):
    vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
    return [min(vals), max(vals)] if vals else [float("nan"), float("nan")]


def group_stats(by_kappa, geo_by_kappa):
    """Per design and family (or alpha and data seed): the statements of Section 5.3 evaluated on that group."""
    ks = sorted(by_kappa)
    lo = [k for k in ks if k <= 100.0]
    mid = [k for k in ks if k >= 1000.0]
    hi = [k for k in ks if k >= 10000.0]
    s = by_kappa
    g = {"speedup": {f"{k:g}": s[k]["speedup"] for k in ks},
         "speedup_seed_range": {f"{k:g}": [s[k].get("speedup_min_over_seeds", np.nan), s[k].get("speedup_max_over_seeds", np.nan)] for k in ks},
         "speedup_le_1e2": _rng(s[k]["speedup"] for k in lo),
         "r_coco": {f"{k:g}": s[k]["r_coco_geomean"] for k in ks},
         "r_coco_ge_1e3": _rng(s[k]["r_coco_geomean"] for k in mid),
         "r_coco_ge_1e4": _rng(s[k]["r_coco_geomean"] for k in hi),
         "r_coco_all": _rng(s[k]["r_coco_geomean"] for k in ks),
         "r_sq_le_1e2": _rng(s[k]["r_sq_geomean"] for k in lo),
         "r_sq": {f"{k:g}": s[k]["r_sq_geomean"] for k in ks},
         "r_var_min": min(s[k]["r_var_geomean"] for k in ks), "r_min_min": min(s[k]["r_min_geomean"] for k in ks),
         "r_sc": {f"{k:g}": s[k]["r_sc_geomean"] for k in ks},
         "mu_eff_over_mu": {f"{k:g}": s[k]["mu_eff_over_mu_geomean"] for k in ks},
         "hess_min_over_lam": {f"{k:g}": geo_by_kappa[k]["hess_min_over_lam"] for k in ks if k in geo_by_kappa},
         "max_share_B": {i: max(s[k]["shares"][i] for k in ks) for i in KEYS},
         "max_share_B_single_seed": {i: max(s[k]["shares_max_over_seeds"][i] for k in ks) for i in KEYS},
         "max_share_A": {i: max(s[k]["shares_orderA"][i] for k in ks) for i in KEYS},
         "shares_B": {f"{k:g}": s[k]["shares"] for k in ks},
         "shares_A": {f"{k:g}": s[k]["shares_orderA"] for k in ks}}
    if lo:
        g["shares_B_le_1e2"] = {i: _rng(s[k]["shares"][i] for k in lo) for i in KEYS}
        g["shares_A_le_1e2"] = {i: _rng(s[k]["shares_orderA"][i] for k in lo) for i in KEYS}
        g["shares_A_le_1e2"]["I2+I4"] = _rng(s[k]["shares_orderA"]["I2"] + s[k]["shares_orderA"]["I4"] for k in lo)
    if hi:
        g["shares_B_ge_1e4"] = {i: _rng(s[k]["shares"][i] for k in hi) for i in KEYS}
        g["shares_B_ge_1e4"]["I2+I4"] = _rng(s[k]["shares"]["I2"] + s[k]["shares"]["I4"] for k in hi)
        g["shares_A_ge_1e4"] = {i: _rng(s[k]["shares_orderA"][i] for k in hi) for i in KEYS}
        g["shares_A_ge_1e4"]["I2+I4"] = _rng(s[k]["shares_orderA"]["I2"] + s[k]["shares_orderA"]["I4"] for k in hi)
        g["shares_seed_spread_ge_1e4"] = max(s[k]["shares_max_over_seeds"][i] - s[k]["shares_min_over_seeds"][i]
                                             for k in hi for i in KEYS)
        g["product_over_speedup_ge_1e4"] = _rng(s[k]["product_over_speedup"] for k in hi)
        g["I1I2_factor_ge_1e4"] = _rng(s[k]["I1I2_factor"] for k in hi)
        g["mu_eff_over_mu_ge_1e4"] = _rng(s[k]["mu_eff_over_mu_geomean"] for k in hi)
        g["I2_largest_ge_1e4"] = bool(all(s[k]["shares"]["I2"] == max(s[k]["shares"].values()) for k in hi))
    if 10000.0 in s and 100000.0 in s:
        g["growth_exponent_1e4_1e5"] = float(np.log10(s[100000.0]["speedup"] / s[10000.0]["speedup"]))
    kref = 10000.0 if 10000.0 in geo_by_kappa else ks[-1]
    g["geometry"] = {k: geo_by_kappa[kref][k] for k in
                     ["max_over_mean_sqnorm", "coherence", "cov_cond", "stable_rank", "mean_L_over_R2", "bayes_acc",
                      "dir_smoothness_median", "dir_smoothness_min", "dir_smoothness_at_smallest_sv"]}
    g["geometry"]["at_kappa"] = kref
    g["Lambda_bottom_over_R2"] = {f"{k:g}": geo_by_kappa[k]["Lambda_bottom_over_R2"] for k in ks if k in geo_by_kappa}
    g["Lambda_median_over_R2"] = {f"{k:g}": geo_by_kappa[k]["Lambda_median_over_R2"] for k in ks if k in geo_by_kappa}
    return g


def derive(res, b3_ref, b3_cache):
    """Groups, coherence and data-seed comparisons, and grid-wide statements."""
    by_group, geo_group = {}, {}
    for key, v in res["configs"].items():
        gk = group_key(v["cfg"])
        by_group.setdefault(gk, {})[v["cfg"]["kappa"]] = v["summary"]
        geo_group.setdefault(gk, {})[v["cfg"]["kappa"]] = v["geometry"]
    # our generator, data seed 0 (exp_B3_tightness.py): summaries of the cached runs (equal to B3_summary.json when
    # the self-check passes; B3_summary.json itself if the cache is absent), geometry recomputed here
    if b3_ref is not None:
        for a in ALPHAS:
            gk = f"C,alpha={a:g},dseed=0"
            for k in KAPPAS:
                key = f"kappa={k:g},alpha={a:g}"
                s = dict(b3_cache[key]) if key in b3_cache else dict(b3_ref["attribution"][key])
                add_products(s)
                by_group.setdefault(gk, {})[k] = s
                prob, _ = make_problem(dict(design="C", alpha=a, data_seed=0, kappa=k))   # rebuilt and re-solved here
                geo = static_geometry(prob)
                geo.update(optimum_geometry(prob))
                geo.update(ours_bayes(N, D, a, 0, prob))
                geo["hess_min_over_lam_recomputed"] = geo["hess_min_over_lam"]
                geo["hess_min_over_lam"] = b3_ref["problems"][key]["mu_local_over_mu"]   # value of the cached eta_*
                geo_group.setdefault(gk, {})[k] = geo
    groups = {gk: group_stats(by_group[gk], geo_group[gk]) for gk in sorted(by_group)}
    comp = {}
    # high versus low coherence (same design, kap, data seed)
    coh = {}
    for gk in groups:
        if ",lc," in gk and gk.replace(",lc,", ",hc,") in groups:
            a, b = by_group[gk], by_group[gk.replace(",lc,", ",hc,")]
            ks = sorted(set(a) & set(b))
            coh[gk.replace(",lc,", ",")] = {
                "r_coco_hc_over_lc": {f"{k:g}": b[k]["r_coco_geomean"] / a[k]["r_coco_geomean"] for k in ks},
                "I2_hc_minus_lc": {f"{k:g}": b[k]["shares"]["I2"] - a[k]["shares"]["I2"] for k in ks},
                "I1_hc_minus_lc": {f"{k:g}": b[k]["shares"]["I1"] - a[k]["shares"]["I1"] for k in ks},
                "speedup_hc_over_lc": {f"{k:g}": b[k]["speedup"] / a[k]["speedup"] for k in ks}}
    allr = [x for v in coh.values() for x in v["r_coco_hc_over_lc"].values()]
    alli = [x for v in coh.values() for x in v["I2_hc_minus_lc"].values()]
    comp["coherence"] = {"pairs": coh, "r_coco_hc_over_lc_range": _rng(allr), "I2_hc_minus_lc_range": _rng(alli),
                         "r_coco_lower_with_hc_everywhere": bool(all(x < 1 for x in allr)),
                         "I2_higher_with_hc_everywhere": bool(all(x > 0 for x in alli))}
    # data-seed variability of our generator (data seed 0: 5 SAGA seeds, B3; data seeds 1, 2: SAGA seeds 0-2)
    dsv, growth = {}, {}
    for a in ALPHAS:
        gks = [f"C,alpha={a:g},dseed={ds}" for ds in [0] + DSEEDS_C]
        if not all(g in by_group for g in gks):
            continue
        for k in KAPPAS:
            ss = [by_group[g][k] for g in gks]
            sp = np.array([x["speedup"] for x in ss])
            seed_spread = [(x["speedup_max_over_seeds"] - x["speedup_min_over_seeds"]) / x["speedup"]
                           for x in ss if "speedup_max_over_seeds" in x]
            dsv[(a, k)] = {
                "speedup_per_data_seed": sp.tolist(), "speedup_rel_spread": float((sp.max() - sp.min()) / sp.mean()),
                "saga_seed_rel_spread_max": float(max(seed_spread)) if seed_spread else float("nan"),
                "shares_per_data_seed": {i: [x["shares"][i] for x in ss] for i in KEYS},
                "shares_range": {i: float(max(x["shares"][i] for x in ss) - min(x["shares"][i] for x in ss)) for i in KEYS},
                "shares_A_range": {i: float(max(x["shares_orderA"][i] for x in ss) - min(x["shares_orderA"][i] for x in ss))
                                   for i in KEYS},
                "r_coco_per_data_seed": [x["r_coco_geomean"] for x in ss],
                "r_sc_per_data_seed": [x["r_sc_geomean"] for x in ss]}
        growth[f"alpha={a:g}"] = [groups[g].get("growth_exponent_1e4_1e5", float("nan")) for g in gks]
    comp["data_seed"] = {"per_config": {f"alpha={a:g},kappa={k:g}": v for (a, k), v in dsv.items()},
                         "growth_exponent_1e4_1e5_per_data_seed": growth}
    for name, pred in [("le_1e2", lambda k: k <= 100.0), ("1e3", lambda k: k == 1000.0), ("ge_1e4", lambda k: k >= 10000.0),
                       ("1e5", lambda k: k == 100000.0), ("all", lambda k: True)]:
        vs = [v for (a, k), v in dsv.items() if pred(k)]
        if vs:
            comp["data_seed"][name] = {
                "speedup_rel_spread": _rng(v["speedup_rel_spread"] for v in vs),
                "saga_seed_rel_spread_max": max(v["saga_seed_rel_spread_max"] for v in vs),
                "shares_range_max": max(max(v["shares_range"].values()) for v in vs),
                "shares_A_range_max": max(max(v["shares_A_range"].values()) for v in vs)}
    # label signal (L against P, kap = 1) and data seed of the designs of Na et al. (R against P, kap = 1)
    for name, other in [("label_signal", "L"), ("ndm_data_seed", "R")]:
        comp[name] = {}
        for fam in ("lc", "hc"):
            gp, go = f"P,{fam},kap=1,dseed=0", f"{other},{fam},kap=1,dseed={1 if other == 'R' else 0}"
            if gp not in by_group or go not in by_group:
                continue
            a, b = by_group[gp], by_group[go]
            ks = sorted(set(a) & set(b))
            comp[name][fam] = {
                "speedup_ratio": {f"{k:g}": b[k]["speedup"] / a[k]["speedup"] for k in ks},
                "shares_absdiff_max": {f"{k:g}": max(abs(b[k]["shares"][i] - a[k]["shares"][i]) for i in KEYS) for k in ks},
                "mu_eff_over_mu": {f"{k:g}": [a[k]["mu_eff_over_mu_geomean"], b[k]["mu_eff_over_mu_geomean"]] for k in ks},
                "share_order_B_same": {f"{k:g}": bool(sorted(KEYS, key=lambda i: -a[k]["shares"][i])
                                                     == sorted(KEYS, key=lambda i: -b[k]["shares"][i])) for k in ks}}
    # grid-wide statements on the new data (P, L, R, and C with data seeds 1, 2)
    new = [gk for gk in groups if not gk.endswith("dseed=0") or not gk.startswith("C")]
    grid = {}
    for name, gks in [("ndm", [g for g in new if not g.startswith("C")]), ("C_new", [g for g in new if g.startswith("C")]),
                      ("all_new", new)]:
        if not gks:
            continue
        G = [groups[g] for g in gks]
        grid[name] = {
            "groups": gks,
            "speedup_le_1e2": _rng(x for g in G for x in g["speedup_le_1e2"]),
            "max_share_B": {i: max(g["max_share_B"][i] for g in G) for i in KEYS},
            "max_share_B_single_seed": {i: max(g["max_share_B_single_seed"][i] for g in G) for i in KEYS},
            "max_share_A": {i: max(g["max_share_A"][i] for g in G) for i in KEYS},
            "r_var_min": min(g["r_var_min"] for g in G), "r_min_min": min(g["r_min_min"] for g in G),
            "r_sq_le_1e2": _rng(x for g in G for x in g.get("r_sq_le_1e2", [])),
            "shares_A_le_1e2": {i: _rng(x for g in G for x in g["shares_A_le_1e2"][i]) for i in ["I5", "I1", "I2+I4", "I3"]}
            if all("shares_A_le_1e2" in g for g in G) else {},
            "shares_B_le_1e2": {i: _rng(x for g in G for x in g["shares_B_le_1e2"][i]) for i in KEYS}
            if all("shares_B_le_1e2" in g for g in G) else {},
            "shares_B_ge_1e4_I2+I4": _rng(x for g in G for x in g.get("shares_B_ge_1e4", {}).get("I2+I4", [])),
            "shares_A_ge_1e4_I2+I4": _rng(x for g in G for x in g.get("shares_A_ge_1e4", {}).get("I2+I4", [])),
            "product_over_speedup_ge_1e4": _rng(x for g in G for x in g.get("product_over_speedup_ge_1e4", [])),
            "I1I2_factor_ge_1e4": _rng(x for g in G for x in g.get("I1I2_factor_ge_1e4", [])),
            "growth_exponent_1e4_1e5": _rng(g.get("growth_exponent_1e4_1e5", np.nan) for g in G),
            "speedup_grows_with_kappa_from_1e3": bool(all(
                g["speedup"][f"{a:g}"] < g["speedup"][f"{b:g}"] for g in G
                for a, b in zip([k for k in KAPPAS if k >= 1000.0][:-1], [k for k in KAPPAS if k >= 1000.0][1:])))}
    comp["grid"] = grid
    return groups, comp


# ----------------------------------------------------------------------------------------------------------------------
def main():
    t_start = time.time()
    res = common.load_result(TAG)
    if res is None:
        res = {"config": dict(n=N, d=D, kappas=KAPPAS, alphas=ALPHAS, kaps=KAPS, seeds_ndm=SEEDS_NDM, seeds_C=SEEDS_C,
                              dseeds_C=DSEEDS_C, passes=PASSES, v_min=V_MIN, stop_D=1e-26), "configs": {}}
    else:
        print("loaded cached", TAG, f"({len(res['configs'])} configurations)", flush=True)
    scales = {fam: label_scale(1.0, fam == "hc", 0) for fam in ("lc", "hc")}
    print("label scales (Bayes accuracy %.6f):" % BAYES_TARGET, scales, flush=True)
    todo = [c for c in configs(scales) if config_key(c) not in res["configs"]]
    print(f"{len(todo)} configurations to run with {PROCS} processes", flush=True)
    if todo:
        with mp.Pool(PROCS) as pool:
            for i, (key, out, msg) in enumerate(pool.imap_unordered(run_config, todo), 1):
                res["configs"][key] = out
                print(f"[{time.time() - t_start:7.1f}s] {i}/{len(todo)} {msg}", flush=True)
                if i % 10 == 0 or i == len(todo):
                    common.save_result(TAG, res)
    checks = {"bayes_target": BAYES_TARGET, "label_scales": scales}
    checks["label_scales_bayes_acc"] = {}
    for fam in ("lc", "hc"):
        pc = problems.make_logistic_ndm(N, D, 10.0, kap=1.0, high_coherence=fam == "hc", seed=0, max_row_norm=scales[fam])
        checks["label_scales_bayes_acc"][fam] = ndm_bayes(pc.X, pc.meta["planted"])
    checks["ours_bayes_dseed0"] = {}
    for a in ALPHAS:
        prob, _ = make_problem(dict(design="C", alpha=a, data_seed=0, kappa=KAPPAS[0]), solve=False)
        checks["ours_bayes_dseed0"][f"alpha={a:g}"] = ours_bayes(N, D, a, 0, prob)
    checks["scale_equivalence"] = scale_check(scales["lc"])
    geo = [v["geometry"] for v in res["configs"].values()]
    checks["optima"] = {"max_grad_norm_mp": max(g["grad_norm_mp"] for g in geo),
                        "max_err_sq_bound": max(g["err_sq_bound"] for g in geo),
                        "max_err_sq_estimate": max(g["err_sq_estimate"] for g in geo)}
    sc, b3_cache = self_check_b3()
    checks["self_check_B3_summary"] = sc
    b3_path = common.result_path(B3_TAG + "_summary", "json")
    b3_ref = None
    if os.path.exists(b3_path):
        with open(b3_path, encoding="utf-8") as f:
            b3_ref = json.load(f)
    groups, comp = derive(res, b3_ref, b3_cache)
    out = {"config": res["config"], "checks": checks,
           "problems": {k: dict(v["problem"], **v["geometry"]) for k, v in sorted(res["configs"].items())},
           "attribution": {k: v["summary"] for k, v in sorted(res["configs"].items())},
           "groups": groups, "comparisons": comp}
    common.save_json(TAG + "_summary", out)
    print(f"done {TAG} in {time.time() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
