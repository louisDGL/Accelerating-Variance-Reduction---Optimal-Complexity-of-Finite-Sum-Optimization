"""Certification of the logistic minimizers eta_* of Section 5.3 (exp_B3_tightness.py; Sections 5.4-5.8 reuse some of
these data sets): high-precision residuals, realized errors, impact of the reference on the analysis, and the rounding
floor of the double-precision SAGA iterates.

Setting: the 15 problems make_logistic(1000, 50, kappa, alpha, seed=0) (the call of common.logistic_problem), kappa in
{10, 1e2, 1e3, 1e4, 1e5}, alpha in {0, 1, 2}, with eta_* rebuilt by solvers.solve_optimum(prob, armijo_only=True), the
Newton solver of commit 451da40 with which every cached logistic result was computed.
  (a) Every field of prob.summary() is compared bit for bit with results/B3_summary.json["problems"].
  (b) Residual r = ||F'(eta_*)|| of the stored problem (X, y, lam and eta_* taken as their binary float64 values) in
      60-digit arithmetic (vrlab.highprec), checked at 100 digits on all 15 problems, and the two float64 residuals
      (prob.grad, and the mean of prob.all_grads stored as grad_star_norm). By strong convexity
      ||eta_* - eta_exact|| <= eps := r/mu and F(eta_*) - F(eta_exact) <= r^2/(2 mu).
      Error propagation to V_t = ||theta_t - eta_*||^2 + Delta sum_i ||z_i - f_i'(eta_*)||^2 (gamma = 1/(4R^2),
      Delta = 3 gamma^2, the constants of the runs): sqrt(V_t) is the Euclidean norm of the vector
      u(eta) = (theta_t - eta, sqrt(Delta)(z_1 - f_1'(eta)), ..., sqrt(Delta)(z_n - f_n'(eta))) of R^{d(n+1)}. Moving eta
      by delta changes its first block by ||delta|| and its block i by sqrt(Delta) ||f_i'(eta + delta) - f_i'(eta)||
      <= sqrt(Delta) L_i ||delta||, since f_i'' = sigma'(.) x_i x_i^T + lam I with 0 < sigma' <= 1/4 has norm at most
      L_i = ||x_i||^2/4 + lam. Hence ||u(eta + delta) - u(eta)|| <= c_V ||delta|| with c_V = (1 + Delta sum_i L_i^2)^{1/2},
      and |sqrt(V_t(eta + delta)) - sqrt(V_t(eta))| <= c_V ||delta|| (triangle inequality). A recorded value V_t = f
      therefore differs from the value for eta_exact by at most 2 c_V eps sqrt(f) + c_V^2 eps^2, i.e. by the relative
      amount 2/sqrt(m) + 1/m with the margin m = f/(c_V eps)^2; for ||theta_t - eta_*||^2 the same holds with c_V = 1.
      Floors: V_t = 1e-18 (Section 5.3) and ||theta_t - eta_*||^2 = 1e-18 (Section 5.5, nine of the problems).
      Which runs reach these floors is read from results/B3.pkl (instrumented V_t < 1e-18) and results/B2.pkl
      (recorded ||theta_t - eta_*||^2 < 1e-18 along SAGA or GD), if present.
  (c) Realized error: e1 = F''(eta_*)^{-1} F'(eta_*) with the 60-digit gradient (highprec.newton_correction) and the
      corrected minimizer eta_c = eta_* - e1 (float64), whose own 60-digit residual r_c certifies ||eta_c - eta_exact||
      <= r_c/mu, hence ||eta_* - eta_exact|| lies in [||eta_* - eta_c|| - r_c/mu, ||eta_* - eta_c|| + r_c/mu]. The
      realized amplification (first order) is (||e1||^2 + Delta sum_i ||f_i''(eta_*) e1||^2)^{1/2} / ||e1||. ||e1||^2 is
      compared with final_D of results/B3_summary.json at kappa = 10, alpha = 1 (the plateau of those runs).
  (d) Impact on the analysis: the five seeds of Section 5.3 at kappa = 10, alpha = 1 and at kappa = 1e3, alpha = 1, run
      with the call of exp_B3_tightness.run_all, first with the legacy eta_* (the statistics, computed by a copy of the
      summary code of exp_B3_tightness.fig3 and main, must reproduce results/B3_summary.json["attribution"], and the
      runs results/B3.pkl if present, bit for bit), then on a copy of the problem whose optimum is set to eta_c. The
      iterates of SAGA do not depend on eta_* (only the recorded and instrumented quantities and the stopping rule
      ||theta_t - eta_*||^2 <= 1e-26 do); this is checked on the instrumented ||F'(theta_t)||^2 and ||zbar||^2.
      Reported: the largest relative change of the instrumented V_t over the records with V_t >= 1e-18, of the
      statistics of passes 2-20 (geometric means of the ratios, shares in orders A and B, speed-up, exponents) and of
      the late window (passes 20-100). Also, on the legacy runs, the largest relative difference between the recorded
      V_t (memory error updated incrementally, re-summed at every instrumented state) and the exact V_t of the
      instrumented state at the same iteration, over the records with V_t >= 1e-18.
  (e) The current (corrected) solver, solve_optimum(prob): which eta_* differ from the cached ones, by how much, and
      their residuals.
  (f) Floor of the double-precision iterates: kappa = 1e3, alpha = 1, SAGA seed 0, 100 passes without the stopping
      rule, gamma R^2 in {1/8, 1/4, 1/2} (Delta = 3 gamma^2), ||theta_t - eta_c||^2 recorded every n/20 iterations,
      plus one run of 300 passes at gamma R^2 = 1/8 (not saturated after 100 passes). Plateau: level = median over the
      last 20 passes; saturated if the medians of the two halves of that window differ by less than 10%; onset = first
      recorded pass from which every record lies within a factor 1.1 of the level. gamma R^2 = 1/4 is the textbook
      step: that run follows the cached run of seed 0 (checked on its final distance to the legacy eta_*).
  (g) Memory error recorded without instrumentation: on the same problem (legacy eta_*), SAGA seed 0 with the textbook
      constants instrumented only at t = 0 and at t = T, for T in {20, 49, 99.95} passes: the record at T carries the
      incremental memory error accumulated since t = 0 (that of a run without instrumentation, whose iterates are the
      same), the instrumented state at T its exact re-summation.
This is careful high-precision evaluation, not interval arithmetic (see vrlab/highprec.py).
BLAS: run with the default (multithreaded) BLAS thread count, OPENBLAS_NUM_THREADS unset, as the cached runs of Section
5.3 were: with OPENBLAS_NUM_THREADS=1 the features differ in their last bits (the rotation X @ Q.T is a BLAS product)
and the cached eta_* are not reproduced. With --threads1 the script also runs (a), (b) and (e) in a subprocess with
OPENBLAS_NUM_THREADS=1 and records how the data, the minimizers and the bounds change.
Usage: python experiments/check_optima_mp.py [--threads1]     (needs mpmath; about ten minutes)
Output: results/optima_mp_summary.json (no cache).
"""
import copy
import glob
import json
import os
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from vrlab import algorithms, analysis, bounds, problems, solvers  # noqa: E402
from vrlab import highprec  # noqa: E402

N, D = 1000, 50
KAPPAS = [10.0, 100.0, 1000.0, 10000.0, 100000.0]
ALPHAS = [0.0, 1.0, 2.0]
B2_PROBLEMS = [(k, a) for k in (100.0, 1000.0, 10000.0) for a in ALPHAS]     # Section 5.5 (exp_B2_mu_eff.py)
DPS, DPS_CHECK = 60, 100
FLOOR = 1e-18                     # analysis floor of Sections 5.3 (on V_t) and 5.5 (on ||theta_t - eta_*||^2)
STOP_D = 1e-26                    # stopping rule of the runs of Section 5.3
SEEDS = list(range(5))
PASSES = 100
RERUN = [(10.0, 1.0), (1000.0, 1.0)]
FLOOR_PROBLEM = (1000.0, 1.0)
FLOOR_RUNS = [(1 / 8, 100), (1 / 4, 100), (1 / 2, 100), (1 / 8, 300)]       # (gamma R^2, passes)
MEMORY_CHECK_T = [20_000, 49_000, 99_950]                                    # (g): iterations

SHARES = ["I1", "I3", "I2", "I4", "I5"]
RATIO_KEYS = ["r_sq", "r_coco", "r_var", "r_sc", "r_min", "mu_eff", "L_eff", "rho_exact", "rho_bound"]
TAG = "optima_mp_summary"


def key(kappa, alpha):
    return f"kappa={kappa:g},alpha={alpha:g}"


def worst_rel(m):
    """Worst-case relative error of a recorded squared norm f whose square root is off by at most sqrt(f/m)."""
    return 2.0 / np.sqrt(m) + 1.0 / m


# ----------------------------------------------------------------------------------------------------------------------
# environment
# ----------------------------------------------------------------------------------------------------------------------
def blas_threads():
    """Number of OpenBLAS threads used by numpy (best effort, through the scipy-openblas library bundled with numpy)."""
    try:
        import ctypes
        for p in glob.glob(os.path.join(os.path.dirname(np.__file__), "..", "numpy.libs", "*openblas*")):
            lib = ctypes.CDLL(p)
            for name in ("scipy_openblas_get_num_threads64_", "openblas_get_num_threads64_", "openblas_get_num_threads"):
                f = getattr(lib, name, None)
                if f is not None:
                    return int(f())
    except Exception:  # pragma: no cover
        pass
    return None


def environment():
    import mpmath
    import scipy
    try:
        blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
    except Exception:  # pragma: no cover
        blas = {}
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "mpmath": mpmath.__version__, "mpmath_backend": mpmath.libmp.BACKEND, "platform": platform.platform(),
            "cpu_count": os.cpu_count(), "blas": blas.get("name"), "blas_version": blas.get("version"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "unset (default)"),
            "openblas_threads_in_use": blas_threads()}


# ----------------------------------------------------------------------------------------------------------------------
# comparisons
# ----------------------------------------------------------------------------------------------------------------------
def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, prefix + str(k) + "/"))
        else:
            out[prefix + str(k)] = v
    return out


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def rel_change(a, b):
    """|b - a| / |a| (0 if a == b, None if not comparable or a == 0 != b)."""
    a, b = _num(a), _num(b)
    if a is None or b is None:
        return None
    if a == b or (np.isnan(a) and np.isnan(b)):
        return 0.0
    if a == 0 or not (np.isfinite(a) and np.isfinite(b)):
        return None
    return abs(b - a) / abs(a)


def bitwise_compare(new, ref):
    """Bit-for-bit comparison of two (nested) dicts of numbers: differing fields with their relative difference."""
    fn, fr = flatten(new), flatten(ref)
    fields = sorted(set(fn) | set(fr))
    diff = {}
    for k in fields:
        if k not in fn or k not in fr:
            diff[k] = "missing"
            continue
        a, b = _num(fn[k]), _num(fr[k])
        if a is None or b is None:
            if fn[k] != fr[k]:
                diff[k] = "different"
        elif not (a == b or (np.isnan(a) and np.isnan(b))):
            diff[k] = abs(a - b) / max(abs(a), abs(b), 1e-300)
    return {"n_fields": len(fields), "bitwise": not diff, "differing_fields": diff}


# ----------------------------------------------------------------------------------------------------------------------
# (a), (b), (c), (e): the 15 problems
# ----------------------------------------------------------------------------------------------------------------------
def build(kappa, alpha):
    """The problem of common.logistic_problem(N, D, kappa, alpha, seed=0), solved with the solver of the cached runs."""
    prob = problems.make_logistic(N, D, kappa=kappa, alpha=alpha, seed=0)
    solvers.solve_optimum(prob, armijo_only=True)
    return prob


def with_optimum(prob, theta):
    """Copy of the problem (data shared) whose optimum is set to theta; SAGA's iterates do not depend on it."""
    p = copy.copy(prob)
    p.set_optimum(theta)
    return p


def bounds_from_residual(prob, r):
    """Strong-convexity bounds from the residual r and the worst-case relative errors at the floors (docstring)."""
    gamma, _ = bounds.saga_textbook_params(prob.R2)
    Delta = 3.0 * gamma ** 2
    eps = r / prob.mu
    cV = float(np.sqrt(1.0 + Delta * float(np.sum(prob.L_i ** 2))))
    mV = FLOOR / (cV * eps) ** 2
    mD = FLOOR / eps ** 2
    return {"err_bound": eps, "sqerr_bound": eps ** 2, "Fgap_bound": r ** 2 / (2.0 * prob.mu), "gamma": gamma,
            "Delta": Delta, "c_V": cV, "V_floor_margin": mV, "V_floor_worst_rel_err": worst_rel(mV),
            "D_floor_margin": mD, "D_floor_worst_rel_err": worst_rel(mD)}


def certify(prob, check_digits=True, realized=True):
    """(b) and (c) for the stored eta_* of prob; returns (record, eta_c)."""
    eta = prob.theta_star
    t0 = time.time()
    e1, r = highprec.newton_correction(prob, eta, DPS)
    rec = {"r64_grad": float(np.linalg.norm(prob.grad(eta))), "r64_stored": float(prob.grad_star_norm), "r_mp": r,
           "newton_iters": int(prob.opt_info["iters"]), "lam": prob.lam, "lmin_hess": float(np.linalg.eigvalsh(prob.hessian(eta))[0])}
    rec["r_mp_over_r64_grad"] = r / rec["r64_grad"]
    rec["r_mp_over_r64_stored"] = r / rec["r64_stored"]
    if check_digits:
        _, r_hi = highprec.mp_logistic_grad(prob.X, prob.y, prob.lam, eta, DPS_CHECK)
        rec["r_mp_100"] = r_hi
        rec["rel_diff_60_100_digits"] = abs(r - r_hi) / r_hi
    rec.update(bounds_from_residual(prob, r))
    eta_c = None
    if realized:
        eta_c = eta - e1
        _, r_c = highprec.mp_logistic_grad(prob.X, prob.y, prob.lam, eta_c, DPS)
        dist = float(np.linalg.norm(eta - eta_c))
        eps_c = r_c / prob.mu
        e1n = float(np.linalg.norm(e1))
        sp = prob.sigma_prime(eta)
        He = (sp * (prob.X @ e1))[:, None] * prob.X + prob.lam * e1[None, :]      # rows f_i''(eta_*) e1
        a = float(np.sqrt(e1n ** 2 + rec["Delta"] * float(np.einsum("ij,ij->", He, He))))
        lo, hi = max(dist - eps_c, 0.0), dist + eps_c
        rec["realized"] = {"e1_norm": e1n, "e1_sq": e1n ** 2, "dist_eta_to_eta_c": dist, "r_c": r_c, "err_bound_eta_c": eps_c,
                           "sqerr_bound_eta_c": eps_c ** 2, "err_low": lo, "err_high": hi, "sqerr_low": lo ** 2,
                           "sqerr_high": hi ** 2, "c_V_realized": a / e1n if e1n > 0 else float("nan"),
                           "V_floor_margin_realized": FLOOR / a ** 2, "V_floor_rel_err_realized": worst_rel(FLOOR / a ** 2),
                           "D_floor_rel_err_realized": worst_rel(FLOOR / e1n ** 2)}
    rec["seconds_mp"] = time.time() - t0
    return rec, eta_c


def current_solver(prob, eta_c=None):
    """(e) the corrected solver on the same data: does it change eta_*, by how much, with which residuals."""
    p = copy.copy(prob)
    solvers.solve_optimum(p)
    same = bool(np.array_equal(p.theta_star, prob.theta_star))
    rec = {"identical_to_cached": same, "newton_iters": int(p.opt_info["iters"]),
           "dist_to_cached": float(np.linalg.norm(p.theta_star - prob.theta_star)),
           "r64_grad": float(np.linalg.norm(p.grad(p.theta_star)))}
    if eta_c is not None:
        rec["dist_to_eta_c"] = float(np.linalg.norm(p.theta_star - eta_c))
    if not same:
        _, r = highprec.mp_logistic_grad(p.X, p.y, p.lam, p.theta_star, DPS)
        rec["r_mp"] = r
        rec.update({k: v for k, v in bounds_from_residual(p, r).items() if k in ("err_bound", "sqerr_bound", "Fgap_bound")})
        rec["summary_vs_B3_json"] = None             # filled by the caller
        rec["_prob"] = p
    return rec


def cached_floor_stats(cache):
    """From results/B3.pkl: per problem, smallest instrumented V_t, smallest kept V_t (>= FLOOR), number of discarded
    records, final D and stopping status of the five runs."""
    out = {}
    if cache is None:
        return out
    for (kappa, alpha), runs in cache["runs"].items():
        V = [np.array([q["V"] for q in r["instr"]]) for r in runs]
        kept = np.concatenate([v[v >= FLOOR] for v in V])
        out[key(kappa, alpha)] = {"min_V": float(min(v.min() for v in V)), "min_V_kept": float(kept.min()) if kept.size else None,
                                  "n_records_discarded": int(sum(int(np.sum(v < FLOOR)) for v in V)),
                                  "reaches_floor": bool(min(v.min() for v in V) < FLOOR),
                                  "final_D_per_seed": [float(r["D"][-1]) for r in runs],
                                  "passes_run_per_seed": [float(r["iters"][-1]) / N for r in runs],
                                  "stopped_per_seed": [r["stopped"] for r in runs]}
    return out


def cached_b2_floor_stats(cache):
    """From results/B2.pkl (exp_B2_mu_eff.py, keys (alpha, kappa)): per problem of Section 5.5, smallest recorded
    ||theta_t - eta_*||^2 along SAGA and GD, number of discarded states and smallest kept value."""
    out = {}
    if cache is None:
        return out
    for (alpha, kappa) in cache["saga"]:
        Ds = np.asarray(cache["saga"][(alpha, kappa)]["D"], dtype=np.float64)
        Dg = np.asarray(cache["gd"][(alpha, kappa)]["D"], dtype=np.float64)
        allD = np.concatenate([Ds, Dg])
        kept = allD[allD >= FLOOR]
        out[key(kappa, alpha)] = {"min_D_saga": float(Ds.min()), "min_D_gd": float(Dg.min()),
                                  "n_states_discarded": int(np.sum(allD < FLOOR)),
                                  "min_D_kept": float(kept.min()) if kept.size else None,
                                  "reaches_floor": bool(allD.min() < FLOOR)}
    return out


def part_abe(js, check_digits=True, realized=True, save_data_dir=None):
    """(a), (b), (c) and (e) on the 15 problems. Returns (records, problems, corrected minimizers)."""
    recs, probs, etas_c = {}, {}, {}
    for kappa in KAPPAS:
        for alpha in ALPHAS:
            k = key(kappa, alpha)
            t0 = time.time()
            prob = build(kappa, alpha)
            t_build = time.time() - t0
            rec = {"a_summary_vs_B3_json": bitwise_compare(prob.summary(), js["problems"][k])}
            b, eta_c = certify(prob, check_digits=check_digits, realized=realized)
            rec["b_c"] = b
            e = current_solver(prob, eta_c)
            p_new = e.pop("_prob", None)
            if p_new is not None:
                e["summary_vs_B3_json"] = {kk: v for kk, v in bitwise_compare(p_new.summary(), js["problems"][k]).items()
                                           if kk != "n_fields"}
            rec["e_current_solver"] = e
            rec["seconds"] = {"build_and_newton": t_build, "total": time.time() - t0}
            recs[k], probs[k], etas_c[k] = rec, prob, eta_c
            if save_data_dir is not None:
                np.save(os.path.join(save_data_dir, f"X_{k}.npy"), prob.X)
                np.save(os.path.join(save_data_dir, f"y_{k}.npy"), prob.y)
                np.save(os.path.join(save_data_dir, f"eta_{k}.npy"), prob.theta_star)
            print(f"{k}: summary bitwise {rec['a_summary_vs_B3_json']['bitwise']}; r64 {b['r64_grad']:.3e}, r({DPS} digits) "
                  f"{b['r_mp']:.4e}; (r/mu)^2 {b['sqerr_bound']:.3e}; c_V {b['c_V']:.2f}; worst rel. err. at V = 1e-18 "
                  f"{b['V_floor_worst_rel_err']:.2e}" + (f"; ||e1||^2 {b['realized']['e1_sq']:.4e}" if realized else "")
                  + f"; current solver identical: {e['identical_to_cached']} [{time.time() - t0:.1f}s]", flush=True)
    return recs, probs, etas_c


# ----------------------------------------------------------------------------------------------------------------------
# (d): impact of the reference on the statistics of Section 5.3
# ----------------------------------------------------------------------------------------------------------------------
def _ratio_curves(runs, k):
    """Verbatim copy of exp_B3_tightness._ratio_curves."""
    T = min(len(r["instr"]) for r in runs)
    t = np.array([q["t"] for q in runs[0]["instr"][:T]])
    X = np.array([[q[k] for q in r["instr"][:T]] for r in runs])
    return t, X


def b3_summary(runs, window=(2, 20)):
    """results/B3_summary.json["attribution"][key] for one configuration: the lines of exp_B3_tightness.fig3 (window
    passes 2-20) and of the per-configuration block of exp_B3_tightness.main, copied verbatim."""
    # ---- exp_B3_tightness.fig3
    sh = []
    sp = []
    for r in runs:
        att = analysis.attribution(r["instr"], r["gamma"], r["Delta"], r["R2"], r["mu"], N,
                                   t_min=window[0] * N, t_max=window[1] * N, v_min=1e-18)
        sh.append(att["shares"])
        sp.append(att["speedup_total"])
    sh = np.array(sh)
    shm = np.nanmean(sh, 0)
    out = {"shares": dict(zip(SHARES, map(float, shm))), "speedup": float(np.nanmean(sp)),
           "shares_min_over_seeds": dict(zip(SHARES, map(float, np.nanmin(sh, 0)))),
           "shares_max_over_seeds": dict(zip(SHARES, map(float, np.nanmax(sh, 0))))}
    # ---- exp_B3_tightness.main
    d = {}
    tV, XV = _ratio_curves(runs, "V")
    for rk in RATIO_KEYS:
        t, X = _ratio_curves(runs, rk)
        sel = (t >= 2 * N) & (t <= 20 * N)
        ok = (XV[:, sel] >= 1e-18)
        vals = np.log(np.clip(X[:, sel], 1e-300, None))[ok]
        d[rk + "_geomean"] = float(np.exp(vals.mean())) if vals.size else float("nan")
    d["r_init"] = float(np.mean([r["r_init"] for r in runs]))
    d["final_D"] = float(np.mean([r["D"][-1] for r in runs]))
    late = [analysis.attribution(r["instr"], r["gamma"], r["Delta"], r["R2"], r["mu"], N, t_min=20 * N, t_max=PASSES * N, v_min=1e-18) for r in runs if len(r["instr"]) > 0]
    late = [a for a in late if np.all(np.isfinite(a["shares"])) and len(a["t"]) >= 10]
    if late:
        d["shares_late_20_100"] = dict(zip(SHARES, map(float, np.mean([a["shares"] for a in late], 0))))
        d["speedup_late_20_100"] = float(np.mean([a["speedup_total"] for a in late]))
    early = [analysis.attribution(r["instr"], r["gamma"], r["Delta"], r["R2"], r["mu"], N, t_min=2 * N, t_max=20 * N, v_min=1e-18) for r in runs]
    d["loo_exponents_per_iter"] = {k: float(np.mean([a["loo_exponents"][k] for a in early])) for k in SHARES}
    orderA = [analysis.attribution_A(r["instr"], r["gamma"], r["Delta"], r["R2"], r["mu"], N, t_min=2 * N, t_max=20 * N, v_min=1e-18) for r in runs]
    d["shares_orderA"] = dict(zip(analysis.STEP_KEYS_A, map(float, np.nanmean([a["shares"] for a in orderA], 0))))
    d["c_exact_per_iter"] = float(np.mean([a["c_exact"] for a in early]))
    d["c_bound_per_iter"] = float(np.mean([a["c_bound"] for a in early]))
    out.update(d)
    return out


def _group(name):
    if name.startswith(("shares_late_20_100/", "speedup_late_20_100")):
        return "late_20_100"
    if name in ("r_init", "final_D"):
        return "other"
    return "window_2_20"


def statistics_change(s_old, s_new):
    """Largest changes of the statistics, by group: relative (over the nonzero legacy values) and, for the shares,
    absolute."""
    fo, fn = flatten(s_old), flatten(s_new)
    out = {}
    for g in ("window_2_20", "late_20_100", "other"):
        names = [k for k in fo if _group(k) == g and k in fn]
        rel = {k: rel_change(fo[k], fn[k]) for k in names}
        rel_ok = {k: v for k, v in rel.items() if v is not None}
        sh = [k for k in names if k.split("/")[0].startswith("shares")]
        absd = {k: abs(float(fn[k]) - float(fo[k])) for k in sh if np.isfinite(float(fo[k])) and np.isfinite(float(fn[k]))}
        kmax = max(rel_ok, key=rel_ok.get) if rel_ok else None
        kabs = max(absd, key=absd.get) if absd else None
        out[g] = {"n_statistics": len(names), "max_rel_change": rel_ok[kmax] if kmax else None, "argmax_rel": kmax,
                  "max_abs_change_shares": absd[kabs] if kabs else None, "argmax_abs_shares": kabs,
                  "not_comparable": sorted(k for k, v in rel.items() if v is None)}
    return out


def run_b3(prob, seed):
    """The call of exp_B3_tightness.run_all, then common.strip_run."""
    theta0 = np.zeros(D)
    r = algorithms.saga(prob, theta0, passes=PASSES, seed=seed, record_every=max(1, N // 20),
                        instrument_every=max(1, N // 10), stop_D=1e-26)
    return common.strip_run(r)


def part_d(prob, eta_c, js_attr, cached_runs):
    p_cor = with_optimum(prob, eta_c)
    runs = {"legacy": [], "corrected": []}
    t0 = time.time()
    for s in SEEDS:
        for lab, p in (("legacy", prob), ("corrected", p_cor)):
            runs[lab].append(run_b3(p, s))
        print(f"   seed {s}: legacy {runs['legacy'][-1]['iters'][-1] / N:g} passes ({runs['legacy'][-1]['stopped']}), "
              f"corrected {runs['corrected'][-1]['iters'][-1] / N:g} passes ({runs['corrected'][-1]['stopped']}) "
              f"[{time.time() - t0:.0f}s]", flush=True)
    s_leg, s_cor = b3_summary(runs["legacy"]), b3_summary(runs["corrected"])
    out = {"legacy_summary_vs_B3_json": bitwise_compare(s_leg, js_attr), "statistics_change": statistics_change(s_leg, s_cor),
           "summary_legacy": s_leg, "summary_corrected": s_cor, "per_seed": []}
    if cached_runs is not None:
        out["legacy_runs_vs_B3_pkl_bitwise"] = bool(all(
            np.array_equal(rl["iters"], rc["iters"]) and np.array_equal(rl["D"], rc["D"]) and np.array_equal(rl["V"], rc["V"])
            and np.array_equal(np.array([q["V"] for q in rl["instr"]]), np.array([q["V"] for q in rc["instr"]]))
            for rl, rc in zip(runs["legacy"], cached_runs)))
    else:
        out["legacy_runs_vs_B3_pkl_bitwise"] = None
    for rl, rc in zip(runs["legacy"], runs["corrected"]):
        m = min(len(rl["instr"]), len(rc["instr"]))
        same_traj = all(ql["t"] == qc["t"] and ql["grad_norm2"] == qc["grad_norm2"] and ql["zbar_norm2"] == qc["zbar_norm2"]
                        for ql, qc in zip(rl["instr"][:m], rc["instr"][:m]))
        Vl = np.array([q["V"] for q in rl["instr"]])
        Vc = np.array([q["V"] for q in rc["instr"]])
        kept = Vl[:m] >= FLOOR
        rel = np.abs(Vc[:m] - Vl[:m])[kept] / Vl[:m][kept]
        # recorded V_t (incremental memory error since the last instrumented state) against the exact V_t of the
        # instrumented state at the same iteration (records with V_t >= FLOOR, t > 0)
        it_l = np.asarray(rl["iters"])
        ti = np.array([q["t"] for q in rl["instr"]])
        _, ia, ib = np.intersect1d(it_l, ti, return_indices=True)
        sel = (it_l[ia] > 0) & (Vl[ib] >= FLOOR)
        rel_trace = np.abs(np.asarray(rl["V"])[ia][sel] - Vl[ib][sel]) / Vl[ib][sel]
        out["per_seed"].append({
            "seed": int(rl["seed"]), "same_iterates_on_common_records": bool(same_traj), "n_common_records": int(m),
            "n_kept_legacy": int(np.sum(Vl >= FLOOR)), "n_kept_corrected": int(np.sum(Vc >= FLOOR)),
            "n_kept_status_changes": int(np.sum((Vl[:m] >= FLOOR) != (Vc[:m] >= FLOOR))),
            "n_kept_legacy_after_corrected_stop": int(np.sum(Vl[m:] >= FLOOR)),
            "max_rel_change_V_kept": float(rel.max()) if rel.size else None,
            "recorded_vs_instrumented_V_max_rel": float(rel_trace.max()) if rel_trace.size else None,
            "n_recorded_vs_instrumented": int(rel_trace.size),
            "final_D_legacy": float(rl["D"][-1]), "final_D_corrected": float(rc["D"][-1]),
            "passes_legacy": float(rl["iters"][-1]) / N, "passes_corrected": float(rc["iters"][-1]) / N,
            "stopped_legacy": rl["stopped"], "stopped_corrected": rc["stopped"]})
    ps = out["per_seed"]
    out["max_rel_change_V_kept"] = max(x["max_rel_change_V_kept"] for x in ps if x["max_rel_change_V_kept"] is not None)
    out["recorded_vs_instrumented_V_max_rel"] = max(x["recorded_vs_instrumented_V_max_rel"] for x in ps
                                                    if x["recorded_vs_instrumented_V_max_rel"] is not None)
    out["same_iterates_all_seeds"] = all(x["same_iterates_on_common_records"] for x in ps)
    out["n_kept_status_changes"] = sum(x["n_kept_status_changes"] + x["n_kept_legacy_after_corrected_stop"] for x in ps)
    out["seconds"] = time.time() - t0
    return out


# ----------------------------------------------------------------------------------------------------------------------
# (f): floor of the double-precision iterates
# ----------------------------------------------------------------------------------------------------------------------
def plateau(passes, Dv, last=20.0, tol=0.1):
    """Plateau of a recorded squared distance (definitions in the module docstring)."""
    P = float(passes[-1])
    win = passes > P - last
    level = float(np.median(Dv[win]))
    m1 = float(np.median(Dv[win & (passes <= P - last / 2)]))
    m2 = float(np.median(Dv[passes > P - last / 2]))
    saturated = bool(abs(m2 / m1 - 1.0) < tol)
    inside = (Dv <= (1 + tol) * level) & (Dv >= level / (1 + tol))
    bad = np.where(~inside)[0]
    i0 = int(bad[-1] + 1) if bad.size else 0
    return {"level": level, "window_passes": [P - last, P], "min_in_window": float(Dv[win].min()),
            "max_in_window": float(Dv[win].max()), "median_first_half": m1, "median_second_half": m2,
            "saturated": saturated, "onset_pass": float(passes[i0]) if (saturated and i0 < len(passes)) else None,
            "D_last": float(Dv[-1])}


def part_f(prob, eta_c, cached_runs, js_attr):
    p_c = with_optimum(prob, eta_c)
    gamma_tb, _ = bounds.saga_textbook_params(p_c.R2)
    out = {"problem": key(*FLOOR_PROBLEM), "seed": 0, "reference": "eta_c = eta_* - e1",
           "sqerr_bound_eta_c": None, "runs": [], "B3_final_D_mean_5_seeds": js_attr["final_D"], "stop_rule": STOP_D}
    for c, passes in FLOOR_RUNS:
        t0 = time.time()
        gamma = c / p_c.R2
        r = algorithms.saga(p_c, np.zeros(D), gamma=gamma, Delta=3.0 * gamma ** 2, passes=passes, seed=0,
                            record_every=max(1, N // 20), record_F=False)
        pa = np.asarray(r["iters"], dtype=np.float64) / N
        Dv = np.asarray(r["D"], dtype=np.float64)
        v = r["theta"] - prob.theta_star                     # distance to the legacy eta_*, as recorded in B3.pkl
        rec = {"gammaR2": c, "gamma": gamma, "textbook_step": bool(gamma == gamma_tb), "passes": passes,
               "plateau": plateau(pa, Dv), "D_legacy_ref_end": float(v @ v),
               "D_per_pass": [float(x) for x in Dv[np.asarray(r["iters"]) % N == 0]], "seconds": time.time() - t0}
        if rec["textbook_step"] and passes == PASSES and cached_runs is not None:
            rec["follows_cached_seed0_run"] = bool(rec["D_legacy_ref_end"] == float(cached_runs[0]["D"][-1]))
        out["runs"].append(rec)
        pl = rec["plateau"]
        print(f"   gamma R^2 = {c:g}, {passes} passes: plateau {pl['level']:.3e} (saturated {pl['saturated']}, onset pass "
              f"{pl['onset_pass']}); D at the end {pl['D_last']:.3e} [{time.time() - t0:.1f}s]", flush=True)
    return out


def part_g(prob):
    """(g): memory error recorded without instrumentation (module docstring)."""
    out = {"problem": key(*FLOOR_PROBLEM), "seed": 0, "reference": "legacy eta_*", "checkpoints": []}
    for T in MEMORY_CHECK_T:
        t0 = time.time()
        r = algorithms.saga(prob, np.zeros(D), n_iters=T + 1, seed=0, record_every=max(1, N // 20), instrument_every=T,
                            record_F=False)
        j = int(np.where(np.asarray(r["iters"]) == T)[0][0])
        q = r["instr"][-1]
        assert q["t"] == T and len(r["instr"]) == 2
        M_rec, V_rec = float(r["M"][j]), float(r["V"][j])
        out["checkpoints"].append({"t": T, "pass": T / N, "M_exact": q["M"], "M_recorded": M_rec, "abs_err_M": abs(M_rec - q["M"]),
                                   "Delta": float(r["Delta"]), "abs_err_V": abs(V_rec - q["V"]), "V_exact": q["V"],
                                   "rel_err_V": abs(V_rec - q["V"]) / q["V"], "seconds": time.time() - t0})
        c = out["checkpoints"][-1]
        print(f"   T = {T}: exact V {c['V_exact']:.3e}, |recorded M - exact M| {c['abs_err_M']:.3e}, relative error of the "
              f"recorded V {c['rel_err_V']:.3e} [{c['seconds']:.1f}s]", flush=True)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# --threads1: (a), (b), (e) with single-threaded BLAS
# ----------------------------------------------------------------------------------------------------------------------
def load_b3_summary():
    with open(common.result_path("B3_summary", "json"), encoding="utf-8") as f:
        return json.load(f)


def worker(outdir):
    js = load_b3_summary()
    recs, _, _ = part_abe(js, check_digits=False, realized=False, save_data_dir=outdir)
    with open(os.path.join(outdir, "abe.json"), "w", encoding="utf-8") as f:
        json.dump({"environment": environment(), "problems": recs}, f, indent=1, default=common._json_default)


def threads1(recs, probs, reach, reach2):
    outdir = tempfile.mkdtemp(prefix="check_optima_mp_")
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1")
    t0 = time.time()
    print("--threads1: (a), (b), (e) in a subprocess with OPENBLAS_NUM_THREADS=1", flush=True)
    subprocess.run([sys.executable, "-B", os.path.abspath(__file__), "--worker", outdir], env=env, check=True)
    with open(os.path.join(outdir, "abe.json"), encoding="utf-8") as f:
        w = json.load(f)
    out = {"environment": w["environment"], "problems": {}}
    for k, r1 in w["problems"].items():
        X1, y1 = np.load(os.path.join(outdir, f"X_{k}.npy")), np.load(os.path.join(outdir, f"y_{k}.npy"))
        eta1 = np.load(os.path.join(outdir, f"eta_{k}.npy"))
        p, b8, b1 = probs[k], recs[k]["b_c"], r1["b_c"]
        dX = np.abs(X1 - p.X)
        out["problems"][k] = {
            "X_identical": bool(np.array_equal(X1, p.X)), "X_frac_entries_differ": float(np.mean(X1 != p.X)),
            "X_max_abs_diff": float(dX.max()), "X_max_diff_in_ulps": float(np.max(dX / np.spacing(np.abs(p.X)))),
            "y_identical": bool(np.array_equal(y1, p.y)),
            "summary_vs_B3_json": {kk: v for kk, v in r1["a_summary_vs_B3_json"].items() if kk != "n_fields"},
            "eta_identical_to_default": bool(np.array_equal(eta1, p.theta_star)),
            "eta_dist_to_default": float(np.linalg.norm(eta1 - p.theta_star)),
            "r_mp": b1["r_mp"], "r_mp_ratio_to_default": b1["r_mp"] / b8["r_mp"],
            "sqerr_bound": b1["sqerr_bound"], "sqerr_bound_ratio_to_default": b1["sqerr_bound"] / b8["sqerr_bound"],
            "V_floor_worst_rel_err": b1["V_floor_worst_rel_err"], "D_floor_worst_rel_err": b1["D_floor_worst_rel_err"],
            "current_solver_identical_to_legacy": r1["e_current_solver"]["identical_to_cached"],
            "current_solver_dist_to_legacy": r1["e_current_solver"]["dist_to_cached"]}
    for name in os.listdir(outdir):
        os.remove(os.path.join(outdir, name))
    os.rmdir(outdir)
    P = out["problems"]
    ratios = [v["sqerr_bound_ratio_to_default"] for v in P.values()]
    rest = [k for k in reach if k != key(10.0, 1.0)]
    out["overview"] = {
        "V_floor_worst_rel_err_kappa10_alpha1": P[key(10.0, 1.0)]["V_floor_worst_rel_err"],
        "V_floor_worst_rel_err_max_other_reaching": max(P[k]["V_floor_worst_rel_err"] for k in rest) if rest else None,
        "B2_D_floor_worst_rel_err_max": max(P[key(*p)]["D_floor_worst_rel_err"] for p in B2_PROBLEMS),
        "B2_D_floor_worst_rel_err_max_reaching": max(P[k]["D_floor_worst_rel_err"] for k in reach2) if reach2 else None,
        "n_problems_X_identical": sum(v["X_identical"] for v in P.values()),
        "n_problems_y_identical": sum(v["y_identical"] for v in P.values()),
        "max_X_diff_in_ulps": max(v["X_max_diff_in_ulps"] for v in P.values()),
        "n_problems_summary_bitwise": sum(v["summary_vs_B3_json"]["bitwise"] for v in P.values()),
        "max_rel_diff_summary_fields": max([x for v in P.values() for x in v["summary_vs_B3_json"]["differing_fields"].values()
                                            if isinstance(x, float)] or [0.0]),
        "max_eta_dist_to_default": max(v["eta_dist_to_default"] for v in P.values()),
        "sqerr_bound_ratio_min": min(ratios), "sqerr_bound_ratio_max": max(ratios),
        "max_sqerr_bound": max(v["sqerr_bound"] for v in P.values()),
        "current_solver_differs_on": sorted(k for k, v in P.items() if not v["current_solver_identical_to_legacy"]),
        "seconds": time.time() - t0}
    return out


# ----------------------------------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------------------------------
def overview(recs, floor_stats, b2_stats, imp, fl, mem, js):
    reach = {k for k, v in floor_stats.items() if v["reaches_floor"]}
    others = {k: v for k, v in recs.items() if k != key(10.0, 1.0)}
    b = {k: v["b_c"] for k, v in recs.items()}
    kmax = max(others, key=lambda k: b[k]["r_mp"])
    o = {"n_problems": len(recs),
         "n_summaries_bitwise": sum(v["a_summary_vs_B3_json"]["bitwise"] for v in recs.values()),
         "r_mp_kappa10_alpha1": b[key(10.0, 1.0)]["r_mp"],
         "r_mp_max_other": b[kmax]["r_mp"], "r_mp_argmax_other": kmax,
         "max_rel_diff_60_100_digits": max(v["rel_diff_60_100_digits"] for v in b.values() if "rel_diff_60_100_digits" in v),
         "r_mp_over_r64_grad_range": [min(v["r_mp_over_r64_grad"] for v in b.values()), max(v["r_mp_over_r64_grad"] for v in b.values())],
         "r_mp_over_r64_stored_range": [min(v["r_mp_over_r64_stored"] for v in b.values()), max(v["r_mp_over_r64_stored"] for v in b.values())],
         "sqerr_bound_max": max(v["sqerr_bound"] for v in b.values()),
         "sqerr_bound_argmax": max(b, key=lambda k: b[k]["sqerr_bound"]),
         "Fgap_bound_max": max(v["Fgap_bound"] for v in b.values()),
         "c_V_range": [min(v["c_V"] for v in b.values()), max(v["c_V"] for v in b.values())],
         "B2_D_floor_worst_rel_err_max": max(b[key(*p)]["D_floor_worst_rel_err"] for p in B2_PROBLEMS),
         "B2_D_floor_worst_rel_err_argmax": max((key(*p) for p in B2_PROBLEMS), key=lambda k: b[k]["D_floor_worst_rel_err"]),
         "B2_D_floor_rel_err_realized_max": max(b[key(*p)]["realized"]["D_floor_rel_err_realized"] for p in B2_PROBLEMS)}
    if reach:
        rest = [k for k in reach if k != key(10.0, 1.0)]
        o.update({
            "problems_reaching_floor": sorted(reach), "problems_not_reaching_floor": sorted(set(recs) - reach),
            "sqerr_bound_max_reaching_floor": max(b[k]["sqerr_bound"] for k in reach),
            "sqerr_bound_argmax_reaching_floor": max(reach, key=lambda k: b[k]["sqerr_bound"]),
            "min_V_cached_not_reaching_floor": min(floor_stats[k]["min_V"] for k in set(recs) - reach),
            "V_floor_worst_rel_err_kappa10_alpha1": b[key(10.0, 1.0)]["V_floor_worst_rel_err"],
            "V_floor_worst_rel_err_max_other_reaching": max(b[k]["V_floor_worst_rel_err"] for k in rest),
            "V_floor_worst_rel_err_argmax_other_reaching": max(rest, key=lambda k: b[k]["V_floor_worst_rel_err"]),
            "V_floor_rel_err_realized_kappa10_alpha1": b[key(10.0, 1.0)]["realized"]["V_floor_rel_err_realized"],
            "V_floor_rel_err_realized_max_other_reaching": max(b[k]["realized"]["V_floor_rel_err_realized"] for k in rest),
            "e1_sq_max_other_reaching": max(b[k]["realized"]["e1_sq"] for k in rest),
            "e1_sq_argmax_other_reaching": max(rest, key=lambda k: b[k]["realized"]["e1_sq"])})
    k10 = key(10.0, 1.0)
    o.update({"e1_sq_kappa10_alpha1": b[k10]["realized"]["e1_sq"],
              "sqerr_interval_kappa10_alpha1": [b[k10]["realized"]["sqerr_low"], b[k10]["realized"]["sqerr_high"]],
              "final_D_B3_summary_kappa10_alpha1": js["attribution"][k10]["final_D"],
              "e1_sq_over_final_D_kappa10_alpha1": b[k10]["realized"]["e1_sq"] / js["attribution"][k10]["final_D"],
              "final_D_per_seed_range_kappa10_alpha1": ([min(floor_stats[k10]["final_D_per_seed"]), max(floor_stats[k10]["final_D_per_seed"])]
                                                        if k10 in floor_stats else None),
              "e1_sq_max_other": max(b[k]["realized"]["e1_sq"] for k in others),
              "e1_sq_argmax_other": max(others, key=lambda k: b[k]["realized"]["e1_sq"]),
              "r_c_max": max(v["realized"]["r_c"] for v in b.values()),
              "current_solver_differs_on": sorted(k for k, v in recs.items() if not v["e_current_solver"]["identical_to_cached"])})
    for k in o["current_solver_differs_on"]:
        e = recs[k]["e_current_solver"]
        o.setdefault("current_solver", {})[k] = {"dist_to_cached": e["dist_to_cached"], "r_mp": e["r_mp"],
                                                 "r_mp_cached": b[k]["r_mp"], "dist_to_eta_c": e["dist_to_eta_c"]}
    o["impact"] = {k: {"legacy_reproduces_B3_summary_json": v["legacy_summary_vs_B3_json"]["bitwise"],
                       "legacy_reproduces_B3_pkl": v["legacy_runs_vs_B3_pkl_bitwise"],
                       "same_iterates": v["same_iterates_all_seeds"], "max_rel_change_V_kept": v["max_rel_change_V_kept"],
                       "window_2_20_max_rel_change": v["statistics_change"]["window_2_20"]["max_rel_change"],
                       "window_2_20_max_abs_change_shares": v["statistics_change"]["window_2_20"]["max_abs_change_shares"],
                       "late_max_abs_change_shares": v["statistics_change"]["late_20_100"]["max_abs_change_shares"],
                       "late_max_rel_change": v["statistics_change"]["late_20_100"]["max_rel_change"],
                       "passes_corrected_range": [min(x["passes_corrected"] for x in v["per_seed"]), max(x["passes_corrected"] for x in v["per_seed"])],
                       "final_D_corrected_range": [min(x["final_D_corrected"] for x in v["per_seed"]), max(x["final_D_corrected"] for x in v["per_seed"])],
                       "n_kept_status_changes": v["n_kept_status_changes"]} for k, v in imp.items()}
    o["iterate_floor"] = {f"gammaR2={r['gammaR2']:g},passes={r['passes']}": {"level": r["plateau"]["level"], "saturated": r["plateau"]["saturated"],
                                                                            "onset_pass": r["plateau"]["onset_pass"], "D_last": r["plateau"]["D_last"]}
                          for r in fl["runs"]}
    reach2 = sorted(k for k, v in b2_stats.items() if v["reaches_floor"])
    if reach2:
        o.update({"B2_problems_reaching_floor": reach2,
                  "B2_D_floor_worst_rel_err_max_reaching": max(b[k]["D_floor_worst_rel_err"] for k in reach2),
                  "B2_D_floor_worst_rel_err_argmax_reaching": max(reach2, key=lambda k: b[k]["D_floor_worst_rel_err"]),
                  "B2_min_D_kept_not_reaching": {k: v["min_D_kept"] for k, v in b2_stats.items() if not v["reaches_floor"]}})
    o["recorded_vs_instrumented_V_max_rel"] = max(v["recorded_vs_instrumented_V_max_rel"] for v in imp.values())
    o["uninstrumented_memory_error"] = [{kk: c[kk] for kk in ("pass", "V_exact", "abs_err_M", "abs_err_V", "rel_err_V")}
                                        for c in mem["checkpoints"]]
    return o


def main():
    if "--worker" in sys.argv:
        worker(sys.argv[sys.argv.index("--worker") + 1])
        return
    t_start = time.time()
    js = load_b3_summary()
    cache = common.load_result("B3")
    print("results/B3.pkl", "loaded" if cache is not None else "absent: no bitwise check of the runs, no floor statistics", flush=True)
    out = {"setting": {"n": N, "d": D, "kappas": KAPPAS, "alphas": ALPHAS, "data_seed": 0, "solver": "solve_optimum(prob, armijo_only=True)",
                       "digits": DPS, "digits_check": DPS_CHECK, "floor": FLOOR, "stop_D": STOP_D, "seeds": SEEDS, "passes": PASSES,
                       "rerun": [key(*p) for p in RERUN], "floor_problem": key(*FLOOR_PROBLEM),
                       "floor_runs": [{"gammaR2": c, "passes": p} for c, p in FLOOR_RUNS], "B2_problems": [key(*p) for p in B2_PROBLEMS]},
           "environment": environment(), "seconds": {}}
    print("environment:", out["environment"], flush=True)
    t0 = time.time()
    print("(a), (b), (c), (e) on the 15 problems", flush=True)
    recs, probs, etas_c = part_abe(js)
    out["seconds"]["abce"] = time.time() - t0
    floor_stats = cached_floor_stats(cache)
    b2_cache = common.load_result("B2")
    b2_stats = cached_b2_floor_stats(b2_cache)
    del b2_cache
    for k in recs:
        recs[k]["cached_runs"] = floor_stats.get(k)
        if k in b2_stats:
            recs[k]["cached_runs_B2"] = b2_stats[k]
    out["problems"] = recs
    t0 = time.time()
    imp = {}
    for kappa, alpha in RERUN:
        k = key(kappa, alpha)
        print(f"(d) re-runs at {k}", flush=True)
        imp[k] = part_d(probs[k], etas_c[k], js["attribution"][k], None if cache is None else cache["runs"][(kappa, alpha)])
        print(f"   legacy summary bitwise {imp[k]['legacy_summary_vs_B3_json']['bitwise']}, runs bitwise "
              f"{imp[k]['legacy_runs_vs_B3_pkl_bitwise']}; max rel. change of V (kept) {imp[k]['max_rel_change_V_kept']:.3e}; "
              f"passes 2-20 {imp[k]['statistics_change']['window_2_20']['max_rel_change']:.3e}; late shares "
              f"{imp[k]['statistics_change']['late_20_100']['max_abs_change_shares']}", flush=True)
    out["impact"] = imp
    out["seconds"]["d"] = time.time() - t0
    t0 = time.time()
    k = key(*FLOOR_PROBLEM)
    print(f"(f) floor of the iterates at {k}", flush=True)
    fl = part_f(probs[k], etas_c[k], None if cache is None else cache["runs"][FLOOR_PROBLEM], js["attribution"][k])
    fl["sqerr_bound_eta_c"] = recs[k]["b_c"]["realized"]["sqerr_bound_eta_c"]
    out["iterate_floor"] = fl
    out["seconds"]["f"] = time.time() - t0
    t0 = time.time()
    print(f"(g) memory error recorded without instrumentation at {k}", flush=True)
    mem = part_g(probs[k])
    out["memory_error_uninstrumented"] = mem
    out["seconds"]["g"] = time.time() - t0
    out["overview"] = overview(recs, floor_stats, b2_stats, imp, fl, mem, js)
    if "--threads1" in sys.argv:
        out["threads1"] = threads1(recs, probs, {k for k, v in floor_stats.items() if v["reaches_floor"]},
                                   {k for k, v in b2_stats.items() if v["reaches_floor"]})
    out["seconds"]["total"] = time.time() - t_start
    common.save_json(TAG, out)
    print(json.dumps(out["overview"], indent=1, default=common._json_default), flush=True)
    print(f"done check_optima_mp [{out['seconds']['total']:.0f}s] -> results/{TAG}.json", flush=True)


if __name__ == "__main__":
    main()
