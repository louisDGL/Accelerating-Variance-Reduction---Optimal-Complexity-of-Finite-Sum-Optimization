"""Hypothesis B1 on the synthetic designs of Na, Derezinski and Mahoney (Math. Program. 201, 2023, Section 5): coherence of
the leverage scores and orientation of the anisotropy (Section exp-B1 of the report, paragraph "Orientation and
coherence"; paragraph "Outcome of the experiments" of the B1 note).

Setting. P-SAGA exactly as experiments/exp_B1_preconditioning.py: theta_t = theta_{t-1} - gamma P^{-1} omega_t with
P in {I, diag(Sigma_hat) + lam I, Sigma_hat + lam I, F''(eta_*)} (function preconditioner, identical), step
gamma = 1/(4 R_P^2) with R_P^2 the upper bound max_i ||x_i||^2_{P^-1}/4 + lam lmax(P^-1) of eq. B1-logistic-RP of the B1
note (vrlab Metric.constants), theta_0 = 0, z_i^(0) = f_i'(theta_0), records every n/20 iterations, exact instrumentation
every n/10 iterations, at most PASSES = 100 passes, stop when ||theta_t - eta_*||_P^2 <= 1e-26. Problems:
vrlab.problems.make_logistic_ndm(1000, 50, kappa, kap=1, high_coherence, seed, max_row_norm=1) (X = c U diag(sigma),
sigma_j equally spaced from 1 to d, so that hat Sigma is exactly diagonal with condition number d^2 = 2500; logistic
labels from a planted vector; lam such that R^2/mu = kappa), eta_* by vrlab.solvers.solve_optimum (default solver),
its residual re-evaluated in 40-digit arithmetic (vrlab.highprec):
  * main design, kappa = 1e3: low coherence ("lc") with data seeds 0-2, high coherence ("hc") with data seeds 0-4;
    metrics I, Sigma_hat + lam I ("stat"), F''(eta_*) ("hess"); sampling seeds 0-4; the Jacobi metric ("diag") on data
    seed 0 of both families (on these axis-aligned designs it equals Sigma_hat + lam I up to rounding, Proposition
    B1-diag(iii));
  * rotated copies X -> X Q^T, Q Haar (QR of a Gaussian matrix drawn by numpy default_rng([20231, family, data seed]),
    signs fixed by the diagonal of R): leverage scores, labels, lam, R^2 and the spectrum of Sigma_hat unchanged,
    Sigma_hat no longer diagonal, eta_* recomputed; metrics I and Jacobi for every data seed, Sigma_hat + lam I and
    F''(eta_*) on data seed 0 (P-SAGA with these three metrics is rotation-equivariant: check);
  * probe at kappa = 1e4 (same data, smaller lam): data seed 0 of both families, metrics I, Sigma_hat + lam I, F''(eta_*),
    sampling seeds 0-2;
  * self-check: the two problems of results/B1_summary.json (make_logistic(1000, 50, 1e3, alpha in {1, 2}, seed 0), eta_*
    with the solver of the cached runs, newton_solve(armijo_only=True)), four metrics, sampling seeds 0-4: rates, constants
    and ratio geometric means compared with results/B1_summary.json, per-state mu_eff^P with results/B1.pkl. With
    OPENBLAS_NUM_THREADS=1 the rotated features of make_logistic differ in their last bits from those of the cached
    (multithreaded) runs, so this reproduction is close, not bit for bit.
Statistics (conventions of exp_B1_preconditioning.py and of the B1 note): exponent per pass = -n x least-squares slope of
log ||theta_t - eta_*||_P^2 against t over passes 2-20 (records above 1e-25), per sampling seed; mean, min and max over
sampling seeds; gain of a metric = ratio of its seed-mean exponent to that of P = I on the same data set; tightness ratios:
geometric means over the instrumented states of passes 2-20 pooled over seeds (runs truncated to the shortest);
kappa_eff^P = R_P^2 / mu_eff^P(theta_{t-1}) with the upper-bound R_P^2, geometric mean over passes 15-20; first bound of
Corollary B1-kappa-traj, (d_inf/4 + lam/(sigma_d + lam)) (sigma_1 + lam)/(m(theta) sigma_1 + lam), along the runs of
Sigma_hat + lam I, with the segment curvature m(theta) = min_i int_0^1 sigma'(y_i x_i^T (eta_* + s (theta - eta_*))) ds
at the instrumented states; parameter share of V^P = ||theta - eta_*||_P^2 / V^P (median over passes 2-20); condition
n >= (4 d_inf + 16)/(9 m) of Remark B1-memory-bottleneck with m = min_i w_i (w_i = sigma'(y_i x_i^T eta_*)) and with the
smallest m(theta_t) met along the runs. Every quantity is measured on these instances; no constant of the method is
tuned to them.
BLAS: run with OPENBLAS_NUM_THREADS=1 (recorded in the summary). --workers K (default 3) worker processes run the
independent (problem, metric) tasks while the parent waits; every task rebuilds its problem deterministically, so the
results do not depend on K (the parent checks that all tasks saw the same eta_*).
Usage: OPENBLAS_NUM_THREADS=1 python experiments/exp_B1_coherence.py [--workers 3] [--rerun] [--resummarize] [--smoke]
  --rerun ignores the cache results/B1_coherence.pkl; --resummarize recomputes the problem summaries (geometry,
  constants, certificates) from the cached runs; --smoke runs two short tasks and writes nothing.
Outputs: results/B1_coherence.pkl (per-run traces and per-state statistics), results/B1_coherence_summary.json.
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from vrlab import algorithms, bounds, problems, solvers  # noqa: E402
from vrlab.instrumentation import Metric  # noqa: E402
from vrlab.problems import sigmoid  # noqa: E402

N, D = 1000, 50
KAPPA, KAPPA_PROBE, KAP = 1e3, 1e4, 1.0
DATA_SEEDS = {"lc": [0, 1, 2], "hc": [0, 1, 2, 3, 4]}
FAMILY_NAME = {"lc": "low coherence", "hc": "high coherence"}
SEEDS, SEEDS_PROBE = [0, 1, 2, 3, 4], [0, 1, 2]
PASSES = 100
WINDOW, LATE = (2, 20), (15, 20)         # passes: rate fit and tightness ratios; late window of the B1 note
FLOOR_FIT = 1e-25                        # records below are not fitted (exp_B1_preconditioning.empirical_rate)
ROT_ENTROPY = 20231
PRECONDS = ["I", "diag", "stat", "hess"]
TAG = "B1_coherence"
STATE_KEYS = ["t", "D", "M", "V", "G", "mu_eff", "L_eff", "r_sq", "r_coco", "r_var", "r_sc", "r_min", "Dshare", "rho_exact"]
RATIOS = ["r_sq", "r_coco", "r_var", "r_sc", "r_min"]


# ----------------------------------------------------------------------------------------------------------------------
# problems and metrics
# ----------------------------------------------------------------------------------------------------------------------
def spec_name(spec):
    """("ndm", family, data seed, rotated, kappa) -> e.g. 'hc|seed=3|rot'; ("b1", alpha) -> 'B1|alpha=2'."""
    if spec[0] == "b1":
        return f"B1|alpha={spec[1]:g}"
    _, fam, ds, rot, kappa = spec
    return f"{fam}|seed={ds}" + ("|rot" if rot else "") + (f"|kappa={kappa:g}" if kappa != KAPPA else "")


def haar(d, fam, ds):
    """Haar-distributed orthogonal matrix (QR of a Gaussian matrix, signs fixed by the diagonal of R)."""
    rng = np.random.default_rng([ROT_ENTROPY, ["lc", "hc"].index(fam), ds])
    Q, R = np.linalg.qr(rng.standard_normal((d, d)))
    return Q * np.sign(np.diag(R))[None, :]


def build(spec):
    """The problem of `spec` with its eta_* (deterministic)."""
    if spec[0] == "b1":
        prob = problems.make_logistic(N, D, kappa=KAPPA, alpha=spec[1], seed=0)
        solvers.solve_optimum(prob, armijo_only=True)          # solver of the cached runs of exp_B1_preconditioning
        return prob
    _, fam, ds, rot, kappa = spec
    prob = problems.make_logistic_ndm(N, D, kappa, kap=KAP, high_coherence=(fam == "hc"), seed=ds, max_row_norm=1.0)
    if rot:
        Q = haar(D, fam, ds)
        meta = dict(prob.meta, rotated=True)
        prob = problems.LogisticProblem(prob.X @ Q.T, prob.y, prob.lam, name=prob.name + "-rotated")
        prob.meta = meta
    solvers.solve_optimum(prob)
    return prob


def preconditioner(prob, kind):
    """Identical to experiments/exp_B1_preconditioning.preconditioner."""
    lam = prob.lam
    if kind == "I":
        return None
    if kind == "diag":
        return np.diag(prob.Sigma_hat) + lam
    if kind == "stat":
        return prob.Sigma_hat + lam * np.eye(prob.d)
    if kind == "hess":
        return prob.hessian(prob.theta_star)
    raise ValueError(kind)


def exact_R2P(prob, P):
    """Supremum of Definition B1-kappaP: max_i lmax(P^{-1/2} (x_i x_i^T / 4 + lam I) P^{-1/2}) (attained at sigma' = 1/4)."""
    if P is None:
        return float(prob.R2)
    Pd = np.diag(P) if np.ndim(P) == 1 else P
    pe, pV = np.linalg.eigh(Pd)
    A = prob.X @ ((pV / np.sqrt(pe)) @ pV.T)                     # rows P^{-1/2} x_i
    lamPinv = prob.lam * ((pV / pe) @ pV.T)
    best = -np.inf
    for i0 in range(0, prob.n, 200):
        Ab = A[i0:i0 + 200]
        M = 0.25 * Ab[:, :, None] * Ab[:, None, :] + lamPinv[None, :, :]
        best = max(best, float(np.linalg.eigvalsh(M)[:, -1].max()))
    return best


def segment_curvatures(prob, theta):
    """bar sigma'_i(theta) = int_0^1 sigma'(a_i^* + s (a_i - a_i^*)) ds = (sigma(a_i) - sigma(a_i^*)) / (a_i - a_i^*),
    a_i = y_i x_i^T theta, a_i^* = y_i x_i^T eta_* (closed form eq. B2-weights-closed of the B2 note); midpoint value
    sigma'((a_i + a_i^*)/2) when |a_i - a_i^*| < 1e-5 (error O(|a_i - a_i^*|^2))."""
    a_star = prob.y * (prob.X @ prob.theta_star)
    a = prob.y * (prob.X @ theta)
    b = a - a_star
    small = np.abs(b) < 1e-5
    q = (sigmoid(a) - sigmoid(a_star)) / np.where(small, 1.0, b)
    mid = 0.5 * (a + a_star)
    return np.where(small, sigmoid(mid) * sigmoid(-mid), q)


def geometry(prob):
    """Metric-independent quantities of the B1 statements."""
    X, lam, n, d = prob.X, prob.lam, prob.n, prob.d
    sig = np.linalg.eigvalsh(prob.Sigma_hat)
    s1, sd = float(sig[-1]), float(sig[0])
    nlev = prob.leverage_scores()                                # n lev_i(lam) = ||x_i||^2_{(Sigma_hat + lam I)^{-1}}
    d_eff = float(np.sum(sig / (sig + lam)))
    d_inf = float(nlev.max())
    U = np.linalg.svd(X, full_matrices=False)[0]
    w = prob.sigma_prime(prob.theta_star)
    wmin = float(w.min())
    dS = np.diag(prob.Sigma_hat)
    cc = np.linalg.eigvalsh(prob.Sigma_hat / np.sqrt(np.outer(dS, dS)))
    R2 = float(prob.R2)
    return {
        "lam": lam, "R2": R2, "kappa": float(prob.kappa), "sigma_1": s1, "sigma_d": sd, "cov_cond": s1 / sd,
        "max_sqnorm": float(prob.sqnorms.max()), "mean_sqnorm": float(prob.sqnorms.mean()),
        "d_eff": d_eff, "d_inf": d_inf, "C": d_inf / d_eff, "nlev_median": float(np.median(nlev)),
        "n_nlev_above_5_d_eff": int(np.sum(nlev > 5 * d_eff)), "sum_lev_minus_d_eff": float(nlev.sum() / n - d_eff),
        "coherence_U": float(n / d * np.einsum("ij,ij->i", U, U).max()),   # (n/d) max_i ||U_i||^2, in [1, n/d]
        "w_min": wmin, "w_median": float(np.median(w)), "w_mean": float(w.mean()), "w_max": float(w.max()),
        "corr_cond": float(cc[-1] / cc[0]), "corr_lmin": float(cc[0]), "corr_lmax": float(cc[-1]),
        "jacobi_cond": float((dS.max() + lam) / (dS.min() + lam)),    # cond(diag(Sigma_hat) + lam I)
        "stat_cond": float((s1 + lam) / (sd + lam)),                   # cond(Sigma_hat + lam I)
        # Sigma_hat + lam I against the Jacobi metric: largest entry of the difference (off-diagonal of Sigma_hat) / lmin(P_J)
        "stat_minus_jacobi_max_over_lmin_PJ": float(np.abs(prob.Sigma_hat - np.diag(dS)).max() / (dS.min() + lam)),
        # Corollary B1-kappa-traj and Remark B1-memory-bottleneck in the limit theta -> eta_*, m(theta) -> min_i w_i
        "cor_stat_limit": (d_inf / 4 + lam / (sd + lam)) * (s1 + lam) / (wmin * s1 + lam),
        "cor_stat_limit_crude": (d_inf / 4 + 1) / wmin,
        "cor_I_limit": R2 / (wmin * sd + lam), "cor_I_lower": R2 / (s1 / 4 + lam),
        "n_required_limit": (4 * d_inf + 16) / (9 * wmin),
        "theta_star_norm": float(np.linalg.norm(prob.theta_star)),
        "max_abs_margin_at_eta": float(np.max(np.abs(X @ prob.theta_star))),
        # Bayes accuracy (1/n) sum_i max(p_i, 1 - p_i), p_i = sigmoid(x_i^T w) with the planted w (unrotated designs only)
        "bayes_accuracy": (float(np.mean(np.maximum(sigmoid(X @ prob.meta["planted"]), 1 - sigmoid(X @ prob.meta["planted"]))))
                           if "planted" in getattr(prob, "meta", {}) and not prob.meta.get("rotated") else None),
        "newton_grad_norm": float(prob.opt_info["grad_norm"]), "newton_iters": int(prob.opt_info["iters"]),
    }


def metric_constants(prob, kind):
    """Global and local constants of the metric (Lemma B1-logistic, Definition B1-kappaP, Lemma B1-asymptotic)."""
    P = preconditioner(prob, kind)
    cst = Metric(P, prob.d).constants(prob, theta_ref=prob.theta_star)
    R2x = exact_R2P(prob, P)
    n = prob.n
    cst.update({
        "R2_P_exact": R2x, "kappa_P_exact": R2x / cst["mu_P"], "R2_P_upper_over_exact": cst["R2_P"] / R2x,
        "lmin_M": cst["mu_P_local"], "cond_M": cst["kappa_P_local"],
        "R2_P_over_lmin_M": cst["R2_P"] / cst["mu_P_local"],                 # bound on limsup kappa_eff^P (B1 note)
        "crit_9n_lmin_M_over_16R2P": 9 * n * cst["mu_P_local"] / (16 * cst["R2_P"]),   # > 1 iff R_P^2/lmin(M_P) < 9n/16
        "rate_certified_pass": n * (1 - bounds.saga_rate(n, cst["R2_P"], cst["mu_P"])),
        "rate_local_pass": n * (1 - bounds.saga_rate(n, cst["R2_P"], cst["mu_P_local"])),
    })
    return cst


def certificate(prob, dps=40):
    """||F'(eta_*)|| re-evaluated in dps-digit arithmetic on the stored data; r/lam bounds ||eta_* - eta_exact||."""
    from vrlab import highprec                                   # needs mpmath
    _, r = highprec.mp_logistic_grad(prob.X, prob.y, prob.lam, prob.theta_star, dps)
    return {"grad_norm_mp": r, "dist_bound": r / prob.lam, "dps": dps}


# ----------------------------------------------------------------------------------------------------------------------
# runs
# ----------------------------------------------------------------------------------------------------------------------
def task_list():
    tasks = []
    for fam, dss in DATA_SEEDS.items():
        for ds in dss:
            for kind in ["I", "stat", "hess"]:
                tasks.append((("ndm", fam, ds, False, KAPPA), kind, SEEDS))
            for kind in ["I", "diag"]:
                tasks.append((("ndm", fam, ds, True, KAPPA), kind, SEEDS))
        tasks.append((("ndm", fam, 0, False, KAPPA), "diag", SEEDS))
        for kind in ["stat", "hess"]:
            tasks.append((("ndm", fam, 0, True, KAPPA), kind, SEEDS))
        for kind in ["I", "stat", "hess"]:
            tasks.append((("ndm", fam, 0, False, KAPPA_PROBE), kind, SEEDS_PROBE))
    for alpha in (1.0, 2.0):
        for kind in PRECONDS:
            tasks.append((("b1", alpha), kind, SEEDS))
    return tasks


def compact(prob, r):
    st = {k: np.array([q[k] for q in r["instr"]], dtype=np.float64) for k in STATE_KEYS}
    st["m_theta"] = np.array([float(segment_curvatures(prob, q["theta"]).min()) for q in r["instr"]])
    rec = {k: np.asarray(r[k], dtype=np.float64) for k in ("iters", "D", "D_P", "Fgap")}
    rec.update({"states": st, "seed": r["seed"], "gamma": r["gamma"], "R2": r["R2"], "mu": r["mu"],
                "r_init": r["r_init"], "stopped": r["stopped"]})
    return rec


def run_task(task, passes=PASSES):
    spec, kind, seeds = task
    t0 = time.time()
    prob = build(spec)
    P = preconditioner(prob, kind)
    cst = Metric(P, prob.d).constants(prob, theta_ref=prob.theta_star)
    runs = []
    for s in seeds:
        r = algorithms.saga(prob, np.zeros(prob.d), P=P, R2=cst["R2_P"], mu=cst["mu_P"], passes=passes, seed=s,
                            record_every=max(1, N // 20), instrument_every=max(1, N // 10), stop_D=1e-26,
                            name=f"P-SAGA[{kind}]", snapshot_theta=True)
        runs.append(compact(prob, r))
    return spec, kind, runs, prob.theta_star.copy(), time.time() - t0


def config():
    return dict(n=N, d=D, kappa=KAPPA, kappa_probe=KAPPA_PROBE, kap=KAP, max_row_norm=1.0, fix_signs=True,
                data_seeds=DATA_SEEDS, sampling_seeds=SEEDS, sampling_seeds_probe=SEEDS_PROBE, passes=PASSES,
                record_every=N // 20, instrument_every=N // 10, stop_D_P=1e-26, theta0="zeros", window_rate_passes=WINDOW,
                window_late_passes=LATE, floor_fit=FLOOR_FIT, rotation="Haar, numpy default_rng([20231, family, data seed])",
                solver="vrlab.solvers.solve_optimum (default); B1 self-check problems: armijo_only=True",
                step="gamma = 1/(4 R_P^2), R_P^2 = max_i ||x_i||^2_{P^-1}/4 + lam lmax(P^-1) (upper bound)")


def run_all(workers, rerun):
    res = None if rerun else common.load_result(TAG)
    if res is not None:
        print("loaded cached", TAG, flush=True)
        return res
    tasks = task_list()
    res = {"config": config(), "runs": {}, "eta": {}, "task_seconds": {}}
    t0 = time.time()
    print(f"{len(tasks)} tasks, {sum(len(t[2]) for t in tasks)} runs, {workers} worker(s)", flush=True)

    def store(out):
        spec, kind, runs, eta, el = out
        res["runs"][(spec, kind)] = runs
        res["task_seconds"][(spec, kind)] = el
        if spec in res["eta"]:
            assert np.array_equal(res["eta"][spec], eta), f"eta_* differs between tasks of {spec_name(spec)}"
        res["eta"][spec] = eta
        rates = [fit_rate(r["iters"], r["D_P"]) for r in runs]
        print(f"[{time.time() - t0:6.0f}s] {spec_name(spec):22s} {kind:4s} exponent/pass {np.mean(rates):.4f} "
              f"(min {np.min(rates):.4f}, max {np.max(rates):.4f}), last pass {max(r['iters'][-1] for r in runs) / N:.1f}  "
              f"({el:.0f}s)", flush=True)

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(run_task, t) for t in tasks]
            for f in as_completed(futs):
                store(f.result())
    else:
        for t in tasks:
            store(run_task(t))
    res["elapsed_runs_s"] = time.time() - t0
    common.save_result(TAG, res)
    return res


def problem_summaries(res):
    """Geometry, constants of every metric run on it and residual certificate, for every problem (parent process)."""
    out = {}
    specs = sorted({spec for (spec, _) in res["runs"]}, key=spec_name)
    for spec in specs:
        prob = build(spec)
        assert np.array_equal(prob.theta_star, res["eta"][spec]), f"eta_* not reproduced for {spec_name(spec)}"
        kinds = [k for k in PRECONDS if (spec, k) in res["runs"]]
        out[spec] = {"name": spec_name(spec), "geometry": geometry(prob), "certificate": certificate(prob),
                     "constants": {k: metric_constants(prob, k) for k in kinds}}
        g = out[spec]["geometry"]
        print(f"   {spec_name(spec):22s} d_eff={g['d_eff']:.2f} d_inf={g['d_inf']:.1f} C={g['C']:.2f} w_min={g['w_min']:.4f} "
              f"cond(C_corr)={g['corr_cond']:.3g} |F'(eta*)|_40={out[spec]['certificate']['grad_norm_mp']:.1e}", flush=True)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------------------------------------------------
def fit_rate(iters, y, window=WINDOW):
    """Exponent per pass: -n x least-squares slope of log y against t over the window (records above FLOOR_FIT)."""
    it = np.asarray(iters, dtype=float)
    y = np.asarray(y, dtype=float)
    sel = (it >= window[0] * N) & (it <= window[1] * N) & (y > FLOOR_FIT)
    return -float(np.polyfit(it[sel], np.log(y[sel]), 1)[0]) * N if sel.sum() >= 3 else float("nan")


def gm(x):
    x = np.asarray(x, dtype=float)
    return float(np.exp(np.mean(np.log(np.clip(x, 1e-300, None)))))


def run_stats(runs, cst, geo, kind):
    T = min(len(r["states"]["t"]) for r in runs)
    t = runs[0]["states"]["t"][:T]
    selw = (t >= WINDOW[0] * N) & (t <= WINDOW[1] * N)
    sell = (t >= LATE[0] * N) & (t <= LATE[1] * N)

    def pool(key, sel):
        return np.concatenate([np.asarray(r["states"][key][:T])[sel] for r in runs])

    def pool_all(key):
        return np.concatenate([np.asarray(r["states"][key]) for r in runs])

    rates = np.array([fit_rate(r["iters"], r["D_P"]) for r in runs])
    R2P, lam, s1, sd = cst["R2_P"], geo["lam"], geo["sigma_1"], geo["sigma_d"]
    mu_w, mu_l = pool("mu_eff", selw), pool("mu_eff", sell)
    s = {"n_runs": len(runs), "rate_pass_mean": float(rates.mean()), "rate_pass_min": float(rates.min()),
         "rate_pass_max": float(rates.max()), "rate_pass_seeds": rates.tolist(),
         "rate_pass_Fgap_mean": float(np.mean([fit_rate(r["iters"], r["Fgap"]) for r in runs])),
         "last_pass": [float(r["iters"][-1] / N) for r in runs], "stopped": [r["stopped"] for r in runs],
         "r_init": float(runs[0]["r_init"]), "n_states_2_20": int(selw.sum() * len(runs))}
    for k in RATIOS:
        s[k + "_gm_2_20"] = gm(pool(k, selw))
    s["r_sq_frac_above_half_2_20"] = float(np.mean(pool("r_sq", selw) > 0.5))
    s["one_minus_r_var_gm_2_20"] = gm(1.0 - pool("r_var", selw))
    s["mu_over_L_eff_gm_2_20"] = gm(mu_w / pool("L_eff", selw))
    s["kappa_eff_gm_15_20"] = gm(R2P / mu_l)
    s["kappa_eff_range_2_20"] = [float((R2P / mu_w).min()), float((R2P / mu_w).max())]
    s["mu_eff_over_lmin_M_gm_15_20"] = gm(mu_l) / cst["mu_P_local"]
    s["Dshare_median_2_20"] = float(np.median(pool("Dshare", selw)))
    s["exact_onestep_exponent_pass_mean_2_20"] = float(np.mean(-N * np.log(pool("rho_exact", selw))))
    m_w, m_all = pool("m_theta", selw), pool_all("m_theta")
    s["m_theta_range_2_20"] = [float(m_w.min()), float(m_w.max())]
    s["m_theta_min_all"] = float(m_all.min())
    s["n_required_traj"] = (4 * geo["d_inf"] + 16) / (9 * s["m_theta_min_all"])
    ke_all = R2P / pool_all("mu_eff")
    if kind == "stat":
        c0 = (geo["d_inf"] / 4 + lam / (sd + lam)) * (s1 + lam)
        b_w = c0 / (m_w * s1 + lam)
        b_all = c0 / (m_all * s1 + lam)
        ke_w = R2P / mu_w
        s["cor_constant_vs_R2P_rel"] = abs(geo["d_inf"] / 4 + lam / (sd + lam) - R2P) / R2P   # both are the upper-bound R_P^2
        s["cor_bound_2_20"] = [float(b_w.min()), float(np.median(b_w)), float(b_w.max())]
        s["cor_bound_gm_15_20"] = gm(c0 / (pool("m_theta", sell) * s1 + lam))
        s["cor_bound_over_kappa_eff_2_20"] = [float((b_w / ke_w).min()), float(np.median(b_w / ke_w)), float((b_w / ke_w).max())]
        s["cor_bound_over_kappa_eff_gm_15_20"] = s["cor_bound_gm_15_20"] / s["kappa_eff_gm_15_20"]
        s["cor_bound_frac_above_kappa_2_20"] = float(np.mean(b_w > geo["kappa"]))
        s["cor_violations_all_states"] = int(np.sum(ke_all > b_all * (1 + 1e-9)))
        s["trajsc_tightness_max_all_states"] = float(np.max((m_all * s1 + lam) / (s1 + lam) / pool_all("mu_eff")))
        s["n_states_all"] = int(m_all.size)
    if kind == "I":
        ub_w = geo["R2"] / (m_w * sd + lam)
        ub_all = geo["R2"] / (m_all * sd + lam)
        s["cor_I_upper_2_20"] = [float(ub_w.min()), float(np.median(ub_w)), float(ub_w.max())]
        s["cor_I_violations_all_states"] = int(np.sum(ke_all > ub_all * (1 + 1e-9)) + np.sum(ke_all < geo["cor_I_lower"] * (1 - 1e-9)))
    return s


def trace_rel_diff(runs_a, runs_b):
    """Largest relative difference of ||theta_t - eta_*||_P^2 between paired runs over the common records of passes 0-20
    (the fitting window and before; later records approach the rounding floor of the iterates)."""
    worst = 0.0
    for a, b in zip(runs_a, runs_b):
        L = min(len(a["D_P"]), len(b["D_P"]))
        ya, yb = a["D_P"][:L], b["D_P"][:L]
        ok = (ya > FLOOR_FIT) & (yb > FLOOR_FIT) & (a["iters"][:L] <= WINDOW[1] * N)
        worst = max(worst, float(np.max(np.abs(ya[ok] - yb[ok]) / yb[ok])))
    return worst


def rates_rel_diff(sa, sb):
    a, b = np.asarray(sa["rate_pass_seeds"]), np.asarray(sb["rate_pass_seeds"])
    return float(np.max(np.abs(a - b) / np.abs(b)))


def minmax(vals):
    vals = [float(v) for v in vals]
    return [min(vals), max(vals)]


def analyze(res, probs):
    runs = {}
    for (spec, kind), rr in res["runs"].items():
        p = probs[spec]
        runs[(spec, kind)] = run_stats(rr, p["constants"][kind], p["geometry"], kind)
    out = {"problems": {}, "runs": {}}
    for spec, p in probs.items():
        out["problems"][spec_name(spec)] = {"geometry": p["geometry"], "certificate": p["certificate"], "constants": p["constants"]}
    for (spec, kind), s in runs.items():
        base = runs.get((spec, "I"))
        if base is not None:
            s["gain_vs_I"] = s["rate_pass_mean"] / base["rate_pass_mean"]
            ratios = np.asarray(s["rate_pass_seeds"]) / np.asarray(base["rate_pass_seeds"])
            s["gain_vs_I_paired_range"] = [float(ratios.min()), float(ratios.max())]
        out["runs"][f"{spec_name(spec)}|{kind}"] = s

    # --- checks: Jacobi = Sigma_hat + lam I on the axis-aligned designs; rotation equivariance of I, stat, hess
    chk = {"jacobi_vs_stat_axis_aligned": {}, "rotation_equivariance": {}}
    for fam in DATA_SEEDS:
        spec = ("ndm", fam, 0, False, KAPPA)
        if (spec, "diag") in runs:
            chk["jacobi_vs_stat_axis_aligned"][spec_name(spec)] = {
                "rates_rel_diff_max": rates_rel_diff(runs[(spec, "diag")], runs[(spec, "stat")]),
                "trace_D_P_rel_diff_max": trace_rel_diff(res["runs"][(spec, "diag")], res["runs"][(spec, "stat")]),
                "stat_minus_jacobi_max_over_lmin_PJ": probs[spec]["geometry"]["stat_minus_jacobi_max_over_lmin_PJ"],
                "corr_lmin": probs[spec]["geometry"]["corr_lmin"], "corr_lmax": probs[spec]["geometry"]["corr_lmax"],
                "R2_P_rel_diff": abs(probs[spec]["constants"]["diag"]["R2_P"] - probs[spec]["constants"]["stat"]["R2_P"])
                / probs[spec]["constants"]["stat"]["R2_P"]}
        for ds in DATA_SEEDS[fam]:
            a, b = ("ndm", fam, ds, True, KAPPA), ("ndm", fam, ds, False, KAPPA)
            for kind in ("I", "stat", "hess"):
                if (a, kind) in runs and (b, kind) in runs:
                    chk["rotation_equivariance"][f"{spec_name(b)}|{kind}"] = {
                        "rates_rel_diff_max": rates_rel_diff(runs[(a, kind)], runs[(b, kind)]),
                        "trace_D_P_rel_diff_max": trace_rel_diff(res["runs"][(a, kind)], res["runs"][(b, kind)])}
    viol = [s.get("cor_violations_all_states", 0) for s in runs.values()] + [s.get("cor_I_violations_all_states", 0) for s in runs.values()]
    chk["corollary_violations_total"] = int(sum(viol))
    out["checks"] = chk

    # --- families at kappa = 1e3 (unrotated): ranges (min-max) over the data seeds
    fams = {}
    for fam, dss in DATA_SEEDS.items():
        specs = [("ndm", fam, ds, False, KAPPA) for ds in dss]
        G = [probs[sp]["geometry"] for sp in specs]
        f = {"data_seeds": dss}
        for key in ["d_eff", "d_inf", "C", "w_min", "w_median", "nlev_median", "coherence_U", "sigma_1", "mean_sqnorm",
                    "cor_stat_limit", "cor_I_limit", "n_required_limit"]:
            f[key] = minmax([g[key] for g in G])
        for kind in ["I", "stat", "hess"]:
            C = [probs[sp]["constants"][kind] for sp in specs]
            S = [runs[(sp, kind)] for sp in specs]
            fk = {}
            for key in ["R2_P", "R2_P_exact", "kappa_P", "kappa_P_exact", "lmin_M", "cond_M", "R2_P_over_lmin_M",
                        "crit_9n_lmin_M_over_16R2P", "rate_local_pass", "rate_certified_pass"]:
                fk[key] = minmax([c[key] for c in C])
            for key in ["rate_pass_mean", "kappa_eff_gm_15_20", "Dshare_median_2_20", "r_sq_gm_2_20", "r_coco_gm_2_20",
                        "r_var_gm_2_20", "r_sc_gm_2_20", "r_sq_frac_above_half_2_20", "r_init", "mu_eff_over_lmin_M_gm_15_20",
                        "exact_onestep_exponent_pass_mean_2_20", "m_theta_min_all", "n_required_traj"]:
                fk[key] = minmax([s[key] for s in S])
            fk["rate_pass_seed_range"] = [min(s["rate_pass_min"] for s in S), max(s["rate_pass_max"] for s in S)]
            if kind != "I":
                fk["gain_vs_I"] = minmax([s["gain_vs_I"] for s in S])
            if kind == "stat":
                fk["cor_bound_2_20"] = [min(s["cor_bound_2_20"][0] for s in S), max(s["cor_bound_2_20"][2] for s in S)]
                fk["cor_bound_gm_15_20"] = minmax([s["cor_bound_gm_15_20"] for s in S])
                fk["cor_bound_over_kappa_eff_min_2_20"] = minmax([s["cor_bound_over_kappa_eff_2_20"][0] for s in S])
                fk["cor_bound_over_kappa_eff_gm_15_20"] = minmax([s["cor_bound_over_kappa_eff_gm_15_20"] for s in S])
                fk["cor_bound_frac_above_kappa_2_20"] = minmax([s["cor_bound_frac_above_kappa_2_20"] for s in S])
                fk["kappa_eff_I_over_stat_15_20"] = minmax([runs[(sp, "I")]["kappa_eff_gm_15_20"] / runs[(sp, "stat")]["kappa_eff_gm_15_20"] for sp in specs])
            f[kind] = fk
        fams[fam] = f
    out["families"] = fams

    # --- rotated copies: Jacobi against P = I on the same rotated data
    rot = {}
    for fam, dss in DATA_SEEDS.items():
        for ds in dss:
            a = ("ndm", fam, ds, True, KAPPA)
            b = ("ndm", fam, ds, False, KAPPA)
            rot[spec_name(a)] = {"rate_diag": runs[(a, "diag")]["rate_pass_mean"], "rate_I": runs[(a, "I")]["rate_pass_mean"],
                                 "diag_over_I": runs[(a, "diag")]["rate_pass_mean"] / runs[(a, "I")]["rate_pass_mean"],
                                 "diag_over_I_paired_range": runs[(a, "diag")]["gain_vs_I_paired_range"],
                                 "rate_stat_unrotated": runs[(b, "stat")]["rate_pass_mean"],
                                 "corr_cond": probs[a]["geometry"]["corr_cond"],
                                 "jacobi_cond": probs[a]["geometry"]["jacobi_cond"], "stat_cond": probs[a]["geometry"]["stat_cond"],
                                 "kappa_P_diag": probs[a]["constants"]["diag"]["kappa_P"],
                                 "R2_P_over_lmin_M_diag": probs[a]["constants"]["diag"]["R2_P_over_lmin_M"],
                                 "cond_M_diag": probs[a]["constants"]["diag"]["cond_M"],
                                 "cond_M_I": probs[a]["constants"]["I"]["cond_M"]}
    out["rotated_jacobi"] = rot
    out["rotated_jacobi_ranges"] = {fam: minmax([rot[spec_name(("ndm", fam, ds, True, KAPPA))]["diag_over_I"] for ds in dss])
                                    for fam, dss in DATA_SEEDS.items()}

    # --- probe kappa = 1e4
    probe = {}
    for fam in DATA_SEEDS:
        sp = ("ndm", fam, 0, False, KAPPA_PROBE)
        g = probs[sp]["geometry"]
        pr = {"d_inf": g["d_inf"], "C": g["C"], "w_min": g["w_min"], "lam": g["lam"], "n_required_limit": g["n_required_limit"]}
        for kind in ["I", "stat", "hess"]:
            c, s = probs[sp]["constants"][kind], runs[(sp, kind)]
            pr[kind] = {"rate_pass_mean": s["rate_pass_mean"], "rate_pass_range": [s["rate_pass_min"], s["rate_pass_max"]],
                        "gain_vs_I": s.get("gain_vs_I"), "R2_P": c["R2_P"], "R2_P_over_lmin_M": c["R2_P_over_lmin_M"],
                        "crit_9n_lmin_M_over_16R2P": c["crit_9n_lmin_M_over_16R2P"], "rate_local_pass": c["rate_local_pass"],
                        "kappa_eff_gm_15_20": s["kappa_eff_gm_15_20"], "Dshare_median_2_20": s["Dshare_median_2_20"]}
        probe[spec_name(sp)] = pr
    lo, hi = probe[spec_name(("ndm", "lc", 0, False, KAPPA_PROBE))], probe[spec_name(("ndm", "hc", 0, False, KAPPA_PROBE))]
    probe["stat_rate_high_over_low"] = {"observed": hi["stat"]["rate_pass_mean"] / lo["stat"]["rate_pass_mean"],
                                        "local_constant_bound": hi["stat"]["rate_local_pass"] / lo["stat"]["rate_local_pass"]}
    out["probe_kappa_1e4"] = probe
    out["selfcheck_B1"] = selfcheck_B1(res, probs, runs)
    return out


def selfcheck_B1(res, probs, runs):
    """Rerun of the two problems of exp_B1_preconditioning against results/B1_summary.json and results/B1.pkl."""
    J = json.load(open(common.result_path("B1_summary", "json"), encoding="utf-8"))["preconditioners"]
    cache = common.load_result("B1")
    out = {"note": "rerun with OPENBLAS_NUM_THREADS=1: features equal to the cached ones up to their last bits"}
    worst = {"rate": 0.0, "constants": 0.0, "ratios": 0.0, "mu_eff_states_passes_0_20": 0.0, "kappa_eff_15_20": 0.0}
    for alpha in (1.0, 2.0):
        spec = ("b1", alpha)
        for kind in PRECONDS:
            ref, s, c = J[f"alpha={alpha:g}|{kind}"], runs[(spec, kind)], probs[spec]["constants"][kind]
            row = {"rate_pass_rerun": s["rate_pass_mean"], "rate_pass_B1_summary": ref["rate_emp_D_P"] * N}
            row["rate_rel_diff"] = abs(row["rate_pass_rerun"] - row["rate_pass_B1_summary"]) / row["rate_pass_B1_summary"]
            row["constants_rel_diff_max"] = max(abs(c[k] - ref[k]) / abs(ref[k]) for k in ("R2_P", "kappa_P", "mu_P_local", "kappa_P_local"))
            row["ratios_rel_diff_max"] = max(abs(s[k + "_gm_2_20"] - ref[k + "_geomean"]) / ref[k + "_geomean"] for k in RATIOS)
            if cache is not None:
                cr = cache["runs"][(alpha, kind)]
                rr = res["runs"][(spec, kind)]
                dmax = 0.0
                for a, b in zip(rr, cr):
                    mb = np.array([q["mu_eff"] for q in b["instr"]])
                    tb = np.array([q["t"] for q in b["instr"]])
                    L = min(len(mb), len(a["states"]["mu_eff"]))
                    sel = tb[:L] <= 20 * N
                    dmax = max(dmax, float(np.max(np.abs(a["states"]["mu_eff"][:L][sel] - mb[:L][sel]) / mb[:L][sel])))
                row["mu_eff_states_rel_diff_max_passes_0_20"] = dmax
                T = min(len(b["instr"]) for b in cr)
                t = np.array([q["t"] for q in cr[0]["instr"][:T]])
                sel = (t >= LATE[0] * N) & (t <= LATE[1] * N)
                mu = np.concatenate([np.array([q["mu_eff"] for q in b["instr"][:T]])[sel] for b in cr])
                R2c = cache["constants"][(alpha, kind)]["R2_P"]
                row["kappa_eff_gm_15_20_cached"] = gm(R2c / mu)
                row["kappa_eff_gm_15_20_rerun"] = s["kappa_eff_gm_15_20"]
                row["kappa_eff_rel_diff"] = abs(row["kappa_eff_gm_15_20_rerun"] - row["kappa_eff_gm_15_20_cached"]) / row["kappa_eff_gm_15_20_cached"]
                worst["mu_eff_states_passes_0_20"] = max(worst["mu_eff_states_passes_0_20"], dmax)
                worst["kappa_eff_15_20"] = max(worst["kappa_eff_15_20"], row["kappa_eff_rel_diff"])
            if kind == "stat":                                       # comment after Corollary B1-kappa-traj: seed 0
                r0 = res["runs"][(spec, kind)][0]
                g = probs[spec]["geometry"]
                c0 = (g["d_inf"] / 4 + g["lam"] / (g["sigma_d"] + g["lam"])) * (g["sigma_1"] + g["lam"])
                b = c0 / (r0["states"]["m_theta"] * g["sigma_1"] + g["lam"])
                t = r0["states"]["t"] / N
                w = (t >= 2) & (t <= 20)
                above = b > g["kappa"]
                first = next((float(t[k]) for k in range(len(t)) if above[k:].all()), None)
                row["cor_bound_seed0_2_20"] = [float(b[w].min()), float(b[w].max())]
                row["cor_bound_seed0_first_pass_above_kappa_for_good"] = first
                row["cor_stat_limit"] = g["cor_stat_limit"]
                row["cor_I_limit"] = g["cor_I_limit"]
                row["n_required_limit"] = g["n_required_limit"]
                row["w_min"] = g["w_min"]
            worst["rate"] = max(worst["rate"], row["rate_rel_diff"])
            worst["constants"] = max(worst["constants"], row["constants_rel_diff_max"])
            worst["ratios"] = max(worst["ratios"], row["ratios_rel_diff_max"])
            out[f"alpha={alpha:g}|{kind}"] = row
    out["worst_rel_diff"] = worst
    return out


def main():
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 3
    threads = os.environ.get("OPENBLAS_NUM_THREADS")
    if threads != "1":
        print(f"warning: OPENBLAS_NUM_THREADS={threads}; the results of the report were computed with OPENBLAS_NUM_THREADS=1", flush=True)
    if "--smoke" in sys.argv:
        for task in [(("ndm", "hc", 0, False, KAPPA), "stat", [0]), (("ndm", "lc", 0, True, KAPPA), "diag", [0])]:
            spec, kind, rr, eta, el = run_task(task, passes=21)
            print(spec_name(spec), kind, f"{fit_rate(rr[0]['iters'], rr[0]['D_P']):.4f}", f"{el:.1f}s")
        return
    t0 = time.time()
    res = run_all(workers, "--rerun" in sys.argv)
    if "problems" not in res or "--resummarize" in sys.argv:
        print("problem summaries (parent process):", flush=True)
        res["problems"] = problem_summaries(res)
        common.save_result(TAG, res)
    summary = analyze(res, res["problems"])
    summary = {"config": res["config"], "environment": {"OPENBLAS_NUM_THREADS": threads, "workers": workers,
                                                         "numpy": np.__version__, "elapsed_runs_s": res.get("elapsed_runs_s"),
                                                         "elapsed_total_s_this_call": time.time() - t0},
               "dropped": [], **summary}
    common.save_json(TAG + "_summary", summary)
    sc = summary["selfcheck_B1"]["worst_rel_diff"]
    print("self-check against results/B1_summary.json and B1.pkl (largest relative differences):", sc, flush=True)
    print("done", TAG, flush=True)


if __name__ == "__main__":
    main()
