"""Checks of Section exp-B2 of the report and of the B2 note on the cached logistic problems.

 (i)   Curvature along the ray tau -> tau eta_* through theta_0 = 0 (Remark B2-ray of the B2 note), on the fifteen problems
       of Section exp-B3 (n = 1000, d = 50, kappa in {10, 1e2, 1e3, 1e4, 1e5}, alpha in {0, 1, 2}, data seed 0; the nine
       problems of Section exp-B2 are those with kappa in {1e2, 1e3, 1e4}): lambda_min(F''(tau eta_*))/mu on a grid of tau
       in [0, 30] (step 0.005 on [0, 2], 0.1 on [2, 30]) and its monotonicity; the monotonicity in tau of every weight
       sigma'(tau y_i x_i^T eta_*) between consecutive grid points (which implies the Loewner order of the Hessians);
       mu_eff(theta_0)/mu at theta_0 = 0, computed as in exp_B2_mu_eff.py and as int_0^1 u^T F''(tau eta_*) u dtau
       (64-point Gauss-Legendre rule), against u^T F''(eta_*) u and u^T F''(0) u, u = eta_*/||eta_*||.
 (ii)  The GD run at alpha = 2, kappa = 1e2 (exp_B2_mu_eff.py: theta_0 = 0, step 1/L_F, at most 400 steps, stop below
       ||theta_t - eta_*||^2 = 1e-26), whose states below the floor 1e-18 include the ten with mu_eff < mu quoted in
       Section exp-B2: mu_eff at every state with the cached eta_* and with eta_c = eta_* - e1, e1 = F''(eta_*)^{-1}
       F'(eta_*) with F'(eta_*) in 40-digit arithmetic (vrlab.highprec.newton_correction), and the gradient F'(theta_t)
       in float64 and in 40-digit arithmetic. ||F'(eta_c)|| evaluated in 40-digit arithmetic at the unrounded
       eta_c (eta_* and e1 kept as two float64 vectors) bounds ||eta_c - eta_exact|| <= ||F'(eta_c)||/mu (strong convexity),
       which gives at every state an interval for mu_eff with respect to the exact minimizer.
 (iii) The GD steps on which mu_eff increases, over the kept states (||theta_t - eta||^2 >= 1e-18, eta the reference used),
       for the nine configurations of Section exp-B2, in particular alpha = 0, kappa = 1e3 and 1e4 (13 of 118 and 56 of 400
       steps in Section exp-B2), recounted with the four combinations of reference (cached, corrected) and gradient
       precision (float64, 40 digits) wherever the run comes close to eta_*; elsewhere the float64 values are compared with
       40-digit ones at five states of the run. For alpha = 0, kappa = 1e3 and 1e4, gradient descent is also run in
       40-digit arithmetic from theta_0 = 0 with the same step, which separates the rounding of the float64 iterates from
       the curvature of F.
The 40-digit gradients use the formula of vrlab.highprec.mp_logistic_grad on the stored float64 data (class MPGrad: data
converted once, dot products by mpmath.fdot), checked against vrlab.highprec.mp_logistic_grad; this is careful
high-precision evaluation, not interval arithmetic.
Problems, BLAS and solver as in the cached runs: vrlab.problems.make_logistic with the default OpenBLAS thread count (with
OPENBLAS_NUM_THREADS=1 the rotated features differ in their last bits) and newton_solve(armijo_only=True) (the solver of
commit 451da40); the script checks that the fifteen problems reproduce every float field of results/B3_summary.json and
that the float64 GD runs reproduce the trajectories of results/B2.pkl, bit for bit. Single process; needs mpmath.
Usage: python experiments/exp_B2_checks.py        Output: results/B2_checks_summary.json
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from vrlab import highprec, problems, solvers  # noqa: E402

N, D = 1000, 50
KAPPAS = [10.0, 100.0, 1000.0, 10000.0, 100000.0]       # Section exp-B3
ALPHAS = [0.0, 1.0, 2.0]
B2_KAPPAS = [100.0, 1000.0, 10000.0]                     # Section exp-B2
GD_ITERS, FLOOR, STOP = 400, 1e-18, 1e-26                # as exp_B2_mu_eff.py
DPS = 40
TAU = np.unique(np.concatenate([np.linspace(0.0, 2.0, 401), np.linspace(2.0, 30.0, 281), [1.0]]))
TAU_REPORT = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0]
GD_MP = [(0.0, 1000.0), (0.0, 10000.0)]                  # gradient descent repeated in 40-digit arithmetic
TAG = "B2_checks"


def key(alpha, kappa):
    return f"alpha={alpha:g},kappa={kappa:g}"


def rebuild(alpha, kappa):
    prob = problems.make_logistic(N, D, kappa=kappa, alpha=alpha, seed=0)
    solvers.solve_optimum(prob, armijo_only=True)
    return prob


class MPGrad:
    """F'(theta) in `dps`-digit arithmetic on the stored float64 data, with the formula of
    vrlab.highprec.mp_logistic_grad, at a point given exactly as a list of mpmath numbers (data converted once)."""

    def __init__(self, prob, dps=DPS):
        import mpmath as mp
        self.mp, self.dps, self.n = mp, dps, prob.n
        with mp.workdps(dps):
            self.rows = [[mp.mpf(float(v)) for v in row] for row in prob.X]
            self.cols = [list(c) for c in zip(*self.rows)]
            self.y = [mp.mpf(float(v)) for v in prob.y]
            self.lam = mp.mpf(float(prob.lam))

    def __call__(self, th):
        mp = self.mp
        with mp.workdps(self.dps):
            c = [-yi / (1 + mp.exp(yi * mp.fdot(xi, th))) for xi, yi in zip(self.rows, self.y)]
            return [mp.fdot(c, col) / self.n + self.lam * t for col, t in zip(self.cols, th)]

    def exact(self, *vectors):
        """The exact sum of float64 vectors as mpmath numbers."""
        mp = self.mp
        with mp.workdps(self.dps):
            return [mp.fsum(mp.mpf(float(v[j])) for v in vectors) for j in range(len(vectors[0]))]

    def norm(self, g):
        mp = self.mp
        with mp.workdps(self.dps):
            return mp.sqrt(mp.fsum(x * x for x in g))

    def grad64(self, theta):
        """40-digit gradient at the float64 point theta, rounded once to float64."""
        return np.array([float(x) for x in self(self.exact(theta))])


# ----------------------------------------------------------------------------------------------------------------------
# (i) the ray through theta_0 = 0
# ----------------------------------------------------------------------------------------------------------------------
def ray_stats(prob, b2=None):
    eta, lam = prob.theta_star, prob.lam
    u = eta / np.linalg.norm(eta)
    lmin = np.empty(len(TAU))
    W = np.empty((len(TAU), prob.n))
    for k, tau in enumerate(TAU):
        th = tau * eta
        W[k] = prob.sigma_prime(th)
        lmin[k] = np.linalg.eigvalsh(prob.hessian(th))[0]
    dl, dW = np.diff(lmin), np.diff(W, axis=0)
    theta0 = np.zeros(prob.d)
    v = theta0 - eta
    mu0 = float(v @ prob.grad(theta0)) / float(v @ v)               # as exp_B2_mu_eff.curvature_stats
    c2 = (prob.X @ u) ** 2
    xg, wg = np.polynomial.legendre.leggauss(64)
    quad = float(sum(0.5 * wk * (lam + np.mean(prob.sigma_prime(0.5 * (xk + 1.0) * eta) * c2)) for xk, wk in zip(xg, wg)))
    ray_star = float(lam + np.mean(prob.sigma_prime(eta) * c2))      # u^T F''(eta_*) u
    ray_zero = float(lam + np.mean(c2) / 4.0)                        # u^T F''(0) u
    idx = {t: int(np.argmin(np.abs(TAU - t))) for t in TAU_REPORT}
    out = {
        "mu": lam, "n_grid": int(len(TAU)),
        "lmin_over_mu": {f"{t:g}": float(lmin[i] / lam) for t, i in idx.items()},
        "lmin_tau0_formula_over_mu": float((prob.cov_eigs[0] / 4.0 + lam) / lam),   # lmin(Sigma_hat)/4 + lam
        "n_lmin_increases": int(np.sum(dl > 0)),
        "max_rel_lmin_increase": float(np.max(dl / lmin[:-1])) if np.any(dl > 0) else 0.0,
        "n_weight_increases": int(np.sum(dW > 0)), "max_weight_increase": float(max(np.max(dW), 0.0)),
        "min_abs_margin_at_eta": float(np.min(np.abs(prob.y * (prob.X @ eta)))),
        "mu_eff_theta0_over_mu": mu0 / lam, "mu_eff_theta0_quadrature_over_mu": quad / lam,
        "quadrature_rel_diff": abs(quad - mu0) / mu0,
        "rayleigh_H_star_dir_eta_over_mu": ray_star / lam, "rayleigh_F0_dir_eta_over_mu": ray_zero / lam,
        "mu_eff_theta0_over_rayleigh_H_star": mu0 / ray_star,
        "ordering_holds": bool(ray_star <= mu0 * (1 + 1e-12) and mu0 <= ray_zero * (1 + 1e-12)),
        "lmin_H_star_over_mu": float(lmin[idx[1.0]] / lam),
    }
    if b2 is not None:                                               # the same quantities in results/B2_summary.json
        out["B2_summary_saga_mu_eff_over_mu_start"] = b2["saga_mu_eff_over_mu_start"]
        out["B2_summary_rayleigh_H_start_over_mu"] = b2["rayleigh_H_start_over_mu"]
        out["mu0_equals_B2_summary"] = bool(mu0 / lam == b2["saga_mu_eff_over_mu_start"])
        out["rayleigh_rel_diff_B2_summary"] = abs(ray_star / lam - b2["rayleigh_H_start_over_mu"]) / b2["rayleigh_H_start_over_mu"]
    return out


# ----------------------------------------------------------------------------------------------------------------------
# (ii)-(iii) gradient descent
# ----------------------------------------------------------------------------------------------------------------------
def gd_float64(prob):
    """GD exactly as exp_B2_mu_eff.run_all (state recorded before the stopping test)."""
    eta = prob.theta_star
    theta = np.zeros(prob.d)
    TH, MU, DD = [], [], []
    for _ in range(GD_ITERS + 1):
        v = theta - eta
        Dv = float(v @ v)
        G = float(v @ prob.grad(theta))
        TH.append(theta.copy())
        MU.append(G / Dv)
        DD.append(Dv)
        if Dv < STOP:
            break
        theta -= prob.grad(theta) / prob.L_F
    return np.array(TH), np.array(MU), np.array(DD)


def increases(mu, Dv, kept):
    """Steps between consecutive kept states on which mu_eff increases."""
    m, d_ = mu[kept], Dv[kept]
    dm = np.diff(m)
    up = dm > 0
    return {"n_kept_states": int(kept.sum()), "n_steps": int(max(kept.sum() - 1, 0)), "n_increasing_steps": int(up.sum()),
            "max_rel_increase": float(np.max(dm[up] / m[:-1][up])) if up.any() else None,
            "D_max_after_increase": float(np.max(d_[1:][up])) if up.any() else None,
            "D_min_after_increase": float(np.min(d_[1:][up])) if up.any() else None}


def gd_analysis(prob, TH, D64, e1, eps_c, mpg, hp_all):
    """mu_eff at every GD state for the four combinations of reference and gradient precision; with hp_all=False the
    40-digit gradient is evaluated at five states only (comparison with float64)."""
    eta, lam = prob.theta_star, prob.lam
    T = len(TH)
    V = TH - eta[None, :]
    VC = V + e1[None, :]                                             # theta_t - eta_c, eta_c = eta_* - e1 unrounded
    G64 = np.array([prob.grad(th) for th in TH])
    mu = {"cached|f64": np.einsum("ij,ij->i", V, G64) / np.einsum("ij,ij->i", V, V),
          "corrected|f64": np.einsum("ij,ij->i", VC, G64) / np.einsum("ij,ij->i", VC, VC)}
    Dc = np.einsum("ij,ij->i", VC, VC)
    out = {"n_states": T, "D_cached_range": [float(D64.min()), float(D64.max())], "D_corrected_min": float(Dc.min())}
    if hp_all:
        t0 = time.time()
        GHP = np.array([mpg.grad64(th) for th in TH])
        out["seconds_40_digit_gradients"] = time.time() - t0
        mu["cached|40d"] = np.einsum("ij,ij->i", V, GHP) / np.einsum("ij,ij->i", V, V)
        mu["corrected|40d"] = np.einsum("ij,ij->i", VC, GHP) / Dc
        # interval for mu_eff with respect to the exact minimizer: ||eta_c - eta_exact|| <= eps_c
        num = np.einsum("ij,ij->i", VC, GHP)
        gn, vn = np.linalg.norm(GHP, axis=1), np.sqrt(Dc)
        lower = (num - gn * eps_c) / (vn + eps_c) ** 2 * (1 - 1e-13)
        upper = np.where(vn > eps_c, (num + gn * eps_c) / np.maximum(vn - eps_c, 1e-300) ** 2 * (1 + 1e-13), np.inf)
        out["exact_reference_interval"] = {
            "eps_c": eps_c, "min_lower_over_mu": float(lower.min() / lam), "max_rel_halfwidth": float(np.max((upper - lower) / (upper + lower))),
            "n_states_certified_above_mu": int(np.sum(lower > lam))}
        out["f64_vs_40d_mu_eff_rel_diff_max_kept_cached"] = float(np.max(np.abs(mu["cached|f64"] - mu["cached|40d"])[D64 >= FLOOR]
                                                                          / mu["cached|40d"][D64 >= FLOOR]))
    else:
        pick = sorted({0, T // 4, T // 2, (3 * T) // 4, T - 1})
        GHP = np.array([mpg.grad64(TH[k]) for k in pick])
        mh = np.einsum("ij,ij->i", VC[pick], GHP) / Dc[pick]
        rel = np.abs(mu["corrected|f64"][pick] - mh) / mh
        dm = np.abs(np.diff(mu["corrected|f64"])) / mu["corrected|f64"][:-1]
        out["f64_check_at_states"] = {"states": pick, "rel_diff_f64_vs_40d": rel.tolist(), "max_rel_diff": float(rel.max()),
                                      "min_abs_rel_step_change": float(dm.min()) if len(dm) else None}
    for name, m in mu.items():
        ref = name.split("|")[0]
        Dref = D64 if ref == "cached" else Dc
        below = Dref < FLOOR
        neg = m < lam
        out[name] = {"n_states_mu_eff_below_mu": int(neg.sum()), "n_below_floor": int(below.sum()),
                     "n_below_floor_mu_eff_below_mu": int((neg & below).sum()),
                     "min_mu_eff_over_mu_all": float(m.min() / lam),
                     "min_mu_eff_over_mu_kept": float(m[~below].min() / lam),
                     "min_mu_eff_over_mu_below_floor": float(m[below].min() / lam) if below.any() else None,
                     "range_mu_eff_over_mu_where_below_mu": [float(m[neg].min() / lam), float(m[neg].max() / lam)] if neg.any() else None,
                     "iterations_mu_eff_below_mu": np.flatnonzero(neg).tolist(),
                     "increases": increases(m, Dref, ~below)}
    return out


def gd_mp(prob, mpg, e1, n_max):
    """Gradient descent in DPS-digit arithmetic from theta_0 = 0 with the step 1/L_F (L_F the stored float64 value); mu_eff
    relative to the unrounded eta_c; stops at the first state below the floor or after n_max steps."""
    mp = mpg.mp
    lam = prob.lam
    with mp.workdps(DPS):
        eta_c = mpg.exact(prob.theta_star, -e1)
        L = mp.mpf(float(prob.L_F))
        th = [mp.mpf(0)] * prob.d
        mus, Ds, TH = [], [], []
        for _ in range(n_max + 1):
            g = mpg(th)
            v = [a - b for a, b in zip(th, eta_c)]
            Dv = mp.fsum(x * x for x in v)
            if Dv < FLOOR:
                break
            mus.append(mp.fdot(v, g) / Dv)
            Ds.append(Dv)
            TH.append(np.array([float(x) for x in th]))
            th = [a - gj / L for a, gj in zip(th, g)]
        dm = [mus[k + 1] - mus[k] for k in range(len(mus) - 1)]
        up = [k for k, x in enumerate(dm) if x > 0]
        return {"n_kept_states": len(mus), "n_steps": len(mus) - 1, "n_increasing_steps": len(up),
                "max_rel_increase": float(max(dm[k] / mus[k] for k in up)) if up else None,
                "min_mu_eff_over_mu": float(min(mus) / lam), "end_mu_eff_over_mu": float(mus[-1] / lam),
                "D_end": float(Ds[-1]), "_TH": np.array(TH)}


def main():
    t_start = time.time()
    threads = os.environ.get("OPENBLAS_NUM_THREADS")
    if threads is not None:
        print(f"warning: OPENBLAS_NUM_THREADS={threads}; the cached problems were built with the default thread count", flush=True)
    B3 = json.load(open(common.result_path("B3_summary", "json"), encoding="utf-8"))["problems"]
    B2s = json.load(open(common.result_path("B2_summary", "json"), encoding="utf-8"))
    B2c = common.load_result("B2")
    out = {"config": dict(n=N, d=D, kappas_B3=KAPPAS, alphas=ALPHAS, kappas_B2=B2_KAPPAS, gd_max_iters=GD_ITERS,
                          floor_D=FLOOR, stop_D=STOP, dps=DPS, tau_grid="0:0.005:2 and 2:0.1:30",
                          solver="newton_solve(armijo_only=True)", OPENBLAS_NUM_THREADS=threads or "default"),
           "rebuild": {}, "ray": {}, "gd": {}}

    # ---- rebuild the fifteen problems and check them against results/B3_summary.json
    probs = {}
    for kappa in KAPPAS:
        for alpha in ALPHAS:
            prob = rebuild(alpha, kappa)
            probs[(alpha, kappa)] = prob
            ref = B3[f"kappa={kappa:g},alpha={alpha:g}"]
            s = prob.summary()
            fields = [k for k, v in ref.items() if isinstance(v, float)]
            out["rebuild"][key(alpha, kappa)] = {"n_float_fields": len(fields), "mismatches": [k for k in fields if s[k] != ref[k]]}
    n_mis = sum(len(v["mismatches"]) for v in out["rebuild"].values())
    print(f"rebuild: {len(probs)} problems, {n_mis} float fields differ from results/B3_summary.json", flush=True)

    # ---- (i) the ray
    for (alpha, kappa), prob in probs.items():
        r = ray_stats(prob, B2s.get(key(alpha, kappa)))
        out["ray"][key(alpha, kappa)] = r
        print(f"(i) {key(alpha, kappa):22s} lmin(F''(tau eta))/mu: tau=0 {r['lmin_over_mu']['0']:.4g}, 1 {r['lmin_over_mu']['1']:.4g}, "
              f"30 {r['lmin_over_mu']['30']:.6g}; increases {r['n_lmin_increases']}, weight increases {r['n_weight_increases']} "
              f"(max {r['max_weight_increase']:.1e}); mu_eff(0)/mu {r['mu_eff_theta0_over_mu']:.4g} >= u'H*u/mu "
              f"{r['rayleigh_H_star_dir_eta_over_mu']:.4g}: {r['ordering_holds']}", flush=True)
    R = out["ray"]
    b2keys = [key(a, k) for a in ALPHAS for k in B2_KAPPAS]
    out["ray_summary"] = {
        "all_lmin_nonincreasing": all(R[k]["n_lmin_increases"] == 0 for k in R),
        "max_weight_increase": max(R[k]["max_weight_increase"] for k in R),
        "all_orderings_hold": all(R[k]["ordering_holds"] for k in R),
        "max_quadrature_rel_diff": max(R[k]["quadrature_rel_diff"] for k in R),
        "lmin_tau30_over_mu_range": [min(R[k]["lmin_over_mu"]["30"] for k in R), max(R[k]["lmin_over_mu"]["30"] for k in R)],
        "B2_rayleigh_H_star_dir_eta_over_mu_range": [min(R[k]["rayleigh_H_star_dir_eta_over_mu"] for k in b2keys),
                                                     max(R[k]["rayleigh_H_star_dir_eta_over_mu"] for k in b2keys)],
        "B2_mu_eff_theta0_over_mu_range": [min(R[k]["mu_eff_theta0_over_mu"] for k in b2keys), max(R[k]["mu_eff_theta0_over_mu"] for k in b2keys)],
        "B2_mu_eff_theta0_over_rayleigh_range": [min(R[k]["mu_eff_theta0_over_rayleigh_H_star"] for k in b2keys),
                                                 max(R[k]["mu_eff_theta0_over_rayleigh_H_star"] for k in b2keys)],
        "B2_mu0_equal_B2_summary_all": all(R[k].get("mu0_equals_B2_summary", False) for k in b2keys),
        "all15_mu_eff_theta0_over_rayleigh_range": [min(R[k]["mu_eff_theta0_over_rayleigh_H_star"] for k in R),
                                                    max(R[k]["mu_eff_theta0_over_rayleigh_H_star"] for k in R)],
    }
    print("(i) summary:", out["ray_summary"], flush=True)

    # ---- check of the 40-digit gradient class against vrlab.highprec
    p0 = probs[(2.0, 100.0)]
    mpg0 = MPGrad(p0)
    g_ref, r_ref = highprec.mp_logistic_grad(p0.X, p0.y, p0.lam, p0.theta_star, DPS)
    out["mpgrad_check"] = {"max_abs_diff_vs_highprec_at_eta": float(np.max(np.abs(mpg0.grad64(p0.theta_star) - g_ref))),
                           "grad_norm_highprec_at_eta": r_ref}

    # ---- (ii)-(iii) gradient descent on the nine configurations of Section exp-B2
    for alpha in ALPHAS:
        for kappa in B2_KAPPAS:
            t0 = time.time()
            prob = probs[(alpha, kappa)]
            k = key(alpha, kappa)
            TH, MU, DD = gd_float64(prob)
            cg = B2c["gd"][(alpha, kappa)]
            rep = {"bitwise_equal_B2_pkl": bool(len(MU) == len(cg["mu_eff"]) and np.array_equal(MU, cg["mu_eff"]) and np.array_equal(DD, cg["D"]))}
            mpg = mpg0 if (alpha, kappa) == (2.0, 100.0) else MPGrad(prob)
            e1, r = highprec.newton_correction(prob, prob.theta_star, DPS)
            r_c = float(mpg.norm(mpg(mpg.exact(prob.theta_star, -e1))))
            rep["reference"] = {"grad_norm_cached_40d": r, "e1_norm": float(np.linalg.norm(e1)), "e1_norm_sq": float(e1 @ e1),
                                "dist_bound_cached": r / prob.lam, "grad_norm_corrected_40d": r_c, "eps_c": r_c / prob.lam}
            close = bool(DD.min() < 1e-12)                            # the run comes close to eta_*: 40-digit gradients at every state
            rep.update(gd_analysis(prob, TH, DD, e1, r_c / prob.lam, mpg, hp_all=close))
            rep["lmin_H_over_mu"] = B2s[k]["lmin_H_over_mu"]
            rep["B2_summary"] = {kk: B2s[k][kk] for kk in ["gd_mu_eff_increasing_steps", "gd_mu_eff_increase_max_rel", "gd_D_max_after_increase",
                                                          "gd_mu_eff_over_mu_min", "gd_mu_eff_over_mu_min_all_states", "gd_end_iter"]}
            if (alpha, kappa) in GD_MP:
                g = gd_mp(prob, mpg, e1, GD_ITERS)
                THm = g.pop("_TH")
                L = min(len(THm), len(TH))
                dev = np.linalg.norm(THm[:L] - TH[:L], axis=1) / np.sqrt(DD[:L])
                g["max_dev_float64_iterates_over_error"] = float(dev.max())
                rep["gd_40_digit"] = g
            out["gd"][k] = rep
            inc = {nm: rep[nm]["increases"]["n_increasing_steps"] for nm in rep if "|" in nm}
            print(f"(ii-iii) {k:22s} bitwise={rep['bitwise_equal_B2_pkl']} states={rep['n_states']} ||e1||={rep['reference']['e1_norm']:.2e} "
                  f"eps_c={rep['reference']['eps_c']:.1e} increases {inc} below-mu "
                  f"{ {nm: rep[nm]['n_states_mu_eff_below_mu'] for nm in rep if '|' in nm} }"
                  + (f" | 40-digit GD: {rep['gd_40_digit']['n_increasing_steps']} increases in {rep['gd_40_digit']['n_steps']} steps" if "gd_40_digit" in rep else "")
                  + f" ({time.time() - t0:.0f}s)", flush=True)
    out["elapsed_s"] = time.time() - t_start
    common.save_json(TAG + "_summary", out)
    print(f"done {TAG} ({out['elapsed_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
