"""Hypothesis B1: SAGA in a quadratic metric P (theta_t = theta_{t-1} - gamma P^{-1} g_t) on anisotropic logistic
regression, with P in {I, diag(Sigma_hat) + lam I, Sigma_hat + lam I, F''(eta_*)}. The transferred Prop. 5.9 gives
the rate 1 - min{1/(3n), 3 mu_P/(16 R_P^2)} with R_P^2 = max_i ||x_i||^2_{P^-1}/4 + lam lmax(P^-1) and the global
mu_P = lam lmin(P^-1); the local mu_P at eta_* is the smallest generalized eigenvalue of (F''(eta_*), P).
This R_P^2 is the upper bound (eq. B1-logistic-RP of the B1 note) of the smoothness constant of Definition B1-kappaP,
exact for P = I and at most 5.2% above it here; it is used both for kappa_P and for the step gamma = 1/(4 R_P^2).
Usage: python experiments/exp_B1_preconditioning.py [--quick]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import plotstyle as ps  # noqa: E402
from vrlab import algorithms, analysis, bounds  # noqa: E402
from vrlab.instrumentation import Metric  # noqa: E402

QUICK = "--quick" in sys.argv
N, D = (300, 20) if QUICK else (1000, 50)
KAPPA = 1e3
ALPHAS = [1.0, 2.0] if not QUICK else [2.0]
SEEDS = list(range(2 if QUICK else 5))
PASSES = 30 if QUICK else 100
TAG = "B1_quick" if QUICK else "B1"
PRECONDS = ["I", "diag", "stat", "hess"]


def preconditioner(prob, kind):
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


def run_all():
    res = common.load_result(TAG)
    if res is not None:
        print("loaded cached", TAG)
        return res
    res = {"config": dict(n=N, d=D, kappa=KAPPA, alphas=ALPHAS, seeds=SEEDS, passes=PASSES), "runs": {}, "constants": {}, "problems": {}}
    for alpha in ALPHAS:
        prob = common.logistic_problem(N, D, KAPPA, alpha, seed=0)
        res["problems"][alpha] = prob.summary()
        theta0 = np.zeros(D)
        for kind in PRECONDS:
            P = preconditioner(prob, kind)
            metric = Metric(P, D)
            cst = metric.constants(prob, theta_ref=prob.theta_star)
            cst["rho_bound_global"] = bounds.saga_rate(N, cst["R2_P"], cst["mu_P"])
            cst["rho_bound_local"] = bounds.saga_rate(N, cst["R2_P"], cst["mu_P_local"])
            res["constants"][(alpha, kind)] = cst
            runs = []
            with common.Timer(f"alpha={alpha:g} P={kind}  R2_P={cst['R2_P']:.3g} mu_P={cst['mu_P']:.3g} (local {cst['mu_P_local']:.3g}) kappa_P={cst['kappa_P']:.3g} (local {cst['kappa_P_local']:.3g})"):
                for s in SEEDS:
                    r = algorithms.saga(prob, theta0, P=P, R2=cst["R2_P"], mu=cst["mu_P"], passes=PASSES, seed=s,
                                        record_every=max(1, N // 20), instrument_every=max(1, N // 10), stop_D=1e-26, name=f"P-SAGA[{kind}]")
                    runs.append(common.strip_run(r))
            res["runs"][(alpha, kind)] = runs
            print("   final F-F*:", [f"{r['Fgap'][-1]:.1e}" for r in runs], flush=True)
    common.save_result(TAG, res)
    return res


def empirical_rate(runs, key="D_P", window=(2, 20)):
    """Least-squares slope of log(key) per iteration over the window (in passes), averaged over seeds."""
    slopes = []
    for r in runs:
        it = np.asarray(r["iters"], dtype=float)
        y = np.asarray(r[key], dtype=float)
        sel = (it >= window[0] * N) & (it <= window[1] * N) & (y > 1e-25)
        if sel.sum() >= 3:
            slopes.append(np.polyfit(it[sel], np.log(y[sel]), 1)[0])
    return -float(np.mean(slopes)) if slopes else np.nan   # contraction exponent per iteration


def plot(res):
    fig = ps.figure(ps.FIG_W, 5.2)
    gs = fig.add_gridspec(2, len(ALPHAS), height_ratios=[1.0, 0.8])   # top: one panel per alpha; bottom: the rates
    summary = {}
    for ai, alpha in enumerate(ALPHAS):
        ax = fig.add_subplot(gs[0, ai])
        for kind in PRECONDS:
            runs = res["runs"][(alpha, kind)]
            it, meanF, allF = analysis.mean_over_seeds(runs, "Fgap")
            ax.semilogy(it / N, np.maximum(meanF, 1e-17), color=ps.PRECOND[kind], label=ps.PRECOND_LABEL[kind])
            ps.band(ax, it / N, np.maximum(allF.min(0), 1e-17), np.maximum(allF.max(0), 1e-17), ps.PRECOND[kind])
            cst = res["constants"][(alpha, kind)]
            summary[f"alpha={alpha:g}|{kind}"] = dict(cst, rate_emp_D_P=empirical_rate(runs, "D_P"), rate_emp_Fgap=empirical_rate(runs, "Fgap"),
                                                       rate_bound_global=1 - cst["rho_bound_global"], rate_bound_local=1 - cst["rho_bound_local"])
            for rk in ["r_sq", "r_coco", "r_var", "r_sc", "r_min"]:
                T = min(len(r["instr"]) for r in runs)
                X = np.array([[q[rk] for q in r["instr"][:T]] for r in runs])
                t = np.array([q["t"] for q in runs[0]["instr"][:T]])
                sel = (t >= 2 * N) & (t <= 20 * N)
                summary[f"alpha={alpha:g}|{kind}"][rk + "_geomean"] = float(np.exp(np.log(np.clip(X[:, sel], 1e-300, None)).mean()))
        ax.set_title(f"κ = 10³, α = {alpha:g}\n(cov. condition number {D ** alpha:g})")
        ax.set_xlabel("effective passes")
        ax.set_ylabel("F(θₜ) − F*  (mean over seeds)")
        ax.set_ylim(1e-17, 10)
        ps.legend(ax, loc="upper right")
    # rates panel (last alpha, full width below): empirical vs bounds
    alpha = ALPHAS[-1]
    ax = fig.add_subplot(gs[1, :])
    x = np.arange(len(PRECONDS))
    w = 0.26
    emp = [summary[f"alpha={alpha:g}|{k}"]["rate_emp_D_P"] * N for k in PRECONDS]
    bg = [summary[f"alpha={alpha:g}|{k}"]["rate_bound_global"] * N for k in PRECONDS]
    bl = [summary[f"alpha={alpha:g}|{k}"]["rate_bound_local"] * N for k in PRECONDS]
    ax.bar(x - w, emp, width=w - 0.03, color=ps.BLUE, label="empirical (slope of log‖θ−η*‖²_P)")
    ax.bar(x, bl, width=w - 0.03, color=ps.ORANGE, label="Prop. 5.9 with local μ_P = λmin(F''(η*), P)")
    ax.bar(x + w, bg, width=w - 0.03, color=ps.AQUA, label="Prop. 5.9 with global μ_P")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([ps.PRECOND_LABEL[k] for k in PRECONDS])
    ax.set_ylim(top=8 * float(np.nanmax(emp + bl + bg)))   # headroom: the legend sits above the bars
    ax.set_ylabel("contraction exponent per pass")
    ax.set_title(f"rates: empirical vs transferred bound (α = {alpha:g})")
    ps.legend(ax, loc="upper left")
    ps.suptitle(fig, "Figure B1 — preconditioned SAGA: convergence and the transferred Proposition 5.9 constants\n"
                "for different metrics P")
    ps.savefig(fig, "figB1_preconditioned_saga")
    return summary


def main():
    ps.setup()
    res = run_all()
    summary = plot(res)
    common.save_json(TAG + "_summary", {"problems": {f"alpha={a:g}": v for a, v in res["problems"].items()}, "preconditioners": summary})
    print("done B1")


if __name__ == "__main__":
    main()
