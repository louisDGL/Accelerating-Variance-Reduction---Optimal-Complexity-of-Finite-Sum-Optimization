"""Hypothesis B2: trajectory-dependent constants. Along SAGA and GD trajectories we measure
  mu_eff(theta) = (theta - eta_*)^T F'(theta) / ||theta - eta_*||^2        (>= mu, -> Rayleigh quotient of F''(eta_*))
  L_eff(theta)  = (1/n) sum_i ||f'_i(theta) - f'_i(eta_*)||^2 / ((theta - eta_*)^T F'(theta))   (<= R^2)
and the spectral decomposition of the error theta - eta_* in the eigenbasis of F''(eta_*).
Usage: python experiments/exp_B2_mu_eff.py [--quick]
"""
import os
import sys

import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import plotstyle as ps  # noqa: E402
from vrlab import algorithms  # noqa: E402

QUICK = "--quick" in sys.argv
N, D = (300, 20) if QUICK else (1000, 50)
KAPPAS = [1e2, 1e3, 1e4] if not QUICK else [1e2, 1e4]
ALPHAS = [0.0, 1.0, 2.0] if not QUICK else [1.0]
PASSES = 20 if QUICK else 60
GD_ITERS = 100 if QUICK else 400
TAG = "B2_quick" if QUICK else "B2"
FLOOR = 1e-18          # states with ||theta - eta_*||^2 below this are discarded (precision of eta_*)


def curvature_stats(prob, theta, U, k_low=(1, 5), w=None):
    """mu_eff, L_eff, spectral profile pi_k = (q_k^T v)^2/||v||^2 of the error in the eigenbasis U of F''(eta_*):
    energy on the k lowest eigenvectors, effective number of directions d_eff = 1/sum pi_k^2 (B2 note, (P1)-(P2)),
    and the quadratic proxy sum_k lambda_k pi_k of mu_eff."""
    v = theta - prob.theta_star
    D = float(v @ v)
    G = float(v @ prob.grad(theta))
    S = float(np.sum((prob.all_grads(theta) - prob.grads_star) ** 2))
    c = U.T @ v
    pi = c ** 2 / D
    out = {"D": D, "mu_eff": G / D, "L_eff": S / (prob.n * G), "d_eff_dirs": float(1.0 / np.sum(pi ** 2))}
    if w is not None:
        out["mu_eff_quadratic_proxy"] = float(np.sum(w * pi))
    for k in k_low:
        out[f"energy_low{k}"] = float(np.sum(pi[:k]))
    return out


def run_all():
    res = common.load_result(TAG)
    if res is not None:
        print("loaded cached", TAG)
        return res
    res = {"config": dict(n=N, d=D, kappas=KAPPAS, alphas=ALPHAS, passes=PASSES, gd_iters=GD_ITERS), "saga": {}, "gd": {}, "spectra": {}}
    for alpha in ALPHAS:
        for kappa in KAPPAS:
            prob = common.logistic_problem(N, D, kappa, alpha, seed=0)
            H = prob.hessian(prob.theta_star)
            w, U = np.linalg.eigh(H)
            Hsq = prob.hessian_i_sq_mean(prob.theta_star)
            gen = eigh(Hsq, H, eigvals_only=True)          # limits of L_eff by direction
            res["spectra"][(alpha, kappa)] = {"hess_eigs": w, "mu": prob.mu, "R2": prob.R2, "L_eff_limit_min": float(gen[0]),
                                              "L_eff_limit_max": float(gen[-1]), "summary": prob.summary()}
            theta0 = np.zeros(D)
            with common.Timer(f"alpha={alpha:g} kappa={kappa:g}"):
                r = algorithms.saga(prob, theta0, passes=PASSES, seed=0, record_every=max(1, N // 20), instrument_every=max(1, N // 10),
                                    snapshot_theta=True, stop_D=1e-26)
                KEYS = ["mu_eff", "L_eff", "energy_low1", "energy_low5", "D", "d_eff_dirs", "mu_eff_quadratic_proxy"]
                sag = {"t": []}
                sag.update({k: [] for k in KEYS})
                for q in r["instr"]:
                    cs = curvature_stats(prob, q["theta"], U, w=w)
                    sag["t"].append(q["t"])
                    for k in KEYS:
                        sag[k].append(cs[k])
                res["saga"][(alpha, kappa)] = {k: np.array(v) for k, v in sag.items()}
                # gradient descent with step 1/L_F, one full gradient per iteration
                theta = theta0.copy()
                gdd = {"t": []}
                gdd.update({k: [] for k in KEYS})
                for it in range(GD_ITERS + 1):
                    cs = curvature_stats(prob, theta, U, w=w)
                    gdd["t"].append(it)
                    for k in KEYS:
                        gdd[k].append(cs[k])
                    if cs["D"] < 1e-26:
                        break
                    theta -= prob.grad(theta) / prob.L_F
                res["gd"][(alpha, kappa)] = {k: np.array(v) for k, v in gdd.items()}
            s = res["saga"][(alpha, kappa)]
            g = res["gd"][(alpha, kappa)]
            print(f"   SAGA: mu_eff/mu {s['mu_eff'][0] / prob.mu:.2f} -> {s['mu_eff'][-1] / prob.mu:.2f}; L_eff/R2 {s['L_eff'][0] / prob.R2:.3f} -> {s['L_eff'][-1] / prob.R2:.3f}; "
                  f"GD: mu_eff/mu -> {g['mu_eff'][-1] / prob.mu:.2f}; lmin(H*)/mu = {w[0] / prob.mu:.2f}; L_eff limits [{gen[0] / prob.R2:.3f}, {gen[-1] / prob.R2:.3f}] R2", flush=True)
    common.save_result(TAG, res)
    return res


def plot(res):
    alpha = 1.0 if 1.0 in ALPHAS else ALPHAS[0]
    fig, axes = ps.new_fig(len(KAPPAS), 3, panel=(4.2, 2.9), sharex="col")
    for j, kappa in enumerate(KAPPAS):
        sp = res["spectra"][(alpha, kappa)]
        s = res["saga"][(alpha, kappa)]
        g = res["gd"][(alpha, kappa)]
        s = {k: np.asarray(v)[np.asarray(s["D"]) >= FLOOR] for k, v in s.items()}
        g = {k: np.asarray(v)[np.asarray(g["D"]) >= FLOOR] for k, v in g.items()}
        mu, R2 = sp["mu"], sp["R2"]
        ax = axes[0][j]
        ax.semilogy(s["t"] / N, s["mu_eff"] / mu, color=ps.ALGO["SAGA"], label="SAGA (x: passes)")
        ax.semilogy(g["t"], g["mu_eff"] / mu, color=ps.ALGO["GD"], label="GD (x: iterations)")
        ax.axhline(sp["hess_eigs"][0] / mu, color=ps.INK["muted"], linewidth=1.0)
        ax.axhline(sp["hess_eigs"][-1] / mu, color=ps.INK["muted"], linewidth=1.0)
        ax.annotate("λmin(F''(η*))/μ", (0, sp["hess_eigs"][0] / mu), xytext=(2, 2), textcoords="offset points", fontsize=7, color=ps.INK["secondary"])
        ax.annotate("λmax(F''(η*))/μ", (0, sp["hess_eigs"][-1] / mu), xytext=(2, 2), textcoords="offset points", fontsize=7, color=ps.INK["secondary"])
        ax.set_title(f"{ps.fmt_kappa(kappa)}, α = {alpha:g}")
        if j == 0:
            ax.set_ylabel("μ_eff(θₜ) / μ")
            ps.legend(ax, loc="upper right", fontsize=7)
        ax = axes[1][j]
        ax.semilogy(s["t"] / N, s["L_eff"] / R2, color=ps.ALGO["SAGA"], label="SAGA")
        ax.semilogy(g["t"], g["L_eff"] / R2, color=ps.ALGO["GD"], label="GD")
        ax.axhline(sp["L_eff_limit_min"] / R2, color=ps.INK["muted"], linewidth=1.0)
        ax.axhline(sp["L_eff_limit_max"] / R2, color=ps.INK["muted"], linewidth=1.0)
        ax.annotate("limits of L_eff at η* (min/max over directions)", (0, sp["L_eff_limit_max"] / R2), xytext=(2, 2), textcoords="offset points", fontsize=7, color=ps.INK["secondary"])
        if j == 0:
            ax.set_ylabel("L_eff(θₜ) / R²")
        ax = axes[2][j]
        ax.plot(s["t"] / N, s["energy_low1"], color=ps.ALGO["SAGA"], label="SAGA")
        ax.plot(g["t"], g["energy_low1"], color=ps.ALGO["GD"], label="GD")
        ax.set_ylim(-0.02, 1.02)
        if j == 0:
            ax.set_ylabel("share of ‖θₜ−η*‖² on the\nslowest eigenvector of F''(η*)")
        ax.set_xlabel("effective passes (SAGA)  /  iterations (GD)")
    fig.suptitle("Figure B2 — effective curvature μ_eff, effective cocoercivity L_eff and error direction along SAGA and GD trajectories", fontsize=10)
    ps.savefig(fig, "figB2_effective_curvature")


def main():
    ps.setup()
    res = run_all()
    plot(res)
    summary = {}
    for (alpha, kappa), sp in res["spectra"].items():
        s, g = res["saga"][(alpha, kappa)], res["gd"][(alpha, kappa)]
        # keep only the states above the precision floor of eta_* (below ~1e-18 the curvature ratios are rounding noise)
        s = {k: np.asarray(v)[np.asarray(s["D"]) >= FLOOR] for k, v in s.items()}
        g = {k: np.asarray(v)[np.asarray(g["D"]) >= FLOOR] for k, v in g.items()}
        mu, R2 = sp["mu"], sp["R2"]
        key = f"alpha={alpha:g},kappa={kappa:g}"
        summary[key] = {"lmin_H_over_mu": float(sp["hess_eigs"][0] / mu), "lmax_H_over_mu": float(sp["hess_eigs"][-1] / mu),
                        "saga_mu_eff_over_mu_start": float(s["mu_eff"][0] / mu), "saga_mu_eff_over_mu_end": float(s["mu_eff"][-1] / mu),
                        "saga_mu_eff_over_mu_geomean": float(np.exp(np.mean(np.log(s["mu_eff"] / mu)))),
                        "gd_mu_eff_over_mu_end": float(g["mu_eff"][-1] / mu),
                        "saga_L_eff_over_R2_geomean": float(np.exp(np.mean(np.log(s["L_eff"] / R2)))), "gd_L_eff_over_R2_end": float(g["L_eff"][-1] / R2),
                        "L_eff_limit_min_over_R2": float(sp["L_eff_limit_min"] / R2), "L_eff_limit_max_over_R2": float(sp["L_eff_limit_max"] / R2),
                        "saga_energy_low1_end": float(s["energy_low1"][-1]), "gd_energy_low1_end": float(g["energy_low1"][-1]),
                        "saga_d_eff_dirs_end": float(s["d_eff_dirs"][-1]) if "d_eff_dirs" in s else None,
                        "saga_d_eff_dirs_geomean": float(np.exp(np.mean(np.log(s["d_eff_dirs"])))) if "d_eff_dirs" in s else None,
                        "gd_d_eff_dirs_end": float(g["d_eff_dirs"][-1]) if "d_eff_dirs" in g else None,
                        "saga_mu_eff_quadratic_proxy_end_over_mu": float(s["mu_eff_quadratic_proxy"][-1] / mu) if "mu_eff_quadratic_proxy" in s else None,
                        "saga_mu_eff_monotone_decreasing_fraction": float(np.mean(np.diff(s["mu_eff"]) <= 0)),
                        "gd_mu_eff_monotone_decreasing_fraction": float(np.mean(np.diff(g["mu_eff"]) <= 0)),
                        "problem": sp["summary"]}
    common.save_json(TAG + "_summary", summary)
    print("done B2")


if __name__ == "__main__":
    main()
