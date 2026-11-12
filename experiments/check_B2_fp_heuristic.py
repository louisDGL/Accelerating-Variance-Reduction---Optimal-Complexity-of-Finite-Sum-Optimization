"""Sanity check of the heuristic remark of the B2 note on gradient descent in floating point (rem:B2-gd-heuristic):
on a random l2-regularized logistic regression with more features than samples (n = 20, d = 40, unit-norm Gaussian
features, labels y = sign(<x, w> + 0.5 noise), lambda = 1e-4), gradient descent with step 1/L_F is run for 3000
iterations from theta_0 = 0 and from a Gaussian theta_0. Recorded: mu_eff(theta_t) = <F'(theta_t), theta_t - eta*> /
||theta_t - eta*||^2 at the end, the target lambda + lambda^+_min (smallest eigenvalue of the data part of F''(eta*)
restricted to the span S of the features, plus lambda), lambda_min(F''(eta*)), the fraction of non-increasing steps of
mu_eff, and the size of the component of theta_t - eta* orthogonal to S.
Usage: python experiments/check_B2_fp_heuristic.py        (seconds; writes results/B2_fp_check_summary.json)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from vrlab import problems, solvers  # noqa: E402

N, D, LAM, ITERS, SEEDS = 20, 40, 1e-4, 3000, [0, 1, 2]


def instance(seed):
    r = np.random.default_rng(seed)
    X = r.standard_normal((N, D))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    y = np.sign(X @ r.standard_normal(D) + 0.5 * r.standard_normal(N))
    y[y == 0] = 1
    prob = problems.LogisticProblem(X, y, LAM)
    solvers.solve_optimum(prob)
    return prob, X, r


def run(seed):
    prob, X, r = instance(seed)
    es = prob.theta_star
    Q, _ = np.linalg.qr(X.T)                                   # orthonormal basis of the span S of the features

    def perp(v):
        return v - Q @ (Q.T @ v)

    p = prob.sigma_prime(es)
    data_hess = (X.T * p) @ X / N
    target = LAM + float(np.linalg.eigvalsh(Q.T @ data_hess @ Q)[0])
    lmin_H = float(np.linalg.eigvalsh(prob.hessian(es))[0])
    out = {"lambda": LAM, "lambda_plus_lambda_plus_min": target, "lambda_min_hessian": lmin_H,
           "grad_norm_at_eta_star": float(np.linalg.norm(prob.grad(es)))}
    for name, th0 in [("theta0_zero", np.zeros(D)), ("theta0_gaussian", r.standard_normal(D))]:
        th = th0.copy()
        mus, perps = [], []
        for _ in range(ITERS + 1):
            v = th - es
            mus.append(float(v @ prob.grad(th)) / float(v @ v))
            perps.append(float(np.linalg.norm(perp(v))))
            th = th - prob.grad(th) / prob.L_F
        mus = np.array(mus)
        out[name] = {"mu_eff_final": float(mus[-1]),
                     "rel_gap_to_target": float((mus[-1] - target) / target),
                     "rel_gap_to_lambda": float((mus[-1] - LAM) / LAM),
                     "frac_nonincreasing_steps": float(np.mean(np.diff(mus) <= 1e-18 * np.abs(mus[1:]))),
                     "perp_norm_initial": perps[0], "perp_norm_max_after_0": float(max(perps[1:])), "perp_norm_final": perps[-1]}
    return out


def main():
    summary = {"config": {"n": N, "d": D, "lambda": LAM, "iterations": ITERS, "step": "1/L_F",
                          "labels": "sign(<x, w> + 0.5 noise), w and noise standard Gaussian", "features": "unit-norm Gaussian"}}
    for seed in SEEDS:
        summary[f"seed{seed}"] = run(seed)
        s = summary[f"seed{seed}"]
        print(f"seed {seed}: target {s['lambda_plus_lambda_plus_min']:.4g}; from 0: mu_eff {s['theta0_zero']['mu_eff_final']:.4g}, "
              f"nonincreasing {s['theta0_zero']['frac_nonincreasing_steps']:.3f}, max perp {s['theta0_zero']['perp_norm_max_after_0']:.2g}; "
              f"from Gaussian: mu_eff {s['theta0_gaussian']['mu_eff_final']:.4g} (rel. gap to lambda {s['theta0_gaussian']['rel_gap_to_lambda']:.2g}), "
              f"perp {s['theta0_gaussian']['perp_norm_initial']:.3g} -> {s['theta0_gaussian']['perp_norm_final']:.3g}")
    common.save_json("B2_fp_check_summary", summary)


if __name__ == "__main__":
    main()
