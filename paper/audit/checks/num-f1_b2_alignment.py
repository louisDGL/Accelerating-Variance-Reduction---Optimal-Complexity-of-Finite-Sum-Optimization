"""Audit num-f1, (C3): 'near the optimum mu_eff is at least about the smallest eigenvalue of the Hessian at eta_*, which
SAGA reaches when its error aligns with the flattest direction, as it does at kappa = 10 n for alpha in {0, 1}'.
results/B2_summary.json holds ONE run (seed 0) per configuration; here seeds 0-4, 60 passes, kappa = 1e4, alpha = 0, 1, 2
(n = 1000, d = 50): share pi_min of the squared error on the bottom eigenvector of F''(eta_*) and mu_eff / lmin(H*) at
the end (last state with D >= 1e-18), and at theta_0.
Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 paper/audit/checks/num-f1_b2_alignment.py
"""
import os
import sys

sys.dont_write_bytecode = True
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import numpy as np  # noqa: E402
import common  # noqa: E402
from vrlab import algorithms  # noqa: E402

N, D = 1000, 50
for alpha in (0.0, 1.0, 2.0):
    prob = common.logistic_problem(N, D, 1e4, alpha, seed=0)
    w, U = np.linalg.eigh(prob.hessian(prob.theta_star))
    v0 = -prob.theta_star
    print(f"alpha = {alpha:g}: lmin(H*)/mu = {w[0]/prob.mu:.3f}; pi_min(theta_0) = {(U[:,0] @ v0)**2 / (v0 @ v0):.4f}")
    for seed in range(5):
        r = algorithms.saga(prob, np.zeros(D), passes=60, seed=seed, record_every=N // 20, instrument_every=N // 10,
                            snapshot_theta=True, stop_D=1e-26)
        recs = [q for q in r["instr"] if q["D"] >= 1e-18]
        q = recs[-1]
        v = q["theta"] - prob.theta_star
        pi = (U.T @ v) ** 2 / (v @ v)
        mins = min(x["mu_eff"] for x in recs) / w[0]
        print(f"    seed {seed}: end at pass {q['t']/N:.1f}, D = {q['D']:.2e}, pi_min = {pi[0]:.4f}, mu_eff/lmin(H*) = "
              f"{q['mu_eff']/w[0]:.4f}, mu_eff/mu = {q['mu_eff']/prob.mu:.3f}; min over the run of mu_eff/lmin(H*) = {mins:.4f}")
