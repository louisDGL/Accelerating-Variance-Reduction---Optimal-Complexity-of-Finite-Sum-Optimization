"""Audit num-f1, (C5): small reproduction of the B1 numbers quoted in the paper (results/B1.pkl is not in the
repository). kappa = 1e3, alpha in {1, 2}, P in {I, Sigma_hat + lam I}, sampling seeds 0-4, 20100 iterations;
observed exponent per pass = minus the least-squares slope of log ||theta - eta_*||_P^2 over passes 2-20 (mean over the
seeds), geometric means of r_coco (tightness of I2) and r_sc (tightness of I4) over passes 2-20; certified exponents.
Also the Euclidean-distance slope for the preconditioned runs (same asymptotic rate, different norm).
Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 paper/audit/checks/num-f1_b1_rerun.py
"""
import json
import os
import sys

sys.dont_write_bytecode = True
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import numpy as np  # noqa: E402
import common  # noqa: E402
from vrlab import algorithms, bounds  # noqa: E402
from vrlab.instrumentation import Metric  # noqa: E402

N, D = 1000, 50
ref = json.load(open(os.path.join(ROOT, "results", "B1_summary.json")))["preconditioners"]


def slope(r, key):
    it = np.asarray(r["iters"], float); y = np.asarray(r[key], float)
    sel = (it >= 2 * N) & (it <= 20 * N) & (y > 1e-25)
    return -np.polyfit(it[sel], np.log(y[sel]), 1)[0] * N


for alpha in (1.0, 2.0):
    prob = common.logistic_problem(N, D, 1e3, alpha, seed=0)
    for kind in ("I", "stat"):
        P = None if kind == "I" else prob.Sigma_hat + prob.lam * np.eye(D)
        cst = Metric(P, D).constants(prob, theta_ref=prob.theta_star)
        sD, sE, rc, rs = [], [], [], []
        for seed in range(5):
            r = algorithms.saga(prob, np.zeros(D), P=P, R2=cst["R2_P"], mu=cst["mu_P"], n_iters=20100, seed=seed,
                                record_every=N // 20, instrument_every=N // 10, stop_D=1e-26)
            sD.append(slope(r, "D_P")); sE.append(slope(r, "D"))
            recs = [q for q in r["instr"] if 2 * N <= q["t"] <= 20 * N]
            rc.append(np.mean(np.log([q["r_coco"] for q in recs]))); rs.append(np.mean(np.log([q["r_sc"] for q in recs])))
        v = ref[f"alpha={alpha:g}|{kind}"]
        cert = (1 - bounds.saga_rate(N, cst["R2_P"], cst["mu_P"])) * N
        print(f"alpha={alpha:g} P={kind:4s}: kappa_P = {cst['kappa_P']:.1f} (json {v['kappa_P']:.1f}); certified exponent/pass "
              f"{cert:.4f} (json {v['rate_bound_global']*N:.4f}); observed exponent/pass (P-norm) {np.mean(sD):.4f} "
              f"[{min(sD):.3f}, {max(sD):.3f}] (json {v['rate_emp_D_P']*N:.4f}); Euclidean-norm slope {np.mean(sE):.4f}; "
              f"r_coco {np.exp(np.mean(rc)):.4f} (json {v['r_coco_geomean']:.4f}); r_sc {np.exp(np.mean(rs)):.5f} "
              f"(json {v['r_sc_geomean']:.5f})")

# per-seed exponents at alpha = 1 (is the gain 1.00 -> 1.11 resolved against the seed spread?), 5 and 10 seeds
prob = common.logistic_problem(N, D, 1e3, 1.0, seed=0)
out = {}
for kind in ("I", "stat"):
    P = None if kind == "I" else prob.Sigma_hat + prob.lam * np.eye(D)
    cst = Metric(P, D).constants(prob, theta_ref=prob.theta_star)
    out[kind] = [slope(algorithms.saga(prob, np.zeros(D), P=P, R2=cst["R2_P"], mu=cst["mu_P"], n_iters=20100, seed=s,
                                       record_every=N // 20, stop_D=1e-26), "D_P") for s in range(10)]
    print(f"alpha=1 P={kind:4s} per-seed exponent/pass, seeds 0-9: " + " ".join(f"{x:.3f}" for x in out[kind]))
for m in (5, 10):
    a, b = np.array(out["I"][:m]), np.array(out["stat"][:m])
    se = np.sqrt(a.var(ddof=1) / m + b.var(ddof=1) / m)
    print(f"    {m} seeds: mean I {a.mean():.4f}, mean P {b.mean():.4f}, difference {b.mean()-a.mean():.4f} +/- {se:.4f} (standard error), "
          f"paired: {np.sum(b > a)}/{m} seeds with P faster")
