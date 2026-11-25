"""Audit num-f1, (C4): constants inside the proof of Prop. 5.9 and the run at gamma R^2 = 4.
(1) Certified exponent as a function of (g, c), g = gamma R^2, Delta = c gamma^2:
        1 - rho_D = (2 g / kappa) (1 - g (1 + c/2)),   1 - rho_M = (1 - 2/c)/n   (admissible iff c > 2 and g (1 + c/2) < 1).
    Gain over the textbook pair (g, c) = (1/4, 3) by a brute-force grid, for several kappa/n; kappa = 1e3, n = 1e3.
(2) results/GammaDelta_summary.json: 17 %, the ratio 'observed / certificate' (4.5 ?), largest admissible step.
(3) Small rerun: SAGA at gamma R^2 = 4 on the problem kappa = 1e3, alpha = 1 (n = 1000, d = 50), 3 seeds, 40 passes as in
    experiments/exp_gamma_delta.py, then 400 passes: does the squared distance keep decreasing?
Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 paper/audit/checks/num-f1_gamma_delta.py
"""
import json
import os
import sys

sys.dont_write_bytecode = True
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import numpy as np  # noqa: E402


def best(n, kappa, g=None):
    """sup over c (and over g if g is None) of min(1 - rho_D, 1 - rho_M), per iteration."""
    c = np.exp(np.linspace(np.log(2.0 + 1e-9), np.log(1e7), 400001))
    if g is None:                       # for fixed c the D-rate is maximal at g = 1/(2 + c): value 1/(kappa (2 + c))
        rD = 1.0 / (kappa * (2.0 + c))
    else:
        rD = (2 * g / kappa) * (1 - g * (1 + c / 2))
    r = np.minimum(rD, (1 - 2 / c) / n)
    j = int(np.argmax(r))
    return float(r[j]), float(c[j])


n = 1000
print("(1) gain of the best (gamma, Delta) over the textbook pair, n = 1000")
for kappa in [1.0001, 3, 10, 100, 562.5, 1e3, 1e4, 1e6]:
    tb = min(3 / (16 * kappa), 1 / (3 * n))
    r, c = best(n, kappa)
    print(f"    kappa = {kappa:<9g} textbook {tb:.4e}  best {r:.4e} (c = {c:.4g}, g = {1/(2+c):.4g})  gain {r/tb:.4f}")
print("    -> bounded by 3 (memory rate 1/n against 1/(3n)); by 4/3 when the parameter term is active (kappa >= 9n/16)")

s = json.load(open(os.path.join(ROOT, "results", "GammaDelta_summary.json")))
kappa = s["problem"]["kappa"]
r, c = best(n, kappa)
r25, c25 = best(n, kappa, g=0.25)
emp, cert = s["empirical_rates_per_pass"], s["certified_rates_per_pass"]
print(f"(2) kappa = {kappa:g}: best certificate {r*n:.4f}/pass (json {s['optimal_within_proof']['rate']*n:.4f}), gain "
      f"{r/(3/(16*kappa)):.4f} (json {s['optimal_within_proof']['gain_over_textbook']:.4f}); at g = 1/4 with the best "
      f"Delta: {r25*n:.4f}/pass (json {cert['0.25']:.4f})")
print(f"    observed exponent at g = 1/4: {emp['0.25']:.4f}/pass (slope of log D, passes 2-40, 3 seeds)")
print(f"    observed(1/4) / certificate at g = 1/4, Delta optimised : {emp['0.25']/cert['0.25']:.3f}")
print(f"    observed(1/4) / best certificate over (gamma, Delta)    : {emp['0.25']/(r*n):.3f}")
print(f"    observed(1/4) / textbook certificate                    : {emp['0.25']/(3*n/(16*kappa)):.3f}")
print(f"    best observed (g = 1/3) / best certificate              : {emp['0.3333333333333333']/(r*n):.3f}")
print("    admissible steps: c > 2 and g (1 + c/2) < 1  =>  g < 1/2 ; 4 / (1/2) = 8")
print("    json, observed exponent per pass by g:", {k: round(v, 4) for k, v in emp.items()})

import common  # noqa: E402
from vrlab import algorithms  # noqa: E402

prob = common.logistic_problem(1000, 50, 1e3, 1.0, seed=0)
print("(3) SAGA at gamma R^2 = 4, kappa = 1e3, alpha = 1")
for seed in range(3):
    r = algorithms.saga(prob, np.zeros(50), gamma=4.0 / prob.R2, passes=400, seed=seed, record_every=100, record_F=False,
                        stop_D=1e-26)
    it = np.asarray(r["iters"], float); y = np.asarray(r["D"], float)
    sel = (it >= 2000) & (it <= 40000) & (y > 1e-25)
    slope = -np.polyfit(it[sel], np.log(y[sel]), 1)[0] * 1000
    pick = [0, 2, 20, 40, 100, 200, 400]
    vals = ", ".join(f"{p}: {y[int(np.argmin(abs(it - p * 1000)))]:.3e}" for p in pick if p * 1000 <= it[-1])
    late = (it >= 100000)
    slope_late = -np.polyfit(it[late], np.log(y[late]), 1)[0] * 1000 if late.sum() > 3 else float("nan")
    print(f"    seed {seed}: slope passes 2-40 = {slope:.4f}/pass; passes 100-400 = {slope_late:.4f}/pass; D at passes {vals}; "
          f"max D = {y.max():.3e}")
