"""Audit num-f1.
(1) Eq. (saga-repr) of the paper checked on instrumented states: E_t[V_{t+1}] = c_D D + c_M Delta M with
    c_D = 1 - mu_eff/(2R^2) (1 - L_eff/R^2 (r_sq/4 + 3/8)),  c_M = 1 - (1/n)(1 - (2/3) r_sq r_var);
    range of the two 'factors' hidden in c_D and c_M: (1 - c_D)/(3 mu_eff/(16 R^2)) <= 8/3 (I1, I2) and
    (1 - c_M)/(1/(3n)) <= 3 (I1, I3).
(2) (C4), last sentence: on n identical components (mu th1^2 + R^2 th2^2)/2 the stated iteration count is a lower bound
    for theta_0 = (r, r), not for every theta_0: theta_0 = (0, 1), gamma = 1/R^2 reaches 0 in one iteration.
Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 paper/audit/checks/num-f1_repr_and_lower.py
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
print("(1) identity (saga-repr) on instrumented states, passes 2-20, seed 0")
for kappa, alpha in ((10.0, 0.0), (1e3, 1.0), (1e5, 0.0), (1e5, 2.0)):
    prob = common.logistic_problem(N, D, kappa, alpha, seed=0)
    r = algorithms.saga(prob, np.zeros(D), n_iters=20100, seed=0, record_every=N // 20, instrument_every=N // 10, stop_D=1e-26)
    R2, mu, Dl = r["R2"], r["mu"], r["Delta"]
    err, fD, fM = 0.0, [], []
    for q in r["instr"]:
        if not (2 * N <= q["t"] <= 20 * N):
            continue
        cD = 1 - q["mu_eff"] / (2 * R2) * (1 - q["L_eff"] / R2 * (q["r_sq"] / 4 + 3 / 8))
        cM = 1 - (1 - 2 / 3 * q["r_sq"] * q["r_var"]) / N
        err = max(err, abs(cD * q["D"] + cM * Dl * q["M"] - q["EV_next"]) / q["V"])
        fD.append((1 - cD) / (3 * q["mu_eff"] / (16 * R2))); fM.append((1 - cM) * 3 * N)
    print(f"    kappa={kappa:g}, alpha={alpha:g}: max |c_D D + c_M Delta M - E V_next| / V = {err:.1e}; factor of I1-I2 in c_D "
          f"in [{min(fD):.3f}, {max(fD):.3f}] (<= 8/3); factor of I1, I3 in c_M in [{min(fM):.3f}, {max(fM):.3f}] (<= 3)")

print("(2) n identical quadratics, SAGA, gamma = 1/R^2")
mu, R2, n = 1e-3, 1.0, 16
Q = np.array([mu, R2])
for theta0 in (np.array([0.0, 1.0]), np.array([1.0, 1.0])):
    rng = np.random.default_rng(0)
    theta = theta0.copy(); z = np.tile(Q * theta, (n, 1)); zbar = z.mean(0)
    out = []
    for t in range(3):
        i = rng.integers(n); g = Q * theta
        theta = theta - (1 / R2) * (g - z[i] + zbar); zbar = zbar + (g - z[i]) / n; z[i] = g
        out.append(float(theta @ theta) / float(theta0 @ theta0))
    print(f"    theta_0 = {theta0}: ||theta_t||^2/||theta_0||^2 for t = 1, 2, 3: {out}; claimed minimal count for eps = 1e-3: "
          f"{(R2 / mu - 1) / 4 * np.log(1 / 2e-3):.0f}")
