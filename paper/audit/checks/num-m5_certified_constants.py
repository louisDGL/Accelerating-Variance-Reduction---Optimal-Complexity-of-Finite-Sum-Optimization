"""Audit num-m5, check 3: the numbers attached to Proposition 5 (lines 578-581 and 648-650 of the paper).

(i)  "its cooldown is 3.3e4 to 8.9e4 times slower, in log(1/q), than q = 0.8": certified factor of the length per
     pass (b = 1: n iterations), q_cert = rho^(n/2), rho = 1 - mu_op / L_eff, L_eff = L_op + 10 A L_max, for the three
     age bounds of Proposition 5(b): A = n (cyclic), 2n - 1 (fresh permutation), ceil(n log(n T / delta)) (independent
     batches, the sampling used in (M5); T = 300 n iterations). Constants recomputed from the data matrix X.
(ii) "gamma <= 1/(14 n L_max), 350 to 1400 times below the tuned steps": tuned steps of the nuclear-scaled joint
     SAGA-Muon at batch one, from M1_summary.json (factor-2 grids) and M5b_summary.json (2^(1/4) grids).

Run from the project root:
    PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        python3 paper/audit/checks/num-m5_certified_constants.py
"""
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "muon", "experiments"))
import common as C                                                    # noqa: E402

SPECS = {
    "soft-cond": C.spec("softmax", 200, 40, 8, 1e4, 2.0),
    "ls-cond": C.spec("ls", 200, 40, 8, 1e4, 2.0),
    "ls-orth": C.spec("ls", 500, 20, 20, 1e5, 2.0, noise=0.1, planted="orthogonal"),
    "soft-sq": C.spec("softmax", 500, 20, 20, 1e4, 2.0),
    "soft-mem": C.spec("softmax", 1000, 50, 10, 1e3, 1.0),
    "ls-mem": C.spec("ls", 1000, 50, 10, 1e3, 1.0),
}
Q_OBS, PASSES = 0.8, 300

const = {}
for key, s in SPECS.items():
    prob = C.problem(s)
    X, lam, n = np.asarray(prob.X), float(prob.lam), prob.n
    K, d = prob.K, prob.d
    r = min(K, d)
    c_l = 0.5 if s["loss"] == "softmax" else 1.0          # Lipschitz constant of the loss gradient in z
    sq = (X ** 2).sum(axis=1)
    Li = c_l * sq
    Lmax, Lmean = float(Li.max()), float(Li.mean())
    ev = np.sort(np.linalg.eigvalsh(X.T @ X / n))[::-1]
    Lop = c_l * float(ev[:r].sum()) + lam * r             # exact for least squares, upper bound for softmax
    mu = lam + (float(ev[-1]) if s["loss"] == "ls" else 0.0)   # best constant (Theorem 1 for softmax: lam)
    R2op = float((c_l * sq + lam * r).max())
    const[key] = dict(n=n, Lmax=Lmax, Lop=Lop, mu=mu, lam=lam, R2op=R2op)
    assert abs(Lop - prob.L_op) < 1e-9 * Lop and abs(R2op - prob.R2_op) < 1e-9 * R2op
    print(f"== {key}: n = {n}, K x d = {K} x {d}, lam = {lam:.4g}, mu_op = {mu:.4g} (lam + lambda_min = "
          f"{lam:.3g} + {ev[-1]:.3g}), L_op = {Lop:.4g}, L_max = {Lmax:.4g}, L_mean = {Lmean:.4g}, "
          f"L_max/L_op = {Lmax / Lop:.2f}, mu_op <= L_max: {mu <= Lmax}")
    if key not in ("soft-cond", "ls-cond"):
        continue
    T = PASSES * n
    ages = [("cyclic, A = n", n), ("fresh permutation, A = 2n - 1", 2 * n - 1)]
    for delta in (0.05, 0.01):
        ages.append((f"independent, A = ceil(n log(nT/delta)), T = {T}, delta = {delta}",
                     math.ceil(n * math.log(n * T / delta))))
    for mu_used, tag in ((mu, "best mu_op"), (lam, "mu_op = lam")):
        if mu_used == mu and tag == "mu_op = lam":
            continue
        for name, A in ages:
            Leff = Lop + 10 * A * Lmax
            log_q_cert = (n / 2) * math.log1p(-mu_used / Leff)
            print(f"   [{tag}] {name:62s} A = {A:5d}: q_cert = 1 - {-math.expm1(log_q_cert):.2e} per pass;"
                  f" log(1/0.8) / log(1/q_cert) = {math.log(Q_OBS) / log_q_cert:.3g}")
        if s["loss"] == "softmax":
            break

# (ii) tuned steps of the nuclear-scaled joint SAGA-Muon at batch one
print("\n== F15: gamma_max = 1 / (14 n L_max) against the tuned steps of 'SAGA-Muon (joint)' at batch one")
m1 = json.load(open(os.path.join(ROOT, "muon", "results", "M1_summary.json")))
m5b = json.load(open(os.path.join(ROOT, "muon", "results", "M5b_summary.json")))
for key, c in const.items():
    gmax = 1.0 / (14 * c["n"] * c["Lmax"])
    row = [r for r in m1[key]["rows"] if r["family"] == "SAGA-Muon (joint)"][0]
    g = row["best"]["gamma"]
    assert g["over"] == "R2_op"
    txt = f"   {key:9s}: gamma_max = {gmax * c['R2op']:.4g} / R2_op; M1 (factor-2 grid): {g['c']:g} / R2_op ->" \
          f" ratio {g['c'] / (gmax * c['R2op']):.0f}; gamma_max L_op = {gmax * c['Lop']:.2g}"
    try:
        rows = m5b["instances"][key]["0"]["batches"]["1"]["fine"]["rows"]
        name = [k for k in rows if "SAGA-Muon" in k][0]
        g2 = rows[name]["best"]["gamma"]
        txt += f"; M5b (2^(1/4) grid, '{name}'): {g2['c']:.4g} / {g2['over']} -> ratio {g2['c'] / (gmax * c['R2op']):.0f}"
    except (KeyError, IndexError) as exc:
        txt += f"; M5b: not available ({exc!r})"
    print(txt)
