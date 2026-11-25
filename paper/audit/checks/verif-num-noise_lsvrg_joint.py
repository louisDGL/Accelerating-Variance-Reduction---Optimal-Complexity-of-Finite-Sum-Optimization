"""Counter-verification of finding F6 of the audit dimension num-noise ((M1), l. 497-498: "Only a finite-sum estimator
reaches rounding precision, with a length that vanishes with it (nuclear-scaled ...)").

(1) Stored rows of muon/results/M1_summary.json for the family "L-SVRG-Muon (joint)" (nuclear-scaled polar step of the
    loopless-SVRG estimator, tuned step) next to "SAGA-Muon (joint)".
(2) Own run (plain numpy loop written here; muon/lab only builds the instance and gives its minimizer) of
        omega = gbar + lam W + (r_j(W) - r_j(w)) x_j^T,   W <- W - gamma ||omega||_* polar(omega),
        with probability 1/n after the step: w <- W, gbar <- loss gradient at w (cost n; 2 per iteration otherwise),
    on the two memory-limited instances, at the step selected by M1 and at smaller ones, 150 passes, three seeds.

Usage (project root):
  PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python3 paper/audit/checks/verif-num-noise_lsvrg_joint.py [json|soft|ls]
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "muon", "experiments"))
part = sys.argv[1] if len(sys.argv) > 1 else "json"

if part == "json":
    d = json.load(open(os.path.join(ROOT, "muon", "results", "M1_summary.json")))
    for inst, blk in d.items():
        tol = blk["tol"]
        for r in blk["rows"]:
            if r["family"] in ("SAGA-Muon (joint)", "L-SVRG-Muon (joint)"):
                s = r["summary"][f"{tol:g}"]
                f = r["final_rel"]
                print(f"{inst:9s} tol {tol:g} budget {blk['passes']:3d} | {r['family']:20s} gamma = "
                      f"{r['best']['gamma']['c']:g}/R2_op edge {r['edge']} | reached {s['reached']}/{s['runs']} "
                      f"median passes {s['median']:.1f} | final rel. sq. distance median {f['median']:.1e} "
                      f"[{f['min']:.1e}, {f['max']:.1e}]")
    sys.exit(0)

import common as C  # noqa: E402  (instance and minimizer only)


def resid_all(loss, Z, y):
    if loss == "ls":
        return Z - y
    E = np.exp(Z - Z.max(axis=1, keepdims=True))
    P = E / E.sum(axis=1, keepdims=True)
    P[np.arange(Z.shape[0]), y] -= 1.0
    return P


def resid_one(loss, z, yj):
    if loss == "ls":
        return z - yj
    e = np.exp(z - z.max())
    r = e / e.sum()
    r[yj] -= 1.0
    return r


def run(loss, c, seed, passes, n=1000, d=50, K=10, kappa=1e3, alpha=1.0):
    prob = C.problem(C.spec(loss, n, d, K, kappa, alpha))
    X, y, lam, Ws = np.array(prob.X), np.array(prob.y), float(prob.lam), np.array(prob.W_star)
    cl = 0.5 if loss == "softmax" else 1.0
    R2op = float(cl * (X ** 2).sum(axis=1).max() + lam * min(K, d))
    gamma = c / R2op
    soft = loss == "softmax"
    if soft:
        Q, _ = np.linalg.qr(np.eye(K) - 1.0 / K)
        Q = Q[:, : K - 1]
    D0 = float((Ws ** 2).sum())
    rng = np.random.default_rng(987654 + 104729 * seed)
    W = np.zeros((K, d))
    w = W.copy()
    Zw = X @ w.T
    gbar = resid_all(loss, Zw, y).T @ X / n
    cost, budget = n, passes * n
    marks = [10, 20, 40, 80, 120, 150]
    out, k = {}, 0
    best = 1.0
    while cost < budget:
        j = int(rng.integers(0, n))
        x = X[j]
        dr = resid_one(loss, W @ x, y[j]) - resid_one(loss, Zw[j].copy(), y[j])
        om = gbar + lam * W + np.outer(dr, x)
        A = Q.T @ om if soft else om
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        O = U @ Vt
        if soft:
            O = Q @ O
        W -= gamma * s.sum() * O
        cost += 2
        if rng.random() < 1.0 / n:
            w = W.copy()
            Zw = X @ w.T
            gbar = resid_all(loss, Zw, y).T @ X / n
            cost += n
        while k < len(marks) and cost >= marks[k] * n:
            v = float(((W - Ws) ** 2).sum()) / D0
            out[marks[k]] = v
            best = min(best, v)
            k += 1
        if not np.isfinite(W).all() or float(((W - Ws) ** 2).sum()) > 1e8 * D0:
            out["diverged"] = cost / n
            break
    return out


loss = "softmax" if part == "soft" else "ls"
grid = (1 / 32, 1 / 64, 1 / 128, 1 / 256) if part == "soft" else (1 / 256, 1 / 512, 1 / 1024)
print(f"{loss}, memory-limited instance (n = 1000, K x d = 10 x 50, kappa = n), loopless-SVRG estimator, nuclear-scaled "
      "polar step, 150 passes; relative squared distance at passes 10, 20, 40, 80, 120, 150")
t0 = time.process_time()
for c in grid:
    for seed in (10, 11, 12):
        o = run(loss, c, seed, 150)
        txt = "  ".join(f"{q}: {o[q]:.1e}" for q in (10, 20, 40, 80, 120, 150) if q in o)
        print(f"  gamma = {c:g}/R2_op  seed {seed}:  {txt}" + (f"   DIVERGED at pass {o['diverged']:.0f}" if "diverged" in o
                                                                 else ""), flush=True)
print(f"  [cpu {time.process_time() - t0:.0f} s]")
