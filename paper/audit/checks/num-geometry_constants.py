"""Audit num-geometry, check 2: independent recomputation of the per-sample condition numbers R_P^2/mu_P of (M4) and
Figure 2(a), from the data of the six instances only (X, labels, lam, W_*), with the formulas of the paper
(paragraph 'Geometries', l. 236-242):

    R_P^2 = max_i c_i x_i^T P^-1 x_i + lam ||P^-1||_op,   mu_P = best constant of D^2F >= mu_P ||Delta P^(1/2)||_F^2,
    one-shot polar metric  P^-1 = V diag(g) V^T + r (I - V V^T), g_k = ||G_0||_* / max(s_k, f s_1), G_0 = grad F(0),
    covariance metric      P^-1 proportional to (c Sigma_hat + lam I)^-1 with eigenvalues floored likewise.

Nothing of muon/lab/vrmuon.py (polar_metric) or of exp_M7_checks.py is used: only the problem generator
(common.problem: data and W_*) and numpy. The Hessian of softmax at W_* is rebuilt here from the probabilities.
Compared with muon/results/M7_summary.json. Read-only.

Run from the project root:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 paper/audit/checks/num-geometry_constants.py
"""
import json
import os
import sys

import numpy as np

sys.dont_write_bytecode = True      # importing the repository must not refresh its __pycache__
ROOT = os.getcwd()
sys.path.insert(0, os.path.join(ROOT, "muon", "experiments"))
import common as C  # noqa: E402  (adds muon/ to the path; creates nothing new)

m7 = json.load(open(os.path.join(ROOT, "muon", "results", "M7_summary.json")))
m1 = json.load(open(os.path.join(ROOT, "muon", "results", "M1_summary.json")))
FLOORS = (0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
POL, COV = "grad F(W_0)", "feature covariance, (c_l Sigma_hat + lam I)^-1"


def sqrtm_psd(A):
    w, U = np.linalg.eigh(A)
    return (U * np.sqrt(np.maximum(w, 0))) @ U.T


worst = 0.0
for key in ("soft-mem", "ls-mem", "soft-cond", "ls-cond", "ls-orth", "soft-sq"):
    s = m1[key]["spec"]
    prob = C.problem(s)
    X, lam, K, d, n = np.asarray(prob.X), float(prob.lam), prob.K, prob.d, prob.n
    soft = s["loss"] == "softmax"
    c = 0.5 if soft else 1.0                       # sup lambda_max of the loss Hessian: softmax 1/2, least squares 1
    Sig = X.T @ X / n
    sq = np.einsum("ij,ij->i", X, X)
    R2 = c * sq.max() + lam
    Ws = np.asarray(prob.W_star)
    if soft:
        y = np.asarray(prob.y).astype(int)
        Z = X @ Ws.T
        Z -= Z.max(axis=1, keepdims=True)
        Pm = np.exp(Z)
        Pm /= Pm.sum(axis=1, keepdims=True)
        # own gradient at W_* (must vanish) and at 0
        Y = np.zeros((n, K))
        Y[np.arange(n), y] = 1.0
        g_star = (Pm - Y).T @ X / n + lam * Ws
        G0 = (np.full((n, K), 1.0 / K) - Y).T @ X / n
        H = np.zeros((K, d, K, d))
        for k in range(K):
            for l in range(K):
                a = (Pm[:, k] * (1.0 if k == l else 0.0)) - Pm[:, k] * Pm[:, l]
                H[k, :, l, :] = (X * a[:, None]).T @ X / n
        H = H.reshape(K * d, K * d) + lam * np.eye(K * d)
        # gauge subspace 1^T W = 0
        Qf, _ = np.linalg.qr(np.concatenate([np.ones((K, 1)) / np.sqrt(K), np.eye(K)[:, :K - 1]], axis=1))
        Q = Qf[:, 1:]
        assert np.abs(Q.T @ np.ones(K)).max() < 1e-12
        PQ = np.kron(Q, np.eye(d))
        Hred = PQ.T @ H @ PQ
        Kg = K - 1
        anchor = Q.T @ G0
    else:
        Yt = np.asarray(prob.Y)
        g_star = (X @ Ws.T - Yt).T @ X / n + lam * Ws
        G0 = (-Yt).T @ X / n
        Hfeat = Sig + lam * np.eye(d)
        Hred = np.kron(np.eye(K), Hfeat)
        Kg = K
        anchor = G0
    ev = np.linalg.eigvalsh(Hred)
    k_loc = R2 / ev[0]
    k_glob = R2 / lam if soft else k_loc
    ref = m7["metric"][key]
    print(f"== {key}: n={n} K={K} d={d} lam={lam:.4e} R2={R2:.6f} |grad F(W_*)|_F={np.linalg.norm(g_star):.1e}  "
          f"Euclid local {k_loc:.3f} (JSON {ref['kappa_euclid']:.3f}) global {k_glob:.3f} (JSON "
          f"{ref['kappa_euclid_global']:.3f});  R2/mu = {k_glob / n:.2f} n (global), {k_loc / n:.2f} n (local)")
    worst = max(worst, abs(k_loc / ref["kappa_euclid"] - 1), abs(k_glob / ref["kappa_euclid_global"] - 1))
    sv = np.linalg.svd(anchor, full_matrices=False)
    sA, VA = sv[1], sv[2].T
    r = int(np.sum(sA > 1e-12 * sA[0]))
    sA, VA = sA[:r], VA[:, :r]
    hC, VC = np.linalg.eigh(c * Sig + lam * np.eye(d))
    hC, VC = hC[::-1], VC[:, ::-1]
    for label in (POL, COV):
        out = []
        for f in FLOORS:
            if label == POL:
                g = sA.sum() / np.maximum(sA, f * sA[0])
                Cm = VA @ np.diag(g) @ VA.T + r * (np.eye(d) - VA @ VA.T)
            else:
                g = hC.sum() / np.maximum(hC, f * hC[0])
                Cm = VC @ np.diag(g) @ VC.T
            Cm = (Cm + Cm.T) / 2
            wC = np.linalg.eigvalsh(Cm)
            R2P = c * np.einsum("ij,jk,ik->i", X, Cm, X).max() + lam * wC[-1]
            T = np.kron(np.eye(Kg), sqrtm_psd(Cm))
            evP = np.linalg.eigvalsh(T @ Hred @ T)
            loc = R2P / evP[0]
            glob = R2P / (lam * wC[0]) if soft else loc
            row = [x for x in ref["rows"] if x["anchor"] == label and x["floor"] == f][0]
            worst = max(worst, abs(loc / row["kappa_P"] - 1), abs(glob / row["kappa_P_global"] - 1))
            iso = (wC[-1] / wC[0] - 1) < 1e-9
            out.append((f, loc, glob, row["kappa_P"], row["kappa_P_global"], iso))
        print(f"   {label[:20]:20s} rank {r if label == POL else d}")
        for f, loc, glob, jl, jg, iso in out:
            print(f"      f={f:<6g} local {loc:10.3f} (JSON {jl:10.3f})  global {glob:11.3f} (JSON {jg:11.3f})  "
                  f"global/Euclid global = {glob / k_glob:.6f}" + ("   <- metric is a multiple of the identity" if iso else ""))
print(f"\nlargest relative difference with M7_summary.json over all cells: {worst:.2e}")
