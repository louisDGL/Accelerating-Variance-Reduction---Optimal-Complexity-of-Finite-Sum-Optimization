"""Adversarial search: can q_1 = v_1^T polar(A Sigma) v_1 be negative for SPD A, Sigma (v_1 top eigenvector of A)?
q_1 is the speed at which ||E||_op decreases along the polar flow on the one-sided quadratic (A = |E|).
Also: tr(A Q) = tr(Sigma^-1 |G|) > 0 (decrease of ||E||_F), tr(Q) (decrease of ||E||_*).
"""
import numpy as np
from scipy.optimize import minimize

rng = np.random.default_rng(3)


def polar(M):
    U, s, Vt = np.linalg.svd(M)
    return U @ Vt


def unpack(x, d):
    La = np.zeros((d, d)); Ls = np.zeros((d, d))
    idx = np.tril_indices(d)
    k = len(idx[0])
    La[idx] = x[:k]; Ls[idx] = x[k:2 * k]
    A = La @ La.T + 1e-10 * np.eye(d)
    S = Ls @ Ls.T + 1e-10 * np.eye(d)
    return A, S


def q1(x, d):
    A, S = unpack(x, d)
    w, V = np.linalg.eigh(A)
    v = V[:, -1]
    return float(v @ polar(A @ S) @ v)


def trQ(x, d):
    A, S = unpack(x, d)
    return float(np.trace(polar(A @ S)))


for d in [2, 3, 4, 5]:
    k = d * (d + 1) // 2
    best_q, best_t = np.inf, np.inf
    for trial in range(60):
        x0 = rng.standard_normal(2 * k) * 10 ** rng.uniform(-2, 1, 2 * k)
        res = minimize(q1, x0, args=(d,), method='Nelder-Mead', options={'maxiter': 6000, 'xatol': 1e-10, 'fatol': 1e-12})
        best_q = min(best_q, res.fun)
        res = minimize(trQ, x0, args=(d,), method='Nelder-Mead', options={'maxiter': 6000, 'xatol': 1e-10, 'fatol': 1e-12})
        best_t = min(best_t, res.fun)
    print(f"d = {d}: adversarial minimum of q_1 = {best_q:+.3e};  adversarial minimum of tr(Q) = {best_t:+.4f} (d - 2 floor(d/2)... compare with -d..d)")
