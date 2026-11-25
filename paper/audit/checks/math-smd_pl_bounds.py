"""Audit math-smd -- the two inequalities used in Theorem 2(c) and Proposition 3(b) (line 415):
   2 mu_op (F(W) - F_*) <= ||grad F(W)||_*^2 <= 2 L_op (F(W) - F_*),
checked on random quadratics F(W) = 1/2 tr(W S W^T) - <B, W> (mu_op = lambda_min(S), L_op = sum of the min(K, d)
largest eigenvalues of S) and on a non-quadratic F = quadratic + c log cosh(<Q, W>) (K = d, L_op = tr S + c d^2).
Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 paper/audit/checks/math-smd_pl_bounds.py
"""
import numpy as np
from scipy.optimize import fsolve
rng = np.random.default_rng(3)
lo, hi = np.inf, -np.inf
for trial in range(3000):
    K, d = [(1, 1), (2, 2), (2, 3), (3, 2), (4, 4), (3, 6)][trial % 6]
    Qm, _ = np.linalg.qr(rng.standard_normal((d, d)))
    ev = np.sort(rng.uniform(0.2, 3.0, d))[::-1]
    S = Qm @ np.diag(ev) @ Qm.T
    B = rng.standard_normal((K, d))
    Ws = B @ np.linalg.inv(S)
    f = lambda W: 0.5 * np.sum((W @ S) * W) - np.sum(B * W)
    W = Ws + rng.standard_normal((K, d)) * 10.0 ** rng.uniform(-3, 1)
    if trial % 5 == 0:                                   # adversarial directions: flattest / all top directions
        W = Ws + np.outer(rng.standard_normal(K), Qm[:, -1])
    if trial % 5 == 1 and K <= d:
        W = Ws + np.linalg.qr(rng.standard_normal((K, K)))[0] @ Qm[:, :K].T @ np.linalg.inv(S)
    G = W @ S - B
    ratio = np.linalg.svd(G, compute_uv=False).sum() ** 2 / (2 * (f(W) - f(Ws)))
    lo = min(lo, ratio / ev[-1]); hi = max(hi, ratio / ev[: min(K, d)].sum())
print(f"quadratics: min ||G||_*^2 / (2 mu_op gap) = {lo:.6f} (>= 1), max ||G||_*^2 / (2 L_op gap) = {hi:.6f} (<= 1)")
lo, hi = np.inf, -np.inf
for trial in range(300):
    d = 2 + trial % 2
    Qm, _ = np.linalg.qr(rng.standard_normal((d, d)))
    ev = np.sort(rng.uniform(0.2, 3.0, d))[::-1]
    S = Qm @ np.diag(ev) @ Qm.T
    B = 0.3 * rng.standard_normal((d, d)); Q, _ = np.linalg.qr(rng.standard_normal((d, d))); c = 0.7
    def f(W):
        t = np.sum(Q * W); return 0.5 * np.sum((W @ S) * W) - np.sum(B * W) + c * (np.logaddexp(t, -t) - np.log(2))
    grad = lambda W: W @ S - B + c * np.tanh(np.sum(Q * W)) * Q
    Ws = fsolve(lambda w: grad(w.reshape(d, d)).ravel(), np.zeros(d * d), xtol=1e-12).reshape(d, d)
    assert np.abs(grad(Ws)).max() < 1e-9
    W = Ws + rng.standard_normal((d, d)) * 10.0 ** rng.uniform(-2, 1)
    ratio = np.linalg.svd(grad(W), compute_uv=False).sum() ** 2 / (2 * (f(W) - f(Ws)))
    lo = min(lo, ratio / ev[-1]); hi = max(hi, ratio / (ev.sum() + c * d * d))
print(f"non-quadratic: min ratio / mu_op = {lo:.6f} (>= 1), max ratio / L_op = {hi:.6f} (<= 1)")
