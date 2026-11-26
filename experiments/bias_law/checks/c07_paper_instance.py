"""Parameter-free prediction of a number measured in the paper (Fig. 3(c); muon/theory_checks/make_paper_figures.py):
finite-sum least squares with an intercept and skewed residuals (n = 2000, K = 2, d = 3), batches drawn with replacement.
The paper measures b * ||grad F(W_b)||_* rising from 0.17 (b=1) to a plateau near 0.3 (0.3079 at b=32).
First-order law (state-dependent noise, Theorem 3):  b * grad F(W_b) -> g* = -(1/6) J^{-1} T[K3] with the covariance
and third cumulant of the per-sample gradients at the least-squares solution.
"""
import numpy as np
from lib import *

def ls_problem(n=2000, seed=4):            # copied verbatim from make_paper_figures.py
    rng = np.random.default_rng(seed)
    K, d = 2, 3
    X = rng.standard_normal((n, d)) * np.array([1.0, 0.6, 0.3])
    X[:, 0] = 1.0
    W0 = rng.standard_normal((K, d))
    Y = X @ W0.T + 0.5 * (rng.standard_exponential((n, K)) - 1.0)
    W_ls = np.linalg.solve(X.T @ X, X.T @ Y).T
    return X, Y, W_ls, X.T @ X / n

X, Y, W_ls, Sig_x = ls_problem()
n = len(X); K, d = 2, 3
R = X @ W_ls.T - Y
g = R[:, :, None] * X[:, None, :]                      # per-sample gradients at W_ls (mean zero)
V = g.reshape(n, -1)
print("mean of per-sample gradients:", np.abs(V.mean(0)).max())
Cov = V.T @ V / n; K3 = np.einsum('na,nb,nc->abc', V, V, V) / n
vals = []
for seed in range(4):
    gstar, J, T = general_prediction(Cov, K3, K, d, S=4_000_000, seed=seed)
    vals.append(nuc(gstar))
print(f"predicted plateau  ||g*||_* = {np.mean(vals):.4f} (+- {np.std(vals):.4f} over 4 Monte-Carlo evaluations of J, T)")
print("measured in the paper: b*||grad F(W_b)||_* = 0.2757 (b=4), 0.2848 (b=8), 0.3032 (b=16), 0.3079 (b=32)")
