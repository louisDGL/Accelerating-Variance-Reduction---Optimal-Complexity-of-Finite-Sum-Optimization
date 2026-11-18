"""Second batch of checks for Part 4 (refinements of check_smoothing_median.py).

  (e')  minibatch offset with an exactly symmetric control: a data set made of pairs (x_i, +eps_i), (x_i, -eps_i) and
        antithetic batches (each batch together with its mirror image) has its mean-field fixed point at the
        least-squares solution for every batch size; with skewed residuals the offset decays like 1/b.
  (f')  momentum: median of an exponential moving average of skewed noise, larger sample.
  (g)   square case K = d: the Hessian at 0 of the smoothed nuclear norm has a surface term; for isotropic Gaussian noise
        c_(K,K) = E[sum_{i<j} 2/(s_i+s_j)]/K^2 + f_K(0)/K^2, f_K(0) = density at 0 of the smallest singular value,
        compared with sqrt(2) Gamma((K+1)/2)/Gamma(K/2) and with the measured response.
  (h)   rank-one (vector) phase: extinction time of the normalized flow = length of the gradient-flow curve, between
        ||e_0||_2 and the l1 norm of e_0 in the eigenbasis of Sigma, whatever the conditioning.
Run: .venv/Scripts/python muon/theory_checks/check_smoothing_median_2.py
"""
import numpy as np
from math import gamma, sqrt
from scipy.optimize import root
from scipy.signal import lfilter

rng = np.random.default_rng(20261117)


def polar_stack(M, tol=1e-12):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    keep = (s > tol * np.maximum(s[..., :1], 1e-300)).astype(float)
    return (U * keep[..., None, :]) @ Vt


print("== (e') minibatch offset: symmetric control and skewed residuals ==")
K, d, half = 2, 3, 4000
Xh = rng.standard_normal((half, d)) * np.array([1.0, 0.6, 0.3])
W0 = rng.standard_normal((K, d))


def fixed_point(X, Y, b, S, rng, W_init, mirror=None):
    B = rng.integers(0, X.shape[0], (S, b))
    if mirror is not None:                                       # antithetic batches: index i <-> mirror[i]
        B = np.concatenate([B, mirror[B]], axis=0)
    XB, YB = X[B], Y[B]
    A = np.einsum('sbi,sbj->sij', XB, XB) / b
    C = np.einsum('sbk,sbj->skj', YB, XB) / b

    def field(w):
        Wm = w.reshape(W_init.shape)
        return polar_stack(Wm[None] @ A - C).mean(0).reshape(-1)

    sol = root(field, W_init.reshape(-1), method='hybr', tol=1e-13)
    return sol.x.reshape(W_init.shape), np.linalg.norm(field(sol.x))


# symmetric control
eps = rng.standard_exponential((half, K)) - 1.0
X = np.concatenate([Xh, Xh]); Y = np.concatenate([Xh @ W0.T + 0.5 * eps, Xh @ W0.T - 0.5 * eps])
mirror = np.concatenate([np.arange(half) + half, np.arange(half)])
W_ls = np.linalg.solve(X.T @ X, X.T @ Y).T
print(f"  symmetric pairs: ||W_ls - W0|| = {np.linalg.norm(W_ls - W0):.1e}")
for b in [2, 8, 32]:
    Wb, res = fixed_point(X, Y, b, 20000, rng, W_ls + 0.01 * rng.standard_normal((K, d)), mirror=mirror)
    print(f"    b = {b:3d}: ||W_b - W_ls||_F = {np.linalg.norm(Wb - W_ls):.2e} (residual {res:.0e})")
# skewed residuals
n = 40000
X = rng.standard_normal((n, d)) * np.array([1.0, 0.6, 0.3])
Y = X @ W0.T + 0.5 * (rng.standard_exponential((n, K)) - 1.0)
W_ls = np.linalg.solve(X.T @ X, X.T @ Y).T
Sig = X.T @ X / n
rows = []
for b in [8, 16, 32, 64, 128]:
    vals = []
    for rep in range(3):
        Wb, res = fixed_point(X, Y, b, 150000, rng, W_ls)
        vals.append(np.linalg.svd((Wb - W_ls) @ Sig, compute_uv=False).sum())
    rows.append((b, np.mean(vals), np.std(vals)))
print("  skewed residuals: (b, ||grad F(W_b)||_* mean over 3 batch draws, std) = " + "; ".join(f"({b}, {m:.5f}, {s:.5f})" for b, m, s in rows))
print("      b * offset = " + ", ".join(f"{b * m:.3f}" for b, m, s in rows))

print("== (f') momentum: median of an exponential moving average of Exp(1) - 1 noise (5 runs of 8e6 samples) ==")
for beta in [0.8, 0.9, 0.95, 0.98]:
    meds = []
    for rep in range(5):
        a = rng.standard_exponential(8_000_000) - 1.0
        m = lfilter([1 - beta], [1, -beta], a)[int(30 / (1 - beta)):]
        meds.append(np.median(m))
    pred = -(1 / 3) * (1 - beta ** 2) / (1 + beta + beta ** 2)
    beff = -(1 / 3) * (1 - beta) / (1 + beta)
    print(f"  beta = {beta:4.2f}: median = {np.mean(meds):+.5f} (+- {np.std(meds) / np.sqrt(5):.5f}); cumulant formula {pred:+.5f}; effective-batch heuristic {beff:+.5f}")

print("== (g) square case: surface term of the smoothed Hessian ==")
for K in [2, 3, 4]:
    S = 600000
    Z = rng.standard_normal((S, K, K))
    s = np.linalg.svd(Z, compute_uv=False)
    acc = np.zeros(S)
    for i in range(K):
        for j in range(i + 1, K):
            acc += 2.0 / (s[:, i] + s[:, j])
    c_reg = acc.mean() / K ** 2
    h = 0.02
    f0_mc = np.mean(s[:, -1] < h) / h
    f0_formula = sqrt(2) * gamma((K + 1) / 2) / gamma(K / 2)
    G = rng.standard_normal((K, K)); G *= 0.02 / np.linalg.norm(G)
    P = 0.5 * (polar_stack(G + Z) + polar_stack(G - Z)).mean(0)
    measured = np.sum(P * G) / np.sum(G * G)
    print(f"  K = {K}: regular part {c_reg:.4f}; density of s_min at 0: Monte Carlo {f0_mc:.3f}, formula {f0_formula:.3f}; "
          f"regular + f(0)/K^2 = {c_reg + f0_formula / K ** 2:.4f}; measured response {measured:.4f}")

print("== (h) rank-one phase: extinction time of e' = - Sigma e / |Sigma e| ==")
for dd, alpha in [(5, 1.0), (5, 4.0), (20, 2.0), (20, 6.0)]:
    lam = np.arange(1, dd + 1.0) ** (-alpha)
    e0 = rng.standard_normal(dd)
    # length of the gradient-flow curve e(t) = exp(-lam t) e0 (eigenbasis), by quadrature in log-time
    ts = np.concatenate([[0.0], np.geomspace(1e-8, 80 / lam.min(), 400001)])
    curve = e0[None, :] * np.exp(-ts[:, None] * lam[None, :])
    length = np.sum(np.linalg.norm(np.diff(curve, axis=0), axis=1))
    print(f"  d = {dd}, kappa = {lam.max() / lam.min():.0e}: extinction time (curve length) = {length:.4f}; ||e_0||_2 = {np.linalg.norm(e0):.4f}; "
          f"||e_0||_1 (eigenbasis) = {np.abs(e0).sum():.4f}; ratio to ||e_0||_2 = {length / np.linalg.norm(e0):.3f} (sqrt(d) = {np.sqrt(dd):.2f})")
