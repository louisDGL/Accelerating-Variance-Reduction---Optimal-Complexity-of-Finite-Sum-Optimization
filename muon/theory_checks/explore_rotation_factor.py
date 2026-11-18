"""Exploration (not a proof): for the one-sided quadratic, polar(E Sigma) = polar(E) Q with Q = polar(|E| Sigma),
|E| = (E^T E)^{1/2}. Questions: (1) is the symmetric part of Q positive definite for all SPD pairs (|E|, Sigma)?
(2) how small can v_1^T Q v_1 be (v_1 top eigenvector of |E|), i.e. the speed at which ||E||_op decreases along the
flow?  (3) singular values of the error along the small-step flow: are they all non-increasing?
"""
import numpy as np

rng = np.random.default_rng(7)


def polar(M):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    return U @ Vt


def rand_spd(d, cond, rng):
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    ev = np.geomspace(1.0, 1.0 / cond, d)
    return (Q * ev) @ Q.T


worst_sym, worst_q1 = np.inf, np.inf
for trial in range(200000):
    d = rng.integers(2, 6)
    A = rand_spd(d, 10 ** rng.uniform(0, 8), rng)
    S = rand_spd(d, 10 ** rng.uniform(0, 8), rng)
    Q = polar(A @ S)
    w = np.linalg.eigvalsh(0.5 * (Q + Q.T)).min()
    wa, Va = np.linalg.eigh(A)
    q1 = Va[:, -1] @ Q @ Va[:, -1]
    worst_sym, worst_q1 = min(worst_sym, w), min(worst_q1, q1)
print(f"random SPD pairs (d in 2..5, condition numbers up to 1e8): min eigenvalue of sym(Q) = {worst_sym:.3e}; min v_1^T Q v_1 = {worst_q1:.3e}")

# 2x2 family: A = diag(1, eps), Sigma = R(theta) diag(1, delta) R(theta)^T
for eps, delta in [(1e-1, 1e-1), (1e-3, 1e-3), (1e-6, 1e-6)]:
    out = []
    for theta in [0.3, 1.0, 1.4, 1.55]:
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        A = np.diag([1.0, eps]); S = R @ np.diag([1.0, delta]) @ R.T
        Q = polar(A @ S)
        out.append((theta, Q[0, 0], np.linalg.eigvalsh(0.5 * (Q + Q.T)).min(), np.cos(theta)))
    print(f"eps = delta = {eps:g}: (theta, e1^T Q e1, lambda_min sym(Q), cos theta) = " + "; ".join(f"({a:.2f}, {b:.4f}, {c:.4f}, {e:.4f})" for a, b, c, e in out))

# singular values of the error along the small-step flow (generic instance)
K, d = 6, 5
lam = np.arange(1, d + 1.0) ** (-2.0)
V, _ = np.linalg.qr(rng.standard_normal((d, d)))
Sigma = (V * lam) @ V.T
E = rng.standard_normal((K, d))
eta = 2e-4
prev = np.linalg.svd(E, compute_uv=False)
max_increase = 0.0
t = 0.0
hist = [prev.copy()]
while prev[0] > 5 * eta:
    U, s, Vt = np.linalg.svd(E @ Sigma, full_matrices=False)
    keep = s > 1e-12 * s[0]
    E = E - eta * (U[:, keep] @ Vt[keep])
    cur = np.linalg.svd(E, compute_uv=False)
    max_increase = max(max_increase, np.max(cur - prev))
    prev = cur; t += eta
    hist.append(cur.copy())
hist = np.array(hist)
print(f"flow on a generic instance (kappa(Sigma) = {lam.max() / lam.min():.0f}): extinction at t = {t:.3f}, ||E_0||_op = {hist[0, 0]:.3f}; "
      f"largest one-step increase of a singular value of E: {max_increase:.2e} (step {eta})")
print("  singular values of E at t = 0, T/4, T/2, 3T/4:")
for f in [0, 0.25, 0.5, 0.75]:
    print("   ", np.round(hist[int(f * (len(hist) - 1))], 4))
