"""Audit math-offset, check 4: equation (3) of paper_saga_to_muon.tex for a size with K = 3 (3 x 5, entries
Exp(1) - 1, b = 32), which the repository does not test against a zero of the mean field. SVD-based polar factor,
one-dimensional root along 1 1^T, S = 1e6, two control variates, delta-method standard error.
Constants from math-offset_constants.py (singular-value formula): a/c = 0.40102 (+- 0.00078).
Run from the project root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 paper/audit/checks/math-offset_3x5.py
"""
import numpy as np
from math import sqrt
from scipy.optimize import brentq

K, d, S, b = 3, 5, 1_000_000, 32
AC, C = 0.40102, 0.40042
pred = (2 / 6) * AC * sqrt(K * d)
rng = np.random.default_rng(31415)
xi = rng.gamma(b, 1.0, size=(S, K, d)) / b - 1.0
Sd = sqrt(b) * xi
X = np.stack([Sd.sum(axis=(1, 2)), (Sd ** 3 - 3 * Sd).sum(axis=(1, 2)) - K * d * 2 / sqrt(b)], axis=1)


def T(g):
    U, s, Vt = np.linalg.svd(g + xi, full_matrices=False)
    return np.einsum('nik,nkj->n', U, Vt)              # sum of the entries of the polar factor


T0 = T(0.0)
beta = np.linalg.lstsq(X - X.mean(0), T0 - T0.mean(), rcond=None)[0]
adj = X @ beta
g0 = pred / (b * sqrt(K * d))
f = lambda g: (T(g) - adj).mean()
gb = brentq(f, 0.5 * g0, 1.5 * g0, xtol=1e-9 * g0 * 1e3, rtol=1e-6)
slope = (f(1.05 * gb) - f(0.95 * gb)) / (0.1 * gb)
se = (T(gb) - adj).std(ddof=1) / sqrt(S) / slope
val, sev = b * sqrt(K * d) * gb, b * sqrt(K * d) * se
print(f"3 x 5, Exp(1) - 1, b = {b}: prediction b ||G_b||_* -> {pred:.4f}; zero of the mean field: {val:.4f} (+- {sev:.4f}); "
      f"ratio {val / pred:.4f} (+- {sev / pred:.4f}); slope / (K d sqrt(b) c) = {slope / (K * d * sqrt(b) * C):.4f}; "
      f"sqrt(b) * mean entry of Phi_b(0) = {sqrt(b) * (T0 - adj).mean() / (K * d):+.5f} (predicted {-(2 / 6) * AC * C:+.5f})")
