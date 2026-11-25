"""Audit math-offset, check 3: is equation (3) of paper_saga_to_muon.tex "parameter-free"? Test on cases that the
repository does not cover: other variance, other skewness, a square size (discontinuity of polar of codimension one)
and a wider one. Entries Gamma(alpha, theta) - alpha theta: sigma^2 = alpha theta^2, kappa_3 = 2 alpha theta^3; the
batch mean is exactly Gamma(b alpha, theta / b) - alpha theta.
Prediction of (3): b ||G_b||_* -> kappa_3 / (6 sigma^2) * (a/c) * sqrt(K d), a/c exact for K = 2 (quadrature).
Zero of the mean field: one-dimensional root along 1 1^T, S = 3e6, two control variates, delta-method standard error.
Run from the project root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 paper/audit/checks/math-offset_other_laws.py
"""
import numpy as np
from math import sqrt, pi
from scipy.optimize import brentq
from scipy.special import gammaln
from numpy.polynomial.legendre import leggauss


def exact_ratio_K2(d, nq=200):
    g, w = leggauss(nq)
    al, wal = (g + 1) * pi / 4, w * pi / 4
    ph, wph = g * pi / 2, w * pi / 2
    A, P = np.meshgrid(al, ph, indexing="ij")
    W = wal[:, None] * wph[None, :] * (np.cos(A) * np.sin(A)) ** (d - 1) * np.cos(P) ** (d - 2)
    W = W / W.sum()
    q = np.sin(2 * A) * np.cos(P)
    h = np.sqrt(1 + q)
    r = (2 * d + 1) * (W * h * (1 - q / 2)).sum() / (W * h).sum()        # E sum s^3 / E sum s
    c = sqrt(2) * np.exp(gammaln(d + 0.5) - gammaln(d)) * (W * h).sum() / (2 * d)
    return 3 * ((2 * 2 + 2 * d + 3) - 2 * r) / (4 * (d + 2)), c


def polar2(M):
    m1, m2 = M[:, 0, :], M[:, 1, :]
    a11 = np.einsum('ij,ij->i', m1, m1); a22 = np.einsum('ij,ij->i', m2, m2); a12 = np.einsum('ij,ij->i', m1, m2)
    if M.shape[2] == 2:                      # square case: |det M| directly (the Gram form cancels near singular M)
        dl = np.abs(m1[:, 0] * m2[:, 1] - m1[:, 1] * m2[:, 0])
    else:
        dl = np.sqrt(np.maximum(a11 * a22 - a12 * a12, 0.0))
    den = dl * np.sqrt(a11 + a22 + 2 * dl)
    den = np.where(den > 0, den, np.inf)     # exactly singular samples (probability zero): polar set to 0
    out = np.empty_like(M)
    out[:, 0, :] = (((a22 + dl) / den)[:, None] * m1 - (a12 / den)[:, None] * m2)
    out[:, 1, :] = ((-(a12 / den))[:, None] * m1 + ((a11 + dl) / den)[:, None] * m2)
    return out


rng = np.random.default_rng(424242)
S, K = 3_000_000, 2
for (d, alpha, theta) in [(2, 1.0, 3.0), (2, 0.25, 2.0), (5, 0.25, 2.0)]:
    sig2, k3 = alpha * theta ** 2, 2 * alpha * theta ** 3
    sig = sqrt(sig2)
    ac, c = exact_ratio_K2(d)
    pred = k3 / (6 * sig2) * ac * sqrt(K * d)
    print(f"2 x {d}, entries Gamma({alpha}, {theta}) centred: sigma^2 = {sig2}, kappa_3 = {k3}, a/c = {ac:.5f}; "
          f"prediction b ||G_b||_* -> {pred:.4f}")
    for b in [16, 64, 256]:
        xi = rng.gamma(b * alpha, theta / b, size=(S, K, d)) - alpha * theta
        Sd = sqrt(b) * xi / sig
        C1 = Sd.sum(axis=(1, 2))
        C3 = (Sd ** 3 - 3 * Sd).sum(axis=(1, 2)) - K * d * (k3 / sig ** 3) / sqrt(b)
        X = np.stack([C1, C3], axis=1)
        T0 = polar2(xi).sum(axis=(1, 2))
        beta = np.linalg.lstsq(X - X.mean(0), T0 - T0.mean(), rcond=None)[0]
        adj = X @ beta

        def f(g):
            return (polar2(g + xi).sum(axis=(1, 2)) - adj).mean()

        g0 = pred / (b * sqrt(K * d))
        gb = brentq(f, -2 * g0, 4 * g0, xtol=1e-14, rtol=1e-12)
        hh = 0.05 * g0
        slope = (f(gb + hh) - f(gb - hh)) / (2 * hh)
        se = (polar2(gb + xi).sum(axis=(1, 2)) - adj).std(ddof=1) / sqrt(S) / slope
        val, sev = b * sqrt(K * d) * gb, b * sqrt(K * d) * se
        print(f"   b = {b:3d}: b ||G_b||_* = {val:.4f} (+- {sev:.4f}); ratio to the prediction {val / pred:.4f} (+- {sev / pred:.4f}); "
              f"slope sigma / (K d sqrt(b) c) = {slope * sig / (K * d * sqrt(b) * c):.4f}", flush=True)
