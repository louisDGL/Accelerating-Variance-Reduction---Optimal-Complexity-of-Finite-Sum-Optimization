"""Audit math-offset, check 2: zero of the mean field for 2 x 3 noise with entries Exp(1) - 1 (lines 482-484 of
paper_saga_to_muon.tex: prediction 0.464, measured 0.450-0.498 for b = 2..256 from 1.5e6 exactly sampled batch means).

Part A (argument "A"): replay of muon/theory_checks/check_minibatch_offset_exact.py with the same generator, seed and
    order of draws (seed 2026, S = 1.5e6, Gamma(b,1)/b - 1, then the sign draw), but with a closed-form polar factor
    of 2 x d matrices instead of the SVD (the SVD version needs about ten minutes); the six-dimensional root is solved
    with the same call (hybr, tol 1e-13). The stored outputs of the original script are not in the repository.
    A standard error of b ||G_b||_* is added (delta method).
Part B (argument "B"): independent estimate. Other seed, S = 3e6 per b, one-dimensional root along the all-ones matrix
    (the law is invariant under permutations of rows and columns, so the unique minimizer is proportional to 1 1^T),
    with two control variates of known mean (sum of the standardized entries, sum of their He_3) and a delta-method
    standard error. Gives b ||G_b||_* (+- se) for b = 1..1024 against the prediction 0.46379 of equation (3).
Run from the project root (one part at a time):
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 paper/audit/checks/math-offset_zero_meanfield.py A
"""
import sys
import numpy as np
from math import sqrt
from scipy.optimize import root, brentq

K, d = 2, 3
PRED = 0.463787           # (2/6) (a/c) sqrt(6), a = 0.286449, c = 0.504295 (exact quadrature, math-offset_constants.py)
C23 = 0.504295


def polar2(M):
    """Polar factor of a stack of 2 x d matrices of rank 2: (M M^T)^(-1/2) M in closed form."""
    m1, m2 = M[:, 0, :], M[:, 1, :]
    a11 = np.einsum('ij,ij->i', m1, m1); a22 = np.einsum('ij,ij->i', m2, m2); a12 = np.einsum('ij,ij->i', m1, m2)
    dl = np.sqrt(np.maximum(a11 * a22 - a12 * a12, 0.0))
    den = dl * np.sqrt(a11 + a22 + 2 * dl)
    out = np.empty_like(M)
    out[:, 0, :] = (((a22 + dl) / den)[:, None] * m1 - (a12 / den)[:, None] * m2)
    out[:, 1, :] = ((-(a12 / den))[:, None] * m1 + ((a11 + dl) / den)[:, None] * m2)
    return out


def self_test():
    r = np.random.default_rng(1)
    M = r.standard_normal((1000, K, d))
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    err = np.abs(polar2(M) - U @ Vt).max()
    print(f"closed-form polar against SVD on 1000 Gaussian 2 x 3 matrices: max entry error {err:.1e}")


def part_A():
    rng = np.random.default_rng(2026)
    S = 1_500_000
    vals = []
    for b in [2, 4, 8, 16, 32, 64, 128, 256]:
        xi = rng.gamma(b, 1.0, size=(S, K, d)) / b - 1.0
        phi0 = polar2(xi).mean(0)

        def field(g):
            return polar2(g.reshape(K, d)[None] + xi).mean(0).reshape(-1)

        sol = root(field, np.zeros(K * d), method='hybr', tol=1e-13)
        Gb = sol.x.reshape(K, d)
        _ = rng.choice([-1.0, 1.0], size=(S, 1, 1))              # keeps the stream of the original script
        nucG = np.linalg.svd(Gb, compute_uv=False).sum()
        cos = Gb.sum() / (np.linalg.norm(Gb) * sqrt(K * d))
        T = polar2(Gb[None] + xi).sum(axis=(1, 2))
        se = sqrt(b) * T.std(ddof=1) / (sqrt(6) * sqrt(S) * C23)   # se of b sqrt(6) g, slope of E T about 6 sqrt(b) c
        vals.append(b * nucG)
        print(f"b = {b:3d}: sqrt(b) ||Phi_b(0)||_F = {sqrt(b) * np.linalg.norm(phi0):.4f};  b ||G_b||_* = {b * nucG:.4f} "
              f"(se of the all-ones component about {se:.4f});  cosine with 1 1^T = {cos:.5f};  residual "
              f"{np.linalg.norm(field(sol.x)):.0e}; nfev = {sol.nfev}", flush=True)
    print(f"range of b ||G_b||_* over b = 2..256: {min(vals):.4f} .. {max(vals):.4f}  (prediction {PRED:.4f})")


def part_B():
    rng = np.random.default_rng(90210)
    S = 3_000_000
    print(f"independent estimate, S = {S}, one-dimensional root along 1 1^T with control variates")
    for b in [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024]:
        xi = rng.gamma(b, 1.0, size=(S, K, d)) / b - 1.0
        Sd = sqrt(b) * xi                                            # standardized entries
        C1 = Sd.sum(axis=(1, 2))                                     # mean 0
        C3 = (Sd ** 3 - 3 * Sd).sum(axis=(1, 2)) - K * d * 2 / sqrt(b)   # mean 0 (skewness of Gamma(b) is 2/sqrt(b))
        X = np.stack([C1, C3], axis=1)
        T0 = polar2(xi).sum(axis=(1, 2))
        beta = np.linalg.lstsq(X - X.mean(0), T0 - T0.mean(), rcond=None)[0]
        adj = X @ beta

        def f(g):
            return (polar2(g + xi).sum(axis=(1, 2)) - adj).mean()

        g0 = PRED / (b * sqrt(6))
        gb = brentq(f, -2 * g0, 4 * g0, xtol=1e-14, rtol=1e-12)
        h = 0.05 * g0
        slope = (f(gb + h) - f(gb - h)) / (2 * h)
        Tg = polar2(gb + xi).sum(axis=(1, 2))
        se_plain = Tg.std(ddof=1) / sqrt(S) / slope
        se_cv = (Tg - adj).std(ddof=1) / sqrt(S) / slope
        val = b * sqrt(6) * gb
        print(f"b = {b:4d}: b ||G_b||_* = {val:.4f} (+- {b * sqrt(6) * se_cv:.4f}; without control variates "
              f"{b * sqrt(6) * se_plain:.4f});  deviation from {PRED:.4f}: {val - PRED:+.4f}, times b: {b * (val - PRED):+.3f};  "
              f"slope / (6 sqrt(b) c) = {slope / (6 * sqrt(b) * C23):.4f};  sqrt(b) ||Phi_b(0)||_F = "
              f"{-sqrt(b) * (T0 - adj).mean() / sqrt(6):.4f}", flush=True)


if __name__ == "__main__":
    self_test()
    {"A": part_A, "B": part_B}[sys.argv[1]]()
