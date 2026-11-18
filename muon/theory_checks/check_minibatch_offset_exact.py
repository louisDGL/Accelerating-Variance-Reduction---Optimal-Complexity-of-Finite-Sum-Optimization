"""Offset of the mean-field fixed point of minibatch polar steps versus the batch size, with continuous skewed noise
whose batch mean can be sampled exactly.

Additive-noise model: stochastic gradient = G + xi_b, xi_b the mean of b i.i.d. centered noise matrices with i.i.d.
entries Exp(1) - 1 (skewness 2). The entries of xi_b are then exactly Gamma(b, 1)/b - 1, so xi_b is sampled directly
for any b. Mean field: Phi_b(G) = E polar(G + xi_b) = gradient of psi_b(G) = E||G + xi_b||_* (Theorem 5.1 of
notes_spectral_theory). Fixed point G_b: Phi_b(G_b) = 0, i.e. -G_b is the nuclear median of xi_b (Theorem 5.3).
Checked: (1) sqrt(b) ||Phi_b(0)||_F tends to a nonzero constant (the bias at the true stationary point is of order
b^(-1/2) in the direction field); (2) b ||G_b|| tends to a constant (offset of order 1/b in gradient space); (3) with
symmetrized noise (random global sign) Phi_b(0) = 0 and G_b = 0 up to Monte-Carlo error.
A small numpy/scipy script; common random numbers (one sample of S noise matrices per b) make the field smooth in G.
Run: .venv/Scripts/python muon/theory_checks/check_minibatch_offset_exact.py
"""
import numpy as np
from scipy.optimize import root

rng = np.random.default_rng(2026)
K, d, S = 2, 3, 1_500_000


def polar_stack(M):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    return U @ Vt


def nuc(M):
    return np.linalg.svd(M, compute_uv=False).sum()


rows = []
for b in [2, 4, 8, 16, 32, 64, 128, 256]:
    xi = rng.gamma(b, 1.0, size=(S, K, d)) / b - 1.0
    phi0 = polar_stack(xi).mean(0)
    se = 1.0 / np.sqrt(S)                                        # order of the Monte-Carlo error per entry

    def field(g):
        return polar_stack(g.reshape(K, d)[None] + xi).mean(0).reshape(-1)

    sol = root(field, np.zeros(K * d), method='hybr', tol=1e-13)
    Gb = sol.x.reshape(K, d)
    # symmetric control: same magnitudes, random global sign
    sgn = rng.choice([-1.0, 1.0], size=(S, 1, 1))
    phi0_sym = polar_stack(sgn * xi).mean(0)
    rows.append((b, np.linalg.norm(phi0), nuc(Gb), np.linalg.norm(field(sol.x)), np.linalg.norm(phi0_sym), se))
    cos_ones = Gb.sum() / (np.linalg.norm(Gb) * np.sqrt(K * d))  # cosine between the fixed point and the all-ones matrix
    print(f"b = {b:3d}: ||Phi_b(0)||_F = {rows[-1][1]:.5f} (sqrt(b) x = {np.sqrt(b) * rows[-1][1]:.4f});  "
          f"||G_b||_* = {rows[-1][2]:.6f} (b x = {b * rows[-1][2]:.4f}), cosine with the all-ones matrix {cos_ones:.5f};  "
          f"residual {rows[-1][3]:.0e};  "
          f"symmetrized noise: ||Phi_b(0)||_F = {rows[-1][4]:.5f} (Monte-Carlo error per entry ~ {se:.5f})", flush=True)
