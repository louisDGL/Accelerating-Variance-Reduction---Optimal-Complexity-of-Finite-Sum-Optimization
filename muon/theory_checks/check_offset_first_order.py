"""First-order formula for the offset of the mean-field fixed point of minibatch polar steps (Proposition 5.5(b)).

Noise model: xi_b = mean of b i.i.d. centered K x d matrices with i.i.d. entries of variance sigma^2 and third cumulant
kappa_3. Formal Edgeworth expansion around the Gaussian limit (Z standard Gaussian K x d):

    E polar(xi_b)       ~  - b^(-1/2) * kappa_3/(6 sigma^3) * a_{K,d} * J          (J = all-ones matrix)
    D E polar(G + xi_b) ~    b^(1/2)  * c_{K,d}/sigma * Id                         (at G = 0)
    fixed point G_b     ~    kappa_3/(6 sigma^2 b) * (a_{K,d}/c_{K,d}) * J,

with  a_{K,d} = - E[polar(Z)_11 (Z_11^3 - 3 Z_11)]  and  c_{K,d} = E||Z||_* / (K d)  (the linear-response constant of
Xie, arXiv:2609.32861, and Chen et al., arXiv:2610.31728). For K = d = 1: a = c = sqrt(2/pi) and the formula is the
Cornish-Fisher value kappa_3/(6 sigma^2 b) of the median of a mean.

For K = 1 (normalized gradient step) the constants are exact: a = 3 E chi_d / (d (d + 2)), c = E chi_d / d, so
a/c = 3/(d + 2) (and in general a_{K,d} = - E[||Z||_* He_4(Z_11)], by Gaussian integration by parts).

For (K, d) = (2, 3) the constants are computed by a two-dimensional quadrature (a = 0.2864, c = 0.5043, d a/c = 1.704).

Checked here: (1) the constants by Monte Carlo, the scalar case and the case K = 1; (2) the prediction against the zero of the mean
field computed directly (entries Gamma(b,1)/b - 1: sigma = 1, kappa_3 = 2), in norm and in direction;
(3) the dependence of a/c on the size of the matrix (about 1/d): the offset per entry is much smaller than for sign
descent, for which every entry has the scalar offset kappa_3/(6 sigma^2 b).
Run: .venv/Scripts/python muon/theory_checks/check_offset_first_order.py
"""
import numpy as np
from math import sqrt, pi
from scipy.optimize import root

rng = np.random.default_rng(20261116)


def constants(K, d, S, chunk):
    a_acc = a_sq = c_acc = 0.0
    n = 0
    for _ in range(S // chunk):
        Z = rng.standard_normal((chunk, K, d))
        U, s, Vt = np.linalg.svd(Z, full_matrices=False)
        v = -((U @ Vt) * (Z ** 3 - 3 * Z)).mean(axis=(1, 2))        # average over the entries (exchangeable)
        a_acc += v.sum(); a_sq += (v ** 2).sum(); c_acc += s.sum(); n += chunk
    a = a_acc / n
    return a, sqrt((a_sq / n - a * a) / n), c_acc / n / (K * d)


print("== constants a_{K,d}, c_{K,d} (Monte Carlo) ==")
print(f"   scalar reference: a = c = sqrt(2/pi) = {sqrt(2 / pi):.5f}")
table = {}
for (K, d, S, ch) in [(1, 1, 2_000_000, 500_000), (2, 2, 2_000_000, 500_000), (2, 3, 2_000_000, 500_000),
                      (3, 5, 1_000_000, 250_000), (4, 4, 1_000_000, 250_000), (8, 8, 400_000, 100_000),
                      (16, 16, 100_000, 25_000), (32, 32, 30_000, 5_000), (4, 16, 400_000, 100_000),
                      (8, 32, 100_000, 25_000), (16, 64, 30_000, 5_000)]:
    a, a_se, c = constants(K, d, S, ch)
    table[(K, d)] = (a, c)
    print(f"   K = {K:2d}, d = {d:2d}: a = {a:.5f} (+- {a_se:.5f}); c = {c:.4f}; a/c = {a / c:.4f}; d * a/c = {d * a / c:.3f}; "
          f"nuclear norm of the offset in units kappa_3/(6 sigma^2 b): (a/c) sqrt(Kd) = {(a / c) * sqrt(K * d):.3f} "
          f"(sign descent, l1 norm: K d = {K * d})", flush=True)

print("== K = 1 (normalized gradient step): exact values a = 3 E chi_d / (d (d + 2)), c = E chi_d / d, a/c = 3/(d + 2) ==")
from scipy.special import gammaln
for d1 in (3, 8):
    a, a_se, c = constants(1, d1, 1_000_000, 250_000)
    chi = sqrt(2) * np.exp(gammaln((d1 + 1) / 2) - gammaln(d1 / 2))
    print(f"   K = 1, d = {d1}: a = {a:.5f} (+- {a_se:.5f}; exact {3 * chi / (d1 * (d1 + 2)):.5f}); c = {c:.5f} "
          f"(exact {chi / d1:.5f}); a/c = {a / c:.4f} (exact {3 / (d1 + 2):.4f})", flush=True)

print("== (K, d) = (2, 3): exact values by quadrature ==")
# Rows z1 = rho1 w1, z2 = rho2 w2 of Z (rho_i ~ chi_3, w_i uniform on the sphere of R^3): ||Z||_* = sqrt(rho1^2 +
# rho2^2 + 2 rho1 rho2 sqrt(1 - t^2)) with t = <w1, w2> uniform on [-1, 1], independent of the first coordinate u of w1,
# also uniform on [-1, 1]; a = - E[||Z||_* He_4(rho1 u)] and E[He_4(rho u) | rho] = rho^4/5 - 2 rho^2 + 3. In polar
# coordinates (rho1, rho2) = R (cos al, sin al) and with t = sin(phi), the integral over R is a Gamma function.
from scipy.special import gamma as _gamma
from numpy.polynomial.legendre import leggauss


def _radial(k):                                       # int_0^infty R^k exp(-R^2/2) dR
    return 2 ** ((k - 1) / 2) * _gamma((k + 1) / 2)


for nq in (40, 80):
    g, w = leggauss(nq)
    al, wal = (g + 1) * pi / 4, w * pi / 4            # al in [0, pi/2]
    ph, wph = g * pi / 2, w * pi / 2                  # phi in [-pi/2, pi/2], density cos(phi)/2
    A, P = np.meshgrid(al, ph, indexing="ij")
    W2 = wal[:, None] * wph[None, :] * np.cos(P) / 2 * (2 / pi) * np.cos(A) ** 2 * np.sin(A) ** 2
    h = np.sqrt(1 + np.sin(2 * A) * np.cos(P))        # ||Z||_* = R h
    a23 = -np.sum(W2 * h * (_radial(10) * np.cos(A) ** 4 / 5 - 2 * _radial(8) * np.cos(A) ** 2 + 3 * _radial(6)))
    c23 = np.sum(W2 * h * _radial(6)) / 6
    mass = np.sum(W2 * _radial(5))                    # total mass of the law of (rho1, rho2, t): must be 1
    print(f"   {nq} x {nq} nodes: mass = {mass:.12f}; a = {a23:.6f}; c = {c23:.6f}; a/c = {a23 / c23:.5f}; "
          f"d * a/c = {3 * a23 / c23:.4f}   (Monte Carlo above: a = {table[(2, 3)][0]:.5f}, c = {table[(2, 3)][1]:.4f})",
          flush=True)

print("== prediction against the zero of the mean field (entries Gamma(b,1)/b - 1: sigma = 1, kappa_3 = 2) ==")
S = 400_000
for (K, d) in [(2, 3), (2, 2)]:
    a, c = table[(K, d)]
    J = np.ones((K, d))
    for b in [8, 32]:
        xi = rng.gamma(b, 1.0, size=(S, K, d)) / b - 1.0

        def field(g):
            U, s, Vt = np.linalg.svd(g.reshape(K, d)[None] + xi, full_matrices=False)
            return (U @ Vt).mean(0).reshape(-1)

        phi0 = field(np.zeros(K * d)).reshape(K, d)
        sol = root(field, np.zeros(K * d), method='hybr', tol=1e-13)
        Gb = sol.x.reshape(K, d)
        pred_phi = -(2 / 6) * a / sqrt(b)
        pred_G = (2 / 6) * (a / c) / b
        cosG = np.sum(Gb * J) / (np.linalg.norm(Gb) * np.linalg.norm(J))
        print(f"   K = {K}, d = {d}, b = {b:2d}: mean entry of E polar(xi_b) = {phi0.mean():+.5f} (predicted {pred_phi:+.5f}, "
              f"Monte-Carlo error ~ {1 / sqrt(S * K * d):.5f}); mean entry of the fixed point = {Gb.mean():+.6f} "
              f"(predicted {pred_G:+.6f}); cosine with the all-ones matrix = {cosG:.4f}; ||G_b||_* x b = "
              f"{b * np.linalg.svd(Gb, compute_uv=False).sum():.4f} (predicted {(2 / 6) * (a / c) * sqrt(K * d):.4f})", flush=True)
