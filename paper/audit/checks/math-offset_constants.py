"""Audit math-offset, check 1: constants a_{K,d}, c_{K,d} of equation (3) of paper_saga_to_muon.tex.

Independent of muon/theory_checks/check_offset_first_order.py:
  (i)   exact Haar average over U, V of Z = U diag(s) V^T (derived for the audit):
            a_{K,d} = 3 [(2K+2d+3) E sum_k s_k - 2 E sum_k s_k^3] / (K (K+2) d (d+2)),
            a/c     = 3 [(2K+2d+3) - 2 E[sum s^3]/E[sum s]] / ((K+2)(d+2)),   c = E sum s / (K d);
        only singular values are sampled (ratio estimator, delta-method standard error);
  (ii)  direct Monte Carlo of a = -E[polar(Z)_11 He_3(Z_11)] (average over entries) with batch standard errors;
  (iii) K = 2: exact two-dimensional quadrature (rows rho_i w_i, angle between the rows);
  (iv)  K = 1: closed form 3/(d+2);
  (v)   c as the Jacobian of the Gaussian mean field at 0 (central finite difference, common random numbers);
  (vi)  scalar case: b * median(Gamma(b,1)/b - 1) -> -1/3 = -kappa_3/(6 sigma^2).
Run from the project root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 paper/audit/checks/math-offset_constants.py
"""
import numpy as np
from math import sqrt, pi
from scipy.special import gammaln
from scipy.stats import gamma as gamma_law
from numpy.polynomial.legendre import leggauss

rng = np.random.default_rng(771203)


def sv_moments(K, d, S, chunk):
    """Samples of p1 = sum s, p3 = sum s^3 for S standard Gaussian K x d matrices."""
    p1 = np.empty(S); p3 = np.empty(S)
    for i in range(0, S, chunk):
        s = np.linalg.svd(rng.standard_normal((chunk, K, d)), compute_uv=False)
        p1[i:i + chunk] = s.sum(1); p3[i:i + chunk] = (s ** 3).sum(1)
    return p1, p3


def ratio_formula(K, d, p1, p3):
    S = len(p1)
    m1, m3 = p1.mean(), p3.mean()
    r = m3 / m1
    # delta method for the ratio estimator
    se_r = np.std(p3 - r * p1, ddof=1) / (m1 * sqrt(S))
    ac = 3 * ((2 * K + 2 * d + 3) - 2 * r) / ((K + 2) * (d + 2))
    se_ac = 6 * se_r / ((K + 2) * (d + 2))
    c = m1 / (K * d)
    se_c = np.std(p1, ddof=1) / sqrt(S) / (K * d)
    a = 3 * ((2 * K + 2 * d + 3) * m1 - 2 * m3) / (K * (K + 2) * d * (d + 2))
    return a, c, se_c, ac, se_ac


def direct_a(K, d, S, chunk):
    vals = []
    for _ in range(S // chunk):
        Z = rng.standard_normal((chunk, K, d))
        U, s, Vt = np.linalg.svd(Z, full_matrices=False)
        vals.append(-((U @ Vt) * (Z ** 3 - 3 * Z)).mean(axis=(1, 2)))
    v = np.concatenate(vals)
    return v.mean(), v.std(ddof=1) / sqrt(len(v))


def exact_K2(d, nq=200):
    """Exact a_{2,d}, c_{2,d}: rows rho_1 w_1, rho_2 w_2, (rho_1, rho_2) = R (cos al, sin al), t = <w_1, w_2> = sin(phi).
    s_1 + s_2 = R h, s_1^3 + s_2^3 = R^3 h (1 - q/2), h = sqrt(1 + q), q = sin(2 al) cos(phi).
    Densities: al ~ (cos al sin al)^(d-1) on [0, pi/2]; phi ~ cos(phi)^(d-2) on [-pi/2, pi/2]; R ~ chi_{2d}."""
    g, w = leggauss(nq)
    al, wal = (g + 1) * pi / 4, w * pi / 4
    ph, wph = g * pi / 2, w * pi / 2
    A, P = np.meshgrid(al, ph, indexing="ij")
    W = wal[:, None] * wph[None, :] * (np.cos(A) * np.sin(A)) ** (d - 1) * np.cos(P) ** (d - 2)
    W = W / W.sum()
    q = np.sin(2 * A) * np.cos(P)
    h = np.sqrt(1 + q)
    Eh, Eh3 = (W * h).sum(), (W * h * (1 - q / 2)).sum()
    ER = sqrt(2) * np.exp(gammaln(d + 0.5) - gammaln(d))            # E chi_{2d}
    ER3 = (2 * d + 1) * ER
    m1, m3 = ER * Eh, ER3 * Eh3
    a = 3 * ((2 * 2 + 2 * d + 3) * m1 - 2 * m3) / (2 * 4 * d * (d + 2))
    c = m1 / (2 * d)
    return a, c


print("== (iii) K = 2, exact quadrature ==")
for d in (2, 3):
    for nq in (100, 200):
        a, c = exact_K2(d, nq)
        print(f"   K = 2, d = {d}, {nq} nodes: a = {a:.6f}, c = {c:.6f}, a/c = {a / c:.6f}, d a/c = {d * a / c:.5f}")
a23, c23 = exact_K2(3)
print(f"   prediction of (3) for 2 x 3, Exp(1)-1 (sigma = 1, kappa_3 = 2): b ||G_b||_* -> (2/6)(a/c) sqrt(6) = "
      f"{(2 / 6) * (a23 / c23) * sqrt(6):.5f};  sqrt(b) ||Phi_b(0)||_F -> (2/6) a sqrt(6) = {(2 / 6) * a23 * sqrt(6):.5f}")

print("== (i)-(ii) ten sizes of the paper (and 1 x 1): singular-value formula and direct Monte Carlo ==")
sizes = [(1, 1), (2, 2), (2, 3), (3, 5), (4, 4), (8, 8), (16, 16), (32, 32), (4, 16), (8, 32), (16, 64)]
res = {}
for (K, d) in sizes:
    S = 400_000 if K * d <= 64 else (200_000 if K * d <= 256 else 100_000)
    p1, p3 = sv_moments(K, d, S, 20_000 if K * d <= 256 else 5_000)
    a, c, se_c, ac, se_ac = ratio_formula(K, d, p1, p3)
    Sd = 400_000 if K * d <= 64 else (100_000 if K * d <= 256 else 30_000)
    ad, se_ad = direct_a(K, d, Sd, 20_000 if K * d <= 256 else 5_000)
    m = max(K, d)
    res[(K, d)] = (a, c, ac, se_ac)
    print(f"   {K:2d} x {d:2d}: c = {c:.5f} (+- {se_c:.5f}); a (sv formula) = {a:.5f}; a (direct MC) = {ad:.5f} (+- {se_ad:.5f}); "
          f"a/c = {ac:.5f} (+- {se_ac:.5f}); max(K,d) a/c = {m * ac:.4f} (+- {m * se_ac:.4f}); "
          f"sqrt(Kd) a/c = {sqrt(K * d) * ac:.4f}", flush=True)
vals = [max(K, d) * res[(K, d)][2] for (K, d) in sizes[1:]]
print(f"   range of max(K,d) a/c over the ten sizes: {min(vals):.4f} .. {max(vals):.4f}")

print("== asymptotics: square (limit 2.4 from the quarter-circle law) and wide (limit 3) ==")
for (K, d, S, ch) in [(64, 64, 20_000, 2_000), (128, 128, 4_000, 500), (2, 64, 200_000, 20_000), (4, 256, 40_000, 4_000),
                      (2, 512, 40_000, 4_000)]:
    p1, p3 = sv_moments(K, d, S, ch)
    a, c, se_c, ac, se_ac = ratio_formula(K, d, p1, p3)
    print(f"   {K:3d} x {d:3d}: a/c = {ac:.6f} (+- {se_ac:.6f}); max(K,d) a/c = {max(K, d) * ac:.4f} (+- {max(K, d) * se_ac:.4f})",
          flush=True)

print("== (iv) K = 1: a/c = 3/(d+2) ==")
for d in (1, 3, 8, 50):
    p1, p3 = sv_moments(1, d, 400_000, 100_000)
    a, c, se_c, ac, se_ac = ratio_formula(1, d, p1, p3)
    chi = sqrt(2) * np.exp(gammaln((d + 1) / 2) - gammaln(d / 2))
    print(f"   d = {d:2d}: a/c = {ac:.5f} (+- {se_ac:.5f}), exact {3 / (d + 2):.5f}; c = {c:.5f}, exact {chi / d:.5f}; "
          f"a exact {3 * chi / (d * (d + 2)):.5f}, formula with exact chi moments: "
          f"{3 * ((2 + 2 * d + 3) * chi - 2 * (d + 1) * chi) / (3 * d * (d + 2)):.5f}")

print("== (v) c_{K,d} as the Jacobian of the Gaussian mean field at 0 (2 x 3; central difference h = 0.1) ==")
K, d, S, h = 2, 3, 2_000_000, 0.1
Z = rng.standard_normal((S, K, d))
E11 = np.zeros((K, d)); E11[0, 0] = 1.0


def polar_stack(M):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    return U @ Vt


dq = (polar_stack(Z + h * E11) - polar_stack(Z - h * E11)) / (2 * h)
m = dq.mean(0); se = dq.std(0, ddof=1) / sqrt(S)
print(f"   D Phi_Z(0)[E_11] = \n{np.array2string(m, precision=4)}\n   standard errors up to {se.max():.4f}; "
      f"entry (1,1) = {m[0, 0]:.4f} (+- {se[0, 0]:.4f}) against c_(2,3) = {c23:.4f} (bias O(h^2))")

print("== (vi) scalar case: b * median(Gamma(b,1)/b - 1), limit -1/3 ==")
print("   " + ", ".join(f"b={b}: {b * (gamma_law.median(b) / b - 1):+.5f}" for b in (1, 2, 4, 8, 16, 32, 256, 4096)))
print("   medians: " + ", ".join(f"{gamma_law.median(b) / b - 1:+.4f}" for b in (1, 2, 4, 8, 16, 32)))
