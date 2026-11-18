"""Checks for the additions to the least-squares section that come from the review of 16 November 2026
(model: F(W) = F* + tr(E Sigma E^T)/2, E = W - W*, G = E Sigma, K >= d). Most of these statements were proved by the
mathematical referee of that section; this script is an independent numerical check of each of them.

  (1) First phase: s_d(E_0) <= tau* <= min{(prod s_i(E_0))^(1/d), tau_ls} <= ||E_0||_op, and the polar factor is the
      same for every schedule of total length below tau* (and for a momentum with non-negative weights).
  (2) Second nuclear-scaled step from any full-rank start: (F_2 - F*)/(F_1 - F*) <= 1 - 4 p_d p_{d-1}/(p_d + p_{d-1}).
  (3) d = 2: Q = polar(|E| Sigma) is the rotation of angle arg(1 + rho_E rho_S exp(2 i theta)); cos >= sqrt(1 - (rho_E rho_S)^2).
  (4) d = 3: w_i^T Q w_i >= [3 g (s_i + g) - s_i (s_j + s_k)] / [(s_i + s_j)(s_i + s_k)], g = (s_1 s_2 s_3)^(1/3).
  (5) d = 2: extinction time of the flow T <= sqrt(s_1 s_2) + sqrt(2) (s_1 - s_2) <= (9 sqrt(2)/8) ||E_0||_op.
  (6) Commuting case: nuclear rank of the gradient on the invariant rays A = {k} in [2, 4 (1 - 1/d)]; spectrum of the
      linearized map {1 - 2 pi, 0, +1 (|A| - 1 times), -1 (|A^c| - 1 times)} on rays of general subsets A; for d = 2 the
      factor after the first step is (p_1 - p_2)^2.
  (7) On a rank-one error the nuclear-scaled step is gradient descent with step 1/tr(Sigma) on the row.
  (8) Selection: tr(Z Sigma Z^T) = tr(V^T Sigma V) + ||P_U^perp C P_V^perp Sigma^(1/2)||_F^2 on the subdifferential.
  (9) [numerical] Tail of the scalar map a -> |a - E_p a| (commuting case) from random starts: the index of the
      dominant amplitude settles within a few steps, is the index of the largest initial amplitude in only part of the
      runs, and the exponent per step of F is close to -2 log(1 - 2 p_k) for that index (50-digit arithmetic).
Run: .venv/Scripts/python muon/theory_checks/check_ls_additions.py
"""
import numpy as np
from math import sqrt
from scipy.integrate import quad
from scipy.linalg import eigh, sqrtm

rng = np.random.default_rng(20261117)


def polar(M, tol=1e-12):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    keep = s > tol * max(s[0], 1e-300)
    return U[:, keep] @ Vt[keep]


def psd_sqrt(A):
    w, V = np.linalg.eigh((A + A.T) / 2)
    return (V * np.sqrt(np.maximum(w, 0))) @ V.T


def random_sigma(d, kappa):
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    lam = np.exp(rng.uniform(0, np.log(kappa), d)); lam[0] = 1.0; lam[-1] = kappa
    return (Q * lam) @ Q.T


def random_error(K, d, cond):
    U, _ = np.linalg.qr(rng.standard_normal((K, d)))
    V, _ = np.linalg.qr(rng.standard_normal((d, d)))
    s = np.exp(rng.uniform(0, np.log(cond), d))
    return (U * s) @ V.T


print("== (1) first phase: bounds on tau*, frozen polar factor ==")
worst = [np.inf, np.inf, np.inf, np.inf]
frozen = 0.0
for trial in range(2000):
    d = int(rng.integers(2, 7)); K = d + int(rng.integers(0, 3))
    Sig = random_sigma(d, 10 ** rng.uniform(0, 3)); E0 = random_error(K, d, 10 ** rng.uniform(0, 2))
    G0 = E0 @ Sig
    O0 = polar(G0); Y0 = O0.T @ G0
    tau_star = eigh((Y0 + Y0.T) / 2, Sig, eigvals_only=True)[0]
    s = np.linalg.svd(E0, compute_uv=False)
    tau_ls = np.trace(Y0) / np.trace(Sig)
    worst[0] = min(worst[0], tau_star / s[-1] - 1)
    worst[1] = min(worst[1], np.exp(np.mean(np.log(s))) / tau_star - 1)
    worst[2] = min(worst[2], tau_ls / tau_star - 1)
    worst[3] = min(worst[3], s[0] / tau_ls - 1)
    if trial < 200:                                      # random schedule of total length 0.98 tau*
        etas = rng.dirichlet(np.ones(6)) * 0.98 * tau_star
        for use_momentum in (False, True):               # plain polar steps, then steps along polar(momentum)
            W = E0.copy(); mom = np.zeros_like(E0)
            for eta in etas:
                G = W @ Sig
                mom = 0.7 * mom + 0.3 * G
                direction = polar(mom) if use_momentum else polar(G)
                frozen = max(frozen, np.linalg.norm(direction - O0))
                W = W - eta * direction
print(f"   2000 instances (d = 2..6, K = d..d+2, cond(Sigma) <= 1e3, cond(E_0) <= 1e2): smallest relative slack of "
      f"tau* >= s_d: {worst[0]:.2e}; geometric mean >= tau*: {worst[1]:.2e}; tau_ls >= tau*: {worst[2]:.2e}; "
      f"||E_0||_op >= tau_ls: {worst[3]:.2e}  (all must be >= 0)")
print(f"   200 schedules of total length 0.98 tau* (6 steps; plain polar steps, and steps along the polar factor of an "
      f"exponential average of the gradients, beta = 0.7): max ||direction - O_0||_F = {frozen:.1e}")

print("== (2) second nuclear-scaled step from a full-rank start (non-commuting) ==")
worst2, eq2 = np.inf, 0.0                                # d >= 3: slack; d = 2: the bound is an equality
for trial in range(20000):
    d = int(rng.integers(2, 7)); K = d + int(rng.integers(0, 2))
    Sig = random_sigma(d, 10 ** rng.uniform(0, 2.5)); E = random_error(K, d, 10 ** rng.uniform(0, 2))
    lam = np.linalg.eigvalsh(Sig); p = lam / lam.sum()
    bound = 1 - 4 * p[0] * p[1] / (p[0] + p[1])
    F = lambda E_: 0.5 * np.trace(E_ @ Sig @ E_.T)
    G = E @ Sig
    E1 = E - np.linalg.svd(G, compute_uv=False).sum() / lam.sum() * polar(G)
    G1 = E1 @ Sig
    E2 = E1 - np.linalg.svd(G1, compute_uv=False).sum() / lam.sum() * polar(G1)
    if F(E1) > 1e-20 * F(E):
        ratio = F(E2) / F(E1)
        if d == 2:
            eq2 = max(eq2, abs(bound - ratio))
        else:
            worst2 = min(worst2, bound - ratio)
print(f"   20000 instances: d >= 3: min (bound - factor of the second step) = {worst2:.3e} (must be >= 0); "
      f"d = 2: max |bound - factor| = {eq2:.1e} (equality); and bound <= 1 - 2/kappa_op by construction")

print("== (3) d = 2: rotation angle of Q = polar(|E| Sigma) ==")
err3, worst3 = 0.0, np.inf
for trial in range(20000):
    s1, s2 = np.sort(np.exp(rng.uniform(0, np.log(1e3), 2)))[::-1]
    l1, l2 = np.sort(np.exp(rng.uniform(0, np.log(1e3), 2)))[::-1]
    a, b = rng.uniform(0, np.pi, 2)
    rot = lambda t: np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    absE = rot(a) @ np.diag([s1, s2]) @ rot(a).T
    Sig = rot(b) @ np.diag([l1, l2]) @ rot(b).T
    Q = polar(absE @ Sig)
    rhoE, rhoS = (s1 - s2) / (s1 + s2), (l1 - l2) / (l1 + l2)
    z = 1 + rhoE * rhoS * np.exp(2j * (a - b))
    err3 = max(err3, abs(Q[0, 0] - z.real / abs(z)), abs(abs(Q[1, 0]) - abs(z.imag) / abs(z)))
    worst3 = min(worst3, Q[0, 0] - sqrt(1 - (rhoE * rhoS) ** 2))
print(f"   20000 instances (condition numbers <= 1e3): max error of the angle formula {err3:.1e}; "
      f"min (cos - sqrt(1 - rho_E^2 rho_S^2)) = {worst3:.2e} (must be >= 0 up to rounding)")

print("== (4) d = 3: lower bound on the diagonal of Q in the basis of the right singular vectors of E ==")
worst4, neg_found, neg_cond = np.inf, 0, np.inf
for trial in range(200000):
    s = np.sort(np.exp(rng.uniform(0, np.log(10 ** rng.uniform(0, 4)), 3)))[::-1]
    Sig = random_sigma(3, 10 ** rng.uniform(0, 6))
    Wb, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    absE = (Wb * s) @ Wb.T
    Q = polar(absE @ Sig)
    q = np.einsum('ji,jk,ki->i', Wb, Q, Wb)
    g = np.prod(s) ** (1 / 3)
    for i in range(3):
        j, k = [x for x in range(3) if x != i]
        lb = (3 * g * (s[i] + g) - s[i] * (s[j] + s[k])) / ((s[i] + s[j]) * (s[i] + s[k]))
        worst4 = min(worst4, q[i] - lb)
    if q[0] < 0:
        neg_found += 1
        neg_cond = min(neg_cond, s[0] ** 2 / (s[1] * s[2]))
print(f"   200000 random pairs: min (q_i - bound_i) = {worst4:.2e} (must be >= 0 up to rounding); instances with "
      f"q_1 < 0: {neg_found}, smallest s_1^2/(s_2 s_3) among them: {neg_cond:.1f} (threshold (2+sqrt 3)^3 = {(2 + sqrt(3)) ** 3:.2f})")
E0 = np.diag([300.0, 200.0, 1.0]); Sig = np.array([[2.0, 4, 12], [4, 9, 20], [12, 20, 89]])
print(f"   integer example: Q_11 = {polar(E0 @ Sig)[0, 0]:+.10f}; s_1^2/(s_2 s_3) = {300 ** 2 / 200:.0f}")

print("== (5) d = 2: extinction time of the flow (exact first phase + arc length of the rank-one phase) ==")
worst5, largest = np.inf, 0.0
for trial in range(3000):
    K = 2 + int(rng.integers(0, 2))
    kap = 10 ** rng.uniform(0, 6)
    Sig = random_sigma(2, kap)
    U, _ = np.linalg.qr(rng.standard_normal((K, 2))); V, _ = np.linalg.qr(rng.standard_normal((2, 2)))
    s = np.array([1.0, rng.uniform(0, 1) ** 2])
    E0 = (U * s) @ V.T
    G0 = E0 @ Sig; O0 = polar(G0); Y0 = O0.T @ G0
    tau_star = eigh((Y0 + Y0.T) / 2, Sig, eigvals_only=True)[0]
    E1 = E0 - tau_star * O0                                # rank one at the end of the first phase
    Uu, ss, Vt = np.linalg.svd(E1)
    e = ss[0] * Vt[0]
    lam, P = np.linalg.eigh(Sig)
    c = P.T @ e
    speed = lambda x: sqrt(sum((lam[k] * c[k]) ** 2 * np.exp(-2 * lam[k] * np.exp(x)) for k in range(2))) * np.exp(x)
    length = quad(speed, -40, 8 - np.log(lam[0]), limit=400, epsabs=1e-12, epsrel=1e-10)[0]
    T = tau_star + length
    bound = sqrt(s[0] * s[1]) + sqrt(2) * (s[0] - s[1])
    worst5 = min(worst5, bound - T)
    largest = max(largest, T / s[0])
    assert abs(ss[0] - (s[0] - s[1])) < 1e-8 * max(1.0, kap) and ss[1] < 1e-8 * max(1.0, kap) + 1e-9
print(f"   3000 instances (K = 2, 3; cond(Sigma) <= 1e6): min (bound - T) = {worst5:.2e} (must be >= 0); "
      f"largest T/||E_0||_op = {largest:.4f} (sqrt 2 = {sqrt(2):.4f}, 9 sqrt(2)/8 = {9 * sqrt(2) / 8:.4f}); "
      f"at tau* the error has rank one with singular value s_1 - s_2 (checked)")

print("== (6) commuting case: invariant rays ==")
lo, hi, spec_err = np.inf, 0.0, 0.0
for trial in range(5000):
    d = int(rng.integers(2, 12))
    p = rng.dirichlet(np.ones(d) * rng.uniform(0.2, 3))
    k = int(np.argmin(p)) if rng.uniform() < 0.5 else int(rng.integers(0, d))
    if p[k] >= 0.5:
        continue
    pi = p[k]
    a = np.full(d, pi); a[k] = 1 - pi                     # ray A = {k}
    nxt = np.abs(a - p @ a)
    assert np.allclose(nxt, (1 - 2 * pi) * a)
    g = p * a                                             # singular values of G up to tr(Sigma)
    nr = g.sum() ** 2 / (g ** 2).sum()
    lo, hi = min(lo, nr - 2), max(hi, nr - 4 * (1 - 1 / d))
    sgn = np.sign(a - p @ a)
    Jac = np.diag(sgn) @ (np.eye(d) - np.outer(np.ones(d), p))
    ev = np.sort(np.linalg.eigvals(Jac).real)
    target = np.sort(np.array([1 - 2 * pi, 0.0] + [-1.0] * (d - 2)))
    spec_err = max(spec_err, np.abs(ev - target).max())
print(f"   5000 rays A = {{k}}: min (nr - 2) = {lo:.2e}, max (nr - 4(1 - 1/d)) = {hi:.2e}; "
      f"spectrum of the linearized map vs {{1 - 2 pi, 0, -1 (d - 2 times)}}: max error {spec_err:.1e}")
spec_err_A, n_rays = 0.0, 0
for trial in range(4000):                                # rays of general subsets A with p(A) < 1/2
    d = int(rng.integers(3, 11))
    p = rng.dirichlet(np.ones(d) * rng.uniform(0.3, 3))
    A = rng.uniform(size=d) < rng.uniform(0.15, 0.6)
    pi = p[A].sum()
    if not (0 < A.sum() < d) or pi >= 0.49:
        continue
    a = np.where(A, 1 - pi, pi)
    assert np.allclose(np.abs(a - p @ a), (1 - 2 * pi) * a)
    Jac = np.diag(np.sign(a - p @ a)) @ (np.eye(d) - np.outer(np.ones(d), p))
    ev = np.sort(np.linalg.eigvals(Jac).real)
    target = np.sort(np.array([1 - 2 * pi, 0.0] + [1.0] * (int(A.sum()) - 1) + [-1.0] * (d - int(A.sum()) - 1)))
    spec_err_A = max(spec_err_A, np.abs(ev - target).max()); n_rays += 1
print(f"   {n_rays} rays of general subsets A (d = 3..10, |A| = 1..d-1): spectrum of the linearized map vs "
      f"{{1 - 2 pi, 0, +1 (|A| - 1 times), -1 (|A^c| - 1 times)}}: max error {spec_err_A:.1e}")
for p1 in (0.1, 0.3, 0.45):
    p = np.array([p1, 1 - p1]); a = rng.uniform(0.2, 2, 2)
    fac = []
    for t in range(4):
        m = p @ a
        fac.append(((p * (a - m) ** 2).sum()) / (p * a * a).sum())
        a = np.abs(a - m)
    print(f"   d = 2, p = ({p1}, {1 - p1}): factors at t = 0..3: " + ", ".join(f"{x:.6f}" for x in fac)
          + f"; (p_1 - p_2)^2 = {(1 - 2 * p1) ** 2:.6f}")

print("== (7) rank-one error: nuclear-scaled step = gradient descent with step 1/tr(Sigma) on the row ==")
d, K = 5, 7
Sig = random_sigma(d, 50.0)
u = rng.standard_normal(K); u /= np.linalg.norm(u); e = rng.standard_normal(d)
E = np.outer(u, e); err7 = 0.0
for t in range(30):
    G = E @ Sig
    E = E - np.linalg.svd(G, compute_uv=False).sum() / np.trace(Sig) * polar(G)
    e = e - Sig @ e / np.trace(Sig)
    err7 = max(err7, np.linalg.norm(E - np.outer(u, e)) / np.linalg.norm(e))
print(f"   30 steps: max relative deviation {err7:.1e}")

print("== (8) selection of the flow: tr(Z Sigma Z^T) on the subdifferential of the nuclear norm ==")
err8, minexcess = 0.0, np.inf
for trial in range(2000):
    K, d = 6, 5
    rk = int(rng.integers(1, 5))
    G = rng.standard_normal((K, rk)) @ rng.standard_normal((rk, d))
    Sig = random_sigma(d, 100.0)
    U, s, Vt = np.linalg.svd(G); U1, V1 = U[:, :rk], Vt[:rk].T
    PU = np.eye(K) - U1 @ U1.T; PV = np.eye(d) - V1 @ V1.T
    C = rng.standard_normal((K, d)); C /= np.linalg.norm(C, 2) * rng.uniform(1, 3)
    N = PU @ C @ PV
    Z = U1 @ V1.T + N
    lhs = np.trace(Z @ Sig @ Z.T)
    rhs = np.trace(V1.T @ Sig @ V1) + np.linalg.norm(N @ psd_sqrt(Sig)) ** 2
    err8 = max(err8, abs(lhs - rhs) / rhs)
    minexcess = min(minexcess, lhs - np.trace(V1.T @ Sig @ V1))
print(f"   2000 rank-deficient gradients with random subgradients: max relative error of the identity {err8:.1e}; "
      f"min excess over the thin polar factor {minexcess:.2e} (>= 0)")

print("== (9) [numerical] commuting case: which amplitude dominates the tail of the scalar map ==")
import mpmath as mp
mp.mp.dps = 50
for name, lam in (("1/k", 1.0 / np.arange(1, 21.0)), ("1/k^2", 1.0 / np.arange(1, 21.0) ** 2)):
    p = [mp.mpf(float(x)) for x in lam / lam.sum()]
    same, last_change, ratios = 0, [], []
    runs, steps = 40, 400
    for run in range(runs):
        a0 = np.abs(rng.standard_normal(20)) if run % 2 == 0 else rng.uniform(0.0, 1.0, 20)
        a = [mp.mpf(float(x)) for x in a0]
        dom, change, F_hist = [], 0, []
        for t in range(steps):
            m = mp.fsum(pk * ak for pk, ak in zip(p, a))
            a = [abs(ak - m) for ak in a]
            k = max(range(20), key=lambda i: a[i])
            if dom and k != dom[-1]:
                change = t
            dom.append(k)
            F_hist.append(mp.fsum(pk * ak * ak for pk, ak in zip(p, a)))
        kf = dom[-1]
        same += int(kf == int(np.argmax(a0)))
        last_change.append(change)
        expo = mp.log(F_hist[steps // 2] / F_hist[-1]) / (steps - 1 - steps // 2)
        ratios.append(float(expo / (-2 * mp.log(1 - 2 * p[kf]))))
    print(f"   spectrum {name}, d = 20, {runs} random starts, {steps} steps: final dominant index = index of the largest "
          f"initial amplitude in {same}/{runs} runs; last change of the dominant index at step <= {max(last_change)} "
          f"(median {int(np.median(last_change))}); exponent of F over the second half / (-2 log(1 - 2 p_k)): "
          f"{min(ratios):.3f} to {max(ratios):.3f}")
