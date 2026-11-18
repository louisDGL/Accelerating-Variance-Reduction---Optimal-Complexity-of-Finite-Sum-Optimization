"""Numerical checks for Part 1 of notes_spectral_theory: condition numbers in the spectral geometry.

Statements checked (q(D) = <D, H D>, H symmetric positive definite on R^{m x n}, r = min(m, n)):
  P1.1a  L_op = max of q over matrices with all singular values equal to 1 (ascent Q <- polar(H Q), restarts);
  P1.1b  1/mu_op = max over unit rank-one u v^T of <u v^T, H^{-1} u v^T>, attained ratio at D* = H^{-1}(u* v*^T);
  P1.2   frame bound kappa_op >= sum_k h_k / min_k h_k >= r, and kappa_op >= kappa_F srank(V_max)/srank(V_min),
         kappa_F / r <= kappa_op <= r kappa_F;
  P1.3a  Kronecker H = A (x) B: L_op = sum_{k<=r} a_k b_k (sorted), mu_op = a_min b_min;
  P1.3b  least squares, K >= d: L_op(F) = tr(Sigma_hat) = mean_i ||x_i||^2 (no averaging gain).
Run: .venv/Scripts/python muon/theory_checks/check_condition_numbers.py
"""
import numpy as np

rng = np.random.default_rng(20261116)


def polar(M):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    return U @ Vt


def opnorm(M):
    return np.linalg.norm(M, 2)


def random_spd_operator(m, n, cond, rng):
    N = m * n
    Q, _ = np.linalg.qr(rng.standard_normal((N, N)))
    ev = np.geomspace(1.0, 1.0 / cond, N)
    return (Q * ev) @ Q.T, ev, Q


def apply(Hm, D):
    return (Hm @ D.reshape(-1)).reshape(D.shape)


def q(Hm, D):
    return float(D.reshape(-1) @ Hm @ D.reshape(-1))


def L_op(Hm, m, n, rng, restarts=200, iters=500):
    best, arg = -np.inf, None
    for _ in range(restarts):
        Q = polar(rng.standard_normal((m, n)))
        for _ in range(iters):
            Qn = polar(apply(Hm, Q))
            if np.linalg.norm(Qn - Q) < 1e-13:
                Q = Qn
                break
            Q = Qn
        val = q(Hm, Q)
        if val > best:
            best, arg = val, Q
    return best, arg


def inv_mu_op(Hinv, m, n, rng, restarts=200, iters=500):
    """max over unit u, v of <u v^T, Hinv (u v^T)> by alternating top-eigenvector updates."""
    T = Hinv.reshape(m, n, m, n)
    best, arg = -np.inf, None
    for _ in range(restarts):
        v = rng.standard_normal(n); v /= np.linalg.norm(v)
        u = rng.standard_normal(m); u /= np.linalg.norm(u)
        val_old = -np.inf
        for _ in range(iters):
            Au = np.einsum('ajbk,j,k->ab', T, v, v)      # m x m form in u
            w, V = np.linalg.eigh(Au); u = V[:, -1]
            Av = np.einsum('ajbk,a,b->jk', T, u, u)      # n x n form in v
            w, V = np.linalg.eigh(Av); v = V[:, -1]
            val = w[-1]
            if abs(val - val_old) < 1e-14 * max(1.0, abs(val)):
                break
            val_old = val
        if val > best:
            best, arg = val, (u.copy(), v.copy())
    return best, arg


def srank(M):
    return np.linalg.norm(M, 'fro') ** 2 / opnorm(M) ** 2


print("== P1.1 / P1.2: generic (non-Kronecker) SPD operators ==")
for (m, n, cond) in [(3, 3, 50.0), (3, 5, 200.0), (5, 3, 1e3), (4, 4, 1e4)]:
    r = min(m, n)
    Hm, ev, Qe = random_spd_operator(m, n, cond, rng)
    Hinv = np.linalg.inv(Hm)
    L, Qstar = L_op(Hm, m, n, rng)
    imu, (u, v) = inv_mu_op(Hinv, m, n, rng)
    mu = 1.0 / imu
    # the claimed minimizer of q / ||.||_op^2
    Dstar = apply(Hinv, np.outer(u, v))
    ratio_star = q(Hm, Dstar) / opnorm(Dstar) ** 2
    # random search: no direction should beat [mu, L]
    lo, hi = np.inf, -np.inf
    for _ in range(20000):
        D = rng.standard_normal((m, n))
        if rng.random() < 0.5:                     # low-rank and structured candidates too
            D = np.outer(rng.standard_normal(m), rng.standard_normal(n))
        val = q(Hm, D) / opnorm(D) ** 2
        lo, hi = min(lo, val), max(hi, val)
    # local refinement of the minimum by subgradient descent on the ratio from the best random points
    kF = ev.max() / ev.min()
    kop = L / mu
    # frame bounds
    best_frame = 0.0
    for _ in range(2000):
        U, _ = np.linalg.qr(rng.standard_normal((m, m)))
        V, _ = np.linalg.qr(rng.standard_normal((n, n)))
        h = np.array([q(Hm, np.outer(U[:, k], V[:, k])) for k in range(r)])
        best_frame = max(best_frame, h.sum() / h.min())
    Vmax = Qe[:, 0].reshape(m, n); Vmin = Qe[:, -1].reshape(m, n)
    sr_bound = kF * srank(Vmax) / srank(Vmin)
    print(f"m={m} n={n} kappa_F={kF:.3g}: L_op={L:.6f} mu_op={mu:.6g} (ratio at D*={ratio_star:.6g}) "
          f"random-search range [{lo:.4g}, {hi:.4g}] in [mu, L]: {lo >= mu * (1 - 1e-9) and hi <= L * (1 + 1e-9)}")
    print(f"      kappa_op={kop:.4g}  >= best frame bound {best_frame:.4g} >= r={r}: {kop >= best_frame * (1 - 1e-9) >= r * (1 - 1e-9)};"
          f"  >= kappa_F*srank(Vmax)/srank(Vmin)={sr_bound:.4g}: {kop >= sr_bound * (1 - 1e-9)};"
          f"  kappa_F/r <= kappa_op <= r kappa_F: {kF / r <= kop * (1 + 1e-9) and kop <= r * kF * (1 + 1e-9)}")

print("== P1.3a: Kronecker H[D] = B D A ==")
for (m, n) in [(3, 5), (5, 3), (4, 4)]:
    r = min(m, n)
    a = np.sort(rng.uniform(0.05, 1.0, n))[::-1]
    b = np.sort(rng.uniform(0.05, 1.0, m))[::-1]
    Qa, _ = np.linalg.qr(rng.standard_normal((n, n))); Qb, _ = np.linalg.qr(rng.standard_normal((m, m)))
    A = (Qa * a) @ Qa.T; B = (Qb * b) @ Qb.T
    Hm = np.kron(B, A)            # row-major vec: vec(B D A) = (B kron A^T) vec(D), A symmetric
    D = rng.standard_normal((m, n))
    assert np.allclose(apply(Hm, D), B @ D @ A)
    L, _ = L_op(Hm, m, n, rng)
    imu, _ = inv_mu_op(np.linalg.inv(Hm), m, n, rng)
    print(f"m={m} n={n}: L_op={L:.8f} vs sum a_k b_k={np.sum(a[:r] * b[:r]):.8f};  mu_op={1 / imu:.8f} vs a_min b_min={a[-1] * b[-1]:.8f};"
          f"  kappa_op/kappa_F={(L * imu) / (a[0] * b[0] / (a[-1] * b[-1])):.4f} in [1, {r}]")

print("== P1.3b: least squares, L_op(F) = tr(Sigma_hat) = mean ||x_i||^2 when K >= d; Ky Fan sum otherwise ==")
for (K, d, nsamp) in [(6, 4, 50), (3, 5, 50)]:
    X = rng.standard_normal((nsamp, d)) * np.geomspace(1, 0.1, d)
    S = X.T @ X / nsamp
    Hm = np.kron(np.eye(K), S)
    L, _ = L_op(Hm, K, d, rng, restarts=50)
    lam = np.sort(np.linalg.eigvalsh(S))[::-1]
    print(f"K={K} d={d}: L_op={L:.8f}; sum of top-min(K,d) eigenvalues={lam[:min(K, d)].sum():.8f}; "
          f"mean_i ||x_i||^2={np.mean(np.sum(X ** 2, 1)):.8f}; L_F=lambda_max={lam[0]:.8f}")
