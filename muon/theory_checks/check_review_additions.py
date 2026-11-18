"""Checks for the statements added or corrected after the review of 16 November 2026.

  (A) Proposition 2.5(c): no norm improves the worst-case condition number of a ridge-regularized softmax model.
      Ingredients checked on an instance: the curvature of the losses vanishes along t W_1, so that
      mu_N(F) = lambda * m_N with m_N = min ||D||_F^2 / N(D)^2; and L_N(f_i) / mu_N(F) >= L_F(f_i) / mu_F(F) for a
      two-sided preconditioned spectral norm N(D) = ||S^1/2 D P^1/2||_op (L_N lower-bounded by an ascent).
  (B) Proposition 4.2(d): for K = d = 2 every polar step keeps s_1(E) - s_2(E) as long as det(E) keeps its sign.
  (C) Proposition 3.4: ||E_0||_op <= T <= sqrt(2 D_0 / mu_op) <= sqrt(kappa_op) ||E_0||_op for the polar flow of the
      least-squares model; t -> ||G(t)||_* is convex and J(t) is non-increasing.
  (D) Theorem 5.2: stochastic polar steps with additive noise are stochastic mirror descent on the smoothed nuclear
      norm; bound  E psi(G_bar_T) - min psi <= (D_F(W_0, W_dagger) + (L_op/2) sum eta^2) / sum eta.
  (E) Theorem 5.3: a zero of the mean polar field is a nuclear median, the converse can fail at rank-deficient points
      (two-point example); every nuclear median C of a centered law has ||C||_* <= E||xi||_*.
  (F) Proposition 5.5(a): the 1/b formula for the median of a mean needs a non-lattice law (two-point example).
  (G) Example 2.4: closed-form operator on r x r matrices with kappa_op / kappa_F close to 1/r
      (L_op = 1 + m (r - 1), mu_op = r m b / (m + (r - 1) b), kappa_F = 1/b).
Run: .venv/Scripts/python muon/theory_checks/check_review_additions.py
"""
import numpy as np
from math import comb, sqrt
from scipy.optimize import root

rng = np.random.default_rng(20261116)


def polar(M, tol=1e-12):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    keep = s > tol * max(s[0], 1e-300)
    return U[:, keep] @ Vt[keep]


def polar_stack(M, tol=1e-12):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    keep = (s > tol * np.maximum(s[..., :1], 1e-300)).astype(float)
    return (U * keep[..., None, :]) @ Vt


def nuc(M):
    return float(np.linalg.svd(M, compute_uv=False).sum())


def spd(n, lo, hi, rng):
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return (Q * np.geomspace(hi, lo, n)) @ Q.T


def sqrtm_spd(S):
    w, V = np.linalg.eigh(S)
    return (V * np.sqrt(w)) @ V.T


print("== (A) ridge-regularized softmax: no free lunch for a two-sided preconditioned spectral norm ==")
n, K, d, lam = 12, 3, 4, 1e-2
X = rng.standard_normal((n, d))
W1 = rng.standard_normal((K, d))


def hess_loss(z):                                   # Hessian of the softmax loss at the logits z
    p = np.exp(z - z.max()); p /= p.sum()
    return np.diag(p) - np.outer(p, p)


for t in [1.0, 10.0, 100.0]:
    print(f"   max_i ||Hess l_i(t W_1 x_i)||_op at t = {t:5.0f}: {max(np.linalg.norm(hess_loss(t * W1 @ x), 2) for x in X):.2e}")
S_, P_ = spd(K, 0.2, 3.0, rng), spd(d, 0.1, 5.0, rng)
Sh, Ph = sqrtm_spd(S_), sqrtm_spd(P_)
Shi, Phi_ = np.linalg.inv(Sh), np.linalg.inv(Ph)
m_N = 1.0 / (np.linalg.eigvalsh(S_).max() * np.linalg.eigvalsh(P_).max())      # min ||D||_F^2 / ||S^1/2 D P^1/2||_op^2
A_top = np.diag([0.5, 0.5, 0.0]) - np.outer([0.5, 0.5, 0.0], [0.5, 0.5, 0.0])   # largest softmax curvature, c = 1/2
worst = np.inf
for i in range(n):
    x = X[i]
    best = 0.0
    for _ in range(20):                              # ascent on D' -> q(S^-1/2 D' P^-1/2) over the spectral unit ball
        Dp = polar(rng.standard_normal((K, d)))
        for _ in range(200):
            D = Shi @ Dp @ Phi_
            grad = Shi @ (2 * A_top @ np.outer(D @ x, x) + 2 * lam * D) @ Phi_
            Dp = polar(grad)
        D = Shi @ Dp @ Phi_
        best = max(best, float((D @ x) @ A_top @ (D @ x) + lam * np.sum(D * D)))
    ratio_N = best / (lam * m_N)                     # lower bound of L_N(f_i) / mu_N(F)
    ratio_F = (0.5 * x @ x + lam) / lam              # L_F(f_i) / mu_F(F)
    worst = min(worst, ratio_N / ratio_F)
print(f"   min over the samples of [L_N(f_i)/mu_N(F)] / [L_F(f_i)/mu_F(F)] (L_N lower-bounded by ascent): {worst:.3f} (>= 1)")

print("== (B) K = d = 2: s_1 - s_2 of the error is conserved by polar steps while det(E) keeps its sign ==")
dev_same, n_same, n_flip = 0.0, 0, 0
for _ in range(20000):
    E = rng.standard_normal((2, 2)) * rng.choice([0.3, 1.0, 3.0])
    Sig = spd(2, 10.0 ** -rng.uniform(0, 3), 1.0, rng)
    eta = rng.uniform(0, 1.5) * np.linalg.norm(E, 2)
    E1 = E - eta * polar(E @ Sig)
    s, s1 = np.linalg.svd(E, compute_uv=False), np.linalg.svd(E1, compute_uv=False)
    if np.linalg.det(E) * np.linalg.det(E1) > 0:
        n_same += 1
        dev_same = max(dev_same, abs((s[0] - s[1]) - (s1[0] - s1[1])))
    else:
        n_flip += 1
print(f"   {n_same} steps keeping the sign of det(E): max |change of s_1 - s_2| = {dev_same:.1e}; {n_flip} steps changing it (not concerned)")

print("== (C) extinction time of the polar flow, least squares: ||E_0||_op <= T <= sqrt(2 D_0/mu_op) <= sqrt(kappa_op) ||E_0||_op ==")
for (K, d, kap) in [(6, 6, 1.0), (6, 6, 36.0), (6, 6, 400.0), (10, 4, 36.0), (10, 4, 400.0)]:
    lam_ = np.geomspace(1.0, 1.0 / kap, d)
    V, _ = np.linalg.qr(rng.standard_normal((d, d)))
    Sig = (V * lam_) @ V.T
    E = rng.standard_normal((K, d)); E /= np.linalg.norm(E, 2)
    D0 = 0.5 * np.sum((E @ Sig) * E)
    eta, t = 2e-4, 0.0
    g_nuc, Js = [], []
    while np.linalg.norm(E, 2) > 3 * eta and t < 50:
        G = E @ Sig
        O = polar(G, tol=1e-9)
        if int(round(t / eta)) % 50 == 0:
            g_nuc.append(nuc(G)); Js.append(float(np.sum((O @ Sig) * O)))
        E = E - eta * O; t += eta
    g_nuc = np.array(g_nuc)
    second = g_nuc[2:] - 2 * g_nuc[1:-1] + g_nuc[:-2]                 # discrete convexity of t -> ||G(t)||_*
    print(f"   K = {K}, d = {d}, cond(Sigma) = {kap:5.0f}: T = {t:.3f}; sqrt(2 D_0/mu_op) = {sqrt(2 * D0 / lam_.min()):.3f}; "
          f"sqrt(kappa_op) = {sqrt(lam_.sum() / lam_.min()):.2f}; min second difference of ||G(t)||_* = {second.min():+.1e} "
          f"(scale {g_nuc[0] * (50 * eta) ** 2:.1e}); largest increase of J between samples = {max(np.diff(Js).max(), 0):.1e} (J_0 = {Js[0]:.3f})")

print("== (D) stochastic polar steps with additive noise = stochastic mirror descent on the smoothed nuclear norm ==")
K, d = 2, 3
lam_ = np.array([1.0, 0.4, 0.1])
V, _ = np.linalg.qr(rng.standard_normal((d, d)))
Sig = (V * lam_) @ V.T
Sig_inv = np.linalg.inv(Sig)
pool = 0.3 * (rng.standard_exponential((200000, K, d)) - 1.0)        # the law of the noise: uniform on this pool
pool -= pool.mean(0)


def psi(G):
    return float(np.linalg.svd(G[None] + pool, compute_uv=False).sum(-1).mean())


def field(g):
    return polar_stack(g.reshape(K, d)[None] + pool).mean(0).reshape(-1)


sol = root(field, np.zeros(K * d), method='hybr', tol=1e-13)
G_dag = sol.x.reshape(K, d)
psi_min = psi(G_dag)
L_op = np.sort(lam_)[::-1][:K].sum()
E0 = rng.standard_normal((K, d))
E_dag = G_dag @ Sig_inv
D0 = 0.5 * np.sum(((E0 - E_dag) @ Sig) * (E0 - E_dag))               # D_F(W_0, W_dagger) for the quadratic
print(f"   nuclear median of the noise: ||G_dagger||_* = {nuc(G_dag):.4f} (mean of the noise = 0); min psi = {psi_min:.5f}, psi(0) = {psi(np.zeros((K, d))):.5f}")
for T in [500, 2000, 8000]:
    eta = sqrt(2 * D0 / (L_op * T))
    gaps = []
    for rep in range(10):
        E = E0.copy(); Gsum = np.zeros((K, d))
        idx = rng.integers(0, pool.shape[0], T)
        for t_ in range(T):
            G = E @ Sig
            Gsum += G
            E = E - eta * polar(G + pool[idx[t_]])
        gaps.append(psi(Gsum / T) - psi_min)
    bound = (D0 + 0.5 * L_op * eta ** 2 * T) / (eta * T)
    print(f"   T = {T:5d}, eta = {eta:.4f}: mean over 10 runs of psi(mean gradient) - min psi = {np.mean(gaps):.5f} (max {np.max(gaps):.5f}); bound {bound:.5f}")

print("== (E) nuclear medians: rank-deficient two-point example, and ||C||_* <= E||xi||_* ==")
Dm = np.diag([1.0, -1.0]) / 2
atoms = np.array([Dm, -Dm])
psi2 = lambda G: float(np.mean([nuc(G + a) for a in atoms]))
G_half = np.eye(2) / 2
lowest = min(psi2(rng.standard_normal((2, 2)) * s_) for s_ in [0.01, 0.1, 0.5, 1.0] for _ in range(5000))
print(f"   xi = +-diag(1,-1)/2: psi(I/2) = {psi2(G_half):.6f}, psi(0) = {psi2(np.zeros((2, 2))):.6f}, lowest value on 20000 random points = {lowest:.6f} "
      f"(lower bound 1); mean polar factor at I/2 = {np.array2string(np.mean([polar(G_half + a) for a in atoms], axis=0), precision=3)}")
worst = 0.0
for _ in range(20):
    S = 4000
    xi = rng.standard_exponential((S, 2, 3)) * rng.uniform(0.2, 2.0, (1, 2, 3)) - 1.0
    xi -= xi.mean(0)
    f = lambda g: polar_stack(g.reshape(2, 3)[None] + xi).mean(0).reshape(-1)
    sol = root(f, np.zeros(6), method='hybr', tol=1e-13)
    if np.linalg.norm(f(sol.x)) < 1e-9:
        C = -sol.x.reshape(2, 3)
        worst = max(worst, nuc(C) / np.linalg.svd(xi, compute_uv=False).sum(-1).mean())
print(f"   20 skewed empirical laws: max ||median||_* / E||xi||_* = {worst:.3f} (<= 1)")

print("== (F) median of the mean of b copies of X in {-1, 2} with P(X = 2) = 1/3 (kappa_3/(6 sigma^2) = 1/6): lattice law ==")
vals = []
for b in range(1, 13):
    cdf, med = 0.0, None
    for k in range(b + 1):                           # k = number of values 2; the mean is (3k - b)/b
        cdf += comb(b, k) * (1 / 3) ** k * (2 / 3) ** (b - k)
        if cdf >= 0.5:
            med = (3 * k - b) / b
            break
    vals.append(b * med)
print("   b x median for b = 1..12:", [int(round(v)) for v in vals], "(the non-lattice formula would give -1/6)")

print("== (G) Example 2.4: operator q(D) = m ||D||_F^2 + (1 - m) D_12^2 - (m - b) (tr D)^2 / r ==")
from scipy.optimize import minimize


def example_check(r, m, b, rng):
    def q(D):
        return m * np.sum(D * D) + (1 - m) * D[0, 1] ** 2 - (m - b) * np.trace(D) ** 2 / r

    def Hop(D):
        out = m * D - (m - b) * np.trace(D) / r * np.eye(r)
        out[0, 1] += (1 - m) * D[0, 1]
        return out

    L_best = 0.0
    for _ in range(200):                                 # ascent on the isometries: Q <- polar(H Q)
        Q = polar(rng.standard_normal((r, r)))
        for _ in range(300):
            Q = polar(Hop(Q))
        L_best = max(L_best, q(Q))
    cyc = np.roll(np.eye(r), 1, axis=1)                  # cyclic permutation with entry (1,2) equal to 1
    L_formula = 1 + m * (r - 1)
    mu_formula = r * m * b / (m + (r - 1) * b)
    c = (m - b) / (m + (r - 1) * b)
    D_star = np.diag([1.0] + [c] * (r - 1))
    mu_best = np.inf
    for _ in range(60):                                  # direct minimization of q(D) / ||D||_op^2
        res = minimize(lambda x: q(x.reshape(r, r)) / np.linalg.norm(x.reshape(r, r), 2) ** 2,
                       rng.standard_normal(r * r), method='Nelder-Mead',
                       options={'maxiter': 20000, 'xatol': 1e-10, 'fatol': 1e-14})
        mu_best = min(mu_best, res.fun)
    return L_best, q(cyc), L_formula, mu_best, q(D_star), mu_formula


for (r, m, b) in [(2, 0.3, 0.05), (3, 0.2, 0.02)]:
    Lb, Lc, Lf, mb, md, mf = example_check(r, m, b, rng)
    print(f"   r = {r}, m = {m}, b = {b}: L_op: ascent {Lb:.6f}, cyclic permutation {Lc:.6f}, formula {Lf:.6f}; "
          f"mu_op: direct minimization {mb:.6f}, at diag(1,c,..,c) {md:.6f}, formula {mf:.6f}; "
          f"kappa_op/kappa_F = {Lf / mf * b:.4f} (1/r = {1 / r:.4f})")
for (r, m, b) in [(10, 1e-2, 1e-5), (20, 1e-2, 1e-6)]:
    print(f"   r = {r}, m = {m:g}, b = {b:g}: kappa_F / kappa_op = {1.0 / ((1 + m * (r - 1)) * (m + (r - 1) * b) / (r * m)):.2f}")
