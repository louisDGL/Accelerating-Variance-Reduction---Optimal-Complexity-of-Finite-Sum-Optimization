"""Numerical checks for Part 4 of notes_spectral_theory: stochastic polar steps, noise smoothing and the nuclear median.

  T4.1  E[polar(G + xi)] is a (sub)gradient of psi_xi(G) = E||G + xi||_*  (exact for an empirical measure).
  T4.2  mean-field fixed points: 0 in E[polar(stochastic gradient)], i.e. grad F(W) = - nuclear median of the noise.
        (d) batch one, least squares, rank-one per-sample gradients: the fixed point is the minimizer of the weighted
            least-absolute-deviation objective (1/n) sum_i ||W x_i - y_i|| / ||x_i||, not the least-squares solution;
        (e) minibatch b, skewed noise: the offset ||grad F(W_b)|| decays like 1/b; scalar case: median of a Gamma mean;
        (f) momentum: offset of the median of an exponential moving average of skewed noise, predicted ratio
            (1 - beta^2)/(1 + beta + beta^2) (third and second cumulants of the average).
  T4.3  low signal-to-noise, isotropic Gaussian noise sigma Z: E polar(G + sigma Z) = (c_{K,d}/sigma) G + o(|G|),
        c_{K,d} = E[sum_{i<j} 2/(s_i+s_j) + (d-K) sum_i 1/s_i] / (K d), s = singular values of a K x d standard Gaussian.
Run: .venv/Scripts/python muon/theory_checks/check_smoothing_median.py
"""
import numpy as np
from scipy import stats
from scipy.optimize import root
from scipy.signal import lfilter

rng = np.random.default_rng(20261116)


def polar_stack(M, tol=1e-12):
    """polar factor of each matrix of a stack (S, K, d), reduced on the nonzero singular values."""
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    keep = (s > tol * np.maximum(s[..., :1], 1e-300)).astype(float)
    return (U * keep[..., None, :]) @ Vt


def nuc_stack(M):
    return np.linalg.svd(M, compute_uv=False).sum(-1)


print("== T4.1 convexity inequality <G - G', E polar(G + xi) - E polar(G' + xi)> >= 0 and subgradient inequality (empirical measure) ==")
K, d, S = 3, 4, 4000
xi = rng.standard_exponential((S, K, d)) - 1.0                  # skewed, centered noise
ok = True
for _ in range(200):
    G, G2 = rng.standard_normal((K, d)) * 0.7, rng.standard_normal((K, d)) * 0.7
    P, P2 = polar_stack(G + xi).mean(0), polar_stack(G2 + xi).mean(0)
    psi, psi2 = nuc_stack(G + xi).mean(), nuc_stack(G2 + xi).mean()
    ok &= np.sum((G - G2) * (P - P2)) >= -1e-10
    ok &= psi2 >= psi + np.sum(P * (G2 - G)) - 1e-10
print("  monotonicity and subgradient inequality hold on 200 random pairs:", ok)

print("== T4.2(d) batch one, least squares: the mean-field fixed point is the weighted LAD solution ==")
n, K, d = 400, 2, 3
X = rng.standard_normal((n, d)) * np.array([1.0, 0.6, 0.3])
W0 = rng.standard_normal((K, d))
noise = rng.standard_exponential((n, K)) - 1.0                  # skewed residuals
Y = X @ W0.T + 0.5 * noise
W_ls = np.linalg.solve(X.T @ X, X.T @ Y).T
xn = np.linalg.norm(X, axis=1)
# weighted LAD by iteratively reweighted least squares (Weiszfeld)
W = W_ls.copy()
for _ in range(5000):
    R = X @ W.T - Y
    w = 1.0 / (xn * np.maximum(np.linalg.norm(R, axis=1), 1e-14))
    W_new = np.linalg.solve((X * w[:, None]).T @ X, (X * w[:, None]).T @ Y).T
    if np.linalg.norm(W_new - W) < 1e-15:
        W = W_new
        break
    W = W_new
W_lad = W


def mean_polar_b1(W):
    R = X @ W.T - Y                                              # per-sample gradient R_i x_i^T, polar = R_i x_i^T/(|R_i||x_i|)
    return ((R / np.linalg.norm(R, axis=1, keepdims=True)).T @ (X / xn[:, None])) / n


print(f"  ||mean_i polar(g_i)||_F at the LAD solution: {np.linalg.norm(mean_polar_b1(W_lad)):.2e}; at the least-squares solution: {np.linalg.norm(mean_polar_b1(W_ls)):.2e}; "
      f"||W_lad - W_ls||_F = {np.linalg.norm(W_lad - W_ls):.4f}")
# stochastic run: batch one, fixed-length polar step (= normalized SGD), decaying length, iterate averaging
W = np.zeros((K, d)); avg = np.zeros((K, d)); cnt = 0
T = 400000
idx = rng.integers(0, n, T)
for t in range(T):
    i = idx[t]
    r = W @ X[i] - Y[i]
    W = W - (0.5 / np.sqrt(t + 100.0)) * np.outer(r / np.linalg.norm(r), X[i] / xn[i])
    if t >= T // 2:
        avg += W; cnt += 1
avg /= cnt
print(f"  batch-one polar steps with decaying length (averaged second half): distance to LAD solution {np.linalg.norm(avg - W_lad):.4f}, "
      f"to least-squares solution {np.linalg.norm(avg - W_ls):.4f}")

print("== T4.2(e) minibatch: offset of the mean-field fixed point versus batch size (skewed and symmetric residuals) ==")


def fixed_point(Xp, Yp, b, S, rng, W_init):
    B = rng.integers(0, Xp.shape[0], (S, b))                     # S fixed batches, with replacement
    XB, YB = Xp[B], Yp[B]                                        # (S, b, d), (S, b, K)
    A = np.einsum('sbi,sbj->sij', XB, XB) / b                    # Sigma_B
    C = np.einsum('sbk,sbj->skj', YB, XB) / b

    def field(w):
        Wm = w.reshape(W_init.shape)
        Gs = Wm[None] @ A - C
        return polar_stack(Gs).mean(0).reshape(-1)

    sol = root(field, W_init.reshape(-1), method='hybr', tol=1e-12)
    return sol.x.reshape(W_init.shape), np.linalg.norm(field(sol.x))


n = 20000
X = rng.standard_normal((n, d)) * np.array([1.0, 0.6, 0.3])
for name, noise in [("skewed (exponential)", rng.standard_exponential((n, K)) - 1.0), ("symmetric (Gaussian)", rng.standard_normal((n, K)))]:
    Y = X @ W0.T + 0.5 * noise
    W_ls = np.linalg.solve(X.T @ X, X.T @ Y).T
    Sig = X.T @ X / n
    rows = []
    for b in [4, 8, 16, 32, 64]:
        Wb, res = fixed_point(X, Y, b, 40000, rng, W_ls)
        gnorm = np.linalg.svd((Wb - W_ls) @ Sig, compute_uv=False).sum()
        rows.append((b, gnorm, res))
    print(f"  {name}: (b, ||grad F(W_b)||_*, residual of the fixed-point equation) = " + "; ".join(f"({b}, {g:.4f}, {r:.0e})" for b, g, r in rows))
    print("      b * ||grad F(W_b)||_* = " + ", ".join(f"{b * g:.3f}" for b, g, r in rows))

print("== scalar case: sign descent with batch b on f_i(w) = (w - a_i)^2/2, a ~ Exp(1) - 1: fixed point = median of the batch mean ==")
for b in [1, 2, 4, 8, 16, 32]:
    med = stats.gamma.median(b) / b - 1.0
    print(f"  b = {b:2d}: median of the batch mean = {med:+.5f}; Edgeworth prediction -kappa_3/(6 sigma^2 b) = {-1 / (3 * b):+.5f}; "
          f"objective gap of the fixed point = {0.5 * med ** 2:.2e}")

print("== T4.2(f) momentum: median of an exponential moving average of skewed noise ==")
N = 4_000_000
a = rng.standard_exponential(N) - 1.0
for beta in [0.0, 0.5, 0.9, 0.95, 0.99]:
    m = lfilter([1 - beta], [1, -beta], a)[int(20 / (1 - beta)) + 10:]
    med = np.median(m)
    pred = -(1 / 3) * (1 - beta ** 2) / (1 + beta + beta ** 2)
    beff = -(1 / 3) * (1 - beta) / (1 + beta)
    print(f"  beta = {beta:4.2f}: median = {med:+.5f}; cumulant prediction -(1/3)(1-b^2)/(1+b+b^2) = {pred:+.5f}; effective-batch heuristic -(1/3)(1-b)/(1+b) = {beff:+.5f}")

print("== T4.3 low signal-to-noise limit with isotropic Gaussian noise ==")
for (K, d) in [(3, 5), (4, 4), (2, 8)]:
    S = 400000
    Z = rng.standard_normal((S, K, d))
    s = np.linalg.svd(Z, compute_uv=False)                       # (S, K), descending
    acc = np.zeros(S)
    for i in range(K):
        for j in range(i + 1, K):
            acc += 2.0 / (s[:, i] + s[:, j])
    acc += (d - K) * np.sum(1.0 / s, axis=1)
    c = acc.mean() / (K * d)
    c_se = acc.std() / np.sqrt(S) / (K * d)
    G = rng.standard_normal((K, d)); G *= 0.05 / np.linalg.norm(G)
    sigma = 1.0
    P = 0.5 * (polar_stack(G + sigma * Z) + polar_stack(G - sigma * Z))   # antithetic pairs remove the odd part
    Pm = P.mean(0)
    cos = np.sum(Pm * G) / (np.linalg.norm(Pm) * np.linalg.norm(G))
    ratio = np.sum(Pm * G) / np.sum(G * G)
    print(f"  K = {K}, d = {d}: c_(K,d) = {c:.4f} (+- {c_se:.4f}; note E 1/s_min may be heavy-tailed when K = d); measured <E polar(G + Z), G>/|G|^2 = {ratio:.4f}; "
          f"cosine between E polar(G + Z) and G = {cos:.4f}")

print("== T4.3 remark: Kronecker noise on the thin side (tall case K >> d): E polar(G + xi) ~ G (K C_R)^(-1/2) ==")
K, d, S = 60, 3, 40000
cr = np.array([1.0, 0.25, 0.04])
Qr, _ = np.linalg.qr(rng.standard_normal((d, d)))
CRh = (Qr * np.sqrt(cr)) @ Qr.T
Z = rng.standard_normal((S, K, d)) @ CRh
G = rng.standard_normal((K, d)); G *= 0.2 / np.linalg.norm(G)
P = 0.5 * (polar_stack(G + Z) + polar_stack(G - Z)).mean(0)
pred = G @ ((Qr * (1 / np.sqrt(K * cr))) @ Qr.T)
print(f"  K = {K}, d = {d}: relative error ||E polar - G (K C_R)^(-1/2)||_F / ||.||_F = {np.linalg.norm(P - pred) / np.linalg.norm(pred):.3f} "
      f"(expected of order (d+1)/(2K) = {(d + 1) / (2 * K):.3f} plus Monte-Carlo error)")
