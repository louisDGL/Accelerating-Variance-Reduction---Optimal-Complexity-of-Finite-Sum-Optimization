"""Audit num-noise, check 2: two statements verified by computation.

(A) (M1), lines 491-492: "With a constant spectral length eta, max(F(W_t), F(W_{t+1})) - F_* >= mu_op eta^2 / 8
    whenever the step is nonzero".
    Proof re-derived: ||W_{t+1} - W_t||_op = eta ||polar(omega)||_op = eta if omega != 0, hence
    max(||W_t - W_*||_op, ||W_{t+1} - W_*||_op) >= eta / 2 (triangle inequality), and mu_op-strong convexity for the
    spectral norm with grad F(W_*) = 0 gives F(W) - F_* >= (mu_op / 2) ||W - W_*||_op^2.
    Numerical test on quadratics F(W) = 1/2 tr((W - W_*) Sigma (W - W_*)^T), for which mu_op = lambda_min(Sigma)
    (rank-one Delta = u v_min^T attains it), with polar steps of (i) exact gradients, (ii) noisy gradients, (iii) an
    adversarial "estimator" (random matrices), plus the tight example (dimension one, 2-cycle W_* +- eta/2).

(B) lines 255-256: "gamma (m + theta c) with theta = 1/n, gamma = 1/lambda is MISO", and lines 499-501: "the
    predictable part m is the gradient of an isotropic quadratic ... the step drives m to zero, which MISO does exactly".
    Independent implementation on a small ridge softmax / least-squares finite sum:
      family   W+ = W - gamma (m + theta c), m = zbar + lam W, c = grad g_j(W) - z_j, then z_j <- grad g_j(W);
      MISO     surrogates d_i(W) = g_i(phi_i) + <grad g_i(phi_i), W - phi_i> + lam/2 ||W||_F^2 (lower bounds of f_i),
               phi_j <- W for the drawn j, W+ = argmin (1/n) sum_i d_i  (computed by solving the quadratic, no formula
               shared with the family).
    Same indices; the two sequences of iterates are compared, and m after each MISO step is printed.

Run from the project root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 paper/audit/checks/num-noise_bound_and_miso.py
"""
import numpy as np


def polar(M, tol=1e-13):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    keep = s > tol * max(s[0], 1e-300)
    return U[:, keep] @ Vt[keep]


# ---------------------------------------------------------------------------------------------------------------------
print("(A) bound max(F_t, F_{t+1}) - F_* >= mu_op eta^2 / 8")
rng = np.random.default_rng(0)
worst = np.inf
n_pairs = 0
for trial in range(60):
    K, d = rng.integers(1, 7), rng.integers(1, 9)
    lam = np.sort(rng.uniform(0.05, 3.0, d))
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    Sigma = (Q * lam) @ Q.T
    mu_op = lam[0]
    # brute-force confirmation that mu_op = lambda_min(Sigma): random directions never go below it
    for _ in range(200):
        D = rng.standard_normal((K, d))
        assert np.sum((D @ Sigma) * D) >= mu_op * np.linalg.norm(D, 2) ** 2 * (1 - 1e-12)
    eta = 10.0 ** rng.uniform(-3, 0)
    mode = trial % 3
    E = rng.standard_normal((K, d)) * 10.0 ** rng.uniform(-3, 1)       # E = W - W_*
    F = lambda E_: 0.5 * float(np.sum((E_ @ Sigma) * E_))               # noqa: E731
    prev = F(E)
    for t in range(400):
        G = E @ Sigma
        if mode == 0:
            om = G
        elif mode == 1:
            om = G + 0.3 * rng.standard_normal((K, d)) * (np.linalg.norm(G) + 1e-3)
        else:
            om = rng.standard_normal((K, d))                           # any nonzero matrix: the bound needs no estimator
        O = polar(om)
        if not O.size or np.linalg.norm(O, 2) < 0.5:
            continue
        E = E - eta * O
        cur = F(E)
        worst = min(worst, max(prev, cur) / (mu_op * eta ** 2 / 8))
        n_pairs += 1
        prev = cur
print(f"    {n_pairs} consecutive pairs on 60 random quadratics (exact, noisy and arbitrary directions): "
      f"min of max(F_t, F_t+1) / (mu_op eta^2 / 8) = {worst:.4f}  (>= 1 required)")
# adversarial search: start at spectral distance eta/2 along the flattest direction, step straight through W_*
mu, eta = 0.37, 0.2
e0 = eta / 2
f = lambda e: 0.5 * mu * e * e                                          # noqa: E731
print(f"    tight case (K = d = 1, 2-cycle +-eta/2): max(F_t, F_t+1) / (mu eta^2 / 8) = "
      f"{max(f(e0), f(e0 - eta)) / (mu * eta ** 2 / 8):.6f}")
# matrix version of the tight case: E = (eta/2) u v_min^T, omega = gradient (rank one): next E = -(eta/2) u v_min^T
K, d = 3, 4
lam = np.array([0.2, 0.5, 1.0, 2.0]); Q, _ = np.linalg.qr(rng.standard_normal((d, d))); Sigma = (Q * lam) @ Q.T
u = rng.standard_normal(K); u /= np.linalg.norm(u)
E = (eta / 2) * np.outer(u, Q[:, 0])
E1 = E - eta * polar(E @ Sigma)
Fm = lambda E_: 0.5 * float(np.sum((E_ @ Sigma) * E_))                  # noqa: E731
print(f"    tight case (3 x 4, rank-one error along the flattest direction): ratio = "
      f"{max(Fm(E), Fm(E1)) / (lam[0] * eta ** 2 / 8):.6f}")

# ---------------------------------------------------------------------------------------------------------------------
print("\n(B) gamma (m + theta c) with theta = 1/n, gamma = 1/lam against MISO (lower-bound surrogates)")


def make(loss, n, d, K, lam, seed):
    r = np.random.default_rng(seed)
    X = r.standard_normal((n, d)) / np.sqrt(d)
    if loss == "ls":
        Y = r.standard_normal((n, K))
        grad = lambda W, i: np.outer(W @ X[i] - Y[i], X[i])            # noqa: E731
    else:
        lab = r.integers(0, K, n)

        def grad(W, i):
            z = W @ X[i]
            p = np.exp(z - z.max()); p /= p.sum()
            p[lab[i]] -= 1.0
            return np.outer(p, X[i])
    return X, grad


for loss in ("ls", "softmax"):
    n, d, K, lam = 40, 6, 3, 0.05
    X, grad = make(loss, n, d, K, lam, 1)
    r = np.random.default_rng(2)
    W0 = r.standard_normal((K, d))
    idx = r.integers(0, n, 400)
    # family member
    Wf = W0.copy()
    Z = np.stack([grad(W0, i) for i in range(n)]); zbar = Z.mean(0)
    # MISO: explicit surrogates, each stored by its linear coefficient A_i = grad g_i(phi_i) (the quadratic part
    # lam/2 ||W||^2 is common); the minimizer of the average surrogate is found by solving lam W + mean(A) = 0
    Wm = W0.copy()
    A = np.stack([grad(W0, i) for i in range(n)])
    gap, mnorm_miso, mnorm_before = 0.0, 0.0, []
    for t, j in enumerate(idx):
        # family step
        m = zbar + lam * Wf
        g = grad(Wf, j)
        c = g - Z[j]
        Wf = Wf - (1.0 / lam) * (m + (1.0 / n) * c)
        zbar = zbar + c / n
        Z[j] = g
        # MISO step
        A[j] = grad(Wm, j)                                             # phi_j <- current iterate
        Wm = np.linalg.solve(lam * np.eye(K * d), -A.mean(0).reshape(-1)).reshape(K, d)
        if t >= 1:                                                     # the first step absorbs the arbitrary W_0
            gap = max(gap, float(np.linalg.norm(Wf - Wm) / max(np.linalg.norm(Wm), 1e-300)))
        mnorm_miso = max(mnorm_miso, float(np.linalg.norm(zbar + lam * Wf)))
    # is m the gradient of an isotropic quadratic for a frozen table? Hessian of W -> zbar + lam W is lam I
    P = r.standard_normal((K, d)); h = 1e-3
    hess_action = ((zbar + lam * (Wf + h * P)) - (zbar + lam * (Wf - h * P))) / (2 * h)
    print(f"    {loss:8s}: max relative gap between the family member and MISO over 399 steps = {gap:.2e}; "
          f"max ||m||_F after a step = {mnorm_miso:.2e}; ||Hess(m)[P] - lam P|| = "
          f"{np.linalg.norm(hess_action - lam * P):.1e}; step 0 gap = "
          f"{np.linalg.norm((W0 - (1 / lam) * ((np.stack([grad(W0, i) for i in range(n)]).mean(0) + lam * W0))) - (-np.stack([grad(W0, i) for i in range(n)]).mean(0) / lam)):.1e}")
