"""Audit math-smd -- Proposition 3 of paper_saga_to_muon.tex (lines 422-447).

 (d1) batch-one polar steps on a small unregularized least-squares problem (n = 60, K = 2, d = 3, intercept, skewed
      residuals), eta_t = 0.3 (1 + t/200)^(-3/4): does the limit minimize Lambda, and does it differ from least squares?
      Control: lengths proportional to the sampled gradient (plain SGD) go to the least-squares solution.
 (d2) d = 1: Lambda(w) = (1/n) sum ||w - y_i/x_i||, minimizer = geometric median of the y_i/x_i (Weiszfeld).
 (d3) existence / non-uniqueness of the minimizer of Lambda (K = d = 1, n even; rank-deficient design).
 (a)  symmetric laws: psi(G) - psi(0) >= 0 with the pointwise bound of the proof; +-I_2 has other medians.
 (b)  centred finite laws: ||C||_* <= E||xi - C||_* <= E||xi||_* for a numerical nuclear median C.
 (c)  SAGA / SVRG fixed points: generic uniqueness, and the identical-Hessian example where, for SVRG, every reference
      point gives omega_j = grad F(W) for all j (index-stationary states (W_*, any reference) unless the refresh of the
      reference is counted as part of the dynamics).

Run from the project root (block: d1 | rest):
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -u paper/audit/checks/math-smd_prop3.py d1
"""
import sys
import numpy as np
from scipy.optimize import minimize

rng = np.random.default_rng(31)


def nuc(M):
    return np.linalg.svd(M, compute_uv=False).sum()


def polar(M, tol=1e-12):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    keep = s > tol * max(1.0, s[0])
    return U[:, keep] @ Vt[keep, :]


def Lam(W, X, Y):
    return np.mean(np.linalg.norm(W @ X.T - Y.T, axis=0) / np.linalg.norm(X, axis=1))


def gradLam(W, X, Y):
    R = W @ X.T - Y.T                                   # K x n
    wgt = 1.0 / (np.linalg.norm(R, axis=0) * np.linalg.norm(X, axis=1))
    return (R * wgt) @ X / X.shape[0]


def argmin_Lam(X, Y, K):
    d = X.shape[1]
    r = minimize(lambda w: (Lam(w.reshape(K, d), X, Y), gradLam(w.reshape(K, d), X, Y).ravel()),
                 np.zeros(K * d), jac=True, method="BFGS", options={"gtol": 1e-12, "maxiter": 5000})
    return r.x.reshape(K, d)


def block_d1():
    print("=== (d1) batch one on least squares, n = 60, K = 2, d = 3 ===")
    n, K, d = 60, 2, 3
    X = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
    W0 = rng.standard_normal((K, d))
    Y = X @ W0.T + 0.5 * (rng.exponential(1.0, (n, K)) - 1.0)
    # lemma: polar(r x^T) = r x^T / (|r||x|) = gradient of ||Wx - y|| / ||x||
    W = rng.standard_normal((K, d))
    j = 5
    r = W @ X[j] - Y[j]
    lem = np.abs(polar(np.outer(r, X[j])) - np.outer(r, X[j]) / (np.linalg.norm(r) * np.linalg.norm(X[j]))).max()
    eps = 1e-6
    num = np.zeros((K, d))
    for a in range(K):
        for b in range(d):
            E = np.zeros((K, d)); E[a, b] = eps
            num[a, b] = (np.linalg.norm((W + E) @ X[j] - Y[j]) - np.linalg.norm((W - E) @ X[j] - Y[j])) / (2 * eps)
    lem2 = np.abs(num / np.linalg.norm(X[j]) - polar(np.outer(r, X[j]))).max()
    print(f"  lemma: |polar(r x^T) - r x^T/(|r||x|)| = {lem:.1e}; |finite-difference gradient of ||Wx-y||/||x|| - polar| = {lem2:.1e}")
    W_ls = np.linalg.solve(X.T @ X, X.T @ Y).T
    W_lad = argmin_Lam(X, Y, K)
    mf = sum(polar(np.outer(W_lad @ X[i] - Y[i], X[i])) for i in range(n)) / n
    print(f"  ||grad Lambda(W_LAD)||_F = {np.linalg.norm(gradLam(W_lad, X, Y)):.1e}; mean field E_j polar(g_j(W_LAD)): "
          f"||.||_F = {np.linalg.norm(mf):.1e}; at W_LS: {np.linalg.norm(gradLam(W_ls, X, Y)):.3f}")
    print(f"  Lambda(W_LAD) = {Lam(W_lad, X, Y):.6f} < Lambda(W_LS) = {Lam(W_ls, X, Y):.6f};  "
          f"||W_LAD - W_LS||_F = {np.linalg.norm(W_lad - W_ls):.4f}")
    xn = np.linalg.norm(X, axis=1)
    N, T = 1500, 40000
    eta = 0.3 * (1.0 + np.arange(T) / 200.0) ** (-0.75)
    for name in ("polar step, deterministic lengths", "length = gamma_t ||g_j||_* (plain SGD)"):
        Wc = np.zeros((N, K, d))
        acc = np.zeros((N, K, d))
        for t in range(T):
            jj = rng.integers(0, n, N)
            x = X[jj]
            R = np.einsum("nkd,nd->nk", Wc, x) - Y[jj]
            if name.startswith("polar"):
                rn = np.linalg.norm(R, axis=1)
                step = (R / (rn * xn[jj])[:, None])[:, :, None] * x[:, None, :]
            else:
                step = 0.3 * R[:, :, None] * x[:, None, :]          # gamma_t = 0.3 eta_t, below 2 / max ||x||^2
            Wc = Wc - eta[t] * step
            if t >= T // 2:
                acc += Wc
        Wbar_chain = acc / (T - T // 2)
        Wbar = Wbar_chain.mean(axis=0)
        dl = np.linalg.norm(Wc - W_lad, axis=(1, 2))
        ds = np.linalg.norm(Wc - W_ls, axis=(1, 2))
        print(f"  [{name}]")
        print(f"     tail average over chains: dist to W_LAD = {np.linalg.norm(Wbar - W_lad):.4f}, dist to W_LS = "
              f"{np.linalg.norm(Wbar - W_ls):.4f};  Lambda(mean) - min Lambda = {Lam(Wbar, X, Y) - Lam(W_lad, X, Y):.2e}")
        print(f"     last iterates: mean dist to W_LAD = {dl.mean():.4f}, to W_LS = {ds.mean():.4f}")


def weiszfeld(P, iters=5000):
    m = P.mean(axis=0)
    for _ in range(iters):
        w = 1.0 / np.linalg.norm(P - m, axis=1)
        m = (P * w[:, None]).sum(axis=0) / w.sum()
    return m


def block_rest():
    print("=== (d2) d = 1: geometric median of the y_i / x_i ===")
    n, K = 25, 2
    x = rng.standard_normal(n) * 2.0
    Yv = rng.standard_normal((n, K)) + np.outer(x, [1.0, -0.5]) + (rng.exponential(1.0, (n, K)) - 1.0)
    X = x[:, None]
    w = rng.standard_normal(K)
    ident = abs(Lam(w[:, None], X, Yv) - np.mean(np.linalg.norm(w[None] - Yv / x[:, None], axis=1)))
    w_lad = argmin_Lam(X, Yv, K).ravel()
    gm = weiszfeld(Yv / x[:, None])
    w_ls = (Yv * x[:, None]).sum(axis=0) / (x ** 2).sum()
    print(f"  |Lambda(w) - (1/n) sum ||w - y_i/x_i||| = {ident:.1e};  argmin Lambda = {w_lad}, Weiszfeld geometric median "
          f"of y_i/x_i = {gm} (diff {np.abs(w_lad - gm).max():.1e});  least squares = {w_ls}")
    print("=== (d3) existence and uniqueness of the minimizer of Lambda ===")
    X1 = np.array([[1.0], [1.0]]); Y1 = np.array([[0.0], [1.0]])
    vals = [Lam(np.array([[w]]), X1, Y1) for w in (-0.5, 0.0, 0.3, 0.7, 1.0, 1.5)]
    print(f"  K = d = 1, n = 2, x = (1, 1), y = (0, 1): Lambda(w) for w = -0.5, 0, 0.3, 0.7, 1, 1.5: {np.round(vals, 3)}"
          f"  -> every w in [0, 1] is a minimizer (not unique)")
    Xr = np.array([[1.0, 2.0], [2.0, 4.0], [-1.0, -2.0]]); Yr = rng.standard_normal((3, 2))
    Wm = argmin_Lam(Xr, Yr, 2)
    Z = np.outer(rng.standard_normal(2), [2.0, -1.0])            # Z x_i = 0 for all i
    print(f"  rank-deficient design (x_i collinear): Lambda(W) = {Lam(Wm, Xr, Yr):.6f} = Lambda(W + Z) = "
          f"{Lam(Wm + Z, Xr, Yr):.6f}: a minimizer exists, the set of minimizers is unbounded")

    print("=== (a) symmetric laws ===")
    xs = rng.standard_normal((200000, 2, 2))
    worst = np.inf
    for _ in range(20):
        G = rng.standard_normal((2, 2)) * 10.0 ** rng.uniform(-2, 0.5)
        sv = lambda M: np.linalg.svd(M, compute_uv=False).sum(axis=1)
        integrand = sv(xs + G) + sv(xs - G) - 2 * sv(xs)
        lower = np.maximum(0.0, 2 * nuc(G) - 2 * sv(xs))
        worst = min(worst, (integrand - lower).min())
        assert integrand.mean() > 0
    print(f"  Gaussian 2x2: min over samples of [||xi+G||+||xi-G||-2||xi||] - max(0, 2||G||-2||xi||) = {worst:.1e} (>= 0), "
          f"psi(G) - psi(0) > 0 for the 20 directions tested")
    I = np.eye(2)
    S = np.array([[0.2, 0.6], [0.6, -0.3]])
    print(f"  xi = +-I_2: psi(S) - psi(0) = {0.5 * (nuc(S + I) + nuc(S - I)) - 2:.1e} for a symmetric S with "
          f"||S||_op = {np.linalg.norm(S, 2):.2f} < 1: 0 is a nuclear median but not the only one")

    print("=== (b) centred finite laws: ||C||_* <= E||xi - C||_* <= E||xi||_* ===")
    worst1 = worst2 = np.inf
    for trial in range(40):
        K, d = [(1, 1), (2, 2), (2, 3), (3, 3)][trial % 4]
        m = rng.integers(2, 6)
        at = rng.exponential(1.0, (m, K, d)) ** 2
        p = rng.dirichlet(np.ones(m))
        at = at - np.tensordot(p, at, axes=1)[None]               # centred
        fun = lambda c: sum(pi * nuc(a - c.reshape(K, d)) for pi, a in zip(p, at))
        best = None
        for a0 in list(at) + [np.zeros((K, d))]:
            r = minimize(fun, a0.ravel(), method="Powell", options={"xtol": 1e-10, "ftol": 1e-13})
            best = r if best is None or r.fun < best.fun else best
        C = best.x.reshape(K, d)
        E0 = sum(pi * nuc(a) for pi, a in zip(p, at))
        worst1 = min(worst1, best.fun - nuc(C))
        worst2 = min(worst2, E0 - best.fun)
    print(f"  40 random centred laws (2-5 atoms, sizes 1x1 to 3x3): min(E||xi-C|| - ||C||) = {worst1:.2e}, "
          f"min(E||xi|| - E||xi-C||) = {worst2:.2e}  (both >= 0)")
    for pp in (0.9, 0.6, 0.51):
        a = pp / (1 - pp)
        print(f"  two-point centred law xi = -1 (prob {pp}), {a:.3f} (prob {1 - pp:.2f}): median -1, E|xi| = {2 * pp:.2f}"
              f" -> ratio |C| / E|xi| = {1 / (2 * pp):.3f}")

    print("=== (c) fixed points of SAGA and SVRG estimators ===")
    n, d = 6, 3
    A = rng.standard_normal((n, d)); bvec = rng.standard_normal(n); lam = 0.3
    gi = lambda w: A * (A @ w - bvec)[:, None] + lam * w[None]          # per-sample gradients of ridge LS (n x d)
    H = A.T @ A / n + lam * np.eye(d)
    w_star = np.linalg.solve(H, A.T @ bvec / n)
    z = gi(w_star)
    saga = gi(w_star) - z + z.mean(axis=0)[None]
    w_ref = w_star + rng.standard_normal(d)
    svrg_at_star = gi(w_star) - gi(w_star) + gi(w_star).mean(axis=0)[None]
    svrg_other = gi(w_star) - gi(w_ref) + gi(w_ref).mean(axis=0)[None]
    print(f"  ridge LS, generic data: max_j |omega_j| at (W_*, exact table) = {np.abs(saga).max():.1e}; SVRG with reference "
          f"W_*: {np.abs(svrg_at_star).max():.1e}; SVRG at W_* with another reference: max_j |omega_j| = "
          f"{np.abs(svrg_other).max():.3f} (moves)")
    # identical Hessians: f_j(w) = 1/2 (w - a_j)^T H (w - a_j)
    Hc = H
    a = rng.standard_normal((n, d))
    gi2 = lambda w: (w[None] - a) @ Hc
    w_star2 = a.mean(axis=0)
    w_ref = w_star2 + 3.0 * rng.standard_normal(d)
    svrg2 = gi2(w_star2) - gi2(w_ref) + gi2(w_ref).mean(axis=0)[None]
    w_any = rng.standard_normal(d)
    svrg3 = gi2(w_any) - gi2(w_ref) + gi2(w_ref).mean(axis=0)[None]
    print(f"  identical Hessians: SVRG at W_* with a reference at distance {np.linalg.norm(w_ref - w_star2):.2f}: "
          f"max_j |omega_j| = {np.abs(svrg2).max():.1e}  -> (W_*, any reference) is moved by no index;"
          f" max_j |omega_j - grad F(W)| at another W: {np.abs(svrg3 - gi2(w_any).mean(axis=0)[None]).max():.1e}")
    z_bad = gi2(w_star2) + rng.standard_normal((n, d))
    saga2 = gi2(w_star2) - z_bad + z_bad.mean(axis=0)[None]
    print(f"  same sum, SAGA at W_* with an inexact table: max_j |omega_j| = {np.abs(saga2).max():.3f} (moves; a table "
          f"that no index changes is exact)")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("d1", "all"):
        block_d1()
    if which in ("rest", "all"):
        block_rest()
