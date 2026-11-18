"""Unit tests for the Muon laboratory (muon/lab). Run with `python -m pytest muon/tests -q -s` or
`python muon/tests/test_lab.py` (plain runner with a summary; -s shows the printed bands and timings)."""
import functools
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))                       # muon/  (package `lab`)
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))      # project root (package `vrlab`, read-only)

from lab import problems, optimum, polar, algorithms as alg, instrumentation as ins  # noqa: E402


# ----------------------------------------------------------------------------------------------
# shared instances
# ----------------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _prob(loss, n=60, d=6, K=4, kappa=30.0, alpha=1.0, seed=3, solve=True):
    mk = problems.make_softmax if loss == "softmax" else problems.make_matrix_ls
    prob = mk(n, d, K, kappa=kappa, alpha=alpha, seed=seed)
    if solve:
        optimum.solve_optimum(prob)
    return prob


def _rand(prob, rng, scale=0.5):
    return scale * rng.standard_normal((prob.K, prob.d))


def _partial_isometry(rng, K, d, r):
    U, _ = np.linalg.qr(rng.standard_normal((K, r)))
    V, _ = np.linalg.qr(rng.standard_normal((d, r)))
    return U @ V.T


# ----------------------------------------------------------------------------------------------
# problems
# ----------------------------------------------------------------------------------------------
def test_features_match_vrlab():
    from vrlab.problems import make_logistic
    for alpha, rotate in [(0.0, True), (1.0, True), (2.0, False)]:
        ref = make_logistic(80, 7, kappa=40.0, alpha=alpha, seed=5, rotate=rotate)
        a = problems.make_softmax(80, 7, 3, kappa=40.0, alpha=alpha, seed=5, rotate=rotate)
        b = problems.make_matrix_ls(80, 7, 3, kappa=40.0, alpha=alpha, seed=5, rotate=rotate)
        assert np.array_equal(a.X, ref.X) and np.array_equal(b.X, ref.X)
        assert abs(np.sqrt(a.sqnorms.max()) - 1.0) < 1e-14
    a = problems.make_softmax(2000, 10, 5, kappa=50.0, seed=0)
    counts = np.bincount(a.y, minlength=5)
    assert counts.min() > 0 and counts.max() < 2000          # all classes present, not degenerate


def test_planted_orthogonal():
    """planted='orthogonal': flat singular values; for K = d and noise = 0 the polar factor of the gradient at 0 is
    minus the planted direction whatever the feature covariance (Q Sigma is its own polar decomposition)."""
    a = problems.make_matrix_ls(300, 8, 8, kappa=1e4, alpha=2.0, seed=2, noise=0.0, planted="orthogonal")
    P = a.meta["planted"]
    sv = np.linalg.svd(P, compute_uv=False)
    assert np.allclose(sv, sv[0], rtol=1e-12)
    Q = P / sv[0]
    assert np.allclose(Q @ Q.T, np.eye(8), atol=1e-12)
    G0 = a.grad(np.zeros((8, 8)))
    assert np.allclose(polar.polar_svd(G0), -Q, atol=1e-9)
    assert np.linalg.cond(a.Sigma_hat) > 50                                  # ... although Sigma_hat is ill-conditioned
    b = problems.make_matrix_ls(300, 12, 5, kappa=50.0, seed=2, planted="orthogonal")
    sv = np.linalg.svd(b.meta["planted"], compute_uv=False)
    assert np.allclose(sv, sv[0], rtol=1e-12)
    # default unchanged, explicit array accepted
    c = problems.make_matrix_ls(300, 12, 5, kappa=50.0, seed=2)
    d = problems.make_matrix_ls(300, 12, 5, kappa=50.0, seed=2, planted="gaussian")
    assert np.array_equal(c.Y, d.Y) and np.array_equal(c.X, b.X)
    e = problems.make_matrix_ls(300, 12, 5, kappa=50.0, seed=2, planted=np.ones((5, 12)))
    assert np.allclose(e.meta["planted"], e.meta["planted"][0, 0])


def _check_derivatives(prob):
    rng = np.random.default_rng(11)
    W, D = _rand(prob, rng), rng.standard_normal((prob.K, prob.d))
    D /= np.linalg.norm(D)
    g = prob.grad(W)
    # full gradient = mean of per-sample gradients = residual formula = grad_batch over all indices
    assert np.allclose(np.mean([prob.grad_i(W, i) for i in range(prob.n)], axis=0), g, atol=1e-14)
    assert np.allclose(prob.loss_residuals(W).T @ prob.X / prob.n + prob.lam * W, g, atol=1e-15)
    assert np.allclose(prob.grad_batch(W, np.arange(prob.n)), g, atol=1e-15)
    # finite differences of F along coordinates and random directions
    h = 1e-5
    for k in range(prob.K):
        for j in range(prob.d):
            E = np.zeros_like(W)
            E[k, j] = h
            fd = (prob.F(W + E) - prob.F(W - E)) / (2 * h)
            assert abs(fd - g[k, j]) < 1e-8 * max(1.0, abs(g[k, j])), (k, j, fd, g[k, j])
    fd = (prob.F(W + h * D) - prob.F(W - h * D)) / (2 * h)
    assert abs(fd - np.vdot(g, D)) < 1e-8
    # per-sample and minibatch gradients
    for i in (0, 7, prob.n - 1):
        fd = (prob.f_i(W + h * D, i) - prob.f_i(W - h * D, i)) / (2 * h)
        assert abs(fd - np.vdot(prob.grad_i(W, i), D)) < 1e-8
    idx = np.array([3, 3, 9, 1, prob.n - 2])                  # duplicates are counted with multiplicity
    gb = prob.grad_batch(W, idx)
    assert np.allclose(gb, np.mean([prob.grad_i(W, i) for i in idx], axis=0), atol=1e-15)
    fd = (prob.F_minibatch(W + h * D, idx) - prob.F_minibatch(W - h * D, idx)) / (2 * h)
    assert abs(fd - np.vdot(gb, D)) < 1e-8
    # Hessian-vector product: finite differences of the gradient, and the dense Hessian
    Hd = prob.hvp(W, D)
    fdH = (prob.grad(W + h * D) - prob.grad(W - h * D)) / (2 * h)
    assert np.linalg.norm(fdH - Hd) < 1e-8 * max(1.0, np.linalg.norm(Hd))
    H = prob.hessian(W)
    assert np.allclose(H, H.T, atol=1e-15)
    assert np.allclose(H @ D.ravel(), Hd.ravel(), atol=1e-14)
    for c in range(0, prob.K * prob.d, 5):                    # columns of H by finite differences of the gradient
        E = np.zeros(prob.K * prob.d)
        E[c] = h
        E = E.reshape(prob.K, prob.d)
        col = ((prob.grad(W + E) - prob.grad(W - E)) / (2 * h)).ravel()
        assert np.allclose(col, H[:, c], atol=1e-8)
    # accurate difference of F at a moderate distance agrees with the plain difference
    W1 = W + 0.3 * D
    assert abs(prob.F_diff(W1, W) - (prob.F(W1) - prob.F(W))) < 1e-13
    # residual differences: accurate formula vs plain difference
    i = 5
    z = W @ prob.X[i]
    delta = 0.1 * rng.standard_normal(prob.K)
    direct = prob.residual_i(z.copy(), i) - prob.residual_i(z - delta, i)
    assert np.allclose(prob.residual_diff_i(z, delta, i), direct, atol=1e-15)


def test_derivatives_softmax():
    _check_derivatives(_prob("softmax", solve=False))


def test_derivatives_ls():
    _check_derivatives(_prob("ls", solve=False))


def test_constants():
    rng = np.random.default_rng(2)
    for loss in ("softmax", "ls"):
        prob = _prob(loss, solve=False)
        assert abs(prob.kappa / 30.0 - 1) < 1e-12
        assert prob.mu == prob.mu_op == prob.lam
        r = prob.r
        for _ in range(5):
            A, B = _rand(prob, rng, 2.0), _rand(prob, rng, 2.0)
            for i in range(0, prob.n, 7):
                dg = prob.grad_i(A, i) - prob.grad_i(B, i)
                # Euclidean and spectral per-sample smoothness
                assert np.linalg.norm(dg) <= prob.L_i[i] * np.linalg.norm(A - B) * (1 + 1e-12)
                assert polar.nuclear_norm(dg) <= prob.L_op_i[i] * polar.spectral_norm(A - B) * (1 + 1e-12)
            # global constants: Rayleigh quotients of the Hessian
            Wr = _rand(prob, rng, 2.0)
            H = prob.hessian(Wr)
            ev = np.linalg.eigvalsh(H)
            assert ev[-1] <= prob.L_F * (1 + 1e-12) and ev[0] >= prob.lam * (1 - 1e-12)
            for rank in range(1, r + 1):
                D = _partial_isometry(rng, prob.K, prob.d, rank)       # ||D||_op = 1
                assert np.vdot(D, prob.hvp(Wr, D)) <= prob.L_op * (1 + 1e-12)
        if loss == "ls":
            # L_F and L_op are attained (exact constants)
            assert abs(np.linalg.eigvalsh(prob.hessian())[-1] - prob.L_F) < 1e-13
            evals, evecs = np.linalg.eigh(prob.Sigma_hat)
            D = evecs[:, ::-1][:, :r].T                                    # rows: top eigenvectors (K <= d)
            assert abs(np.vdot(D, prob.hvp(np.zeros((prob.K, prob.d)), D)) - prob.L_op) < 1e-13
        else:
            # mu = mu_op = lam is attained along D = 1 v^T (softmax redundancy)
            v = rng.standard_normal(prob.d)
            D = np.outer(np.ones(prob.K), v)
            q = np.vdot(D, prob.hvp(_rand(prob, rng), D))
            assert abs(q - prob.lam * np.vdot(D, D)) < 1e-14 * np.vdot(D, D)
            # c_l = 1/2: lambda_max(diag(p) - p p^T) <= 1/2, approached at p = (1/2, 1/2, 0, ...)
            for _ in range(200):
                p = rng.dirichlet(np.full(prob.K, 0.3))
                assert np.linalg.eigvalsh(np.diag(p) - np.outer(p, p))[-1] <= 0.5 + 1e-15
            p = np.zeros(prob.K)
            p[:2] = 0.5
            assert abs(np.linalg.eigvalsh(np.diag(p) - np.outer(p, p))[-1] - 0.5) < 1e-15


# ----------------------------------------------------------------------------------------------
# polar maps
# ----------------------------------------------------------------------------------------------
def test_polar_svd():
    rng = np.random.default_rng(4)
    for shape in [(6, 15), (15, 6), (7, 7)]:
        M = rng.standard_normal(shape)
        O, s = polar.polar_svd(M, return_sv=True)
        k = min(shape)
        assert np.allclose(O.T @ O if shape[0] >= shape[1] else O @ O.T, np.eye(k), atol=1e-14)
        assert abs(np.vdot(M, O) - s.sum()) < 1e-13 * s.sum()
        assert abs(polar.nuclear_norm(M) - np.linalg.svd(M, compute_uv=False).sum()) < 1e-13 * s.sum()
        assert abs(polar.spectral_norm(M) - np.linalg.norm(M, 2)) < 1e-14 * s[0]
        # O = (M M^T)^{-1/2} M for a full-rank M
        if shape[0] <= shape[1]:
            w, U = np.linalg.eigh(M @ M.T)
            assert np.allclose(O, (U / np.sqrt(w)) @ U.T @ M, atol=1e-12)
        # non-contiguous input
        assert np.allclose(polar.polar_svd(np.asfortranarray(M)), O, atol=1e-14)
        assert np.allclose(polar.polar_svd(M.T), O.T, atol=1e-14)
    # rank-deficient: rank 3 in R^{6 x 15}
    M = rng.standard_normal((6, 3)) @ rng.standard_normal((3, 15))
    O = polar.polar_svd(M)
    so = np.linalg.svd(O, compute_uv=False)
    assert np.allclose(so[:3], 1.0, atol=1e-14) and np.all(so[3:] < 1e-14)
    P = O.T @ O                                     # orthogonal projector onto the row space of M
    assert np.allclose(P @ P, P, atol=1e-14) and np.allclose(P @ M.T, M.T, atol=1e-13)
    assert abs(np.vdot(M, O) - polar.nuclear_norm(M)) < 1e-12 * polar.nuclear_norm(M)
    assert not np.any(polar.polar_svd(np.zeros((3, 5))))
    assert np.all(np.isnan(polar.polar_svd(np.full((3, 5), np.nan))))


def test_polar_softmax_gauge():
    """Near W_* the computed softmax gradient has an absolute rounding component along 1 that the exact polar factor
    turns into a unit singular direction; remove_gauge removes it."""
    prob = _prob("softmax", n=200, d=10, K=4, kappa=100.0, seed=2)
    rng = np.random.default_rng(0)
    worst_raw = worst_proj = 0.0
    for _ in range(100):
        P = rng.standard_normal((prob.K, prob.d)) * 10.0 ** rng.uniform(-12, -4)
        W = prob.W_star + P
        W -= W.mean(axis=0, keepdims=True)
        G = prob.grad(W)
        worst_raw = max(worst_raw, np.abs(np.ones(prob.K) @ polar.polar_svd(G)).max())
        worst_proj = max(worst_proj, np.abs(np.ones(prob.K) @ polar.polar_svd(prob.remove_gauge(G))).max())
    print(f"\n  softmax gradients near W_*: max |1^T polar(G)| = {worst_raw:.2e} raw, {worst_proj:.2e} after remove_gauge")
    assert worst_proj < 1e-13
    assert worst_raw > 0.1                          # the phenomenon is real (documented in README / remove_gauge)


def test_ns_cubic_converges():
    rng = np.random.default_rng(5)
    for shape in [(8, 20), (20, 8)]:
        U, _ = np.linalg.qr(rng.standard_normal((shape[0], min(shape))))
        V, _ = np.linalg.qr(rng.standard_normal((shape[1], min(shape))))
        s = np.geomspace(1.0, 1e-3, min(shape)) * 7.0
        M = (U * s) @ V.T
        assert np.abs(polar.ns_cubic(M, steps=60) - polar.polar_svd(M)).max() < 1e-13
        # implementation check against the scalar map
        sv = np.sort(polar.singular_values(polar.ns_cubic(M, steps=6)))
        pred = np.sort(polar.ns_scalar_map(s / np.linalg.norm(M), 6, polar.CUBIC_COEFFS))
        assert np.allclose(sv, pred, atol=1e-13)


def test_ns5():
    rng = np.random.default_rng(6)
    a, b, c = polar.NS5_COEFFS

    def keller_literal(G, steps=5, eps=1e-7):          # line-by-line transcription of the PyTorch code (float64)
        X = G
        if G.shape[0] > G.shape[1]:
            X = X.T
        X = X / (np.linalg.norm(X) + eps)
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X
        if G.shape[0] > G.shape[1]:
            X = X.T
        return X

    lo, hi, lo_all, x_min = np.inf, -np.inf, np.inf, np.inf
    for trial in range(200):
        shape = [(10, 50), (50, 10), (16, 16), (4, 30)][trial % 4]
        M = rng.standard_normal(shape) * rng.uniform(0.01, 100)
        X = polar.ns5(M)
        assert np.allclose(X, keller_literal(M), atol=1e-13)
        if shape[0] != shape[1]:                     # same arithmetic on M and M^T: the smaller Gram matrix is used
            assert np.array_equal(polar.ns5(M.T), X.T)
        else:                                        # square: X X^T for M, X^T X for M^T (equal up to rounding only)
            assert np.allclose(polar.ns5(M.T), X.T, atol=1e-13)
        sv = np.sort(polar.singular_values(X))
        s = polar.singular_values(M)
        xin = s / (np.linalg.norm(M) + 1e-7)                      # normalized singular values fed to the polynomial
        out = polar.ns_scalar_map(xin)
        assert np.allclose(sv, np.sort(out), atol=1e-12)
        in_band = xin >= 0.01                                     # the band [0.68, 1.20] holds for inputs >= 0.01 only
        lo, hi = min(lo, out[in_band].min()), max(hi, out.max())
        lo_all, x_min = min(lo_all, sv[0]), min(x_min, xin.min())
    # theoretical band of p^5 on normalized singular values x in [0.01, 1]
    x = np.linspace(0.01, 1.0, 400001)
    y = polar.ns_scalar_map(x)
    print(f"\n  ns5 (Keller coefficients, 5 steps) on 200 Gaussian matrices: output singular values in "
          f"[{lo:.4f}, {hi:.4f}] for normalized inputs >= 0.01; smallest output {lo_all:.4f} (square matrices: "
          f"smallest normalized input {x_min:.1e}); band of p^5 over x = s/||M||_F in [0.01, 1]: "
          f"[{y.min():.4f}, {y.max():.4f}]; p(1) = {a + b + c:.4f}")
    assert 0.6 < lo and hi < 1.25 and 0.6 < y.min() and y.max() < 1.21
    assert lo_all < 0.6                                           # ill-conditioned (square) inputs fall below the band
    # bf16 emulation: close to the float64 iteration, singular values in a slightly wider band
    M = rng.standard_normal((10, 50))
    Xb, X = polar.ns5(M, bf16=True), polar.ns5(M)
    svb = polar.singular_values(Xb)
    print(f"  ns5 bf16 emulation: singular values in [{svb.min():.4f}, {svb.max():.4f}],"
          f" max |X_bf16 - X_f64| = {np.abs(Xb - X).max():.2e}")
    assert np.abs(Xb - X).max() < 0.1 and 0.5 < svb.min() and svb.max() < 1.3
    assert np.array_equal(polar.bf16_round(Xb), Xb)                    # output is bf16-representable
    # dispatcher
    assert np.allclose(polar.orth(M, "svd"), polar.polar_svd(M)) and np.allclose(polar.orth(M, "ns5"), X)
    assert np.allclose(polar.orth(M, "ns_cubic", steps=3), polar.ns_cubic(M, steps=3))
    assert np.allclose(polar.orth_singular_values(M, "svd"), 1.0)


def test_bf16_round():
    r = polar.bf16_round
    for v in [1.0, 0.5, 3.0, 1 + 2 ** -7, -2.75, 0.0]:
        assert r(v) == v
    assert r(1 + 2 ** -8) == 1.0                         # tie -> even (1.0)
    assert r(1 + 3 * 2 ** -8) == 1 + 2 ** -6             # tie -> even (1 + 2^-6)
    assert r(1 + 2 ** -8 + 2 ** -20) == 1 + 2 ** -7      # above the tie
    assert r(-(1 + 3 * 2 ** -8)) == -(1 + 2 ** -6)
    x = np.random.default_rng(7).standard_normal(10000) * 10.0 ** np.random.default_rng(8).uniform(-20, 20, 10000)
    y = r(x)
    assert np.all(np.abs(y - x) <= 2 ** -8 * np.abs(x) * (1 + 1e-6))
    assert np.array_equal(r(y), y)
    assert np.isnan(r(np.array([np.nan]))[0])


# ----------------------------------------------------------------------------------------------
# optimum and accurate gaps
# ----------------------------------------------------------------------------------------------
def test_optimum_ls():
    prob = _prob("ls", n=200, d=10, K=4, kappa=1e3, alpha=1.0, seed=1)
    info = prob.opt_info
    assert info["grad_norm"] <= 1e-14 * max(1.0, info["grad_norm0"])
    assert abs(info["err_bound"] - info["grad_norm"] / prob.lam) < 1e-30
    # agrees with a dense solve of the vectorized normal equations (H vec W = vec(Y^T X / n))
    B = prob.Y.T @ prob.X / prob.n
    Wd = np.linalg.solve(prob.hessian(), B.ravel()).reshape(prob.K, prob.d)
    assert np.linalg.norm(Wd - prob.W_star) <= 1e-12 * np.linalg.norm(prob.W_star)


def test_optimum_softmax():
    prob = _prob("softmax", n=300, d=10, K=5, kappa=1e3, alpha=1.0, seed=1)
    info = prob.opt_info
    print(f"\n  softmax Newton: ||grad F(W_*)||_F = {info['grad_norm']:.2e} (tol {info['tol']:.1e}), "
          f"{info['iters']} iterations, error bound {info['err_bound']:.2e}")
    assert info["converged"] and info["grad_norm"] <= 1e-14 * max(1.0, info["grad_norm0"])
    assert abs(info["err_bound"] - info["grad_norm"] / prob.lam) < 1e-30
    assert abs(np.ones(prob.K) @ prob.W_star).max() < 1e-13           # 1^T W_* = 0
    # a second solve from another start agrees within the two strong-convexity error bounds (plus rounding)
    W2, info2 = optimum.newton_softmax(prob, W0=np.random.default_rng(1).standard_normal((prob.K, prob.d)))
    assert np.linalg.norm(W2 - prob.W_star) <= info["err_bound"] + info2["err_bound"] + 1e-13


def _mp_F(prob, W, dps=50):
    """F(W) in `dps`-digit arithmetic from the stored float64 data."""
    import mpmath as mp
    with mp.workdps(dps):
        X = [[mp.mpf(float(v)) for v in row] for row in prob.X]
        Wm = [[mp.mpf(float(v)) for v in row] for row in np.asarray(W)]
        tot = mp.mpf(0)
        for i in range(prob.n):
            z = [mp.fsum(Wm[k][j] * X[i][j] for j in range(prob.d)) for k in range(prob.K)]
            if prob.loss == "softmax":
                tot += mp.log(mp.fsum(mp.exp(zk) for zk in z)) - z[int(prob.y[i])]
            else:
                tot += mp.fsum((z[k] - mp.mpf(float(prob.Y[i, k]))) ** 2 for k in range(prob.K)) / 2
        reg = mp.mpf(float(prob.lam)) / 2 * mp.fsum(w * w for row in Wm for w in row)
        return tot / prob.n + reg


def test_Fgap_accuracy_mpmath():
    """Fgap via the Bregman identity is accurate to O(eps ||W - W_*||) (the plain difference has a floor ~eps |F_*|)."""
    import mpmath as mp
    rng = np.random.default_rng(9)
    for loss in ("softmax", "ls"):
        prob = _prob(loss, n=12, d=3, K=3, kappa=20.0, alpha=0.0, seed=4)
        F_star_mp = _mp_F(prob, prob.W_star)
        D = rng.standard_normal((prob.K, prob.d))
        D -= D.mean(axis=0, keepdims=True)
        for h in (1e-2, 1e-5, 1e-8, 1e-11):
            W = prob.W_star + h * D
            E = W - prob.W_star
            with mp.workdps(50):
                exact = float(_mp_F(prob, W) - F_star_mp)
            err = abs(prob.Fgap(W) - exact)
            naive = abs((prob.F(W) - prob.F_star) - exact)
            assert err <= 1e-15 * np.linalg.norm(E) + 1e-13 * abs(exact), (loss, h, err, exact)
            if h <= 1e-8:
                assert naive > 1e3 * err                 # the plain difference is useless there
    # residual differences for tiny delta (softmax) against mpmath
    prob = _prob("softmax", n=12, d=3, K=3, kappa=20.0, alpha=0.0, seed=4)
    z = np.array([0.3, -1.2, 2.0])
    delta = 1e-10 * np.array([1.0, -2.0, 0.5])
    with mp.workdps(50):
        def sm(v):
            e = [mp.exp(mp.mpf(float(t))) for t in v]
            s = mp.fsum(e)
            return [x / s for x in e]
        zm = [mp.mpf(float(t)) for t in z]
        dm = [mp.mpf(float(t)) for t in delta]
        p1, p0 = sm(z), [None] * 3
        e = [mp.exp(zm[k] - dm[k]) for k in range(3)]
        p0 = [x / mp.fsum(e) for x in e]
        exact = np.array([float(p1[k] - p0[k]) for k in range(3)])
    got = prob.residual_diff_i(z, delta, 0)
    assert np.abs(got - exact).max() <= 1e-14 * np.abs(exact).max()


# ----------------------------------------------------------------------------------------------
# estimators (exact enumeration of all indices)
# ----------------------------------------------------------------------------------------------
def test_unbiased_saga_estimator():
    rng = np.random.default_rng(12)
    for loss in ("softmax", "ls"):
        prob = _prob(loss)
        W = _rand(prob, rng)
        # memory at random past points: each sample's residual taken at one of 4 random matrices
        tables = [prob.loss_residuals(_rand(prob, rng)) for _ in range(4)]
        pick = rng.integers(0, 4, prob.n)
        rtab = np.array([tables[pick[i]][i] for i in range(prob.n)])
        zbar = rtab.T @ prob.X / prob.n
        ests = [alg.saga_estimator(prob, W, j, rtab, zbar)[0] for j in range(prob.n)]
        g = prob.grad(W)
        assert np.linalg.norm(np.mean(ests, axis=0) - g) <= 1e-14 * max(1.0, np.linalg.norm(g))
        # incremental memory updates keep zbar = (1/n) rtab^T X
        rt, zb = rtab.copy(), zbar.copy()
        for j in rng.integers(0, prob.n, 500):
            _, rj, outer = alg.saga_estimator(prob, _rand(prob, rng), j, rt, zb)
            zb += outer / prob.n
            rt[j] = rj
        assert np.linalg.norm(zb - rt.T @ prob.X / prob.n) < 1e-14


def test_unbiased_lsvrg_estimator():
    rng = np.random.default_rng(13)
    for loss in ("softmax", "ls"):
        prob = _prob(loss)
        W, w = _rand(prob, rng), _rand(prob, rng)
        Rw = prob.loss_residuals(w)
        gbar = Rw.T @ prob.X / prob.n
        ests = [alg.lsvrg_estimator(prob, W, j, Rw, gbar)[0] for j in range(prob.n)]
        g = prob.grad(W)
        assert np.linalg.norm(np.mean(ests, axis=0) - g) <= 1e-14 * max(1.0, np.linalg.norm(g))
        # the estimator is f_j'(W) - f_j'(w) + grad F(w)
        j = 7
        ref = prob.grad_i(W, j) - prob.grad_i(w, j) + prob.grad(w)
        assert np.allclose(ests[j], ref, atol=1e-15)


# ----------------------------------------------------------------------------------------------
# convergence
# ----------------------------------------------------------------------------------------------
def test_gd_and_saga_converge():
    for loss in ("softmax", "ls"):
        prob = _prob(loss, n=100, d=8, K=4, kappa=20.0, alpha=1.0, seed=0)
        W0 = np.zeros((prob.K, prob.d))
        r = alg.gd(prob, W0, 3000, record_every=100, stop_dist2=1e-24)
        assert r["dist2_F"][-1] <= 1e-24 and r["stopped"] == "stop_dist2"
        assert r["grad_evals"][-1] == prob.n * r["iters"][-1]
        T = 60 * prob.n
        r = alg.saga(prob, W0, T)
        assert r["dist2_F"][-1] <= 1e-18 * r["dist2_F"][0], (loss, r["dist2_F"][-1])
        assert r["grad_evals"][-1] == prob.n + T and r["iters"][-1] == T
        assert r["zbar_err"] < 1e-14
        assert r["Fgap"][-1] < 1e-18 * r["Fgap"][0]


def test_muon_full_monotone_ls():
    prob = _prob("ls", n=200, d=10, K=4, kappa=50.0, alpha=1.0, seed=1)
    W0 = np.zeros((prob.K, prob.d))
    for rule in ("sharp", "linesearch"):
        r = alg.muon_full(prob, W0, 400, rule=rule, record_every=1)
        fg, pair = r["Fgap"], r["pair"]
        guaranteed = pair[1:] ** 2 / (2 * prob.L_op)            # ||G||_*^2 / (2 L_op) with an exact polar factor
        dec = fg[:-1] - fg[1:]
        live = fg[:-1] > 1e-22 * fg[0]
        assert live.sum() > 50
        assert np.all(dec[live] > 0), rule
        assert np.all(dec[live] >= guaranteed[live] * (1 - 1e-8) - 1e-12 * fg[:-1][live]), rule
        assert r["dist2_F"][-1] <= 1e-25 * r["dist2_F"][0], (rule, r["dist2_F"][-1])
        per = 2 if rule == "linesearch" else 1
        assert r["grad_evals"][-1] == per * prob.n * 400 and r["hvp_evals"] == (400 if rule == "linesearch" else 0)


def test_saga_muon_sharp_converges_softmax():
    prob = _prob("softmax", n=100, d=8, K=4, kappa=20.0, alpha=1.0, seed=0)
    W0 = np.zeros((prob.K, prob.d))
    T = 60 * prob.n
    r = alg.saga_muon(prob, W0, T, rule="sharp")
    print(f"\n  saga_muon sharp (softmax n=100, d=8, K=4, kappa=20): dist2 {r['dist2_F'][0]:.2e} -> {r['dist2_F'][-1]:.2e}"
          f" in 60 passes (SAGA: {alg.saga(prob, W0, T)['dist2_F'][-1]:.2e})")
    assert r["dist2_F"][-1] <= 1e-16 * r["dist2_F"][0]
    assert r["grad_evals"][-1] == prob.n + T
    # the same index stream as SAGA (same seed): first estimator identical
    assert abs(r["g_sharp"] - 1.0 / (4 * prob.R2_op)) < 1e-18


def test_other_methods_run_and_decrease():
    prob = _prob("softmax", n=100, d=8, K=4, kappa=20.0, alpha=1.0, seed=0)
    W0 = np.zeros((prob.K, prob.d))
    n = prob.n
    d0 = alg.gd(prob, W0, 0)["dist2_F"][0]
    runs = {
        "muon_full const": alg.muon_full(prob, W0, 50, rule="const", lr=0.05),
        "muon_full bb": alg.muon_full(prob, W0, 200, rule="bb"),
        "muon_full sharp ns5 momentum": alg.muon_full(prob, W0, 200, rule="sharp", orth="ns5", beta=0.5, nesterov=True),
        "muon_sgd ns5": alg.muon_sgd(prob, W0, 600, batch=10, lr=0.02),
        "muon_sgd svd cosine reshuffle": alg.muon_sgd(prob, W0, 600, batch=10, lr=0.05, orth="svd", schedule="cosine",
                                                      sampling="reshuffle", weight_decay=0.01),
        "muon_sgd ns_cubic wsd": alg.muon_sgd(prob, W0, 600, batch=10, lr=0.05, orth="ns_cubic", schedule="wsd"),
        "sgd linear momentum": alg.sgd(prob, W0, 600, batch=10, lr=1.0, schedule="linear", beta=0.9),
        "storm_muon": alg.storm_muon(prob, W0, 600, batch=10, lr=0.02, a=0.2),
        "saga_muon const": alg.saga_muon(prob, W0, 20 * n, rule="const", lr=0.01),
        "saga_muon decay ns5": alg.saga_muon(prob, W0, 20 * n, rule="decay", lr=0.05, orth="ns5"),
        "saga_muon bb same_sample": alg.saga_muon(prob, W0, 20 * n, rule="bb"),
        "saga_muon bb consecutive": alg.saga_muon(prob, W0, 20 * n, rule="bb", bb_secant="consecutive"),
        "saga_muon sharp momentum": alg.saga_muon(prob, W0, 20 * n, rule="sharp", beta=0.5),
        "lsvrg_muon sharp": alg.lsvrg_muon(prob, W0, 20 * n, rule="sharp"),
        "lsvrg": alg.lsvrg(prob, W0, 20 * n),
    }
    for name, r in runs.items():
        assert np.all(np.isfinite(r["dist2_F"])) and r["stopped"] is None, name
        assert r["dist2_F"][-1] < 0.1 * d0, (name, r["dist2_F"][-1])
        assert r["iters"][-1] == {"muon_full const": 50, "muon_full bb": 200}.get(
            name, 200 if name.startswith("muon_full") else (600 if r["iters"][-1] <= 600 else 20 * n)), name
    # gradient accounting
    r = runs["storm_muon"]
    assert r["grad_evals"][-1] == 10 + 2 * 10 * 599
    r = runs["muon_sgd ns5"]
    assert r["grad_evals"][-1] == 10 * 600
    for name in ("lsvrg_muon sharp", "lsvrg"):
        r = runs[name]
        assert r["grad_evals"][-1] == n + 2 * 20 * n + n * r["refreshes"]
    r = runs["saga_muon bb same_sample"]
    assert r["grad_evals"][-1] == n + 2 * 20 * n and np.isfinite(r["L_hat"][-1])
    assert prob.mu_op <= r["L_hat"][-1] <= prob.R2_op
    # schedules
    f = alg.lr_factors("wsd", 10, decay_frac=0.2)
    assert np.allclose(f, [1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5])
    assert np.allclose(alg.lr_factors("cosine", 4), [1.0, 0.5 * (1 + np.cos(np.pi / 4)), 0.5, 0.5 * (1 + np.cos(3 * np.pi / 4))])
    assert np.allclose(alg.lr_factors("const", 5, warmup=0.4), [0.5, 1, 1, 1, 1])
    # stop rule and final record
    r = alg.saga(prob, W0, 100 * n, stop_dist2=1e-8 * d0)
    assert r["stopped"] == "stop_dist2" and r["dist2_F"][-1] <= 1e-8 * d0
    r = alg.saga(prob, W0, 1234, record_every=500)
    assert list(r["iters"]) == [0, 500, 1000, 1234]


def test_instrument():
    rng = np.random.default_rng(14)
    for loss in ("softmax", "ls"):
        prob = _prob(loss, n=100, d=8, K=4, kappa=20.0, alpha=1.0, seed=0)
        for scale in (1.0, 1e-3):
            W = prob.W_star + scale * rng.standard_normal((prob.K, prob.d))
            st = ins.instrument(prob, W, eta=0.3 * scale)
            assert abs(st["mu_eff_op"] - st["srank_E"] * st["mu_eff_F"]) < 1e-12 * st["mu_eff_op"]
            assert st["mu_eff_F"] >= prob.lam * (1 - 1e-12) and st["mu_eff_op"] >= prob.mu_op * (1 - 1e-12)
            assert 1 - 1e-12 <= st["srank_E"] <= prob.r + 1e-12 and 1 - 1e-12 <= st["nrank_G"] <= prob.r + 1e-12
            assert st["L_eff_F"] <= prob.c_loss * prob.sqnorms.max() * (1 + 1e-12)
            assert st["L_eff_F_reg"] <= prob.R2 * (1 + 1e-12)
            assert st["L_eff_op_reg"] <= prob.R2_op * (1 + 1e-12)
            assert st["L_eff_op"] == st["L_eff_F"]
            assert abs(st["align_dual"] - 1.0) < 1e-12 and abs(st["cos_G"] - 1.0) < 1e-12   # G = grad, O = polar(G)
            assert st["decrease_bound"] <= st["decrease"] * (1 + 1e-10) + 1e-30
            if loss == "ls":
                assert abs(st["decrease"] - st["decrease_model"]) <= 1e-10 * abs(st["decrease"])
            # curvature along O from the dense Hessian
            O = polar.polar_svd(prob.grad(W))
            assert abs(st["curv_O"] - O.ravel() @ prob.hessian(W) @ O.ravel()) < 1e-12 * st["curv_O"]
            # vrlab's L_eff (lam inside f_i) from explicit per-sample gradients
            S = np.mean([np.linalg.norm(prob.grad_i(W, i) - prob.grad_i(prob.W_star, i)) ** 2 for i in range(prob.n)])
            assert abs(st["L_eff_F_reg"] - S / st["s"]) < 1e-10 * st["L_eff_F_reg"]


# ----------------------------------------------------------------------------------------------
# timing
# ----------------------------------------------------------------------------------------------
def test_timing():
    """Iterations per second at n = 1000, d = 50, K = 10 (printed; the assertion is only a loose floor)."""
    lines = []
    for loss in ("softmax", "ls"):
        prob = _prob(loss, n=1000, d=50, K=10, kappa=100.0, alpha=0.5, seed=0)
        W0 = np.zeros((prob.K, prob.d))
        for label, f, T in [
            ("saga", lambda T: alg.saga(prob, W0, T), 20000),
            ("saga (record every n)", lambda T: alg.saga(prob, W0, T, record_every=prob.n), 20000),
            ("saga_muon svd", lambda T: alg.saga_muon(prob, W0, T), 10000),
            ("saga_muon svd (record every n)", lambda T: alg.saga_muon(prob, W0, T, record_every=prob.n), 10000),
            ("saga_muon ns5", lambda T: alg.saga_muon(prob, W0, T, orth="ns5"), 10000),
        ]:
            t0 = time.perf_counter()
            r = f(T)
            rate = r["iters"][-1] / (time.perf_counter() - t0)
            lines.append(f"  {loss:7s} {label:32s} {rate:8.0f} it/s")
            assert rate > 1000, (loss, label, rate)
    print("\n  timing at n=1000, d=50, K=10 (record_every = n/10 unless stated):\n" + "\n".join(lines))


# ----------------------------------------------------------------------------------------------
# plain runner
# ----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items(), key=lambda kv: kv[1].__code__.co_firstlineno
                                        if callable(kv[1]) and hasattr(kv[1], "__code__") else 0)
             if k.startswith("test_") and callable(v)]
    failed = []
    t_all = time.perf_counter()
    for name, fn in tests:
        t0 = time.perf_counter()
        try:
            fn()
            print(f"PASS {name} ({time.perf_counter() - t0:.1f}s)")
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            import traceback
            traceback.print_exc()
            print(f"FAIL {name}: {exc!r}")
    print(f"\n{len(tests) - len(failed)} passed, {len(failed)} failed in {time.perf_counter() - t_all:.1f}s")
    sys.exit(1 if failed else 0)
