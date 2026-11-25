"""Audit math-smd -- Theorem 2(a) of paper_saga_to_muon.tex (lines 395-416).

Checks, for a noise xi with FINITE support (so that every expectation is exact) and for a quadratic and a
non-quadratic F on K x d matrices:
  (i)   the pathwise three-point identity
        D_F(W+, W') = D_F(W, W') - eta <O, G - G'> + D_F(W+, W),  O = polar(G + xi), W+ = W - eta O;
  (ii)  D_F(W+, W) <= L_op eta^2 / 2;
  (iii) <Phi_xi(G), G - G'> >= psi_xi(G) - psi_xi(G')   (subgradient inequality, Phi_xi(G) in d psi_xi(G));
  (iv)  inequality (a): E[D_F(W+, W^dag) | W] <= D_F(W, W^dag) - eta (psi_xi(G) - min psi_xi) + L_op eta^2 / 2,
        with G^dag a numerical minimizer of psi_xi and W^dag = grad F^*(G^dag);
  (v)   the same inequality with an arbitrary reference point W' in the place of W^dag (the proof does not use
        minimality), and a Monte-Carlo version with a continuous law (entries Exp(1) - 1).
Also includes rank-deficient atoms and the atom G + xi = 0 (polar(0) = 0).

Run from the project root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 paper/audit/checks/math-smd_thm2a_inequality.py
"""
import numpy as np
from scipy.optimize import minimize

rng = np.random.default_rng(20261120)


def polar(M, tol=1e-12):
    """U V^T over the singular values > 0 (polar(0) = 0), as defined in the paper (line 197)."""
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    keep = s > tol * max(1.0, s[0] if s.size else 1.0)
    return (U[:, keep]) @ Vt[keep, :]


def nuc(M):
    return np.linalg.svd(M, compute_uv=False).sum()


class Quad:
    """F(W) = 1/2 tr(W S W^T) - <B, W>."""

    def __init__(self, K, d):
        A = rng.standard_normal((d, d))
        ev = np.sort(rng.uniform(0.3, 3.0, d))[::-1]
        Q, _ = np.linalg.qr(A)
        self.S = Q @ np.diag(ev) @ Q.T
        self.B = rng.standard_normal((K, d))
        self.Lop = ev[: min(K, d)].sum()       # sum of the r largest eigenvalues (exact)
        self.muop = ev[-1]
        self.K, self.d = K, d

    def f(self, W):
        return 0.5 * np.sum((W @ self.S) * W) - np.sum(self.B * W)

    def grad(self, W):
        return W @ self.S - self.B

    def grad_conj(self, G):
        return (G + self.B) @ np.linalg.inv(self.S)


class NonQuad:
    """F(W) = 1/2 tr(W S W^T) + c log cosh(<Q, W>) - <B, W>, K = d, Q orthogonal: L_op = tr S + c d^2 exactly
    (tr(D S D^T) = tr S and <Q, D>^2 <= d^2, both attained at D = Q, where <Q, W> = 0)."""

    def __init__(self, d, c=0.7):
        self.q = Quad(d, d)
        self.Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        self.c = c
        self.Lop = np.trace(self.q.S) + c * d * d
        self.muop = self.q.muop
        self.K = self.d = d

    def f(self, W):
        t = np.sum(self.Q * W)
        return self.q.f(W) + self.c * (np.logaddexp(t, -t) - np.log(2.0))

    def grad(self, W):
        return self.q.grad(W) + self.c * np.tanh(np.sum(self.Q * W)) * self.Q

    def grad_conj(self, G):
        # Newton on W -> grad F(W) - G (strongly convex, C^2)
        W = self.q.grad_conj(G)
        Sinv = np.linalg.inv(self.q.S)
        for _ in range(200):
            r = self.grad(W) - G
            if np.abs(r).max() < 1e-13:
                break
            t = np.sum(self.Q * W)
            h = self.c / np.cosh(t) ** 2
            # Hessian: D -> D S + h <Q, D> Q ; solve by Sherman-Morrison in vectorised form
            QS = self.Q @ Sinv
            a = r @ Sinv
            D = a - h * np.sum(self.Q * a) / (1.0 + h * np.sum(self.Q * QS)) * QS
            W = W - D
        return W


def breg(F, A, Bp):
    """D_F(A, B) = F(A) - F(B) - <grad F(B), A - B>."""
    return F.f(A) - F.f(Bp) - np.sum(F.grad(Bp) * (A - Bp))


def psi(G, atoms, p):
    return sum(pi * (nuc(G + x) - nuc(x)) for pi, x in zip(p, atoms))


def Phi(G, atoms, p):
    return sum(pi * polar(G + x) for pi, x in zip(p, atoms))


def argmin_psi(atoms, p, shape):
    fun = lambda g: psi(g.reshape(shape), atoms, p)
    best = None
    for _ in range(6):
        x0 = rng.standard_normal(shape[0] * shape[1])
        r = minimize(fun, x0, method="Powell", options={"xtol": 1e-10, "ftol": 1e-14, "maxiter": 40000})
        r = minimize(fun, r.x, method="Nelder-Mead", options={"xatol": 1e-11, "fatol": 1e-15, "maxiter": 40000})
        if best is None or r.fun < best.fun:
            best = r
    return best.x.reshape(shape), best.fun


def run_case(name, F, atoms, p, n_trials=4000):
    K, d = F.K, F.d
    Gd, pmin = argmin_psi(atoms, p, (K, d))
    Wd = F.grad_conj(Gd)
    assert np.abs(F.grad(Wd) - Gd).max() < 1e-9
    worst = dict(id3=0.0, smooth=-np.inf, subg=np.inf, a_dag=-np.inf, a_any=-np.inf, psi_gap=np.inf)
    tight = np.inf
    for trial in range(n_trials):
        scale = 10.0 ** rng.uniform(-2, 1)
        W = Wd + scale * rng.standard_normal((K, d))
        if trial % 7 == 0:                       # put G + xi = 0 exactly for one atom (polar(0) = 0)
            W = F.grad_conj(-atoms[trial % len(atoms)])
        eta = 10.0 ** rng.uniform(-3, 0.5)
        G = F.grad(W)
        Wp_ref = Wd + rng.standard_normal((K, d))        # arbitrary reference point W'
        Gp_ref = F.grad(Wp_ref)
        lhs_dag = lhs_any = 0.0
        for pi, x in zip(p, atoms):
            O = polar(G + x)
            Wn = W - eta * O
            # (i) three-point identity, pathwise
            id3 = breg(F, Wn, Wd) - (breg(F, W, Wd) - eta * np.sum(O * (G - Gd)) + breg(F, Wn, W))
            worst["id3"] = max(worst["id3"], abs(id3) / (1.0 + breg(F, W, Wd)))
            # (ii) smoothness
            worst["smooth"] = max(worst["smooth"], breg(F, Wn, W) - 0.5 * F.Lop * eta ** 2)
            lhs_dag += pi * breg(F, Wn, Wd)
            lhs_any += pi * breg(F, Wn, Wp_ref)
        ps = psi(G, atoms, p)
        Ph = Phi(G, atoms, p)
        worst["psi_gap"] = min(worst["psi_gap"], ps - pmin)
        # (iii) subgradient inequality against G^dag and against an arbitrary point
        worst["subg"] = min(worst["subg"], np.sum(Ph * (G - Gd)) - (ps - pmin),
                            np.sum(Ph * (G - Gp_ref)) - (ps - psi(Gp_ref, atoms, p)))
        rhs_dag = breg(F, W, Wd) - eta * (ps - pmin) + 0.5 * F.Lop * eta ** 2
        rhs_any = breg(F, W, Wp_ref) - eta * (ps - psi(Gp_ref, atoms, p)) + 0.5 * F.Lop * eta ** 2
        worst["a_dag"] = max(worst["a_dag"], (lhs_dag - rhs_dag) / (1.0 + abs(rhs_dag)))
        worst["a_any"] = max(worst["a_any"], (lhs_any - rhs_any) / (1.0 + abs(rhs_any)))
        tight = min(tight, (rhs_dag - lhs_dag) / (0.5 * F.Lop * eta ** 2))
    print(f"[{name}] K x d = {K} x {d}, {len(atoms)} atoms, L_op = {F.Lop:.4f}, mu_op = {F.muop:.4f}")
    print(f"   min psi = {pmin:.6f}, ||G^dag||_* = {nuc(Gd):.6f}, ||Phi(G^dag)||_F = {np.linalg.norm(Phi(Gd, atoms, p)):.2e}")
    print(f"   (i)   max |three-point identity residual| (relative)      = {worst['id3']:.2e}   (expected ~1e-15)")
    print(f"   (ii)  max  D_F(W+,W) - L_op eta^2/2                       = {worst['smooth']:.2e}   (expected <= 0)")
    print(f"   (iii) min  <Phi(G), G-G'> - (psi(G)-psi(G'))              = {worst['subg']:.2e}   (expected >= 0)")
    print(f"         min  psi(G) - psi(G^dag) over the trials            = {worst['psi_gap']:.2e}   (expected >= 0)")
    print(f"   (iv)  max  (LHS - RHS) of (a), reference W^dag (relative) = {worst['a_dag']:.2e}   (expected <= 0)")
    print(f"   (v)   max  (LHS - RHS) of (a), arbitrary reference W'     = {worst['a_any']:.2e}   (expected <= 0)")
    print(f"         min slack of (a) in units of L_op eta^2/2           = {tight:.3e}   (>= 0; small = tight)")
    return worst


def atoms_generic(K, d, m, skew=True):
    at = [rng.exponential(1.0, (K, d)) - 1.0 if skew else rng.standard_normal((K, d)) for _ in range(m)]
    p = rng.dirichlet(np.ones(m))
    return at, p


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    ok = True
    # 1. scalar, the two-point law of the paper (line 393)
    F = Quad(1, 1)
    w = run_case("scalar, P(xi=1)=0.4, P(xi=-1)=0.6, quadratic F", F, [np.array([[1.0]]), np.array([[-1.0]])],
                 [0.4, 0.6], 2000)
    ok &= w["a_dag"] <= 1e-9 and w["a_any"] <= 1e-9
    # 2. 2 x 2 quadratic, skewed atoms
    F = Quad(2, 2)
    at, p = atoms_generic(2, 2, 7)
    w = run_case("2x2 quadratic, 7 skewed atoms", F, at, p)
    ok &= w["a_dag"] <= 1e-9 and w["a_any"] <= 1e-9 and w["smooth"] <= 1e-10 and w["subg"] >= -1e-9
    # 3. 2 x 3 quadratic with rank-deficient atoms and an atom equal to 0
    F = Quad(2, 3)
    at, p = atoms_generic(2, 3, 5)
    at.append(np.outer(rng.standard_normal(2), rng.standard_normal(3)))   # rank one
    at.append(np.zeros((2, 3)))
    p = rng.dirichlet(np.ones(len(at)))
    w = run_case("2x3 quadratic, rank-deficient atoms", F, at, p)
    ok &= w["a_dag"] <= 1e-9 and w["a_any"] <= 1e-9 and w["smooth"] <= 1e-10 and w["subg"] >= -1e-9
    # 4. 3 x 3 non-quadratic F with exact L_op
    F = NonQuad(3)
    at, p = atoms_generic(3, 3, 6)
    w = run_case("3x3 NON-quadratic F (log cosh), 6 skewed atoms", F, at, p, 2500)
    ok &= w["a_dag"] <= 1e-9 and w["a_any"] <= 1e-9 and w["smooth"] <= 1e-10 and w["subg"] >= -1e-9
    # 5. Monte-Carlo version with a continuous law (entries Exp(1)-1), 2x2 non-quadratic F, arbitrary reference
    F = NonQuad(2)
    N = 200000
    xs = rng.exponential(1.0, (N, 2, 2)) - 1.0

    def nuc_b(M):
        return np.linalg.svd(M, compute_uv=False).sum(axis=1)

    def polar_b(M):
        U, s, Vt = np.linalg.svd(M)
        return U @ Vt

    worst_z = np.inf
    for trial in range(12):
        W = rng.standard_normal((2, 2)) * 10.0 ** rng.uniform(-1, 0.5)
        Wp = rng.standard_normal((2, 2))
        eta = 10.0 ** rng.uniform(-2, 0)
        G, Gp = F.grad(W), F.grad(Wp)
        O = polar_b(G[None] + xs)
        Wn = W[None] - eta * O
        # D_F(Wn, Wp) for each sample
        t = np.einsum("ij,nij->n", F.Q, Wn)
        fWn = 0.5 * np.einsum("nij,jk,nik->n", Wn, F.q.S, Wn) - np.einsum("ij,nij->n", F.q.B, Wn) \
            + F.c * (np.logaddexp(t, -t) - np.log(2.0))
        dvals = fWn - F.f(Wp) - np.einsum("ij,nij->n", Gp, Wn - Wp[None])
        psi_diff = nuc_b(G[None] + xs) - nuc_b(Gp[None] + xs)      # sample of psi(G) - psi(G')
        slack = breg(F, W, Wp) - eta * psi_diff + 0.5 * F.Lop * eta ** 2 - dvals   # mean must be >= 0
        z = slack.mean() / (slack.std() / np.sqrt(N))
        worst_z = min(worst_z, z)
    print(f"[Monte Carlo, 2x2 non-quadratic F, entries Exp(1)-1, N={N}] min z-score of RHS-LHS of (a) over 12 "
          f"random (W, W', eta): {worst_z:.1f}  (expected > 0)")
    ok &= worst_z > -3
    print("ALL CHECKS PASSED" if ok else "SOME CHECK FAILED")
