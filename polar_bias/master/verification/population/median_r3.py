"""ITEM 6. Numerical test of Theorem "Finite-population bias" for the Euclidean norm on R^3.

N = 3, G = SO(3) (orbits are spheres; the orbital integral is sin(u)/u, so (eq:orbits) holds with kappa = 1/2,
gamma = kappa/(2(N-kappa)) = 1/10).

 A. g_* = -(1/6) J^{-1} D^4P(0)[K_3,.] by quadrature for a general covariance, and check of the closed form
    g_* = E[zeta |zeta|^2] / (2 sigma^2 (N+2)) of [pb] for Sigma = sigma^2 Id.
 B. Lattice population (product law on {-1,2}^3, probabilities 2/3,1/3, realised by repetition; n multiple of 108):
    EXACT law of the batch sum by dynamic programming over the 8 categories (no Monte Carlo), exact geometric median.
 C. Same population, larger n: Monte Carlo via multivariate hypergeometric counts (histogram on the lattice).
 D. Non-isotropic, non-product population with 12 distinct atoms: Monte Carlo, with standard errors.
 E. Scalar sanity check (N = 1, NOT covered by the theorem): exact medians of the lattice population.
"""
import itertools, math, sys, time
import numpy as np
from scipy.special import gammaln

np.set_printoptions(precision=5, suppress=True, linewidth=150)
rng = np.random.default_rng(7)


# ----------------------------------------------------------------------------------------------------------------
# A. the coefficient
# ----------------------------------------------------------------------------------------------------------------
def gstar_quadrature(Sigma, K3, nth=48, nph=96):
    """D^2P(0), D^4P(0)[K3,.] for P(x)=E|x+Z|, Z~N(0,Sigma), via Gaussian integration by parts
    d_{abcl} P(0) = E[ |Z| He_{abcl}(Z) ],  in polar coordinates Z = S^{1/2} r w  (r ~ chi_N exact, sphere by product rule)."""
    N = 3
    L = np.linalg.cholesky(Sigma)
    Lam = np.linalg.inv(Sigma)
    x, wx = np.polynomial.legendre.leggauss(nth)            # cos(theta)
    ph = 2 * np.pi * (np.arange(nph) + 0.5) / nph
    ct, PH = np.meshgrid(x, ph, indexing="ij")
    st = np.sqrt(1 - ct ** 2)
    W = np.stack([st * np.cos(PH), st * np.sin(PH), ct], -1).reshape(-1, 3)
    wt = (wx[:, None] * np.ones(nph)[None, :] / (2 * nph)).reshape(-1)     # sums to 1
    V = W @ L.T                                             # S^{1/2} w
    nu = np.linalg.norm(V, axis=1)                          # |Z| = r * nu
    Y = V @ Lam                                             # Sigma^{-1} Z = r * Y
    Er = lambda k: 2 ** (k / 2) * math.exp(gammaln((N + k) / 2) - gammaln(N / 2))   # E r^k, r ~ chi_N
    # Hessian: E[|Z| (Y Y' - Lam)]
    J = Er(3) * np.einsum("s,s,sa,sb->ab", wt, nu, Y, Y) - Er(1) * (wt @ nu) * Lam
    # fourth derivative contracted with K3 in three slots:
    # He4_{abcl} = Y_aY_bY_cY_l - [Lam_ab Y_cY_l + 5 perms] + [Lam_ab Lam_cl + 2 perms]
    K3Y3 = np.einsum("abc,sa,sb,sc->s", K3, Y, Y, Y)        # K3[Y,Y,Y]
    K3L = np.einsum("abc,ab->c", K3, Lam)                   # K3 contracted with Lam (K3 symmetric)
    K3LY = Y @ K3L                                          # K3[Lam, Y]
    K3YY = np.einsum("abc,sa,sb->sc", K3, Y, Y)             # K3[Y,Y,.]
    # contraction over a,b,c of the 6 pair terms: 3 with pair inside (a,b,c): K3[Lam,Y] Y_l ; 3 with pair (a,l): Lam K3[.,Y,Y]
    t4 = Er(5) * np.einsum("s,s,s,sl->l", wt, nu, K3Y3, Y)
    t2 = Er(3) * (3 * np.einsum("s,s,s,sl->l", wt, nu, K3LY, Y) + 3 * np.einsum("s,s,sc,cl->l", wt, nu, K3YY, Lam))
    t0 = Er(1) * (wt @ nu) * 3 * (Lam @ K3L)
    T = t4 - t2 + t0
    return -np.linalg.solve(J, T) / 6, J, T


def moments(z, w=None):
    w = np.full(len(z), 1 / len(z)) if w is None else w
    Sig = np.einsum("i,ia,ib->ab", w, z, z)
    K3 = np.einsum("i,ia,ib,ic->abc", w, z, z, z)
    return Sig, K3


# ----------------------------------------------------------------------------------------------------------------
# geometric median of a weighted point cloud (Newton with line search; exact optimality check returned)
# ----------------------------------------------------------------------------------------------------------------
def geomed(X, w, x0=None, iters=60):
    m = (w @ X) if x0 is None else x0.copy()
    m = m + 1e-7                                            # stay off the lattice
    f = lambda mm: w @ np.linalg.norm(X - mm, axis=1)
    fm = f(m)
    for _ in range(iters):
        D = m - X
        d = np.linalg.norm(D, axis=1)
        d = np.maximum(d, 1e-300)
        U = D / d[:, None]
        g = w @ U
        Hs = (w / d).sum() * np.eye(X.shape[1]) - np.einsum("s,sa,sb->ab", w / d, U, U)
        step = np.linalg.solve(Hs, g)
        lam = 1.0
        while True:
            m_new = m - lam * step
            f_new = f(m_new)
            if f_new <= fm or lam < 1e-12:
                break
            lam /= 2
        if np.linalg.norm(m_new - m) < 1e-15:
            m = m_new; break
        m, fm = m_new, f_new
    D = m - X; d = np.linalg.norm(D, axis=1)
    g = w @ (D / np.maximum(d, 1e-300)[:, None])
    return m, np.linalg.norm(g), Hs


# ----------------------------------------------------------------------------------------------------------------
# B. exact law for the lattice population
# ----------------------------------------------------------------------------------------------------------------
def lattice_population(n):
    assert n % 27 == 0
    cats = list(itertools.product([0, 1], repeat=3))        # indicator "coordinate equals 2"
    mult = [n * (1 ** sum(c)) * (2 ** (3 - sum(c))) // 27 for c in cats]
    assert sum(mult) == n
    return cats, mult


def exact_law_lattice(n, b):
    """joint law of (N1,N2,N3) = numbers of sampled points whose k-th coordinate equals 2, for a uniform b-subset.
    DP over the 8 categories on the array [y, n1, n2, n3], y = number of points taken so far."""
    cats, mult = lattice_population(n)
    Mk = n // 3
    top = min(b, Mk)
    A = np.zeros((b + 1, top + 1, top + 1, top + 1))
    A[0, 0, 0, 0] = 1.0
    logC = lambda M, j: gammaln(M + 1) - gammaln(j + 1) - gammaln(M - j + 1)
    scale = 0.0
    for c, M in zip(cats, mult):
        Bn = np.zeros_like(A)
        for j in range(0, min(M, b) + 1):
            sh = [j * ck for ck in c]
            if max(sh) > top: break
            src = A[: b + 1 - j, : top + 1 - sh[0], : top + 1 - sh[1], : top + 1 - sh[2]]
            Bn[j:, sh[0]:, sh[1]:, sh[2]:] += math.exp(logC(M, j) - logC(M, min(M, b) // 2)) * src
        scale += logC(M, min(M, b) // 2)
        A = Bn
        s_ = A.sum(); A /= s_; scale += math.log(s_)
    P = A[b]
    tot = P.sum()
    # check normalisation against C(n,b)
    logtot = math.log(tot) + scale
    assert abs(logtot - logC(n, b)) < 1e-8 * abs(logC(n, b)), (logtot, logC(n, b))
    return P / tot


def law_to_points(P, b):
    idx = np.argwhere(P > 0)
    w = P[P > 0]
    X = (-b + 3.0 * idx) / b                                # batch mean, coordinates -1 + 3 N_k/b
    return X, w


def run_lattice(n, b, P, gst, label):
    X, w = law_to_points(P, b)
    mean = w @ X
    cov = np.einsum("s,sa,sb->ab", w, X, X)
    m, gn, _ = geomed(X, w)
    pred = -((n - 2 * b) / (n - 2)) * gst
    v = b * (n - b) / n
    print(f"  n={n:5d} b={b:5d} p={b / n:.3f} v={v:7.1f} [{label}]  b*m = {b * m}   predicted {pred}   "
          f"diff(max) = {np.abs(b * m - pred).max():.5f}   |mean|={np.abs(mean).max():.1e}  |grad|={gn:.1e}")
    return b * m, pred


if __name__ == "__main__":
    t_start = time.time()
    print("=" * 110 + "\nA. coefficient g_* for the Euclidean norm on R^3\n" + "=" * 110)
    # isotropic product law {-1,2}: Sigma = 2 Id, K3_{aaa} = 2
    Sig = 2 * np.eye(3); K3 = np.zeros((3, 3, 3))
    for a in range(3): K3[a, a, a] = 2.0
    g, J, T = gstar_quadrature(Sig, K3)
    Ez = np.einsum("abb->a", K3)
    print("lattice product law: g_* (quadrature) =", g, "  closed form E[z|z|^2]/(2 s^2 (N+2)) =", Ez / (2 * 2 * 5))
    print("   J = (1-1/N) E[1/|Z|] Id: quadrature", J[0, 0], " exact", (2 / 3) * math.sqrt(2 / math.pi) / math.sqrt(2))
    for nth in (12, 24, 48):
        print(f"   quadrature convergence nth={nth}:", gstar_quadrature(Sig, K3, nth, 2 * nth)[0])
    # random isotropic test with a full K3
    zt = rng.standard_normal((40, 3)) ** 3; zt -= zt.mean(0)
    _, K3t = moments(zt)
    for s2 in (0.5, 3.0):
        gq = gstar_quadrature(s2 * np.eye(3), K3t)[0]
        print(f"   Sigma={s2} Id, generic K3: quadrature {gq}  closed form {np.einsum('abb->a', K3t) / (2 * s2 * 5)}")
    # independent finite-difference check of the quadrature for a NON-isotropic Sigma: Monte Carlo free, uses
    # P(x) = E|x+Z| computed by the same polar rule but differentiated numerically (different formula path)
    Sg = np.array([[2.0, 0.6, -0.3], [0.6, 1.0, 0.2], [-0.3, 0.2, 0.5]])
    gq, Jq, Tq = gstar_quadrature(Sg, K3t, 64, 128)
    # check J via the smooth formula J = E[(I-uu')/|Z|]
    x_, wx_ = np.polynomial.legendre.leggauss(64); ph_ = 2 * np.pi * (np.arange(128) + 0.5) / 128
    ct_, PH_ = np.meshgrid(x_, ph_, indexing="ij"); st_ = np.sqrt(1 - ct_ ** 2)
    W_ = np.stack([st_ * np.cos(PH_), st_ * np.sin(PH_), ct_], -1).reshape(-1, 3); wt_ = (wx_[:, None] * np.ones(128)[None] / 256).reshape(-1)
    V_ = W_ @ np.linalg.cholesky(Sg).T; nv = np.linalg.norm(V_, axis=1); Uv = V_ / nv[:, None]
    Einv_r = math.sqrt(2 / math.pi)                          # E[1/r], r~chi_3
    J2 = Einv_r * (np.einsum("s,s->", wt_, 1 / nv) * np.eye(3) - np.einsum("s,s,sa,sb->ab", wt_, 1 / nv, Uv, Uv))
    print("   non-isotropic Sigma: max|J(Hermite form) - J(E[(I-uu')/|Z|])| =", np.abs(Jq - J2).max())
    # check T via Monte Carlo-free 3rd-order form: D^4P(0)[K3,.]_l = E[ d_l|Z| * He3[K3](Z) ], d_l|Z| = Z_l/|Z|
    Y_ = V_ @ np.linalg.inv(Sg); Lam_ = np.linalg.inv(Sg)
    Er = lambda k: 2 ** (k / 2) * math.exp(gammaln((3 + k) / 2) - gammaln(1.5))
    He3 = Er(3) * np.einsum("abc,sa,sb,sc->s", K3t, Y_, Y_, Y_)[:, None] * Uv - 3 * Er(1) * (Y_ @ np.einsum("abc,ab->c", K3t, Lam_))[:, None] * Uv
    T2 = np.einsum("s,sl->l", wt_, He3)
    print("   non-isotropic Sigma: D^4P(0)[K3,.] (4th Hermite form)", Tq, " (gradient x 3rd Hermite form)", T2)

    print("\n" + "=" * 110 + "\nB. lattice population {-1,2}^3 (2/3,1/3), EXACT law and exact geometric median;  g_* = 0.1*(1,1,1)\n" + "=" * 110)
    gst = Ez / 20
    results = []
    for n, bs in [(108, [9, 27, 54, 81, 99]), (216, [18, 54, 81, 108, 162]), (432, [36, 108])]:
        for b in bs:
            t0 = time.time()
            P = exact_law_lattice(n, b)
            bm, pred = run_lattice(n, b, P, gst, f"exact, {time.time() - t0:.0f}s")
            results.append((n, b, bm[0], pred[0]))
            sys.stdout.flush()
    print("\n  summary (first coordinate):   n     b     p      v     b*m      -(n-2b)/(n-2) g*   -(1-2p) g*    diff    diff*v   diff*sqrt(v)")
    for n, b, bm, pr in results:
        v = b * (n - b) / n
        print(f"                             {n:5d} {b:5d} {b / n:.3f} {v:7.1f} {bm:9.5f} {pr:12.5f} {-(1 - 2 * b / n) * 0.1:14.5f} {bm - pr:10.5f} {(bm - pr) * v:8.3f} {(bm - pr) * math.sqrt(v):8.3f}")
    # duality check on the exact numbers
    d = {(n, b): bm for n, b, bm, _ in results}
    for (n, b) in [(108, 27), (216, 54), (108, 9)]:
        if (n, n - b) in d:
            print(f"  duality check n={n}: (n-b) m_(n-b) = {d[(n, n - b)]:.6f}  vs  -b m_b = {-d[(n, b)]:.6f}")

    print("\n" + "=" * 110 + "\nC. same lattice population, larger n: Monte Carlo (multivariate hypergeometric counts, histogram on the lattice)\n" + "=" * 110)
    def mc_lattice(n, b, S, seed):
        r_ = np.random.default_rng(seed)
        cats, mult = lattice_population(n)
        C = np.array(cats)
        Mk = n // 3; top = min(b, Mk)
        hist = np.zeros((top + 1,) * 3)
        chunk = 2_000_000
        done = 0
        while done < S:
            k = min(chunk, S - done)
            cnt = r_.multivariate_hypergeometric(mult, b, size=k, method="marginals")
            Nk = cnt @ C
            np.add.at(hist, (Nk[:, 0], Nk[:, 1], Nk[:, 2]), 1.0)
            done += k
        return hist / S
    def mc_se(X, w, m, S, b):
        D = m - X; d = np.linalg.norm(D, axis=1); U = D / d[:, None]
        Jh = (w / d).sum() * np.eye(3) - np.einsum("s,sa,sb->ab", w / d, U, U)
        Cu = np.einsum("s,sa,sb->ab", w, U, U)
        cov = np.linalg.solve(Jh, np.linalg.solve(Jh, Cu).T) / S
        return b * np.sqrt(np.diag(cov))
    S = 40_000_000
    # validation of the MC pipeline against the exact DP
    for n, b in [(216, 54)]:
        P = mc_lattice(n, b, S, 1); X, w = law_to_points(P, b); m, gn, _ = geomed(X, w)
        print(f"  validation n={n} b={b}: MC b*m = {b * m} +- {mc_se(X, w, m, S, b)}   (exact above: {d[(n, b)]:.5f})")
    for n, bs in [(864, [216, 432, 648]), (1728, [432, 1296]), (3456, [864]), (1728, [144])]:
        for b in bs:
            t0 = time.time()
            P = mc_lattice(n, b, S, 100 + n + b); X, w = law_to_points(P, b); m, gn, _ = geomed(X, w)
            se = mc_se(X, w, m, S, b)
            pred = -((n - 2 * b) / (n - 2)) * gst
            avg = (b * m).mean(); avg_se = np.linalg.norm(se) / 3      # coordinates exchangeable; conservative-ish
            print(f"  n={n:5d} b={b:5d} p={b / n:.3f} v={b * (n - b) / n:7.1f}: b*m = {b * m} (se {se[0]:.4f}/coord)  mean over coords {avg:.5f}   predicted {pred[0]:.5f}"
                  f"   diff = {avg - pred[0]:+.5f}  [{time.time() - t0:.0f}s]")
            sys.stdout.flush()

    print("\n" + "=" * 110 + "\nD. non-isotropic 12-atom population (not a product, not symmetric), Monte Carlo with standard errors\n" + "=" * 110)
    r2 = np.random.default_rng(12345)
    q = r2.exponential(1.0, size=(12, 3)) @ np.array([[1.0, 0.5, 0.0], [0.0, 1.0, -0.4], [0.3, 0.0, 0.7]])
    q[:4] *= 1.8
    q -= q.mean(0)
    Sg, K3g = moments(q)
    gq, Jq, Tq = gstar_quadrature(Sg, K3g, 64, 128)
    print("  atoms:\n", q)
    print("  Sigma_n =\n", Sg, "\n  eigenvalues", np.linalg.eigvalsh(Sg))
    print("  g_*^(n) (quadrature, 64x128) =", gq, "   (32x64):", gstar_quadrature(Sg, K3g, 32, 64)[0])
    S = 12_000_000
    for n, bs in [(120, [30, 60, 90]), (480, [120, 360, 48]), (1920, [480, 1440])]:
        for b in bs:
            t0 = time.time()
            rr = np.random.default_rng(n * 7 + b)
            sums = np.empty((S, 3))
            for st in range(0, S, 2_000_000):
                cnt = rr.multivariate_hypergeometric([n // 12] * 12, b, size=2_000_000, method="marginals")
                sums[st: st + 2_000_000] = cnt @ q
            X = sums / b; w = np.full(S, 1.0 / S)
            m, gn, _ = geomed(X, w, iters=25)
            se = mc_se(X, w, m, S, b)
            pred = -((n - 2 * b) / (n - 2)) * gq
            zsc = (b * m - pred) / se
            print(f"  n={n:5d} b={b:5d} p={b / n:.3f} v={b * (n - b) / n:6.1f}: b*m = {b * m} +- {se}\n"
                  f"{'':40s} predicted {pred}   diff/se = {zsc}   |grad|={gn:.1e} [{time.time() - t0:.0f}s]")
            sys.stdout.flush()

    print("\n" + "=" * 110 + "\nE. scalar sanity check (N=1; NOT covered by the theorem): lattice population {-1,2}, exact medians\n" + "=" * 110)
    from scipy.stats import hypergeom
    print("  formal coefficient: g_* = kappa_3/(6 sigma^2) = 2/12 = 0.16667; the batch sum lives on -b+3Z (spacing 3 >> g_*).")
    for n, b in [(108, 27), (432, 108), (1728, 432), (1729 * 3, 1296), (1728, 1296), (6912, 1728), (6912, 1729), (6912, 1730)]:
        K = n // 3
        k = np.arange(0, min(b, K) + 1)
        cdf = hypergeom.cdf(k, n, K, b)
        kmed = k[np.searchsorted(cdf, 0.5)]
        # smooth "interpolated" median of the count, to see the coefficient through the lattice
        i = np.searchsorted(cdf, 0.5); pmf = hypergeom.pmf(k[i], n, K, b)
        kint = k[i] - 0.5 + (0.5 - (cdf[i] - pmf)) / pmf
        print(f"  n={n:5d} b={b:5d}: b*median = {-b + 3 * kmed:+d} (true median: O(1) lattice value, no convergence)   "
              f"mid-point-interpolated b*m = {-b + 3 * kint:+.4f}   formal -(n-2b)/(n-2) g_* = {-(n - 2 * b) / (n - 2) / 6:+.4f}")
    print(f"\n total time {time.time() - t_start:.0f}s")
