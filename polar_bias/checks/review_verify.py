"""Referee scratch checks for polar_bias.tex (independent of c01..c07)."""
import itertools, json, sys
import numpy as np
import sympy as sp
import mpmath as mp

print("== A. symbolic 2x2 system")
K, d, q, m1 = sp.symbols('K d q m1', positive=True)
al, be = sp.symbols('alpha beta')
N = K * d
eq1 = sp.Eq(N / 3 * ((N + 2) * al + (K + d + 1) * be), -m1)
eq2 = sp.Eq(N / 3 * ((K + d + 1) * al + sp.Rational(1, 2) * (N + K + d + 3) * be), (K + d - q) * m1 - (K + d + 1) * m1)
sol = sp.solve([eq1, eq2], [al, be], dict=True)[0]
c = m1 / N
a1 = sp.simplify(-sol[al] / c); a2 = sp.simplify(-sol[be] / c)
D = (K - 1) * (d - 1) * (K + 2) * (d + 2)
a1p = 3 * ((K - 1) * (d - 1) - 2 * (K + d + 1) * q) / D
a2p = 6 * ((K * d + 2) * q + (K - 1) * (d - 1)) / D
print(" a1 matches:", sp.simplify(a1 - a1p) == 0, " a2 matches:", sp.simplify(a2 - a2p) == 0)
det = sp.factor((N / 3) ** 2 * ((N + 2) * (N + K + d + 3) / 2 - (K + d + 1) ** 2))
print(" det:", det, " claimed-diff:", sp.simplify(det - N ** 2 / 18 * D))
print(" a1+a2 diff:", sp.simplify(a1p + a2p - 3 * (3 + 2 * q) / ((K + 2) * (d + 2))))
print(" K a1+a2 diff (K<=d):", sp.simplify(K * a1p + a2p - 3 * (d - 1 - 2 * q) / ((d - 1) * (d + 2))))

print("== B. brute force pairing tensors / orbit sums")
def pairing(n, p):
    # p in {1,2,3}: output index (slot 0) paired with input slot p; other two paired
    T = np.zeros((n,) * 4)
    others = [s for s in (1, 2, 3) if s != p]
    for i in range(n):
        for k in range(n):
            idx = [0] * 4
            idx[0] = i; idx[p] = i; idx[others[0]] = k; idx[others[1]] = k
            T[tuple(idx)] = 1
    return T
def prod(Kn, dn, p, pp):
    R = pairing(Kn, p); C = pairing(dn, pp)
    # tensor on E^{4}: index (i,j),(a,b),(c,e),(f,g)
    return np.einsum('iacf,jbeg->ijabcefg', R, C)
for Kn, dn in [(2, 2), (2, 3), (3, 3), (3, 4), (1, 3)]:
    Nn = Kn * dn
    P = {(p, pp): prod(Kn, dn, p, pp) for p in (1, 2, 3) for pp in (1, 2, 3)}
    Abar = sum(P[(p, p)] for p in (1, 2, 3)) / 3
    Bbar = sum(P[(p, pp)] for p in (1, 2, 3) for pp in (1, 2, 3) if p != pp) / 6
    ip = lambda X, Y: float((X * Y).sum())
    A1 = P[(1, 1)]; B1 = P[(1, 2)]
    vals = (ip(A1, Abar), ip(A1, Bbar), ip(B1, Abar), ip(B1, Bbar))
    cl = (Nn * (Nn + 2) / 3, Nn * (Kn + dn + 1) / 3, Nn * (Kn + dn + 1) / 3, Nn * (Nn + Kn + dn + 3) / 6)
    # index formulas of the paper applied to zeta^{x3}
    rng = np.random.default_rng(0); z = rng.standard_normal((Kn, dn))
    K3 = np.einsum('ab,ce,fg->abcefg', z, z, z)
    Aout = np.einsum('ijabcefg,abcefg->ij', Abar, K3); Bout = np.einsum('ijabcefg,abcefg->ij', Bbar, K3)
    # paper's index formula for B: sum_{k,l} K3^{(kj),(kl),(il)}
    Bidx = np.einsum('kjklil->ij', K3)
    print(f" {Kn}x{dn}: counts {vals} claimed {cl} ok={np.allclose(vals, cl)}; "
          f"A=z|z|^2 {np.allclose(Aout, z * (z ** 2).sum())}; B=zz^Tz {np.allclose(Bout, z @ z.T @ z)}; index formula {np.allclose(Bidx, z @ z.T @ z)}")

print("== C. divergence of f(Z)=(ZZ^T)^{1/2} Z by finite differences")
rng = np.random.default_rng(1)
def f(Z):
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    return (U * s ** 2) @ Vt
for Kn, dn in [(2, 2), (2, 4), (3, 5), (4, 3)]:
    Z = rng.standard_normal((Kn, dn)); h = 1e-6; div = 0.0
    for i in range(Kn):
        for j in range(dn):
            E = np.zeros((Kn, dn)); E[i, j] = h
            div += (f(Z + E)[i, j] - f(Z - E)[i, j]) / (2 * h)
    s = np.linalg.svd(Z, compute_uv=False)
    form = (Kn + dn) * s.sum() - sum(s[i] * s[j] / (s[i] + s[j]) for i in range(len(s)) for j in range(len(s)) if i != j)
    print(f" {Kn}x{dn}: FD div {div:.6f}  formula {form:.6f}")

print("== D. Marchenko-Pastur fractional moments vs 2F1, kappa(gamma)")
def mu_num(s, g):
    lo, hi = (1 - mp.sqrt(g)) ** 2, (1 + mp.sqrt(g)) ** 2
    return mp.quad(lambda x: x ** s * mp.sqrt((hi - x) * (x - lo)) / (2 * mp.pi * g * x), [lo, hi])
def kap(g):
    a = mp.hyp2f1(-0.5, 0.5, 2, g); b3 = mp.hyp2f1(-1.5, -0.5, 2, g)
    return ((1 + g) * a - b3) / (g * a)
for g in [0.05, 0.25, 0.5, 1.0]:
    print(f" gamma={g}: mu_1/2 num {float(mu_num(0.5, g)):.8f} 2F1 {float(mp.hyp2f1(-0.5, 0.5, 2, g)):.8f};"
          f" mu_3/2 num {float(mu_num(1.5, g)):.8f} 2F1 {float(mp.hyp2f1(-1.5, -0.5, 2, g)):.8f}; kappa {float(kap(g)):.6f}"
          f" 6kappa {float(6 * kap(g)):.4f}  1-2(1+g)kappa {float(1 - 2 * (1 + g) * kap(g)):.5f}")
gs = np.linspace(1e-3, 1, 400); ks = np.array([float(kap(g)) for g in gs])
print(" kappa monotone decreasing on grid:", bool(np.all(np.diff(ks) < 0)), " range", ks.min(), ks.max(),
      " max of 1-2(1+g)kappa:", float(np.max(1 - 2 * (1 + gs) * ks)))

print("== E. Monte Carlo: T_I[D^{x3}] vs -c(a1|D|^2 D + a2 D D^T D), and q, m3 identity")
tab = {(r['K'], r['d']): r for r in json.load(open('../results/c02_constants.json'))}
def polar2(Z):  # K=2 rows, batch
    S = np.einsum('nij,nkj->nik', Z, Z)
    det = S[:, 0, 0] * S[:, 1, 1] - S[:, 0, 1] ** 2; tr = S[:, 0, 0] + S[:, 1, 1]
    sd = np.sqrt(det); t = np.sqrt(tr + 2 * sd)
    # sqrt(S) = (S + sd I)/t ; inverse = adj/det_sqrt, det(sqrt S)=sd
    R = S.copy(); R[:, 0, 0] += sd; R[:, 1, 1] += sd; R /= t[:, None, None]
    inv = np.empty_like(R)
    inv[:, 0, 0] = R[:, 1, 1]; inv[:, 1, 1] = R[:, 0, 0]; inv[:, 0, 1] = -R[:, 0, 1]; inv[:, 1, 0] = -R[:, 1, 0]
    inv /= sd[:, None, None]
    return np.einsum('nik,nkj->nij', inv, Z), sd, t
rng = np.random.default_rng(7)
for (Kn, dn), Dm in [((2, 2), np.array([[1.0, 0], [0, 2.0]])), ((2, 3), np.array([[1.0, 0, 0], [0, 2.0, 0]])),
                     ((2, 3), np.array([[1.0, 0, 0], [0, 0.0, 0]]))]:
    acc = np.zeros((Kn, dn)); acc2 = np.zeros((Kn, dn)); n = 0; qs = 0.0; m1s = 0.0; m3s = 0.0
    for _ in range(40):
        Z = rng.standard_normal((500000, Kn, dn))
        for sgn in (1,):
            O, sd, t = polar2(Z)
            zd = np.einsum('nij,ij->n', Z, Dm); w = zd ** 3 - 3 * zd * (Dm ** 2).sum()
            # antithetic: polar odd, w odd -> product even; nothing gained
            x = O * w[:, None, None]
            acc += x.sum(0); acc2 += (x ** 2).sum(0); n += len(Z)
            # singular values: s1+s2 = t, s1 s2 = sd
            m1s += t.sum(); qs += (2 * sd / t).sum(); m3s += (t ** 3 - 3 * sd * t).sum()
    T = acc / n; se = np.sqrt(acc2 / n - T ** 2) / np.sqrt(n)
    m1e = m1s / n; qe = qs / m1s; m3e = m3s / n
    r = tab[(Kn, dn)]; cc = r['m1'] / (Kn * dn)
    pred = -cc * (r['a1'] * (Dm ** 2).sum() * Dm + r['a2'] * Dm @ Dm.T @ Dm)
    print(f" {Kn}x{dn} D={Dm.tolist()}: q MC {qe:.4f} (json {r['q']:.4f}); m3 {m3e:.4f} vs (K+d-q)m1 {(Kn + dn - qe) * m1e:.4f}")
    print("   T MC   ", np.round(T, 4).tolist(), " se ", np.round(se, 4).max())
    print("   T pred ", np.round(pred, 4).tolist())

print("== F. cumulant ratios")
b = sp.symbols('beta', positive=True)
r2 = (1 - b) ** 2 / (1 - b ** 2); r3 = (1 - b) ** 3 / (1 - b ** 3)
print(" EMA rho3/rho2 - (1-b^2)/(1+b+b^2):", sp.simplify(r3 / r2 - (1 - b ** 2) / (1 + b + b ** 2)))
n_, bb = sp.symbols('n b', positive=True)
kap3 = (n_ - bb) * (n_ - 2 * bb) / (bb ** 2 * (n_ - 1) * (n_ - 2)); cv = (n_ - bb) / (bb * (n_ - 1))
print(" WOR factor kappa3/cov:", sp.simplify(kap3 / cv))
# brute-force finite-population third cumulant factor
import itertools as it
v = np.array([1.0, -2.0, 0.5, 3.0, -1.0, -1.5, 0.7]); v -= v.mean(); nn = len(v)
for bsz in (2, 3, 5):
    means = np.array([v[list(S)].mean() for S in it.combinations(range(nn), bsz)])
    print(f" n={nn} b={bsz}: var ratio {means.var() / (v ** 2).mean():.6f} claimed {(nn - bsz) / (bsz * (nn - 1)):.6f};"
          f" k3 ratio {(means ** 3).mean() / (v ** 3).mean():.6f} claimed {(nn - bsz) * (nn - 2 * bsz) / (bsz ** 2 * (nn - 1) * (nn - 2)):.6f}")
