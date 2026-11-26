"""Validation of the first-order law  G_b = g*/b + o(1/b)  on noise laws that are NOT i.i.d.-entry.
Isotropic covariance (closed form, two constants):  g* = (a1 A + a2 B)/6,  A = E[z |z|_F^2], B = E[z z^T z].
General covariance:                                   g* = -(1/6) J_Sigma^{-1} T_Sigma[K3]   (Monte Carlo operators).
Zeros of the mean field are computed from S exactly sampled batch means (common random numbers).
"""
import numpy as np, json, sys, time, functools
print = functools.partial(print, flush=True)
from lib import *
K, d = 2, 3
rng = np.random.default_rng(11)
q, m1 = q_const(K, d, 4_000_000)
a1, a2 = a_consts(K, d, q)
print(f"K={K} d={d}: q={q:.4f} a1={a1:.4f} a2={a2:.4f} a1+a2={a1+a2:.4f}")

def gam(shape, size, sign=1.0):          # centred, unit variance, skewness sign*2/sqrt(shape)
    return sign * (rng.gamma(shape, 1.0, size) - shape) / np.sqrt(shape)

# ---- per-sample noise laws (return (n,K,d) samples) ------------------------------------------------
def law_iid(n):                           # L1: i.i.d. Exp-1 entries (the paper's case), kappa3 = 2
    return gam(1.0, (n, K, d))
def law_rank1(n):                         # L2: rank-one z = delta x^T, i.i.d. Exp-1 coordinates
    return gam(1.0, (n, K, 1)) * gam(1.0, (n, 1, d))
RHO = 0.6; KSH = (2 / (RHO ** 3 * 2 * np.sqrt(2) / (1 - RHO ** 2) ** 1.5)) ** 2
def law_coskew(n):                        # L3: Gaussian entries, z_11 co-skewed with z_12 but with zero own skewness
    z = rng.standard_normal((n, K, d))
    z[:, 0, 0] = RHO * (z[:, 0, 1] ** 2 - 1) / np.sqrt(2) + np.sqrt(1 - RHO ** 2) * gam(KSH, n, -1.0)
    return z
Q = np.linalg.qr(np.random.default_rng(5).standard_normal((d, K)))[0].T      # K x d, Q Q^T = I
RP = 0.9; Mfac = RP * Q / np.sqrt(K)
Cfill = np.linalg.cholesky(np.eye(K * d) - np.outer(Mfac.ravel(), Mfac.ravel()))
def law_factor(n):                        # L4: one skewed factor along a full-rank partial isometry + Gaussian filler
    return gam(1.0, (n, 1, 1)) * Mfac[None] + (rng.standard_normal((n, K * d)) @ Cfill.T).reshape(n, K, d)
Ld = np.array([[1.0, 0.0], [0.6, 0.8]]) * np.array([[1.5], [0.7]]); Lx = np.diag([2.0, 1.0, 0.5]) @ np.array([[1, 0, 0], [0.5, 1, 0], [-0.3, 0.4, 1.0]])
def law_kron(n):                          # L5: rank-one, anisotropic Kronecker covariance (general formula)
    de = gam(1.0, (n, K)) @ Ld.T; x = gam(2.0, (n, d)) @ Lx.T
    return de[:, :, None] * x[:, None, :]
LAWS = dict(iid=law_iid, rank1=law_rank1, coskew=law_coskew, factor=law_factor, kron=law_kron)

def batch_means(law, b, S, chunk=20000):
    out = np.empty((S, K, d))
    for s in range(0, S, chunk):
        m = min(chunk, S - s)
        out[s:s + m] = law(m * b).reshape(m, b, K, d).mean(1)
    return out

res = {}
for name in sys.argv[1:] or LAWS:
    law = LAWS[name]
    samp = law(4_000_000)
    Sig = np.cov(samp.reshape(len(samp), -1).T); K3 = third_cumulant(samp)
    iso = np.linalg.norm(Sig - np.eye(K * d)) < 0.02
    A, B = contractions(K3, K, d)
    if iso:
        gstar = (a1 * A + a2 * B) / 6
    gen, J, T = general_prediction(Sig, K3, K, d)
    print(f"\n== law {name}: isotropic={iso}; |A|={np.linalg.norm(A):.3f} |B|={np.linalg.norm(B):.3f}")
    if iso:
        print(f"   closed form g* = {np.round(gstar.ravel(),4)}\n   general MC  g* = {np.round(gen.ravel(),4)}")
    else:
        gstar = gen; print(f"   general MC g* = {np.round(gen.ravel(),4)}")
    res[name] = dict(gstar=gstar.tolist(), rows=[])
    import os
    BS = eval(os.environ.get('BS', '[(4, 3_000_000), (16, 3_000_000), (64, 2_000_000)]'))
    for b, S in BS:
        t0 = time.time()
        xi = batch_means(law, b, S)
        G, r = field_zero(xi, gstar / b)
        se = field_zero_se(xi, G) / (np.sqrt(b) * m1 / (K * d) / np.sqrt(np.trace(Sig) / (K * d)))  # ~ J^{-1} se, isotropic proxy
        err = np.linalg.norm(b * G - gstar)
        cos = (G * gstar).sum() / (np.linalg.norm(G) * np.linalg.norm(gstar))
        sm = np.sign(np.median(xi.reshape(S, -1), axis=0)) * 0  # placeholder
        med = -b * np.median(xi.reshape(S, -1), axis=0)          # sign descent: b * (minus coordinatewise median)
        print(f"   b={b:3d}: b*G = {np.round((b*G).ravel(),4)}  |b G - g*|_F/|g*|_F = {err/np.linalg.norm(gstar):.3f}  cos = {cos:.4f}"
              f"  (MC se of b*G ~ {b*se:.3f}; resid {r:.0e}; {time.time()-t0:.0f}s)")
        print(f"          sign descent, b * limit gradient per coordinate = {np.round(med,3)}")
        res[name]['rows'].append(dict(b=b, bG=(b * G).tolist(), relerr=err / np.linalg.norm(gstar), cos=cos, se=b * se, sign=med.tolist()))
    json.dump(res, open(f'../../../results/bias_law/checks/c03_general_law_{"_".join(sys.argv[1:]) or "all"}{os.environ.get("TAG","")}.json', 'w'), indent=1)
