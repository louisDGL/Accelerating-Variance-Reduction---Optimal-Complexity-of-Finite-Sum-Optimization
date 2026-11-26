"""Fit exact polynomials E tr W^k (k<=8) in (K,d): total degree k+1 (Wick/genus expansion), from exact integer values."""
import mpmath as mp, sympy as sp, pickle, sys
from wish import Wish
mp.mp.dps = 90
KMAX = 8
pts = [(K, d) for K in range(1, 11) for d in range(K, 14)]   # K<=d (symmetric)
vals = {}
for (K, d) in pts:
    w = Wish(K, d)
    for k in range(1, KMAX+1):
        v = w.Etr(k); r = mp.nint(v)
        assert abs(v-r) < mp.mpf(10)**(-25), (K, d, k, v-r)
        vals[(K, d, k)] = int(r)
Ks, ds = sp.symbols('K d')
polys = {}
for k in range(1, KMAX+1):
    # symmetric basis: (K+d)^a (Kd)^b, a+2b<=k+1, b>=1
    basis = [(a, b) for b in range(1, (k+1)//2+1) for a in range(0, k+2-2*b)]
    fit = pts[:]  # overdetermined exact solve
    A = sp.Matrix([[sp.Integer(K+d)**a*sp.Integer(K*d)**b for (a, b) in basis] for (K, d) in fit])
    y = sp.Matrix([vals[(K, d, k)] for (K, d) in fit])
    n = len(basis)
    # choose n rows giving invertible system, then verify on all rows
    sol = (A.T*A).LUsolve(A.T*y)
    assert A*sol == y, k
    P = sp.factor(sum(c*(Ks+ds)**a*(Ks*ds)**b for c, (a, b) in zip(sol, basis)))
    polys[k] = sp.expand(P)
    print(k, len(pts)-n, "extra exact checks;", P); sys.stdout.flush()
pickle.dump({k: sp.srepr(p) for k, p in polys.items()}, open("mom_polys.pkl", "wb"))
