"""Rigorous (exact rational arithmetic) confirmation of the polynomials E tr W^k, k<=8.
For d-K odd, a=(d-K-1)/2 is an integer and all Pfaffian entries are rational."""
import sympy as sp, pickle
from sympy import Rational as R, factorial, binomial
from functools import lru_cache
@lru_cache(None)
def J(A, B):   # integers A,B>=1
    inc = sum(binomial(B-1, j)*(-1)**j*R(1, 2**(A+j))/(A+j) for j in range(B))
    return 2**(A+B)*factorial(A+B-1)*inc
def Etr_all(K, d, kmax):
    a = (d-K-1)//2; assert (d-K) % 2 == 1
    n = K if K % 2 == 0 else K+1
    I = lambda al, be: J(al+a+1, be+a+1)-J(be+a+1, al+a+1)
    m1 = lambda al: 2**(al+a+1)*factorial(al+a)
    def mat(s):
        M = sp.zeros(n, n)
        for i in range(K):
            for j in range(i+1, K):
                v = I(i, j) if s is None else I(i+s, j)+I(i, j+s)
                M[i, j] = v; M[j, i] = -v
            if n > K:
                v = m1(i) if s is None else m1(i+s)
                M[i, K] = v; M[K, i] = -v
        return M
    Minv = mat(None).inv()
    return [K]+[(Minv*mat(s)).trace()/2 for s in range(1, kmax+1)]
Ks, ds = sp.symbols('K d', positive=True)
_p = pickle.load(open("mom_polys.pkl", "rb"))
P = {k: sp.sympify(v).subs({sp.Symbol('K'): Ks, sp.Symbol('d'): ds}) for k, v in _p.items()}
pts = [(K, K+o) for K in range(1, 9) for o in (1, 3, 5, 7, 9)]
vals = {pt: Etr_all(*pt, 8) for pt in pts}
for k in range(1, 9):
    basis = [(a, b) for b in range(1, (k+1)//2+1) for a in range(0, k+2-2*b)]
    A = sp.Matrix([[sp.Integer(K+d)**a*sp.Integer(K*d)**b for (a, b) in basis] for (K, d) in pts])
    ok = all(P[k].subs({Ks: K, ds: d}) == vals[(K, d)][k] for (K, d) in pts)
    print("k", k, "basis size", len(basis), "rank on exact points", A.rank(), "polynomial matches all", len(pts), "exact values:", ok)
