"""Exact rational Hermite certificates for the 52 exceptional (K,d).
g(x)=(c-x)/sqrt(x); H = Hermite interpolant of g (value+derivative) at 4 nodes x_i=t_i^2, t_i rational.
g-H = g^{(8)}(xi)/8! prod (x-x_i)^2 >= 0, so E_nu g >= E_nu H = exact rational."""
import sympy as sp, mpmath as mp, pickle
from sympy import Rational as R
mp.mp.dps = 50
Ks, ds = sp.symbols('K d')
_p = pickle.load(open("mom_polys.pkl", "rb"))
P = {k: sp.sympify(v) for k, v in _p.items()}
def gauss_nodes(nu, n):
    H = mp.matrix(n, n); b = mp.matrix(n, 1)
    for i in range(n):
        for j in range(n): H[i, j] = mp.mpf(nu[i+j].p)/nu[i+j].q
        b[i] = -mp.mpf(nu[i+n].p)/nu[i+n].q
    c = mp.lu_solve(H, b)
    return sorted(mp.re(z) for z in mp.polyroots([1]+[c[n-1-i] for i in range(n)], maxsteps=500, extraprec=300))
exc = {3: range(3, 22), 4: range(4, 17), 5: range(5, 14), 6: range(6, 12), 7: range(7, 11), 8: range(8, 9)}
x = sp.Symbol('x')
worst = None; cnt = 0
for K, rng in exc.items():
    for d in rng:
        nu = [R(P[k+1].subs({Ks: K, ds: d}), K*d) for k in range(8)]
        c = R(2*K*K+2*d*d+3*K*d+3*K+3*d-1, 2*(K+d+1))
        ts = [R(int(mp.nint(mp.sqrt(z)*16)), 16) for z in gauss_nodes(nu, 4)]
        assert len(set(ts)) == 4 and min(ts) > 0
        rows, rhs = [], []
        for t in ts:
            xi = t*t
            rows.append([xi**j for j in range(8)]); rhs.append((c-xi)/t)
            rows.append([j*xi**(j-1) if j else 0 for j in range(8)]); rhs.append(-c/(2*t**3)-1/(2*t))
        coef = sp.Matrix(rows).LUsolve(sp.Matrix(rhs))
        val = sum(coef[j]*nu[j] for j in range(8))
        assert val > 0, (K, d, val)
        cnt += 1
        if worst is None or val < worst[0]: worst = (val, K, d)
        print(K, d, "nodes t_i:", ts, "exact lower bound E_nu H = %.6f > 0" % float(val))
print("all", cnt, "exceptional cases certified; smallest bound", float(worst[0]), "at", worst[1:])
