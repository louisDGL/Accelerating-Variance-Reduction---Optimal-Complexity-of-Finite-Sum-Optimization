import sympy as sp, pickle
K, d, th = sp.symbols('K d theta', positive=True)
_p = pickle.load(open("mom_polys.pkl", "rb"))
M = {k: sp.sympify(sp.sympify(v)).subs({sp.Symbol('K'): K, sp.Symbol('d'): d}) for k, v in _p.items()}
M[0] = K
m = K+d+1
cst = (2*K**2+2*d**2+3*K*d+3*K+3*d-1)/(2*m)
def cert(N, theta=th):
    """sqrt(x0) * E_nu[T_N], T_N = degree-N Taylor polynomial of g(x)=(c-x)/sqrt(x) at x0=theta*m.
    g >= T_N on (0,inf) for odd N (g^{(N+1)}>0).  nu_j = E tr W^{j+1}/(Kd)."""
    x0 = theta*m
    tot = 0
    for k in range(N+1):
        Ey = sum(sp.binomial(k, j)*M[j+1]/(K*d)/x0**j*(-1)**(k-j) for j in range(k+1))
        tot += (cst*sp.binomial(-sp.Rational(1, 2), k) - x0*sp.binomial(sp.Rational(1, 2), k))*Ey
    return sp.together(tot)
if __name__ == "__main__":
    for N in (5, 7):
        C = cert(N)
        f = sp.lambdify((K, d, th), C, 'mpmath')
        for t in (1, 1.05, 1.1, 1.15, 1.2, 1.25):
            bad = [(a, b) for a in range(2, 61) for b in range(a, 121) if f(a, b, t) <= 0]
            byK = {}
            for a, b in bad: byK.setdefault(a, []).append(b)
            print(N, t, "fail count", len(bad), {a: (min(v), max(v), len(v)) for a, v in byK.items()})
