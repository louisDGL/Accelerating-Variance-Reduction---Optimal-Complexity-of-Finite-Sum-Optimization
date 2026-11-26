"""Independent verification of the proof of a_1<0 (q_{K,d} > (K-1)(d-1)/(2(K+d+1)))  for K,d>=3 (K=2: by hand).
(1) Wick moments (v01) agree with the polynomials used by the certificate scripts in ../a1.
(2) E_nu[T_5] recomputed from scratch; positivity of the numerator on the stated ranges."""
import pickle, sympy as sp
K, d, x = sp.symbols('K d x', positive=True)
mine = {k: sp.sympify(v, locals={'K': K, 'd': d}) for k, v in pickle.load(open('../results/wick_moments.pkl', 'rb')).items()}
theirs = pickle.load(open('../a1/mom_polys.pkl', 'rb'))
Ks, ds = sp.symbols('K d')
for k in sorted(mine):
    t = sp.sympify(theirs[k]).subs({Ks: K, ds: d})
    print('E tr W^%d: Wick enumeration == certificate polynomial:' % k, sp.expand(mine[k]-t) == 0)
m = K+d+1
cst = (2*K**2+2*d**2+3*K*d+3*K+3*d-1)/(2*m)
# check c* = K+d - (K-1)(d-1)/(2m)
print('c* identity:', sp.simplify(cst-(K+d-(K-1)*(d-1)/(2*m))) == 0)
g = (cst-x)/sp.sqrt(x)
T5 = sum(sp.diff(g, x, n).subs(x, m)*(x-m)**n/sp.factorial(n) for n in range(6))
T5 = sp.expand(T5)
E = sum(sp.Poly(T5, x).coeff_monomial(x**j)*mine[j+1]/(K*d) for j in range(6))
E = sp.simplify(E*sp.sqrt(m))          # rational function
num, den = sp.fraction(sp.together(E))
num = sp.expand(num); den = sp.factor(den)
print('sqrt(m) E_nu[T_5] = NUM/DEN,  DEN =', den)
ct = sp.Poly(num, K, d).content()
NUM = sp.expand(num/ct)
print('NUM =', sp.collect(NUM, K))
a, b = sp.symbols('a b', nonnegative=True)
def allpos(expr, vars_):
    return all(c > 0 for c in sp.Poly(sp.expand(expr), *vars_).coeffs())
sign = 1 if sp.Poly(den/ct if False else den, K, d).LC() > 0 else -1
print('all coefficients of NUM(9+a, 9+a+b) positive:', allpos(NUM.subs({K: 9+a, d: 9+a+b}, simultaneous=True), (a, b)))
for K0, d0 in [(3, 22), (4, 17), (5, 14), (6, 12), (7, 11), (8, 9)]:
    print('NUM(%d, %d+b) all coefficients positive:' % (K0, d0), allpos(NUM.subs({K: K0, d: d0+b}, simultaneous=True), (b,)))
