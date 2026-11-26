import sympy as sp, mpmath as mp
from cert import cert, K, d, M, cst, m
from wish import Wish
mp.mp.dps = 50
a, b = sp.symbols('a b', nonnegative=True)
C = cert(5, sp.Integer(1))
num, den = sp.fraction(sp.together(C)); num = sp.expand(num)
g = sp.gcd_list(sp.Poly(num, K, d).coeffs()); 
print("den =", sp.factor(den), "; content", g)
print("NUM(K,d) =", sp.collect(sp.expand(num/g), K))
print("NUM symmetric:", sp.expand(num - num.subs({K: d, d: K}, simultaneous=True)) == 0)
p = sp.Poly(sp.expand((num/g).subs({K: 9+a, d: 9+a+b}, simultaneous=True)), a, b)
print("K=9+a,d=9+a+b: #coeffs", len(p.coeffs()), "min coeff", min(p.coeffs()))
print(p.as_expr())
for k, dd in [(3,22),(4,17),(5,14),(6,12),(7,11),(8,9)]:
    pp = sp.Poly(sp.expand((num/g).subs({K: k, d: dd+b})), b)
    print("K=%d,d=%d+b:" % (k, dd), pp.as_expr(), "| value at d=%d:" % (dd-1), (num/g).subs({K: k, d: dd-1}))
# K=2 closed form check: q = E[c/sqrt(1+c)]/E[sqrt(1+c)], c~Beta(d-1,1)
for dd in (2, 3, 5, 9):
    f1 = mp.quad(lambda c: (dd-1)*c**(dd-2)*c/mp.sqrt(1+c), [0, 1])
    f2 = mp.quad(lambda c: (dd-1)*c**(dd-2)*mp.sqrt(1+c), [0, 1])
    print("K=2 d=%d closed form q" % dd, mp.nstr(f1/f2, 20), "Pfaffian", mp.nstr(Wish(2, dd).q(), 20))
# non-concavity of r at large p
for (k, dd) in [(2, 2), (2, 3), (3, 3), (3, 5)]:
    w = Wish(k, dd)
    ps = [mp.mpf(i)/2 for i in range(0, 121)]
    rs = [w.r(p) for p in ps]
    sd = [(ps[i], rs[i-1]+rs[i+1]-2*rs[i]) for i in range(1, len(rs)-1)]
    pos = [float(p_) for p_, v in sd if v > 0]
    print((k, dd), "r(p)-p at p=4,20,60:", [mp.nstr(w.r(p)-p, 8) for p in (4, 20, 60)], "limit K+d-1 =", k+dd-1,
          "| second difference >0 for p in", (pos[0], pos[-1]) if pos else None, "| concave on [0,4] grid:", all(v < 0 for p_, v in sd if p_ <= 3.5))
