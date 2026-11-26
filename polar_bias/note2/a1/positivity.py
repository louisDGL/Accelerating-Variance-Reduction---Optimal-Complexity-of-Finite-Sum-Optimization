import sympy as sp
from cert import cert, K, d
a, b = sp.symbols('a b', nonnegative=True)
def analyze(N, theta, regions):
    C = cert(N, sp.Rational(theta))
    num, den = sp.fraction(sp.together(C))
    num = sp.expand(num); den = sp.factor(den)
    print("N", N, "theta", theta, "den:", den, " num degree", sp.Poly(num, K, d).total_degree())
    for name, sub in regions:
        p = sp.Poly(sp.expand(num.subs(sub, simultaneous=True)), a, b)
        cs = p.coeffs()
        dn = sp.expand(den.subs(sub, simultaneous=True))
        dcs = sp.Poly(dn, a, b).coeffs()
        print("  ", name, "num coeffs: n=%d min=%s all>0:%s" % (len(cs), min(cs), all(c > 0 for c in cs)),
              "| den coeffs all>0:", all(c > 0 for c in dcs))
    return num, den
if __name__ == "__main__":
    analyze(7, "6/5", [("K=5+a,d=5+a+b", {K: 5+a, d: 5+a+b}), ("K=4,d=6+b", {K: 4, d: 6+b}), ("K=3,d=8+b", {K: 3, d: 8+b}),
                       ("K=6+a,d=6+a+b", {K: 6+a, d: 6+a+b}),("K=5,d=5+b", {K: 5, d: 5+b})])
    analyze(5, "1", [("K=9+a,d=9+a+b", {K: 9+a, d: 9+a+b})] + [("K=%d,d=%d+b" % (k, dd), {K: k, d: dd+b}) for k, dd in [(3,22),(4,17),(5,14),(6,12),(7,11),(8,9)]])
