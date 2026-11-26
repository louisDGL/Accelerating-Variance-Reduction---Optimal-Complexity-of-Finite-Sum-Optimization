"""Independent check of E tr (Z Z^T)^k, Z a K x d real Gaussian matrix, as a polynomial in (K,d), by brute-force Wick
enumeration: sum over pairings of the 2k factors of K^{#row classes} d^{#column classes}."""
import sys, pickle, sympy as sp
from collections import Counter
K, d = sp.symbols('K d')
def moment_poly(k):
    n = 2*k
    rows = [(m//2 + (m % 2)) % k for m in range(n)]      # factor 2m: Z[i_m, j_m]; factor 2m+1: Z[i_{m+1}, j_m]
    cols = [m//2 for m in range(n)]
    cnt = Counter()
    def find(p, x):
        while p[x] != x:
            x = p[x]
        return x
    def rec(free, pr, pc, nr, nc):
        if not free:
            cnt[(nr, nc)] += 1; return
        a = free[0]
        for idx in range(1, len(free)):
            b = free[idx]
            ra, rb = find(pr, rows[a]), find(pr, rows[b])
            ca, cb = find(pc, cols[a]), find(pc, cols[b])
            pr2, nr2 = pr, nr
            if ra != rb:
                pr2 = pr.copy(); pr2[ra] = rb; nr2 = nr-1
            pc2, nc2 = pc, nc
            if ca != cb:
                pc2 = pc.copy(); pc2[ca] = cb; nc2 = nc-1
            rec(free[1:idx]+free[idx+1:], pr2, pc2, nr2, nc2)
    rec(list(range(n)), list(range(k)), list(range(k)), k, k)
    return sp.expand(sum(c*K**a*d**b for (a, b), c in cnt.items()))
kmax = int(sys.argv[1]) if len(sys.argv) > 1 else 6
mine = {k: moment_poly(k) for k in range(1, kmax+1)}
for k in (2, 3, 4):
    print(k, sp.factor(mine[k]))
try:
    theirs = pickle.load(open('../a1/mom_polys.pkl', 'rb'))
    print('type of stored object:', type(theirs), list(theirs)[:3] if hasattr(theirs, '__iter__') else '')
except Exception as e:
    print('pickle load failed', e); theirs = None
pickle.dump({k: str(v) for k, v in mine.items()}, open('../../../results/bias_law/lattice/wick_moments.pkl', 'wb'))
for k, v in mine.items():
    print(k, 'value at (K,d)=(1,d):', sp.factor(v.subs(K, 1)))
