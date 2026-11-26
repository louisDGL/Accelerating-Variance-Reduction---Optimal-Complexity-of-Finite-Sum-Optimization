"""Sampling without replacement from a finite population of n matrices.
(1) Exact complement duality:  xi_{n-b} = -(b/(n-b)) xi_b in law, hence Med(xi_{n-b}) = -(b/(n-b)) Med(xi_b);
    checked on an enumerated population (all C(n,b) batches, K=d=1 and K=2,d=2 with small n).
(2) First-order law with finite-population cumulants:  G_b ~ g* (n-2b)/(b(n-2)): sign change at b = n/2.
"""
import numpy as np, itertools, functools
from lib import *
print = functools.partial(print, flush=True)
rng = np.random.default_rng(3)

# (1) exact duality, enumeration
K, d, n = 2, 2, 12
pop = rng.gamma(1.0, 1.0, (n, K, d)) - 1.0; pop -= pop.mean(0)
def all_means(b):
    idx = np.array(list(itertools.combinations(range(n), b)))
    return pop[idx].mean(1)
print("(1) complement duality on an enumerated population, n=12, K=d=2 (zeros of the exact mean field)")
for b in [3, 4, 5]:
    Gb, r1 = field_zero(all_means(b)); Gc, r2 = field_zero(all_means(n - b))
    print(f"   b={b}: |G_(n-b) + (b/(n-b)) G_b|_F = {np.linalg.norm(Gc + b/(n-b)*Gb):.2e}   (|G_b|={np.linalg.norm(Gb):.4f}, residuals {r1:.0e},{r2:.0e})")

# (2) first-order law
K, d, n = 2, 3, 400
pop = rng.gamma(1.0, 1.0, (n, K, d)) - 1.0; pop -= pop.mean(0)
X = pop.reshape(n, -1)
Sig = X.T @ X / n; K3 = np.einsum('na,nb,nc->abc', X, X, X) / n
gstar, J, T = general_prediction(Sig, K3, K, d)
print(f"\n(2) n={n}, K={K}, d={d}: with-replacement first-order constant g* = {np.round(gstar.ravel(),3)}")
S = 1_000_000
for b in [8, 25, 50, 100, 150, 200, 250, 300, 350]:
    xi = np.empty((S, K, d)); ch = 20000
    for s in range(0, S, ch):
        idx = np.argsort(rng.random((ch, n)), axis=1)[:, :b]
        xi[s:s + ch] = pop[idx].mean(1)
    pred = gstar * (n - 2 * b) / (b * (n - 2))
    G, r = field_zero(xi, pred)
    c = (G * gstar).sum() / (gstar * gstar).sum()          # coefficient of G on g*
    print(f"   b={b:3d}: <G,g*>/|g*|^2 = {c:+.5f}   predicted (n-2b)/(b(n-2)) = {(n-2*b)/(b*(n-2)):+.5f}   with replacement 1/b = {1/b:.5f}"
          f"   |G-pred|/|g*| = {np.linalg.norm(G-pred)/np.linalg.norm(gstar):.5f}")
