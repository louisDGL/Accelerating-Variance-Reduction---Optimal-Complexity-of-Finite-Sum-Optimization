"""Validation of the Monte Carlo pipeline of median_r3.py (section C) against the exact DP law (section B)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from median_r3 import exact_law_lattice, law_to_points, geomed, lattice_population

def mc_hist(n, b, S, seed):
    r_ = np.random.default_rng(seed); cats, mult = lattice_population(n); C = np.array(cats)
    top = min(b, n // 3); hist = np.zeros((top + 1,) * 3)
    for st in range(0, S, 2_000_000):
        cnt = r_.multivariate_hypergeometric(mult, b, size=2_000_000, method="marginals"); Nk = cnt @ C
        np.add.at(hist, (Nk[:, 0], Nk[:, 1], Nk[:, 2]), 1.0)
    return hist / S

S = 40_000_000
for n, b in [(216, 54), (108, 27), (216, 81)]:
    P = exact_law_lattice(n, b); X, w = law_to_points(P, b); m, _, _ = geomed(X, w)
    print(f"n={n} b={b}: exact b*m = {b*m[0]:.7f}")
    for seed in (11, 22, 33, 44):
        Ph = mc_hist(n, b, S, seed)
        print(f"   seed {seed}: max|P_mc - P_exact| = {np.abs(Ph-P).max():.2e} (expected ~ {np.sqrt(P.max()/S):.1e});", end=" ")
        Xh, wh = law_to_points(Ph, b); mh, _, _ = geomed(Xh, wh)
        D = mh - Xh; d = np.linalg.norm(D, axis=1); U = D / d[:, None]
        Jh = (wh / d).sum() * np.eye(3) - np.einsum("s,sa,sb->ab", wh / d, U, U)
        cov = np.linalg.solve(Jh, np.linalg.solve(Jh, np.einsum("s,sa,sb->ab", wh, U, U)).T) / S
        print("MC b*m =", b * mh, " se", b * np.sqrt(np.diag(cov)), " z =", (b * mh - b * m) / (b * np.sqrt(np.diag(cov))))
