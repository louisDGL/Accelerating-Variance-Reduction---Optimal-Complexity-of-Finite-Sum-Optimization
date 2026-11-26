"""State-dependent noise: population least squares, per-sample gradient (W x - y) x^T, y = W0 x + eps.
The noise law depends on W (through (W-W0)(x x^T - C)).  Prediction: every zero W_b of the mean field
h_b(W) = E polar(mean_b (W x - y) x^T) near W0 satisfies  W_b - W0 = (g*/b) C^{-1} + o(1/b),
g* = -(1/6) J^{-1} T[K3] computed at W0 (noise -eps x^T), C = E x x^T.
"""
import numpy as np, functools
from scipy.optimize import root
from lib import *
print = functools.partial(print, flush=True)
rng = np.random.default_rng(21)
K, d = 2, 3
W0 = np.array([[1.0, -0.5, 0.3], [0.2, 0.8, -1.0]])
Lx = np.array([[1.0, 0, 0], [0.4, 0.9, 0], [-0.2, 0.3, 0.7]])
Le = np.array([[1.0, 0.0], [0.5, 0.8]])
def sample(n):
    u = rng.gamma(2.0, 1.0, (n, d)) / np.sqrt(2.0)            # skewed features with nonzero mean (acts as an intercept)
    x = u @ Lx.T
    eps = (rng.gamma(1.0, 1.0, (n, K)) - 1.0) @ Le.T           # skewed residuals, independent of x, mean zero
    return x, eps
x, eps = sample(4_000_000)
C = x.T @ x / len(x)
noise = -(eps[:, :, None] * x[:, None, :])
Sig = np.cov(noise.reshape(len(x), -1).T); K3 = third_cumulant(noise)
gstar, J, T = general_prediction(Sig, K3, K, d)
pred = gstar @ np.linalg.inv(C)
print("C =", np.round(C, 3).tolist()); print("g* =", np.round(gstar.ravel(), 4)); print("predicted b (W_b - W0) =", np.round(pred.ravel(), 4))
for b, S in [(16, 4_000_000), (64, 4_000_000), (256, 2_000_000)]:
    Ch = np.empty((S, d, d)); Eh = np.empty((S, K, d)); ch = max(1, 2_000_000 // b)
    for s in range(0, S, ch):
        m = min(ch, S - s); xx, ee = sample(m * b); xx = xx.reshape(m, b, d); ee = ee.reshape(m, b, K)
        Ch[s:s + m] = np.einsum('mbi,mbj->mij', xx, xx) / b; Eh[s:s + m] = np.einsum('mbk,mbj->mkj', ee, xx) / b
    def solve(sl, frozen):
        if frozen:
            f = lambda w: polar((w.reshape(K, d) @ C)[None] - Eh[sl]).mean(0).reshape(-1)
        else:
            f = lambda w: polar(np.einsum('kl,mlj->mkj', w.reshape(K, d), Ch[sl]) - Eh[sl]).mean(0).reshape(-1)
        return root(f, pred.reshape(-1) / b, method='hybr', tol=1e-13).x.reshape(K, d)
    nb = 8; blocks = [slice(i * S // nb, (i + 1) * S // nb) for i in range(nb)]
    Ds = np.array([solve(sl, False) for sl in blocks]); D0s = np.array([solve(sl, True) for sl in blocks])
    D, D0 = Ds.mean(0), D0s.mean(0); se = np.linalg.norm(Ds.std(0, ddof=1)) / np.sqrt(nb)
    print(f"   b={b:4d}: b(W_b-W0) = {np.round((b*D).ravel(),4)}  rel.err to prediction {np.linalg.norm(b*D-pred)/np.linalg.norm(pred):.3f}"
          f" (MC se, same units: {b*se/np.linalg.norm(pred):.3f})"
          f"  | frozen-noise zero rel.err {np.linalg.norm(b*D0-pred)/np.linalg.norm(pred):.3f}; state-dep vs frozen {np.linalg.norm(D-D0)/np.linalg.norm(D0):.3f}")
