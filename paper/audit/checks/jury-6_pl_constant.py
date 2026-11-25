"""Controle numerique du constat jury-6 : pour un modele softmax ridge, la constante de Polyak-Lojasiewicz
en norme duale, inf_W ||grad F(W)||_{N*}^2 / (2 (F(W) - F_*)), vaut lambda / c_N^2 (c_N = max N / ||.||_F),
comme la constante de forte convexite mu_N du Theoreme 1. Trois normes : spectrale, max des entrees,
Frobenius ponderee ||Delta P^{1/2}||_F.  Usage : python3 paper/audit/checks/jury-6_pl_constant.py"""
import numpy as np
from scipy.optimize import minimize

rng = np.random.default_rng(0)
K, d, n, lam = 3, 4, 30, 0.1
X = rng.standard_normal((n, d))
y = rng.integers(0, K, size=n)
Y = np.eye(K)[y]

def F(W):
    Z = X @ W.T
    Z = Z - Z.max(axis=1, keepdims=True)
    lse = np.log(np.exp(Z).sum(axis=1))
    return float(np.mean(lse - Z[np.arange(n), y]) + 0.5 * lam * np.sum(W * W))

def gradF(W):
    Z = X @ W.T
    Z = Z - Z.max(axis=1, keepdims=True)
    P = np.exp(Z); P /= P.sum(axis=1, keepdims=True)
    return (P - Y).T @ X / n + lam * W

res = minimize(lambda w: F(w.reshape(K, d)), np.zeros(K * d), jac=lambda w: gradF(w.reshape(K, d)).ravel(),
               method="L-BFGS-B", options=dict(gtol=1e-13, ftol=1e-16, maxiter=5000))
Fstar = res.fun
print("F_* = %.12f, ||grad|| at optimum = %.1e" % (Fstar, np.linalg.norm(res.jac)))

Pm = np.diag([4.0, 1.0, 0.5, 0.25])                      # metrique ponderee, lambda_max(P) = 4
u = np.zeros(K); u[0] = 1.0
e11 = np.zeros((K, d)); e11[0, 0] = 1.0
norms = {
    # nom : (norme duale, Delta_0 de norme de Frobenius 1 maximisant N / ||.||_F, c_N^2)
    "spectrale": (lambda G: np.linalg.svd(G, compute_uv=False).sum(), np.outer(u, np.eye(d)[1]), 1.0),
    "max des entrees": (lambda G: np.abs(G).sum(), e11, 1.0),
    "Frobenius ponderee": (lambda G: np.linalg.norm(G @ np.diag(np.diag(Pm) ** -0.5)), np.outer(u, np.eye(d)[0]), 4.0),
}
ok = True
for name, (dual, D0, c2) in norms.items():
    target = lam / c2
    ratios = []
    for t in [1e1, 1e2, 1e3, 1e4, 1e5]:
        W = t * D0
        ratios.append(dual(gradF(W)) ** 2 / (2 * (F(W) - Fstar)))
    # la borne inferieure mu_PL >= lambda / c_N^2 sur des points aleatoires
    worst = np.inf
    for _ in range(20000):
        W = res.x.reshape(K, d) + rng.standard_normal((K, d)) * 10 ** rng.uniform(-3, 3)
        worst = min(worst, dual(gradF(W)) ** 2 / (2 * (F(W) - Fstar)))
    print("%-20s lambda/c_N^2 = %.5f | rapport le long de t*Delta_0, t = 1e1..1e5 : %s | min sur 20000 points : %.5f"
          % (name, target, " ".join("%.5f" % r for r in ratios), worst))
    ok &= abs(ratios[-1] - target) < 2e-3 * target and worst >= target * (1 - 1e-9)
print("CONFORME" if ok else "NON CONFORME")
