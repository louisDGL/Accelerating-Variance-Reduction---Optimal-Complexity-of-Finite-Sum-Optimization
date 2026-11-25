"""Audit math-nfl, controle (f) : metrique polaire (l. 240-242) et (M4) "Computed once, the polar metric makes the
method SAGA in a fixed metric, certified with R_P^2, mu_P".

Sur un petit softmax ridge (K = 3, d = 5, n = 24) et un petit moindres carres (K = 3, d = 5, n = 24) :
  (1) G_0 C = ||G_0||_* polar(G_0) pour C = V diag(||G_0||_*/s_k) V^T + r (I - V V^T) (sans plancher) ;
  (2) R_P^2 = max_i c_i x_i^T C x_i + lam ||C||_op majore bien max_i L_N(f_i) pour N = ||. C^{-1/2}||_F
      (valeur exacte lmax(c_i C^{1/2} x_i x_i^T C^{1/2} + lam C)) ;
  (3) mu_P : softmax -> lam lmin(C) (Theoreme 1 avec c_N^2 = 1/lmin(C)) ; moindres carres -> lmin(C^{1/2} Sigma C^{1/2}) ;
  (4) certificat : le long d'une course de SAGA dans la metrique figee (table des gradients des pertes, gradient
      exact du regularisateur, gamma = 1/(4 R_P^2), table initialisee en W_0), l'esperance conditionnelle exacte
      (enumeration des n indices) de V = ||W - W*||_{C^-1}^2 + 3 gamma^2 sum_i ||z_i - g_i'(W*)||_C^2 verifie
      E_j[V+] <= (1 - min{1/(3n), 3 mu_P/(16 R_P^2)}) V a chaque etat visite ;
  (5) kappa_P >= kappa (softmax) pour des planchers f dans {0, 0.1, 0.3, 1}.

Lancer depuis la racine du depot :
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 paper/audit/checks/math-nfl_polar_metric.py
"""
import numpy as np
from scipy.optimize import minimize

rng = np.random.default_rng(3)
K, d, n, lam = 3, 5, 24, 5e-2
X = rng.standard_normal((n, d)) * np.linspace(1.5, 0.3, d)
ycls = rng.integers(0, K, n)
Yls = rng.standard_normal((n, K))


def loss_grads(kind, W):
    """residus n x K : grad l_i(W x_i) ; g_i'(W) = r_i x_i^T."""
    Z = X @ W.T
    if kind == "softmax":
        P = np.exp(Z - Z.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
        P[np.arange(n), ycls] -= 1
        return P
    return Z - Yls


def Fval(kind, Wv):
    W = Wv.reshape(K, d); Z = X @ W.T
    if kind == "softmax":
        m = Z.max(1, keepdims=True)
        val = np.mean(m[:, 0] + np.log(np.exp(Z - m).sum(1)) - Z[np.arange(n), ycls])
    else:
        val = 0.5 * np.mean(((Z - Yls) ** 2).sum(1))
    return val + 0.5 * lam * np.sum(W * W)


def Fgrad(kind, Wv):
    W = Wv.reshape(K, d)
    return (loss_grads(kind, W).T @ X / n + lam * W).reshape(-1)


def metric(G0, f):
    U, s, Vt = np.linalg.svd(G0, full_matrices=False)
    keep = s > 1e-12 * s[0]
    s, V = s[keep], Vt[keep].T
    r = len(s)
    g = s.sum() / np.maximum(s, f * s[0])
    return V @ np.diag(g) @ V.T + r * (np.eye(d) - V @ V.T), U[:, keep], s, V


for kind, c_l in [("softmax", 0.5), ("ls", 1.0)]:
    W0 = np.zeros((K, d))
    res = minimize(lambda v: Fval(kind, v), W0.reshape(-1), jac=lambda v: Fgrad(kind, v), method="L-BFGS-B",
                   options=dict(gtol=1e-14, ftol=1e-18, maxiter=20000))
    Ws = res.x.reshape(K, d)
    for _ in range(50):                                   # polissage par gradient (quadratique ou fortement convexe)
        Ws = Ws - 0.2 * Fgrad(kind, Ws.reshape(-1)).reshape(K, d)
    G0 = Fgrad(kind, W0.reshape(-1)).reshape(K, d)
    sq = (X ** 2).sum(1)
    kappa = (c_l * sq.max() + lam) / (lam if kind == "softmax" else np.linalg.eigvalsh(X.T @ X / n + lam * np.eye(d))[0])
    print("=" * 90)
    print("%s : |grad F(W*)| = %.1e ; kappa euclidien global R^2/mu = %.2f" % (kind, np.abs(Fgrad(kind, Ws.reshape(-1))).max(), kappa))
    for f in [0.0, 0.1, 0.3, 1.0]:
        C, U, s, V = metric(G0, f)
        if f == 0.0:
            print("(1) |G_0 C - ||G_0||_* polar(G_0)|_max = %.2e (rang %d)" % (np.abs(G0 @ C - s.sum() * U @ V.T).max(), len(s)))
        wC, VC = np.linalg.eigh(C)
        Ch = (VC * np.sqrt(wC)) @ VC.T
        q = np.einsum("ij,jk,ik->i", X, C, X)
        R2P = c_l * q.max() + lam * wC[-1]
        LN_exact = max(np.linalg.eigvalsh(c_l * np.outer(Ch @ x, Ch @ x) + lam * C)[-1] for x in X)
        if kind == "softmax":
            muP = lam * wC[0]
        else:
            muP = np.linalg.eigvalsh(Ch @ (X.T @ X / n + lam * np.eye(d)) @ Ch)[0]
        kP = R2P / muP
        rho = 1 - min(1 / (3 * n), 3 * muP / (16 * R2P))
        # (4) course de SAGA dans la metrique, esperance conditionnelle exacte de V
        gam = 1 / (4 * R2P)
        Cinv = np.linalg.inv(C)
        Rs = loss_grads(kind, Ws)                          # residus au minimiseur
        W = W0.copy(); Rtab = loss_grads(kind, W)          # table initialisee en W_0
        normC = lambda A: np.sum((A @ C) * A)
        normCi = lambda A: np.sum((A @ Cinv) * A)

        def V_of(W, Rtab):
            mem = sum(normC(np.outer(Rtab[i] - Rs[i], X[i])) for i in range(n))
            return normCi(W - Ws) + 3 * gam ** 2 * mem

        worst = -np.inf
        rs = np.random.default_rng(5)
        for t in range(600):
            Vt = V_of(W, Rtab)
            if Vt < 1e-22:
                break
            Rcur = loss_grads(kind, W)
            zbar = Rtab.T @ X / n
            EV = 0.0
            for j in range(n):
                om = np.outer(Rcur[j] - Rtab[j], X[j]) + zbar + lam * W
                Rn = Rtab.copy(); Rn[j] = Rcur[j]
                EV += V_of(W - gam * om @ C, Rn) / n
            worst = max(worst, EV / (rho * Vt))
            j = rs.integers(n)
            om = np.outer(Rcur[j] - Rtab[j], X[j]) + zbar + lam * W
            W = W - gam * om @ C; Rtab[j] = Rcur[j]
        print("f=%.1f : R_P^2=%.4f >= max_i L_N(f_i)=%.4f ; mu_P=%.5f ; kappa_P=%.2f (kappa_P/kappa=%.4f) ; "
              "max_t E[V+]/(rho V) = %.6f (<= 1 attendu, %d etats)" % (f, R2P, LN_exact, muP, kP, kP / kappa, worst, t + 1))
