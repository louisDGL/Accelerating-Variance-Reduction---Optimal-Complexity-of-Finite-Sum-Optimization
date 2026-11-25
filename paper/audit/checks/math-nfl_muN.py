"""Audit math-nfl, controle (b) : Theoreme 1 (no free lunch, for every norm) sur de petits problemes.

Modeles (GLM) ridge : softmax (K = 4, d = 6, n = 30) et logistique binaire (K = 1, d = 6, n = 30).
Normes N testees sur R^{K x d} :
  op      : spectrale                         c_N = 1          Delta_0 = u v^T
  linf    : max des |entrees| (descente de signe)  c_N = 1     Delta_0 = E_11
  wF      : ||Delta P^{1/2}||_F               c_N^2 = lmax(P)  Delta_0 = u v_max(P)^T
  kron    : ||S^{1/2} Delta P^{1/2}||_F       c_N^2 = lmax(S) lmax(P)
  pop     : ||Delta P^{1/2}||_op              c_N^2 = lmax(P)
  l1      : somme des |entrees|               c_N = sqrt(K d)  Delta_0 = matrice de signes / sqrt(K d)
Controles :
  (1) c_N : la valeur annoncee est atteinte en Delta_0 et jamais depassee sur 2e5 directions aleatoires ;
  (2) hypothese : max_i ||hess l_i(t W_1 x_i)||_op -> 0 le long de t W_1, W_1 gaussien ; contre-exemples : W_1 a deux
      lignes egales dominantes (softmax), x_i = 0 ;
  (3) mu_N(F) = lam / c_N^2 : D^2F(t W_1)[Delta_0, Delta_0] / N(Delta_0)^2 -> lam / c_N^2 (majorant de mu_N),
      et sur 4e4 couples (W, Delta) aleatoires + raffinement le quotient ne descend jamais sous lam / c_N^2 ;
  (4) L_N(g) >= L_F(g) / c_N^2 pour g = F et g = f_i : minorant numerique de L_N par ascension, compare a
      L_F / c_N^2 avec L_F(f_i) = c_i |x_i|^2 + lam (c_i = 1/2 softmax, 1/4 logistique) ;
  (5) consequences : kappa_N(F) >= kappa_F(F) et L_N(f_i)/mu_N(F) >= (c_i |x_i|^2 + lam)/lam.

Lancer depuis la racine du depot :
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 paper/audit/checks/math-nfl_muN.py
"""
import numpy as np
from scipy.optimize import minimize

rng = np.random.default_rng(11)
lam = 1e-2


def rand_spd(m, cond):
    Q, _ = np.linalg.qr(rng.standard_normal((m, m)))
    ev = np.exp(np.linspace(0, np.log(cond), m))
    return (Q * ev) @ Q.T


def sqrtm_spd(A):
    w, V = np.linalg.eigh(A)
    return (V * np.sqrt(w)) @ V.T


class Model:
    def __init__(self, kind, K, d, n):
        self.kind, self.K, self.d, self.n = kind, K, d, n
        self.X = rng.standard_normal((n, d))
        self.c = 0.5 if kind == "softmax" else 0.25

    def hess_blocks(self, W):
        """liste des hessiennes K x K de l_i en W x_i."""
        Z = self.X @ W.T
        if self.kind == "softmax":
            P = np.exp(Z - Z.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
            return np.einsum("ik,kl->ikl", P, np.eye(self.K)) - np.einsum("ik,il->ikl", P, P)
        a = np.exp(-np.abs(Z))                  # logistique : l''(z) = s (1 - s) = a / (1 + a)^2, a = exp(-|z|)
        return (a / (1 + a) ** 2)[:, :, None]

    def d2(self, W, D, idx=None):
        """D^2 F(W)[D, D] (idx None) ou D^2 f_i(W)[D, D]."""
        Hs = self.hess_blocks(W)
        V = self.X @ D.T
        q = np.einsum("ik,ikl,il->i", V, Hs, V)
        loss = q.mean() if idx is None else q[idx]
        return loss + lam * np.sum(D * D)

    def max_curv(self, W):
        return max(np.linalg.eigvalsh(H)[-1] for H in self.hess_blocks(W))


def norms(K, d):
    P = rand_spd(d, 50.0); S = rand_spd(K, 20.0) if K > 1 else np.eye(1)
    Ph, Sh = sqrtm_spd(P), sqrtm_spd(S)
    wP, VP = np.linalg.eigh(P); wS, VS = np.linalg.eigh(S)
    u = np.zeros(K); u[0] = 1.0
    E11 = np.zeros((K, d)); E11[0, 0] = 1.0
    sgn = np.sign(rng.standard_normal((K, d)))
    return {
        "op":   (lambda D: np.linalg.norm(D, 2), 1.0, np.outer(u, np.eye(d)[0])),
        "linf": (lambda D: np.abs(D).max(), 1.0, E11),
        "wF":   (lambda D: np.linalg.norm(D @ Ph), np.sqrt(wP[-1]), np.outer(u, VP[:, -1])),
        "kron": (lambda D: np.linalg.norm(Sh @ D @ Ph), np.sqrt(wS[-1] * wP[-1]), np.outer(VS[:, -1], VP[:, -1])),
        "pop":  (lambda D: np.linalg.norm(D @ Ph, 2), np.sqrt(wP[-1]), np.outer(u, VP[:, -1])),
        "l1":   (lambda D: np.abs(D).sum(), np.sqrt(K * d), sgn / np.sqrt(K * d)),
    }


for kind, K, d, n in [("softmax", 4, 6, 30), ("logistic", 1, 6, 30)]:
    M = Model(kind, K, d, n)
    print("=" * 100)
    print("%s : K=%d d=%d n=%d lam=%g ; L_F(f_i) = c_i|x_i|^2 + lam avec c_i = %g" % (kind, K, d, n, lam, M.c))
    W1 = rng.standard_normal((K, d))
    ts = [1, 10, 100, 1000] if kind == "softmax" else [1, 1e2, 1e4, 1e6]
    print("    min_i |W_1 x_i| = %.3e (logistique) ou plus petit ecart entre les deux plus grandes coordonnees de W_1 x_i = %.3e (softmax)"
          % (np.abs(M.X @ W1.T).min(), np.min(np.sort(M.X @ W1.T, axis=1)[:, -1] - np.sort(M.X @ W1.T, axis=1)[:, -2]) if K > 1 else float("nan")))
    print("(2) max_i lmax(hess l_i(t W_1 x_i)), t = %s : %s" % (ts, ["%.2e" % M.max_curv(t * W1) for t in ts]))
    if kind == "softmax":
        Wtie = W1.copy(); Wtie[1] = Wtie[0]           # deux lignes egales : ex aequo pour les i ou elles dominent
        print("    contre-exemple (deux lignes egales)   : %s  -> ne tend pas vers 0 (W_1 non generique)"
              % ["%.2e" % M.max_curv(t * Wtie) for t in ts])
    Xsave = M.X.copy(); M.X = Xsave.copy(); M.X[0] = 0.0
    print("    contre-exemple (x_1 = 0)              : %s  -> %s" % (["%.2e" % M.max_curv(t * W1) for t in ts],
          "ne tend pas vers 0" if kind == "softmax" else "l''(0) = 1/4"))
    M.X = Xsave
    # constantes Frobenius exactes par echantillon : verification de L_F(f_i) = c_i |x_i|^2 + lam
    i0 = int(np.argmax((M.X ** 2).sum(1)))
    xi = M.X[i0]
    if kind == "softmax":
        zt = np.zeros(K); zt[0] = zt[1] = 30.0      # p -> (1/2, 1/2, 0, 0) : lmax -> 1/2
        e = np.zeros(K); e[0], e[1] = 1 / np.sqrt(2), -1 / np.sqrt(2)
    else:
        zt = np.zeros(K); e = np.ones(1)            # l''(0) = 1/4
    Wz = np.outer(zt, xi) / (xi @ xi)
    Dz = np.outer(e, xi) / np.linalg.norm(xi)
    LF_fi = M.c * (xi @ xi) + lam
    print("    L_F(f_i) : quotient en (W, e x_i^T/|x_i|) = %.10f, formule c_i|x_i|^2 + lam = %.10f" % (M.d2(Wz, Dz, i0), LF_fi))
    # borne Frobenius globale L_F(F) : minorant par ascension aleatoire (sert au point (5))
    best_LF = 0.0
    for _ in range(3000):
        W = rng.standard_normal((K, d)) * rng.uniform(0, 2); D = rng.standard_normal((K, d)); D /= np.linalg.norm(D)
        best_LF = max(best_LF, M.d2(W, D))
    f = lambda v: -M.d2(v[:K * d].reshape(K, d), v[K * d:].reshape(K, d)) / np.sum(v[K * d:] ** 2)
    for _ in range(6):
        v0 = np.r_[rng.standard_normal(K * d) * 0.3, rng.standard_normal(K * d)]
        best_LF = max(best_LF, -minimize(f, v0, method="L-BFGS-B", options=dict(maxiter=300)).fun)
    print("    L_F(F) (minorant numerique) = %.6f ; kappa_F(F) >= %.2f" % (best_LF, best_LF / lam))
    print("%-5s %8s %10s %10s | %s | %12s %12s | %10s %12s %12s" % ("N", "c_N", "N(D0)/|D0|", "max alea", "D2F(tW1)[D0]/N(D0)^2 / (lam/c_N^2), le long de t W_1 (memes t)",
          "min quotient", "/(lam/c_N^2)", "L_N(F)>=", "L_F(F)/c_N^2", "kappa_N>="))
    for name, (N, cN, D0) in norms(K, d).items():
        ratio0 = N(D0) / np.linalg.norm(D0)
        rnd = max(N(D) / np.linalg.norm(D) for D in rng.standard_normal((20000, K, d)))
        target = lam / cN ** 2
        along = [M.d2(t * W1, D0) / N(D0) ** 2 / target for t in ts]
        # (3) minimum du quotient de courbure sur des couples aleatoires, puis raffinement local
        qmin = np.inf
        for _ in range(2000):
            W = rng.standard_normal((K, d)) * 10 ** rng.uniform(-1, 3); D = rng.standard_normal((K, d))
            qmin = min(qmin, M.d2(W, D) / N(D) ** 2)
        g = lambda v: M.d2(v[:K * d].reshape(K, d), v[K * d:].reshape(K, d)) / N(v[K * d:].reshape(K, d)) ** 2
        for _ in range(4):
            v0 = np.r_[(rng.standard_normal(K * d) * 30), (D0 + 0.1 * rng.standard_normal((K, d))).reshape(-1)]
            qmin = min(qmin, minimize(g, v0, method="Nelder-Mead", options=dict(maxiter=3000, xatol=1e-9, fatol=1e-14)).fun)
        # (4) minorant de L_N(F) par evaluation + raffinement
        LN = 0.0
        for _ in range(3000):
            W = rng.standard_normal((K, d)) * rng.uniform(0, 2); D = rng.standard_normal((K, d))
            LN = max(LN, M.d2(W, D) / N(D) ** 2)
        h = lambda v: -g(v)
        for _ in range(4):
            v0 = np.r_[rng.standard_normal(K * d) * 0.3, rng.standard_normal(K * d)]
            LN = max(LN, -minimize(h, v0, method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-9, fatol=1e-14)).fun)
        # et en la direction optimale Frobenius par echantillon : L_N(f_i) >= D2 f_i[Dz]/N(Dz)^2 >= L_F(f_i)/c_N^2
        LN_fi = M.d2(Wz, Dz, i0) / N(Dz) ** 2
        ok_fi = LN_fi >= LF_fi / cN ** 2 * (1 - 1e-6)
        print("%-5s %8.4f %10.6f %10.6f | %s | %12.3e %12.6f | %10.4f %12.4f %12.2f | per-sample %s (%.4g >= %.4g)"
              % (name, cN, ratio0, rnd, " ".join("%.6f" % a for a in along), qmin, qmin / target, LN, best_LF / cN ** 2,
                 LN / target, "OK" if ok_fi else "VIOLATION", LN_fi / target, LF_fi / lam))
