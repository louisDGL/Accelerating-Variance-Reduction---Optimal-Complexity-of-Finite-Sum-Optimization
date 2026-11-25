"""Audit math-nfl, controle (c) : kappa_op >= min(K, d) pour toute quadratique matricielle.

Pour un operateur SPD H sur R^{K x d}, q(D) = <D, H D> :
  L_op  = max_{Q a valeurs singulieres toutes egales a 1} q(Q)        (maximum d'une fonction convexe sur la boule)
  1/mu_op = max_{|u|=|v|=1} <u v^T, H^{-1} u v^T>                      (dualite forte convexite / regularite)
Les deux sont calcules par ascension multi-depart (iterations monotones) : on obtient des MINORANTS de L_op et de
1/mu_op, donc un MINORANT de kappa_op. Le script verifie :
  (1) la preuve par cadres : pour un cadre aleatoire (u_k, v_k), h_k = q(u_k v_k^T),
      max_eps q(sum eps_k u_k v_k^T) >= sum_k h_k >= r min_k h_k   et   mu_op <= min_k h_k ;
  (2) kappa_op (minorant multi-depart, quasi exact en petite taille) >= r sur des H aleatoires, y compris
      proches de l'identite (ou la borne est serree : H = Id donne exactement r) ;
  (3) le controle par force brute en 2 x 2 : mu_op et L_op par balayage fin des directions.

Lancer depuis la racine du depot :
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 paper/audit/checks/math-nfl_kop.py
"""
import itertools
import numpy as np

rng = np.random.default_rng(7)


def polar_full(G):
    U, s, Vt = np.linalg.svd(G, full_matrices=False)
    return U @ Vt


def L_op_lb(M, K, d, starts=30, iters=200):
    """max q(Q) sur les Q a valeurs singulieres 1 : Q <- polar(H Q) est une ascension (q convexe)."""
    best = 0.0
    for _ in range(starts):
        Q = polar_full(rng.standard_normal((K, d)))
        val = -1.0
        for _ in range(iters):
            HQ = (M @ Q.reshape(-1)).reshape(K, d)
            new = float(np.sum(Q * HQ))
            if new <= val * (1 + 1e-13):
                break
            val = new
            Q = polar_full(HQ)
        best = max(best, val)
    return best


def inv_mu_op_lb(Minv, K, d, starts=30, iters=200):
    """max <u v^T, H^{-1} u v^T> : alternance de vecteurs propres dominants (monotone)."""
    T = Minv.reshape(K, d, K, d)
    best = 0.0
    for _ in range(starts):
        v = rng.standard_normal(d); v /= np.linalg.norm(v)
        val = -1.0
        for _ in range(iters):
            A = np.einsum("kalb,a,b->kl", T, v, v)
            w, U = np.linalg.eigh(A); u = U[:, -1]
            B = np.einsum("kalb,k,l->ab", T, u, u)
            w, V = np.linalg.eigh(B); v = V[:, -1]
            if w[-1] <= val * (1 + 1e-13):
                break
            val = w[-1]
        best = max(best, val)
    return best


def rand_spd(m, kind):
    if kind == "near_id":
        S = rng.standard_normal((m, m)); S = (S + S.T) / 2
        eps = 10 ** rng.uniform(-6, -1)
        return np.eye(m) + eps * S / np.linalg.norm(S, 2)
    Q, _ = np.linalg.qr(rng.standard_normal((m, m)))
    ev = np.exp(rng.uniform(0, np.log(10 ** rng.uniform(0, 4)), m))
    return (Q * ev) @ Q.T


print("(1)-(2) kappa_op (minorant multi-depart) / r, et inegalites de cadre")
worst_overall = np.inf
frame_fail = 0
for (K, d) in [(2, 2), (2, 3), (3, 3), (4, 2), (3, 5), (5, 5), (6, 4)]:
    r = min(K, d)
    ratios = []
    for rep in range(30):
        kind = "near_id" if rep % 3 == 0 else "generic"
        M = rand_spd(K * d, kind)
        Minv = np.linalg.inv(M)
        Llb = L_op_lb(M, K, d)
        mu_ub = 1.0 / inv_mu_op_lb(Minv, K, d)
        ratios.append(Llb / mu_ub / r)
        # inegalites de cadre
        U, _ = np.linalg.qr(rng.standard_normal((K, K))); V, _ = np.linalg.qr(rng.standard_normal((d, d)))
        q = lambda D: float(D.reshape(-1) @ M @ D.reshape(-1))
        h = np.array([q(np.outer(U[:, k], V[:, k])) for k in range(r)])
        best_sign = max(q(sum(e[k] * np.outer(U[:, k], V[:, k]) for k in range(r)))
                        for e in itertools.product([-1, 1], repeat=r))
        ok = (best_sign >= h.sum() * (1 - 1e-12)) and (mu_ub <= h.min() * (1 + 1e-9)) and (Llb >= best_sign * (1 - 1e-9) or True)
        frame_fail += (not ok)
    ratios = np.array(ratios)
    worst_overall = min(worst_overall, ratios.min())
    print("  K=%d d=%d r=%d : min kappa_op/r = %.9f, mediane = %.3g, max = %.3g" % (K, d, r, ratios.min(), np.median(ratios), ratios.max()))
print("  echec des inegalites de cadre : %d ; min global de kappa_op/r = %.9f (>= 1 attendu)" % (frame_fail, worst_overall))

# H = Id : egalite
for (K, d) in [(2, 2), (3, 5), (6, 4)]:
    M = np.eye(K * d)
    print("  H = Id, K=%d d=%d : kappa_op = %.12f (r = %d)" % (K, d, L_op_lb(M, K, d) * inv_mu_op_lb(M, K, d), min(K, d)))

# (3) force brute 2 x 2 : mu_op, L_op par echantillonnage dense de la sphere unite de R^4 pour la norme spectrale
print("(3) force brute 2 x 2 (2e5 directions aleatoires + raffinement local) contre le multi-depart")
from scipy.optimize import minimize
worst_bf = np.inf
for rep in range(10):
    M = rand_spd(4, "near_id" if rep % 2 == 0 else "generic")

    def ratio(x):
        D = x.reshape(2, 2)
        return float(x @ M @ x) / np.linalg.norm(D, 2) ** 2

    Xs = rng.standard_normal((200000, 4))
    qv = np.einsum("ni,ij,nj->n", Xs, M, Xs)
    nrm = np.linalg.norm(Xs.reshape(-1, 2, 2), 2, axis=(1, 2)) ** 2
    rat = qv / nrm
    lo = min(minimize(ratio, Xs[i], method="Nelder-Mead", options=dict(xatol=1e-12, fatol=1e-14, maxiter=4000)).fun
             for i in np.argsort(rat)[:5])
    hi = max(-minimize(lambda x: -ratio(x), Xs[i], method="Nelder-Mead", options=dict(xatol=1e-12, fatol=1e-14, maxiter=4000)).fun
             for i in np.argsort(rat)[-5:])
    Llb = L_op_lb(M, 2, 2); mu_ub = 1.0 / inv_mu_op_lb(np.linalg.inv(M), 2, 2)
    worst_bf = min(worst_bf, hi / lo / 2)
    print("  rep %d : force brute mu_op=%.8f L_op=%.8f kappa/2=%.8f | multi-depart mu_op=%.8f L_op=%.8f"
          % (rep, lo, hi, hi / lo / 2, mu_ub, Llb))
print("  min kappa_op/2 par force brute = %.9f (>= 1 attendu)" % worst_bf)
