"""Audit math-nfl, controle (e) : "On least squares ... the gap is about a factor 4 on the ray that attracts
generic runs" (l. 313-314).

Pas polaire a longueur nucleaire E+ = E - (||E Sigma||_* / L_op) polar(E Sigma) sur la quadratique unilaterale
F - F* = (1/2) tr(E Sigma E^T), L_op = somme des r = min(K, d) plus grandes valeurs propres de Sigma,
mu_op = lmin(Sigma). On mesure, depuis des departs generiques (gaussiens) :
  - l'exposant exact par pas dans la queue (moyenne de -log(D_{t+1}/D_t) sur une fenetre) rapporte a l'exposant
    certifie -log(1 - 1/kappa_op) ;
  - les deux marges de (2) moyennees sur la fenetre : mu_PL/mu_op et 2 - J/L_op ;
  - le rapport predit sur le rayon invariant de la direction la plus plate (K >= d) :
    2 log(1 - 2 p_d) / log(1 - p_d), p_d = lmin / tr Sigma ;
  - l'alignement du vecteur singulier droit dominant de l'erreur avec le vecteur propre le plus plat.

Lancer depuis la racine du depot :
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 paper/audit/checks/math-nfl_ls_tail.py
"""
import numpy as np


def run(K, d, ev, seed, T=6000, floor=1e-22, rotate=True):
    rng = np.random.default_rng(seed)
    r = min(K, d)
    ev = np.sort(np.asarray(ev, float))[::-1]
    if rotate:
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    else:
        Q = np.eye(d)
    Sig = (Q * ev) @ Q.T
    L_op, mu_op = ev[:r].sum(), ev[-1]
    E = rng.standard_normal((K, d))
    D0 = 0.5 * np.sum((E @ Sig) * E)
    logs, pl, sm, cosv = [], [], [], []
    D = D0
    for t in range(T):
        G = E @ Sig
        U, s, Vt = np.linalg.svd(G, full_matrices=False)
        O = U @ Vt
        nuc = s.sum()
        J = np.sum((O @ Sig) * O)
        E = E - (nuc / L_op) * O
        Dn = 0.5 * np.sum((E @ Sig) * E)
        logs.append(-np.log(Dn / D)); pl.append(nuc ** 2 / (2 * D) / mu_op); sm.append(2 - J / L_op)
        _, _, Vte = np.linalg.svd(E, full_matrices=False)
        cosv.append(abs(Vte[0] @ Q[:, -1]))
        D = Dn
        if D < floor * D0:
            break
    logs = np.array(logs); n = len(logs)
    w = slice(n // 2, n)                       # seconde moitie de la course (queue)
    cert = -np.log(1 - mu_op / L_op)
    return dict(steps=n, ratio=logs[w].mean() / cert, pl=np.mean(np.array(pl)[w]), sm=np.mean(np.array(sm)[w]),
                first_order=np.mean(np.array(pl)[w] * np.array(sm)[w]), cos=np.mean(np.array(cosv)[w]),
                kop=L_op / mu_op, pd=mu_op / ev.sum(), head=logs[: max(5, n // 20)].mean() / cert,
                minstep=logs[w].min() / cert)


def spectrum(kind, d):
    k = np.arange(1, d + 1)
    if kind == "1/k":
        return 1.0 / k
    if kind == "1/k^2":
        return 1.0 / k ** 2
    if kind == "geo100":
        return 100.0 ** (-(k - 1) / max(d - 1, 1))
    if kind == "gap":           # une petite valeur propre isolee
        return np.r_[np.ones(d - 1), 0.02]
    raise ValueError(kind)


print("%-8s %-7s %9s %6s | %27s | %8s %8s %8s %6s %7s" % ("K x d", "spectre", "kappa_op", "pas",
      "exposant exact/certifie", "mu_PL/mu", "2-J/L", "rayon", "cos", "min/pas"))
print("%-8s %-7s %9s %6s | %8s %8s %9s |" % ("", "", "", "", "min", "mediane", "max"))
allr = {}
for (K, d) in [(20, 20), (10, 4), (40, 8), (6, 6), (2, 2), (4, 10), (8, 40), (10, 50)]:
    for kind in ["1/k", "1/k^2", "geo100", "gap"]:
        ev = spectrum(kind, d)
        res = [run(K, d, ev, seed) for seed in range(5)]
        rat = np.array([x["ratio"] for x in res])
        pd = res[0]["pd"]
        ray = 2 * np.log(1 - 2 * pd) / np.log(1 - pd) if (K >= d and pd < 0.5) else float("nan")
        allr[(K, d, kind)] = rat
        print("%-8s %-7s %9.1f %6d | %8.2f %8.2f %9.2f | %8.2f %8.3f %8.2f %6.3f %7.2f"
              % ("%dx%d" % (K, d), kind, res[0]["kop"], int(np.median([x["steps"] for x in res])), rat.min(),
                 np.median(rat), rat.max(), np.median([x["pl"] for x in res]), np.median([x["sm"] for x in res]),
                 ray, np.median([x["cos"] for x in res]), min(x["minstep"] for x in res)))

sq = np.concatenate([v for (K, d, k), v in allr.items() if K >= d])
wide = np.concatenate([v for (K, d, k), v in allr.items() if K < d])
print("\nK >= d : exposant exact / certifie dans la queue, min %.2f, mediane %.2f, max %.2f (%d courses)"
      % (sq.min(), np.median(sq), sq.max(), len(sq)))
print("K <  d : exposant exact / certifie dans la queue, min %.2f, mediane %.2f, max %.2f (%d courses)"
      % (wide.min(), np.median(wide), wide.max(), len(wide)))
