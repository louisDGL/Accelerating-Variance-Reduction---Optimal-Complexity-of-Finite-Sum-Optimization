"""Audit math-nfl, complement du controle (e) : sur les moindres carres, l'ecart exact/certifie du pas polaire a
longueur nucleaire vaut ~4 dans la QUEUE des courses generiques ; qu'en est-il sur une course entiere jusqu'a une
precision donnee (tete comprise) ? On mesure, depuis un depart gaussien, le nombre de pas T(eps) pour
D_t/D_0 <= eps et le rapport [log(1/eps)/T(eps)] / [-log(1 - 1/kappa_op)] (exposant moyen exact / certifie),
ainsi que mu_PL/mu_op au premier pas.

Lancer depuis la racine du depot :
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 paper/audit/checks/math-nfl_ls_head.py
"""
import numpy as np


def run(K, d, ev, seed, T=20000):
    rng = np.random.default_rng(seed)
    r = min(K, d)
    ev = np.sort(np.asarray(ev, float))[::-1]
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    Sig = (Q * ev) @ Q.T
    L_op, mu_op = ev[:r].sum(), ev[-1]
    E = rng.standard_normal((K, d))
    D0 = 0.5 * np.sum((E @ Sig) * E)
    cert = -np.log(1 - mu_op / L_op)
    hits = {}
    pl0 = None
    for t in range(T):
        G = E @ Sig
        U, s, Vt = np.linalg.svd(G, full_matrices=False)
        D = 0.5 * np.sum(G * E)
        if pl0 is None:
            pl0 = s.sum() ** 2 / (2 * D) / mu_op
        for eps in (1e-2, 1e-6, 1e-12):
            if eps not in hits and D <= eps * D0:
                hits[eps] = t
        if len(hits) == 3:
            break
        E = E - (s.sum() / L_op) * (U @ Vt)
    return L_op / mu_op, pl0, {e: np.log(1 / e) / max(t, 1) / cert for e, t in hits.items()}, hits


k = lambda d: np.arange(1, d + 1)
print("%-7s %-7s %9s %12s | exposant moyen exact / certifie jusqu'a eps (pas)" % ("K x d", "spectre", "kappa_op", "mu_PL/mu(0)"))
for (K, d) in [(20, 20), (40, 8), (8, 40), (10, 50)]:
    for name, ev in [("1/k", 1.0 / k(d)), ("1/k^2", 1.0 / k(d) ** 2), ("geo100", 100.0 ** (-(k(d) - 1) / (d - 1)))]:
        out = [run(K, d, ev, s) for s in range(3)]
        kop = out[0][0]
        med = {e: np.median([o[2].get(e, np.nan) for o in out]) for e in (1e-2, 1e-6, 1e-12)}
        st = {e: int(np.median([o[3].get(e, -1) for o in out])) for e in (1e-2, 1e-6, 1e-12)}
        print("%-7s %-7s %9.1f %12.1f | 1e-2: %6.1f (%d) ; 1e-6: %6.1f (%d) ; 1e-12: %6.1f (%d)"
              % ("%dx%d" % (K, d), name, kop, np.median([o[1] for o in out]), med[1e-2], st[1e-2], med[1e-6], st[1e-6], med[1e-12], st[1e-12]))
