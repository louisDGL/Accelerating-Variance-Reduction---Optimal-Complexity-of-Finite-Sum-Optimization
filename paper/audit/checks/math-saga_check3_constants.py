"""Audit math-saga, controle 3 : (C4) premiere phrase et (C5).

(a) Meilleur couple (gamma, Delta) "dans la preuve" : la chaine I1-I5 certifie, pour tout gamma, Delta > 0,
      tau(gamma, Delta) = min{ 2 gamma mu (1 - gamma R^2 - R^2 Delta/(2 gamma)),  (1/n)(1 - 2 gamma^2/Delta) }
    (recalcule ici a la main, cf. rapport).  On maximise par force brute sur une grille et on compare au taux du livre
    min{1/(3n), 3/(16 kappa)} : gain <= 3, <= 4/3 si kappa >= 9n/16, = 17 % a kappa = n = 1000 ; plus grand pas admis.
(b) (C5) : kappa_P >= kappa pour toute metrique P (regression logistique ridge), avec le R_P^2 du papier (l. 237)
    et avec la constante de regularite exacte dans la metrique P ; mu_P = lambda / lambda_max(P).

Lancer depuis la racine du depot :
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 paper/audit/checks/math-saga_check3_constants.py
"""
import math
import numpy as np
from scipy.special import expit


def tau_grid(n, kappa):
    """max sur une grille fine de (x = gamma R^2, c = Delta/gamma^2) du taux certifie (R^2 = 1, mu = 1/kappa)."""
    x = np.exp(np.linspace(math.log(1e-7), math.log(0.5), 6000))[:, None]      # gamma R^2
    c = (2 + np.exp(np.linspace(math.log(1e-6), math.log(1e7), 3000)))[None, :]  # Delta = c gamma^2, c > 2
    alpha = 1 - x - c * x / 2
    tD = 2 * x / kappa * alpha
    tM = (1 - 2 / c) / n
    t = np.minimum(tD, tM)
    t[alpha <= 0] = -np.inf
    k = np.unravel_index(np.argmax(t), t.shape)
    return float(t[k]), float(x[k[0], 0]), float(c[0, k[1]])


def tau_star(n, kappa):
    return 2 / (4 * kappa + n + math.sqrt(n * n + 16 * kappa * kappa))


def part_a():
    print("=== (a) meilleur (gamma, Delta) dans la preuve ===")
    print("%-9s %-9s %-12s %-12s %-10s %-10s %-10s %s" % ("n", "kappa", "tau grille", "tau* formule", "Q=tau/livre", "gamma R^2", "pas Hofmann", "c=Delta/gamma^2"))
    Qmax, Qmax_ill = 0.0, 0.0
    for n, kappa in ((1000, 10.0), (1000, 100.0), (1000, 562.5), (1000, 1e3), (1000, 1e4), (1000, 1e5), (10, 1e6),
                     (100000, 2.0), (10 ** 7, 1.5), (20, 20.0)):
        tg, xg, cg = tau_grid(n, kappa)
        ts = tau_star(n, kappa)
        book = min(1 / (3 * n), 3 / (16 * kappa))
        K = 4 * kappa / n
        a_h = 2 * K / (1 + K + math.sqrt(1 + K * K))          # a*(K) de Hofmann et al. (transcription de la note B3), gamma* = a*/(4L)
        print("%-9d %-9.4g %-12.5e %-12.5e %-10.4f %-10.5f %-10.5f %.4g" % (n, kappa, tg, ts, tg / book, xg, a_h / 4, cg))
        assert abs(tg / ts - 1) < 2e-3
        Qmax = max(Qmax, ts / book)
        if kappa >= 9 * n / 16:
            Qmax_ill = max(Qmax_ill, ts / book)
    s = np.exp(np.linspace(math.log(1e-8), math.log(1e8), 200001))
    Q = np.where(s <= 16 / 9, 32 / (3 * (4 + s + np.sqrt(s * s + 16))), 6 * s / (4 + s + np.sqrt(s * s + 16)))
    print("sup_s Q(s) = %.6f (-> 3) ; sup_{s<=16/9} Q = %.6f (-> 4/3 = %.6f) ; min Q = %.4f en s = %.4f ; Q(1) = %.4f (kappa = n : +%.1f %%)"
          % (Q.max(), Q[s <= 16 / 9].max(), 4 / 3, Q.min(), s[np.argmin(Q)], 32 / (3 * (5 + math.sqrt(17))), 100 * (32 / (3 * (5 + math.sqrt(17))) - 1)))
    print("plus grand pas admis par la preuve : alpha > 0 et Delta > 2 gamma^2  =>  gamma R^2 < 1/2 (borne superieure non atteinte) ; 4 / (1/2) = 8")


def part_b():
    print("\n=== (b) (C5) kappa_P >= kappa pour toute metrique P ===")
    rng = np.random.default_rng(1)
    n, d = 20, 5
    X = rng.standard_normal((n, d)) * (np.arange(1, d + 1) ** -0.7)[None, :]
    y = rng.choice([-1.0, 1.0], n)
    lam = 1e-2
    R2 = 0.25 * (X ** 2).sum(1).max() + lam
    kappa = R2 / lam
    print("kappa = R^2/mu = %.4f" % kappa)

    def hessF(th):
        m = y * (X @ th)
        w = expit(m) * expit(-m)
        return (X * w[:, None]).T @ X / n + lam * np.eye(d)

    def consts(P):
        w, U = np.linalg.eigh(P)
        Pih = U @ np.diag(w ** -0.5) @ U.T                    # P^{-1/2}
        Pinv = U @ np.diag(1 / w) @ U.T
        R2P_paper = 0.25 * np.einsum("ij,jk,ik->i", X, Pinv, X).max() + lam / w[0]
        R2P_exact = max(np.linalg.eigvalsh(Pih @ (0.25 * np.outer(x, x) + lam * np.eye(d)) @ Pih)[-1] for x in X)
        muP = lam / w[-1]
        return R2P_paper, R2P_exact, muP, Pih

    worst_p, worst_e, worst_gap = np.inf, np.inf, 0.0
    for trial in range(4000):
        A = rng.standard_normal((d, d)) * rng.choice([0.1, 1.0, 3.0])
        P = A @ A.T + 10 ** rng.uniform(-3, 1) * np.eye(d)
        if trial % 4 == 0:                                    # perturbations de l'identite et metriques de covariance
            P = np.eye(d) + 10 ** rng.uniform(-4, -1) * (A + A.T)
        if trial % 4 == 1:
            P = X.T @ X / n * rng.uniform(0.01, 1) + lam * 10 ** rng.uniform(-1, 2) * np.eye(d)
        if np.linalg.eigvalsh(P)[0] <= 0:
            continue
        rp, re, mp, _ = consts(P)
        assert rp >= re * (1 - 1e-12)
        worst_p = min(worst_p, rp / mp / kappa)
        worst_e = min(worst_e, re / mp / kappa)
        worst_gap = max(worst_gap, rp / re)
    print("4000 metriques P : min kappa_P/kappa avec R_P^2 du papier = %.6f ; avec la constante exacte = %.6f (>= 1 attendu)" % (worst_p, worst_e))
    print("R_P^2 du papier / constante de regularite exacte dans la metrique P : toujours >= 1, max observe %.3f (majorant, exact pour P = I)" % worst_gap)
    # mu_P exact = lambda / lambda_max(P) : infimum de lambda_min(P^{-1/2} F'' P^{-1/2}) le long de t * theta_1
    A = rng.standard_normal((d, d))
    P = A @ A.T + 0.3 * np.eye(d)
    rp, re, mp, Pih = consts(P)
    th1 = rng.standard_normal(d)
    for t in (1, 10, 100, 1000):
        print("  t=%5d : lambda_min(P^-1/2 F''(t th1) P^-1/2) / (lambda/lmax(P)) - 1 = %.3e" % (t, np.linalg.eigvalsh(Pih @ hessF(t * th1) @ Pih)[0] / mp - 1))
    # contre-exemple explicite a l'exactitude de la formule R_P^2 : P = diag(1,4), x = (0,1), c = 1, lambda = 0.1
    c, l = 1.0, 0.1
    Pinv = np.diag([1.0, 0.25])
    x = np.array([0.0, 1.0])
    u = np.sqrt(Pinv) @ x
    exact = np.linalg.eigvalsh(c * np.outer(u, u) + l * Pinv)[-1]
    print("exemple P = diag(1,4), x = (0,1), c = 1, lambda = 0.1 : formule du papier %.3f, constante exacte %.3f" % (c * x @ Pinv @ x + l * 1.0, exact))


if __name__ == "__main__":
    part_a()
    part_b()
