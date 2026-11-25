"""Audit math-saga, controle 2 : borne inferieure de (C4) (papier l. 321-324).

n composantes identiques f_i(theta) = (mu theta_1^2 + R^2 theta_2^2)/2, eta* = 0, SAGA a pas constant gamma.
Les deux coordonnees sont decouplees et la dynamique est lineaire : pour une coordonnee de courbure q, l'etat
s = (theta, z_1..z_n) suit s' = A_j s (j uniforme), donc
   E[s]      : m'     = Abar m,
   E[s s^T]  : Sigma' = (1/n) sum_j A_j Sigma A_j^T          (recurrence EXACTE, pas de Monte-Carlo).
E||theta_t||^2 = somme sur les deux coordonnees de Sigma_t[0,0].

On compare le premier t tel que E||theta_t||^2 <= eps ||theta_0||^2 a N = (kappa-1)/4 * log(1/(2 eps)),
pour plusieurs pas, points initiaux et initialisations de la table, et on verifie E[theta_t] = (I - gamma Q)^t theta_0.

Lancer depuis la racine du depot :
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 paper/audit/checks/math-saga_check2_identical.py
"""
import math
import numpy as np


def maps(n, q, gamma):
    """A_j (j = 0..n-1) agissant sur s = (theta, z_1..z_n) pour une coordonnee de courbure q."""
    A = np.zeros((n, n + 1, n + 1))
    for j in range(n):
        Aj = np.eye(n + 1)
        # theta' = theta - gamma (q theta - z_j + zbar)
        Aj[0, 0] = 1 - gamma * q
        Aj[0, 1:] = -gamma / n
        Aj[0, 1 + j] += gamma
        # z_j' = q theta
        Aj[1 + j, :] = 0.0
        Aj[1 + j, 0] = q
        A[j] = Aj
    return A


def run(n, mu, R2, gamma, theta0, table, T, rng=None):
    """Retourne E||theta_t||^2 (t = 0..T) et E[theta_t] par recurrence exacte.
    table : 'grad0' (z_i = f_i'(theta_0)), 'zero' (z_i = 0 = f_i'(eta*)), 'rand' (deterministe arbitraire)."""
    e2 = np.zeros(T + 1)
    mean = np.zeros((T + 1, 2))
    for k, q in enumerate((mu, R2)):
        A = maps(n, q, gamma)
        Abar = A.mean(0)
        s0 = np.zeros(n + 1)
        s0[0] = theta0[k]
        if table == "grad0":
            s0[1:] = q * theta0[k]
        elif table == "zero":
            s0[1:] = 0.0
        else:
            s0[1:] = rng.standard_normal(n) * q
        Sig = np.outer(s0, s0)
        m = s0.copy()
        e2[0] += Sig[0, 0]
        mean[0, k] = m[0]
        for t in range(1, T + 1):
            Sig = np.einsum("jab,bc,jdc->ad", A, Sig, A, optimize=True) / n
            m = Abar @ m
            e2[t] += Sig[0, 0]
            mean[t, k] = m[0]
            if not np.isfinite(e2[t]) or e2[t] > 1e200:
                e2[t:] = np.inf
                break
    return e2, mean


def first_hit(e2, thr):
    idx = np.nonzero(e2 <= thr)[0]
    return int(idx[0]) if len(idx) else None


def main():
    rng = np.random.default_rng(0)
    print("=== A. Moyenne : E[theta_t] = (I - gamma Q)^t theta_0, quelle que soit la table ===")
    worst = 0.0
    for n in (1, 3, 8):
        for table in ("grad0", "zero", "rand"):
            for g in (0.25, 1.0, 1.7):
                mu, R2 = 0.05, 1.0
                th0 = np.array([0.7, -1.3])
                e2, mean = run(n, mu, R2, g / R2, th0, table, 60, rng)
                ref = np.stack([(1 - g / R2 * mu) ** np.arange(61) * th0[0], (1 - g) ** np.arange(61) * th0[1]], 1)
                worst = max(worst, np.abs(mean - ref).max())
    print("ecart max |E theta_t - (I-gamma Q)^t theta_0| = %.2e" % worst)

    print("\n=== B. theta_0 = (r, r) : premier t avec E||theta_t||^2 <= eps ||theta_0||^2, contre N = (kappa-1)/4 log(1/(2 eps)) ===")
    print("   (la borne du papier doit tenir : t_hit >= N ; '-' = non atteint en T iterations, ce qui respecte aussi la borne)")
    viol = 0
    cases = 0
    min_ratio = np.inf
    for kappa in (3.0, 10.0, 40.0):
        mu, R2 = 1.0 / kappa, 1.0
        for n in (1, 2, 5, 10):
            for table in ("grad0", "zero", "rand"):
                for g in (0.02, 0.1, 0.25, 0.5, 1.0, 2 / (1 + 1 / kappa), 1.9, 2.5):
                    T = int(40 * kappa) + 60 * n
                    th0 = np.array([1.0, 1.0])
                    e2, _ = run(n, mu, R2, g / R2, th0, table, T, rng)
                    # borne de la preuve : E||theta_t||^2 >= ((kappa-1)/(kappa+1))^{2t} * ||theta_0||^2 / 2
                    tt = np.arange(T + 1)
                    lb = ((kappa - 1) / (kappa + 1)) ** (2 * tt) * 1.0
                    fin = np.isfinite(e2)
                    if np.any(e2[fin] < lb[fin] * (1 - 1e-9)):
                        viol += 1
                    for eps in (1e-1, 1e-2, 1e-4):
                        N = (kappa - 1) / 4 * math.log(1 / (2 * eps))
                        h = first_hit(e2, eps * 2.0)
                        cases += 1
                        if h is not None:
                            min_ratio = min(min_ratio, h / N)
                            if h < N:
                                viol += 1
    print("cas testes : %d ; violations : %d ; plus petit rapport t_hit / N observe : %.3f" % (cases, viol, min_ratio))

    print("\n=== C. Autres points initiaux : l'enonce du papier ne fixe pas theta_0 ===")
    print("%-6s %-3s %-14s %-9s %-7s %-7s %-9s %-10s %s" % ("kappa", "n", "theta_0", "gamma*R2", "table", "eps", "t_hit", "N(papier)", "N corrige = (kappa-1)/4 log(m/eps), m = min(th0_k^2)/||th0||^2"))
    for kappa in (10.0, 100.0):
        mu, R2 = 1.0 / kappa, 1.0
        for n in (5,):
            for th0 in ((0.0, 1.0), (0.1, 1.0), (0.3, 1.0), (1.0, 0.1), (1.0, 1.0)):
                for g, table in ((1.0, "grad0"), (0.25, "grad0"), (1.0, "zero")):
                    T = int(60 * kappa)
                    th = np.array(th0)
                    e2, _ = run(n, mu, R2, g / R2, th, table, T, rng)
                    for eps in (1e-2,):
                        N = (kappa - 1) / 4 * math.log(1 / (2 * eps))
                        m = min(th[0] ** 2, th[1] ** 2) / (th ** 2).sum()
                        Nc = (kappa - 1) / 4 * math.log(m / eps) if m > 0 else float("-inf")
                        h = first_hit(e2, eps * (th ** 2).sum())
                        flag = "  <-- VIOLE l'enonce du papier" if (h is not None and h < N) else ""
                        print("%-6.0f %-3d %-14s %-9.3f %-7s %-7.0e %-9s %-10.1f %.1f%s" % (kappa, n, str(th0), g, table, eps, h, N, Nc, flag))


if __name__ == "__main__":
    main()
