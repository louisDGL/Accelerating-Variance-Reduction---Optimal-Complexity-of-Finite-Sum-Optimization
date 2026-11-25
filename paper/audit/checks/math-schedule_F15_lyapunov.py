"""Audit math-schedule, controle 3 : fait F15 (muon/notes_theory.md), cite aux lignes 648-650 du papier.

Pas a longueur nucleaire S_j = gamma ||omega_j||_* polar(omega_j), lot un, indice uniforme, table SAGA
z_i = grad g_i(phi_i) ; Theta = (1/n) sum_i ||W - phi_i||_op^2, Psi = D(W) + Theta / (8 n gamma).
Enonce : si gamma <= 1/(14 n L_max) et gamma L_op <= 1,
    E_j[Psi_+] <= (1 - min{(1 - gamma L_op) gamma mu_op / 2, 1/(4n)}) Psi.
Controle par enumeration exacte des n indices, en des etats (W, phi_1..phi_n) aleatoires et le long de runs, sur le
probleme de moindres carres du controle 1 (K = 3, d = 4, n = 12) ; inegalites intermediaires (I1), (I2), (I3).
Aussi : la constante 10 de la forme close L_eff = L_op + 10 A L_max de la Proposition 5 (b).

Lancer depuis la racine du projet :
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 paper/audit/checks/math-schedule_F15_lyapunov.py
"""
import importlib
import math
import os
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sim = importlib.import_module("math-schedule_prop5_sim")
LS, polar, nuc, opn, K, d, n = sim.LS, sim.polar, sim.nuc, sim.opn, sim.K, sim.d, sim.n


def one_state(p, W, Phi, gamma, theta):
    """E_j Psi_+ / Psi exact et controles (I1)-(I3) en un etat. Renvoie (rapport, facteur certifie, nb de violations)."""
    X, Y = p.X, p.Y
    R = np.einsum("ikd,id->ki", Phi, X) - Y.T              # residus stockes aux points phi_i  (K x n)
    zbar = R @ X / n
    G = p.grad(W)
    nG, D = nuc(G), p.gap(W)
    l = np.linalg.svd(W[None] - Phi, compute_uv=False)[:, 0]
    Theta = float((l ** 2).mean())
    Psi = D + Theta / (8 * n * gamma)
    a = 1 - gamma * p.L_op
    viol = 0
    Psi_plus, x2, S2 = np.empty(n), np.empty(n), np.empty(n)
    for j in range(n):
        c = np.outer(W @ X[j] - Y[j] - R[:, j], X[j])
        omega = zbar + p.lam * W + theta * c
        s = nuc(omega)
        S = gamma * s * polar(omega)
        x = nuc(omega - G)
        Wp = W - S
        Dp = p.gap(Wp)
        Phi_p = Phi.copy()
        Phi_p[j] = W
        lp = np.linalg.svd(Wp[None] - Phi_p, compute_uv=False)[:, 0]
        Psi_plus[j] = Dp + float((lp ** 2).mean()) / (8 * n * gamma)
        x2[j], S2[j] = x ** 2, (gamma * s) ** 2
        tol = 1e-10 * (abs(D) + gamma * (s * s + x * x + nG * nG))
        # (I1) D_+ - D <= -||S_j||_op^2/(4 gamma) - (gamma a/4) ||G||_*^2 + (3 gamma/2) x^2
        if Dp - D > -S2[j] / (4 * gamma) - gamma * a / 4 * nG ** 2 + 1.5 * gamma * x2[j] + tol:
            viol += 1
        # F12 a b = 1 : x <= theta L_j l_j + (1/n) sum_i L_i l_i
        if x > (theta * p.Li[j] * l[j] + float((p.Li * l).mean())) * (1 + 1e-9) + 1e-13 * nG:
            viol += 1
    # (I2) E_j x^2 <= 4 L_max^2 Theta
    if x2.mean() > 4 * p.Lmax ** 2 * Theta * (1 + 1e-9) + 1e-26:
        viol += 1
    cert = 1 - min(a * gamma * p.mu_op / 2, 1 / (4 * n))
    return float(Psi_plus.mean()) / Psi, cert, viol


def main():
    t0 = time.process_time()
    rng = np.random.default_rng(3)
    print("=== F15 : E_j Psi_+ <= (1 - min{(1 - gamma L_op) gamma mu_op / 2, 1/(4n)}) Psi, enumeration exacte ===")
    for pseed in (1, 2):
        p = LS(pseed)
        g_max = 1.0 / (14 * n * p.Lmax)
        print(f"probleme {pseed}: gamma_max = 1/(14 n L_max) = {g_max:.3e}, gamma_max L_op = {g_max * p.L_op:.3e} "
              f"(<= 1), mu_op <= 7 L_max : {p.mu_op <= 7 * p.Lmax}")
        for mult in (1.0, 0.25, 4.0, 16.0, 64.0, 350.0, 1400.0):
            gamma = mult * g_max
            worst, nviol, nfail, N = -np.inf, 0, 0, 0
            for trial in range(250):
                theta = float(rng.choice([1.0, 1.0 / n, 0.5]))
                W = p.Wstar + rng.standard_normal((K, d)) * 10 ** rng.uniform(-3, 1)
                mode = trial % 3
                if mode == 0:                                # table dispersee, ecarts quelconques
                    Phi = W[None] + rng.standard_normal((n, K, d)) * 10 ** rng.uniform(-4, 1, size=(n, 1, 1))
                elif mode == 1:                              # ecarts de rang un alignes sur x_i
                    Phi = np.array([W - 10 ** rng.uniform(-3, 0.5) * np.outer(rng.standard_normal(K), p.X[i])
                                    for i in range(n)])
                else:                                        # table fraiche (Theta = 0)
                    Phi = np.repeat(W[None], n, axis=0)
                if gamma * p.L_op > 1:
                    continue
                ratio, cert, v = one_state(p, W, Phi, gamma, theta)
                N += 1
                worst = max(worst, ratio - cert)
                nfail += int(ratio > cert * (1 + 1e-10))
                nviol += v if mult <= 1.0 else 0
            tag = "dans l'hypothese" if mult <= 1.0 else "HORS hypothese"
            print(f"   gamma = {mult:g} gamma_max ({tag}) : {N} etats, max (E_j Psi_+/Psi - facteur certifie) = "
                  f"{worst:+.3e}, etats ou l'inegalite finale echoue : {nfail}"
                  + (f", violations de (I1)/(I2)/F12 : {nviol}" if mult <= 1.0 else ""))
    # le long d'un run a gamma_max (table initialisee en W_0, indices independants)
    p = LS(1)
    gamma = 1.0 / (14 * n * p.Lmax)
    W = np.random.default_rng(101).standard_normal((K, d))
    Phi = np.repeat(W[None], n, axis=0)
    rr = np.random.default_rng(9)
    worst, nfail, gain = -np.inf, 0, []
    for t in range(3000):
        if t % 25 == 0:
            ratio, cert, v = one_state(p, W, Phi, gamma, 1.0)
            worst = max(worst, ratio - cert)
            nfail += int(ratio > cert * (1 + 1e-10)) + v
            gain.append((1 - ratio) / (1 - cert))
        j = int(rr.integers(n))
        R = np.einsum("ikd,id->ki", Phi, p.X) - p.Y.T
        omega = R @ p.X / n + p.lam * W + np.outer(W @ p.X[j] - p.Y[j] - R[:, j], p.X[j])
        S = gamma * nuc(omega) * polar(omega)
        Phi[j] = W
        W = W - S
    print(f"run a gamma_max (3000 iterations, 120 etats) : echecs = {nfail}, max ecart = {worst:+.3e}, decroissance "
          f"exacte / certifiee entre {min(gain):.1f} et {max(gain):.1f}")

    print("=== Proposition 5 (b) : constante 10 de L_eff = L_op + 10 A L_max ===")
    # pire cas : theta = 1, L_bar = L_max, mu_op = L_max, L_op = mu_op (plus petit rho possible sous mu_op <= L_max)
    worst = 0.0
    for A in list(range(1, 400)) + [10 ** 3, 10 ** 4, 10 ** 5]:
        rho = 1 - 1.0 / (1 + 10 * A)
        sig = sum(rho ** (-k / 2) for k in range(1, A + 1)) if A < 400 else A * rho ** (-A / 2)
        worst = max(worst, 4 * 2 * sig / (10 * A))
    print(f"   max_A 4 (theta L_max + L_bar) sum_(k<=A) rho^(-k/2) / (10 A L_max) au pire cas = {worst:.4f} (<= 1 requis) ; "
          f"L_eff >= 11 mu_op >= 2 mu_op")
    print(f"CPU {time.process_time() - t0:.1f} s")


if __name__ == "__main__":
    main()
