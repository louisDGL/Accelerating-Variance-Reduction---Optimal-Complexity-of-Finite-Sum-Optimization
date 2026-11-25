"""Audit math-schedule, controle 2 : chiffres des remarques (l. 577-581) et de la phrase F15 (l. 648-650) du papier,
recalcules sur les instances du depot (lecture seule de muon/lab et de muon/results/M1_summary.json).

(i)   Constantes de soft-cond et ls-cond (n = 200, K x d = 8 x 40, kappa = 1e4, alpha = 2, graine 0) et refroidissement
      certifie par la Proposition 5 (b) : q_cert = rho^(n/(2b)) par passe, rapport log(1/0.8) / log(1/q_cert), pour
      l'ordre cyclique (A = n), la permutation fraiche (A = 2n - 1) et les lots independants
      (A = ceil(n log(nT/delta)), T = 300 passes) ; longueur initiale prescrite eta_0 contre lr_0 = 0.1.
(ii)  Le long d'un run de la recette a q = 0.8 (facteur polaire exact, b = 1, lr_0 = 0.1, tirages independants,
      graine 10) : E_j ||omega_j - grad F||_* / ||grad F||_* aux passes 2, 6, ..., 26 (enumeration des 200 indices),
      E_j D_+ / D, part des indices dont le pas augmente F, part des indices avec <grad F, polar(omega_j)> < 0.
(iii) F15 : 1 / (14 n L_max) contre les pas regles de "SAGA-Muon (joint)" dans M1_summary.json ; hypothese
      gamma L_op <= 1.

Lancer depuis la racine du projet :
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 paper/audit/checks/math-schedule_benchmark_numbers.py
"""
import copy
import json
import math
import os
import sys
import time

sys.dont_write_bytecode = True                       # lecture seule sur muon/
ROOT = os.getcwd()
sys.path.insert(0, os.path.join(ROOT, "muon"))
sys.path.insert(0, ROOT)

import numpy as np                                   # noqa: E402
from lab import problems, optimum, vrmuon as vm      # noqa: E402


def nuc(M):
    return float(np.linalg.svd(M, compute_uv=False).sum())


def opn(M):
    return float(np.linalg.svd(M, compute_uv=False)[0])


def build(loss, n, d, K, kappa, alpha, seed):
    mk = problems.make_softmax if loss == "softmax" else problems.make_matrix_ls
    prob = mk(n, d, K, kappa=kappa, alpha=alpha, seed=seed)
    optimum.solve_optimum(prob)
    return prob


def constants(prob):
    Li = prob.c_loss * prob.sqnorms                  # constante de la partie perte g_i (sans le regularisateur)
    mu = float(prob.lam + (prob.cov_eigs[0] if prob.loss == "ls" else 0.0))
    return Li, float(Li.max()), float(Li.mean()), mu, float(prob.L_op)


def main():
    t0 = time.process_time()
    q_obs, lr0 = 0.8, 0.1
    M1 = json.load(open(os.path.join(ROOT, "muon", "results", "M1_summary.json")))
    print("=== (i) refroidissement certifie contre q = 0.8 ===")
    probs = {}
    for name in ("soft-cond", "ls-cond"):
        s = M1[name]["spec"]
        prob = build(s["loss"], s["n"], s["d"], s["K"], s["kappa"], s["alpha"], s["seed"])
        probs[name] = prob
        n = prob.n
        Li, Lmax, Lbar, mu, Lop = constants(prob)
        W0 = np.zeros((prob.K, prob.d))
        D0 = prob.Fgap(W0)
        print(f"{name}: spec={s}  lam={prob.lam:.4g} L_max={Lmax:.4g} L_bar={Lbar:.4g} L_op={Lop:.4g} "
              f"mu_op={mu:.4g} (mu_op <= L_max: {mu <= Lmax}) L_max/L_op={Lmax / Lop:.3f} L_max/mu_op={Lmax / mu:.4g}")
        rows = [("cyclique", n), ("permutation fraiche", 2 * n - 1)]
        T = 300 * n
        for delta in (0.05, 1e-2):
            rows.append((f"independants T=300 passes delta={delta:g}", math.ceil(n * math.log(n * T / delta))))
        for label, A in rows:
            for form, Leff in (("L_op+10 A L_max", Lop + 10 * A * Lmax),):
                rho = 1 - mu / Leff
                log_q = (n / 2) * math.log1p(-mu / Leff)              # log du facteur de la longueur par passe
                eta0 = math.sqrt(2 * mu * D0) / Leff
                print(f"   {label:42s} A={A:6d}: 1-rho={mu / Leff:.3e}, q_cert=1-{-math.expm1(log_q):.3e}, "
                      f"log(1/0.8)/log(1/q_cert)={math.log(q_obs) / log_q:.3g} ; eta_0 prescrit={eta0:.3e} "
                      f"(lr_0 regle 0.1 a 0.3 : {lr0 / eta0:.3g} a {0.3 / eta0:.3g} fois plus grand)")
    print(f"CPU {time.process_time() - t0:.1f} s")

    print("=== (ii) run de la recette a q = 0.8 : erreur de l'estimateur contre gradient ===")
    for name, prob in probs.items():
        n = prob.n
        Li, Lmax, Lbar, mu, Lop = constants(prob)
        algo = vm.VRMuon(prob, estimator="saga", length="const", lr=lr0, schedule="geometric", lr_decay=q_obs)
        W = np.zeros((prob.K, prob.d))
        W0 = W.copy()
        st = algo.init_state(W)
        batches = [B for B, _ in vm._draws(np.random.default_rng(10), n, 1, 28 * n, False, None, "iid")]
        at = set(p * n for p in (2, 6, 10, 14, 18, 22, 26))
        Phi = np.repeat(W[None], n, axis=0)
        t_ref = np.zeros(n, dtype=int)
        noise, fac, up, neg, ages, chk = [], [], [], [], [], []
        for t, B in enumerate(batches):
            if t in at:
                G = prob.grad(W)
                nG, D = nuc(G), prob.Fgap(W)
                x = np.empty(n)
                dF = np.empty(n)
                ip = np.empty(n)
                eta = lr0 * q_obs ** (t / n) * algo.scale
                l_true = np.linalg.svd(W[None] - Phi, compute_uv=False)[:, 0]
                for j in range(n):
                    m, c, aux = algo.pieces(W, st, j)
                    S, info = algo.step(W, m, c, st, t)
                    x[j] = nuc(m + c - G)
                    dF[j] = prob.F_diff(W - S, W, G0=G)
                    ip[j] = float(np.vdot(G, S)) / eta
                    assert abs(opn(S) - eta) <= 1e-9 * eta           # pas de longueur spectrale eta_t exactement
                    assert x[j] <= (Li[j] * l_true[j] + float((Li * l_true).mean())) * (1 + 1e-8)   # F12
                    assert x[j] <= (Lmax + Lbar) * l_true.max() * (1 + 1e-8)                        # Prop. 5 (a)
                noise.append(float(x.mean()) / nG)
                fac.append(1 + float(dF.mean()) / D)
                up.append(float((dF > 0).mean()))
                neg.append(float((ip < 0).mean()))
                ages.append(int((t - t_ref).max()))
            m, c, aux = algo.pieces(W, st, B)
            S, info = algo.step(W, m, c, st, t)
            algo.commit(W, st, B, 1.0, aux, info)
            Phi[B], t_ref[B] = W, t
            W = algo.advance(W, S, st)
        rel = float(np.sum((W - prob.W_star) ** 2) / np.sum((W0 - prob.W_star) ** 2))
        print(f"{name}: E_j||omega_j - grad F||_* / ||grad F||_* aux 7 etats = "
              f"[{', '.join(f'{v:.2f}' for v in noise)}] -> intervalle [{min(noise):.2f}, {max(noise):.2f}]")
        print(f"   E_j D_+ / D = [{', '.join(f'{v:.4f}' for v in fac)}] ; part des indices j dont le pas AUGMENTE F = "
              f"[{', '.join(f'{v:.2f}' for v in up)}] ; part des j avec <grad F, polar(omega_j)> < 0 = "
              f"[{', '.join(f'{v:.2f}' for v in neg)}] ; plus grand age aux etats = {ages} (n = {n}) ; distance "
              f"relative finale au carre apres 28 passes = {rel:.1e}")
    print(f"CPU {time.process_time() - t0:.1f} s")

    print("=== (iii) F15 : 1/(14 n L_max) contre les pas regles (M1_summary.json, 'SAGA-Muon (joint)') ===")
    for name in M1:
        row = [r for r in M1[name]["rows"] if r.get("family") == "SAGA-Muon (joint)"]
        if not row:
            continue
        best = row[0]["best"]["gamma"]
        s = M1[name]["spec"]
        prob = probs.get(name)
        if prob is None:
            mk = problems.make_softmax if s["loss"] == "softmax" else problems.make_matrix_ls
            prob = mk(s["n"], s["d"], s["K"], kappa=s["kappa"], alpha=s["alpha"], seed=s["seed"])   # constantes seules
        Li = prob.c_loss * prob.sqnorms
        Lmax, Lbar = float(Li.max()), float(Li.mean())
        g_cert = 1.0 / (14 * prob.n * Lmax)
        unit = float(getattr(prob, best["over"]))
        g_tuned = best["c"] / unit
        print(f"{name:10s} n={prob.n} K x d={prob.K}x{prob.d}: 1/(14 n L_max) = {g_cert * prob.R2_op:.3e}/R2_op ; pas "
              f"regle = {best['c']}/{best['over']} -> rapport {g_tuned / g_cert:.0f} ; gamma_cert L_op = "
              f"{g_cert * prob.L_op:.2e} (<= 1 : {g_cert * prob.L_op <= 1}) ; lam r = {prob.lam * prob.r:.3g} <= "
              f"(14 n - 1) L_max = {(14 * prob.n - 1) * Lmax:.3g} : {prob.lam * prob.r <= (14 * prob.n - 1) * Lmax} ; "
              f"L_op <= L_bar + lam r : {prob.L_op <= Lbar + prob.lam * prob.r + 1e-12}")
    print(f"CPU {time.process_time() - t0:.1f} s")


if __name__ == "__main__":
    main()
