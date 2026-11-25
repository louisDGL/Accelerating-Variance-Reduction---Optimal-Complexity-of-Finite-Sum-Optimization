"""Audit math-nfl, controle (d) : identite (2) du pas polaire a longueur nucleaire.

  (F(W+) - F*) / (F(W) - F*) = 1 - (mu_PL / L) (2 - J / L),   W+ = W - ||G||_* O / L,  O = polar(G),
  J = <O, H O>,  mu_PL = ||G||_*^2 / (2 (F(W) - F*)).

Verifie, sur des quadratiques matricielles aleatoires :
  (i)   l'identite, pour trois familles de hessiennes (unilaterale Delta -> Delta Sigma ; Kronecker
        Delta -> B Delta A ; operateur SPD general sur R^{K x d}) et plusieurs L ;
  (ii)  J dans [mu_op, L_op] et mu_PL dans [mu_op, J] quand mu_op, L_op sont connus en forme close
        (unilaterale : mu_op = lmin(Sigma), L_op = somme des min(K,d) plus grandes valeurs propres ;
        Kronecker : mu_op = a_d b_K, L_op = sum_{k<=r} a_k b_k) ; mu_PL <= J dans tous les cas ;
  (iii) le facteur certifie 1 - 1/kappa_op = valeur de (2) en J = L_op, mu_PL = mu_op, et c'est bien un majorant ;
  (iv)  extension non quadratique (softmax ridge) : la meme identite avec
        Jt = 2 int_0^1 (1 - s) D^2F(W - s eta O)[O, O] ds a la place de J.

Lancer depuis la racine du depot :
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 paper/audit/checks/math-nfl_identity2.py
"""
import numpy as np
from scipy.optimize import minimize

rng = np.random.default_rng(20261120)


def polar(G, tol=1e-12):
    U, s, Vt = np.linalg.svd(G, full_matrices=False)
    keep = s > tol * max(s[0], 1e-300)
    return U[:, keep] @ Vt[keep], s[keep].sum()


def rand_spd(m, cond):
    Q, _ = np.linalg.qr(rng.standard_normal((m, m)))
    ev = np.exp(rng.uniform(0, np.log(cond), m))
    return (Q * ev) @ Q.T, np.sort(ev)[::-1]


worst_id = 0.0
viol = 0
n_cases = 0
min_gap_J_lo = np.inf; min_gap_J_hi = np.inf; min_gap_pl_lo = np.inf; min_gap_pl_hi = np.inf
worst_cert = np.inf

for (K, d) in [(6, 6), (10, 4), (4, 10), (1, 5), (5, 1), (3, 7)]:
    r = min(K, d)
    for fam in ["onesided", "kron", "general"]:
        for rep in range(40):
            if fam == "onesided":
                Sig, ev = rand_spd(d, 10 ** rng.uniform(0, 4))
                H = lambda D, Sig=Sig: D @ Sig
                mu_op, L_op = ev[-1], ev[:r].sum()
            elif fam == "kron":
                A, a = rand_spd(d, 10 ** rng.uniform(0, 3))
                B, b = rand_spd(K, 10 ** rng.uniform(0, 3))
                H = lambda D, A=A, B=B: B @ D @ A
                mu_op, L_op = a[-1] * b[-1], (a[:r] * b[:r]).sum()
            else:
                M, _ = rand_spd(K * d, 10 ** rng.uniform(0, 3))
                H = lambda D, M=M: (M @ D.reshape(-1)).reshape(K, d)
                mu_op = L_op = None
            Wstar = rng.standard_normal((K, d))
            E = rng.standard_normal((K, d))
            if rep % 5 == 0 and r >= 2:          # erreur de rang deficient : gradient de rang < r dans les cas structures
                E = rng.standard_normal((K, 1)) @ rng.standard_normal((1, d))
            F = lambda W: 0.5 * np.sum((W - Wstar) * H(W - Wstar))
            W = Wstar + E
            G = H(E)
            O, nuc = polar(G)
            J = np.sum(O * H(O))
            gap = F(W)
            mu_pl = nuc ** 2 / (2 * gap)
            Ls = [J, 2 * J, 0.7 * J] + ([L_op] if L_op is not None else [])
            for L in Ls:
                lhs = F(W - nuc * O / L) / gap
                rhs = 1 - (mu_pl / L) * (2 - J / L)
                worst_id = max(worst_id, abs(lhs - rhs))
            n_cases += 1
            if mu_pl > J * (1 + 1e-10):
                viol += 1
            if L_op is not None:
                min_gap_J_lo = min(min_gap_J_lo, J / mu_op - 1)
                min_gap_J_hi = min(min_gap_J_hi, 1 - J / L_op)
                min_gap_pl_lo = min(min_gap_pl_lo, mu_pl / mu_op - 1)
                min_gap_pl_hi = min(min_gap_pl_hi, 1 - mu_pl / J)
                exact = 1 - (mu_pl / L_op) * (2 - J / L_op)
                cert = 1 - mu_op / L_op
                worst_cert = min(worst_cert, cert - exact)

print("(i)  identite (2) : %d cas, ecart absolu max |gauche - droite| = %.2e" % (n_cases, worst_id))
print("(ii) mu_PL <= J viole dans %d cas ; marges minimales (>= 0 attendu a l'arrondi pres) :" % viol)
print("     J/mu_op - 1 >= %.3e ; 1 - J/L_op >= %.3e ; mu_PL/mu_op - 1 >= %.3e ; 1 - mu_PL/J >= %.3e"
      % (min_gap_J_lo, min_gap_J_hi, min_gap_pl_lo, min_gap_pl_hi))
print("(iii) certifie - exact (>= 0 attendu) : min = %.3e" % worst_cert)

# ---- (iv) extension non quadratique : softmax ridge, Jt = 2 int (1-s) D^2F(W - s eta O)[O,O] ds
K, d, n, lam = 4, 6, 40, 1e-2
X = rng.standard_normal((n, d)); y = rng.integers(0, K, n)


def Fsm(Wv):
    W = Wv.reshape(K, d); Z = X @ W.T
    m = Z.max(1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(Z - m).sum(1))
    return np.mean(lse - Z[np.arange(n), y]) + 0.5 * lam * np.sum(W * W)


def Gsm(Wv):
    W = Wv.reshape(K, d); Z = X @ W.T
    P = np.exp(Z - Z.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    P[np.arange(n), y] -= 1
    return (P.T @ X / n + lam * W).reshape(-1)


def D2(W, Dl):
    Z = X @ W.T
    P = np.exp(Z - Z.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    V = X @ Dl.T
    return np.mean((P * V * V).sum(1) - (P * V).sum(1) ** 2) + lam * np.sum(Dl * Dl)


res = minimize(Fsm, np.zeros(K * d), jac=Gsm, method="L-BFGS-B", options=dict(gtol=1e-13, ftol=1e-16, maxiter=5000))
Fstar = res.fun
xs, ws = np.polynomial.legendre.leggauss(60)
s_nodes = 0.5 * (xs + 1); s_w = 0.5 * ws
worst_nq = 0.0
for rep in range(20):
    W = rng.standard_normal((K, d)) * rng.uniform(0.1, 3)
    G = Gsm(W.reshape(-1)).reshape(K, d)
    O, nuc = polar(G)
    gap = Fsm(W.reshape(-1)) - Fstar
    mu_pl = nuc ** 2 / (2 * gap)
    for L in [0.5 * np.linalg.norm(X, 2) ** 2 / n + lam * min(K, d), 0.3, 1.0]:
        eta = nuc / L
        Jt = 2 * sum(w * (1 - s) * D2(W - s * eta * O, O) for s, w in zip(s_nodes, s_w))
        lhs = (Fsm((W - eta * O).reshape(-1)) - Fstar) / gap
        rhs = 1 - (mu_pl / L) * (2 - Jt / L)
        worst_nq = max(worst_nq, abs(lhs - rhs))
print("(iv) softmax ridge, identite avec Jt (courbure integree) : ecart max = %.2e (|grad| au minimiseur %.1e)"
      % (worst_nq, np.abs(Gsm(res.x)).max()))
