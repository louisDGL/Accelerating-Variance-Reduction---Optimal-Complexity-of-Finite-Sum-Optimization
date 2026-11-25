"""Audit code-fidelity, bloc (d) : partie SAGA (vrlab).

1. Le pas "textbook" de vrlab.algorithms.saga est-il gamma = 1/(4 R^2), R^2 = max_i (||x_i||^2 / 4 + lam), Delta = 3 gamma^2 ?
2. SAGA de vrlab = SAGA du papier (l. 190-192) : implementation independante, memes indices.
3. L'"enumeration exacte" de vrlab.instrumentation.saga_state_stats est-elle exacte ? Comparaison a la force brute
   (les n pas possibles depuis un etat, Lyapunov recalcule de zero), en metrique euclidienne et en metrique P dense.
4. Ordres d'echantillonnage : 'rr' = permutation nouvelle a chaque passe, 'so' = une seule, 'cyclic'.

Lancer depuis la racine du projet :
    PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        python3 paper/audit/checks/code-fidelity_saga_exact.py
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, ROOT)
from vrlab import algorithms, problems, solvers                      # noqa: E402
from vrlab.instrumentation import Metric, saga_state_stats           # noqa: E402

OK = []


def check(name, cond, detail=""):
    OK.append(bool(cond))
    print(f"[{'OK ' if cond else 'FAIL'}] {name}  {detail}")


n, d, kappa = 200, 10, 300.0
prob = problems.make_logistic(n, d, kappa=kappa, alpha=1.0, seed=0)
solvers.solve_optimum(prob)
X, lam = prob.X, prob.lam

# 1. constantes et pas du livre
R2 = float((np.einsum("ij,ij->i", X, X) / 4.0 + lam).max())
check("R^2 = max_i (||x_i||^2 / 4 + lam), mu = lam", abs(prob.R2 - R2) < 1e-14 and prob.mu == lam,
      f"R2 = {prob.R2:.6g}, kappa = {prob.R2 / prob.mu:.6g}")
theta0 = np.zeros(d)
T = 3 * n + 37
run = algorithms.saga(prob, theta0, n_iters=T, seed=3, record_every=T)
check("pas par defaut gamma = 1/(4 R^2), Delta = 3 gamma^2",
      abs(run["gamma"] - 1 / (4 * R2)) < 1e-15 and abs(run["Delta"] - 3 * run["gamma"] ** 2) < 1e-18)
gamma, Delta = run["gamma"], run["Delta"]

# 2. implementation independante de SAGA (l. 190-192 du papier), memes indices
idx = np.random.default_rng(3).integers(0, n, size=T)


def fprime(theta, i):                                  # f_i'(theta) pour f_i = log(1 + exp(-y_i x_i.theta)) + lam/2 |theta|^2
    u = prob.y[i] * (X[i] @ theta)
    return -prob.y[i] * X[i] / (1.0 + np.exp(u)) + lam * theta


theta = theta0.copy()
z = np.array([fprime(theta, i) for i in range(n)])     # z_i^(0) = f_i'(theta_0)
for i in idx:
    g = fprime(theta, i)
    theta = theta - gamma * (g - z[i] + z.mean(0))
    z[i] = g
err = np.abs(theta - run["theta"]).max()
check("vrlab.saga = SAGA du papier (omega = f_i' - z_i + moyenne des z, z_i <- f_i')", err < 1e-12, f"ecart {err:.1e}")
check("cout : n initial + 1 par iteration", run["grad_evals"][-1] == n + T)

# 3. enumeration exacte contre la force brute
gs = np.array([fprime(prob.theta_star, i) for i in range(n)])
check("somme des f_i'(eta_*) nulle (optimum precis)", np.abs(gs.mean(0)).max() < 1e-13, f"{np.abs(gs.mean(0)).max():.1e}")
rngP = np.random.default_rng(7)
A = rngP.standard_normal((d, d))
Pdense = A @ A.T / d + 0.5 * np.eye(d)
for label, P in (("euclidienne", None), ("P dense", Pdense)):
    met = Metric(P, d)
    Pinv = np.eye(d) if P is None else np.linalg.inv(P)
    Pm = np.eye(d) if P is None else P
    zbar = z.mean(0)
    st = saga_state_stats(prob, theta, z, zbar, gamma, Delta, prob.R2, prob.mu, met)
    V = (theta - prob.theta_star) @ Pm @ (theta - prob.theta_star) \
        + Delta * np.einsum("ij,jk,ik->", z - gs, Pinv, z - gs)
    EV = 0.0
    for i in range(n):
        g = fprime(theta, i)
        th = theta - gamma * Pinv @ (g - z[i] + zbar)
        z2 = z.copy()
        z2[i] = g
        EV += ((th - prob.theta_star) @ Pm @ (th - prob.theta_star)
               + Delta * np.einsum("ij,jk,ik->", z2 - gs, Pinv, z2 - gs)) / n
    check(f"metrique {label} : V et E[V_+ | etat] de l'instrumentation = force brute sur les n indices",
          abs(st["V"] - V) < 1e-12 * V and abs(st["EV_next"] - EV) < 1e-12 * EV,
          f"rho_exact {st['EV_next'] / st['V']:.12f} contre {EV / V:.12f}")
    # identite (1) du papier, avec les rapports mesures (pas du livre, metrique euclidienne seulement)
    if P is None:
        me, Le, rsq, rvar = st["mu_eff"], st["L_eff"], st["r_sq"], st["r_var"]
        cD = 1 - me / (2 * R2) * (1 - Le / R2 * (rsq / 4 + 3.0 / 8))
        cM = 1 - (1 - 2.0 / 3 * rsq * rvar) / n
        rep = cD * st["D"] + cM * Delta * st["M"]
        check("identite (1) du papier : E[V_+] = c_D D + c_M delta M avec mu_eff, L_eff, r_sq, r_var mesures",
              abs(rep - EV) < 1e-10 * EV, f"ecart relatif {abs(rep - EV) / EV:.1e}")
        check("mu_eff = <theta - eta_*, F'> / ||theta - eta_*||^2 et L_eff = (1/n) sum ||f_i' - f_i'(eta_*)||^2 / <.,.>",
              abs(me - st["G"] / st["D"]) < 1e-12 and abs(Le - st["S"] / n / st["G"]) < 1e-12)

# 4. ordres d'echantillonnage
chunks = list(algorithms._index_stream(np.random.default_rng(0), 50, 150, sampling="rr"))
check("'rr' : une permutation nouvelle par passe",
      all(sorted(c) == list(range(50)) for c in chunks) and not np.array_equal(chunks[0], chunks[1]))
chunks = list(algorithms._index_stream(np.random.default_rng(0), 50, 150, sampling="so"))
check("'so' : la meme permutation a chaque passe", np.array_equal(chunks[0], chunks[1]) and sorted(chunks[0]) == list(range(50)))
try:
    algorithms.saga(prob, theta0, n_iters=10, sampling="rr", instrument_every=1)
    refused = False
except ValueError:
    refused = True
check("l'instrumentation exacte (esperance sur un indice uniforme) est refusee hors echantillonnage independant", refused)

print(f"\n{sum(OK)}/{len(OK)} controles passes")
