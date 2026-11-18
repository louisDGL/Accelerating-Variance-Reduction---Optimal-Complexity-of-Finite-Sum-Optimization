"""Counterexample search with bounded conditioning, then verification in 50-digit arithmetic:
for the one-sided quadratic, d/dt ||E||_op = - q_1 along the polar flow, q_1 = v_1^T Q v_1, Q = polar(|E| Sigma),
v_1 the top eigenvector of |E|. In dimension 2, q_1 > 0. In dimension >= 3, q_1 can be negative: the spectral norm of
the error can increase along the flow, although F and ||E||_F decrease (d/dt ||E||_F^2/2 = - tr(Sigma^-1 |G|) < 0).
Run: .venv/Scripts/python muon/theory_checks/counterexample_opnorm.py
"""
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm
import mpmath as mp

rng = np.random.default_rng(11)
d = 3
LOGC = np.log(1e3)            # eigenvalues of A and Sigma confined to [1e-3, 1]: condition numbers <= 1e3


def build(x):
    ka = d * (d - 1) // 2
    Sa = np.zeros((d, d)); Ss = np.zeros((d, d))
    iu = np.triu_indices(d, 1)
    Sa[iu] = x[:ka]; Sa = Sa - Sa.T
    Ss[iu] = x[ka:2 * ka]; Ss = Ss - Ss.T
    ea = np.exp(-LOGC * (0.5 + 0.5 * np.tanh(x[2 * ka:2 * ka + d])))
    es = np.exp(-LOGC * (0.5 + 0.5 * np.tanh(x[2 * ka + d:2 * ka + 2 * d])))
    Ra, Rs = expm(Sa), expm(Ss)
    return (Ra * ea) @ Ra.T, (Rs * es) @ Rs.T


def q1(x):
    A, S = build(x)
    w, V = np.linalg.eigh(A)
    U, s, Vt = np.linalg.svd(A @ S)
    return float(V[:, -1] @ (U @ Vt) @ V[:, -1])


best = (np.inf, None)
for trial in range(300):
    x0 = rng.standard_normal(2 * (d * (d - 1) // 2) + 2 * d) * 1.5
    res = minimize(q1, x0, method='Nelder-Mead', options={'maxiter': 4000, 'xatol': 1e-9, 'fatol': 1e-12})
    if res.fun < best[0]:
        best = (res.fun, res.x)
val, x = best
A, S = build(x)
print(f"minimum of q_1 found with condition numbers <= 1e3 (d = {d}): {val:+.6f}")
np.set_printoptions(precision=6, suppress=False, linewidth=150)
print("A = |E| =\n", A, "\neigenvalues", np.linalg.eigvalsh(A))
print("Sigma =\n", S, "\neigenvalues", np.linalg.eigvalsh(S))

# verification in 50-digit arithmetic, from the printed double-precision matrices (symmetrized)
mp.mp.dps = 50
A = 0.5 * (A + A.T); S = 0.5 * (S + S.T)
Am = mp.matrix(A.tolist()); Sm = mp.matrix(S.tolist())
M = Am * Sm
U, sv, Vh = mp.svd_r(M)
Q = U * Vh
evals, evecs = mp.eigsy(Am)
k = max(range(d), key=lambda i: evals[i])
v = evecs[:, k]
q1_mp = (v.T * Q * v)[0]
Y = Vh.T * mp.diag([sv[i] for i in range(d)]) * Vh
trSinvY = sum((Sm ** -1 * Y)[i, i] for i in range(d))
trQ = sum(Q[i, i] for i in range(d))
print("50-digit check: q_1 =", mp.nstr(q1_mp, 15), "; min singular value of A Sigma =", mp.nstr(min(sv), 5),
      "; tr(Q) =", mp.nstr(trQ, 10), "; tr(Sigma^-1 |G|) =", mp.nstr(trSinvY, 10), "(> 0: ||E||_F decreases)")

# direct confirmation on the flow: E_0 = A (K = d), small polar steps, ||E||_op at a few times
E = A.copy(); eta = 1e-6
n0 = np.linalg.norm(E, 2)
for it in range(2000):
    U_, s_, Vt_ = np.linalg.svd(E @ S)
    E = E - eta * (U_ @ Vt_)
print(f"flow from E_0 = A: ||E||_op goes from {n0:.9f} to {np.linalg.norm(E, 2):.9f} after time {2000 * eta:.4f} "
      f"(predicted change {-val * 2000 * eta:+.3e}); ||E||_F from {np.linalg.norm(A):.9f} to {np.linalg.norm(E):.9f}")
