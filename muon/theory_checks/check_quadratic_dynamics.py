"""Numerical checks for Part 3 of notes_spectral_theory: exact dynamics of polar steps on the one-sided quadratic
F(W) = (1/2) tr((W - W*) Sigma (W - W*)^T), W in R^{K x d}, K >= d  (multi-output least squares).

  T3.1  first phase: the polar factor is constant, W_t = W_0 - tau_t O_0, as long as tau_t < tau* = lambda_min(Sigma^-1/2 Y_0 Sigma^-1/2);
        F(tau) = F_0 - tau tr(Y_0) + tau^2 tr(Sigma)/2; exact line search tau_ls = tr(Y_0)/tr(Sigma) >= tau*, equality iff Y_0 ~ Sigma.
  T3.2  one-step contraction of the nuclear-scaled step: 1 - (tr|M|)^2 / (tr Sigma * tr(M Sigma^-1 M^T)).
  T3.3  commuting case: error singular values follow a_k <- | a_k - eta |; with the nuclear-scaled step eta = E_p[a]
        (p_k = lambda_k / tr Sigma): a <- |a - E_p a|, contraction Var_p(a)/E_p(a^2) = 1 - MAD^2/Var afterwards;
        small-step limit: a_k(t) = (a_k(0) - t)_+ (singular value soft-thresholding of the initial error), extinction
        time = ||E_0||_op whatever Sigma;  inf of MAD_p^2/Var_p = 4 p1 p2/(p1 + p2) over the two smallest atoms.
  Exploration: long-run contraction of the nuclear-scaled step vs GD on commuting and generic instances.
Run: .venv/Scripts/python muon/theory_checks/check_quadratic_dynamics.py
"""
import numpy as np

rng = np.random.default_rng(20261116)


def polar(M, tol=1e-13):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    keep = s > tol * max(s[0], 1e-300)
    return (U[:, keep]) @ Vt[keep]


def nuc(M):
    return np.linalg.svd(M, compute_uv=False).sum()


def psd_sqrt(S):
    w, V = np.linalg.eigh(S)
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T


def make(K, d, alpha, rng, commuting=False, a=None):
    lam = np.arange(1, d + 1.0) ** (-alpha)
    V, _ = np.linalg.qr(rng.standard_normal((d, d)))
    Sigma = (V * lam) @ V.T
    if commuting:
        U, _ = np.linalg.qr(rng.standard_normal((K, d)))
        if a is None:
            a = np.abs(rng.standard_normal(d)) + 0.1
        E0 = (U * a) @ V.T
    else:
        E0 = rng.standard_normal((K, d))
    return Sigma, lam, V, E0


def F(E, Sigma):
    return 0.5 * np.trace(E @ Sigma @ E.T)


print("== T3.1 first phase (generic, non-commuting) ==")
K, d = 7, 5
Sigma, lam, V, E0 = make(K, d, 1.0, rng)
G0 = E0 @ Sigma
Y0 = psd_sqrt(G0.T @ G0)
O0 = G0 @ np.linalg.inv(Y0)
Si = np.linalg.inv(psd_sqrt(Sigma))
tau_star = np.linalg.eigvalsh(Si @ Y0 @ Si).min()
tau_ls = np.trace(Y0) / np.trace(Sigma)
print(f"tau* = {tau_star:.6f}, tau_ls = {tau_ls:.6f}, tau_ls >= tau*: {tau_ls >= tau_star}")
for frac in [0.3, 0.7, 0.999, 1.001, 1.5]:
    tau = frac * tau_star
    E = E0 - tau * O0
    G = E @ Sigma
    same = np.linalg.norm(polar(G) - O0)
    Fpred = F(E0, Sigma) - tau * np.trace(Y0) + 0.5 * tau ** 2 * np.trace(Sigma)
    print(f"  tau = {frac:5.3f} tau*: ||polar(G_tau) - O_0||_F = {same:.2e}; F - formula = {F(E, Sigma) - Fpred:+.2e}; "
          f"min eig(Y0 - tau Sigma) = {np.linalg.eigvalsh(Y0 - tau * Sigma).min():+.3e}")
# several small steps inside the phase give the same polar factor
E = E0.copy(); tau = 0.0
for eta in rng.uniform(0, 0.3 * tau_star, 3):
    if tau + eta < tau_star:
        E = E - eta * polar(E @ Sigma); tau += eta
print(f"  3 polar steps of total length {tau / tau_star:.3f} tau*: ||W - (W0 - tau O0)|| = {np.linalg.norm(E - (E0 - tau * O0)):.2e}")
# white error: tau_ls = tau*
Q, _ = np.linalg.qr(rng.standard_normal((K, d)))
Ew = 2.3 * Q
Gw = Ew @ Sigma; Yw = psd_sqrt(Gw.T @ Gw)
print(f"  white error E0 = s Q: tau* = {np.linalg.eigvalsh(Si @ Yw @ Si).min():.6f}, tau_ls = {np.trace(Yw) / np.trace(Sigma):.6f} (= s = 2.3), "
      f"F after the line-search step = {F(Ew - (np.trace(Yw) / np.trace(Sigma)) * polar(Gw), Sigma):.2e}")

print("== T3.2 one-step contraction of the nuclear-scaled step (generic) ==")
for _ in range(3):
    Sigma, lam, V, E0 = make(K, d, 1.5, rng)
    G = E0 @ Sigma
    eta = nuc(G) / np.trace(Sigma)
    E1 = E0 - eta * polar(G)
    M = G                                                # tr|M| = ||G||_*, tr(M Sigma^-1 M^T) = 2 F
    pred = 1 - nuc(M) ** 2 / (np.trace(Sigma) * np.trace(M @ np.linalg.inv(Sigma) @ M.T))
    print(f"  measured {F(E1, Sigma) / F(E0, Sigma):.12f}  formula {pred:.12f}")

print("== T3.3 commuting case: scalar recursion and the mean-absolute-deviation map ==")
K, d = 8, 6
Sigma, lam, V, E0 = make(K, d, 1.0, rng, commuting=True)
p = lam / lam.sum()
a = np.linalg.norm(E0 @ V, axis=0)                      # singular value of E along v_k
E = E0.copy()
ok = True
for t in range(12):
    G = E @ Sigma
    eta = nuc(G) / np.trace(Sigma)
    m = p @ a
    ratio_pred = (p @ (a - m) ** 2) / (p @ a ** 2)
    E_new = E - eta * polar(G)
    a_new = np.abs(a - m)
    a_meas = np.linalg.norm(E_new @ V, axis=0)
    ok &= np.allclose(a_meas, a_new, rtol=1e-7, atol=1e-12) and abs(eta - m) < 1e-10 * max(m, 1e-300)
    ok &= abs(F(E_new, Sigma) / F(E, Sigma) - ratio_pred) < 1e-8
    E, a = E_new, a_new
print(f"  12 steps: matrix iteration == map a <- |a - E_p a|, and F ratio == Var_p(a)/E_p(a^2): {ok}")

# infimum of MAD^2/Var
p_sorted = np.sort(p)
bound = 4 * p_sorted[0] * p_sorted[1] / (p_sorted[0] + p_sorted[1])
worst = np.inf
for _ in range(200000):
    x = rng.standard_normal(d) * rng.choice([1.0, 10.0, 100.0], d)
    x = x - p @ x
    worst = min(worst, (p @ np.abs(x)) ** 2 / (p @ x ** 2))
i1, i2 = np.argsort(p)[:2]
x = np.zeros(d); x[i1] = 1.0 / p[i1]; x[i2] = -1.0 / p[i2]
print(f"  inf MAD_p^2/Var_p: random search {worst:.5f} >= bound 4 p1 p2/(p1+p2) = {bound:.5f}; extremal configuration gives "
      f"{(p @ np.abs(x)) ** 2 / (p @ x ** 2):.5f}; 2 p_min = {2 * p_sorted[0]:.5f}, 1/kappa_op = p_min = {p_sorted[0]:.5f}")

print("== T3.3 small-step limit: soft-thresholding path and extinction time (commuting) ==")
for alpha in [0.0, 1.0, 2.0]:
    Sigma, lam, V, E0 = make(K, d, alpha, rng, commuting=True, a=np.array([2.0, 0.3, 1.1, 0.7, 1.6, 0.05]))
    a0 = np.linalg.norm(E0 @ V, axis=0)
    eta = 1e-3
    E = E0.copy(); t = 0.0; T_ext = None; err_path = 0.0
    U0 = (E0 @ V) / a0
    while t < 2.5:
        E = E - eta * polar(E @ Sigma); t += eta
        svt = (U0 * np.clip(a0 - t, 0, None)) @ V.T
        err_path = max(err_path, np.linalg.norm(E - svt, 2))
        if T_ext is None and np.linalg.norm(E, 2) <= 2 * eta:
            T_ext = t
    print(f"  alpha = {alpha}: kappa(Sigma) = {lam.max() / lam.min():.0f}; extinction time {T_ext:.3f} vs ||E_0||_op = {a0.max():.3f}; "
          f"max_t ||E(t) - SVT_t(E_0)||_op = {err_path:.1e} (step {eta})")

print("== extinction time, generic (non-commuting): bounds ||G0||_*/L_op <= T <= ||G0||_*/mu_op and T >= ||E0||_op ==")
for alpha in [0.0, 1.0, 2.0]:
    Sigma, lam, V, E0 = make(K, d, alpha, rng)
    G0 = E0 @ Sigma
    eta = 1e-3
    E = E0.copy(); t = 0.0
    while np.linalg.norm(E, 2) > 3 * eta and t < 400:
        E = E - eta * polar(E @ Sigma); t += eta
    lo, hi = nuc(G0) / np.trace(Sigma), nuc(G0) / lam.min()
    print(f"  alpha = {alpha}: T = {t:.3f}; ||G0||_*/L_op = {lo:.3f}; ||G0||_*/mu_op = {hi:.3f}; ||E0||_op = {np.linalg.norm(E0, 2):.3f}; "
          f"||E0||_* = {nuc(E0):.3f}")

print("== exploration: long-run contraction per step, nuclear-scaled step vs GD (step 1/L_F) vs GD exact line search ==")


def run_sharp(E0, Sigma, T):
    E = E0.copy(); out = [F(E, Sigma)]
    for _ in range(T):
        G = E @ Sigma
        O = polar(G)
        J = np.trace(O @ Sigma @ O.T)
        E = E - (nuc(G) / J) * O                        # exact line search along O (= nuclear-scaled with L = J)
        out.append(F(E, Sigma))
        if out[-1] < 1e-28 * out[0]:
            break
    return np.array(out)


def run_gd(E0, Sigma, T, ls=False):
    E = E0.copy(); out = [F(E, Sigma)]
    L = np.linalg.eigvalsh(Sigma).max()
    for _ in range(T):
        G = E @ Sigma
        step = (np.sum(G * G) / np.trace(G @ Sigma @ G.T)) if ls else 1.0 / L
        E = E - step * G
        out.append(F(E, Sigma))
        if out[-1] < 1e-28 * out[0]:
            break
    return np.array(out)


def iters_to(tr, tol):
    idx = np.nonzero(tr <= tol * tr[0])[0]
    return int(idx[0]) if len(idx) else None


K, d = 20, 20
for alpha in [1.0, 2.0]:
    for commuting in [True, False]:
        res = []
        for seed in range(5):
            r2 = np.random.default_rng(100 + seed)
            Sigma, lam, V, E0 = make(K, d, alpha, r2, commuting=commuting)
            ts, tg, tl = run_sharp(E0, Sigma, 200000), run_gd(E0, Sigma, 200000), run_gd(E0, Sigma, 200000, ls=True)
            res.append((iters_to(ts, 1e-12), iters_to(tg, 1e-12), iters_to(tl, 1e-12)))
        kF = lam.max() / lam.min(); kop = lam.sum() / lam.min()
        print(f"  alpha = {alpha}, commuting = {commuting}: kappa_F = {kF:.0f}, kappa_op = {kop:.0f}; iterations to 1e-12 "
              f"(sharp/line-search polar, GD 1/L, GD line search) over 5 seeds: {res}")
