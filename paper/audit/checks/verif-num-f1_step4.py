"""Contre-verification independante du constat N1-STEP4 (papier l. 320-321 : "SAGA still converges at gamma R^2 = 4,
eight times the largest step the proof admits").

Code autonome : ni vrlab, ni les scripts num-f1_* de l'auditeur. L'instance (n = 1000, d = 50, kappa = 1e3, alpha = 1,
graine de donnees 0) est reconstruite ici d'apres la description de vrlab/problems.py:make_logistic et ses constantes
sont comparees a results/GammaDelta_summary.json (lecture seule). SAGA est reecrit (table de gradients complets
z_i = f_i'(theta), initialisee en theta_0 = 0, indices i.i.d. uniformes), en lot sur plusieurs graines.

Usage : python3 verif-num-f1_step4.py MODE
  repro : 40 passes, graines 0-2, grille du papier ; pente de log D sur les passes 2-40 (le chiffre du JSON)
  long4 : gamma R^2 = 4, 8 graines, 1000 passes
  long2 : gamma R^2 = 2, 8 graines, 150 passes
  scan  : gamma R^2 dans {2.5, 3, 3.5}, 4 graines, 400 passes
  lin   : SAGA sur le modele quadratique de F en eta_* (dynamique linearisee) : exposant de Lyapunov par passe
"""
import json
import os
import sys
import time

import numpy as np

N, D, KAPPA, ALPHA = 1000, 50, 1e3, 1.0
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))


def build_instance():
    rng = np.random.default_rng(0)
    ev = np.arange(1, D + 1, dtype=float) ** (-ALPHA)
    X = rng.standard_normal((N, D)) * np.sqrt(ev)[None, :]
    Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
    X = X @ Q.T
    X *= 1.0 / np.sqrt((X * X).sum(1).max())
    w = rng.standard_normal(D)
    logits = X @ w
    logits *= 2.0 / logits.std()
    y = np.sign(logits + rng.standard_normal(N))
    y[y == 0] = 1.0
    lam = 0.25 / (KAPPA - 1.0)
    A = y[:, None] * X                      # f_i(theta) = log(1 + exp(-a_i.theta)) + lam/2 |theta|^2
    R2 = float((A * A).sum(1).max() / 4.0 + lam)
    return A, lam, R2


def full_grad(A, lam, th):
    m = A @ th
    return -(A * (1.0 / (1.0 + np.exp(m)))[:, None]).mean(0) + lam * th


def hess(A, lam, th):
    m = A @ th
    sp = 1.0 / (2.0 + np.exp(m) + np.exp(-m))        # sigma(m) sigma(-m)
    return (A.T * sp) @ A / A.shape[0] + lam * np.eye(A.shape[1]), sp


def newton(A, lam):
    th = np.zeros(A.shape[1])
    for _ in range(100):
        g = full_grad(A, lam, th)
        H, _ = hess(A, lam, th)
        step = np.linalg.solve(H, g)
        t = 1.0
        gn = np.linalg.norm(g)
        while np.linalg.norm(full_grad(A, lam, th - t * step)) >= gn and t > 1e-12:
            t *= 0.5
        if t <= 1e-12:
            break
        th = th - t * step
    return th


def indices(seeds, n_iters):
    """Meme flux que numpy default_rng(seed).integers(0, n, size=min(1e6, reste)) : graines 0-2 = celles du depot."""
    out = np.empty((len(seeds), n_iters), dtype=np.int64)
    for k, s in enumerate(seeds):
        rng = np.random.default_rng(s)
        done = 0
        while done < n_iters:
            c = min(1_000_000, n_iters - done)
            out[k, done:done + c] = rng.integers(0, N, size=c)
            done += c
    return out


def saga_batch(A, lam, th_star, gR2, R2, seeds, passes, rec=100):
    """SAGA, pas gamma = gR2 / R2, theta_0 = 0, un run par graine (en lot). Renvoie iters, D (S x T)."""
    S = len(seeds)
    gamma = gR2 / R2
    n_iters = passes * N
    idx = indices(seeds, n_iters)
    ar = np.arange(S)
    th = np.zeros((S, D))
    z0 = -(A * 0.5) + 0.0                                # f_i'(0) = -a_i / 2
    Z = np.repeat(z0[None, :, :], S, axis=0).copy()
    zbar = Z.mean(1)
    its = [0]
    Ds = [((th - th_star) ** 2).sum(1)]
    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(n_iters):
            i = idx[:, t]
            a = A[i]
            m = np.einsum("sd,sd->s", a, th)
            g = a * (-1.0 / (1.0 + np.exp(m)))[:, None] + lam * th
            zi = Z[ar, i]
            diff = g - zi
            th -= gamma * (diff + zbar)
            zbar += diff / N
            Z[ar, i] = g
            if (t + 1) % N == 0:
                zbar = Z.mean(1)
            if (t + 1) % rec == 0:
                its.append(t + 1)
                Ds.append(((th - th_star) ** 2).sum(1))
    return np.array(its), np.array(Ds).T, th


def slope(its, Dv, p0, p1):
    """Exposant par passe : -n * pente des moindres carres de log D sur les passes [p0, p1] (regle du depot)."""
    sel = (its >= p0 * N) & (its <= p1 * N) & (Dv > 1e-25) & np.isfinite(Dv)
    if sel.sum() < 3 or not np.all(np.isfinite(Dv)):
        return float("nan")
    return float(-np.polyfit(its[sel].astype(float), np.log(Dv[sel]), 1)[0] * N)


def at_pass(its, Dv, p):
    return float(Dv[np.searchsorted(its, p * N)])


def lin_exponent(A, lam, th_star, gR2, R2, seeds, passes):
    """SAGA sur le modele quadratique en eta_* : e <- e - gamma (H_i e - u_i + ubar), u_i <- H_i e, avec
    H_i = sp_i a_i a_i^T + lam I. Exposant de log(|e|^2 + gamma^2 mean_i |u_i|^2) par passe (renormalisation a chaque
    passe), c'est-a-dire l'exposant de Lyapunov de la distance au carre."""
    S = len(seeds)
    gamma = gR2 / R2
    _, sp = hess(A, lam, th_star)
    idx = indices(seeds, passes * N)
    ar = np.arange(S)
    rng = np.random.default_rng(12345)
    e = rng.standard_normal((S, D))
    U = sp[None, :, None] * np.einsum("nd,sd->sn", A, e)[:, :, None] * A[None, :, :] + lam * e[:, None, :]
    ubar = U.mean(1)
    logn = np.zeros((S, passes + 1))
    acc = np.zeros(S)
    for t in range(passes * N):
        i = idx[:, t]
        a = A[i]
        h = a * (sp[i] * np.einsum("sd,sd->s", a, e))[:, None] + lam * e
        ui = U[ar, i]
        diff = h - ui
        e -= gamma * (diff + ubar)
        ubar += diff / N
        U[ar, i] = h
        if (t + 1) % N == 0:
            ubar = U.mean(1)
            nrm2 = (e * e).sum(1) + gamma ** 2 * (U * U).sum(2).mean(1)
            sc = np.sqrt(nrm2)
            e /= sc[:, None]
            U /= sc[:, None, None]
            ubar /= sc[:, None]
            acc += np.log(nrm2)
            logn[:, (t + 1) // N] = acc
    return logn


def main():
    mode = sys.argv[1]
    t0 = time.process_time()
    A, lam, R2 = build_instance()
    th_star = newton(A, lam)
    H, sp = hess(A, lam, th_star)
    ev = np.linalg.eigvalsh(H)
    ref = json.load(open(os.path.join(ROOT, "results", "GammaDelta_summary.json")))["problem"]
    print(f"instance : R2 = {R2:.16g} (JSON {ref['R2']:.16g}), lam = {lam:.16g} (JSON {ref['lam']:.16g}), "
          f"|eta*| = {np.linalg.norm(th_star):.12g} (JSON {ref['theta_star_norm']:.12g}), "
          f"lmin H* = {ev[0]:.10g} (JSON {ref['hess_star_min']:.10g}), lmax H* = {ev[-1]:.10g} "
          f"(JSON {ref['hess_star_max']:.10g}), |F'(eta*)| = {np.linalg.norm(full_grad(A, lam, th_star)):.2e}")
    D0 = float(th_star @ th_star)
    print(f"D_0 = |eta*|^2 = {D0:.6g} ; kappa = {R2 / lam:.6g}")
    sq = (A * A).sum(1)
    for g in (2.0, 4.0):
        c = g / R2 * (sp * sq + lam)
        print(f"gamma R^2 = {g:g} : gamma * f_i''(eta*) le long de x_i : max {c.max():.3f}, part des i avec > 2 : "
              f"{(c > 2).mean():.3f}, > 1 : {(c > 1).mean():.3f} ; gamma * lmax(H*) = {g / R2 * ev[-1]:.3f}")

    if mode == "repro":
        paper = json.load(open(os.path.join(ROOT, "results", "GammaDelta_summary.json")))["empirical_rates_per_pass"]
        for g in (0.125, 0.25, 1 / 3, 0.5, 1.0, 2.0, 4.0):
            its, Dv, _ = saga_batch(A, lam, th_star, g, R2, [0, 1, 2], 40)
            sl = [slope(its, Dv[k], 2, 40) for k in range(3)]
            print(f"gamma R^2 = {g:.4g} : pentes passes 2-40 par graine {np.round(sl, 6).tolist()} moyenne "
                  f"{np.mean(sl):.6f} (JSON {paper[str(g)]:.6f}) ; D a 40 passes {['%.3g' % x for x in Dv[:, -1]]}")
    elif mode in ("long4", "long2", "scan"):
        if mode == "long4":
            cfg = [(4.0, list(range(8)), 1000)]
        elif mode == "long2":
            cfg = [(2.0, list(range(8)), 150)]
        else:
            cfg = [(2.5, [0, 1, 2, 3], 400), (3.0, [0, 1, 2, 3], 400), (3.5, [0, 1, 2, 3], 400)]
        for g, seeds, passes in cfg:
            its, Dv, th = saga_batch(A, lam, th_star, g, R2, seeds, passes)
            print(f"\n=== gamma R^2 = {g:g}, graines {seeds}, {passes} passes, D_0 = {D0:.4g} ===")
            marks = [p for p in (1, 2, 5, 10, 20, 40, 60, 100, 150, 200, 400, 700, 1000) if p <= passes]
            print("passe  " + "  ".join(f"g{s:<8d}" for s in seeds))
            for p in marks:
                print(f"{p:5d}  " + "  ".join(f"{at_pass(its, Dv[k], p):9.3e}" for k in range(len(seeds))))
            wins = [(2, 40)] + [w for w in ((40, 100), (100, 400), (100, 1000), (400, 1000)) if w[1] <= passes]
            for w in wins:
                print(f"pente par passe sur les passes {w[0]}-{w[1]} : "
                      + "  ".join(f"{slope(its, Dv[k], *w):+.5f}" for k in range(len(seeds))))
            if passes >= 200:
                late = Dv[:, its >= 100 * N]
                print("passes >= 100 : mediane D " + "  ".join(f"{np.median(late[k]):.3g}" for k in range(len(seeds))))
                print("passes >= 100 : min D     " + "  ".join(f"{late[k].min():.3g}" for k in range(len(seeds))))
                print("passes >= 100 : max D     " + "  ".join(f"{late[k].max():.3g}" for k in range(len(seeds))))
                print(f"toutes graines, passes >= 100 : mediane {np.median(late):.4g} = {np.median(late) / D0:.4f} D_0, "
                      f"min {late.min():.4g}, max {late.max():.4g}")
            gn = [float(np.linalg.norm(full_grad(A, lam, th[k]))) for k in range(len(seeds))]
            print("fin : |F'(theta)| " + "  ".join(f"{x:.3g}" for x in gn) + f" ; min D sur tout le run {Dv.min():.3g}")
    elif mode == "lin":
        for g, passes in ((0.25, 60), (1.0, 60), (2.0, 150), (3.0, 150), (3.5, 150), (3.75, 150), (4.0, 150)):
            logn = lin_exponent(A, lam, th_star, g, R2, [0, 1, 2, 3], passes)
            h = passes // 3
            ex = (logn[:, -1] - logn[:, h]) / (passes - h)
            print(f"modele quadratique en eta*, gamma R^2 = {g:g} : exposant de -log D par passe (passes {h}-{passes}) "
                  f"par graine {np.round(-ex, 4).tolist()} moyenne {-ex.mean():+.4f}")
    print(f"[cpu {time.process_time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
