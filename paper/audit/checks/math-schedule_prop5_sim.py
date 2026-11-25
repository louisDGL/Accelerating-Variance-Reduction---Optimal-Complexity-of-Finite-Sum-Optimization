"""Audit math-schedule, controle 1 : Proposition 5 du papier, items (a), (b), (c), implementation independante du lab.

Probleme : moindres carres ridge matriciels, F(W) = (1/n) sum_i ||W x_i - y_i||^2 / 2 + (lam/2) ||W||_F^2,
K = 3, d = 4, n = 12 (numpy seul, facteur polaire par SVD).

(b) recette de l'item (b) : table SAGA initialisee par les gradients en W_0, facteur polaire exact,
    eta_t = eta_0 rho^(t/2), eta_0 = (2 mu_op D_0)^(1/2) / L_eff, L_eff = L_op + 10 A L_max, rho = 1 - mu_op / L_eff ;
    ordre cyclique (A = n/b), permutation fraiche (A = 2n/b - 1), adversaire glouton a ages <= A, lots independants
    (A = ceil((n/b) log(nT/delta))). A chaque iteration : D_t <= D_0 rho^t, recurrence a un pas, borne (a), ages.
(a) borne ||omega - grad F||_* <= (theta L_max + L_bar) max_i ||W - phi_i||_op sur des etats aleatoires
    (moindres carres c_l = 1, softmax c_l = 1/2).
(c) covariance d'echelle : deux trajectoires aux echelles 1 et s, memes lots.
Controles negatifs : table initialisee a zero (hypothese implicite de la Proposition), mu_op > L_max.

Lancer depuis la racine du projet :
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 paper/audit/checks/math-schedule_prop5_sim.py
"""
import math
import sys
import time

import numpy as np

K, d, n = 3, 4, 12


def polar(M):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    if s[0] == 0.0:
        return np.zeros_like(M)
    keep = s > 1e-14 * s[0]
    return U[:, keep] @ Vt[keep]


def nuc(M):
    return float(np.linalg.svd(M, compute_uv=False).sum())


def opn(M):
    return float(np.linalg.svd(M, compute_uv=False)[0])


class LS:
    def __init__(self, seed, lam=0.05, scale=None):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, d))
        if scale is not None:
            X = X * scale
        else:
            X = X * np.array([1.0, 0.7, 0.4, 0.2])          # spectre de covariance anisotrope
        Y = rng.standard_normal((n, K))
        self.X, self.Y, self.lam = X, Y, lam
        self.Sig = X.T @ X / n
        self.S = self.Sig + lam * np.eye(d)                  # hessienne : D^2F[D, D] = tr(D S D^T)
        ev = np.linalg.eigvalsh(self.S)
        self.mu_op = float(ev[0])                            # min_{||D||_op = 1} tr(D S D^T) = lambda_min(S)
        self.L_op = float(np.sort(ev)[::-1][:min(K, d)].sum())   # Ky Fan : somme des r plus grandes valeurs propres
        self.Li = np.einsum("ij,ij->i", X, X)                # c_l = 1
        self.Lmax, self.Lbar = float(self.Li.max()), float(self.Li.mean())
        self.Wstar = np.linalg.solve(self.S, X.T @ Y / n).T

    def grad(self, W):
        return (W @ self.X.T - self.Y.T) @ self.X / n + self.lam * W

    def gap(self, W):
        E = W - self.Wstar
        return 0.5 * float(np.einsum("ij,jk,ik->", E, self.S, E))

    def res(self, W):
        return W @ self.X.T - self.Y.T                       # K x n, colonne i = W x_i - y_i


def check_constants(p, rng, trials=4000):
    """mu_op et L_op sont bien les meilleures constantes : recherche aleatoire + directions extremales."""
    lo, hi = np.inf, 0.0
    for _ in range(trials):
        Dl = rng.standard_normal((K, d))
        if rng.random() < 0.5:                               # rang un
            Dl = np.outer(rng.standard_normal(K), rng.standard_normal(d))
        q = float(np.einsum("ij,jk,ik->", Dl, p.S, Dl)) / opn(Dl) ** 2
        lo, hi = min(lo, q), max(hi, q)
    ev, V = np.linalg.eigh(p.S)
    q_min = float(V[:, 0] @ p.S @ V[:, 0])                   # Delta = e v_min^T
    Dtop = np.eye(K, d) @ V[:, ::-1].T                       # lignes = K vecteurs propres dominants : ||.||_op = 1
    q_max = float(np.einsum("ij,jk,ik->", Dtop, p.S, Dtop))
    assert lo >= p.mu_op * (1 - 1e-12) and hi <= p.L_op * (1 + 1e-12)
    assert abs(q_min - p.mu_op) < 1e-12 and abs(q_max - p.L_op) < 1e-10
    return lo / p.mu_op, hi / p.L_op


def batches_stream(order, b, T, rng):
    nb = n // b
    if order == "cyclic":
        perm = rng.permutation(n)
        for t in range(T):
            k = t % nb
            yield perm[k * b:(k + 1) * b]
    elif order == "reshuffle":
        t = 0
        while t < T:
            perm = rng.permutation(n)
            for k in range(nb):
                if t >= T:
                    return
                yield perm[k * b:(k + 1) * b]
                t += 1
    elif order == "iid":
        for t in range(T):
            yield rng.choice(n, size=b, replace=False)
    else:
        raise ValueError(order)


def run_b(p, W0, order, b, theta, A, T, seed, zero_table=False, adversary=False, full_checks_until=3000,
          check_every=97, A_for_schedule=None):
    """Recette de l'item (b). Renvoie un dict de statistiques ; aucune assertion (les violations sont comptees)."""
    rng = np.random.default_rng(seed)
    A_s = A if A_for_schedule is None else A_for_schedule
    L_eff = p.L_op + 10 * A_s * p.Lmax
    rho = 1.0 - p.mu_op / L_eff
    D0 = p.gap(W0)
    eta0 = math.sqrt(2 * p.mu_op * D0) / L_eff
    X = p.X
    W = W0.copy()
    R = np.zeros((K, n)) if zero_table else p.res(W0)        # residus stockes : z_i = R[:, i] x_i^T
    Phi = np.repeat(W0[None], n, axis=0)
    t_ref = np.zeros(n, dtype=int)
    out = dict(rho=rho, eta0=eta0, L_eff=L_eff, D0=D0, max_ratio=0.0, t_max_ratio=0, viol_bound=0, viol_rec=0,
               viol_a=0, viol_inner=0, max_age=0, max_a_ratio=0.0, min_rec_slack=np.inf, T=T)
    stream = None if adversary else batches_stream(order, b, T, rng)
    D = D0
    for t in range(T):
        eta = eta0 * rho ** (t / 2)
        age = t - t_ref
        if adversary:
            # regle "quelconque" : parmi les indices qui laissent les ages <= A realisables, celui qui maximise D_{t+1}
            slack = A - age                                   # l'entree i doit etre tiree au plus tard a t + slack_i
            best, bestD = None, -np.inf
            zbar = R @ X / n
            for j in range(n):
                s2 = slack - 1
                s2[j] = A - 1
                ss = np.sort(s2)
                s2 = None
                if np.any(ss < np.arange(n)):                 # echeances non realisables a b = 1
                    continue
                c = np.outer(W @ X[j] - p.Y[j] - R[:, j], X[j])
                om = zbar + p.lam * W + theta * c
                Dj = p.gap(W - eta * polar(om))
                if Dj > bestD:
                    best, bestD = j, Dj
            B = np.array([best])
        else:
            B = next(stream)
        out["max_age"] = max(out["max_age"], int(age.max()))
        zbar = R @ X / n
        Rb = W @ X[B].T - p.Y[B].T                           # K x b
        c = (Rb - R[:, B]) @ X[B] / b
        omega = zbar + p.lam * W + theta * c
        O = polar(omega)
        G = p.grad(W)
        full = (t < full_checks_until) or (t % check_every == 0)
        if full and not zero_table:
            err = nuc(omega - G)
            l_true = np.linalg.svd(W[None] - Phi, compute_uv=False)[:, 0]
            rhs_a = (theta * p.Lmax + p.Lbar) * float(l_true.max())
            if rhs_a > 0:
                out["max_a_ratio"] = max(out["max_a_ratio"], err / rhs_a)
            nG = nuc(G)
            if err > rhs_a * (1 + 1e-9) + 1e-13 * nG:        # a t = 0 les deux membres sont nuls a l'arrondi pres
                out["viol_a"] += 1
            if float(np.vdot(G, O)) < nG - 2 * err - 1e-9 * (nG + err):
                out["viol_inner"] += 1
        W_new = W - eta * O
        D_new = p.gap(W_new)
        rec = D - eta * math.sqrt(2 * p.mu_op * D) + 0.5 * L_eff * eta ** 2
        if not zero_table and int(age.max()) <= A_s:
            out["min_rec_slack"] = min(out["min_rec_slack"], (rec - D_new) / max(rec, 1e-300))
            if D_new > rec * (1 + 1e-9):
                out["viol_rec"] += 1
        R[:, B] = Rb
        Phi[B], t_ref[B] = W, t
        W, D = W_new, D_new
        bound = D0 * rho ** (t + 1)
        ratio = D / bound
        if ratio > out["max_ratio"]:
            out["max_ratio"], out["t_max_ratio"] = ratio, t + 1
        if ratio > 1 + 1e-9:
            out["viol_bound"] += 1
    out["D_T_over_D0"] = D / D0
    out["rho_T"] = rho ** T
    return out


def fmt(o):
    return (f"rho=1-{1 - o['rho']:.3e} eta0={o['eta0']:.3e} T={o['T']} max_age={o['max_age']} "
            f"max_t D_t/(D_0 rho^t)={o['max_ratio']:.6f} (t={o['t_max_ratio']}) viol(D_t<=D_0 rho^t)={o['viol_bound']} "
            f"viol(rec)={o['viol_rec']} viol(a)={o['viol_a']} viol(<G,O>)={o['viol_inner']} "
            f"max (a) lhs/rhs={o['max_a_ratio']:.3f} D_T/D_0={o['D_T_over_D0']:.2e} rho^T={o['rho_T']:.2e}")


def part_b():
    print("=== (b) recette prescrite, D_t <= D_0 rho^t a chaque iteration ===")
    rng = np.random.default_rng(0)
    worst = 0.0
    total_viol = 0
    for pseed in (1, 2):
        p = LS(pseed)
        r1, r2 = check_constants(p, rng)
        print(f"probleme {pseed}: mu_op={p.mu_op:.4f} L_op={p.L_op:.4f} L_max={p.Lmax:.4f} L_bar={p.Lbar:.4f} "
              f"(mu_op <= L_max: {p.mu_op <= p.Lmax}); recherche aleatoire: min quotient/mu_op={r1:.3f}, "
              f"max quotient/L_op={r2:.3f}")
        W0 = np.random.default_rng(100 + pseed).standard_normal((K, d))
        for order in ("cyclic", "reshuffle"):
            for b in (1, 3):
                A = n // b if order == "cyclic" else 2 * n // b - 1
                for theta in (1.0, b / n):
                    o = run_b(p, W0, order, b, theta, A, 20000, seed=7)
                    assert o["max_age"] <= A, (order, b, o["max_age"], A)
                    worst = max(worst, o["max_ratio"])
                    total_viol += o["viol_bound"] + o["viol_rec"] + o["viol_a"] + o["viol_inner"]
                    print(f"  {order:9s} b={b} theta={theta:.3f} A={A}: {fmt(o)}")
        # depart ou l'inegalite de Polyak-Lojasiewicz duale est une egalite : W_0 - W_* = u v_min^T
        ev, V = np.linalg.eigh(p.S)
        W0t = p.Wstar + np.outer(np.random.default_rng(5).standard_normal(K), V[:, 0])
        G0 = p.grad(W0t)
        print(f"  depart PL-serre : ||grad F||_*^2 / (2 mu_op D_0) = {nuc(G0) ** 2 / (2 * p.mu_op * p.gap(W0t)):.12f}")
        for order, b, A in (("cyclic", 1, n), ("reshuffle", 1, 2 * n - 1), ("cyclic", 3, n // 3)):
            o = run_b(p, W0t, order, b, 1.0, A, 20000, seed=7)
            worst = max(worst, o["max_ratio"])
            total_viol += o["viol_bound"] + o["viol_rec"] + o["viol_a"] + o["viol_inner"]
            print(f"  PL-serre {order:9s} b={b} A={A}: {fmt(o)}")
    # depart W_0 = 0 (celui des experiences) et un depart proche de l'optimum
    p = LS(1)
    for name, W0 in (("W0=0", np.zeros((K, d))), ("W0 proche", p.Wstar + 1e-3 * np.ones((K, d)))):
        o = run_b(p, W0, "reshuffle", 1, 1.0, 2 * n - 1, 40000, seed=3)
        worst = max(worst, o["max_ratio"])
        total_viol += o["viol_bound"] + o["viol_rec"] + o["viol_a"] + o["viol_inner"]
        print(f"  {name:9s} reshuffle b=1 theta=1: {fmt(o)}")
    print(f"  => plus grand D_t/(D_0 rho^t) sur ces runs: {worst:.6f} ; violations totales: {total_viol}")
    return worst, total_viol


def part_b_adversary():
    print("=== (b) regle adverse (gloutonne) a ages <= A, b = 1 ===")
    tot = 0
    for pseed in (1, 2):
        p = LS(pseed)
        ev, V = np.linalg.eigh(p.S)
        starts = (("aleatoire", np.random.default_rng(100 + pseed).standard_normal((K, d))),
                  ("PL-serre", p.Wstar + np.outer(np.random.default_rng(5).standard_normal(K), V[:, 0])))
        for sname, W0 in starts:
            for A in (n, 2 * n - 1):
                o = run_b(p, W0, None, 1, 1.0, A, 4000, seed=0, adversary=True)
                assert o["max_age"] <= A
                tot += o["viol_bound"] + o["viol_rec"] + o["viol_a"] + o["viol_inner"]
                print(f"  probleme {pseed} depart {sname} A={A}: {fmt(o)}")
    print(f"  => violations totales (adversaire): {tot}")
    return tot


def part_b_iid():
    print("=== (b) lots independants : A = ceil((n/b) log(n T / delta)), horizon T fixe a l'avance ===")
    p = LS(1)
    W0 = np.random.default_rng(101).standard_normal((K, d))
    b, delta = 1, 0.05
    for T in (2000, 20000):
        A = math.ceil((n / b) * math.log(n * T / delta))
        exceed, viol, worst = 0, 0, 0.0
        nrun = 40 if T == 2000 else 6
        for s in range(nrun):
            o = run_b(p, W0, "iid", b, 1.0, A, T, seed=1000 + s, full_checks_until=0, check_every=10 ** 9)
            exceed += int(o["max_age"] > A)
            viol += o["viol_bound"]
            worst = max(worst, o["max_ratio"])
        L_eff = p.L_op + 10 * A * p.Lmax
        rho = 1 - p.mu_op / L_eff
        print(f"  T={T} delta={delta}: A={A} (n/b={n // b}), rho=1-{1 - rho:.3e}, rho^T={rho ** T:.3e} ; "
              f"{nrun} runs: runs avec un age > A: {exceed}, violations de D_t <= D_0 rho^t: {viol}, "
              f"max ratio {worst:.6f}")
    # dependance du certificat en T : facteur garanti a l'horizon T
    print("  facteur certifie rho^T en fonction de T (delta = 0.05) :")
    for T in (10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 8):
        A = math.ceil((n / b) * math.log(n * T / delta))
        rho = 1 - p.mu_op / (p.L_op + 10 * A * p.Lmax)
        print(f"    T={T:>9d}: A={A:4d}, 1-rho={1 - rho:.3e}, T log(1/rho)={-T * math.log(rho):.3e} nats")


def part_negative():
    print("=== controles negatifs ===")
    # (i) table initialisee a zero : z_i n'est le gradient en aucun itere, l'age n'est pas defini avant le 1er passage.
    #     Depart proche de W_* : grad F(W_0) est petit, omega_0 = lam W_0 + grad g_j(W_0) ne l'est pas et sa direction
    #     polaire n'est pas correlee au gradient.
    nviol, nrun, worst, first = 0, 0, 0.0, None
    for pseed in (1, 2, 3):
        p = LS(pseed)
        for dist in (1.0, 1e-3):
            W0 = p.Wstar + dist * np.random.default_rng(100 + pseed).standard_normal((K, d))
            for seed in range(4):
                o = run_b(p, W0, "cyclic", 1, 1.0, n, 2000, seed=seed, zero_table=True)
                nrun += 1
                if o["viol_bound"]:
                    nviol += 1
                    worst = max(worst, o["max_ratio"])
                    if first is None:
                        first = (pseed, dist, seed, round(o["max_ratio"], 6), o["t_max_ratio"], o["viol_bound"])
                # meme run avec la table initialisee en W_0 : temoin
                o2 = run_b(p, W0, "cyclic", 1, 1.0, n, 2000, seed=seed)
                assert o2["viol_bound"] == 0
    print(f"  table nulle au depart (cyclique, b=1, A=n, 2000 iterations) : {nviol}/{nrun} runs violent "
          f"D_t <= D_0 rho^t (temoin avec table en W_0 : 0/{nrun}) ; plus grand D_t/(D_0 rho^t) = {worst:.6f} ; premier "
          f"cas (probleme, ||W_0-W_*|| ~, graine, max ratio, t, nb d'iterations en violation) = {first}")
    # (ii) mu_op > L_max : la forme close L_eff = L_op + 10 A L_max ne garantit plus la recurrence
    for lam in (200.0, 1e4):
        p = LS(1, lam=lam)
        W0 = np.random.default_rng(101).standard_normal((K, d))
        for order, A in (("cyclic", n), ("reshuffle", 2 * n - 1)):
            L_eff = p.L_op + 10 * A * p.Lmax
            rho = 1 - p.mu_op / L_eff
            need = p.L_op + 4 * (p.Lmax + p.Lbar) * sum(rho ** (-k / 2) for k in range(1, A + 1))
            T = int(min(400, 40.0 / -math.log(rho)))         # D_0 rho^T >= e^-40 D_0 : au-dessus de l'arrondi
            o = run_b(p, W0, order, 1, 1.0, A, T, seed=7)
            print(f"  lam={lam}: mu_op={p.mu_op:.2f} > L_max={p.Lmax:.2f}; {order} A={A}: rho={rho:.4f}, "
                  f"L_eff={L_eff:.1f} contre L_op+4(L_max+L_bar)sum rho^(-k/2)={need:.3e} (condition suffisante de "
                  f"la preuve {'satisfaite' if L_eff >= need else 'NON satisfaite'}) ; run de {T} iterations: "
                  f"viol(rec)={o['viol_rec']} viol(D_t<=D_0 rho^t)={o['viol_bound']} max ratio={o['max_ratio']:.4f}")
    return nviol


def part_a():
    print("=== (a) ||omega - grad F||_* <= (theta L_max + L_bar) max_i ||W - phi_i||_op, etats aleatoires ===")
    rng = np.random.default_rng(5)

    def softmax_res(Z, lab):                                  # gradient en z de l'entropie croisee : p - e_y
        Z = Z - Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(lab)), lab] -= 1.0
        return P

    for loss, c_l in (("ls", 1.0), ("softmax", 0.5)):
        worst, cnt = 0.0, 0
        worst_tri = 0.0
        for trial in range(6000):
            X = rng.standard_normal((n, d)) * rng.uniform(0.2, 3.0, size=(n, 1))
            Y = rng.standard_normal((n, K))
            lab = rng.integers(0, K, size=n)
            lam = 10 ** rng.uniform(-3, 0)
            W = rng.standard_normal((K, d)) * 10 ** rng.uniform(-1, 1)
            mode = trial % 3
            if mode == 0:                                     # points de table arbitraires
                Phi = W[None] + rng.standard_normal((n, K, d)) * 10 ** rng.uniform(-3, 1, size=(n, 1, 1))
            elif mode == 1:                                   # ecarts de rang un alignes sur x_i (cas defavorable)
                Phi = np.array([W - 10 ** rng.uniform(-2, 0.5) * np.outer(rng.standard_normal(K), X[i])
                                for i in range(n)])
            else:                                             # seul le lot est perime
                Phi = np.repeat(W[None], n, axis=0)
            b = int(rng.choice([1, 2, 3, 4, 6, 12]))
            B = rng.choice(n, size=b, replace=False)
            if mode == 2:
                for j in B:
                    Phi[j] = W - np.outer(rng.standard_normal(K), X[j])
            theta = float(rng.choice([0.0, b / n, 0.5, 1.0, rng.uniform()]))
            if loss == "ls":
                res = lambda V, i: V @ X[i] - Y[i]
            else:
                res = lambda V, i: softmax_res((V @ X[i])[None], lab[i:i + 1])[0]
            gW = np.array([np.outer(res(W, i), X[i]) for i in range(n)])
            z = np.array([np.outer(res(Phi[i], i), X[i]) for i in range(n)])
            gradF = gW.mean(axis=0) + lam * W
            omega = z.mean(axis=0) + lam * W + (theta / b) * (gW[B] - z[B]).sum(axis=0)
            Li = c_l * np.einsum("ij,ij->i", X, X)
            l = np.linalg.svd(W[None] - Phi, compute_uv=False)[:, 0]
            lhs = nuc(omega - gradF)
            rhs = (theta * Li.max() + Li.mean()) * l.max()
            # ligne intermediaire : sum_i |w_i| ||c_i||_* avec w_i = theta 1[i in B]/b - 1/n
            w = -np.ones(n) / n
            w[B] += theta / b
            tri = sum(abs(w[i]) * nuc(gW[i] - z[i]) for i in range(n))
            assert lhs <= tri * (1 + 1e-9) + 1e-12
            assert np.all(np.abs(w) <= theta / b + 1 / n + 1e-15)
            cnt += 1
            if rhs > 0:
                worst = max(worst, lhs / rhs)
                worst_tri = max(worst_tri, tri / rhs)
            assert lhs <= rhs * (1 + 1e-9) + 1e-12, (loss, trial, lhs, rhs)
        print(f"  {loss:8s} c_l={c_l}: {cnt} etats, max lhs/rhs = {worst:.4f}, max (somme triangulaire)/rhs = "
              f"{worst_tri:.4f}  (<= 1 attendu)")
    # la lecture litterale "|w_i| <= theta/b + 1/n pour tout i" ne donne PAS (a) : il faut w_i = -1/n hors du lot
    theta, b = 1.0, 1
    print(f"  remarque : sum_i (theta/b + 1/n) L_i = (theta n / b + 1) L_bar = {theta * n / b + 1:.0f} L_bar, "
          f"alors que sum_i |w_i| L_i <= theta L_max + L_bar : la borne (a) utilise |w_i| = 1/n hors du lot")


NS = (3.4445, -4.7750, 2.0315)


def ns5(M, eps):
    X = M.T if M.shape[0] > M.shape[1] else M
    X = X / (np.sqrt(np.vdot(X, X)) + eps)
    a, b_, c_ = NS
    for _ in range(5):
        A_ = X @ X.T
        Bm = b_ * A_ + c_ * A_ @ A_
        X = a * X + Bm @ X
    return X.T if M.shape[0] > M.shape[1] else X


def part_c():
    print("=== (c) covariance d'echelle exacte sur moindres carres ===")
    p = LS(1)
    X, Y = p.X, p.Y
    Rstar = p.res(p.Wstar)                                    # residus a l'optimum

    def step(E, e, B, eta, omap, theta, b):
        """Un pas sur l'etat u = (E, e) : E = W - W_*, e[:, i] = residu stocke - residu optimal."""
        W = p.Wstar + E
        R = Rstar + e
        zbar = R @ X / n
        Rb = W @ X[B].T - Y[B].T
        omega = zbar + p.lam * W + theta * (Rb - R[:, B]) @ X[B] / b
        E2 = E - eta * omap(omega)
        e2 = e.copy()
        e2[:, B] = Rb - Rstar[:, B]
        return E2, e2

    def step_lin(E, e, B, eta, omap, theta, b):
        """Meme pas ecrit directement sur u (omega lineaire en u, grad F(W_*) = 0) : sans l'arrondi de W_* + E."""
        omega = e @ X / n + p.lam * E + theta * (E @ X[B].T - e[:, B]) @ X[B] / b
        E2 = E - eta * omap(omega)
        e2 = e.copy()
        e2[:, B] = E @ X[B].T
        return E2, e2

    rng = np.random.default_rng(11)
    maps = {"polar": polar, "ns5 eps=0": lambda M: ns5(M, 0.0), "ns5 eps=1e-7": lambda M: ns5(M, 1e-7),
            "frob": lambda M: M / np.sqrt(np.vdot(M, M)), "sign": np.sign}
    # (i) omega est lineaire en u : les deux ecritures du pas coincident
    E = rng.standard_normal((K, d)); e = rng.standard_normal((K, n)); B = np.array([3, 7])
    a1 = step(E, e, B, 0.1, polar, 1.0, 2); a2 = step_lin(E, e, B, 0.1, polar, 1.0, 2)
    print(f"  linearite de omega en u : ecart entre les deux ecritures du pas = "
          f"{max(np.abs(a1[0] - a2[0]).max(), np.abs(a1[1] - a2[1]).max()):.1e}")
    # (ii) identite a un pas Phi_{s eta}(s u; B) = s Phi_eta(u; B), etats et tables quelconques
    for name, omap in maps.items():
        for s, scale_u in ((1e-3, 1.0), (1e-6, 1e-3)):
            worst = 0.0
            for trial in range(300):
                b = int(rng.choice([1, 2, 3]))
                B = rng.choice(n, size=b, replace=False)
                theta = float(rng.choice([1.0, b / n, 0.5]))
                E = scale_u * rng.standard_normal((K, d))
                e = scale_u * rng.standard_normal((K, n))
                eta = scale_u * 10 ** rng.uniform(-2, 0)
                E1, e1 = step_lin(E, e, B, eta, omap, theta, b)
                E2, e2 = step_lin(s * E, s * e, B, s * eta, omap, theta, b)
                num = np.sqrt(np.sum((E2 - s * E1) ** 2) + np.sum((e2 - s * e1) ** 2))
                den = s * np.sqrt(np.sum(E1 ** 2) + np.sum(e1 ** 2))
                worst = max(worst, num / den)
            print(f"  un pas, {name:13s} s={s:g}, ||u||~{scale_u:g}: max ecart relatif = {worst:.2e}")
    # (iii) deux trajectoires aux echelles 1 et s, memes lots independants, eta_t = eta_0 r^t
    for name in ("polar", "ns5 eps=0", "ns5 eps=1e-7"):
        omap = maps[name]
        for s in (1e-3, 1e-7):
            rngb = np.random.default_rng(21)
            E = np.random.default_rng(101).standard_normal((K, d))
            e = E @ X.T                                       # table initialisee en W_0 : e_i = E_0 x_i
            Es, es = s * E, s * e
            eta0, r = 0.3, 0.995
            line = []
            for t in range(121):
                if t in (1, 10, 30, 60, 120):
                    num = np.sqrt(np.sum((Es - s * E) ** 2) + np.sum((es - s * e) ** 2))
                    den = s * np.sqrt(np.sum(E ** 2) + np.sum(e ** 2))
                    line.append(f"t={t}: {num / den:.1e}")
                B = rngb.choice(n, size=1, replace=False)
                eta = eta0 * r ** t
                E, e = step_lin(E, e, B, eta, omap, 1.0, 1)
                Es, es = step_lin(Es, es, B, s * eta, omap, 1.0, 1)
            print(f"  trajectoires, {name:13s} s={s:g}: ecart relatif de u_t(s)/s a u_t : " + ", ".join(line))


if __name__ == "__main__":
    t0 = time.process_time()
    which = sys.argv[1:] or ["a", "b", "adv", "iid", "neg", "c"]
    if "a" in which:
        part_a()
    if "b" in which:
        part_b()
    if "adv" in which:
        part_b_adversary()
    if "iid" in which:
        part_b_iid()
    if "neg" in which:
        part_negative()
    if "c" in which:
        part_c()
    print(f"CPU {time.process_time() - t0:.1f} s")
