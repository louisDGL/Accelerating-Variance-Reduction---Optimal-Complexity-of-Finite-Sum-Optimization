"""Audit code-fidelity, bloc (a)/(b) : le code de muon/lab fait-il ce que le papier decrit (lignes 196-206, 253-262,
508-515, Proposition 5) ? Controles unitaires independants, sur de petites instances, sans cache.

Lancer depuis la racine du projet :
    PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        python3 paper/audit/checks/code-fidelity_lab_units.py
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "muon", "experiments"))
import common as C                                                    # noqa: E402
from lab import algorithms as alg, maps, mvr, polar, vrmuon as vm     # noqa: E402

rng = np.random.default_rng(123)
OK = []


def check(name, cond, detail=""):
    OK.append(bool(cond))
    print(f"[{'OK ' if cond else 'FAIL'}] {name}  {detail}")


# ------------------------------------------------------------------ 1. ns5 = routine publique de Muon
def ns5_reference(G, steps=5, eps=1e-7):
    """zeropower_via_newtonschulz5 du depot public de Muon, recopie de memoire, en float64."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (np.linalg.norm(G) + eps)
    tr = G.shape[0] > G.shape[1]
    if tr:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.T if tr else X


for shape in ((8, 40), (20, 20), (40, 8)):
    M = rng.standard_normal(shape)
    d = np.abs(polar.ns5(M) - ns5_reference(M)).max()
    check(f"ns5 {shape} = routine de reference (5 iterations, eps 1e-7)", d < 1e-12, f"ecart max {d:.1e}")
M = rng.standard_normal((8, 40))
sv = np.linalg.svd(polar.ns5(M), compute_uv=False)
check("ns5 : valeurs singulieres dans une bande autour de 1 (pas exactement 1)", 0.6 < sv.min() and sv.max() < 1.25,
      f"[{sv.min():.3f}, {sv.max():.3f}]")
r1 = np.linalg.norm(polar.ns5(1e-12 * M)) / np.linalg.norm(polar.ns5(M))
check("ns5 n'est pas invariant d'echelle a cause de eps (||ns5(1e-12 M)|| / ||ns5(M)|| << 1)", r1 < 0.1, f"{r1:.2e}")
r2 = np.abs(polar.ns5(1e-12 * M, eps=1e-30) - polar.ns5(M, eps=1e-30)).max()
check("ns5 avec eps = 1e-30 : invariant d'echelle", r2 < 1e-9, f"ecart {r2:.1e}")

# ------------------------------------------------------------------ 2. facteur polaire
U, s, Vt = np.linalg.svd(M, full_matrices=False)
check("polar_svd(M) = U V^T", np.abs(polar.polar_svd(M) - U @ Vt).max() < 1e-12)
check("polar_svd(0) = 0", not polar.polar_svd(np.zeros((3, 5))).any())
check("frob : M / ||M||_F", np.abs(polar.frob_normalize(M) - M / np.linalg.norm(M)).max() < 1e-15)
check("sign : signe coordonnee par coordonnee", (polar.sign_map(M) == np.sign(M)).all())

# ------------------------------------------------------------------ 3. estimateur SAGA, table, longueurs
for loss in ("softmax", "ls"):
    s_ = C.spec(loss, 60, 6, 4, 50.0, 1.0)
    prob = C.problem(s_)
    n, X, lam = prob.n, prob.X, prob.lam
    W = prob.remove_gauge(0.3 * rng.standard_normal((prob.K, prob.d)))
    W0 = np.zeros_like(W)

    def gi(Wm, i):                                   # gradient de la partie perte g_i (sans le regulariseur)
        return prob.grad_i(Wm, i) - lam * Wm

    # table explicite z_i = grad g_i(phi_i), comparee a l'etat du laboratoire le long de quelques pas
    for b, theta in ((1, 1.0), (4, 0.3)):
        algo = vm.VRMuon(prob, estimator="saga", batch=b, theta=theta, placement="joint", length="sharp",
                         gamma=0.1 / prob.R2_op)
        st = algo.init_state(W0)
        Z = np.array([gi(W0, i) for i in range(n)])
        Wc = W0.copy()
        worst = 0.0
        r2 = np.random.default_rng(5)
        for t in range(25):
            B = int(r2.integers(n)) if b == 1 else r2.choice(n, size=b, replace=False)
            Bl = [B] if b == 1 else list(B)
            m, c, aux = algo.pieces(Wc, st, B)
            omega_lab = m + theta * c
            omega_ref = Z.mean(0) + lam * Wc + theta / b * sum(gi(Wc, j) - Z[j] for j in Bl)
            worst = max(worst, np.abs(omega_lab - omega_ref).max())
            S, info = algo.step(Wc, m, c, st, t)
            # pas a longueur nucleaire : S = gamma ||omega||_* polar(omega) (dans la jauge pour softmax)
            Og = polar.polar_svd(prob.remove_gauge(omega_ref), rtol=1e-12)
            nuc = np.linalg.svd(prob.remove_gauge(omega_ref), compute_uv=False).sum()
            worst = max(worst, np.abs(S - algo.gamma * nuc * Og).max() / max(1e-300, np.abs(S).max()))
            algo.commit(Wc, st, B, 1.0, aux, info)
            for j in Bl:
                Z[j] = gi(Wc, j)                     # table remplacee par le gradient au point d'AVANT le pas
            Wc = Wc - S
        check(f"{loss}, b={b}, theta={theta} : omega = zbar + lam W + (theta/b) sum (grad g_j - z_j), table au point "
              f"courant, pas gamma ||omega||_* polar(omega)", worst < 1e-9, f"ecart max {worst:.1e}")

    # longueur fixe, ecretee, carte soft
    algo = vm.VRMuon(prob, estimator="saga", length="const", lr=0.05, schedule="geometric", lr_decay=0.8)
    st = algo.init_state(W0)                         # table en W_0 = 0, estimateur evalue en W : innovation non nulle
    m, c, aux = algo.pieces(W, st, 3)
    assert np.linalg.norm(c) > 0
    S, info = algo.step(W, m, c, st, 7)
    check(f"{loss} : longueur fixe, ||S||_op = lr_0 q^(t b / n)",
          abs(np.linalg.norm(S, 2) - 0.05 * 0.8 ** (7 / n)) < 1e-12, f"{np.linalg.norm(S, 2):.6g}")
    algo = vm.VRMuon(prob, estimator="saga", batch=4, length="const", lr=0.05, schedule="geometric", lr_decay=0.8)
    check(f"{loss} : lr_t = lr_0 q^(t b / n) pour b = 4", abs(algo._lr_t(30) - 0.05 * 0.8 ** (30 * 4 / n)) < 1e-15)
    nuc = np.linalg.svd(prob.remove_gauge(m + c), compute_uv=False).sum()
    for rho in (1e-3, 1e3):
        algo = vm.VRMuon(prob, estimator="saga", length="clip", rho=rho, gamma=0.2 / prob.R2_op)
        S, info = algo.step(W, m, c, algo.init_state(W0), 0)
        check(f"{loss} : ecrete, ||S||_op = min(rho, gamma ||omega||_*), rho = {rho:g}",
              abs(np.linalg.norm(S, 2) - min(rho, algo.gamma * nuc)) < 1e-10 * max(1, rho))
    algo = vm.VRMuon(prob, estimator="saga", orth="soft", eps_innov=0.3, gamma=0.2 / prob.R2_op)
    S, info = algo.step(W, m, c, algo.init_state(W0), 0)
    om = prob.remove_gauge(m + c)
    Us, ss, Vs = np.linalg.svd(om, full_matrices=False)
    eps = 0.3 * np.linalg.norm(c)
    Osoft = Us @ np.diag(ss / np.sqrt(ss ** 2 + eps ** 2)) @ Vs
    check(f"{loss} : carte soft = U diag(s / sqrt(s^2 + eps^2)) V^T avec eps = eps_innov ||theta c||_F (variable), "
          f"longueur gamma <omega, O>", np.abs(S - algo.gamma * np.vdot(om, Osoft) * Osoft).max() < 1e-10)

    # MISO : theta = 1/n, gamma = 1/lam  =>  W_t = - zbar_t / lam
    cfg = C.resolve(dict(estimator="saga", placement="euclid", miso_delta=1.0), prob)
    check(f"{loss} : MISO = (theta = 1/n, gamma = 1/lam, table remplacee)",
          abs(cfg["theta"] - 1 / n) < 1e-15 and abs(cfg["gamma"] - 1 / lam) < 1e-9 / lam and cfg["mem_relax"] == 1.0)
    algo = vm.VRMuon(prob, **cfg)
    st = algo.init_state(W0)
    Wc, worst = W0.copy(), 0.0
    for t in range(40):
        j = int(rng.integers(n))
        m, c, aux = algo.pieces(Wc, st, j)
        S, info = algo.step(Wc, m, c, st, t)
        algo.commit(Wc, st, j, 1.0, aux, info)
        Wc = Wc - S
        worst = max(worst, np.abs(Wc + st.zbar / lam).max() / max(1e-300, np.abs(Wc).max()))
    check(f"{loss} : MISO garde l'itere au minimiseur du substitut, W = - zbar / lam", worst < 1e-9, f"{worst:.1e}")

    # metrique polaire figee en W_0 : C = V diag(nu / max(s_k, f s_1)) V^T + r (I - V V^T), pas c / R_P^2
    f = 0.1
    algo = vm.VRMuon(prob, estimator="saga", placement="metric", metric_every=10 ** 9, metric_floor=f, c_metric=0.5,
                     theta=0.6)
    st = algo.init_state(W0)
    m, c, aux = algo.pieces(W0, st, 5)
    S, info = algo.step(W0, m, c, st, 0)
    G0 = prob.grad(W0)
    check(f"{loss} : l'ancre de la metrique est le gradient complet en W_0", np.abs(m - G0).max() < 1e-12)
    Q = maps.gauge_basis(prob)
    R0 = G0 if Q is None else Q.T @ G0
    _, sg, Vg = np.linalg.svd(R0, full_matrices=False)
    r = len(sg)
    Cm = Vg.T @ np.diag(sg.sum() / np.maximum(sg, f * sg[0])) @ Vg + r * (np.eye(prob.d) - Vg.T @ Vg)
    R2P = prob.c_loss * np.einsum("ij,jk,ik->i", X, Cm, X).max() + lam * np.linalg.norm(Cm, 2)
    check(f"{loss} : pas de la metrique polaire = (c / R_P^2) (m + theta c) C, R_P^2 = max_i c x_i^T C x_i + lam ||C||_op",
          np.abs(S - 0.5 / R2P * (m + 0.6 * c) @ Cm).max() < 1e-10 * np.abs(S).max(), f"R2P {R2P:.4g}")
    for t in range(3 * n):                              # jamais rafraichie (one-shot)
        j = int(rng.integers(n))
        m, c, aux = algo.pieces(W0, st, j)
        S, info = algo.step(W0, m, c, st, t)
        algo.commit(W0, st, j, 1.0, aux, info)
    check(f"{loss} : metrique one-shot non rafraichie apres 3 passes", st.metric is not None)

    # echantillonnage d'importance : estimateur sans biais
    algo = vm.VRMuon(prob, estimator="saga", placement="euclid", sampling=1.0)
    st = algo.init_state(W0)
    mean = sum(algo.probs[i] * (algo.pieces(W, st, i)[0] + algo.pieces(W, st, i)[1]) for i in range(n))
    check(f"{loss} : echantillonnage d'importance p_i ~ L_i, estimateur sans biais",
          np.abs(mean - prob.grad(W)).max() < 1e-10, f"{np.abs(mean - prob.grad(W)).max():.1e}")
    check(f"{loss} : L_i = c ||x_i||^2 + lam", np.abs(prob.L_i - (prob.c_loss * (X ** 2).sum(1) + lam)).max() < 1e-12)

    # moment de Nesterov 0.9 : D = (1 - beta)(omega + beta buf), buf = beta buf + omega  ==  forme lerp de Muon
    algo = vm.VRMuon(prob, estimator="saga", length="const", lr=0.01, orth="frob", beta=0.9, nesterov=True)
    st = algo.init_state(W0)
    ema, Wc, worst = np.zeros_like(W0), W0.copy(), 0.0
    for t in range(15):
        j = int(rng.integers(n))
        m, c, aux = algo.pieces(Wc, st, j)
        S, info = algo.step(Wc, m, c, st, t)
        om = m + c
        ema = 0.9 * ema + 0.1 * om                       # buf.lerp_(g, 1 - beta)
        D = 0.1 * om + 0.9 * ema                         # g.lerp_(buf, beta)
        Dg = prob.remove_gauge(D)
        worst = max(worst, np.abs(S - 0.01 * Dg / np.linalg.norm(Dg)).max())
        algo.commit(Wc, st, j, 1.0, aux, info)
        Wc = Wc - S
    check(f"{loss} : moment de Nesterov 0.9 avant la carte (forme lerp de Muon), controle frob", worst < 1e-12,
          f"{worst:.1e}")

# ------------------------------------------------------------------ 4. tirage des lots, comptage des passes
n, b = 1000, 20
dr = list(vm._draws(np.random.default_rng(0), n, b, 200, False, None, "iid"))
check("lots independants (b > 1) : indices distincts dans un lot", all(len(set(B)) == b for B, _ in dr))
dr = list(vm._draws(np.random.default_rng(0), n, b, 150, False, None, "reshuffle"))
ok = all(sorted(np.concatenate([B for B, _ in dr[p * 50:(p + 1) * 50]])) == list(range(n)) for p in range(3))
diff = any((np.concatenate([B for B, _ in dr[:50]]) != np.concatenate([B for B, _ in dr[50:100]])))
check("'reshuffle' : chaque indice une fois par passe, permutation nouvelle a chaque passe", ok and diff)
bs = list(alg._batch_stream(np.random.default_rng(0), 200, 32, 400, "iid"))
dup = np.mean([len(set(B)) < 32 for B in bs])
check("methodes sans table (muon_sgd, mvr) : lots tires AVEC remise (doublons possibles)", dup > 0.5,
      f"fraction de lots avec doublon {dup:.2f} (n = 200, b = 32)")

s_ = C.spec("ls", 60, 6, 4, 50.0, 1.0)
prob = C.problem(s_)
W0 = np.zeros((prob.K, prob.d))
r = vm.vr_muon(prob, W0, 30, seed=0, estimator="saga", batch=4, gamma=0.1 / prob.R2_op, record_every=1)
check("SAGA : cout initial n compte (1 passe), puis b par iteration",
      r["grad_evals"][0] == prob.n and np.all(np.diff(r["grad_evals"]) == 4), f"ge[:3] = {r['grad_evals'][:3]}")
r = vm.vr_muon(prob, W0, 30, seed=0, estimator="sgd", batch=4, length="const", lr=0.01, record_every=1)
check("gradient de lot sans table : cout initial 0", r["grad_evals"][0] == 0)
r = mvr.mvr_muon(prob, W0, 10, batch=4, lr=0.01, g=0.025, clip=1.0, record_every=1)
check("MARS-M (meme lot) : b au premier pas puis 2b", list(np.diff(r["grad_evals"])[:3]) == [4, 8, 8],
      f"{list(np.diff(r['grad_evals'])[:4])}")

# ------------------------------------------------------------------ 5. MARS-M / STORM (mvr.py)
f = alg.lr_factors(["geometric", 0.75, 200 / (2.0 * 1)], 1000)
check("MARS-M / STORM, refroidissement : lr_t = lr_0 q^(2 b t / n) (par passe de gradients)",
      abs(f[400] - 0.75 ** 4) < 1e-12)

print(f"\n{sum(OK)}/{len(OK)} controles passes")
