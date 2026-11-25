"""Audit num-m5, check 4: an independent implementation of recipe (4) of the paper,
    W_{t+1} = W_t - lr_t o(omega_t),  lr_t = lr_0 q^(t b / n),  b = 1, SAGA table, independent uniform indices,
written from the text of the paper (the loop, the table, the polar factor, ns5 and the Nesterov buffer are rewritten
here; only the data, the loss residuals and W_* come from muon/lab). It checks, on soft-cond and ls-cond:
  (1) the floors of (M5): exact polar factor (rounding), Muon's recipe = ns5 (5 steps, eps = 1e-7) + Nesterov 0.9 at
      q = 0.8, lr_0 = 0.1 (near 1e-18), the same with eps = 1e-30;
  (2) why ns5 stalls: ||D_t||_F (the matrix fed to ns5) against eps when the run leaves the law;
  (3) why the exact polar factor stalls: lr_t against the spacing of the floating-point numbers at the size of W;
  (4) line 579-580: "the error of the estimator is 2.3 to 3.5 times the gradient in nuclear norm" (mean over the
      index of ||omega_j - grad F||_* / ||grad F||_*, exact enumeration, passes 2, 6, ..., 26, polar factor, q = 0.8).

Run from the project root (about 1.5 minutes):
    PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        python3 paper/audit/checks/num-m5_independent_recipe.py
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "muon", "experiments"))
import common as C                                                    # noqa: E402

SPECS = {"soft-cond": C.spec("softmax", 200, 40, 8, 1e4, 2.0), "ls-cond": C.spec("ls", 200, 40, 8, 1e4, 2.0)}
NS = (3.4445, -4.7750, 2.0315)


def nuc(M):
    return float(np.linalg.svd(M, compute_uv=False).sum())


def polar(M, rank):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    return U[:, :rank] @ Vt[:rank]


def ns5(M, eps):
    a, b, c = NS
    X = M / (np.sqrt((M * M).sum()) + eps)
    for _ in range(5):
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X
    return X


def run(prob, o, q, lr0, passes, seed, beta=0.0, eps=None, snapshots=()):
    """Recipe (4) at batch one with a SAGA table. o in {'polar', 'ns5'}; beta > 0: Nesterov momentum before the map."""
    n, K, d, lam = prob.n, prob.K, prob.d, float(prob.lam)
    X, Ws = np.asarray(prob.X), prob.W_star
    soft = prob.loss == "softmax"
    rank = K - 1 if soft else min(K, d)                    # softmax gradients have zero column sums: rank K - 1
    center = (lambda M: M - M.mean(axis=0, keepdims=True)) if soft else (lambda M: M)
    D0 = float((Ws * Ws).sum())
    rng = np.random.default_rng(1000 + seed)
    W = np.zeros((K, d))
    R = prob.loss_residuals(W).copy()                      # table: z_i = R[i] x_i^T (loss part)
    zbar = R.T @ X / n
    buf = np.zeros((K, d))
    rec, noise = [], []
    T = int((passes - 1) * n)                              # the first pass pays for the table
    for t in range(T + 1):
        lr = lr0 * q ** (t / n)
        if t in snapshots:                                 # exact enumeration of the index
            G = prob.grad(W)
            Rn = prob.loss_residuals(W)
            x = [nuc(np.outer(Rn[j] - R[j], X[j]) + zbar + lam * W - G) for j in range(n)]
            noise.append(float(np.mean(x)) / nuc(G))
        j = int(rng.integers(n))
        r_new = prob.residual_i(W @ X[j], j)
        c = np.outer(r_new - R[j], X[j])
        omega = center(zbar + lam * W + c)
        Dm = omega
        if beta:
            buf = beta * buf + omega
            Dm = (1.0 - beta) * (omega + beta * buf)
        O = polar(Dm, rank) if o == "polar" else center(ns5(Dm, eps))
        if t % n == 0:
            rel = float(((W - Ws) ** 2).sum()) / D0
            rec.append((1 + t / n, rel, lr, float(np.sqrt((Dm * Dm).sum()))))
        W = W - lr * O
        zbar = zbar + c / n
        R[j] = r_new
    return np.array(rec), noise


for key, s in SPECS.items():
    prob = C.problem(s)
    Ws = prob.W_star
    D0 = float((Ws * Ws).sum())
    # sanity: the gradient of the lab against finite differences of F, and grad F(W_*) = 0
    rngc = np.random.default_rng(0)
    W, E = rngc.standard_normal(Ws.shape), rngc.standard_normal(Ws.shape)
    fd = (prob.F(W + 1e-6 * E) - prob.F(W - 1e-6 * E)) / 2e-6
    print(f"\n===== {key}: finite-difference check of grad F: {fd:.8g} vs {float((prob.grad(W) * E).sum()):.8g};"
          f" ||grad F(W_*)||_F = {np.linalg.norm(prob.grad(Ws)):.1e}; max |W_*| = {np.abs(Ws).max():.2f},"
          f" median |W_*| = {np.median(np.abs(Ws)):.2f}")
    lr_polar = 0.1 if key == "soft-cond" else 0.3          # selected points of M9 part B
    confs = [("exact polar factor, q = 0.8, lr_0 = %g" % lr_polar, dict(o="polar", q=0.8, lr0=lr_polar)),
             ("Muon's recipe: ns5 (eps = 1e-7) + Nesterov 0.9, q = 0.8, lr_0 = 0.1",
              dict(o="ns5", q=0.8, lr0=0.1, beta=0.9, eps=1e-7)),
             ("the same with eps = 1e-30", dict(o="ns5", q=0.8, lr0=0.1, beta=0.9, eps=1e-30))]
    for name, kw in confs:
        floors, widths, at = [], [], []
        for seed in (10, 11, 12):
            rec, _ = run(prob, passes=300, seed=seed, **kw)
            p, rel, lr, dn = rec.T
            ratio = rel * D0 / lr ** 2
            win = (rel >= 1e-12) & (rel <= 1e-6)
            level = np.median(ratio[win])
            k0 = int(np.argmax(rel <= 1e-6))
            k1 = k0 + int(np.nonzero(ratio[k0:] <= 3 * level)[0][-1])
            x = ratio[(p >= 0.5 * p[k1]) & (p <= p[k1])]
            floors.append(rel.min())
            widths.append(x.max() / x.min())
            at.append((p[k1], lr[k1], dn[k1]))
        at = np.median(np.array(at), axis=0)
        print(f"-- {name}\n   lowest relative squared distance, 3 seeds: {', '.join(f'{v:.2e}' for v in floors)};"
              f" width of dist2/lr2 (second half): {', '.join(f'{v:.1f}' for v in widths)}\n"
              f"   leaves the law near pass {at[0]:.0f}: lr_t = {at[1]:.2e}, ||D_t||_F = {at[2]:.2e}"
              f" (ns5 divides by ||D_t||_F + eps; double-precision spacing at 1.0: 2.2e-16)", flush=True)
    for lr0 in ((0.1,) if key == "soft-cond" else (0.1, 0.3)):
        snaps = [(p - 1) * prob.n for p in (2, 6, 10, 14, 18, 22, 26)]
        vals = []
        for seed in (10, 11):
            _, noise = run(prob, o="polar", q=0.8, lr0=lr0, passes=28, seed=seed, snapshots=snaps)
            vals.append(noise)
            print(f"-- E_j ||omega_j - grad F||_* / ||grad F||_*, polar, q = 0.8, lr_0 = {lr0}, seed {seed}, passes 2..26:"
                  f" {', '.join(f'{v:.2f}' for v in noise)}", flush=True)
        print(f"   range over the two seeds: [{np.min(vals):.2f}, {np.max(vals):.2f}]")
