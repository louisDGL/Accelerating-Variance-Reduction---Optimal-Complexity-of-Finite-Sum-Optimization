"""Counter-verification of finding F1 of the audit dimension num-noise ((M2), l. 499-503 of the paper):
"[the joint SAGA-Muon] fails where MISO fails (kappa >> n): it is not a spectral method".

Independent implementation (plain numpy loops written here; muon/lab is used ONLY to build the problem instance and to
read its minimizer, so that the instances are those of exp_M6 and exp_M1; the auditor's script is not used):

  joint SAGA-Muon   omega = zbar + lam W + (r_j(W) - rtab_j) x_j^T,   W <- W - gamma ||omega||_* polar(omega),
                    gamma = c / R2_op,  R2_op = max_i c_l ||x_i||^2 + lam min(K, d)   (softmax: polar factor taken in
                    the subspace orthogonal to the all-ones vector)
  MISO (delta = 1)  W <- W - (1 / lam) (zbar + lam W + (1 / n) (r_j(W) - rtab_j) x_j^T)
  table             rtab_j <- r_j(W) at the sampled index (both methods), initialized at W_0 = 0 (cost: one pass).

Parts (argument):
  m6soft, m6ls   instances of exp_M6 with kappa / n = 10 (n = 1000, d = 50, K = 10, alpha = 1): MISO and the joint step
                 for c = 1, 1/2, ..., 1/128, 40 passes (the budget of M6), three seeds;
  m6ref          softmax instance of exp_M6 with kappa = n: MISO and joint c = 1/4 (the paper quotes 11.9 and 11.8);
  m1             the two non-square conditioning-limited instances of the paper (soft-cond, ls-cond; n = 200, d = 40,
                 K = 8, kappa = 50 n): MISO and the joint step for c = 1, ..., 1/32, 100 passes, three seeds.
Printed per run: passes to 1e-6 (relative squared distance, the initial pass of the table included), final value,
and the medians over the run (states above 1e-14) of ||m||_* / ||grad F||_* and of the nuclear rank of omega.

Usage (project root):
  PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python3 paper/audit/checks/verif-num-noise_joint_tuned.py m6soft
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "muon", "experiments"))
import common as C  # noqa: E402  (instance and minimizer only)


def instance(loss, n, d, K, kappa, alpha):
    prob = C.problem(C.spec(loss, n, d, K, kappa, alpha))
    X = np.array(prob.X)
    lam = float(prob.lam)
    cl = 0.5 if loss == "softmax" else 1.0
    sq = (X ** 2).sum(axis=1)
    R2 = float(cl * sq.max() + lam)
    R2op = float(cl * sq.max() + lam * min(K, d))
    Ws = np.array(prob.W_star)
    y = np.array(prob.y)
    # own check of the minimizer
    Z = X @ Ws.T
    Rs = resid_all(loss, Z, y)
    g = Rs.T @ X / n + lam * Ws
    return dict(loss=loss, X=X, y=y, lam=lam, R2=R2, R2op=R2op, Ws=Ws, n=n, d=d, K=K,
                gstar=float(np.linalg.norm(g)), kappa=R2 / lam)


def resid_all(loss, Z, y):
    if loss == "ls":
        return Z - y
    E = np.exp(Z - Z.max(axis=1, keepdims=True))
    P = E / E.sum(axis=1, keepdims=True)
    P[np.arange(Z.shape[0]), y] -= 1.0
    return P


def run(inst, method, c, seed, passes, tol=1e-6):
    loss, X, y, lam, n, d, K = (inst[k] for k in ("loss", "X", "y", "lam", "n", "d", "K"))
    Ws = inst["Ws"]
    D0 = float((Ws ** 2).sum())
    soft = loss == "softmax"
    if soft:                                   # orthonormal basis of the complement of the all-ones vector
        Q, _ = np.linalg.qr(np.eye(K) - 1.0 / K)
        Q = Q[:, : K - 1]
    gamma = c / inst["R2op"] if method == "joint" else 1.0 / lam
    rng = np.random.default_rng(424242 + 7919 * seed)
    W = np.zeros((K, d))
    R = resid_all(loss, X @ W.T, y)
    zbar = R.T @ X / n
    T = int(round((passes - 1) * n))
    idx = rng.integers(0, n, size=T)
    rec = max(1, n // 10)
    ps, rel, base, nrank = [1.0], [1.0], [], []
    diverged = False
    for t in range(T):
        j = idx[t]
        x = X[j]
        z = W @ x
        if soft:
            e = np.exp(z - z.max())
            r = e / e.sum()
            r[y[j]] -= 1.0
        else:
            r = z - y[j]
        dr = r - R[j]
        m = zbar + lam * W
        if method == "joint":
            om = m + np.outer(dr, x)
            A = Q.T @ om if soft else om
            U, s, Vt = np.linalg.svd(A, full_matrices=False)
            O = U @ Vt
            if soft:
                O = Q @ O
            S = gamma * s.sum() * O
            if t % n == n // 2:
                Gf = resid_all(loss, X @ W.T, y).T @ X / n + lam * W
                nm = np.linalg.svd(m, compute_uv=False).sum()
                ng = np.linalg.svd(Gf, compute_uv=False).sum()
                if rel[-1] > 1e-14 and ng > 0:
                    base.append(nm / ng)
                    nrank.append(s.sum() ** 2 / (s ** 2).sum())
        else:
            S = (m + np.outer(dr, x) / n) / lam
        W -= S
        zbar += np.outer(dr, x) / n
        R[j] = r
        if (t + 1) % rec == 0:
            v = float(((W - Ws) ** 2).sum()) / D0
            ps.append(1.0 + (t + 1) / n)
            rel.append(v)
            if not np.isfinite(v) or v > 1e8:
                diverged = True
                break
            if v < 1e-22:
                break
    ps, rel = np.array(ps), np.array(rel)
    k = np.nonzero(rel <= tol)[0]
    return dict(passes_to=float(ps[k[0]]) if len(k) else float("inf"), final=float(rel[-1]), last=float(ps[-1]),
                diverged=diverged, base=float(np.median(base)) if base else float("nan"),
                nrank=float(np.median(nrank)) if nrank else float("nan"), ps=ps, rel=rel)


def show(label, rs):
    p = [r["passes_to"] for r in rs]
    f = [r["final"] for r in rs]
    div = sum(r["diverged"] for r in rs)
    base = np.nanmedian([r["base"] for r in rs]) if np.isfinite([r["base"] for r in rs]).any() else float("nan")
    nr = np.nanmedian([r["nrank"] for r in rs]) if np.isfinite([r["nrank"] for r in rs]).any() else float("nan")
    print(f"  {label:26s} passes to 1e-6: {'  '.join(f'{v:6.1f}' for v in p)}   final: "
          f"{'  '.join(f'{v:8.1e}' for v in f)}   diverged {div}/{len(rs)}   ||m||*/||gradF||* {base:6.3f}   "
          f"nrank(omega) {nr:5.2f}", flush=True)


def block(title, inst, cs, seeds, passes):
    print(f"\n{title}: n = {inst['n']}, K x d = {inst['K']} x {inst['d']}, R2/lam = {inst['kappa']:.0f} = "
          f"{inst['kappa'] / inst['n']:.1f} n, R2_op = {inst['R2op']:.4f}, ||grad F(W_*)||_F = {inst['gstar']:.1e}; "
          f"{passes} passes, seeds {seeds}")
    t0 = time.process_time()
    show("MISO (delta = 1)", [run(inst, "miso", None, s, passes) for s in seeds])
    for c in cs:
        show(f"joint, gamma = {c:g}/R2_op", [run(inst, "joint", c, s, passes) for s in seeds])
    print(f"  [cpu {time.process_time() - t0:.0f} s]", flush=True)


part = sys.argv[1] if len(sys.argv) > 1 else "m1"
GRID = (1.0, 0.5, 0.25, 0.125, 1 / 16, 1 / 32, 1 / 64, 1 / 128)
if part == "m6soft":
    block("exp_M6 softmax, kappa/n = 10", instance("softmax", 1000, 50, 10, 1e4, 1.0), GRID, (0, 1, 2), 40)
elif part == "m6ls":
    block("exp_M6 least squares, kappa/n = 10", instance("ls", 1000, 50, 10, 1e4, 1.0), GRID, (0, 1, 2), 40)
elif part == "m6ref":
    block("exp_M6 softmax, kappa/n = 1", instance("softmax", 1000, 50, 10, 1e3, 1.0), (1.0, 0.25), (0, 1, 2), 40)
elif part == "m1":
    block("soft-cond (M1)", instance("softmax", 200, 40, 8, 1e4, 2.0), GRID[:6], (10, 11, 12), 100)
    block("ls-cond (M1)", instance("ls", 200, 40, 8, 1e4, 2.0), GRID[:6], (10, 11, 12), 100)
