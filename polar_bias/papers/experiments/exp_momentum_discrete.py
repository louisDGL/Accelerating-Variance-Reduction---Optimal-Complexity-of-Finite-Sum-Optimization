"""Momentum iteration with discrete (lattice) gradient noise, 3 x 3 matrices.

    M_t = beta M_{t-1} + (1 - beta) (grad F(W_t) + xi_t),   W_{t+1} = W_t - eta_t o(M_t),   eta_t = (t + 1)^(-3/4),
    F(W) = 1/2 sum_ij A_ij (W_ij - Wstar_ij)^2  (strongly convex quadratic, A_ij in [0.5, 2]),
    xi_t = mean of b i.i.d. matrices with i.i.d. entries in {-1, 2}, P = (2/3, 1/3),
    o = polar factor (kind 'nuc')  or  entrywise sign (kind 'sign').  M_0 = 0, W_0 = 0.

For each of R independent runs we record the time average of grad F(W_t) over the last half of the run (and, as a
drift diagnostic, over the third quarter), averaged over the 9 entries, together with the same average of xi_t
(exact mean zero: a control variate, used only for the variance-reduced numbers).

usage: exp_momentum_discrete.py [nproc] [T]   -> papers/figures/data/momentum_runs.jsonl
"""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures", "data", "momentum_runs.jsonl")
A = np.array([[1.0, 2.0, 0.5], [1.5, 0.75, 1.25], [0.6, 1.8, 1.0]])
WSTAR = np.array([[0.7, -1.3, 0.4], [2.0, -0.5, 1.1], [-0.9, 0.3, 1.6]])


def run(args):
    kind, b, beta, T, R, seed = args
    t0 = time.time()
    rng = np.random.default_rng(seed)
    W = np.zeros((R, 3, 3)); M = np.zeros((R, 3, 3))
    half, q3 = T // 2, (T // 2, 3 * T // 4)
    accG = np.zeros((R, 3, 3)); accX = np.zeros((R, 3, 3)); accG3 = np.zeros((R, 3, 3))
    chunk = 1024; t = 0
    while t < T:
        m = min(chunk, T - t)
        X = (3.0 * rng.binomial(b, 1 / 3, (m, R, 3, 3)) - b) / b
        etas = (np.arange(t + 1, t + m + 1) + 1.0) ** (-0.75)
        for i in range(m):
            G = A * (W - WSTAR)
            M = beta * M + (1 - beta) * (G + X[i])
            if kind == "nuc":
                U, s, Vt = np.linalg.svd(M)
                O = U @ Vt
            else:
                O = np.sign(M)
            if t + i >= half:
                accG += G; accX += X[i]
                if t + i < q3[1]: accG3 += G
            W -= etas[i] * O
        t += m
    n = T - half
    return dict(kind=kind, b=b, beta=beta, T=T, R=R, seed=seed, seconds=time.time() - t0,
                G_run=(accG / n).mean(axis=(1, 2)).tolist(),          # per run: entrywise mean of time-averaged gradient
                X_run=(accX / n).mean(axis=(1, 2)).tolist(),
                G3_run=(accG3 / (q3[1] - q3[0])).mean(axis=(1, 2)).tolist(),
                G_entry=(accG / n).mean(axis=0).tolist())             # per entry: mean over runs


if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    T = int(float(sys.argv[2])) if len(sys.argv) > 2 else 1_000_000
    R = 64
    tasks = []
    for beta in (0.0, 0.5, 0.9):
        for b in (1, 2, 4, 8, 16):
            nseed = {8: 3, 16: 4}.get(b, 1) if beta == 0.9 else 1     # noisiest configurations: 192 and 256 runs
            tasks += [("nuc", b, beta, T, R, 100 * b + int(10 * beta) + 7919 * k) for k in range(nseed)]
    tasks += [("nuc", b, 0.0, T, R, 100 * b) for b in (3, 12)]    # class b = 0 mod 3 (pure batch only)
    tasks.sort(key=lambda a: -a[1] * (1 + a[2]))
    tasks += [("sign", b, beta, T, R, 50_000 + 100 * b + int(10 * beta)) for beta in (0.0, 0.5, 0.9)
              for b in (1, 2, 3, 4, 6, 8, 12, 16)]
    t0 = time.time()
    with Pool(nproc) as p, open(OUT, "w") as fh:
        for rec in p.imap_unordered(run, tasks, chunksize=1):
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            g = np.array(rec["G_run"]); beff = rec["b"] * (1 + rec["beta"] + rec["beta"] ** 2) / (1 - rec["beta"] ** 2)
            print(f"[{time.time()-t0:6.0f}s] {rec['kind']:4s} b={rec['b']:3d} beta={rec['beta']} b_eff*G = {beff*g.mean():+.4f} "
                  f"+- {beff*g.std(ddof=1)/np.sqrt(len(g)):.4f}  ({rec['seconds']:.0f}s)", flush=True)
