"""Diagnostic for the digits instance: FROZEN noise.  Zero G_b of the mean field G -> E polar(G + xi_b), xi_b the mean of b
per-sample gradients at W_* drawn with replacement (multinomial counts), by Robbins-Monro with Polyak averaging.
Separates the convergence of the law-level statement  b G_b -> g_*  from the parameter dependence of the noise.
Memory: O(R n) integers per step.  Usage: digits_frozen.py b [T] [R]"""
import os
for v in ('OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[v] = '2'
import sys, json, time, numpy as np
from digits_problem import *
b = int(sys.argv[1]); T = int(float(sys.argv[2])) if len(sys.argv) > 2 else 100_000; R = int(sys.argv[3]) if len(sys.argv) > 3 else 32
p = problem(); K, d, n = p['K'], p['d'], p['n']
V = per_sample_grads(p); V = V - V.mean(0)
gs = np.load(os.path.join(DATA, 'digits_gstar.npz'))['gs'].mean(0).reshape(-1)
rng = np.random.default_rng([7, b])
pv = np.full(n, 1.0 / n)
sc = np.sqrt(np.trace(V.T @ V / n) / (K * d))            # noise scale per entry
G = np.tile(gs / b, (R, 1)) * 0.0                         # start at 0 (not at the prediction)
acc = np.zeros((2, R, K * d)); cnt = [0, 0]; tic = time.time()
eta0 = 0.5 * sc / np.sqrt(b); t0 = 200.0
for t in range(T):
    C = rng.multinomial(b, pv, size=R)                    # (R,n)
    xi = (C @ V) / b
    U, s, Vt = np.linalg.svd((G + xi).reshape(R, K, d), full_matrices=False)
    G -= eta0 * (1 + t / t0) ** -0.6 * (U @ Vt).reshape(R, -1)
    if t >= T // 2:
        h = int(t >= 3 * T // 4); acc[h] += G; cnt[h] += 1
G1, G2 = acc[0] / cnt[0], acc[1] / cnt[1]; Gm = 0.5 * (G1 + G2)
proj = b * (Gm @ gs) / (gs @ gs)
nuc = lambda g: np.linalg.svd(g.reshape(K, d), compute_uv=False).sum()
out = dict(b=b, T=T, R=R, proj_mean=float(proj.mean()), proj_se=float(proj.std(ddof=1) / np.sqrt(R)),
           proj_half1=float((b * (G1 @ gs) / (gs @ gs)).mean()), proj_half2=float((b * (G2 @ gs) / (gs @ gs)).mean()),
           nuc_ratio_of_mean=float(b * nuc(Gm.mean(0)) / nuc(gs)),
           cos=float(Gm.mean(0) @ gs / np.linalg.norm(Gm.mean(0)) / np.linalg.norm(gs)), seconds=time.time() - tic)
print(json.dumps(out), flush=True)
with open(os.path.join(DATA, 'digits_frozen.jsonl'), 'a') as f: f.write(json.dumps(out) + '\n')
