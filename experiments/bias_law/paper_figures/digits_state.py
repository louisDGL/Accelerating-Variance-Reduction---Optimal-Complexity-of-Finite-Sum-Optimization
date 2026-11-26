"""Digits instance, parameter-dependent noise, large batches: the iteration W_{t+1} = W_t - eta_t polar(g_t(W_t)) with
g_t the minibatch gradient of b samples drawn WITH replacement, sampled through multinomial counts (cost independent of b).
Started at W_* (not at the prediction), Polyak average over the second half; reports b*grad F(W_b) against g_*,
b^2 * excess loss against <g_*, H^{-1} g_*>/2, and the two quarters of the tail.  Usage: digits_state.py b [T] [R]"""
import os
for v in ('OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[v] = '2'
import sys, json, time, numpy as np
from digits_problem import *
b = int(sys.argv[1]); T = int(float(sys.argv[2])) if len(sys.argv) > 2 else 60_000; R = int(sys.argv[3]) if len(sys.argv) > 3 else 16
p = problem(); K, d, n = p['K'], p['d'], p['n']; X, Y, Hx, Ws = p['X'], p['Y'], p['Hx'], p['Ws']
V = per_sample_grads(p)
gs = np.load(os.path.join(DATA, 'digits_gstar.npz'))['gs'].mean(0)              # (K,d)
pred_loss = 0.5 * np.sum(gs * np.linalg.solve(Hx, gs.T).T)
sc = np.sqrt(np.trace(V.T @ V / n) / (K * d)); lmax = np.linalg.eigvalsh(Hx)[-1]
rng = np.random.default_rng([11, b]); pv = np.full(n, 1.0 / n)
W = np.tile(Ws, (R, 1, 1)); acc = np.zeros((2, R, K, d)); cnt = [0, 0]; tic = time.time()
eta0 = 0.5 * sc / np.sqrt(b) / lmax; t0 = 200.0
for t in range(T):
    C = rng.multinomial(b, pv, size=R) / b                                      # (R,n)
    A = np.einsum('rn,ni,nj->rij', C, X, X, optimize=True)                      # (R,d,d)
    B = np.einsum('rn,nk,nj->rkj', C, Y, X, optimize=True)                      # (R,K,d)
    G = W @ A - B + LAM * W
    U, s, Vt = np.linalg.svd(G, full_matrices=False)
    W -= eta0 * (1 + t / t0) ** -0.6 * (U @ Vt)
    if t >= T // 2:
        h = int(t >= 3 * T // 4); acc[h] += W; cnt[h] += 1
nuc = lambda g: np.linalg.svd(g, compute_uv=False).sum()
def stats(Wm):
    D = Wm - Ws; g = D @ Hx                                                     # grad F at the averaged iterate (F quadratic)
    return dict(nuc_ratio=float(b * nuc(g) / nuc(gs)), proj=float(b * np.sum(g * gs) / np.sum(gs * gs)),
                loss_ratio=float(b * b * 0.5 * np.sum(D * (D @ Hx)) / pred_loss),
                cos_W=float(np.sum(D * np.linalg.solve(Hx, gs.T).T) / np.linalg.norm(D) / np.linalg.norm(np.linalg.solve(Hx, gs.T))))
W1, W2 = acc[0] / cnt[0], acc[1] / cnt[1]
per = [stats(0.5 * (W1[r] + W2[r]))['proj'] for r in range(R)]
out = dict(b=b, T=T, R=R, all=stats(0.5 * (W1 + W2).mean(0)), q3=stats(W1.mean(0)), q4=stats(W2.mean(0)),
           proj_se=float(np.std(per, ddof=1) / np.sqrt(R)), seconds=time.time() - tic)
print(json.dumps(out), flush=True)
with open(os.path.join(DATA, 'digits_state.jsonl'), 'a') as f: f.write(json.dumps(out) + '\n')
