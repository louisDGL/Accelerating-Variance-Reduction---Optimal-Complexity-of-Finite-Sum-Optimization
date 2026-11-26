"""Run the stochastic polar iteration on the digits least-squares problem for one (beta, b):
W_{t+1} = W_t - eta_t polar(M_t), M_t = beta M_{t-1} + (1-beta) g_t, g_t = minibatch gradient (b samples WITH replacement),
eta_t = eta0 (1 + t/t0)^(-0.75), R parallel runs from W = 0, exact polar factor by batched SVD.
Saves per-run averages of W over the two halves of the last third of the run."""
import os
for v in ('OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[v] = '1'
import sys, time, numpy as np
from digits_problem import *


def simulate(beta, b, R, T, eta0, t0, seed=0, log=None):
    p = problem(); X, Y, n, K, d = p['X'], p['Y'], p['n'], p['K'], p['d']
    rng = np.random.default_rng([seed, b, int(round(100 * beta))])
    W = np.zeros((R, K, d)); M = np.zeros((R, K, d))
    t1 = T - T // 3; th = t1 + (T - t1) // 2
    acc = np.zeros((2, R, K, d)); cnt = [0, 0]; tic = time.time()
    for t in range(T):
        idx = rng.integers(n, size=(R, b))
        Xb = X[idx]                                             # (R,b,d)
        res = Xb @ W.transpose(0, 2, 1) - Y[idx]                # (R,b,K)
        G = res.transpose(0, 2, 1) @ Xb / b + LAM * W
        M = beta * M + (1 - beta) * G
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        W -= eta0 * (1 + t / t0) ** -0.75 * (U @ Vt)
        if t >= t1:
            h = int(t >= th); acc[h] += W; cnt[h] += 1
        if log and t % (T // 10) == 0:
            print(f'beta={beta} b={b} t={t} F-F*={F(p, W.mean(0)) - F(p, p["Ws"]):.3e} {time.time()-tic:.0f}s', file=log, flush=True)
    return acc[0] / cnt[0], acc[1] / cnt[1], eta0 * (1 + T / t0) ** -0.75


if __name__ == '__main__':
    beta, b, R, T, eta0, t0 = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])
    tagname = sys.argv[7] if len(sys.argv) > 7 else 'main'
    os.makedirs(os.path.join(DATA, 'digits_runs'), exist_ok=True)
    tic = time.time()
    W1, W2, eta_end = simulate(beta, b, R, T, eta0, t0, log=sys.stdout)
    np.savez(os.path.join(DATA, 'digits_runs', f'{tagname}_beta{beta}_b{b}.npz'), W1=W1, W2=W2, beta=beta, b=b, R=R, T=T, eta0=eta0, t0=t0,
             eta_end=eta_end, seconds=time.time() - tic)
    print('done', beta, b, time.time() - tic)
