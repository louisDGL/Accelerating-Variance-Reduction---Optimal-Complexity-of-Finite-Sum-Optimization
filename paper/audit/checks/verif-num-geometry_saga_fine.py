"""Counter-check of finding G2 (num-geometry): passes of plain SAGA at batch one when its single step is tuned on a
grid of ratio 2^(1/4) instead of the factor-2 grid of M1 / M8, on the three data draws of soft-cond and ls-cond.

Independent implementation: the SAGA loop below is written from scratch (uniform i.i.d. indices, loss-part table,
W_0 = 0, initial full gradient counted as one pass, record every 0.1 pass, target: relative squared distance 1e-6).
The repository is used only to build the data (X, labels / targets, lambda) and for W_* (whose gradient is rechecked
here), and, with the option `repo`, to rerun the repository's own SAGA at chosen steps on the confirmation seeds.

Mini-protocol (mirrors the repository: tune on seeds 0-4, confirm on fresh seeds 10-19):
    grid c = c0 * 2^(k/4), gamma = c / R2 ; selected step = smallest median over the 5 tuning seeds ;
    reported = median over the 10 confirmation seeds. The RNG streams differ from the repository's.

Run (one instance and data draw per call, well under 3 minutes of CPU):
    PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        python3 paper/audit/checks/verif-num-geometry_saga_fine.py softmax 2 [repo]
Read-only on the repository.
"""
import os
import statistics as st
import sys
import time

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'muon', 'experiments'))
sys.path.insert(0, os.path.join(ROOT, 'muon'))
import common as C  # noqa: E402

loss = sys.argv[1] if len(sys.argv) > 1 else 'softmax'
data_seed = int(sys.argv[2]) if len(sys.argv) > 2 else 2
with_repo = len(sys.argv) > 3 and sys.argv[3] == 'repo'
TOL = 1e-6
BUDGET = 300.0
Q = 2 ** 0.25

spec = C.spec(loss, 200, 40, 8, 1e4, 2.0, seed=data_seed)
prob = C.problem(spec)
X = np.array(prob.X, dtype=float)
n, d = X.shape
K = prob.K
lam = float(prob.lam)
Wstar = np.array(prob.W_star, dtype=float)
D0 = float(np.sum(Wstar ** 2))
if loss == 'softmax':
    y = np.asarray(prob.y, dtype=int)
    c_loss = 0.5
else:
    Y = np.array(prob.Y, dtype=float)
    c_loss = 1.0
R2 = c_loss * float((X ** 2).sum(axis=1).max()) + lam


def resid_all(W):
    Z = X @ W.T
    if loss == 'softmax':
        Z = Z - Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(n), y] -= 1.0
        return P
    return Z - Y


def resid_one(z, j):
    if loss == 'softmax':
        e = np.exp(z - z.max())
        e /= e.sum()
        e[y[j]] -= 1.0
        return e
    return z - Y[j]


g_star = resid_all(Wstar).T @ X / n + lam * Wstar
print('instance %s n=%d d=%d K=%d data seed %d : lam = %.6e, R2 = %.6f (repo %.6f), ||grad F(W_*)||_F = %.2e' % (
    loss, n, d, K, data_seed, lam, R2, prob.R2, np.linalg.norm(g_star)))


def saga(c, seed):
    """Plain SAGA, gamma = c / R2. Returns passes to TOL (inf if not reached within BUDGET or diverged)."""
    gamma = c / R2
    rng = np.random.default_rng(100003 * (seed + 1) + 17)
    W = np.zeros((K, d))
    Rt = resid_all(W)
    zbar = Rt.T @ X / n
    its = int(round((BUDGET - 1.0) * n))
    idx = rng.integers(0, n, size=its)
    rec = n // 10
    shrink = 1.0 - gamma * lam
    for t in range(its):
        j = idx[t]
        x = X[j]
        r = resid_one(W @ x, j)
        dr = r - Rt[j]
        o = np.outer(dr, x)
        W *= shrink
        W -= gamma * (zbar + o)
        zbar += o / n
        Rt[j] = r
        if (t + 1) % rec == 0:
            rel = float(np.sum((W - Wstar) ** 2)) / D0
            if not np.isfinite(rel) or rel > 1e6:
                return float('inf')
            if rel <= TOL:
                return (n + t + 1) / n
    return float('inf')


def repo_saga(c, seed):
    r = C.run_one({'spec': spec, 'method': 'vr', 'cfg': dict(estimator='saga', placement='euclid',
                                                             gamma=C.over('R2', c)),
                   'passes': BUDGET, 'seed': seed, 'stop_rel': 0.5 * TOL})
    return r['passes_to']['1e-06']


def med(xs):
    return st.median(xs)


c0 = 2.0 if loss == 'softmax' else 1.0
grid = [c0 * Q ** k for k in range(9)]
factor2 = [grid[0], grid[4], grid[8]]
TUNE = [0, 1, 2, 3, 4]
CONF = list(range(10, 20))

t0 = time.process_time()
print('own SAGA, tuning seeds 0-4 (median passes to 1e-6; budget %g passes):' % BUDGET)
tune = {}
for c in grid:
    p = [saga(c, s) for s in TUNE]
    tune[c] = med(p)
    print('   c = %6.3f  %s  median %s  [%s]' % (
        c, '(factor-2 grid)' if any(abs(c - f) < 1e-9 for f in factor2) else '               ', tune[c],
        ', '.join('%.1f' % v for v in p)), flush=True)
best_fine = min(grid, key=lambda c: tune[c])
best_f2 = min(factor2, key=lambda c: tune[c])
print('selected on the 2^(1/4) grid: c = %.3f ; on the factor-2 grid: c = %.3f' % (best_fine, best_f2))
res = {}
for label, c in (('factor-2', best_f2), ('2^(1/4)', best_fine)):
    p = [saga(c, s) for s in CONF]
    res[label] = med(p)
    print('own SAGA, confirmation seeds 10-19, %-9s c = %.3f : median %.2f  [%.1f, %.1f]  reached %d/10' % (
        label, c, med(p), min(p), max(p), sum(np.isfinite(p))), flush=True)
print('fine / factor-2 = %.3f  (CPU %.0f s)' % (res['2^(1/4)'] / res['factor-2'], time.process_time() - t0))

if with_repo:
    for label, c in (('factor-2', best_f2), ('2^(1/4)', best_fine)):
        p = [repo_saga(c, s) for s in CONF]
        print('repo SAGA (common.run_one), seeds 10-19, %-9s c = %.3f : median %.2f  [%.1f, %.1f]  reached %d/10' % (
            label, c, med(p), min(p), max(p), sum(np.isfinite(p))), flush=True)
    print('(CPU %.0f s)' % (time.process_time() - t0))
