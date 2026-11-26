"""FIG B data.  (a,c) zeros of the frozen mean field G -> E polar(G + xi_b) for the five noise laws of checks/c03_general_law.py
(2 x 3), with block-wise Monte-Carlo standard errors, plus the sign-descent limit (-median);  (b) momentum (EMA) filtered noise.
Every (law, b) is split into independent blocks of BLK batch means; the zero is solved per block (common random numbers,
exact closed-form 2 x d polar factor) and the blocks are averaged; the standard error is the block-to-block one.
Predictions use EXACT cumulants of the laws (the c03 script used empirical ones) and the exact q_{2,3}."""
import os
for v in ('OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[v] = '1'
import numpy as np, time, sys
from scipy.optimize import root
from common import *
import lib

K, d = 2, 3
BLK = 2_000_000
# ---- constants of the laws, copied from checks/c03_general_law.py ---------------------------------------------------
RHO = 0.6; KSH = (2 / (RHO ** 3 * 2 * np.sqrt(2) / (1 - RHO ** 2) ** 1.5)) ** 2
Q = np.linalg.qr(np.random.default_rng(5).standard_normal((d, K)))[0].T
RP = 0.9; Mfac = RP * Q / np.sqrt(K)
Cfill = np.linalg.cholesky(np.eye(K * d) - np.outer(Mfac.ravel(), Mfac.ravel()))
Ld = np.array([[1.0, 0.0], [0.6, 0.8]]) * np.array([[1.5], [0.7]])
Lx = np.diag([2.0, 1.0, 0.5]) @ np.array([[1, 0, 0], [0.5, 1, 0], [-0.3, 0.4, 1.0]])


def cgam(rng, shape, size):               # centred unit-variance Gamma(shape)
    return (rng.standard_gamma(shape, size) - shape) / np.sqrt(shape)


def bm_iid(rng, b, m):                    # exact: mean of b Exp(1)-1
    return rng.standard_gamma(b, (m, K, d)) / b - 1.0


def bm_coskew(rng, b, m):                 # exact batch mean of the c03 'coskew' law
    z = rng.standard_normal((m, K, d)) / np.sqrt(b)
    chi = rng.chisquare(b - 1, m) if b > 1 else 0.0
    m2 = (b * z[:, 0, 1] ** 2 + chi) / b - 1.0                      # mean of z12^2 - 1
    gm = -(rng.standard_gamma(b * KSH, m) - b * KSH) / (b * np.sqrt(KSH))
    z[:, 0, 0] = RHO * m2 / np.sqrt(2) + np.sqrt(1 - RHO ** 2) * gm
    return z


def bm_factor(rng, b, m):                 # exact batch mean of the c03 'factor' law
    f = rng.standard_gamma(b, (m, 1, 1)) / b - 1.0
    return f * Mfac[None] + (rng.standard_normal((m, K * d)) @ Cfill.T).reshape(m, K, d) / np.sqrt(b)


def nblk(name, b):                        # block size: smaller for the brute-force laws at large b
    return 1_000_000 if (name in ('rank1', 'kron') and b >= 256) else BLK


def _brute(rng, b, m, one):
    out = np.empty((m, K, d)); ch = max(1, 2_000_000 // b)
    for s in range(0, m, ch):
        mm = min(ch, m - s); out[s:s + mm] = one(mm * b).reshape(mm, b, K, d).mean(1)
    return out


def bm_rank1(rng, b, m):
    return _brute(rng, b, m, lambda n: cgam(rng, 1.0, (n, K, 1)) * cgam(rng, 1.0, (n, 1, d)))


def bm_kron(rng, b, m):
    def one(n):
        de = cgam(rng, 1.0, (n, K)) @ Ld.T; x = cgam(rng, 2.0, (n, d)) @ Lx.T
        return de[:, :, None] * x[:, None, :]
    return _brute(rng, b, m, one)


BM = dict(iid=bm_iid, rank1=bm_rank1, coskew=bm_coskew, factor=bm_factor, kron=bm_kron)


def zero(xi, x0):
    f = lambda g: polar2(g.reshape(1, K, d) + xi).mean(0).ravel()
    sol = root(f, x0.ravel(), method='hybr', tol=1e-13)
    return sol.x, float(np.linalg.norm(f(sol.x)))


def task_law(a):
    name, b, blk, x0 = a
    rng = np.random.default_rng([abs(hash(name)) % 1000 if False else sorted(BM).index(name), b, blk, 2024])
    S = nblk(name, b)
    xi = BM[name](rng, b, S)
    G, r = zero(xi, np.asarray(x0) / b)
    med = -np.median(xi.reshape(S, -1), axis=0)
    return ('law', name, b, (b * G).tolist(), (b * med).tolist(), r)


def weights(beta, tol=1e-7):
    if beta == 0:
        return np.ones(1)
    k = np.arange(int(np.ceil(np.log(tol) / np.log(beta))) + 1)
    return (1 - beta) * beta ** k


def task_mom(a):
    beta, b, blk, x0 = a
    rng = np.random.default_rng([int(round(beta * 100)), b, blk, 4048])
    S = 1_000_000
    N = np.zeros((S, K, d))
    for wk in weights(beta):
        N += wk * (rng.standard_gamma(b, (S, K, d)) / b - 1.0)
    G, r = zero(N, np.asarray(x0))
    return ('mom', beta, b, G.tolist(), None, r)


def run(a):
    t = time.time(); out = (task_law if a[0] == 'law' else task_mom)(a[1:])
    return out + (time.time() - t,)


def exact_predictions():
    q = exact_q(K, d)[0]; a1, a2 = a_consts(K, d, q)
    g = {}
    g['iid'] = (a1 + a2) * 2.0 / 6 * np.ones((K, d))                 # A = B = kappa3 * ones, kappa3 = 2
    g['rank1'] = (a1 + a2) * 4.0 / 6 * np.ones((K, d))               # A = B = (kappa3)^2 * ones
    e11 = np.zeros((K, d)); e11[0, 0] = 1
    g['coskew'] = (a1 + a2) * RHO * np.sqrt(2) / 6 * e11             # A = B = rho sqrt2 e_11
    g['factor'] = (a1 * 2 * (Mfac ** 2).sum() * Mfac + a2 * 2 * Mfac @ Mfac.T @ Mfac) / 6   # K3 = 2 M(x)M(x)M
    # kron: exact covariance and third cumulant, operators J, T by Monte Carlo (lib.general_prediction, 4 seeds)
    Sig = np.kron(Ld @ Ld.T, Lx @ Lx.T)
    k3d = 2.0 * np.einsum('am,bm,cm->abc', Ld, Ld, Ld); k3x = np.sqrt(2.0) * np.einsum('am,bm,cm->abc', Lx, Lx, Lx)
    K3 = np.einsum('ace,bdf->abcdef', k3d, k3x)
    gs = [lib.general_prediction(Sig, K3, K, d, S=4_000_000, seed=s, direction=polar2)[0] for s in range(4)]
    g['kron'] = np.mean(gs, 0)
    kron_se = float(np.linalg.norm(np.std(gs, 0, ddof=1)) / 2)
    return dict(q=q, a1=a1, a2=a2), g, kron_se


if __name__ == '__main__':
    t0 = time.time()
    consts, gst, kron_se = exact_predictions()
    print('constants', consts, 'kron g* MC se (Frobenius)', kron_se, flush=True)
    for k, v in gst.items():
        print(k, np.round(v.ravel(), 5), flush=True)
    NB = dict(iid={1: 8, 4: 8, 16: 8, 64: 8, 256: 12, 1024: 30}, coskew={1: 8, 4: 8, 16: 8, 64: 10, 256: 24, 1024: 80},
              factor={1: 8, 4: 8, 16: 10, 64: 16, 256: 48, 1024: 100}, rank1={4: 8, 16: 8, 64: 8, 256: 8, 1024: 8},
              kron={4: 8, 16: 8, 64: 8, 256: 8, 1024: 8})
    tasks = [('law', n, b, k, gst[n].tolist()) for n in NB for b in NB[n] for k in range(NB[n][b])]
    gper = float(gst['iid'][0, 0])
    MB = {1: 4, 4: 6, 16: 12}
    for beta in (0.0, 0.5, 0.9, 0.95):
        beff = (1 + beta + beta ** 2) / (1 - beta ** 2)
        tasks += [('mom', beta, b, k, (gper / (b * beff) * np.ones(6)).tolist()) for b in MB for k in range(MB[b])]
    cost = lambda a: (a[2] * (5 if a[1] in ('rank1', 'kron') else 0.02) if a[0] == 'law' else 60 * len(weights(a[1])))
    tasks.sort(key=cost, reverse=True)
    cache = os.path.join(DATA, 'fig_law_blocks.jsonl')             # one line per finished block (survives interruptions)
    res = {}; done = set()
    if os.path.exists(cache):
        for l in open(cache):
            r = json.loads(l); res.setdefault(r['key'], []).append(r); done.add((r['key'], r['blk']))
    key = lambda a: f'{a[0]}|{a[1]}|{a[2]}'
    dump(dict(consts=consts, gstar={k: v.tolist() for k, v in gst.items()}, kron_gstar_se=kron_se, block=BLK, block_brute_large_b=1_000_000), 'fig_law_pred.json')
    tasks = [a for a in tasks if (key(a), a[3]) not in done]
    tasks.sort(key=cost)                                           # cheapest first
    limit = float(sys.argv[1]) if len(sys.argv) > 1 else 480       # seconds; rerun to continue (resumable)
    print(len(done), 'blocks cached;', len(tasks), 'to do', flush=True)
    # strictly sequential, one block (<= 2e6 2x3 matrices, ~0.5 GB peak) in memory at a time
    for i, a in enumerate(tasks):
        if time.time() - t0 > limit:
            print('time limit reached;', len(tasks) - i, 'blocks left'); break
        out = run(a)
        r = dict(key=key(a), blk=a[3], v=out[3], sign=out[4], resid=out[5], sec=out[6])
        with open(cache, 'a') as f:
            f.write(json.dumps(r) + '\n')
        if i % 10 == 0:
            print(i, len(tasks), f'{time.time()-t0:.0f}s', r['key'], f'{out[6]:.0f}s', flush=True)
    print('done', time.time() - t0)
