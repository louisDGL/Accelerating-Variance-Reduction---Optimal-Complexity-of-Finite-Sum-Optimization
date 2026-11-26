"""Sampling WITHOUT replacement from a finite population of n matrices, nuclear norm.

G_b := zero of G -> E polar(G + xi_b), xi_b = mean of a uniformly random b-subset of the (centred) population.
Prediction (first order, finite-population cumulants):  G_b ~ g_* (n - 2b) / (b (n - 2)),  i.e. the coefficient
    c_b := b <G_b, g_*> / |g_*|^2  ~  (n - 2b) / (n - 2)        (with replacement: 1),
with a sign change at b = n/2.

Population A ("generic"): the n = 400, 2 x 3 population of checks/c05_without_replacement.py (same seed and draw
    order; centred Gamma(1) entries).  g_* from checks/lib.general_prediction (Monte Carlo, 4e6 antithetic pairs).
    For b > n/2 the complement is sampled and the exact identity xi_b = -((n-b)/b) xi_{n-b} is used.
    The zero is computed by common random numbers (6 unknowns) on each of NB independent batches.
Population B ("lattice"): 2 x 2 matrices with entries in {-1, 2}; the 16 patterns with multiplicities
    mult * 2^(number of entries equal to -1): n = 81 * mult (mult = 8: n = 648; mult = 128: n = 10368).  Its mean, covariance and third
    cumulant coincide exactly with those of the i.i.d. law (2/3, 1/3), so g_* = 0.113774 11^T (closed form).
    Batches are drawn exactly through the multivariate hypergeometric law of the pattern counts.  By symmetry
    G_b = (m_b / b) 11^T and c_b = m_b / 0.113774; m_b minimises the sample mean of ||m 11^T + S_b||_* (with a
    mean-zero linear control variate), S_b = b xi_b.
Standard errors: standard deviation over the independent batches / sqrt(number of batches)
(16 batches; 64 for the lattice population with n = 10368).  Finished tasks are checkpointed in wor_tasks.jsonl.

usage: exp_wor.py [nproc | assemble]   -> papers/figures/data/wor.json
"""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import numpy as np
from scipy.optimize import root, minimize_scalar
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures", "data", "wor.json")
sys.path.insert(0, os.path.join(HERE, "..", "..", "checks"))
G22 = 3 * (3 + 2 * (3 * np.sqrt(2) - 2) / 7) / 16 / 6          # 0.113774


def population_A():
    rng = np.random.default_rng(3)
    _ = rng.gamma(1.0, 1.0, (12, 2, 2))                         # c05 draws its small population first
    pop = rng.gamma(1.0, 1.0, (400, 2, 3)) - 1.0
    return pop - pop.mean(0)


def polar23(M):
    """polar factor of (S, 2, 3) arrays: (M M^T)^(-1/2) M in closed form."""
    x, y = M[:, 0, :], M[:, 1, :]
    a = (x * x).sum(1); c = (y * y).sum(1); bb = (x * y).sum(1)
    m12 = x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0]; m13 = x[:, 0] * y[:, 2] - x[:, 2] * y[:, 0]
    m23 = x[:, 1] * y[:, 2] - x[:, 2] * y[:, 1]
    s = np.sqrt(m12 * m12 + m13 * m13 + m23 * m23); den = s * np.sqrt(a + c + 2 * s)
    P = np.empty_like(M)
    P[:, 0, :] = ((c + s)[:, None] * x - bb[:, None] * y) / den[:, None]
    P[:, 1, :] = ((a + s)[:, None] * y - bb[:, None] * x) / den[:, None]
    return P


def task_A(args):
    b, seed, S, gstar = args
    pop = population_A(); n = len(pop); X = pop.reshape(n, -1)
    rng = np.random.default_rng(seed)
    bb = min(b, n - b); xi = np.empty((S, 6)); ch = 25_000
    for s0 in range(0, S, ch):
        u = rng.random((ch, n))
        kth = np.partition(u, bb - 1, axis=1)[:, bb - 1:bb]
        xi[s0:s0 + ch] = ((u <= kth).astype(float) @ X) / bb
    if bb != b: xi *= -(n - b) / b                               # exact complement identity
    xi = xi.reshape(S, 2, 3)
    pred = gstar * (n - 2 * b) / (b * (n - 2))
    f = lambda g: polar23(g.reshape(1, 2, 3) + xi).mean(0).ravel()
    sol = root(f, pred.ravel(), method="hybr", tol=1e-12)
    G = sol.x.reshape(2, 3)
    return "A", b, float(b * (G * gstar).sum() / (gstar * gstar).sum()), float(np.linalg.norm(f(sol.x))), \
        float(b * np.linalg.norm(G - pred) / np.linalg.norm(gstar))


PAT = np.array([[(i >> j) & 1 for j in range(4)] for i in range(16)])        # 1 = entry equal to 2
def nuc2(a, b, c, d):
    return np.sqrt(a * a + b * b + c * c + d * d + 2 * np.abs(a * d - b * c))


def task_B(args):
    b, seed, S, mult = args
    colors = mult * 2 ** (4 - PAT.sum(1)); n = int(colors.sum())
    rng = np.random.default_rng(seed)
    cnt = rng.multivariate_hypergeometric(colors, b, size=S, method="marginals")
    Sb = (-b + 3.0 * (cnt @ PAT)).T                             # (4, S): entries of the batch SUM
    T = Sb.sum(0); h = 0.05
    D = (nuc2(*(Sb + h)) - nuc2(*(Sb - h))) / (2 * h)
    beta = np.cov(D, T)[0, 1] / T.var(); Tm = T.mean()
    f = lambda m: nuc2(*(Sb + m)).mean() - beta * m * Tm
    r = minimize_scalar(f, bounds=(-1.5, 1.5), method="bounded", options={"xatol": 2e-5})
    return "B%d" % n, b, float(r.x / G22), 0.0, 0.0


if __name__ == "__main__":
    assemble = len(sys.argv) > 1 and sys.argv[1] == "assemble"   # only rebuild wor.json from the checkpoint (no computation)
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 and not assemble else 5
    t0 = time.time()
    from lib import general_prediction
    pop = population_A(); n = len(pop); X = pop.reshape(n, -1)
    Sig = X.T @ X / n; K3 = np.einsum("na,nb,nc->abc", X, X, X) / n
    GC = os.path.join(HERE, "..", "figures", "data", "wor_gstar.json")     # checkpoint of g_* (slow to compute)
    if os.path.exists(GC):
        gstar, gstar2 = [np.array(x) for x in json.load(open(GC))]
    else:
        gstar, _, _ = general_prediction(Sig, K3, 2, 3)                       # seed 1 (as c05)
        gstar2, _, _ = general_prediction(Sig, K3, 2, 3, seed=2)              # independent repeat: Monte-Carlo error of g_*
        json.dump([gstar.tolist(), gstar2.tolist()], open(GC, "w"))
    print("g* (A) =", np.round(gstar.ravel(), 4), " repeat:", np.round(gstar2.ravel(), 4),
          " rel. diff %.4f" % (np.linalg.norm(gstar - gstar2) / np.linalg.norm(gstar)), " (%.0fs)" % (time.time() - t0), flush=True)
    NB = 16
    bA = [8] + list(range(25, 400, 25)) + [392]
    tasks = [task for b in bA for task in [(task_A, (b, 7_000_000 + 1000 * b + k, 250_000, gstar)) for k in range(NB)]]
    for mult in (8, 128):
        nB = 81 * mult
        st = nB // 12
        bB = sorted({b0 + r for b0 in range(st, nB - 2, st) for r in (0, 1, 2)} | {st // 4 + r for r in (0, 1, 2)} | {nB - st // 4 - r for r in (0, 1, 2)})
        tasks += [(task_B, (b, 9_000_000 + 1000 * b + k, 500_000, mult)) for b in bB for k in range(NB if mult == 8 else 4 * NB)]   # 64 batches for the large lattice population
    CK = os.path.join(HERE, "..", "figures", "data", "wor_tasks.jsonl")    # per-task checkpoint (resume after interruption)
    res = {}; done = set()
    if os.path.exists(CK):
        for line in open(CK):
            name, b, seed, c, resid, dev = json.loads(line)
            done.add(seed); res.setdefault(name, {}).setdefault(b, []).append((c, resid, dev))
    todo = [(fn, a) for fn, a in tasks if a[1] not in done]
    print("%d tasks, %d already done" % (len(tasks), len(tasks) - len(todo)), flush=True)
    if assemble: todo = []
    with Pool(nproc if todo else 1) as p, open(CK, "a") as fh:
        futs = [(a[1], p.apply_async(fn, (a,))) for fn, a in todo]
        for seed, fu in futs:
            name, b, c, resid, dev = fu.get()
            fh.write(json.dumps([name, b, seed, c, resid, dev]) + "\n"); fh.flush()
            res.setdefault(name, {}).setdefault(b, []).append((c, resid, dev))
    out = dict(gstar_A=gstar.tolist(), gstar_A_repeat=gstar2.tolist(), gstar_B=G22, NB=NB,
               samples_per_batch=dict(A=250_000, B=500_000), seconds=time.time() - t0, populations={})
    for name, d in res.items():
        n = 400 if name == "A" else int(name[1:])
        rows = []
        for b in sorted(d):
            v = np.array(d[b]); c = v[:, 0]
            rows.append(dict(b=b, frac=b / n, mod3=b % 3, nbatch=len(c), coef=float(c.mean()), se=float(c.std(ddof=1) / np.sqrt(len(c))),
                             pred=(n - 2 * b) / (n - 2), max_resid=float(v[:, 1].max()), dev_norm=float(v[:, 2].mean())))
            print(f"{name:5s} b={b:4d} b/n={b/n:.3f}  coef = {rows[-1]['coef']:+.4f} +- {rows[-1]['se']:.4f}   pred {rows[-1]['pred']:+.4f}", flush=True)
        out["populations"][name] = dict(n=n, rows=rows)
    json.dump(out, open(OUT, "w"), indent=1)
    print("time %.0fs" % (time.time() - t0))
