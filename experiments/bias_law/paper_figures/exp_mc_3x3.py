"""Monte-Carlo estimate of m_b = b G_b for 3 x 3 nuclear norm, lattice noise (entries of S_b: -b + 3 Bin(b, 1/3)).

Same estimator as master/verification/orbital/t6_median.py (not modified): for each independent batch of n samples,
minimise over m the sample average of ||m 11^T + S||_* minus a mean-zero linear control variate beta m <S, 11^T>
(common random numbers inside a batch).  Reported value = mean of the batch minimisers; standard error = standard
deviation of the batch minimisers / sqrt(number of batches)  (batch means over independent seeds).
The nuclear norm of a 3 x 3 matrix is evaluated in closed form (trigonometric eigenvalues of M M^T).

usage: exp_mc_3x3.py [nproc] [scale] [b1,b2,...]  -> results/bias_law/figure_data/mc_3x3.json
(with a list of b: only these are computed, with 16 batches of 5e5, and merged into the existing file)
"""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import numpy as np
from scipy.optimize import minimize_scalar
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "..", "results", "bias_law", "figure_data", "mc_3x3.json")
Q33 = 0.68691                      # t6_output.txt: 0.68691 +- 0.00005
GSTAR = (3 + 2 * Q33) / 50


def nuc3(e):
    """nuclear norms of 3 x 3 matrices given as a list of 9 arrays (row-major)."""
    a11 = e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; a22 = e[3] * e[3] + e[4] * e[4] + e[5] * e[5]
    a33 = e[6] * e[6] + e[7] * e[7] + e[8] * e[8]
    a12 = e[0] * e[3] + e[1] * e[4] + e[2] * e[5]; a13 = e[0] * e[6] + e[1] * e[7] + e[2] * e[8]
    a23 = e[3] * e[6] + e[4] * e[7] + e[5] * e[8]
    q = (a11 + a22 + a33) / 3
    d1, d2, d3 = a11 - q, a22 - q, a33 - q
    p2 = d1 * d1 + d2 * d2 + d3 * d3 + 2 * (a12 * a12 + a13 * a13 + a23 * a23)
    p = np.sqrt(p2 / 6); ps = np.where(p > 0, p, 1.0)
    det = d1 * (d2 * d3 - a23 * a23) - a12 * (a12 * d3 - a23 * a13) + a13 * (a12 * a23 - d2 * a13)
    phi = np.arccos(np.clip(det / (2 * ps ** 3), -1.0, 1.0)) / 3
    l1 = q + 2 * p * np.cos(phi); l3 = q + 2 * p * np.cos(phi + 2 * np.pi / 3); l2 = 3 * q - l1 - l3
    return np.sqrt(np.maximum(l1, 0)) + np.sqrt(np.maximum(l2, 0)) + np.sqrt(np.maximum(l3, 0))


def batch(args):
    b, seed, n = args
    rng = np.random.default_rng(seed)
    mins = []
    for _ in range(max(1, n // 500_000)):          # sub-batches of 5e5 (memory); each gives one minimiser
        nn = min(n, 500_000)
        S = (-b + 3.0 * rng.binomial(b, 1 / 3, size=(9, nn)))
        T = S.sum(axis=0)                                   # exact mean 0, variance 18 b
        h = 0.05
        D = (nuc3([s + GSTAR + h for s in S]) - nuc3([s + GSTAR - h for s in S])) / (2 * h)
        beta = np.cov(D, T)[0, 1] / (18 * b); Tm = T.mean()
        f = lambda m: nuc3([s + m for s in S]).mean() - beta * m * Tm
        r = minimize_scalar(f, bounds=(-1.0, 1.2), method="bounded", options={"xatol": 1e-4})
        mins.append(r.x)
    return b, mins


if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    plan = {}
    for b in (6, 7, 8, 12, 13, 14, 30, 31, 32, 99, 100, 101, 300, 301, 302): plan[b] = int(16 * scale)     # x 5e5 samples
    for b in (999, 1000, 1001, 3000, 3001, 3002): plan[b] = int(128 * scale)
    old = []
    if len(sys.argv) > 3:
        plan = {int(x): int(16 * scale) for x in sys.argv[3].split(",")}
        old = [r for r in json.load(open(OUT))["results"] if r["b"] not in plan]
    tasks = [(b, 10_000 * b + k, 2_000_000) for b, nsub in plan.items() for k in range(nsub // 4)]
    tasks.sort(key=lambda a: -a[0])
    t0 = time.time(); res = {b: [] for b in plan}
    with Pool(nproc) as p:
        for b, mins in p.imap_unordered(batch, tasks, chunksize=1):
            res[b] += mins
    out = []
    for b in sorted(res):
        x = np.array(res[b])
        out.append(dict(b=b, mod3=b % 3, m=float(x.mean()), se=float(x.std(ddof=1) / np.sqrt(len(x))), nbatch=len(x),
                        nsamples=len(x) * 500_000, gstar=GSTAR, minimisers=x.tolist()))
        print(f"b={b:5d} mod3={b%3}  m = {x.mean():+.5f} +- {out[-1]['se']:.5f}   ratio {x.mean()/GSTAR:.3f} +- {out[-1]['se']/GSTAR:.3f}  ({len(x)} batches)")
    out = sorted(out + old, key=lambda r: r["b"])
    json.dump(dict(gstar=GSTAR, q33=Q33, seconds=time.time() - t0, batch_size=500_000,
                   note="m = mean of batch minimisers; se = sd of batch minimisers / sqrt(nbatch)", results=out),
              open(OUT, "w"), indent=1)
    print("time %.0fs" % (time.time() - t0))
