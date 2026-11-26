"""FIG A data: exact q_{K,d}, a1, a2 for many sizes (wish.py), proved envelope, Marchenko-Pastur limits."""
import numpy as np, time
from multiprocessing import Pool
from scipy import integrate
from common import *

import os
MAXK = int(os.environ.get('MAXK', 256))          # largest min(K,d) attempted (cost grows steeply: matrix size K, precision 60+3K digits)
SQ = [k for k in [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256] if k <= MAXK]
SIZES = {'1': [(k, k) for k in SQ], '2': [(k, 2 * k) for k in SQ if 2 * k <= 256], '4': [(k, 4 * k) for k in SQ if 4 * k <= 256]}
TABLE = [(2, 2), (2, 3), (3, 3), (3, 5), (4, 4), (8, 8), (16, 16), (64, 64), (8, 32), (32, 128), (128, 128)]


def row(Kd):
    K, d = Kd; t = time.time()
    q, chk, dps = exact_q(K, d); a1, a2 = a_consts(K, d, q); m = max(K, d)
    return dict(K=K, d=d, q=q, a1=a1, a2=a2, a1_plus_a2=a1 + a2, lower=3 / (K + d + 1), upper=3 / (m + 2),
                scaled=m * (a1 + a2), scaled_lower=3 * m / (K + d + 1), scaled_upper=3 * m / (m + 2),
                identity_check=chk, dps=dps, seconds=time.time() - t)


def kappa(g):
    """kappa(gamma) = [(1+gamma) mu_{1/2} - mu_{3/2}]/(gamma mu_{1/2}), mu_s = int x^s dMP_gamma(x) (eigenvalue law of ZZ^T/d)."""
    lo, hi = (1 - np.sqrt(g)) ** 2, (1 + np.sqrt(g)) ** 2
    dens = lambda x: np.sqrt(max((hi - x) * (x - lo), 0.0)) / (2 * np.pi * g * x)
    mu = lambda s: integrate.quad(lambda x: x ** s * dens(x), lo, hi, limit=400, epsabs=1e-13, epsrel=1e-13)[0]
    return ((1 + g) * mu(0.5) - mu(1.5)) / (g * mu(0.5)), mu(0.0)


if __name__ == '__main__':
    todo = sorted(set(sum(SIZES.values(), []) + TABLE), key=lambda s: min(s))
    cache = os.path.join(DATA, 'fig_constants_rows.jsonl')          # one line per finished size (survives interruptions)
    rows = {}
    if os.path.exists(cache):
        for l in open(cache):
            r = json.loads(l); rows[(r['K'], r['d'])] = r
    with Pool(int(os.environ.get('NPROC', 12))) as p:
        for r in p.imap_unordered(row, [s for s in todo if s not in rows], chunksize=1):
            rows[(r['K'], r['d'])] = r
            with open(cache, 'a') as f:
                f.write(json.dumps(r) + '\n')
    print('all sizes done; run figA_plot.py to assemble fig_constants.json / table_constants.json and draw the figure')
