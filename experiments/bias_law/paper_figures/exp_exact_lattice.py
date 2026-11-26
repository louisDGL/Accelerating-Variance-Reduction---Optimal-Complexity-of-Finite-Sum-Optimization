"""Exact minimisers for lattice noise (extends experiments/bias_law/lattice/e01_exact_lattice.py; that file is not modified).

Noise: K x d matrix with i.i.d. entries X in {-1, 2}, P = (2/3, 1/3) (mean 0, variance 2, third cumulant 2).
S_b = sum of b copies: each entry is -b + 3 Bin(b, 1/3).  By symmetry the minimiser of G -> E nu(G + S_b / b) is
(m_b / b) 11^T; we compute m_b = b G_b (one entry) as the zero / sign change of the monotone field
    f(m) = < E o(m 11^T + S_b), 11^T >,       o = sign | x/|x| (1 x N) | polar factor (2 x 2, 2 x 3).
The binomial law is used exactly; the only approximation is a truncation of the support:
  1 x N : box, each one-dimensional tail < 1e-13 (N = 2) or 1e-11 (N = 3);
  2 x d : all lattice points whose (product) weight exceeds a threshold chosen such that the dropped mass is
          about 1e-8 (the dropped mass is computed and stored); column-permutation symmetry is used.
If b = 0 mod 3 the field may jump across 0 at m = 0 (an atom of rank-deficient matrices): this is tested first by
evaluating f(-1e-6), f(+1e-6); in that case the minimiser is exactly 0 ("locked").

usage: exp_exact_lattice.py [nproc]      -> results/bias_law/figure_data/exact_lattice.jsonl (one JSON record per line)
"""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import numpy as np
from scipy.stats import binom, chi2
from scipy.optimize import brentq
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "..", "results", "bias_law", "figure_data", "exact_lattice.jsonl")
P = 1.0 / 3.0
Q22 = (3 * np.sqrt(2) - 2) / 7
Q23 = 0.39347028878
GSTAR = {"sign": 1 / 6, "1x2": 0.125, "1x3": 0.1,
         "2x2": 3 * (3 + 2 * Q22) / 16 / 6, "2x3": 3 * (3 + 2 * Q23) / 20 / 6}
DELTA = 1e-6


def axis(b, eps):
    lo = int(max(0, binom.ppf(eps, b, P) - 1)); hi = int(min(b, binom.isf(eps, b, P) + 1))
    k = np.arange(lo, hi + 1)
    return -b + 3.0 * k, binom.pmf(k, b, P)


# ---------------------------------------------------------------- sign
def sign_median(b):
    k = np.arange(0, b + 1); v = -b + 3 * k; cdf = binom.cdf(k, b, P)
    i = int(np.searchsorted(cdf, 0.5))
    assert abs(cdf[i] - 0.5) > 1e-12 and (i == 0 or cdf[i - 1] < 0.5)       # unique median
    return float(-v[i])                                                     # b G_b = -median(S_b)


# ---------------------------------------------------------------- 1 x N
class Vec:
    def __init__(self, b, N):
        self.N = N; self.v, self.w = axis(b, 1e-13 if N == 2 else 1e-11)
        self.dropped = 1 - self.w.sum() ** N; self.npts = len(self.v) ** N

    def f(self, m):
        v, w = self.v + m, self.w
        if self.N == 2:
            X, Y = v[:, None], v[None, :]
            r = np.sqrt(X * X + Y * Y); r[r == 0] = np.inf
            return float(w @ ((X + Y) / r) @ w)
        tot = 0.0
        Y, Z = v[:, None], v[None, :]
        yz2 = Y * Y + Z * Z; yz = Y + Z
        for i in range(len(v)):
            r = np.sqrt(v[i] ** 2 + yz2); r[r == 0] = np.inf
            tot += w[i] * float(w @ ((v[i] + yz) / r) @ w)
        return tot


# ---------------------------------------------------------------- 2 x d (d = 2, 3)
class Mat:
    """Columns (x, y) sorted by decreasing weight; enumerate multisets c1 <= c2 (<= c3) with multiplicities."""

    def __init__(self, b, d, mass=1e-8):
        self.d = d
        v, w = axis(b, 1e-13)
        X, Y = np.meshgrid(v, v, indexing="ij"); W = np.multiply.outer(w, w)
        o = np.argsort(-W.ravel(), kind="stable")
        self.x, self.y, self.w = X.ravel()[o], Y.ravel()[o], W.ravel()[o]
        R2 = chi2.isf(mass, 2 * d)
        self.thr = self.w[0] ** d * np.exp(-R2 / 2)
        nc = int(np.searchsorted(-self.w, -self.thr / self.w[0] ** (d - 1), side="right"))
        self.x, self.y, self.w = self.x[:nc], self.y[:nc], self.w[:nc]
        self.negw = -self.w
        self.plan = []            # per c1: index arrays and multiplicity-weights (built lazily, cached if small)
        kept = 0.0; npts = 0
        for c1, i2, i3, wt in self._iter():
            kept += wt.sum(); npts += len(wt)
        self.dropped = 1 - kept; self.npts = npts

    def _hi(self, t):             # number of columns with weight >= t
        return np.searchsorted(self.negw, -t, side="right")

    def _iter(self):
        w = self.w; nc = len(w)
        for c1 in range(nc):
            t1 = self.thr / w[c1]
            if self.d == 2:
                hi = int(self._hi(t1))
                if hi <= c1: break
                i2 = np.arange(c1, hi)
                wt = 2.0 * w[c1] * w[i2]; wt[0] = w[c1] * w[c1]
                yield c1, i2, None, wt
            else:
                hi2 = int(self._hi(t1 / w[c1]))      # c2 >= c1 needs w[c2] w[c3] >= t1 with w[c3] <= w[c2] <= w[c1]
                if hi2 <= c1: break
                c2 = np.arange(c1, hi2)
                L = self._hi(t1 / w[c2]) - c2; ok = L > 0
                c2, L = c2[ok], L[ok]
                if len(c2) == 0: break
                i2 = np.repeat(c2, L)
                start = np.cumsum(L) - L
                i3 = np.arange(L.sum()) - np.repeat(start, L) + i2
                mult = np.full(len(i2), 6.0)
                mult[(i2 == c1) | (i3 == i2)] = 3.0
                mult[(i2 == c1) & (i3 == i2)] = 1.0
                yield c1, i2, i3, mult * w[c1] * w[i2] * w[i3]

    def f(self, m):
        x, y = self.x + m, self.y + m
        t = y - x; sq = x * x + y * y; sm = x + y
        tot = 0.0
        for c1, i2, i3, wt in self._iter():
            x1, y1, t1 = x[c1], y[c1], t[c1]
            x2, y2, t2 = x[i2], y[i2], t[i2]
            m12 = x1 * y2 - x2 * y1
            if self.d == 2:
                s = np.abs(m12); fro2 = sq[c1] + sq[i2]; r = sm[c1] + sm[i2]
                br = m12 * (t2 - t1)
            else:
                x3, y3, t3 = x[i3], y[i3], t[i3]
                m13 = x1 * y3 - x3 * y1; m23 = x2 * y3 - x3 * y2
                s = np.sqrt(m12 * m12 + m13 * m13 + m23 * m23)
                fro2 = sq[c1] + sq[i2] + sq[i3]; r = sm[c1] + sm[i2] + sm[i3]
                br = m12 * (t2 - t1) + m13 * (t3 - t1) + m23 * (t3 - t2)
            # <polar(M), 11^T> = (sum(M) + bracket / s) / (s1 + s2),  s = s1 s2,  (s1+s2)^2 = |M|_F^2 + 2 s
            good = s > 1e-11 * (fro2 + 1e-300)
            den = np.sqrt(fro2 + 2 * s); den[den == 0] = np.inf
            val = (r + np.where(good, br / np.where(good, s, 1.0), 0.0)) / den
            tot += float(wt @ val)
        return tot


def solve(task):
    name, b = task
    t0 = time.time(); g = GSTAR[name]
    if name == "sign":
        m = sign_median(b); rec = dict(geom=name, b=b, m=m, locked=False, dropped=0.0, npts=b + 1, nev=0)
    else:
        obj = Vec(b, int(name[2])) if name[0] == "1" else Mat(b, int(name[2]))
        nev = [0]
        def f(m):
            nev[0] += 1; return obj.f(m)
        locked = False; m = None
        if b % 3 == 0:
            fm, fp = f(-DELTA), f(DELTA)
            if fm < 0 < fp: m, locked = 0.0, True
            lo, hi = (DELTA, 0.6) if fp <= 0 else (-0.6, -DELTA)
        else:
            lo, hi = -0.6, 0.8
        if m is None:
            m = brentq(f, lo, hi, xtol=1e-9 if obj.npts < 3e7 else 2e-8)
        rec = dict(geom=name, b=b, m=float(m), locked=locked, dropped=float(obj.dropped), npts=int(obj.npts), nev=nev[0])
    rec.update(mod3=b % 3, gstar=g, err=rec["m"] - g, sqrtb_err=np.sqrt(b) * (rec["m"] - g), b_err=b * (rec["m"] - g),
               seconds=time.time() - t0)
    return rec


def geo_classes(lo, hi, n):
    out = set()
    for x in np.geomspace(lo, hi, n):
        x = int(round(x))
        for r in range(3):
            c = x + ((r - x) % 3)
            out.add(c if c <= hi + 2 else c - 3)
    return sorted(out)


def tasks():
    T = [("sign", b) for b in list(range(1, 61)) + [300, 301, 302, 3000, 3001, 3002]]
    T += [("1x2", b) for b in sorted(set(geo_classes(4, 4096, 11) + [16, 17, 18, 64, 65, 66, 256, 257, 258, 1024, 1025, 1026]))]
    T += [("1x3", b) for b in sorted(set(geo_classes(4, 1000, 9) + [16, 17, 18, 64, 65, 66, 128, 129, 130, 256, 400]))]
    T += [("2x3", b) for b in list(range(3, 37)) + [40, 41, 42, 46, 47, 48, 52, 53, 54, 58, 59, 60, 64, 65, 66, 79, 80, 81, 100, 101, 102]]
    T += [("2x2", b) for b in list(range(3, 67)) + [96, 97, 98, 129, 130, 131, 192, 193, 194, 255, 256, 257, 384, 385, 386,
                                                    513, 514, 515, 768, 769, 770, 1026, 1027, 1028, 1536, 1537, 1538, 2049]]
    return T


def cost(t):
    name, b = t
    return {"sign": 0, "1x2": 1e-6 * b, "1x3": 1e-6 * b ** 1.5, "2x2": 0.25 * b ** 2, "2x3": b ** 3}[name]


if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    if len(sys.argv) > 2:      # quick check: exp_exact_lattice.py 1 2x3:8,2x3:9
        for s in sys.argv[2].split(","):
            n_, b_ = s.split(":"); print(solve((n_, int(b_))))
        sys.exit()
    done = set()
    if os.path.exists(OUT):                      # resume: records already present are kept
        done = {(r["geom"], r["b"]) for r in map(json.loads, open(OUT))}
    T = sorted([t for t in tasks() if t not in done], key=cost, reverse=True)
    t0 = time.time()
    with Pool(nproc) as p, open(OUT, "a") as fh:
        for rec in p.imap_unordered(solve, T, chunksize=1):
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            print(f"[{time.time()-t0:6.0f}s] {rec['geom']:4s} b={rec['b']:5d} m={rec['m']:+.6f} err={rec['err']:+.6f} "
                  f"locked={rec['locked']} dropped={rec['dropped']:.1e} npts={rec['npts']:.2e} nev={rec['nev']} ({rec['seconds']:.0f}s)", flush=True)
