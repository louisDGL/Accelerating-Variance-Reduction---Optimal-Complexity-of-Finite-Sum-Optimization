"""Exact medians for lattice noise.  Entries of zeta i.i.d. X in {-1,2}, P=(2/3,1/3): mean 0, var 2, kappa3 2.
Sum of b copies: entry = -b + 3*Bin(b,1/3), on the lattice -b+3Z.  We compute, with the exact binomial law
(truncated at +-T standard deviations), the minimizer m_b of  m -> E nu(m 11^T + S_b)  (by symmetry the minimizer
is a multiple of the all-ones matrix when unique), i.e. b times the limit gradient, and compare with the
continuous-law prediction g* (Theorem 4/6 of polar_bias).
usage: python e01_exact_lattice.py {sign|1x2|1x3|2x2|2x3}"""
import sys, numpy as np
from scipy.stats import binom
from scipy.optimize import brentq

P = 1/3
def axis(b, T):
    mu, sd = b*P, np.sqrt(b*P*(1-P))
    lo, hi = max(0, int(np.floor(mu-T*sd))), min(b, int(np.ceil(mu+T*sd)))
    k = np.arange(lo, hi+1)
    return (-b+3.0*k), binom.pmf(k, b, P)

def field_sum(shape, b, m, T):
    """<E o(m 11^T + S_b), 11^T> for o = polar (K<=d) / normalisation (K=1)."""
    K, d = shape
    v, w = axis(b, T)
    n = len(v)
    N = K*d
    # iterate over the first coordinate in chunks, full grid on the others
    tot = 0.0
    if N == 1:
        return float(np.sum(w*np.sign(v+m)))
    grids = np.meshgrid(*([v]*(N-1)), indexing='ij')
    wg = w
    for _ in range(N-2):
        wg = np.multiply.outer(wg, w)
    rest = [g.ravel()+m for g in grids]
    wr = wg.ravel()
    for i0 in range(n):
        x0 = v[i0]+m
        ent = [np.full_like(rest[0], x0)] + rest
        if K == 1:
            nrm = np.sqrt(sum(e*e for e in ent))
            val = sum(ent)/np.where(nrm > 0, nrm, 1.0)
        else:
            M = [[ent[i*d+j] for j in range(d)] for i in range(K)]   # K=2
            a = sum(M[0][j]**2 for j in range(d)); c = sum(M[1][j]**2 for j in range(d))
            bb = sum(M[0][j]*M[1][j] for j in range(d))
            det = np.maximum(a*c-bb*bb, 0.0); sd_ = np.sqrt(det)
            tr = np.sqrt(a+c+2*sd_)                      # s1+s2
            r0 = sum(M[0]); r1 = sum(M[1])               # M 1
            # A^{-1/2} = (A + sqrt(det) I)^{-1} * (s1+s2) ;  (A+sI)^{-1} = [[c+s,-bb],[-bb,a+s]]/((a+s)(c+s)-bb^2)
            den = (a+sd_)*(c+sd_)-bb*bb
            if d == 2:
                dm = M[0][0]*M[1][1]-M[0][1]*M[1][0]          # determinant itself: no cancellation in the Gram matrix
                det = dm*dm; sd_ = np.abs(dm); tr = np.sqrt(a+c+2*sd_); den = (a+sd_)*(c+sd_)-bb*bb
                good = sd_ > 1e-10*(a+c+1e-300)
            else:
                good = det > 1e-13*(a+c+1e-300)**2
            den = np.where(good, den, 1.0)
            y0 = ((c+sd_)*r0-bb*r1)/den*tr; y1 = (-bb*r0+(a+sd_)*r1)/den*tr
            # rank-deficient atoms: partial isometry u v^T of a rank-one matrix, <u v^T, 11^T> = sum(M)/|M|_F
            fro = np.sqrt(a+c)
            val = np.where(good, y0+y1, np.where(fro > 0, (r0+r1)/np.where(fro > 0, fro, 1.0), 0.0))
        tot += w[i0]*np.dot(wr, val)
    return float(tot)

def run(name):
    cfg = {'sign': ((1,1), [3,4,5,30,31,32,300,301,302,3000,3001,3002], 12),
           '1x2': ((1,2), [4,8,16,17,18,32,64,65,66,128,256,257,258,512,1024,1025,1026,2048,4096], 7.5),
           '1x3': ((1,3), [4,8,16,17,18,32,64,65,66,128,129,130,256,400], 6.5),
           '2x2': ((2,2), [4,8,16,17,18,32,33,34,64,65,66,128,129,130,200,256], 5.5),
           '2x3': ((2,3), [4,8,9,10,16,17,18,25,32], 4.6)}[name]
    (K,d), bs, T = cfg
    if len(sys.argv) > 2: bs = [int(x) for x in sys.argv[2].split(',')]
    if len(sys.argv) > 3: T = float(sys.argv[3])
    N = K*d
    if name == 'sign': gstar = 2/(6*2)
    elif K == 1: gstar = 2/(2*2*(N+2))
    else:
        q = {(2,2): (3*np.sqrt(2)-2)/7, (2,3): 0.39347028878}[(K,d)]
        gstar = 3*(3+2*q)/((K+2)*(d+2))*2/(6*2)
    print(f"{name}: continuous-law prediction for b*G_b (each entry): g* = {gstar:.6f}", flush=True)
    for b in bs:
        if name == 'sign':
            v, w = axis(b, T); cdf = np.cumsum(w)
            # medians of S_b: lattice points l with P(S<l)<=1/2<=P(S<=l); b*G = -median
            i = np.searchsorted(cdf, 0.5); med = v[i]
            print(f"b={b:5d}  b mod 3={b%3}  median of S_b={med:+.0f}  b*G_b={-med:+.0f}   nearest pt of (-b+3Z) to -1/6: {(-b%3) if (-b%3)<1.5-1/6 else (-b%3)-3:+d}", flush=True)
            continue
        f = lambda m: field_sum((K,d), b, m, T)
        m = brentq(f, -1.5, 1.5, xtol=1e-10)
        # gradient-side: the limit gradient is the argmin of psi(G)=E nu(G+xi): G = m/b, so b*G = m
        print(f"b={b:5d} mod3={b%3}  b*G_b={m:+.6f}  err={m-gstar:+.6f}  sqrt(b)*err={np.sqrt(b)*(m-gstar):+.4f}  b*err={b*(m-gstar):+.3f}", flush=True)

if __name__ == '__main__':
    run(sys.argv[1])
