"""2x2 nuclear-norm (polar) descent with momentum and discrete noise vs. direct minimisation of g -> E||g+N_beta||_*.

xi_t = mean of b i.i.d. matrices with i.i.d. entries in {-1,2} (prob 2/3,1/3).  F(W)=1/2 sum_ij A_ij (W_ij-Wstar_ij)^2.
usage: matrix_nuclear.py sim  BETA B THETA T R SEED     |   matrix_nuclear.py ref BETA B NSAMP NBATCH
"""
import sys, numpy as np
from multiprocessing import Pool

A = np.array([1.0, 2.0, 0.5, 1.5]); WSTAR = np.array([0.7, -1.3, 0.4, 2.0])

def polar4(a, b, c, d):
    """polar factor of [[a,b],[c,d]] componentwise: (M + sgn(det) cof(M))/(s1+s2).  At det=0 this returns an
    orthogonal matrix uv^T +- u_perp v_perp^T, still an element of the nuclear subdifferential."""
    s = np.where(a * d - b * c >= 0, 1.0, -1.0)
    p, q, r, u = a + s * d, b - s * c, c - s * b, d + s * a
    nrm = np.sqrt(0.5 * (p * p + q * q + r * r + u * u))
    nrm = np.where(nrm == 0, 1.0, nrm)
    return p / nrm, q / nrm, r / nrm, u / nrm

def nuc(M):   # M shape (...,4): a,b,c,d
    return np.sqrt((M * M).sum(-1) + 2 * np.abs(M[..., 0] * M[..., 3] - M[..., 1] * M[..., 2]))

def sim(beta, b, theta, T, R, seed, eta0=1.0, t0=1.0):
    rng = np.random.default_rng(seed)
    W = np.zeros((4, R)); M = np.zeros((4, R)); N = np.zeros((4, R))
    for _ in range(400):
        N = beta * N + (1 - beta) * (3.0 * rng.binomial(b, 1/3, (4, R)) - b) / b
    cps = sorted(set([T // 64, T // 16, T // 4, T // 2, T])); out = {}
    Abar = A[:, None]; Ws = WSTAR[:, None]; chunk = 2048; t = 0; minslack = np.inf
    while t < T:
        m = min(chunk, T - t)
        X = (3.0 * rng.binomial(b, 1/3, (m, 4, R)) - b) / b
        etas = eta0 * (np.arange(t + 1, t + m + 1) + t0) ** (-theta)
        for i in range(m):
            G = Abar * (W - Ws)
            M = beta * M + (1 - beta) * (G + X[i])
            O = np.stack(polar4(M[0], M[1], M[2], M[3]))
            if t + i < 5000:      # (eq:path) with a := 0 replaced by generic a: inequality holds for ANY a
                N = beta * N + (1 - beta) * X[i]
                e = M - G - N; a = 0.01
                lhs = (O * (G - a)).sum(0)
                rhs = nuc((G + N).T) - nuc((a + N).T) - 2 * np.sqrt(2) * np.sqrt((e * e).sum(0))
                minslack = min(minslack, (lhs - rhs).min())
            W = W - etas[i] * O
            if t + i + 1 in cps:
                out[t + i + 1] = (Abar * (W - Ws)).copy()
        t += m
    return out, minslack

def _ref_batch(args):
    beta, b, n, seed = args
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    J = int(np.ceil(np.log(1e-13) / np.log(beta)))
    N = np.zeros((n, 4))
    for j in range(J):   # N = (1-beta) sum beta^j xi_j
        N += (1 - beta) * beta ** j * (3.0 * rng.binomial(b, 1/3, (n, 4)) - b) / b
    def fg(g):
        Mx = N + g
        val = nuc(Mx).mean()
        O = polar4(Mx[:, 0], Mx[:, 1], Mx[:, 2], Mx[:, 3])
        return val, np.array([o.mean() for o in O])
    r = minimize(fg, np.zeros(4), jac=True, method='L-BFGS-B', options=dict(maxiter=500, ftol=1e-15, gtol=1e-9))
    return r.x, np.abs(r.jac).max()

if __name__ == '__main__':
    mode, beta, b = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    if mode == 'sim':
        theta, T, R, seed = float(sys.argv[4]), int(float(sys.argv[5])), int(sys.argv[6]), int(sys.argv[7])
        out, minslack = sim(beta, b, theta, T, R, seed)
        print(f"SIM beta={beta} b={b} theta={theta} T={T} R={R}; (eq:path) min slack (K=sqrt2) = {minslack:.3e}")
        for T_, G in out.items():
            print(f"  t={T_:>9d}: mean G_t = {np.round(G.mean(1),5)} SE {np.round(G.std(1,ddof=1)/np.sqrt(R),5)} sd-across-runs {np.round(G.std(1,ddof=1),5)}")
        np.save(f"sim_b{b}_beta{beta}_theta{theta}.npy", out[max(out)])
    else:
        n, nb = int(float(sys.argv[4])), int(sys.argv[5])
        with Pool(min(nb, 8)) as p:
            res = p.map(_ref_batch, [(beta, b, n, 1000 + k) for k in range(nb)])
        X = np.array([r[0] for r in res])
        print(f"REF beta={beta} b={b}: argmin_g E||g+N_beta||_* from {nb} independent batches of {n} samples (4 free coordinates)")
        print(f"  mean = {np.round(X.mean(0),5)}  SE = {np.round(X.std(0,ddof=1)/np.sqrt(nb),5)}  (max |grad| at optimum {max(r[1] for r in res):.1e})")
        print(f"  pooled over the 4 entries: {X.mean():.5f} +- {X.mean(1).std(ddof=1)/np.sqrt(nb):.5f}")
        np.save(f"ref_b{b}_beta{beta}.npy", X)
