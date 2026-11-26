"""Scalar sign descent with momentum: referee check of Theorem 'Convergence with atomic innovations'.

F(w)=w^2/2 (so G=W), nu=|.|, O_t=sign(M_t) (sign(0):=0, a valid subgradient).
case A: xi in {-1,2} w.p. 2/3,1/3, beta=1/2  -> N_beta = -1+3U, U has i.i.d. Bernoulli(1/3) binary digits
        (singular continuous law, full support, strictly increasing cdf => unique median, computed exactly).
case C: xi = +-1, beta=1/3 -> N_beta lives on a Cantor set, Med = [-1/3,1/3] (a whole interval).
usage: scalar_sign.py CASE THETA T R SEED
"""
import sys, numpy as np

def median_caseA():
    # U = sum_{j>=0} 2^{-(j+1)} B_j, B_j ~ Bern(1/3).  Solve F_U(u)=1/2 digit by digit (exact rational recursion).
    from fractions import Fraction as Fr
    p, u, w = Fr(1, 2), Fr(0), Fr(1, 2)
    for _ in range(80):
        if p < Fr(2, 3):
            p = p / Fr(2, 3)
        else:
            u += w; p = (p - Fr(2, 3)) / Fr(1, 3)
        w /= 2
    return float(-1 + 3 * u)       # median of N_beta

def run(case, theta, T, R, seed, eta0=1.0, t0=1.0):
    rng = np.random.default_rng(seed)
    if case == 'A':
        beta = 0.5; draw = lambda shape: 3.0 * (rng.random(shape) < 1/3) - 1.0
        W = np.where(np.arange(R) % 2 == 0, 2.0, -2.0)
    else:
        beta = 1/3; draw = lambda shape: 2.0 * (rng.random(shape) < 0.5) - 1.0
        W = np.repeat(np.array([0.0, 2.0, -2.0, 0.2]), R // 4)
    W0 = W.copy()
    # stationary N_0 from 200 negative-time innovations (two-sided extension), M_0 := 0 deterministic
    N = np.zeros(R)
    for _ in range(200):
        N = beta * N + (1 - beta) * draw(R)
    M = np.zeros(R)
    e = M - W - N                      # e_0
    sum_eta_e = np.zeros(R); min_path_slack = np.inf; max_rec_err = 0.0
    a = -median_caseA() if case == 'A' else 0.0   # a in -Med for (eq:path) check
    cps = sorted(set([T // 64, T // 16, T // 4, T // 2, T])); out = {}
    chunk = 4096; t = 0
    while t < T:
        m = min(chunk, T - t)
        X = draw((m, R))
        etas = eta0 * (np.arange(t + 1, t + m + 1) + t0) ** (-theta)
        for i in range(m):
            Gprev = W
            # step t+i+1: G_t=W_t, M_t = beta M_{t-1} + (1-beta)(G_t+xi_t)
            N = beta * N + (1 - beta) * X[i]
            M = beta * M + (1 - beta) * (W + X[i])
            enew = M - W - N
            if t + i < 20000:   # diagnostics on the early part (cheap)
                # recursion e_t = beta e_{t-1} + beta (G_{t-1}-G_t) checked below via stored Glast
                if t + i > 0:
                    max_rec_err = max(max_rec_err, np.abs(enew - (beta * e + beta * (Glast - W))).max())
                O = np.sign(M)
                slack = O * (W - a) - ((np.abs(W + N) - np.abs(a + N)) - 2 * np.abs(enew))
                min_path_slack = min(min_path_slack, slack.min())
            else:
                O = np.sign(M)
            e = enew; Glast = W
            sum_eta_e += etas[i] * np.abs(e)
            W = W - etas[i] * O
            if t + i + 1 in cps:
                out[t + i + 1] = W.copy()
        t += m
    return dict(W0=W0, cps=out, sum_eta_e=sum_eta_e, min_path_slack=min_path_slack, max_rec_err=max_rec_err)

if __name__ == '__main__':
    case, theta, T, R, seed = sys.argv[1], float(sys.argv[2]), int(float(sys.argv[3])), int(sys.argv[4]), int(sys.argv[5])
    res = run(case, theta, T, R, seed)
    print(f"case {case} theta={theta} T={T} R={R} seed={seed}")
    print(f"  recursion e_t check (first 2e4 steps): max abs err {res['max_rec_err']:.2e}")
    print(f"  (eq:path) slack min over runs/steps (must be >= 0 up to rounding): {res['min_path_slack']:.3e}")
    print(f"  sum_t eta_t|e_t| up to T: mean {res['sum_eta_e'].mean():.4f} max {res['sum_eta_e'].max():.4f}")
    if case == 'A':
        med = median_caseA()
        rng = np.random.default_rng(1); n = 2_000_000; Ns = np.zeros(n)
        for _ in range(60): Ns = 0.5 * Ns + 0.5 * (3.0 * (rng.random(n) < 1/3) - 1.0)
        print(f"  exact median of N_beta = {med:.12f}; MC median (2e6 samples) = {np.median(Ns):.5f}; target G_inf = {-med:.12f}")
        for T_, W in res['cps'].items():
            d = W + med
            print(f"  t={T_:>9d}: mean(G_t)-target = {d.mean():+.3e} +- {d.std(ddof=1)/np.sqrt(len(d)):.1e} (SE); sd across runs {d.std(ddof=1):.3e}; max|.| {np.abs(d).max():.3e}")
        ks = sorted(res['cps']); print(f"  max |G_T - G_T/2| over runs: {np.abs(res['cps'][ks[-1]]-res['cps'][ks[-2]]).max():.3e}")
    else:
        ks = sorted(res['cps']); WT = res['cps'][ks[-1]]; Wh = res['cps'][ks[-2]]
        for w0 in np.unique(res['W0']):
            s = res['W0'] == w0; x = WT[s]
            print(f"  W0={w0:+.1f}: G_T min {x.min():+.4f} max {x.max():+.4f} mean {x.mean():+.4f} sd {x.std(ddof=1):.4f}; "
                  f"frac in [-1/3,1/3]+-1e-2: {np.mean(np.abs(x) <= 1/3 + 1e-2):.3f}; max excess over 1/3: {max(0, np.abs(x).max()-1/3):.2e}; "
                  f"max|G_T-G_T/2| {np.abs(x - Wh[s]).max():.2e}; quantiles(10,50,90) {np.round(np.quantile(x,[.1,.5,.9]),4)}")
