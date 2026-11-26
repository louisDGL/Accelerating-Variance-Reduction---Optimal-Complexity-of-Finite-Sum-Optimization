"""Momentum.  M_t = beta M_{t-1} + (1-beta)(grad F(W_t) + xi_t),  W_{t+1} = W_t - eta_t polar(M_t).
(i)  Frozen iterate: zero of the mean field of the stationary filtered noise N = sum_k w_k xi_k versus the first-order law
     G = g* (rho3/rho2)/b,  rho_r = sum_k w_k^r  (EMA: rho3/rho2 = (1-beta^2)/(1+beta+beta^2); Nesterov weights too),
     and versus the variance-based effective batch (1-beta)/(1+beta).
(ii) Coupled dynamics with Robbins-Monro lengths on a quadratic: the tail average of grad F(W_t) against the zero of (i).
Noise: i.i.d. Exp(1)-1 entries (kappa3 = 2), 2 x 3, batch b (exact Gamma batch means).
"""
import numpy as np, functools, sys, json
from lib import *
print = functools.partial(print, flush=True)
rng = np.random.default_rng(17)
K, d = 2, 3
q, m1 = q_const(K, d, 4_000_000); a1, a2 = a_consts(K, d, q)
gstar = (a1 + a2) * 2.0 / 6 * np.ones((K, d))
print(f"g* per entry = {gstar[0,0]:.5f}")

def weights(beta, nesterov, tol=1e-9):
    k = np.arange(int(np.ceil(np.log(tol) / np.log(beta))) + 1)
    w = (1 - beta) * beta ** k
    if nesterov:
        w = np.concatenate([[1 - beta ** 2], (1 - beta) * beta ** (k[1:] + 1)])
    return w

def filtered(w, b, S):
    N = np.zeros((S, K, d))
    for wk in w:
        N += wk * (rng.gamma(b, 1.0, (S, K, d)) / b - 1.0)
    return N

out = []
if 'frozen' in sys.argv or len(sys.argv) == 1:
    print("(i) frozen iterate:  beta  nesterov  b   b*G(mean entry)   first-order g*rho3/rho2   variance-based g*rho2    ratio measured/first-order")
    for beta, nes, b, S in [(0.5, False, 1, 3_000_000), (0.5, False, 8, 3_000_000), (0.9, False, 1, 2_000_000), (0.9, False, 4, 2_000_000),
                            (0.95, False, 1, 2_000_000), (0.9, True, 1, 2_000_000), (0.95, True, 1, 2_000_000)]:
        w = weights(beta, nes); r2, r3 = (w ** 2).sum(), (w ** 3).sum()
        N = filtered(w, b, S)
        pred = gstar * (r3 / r2) / b
        G, r = field_zero(N, pred)
        se = field_zero_se(N, G) / (m1 / (K * d) / np.sqrt(r2 / b)) / np.sqrt(K * d)     # per-entry se of G (isotropic J)
        print(f"   {beta:4.2f}  {str(nes):5s}  {b:2d}   {b*G.mean():.5f} (+-{b*se/np.sqrt(K*d):.5f})   {b*pred[0,0]:.5f}   {gstar[0,0]*r2:.5f}"
              f"   {G.mean()/pred[0,0]:.3f}   [b_eff = b*rho2/rho3 = {b*r2/r3:.2f}]")
        out.append(dict(beta=beta, nesterov=nes, b=b, bG=float(b * G.mean()), pred=float(b * pred[0, 0]), var_based=float(gstar[0, 0] * r2)))
    json.dump(out, open('../results/c04_frozen.json', 'w'), indent=1)

if 'coupled' in sys.argv or len(sys.argv) == 1:
    print("(ii) coupled dynamics, F(W) = 0.5 <W, W H>, H = diag(1,2,0.5); R parallel runs; tail average of grad F(W_t)")
    H = np.array([1.0, 2.0, 0.5])
    for beta, b, eta0 in [(0.9, 1, 0.05), (0.9, 1, 0.025), (0.5, 1, 0.05)]:
        w = weights(beta, False); N = filtered(w, b, 2_000_000); Gz, _ = field_zero(N, gstar * ((w**3).sum() / (w**2).sum()) / b)
        R, T, tau = 2000, 40000, 500.0
        W = np.zeros((R, K, d)); M = np.zeros((R, K, d)); acc = np.zeros((K, d)); cnt = 0
        for t in range(T):
            G = W * H
            xi = rng.gamma(b, 1.0, (R, K, d)) / b - 1.0
            M = beta * M + (1 - beta) * (G + xi)
            W -= eta0 / (1 + t / tau) ** 0.75 * polar(M)
            if t >= T // 2:
                acc += G.mean(0); cnt += 1
        Gbar = acc / cnt
        print(f"   beta={beta} b={b} eta0={eta0}: tail-averaged gradient = {np.round(Gbar.ravel(),5)}\n"
              f"        zero of the frozen stationary mean field = {np.round(Gz.ravel(),5)};  relative distance {np.linalg.norm(Gbar-Gz)/np.linalg.norm(Gz):.3f};"
              f"  final eta = {eta0/(1+T/tau)**0.75:.2e}")
