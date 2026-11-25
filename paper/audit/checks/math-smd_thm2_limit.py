"""Audit math-smd -- Theorem 2(b)-(c) and lines 389-393, 418-420 of paper_saga_to_muon.tex.

Simulations of W_{t+1} = W_t - eta_t polar(grad F(W_t) + xi_t), eta_t = eta_0 (1 + t/t_0)^(-3/4)
(sum eta = infinity, sum eta^2 < infinity), many independent chains in parallel.

 A. scalar, centred skewed noise xi = Exp(1) - 1: predicted limit gradient -median(xi) = 1 - log 2, for a quadratic
    and for a non-quadratic F (the limit GRADIENT must not depend on F).
 B. scalar two-point laws: the law of the paper, P(xi=1)=0.4, P(xi=-1)=0.6 (psi minimal at g=1, |Phi| >= 0.2), and a
    CENTRED two-point law P(xi=2)=1/3, P(xi=-1)=2/3 (E xi = 0, Med = {-1}, limit gradient 1).
 C. 2 x 2, entries i.i.d. Exp(1) - 1 (E xi = 0, full rank a.s.): G^dag = argmin of psi_xi computed from 1e6 samples;
    check that it is a zero of the mean field on an independent sample, that the runs converge to it for a quadratic and
    a non-quadratic F, and the bounds of Theorem 2(c) and Proposition 3(b).
 D. xi = +-I_2 (symmetric, 0 not in the support): 0 in Med(xi) but Med(xi) != {0}; the limit is a random point.
 E. 2 x 2 Cauchy entries (no moment, symmetric, 0 in the support): W_t -> W_* (Proposition 3(a), "no moment").

Run from the project root, one block per call (scalar | 2x2 | pmI):
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -u paper/audit/checks/math-smd_thm2_limit.py scalar
"""
import numpy as np
from scipy.optimize import minimize, brentq

rng = np.random.default_rng(7)


def eta_sched(T, eta0, t0):
    return eta0 * (1.0 + np.arange(T) / t0) ** (-0.75)


# ---------------------------------------------------------------- A, B: scalar
def run_scalar(grad, sampler, T, N, eta0=0.5, t0=50.0, w0=0.0):
    eta = eta_sched(T, eta0, t0)
    w = np.full(N, w0)
    acc = np.zeros(N)
    for t in range(T):
        g = grad(w)
        w = w - eta[t] * np.sign(g + sampler(N))          # np.sign(0) = 0 = polar(0)
        if t >= T // 2:
            acc += grad(w)
    return grad(w), acc / (T - T // 2)


def scalar_block():
    print("=== A. scalar, xi = Exp(1) - 1 (E xi = 0, median = log 2 - 1) ===")
    target = 1.0 - np.log(2.0)
    T, N = 30000, 4000
    samp = lambda n: rng.exponential(1.0, n) - 1.0
    for name, grad in [("F = w^2/2", lambda w: w),
                       ("F = w^2/2 + 0.8 log cosh(w - 0.5)", lambda w: w + 0.8 * np.tanh(w - 0.5)),
                       ("F = 2 w^2 + sqrt(1 + (w+1)^2)", lambda w: 4 * w + (w + 1) / np.sqrt(1 + (w + 1) ** 2))]:
        gT, gbar = run_scalar(grad, samp, T, N)
        print(f"  {name:36s}: mean G_T = {gT.mean():+.4f} (std over chains {gT.std():.4f}), "
              f"tail average = {gbar.mean():+.4f} +- {gbar.std() / np.sqrt(N):.4f};  predicted {target:+.4f};  "
              f"E xi = 0 would give 0")
    print("=== B. scalar two-point laws ===")
    # law of the paper
    Phi = lambda g: 0.4 * np.sign(g + 1) + 0.6 * np.sign(g - 1)
    psi = lambda g: 0.4 * (abs(g + 1) - 1) + 0.6 * (abs(g - 1) - 1)
    grid = np.concatenate([np.linspace(-3, 3, 600001), [-1.0, 1.0]])
    print(f"  P(xi=1)=0.4, P(xi=-1)=0.6: argmin psi on a grid = {grid[np.argmin([psi(g) for g in grid])]:.4f}, "
          f"min |Phi| on the grid (incl. g = -1, 1) = {np.abs(Phi(grid)).min():.3f}, Phi(1) = {Phi(1.0):.2f}, "
          f"Phi(1-) = {Phi(0.999):.2f}, Phi(-1) = {Phi(-1.0):.2f}")
    samp = lambda n: np.where(rng.random(n) < 0.4, 1.0, -1.0)
    gT, gbar = run_scalar(lambda w: w, samp, 30000, 2000)
    print(f"     run: mean G_T = {gT.mean():.4f} (std {gT.std():.4f}), tail average {gbar.mean():.4f}; predicted 1 "
          f"(no zero of the mean field)")
    samp = lambda n: np.where(rng.random(n) < 1.0 / 3.0, 2.0, -1.0)
    gT, gbar = run_scalar(lambda w: w, samp, 30000, 2000)
    print(f"  centred law P(xi=2)=1/3, P(xi=-1)=2/3 (E xi = 0, E|xi| = 4/3): mean G_T = {gT.mean():.4f} "
          f"(std {gT.std():.4f}), tail average {gbar.mean():.4f}; predicted -median = 1 <= E|xi|")


# ---------------------------------------------------------------- C: 2 x 2
def nuc2(M):
    det = M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]
    return np.sqrt((M ** 2).sum(axis=(-1, -2)) + 2.0 * np.abs(det))


def polar2(M):
    """Closed form for 2 x 2 matrices of full rank: (M + sign(det) cof(M)) / ||M||_*."""
    a, b, c, d = M[..., 0, 0], M[..., 0, 1], M[..., 1, 0], M[..., 1, 1]
    det = a * d - b * c
    s = np.sign(det)
    out = np.empty_like(M)
    out[..., 0, 0] = a + s * d
    out[..., 0, 1] = b - s * c
    out[..., 1, 0] = c - s * b
    out[..., 1, 1] = d + s * a
    nrm = np.sqrt(a * a + b * b + c * c + d * d + 2.0 * np.abs(det))
    return out / nrm[..., None, None]


def block_2x2():
    print("=== C. 2 x 2, entries i.i.d. Exp(1) - 1 ===")
    # self-test of the closed forms against the SVD
    M = rng.standard_normal((1000, 2, 2))
    U, s, Vt = np.linalg.svd(M)
    assert np.abs(polar2(M) - U @ Vt).max() < 1e-9 and np.abs(nuc2(M) - s.sum(axis=1)).max() < 1e-10
    n1 = 1_000_000
    xs = rng.exponential(1.0, (n1, 2, 2)) - 1.0
    base = nuc2(xs).mean()

    def fun(g):
        G = g.reshape(2, 2)
        return nuc2(G[None] + xs).mean() - base, polar2(G[None] + xs).mean(axis=0).ravel()

    r = minimize(fun, np.zeros(4), jac=True, method="BFGS", options={"gtol": 1e-9})
    Gd = r.x.reshape(2, 2)
    xs2 = rng.exponential(1.0, (n1, 2, 2)) - 1.0
    Phi_ind = polar2(Gd[None] + xs2).mean(axis=0)
    Phi0 = polar2(xs2).mean(axis=0)
    Enuc = nuc2(xs2).mean()
    print(f"  G^dag (argmin of psi on {n1} samples) =\n{Gd}")
    print(f"  ||G^dag||_* = {nuc2(Gd):.4f}, min psi = {r.fun:.5f} (psi(0) = 0);  E||xi||_* = {Enuc:.4f}")
    print(f"  mean field on an independent sample: ||Phi(G^dag)||_F = {np.linalg.norm(Phi_ind):.2e} "
          f"(MC noise ~ {1 / np.sqrt(n1):.1e});  ||Phi(0)||_F = {np.linalg.norm(Phi0):.4f}  (E xi = 0 but Phi(0) != 0)")
    del xs, xs2

    # two objectives
    A = np.array([[2.0, 0.6], [0.6, 0.5]])
    ev = np.linalg.eigvalsh(A)
    B = np.array([[0.15, -0.2], [0.25, 0.1]])
    Q = np.array([[np.cos(2.0), -np.sin(2.0)], [np.sin(2.0), np.cos(2.0)]])
    c = 0.7

    def grad_quad(W):
        return W @ A - B

    def grad_nq(W):
        t = (W * Q).sum(axis=(-1, -2))
        return W @ A - B + c * np.tanh(t)[..., None, None] * Q

    def f_quad(W):
        return 0.5 * ((W @ A) * W).sum(axis=(-1, -2)) - (B * W).sum(axis=(-1, -2))

    def f_nq(W):
        t = (W * Q).sum(axis=(-1, -2))
        return f_quad(W) + c * (np.logaddexp(t, -t) - np.log(2.0))

    T, N = 20000, 2000
    eta = eta_sched(T, 0.3, 100.0)
    for name, grad, f, Lop, muop in [("quadratic F", grad_quad, f_quad, ev.sum(), ev.min()),
                                     ("NON-quadratic F (log cosh)", grad_nq, f_nq, ev.sum() + 4 * c, ev.min())]:
        W = np.zeros((N, 2, 2))
        acc = np.zeros((N, 2, 2))
        std_mid = None
        for t in range(T):
            G = grad(W)
            W = W - eta[t] * polar2(G + rng.exponential(1.0, (N, 2, 2)) - 1.0)
            if t == T // 8:
                std_mid = np.linalg.norm(grad(W) - Gd, axis=(1, 2)).mean()
            if t >= T // 2:
                acc += grad(W)
        GT = grad(W)
        Gbar = (acc / (T - T // 2)).mean(axis=0)
        # W_* and W^dag by Newton-free fixed point: solve grad(W) = target with scipy
        from scipy.optimize import fsolve
        Wstar = fsolve(lambda w: grad(w.reshape(2, 2)).ravel(), np.zeros(4), xtol=1e-12).reshape(2, 2)
        Wdag = fsolve(lambda w: (grad(w.reshape(2, 2)) - Gd).ravel(), np.zeros(4), xtol=1e-12).reshape(2, 2)
        assert np.abs(grad(Wstar)).max() < 1e-10 and np.abs(grad(Wdag) - Gd).max() < 1e-10
        tdag = (Wdag * Q).sum()
        print(f"     <Q, W^dag> = {tdag:+.3f}, sech^2 = {1 / np.cosh(tdag) ** 2:.3f} (weight of the non-quadratic "
              f"curvature at the limit; irrelevant for the quadratic F)")
        gap_lim = f(W).mean() - f(Wstar)
        gap_dag = f(Wdag) - f(Wstar)
        print(f"  [{name}] L_op = {Lop:.3f}, mu_op = {muop:.3f}")
        print(f"     tail-averaged gradient (mean over {N} chains) =\n{Gbar}")
        print(f"     ||mean G - G^dag||_F = {np.linalg.norm(Gbar - Gd):.4f}  vs  ||G^dag||_F = {np.linalg.norm(Gd):.4f}"
              f"  (distance to 0, the limit under 'E xi = 0')")
        print(f"     mean over chains of ||G_t - G^dag||_F: t = T/8: {std_mid:.4f}, t = T: "
              f"{np.linalg.norm(GT - Gd, axis=(1, 2)).mean():.4f}   (shrinks with eta_t)")
        print(f"     F(W^dag) - F_* = {gap_dag:.5f}; mean F(W_T) - F_* = {gap_lim:.5f};  lower bound of Thm 2(c) "
              f"||G^dag||_*^2/(2 L_op) = {nuc2(Gd) ** 2 / (2 * Lop):.5f};  upper bounds of Prop 3(b): "
              f"||G^dag||_*^2/(2 mu_op) = {nuc2(Gd) ** 2 / (2 * muop):.5f} <= (E||xi||_*)^2/(2 mu_op) = "
              f"{Enuc ** 2 / (2 * muop):.4f}")


def block_pmI():
    print("=== D. xi = +-I_2 with probability 1/2, F = ||W||_F^2 / 2, W_0 = 0 ===")
    I = np.eye(2)
    psi = lambda G: 0.5 * (nuc2(G + I) + nuc2(G - I)) - 2.0
    for w in (0.0, 0.5, -0.9, 1.0, 1.2):
        print(f"  psi({w:+.1f} I) = {psi(w * I):+.4f}", end=";")
    S = np.array([[0.3, 0.5], [0.5, -0.4]])
    print(f"  psi(symmetric S, ||S||_op = {np.linalg.norm(S, 2):.2f}) = {psi(S):+.2e}")
    T, N = 10000, 2000
    eta = eta_sched(T, 0.05, 100.0)
    W = np.zeros((N, 2, 2))
    for t in range(T):
        sgn = np.where(rng.random(N) < 0.5, 1.0, -1.0)
        W = W - eta[t] * polar2(W + sgn[:, None, None] * I)
    w = W[:, 0, 0]
    off = np.abs(W - w[:, None, None] * I).max()
    print(f"  limits W_T = w I (max deviation from w I: {off:.1e}); w over {N} chains: mean {w.mean():+.4f}, "
          f"std {w.std():.4f} (sqrt(sum eta^2) = {np.sqrt((eta ** 2).sum()):.4f}), share with |w| < 1e-3: "
          f"{(np.abs(w) < 1e-3).mean():.4f}; all psi(W_T) = min: {np.abs(psi(W)).max():.1e}")
    print("  -> 0 is in Med(xi) but the limit is a random nuclear median, almost surely nonzero.")


def block_cauchy():
    print("=== E. 2 x 2, entries i.i.d. standard Cauchy (no moment; symmetric, 0 in the support: Med = {0}) ===")
    A = np.array([[2.0, 0.6], [0.6, 0.5]])
    B = np.array([[0.15, -0.2], [0.25, 0.1]])
    T, N = 20000, 2000
    eta = eta_sched(T, 0.3, 100.0)
    W = np.zeros((N, 2, 2))
    acc = np.zeros((N, 2, 2))
    for t in range(T):
        W = W - eta[t] * polar2(W @ A - B + rng.standard_cauchy((N, 2, 2)))
        if t >= T // 2:
            acc += W @ A - B
    Gbar = (acc / (T - T // 2)).mean(axis=0)
    GT = W @ A - B
    print(f"  ||grad F(W_0)||_F = {np.linalg.norm(B):.4f};  tail-averaged gradient, mean over chains: ||.||_F = "
          f"{np.linalg.norm(Gbar):.4f};  mean over chains of ||G_T||_F = {np.linalg.norm(GT, axis=(1, 2)).mean():.4f}"
          f"  (-> 0: W_t -> W_* without any moment)")


if __name__ == "__main__":
    import sys
    np.set_printoptions(precision=4, suppress=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"      # 'scalar', '2x2', 'pmI' or 'all' (one block per call
    if which in ("scalar", "all"):                            #  keeps each call under 3 minutes of CPU)
        scalar_block()
    if which in ("2x2", "all"):
        block_2x2()
    if which in ("pmI", "all"):
        block_pmI()
        block_cauchy()
