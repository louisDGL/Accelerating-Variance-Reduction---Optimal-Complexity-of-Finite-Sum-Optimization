"""High-precision computation of the minimizer eta_* of F."""
from __future__ import annotations

import numpy as np

from .problems import BlockQuadraticProblem, SharedQuadraticProblem, AlignedChainProblem

# Below this relative size of the Newton decrement the Armijo test on F is no longer meaningful: the predicted decrease
# dec/2 is within a few orders of magnitude of the rounding error of F (about 1e-16 |F|, more for long sums).
DEC_ROUNDING = 1e-12


def newton_solve(prob, theta0=None, tol=1e-15, max_iter=200, verbose=False):
    """Damped Newton method, run until ||F'(theta)|| <= tol or stagnation.

    Step size: Armijo backtracking on F while the Newton decrement dec = F'(theta)^T F''(theta)^{-1} F'(theta)
    exceeds DEC_ROUNDING * max(1, |F(theta)|). Below that level the Armijo test would be decided by the rounding error
    of F (it stalled at ||F'|| = 1.3e-12 on the B3 problem kappa = 10, alpha = 1), so the step is instead accepted as
    soon as it decreases ||F'|| (starting from the full Newton step, which does in the quadratic-convergence region).
    The iteration stops when ||F'|| <= tol or when ||F'|| has not decreased for 3 iterations (rounding floor of the
    gradient). Returns (theta, info) for the iterate with the smallest ||F'||, with
    info = {'grad_norm', 'iters', 'err_bound'} where err_bound = ||F'(theta)|| / mu bounds ||theta - eta_*||
    (strong convexity).
    """
    d = prob.d
    theta = np.zeros(d) if theta0 is None else np.array(theta0, dtype=np.float64)
    g = prob.grad(theta)
    gn = float(np.linalg.norm(g))
    best = (gn, theta.copy())
    it = 0
    stagnation = 0
    while gn > tol and it < max_iter:
        H = prob.hessian(theta)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        dec = float(g @ step)
        F0 = prob.F(theta)
        t = 1.0
        if dec > DEC_ROUNDING * max(1.0, abs(F0)):
            # Armijo backtracking on F
            while prob.F(theta - t * step) > F0 - 1e-4 * t * dec and t > 1e-10:
                t *= 0.5
            theta = theta - t * step
            g = prob.grad(theta)
        else:
            # near the optimum: accept the first step (t = 1, 1/2, ...) that decreases ||F'||
            while True:
                theta_t = theta - t * step
                g_t = prob.grad(theta_t)
                if float(np.linalg.norm(g_t)) < gn or t <= 1e-10:
                    break
                t *= 0.5
            theta, g = theta_t, g_t
        gn = float(np.linalg.norm(g))
        it += 1
        if verbose:
            print(f"newton it={it} t={t:.3g} dec={dec:.3e} |g|={gn:.3e}")
        if gn < best[0]:
            best = (gn, theta.copy())
            stagnation = 0
        else:
            stagnation += 1
            if stagnation >= 3:          # ||F'|| no longer decreases: rounding floor of the gradient
                break
    gn, theta = best
    info = {"grad_norm": gn, "iters": it, "err_bound": gn / prob.mu}
    return theta, info


def solve_optimum(prob, **kw):
    """Compute eta_* (exactly for block quadratics, Newton otherwise) and store it in `prob`."""
    if isinstance(prob, (BlockQuadraticProblem, SharedQuadraticProblem, AlignedChainProblem)):
        theta = prob.exact_minimizer()
        info = {"grad_norm": float(np.linalg.norm(prob.grad(theta))), "iters": 0}
        info["err_bound"] = info["grad_norm"] / prob.mu
    else:
        theta, info = newton_solve(prob, **kw)
    prob.set_optimum(theta)
    prob.opt_info = info
    return theta, info
