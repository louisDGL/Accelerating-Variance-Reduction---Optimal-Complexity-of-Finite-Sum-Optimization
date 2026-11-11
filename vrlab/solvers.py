"""High-precision computation of the minimizer eta_* of F."""
from __future__ import annotations

import numpy as np

from .problems import BlockQuadraticProblem, SharedQuadraticProblem, AlignedChainProblem


def newton_solve(prob, theta0=None, tol=1e-15, max_iter=200, verbose=False):
    """Damped Newton method with Armijo backtracking, run until ||F'(theta)|| <= tol or stagnation.

    Returns (theta, info) with info = {'grad_norm', 'iters', 'err_bound'} where
    err_bound = ||F'(theta)|| / mu bounds ||theta - eta_*|| (strong convexity).
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
        while prob.F(theta - t * step) > F0 - 1e-4 * t * dec and t > 1e-10:
            t *= 0.5
        theta = theta - t * step
        g = prob.grad(theta)
        gn = float(np.linalg.norm(g))
        it += 1
        if verbose:
            print(f"newton it={it} t={t:.3g} |g|={gn:.3e}")
        if gn < best[0]:
            best = (gn, theta.copy())
            stagnation = 0
        else:
            stagnation += 1
            if stagnation >= 3:          # rounding floor reached
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
