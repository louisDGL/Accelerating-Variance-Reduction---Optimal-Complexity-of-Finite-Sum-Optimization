"""High-precision minimizer W_* of F.

  * LS: closed form W_* = (Y^T X / n)(X^T X / n + lam I)^{-1}, computed with a Cholesky solve (never an inverse), then
    at most `refine` steps of iterative refinement W <- W - grad F(W) (Sigma + lam I)^{-1} (kept only if they decrease
    ||grad F||_F);
  * softmax: damped Newton on vec(W) (dense Hessian) with Armijo backtracking, then a final pure-Newton polish.
    The Armijo test uses the exact Bregman form of F(W_new) - F(W) (`MatrixFiniteSum.F_diff`), whose rounding error is
    O(eps ||W_new - W||), so the test stays meaningful near the optimum (vrlab's solver had to switch to a gradient-norm
    test there because the plain difference of two values of F is decided by rounding).
Both report the achieved residual ||grad F(W_*)||_F and the strong-convexity error bound
||W_* - W_exact||_F <= ||grad F(W_*)||_F / lam (and F(W_*) - F(W_exact) <= ||grad F(W_*)||_F^2 / (2 lam)).
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla


def _info(prob, W, method, iters, tol=None, **extra):
    gn = float(np.linalg.norm(prob.grad(W)))
    g0 = float(np.linalg.norm(prob.grad(np.zeros((prob.K, prob.d)))))
    info = {"method": method, "iters": iters, "grad_norm": gn, "grad_norm0": g0, "err_bound": gn / prob.lam,
            "F_bound": gn ** 2 / (2.0 * prob.lam)}
    if tol is not None:
        info["tol"] = tol
        info["converged"] = bool(gn <= tol)
    info.update(extra)
    return info


def solve_ls(prob, refine=2):
    """Closed-form minimizer of the matrix least-squares problem. Returns (W, info)."""
    A = prob.Sigma_hat + prob.lam * np.eye(prob.d)
    cf = sla.cho_factor(A, lower=True)
    B = prob.Y.T @ prob.X / prob.n                          # K x d
    W = sla.cho_solve(cf, B.T).T                             # W A = B, A symmetric
    g = prob.grad(W)
    gn = float(np.linalg.norm(g))
    done = 0
    for _ in range(refine):
        W_new = W - sla.cho_solve(cf, g.T).T
        g_new = prob.grad(W_new)
        gn_new = float(np.linalg.norm(g_new))
        if not gn_new < gn:
            break
        W, g, gn = W_new, g_new, gn_new
        done += 1
    return W, _info(prob, W, "closed_form", 0, refine_steps=done)


def newton_softmax(prob, W0=None, tol_rel=1e-14, max_iter=100, polish=2, c1=1e-4, verbose=False):
    """Damped Newton method for the softmax problem, run until ||grad F||_F <= tol_rel * max(1, ||grad F(0)||_F).

    Each iteration: H = hess F(W) (dense), step S = H^{-1} grad F(W), decrement dec = <grad F(W), S>; the step size
    t in {1, 1/2, 1/4, ...} is the first one with F(W - t S) - F(W) <= -c1 t dec (difference evaluated by F_diff).
    The loop stops at the tolerance, after `max_iter` iterations, or when ||grad F|| has not decreased for 3 iterations
    (rounding floor). Then `polish` pure Newton steps (t = 1) are tried from the best iterate, keeping the iterate with
    the smallest ||grad F||_F. Returns (W, info) with info['grad_norm'], info['err_bound'], info['converged'], ..."""
    K, d = prob.K, prob.d
    g0n = float(np.linalg.norm(prob.grad(np.zeros((K, d)))))
    tol = tol_rel * max(1.0, g0n)
    W = np.zeros((K, d)) if W0 is None else np.array(W0, dtype=np.float64, copy=True)
    g = prob.grad(W)
    gn = float(np.linalg.norm(g))
    best_gn, best_W = gn, W.copy()
    it, stagnation, backtracks = 0, 0, 0
    while gn > tol and it < max_iter:
        H = prob.hessian(W)
        S = _newton_step(H, g)
        dec = float(np.vdot(g, S))
        t = 1.0
        while True:
            W_t = W - t * S
            if prob.F_diff(W_t, W, G0=g) <= -c1 * t * dec or t < 1e-10:
                break
            t *= 0.5
            backtracks += 1
        W = W_t
        g = prob.grad(W)
        gn = float(np.linalg.norm(g))
        it += 1
        if verbose:
            print(f"newton it={it} t={t:.3g} dec={dec:.3e} |g|={gn:.3e}")
        if gn < best_gn:
            best_gn, best_W = gn, W.copy()
            stagnation = 0
        else:
            stagnation += 1
            if stagnation >= 3:
                break
    # pure Newton polish from the best iterate
    W = best_W.copy()
    polished = 0
    for _ in range(polish):
        g = prob.grad(W)
        W = W - _newton_step(prob.hessian(W), g)
        gn = float(np.linalg.norm(prob.grad(W)))
        if gn < best_gn:
            best_gn, best_W = gn, W.copy()
            polished += 1
    return best_W, _info(prob, best_W, "newton", it, tol=tol, polish_improved=polished, backtracks=backtracks)


def _newton_step(H, g):
    """H^{-1} vec(g) reshaped like g (Cholesky; LU / least squares as fallbacks)."""
    shape = g.shape
    try:
        s = sla.cho_solve(sla.cho_factor(H, lower=True), g.ravel())
    except np.linalg.LinAlgError:
        try:
            s = np.linalg.solve(H, g.ravel())
        except np.linalg.LinAlgError:
            s = np.linalg.lstsq(H, g.ravel(), rcond=None)[0]
    return s.reshape(shape)


def solve_optimum(prob, **kw):
    """Compute W_* (closed form for LS, Newton for softmax), store it with `prob.set_optimum` and return (W, info)."""
    if prob.loss == "ls":
        W, info = solve_ls(prob, **kw)
    elif prob.loss == "softmax":
        W, info = newton_softmax(prob, **kw)
    else:
        raise ValueError(f"unknown loss {prob.loss!r}")
    prob.set_optimum(W, info)
    return W, info
