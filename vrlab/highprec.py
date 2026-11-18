"""High-precision re-evaluation of the logistic gradient on the stored float64 data (mpmath), to certify computed
minimizers. Not imported by vrlab/__init__.py: mpmath is needed only by the scripts that use this module
(experiments/check_optima_mp.py, experiments/exp_B2_checks.py).

The inputs X, y, lam and theta are the binary float64 values, converted exactly to mpmath numbers, so that the result
is the gradient of the problem actually solved (not of an unrounded ideal data set). This is careful high-precision
evaluation, not interval arithmetic.
"""
from __future__ import annotations

import numpy as np


def mp_logistic_grad(X, y, lam, theta, dps=60):
    """Gradient of F(theta) = (1/n) sum_i log(1 + exp(-y_i x_i^T theta)) + (lam/2) ||theta||^2 in `dps`-digit
    arithmetic. Returns (g, norm): g as a float64 array (the high-precision value rounded once), norm as a float
    (rounded from the high-precision norm of the unrounded gradient)."""
    import mpmath as mp
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    with mp.workdps(dps):
        th = [mp.mpf(float(t)) for t in theta]
        acc = [mp.mpf(0)] * d
        for i in range(n):
            xi = [mp.mpf(float(v)) for v in X[i]]
            m = mp.fsum(xi[j] * th[j] for j in range(d))
            yi = mp.mpf(float(y[i]))
            c = -yi / (1 + mp.exp(yi * m))          # derivative of log(1 + exp(-y m)) with respect to m, times y
            for j in range(d):
                acc[j] += c * xi[j]
        lam_mp = mp.mpf(float(lam))
        g = [acc[j] / n + lam_mp * th[j] for j in range(d)]
        norm = mp.sqrt(mp.fsum(v * v for v in g))
        g64 = np.array([float(v) for v in g])
        return g64, float(norm)


def newton_correction(prob, theta, dps=60):
    """One Newton step from the high-precision gradient at theta: returns (e1, r) with r = ||F'(theta)|| evaluated in
    `dps` digits and e1 = F''(theta)^{-1} F'(theta) solved with the float64 Hessian. theta - e1 estimates the exact
    minimizer of the stored problem up to O(||e1||^2) and the rounding of the subtraction; ||e1|| estimates the actual
    error of theta, whereas r / mu bounds it rigorously (strong convexity)."""
    g, r = mp_logistic_grad(prob.X, prob.y, prob.lam, theta, dps)
    e1 = np.linalg.solve(prob.hessian(theta), g)
    return e1, r
