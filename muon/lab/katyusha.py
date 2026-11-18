"""Loopless Katyusha on matrix finite sums, with an optional spectral (sharp) step on the y sequence.

L-Katyusha (Kovalev, Horvath and Richtarik, ALT 2020), in the notation of vrlab.algorithms.lkatyusha:

    x   = theta1 z + theta2 w + (1 - theta1 - theta2) y
    g   = f_j'(x) - f_j'(w) + grad F(w)                     (= (r_j(x) - Rw_j) x_j^T + gbar + lam x, regularizer exact)
    z+  = (eta sigma x + z - (eta / L) g) / (1 + eta sigma),      sigma = mu / L
    y+  = x + theta1 (z+ - z)                                 ystep = 'coupling' (the published method)
    w+  = y with probability p, w otherwise.
Defaults: L = R2, mu = lam, theta2 = 1/2, theta1 = min{sqrt(2 sigma n / 3), 1/2}, eta = theta2 / ((1 + theta2) theta1),
p = 1/n. The z step is a Euclidean (mirror) step, linear in the estimator g, which is where its unbiasedness is used.

Katyusha in a general norm (Allen-Zhu, JMLR 2018, Section 6, "Katyusha2") keeps the linear z step for a
distance-generating function that is strongly convex with respect to the norm (here ||.||_F^2 / 2, which is 1-strongly
convex with respect to the spectral norm) and takes for y a steepest-descent step in the norm. For the spectral norm
this step is the sharp (nuclear-norm-scaled polar) step of Muon-type methods:

    y+  = x - gamma_y sharp(g) = x - gamma_y ||g||_* polar(g)        ystep = 'sharp'   (default gamma_y = 1/(9 L_y),
                                                                      L_y = R2_op; Allen-Zhu's step is 1/(9L))
    y+  = x - gamma_y g                                             ystep = 'euclid'  (same scheme with the Euclidean
                                                                      gradient step, default gamma_y = 1/(3 L))
The sharp variant is the instance of the general-norm template in the spectral geometry. It differs from Katyusha2
on three points, so that no theorem of the paper covers it: the regularizer is kept in the estimator instead of a
proximal term (Katyusha2 is composite, with psi in both steps), the indices are sampled uniformly (Katyusha2 samples
proportionally to L_i), and the scheme is loopless. It is the only place of the lab where a Muon-type step sits
inside an accelerated variance-reduced scheme. Experiment M7 measures the weight of the y step in this scheme (the
coupling step, the sharp step for several gamma_y, and no y step at all, y = x).

Gradient accounting as lsvrg: n initially, 2 per iteration, n per refresh of w. Traces are recorded at y.
"""
from __future__ import annotations

import numpy as np

from . import maps
from .algorithms import _Monitor, _default_record_every, _index_chunks


def lkatyusha_params(n, L, mu, theta2=0.5):
    sigma = mu / L
    theta1 = min(np.sqrt(2.0 * sigma * n / 3.0), 0.5)
    eta = theta2 / ((1.0 + theta2) * theta1)
    return dict(sigma=sigma, theta1=theta1, theta2=theta2, eta=eta, p=1.0 / n)


def lkatyusha(prob, W0, n_iters, L=None, mu=None, ystep="coupling", gamma_y=None, theta1=None, theta2=0.5, eta=None,
              p=None, seed=0, gauge="auto", record_every=None, stop_Fgap=None, stop_dist2=None, max_time=None,
              max_grad_evals=None, name=None):
    """Loopless Katyusha with ystep in {'coupling', 'sharp', 'euclid'} (see the module docstring). Returns the dict of
    algorithms._Monitor (traces at y) plus the parameters and the number of refreshes."""
    if ystep not in ("coupling", "sharp", "euclid"):
        raise ValueError(f"unknown ystep {ystep!r}")
    n, lam, X = prob.n, prob.lam, prob.X
    L = prob.R2 if L is None else float(L)
    mu = prob.mu if mu is None else float(mu)
    prm = lkatyusha_params(n, L, mu, theta2)
    sigma = prm["sigma"]
    p = prm["p"] if p is None else float(p)
    theta1 = prm["theta1"] if theta1 is None else float(theta1)
    eta = theta2 / ((1.0 + theta2) * theta1) if eta is None else float(eta)
    if theta1 + theta2 > 1.0 + 1e-12:
        raise ValueError("theta1 + theta2 must be <= 1")
    if gamma_y is None:
        gamma_y = 1.0 / (9.0 * prob.R2_op) if ystep == "sharp" else 1.0 / (3.0 * L)
    basis = maps.gauge_basis(prob) if (isinstance(gauge, str) and gauge == "auto") else None
    rng = np.random.default_rng(seed)
    y = np.array(W0, dtype=np.float64, copy=True)
    z = y.copy()
    w = y.copy()
    Rw = prob.loss_residuals(w)
    gbar = Rw.T @ X / n
    ge = n
    record_every = _default_record_every(prob) if record_every is None else record_every
    mon = _Monitor(prob, name or f"L-Katyusha[{ystep}]", record_every, stop_Fgap, stop_dist2, max_time)
    mon.record(0, ge, y)
    es = eta * sigma
    refreshes = 0
    t = 0
    stop = False
    for idx in _index_chunks(rng, n, n_iters):
        coins = rng.random(len(idx))
        for k in range(len(idx)):
            j = idx[k]
            x = theta1 * z + theta2 * w + (1.0 - theta1 - theta2) * y
            xj = X[j]
            rj = prob.residual_i(np.dot(x, xj), j)
            g = (rj - Rw[j])[:, None] * xj
            g += gbar
            g += lam * x
            z_new = (es * x + z - (eta / L) * g) / (1.0 + es)
            y_old = y
            if ystep == "coupling":
                y = x + theta1 * (z_new - z)
            elif ystep == "sharp":
                y = x - gamma_y * maps.sharp(g, basis=basis)
            else:
                y = x - gamma_y * g
            z = z_new
            if coins[k] < p:
                w = y_old.copy()
                Rw = prob.loss_residuals(w)
                gbar = Rw.T @ X / n
                ge += n
                refreshes += 1
            t += 1
            ge += 2
            if t % mon.record_every == 0 and mon.record(t, ge, y):
                stop = True
                break
            if max_grad_evals is not None and ge >= max_grad_evals:
                stop = True
                break
        if stop:
            break
    mon.finish(t, ge, y)
    return mon.result(y, L=L, mu=mu, sigma=sigma, theta1=theta1, theta2=theta2, eta=eta, p=p, ystep=ystep,
                      gamma_y=gamma_y, seed=seed, refreshes=refreshes, n=n)
