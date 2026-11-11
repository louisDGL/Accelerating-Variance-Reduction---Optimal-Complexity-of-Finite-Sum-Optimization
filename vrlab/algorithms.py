"""Optimization algorithms with gradient-evaluation accounting and (for SAGA) full instrumentation.

All stochastic methods draw indices uniformly from {0..n-1} with a numpy Generator seeded by `seed`.
Gradient accounting: one component gradient f'_i costs 1; a full gradient F' costs n.
Traces record ||theta - eta_*||^2 (Euclidean, and in the P-metric for preconditioned SAGA) and F - F_*
at `record_every` iterations.
"""
from __future__ import annotations

import time

import numpy as np

from .instrumentation import Metric, saga_state_stats, init_ratio, lkatyusha_state_stats
from .bounds import saga_textbook_params, lkatyusha_params


class Trace:
    """Container for recorded quantities along a run."""

    def __init__(self, name):
        self.name = name
        self.iters = []
        self.grad_evals = []
        self.D = []            # ||theta - eta_*||^2 (Euclidean)
        self.Fgap = []         # F(theta) - F_*
        self.extra = {}        # name -> list

    def add(self, it, ge, D, Fgap, **extra):
        self.iters.append(it)
        self.grad_evals.append(ge)
        self.D.append(D)
        self.Fgap.append(Fgap)
        for k, v in extra.items():
            self.extra.setdefault(k, []).append(v)

    def arrays(self):
        out = {"iters": np.array(self.iters), "grad_evals": np.array(self.grad_evals, dtype=np.float64),
               "D": np.array(self.D), "Fgap": np.array(self.Fgap)}
        for k, v in self.extra.items():
            out[k] = np.array(v)
        return out


def _dist2(prob, theta):
    v = theta - prob.theta_star
    return float(v @ v)


def _fgap(prob, theta):
    return float(prob.F(theta) - prob.F_star)


def _index_stream(rng, n, n_iters, chunk=1_000_000):
    done = 0
    while done < n_iters:
        k = min(chunk, n_iters - done)
        yield rng.integers(0, n, size=k)
        done += k


# ----------------------------------------------------------------------------------------------
# SAGA (optionally preconditioned), with exact instrumentation of the proof of Prop. 5.9
# ----------------------------------------------------------------------------------------------
def saga(prob, theta0, gamma=None, n_iters=None, seed=0, P=None, Delta=None, R2=None, mu=None,
         record_every=None, instrument_every=None, instrument_max=None, record_F=True,
         stop_D=None, max_time=None, resync_every=None, name="SAGA", passes=None, snapshot_theta=False):
    """SAGA: theta_t = theta_{t-1} - gamma P^{-1} [f'_{i}(theta_{t-1}) + zbar - z_i], z_i <- f'_i(theta_{t-1}).

    Parameters
    ----------
    gamma, Delta : step size and Lyapunov weight (defaults: textbook 1/(4R^2), 3 gamma^2 with R^2 = `R2`)
    R2, mu       : constants used in the ratios/bounds (defaults: prob.R2, prob.mu; for a metric P pass
                   the P-constants R2_P, mu_P)
    P            : None | diagonal (d,) | dense SPD (d,d) preconditioner
    instrument_every : compute the exact conditional expectations (O(nd)) every k iterations
    passes       : alternative to n_iters (number of effective passes, n iterations each)
    Returns a dict with the trace arrays, the list of instrumentation records, and run metadata.
    """
    n, d = prob.n, prob.d
    metric = Metric(P, d)
    R2 = prob.R2 if R2 is None else R2
    mu = prob.mu if mu is None else mu
    if gamma is None:
        gamma, Delta_tb = saga_textbook_params(R2)
    if Delta is None:
        Delta = 3.0 * gamma ** 2
    if n_iters is None:
        n_iters = int(passes * n)
    record_every = max(1, n // 10) if record_every is None else record_every
    resync_every = n if resync_every is None else resync_every
    rng = np.random.default_rng(seed)
    gs = prob.grads_star

    theta = np.array(theta0, dtype=np.float64).copy()
    z = prob.all_grads(theta)                     # z_i^(0) = f'_i(theta_0)
    zbar = z.mean(0)
    M = float(metric.dual_norm2_rows(z - gs).sum())
    ge = n                                        # initial gradient evaluations
    r_init, S0, D0 = init_ratio(prob, theta, R2, metric)

    tr = Trace(name)
    instr = []
    D = metric.norm2(theta - prob.theta_star)
    tr.add(0, ge, _dist2(prob, theta), _fgap(prob, theta) if record_F else np.nan, D_P=D, M=M, V=D + Delta * M)
    t = 0
    t0 = time.time()
    stopped = None
    for idx in _index_stream(rng, n, n_iters):
        for i in idx:
            # ---- instrumentation at the state before the step
            if instrument_every and t % instrument_every == 0 and (instrument_max is None or len(instr) < instrument_max):
                st = saga_state_stats(prob, theta, z, zbar, gamma, Delta, R2, mu, metric)
                st["t"] = t
                st["grad_evals"] = ge
                if snapshot_theta:
                    st["theta"] = theta.copy()
                instr.append(st)
                M = st["M"]                        # resync the incremental memory error
            # ---- SAGA step
            gi = prob.grad_i(theta, i)
            omega = gi + zbar - z[i]
            theta -= gamma * metric.apply_inv(omega)
            M += metric.dual_norm2(gi - gs[i]) - metric.dual_norm2(z[i] - gs[i])
            zbar += (gi - z[i]) / n
            z[i] = gi
            t += 1
            ge += 1
            if t % resync_every == 0:
                zbar = z.mean(0)
            # ---- recording
            if t % record_every == 0:
                D = metric.norm2(theta - prob.theta_star)
                tr.add(t, ge, _dist2(prob, theta), _fgap(prob, theta) if record_F else np.nan,
                       D_P=D, M=M, V=D + Delta * M)
                if stop_D is not None and D <= stop_D:
                    stopped = "stop_D"
                    break
                if max_time is not None and time.time() - t0 > max_time:
                    stopped = "max_time"
                    break
        if stopped:
            break
    out = tr.arrays()
    out.update({"instr": instr, "gamma": gamma, "Delta": Delta, "R2": R2, "mu": mu, "n": n, "d": d,
                "r_init": r_init, "S0": S0, "D0": D0, "seed": seed, "stopped": stopped,
                "elapsed": time.time() - t0, "theta": theta, "name": name, "metric": metric.kind})
    return out


# ----------------------------------------------------------------------------------------------
# loopless SVRG
# ----------------------------------------------------------------------------------------------
def lsvrg(prob, theta0, gamma, n_iters=None, p=None, seed=0, record_every=None, record_F=True,
          stop_D=None, max_time=None, name="L-SVRG", passes=None, count_reference_grad=True):
    """Loopless SVRG (Hofmann et al. 2015; Kovalev et al. 2020): g = f'_i(theta) - f'_i(w) + F'(w),
    theta <- theta - gamma g; with probability p the reference w is reset to the current theta (full gradient).
    Accounting: 2 gradients per iteration (f'_i(theta) and f'_i(w)) + n per reference update."""
    n, d = prob.n, prob.d
    p = 1.0 / n if p is None else p
    if n_iters is None:
        n_iters = int(passes * n)
    record_every = max(1, n // 10) if record_every is None else record_every
    rng = np.random.default_rng(seed)
    theta = np.array(theta0, dtype=np.float64).copy()
    w = theta.copy()
    Gw = prob.all_grads(w)
    gw = Gw.mean(0)
    ge = n
    tr = Trace(name)
    tr.add(0, ge, _dist2(prob, theta), _fgap(prob, theta) if record_F else np.nan)
    t = 0
    t0 = time.time()
    stopped = None
    per_iter = 2 if count_reference_grad else 1
    for idx in _index_stream(rng, n, n_iters):
        coins = rng.random(len(idx))
        for i, c in zip(idx, coins):
            g = prob.grad_i(theta, i) - Gw[i] + gw
            new_ref = c < p
            if new_ref:
                w = theta.copy()
            theta -= gamma * g
            if new_ref:
                Gw = prob.all_grads(w)
                gw = Gw.mean(0)
                ge += n
            t += 1
            ge += per_iter
            if t % record_every == 0:
                D = _dist2(prob, theta)
                tr.add(t, ge, D, _fgap(prob, theta) if record_F else np.nan)
                if stop_D is not None and D <= stop_D:
                    stopped = "stop_D"
                    break
                if max_time is not None and time.time() - t0 > max_time:
                    stopped = "max_time"
                    break
        if stopped:
            break
    out = tr.arrays()
    out.update({"gamma": gamma, "p": p, "seed": seed, "stopped": stopped, "elapsed": time.time() - t0,
                "theta": theta, "name": name})
    return out


# ----------------------------------------------------------------------------------------------
# loopless Katyusha (Kovalev, Horvath, Richtarik 2020, Algorithm 2) and the naive variant theta_2 = 0
# ----------------------------------------------------------------------------------------------
def lkatyusha(prob, theta0, L=None, mu=None, n_iters=None, p=None, theta1=None, theta2=0.5, eta=None,
              seed=0, record_every=None, record_F=True, stop_D=None, max_time=None, name="L-Katyusha",
              passes=None, count_reference_grad=True, instrument_every=None, instrument_max=None):
    """L-Katyusha:
        x = theta1 z + theta2 w + (1 - theta1 - theta2) y
        g = f'_i(x) - f'_i(w) + F'(w)
        z+ = (1/(1 + eta sigma)) (eta sigma x + z - (eta/L) g),   sigma = mu/L
        y+ = x + theta1 (z+ - z)
        w+ = y (with probability p), w otherwise.
    Defaults (uniform sampling): theta2 = 1/2, theta1 = min{sqrt(2 sigma n/3), 1/2}, eta = theta2/((1+theta2) theta1), p = 1/n.
    Setting theta2 = 0 (and an explicit eta) gives the "naive" Nesterov-type coupling without negative momentum.
    """
    n, d = prob.n, prob.d
    L = prob.R2 if L is None else L
    mu = prob.mu if mu is None else mu
    prm = lkatyusha_params(n, L, mu, p, theta2 if theta2 > 0 else 0.5)
    sigma = prm["sigma"]
    p = prm["p"] if p is None else p
    theta1 = prm["theta1"] if theta1 is None else theta1
    if eta is None:
        eta = theta2 / ((1.0 + theta2) * theta1) if theta2 > 0 else 1.0 / (3.0 * theta1)
    assert theta1 + theta2 <= 1.0 + 1e-12
    if n_iters is None:
        n_iters = int(passes * n)
    record_every = max(1, n // 10) if record_every is None else record_every
    rng = np.random.default_rng(seed)
    y = np.array(theta0, dtype=np.float64).copy()
    z = y.copy()
    w = y.copy()
    Gw = prob.all_grads(w)
    gw = Gw.mean(0)
    ge = n
    tr = Trace(name)
    tr.add(0, ge, _dist2(prob, y), _fgap(prob, y) if record_F else np.nan)
    t = 0
    t0 = time.time()
    stopped = None
    es = eta * sigma
    per_iter = 2 if count_reference_grad else 1
    instr = []
    for idx in _index_stream(rng, n, n_iters):
        coins = rng.random(len(idx))
        for i, c in zip(idx, coins):
            if instrument_every and t % instrument_every == 0 and (instrument_max is None or len(instr) < instrument_max):
                st = lkatyusha_state_stats(prob, y, z, w, theta1, theta2, eta, L, mu, p, Gw=Gw, gw=gw)
                st["t"] = t
                st["grad_evals"] = ge
                instr.append(st)
            x = theta1 * z + theta2 * w + (1.0 - theta1 - theta2) * y
            g = prob.grad_i(x, i) - Gw[i] + gw
            z_new = (es * x + z - (eta / L) * g) / (1.0 + es)
            y_old = y
            y = x + theta1 * (z_new - z)
            z = z_new
            if c < p:
                w = y_old.copy()
                Gw = prob.all_grads(w)
                gw = Gw.mean(0)
                ge += n
            t += 1
            ge += per_iter
            if t % record_every == 0:
                D = _dist2(prob, y)
                tr.add(t, ge, D, _fgap(prob, y) if record_F else np.nan)
                if stop_D is not None and D <= stop_D:
                    stopped = "stop_D"
                    break
                if max_time is not None and time.time() - t0 > max_time:
                    stopped = "max_time"
                    break
                if not np.isfinite(D) or D > 1e30:
                    stopped = "diverged"
                    break
        if stopped:
            break
    out = tr.arrays()
    out.update({"theta1": theta1, "theta2": theta2, "eta": eta, "p": p, "L": L, "mu": mu, "seed": seed,
                "stopped": stopped, "elapsed": time.time() - t0, "theta": y, "name": name, "instr": instr,
                "n": n})
    return out


# ----------------------------------------------------------------------------------------------
# naive acceleration: Nesterov momentum with the loopless-SVRG estimator (hypothesis A2)
# ----------------------------------------------------------------------------------------------
def nesterov_svrg(prob, theta0, gamma, beta, n_iters=None, p=None, seed=0, record_every=None, record_F=True,
                  stop_D=None, max_time=None, name="Nesterov+SVRG", passes=None, count_reference_grad=True):
    """v = theta + beta (theta - theta_prev);  g = f'_i(v) - f'_i(w) + F'(w);  theta+ = v - gamma g;
    reference w <- v with probability p (loopless). beta = (sqrt(kappa)-1)/(sqrt(kappa)+1) is the usual choice."""
    n, d = prob.n, prob.d
    p = 1.0 / n if p is None else p
    if n_iters is None:
        n_iters = int(passes * n)
    record_every = max(1, n // 10) if record_every is None else record_every
    rng = np.random.default_rng(seed)
    theta = np.array(theta0, dtype=np.float64).copy()
    theta_prev = theta.copy()
    w = theta.copy()
    Gw = prob.all_grads(w)
    gw = Gw.mean(0)
    ge = n
    tr = Trace(name)
    tr.add(0, ge, _dist2(prob, theta), _fgap(prob, theta) if record_F else np.nan)
    t = 0
    t0 = time.time()
    stopped = None
    per_iter = 2 if count_reference_grad else 1
    for idx in _index_stream(rng, n, n_iters):
        coins = rng.random(len(idx))
        for i, c in zip(idx, coins):
            v = theta + beta * (theta - theta_prev)
            g = prob.grad_i(v, i) - Gw[i] + gw
            theta_prev = theta
            theta = v - gamma * g
            if c < p:
                w = v.copy()
                Gw = prob.all_grads(w)
                gw = Gw.mean(0)
                ge += n
            t += 1
            ge += per_iter
            if t % record_every == 0:
                D = _dist2(prob, theta)
                tr.add(t, ge, D, _fgap(prob, theta) if record_F else np.nan)
                if stop_D is not None and D <= stop_D:
                    stopped = "stop_D"
                    break
                if max_time is not None and time.time() - t0 > max_time:
                    stopped = "max_time"
                    break
                if not np.isfinite(D) or D > 1e30:
                    stopped = "diverged"
                    break
        if stopped:
            break
    out = tr.arrays()
    out.update({"gamma": gamma, "beta": beta, "p": p, "seed": seed, "stopped": stopped,
                "elapsed": time.time() - t0, "theta": theta, "name": name})
    return out


# ----------------------------------------------------------------------------------------------
# full-gradient methods
# ----------------------------------------------------------------------------------------------
def gd(prob, theta0, n_iters, step=None, record_every=1, record_F=True, stop_D=None, name="GD", max_time=None):
    """Gradient descent theta <- theta - step F'(theta), step = 1/L_F by default. n gradients per iteration."""
    n = prob.n
    step = 1.0 / prob.L_F if step is None else step
    theta = np.array(theta0, dtype=np.float64).copy()
    tr = Trace(name)
    ge = 0
    tr.add(0, ge, _dist2(prob, theta), _fgap(prob, theta) if record_F else np.nan)
    t0 = time.time()
    stopped = None
    for t in range(1, n_iters + 1):
        theta -= step * prob.grad(theta)
        ge += n
        if t % record_every == 0:
            D = _dist2(prob, theta)
            tr.add(t, ge, D, _fgap(prob, theta) if record_F else np.nan)
            if stop_D is not None and D <= stop_D:
                stopped = "stop_D"
                break
            if max_time is not None and time.time() - t0 > max_time:
                stopped = "max_time"
                break
    out = tr.arrays()
    out.update({"step": step, "stopped": stopped, "elapsed": time.time() - t0, "theta": theta, "name": name})
    return out


def agd(prob, theta0, n_iters, L=None, mu=None, record_every=1, record_F=True, stop_D=None, name="AGD", max_time=None):
    """Nesterov's accelerated gradient for strongly convex F (constant momentum):
    y = theta + beta (theta - theta_prev), theta+ = y - (1/L) F'(y), beta = (sqrt(L/mu) - 1)/(sqrt(L/mu) + 1)."""
    n = prob.n
    L = prob.L_F if L is None else L
    mu = prob.mu if mu is None else mu
    q = np.sqrt(L / mu)
    beta = (q - 1.0) / (q + 1.0)
    theta = np.array(theta0, dtype=np.float64).copy()
    theta_prev = theta.copy()
    tr = Trace(name)
    ge = 0
    tr.add(0, ge, _dist2(prob, theta), _fgap(prob, theta) if record_F else np.nan)
    t0 = time.time()
    stopped = None
    for t in range(1, n_iters + 1):
        y = theta + beta * (theta - theta_prev)
        theta_prev = theta
        theta = y - prob.grad(y) / L
        ge += n
        if t % record_every == 0:
            D = _dist2(prob, theta)
            tr.add(t, ge, D, _fgap(prob, theta) if record_F else np.nan)
            if stop_D is not None and D <= stop_D:
                stopped = "stop_D"
                break
            if max_time is not None and time.time() - t0 > max_time:
                stopped = "max_time"
                break
    out = tr.arrays()
    out.update({"L": L, "mu": mu, "beta": beta, "stopped": stopped, "elapsed": time.time() - t0, "theta": theta, "name": name})
    return out


def sgd(prob, theta0, n_iters=None, gamma0=None, decay=True, seed=0, record_every=None, record_F=True,
        name="SGD", passes=None):
    """Plain SGD with step gamma_t = gamma0 / (1 + gamma0 mu t) (decay) or constant gamma0 (Bach, Sec. 5.4)."""
    n = prob.n
    if n_iters is None:
        n_iters = int(passes * n)
    record_every = max(1, n // 10) if record_every is None else record_every
    gamma0 = 1.0 / prob.R2 if gamma0 is None else gamma0
    rng = np.random.default_rng(seed)
    theta = np.array(theta0, dtype=np.float64).copy()
    tr = Trace(name)
    ge = 0
    tr.add(0, ge, _dist2(prob, theta), _fgap(prob, theta) if record_F else np.nan)
    t = 0
    for idx in _index_stream(rng, n, n_iters):
        for i in idx:
            t += 1
            g = gamma0 / (1.0 + gamma0 * prob.mu * t) if decay else gamma0
            theta -= g * prob.grad_i(theta, i)
            ge += 1
            if t % record_every == 0:
                tr.add(t, ge, _dist2(prob, theta), _fgap(prob, theta) if record_F else np.nan)
    out = tr.arrays()
    out.update({"gamma0": gamma0, "seed": seed, "theta": theta, "name": name})
    return out


# ----------------------------------------------------------------------------------------------
# utilities
# ----------------------------------------------------------------------------------------------
def grads_to_accuracy(run, eps, key="D", relative=True):
    """First gradient-evaluation count at which run[key] <= eps * run[key][0] (relative) or <= eps.
    Log-linear interpolation between records; returns np.nan if never reached."""
    x = np.asarray(run["grad_evals"], dtype=np.float64)
    y = np.asarray(run[key], dtype=np.float64)
    target = eps * y[0] if relative else eps
    ok = np.where(y <= target)[0]
    if len(ok) == 0:
        return np.nan
    k = ok[0]
    if k == 0:
        return x[0]
    y0, y1 = max(y[k - 1], 1e-300), max(y[k], 1e-300)
    if y0 <= target:
        return x[k - 1]
    # interpolate log y linearly in x
    frac = (np.log(y0) - np.log(target)) / max(np.log(y0) - np.log(y1), 1e-300)
    return x[k - 1] + frac * (x[k] - x[k - 1])
