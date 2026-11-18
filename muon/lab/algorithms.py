"""Muon-type (orthogonalized) and Euclidean first-order methods on matrix finite sums, with gradient accounting.

Methods (W in R^{K x d}; orth in {'svd', 'ns5', 'ns_cubic'} or a callable, see polar.py):
  gd          W <- W - step grad F(W)                                    Euclidean baseline, default step 1/L_F
  muon_full   W <- W - eta_t O_t, O_t = orth(M_t), M_t = full gradient (or its momentum); rules const / sharp /
              linesearch / bb
  sgd         minibatch SGD, optional momentum                           Euclidean baseline
  muon_sgd    Keller Jordan's Muon on minibatch gradients (momentum, Nesterov, ns5, shape scale, decoupled decay)
  storm_muon  STORM momentum-based variance reduction then orthogonalization (MARS-M / Muon-MVR family)
  saga        Euclidean SAGA, memory of the loss parts only (regularizer gradient exact)
  lsvrg       Euclidean loopless SVRG
  saga_muon   W <- W - eta_t orth(omega_t), omega_t the SAGA estimator; rules const / decay / sharp / bb
  lsvrg_muon  the same with the L-SVRG estimator

Gradient accounting (as vrlab): one per-sample gradient = 1, a full gradient = n, a minibatch of size b = b.
SAGA: n initially + 1 per iteration. L-SVRG: n initially + 2 per iteration (f_j' at W and at the reference point w,
the latter charged although the table of the f_j'(w) is stored) + n per refresh of w (n p = 1 per iteration in
expectation for p = 1/n). STORM: b at the first iteration, then 2b (the same minibatch at W_t and at W_{t-1}).
A full Hessian-vector product is charged as one full gradient (n) and also counted in out['hvp_evals']; the
'same_sample' BB secant of the VR-Muon methods charges 1 more per iteration.

Traces (`Trace`): iterations, grad_evals, Fgap = F(W) - F_* (accurate Bregman form, problems.F_diff),
dist2_F = ||W - W_*||_F^2, dist2_op = ||W - W_*||_op^2, plus method-specific fields (eta, ...), recorded at t = 0,
every `record_every` iterations and at the last iteration. Stop rules: stop_Fgap (Fgap <= stop_Fgap), stop_dist2
(dist2_F <= stop_dist2), max_time (CPU seconds, time.process_time); 'diverged' when dist2_F is not finite or > 1e30.
Every method returns a dict: the trace arrays, the final iterate 'W', 'stopped', 'elapsed' (CPU s), 'wall' (s) and
its parameters.

Randomness: numpy Generators seeded by `seed`. The single-index methods draw their indices with rng.integers in chunks
of 10^6 (as vrlab), so saga and saga_muon see the same index sequence for the same seed (lsvrg / lsvrg_muon draw one
coin per index after each chunk, as vrlab.lsvrg). Minibatches: sampling='iid' (indices uniform with replacement,
counted with multiplicity) or 'reshuffle' (random reshuffling: consecutive batches of a fresh permutation every epoch,
the incomplete last batch of an epoch dropped).
"""
from __future__ import annotations

import time

import numpy as np

from .instrumentation import instrument
from .polar import get_orth, spectral_norm

_CHUNK = 1_000_000


# ----------------------------------------------------------------------------------------------
# trace and monitoring
# ----------------------------------------------------------------------------------------------
class Trace:
    """Recorded quantities along a run (lists; `arrays()` converts them)."""

    def __init__(self, name):
        self.name = name
        self.iters = []
        self.grad_evals = []
        self.Fgap = []
        self.dist2_F = []
        self.dist2_op = []
        self.extra = {}

    def add(self, it, ge, Fgap, dist2_F, dist2_op, **extra):
        self.iters.append(int(it))
        self.grad_evals.append(float(ge))
        self.Fgap.append(float(Fgap))
        self.dist2_F.append(float(dist2_F))
        self.dist2_op.append(float(dist2_op))
        for k, v in extra.items():
            self.extra.setdefault(k, []).append(v)

    def arrays(self):
        out = {"iters": np.array(self.iters), "grad_evals": np.array(self.grad_evals),
               "Fgap": np.array(self.Fgap), "dist2_F": np.array(self.dist2_F), "dist2_op": np.array(self.dist2_op)}
        for k, v in self.extra.items():
            out[k] = np.array(v)
        return out


class _Monitor:
    """Recording, stop rules and timing shared by all methods."""

    def __init__(self, prob, name, record_every, stop_Fgap=None, stop_dist2=None, max_time=None):
        if prob.W_star is None:
            raise ValueError("set the optimum first (lab.optimum.solve_optimum(prob))")
        self.prob = prob
        self.tr = Trace(name)
        self.record_every = max(1, int(record_every))
        self.stop_Fgap, self.stop_dist2, self.max_time = stop_Fgap, stop_dist2, max_time
        self.stopped = None
        self.last = -1
        self.cpu0 = time.process_time()
        self.wall0 = time.perf_counter()

    def due(self, it):
        return it % self.record_every == 0

    def record(self, it, ge, W, **extra):
        """Record the state; returns True if a stop rule fired."""
        prob = self.prob
        E = W - prob.W_star
        d2F = float(np.vdot(E, E))
        if np.isfinite(d2F):
            d2op = spectral_norm(E) ** 2
            fg = prob.Fgap(W)
        else:
            d2op = fg = float("nan")
        self.tr.add(it, ge, fg, d2F, d2op, **extra)
        self.last = it
        if not np.isfinite(d2F) or d2F > 1e30:
            self.stopped = "diverged"
        elif self.stop_Fgap is not None and fg <= self.stop_Fgap:
            self.stopped = "stop_Fgap"
        elif self.stop_dist2 is not None and d2F <= self.stop_dist2:
            self.stopped = "stop_dist2"
        elif self.max_time is not None and time.process_time() - self.cpu0 > self.max_time:
            self.stopped = "max_time"
        return self.stopped is not None

    def finish(self, it, ge, W, **extra):
        if self.last != it:
            self.record(it, ge, W, **extra)

    def result(self, W, **meta):
        out = self.tr.arrays()
        out.update(meta)
        out.update({"name": self.tr.name, "W": W, "stopped": self.stopped,
                    "elapsed": time.process_time() - self.cpu0, "wall": time.perf_counter() - self.wall0})
        return out


# ----------------------------------------------------------------------------------------------
# schedules and index streams
# ----------------------------------------------------------------------------------------------
def lr_factors(schedule, T, warmup=0.0, decay_frac=0.2):
    """Multipliers f_t (t = 0..T-1) of the base learning rate, with s = t / T:
        'const' 1;  'cosine' (1 + cos(pi s)) / 2;  'linear' 1 - s;
        'wsd' (warmup-stable-decay) min(1, (1 - s) / decay_frac): constant, then linear decay to 0 over the last
        decay_frac of the run;  a callable schedule(t, T) -> factor.
    warmup > 0 multiplies by min(1, (t + 1) / (warmup T)) (linear warmup over the first `warmup` fraction)."""
    t = np.arange(T, dtype=np.float64)
    s = t / max(T, 1)
    if callable(schedule):
        f = np.array([float(schedule(int(k), T)) for k in range(T)])
    elif schedule == "const":
        f = np.ones(T)
    elif schedule == "cosine":
        f = 0.5 * (1.0 + np.cos(np.pi * s))
    elif schedule == "linear":
        f = 1.0 - s
    elif schedule == "wsd":
        f = np.minimum(1.0, (1.0 - s) / decay_frac)
    else:
        raise ValueError(f"unknown schedule {schedule!r}")
    if warmup > 0:
        f = f * np.minimum(1.0, (t + 1.0) / (warmup * T))
    return f


def _index_chunks(rng, n, n_iters):
    done = 0
    while done < n_iters:
        k = min(_CHUNK, n_iters - done)
        yield rng.integers(0, n, size=k)
        done += k


def _batch_stream(rng, n, batch, n_iters, sampling="iid"):
    if sampling == "iid":
        rows = max(1, _CHUNK // batch)
        done = 0
        while done < n_iters:
            k = min(rows, n_iters - done)
            for row in rng.integers(0, n, size=(k, batch)):
                yield row
            done += k
    elif sampling == "reshuffle":
        if batch > n:
            raise ValueError("batch > n is impossible without replacement")
        perm, pos = rng.permutation(n), 0
        for _ in range(n_iters):
            if pos + batch > n:
                perm, pos = rng.permutation(n), 0
            yield perm[pos:pos + batch]
            pos += batch
    else:
        raise ValueError(f"unknown sampling {sampling!r}")


def _default_record_every(prob, batch=1):
    return max(1, prob.n // (10 * batch))


# ----------------------------------------------------------------------------------------------
# variance-reduced estimators (used by the loops and by the exact unbiasedness tests)
# ----------------------------------------------------------------------------------------------
def saga_estimator(prob, W, j, rtab, zbar):
    """SAGA estimator with the memory of the loss parts only:
        omega = (r_j(W) - rtab_j) x_j^T + zbar + lam W,   zbar = (1/n) rtab^T X = (1/n) sum_i rtab_i x_i^T.
    E_j[omega] = (1/n) R(W)^T X + lam W = grad F(W) for any table rtab. Returns (omega, r_j(W), outer) with
    outer = (r_j(W) - rtab_j) x_j^T, the increment of n zbar when the memory of j is refreshed."""
    xj = prob.X[j]
    rj = prob.residual_i(np.dot(W, xj), j)
    outer = (rj - rtab[j])[:, None] * xj
    omega = outer + zbar
    omega += prob.lam * W
    return omega, rj, outer


def lsvrg_estimator(prob, W, j, Rw, gbar):
    """L-SVRG estimator f_j'(W) - f_j'(w) + grad F(w) = (r_j(W) - Rw_j) x_j^T + gbar + lam W, with Rw the residual
    table at the reference point w and gbar = (1/n) Rw^T X (the regularizer terms lam W - lam w + lam w = lam W are
    exact, so this is the same estimator whether or not lam is put inside the f_i). Returns (omega, r_j(W))."""
    xj = prob.X[j]
    rj = prob.residual_i(np.dot(W, xj), j)
    omega = (rj - Rw[j])[:, None] * xj
    omega += gbar
    omega += prob.lam * W
    return omega, rj


# ----------------------------------------------------------------------------------------------
# full-gradient methods
# ----------------------------------------------------------------------------------------------
def gd(prob, W0, n_iters, step=None, record_every=1, stop_Fgap=None, stop_dist2=None, max_time=None, name="GD"):
    """Gradient descent W <- W - step grad F(W), step = 1/L_F by default. n gradients per iteration."""
    n = prob.n
    step = 1.0 / prob.L_F if step is None else float(step)
    W = np.array(W0, dtype=np.float64, copy=True)
    mon = _Monitor(prob, name, record_every, stop_Fgap, stop_dist2, max_time)
    ge = 0
    mon.record(0, ge, W)
    t = 0
    for t in range(1, n_iters + 1):
        W -= step * prob.grad(W)
        ge += n
        if mon.due(t) and mon.record(t, ge, W):
            break
    mon.finish(t, ge, W)
    return mon.result(W, step=step)


def agd(prob, W0, n_iters, L=None, mu=None, record_every=1, stop_Fgap=None, stop_dist2=None, max_time=None,
        name="AGD"):
    """Nesterov's accelerated gradient method for a strongly convex objective, constant momentum:
        Y_{t+1} = V_t - (1/L) grad F(V_t),    V_{t+1} = Y_{t+1} + beta (Y_{t+1} - Y_t),
        beta = (sqrt(L/mu) - 1) / (sqrt(L/mu) + 1),
    with L = prob.L_F and mu = prob.lam by default (L and mu are tuned in the experiments: the local constants at the
    optimum can be far from these global bounds). The recorded iterate is Y_t. n gradients per iteration."""
    n = prob.n
    L = prob.L_F if L is None else float(L)
    mu = prob.lam if mu is None else float(mu)
    if not (L > 0 and 0 < mu <= L):
        raise ValueError("agd needs 0 < mu <= L")
    q = (L / mu) ** 0.5
    beta = (q - 1.0) / (q + 1.0)
    Y = np.array(W0, dtype=np.float64, copy=True)
    V = Y.copy()
    mon = _Monitor(prob, name, record_every, stop_Fgap, stop_dist2, max_time)
    ge = 0
    mon.record(0, ge, Y)
    t = 0
    for t in range(1, n_iters + 1):
        Y_new = V - prob.grad(V) / L
        V = Y_new + beta * (Y_new - Y)
        Y = Y_new
        ge += n
        if mon.due(t) and mon.record(t, ge, Y):
            break
    mon.finish(t, ge, Y)
    return mon.result(Y, L=L, mu=mu, beta=beta)


def lbfgs(prob, W0, n_evals, memory=10, stop_Fgap=None, stop_dist2=None, max_time=None, name="L-BFGS"):
    """Limited-memory BFGS (scipy.optimize, method 'L-BFGS-B' without bounds) on the full objective. Every evaluation
    of (F, grad F) by the line search is charged n gradients; the trace has one record per iteration, at the number
    of evaluations spent so far, and the run uses at most `n_evals` evaluations. The stopping tests of the solver are
    disabled (ftol = gtol = 0): it runs until the budget, a stop rule of the monitor, or until its line search can
    make no further progress (rounding level; 'stopped' is then 'linesearch')."""
    from scipy.optimize import minimize

    n = prob.n
    K, d = prob.K, prob.d
    mon = _Monitor(prob, name, 1, stop_Fgap, stop_dist2, max_time)
    W = np.array(W0, dtype=np.float64, copy=True)
    state = {"evals": 0, "it": 0, "x": W.ravel().copy()}

    def fun(x):
        Wx = x.reshape(K, d)
        state["evals"] += 1
        return prob.F(Wx), prob.grad(Wx).ravel()

    def callback(x):
        state["it"] += 1
        state["x"] = np.array(x, dtype=np.float64, copy=True)
        if mon.record(state["it"], state["evals"] * n, state["x"].reshape(K, d)):
            raise StopIteration

    mon.record(0, 0, W)
    try:
        minimize(fun, W.ravel(), jac=True, method="L-BFGS-B", callback=callback,
                 options={"maxcor": int(memory), "maxfun": int(n_evals), "maxiter": int(n_evals),
                          "ftol": 0.0, "gtol": 0.0})
    except StopIteration:                              # scipy < 1.11 lets the exception through
        pass
    if mon.stopped is None and state["evals"] < n_evals:
        mon.stopped = "linesearch"
    return mon.result(state["x"].reshape(K, d), memory=int(memory), evals=state["evals"])


def muon_full(prob, W0, n_iters, rule="sharp", lr=None, L=None, orth="svd", beta=0.0, nesterov=False, orth_kw=None,
              L_min=None, L_max=None, record_every=1, stop_Fgap=None, stop_dist2=None, max_time=None,
              instrument_every=None, instrument_max=None, name=None):
    """Deterministic orthogonalized descent W_{t+1} = W_t - eta_t O_t, O_t = orth(M_t), with M_t = G_t = grad F(W_t)
    (beta = 0) or the momentum B_t = beta B_{t-1} + G_t, M_t = G_t + beta B_t (nesterov) or B_t.

    Step rules (all use the pairing <G_t, O_t> with the current gradient; it equals ||G_t||_* when O_t = polar(G_t)):
      'const'      eta = lr.
      'sharp'      eta = <G, O> / L, L = prob.L_op by default: the dualized steepest-descent step of Bernstein and
                   Newhouse. Op-norm smoothness gives F(W - eta O) <= F(W) - eta <G, O> + (L/2) eta^2 ||O||_op^2, so
                   with an exact polar factor (||O||_op = 1) and L >= the true L_op the decrease is >= ||G||_*^2/(2L);
                   for an inexact O it is >= (<G, O>^2 / L)(1 - ||O||_op^2 / 2) > 0 as long as ||O||_op < sqrt(2).
      'linesearch' eta = <G, O> / <O, hess F(W) O>: exact minimizer of the quadratic model along O (exact line search
                   for LS); one Hessian-vector product per iteration (charged n gradients).
      'bb'         secant (Barzilai-Borwein) curvature in the spectral norm
                   L_t = <G_t - G_{t-1}, W_t - W_{t-1}> / ||W_t - W_{t-1}||_op^2, clipped to [L_min, L_max]
                   (defaults prob.mu_op and prob.L_op: for convex F the secant lies in this interval in exact
                   arithmetic, so the clip only guards against rounding when the steps become tiny), and
                   eta = <G_t, O_t> / L_t; the first step is 'sharp'.
    n gradients per iteration (2n with 'linesearch'). Trace extras: eta, pair = <G, O>, L_t (bb)."""
    n = prob.n
    if rule not in ("const", "sharp", "linesearch", "bb"):
        raise ValueError(f"unknown rule {rule!r}")
    if rule == "const" and lr is None:
        raise ValueError("rule 'const' needs lr")
    orth_fn = get_orth(orth, **(orth_kw or {}))
    L = prob.L_op if L is None else float(L)
    L_min = prob.mu_op if L_min is None else float(L_min)
    L_max = prob.L_op if L_max is None else float(L_max)
    W = np.array(W0, dtype=np.float64, copy=True)
    buf = np.zeros_like(W) if beta else None
    name = name or f"Muon-full[{rule},{orth if isinstance(orth, str) else 'custom'}]"
    mon = _Monitor(prob, name, record_every, stop_Fgap, stop_dist2, max_time)
    ge, hv = 0, 0
    eta = pair = L_t = float("nan")
    mon.record(0, ge, W, eta=eta, pair=pair, L_t=L_t)
    G_prev = W_prev = None
    instr = []
    t = 0
    for t in range(1, n_iters + 1):
        G = prob.grad(W)
        ge += n
        if beta:
            buf *= beta
            buf += G
            Dir = G + beta * buf if nesterov else buf
        else:
            Dir = G
        O = orth_fn(Dir)
        pair = float(np.vdot(G, O))
        if rule == "const":
            eta = float(lr)
        elif rule == "sharp":
            eta = pair / L
        elif rule == "linesearch":
            curv = float(np.vdot(O, prob.hvp(W, O)))
            hv += 1
            ge += n
            eta = pair / curv if curv > 0 else 0.0
        else:
            if G_prev is None:
                L_t = L
            else:
                dW = W - W_prev
                den = spectral_norm(dW) ** 2
                if den > 0:
                    L_t = min(max(float(np.vdot(G - G_prev, dW)) / den, L_min), L_max)
            eta = pair / L_t
            G_prev, W_prev = G, W
        if instrument_every and (t - 1) % instrument_every == 0 and (instrument_max is None or len(instr) < instrument_max):
            st = instrument(prob, W, G=Dir, O=O, eta=eta)
            st.update(t=t - 1, grad_evals=ge)
            instr.append(st)
        W = W - eta * O
        if mon.due(t) and mon.record(t, ge, W, eta=eta, pair=pair, L_t=L_t):
            break
    mon.finish(t, ge, W, eta=eta, pair=pair, L_t=L_t)
    return mon.result(W, rule=rule, lr=lr, L=L, L_min=L_min, L_max=L_max, orth=orth, beta=beta, nesterov=nesterov,
                      hvp_evals=hv, instr=instr)


# ----------------------------------------------------------------------------------------------
# minibatch methods
# ----------------------------------------------------------------------------------------------
def sgd(prob, W0, n_iters, batch, lr, schedule="const", seed=0, beta=0.0, nesterov=False, sampling="iid",
        warmup=0.0, decay_frac=0.2, record_every=None, stop_Fgap=None, stop_dist2=None, max_time=None, name="SGD"):
    """Minibatch SGD W <- W - lr_t D_t, D_t = G_t (beta = 0) or the momentum B_t = beta B_{t-1} + G_t,
    D_t = G_t + beta B_t (nesterov) or B_t; G_t = grad_batch (lam W included); lr_t = lr * lr_factors(schedule)[t].
    b gradients per iteration."""
    W = np.array(W0, dtype=np.float64, copy=True)
    lrs = float(lr) * lr_factors(schedule, n_iters, warmup, decay_frac)
    rng = np.random.default_rng(seed)
    record_every = _default_record_every(prob, batch) if record_every is None else record_every
    mon = _Monitor(prob, name, record_every, stop_Fgap, stop_dist2, max_time)
    buf = np.zeros_like(W) if beta else None
    ge = 0
    mon.record(0, ge, W, lr=float("nan"))
    t = 0
    for t, idx in enumerate(_batch_stream(rng, prob.n, batch, n_iters, sampling), start=1):
        G = prob.grad_batch(W, idx)
        ge += batch
        if beta:
            buf *= beta
            buf += G
            Dir = G + beta * buf if nesterov else buf
        else:
            Dir = G
        W -= lrs[t - 1] * Dir
        if mon.due(t) and mon.record(t, ge, W, lr=lrs[t - 1]):
            break
    mon.finish(t, ge, W, lr=lrs[t - 1] if t else float("nan"))
    return mon.result(W, batch=batch, lr=lr, schedule=schedule, seed=seed, beta=beta, nesterov=nesterov,
                      sampling=sampling)


def muon_sgd(prob, W0, n_iters, batch, lr, beta=0.95, nesterov=True, orth="ns5", schedule="const", seed=0,
             weight_decay=0.0, sampling="iid", warmup=0.0, decay_frac=0.2, orth_kw=None, record_every=None,
             stop_Fgap=None, stop_dist2=None, max_time=None, name=None):
    """Keller Jordan's Muon on minibatch gradients of F (lam W included in the gradient):
        G = grad_batch(W, idx);  B <- beta B + G;  D = G + beta B (nesterov) or B;  O = orth(D);
        W <- (1 - lr_t weight_decay) W - lr_t scale O,   scale = max(1, K/d)^(1/2),  lr_t = lr * lr_factors(...)[t].
    (The current public code keeps an EMA m = beta m + (1 - beta) G and uses (1 - beta) G + beta m; both directions are
    (1 - beta) times the ones above, a factor removed by the orthogonalization up to the eps of ns5.)
    b gradients per iteration."""
    K, d = prob.K, prob.d
    orth_fn = get_orth(orth, **(orth_kw or {}))
    scale = max(1.0, K / d) ** 0.5
    W = np.array(W0, dtype=np.float64, copy=True)
    lrs = float(lr) * lr_factors(schedule, n_iters, warmup, decay_frac)
    rng = np.random.default_rng(seed)
    record_every = _default_record_every(prob, batch) if record_every is None else record_every
    name = name or f"Muon-SGD[{orth if isinstance(orth, str) else 'custom'}]"
    mon = _Monitor(prob, name, record_every, stop_Fgap, stop_dist2, max_time)
    buf = np.zeros_like(W)
    ge = 0
    mon.record(0, ge, W, lr=float("nan"))
    t = 0
    for t, idx in enumerate(_batch_stream(rng, prob.n, batch, n_iters, sampling), start=1):
        G = prob.grad_batch(W, idx)
        ge += batch
        buf *= beta
        buf += G
        Dir = G + beta * buf if nesterov else buf
        O = orth_fn(Dir)
        lr_t = lrs[t - 1]
        if weight_decay:
            W *= 1.0 - lr_t * weight_decay
        W -= (lr_t * scale) * O
        if mon.due(t) and mon.record(t, ge, W, lr=lr_t):
            break
    mon.finish(t, ge, W, lr=lrs[t - 1] if t else float("nan"))
    return mon.result(W, batch=batch, lr=lr, beta=beta, nesterov=nesterov, orth=orth, schedule=schedule, seed=seed,
                      weight_decay=weight_decay, sampling=sampling, scale=scale)


def storm_muon(prob, W0, n_iters, batch, lr, a=0.1, orth="ns5", schedule="const", seed=0, weight_decay=0.0,
               sampling="iid", warmup=0.0, decay_frac=0.2, orth_kw=None, record_every=None, stop_Fgap=None,
               stop_dist2=None, max_time=None, name=None):
    """STORM estimator then orthogonalization (the MARS-M / Muon-MVR family):
        d_0 = g_0(W_0);  d_t = g_t(W_t) + (1 - a) (d_{t-1} - g_t(W_{t-1}))   (the SAME minibatch at W_t and W_{t-1}),
        W_{t+1} = (1 - lr_t weight_decay) W_t - lr_t scale orth(d_t),  scale = max(1, K/d)^(1/2).
    a = 1 gives Muon without momentum; small a keeps a long, corrected memory. b gradients at t = 0, then 2b."""
    K, d = prob.K, prob.d
    orth_fn = get_orth(orth, **(orth_kw or {}))
    scale = max(1.0, K / d) ** 0.5
    W = np.array(W0, dtype=np.float64, copy=True)
    lrs = float(lr) * lr_factors(schedule, n_iters, warmup, decay_frac)
    rng = np.random.default_rng(seed)
    record_every = _default_record_every(prob, batch) if record_every is None else record_every
    name = name or f"STORM-Muon[{orth if isinstance(orth, str) else 'custom'}]"
    mon = _Monitor(prob, name, record_every, stop_Fgap, stop_dist2, max_time)
    ge = 0
    mon.record(0, ge, W, lr=float("nan"))
    Dm = W_prev = None
    t = 0
    for t, idx in enumerate(_batch_stream(rng, prob.n, batch, n_iters, sampling), start=1):
        G = prob.grad_batch(W, idx)
        ge += batch
        if Dm is None:
            Dm = G
        else:
            Gp = prob.grad_batch(W_prev, idx)
            ge += batch
            Dm = G + (1.0 - a) * (Dm - Gp)
        O = orth_fn(Dm)
        W_prev = W.copy()
        lr_t = lrs[t - 1]
        if weight_decay:
            W *= 1.0 - lr_t * weight_decay
        W -= (lr_t * scale) * O
        if mon.due(t) and mon.record(t, ge, W, lr=lr_t):
            break
    mon.finish(t, ge, W, lr=lrs[t - 1] if t else float("nan"))
    return mon.result(W, batch=batch, lr=lr, a=a, orth=orth, schedule=schedule, seed=seed,
                      weight_decay=weight_decay, sampling=sampling, scale=scale)


# ----------------------------------------------------------------------------------------------
# variance-reduced methods (single-index)
# ----------------------------------------------------------------------------------------------
def saga(prob, W0, n_iters, gamma=None, seed=0, record_every=None, stop_Fgap=None, stop_dist2=None, max_time=None,
         resync_every=None, instrument_every=None, instrument_max=None, name="SAGA"):
    """Euclidean SAGA with the memory of the LOSS parts only:
        omega_t = (r_j(W) - rtab_j) x_j^T + zbar + lam W,  W <- W - gamma omega_t,  rtab_j <- r_j(W),
    with the n x K residual table rtab (rank-one memories z_i = rtab_i x_i^T) and zbar = (1/n) rtab^T X maintained
    incrementally in O(K d) per iteration (recomputed exactly every `resync_every` = n iterations). Default
    gamma = 1/(4 R2) (Bach, Prop. 5.9).

    Difference with vrlab.saga, which puts lam inside every f_i and stores the full f_i'(theta): here the gradient
    lam W of the regularizer is exact at every step, so (i) the memory is n K numbers instead of n K d, and (ii) the
    estimator does not carry the stale terms lam (W - W_{last visit of j}) (less variance). Prop. 5.9 still holds with
    the same constants: with M = sum_i ||z_i - g_i'(W_*)||_F^2, the memory obeys E[M_+] = (1 - 1/n) M + S_g / n with
    S_g = sum_i ||g_i'(W) - g_i'(W_*)||^2 <= S = sum_i ||f_i'(W) - f_i'(W_*)||^2 (monotonicity of g_i'), and
    E||omega||^2 <= 2 S/n + 2 M/n as in the book, so the Lyapunov recursion of the proof is unchanged. (The
    comparison S_g <= S uses the isotropy of the regularizer; in a non-Euclidean metric it is replaced by the
    co-coercivity of the loss parts, fact F11 of notes_theory.md.)
    Accounting: n initially, 1 per iteration."""
    gamma = 1.0 / (4.0 * prob.R2) if gamma is None else float(gamma)
    return _vr_run(prob, W0, n_iters, "saga", "euclid", gamma=gamma, seed=seed, record_every=record_every,
                   stop_Fgap=stop_Fgap, stop_dist2=stop_dist2, max_time=max_time, resync_every=resync_every,
                   instrument_every=instrument_every, instrument_max=instrument_max, name=name)


def lsvrg(prob, W0, n_iters, gamma=None, p=None, seed=0, record_every=None, stop_Fgap=None, stop_dist2=None,
          max_time=None, instrument_every=None, instrument_max=None, name="L-SVRG"):
    """Euclidean loopless SVRG (Hofmann et al. 2015; Kovalev, Horvath, Richtarik 2020): omega = f_j'(W) - f_j'(w) +
    grad F(w), W <- W - gamma omega, and with probability p (default 1/n) the reference w is reset to the pre-step
    point. Default gamma = 1/(6 R2) (the step of Kovalev et al.). Accounting: n + 2 per iteration + n per refresh."""
    gamma = 1.0 / (6.0 * prob.R2) if gamma is None else float(gamma)
    return _vr_run(prob, W0, n_iters, "lsvrg", "euclid", gamma=gamma, p=p, seed=seed, record_every=record_every,
                   stop_Fgap=stop_Fgap, stop_dist2=stop_dist2, max_time=max_time, instrument_every=instrument_every,
                   instrument_max=instrument_max, name=name)


_VR_DOC = """
    Step rules (O_t = orth(D_t), D_t = omega_t, or the momentum B_t = beta B_{t-1} + omega_t, D_t = omega_t + beta B_t
    (nesterov) or B_t; pair_t = <omega_t, O_t>, which equals ||omega_t||_* when O_t = polar(omega_t)):
      'const'  eta_t = lr.
      'decay'  eta_t = lr (1 + t / decay_T0)^(-decay_power), defaults decay_T0 = n, decay_power = 1/2.
      'sharp'  eta_t = gamma pair_t (= ||omega_t||_* / L with L = 1/gamma). Default gamma = 1/(4 R2_op) for SAGA and
               1/(6 R2_op) for L-SVRG, R2_op = max_i L_op_i: the textbook steps 1/(4 R^2) (Bach, Prop. 5.9) and
               1/(6 R^2) (Kovalev et al.) with the per-sample Euclidean smoothness R^2 replaced by the per-sample
               spectral smoothness, i.e. a transfer of the constants, not a theorem (the map
               omega -> ||omega||_* polar(omega) is nonlinear, so the orthogonalized step is biased). As the estimator
               error vanishes at W_* (memory converged), the step vanishes with it. `L` may be given instead of gamma.
               Observed (see muon/README.md): stable for SAGA on the test instances; unstable for L-SVRG on matrix LS
               (the estimator is then nearly rank one and dominated by the noise (W - w)(x_j x_j^T - Sigma), and its
               nuclear norm sets a step that the polar factor spreads over the weak gradient directions).
      'bb'     eta_t = pair_t / (bb_safety L_hat_t): L_hat_t is a secant estimate of the spectral curvature along the
               recent steps, built without full gradients from estimator differences,
                 L_hat_t = clip( Nbar_t / Dbar_t, L_min, L_max ),   Nbar, Dbar = EMAs (weight 1/bb_window) of
                 N_t = <Delta omega_t, W_t - W_{t-1}>,  D_t = ||W_t - W_{t-1}||_op^2,
               with Delta omega_t =
                 'same_sample' (default): omega_{j_t}(W_t) - omega_{j_t}(W_{t-1}) = f_j'(W_t) - f_j'(W_{t-1}) (same
                   index and memory at both points: the memory terms cancel), computed without cancellation as
                   (r_j(W_t) - r_j(W_{t-1})) x_j^T + lam (W_t - W_{t-1}); E_j[N_t] is the exact secant
                   <grad F(W_t) - grad F(W_{t-1}), W_t - W_{t-1}>; costs 1 more per-sample gradient per iteration;
                 'consecutive': omega_t - omega_{t-1} (free, but biased: given the past,
                   E[N_t] = eta_{t-1} (<omega_{t-1} - grad F(W_t), O_{t-1}>), whose noise part
                   eta_{t-1} <omega_{t-1} - grad F(W_{t-1}), O_{t-1}> is positive because O_{t-1} = orth(omega_{t-1})
                   is aligned with the estimator noise; when the noise dominates (the usual SAGA regime) N_t / D_t is
                   about pair_{t-1} / eta_{t-1} = bb_safety L_hat_{t-1}, so the estimate does not track the curvature).
               Defaults: bb_window = n, bb_safety = 4 (the factor of the textbook step), L_min = mu_op, L_max = R2_op
               (for the same-sample secant N_t / D_t lies in [lam, L_op_j] in exact arithmetic, so the clip is inactive
               except under rounding); before the first secant the step is 'sharp'.
    Trace extras: eta (last step), pair (last pairing), L_hat."""


def saga_muon(prob, W0, n_iters, rule="sharp", gamma=None, lr=None, L=None, orth="svd", seed=0, beta=0.0,
              nesterov=False, decay_T0=None, decay_power=0.5, bb_secant="same_sample", bb_window=None, bb_safety=4.0,
              L_min=None, L_max=None, orth_kw=None, record_every=None, stop_Fgap=None, stop_dist2=None,
              max_time=None, resync_every=None, instrument_every=None, instrument_max=None, name=None):
    """Orthogonalized SAGA: W <- W - eta_t orth(D_t) with the SAGA estimator omega_t of `saga` (loss-part memory,
    exact regularizer gradient). Accounting: n initially, 1 per iteration (2 with the 'same_sample' BB secant)."""
    return _vr_run(prob, W0, n_iters, "saga", rule, gamma=gamma, lr=lr, L=L, orth=orth, orth_kw=orth_kw, seed=seed,
                   beta=beta, nesterov=nesterov, decay_T0=decay_T0, decay_power=decay_power, bb_secant=bb_secant,
                   bb_window=bb_window, bb_safety=bb_safety, L_min=L_min, L_max=L_max, record_every=record_every,
                   stop_Fgap=stop_Fgap, stop_dist2=stop_dist2, max_time=max_time, resync_every=resync_every,
                   instrument_every=instrument_every, instrument_max=instrument_max, name=name)


def lsvrg_muon(prob, W0, n_iters, rule="sharp", p=None, gamma=None, lr=None, L=None, orth="svd", seed=0, beta=0.0,
               nesterov=False, decay_T0=None, decay_power=0.5, bb_secant="same_sample", bb_window=None,
               bb_safety=4.0, L_min=None, L_max=None, orth_kw=None, record_every=None, stop_Fgap=None,
               stop_dist2=None, max_time=None, instrument_every=None, instrument_max=None, name=None):
    """Orthogonalized loopless SVRG: W <- W - eta_t orth(D_t) with the L-SVRG estimator f_j'(W) - f_j'(w) + grad F(w);
    the reference w is reset to the pre-step point with probability p (default 1/n). Accounting: n initially,
    2 per iteration (3 with the 'same_sample' BB secant), n per refresh."""
    return _vr_run(prob, W0, n_iters, "lsvrg", rule, gamma=gamma, lr=lr, L=L, orth=orth, orth_kw=orth_kw, seed=seed,
                   beta=beta, nesterov=nesterov, p=p, decay_T0=decay_T0, decay_power=decay_power,
                   bb_secant=bb_secant, bb_window=bb_window, bb_safety=bb_safety, L_min=L_min, L_max=L_max,
                   record_every=record_every, stop_Fgap=stop_Fgap, stop_dist2=stop_dist2, max_time=max_time,
                   instrument_every=instrument_every, instrument_max=instrument_max, name=name)


saga_muon.__doc__ += _VR_DOC
lsvrg_muon.__doc__ += _VR_DOC


def _vr_run(prob, W0, n_iters, estimator, rule, gamma=None, lr=None, L=None, orth="svd", orth_kw=None, seed=0,
            beta=0.0, nesterov=False, p=None, decay_T0=None, decay_power=0.5, bb_secant="same_sample",
            bb_window=None, bb_safety=4.0, L_min=None, L_max=None, record_every=None, stop_Fgap=None,
            stop_dist2=None, max_time=None, resync_every=None, instrument_every=None, instrument_max=None, name=None):
    """Shared loop of saga, lsvrg, saga_muon and lsvrg_muon (estimator in {'saga', 'lsvrg'};
    rule in {'euclid', 'const', 'decay', 'sharp', 'bb'})."""
    n, lam, X = prob.n, prob.lam, prob.X
    if estimator not in ("saga", "lsvrg"):
        raise ValueError(f"unknown estimator {estimator!r}")
    if rule not in ("euclid", "const", "decay", "sharp", "bb"):
        raise ValueError(f"unknown rule {rule!r}")
    if rule in ("const", "decay") and lr is None:
        raise ValueError(f"rule {rule!r} needs lr")
    if bb_secant not in ("same_sample", "consecutive"):
        raise ValueError(f"unknown bb_secant {bb_secant!r}")
    is_saga = estimator == "saga"
    euclid = rule == "euclid"
    use_bb = rule == "bb"
    same_sample = use_bb and bb_secant == "same_sample"
    if euclid:
        gamma = float(gamma)
        orth_fn, exact_polar = None, False
    else:
        orth_fn = get_orth(orth, **(orth_kw or {}))
        exact_polar = orth == "svd"
    if gamma is not None:
        g_sharp = float(gamma)
    elif L is not None:
        g_sharp = 1.0 / float(L)
    else:                                 # textbook constants with R^2 -> R2_op: 1/(4 R^2) SAGA, 1/(6 R^2) L-SVRG
        g_sharp = 1.0 / ((4.0 if is_saga else 6.0) * prob.R2_op)
    T0 = float(n if decay_T0 is None else decay_T0)
    L_min = prob.mu_op if L_min is None else float(L_min)
    L_max = prob.R2_op if L_max is None else float(L_max)
    a_bb = 1.0 / float(n if bb_window is None else bb_window)
    p = 1.0 / n if p is None else float(p)
    record_every = _default_record_every(prob) if record_every is None else record_every
    resync_every = n if resync_every is None else int(resync_every)
    per_iter = (1 if is_saga else 2) + (1 if same_sample else 0)
    if name is None:
        base = "SAGA" if is_saga else "L-SVRG"
        name = base if euclid else f"{base}-Muon[{rule},{orth if isinstance(orth, str) else 'custom'}]"

    W = np.array(W0, dtype=np.float64, copy=True)
    rng = np.random.default_rng(seed)
    if is_saga:
        rtab = prob.loss_residuals(W)                 # memory r_i at W_0
        zbar = rtab.T @ X / n
    else:
        Rw = prob.loss_residuals(W)                   # reference w = W_0
        gbar = Rw.T @ X / n
    refreshes = 0
    ge = n
    mon = _Monitor(prob, name, record_every, stop_Fgap, stop_dist2, max_time)
    eta = gamma if euclid else float("nan")
    pair = L_hat = float("nan")
    mon.record(0, ge, W, eta=eta, pair=pair, L_hat=L_hat)
    buf = np.zeros_like(W) if beta else None
    Nbar = Dbar = 0.0
    dW = omega_prev = None
    dW_op2 = 0.0
    instr = []
    t = 0
    stop = False
    for idx in _index_chunks(rng, n, n_iters):
        coins = None if is_saga else rng.random(len(idx))
        for k in range(len(idx)):
            j = idx[k]
            if is_saga:
                omega, rj, outer = saga_estimator(prob, W, j, rtab, zbar)
            else:
                omega, rj = lsvrg_estimator(prob, W, j, Rw, gbar)
            # ---- secant of the previous step (BB rule)
            if use_bb and dW is not None and dW_op2 > 0.0:
                if same_sample:
                    xj = X[j]
                    delta = np.dot(dW, xj)                                   # (W_t - W_{t-1}) x_j
                    dr = prob.residual_diff_i(np.dot(W, xj), delta, j)        # r_j(W_t) - r_j(W_{t-1})
                    num = float(np.dot(dr, delta)) + lam * float(np.vdot(dW, dW))
                else:
                    num = float(np.vdot(omega - omega_prev, dW))
                Nbar += a_bb * (num - Nbar)
                Dbar += a_bb * (dW_op2 - Dbar)
            # ---- direction
            if beta:
                buf *= beta
                buf += omega
                Dir = omega + beta * buf if nesterov else buf
            else:
                Dir = omega
            # ---- step
            if euclid:
                step = gamma * Dir
                O = None
            else:
                O = orth_fn(Dir)
                pair = float(np.vdot(omega, O))
                if rule == "sharp":
                    eta = g_sharp * pair
                elif rule == "const":
                    eta = float(lr)
                elif rule == "decay":
                    eta = float(lr) * (1.0 + t / T0) ** (-decay_power)
                elif Dbar > 0.0:
                    L_hat = min(max(Nbar / Dbar, L_min), L_max)
                    eta = pair / (bb_safety * L_hat)
                else:
                    eta = g_sharp * pair
                step = eta * O
            if instrument_every and t % instrument_every == 0 and (instrument_max is None or len(instr) < instrument_max):
                st = instrument(prob, W, G=Dir, O=O, eta=None if euclid else eta)
                st.update(t=t, grad_evals=ge)
                instr.append(st)
            W_ref = W.copy() if (not is_saga and coins[k] < p) else None
            if use_bb:
                dW = -step
                if exact_polar:                                   # ||polar(D)||_op = 1 unless D = 0
                    dW_op2 = eta * eta if np.any(O) else 0.0
                else:
                    dW_op2 = spectral_norm(dW) ** 2
                if not same_sample:
                    omega_prev = omega
            W -= step
            # ---- memory / reference update
            if is_saga:
                outer *= 1.0 / n
                zbar += outer
                rtab[j] = rj
            elif W_ref is not None:
                Rw = prob.loss_residuals(W_ref)
                gbar = Rw.T @ X / n
                ge += n
                refreshes += 1
            t += 1
            ge += per_iter
            if is_saga and t % resync_every == 0:
                zbar = rtab.T @ X / n
            if t % mon.record_every == 0 and mon.record(t, ge, W, eta=eta, pair=pair, L_hat=L_hat):
                stop = True
                break
        if stop:
            break
    mon.finish(t, ge, W, eta=eta, pair=pair, L_hat=L_hat)
    meta = dict(estimator=estimator, rule=rule, gamma=gamma, g_sharp=g_sharp, lr=lr, orth=None if euclid else orth,
                seed=seed, beta=beta, nesterov=nesterov, per_iter=per_iter, instr=instr, n=n)
    if is_saga:
        meta["zbar_err"] = float(np.linalg.norm(zbar - rtab.T @ X / n))
    else:
        meta.update(p=p, refreshes=refreshes)
    if use_bb:
        meta.update(bb_secant=bb_secant, bb_window=1.0 / a_bb, bb_safety=bb_safety, L_min=L_min, L_max=L_max)
    if rule == "decay":
        meta.update(decay_T0=T0, decay_power=decay_power)
    return mon.result(W, **meta)
