"""Generalized variance-reduced Muon: estimator x placement of the nonlinear map x step-length rule.

Every variance-reduced estimator of grad F(W) used here splits, given the past, into a PREDICTABLE part m (known before
the index is drawn) and an INNOVATION c (depends on the sampled batch B, |B| = b):

    estimator   m (predictable)                 c (innovation)                               E[c | past]
    'saga'      zbar + lam W                    (1/b) sum_{j in B} (r_j(W) - rtab_j) x_j^T   grad F(W) - m
    'lsvrg'     gbar + lam W                    (1/b) sum_{j in B} (r_j(W) - Rw_j) x_j^T     grad F(W) - m
    'page'      v_{t-1} + lam W                 (1/b) sum_{j in B} (r_j(W_t) - r_j(W_{t-1})) x_j^T   (recursive, biased)
    'sgd'       lam W                           (1/b) sum_{j in B} r_j(W) x_j^T              grad F(W) - m

(rtab: SAGA table of loss residuals, zbar = rtab^T X / n; Rw: residuals at the L-SVRG reference point w,
gbar = Rw^T X / n; v: PAGE / loopless-SARAH estimate of the loss gradient, reset to the exact loss gradient with
probability p.) The regularizer gradient lam W is always exact. For b = 1 the innovation is RANK ONE while m is a
full-rank matrix: the memory of the variance-reduced method is the only source of spectral structure at batch size one
(the sharp map is the identity on rank-one matrices, see maps.py).

The damped estimator is omega_theta = m + theta c: theta = 1 is the unbiased estimator (SAGA, L-SVRG), theta = 1/n with
the SAGA memory is the SAG direction, theta = 0 the frozen predictable part.

PLACEMENT of the nonlinear step map T (defined below) relative to the innovation:
    'joint'    S = T(omega_theta)                                    map of the corrected estimator (saga_muon, lsvrg_muon)
    'split'    S = T(m) + gamma_c theta c                            map of the predictable part, innovation added linearly
    'tangent'  S = T(m) + DT(m)[theta c]                             first-order expansion of T at the predictable part
    'radial'   S = gamma_c omega_theta + gamma <omega_theta, O_m> O_m    O_m = polar(m): a predictable direction with an
                                                                     unbiased length (E[S] = gamma_c grad F +
                                                                     gamma <grad F, O_m> O_m for theta = 1)
    'metric'   S = gamma_P omega_theta C                             linear map by a frozen "polar metric" C (below)
    'euclid'   S = gamma omega_theta                                 no map (SAGA, L-SVRG, SAG-like, PAGE, SGD)
and W <- W - S. For 'split' and 'tangent' the conditional expectation of the step is explicit:
E[S] = T(m) + gamma_c theta (grad F(W) - m), resp. T(m) + DT(m)[theta (grad F(W) - m)] (= T(grad F(W)) up to second
order in grad F(W) - m when theta = 1), whereas E[T(omega)] has no closed form.
With `lazy=True` (L-SVRG, 'split' or 'tangent') the map is evaluated at the reference point only, T(grad F(w)), once
per refresh, and the innovation becomes c + lam (W - w): no factorization inside the inner loop (constant learning
rate only: schedules are refused).
Options that a placement does not use are refused rather than ignored: 'radial' always uses the exact polar factor
of m and the signed pairing, 'metric' the step c_metric / R2P; `eps_innov` and the momentum before the map (`beta`,
`nesterov`) exist for the joint placement only; `schedule` and `warmup` for the fixed-length rule only.

POLAR METRIC ('metric' placement). Let D be an anchor matrix (L-SVRG: the exact gradient at the reference point,
refreshed with it; SAGA: the predictable part m, refreshed every `metric_every` iterations, default n) with thin SVD
D = U diag(s) V^T, nu = ||D||_*, r = min(K, d) (K - 1 in the softmax gauge) and floored values
sh_k = max(s_k, metric_floor * s_1). The feature-side metric
    C = V diag(nu / sh_k) V^T + r (I_d - V V^T)
satisfies D C = nu U V^T = sharp(D) when no floor is active: the linear map omega -> omega C reproduces the
nuclear-scaled polar step at the anchor and is LINEAR in the estimator elsewhere, so that E[S] = gamma_P grad F(W) C
exactly: it is a variance-reduced method preconditioned by a frozen metric (the setting of the note B1 of the main
project, here with the loss-part memory; when C is never refreshed, i.e. `metric_every` larger than the run, the rate
of Proposition 5.9 holds in the metric: fact F11 of notes_theory.md). The step is gamma_P = c_metric / R2P with the
per-sample smoothness in the metric
R2P = max_i c_l x_i^T C x_i + lam ||C||_op, recomputed at each refresh of C. For least squares with K = d, an anchor
at a point where W - W_* is a multiple of an orthogonal matrix (problems.make_matrix_ls(planted='orthogonal') at
W_0 = 0, up to the label noise) and no active floor, C is proportional to the inverse Hessian. This does NOT extend to
later anchors (the gradient is small in the directions already converged, which then receive the largest gains) nor
to an active floor; with metric_floor = 1 the metric is isotropic on the row space of the anchor (see
`polar_metric`): the placement then contains Euclidean SAGA / L-SVRG.
With `metric_source='cov'` the factors of the metric are those of the model Hessian on the feature side,
H = c_l Sigma_hat + lam I = V diag(h) V^T, instead of those of an anchor gradient:
C = V diag(tr(H) / max(h_k, metric_floor h_1)) V^T, a fixed metric that uses the second moments of the features and no
gradient (the statistical preconditioner of the note B1; the Newton metric for least squares when metric_floor = 0).
It is the control against which the polar metric is compared.

ANCHOR STEP (`anchor_step` = c_a > 0, L-SVRG estimator): right after each refresh of the reference point w (and at
W_0), the iterate is set to w - (c_a / L_op) sharp(grad F(w)): a PolarGrad step that uses the exact gradient computed
by the refresh. The nonlinear map is then applied to exact gradients only; for c_a <= 1 the step decreases F by at
least c_a (1 - c_a / 2) ||grad F(w)||_*^2 / L_op.

STEP MAP T(M) = eta(M) O(M), O = orth(M):
    orth     'svd' exact polar factor; 'ns5', 'ns_cubic' (polar.py); 'soft': U diag(s / sqrt(s^2 + eps^2)) V^T with
             eps = `eps` (absolute) or eps = eps_innov * ||theta c||_F (relative to the innovation of the current step:
             singular directions of the estimator below the innovation level are not inflated to one)
    length   'sharp'  eta = gamma * pair                      (nuclear-norm scaling; pair = <M, O> = ||M||_* for 'svd')
             'const'  eta = lr_t * scale                      (fixed-length LMO step of Muon; lr_t follows `schedule`,
                                                               scale = max(1, K/d)^(1/2) as in muon_sgd)
             'clip'   eta = min(rho, gamma * pair)            (generalized gradient norm clipping: fixed length rho
                                                               far from the optimum, sharp step near it)
    pair_with (joint placement) 'omega': pair = <omega, O> (default; saga_muon's rule); 'dir': pair = <D, O> with D the
             argument of the map (differs from 'omega' only with momentum, D being then the EMA-normalized buffer);
             'mean': pair = max(<m, O>, 0), the pairing of the direction with the predictable part (a low-noise proxy of
             ||grad F||_*).
Momentum: `beta` (before the map, joint placement only): buf <- beta buf + omega, D = omega + beta buf (nesterov) or
buf; `beta_post` (after the map, any placement): the applied step is the EMA Sbar <- beta_post Sbar + (1 - beta_post) S.

Defaults: gamma = 1/(4 R2_op) ('saga', 'sgd'), 1/(6 R2_op) ('lsvrg', 'page') for the mapped placements and the same
with R2 for 'euclid'; gamma_c = 1/(4 R2) resp. 1/(6 R2); p = b/n (L-SVRG: one refresh per pass in expectation),
b/(n + b) (PAGE). With these defaults vr_muon(estimator='saga') reproduces saga_muon(rule='sharp'),
vr_muon(estimator='saga', placement='euclid') reproduces saga, and likewise for L-SVRG (same index streams for b = 1).

Relaxed memory (`mem_relax` = delta in (0, 1], SAGA memory only): the sampled entries of the table are moved by a
fraction delta towards the fresh gradients, z_j <- z_j + delta (g_j'(W) - z_j), instead of being replaced. The
estimator stays unbiased for any table, and the recursion of the predictable part becomes
m_{t+1} = m_t - lam S_t + delta (b/n) c_t. The Euclidean member with theta = delta b/n and gamma = 1/lam keeps m = 0:
it is dual-free SDCA with step delta / (lam n) and uniform sampling (Shalev-Shwartz, "SDCA without Duality",
arXiv:1502.06177, Theorem 1: step <= 1 / (L + lam n), i.e. delta <= lam n / (R2 + lam n) = 1 / (1 + kappa/n),
the value used by mem_relax='sdca'; the ICML 2016 version of the paper has the more restrictive condition
step <= min(1/(4L), 1/(4 lam n))); delta = 1 is MISO with lower-bound surrogates (Mairal 2015; the Finito update
with alpha = 1), guaranteed for n >= 2 kappa only.

Importance sampling (`sampling` = a > 0, SAGA and L-SVRG at batch 1): the index is drawn with probability
p_i proportional to L_i^a (L_i = c_l ||x_i||^2 + lam, the per-sample smoothness) and the innovation is weighted by
1 / (n p_i), which keeps the estimator unbiased, E[c | past] = grad F(W) - m; the table and its mean are updated with
the unweighted gradient. a = 1 is sampling proportional to the smoothness constants (Xiao and Zhang 2014; Schmidt et
al. 2015), a = 1/2 sampling proportional to the norm of the features, which is the variance-optimal choice for
gradients of norm proportional to ||x_i||. Entry i of the SAGA table is then refreshed with probability p_i.

Softmax gauge: with gauge='auto' (default) every factorization is done in the subspace orthogonal to the all-ones
vector (maps.gauge_basis), which requires 1^T W_0 = 0 (checked by `init_state` for the mapped placements).

Gradient accounting (as algorithms.py): initial full gradient n (0 for 'sgd'); per iteration b ('saga', 'sgd'),
2b ('lsvrg', and 'page' without refresh), n per refresh of the L-SVRG reference or of the PAGE estimate.

The class `VRMuon` exposes the three stages of one iteration, `pieces` (m, c), `step` (S; no side effect) and `commit`
(memory update), so that the exact conditional expectations of exact.py enumerate the very steps taken by the loop.
"""
from __future__ import annotations

import numpy as np

from . import maps
from .algorithms import _Monitor, _default_record_every, _index_chunks, lr_factors
from .instrumentation import instrument
from .polar import get_orth, svd_thin

ESTIMATORS = ("saga", "lsvrg", "page", "sgd")
PLACEMENTS = ("joint", "split", "tangent", "radial", "metric", "euclid")
LENGTHS = ("sharp", "const", "clip")
ORTHS = ("svd", "soft", "ns5", "ns_cubic")
PAIRINGS = ("omega", "dir", "mean")


class _State:
    """Mutable state of a run (memory tables, reference point, momentum buffers, counters)."""


def residual_diff_rows(prob, Z, Delta):
    """Rows r_i(z_i) - r_i(z_i - delta_i) for the rows of Z and Delta, without cancellation for small delta (the
    labels cancel): Delta for least squares; for softmax, with p = softmax(z) and A = sum_l p_l exp(-delta_l),
    p_k exp(-delta_k) sum_l p_l expm1(delta_k - delta_l) / A (see SoftmaxProblem.residual_diff_i)."""
    if prob.loss == "ls":
        return np.array(Delta, dtype=np.float64, copy=True)
    P = np.exp(Z - Z.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)
    E = np.exp(-Delta)
    A = (P * E).sum(axis=1, keepdims=True)
    inner = np.einsum("bkl,bl->bk", np.expm1(Delta[:, :, None] - Delta[:, None, :]), P)
    return P * E * inner / A


class VRMuon:
    """One configuration of the generalized variance-reduced Muon (see the module docstring)."""

    def __init__(self, prob, estimator="saga", batch=1, theta=1.0, placement="joint", length="sharp",
                 gamma=None, gamma_c=None, lr=None, rho=None, orth="svd", eps=None, eps_innov=None, orth_kw=None,
                 pair_with="omega", beta=0.0, nesterov=False, beta_post=0.0, p=None, lazy=False, gauge="auto",
                 schedule="const", n_iters=None, warmup=0.0, decay_frac=0.2, resync_every=None, mem_relax=1.0,
                 anchor_step=0.0, c_metric=None, metric_floor=0.1, metric_every=None, sampling=None,
                 metric_source="polar"):
        if estimator not in ESTIMATORS:
            raise ValueError(f"unknown estimator {estimator!r}")
        if placement not in PLACEMENTS:
            raise ValueError(f"unknown placement {placement!r}")
        if length not in LENGTHS:
            raise ValueError(f"unknown length rule {length!r}")
        if pair_with not in PAIRINGS:
            raise ValueError(f"unknown pair_with {pair_with!r}")
        if not (isinstance(orth, str) and orth in ORTHS) and not callable(orth):
            raise ValueError(f"unknown orth {orth!r}")
        mapped = placement != "euclid"
        # placements whose step goes through the configurable map T (orth, length rule, pairing)
        uses_map = placement in ("joint", "split", "tangent")
        if uses_map and length == "const" and lr is None:
            raise ValueError("length 'const' needs lr")
        if uses_map and length == "clip" and rho is None:
            raise ValueError("length 'clip' needs rho")
        if not uses_map:
            # 'radial' always uses the exact polar factor of m with the signed pairing, 'metric' the frozen metric
            # with the step c_metric / R2P, 'euclid' no map: refuse options that would be silently ignored
            ignored = [k for k, v, default in (("length", length, "sharp"), ("orth", orth, "svd"),
                                               ("pair_with", pair_with, "omega"), ("lr", lr, None),
                                               ("rho", rho, None)) if v != default]
            if placement == "metric" and gamma is not None:
                ignored.append("gamma (use c_metric)")
            if ignored:
                raise ValueError(f"placement {placement!r} does not use: {', '.join(ignored)}")
        if orth == "soft":
            if (eps is None) == (eps_innov is None):
                raise ValueError("orth 'soft' needs exactly one of eps (absolute) and eps_innov (relative to the "
                                 "innovation)")
            if not (eps if eps is not None else eps_innov) > 0:
                raise ValueError("orth 'soft' needs eps > 0 (use orth 'svd' for the exact polar factor)")
            if eps_innov is not None and placement != "joint":
                raise ValueError("eps_innov (bounded gain relative to the innovation) is defined for the joint "
                                 "placement; use an absolute eps")
        elif eps is not None or eps_innov is not None:
            raise ValueError("eps and eps_innov are parameters of orth 'soft'")
        if placement != "joint" and (beta or nesterov):
            raise ValueError("momentum before the map (beta, nesterov) is only defined for the joint placement; use "
                             "beta_post")
        if not (0.0 <= beta < 1.0 and 0.0 <= beta_post < 1.0):
            raise ValueError("beta and beta_post must be in [0, 1)")
        if placement == "tangent" and not (orth in ("svd", "soft")):
            raise ValueError("the tangent placement needs orth in {'svd', 'soft'}")
        if lazy and not (estimator == "lsvrg" and placement in ("split", "tangent")):
            raise ValueError("lazy=True is defined for estimator 'lsvrg' with placement 'split' or 'tangent'")
        scheduled = not (isinstance(schedule, str) and schedule == "const") or bool(warmup)
        if scheduled and not (uses_map and length == "const"):
            raise ValueError("schedule and warmup apply to the fixed-length rule (length 'const') only")
        if scheduled and lazy:
            raise ValueError("lazy=True caches the map of the reference point: it needs a constant learning rate "
                             "(no schedule, no warmup)")
        self.prob = prob
        n = prob.n
        self.n, self.lam, self.X = n, prob.lam, prob.X
        self.estimator, self.batch, self.theta = estimator, int(batch), float(theta)
        if not 1 <= self.batch <= n:
            raise ValueError("batch must be in [1, n]")
        self.placement, self.length, self.orth, self.pair_with = placement, length, orth, pair_with
        c0 = 4.0 if estimator in ("saga", "sgd") else 6.0
        if gamma is None:
            gamma = 1.0 / (c0 * (prob.R2_op if mapped else prob.R2))
        self.gamma = float(gamma)
        self.gamma_c = 1.0 / (c0 * prob.R2) if gamma_c is None else float(gamma_c)
        self.lr = None if lr is None else float(lr)
        self.rho = None if rho is None else float(rho)
        self.eps = None if eps is None else float(eps)
        self.eps_innov = None if eps_innov is None else float(eps_innov)
        self.beta, self.nesterov, self.beta_post = float(beta), bool(nesterov), float(beta_post)
        b = self.batch
        if p is None:
            p = b / n if estimator == "lsvrg" else b / (n + b)
        self.p = float(p)
        self.lazy = bool(lazy)
        if isinstance(gauge, str):
            if gauge != "auto":
                raise ValueError("gauge must be 'auto', None/False or an orthonormal basis (K x K')")
            self.basis = maps.gauge_basis(prob)
        else:
            self.basis = None if (gauge is None or gauge is False) else np.asarray(gauge, dtype=np.float64)
        self._orth_fn = None
        if callable(orth) or orth in ("ns5", "ns_cubic"):
            fn = get_orth(orth, **(orth_kw or {}))
            self._orth_fn = (lambda M: fn(prob.remove_gauge(M))) if self.basis is not None else fn
        self.scale = max(1.0, prob.K / prob.d) ** 0.5
        self.schedule = schedule
        self.warmup, self.decay_frac, self.orth_kw = float(warmup), float(decay_frac), dict(orth_kw or {})
        self.sched_iters = None                                       # span of the schedule (None: constant rate)
        self._lrs = None
        if scheduled:
            if n_iters is None or n_iters > 50_000_000:
                raise ValueError("a learning-rate schedule needs the number of iterations n_iters it spans "
                                 "(at most 5e7)")
            self.sched_iters = int(n_iters)
            self._lrs = self.lr * lr_factors(schedule, n_iters, warmup, decay_frac)
        self.resync_every = n if resync_every is None else int(resync_every)
        if isinstance(mem_relax, str):
            if mem_relax != "sdca":
                raise ValueError("mem_relax must be a number in (0, 1] or 'sdca'")
            mem_relax = prob.lam * n / (prob.R2 + prob.lam * n)        # = 1 / (1 + kappa / n) up to the lam in R2
        self.mem_relax = float(mem_relax)
        if not 0.0 < self.mem_relax <= 1.0:
            raise ValueError("mem_relax must be in (0, 1]")
        if self.mem_relax != 1.0 and estimator != "saga":
            raise ValueError("mem_relax is defined for the SAGA memory")
        self.anchor_step = float(anchor_step)
        if self.anchor_step and estimator != "lsvrg":
            raise ValueError("anchor_step is defined for the L-SVRG estimator")
        if placement == "metric" and estimator not in ("saga", "lsvrg"):
            raise ValueError("the metric placement needs the SAGA or L-SVRG estimator")
        if metric_source not in ("polar", "cov"):
            raise ValueError(f"unknown metric_source {metric_source!r}")
        if metric_source != "polar" and placement != "metric":
            raise ValueError("metric_source is an option of the metric placement")
        if metric_source == "cov" and metric_every is not None:
            raise ValueError("the covariance metric is fixed: metric_every does not apply")
        if not 0.0 <= float(metric_floor) <= 1.0:
            raise ValueError("metric_floor must be in [0, 1]")
        self.metric_source = metric_source
        self._cov = None                                              # eigen-factors of the model Hessian ('cov')
        if c_metric is None:
            c_metric = 0.25 if estimator == "saga" else 1.0 / 6.0
        self.c_metric = float(c_metric)
        self.metric_floor = float(metric_floor)
        self.metric_every = n if metric_every is None else int(metric_every)
        if self.resync_every < 1 or self.metric_every < 1:
            raise ValueError("resync_every and metric_every must be >= 1")
        self.needs_coin = estimator in ("lsvrg", "page")
        self.init_cost = 0 if estimator == "sgd" else n
        self.fallbacks = 0
        # importance sampling of the index: p_i proportional to L_i^a, innovation weighted by 1 / (n p_i)
        self.sampling = 0.0 if sampling in (None, "uniform") else float(sampling)
        self.probs = self.iw = None
        if self.sampling:
            if self.batch != 1 or estimator not in ("saga", "lsvrg"):
                raise ValueError("importance sampling is defined for the SAGA and L-SVRG estimators at batch 1")
            if self.sampling < 0:
                raise ValueError("sampling must be None ('uniform') or a nonnegative exponent")
            w = np.asarray(prob.L_i, dtype=np.float64) ** self.sampling
            self.probs = w / w.sum()
            self.iw = 1.0 / (n * self.probs)
        # keys of the per-state caches (the cached quantities depend on these parameters only)
        gauged = self.basis is not None
        self._metric_key = (self.metric_source, self.metric_floor, self.c_metric, gauged)
        self._anchor_key = (placement, length, self.gamma, self.lr, self.rho,
                            orth if isinstance(orth, str) else id(orth), self.eps, pair_with, gauged,
                            tuple(sorted(self.orth_kw.items())))

    # ------------------------------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------------------------------
    def init_state(self, W0):
        prob, n, X = self.prob, self.n, self.X
        if self.basis is not None and self.placement in ("joint", "split", "tangent"):
            # the mapped part of the step lives in span(basis): a component of W_0 outside it would never move
            out = W0 - self.basis @ (self.basis.T @ W0)
            if float(np.linalg.norm(out)) > 1e-9 * max(1.0, float(np.linalg.norm(W0))):
                raise ValueError("W0 has a component outside the gauge subspace (softmax: the columns of W0 must sum "
                                 "to zero); project it (prob.remove_gauge) or pass gauge=None")
        st = _State()
        st.t = 0
        st.refreshes = 0
        st.jump = None                            # iterate imposed by an anchor step (see `advance`)
        st.metric = None                          # frozen polar metric: {'svd': (s, V), key: derived quantities}
        st.buf = np.zeros_like(W0) if self.beta else None
        st.Sbar = np.zeros_like(W0) if self.beta_post else None
        est = self.estimator
        if est == "saga":
            st.rtab = prob.loss_residuals(W0)
            st.zbar = st.rtab.T @ X / n
        elif est == "lsvrg":
            st.Rw = prob.loss_residuals(W0)
            st.gbar = st.Rw.T @ X / n
            st.w = np.array(W0, dtype=np.float64, copy=True)
            st.anchor = None                      # cached maps at the reference point (lazy): {key: tuple}
        elif est == "page":
            st.v = prob.loss_residuals(W0).T @ X / n
            st.W_prev = None                      # None: v is the exact loss gradient at the current point
        return st

    # ------------------------------------------------------------------------------------------
    # stage 1: predictable part and innovation
    # ------------------------------------------------------------------------------------------
    def pieces(self, W, st, B, coin=1.0):
        """Return (m, c, aux) for the batch B (an index if batch = 1, an index array otherwise) and the coin of the
        iteration (L-SVRG: refresh after the step if coin < p; PAGE: exact loss gradient at W if coin < p).
        c is None when the innovation is identically zero (PAGE at a refresh)."""
        prob, X, lam, est, b = self.prob, self.X, self.lam, self.estimator, self.batch
        if est == "page":
            if st.W_prev is None or coin < self.p:
                v = st.v if st.W_prev is None else prob.loss_residuals(W).T @ X / self.n
                return v + lam * W, None, ("full", v)
            dW = W - st.W_prev
            if b == 1:
                xj = X[B]
                dr = prob.residual_diff_i(np.dot(W, xj), np.dot(dW, xj), B)
                c = dr[:, None] * xj
            else:
                Xb = X[B]
                c = residual_diff_rows(prob, Xb @ W.T, Xb @ dW.T).T @ Xb / b
            return st.v + lam * W, c, ("rec", c)
        if b == 1:
            xj = X[B]
            rj = prob.residual_i(np.dot(W, xj), B)
            if est == "saga":
                c = (rj - st.rtab[B])[:, None] * xj
                if self.iw is not None:                      # the table update (aux) uses the unweighted difference
                    return st.zbar + lam * W, self.iw[B] * c, (rj, c)
                return st.zbar + lam * W, c, (rj, c)
            if est == "lsvrg":
                c = (rj - st.Rw[B])[:, None] * xj
                if self.iw is not None:
                    c *= self.iw[B]
                return st.gbar + lam * W, c, None
            return lam * W, rj[:, None] * xj, None                                # sgd
        Xb = X[B]
        Rb = prob._residual_rows(Xb @ W.T, B)
        if est == "saga":
            dR = Rb - st.rtab[B]
            return st.zbar + lam * W, dR.T @ Xb / b, Rb
        if est == "lsvrg":
            return st.gbar + lam * W, (Rb - st.Rw[B]).T @ Xb / b, None
        return lam * W, Rb.T @ Xb / b, None                                        # sgd

    # ------------------------------------------------------------------------------------------
    # stage 2: the step (no side effect)
    # ------------------------------------------------------------------------------------------
    def _lr_t(self, t):
        if self._lrs is None:
            return self.lr
        return float(self._lrs[min(t, len(self._lrs) - 1)])

    def _orthogonalize(self, D, c_norm):
        """Return (O, pair_self) with pair_self = <D, O> (None if not computed)."""
        orth = self.orth
        if orth == "svd":
            _, O, nuc = maps.sharp(D, basis=self.basis, return_parts=True)
            return O, nuc
        if orth == "soft":
            eps = self.eps if self.eps is not None else self.eps_innov * c_norm
            if eps > 0:
                return maps.soft_polar(D, eps, basis=self.basis), None
            _, O, nuc = maps.sharp(D, basis=self.basis, return_parts=True)       # no innovation: exact polar
            return O, nuc
        return self._orth_fn(D), None

    def _eta(self, pair, t):
        if self.length == "sharp":
            return self.gamma * pair
        if self.length == "const":
            return self._lr_t(t) * self.scale
        return min(self.rho, self.gamma * pair)

    def _map(self, D, t, c_norm=0.0, omega=None, m=None):
        """T(D) = eta O; returns (S, eta, pair)."""
        O, pair_self = self._orthogonalize(D, c_norm)
        pw = self.pair_with
        if omega is None or pw == "dir":
            pair = float(np.vdot(D, O)) if pair_self is None else pair_self
        elif pw == "omega":
            pair = pair_self if (pair_self is not None and omega is D) else float(np.vdot(omega, O))
        else:
            pair = max(float(np.vdot(m, O)), 0.0)
        eta = self._eta(pair, t)
        return eta * O, eta, pair

    def _map_derivative(self, m, t):
        """(T(m), DT(m)) for the tangent placement; DT is a function E -> matrix. Raises LinAlgError if m is rank
        deficient (exact polar)."""
        if self.orth == "soft":
            SD = maps.SpectralDerivative(m, "soft", eps=self.eps, basis=self.basis)
            O = SD.value
            pair = float(np.vdot(m, O))
            if self.length == "const":
                lr = self._lr_t(t) * self.scale
                return lr * O, (lambda E: lr * SD(E)), lr, pair
            g = self.gamma
            if self.length == "clip" and g * pair >= self.rho:
                rho = self.rho
                return rho * O, (lambda E: rho * SD(E)), rho, pair
            # eta(M) O(M) with eta = gamma <M, O(M)>:  D[eta O][E] = gamma (<E, O> + <m, DO[E]>) O + eta DO[E]
            eta = g * pair

            def DT(E, SD=SD, O=O, m=m, g=g, eta=eta):
                DO = SD(E)
                return g * (float(np.vdot(E, O)) + float(np.vdot(m, DO))) * O + eta * DO
            return eta * O, DT, eta, pair
        if self.length == "const":
            SD = maps.SpectralDerivative(m, "polar", basis=self.basis)
            lr = self._lr_t(t) * self.scale
            return lr * SD.value, (lambda E: lr * SD(E)), lr, SD.nuc
        SD = maps.SpectralDerivative(m, "sharp", basis=self.basis)
        g = self.gamma
        if self.length == "clip" and g * SD.nuc >= self.rho:
            SDp = maps.SpectralDerivative(m, "polar", basis=self.basis)
            rho = self.rho
            return rho * SDp.value, (lambda E: rho * SDp(E)), rho, SD.nuc
        return g * SD.value, (lambda E: g * SD(E)), g * SD.nuc, SD.nuc

    def step(self, W, m, c, st, t=0, cache=None):
        """The step S of the iteration (W <- W - S), a function of (W, m, c, state). Returns (S, info) with
        info = {'eta', 'pair', 'buf' (new momentum buffer or None), 'Sbar' (new post-map EMA or None)}.
        The memory, the buffers and the counters of the state are not modified; the only writes are the caches
        st.metric ('metric' placement) and st.anchor (lazy placements), which are filled on demand, keyed by the
        parameters they depend on and invalidated by `commit`, so that another configuration evaluated at the same
        state never reads them by mistake.
        `cache` (a dict) may be passed to reuse the map of the predictable part across calls with the same m
        ('split', 'tangent' and 'radial' placements; used by the exact enumeration)."""
        th, pl = self.theta, self.placement
        info = {"eta": float("nan"), "pair": float("nan"), "buf": None, "Sbar": None}
        if pl == "euclid":
            S = self.gamma * (m if c is None else m + th * c)
        elif pl == "joint":
            omega = m if c is None else m + th * c
            D = omega
            if self.beta:
                buf = self.beta * st.buf + omega
                info["buf"] = buf
                D = (1.0 - self.beta) * (omega + self.beta * buf if self.nesterov else buf)
            c_norm = 0.0
            if self.eps_innov is not None and c is not None:
                c_norm = abs(th) * float(np.sqrt(np.vdot(c, c)))
            S, info["eta"], info["pair"] = self._map(D, t, c_norm, omega=omega, m=m)
        elif pl == "radial":
            if cache is not None and "radial" in cache:
                O, nuc = cache["radial"]
            else:
                _, O, nuc = maps.sharp(m, basis=self.basis, return_parts=True)
                if cache is not None:
                    cache["radial"] = (O, nuc)
            pair = nuc if c is None else nuc + th * float(np.vdot(c, O))        # <omega_theta, O_m>, signed
            S = self.gamma * pair * O
            S += self.gamma_c * (m if c is None else m + th * c)
            info["eta"], info["pair"] = self.gamma * pair, pair
        elif pl == "metric":
            V, gains, r, g_P, _ = self._metric(st, m)
            omega = m if c is None else m + th * c
            S = np.dot(np.dot(omega, V) * (gains - r), V.T)
            S += r * omega
            S *= g_P
            info["eta"] = g_P
        else:
            lazy = self.lazy
            corr = None
            if lazy:
                corr = self.lam * (W - st.w)
                if c is not None:
                    corr = corr + th * c
            elif c is not None:
                corr = th * c
            if pl == "split":
                if lazy:
                    S0, _, eta, pair = self._anchor(st, t)
                elif cache is not None and "split" in cache:
                    S0, eta, pair = cache["split"]
                else:
                    S0, eta, pair = self._map(m, t)
                    if cache is not None:
                        cache["split"] = (S0, eta, pair)
                S = S0 if corr is None else S0 + self.gamma_c * corr
            elif lazy:                                           # lazy tangent (the fallback is handled by _anchor)
                S0, DT, eta, pair = self._anchor(st, t)
                S = S0 if corr is None else S0 + DT(corr)
            else:
                try:
                    if cache is not None and "tangent" in cache:
                        S0, DT, eta, pair = cache["tangent"]
                    else:
                        S0, DT, eta, pair = self._map_derivative(m, t)
                        if cache is not None:
                            cache["tangent"] = (S0, DT, eta, pair)
                    S = S0 if corr is None else S0 + DT(corr)
                except np.linalg.LinAlgError:                    # rank-deficient predictable part: split step
                    self.fallbacks += 1
                    S0, eta, pair = self._map(m, t)
                    S = S0 if corr is None else S0 + self.gamma_c * corr
            info["eta"], info["pair"] = eta, pair
        if self.beta_post:
            S = self.beta_post * st.Sbar + (1.0 - self.beta_post) * S
            info["Sbar"] = S
        return S, info

    def anchor_svd(self, D):
        """(s, V): singular values and right singular vectors of the anchor matrix D (reduced to the gauge subspace)."""
        R = D if self.basis is None else self.basis.T @ D
        _, s, Vt = svd_thin(R)
        return s, np.ascontiguousarray(Vt.T)

    def covariance_factors(self):
        """(h, V): eigenvalues (decreasing) and eigenvectors of the model Hessian c_l Sigma_hat + lam I on the feature
        side, the factors of the 'cov' metric C = V diag(tr(H) / max(h_k, metric_floor h_1)) V^T. They do not depend
        on the iterate and are computed once."""
        if self._cov is None:
            prob = self.prob
            h, V = np.linalg.eigh(prob.c_loss * prob.Sigma_hat + prob.lam * np.eye(prob.d))
            self._cov = (np.ascontiguousarray(h[::-1]), np.ascontiguousarray(V[:, ::-1]))
        return self._cov

    def polar_metric(self, D=None, svd=None):
        """Frozen metric of the anchor matrix D (or of its factors svd = (s, V)): returns (V, gains, r, gamma_P, R2P)
        with C = V diag(gains) V^T + r (I - V V^T), gains_k = ||D||_* / max(s_k, metric_floor s_1),
        R2P = max_i c_l x_i^T C x_i + lam ||C||_op and gamma_P = c_metric / R2P. A zero anchor gives the Euclidean
        metric (C = I). With metric_floor = 1 all gains equal ||D||_* / s_1: C is a multiple of the identity as soon
        as the reduced anchor has d rows or more (the method is then Euclidean SAGA / L-SVRG with the step
        c_metric / R2), a two-level scaling (row space of the anchor / its complement) otherwise."""
        prob = self.prob
        s, V = self.anchor_svd(D) if svd is None else svd
        r = len(s)
        if not (s.size and s[0] > 0):
            V = np.zeros((prob.d, 0))
            gains = np.zeros(0)
            r_eff, R2P = 1.0, prob.R2
            return V, gains, r_eff, self.c_metric / R2P, R2P
        gains = float(s.sum()) / np.maximum(s, self.metric_floor * s[0])
        XV = prob.X @ V
        quad = r * prob.sqnorms + np.einsum("ik,ik->i", XV * (gains - r), XV)          # x_i^T C x_i
        c_op = float(max(gains.max(), r if r < prob.d else 0.0))
        R2P = float(prob.c_loss * quad.max() + prob.lam * c_op)
        return V, gains, float(r), self.c_metric / R2P, R2P

    def _metric(self, st, m):
        """Current frozen metric. The factorization of the anchor (L-SVRG: exact gradient at the reference point;
        SAGA: the predictable part at the first step after an invalidation) is computed on demand and kept in the
        state until `commit` invalidates it; the quantities derived from it are cached under the key
        (metric_source, metric_floor, c_metric, gauge) of the configuration. The covariance metric
        (metric_source='cov') is fixed and never invalidated."""
        gauged = self.basis is not None
        src = self.metric_source
        if st.metric is None or st.metric["gauged"] != gauged or st.metric.get("source", "polar") != src:
            if src == "cov":
                factors = self.covariance_factors()
            else:
                factors = self.anchor_svd((st.gbar + self.lam * st.w) if self.estimator == "lsvrg" else m)
            st.metric = {"svd": factors, "gauged": gauged, "source": src}
        key = self._metric_key
        if key not in st.metric:
            st.metric[key] = self.polar_metric(svd=st.metric["svd"])
        return st.metric[key]

    def anchor_displacement(self, st):
        """(c_a / L_op) sharp(grad F(w)) at the L-SVRG reference point w."""
        G_w = st.gbar + self.lam * st.w
        return (self.anchor_step / self.prob.L_op) * maps.sharp(G_w, basis=self.basis)

    def advance(self, W, S, st):
        """New iterate after the step S: W - S, or the iterate imposed by an anchor step at a refresh."""
        if st.jump is not None:
            W_new, st.jump = st.jump, None
            return W_new
        return W - S

    def _anchor(self, st, t):
        """Cached map at the L-SVRG reference point, m_w = gbar + lam w: (T(m_w), DT, eta, pair), with DT the
        derivative E -> DT(m_w)[E] for the tangent placement (None for the split placement). If m_w is rank deficient
        the derivative does not exist: the tangent step falls back, for the whole reference period, to the lazy
        split step T(m_w) + gamma_c (.), counted once in `fallbacks`. The cache is kept in the state under the key of
        the configuration (placement, length rule, steps, orthogonalization) until the next refresh."""
        if st.anchor is None:
            st.anchor = {}
        key = self._anchor_key
        if key not in st.anchor:
            m_w = st.gbar + self.lam * st.w
            if self.placement == "split":
                S0, eta, pair = self._map(m_w, t)
                st.anchor[key] = (S0, None, eta, pair)
            else:
                try:
                    st.anchor[key] = self._map_derivative(m_w, t)
                except np.linalg.LinAlgError:
                    self.fallbacks += 1
                    S0, eta, pair = self._map(m_w, t)
                    g_c = self.gamma_c
                    st.anchor[key] = (S0, (lambda E: g_c * E), eta, pair)
        return st.anchor[key]

    # ------------------------------------------------------------------------------------------
    # stage 3: memory update (at the pre-step point W)
    # ------------------------------------------------------------------------------------------
    def commit(self, W, st, B, coin, aux, info):
        """Update the memory with the quantities computed at the pre-step point W; returns the number of per-sample
        gradient evaluations charged to the iteration."""
        est, b, n = self.estimator, self.batch, self.n
        if info["buf"] is not None:
            st.buf = info["buf"]
        if info["Sbar"] is not None:
            st.Sbar = info["Sbar"]
        st.t += 1
        if est == "saga":
            dl = self.mem_relax
            if b == 1:
                rj, c = aux
                if dl == 1.0:
                    st.zbar += c * (1.0 / n)
                    st.rtab[B] = rj
                else:                                        # relaxed replacement: z_j <- z_j + delta (g_j'(W) - z_j)
                    st.zbar += c * (dl / n)
                    st.rtab[B] += dl * (rj - st.rtab[B])
            else:
                Xb = self.X[B]
                dR = aux - st.rtab[B]
                if dl == 1.0:
                    st.zbar += dR.T @ Xb / n
                    st.rtab[B] = aux
                else:
                    st.zbar += dR.T @ Xb * (dl / n)
                    st.rtab[B] += dl * dR
            if st.t % self.resync_every == 0:
                st.zbar = st.rtab.T @ self.X / n
            if self.placement == "metric" and self.metric_source == "polar" and st.t % self.metric_every == 0:
                st.metric = None                              # recomputed from the predictable part at the next step
            return b
        if est == "lsvrg":
            cost = 2 * b
            if coin < self.p:
                st.w = np.array(W, dtype=np.float64, copy=True)
                st.Rw = self.prob.loss_residuals(st.w)
                st.gbar = st.Rw.T @ self.X / n
                st.anchor = None
                if self.metric_source == "polar":
                    st.metric = None
                st.refreshes += 1
                cost += n
                if self.anchor_step:
                    st.jump = st.w - self.anchor_displacement(st)
            return cost
        if est == "page":
            kind, v = aux
            first = st.W_prev is None
            st.W_prev = np.array(W, dtype=np.float64, copy=True)
            if kind == "full":
                st.v = v
                if first:
                    return 0                                    # the exact gradient at W_0 is the initial cost
                st.refreshes += 1
                return n
            st.v = st.v + v                              # v_t = v_{t-1} + c (undamped recursion)
            return 2 * b
        return b                                                # sgd

    # ------------------------------------------------------------------------------------------
    def memory_error(self, st):
        """Mean squared Frobenius error of the stored loss gradients with respect to their values at W_*:
        (1/n) sum_i ||z_i - g_i'(W_*)||_F^2 for the SAGA table, the same with g_i'(w) for the L-SVRG reference point
        (rank-one matrices: ||(r_i - r_i^*) x_i^T||_F^2 = ||r_i - r_i^*||^2 ||x_i||^2); NaN for the other estimators.
        Whatever the step rule, the SAGA memory (mem_relax = 1) obeys
        E[M_{t+1} | past] = (1 - b/n) M_t + (b/n) S_t >= (1 - b/n) M_t,
        with S_t = (1/n) sum_i ||g_i'(W_t) - g_i'(W_*)||_F^2. With a relaxed table (mem_relax = delta < 1) convexity
        gives E[M_{t+1} | past] <= (1 - delta b/n) M_t + delta (b/n) S_t, and the floor becomes
        E[M_{t+1} | past] >= (1 - (b/n) delta (2 - delta)) M_t - 2 (b/n) delta (1 - delta) (M_t S_t)^(1/2)."""
        prob = self.prob
        if self.estimator == "saga":
            R = st.rtab
        elif self.estimator == "lsvrg":
            R = st.Rw
        else:
            return float("nan")
        dR = R - prob.R_star
        return float((np.einsum("ij,ij->i", dR, dR) * prob.sqnorms).mean())

    def describe(self):
        d = dict(estimator=self.estimator, batch=self.batch, theta=self.theta, placement=self.placement,
                 length=self.length, gamma=self.gamma, gamma_c=self.gamma_c, lr=self.lr, rho=self.rho,
                 orth=self.orth if isinstance(self.orth, str) else "custom", eps=self.eps, eps_innov=self.eps_innov,
                 pair_with=self.pair_with, beta=self.beta, nesterov=self.nesterov, beta_post=self.beta_post,
                 p=self.p, lazy=self.lazy, schedule=self.schedule if isinstance(self.schedule, str) else "custom",
                 gauge=self.basis is not None, mem_relax=self.mem_relax, anchor_step=self.anchor_step,
                 c_metric=self.c_metric, metric_floor=self.metric_floor, resync_every=self.resync_every,
                 metric_every=self.metric_every, warmup=self.warmup, decay_frac=self.decay_frac,
                 orth_kw=dict(self.orth_kw), sched_iters=self.sched_iters, sampling=self.sampling,
                 metric_source=self.metric_source)
        return d


def _draws(rng, n, b, n_iters, need_coin, probs=None):
    """Index (b = 1) or index array (b > 1, a uniform subset without replacement) and coin of every iteration. For
    b = 1 the streams are those of algorithms._vr_run (indices in chunks of min(n_iters, 10^6), then one coin per
    index of the chunk). Consequence for the estimators with a coin (L-SVRG, PAGE) at b = 1: the realization depends
    on n_iters through the chunk size, so two runs with the same seed and different n_iters are not prefixes of one
    another (SAGA, SGD and b > 1 are prefix-consistent). With `probs` (importance sampling, b = 1) the indices are
    drawn from that distribution, in the same chunks."""
    if b == 1 and probs is not None:
        done = 0
        while done < n_iters:
            k = min(1_000_000, n_iters - done)
            idx = rng.choice(n, size=k, p=probs)
            coins = rng.random(k) if need_coin else None
            for q in range(k):
                yield idx[q], (coins[q] if need_coin else 1.0)
            done += k
    elif b == 1:
        for idx in _index_chunks(rng, n, n_iters):
            coins = rng.random(len(idx)) if need_coin else None
            for k in range(len(idx)):
                yield idx[k], (coins[k] if need_coin else 1.0)
    else:
        for _ in range(n_iters):
            B = rng.choice(n, size=b, replace=False, shuffle=False)
            yield B, (rng.random() if need_coin else 1.0)


def default_name(cfg):
    est = {"saga": "SAGA", "lsvrg": "L-SVRG", "page": "PAGE", "sgd": "SGD"}[cfg["estimator"]]
    if cfg["placement"] == "euclid":
        tag = est
    else:
        tag = f"{est}-Muon[{cfg['placement']},{cfg['length']},{cfg['orth']}]"
    if cfg["placement"] in ("radial", "metric"):                  # these placements have no length rule / orth option
        tag = f"{est}-Muon[{cfg['placement']}]"
    if cfg["placement"] == "metric" and cfg.get("metric_source", "polar") == "cov":
        tag = f"{est}[cov-metric]"
    extra = []
    if cfg["theta"] != 1.0:
        extra.append(f"theta={cfg['theta']:g}")
    if cfg["batch"] != 1:
        extra.append(f"b={cfg['batch']}")
    if cfg["lazy"]:
        extra.append("lazy")
    if cfg.get("anchor_step"):
        extra.append(f"anchor={cfg['anchor_step']:g}")
    for k, default in (("beta", 0.0), ("beta_post", 0.0), ("pair_with", "omega"), ("mem_relax", 1.0),
                       ("eps", None), ("eps_innov", None), ("sampling", 0.0)):
        if cfg.get(k, default) != default:
            v = cfg[k]
            extra.append(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}")
    return tag + ("(" + ",".join(extra) + ")" if extra else "")


def vr_muon(prob, W0, n_iters, seed=0, record_every=None, stop_Fgap=None, stop_dist2=None, max_time=None,
            max_grad_evals=None, instrument_every=None, instrument_max=None, name=None, **config):
    """Run one configuration (keyword arguments of `VRMuon`) for at most n_iters iterations (or until
    `max_grad_evals` per-sample gradient evaluations are spent). Returns the dict of algorithms._Monitor (traces
    iters, grad_evals, Fgap, dist2_F, dist2_op, eta, pair, mem (VRMuon.memory_error); final W; 'stopped'; timings)
    plus the configuration,
    'refreshes', 'fallbacks' (tangent steps replaced by split steps because the predictable part was rank deficient)
    and, for SAGA, 'zbar_err'.

    Conventions. The budget test follows the iteration: the last iteration may exceed `max_grad_evals` by its own
    cost (at most one refresh, n + 2b), and one iteration is run even if the budget is below the initial cost. With
    `anchor_step` the record of iteration 0 is taken at W_0, before the (free) anchor step at W_0, whose effect is
    seen from the next record on. A run whose iterate or estimator becomes non-finite ends with
    stopped = 'diverged' (the last record then holds NaN)."""
    algo = VRMuon(prob, n_iters=n_iters, **config)
    cfg = algo.describe()
    W = np.array(W0, dtype=np.float64, copy=True)
    st = algo.init_state(W)
    rng = np.random.default_rng(seed)
    b = algo.batch
    record_every = _default_record_every(prob, b) if record_every is None else record_every
    mon = _Monitor(prob, name or default_name(cfg), record_every, stop_Fgap, stop_dist2, max_time)
    ge = algo.init_cost
    mon.record(0, ge, W, eta=float("nan"), pair=float("nan"), mem=algo.memory_error(st))
    if algo.anchor_step:                               # anchor step at W_0 (the exact gradient is the initial cost)
        W = W - algo.anchor_displacement(st)
    instr = []
    t = 0
    info = {"eta": float("nan"), "pair": float("nan")}
    rec = mon.record_every
    for B, coin in _draws(rng, prob.n, b, n_iters, algo.needs_coin, algo.probs):
        m, c, aux = algo.pieces(W, st, B, coin)
        try:
            S, info = algo.step(W, m, c, st, t)
        except np.linalg.LinAlgError:
            if np.isfinite(m).all() and (c is None or np.isfinite(c).all()):
                raise
            W = np.full_like(W, np.nan)               # non-finite estimator (overflow): the run has diverged
            t += 1
            mon.record(t, ge, W, eta=float("nan"), pair=float("nan"), mem=float("nan"))
            break
        if instrument_every and t % instrument_every == 0 and (instrument_max is None or len(instr) < instrument_max):
            omega = m if c is None else m + algo.theta * c
            sdat = instrument(prob, W, G=omega, O=S, eta=1.0, per_sample=False)
            sdat.update(t=t, grad_evals=ge, nuc_m=float(svd_thin(m)[1].sum()))
            instr.append(sdat)
        ge += algo.commit(W, st, B, coin, aux, info)
        W = algo.advance(W, S, st)
        t += 1
        if not np.isfinite(W).all():                  # caught here, before the next factorization
            mon.record(t, ge, W, eta=info["eta"], pair=info["pair"], mem=float("nan"))
            break
        if t % rec == 0 and mon.record(t, ge, W, eta=info["eta"], pair=info["pair"], mem=algo.memory_error(st)):
            break
        if max_grad_evals is not None and ge >= max_grad_evals:
            break
    mon.finish(t, ge, W, eta=info["eta"], pair=info["pair"], mem=algo.memory_error(st))
    meta = dict(cfg)
    meta.update(seed=seed, n=prob.n, refreshes=st.refreshes, fallbacks=algo.fallbacks, instr=instr)
    if algo.estimator == "saga":
        meta["zbar_err"] = float(np.linalg.norm(st.zbar - st.rtab.T @ prob.X / prob.n))
    return mon.result(W, **meta)
