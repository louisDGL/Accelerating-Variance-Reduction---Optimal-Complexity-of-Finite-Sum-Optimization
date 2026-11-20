"""Exact conditional expectations of one iteration of a variance-reduced Muon configuration (batch size 1).

Given the state (the iterate W and the memory of the estimator), the next step of a single-index method takes exactly n
values S_1, ..., S_n (one per sampled index j, each with probability 1/n). Enumerating them gives, with no Monte-Carlo
error, the conditional expectation of any one-step quantity; this is the device used for SAGA in vrlab
(instrumentation.py), extended here to steps that are NONLINEAR in the estimator, for which no closed form exists.

`enumerate_steps(algo, W, st)` returns the n steps of the configuration `algo` (a vrmuon.VRMuon) at the state (W, st),
computed by the very functions used by the optimization loop (algo.pieces, algo.step), without modifying the state.

`one_step_stats(prob, algo, W, st)` returns (E = W - W_*, G = grad F(W), Sbar = E_j[S_j]; all expectations are exact):
  dec_F          E_j[F(W) - F(W - S_j)]          expected decrease of the objective (Bregman form, accurate)
  dec_dist2      E_j[||E||_F^2 - ||E - S_j||_F^2] = 2 <E, Sbar> - E_j ||S_j||_F^2
  lin            <G, Sbar>                        first-order part of dec_F (dec_F = lin - second-order terms)
  S2_F, S2_op    E_j ||S_j||_F^2, E_j ||S_j||_op^2
  var_F          E_j ||S_j - Sbar||_F^2
  mean_norm_F    ||Sbar||_F
  cos_grad       <Sbar, G> / (||Sbar||_F ||G||_F)                         Euclidean alignment of the mean step
  dual_align     <G, Sbar> / (||G||_* ||Sbar||_op)                        in [-1, 1]; 1 iff Sbar is a multiple of polar(G) on
                                                                           the row and column spaces of G
  cos_sharp      <Sbar, sharp(G)> / (||Sbar||_F ||sharp(G)||_F)           alignment with the full-gradient sharp step
  cos_dist       <Sbar, E> / (||Sbar||_F ||E||_F)                         alignment with the error
  rel_dec_F      dec_F / (F(W) - F_*),  rel_dec_dist2 = dec_dist2 / ||E||_F^2     per-iteration contraction rates
  cost           expected per-sample gradient evaluations of one iteration (refreshes included: the refresh of the
                 L-SVRG reference or of the PAGE estimate, and the `refresh_extra` entries of the SAGA table)
  eff_F, eff_dist2   rel_dec_F / cost, rel_dec_dist2 / cost                relative progress per gradient evaluation
                 (SAGA, SGD and L-SVRG: the step does not depend on the refresh coin, the expectations are complete.
                 PAGE: the expectations are conditional on the no-refresh branch while `cost` includes the refreshes,
                 so eff_* are approximations for PAGE. Configurations with anchor_step are refused.)
and the state descriptors
  noise_nuc      E_j ||omega_j - G||_* / ||G||_*      nuclear-norm error of the (damped) estimator relative to the gradient
  noise_F        (E_j ||omega_j - G||_F^2)^(1/2) / ||G||_F
  bias_F         ||E_j[omega_j] - G||_F / ||G||_F     (0 for theta = 1; the staleness of the memory mean for theta < 1)
  stale_F        ||m - G||_F / ||G||_F                relative staleness of the predictable part
  base_ratio     ||m||_F / ||G||_F                    size of the predictable part relative to the gradient. For the
                                                      SAGA and L-SVRG memories m = zbar + lam W vanishes at the point
                                                      W_B = -zbar / lam, the minimizer of the average of the stored
                                                      linearized losses plus the exact regularizer (the iterate of
                                                      MISO / Finito / dual-free SDCA for this memory); a small
                                                      base_ratio means that the method has driven W to W_B and that
                                                      the gradient is carried by the mean innovation E_j[c_j] = G - m
  surrogate_gap  ||W - W_B||_F / ||W - W_*||_F = ||m||_F / (lam ||W - W_*||_F)   (SAGA and L-SVRG memories)
  cos_m_grad     <m, G> / (||m||_F ||G||_F)           Euclidean alignment of the predictable part with the gradient
  smin_ratio     E_j ||theta c_j||_op / s_min(G)      innovation size relative to the smallest singular value of the
                                                      gradient (computed in the gauge subspace for softmax): the polar
                                                      factor of omega_j is close to polar(G) only if this is << 1
  nrank_G, nrank_m, nrank_omega (mean)   nuclear ranks ||.||_*^2 / ||.||_F^2 of the gradient, of the predictable part and
                                         of the estimator
  srank_E        stable rank ||E||_F^2 / ||E||_op^2 of the error.

`reference_steps(prob, W)` gives the same progress measures for deterministic full-gradient steps (GD with step 1/L_F,
the sharp step with 1/L_op, each also with its exact line search on the quadratic model), per gradient evaluation
(cost n), as yardsticks.

`collect_states(prob, W0, n_iters, at, seed, **config)` runs a configuration and returns copies of (t, W, state) at the
requested iteration counts, to evaluate several configurations FROM THE SAME STATES (they must share the estimator).

The expectations are over an index drawn UNIFORMLY and independently of the past (order 'iid'). Under a permutation
order (vrmuon: 'reshuffle', 'shuffle_once') the next index is uniform among the entries not yet visited in the pass,
or deterministic: `one_step_stats` refuses such configurations, as it refuses importance sampling. The extra
refreshes of the SAGA table (`refresh_extra`) happen after the step and do not change it: they enter the statistics
through `cost` only (and through the states, which `collect_states` advances with them).
"""
from __future__ import annotations

import copy

import numpy as np

from . import maps
from .polar import singular_values, svd_thin
from .vrmuon import VRMuon, _draws, extra_rng


def enumerate_steps(algo, W, st, t=0, idx=None, return_pieces=False):
    """Steps S_j of `algo` at (W, st) for j in idx (default: all n indices), as an array (len(idx), K, d).
    With return_pieces=True also returns the predictable part m and the array of the innovations c_j.

    The state is not modified (the on-demand caches of VRMuon.step are filled on a shallow copy). The enumeration
    covers the randomness of the index only: configurations whose next iterate also depends on the refresh coin
    through something else than the memory are refused (anchor_step: the iterate jumps at a refresh) or documented
    (PAGE: the n steps are those of the no-refresh branch, which has probability 1 - p; the refresh branch takes a
    full-gradient step). With importance sampling the step of index j has probability algo.probs[j]: expectations
    must be weighted accordingly (one_step_stats assumes uniform sampling and refuses such configurations). The
    steps themselves do not depend on the sampling order nor on refresh_extra (the extra refreshes follow the step)."""
    if algo.batch != 1:
        raise ValueError("exact enumeration is defined for batch = 1")
    if algo.beta_post or algo.beta:
        raise ValueError("exact one-step enumeration does not support momentum (the step would depend on the buffer; "
                         "use beta = beta_post = 0)")
    if algo.anchor_step:
        raise ValueError("exact one-step enumeration does not support anchor_step (with probability p the next "
                         "iterate is the anchor jump, not W - S_j)")
    st = copy.copy(st)                                # caches written by `step` stay local to this call
    n = algo.n
    idx = np.arange(n) if idx is None else np.asarray(idx)
    K, d = W.shape
    S = np.empty((len(idx), K, d))
    C = np.empty((len(idx), K, d)) if return_pieces else None
    cache = {}
    m = None
    for k, j in enumerate(idx):
        m, c, _ = algo.pieces(W, st, int(j), coin=1.0)
        S[k], _ = algo.step(W, m, c, st, t, cache=cache)
        if return_pieces:
            C[k] = 0.0 if c is None else c
    return (S, m, C) if return_pieces else S


def _dec_F(prob, W, S, G, chunk=64):
    """F(W) - F(W - S_j) for every step of the stack S (Bregman identity of problems.F_diff)."""
    Z = prob.outputs(W)
    P = None
    if prob.loss == "softmax":
        from .problems import softmax_rows
        P = softmax_rows(Z)
    out = np.empty(len(S))
    X = prob.X
    for lo in range(0, len(S), chunk):
        Sc = S[lo:lo + chunk]
        Delta = -np.einsum("id,jkd->jik", X, Sc)                    # Delta[j, i] = -(S_j x_i)
        m, n, K = Delta.shape
        breg = prob._bregman_rows(Delta.reshape(m * n, K),
                                  None if Z is None else np.tile(Z, (m, 1)),
                                  None if P is None else np.tile(P, (m, 1))).reshape(m, n).mean(axis=1)
        lin = np.einsum("kd,jkd->j", G, Sc)
        sq = np.einsum("jkd,jkd->j", Sc, Sc)
        out[lo:lo + chunk] = -(breg - lin + 0.5 * prob.lam * sq)
    return out


def _nuc(M, basis=None):
    return float(singular_values(M if basis is None else basis.T @ M).sum())


def _op(M):
    s = singular_values(M)
    return float(s[0]) if s.size else 0.0


def _cos(A, B):
    den = np.sqrt(np.vdot(A, A) * np.vdot(B, B))
    return float(np.vdot(A, B) / den) if den > 0 else float("nan")


def expected_cost(algo):
    """Expected per-sample gradient evaluations of one iteration (refreshes included; b + refresh_extra for SAGA
    with extra refreshes of its table)."""
    b, n, est = algo.batch, algo.n, algo.estimator
    if est in ("saga", "sgd"):
        return float(b + algo.refresh_extra)
    if est == "lsvrg":
        return 2.0 * b + algo.p * n
    return (1.0 - algo.p) * 2.0 * b + algo.p * n                        # page


def one_step_stats(prob, algo, W, st, t=0, with_state=True):
    """Exact one-step statistics of `algo` at the state (W, st) (see the module docstring). Returns a dict."""
    if algo.probs is not None or algo.order != "iid":
        raise ValueError("one_step_stats assumes uniform sampling of the index (no importance sampling, order 'iid')")
    S, m, C = enumerate_steps(algo, W, st, t, return_pieces=True)
    n = len(S)
    E = W - prob.W_star
    G = prob.grad(W)
    Sbar = S.mean(axis=0)
    S2 = np.einsum("jkd,jkd->j", S, S)
    dF = _dec_F(prob, W, S, G)
    Fgap = prob.Fgap(W)
    d2 = float(np.vdot(E, E))
    lin = float(np.vdot(G, Sbar))
    dec_dist2 = 2.0 * float(np.vdot(E, Sbar)) - float(S2.mean())
    sop2 = np.array([_op(S[j]) ** 2 for j in range(n)])
    nucG = _nuc(G, algo.basis)
    cost = expected_cost(algo)
    out = {
        "dec_F": float(dF.mean()), "dec_dist2": dec_dist2, "lin": lin,
        "S2_F": float(S2.mean()), "S2_op": float(sop2.mean()),
        "var_F": float(S2.mean() - np.vdot(Sbar, Sbar)), "mean_norm_F": float(np.sqrt(np.vdot(Sbar, Sbar))),
        "cos_grad": _cos(Sbar, G), "cos_dist": _cos(Sbar, E),
        "dual_align": lin / (nucG * _op(Sbar)) if nucG > 0 and np.any(Sbar) else float("nan"),
        "cos_sharp": _cos(Sbar, maps.polar(G, basis=algo.basis)),
        "Fgap": Fgap, "dist2_F": d2, "cost": cost,
        "rel_dec_F": float(dF.mean()) / Fgap if Fgap > 0 else float("nan"),
        "rel_dec_dist2": dec_dist2 / d2 if d2 > 0 else float("nan"),
        "frac_increase_F": float(np.mean(dF < 0)),
    }
    out["eff_F"] = out["rel_dec_F"] / cost
    out["eff_dist2"] = out["rel_dec_dist2"] / cost
    if with_state:
        th = algo.theta
        Om = m[None] + th * C                                            # the n (damped) estimators
        Err = Om - G[None]
        gF = float(np.sqrt(np.vdot(G, G)))
        basis = algo.basis
        nuc_err = np.array([_nuc(Err[j], basis) for j in range(n)])
        sG = singular_values(G if basis is None else basis.T @ G)
        c_op = np.array([_op(C[j]) for j in range(n)]) * abs(th)
        nr_om = []
        for j in range(n):
            s = singular_values(Om[j] if basis is None else basis.T @ Om[j])
            nr_om.append(float(s.sum() ** 2 / (s ** 2).sum()) if s.any() else float("nan"))
        sm = singular_values(m if basis is None else basis.T @ m)
        sE = singular_values(E)
        out.update({
            "noise_nuc": float(nuc_err.mean()) / nucG if nucG > 0 else float("nan"),
            "noise_F": float(np.sqrt(np.einsum("jkd,jkd->j", Err, Err).mean())) / gF if gF > 0 else float("nan"),
            "bias_F": float(np.linalg.norm(Om.mean(axis=0) - G)) / gF if gF > 0 else float("nan"),
            "stale_F": float(np.linalg.norm(m - G)) / gF if gF > 0 else float("nan"),
            "base_ratio": float(np.linalg.norm(m)) / gF if gF > 0 else float("nan"),
            "surrogate_gap": (float(np.linalg.norm(m)) / (prob.lam * np.sqrt(d2))
                              if (algo.estimator in ("saga", "lsvrg") and d2 > 0) else float("nan")),
            "cos_m_grad": _cos(m, G),
            "smin_ratio": float(c_op.mean()) / float(sG[-1]) if sG[-1] > 0 else float("inf"),
            "nrank_G": float(sG.sum() ** 2 / (sG ** 2).sum()),
            "nrank_m": float(sm.sum() ** 2 / (sm ** 2).sum()) if sm.any() else float("nan"),
            "nrank_omega": float(np.nanmean(nr_om)),
            "srank_E": float((sE ** 2).sum() / sE[0] ** 2) if sE[0] > 0 else float("nan"),
            "grad_norm_F": gF, "nuc_grad": nucG,
        })
    return out


def reference_steps(prob, W, basis=None):
    """Progress of deterministic full-gradient steps from W, per iteration and per gradient evaluation (cost n):
    'gd' (step 1/L_F), 'gd_ls' (exact line search on the quadratic model), 'sharp' (gamma = 1/L_op, the PolarGrad /
    dualized steepest-descent step), 'sharp_ls' (same direction, exact line search on the quadratic model). Each entry
    is a dict with rel_dec_F, rel_dec_dist2, eff_F, eff_dist2."""
    G = prob.grad(W)
    E = W - prob.W_star
    Fgap = prob.Fgap(W)
    d2 = float(np.vdot(E, E))
    O = maps.polar(G, basis=basis)
    nuc = _nuc(G, basis)
    steps = {"gd": G / prob.L_F, "sharp": (nuc / prob.L_op) * O}
    cG = float(np.vdot(G, prob.hvp(W, G)))
    cO = float(np.vdot(O, prob.hvp(W, O)))
    if cG > 0:
        steps["gd_ls"] = (float(np.vdot(G, G)) / cG) * G
    if cO > 0:
        steps["sharp_ls"] = (nuc / cO) * O
    out = {}
    for k, S in steps.items():
        dF = float(_dec_F(prob, W, S[None], G)[0])
        dd = 2.0 * float(np.vdot(E, S)) - float(np.vdot(S, S))
        out[k] = {"rel_dec_F": dF / Fgap if Fgap > 0 else float("nan"),
                  "rel_dec_dist2": dd / d2 if d2 > 0 else float("nan")}
        out[k]["eff_F"] = out[k]["rel_dec_F"] / prob.n
        out[k]["eff_dist2"] = out[k]["rel_dec_dist2"] / prob.n
    return out


def collect_states(prob, W0, n_iters, at, seed=0, **config):
    """Run the configuration `config` (keyword arguments of vrmuon.VRMuon) for n_iters iterations and return the list
    of (t, W copy, state deep copy) at the iteration counts in `at` (states BEFORE iteration t is performed).
    The trajectory is that of vrmuon.vr_muon(prob, W0, n_iters, seed, **config) with the SAME n_iters: for L-SVRG
    and PAGE at batch 1 the coin stream depends on n_iters (see vrmuon._draws), so a run launched with another
    n_iters (e.g. a budget-limited run of the experiments) is a different realization; SAGA is prefix-consistent.
    The caches of the collected states (frozen metric, lazy anchor) are keyed by the configuration: evaluating
    another configuration at these states recomputes what it needs from the anchor stored in the state."""
    algo = VRMuon(prob, n_iters=n_iters, **config)
    W = np.array(W0, dtype=np.float64, copy=True)
    st = algo.init_state(W)
    rng = np.random.default_rng(seed)
    rng_x = extra_rng(seed) if algo.refresh_extra else None
    at = sorted(set(int(a) for a in at))
    out = []
    t = 0
    if algo.anchor_step:
        W = W - algo.anchor_displacement(st)
    if at and at[0] == 0:
        out.append((0, W.copy(), copy.deepcopy(st)))
    for B, coin in _draws(rng, prob.n, algo.batch, n_iters, algo.needs_coin, algo.probs, algo.order):
        m, c, aux = algo.pieces(W, st, B, coin)
        S, info = algo.step(W, m, c, st, t)
        algo.commit(W, st, B, coin, aux, info)
        W = algo.advance(W, S, st)
        t += 1
        if rng_x is not None and np.isfinite(W).all():        # extra refreshes of the table, as vrmuon.vr_muon
            algo.refresh(W, st, B, rng_x)
        if t in at:
            out.append((t, W.copy(), copy.deepcopy(st)))
        if not np.isfinite(W).all():
            break
    return out
