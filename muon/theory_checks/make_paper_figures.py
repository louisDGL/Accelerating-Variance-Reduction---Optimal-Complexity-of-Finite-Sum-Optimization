"""Figures of the condensed paper (muon/paper_spectral_muon.tex), with the data they show.

  figT1_ls_model   the least-squares model F(W) = (1/2) tr((W - W*) Sigma (W - W*)^T):
                   (a) polar flow, commuting start: amplitudes of the error along the eigenvectors of Sigma for two
                       condition numbers of Sigma, against the soft-thresholding path (a_k - t)_+;
                   (b) polar flow, generic start: spectral norm of the error for three condition numbers;
                   (c) discrete steps: objective gap of the nuclear-scaled polar step against gradient descent.
  figT2_noise      stochastic polar steps on a finite-sum least-squares problem with skewed residuals:
                   (a) batch one, same decaying length: minibatch estimator (settles at the weighted
                       least-absolute-deviation solution) against the SAGA estimator (goes to the least-squares one);
                   (b) offset of the mean-field stationary point against the batch size (roots of the mean field,
                       stochastic runs, SAGA estimator);
                   (c) scalar case with momentum: median of an exponential moving average of skewed noise.
  figT2s_noise     panels (a) and (b) of figT2_noise only; built only when "T2s" is requested.
  figR3_theory     figure of the short report (muon/report_spectral_muon.tex): panels (a), (b) of figT2_noise and
                   panel (c) of figT1_ls_model; built only when "R3" is requested.

The data are computed once and cached in muon/theory_checks/paper_figures_data.json (delete it to recompute).
Run: .venv/Scripts/python muon/theory_checks/make_paper_figures.py [--data-only] [T1] [T2] [T2s] [R3]
"""
from __future__ import annotations

import json
import os
import sys
import time
import types

import numpy as np
from scipy.optimize import root
from scipy.signal import lfilter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CACHE = os.path.join(HERE, "paper_figures_data.json")
FIGURES = os.path.join(ROOT, "muon", "figures")


# ----------------------------------------------------------------------------------------------------------------------
# linear algebra helpers
# ----------------------------------------------------------------------------------------------------------------------
def polar(M, tol=1e-13):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    keep = s > tol * max(s[0], 1e-300)
    return U[:, keep] @ Vt[keep]


def polar_stack(M, tol=1e-12):
    """Polar factor of each matrix of a stack, reduced on the nonzero singular values (a rank-one minibatch gradient
    has a rank-one polar factor)."""
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    keep = (s > tol * np.maximum(s[..., :1], 1e-300)).astype(float)
    return (U * keep[..., None, :]) @ Vt


def nuc(M):
    return float(np.linalg.svd(M, compute_uv=False).sum())


def geom_spectrum(d, kappa):
    return kappa ** (-np.arange(d) / (d - 1.0))


# ----------------------------------------------------------------------------------------------------------------------
# T1: least-squares model
# ----------------------------------------------------------------------------------------------------------------------
def data_flow_commuting():
    """Polar flow (explicit steps of length 1e-3) from an error that commutes with Sigma: amplitudes along v_k."""
    rng = np.random.default_rng(1)
    d = 5
    a0 = np.array([2.0, 1.6, 1.1, 0.7, 0.3])
    U, _ = np.linalg.qr(rng.standard_normal((d, d)))
    V, _ = np.linalg.qr(rng.standard_normal((d, d)))
    E0 = (U * a0) @ V.T
    eta, steps, every = 1e-3, 2400, 20
    out = {"a0": a0.tolist(), "eta": eta, "runs": {}}
    for kappa in (1.0, 1e4):
        Sigma = (V * geom_spectrum(d, kappa)) @ V.T
        E = E0.copy()
        ts, amps = [0.0], [np.linalg.norm(E @ V, axis=0).tolist()]
        for k in range(1, steps + 1):
            E = E - eta * polar(E @ Sigma)
            if k % every == 0:
                ts.append(k * eta)
                amps.append(np.linalg.norm(E @ V, axis=0).tolist())
        dev = max(np.max(np.abs(np.array(a) - np.clip(a0 - t, 0, None))) for t, a in zip(ts, amps))
        out["runs"][f"{kappa:g}"] = {"t": ts, "amps": amps, "max_dev_from_svt": float(dev)}
        print(f"  commuting flow, kappa(Sigma) = {kappa:g}: max deviation from (a_k - t)_+ = {dev:.2e} (step {eta})")
    return out


def data_flow_generic():
    """Polar flow from a generic error (||E_0||_op = 1) for three condition numbers of Sigma: ||E(t)||_op."""
    rng = np.random.default_rng(2)
    d = 5
    V, _ = np.linalg.qr(rng.standard_normal((d, d)))
    E0 = rng.standard_normal((d, d))
    E0 /= np.linalg.norm(E0, 2)
    eta, every = 2e-4, 25
    out = {"eta": eta, "runs": {}, "E0_nuc": nuc(E0)}
    for kappa in (1.0, 1e2, 1e4):
        lam = geom_spectrum(d, kappa)
        Sigma = (V * lam) @ V.T
        E = E0.copy()
        ts, ops = [0.0], [1.0]
        T_ext, k = None, 0
        while k * eta < 1.5:
            E = E - eta * polar(E @ Sigma)
            k += 1
            nrm = np.linalg.norm(E, 2)
            if T_ext is None and nrm <= 3 * eta:
                T_ext = k * eta
            if k % every == 0:
                ts.append(k * eta)
                ops.append(float(nrm))
        G0 = E0 @ Sigma
        out["runs"][f"{kappa:g}"] = {"t": ts, "op": ops, "T_ext": T_ext, "upper": nuc(G0) / lam.min(),
                                     "lower_grad": nuc(G0) / lam.sum()}
        print(f"  generic flow, kappa(Sigma) = {kappa:g}: extinction time {T_ext:.3f} x ||E_0||_op; bounds of Prop. 3.4: "
              f"[1.000, {np.sqrt(np.sum(G0 * E0) / lam.min()):.2f}] (sqrt(2 D_0 / mu_op), with ||E_0||_op = 1)")
    return out


def data_discrete():
    """Objective gap of the nuclear-scaled polar step (L = tr Sigma), gradient descent with step 1/lambda_1 and with
    exact line search, on K = d = 100, spectrum 1/k, generic start."""
    rng = np.random.default_rng(3)
    d = 100
    lam = 1.0 / np.arange(1, d + 1.0)
    V, _ = np.linalg.qr(rng.standard_normal((d, d)))
    Sigma = (V * lam) @ V.T
    E0 = rng.standard_normal((d, d))
    T = 3000

    def F(E):
        return 0.5 * float(np.sum((E @ Sigma) * E))

    runs = {}
    E = E0.copy(); tr = [F(E)]
    for _ in range(T):
        G = E @ Sigma
        E = E - (nuc(G) / lam.sum()) * polar(G)
        tr.append(F(E))
    runs["polar"] = tr
    amps = np.linalg.norm(E @ V, axis=0)                       # amplitudes along v_k at the end of the polar run
    # state of the error at the end of the polar run: is it on the flattest invariant ray?
    _, s_end, Vt_end = np.linalg.svd(E)
    end_state = {"share_s1": float(s_end[0] ** 2 / np.sum(s_end ** 2)),
                 "cos_top_vd": float(abs(Vt_end[0] @ V[:, -1])),
                 "nuclear_rank_gradient": float(nuc(E @ Sigma) ** 2 / np.sum((E @ Sigma) ** 2))}
    for name, ls in (("gd", False), ("gd_ls", True)):
        E = E0.copy(); tr = [F(E)]
        for _ in range(T):
            G = E @ Sigma
            step = (np.sum(G * G) / np.sum((G @ Sigma) * G)) if ls else 1.0
            E = E - step * G
            tr.append(F(E))
        runs[name] = tr
    out = {"d": d, "T": T, "kappa_F": float(lam[0] / lam[-1]), "kappa_op": float(lam.sum() / lam[-1]),
           "trace_over_lmax": float(lam.sum() / lam[0])}
    idx = np.unique(np.round(np.geomspace(1, T, 400)).astype(int))
    out["iters"] = [0] + idx.tolist()
    for name, tr in runs.items():
        tr = np.array(tr) / tr[0]
        out[name] = [1.0] + tr[idx].tolist()
        tail = np.log(tr[T - 500] / tr[T]) / 500.0              # exponent per step over the last 500 iterations
        out[name + "_tail_exponent"] = float(tail)
    out["ray_exponent"] = float(-2 * np.log(1 - 2 * lam[-1] / lam.sum()))      # flattest invariant ray (Thm 4.3(c))
    out["gd_exponent"] = float(-2 * np.log(1 - lam[-1] / lam[0]))
    out["two_over_kop"] = float(-np.log(1 - 2 * lam[-1] / lam.sum()))            # bound of Thm 4.3(b)
    out["end_state"] = end_state
    trp = np.array(runs["polar"])
    out["polar_window_exponents"] = {f"{a}-{b}": float(np.log(trp[a] / trp[b]) / (b - a))
                                     for (a, b) in ((1000, 2000), (2000, 2500), (2500, 3000))}
    rays = -2 * np.log(1 - 2 * lam[-10:] / lam.sum())            # rays A = {k} of the ten flattest directions
    out["ray_exponents_ten_flattest"] = [float(rays.min()), float(rays.max())]
    print(f"  discrete runs: kappa_F = {out['kappa_F']:.0f}, kappa_op = {out['kappa_op']:.0f}; relative gap after 10 / 100 / "
          f"1000 / {T} iterations: polar {runs['polar'][10] / runs['polar'][0]:.2e} / {runs['polar'][100] / runs['polar'][0]:.2e} / "
          f"{runs['polar'][1000] / runs['polar'][0]:.2e} / {runs['polar'][T] / runs['polar'][0]:.2e}; "
          f"GD 1/L {runs['gd'][10] / runs['gd'][0]:.2e} / {runs['gd'][100] / runs['gd'][0]:.2e} / {runs['gd'][1000] / runs['gd'][0]:.2e} / "
          f"{runs['gd'][T] / runs['gd'][0]:.2e}; GD line search {runs['gd_ls'][10] / runs['gd_ls'][0]:.2e} / "
          f"{runs['gd_ls'][100] / runs['gd_ls'][0]:.2e} / {runs['gd_ls'][1000] / runs['gd_ls'][0]:.2e} / {runs['gd_ls'][T] / runs['gd_ls'][0]:.2e}")
    print(f"     exponent per step over the last 500 iterations: polar {out['polar_tail_exponent']:.5f} "
          f"(flattest invariant ray: {out['ray_exponent']:.5f}; worst case 2/kappa_op: {out['two_over_kop']:.5f}), "
          f"GD 1/L {out['gd_tail_exponent']:.5f} (asymptotic {out['gd_exponent']:.5f}), GD line search {out['gd_ls_tail_exponent']:.5f}")
    print(f"     amplitudes of the error along v_k at the end of the polar run (largest five, index): "
          f"{[(int(i) + 1, float(f'{amps[i]:.3e}')) for i in np.argsort(-amps)[:5]]}")
    print(f"     end of the polar run: share of s_1^2 in ||E||_F^2 = {end_state['share_s1']:.4f}; |cos(top right singular "
          f"vector, v_d)| = {end_state['cos_top_vd']:.3f}; nuclear rank of the gradient = {end_state['nuclear_rank_gradient']:.2f}; "
          f"exponent per step on [1000,2000], [2000,2500], [2500,3000]: "
          + ", ".join(f"{v:.5f}" for v in out["polar_window_exponents"].values())
          + f"; exponents of the rays of the ten flattest directions: {rays.min():.5f} to {rays.max():.5f}")
    return out


# ----------------------------------------------------------------------------------------------------------------------
# T2: noise
# ----------------------------------------------------------------------------------------------------------------------
def ls_problem(n=2000, seed=4):
    rng = np.random.default_rng(seed)
    K, d = 2, 3
    X = rng.standard_normal((n, d)) * np.array([1.0, 0.6, 0.3])
    X[:, 0] = 1.0                                                        # intercept: without it the inputs are symmetric
    W0 = rng.standard_normal((K, d))                                     # and the gradient noise eps x^T is symmetric in
    Y = X @ W0.T + 0.5 * (rng.standard_exponential((n, K)) - 1.0)        # law whatever the skewness of the residuals
    W_ls = np.linalg.solve(X.T @ X, X.T @ Y).T
    Sig = X.T @ X / n
    return X, Y, W_ls, Sig


def weighted_lad(X, Y, W_init):
    """Minimizer of (1/n) sum_i ||W x_i - y_i|| / ||x_i|| (Weiszfeld iterations)."""
    xn = np.linalg.norm(X, axis=1)
    W = W_init.copy()
    for _ in range(20000):
        R = X @ W.T - Y
        w = 1.0 / (xn * np.maximum(np.linalg.norm(R, axis=1), 1e-14))
        W_new = np.linalg.solve((X * w[:, None]).T @ X, (X * w[:, None]).T @ Y).T
        done = np.linalg.norm(W_new - W) < 1e-15
        W = W_new
        if done:
            break
    return W


def length(t, eta0=0.3, t0=200.0, power=0.75):
    return eta0 / (1.0 + t / t0) ** power


def run_minibatch(X, Y, b, T, chains, seed, checkpoints):
    """Fixed-length polar steps with the minibatch gradient (sampling with replacement), `chains` independent runs in
    parallel; returns the average over the chains of the mean iterate over (c/2, c] for each checkpoint c."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    K = Y.shape[1]
    W = np.zeros((chains, K, d))
    acc = np.zeros((K, d)); sums = {}
    cps = sorted(set(checkpoints) | {c // 2 for c in checkpoints})
    for t in range(1, T + 1):
        B = rng.integers(0, n, (chains, b))
        XB = X[B]                                                  # (chains, b, d)
        R = np.einsum('ckd,cbd->cbk', W, XB) - Y[B]                # residuals (chains, b, K)
        G = np.einsum('cbk,cbd->ckd', R, XB) / b
        W = W - length(t) * polar_stack(G)
        acc += W.mean(0)
        if t in cps:
            sums[t] = acc.copy()
    return {c: (sums[c] - sums[c // 2]) / (c - c // 2) for c in checkpoints}


def run_saga(X, Y, b, T, chains, seed, checkpoints, nuclear_scaled=False, gamma=None):
    """Polar steps with the SAGA estimator (batch b, table of per-sample residuals initialized at W = 0). Either the
    same fixed length as run_minibatch, or the nuclear-scaled length gamma * ||estimator||_*; returns the tail averages
    as run_minibatch (fixed length) or the last iterate of every chain at the checkpoints (nuclear-scaled)."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    K = Y.shape[1]
    W = np.zeros((chains, K, d))
    Rtab = np.broadcast_to(-Y, (chains, n, K)).copy()              # stored residuals W_i x_i - y_i with W_i = 0
    Zbar = np.einsum('cnk,nd->ckd', Rtab, X) / n                   # mean of the stored gradients
    acc = np.zeros((K, d)); sums = {}; last = {}
    cps = sorted(set(checkpoints) | {c // 2 for c in checkpoints})
    ar = np.arange(chains)[:, None]
    for t in range(1, T + 1):
        B = np.stack([rng.choice(n, b, replace=False) for _ in range(chains)]) if b > 1 else rng.integers(0, n, (chains, 1))
        XB = X[B]
        Rnew = np.einsum('ckd,cbd->cbk', W, XB) - Y[B]
        Rold = Rtab[ar, B]
        diff = np.einsum('cbk,cbd->ckd', Rnew - Rold, XB)
        G = diff / b + Zbar
        if nuclear_scaled:
            U, s, Vt = np.linalg.svd(G, full_matrices=False)
            W = W - gamma * s.sum(-1)[:, None, None] * (U @ Vt)
        else:
            W = W - length(t) * polar_stack(G)
        Zbar = Zbar + diff / n
        Rtab[ar, B] = Rnew
        acc += W.mean(0)
        if t in cps:
            sums[t] = acc.copy()
            last[t] = W.copy()
    if nuclear_scaled:
        return {c: last[c] for c in checkpoints}
    return {c: (sums[c] - sums[c // 2]) / (c - c // 2) for c in checkpoints}


def mean_field_root(X, Y, b, S, seed, W_init, chunk=100000):
    """Zero of the mean field W -> E polar(minibatch gradient at W), estimated with S fixed batches (common random
    numbers); returns the root and the Frobenius norm of the residual field."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    A, C = [], []
    for s0 in range(0, S, chunk):
        B = rng.integers(0, n, (min(chunk, S - s0), b))
        XB, YB = X[B], Y[B]
        A.append(np.einsum('sbi,sbj->sij', XB, XB) / b)
        C.append(np.einsum('sbk,sbj->skj', YB, XB) / b)
    A, C = np.concatenate(A), np.concatenate(C)

    def field(w):
        return polar_stack(w.reshape(W_init.shape)[None] @ A - C).mean(0).reshape(-1)

    sol = root(field, W_init.reshape(-1), method='hybr', tol=1e-13)
    return sol.x.reshape(W_init.shape), float(np.linalg.norm(field(sol.x)))


def data_noise_trace():
    """Batch one, same decaying length, two estimators (panel a)."""
    X, Y, W_ls, Sig = ls_problem()
    n = X.shape[0]
    W_lad = weighted_lad(X, Y, W_ls)
    gnorm = lambda W: nuc((W - W_ls) @ Sig)                        # ||grad F(W)||_*
    out = {"n": n, "dist_lad_ls": float(np.linalg.norm(W_lad - W_ls)), "offset_b1_exact": gnorm(W_lad)}
    print(f"  finite-sum least squares, n = {n}: ||W_lad - W_ls||_F = {out['dist_lad_ls']:.4f}; "
          f"||grad F(W_lad)||_* = {out['offset_b1_exact']:.4f}")
    T, chains = 100000, 32
    cps = [int(c) for c in np.unique(np.round(np.geomspace(200, T, 25)).astype(int))]
    t0 = time.time()
    mb = run_minibatch(X, Y, 1, T, chains, 10, cps)
    sg = run_saga(X, Y, 1, T, chains, 11, cps)
    out.update({"T": T, "chains": chains, "checkpoints": cps, "length": [float(length(c)) for c in cps],
                "mb_to_ls": [float(np.linalg.norm(mb[c] - W_ls)) for c in cps],
                "mb_to_lad": [float(np.linalg.norm(mb[c] - W_lad)) for c in cps],
                "saga_to_ls": [float(np.linalg.norm(sg[c] - W_ls)) for c in cps]})
    print(f"  (a) batch one, {chains} chains x {T} steps ({time.time() - t0:.0f} s): tail-averaged iterate at the end: "
          f"minibatch: distance to LAD {out['mb_to_lad'][-1]:.2e}, to LS {out['mb_to_ls'][-1]:.2e}; "
          f"SAGA estimator: distance to LS {out['saga_to_ls'][-1]:.2e}")
    return out


def data_noise_offsets():
    """Offset of the limit point against the batch size (panel b)."""
    X, Y, W_ls, Sig = ls_problem()
    W_lad = weighted_lad(X, Y, W_ls)
    gnorm = lambda W: nuc((W - W_ls) @ Sig)
    rows = []
    W_prev = W_lad
    for b in (1, 2, 4, 8, 16, 32):
        t0 = time.time()
        if b == 1:
            roots = [gnorm(W_lad)] * 3
            W_prev = W_lad
        else:
            roots = []
            for rep in range(2):
                Wb, res = mean_field_root(X, Y, b, 200000, 100 * b + rep, W_prev)
                roots.append(gnorm(Wb))
            W_prev = Wb
        Tb, chains_b = 20000, 64
        end = run_minibatch(X, Y, b, Tb, chains_b, 20 + b, [Tb])[Tb]
        gamma = 1.0 / (3.0 * np.max(np.sum(X * X, axis=1)))
        Ts = 200000 // b + 2000                                    # about 100 passes over the n = 2000 samples
        sg_end = run_saga(X, Y, b, Ts, 2, 40 + b, [Ts], nuclear_scaled=True, gamma=gamma)
        sg_g = float(np.max([gnorm(Wc) for Wc in list(sg_end.values())[0]]))
        rows.append({"b": b, "root_mean": float(np.mean(roots)), "root_std": float(np.std(roots)),
                     "run": gnorm(end), "saga": sg_g})
        print(f"  (b) b = {b:2d}: mean-field root ||grad F||_* = {np.mean(roots):.5f} (+- {np.std(roots):.5f}, b x = "
              f"{b * np.mean(roots):.4f}); stochastic run (decaying length, tail average) {gnorm(end):.5f}; "
              f"SAGA estimator, nuclear-scaled length: {sg_g:.1e} ({time.time() - t0:.0f} s)", flush=True)
    return {"rows": rows}


def data_noise_momentum():
    """Scalar case with momentum: median of an exponential moving average of Exp(1) - 1 noise (panel c)."""
    rng = np.random.default_rng(5)
    rows = []
    for beta in (0.0, 0.5, 0.8, 0.9, 0.95, 0.98):
        meds = []
        for rep in range(5):
            a = rng.standard_exponential(8_000_000) - 1.0
            m = lfilter([1 - beta], [1, -beta], a)[int(30 / (1 - beta)):]
            meds.append(np.median(m))
        rows.append({"beta": beta, "median": float(np.mean(meds)), "se": float(np.std(meds) / np.sqrt(5)),
                     "formula": -(1 / 3) * (1 - beta ** 2) / (1 + beta + beta ** 2),
                     "heuristic": -(1 / 3) * (1 - beta) / (1 + beta)})
        print(f"  (c) beta = {beta:4.2f}: median {rows[-1]['median']:+.5f} (+- {rows[-1]['se']:.5f}); cumulant formula "
              f"{rows[-1]['formula']:+.5f}; effective-batch heuristic {rows[-1]['heuristic']:+.5f}")
    return {"rows": rows}


# ----------------------------------------------------------------------------------------------------------------------
# cache
# ----------------------------------------------------------------------------------------------------------------------
BUILDERS = {"flow_commuting": data_flow_commuting, "flow_generic": data_flow_generic, "discrete": data_discrete,
            "noise_trace": data_noise_trace, "noise_offsets": data_noise_offsets, "noise_momentum": data_noise_momentum}


def load_data(keys):
    data = {}
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as f:
            data = json.load(f)
    for k in keys:
        if k not in data:
            print(f"computing {k} ...", flush=True)
            data[k] = BUILDERS[k]()
            with open(CACHE, "w", encoding="utf-8") as f:      # saved after every block
                json.dump(data, f)
    return data


# ----------------------------------------------------------------------------------------------------------------------
# figures (style of the main project: experiments/plotstyle.py)
# ----------------------------------------------------------------------------------------------------------------------
def _plotstyle():
    sys.modules.setdefault("common", types.SimpleNamespace(FIGURES=FIGURES))   # plotstyle only needs common.FIGURES
    sys.path.insert(0, os.path.join(ROOT, "experiments"))
    import plotstyle as ps
    ps.setup()
    return ps


def fig_ls_model(data, ps):
    plt, INK = ps.plt, ps.INK
    fig = ps.figure(width=ps.FIG_W, height=2.55)
    gs = fig.add_gridspec(1, 3)
    # (a) commuting flow
    ax = fig.add_subplot(gs[0, 0])
    fc = data["flow_commuting"]
    a0 = np.array(fc["a0"])
    r1, r2 = fc["runs"]["1"], fc["runs"]["10000"]
    for k in range(len(a0)):
        ax.plot(r1["t"], np.array(r1["amps"])[:, k], color=ps.BLUE, lw=3.2, alpha=0.45, solid_capstyle="butt",
                label="κ(Σ) = 1" if k == 0 else None)
        ax.plot(r2["t"], np.array(r2["amps"])[:, k], color=ps.ORANGE, lw=1.3, ls=(0, (4, 2)),
                label="κ(Σ) = 10⁴" if k == 0 else None)
    ax.set_xlabel("time t (total length of the steps)")
    ax.set_ylabel("amplitudes of the error, ‖E(t) vₖ‖")
    ax.set_title("(a) commuting start", loc="left")
    ax.set_xlim(0, 2.4); ax.set_ylim(0, 2.1)
    ps.legend(ax, loc="upper right")
    # (b) generic flow
    ax = fig.add_subplot(gs[0, 1])
    fg = data["flow_generic"]
    for (key, lab, col, ls, lw) in (("1", "κ(Σ) = 1", ps.BLUE, "-", 3.2), ("100", "κ(Σ) = 10²", ps.ORANGE, "-", 1.8),
                                    ("10000", "κ(Σ) = 10⁴", ps.AQUA, (0, (4, 2)), 1.5)):
        r = fg["runs"][key]
        ax.plot(r["t"], r["op"], color=col, lw=lw, ls=ls, alpha=0.5 if key == "1" else 1.0, label=lab)
    ax.set_xlabel("time t")
    ax.set_ylabel("‖E(t)‖ₒₚ / ‖E(0)‖ₒₚ")
    ax.set_title("(b) generic start", loc="left")
    ax.set_xlim(0, 1.3); ax.set_ylim(0, 1.05)
    ps.legend(ax, loc="upper right")
    # (c) discrete steps
    _ls_panel_discrete(fig.add_subplot(gs[0, 2]), data["discrete"], ps)
    ps.savefig(fig, "figT1_ls_model")


def _ls_panel_discrete(ax, dd, ps, title="(c) discrete steps"):
    """Discrete steps on the least-squares model: nuclear-scaled polar step against gradient descent."""
    it = np.array(dd["iters"]); it_plot = np.maximum(it, 0.8)
    for (key, lab, col, ls) in (("polar", "polar step", ps.BLUE, "-"), ("gd", "GD, step 1/λ₁", ps.ORANGE, "-"),
                                ("gd_ls", "GD, line search", ps.AQUA, (0, (4, 2)))):
        ax.plot(it_plot, dd[key], color=col, ls=ls, lw=1.8, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("(F − F*) / (F₀ − F*)")
    ax.set_title(title, loc="left")
    ax.set_xlim(0.8, dd["T"])
    ax.set_ylim(1e-16, 3.0)                              # below 1e-16 the runs are at rounding level
    ps.legend(ax, loc="lower left")


def _noise_panel_trace(ax, tr, ps):
    """(a) traces, batch one."""
    INK = ps.INK
    c = np.array(tr["checkpoints"])
    ax.plot(c, tr["mb_to_ls"], color=ps.BLUE, marker="o", ms=3.2, lw=1.6, label="minibatch, to LS")
    ax.plot(c, tr["mb_to_lad"], color=ps.BLUE, marker="o", ms=3.2, lw=1.6, ls=(0, (4, 2)), mfc=INK["surface"],
            label="minibatch, to LAD")
    ax.plot(c, tr["saga_to_ls"], color=ps.ORANGE, marker="s", ms=3.2, lw=1.6, label="SAGA, to LS")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("iteration (batch size 1)")
    ax.set_ylabel("distance to the solution")
    ax.set_title("(a) batch one", loc="left")
    ax.set_ylim(1.5e-6, 40.0)                             # room for the legend below the curves
    ps.legend(ax, loc="lower left")


def _noise_panel_offsets(ax, rows, ps):
    """(b) batch size times the offset (linear axis: the plateau and the zero of the SAGA estimator are both visible)."""
    INK = ps.INK
    b = np.array([r["b"] for r in rows], dtype=float)
    ax.plot(b, [r["b"] * r["root_mean"] for r in rows], color=ps.BLUE, lw=1.6, label="zero of the mean field")
    ax.plot(b, [r["b"] * r["run"] for r in rows], color=ps.BLUE, marker="o", ms=4.5, ls="", mec=INK["surface"], mew=0.8,
            label="minibatch runs")
    ax.plot(b, [r["b"] * r["saga"] for r in rows], color=ps.ORANGE, marker="s", ms=4.5, ls="", mec=INK["surface"],
            mew=0.8, label="SAGA estimator", clip_on=False, zorder=5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(b); ax.set_xticklabels([f"{int(x)}" for x in b])
    ax.minorticks_off()
    ax.set_ylim(0.0, 0.36)
    ax.set_xlabel("batch size b")
    ax.set_ylabel("b · ‖∇F‖* at the limit point")
    ax.set_title("(b) b × offset", loc="left")
    ps.legend(ax, loc="lower right", bbox_to_anchor=(1.0, 0.07))


def _noise_panel_momentum(ax, mm, ps):
    """(c) momentum, scalar case."""
    INK = ps.INK
    beta = np.array([m["beta"] for m in mm])
    x = 1.0 - beta
    grid = np.geomspace(0.02, 1.0, 100); bg = 1 - grid
    ax.plot(grid, (1 / 3) * (1 - bg ** 2) / (1 + bg + bg ** 2), color=ps.BLUE, lw=1.6, label="cumulant formula")
    ax.plot(grid, (1 / 3) * (1 - bg) / (1 + bg), color=ps.ORANGE, lw=1.6, ls=(0, (4, 2)), label="effective batch")
    ax.plot(x, [-m["median"] for m in mm], color=INK["primary"], marker="o", ms=4.5, ls="", mec=INK["surface"], mew=0.8,
            label="measured")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("1 − β (momentum β)")
    ax.set_ylabel("|median| of the averaged noise")
    ax.set_title("(c) momentum, scalar", loc="left")
    ps.legend(ax, loc="upper left")


def fig_noise(data, ps):
    tr, rows, mm = data["noise_trace"], data["noise_offsets"]["rows"], data["noise_momentum"]["rows"]
    fig = ps.figure(width=ps.FIG_W, height=2.65)
    gs = fig.add_gridspec(1, 3)
    _noise_panel_trace(fig.add_subplot(gs[0, 0]), tr, ps)
    _noise_panel_offsets(fig.add_subplot(gs[0, 1]), rows, ps)
    _noise_panel_momentum(fig.add_subplot(gs[0, 2]), mm, ps)
    ps.savefig(fig, "figT2_noise")


def fig_noise_short(data, ps):
    """Panels (a) and (b) of figT2_noise only: the figure of the short report (muon/report_spectral_muon.tex)."""
    tr, rows = data["noise_trace"], data["noise_offsets"]["rows"]
    fig = ps.figure(width=ps.FIG_W, height=2.0)
    gs = fig.add_gridspec(1, 2)
    _noise_panel_trace(fig.add_subplot(gs[0, 0]), tr, ps)
    _noise_panel_offsets(fig.add_subplot(gs[0, 1]), rows, ps)
    ps.savefig(fig, "figT2s_noise")


def fig_theory_short(data, ps):
    """Figure of the short report (muon/report_spectral_muon.tex): panels (a), (b) of figT2_noise and the discrete
    steps of figT1_ls_model."""
    tr, rows = data["noise_trace"], data["noise_offsets"]["rows"]
    fig = ps.figure(width=ps.FIG_W, height=2.3)
    gs = fig.add_gridspec(1, 3)
    _noise_panel_trace(fig.add_subplot(gs[0, 0]), tr, ps)
    _noise_panel_offsets(fig.add_subplot(gs[0, 1]), rows, ps)
    _ls_panel_discrete(fig.add_subplot(gs[0, 2]), data["discrete"], ps, title="(c) polar step vs GD")
    ps.savefig(fig, "figR3_theory")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    want = args or ["T1", "T2"]                           # "T2s" and "R3" (short report) are built only on request
    keys = (["flow_commuting", "flow_generic", "discrete"] if "T1" in want else []) + \
           (["noise_trace", "noise_offsets", "noise_momentum"] if "T2" in want else []) + \
           (["noise_trace", "noise_offsets"] if "T2s" in want else []) + \
           (["noise_trace", "noise_offsets", "discrete"] if "R3" in want else [])
    data = load_data(keys)
    if "--data-only" not in sys.argv:
        ps = _plotstyle()
        if "T1" in want:
            fig_ls_model(data, ps)
        if "T2" in want:
            fig_noise(data, ps)
        if "T2s" in want:
            fig_noise_short(data, ps)
        if "R3" in want:
            fig_theory_short(data, ps)
