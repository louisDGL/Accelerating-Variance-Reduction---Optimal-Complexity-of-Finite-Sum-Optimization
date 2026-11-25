"""Figure E3 of the appendix of explanatory figures of the paper (paper/figures/figE3_polar_median): where polar steps
go, in the plane. Explanatory figure; it adds no result to the paper. It is the planar companion of figT10_sign_median
(dimension one) and explains Theorem 2(b),(c) and Proposition 3(b),(c),(d) of paper/paper_saga_to_muon.tex, with (M1)
and (M5) for the runs on a geometric cooldown.

The toy (an exact instance of Theorem 2 and of Proposition 3(d); nothing is drawn by hand):
    W in R^{2 x 1} (K = 2, d = 1, features x_i = 1),  F(W) = (1/n) sum_i f_i(W),  f_i(W) = |W - y_i|^2 / 2,  n = 200,
    y_i = ((e_i1 - 1)/2, e_i2 - 1) with e_ik i.i.d. Exp(1), drawn once (seed POP_SEED): a skewed cloud of the plane.
F(W) = |W - ybar|^2 / 2 + const, mu_op = L_op = 1, and the minimizer W_* is the mean ybar of the cloud. The gradient
of sample j is grad F(W) + xi_j with xi_j = ybar - y_j: additive noise, independent of W, of mean zero (the hypothesis
of Theorem 2). For a K x 1 matrix the polar factor is the normalization omega / |omega| and the nuclear and spectral
norms are the Euclidean norm: this is the VECTOR case (the classical one of Cardot, Cenac and Zitt, cited in the
paper), not evidence about the spectral map of matrices with K, d >= 2. A nuclear median of xi is ybar minus a
geometric median of the y_i, so the limit of Theorem 2(b) is
    W_dagger = the geometric median of the y_i  (Weiszfeld's iteration, checked by Newton's iteration on the mean
               field; the script stops if the two disagree or if the mean field does not vanish there),
and the mean step Phi(W) = (1/n) sum_i (W - y_i)/|W - y_i| is the gradient of Lambda(W) = (1/n) sum_i |W - y_i|, which
is the noise-smoothed nuclear norm psi_xi(grad F(W)) up to a constant (the objective of Proposition 3(d)).

Runs, all from W_0 (the batch-one runs of one seed use the same indices):
  no table             W <- W - eta_t polar(g_i(W)), i uniform, eta_t = ETA0 (1 + t/n)^(-3/4): deterministic lengths
                       with sum eta_t = infinity and sum eta_t^2 < infinity, the class of Theorem 2;
  no table, batch b    the same with the mean of b sampled gradients (indices with replacement: the noise of Theorem
                       2 is ybar minus the mean of b sampled y_i); CHAINS chains of T_BATCH iterations, iterate
                       averaged over the second half of the run and over the chains;
  SAGA table           W <- W - lr_t polar(omega_t), omega_t the SAGA estimator (table at W_0 at the start),
                       lr_t = lr_0 Q^(t b / n): the recipe (4) of the paper with the exact polar factor, no momentum.
                       Batch one: lr_0 = LR0. Batches of b distinct indices (panel (c)): lr_0 = LR0 * b, which keeps
                       the total path sum_t lr_t (with lr_0 = LR0 the path at b = 32 is shorter than |W_0 - W_*| and
                       the run cannot arrive: the script prints it);
  cooldown, no table   the same lr_t on the sampled gradient, batch one (the control of (M5), here on the very
                       schedule of the run with the table). Its total path is 21 times |W_0 - W_*|: the budget
                       allowed convergence. It stops next to W_dagger (0.01 to 0.02 from it), 0.29 from W_*.
A peculiarity of the run with the table, stated because its path in panel (a) is a straight segment: with x_i = 1 the
SAGA estimator is W - phi_j + mean_i phi_i - ybar (phi_i: the iterate at which entry i was last replaced); the sampled
y_j has cancelled, and in exact arithmetic the iterate stays on the line through W_0 and W_*, where it is the sign
iteration of figT10_sign_median (run_table_on_line computes it). It first stops short of W_*, crosses it after about
one pass and overshoots it by 0.25 to 0.39 around pass 2, while part of the table still holds gradients at W_0; then
it settles. In floating point the component across the line starts at rounding level (1e-14 during the first pass),
is amplified by the normalization whenever the estimator is short, exceeds 1e-9 after about 5 passes and stays below
7e-3: the digits of the final distances depend on the rounding (hence on the platform), their order (lr_t) does not,
and the exact one-dimensional iteration ends at the same order (printed).
Zeros of the mean field for batches of b: the geometric median of the law of the mean of b sampled y_i. b = 1: the
n atoms (exact); b = 2: the n(n+1)/2 pairs (exact) and, as a check of the sampler, Monte Carlo; b >= 4: REPS
independent samples of S_MC batch means each (mean over the replicates, standard error from their dispersion).

Panels:
  (a) the plane of W: the cloud (the part in the window), level sets of Lambda, W_* and W_dagger, the run without a
      table of seed SEEDS[0] (thin: the iterates; thick: the iterate averaged over the window (t/2, t]) and the run
      with the SAGA table (same indices), a ring in the colour of each run where it ends; inset: the segment from
      W_dagger to W_* enlarged ZOOM times, with the zeros of the mean field for b = 2, 4, 8, 16 (open blue circles,
      labelled by b; b = 1 is W_dagger). The direct labels have a halo in the surface colour (function halo);
  (b) distances against passes (median over the SEEDS seeds, band = range over the seeds): the averaged iterate
      without a table, to W_dagger (blue); the iterate with the table, to W_* (orange); the cooldown without a table,
      to W_* (grey ink, as Figure 4(a) of the paper draws the runs without a table on the cooldown; the lecture
      version figT11 draws it in blue, dashed). The distance to W_* of the averaged iterate without a table is
      printed, not drawn: it lies on the grey curve;
  (c) b x |limit - W_*| for b = 1 ... 32: zeros of the mean field (line; bars = two standard errors), averaged runs
      (dots; bars = two standard errors over the chains), SAGA estimator on the cooldown (squares, at their measured
      value, zero at the scale of the axis).

Status. Proved: the limit without a table (Theorem 2(b) with Med(xi) = {ybar - geometric median}; Proposition 3(d));
that the minimizer is the only fixed point with the table (Proposition 3(c), which says nothing about attraction).
Computed exactly or by Monte Carlo with its error: the points W_*, W_dagger, the level sets, the zeros of the mean
field. Observed on this instance only (illustration): that the run with the table reaches that fixed point on the
cooldown q = 0.8 (a geometric cooldown is not in the class of Theorem 2; Proposition 5 certifies the mechanism at
much slower cooldowns), where the cooldown without a table freezes, and the ends of the 60-pass and 100-pass runs
(the averaged iterate without a table is still 0.008 to 0.014 from W_dagger: its limit is proved, not shown).
Panel (c) shows the 1/b scaling only: the two entries of the noise are not identically distributed, formula (3) of
the paper and its constant are not drawn.

The script prints every number it plots and stops if two computations of the same number disagree. Nothing is read
from the results of the project. The functions data, print_all, halo, path_point, panel_plane, fit_window,
panel_distances, below_no_table and table_label_anchor are also used by make_figT11_polar_median.py (the lecture
version of panels (a) and (b)).
Run:    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/scripts/make_figE3_polar_median.py   (about 45 s)
Output: paper/figures/figE3_polar_median.pdf and .png
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import explain_style as es  # noqa: E402

NAME = "figE3_polar_median"
N = 200
POP_SEED = 0
W0 = (1.30, 0.45)             # starting point of every run
PASSES = 60                   # batch-one runs of panels (a), (b)
SEEDS = (0, 1, 2, 3, 4)
ETA0 = 0.03                   # no table: eta_t = ETA0 (1 + t/n)^(-3/4)
LR0, Q = 0.03, 0.8            # cooldown: lr_t = LR0 * Q^(t b / n) at batch one (LR0 * b for batches of b)
RECORD = 10                   # panel (b): one point every RECORD iterations
BATCHES = (1, 2, 4, 8, 16, 32)
S_MC, REPS = 500_000, 16      # zeros of the mean field, b >= 2: REPS independent samples of S_MC batch means
CHAINS, T_BATCH = 64, 20_000  # averaged runs of panel (c)
SAGA_PASSES = 100             # SAGA estimator of panel (c)
WINDOW = (-1.28, 1.44, -1.00, 0.92)      # panel (a): x_min, x_max, y_min, y_max
LEVELS = (0.01, 0.05, 0.12, 0.22, 0.35)          # level sets of Lambda - min Lambda
GRID_MARGIN = 0.7             # Lambda is evaluated on the window enlarged by this margin
ZOOM = 3.3                    # enlargement of the inset of panel (a)


# ----------------------------------------------------------------------------------------------------------------------
# the toy
# ----------------------------------------------------------------------------------------------------------------------
def population():
    e = np.random.default_rng(POP_SEED).standard_exponential((N, 2))
    return np.stack([0.5 * (e[:, 0] - 1.0), e[:, 1] - 1.0], axis=1)


def mean_unit(P, w, weights=None):
    """Mean of the unit vectors (w - p)/|w - p| over the points p of P: the mean field of the polar step at w."""
    dx, dy = w[0] - P[:, 0], w[1] - P[:, 1]
    r = np.hypot(dx, dy)
    if weights is None:
        return np.array([np.mean(dx / r), np.mean(dy / r)])
    return np.array([np.sum(weights * dx / r), np.sum(weights * dy / r)]) / np.sum(weights)


def weiszfeld(P, start, weights=None, tol=1e-14, itmax=100_000):
    """Geometric median of the points P (weights optional) by Weiszfeld's fixed point."""
    w = np.array(start, dtype=float)
    wt = np.ones(len(P)) if weights is None else weights
    for _ in range(itmax):
        iw = wt / np.hypot(P[:, 0] - w[0], P[:, 1] - w[1])
        new = np.array([np.sum(P[:, 0] * iw), np.sum(P[:, 1] * iw)]) / np.sum(iw)
        if np.hypot(*(new - w)) < tol:
            return new
        w = new
    raise SystemExit("Weiszfeld's iteration did not converge")


def newton_median(P, start, weights=None, tol=1e-13, itmax=60):
    """The same point by Newton's iteration on the mean field (the gradient of the mean distance)."""
    w = np.array(start, dtype=float)
    wt = np.full(len(P), 1.0 / len(P)) if weights is None else weights / np.sum(weights)
    for _ in range(itmax):
        dx, dy = w[0] - P[:, 0], w[1] - P[:, 1]
        r = np.hypot(dx, dy)
        ux, uy = dx / r, dy / r
        g = np.array([np.sum(wt * ux), np.sum(wt * uy)])
        k = wt / r
        H = np.array([[np.sum(k * uy * uy), -np.sum(k * ux * uy)], [-np.sum(k * ux * uy), np.sum(k * ux * ux)]])
        step = np.linalg.solve(H, g)
        w = w - step
        if np.hypot(*step) < tol:
            return w
    raise SystemExit("Newton's iteration for the geometric median did not converge")


def mean_distance(y, pts):
    """Lambda at the rows of pts."""
    return np.hypot(pts[:, None, 0] - y[None, :, 0], pts[:, None, 1] - y[None, :, 1]).mean(axis=1)


def batch_means(y, b, size, rng):
    acc = np.zeros((size, 2))
    for _ in range(b):
        acc += y[rng.integers(0, N, size)]
    return acc / b


def zeros_of_mean_field(y, mean, gm, monte_carlo=True):
    """b -> (zero of the mean field, standard error of b x its distance to the mean, how it was computed).
    monte_carlo = False: b = 1 and 2 only (exact)."""
    out = {1: (gm, 0.0, "exact, n atoms")}
    i, j = np.triu_indices(N)
    pairs, wts = 0.5 * (y[i] + y[j]), np.where(i == j, 1.0, 2.0)
    z2 = weiszfeld(pairs, gm, wts)
    if np.hypot(*(newton_median(pairs, gm, wts) - z2)) > 1e-10:
        raise SystemExit("b = 2: Weiszfeld and Newton disagree")
    out[2] = (z2, 0.0, "exact, n(n+1)/2 pairs")
    if not monte_carlo:
        return out
    start = z2
    for b in BATCHES[1:]:
        reps = np.array([newton_median(batch_means(y, b, S_MC, np.random.default_rng(1000 * b + r)), start)
                         for r in range(REPS)])
        z = reps.mean(axis=0)
        se = float(b * np.std(np.hypot(*(reps - mean).T), ddof=1) / np.sqrt(REPS))
        if b == 2:                                              # check of the sampler against the enumeration
            gap = 2.0 * abs(np.hypot(*(z - mean)) - np.hypot(*(z2 - mean)))
            if gap > 4.0 * se + 1e-4:
                raise SystemExit(f"b = 2: Monte Carlo and enumeration disagree ({gap:.2e}, standard error {se:.2e})")
            out["mc2"] = (z, se)
        else:
            out[b] = (z, se, f"{REPS} x {S_MC:,} sampled batch means")
        start = z
    return out


# ----------------------------------------------------------------------------------------------------------------------
# runs
# ----------------------------------------------------------------------------------------------------------------------
def run_no_table(y, seed):
    """Batch-one polar steps on the sampled gradient, eta_t = ETA0 (1 + t/n)^(-3/4). Returns every iterate."""
    idx = np.random.default_rng(seed).integers(0, N, PASSES * N)
    wx, wy = W0
    out = np.empty((PASSES * N + 1, 2))
    out[0] = W0
    for t in range(PASSES * N):
        rx, ry = wx - y[idx[t], 0], wy - y[idx[t], 1]
        s = ETA0 * (1.0 + t / N) ** (-0.75) / np.hypot(rx, ry)
        wx, wy = wx - s * rx, wy - s * ry
        out[t + 1] = wx, wy
    return out


def run_cooldown(y, seed, table, b=1, passes=PASSES, lr0=LR0):
    """Polar steps of length lr_t = lr0 Q^(t b / n). table: the SAGA estimator (f_i'(W) = W - y_i is stored as the
    point phi_i where it was computed; table at W_0 at the start), else the mean of the b sampled gradients. b = 1:
    the indices of run_no_table with the same seed; b > 1: b distinct indices. Returns every iterate."""
    rng = np.random.default_rng(seed)
    T = passes * N // b
    idx = rng.integers(0, N, T)[:, None] if b == 1 else np.array([rng.choice(N, b, replace=False) for _ in range(T)])
    w = np.array(W0, dtype=float)
    ybar = y.mean(axis=0)
    phi = np.tile(w, (N, 1))
    phibar = phi.mean(axis=0)
    out = np.empty((T + 1, 2))
    out[0] = w
    for t in range(T):
        B = idx[t]
        if table:                                   # omega = mean_B (f_j'(W) - z_j) + mean_i z_i, z_i = phi_i - y_i
            diff = (w - phi[B]).sum(axis=0)
            omega = diff / b + phibar - ybar
            phibar = phibar + diff / N
            phi[B] = w
        else:
            omega = w - y[B].mean(axis=0)
        nrm = np.hypot(omega[0], omega[1])
        if nrm > 0.0:
            w = w - lr0 * Q ** (t * b / N) * omega / nrm
        out[t + 1] = w
        if table and (t + 1) % max(N // b, 1) == 0:
            phibar = phi.mean(axis=0)               # recomputed once per pass, as the SAGA loops of the project do
    return out


def run_table_on_line(d0, seed, passes=PASSES, lr0=LR0):
    """The batch-one run with the table as exact arithmetic would give it. With x_i = 1 the SAGA estimator is
    W - phi_j + mean_i phi_i - ybar: the sampled y_j has cancelled, and as long as the phi_i are on the line through
    W_0 and W_* so are the estimator and the next iterate. The coordinate s of W - W_* along (W_0 - W_*)/|W_0 - W_*|
    then follows the sign iteration of figT10_sign_median: s <- s - lr_t sign(s - sigma_j + mean_i sigma_i),
    sigma_j <- s, every sigma_i = d0 = |W_0 - W_*| at first (the indices of run_cooldown). Returns |s| at the end."""
    idx = np.random.default_rng(seed).integers(0, N, passes * N)
    sig = np.full(N, d0)
    s = d0
    for t in range(passes * N):
        j = idx[t]
        omega = s - sig[j] + sig.mean()
        sig[j] = s
        s -= lr0 * Q ** (t / N) * np.sign(omega)
    return abs(s)


def window_average(path):
    """Average of the iterates over the window (t/2, t], for t = 1, 2, ... (row t - 1)."""
    c = np.cumsum(path, axis=0)
    t = np.arange(1, len(path))
    return (c[t] - c[t // 2]) / (t - t // 2)[:, None]


def averaged_batch_run(y, b, seed, chunk=250):
    """CHAINS chains of polar steps on the mean of b sampled gradients (with replacement), eta_t as run_no_table;
    returns the average of each chain over the second half of the run. The batch means are drawn `chunk` iterations
    at a time (speed only)."""
    rng = np.random.default_rng(seed)
    W = np.tile(np.array(W0, dtype=float), (CHAINS, 1))
    acc = np.zeros((CHAINS, 2))
    eta = ETA0 * (1.0 + np.arange(T_BATCH) / N) ** (-0.75)
    for t0 in range(0, T_BATCH, chunk):
        means = y[rng.integers(0, N, (min(chunk, T_BATCH - t0), CHAINS, b))].mean(axis=2)
        for k in range(len(means)):
            r = W - means[k]
            W = W - (eta[t0 + k] / np.hypot(r[:, 0], r[:, 1]))[:, None] * r
            if t0 + k >= T_BATCH // 2:
                acc += W
    return acc / (T_BATCH - T_BATCH // 2)


# ----------------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------------
def data(offsets=True):
    """offsets = False skips what only panel (c) and the inset of panel (a) need (the sampled zeros of the mean field,
    the averaged minibatch runs and the SAGA runs with batches)."""
    y = population()
    mean = y.mean(axis=0)
    gm = weiszfeld(y, mean)
    gm_newton = newton_median(y, mean)
    if np.hypot(*(gm - gm_newton)) > 1e-10 or np.hypot(*mean_unit(y, gm)) > 1e-12:
        raise SystemExit("the two computations of the geometric median disagree, or the mean field does not vanish")
    off = float(np.hypot(*(gm - mean)))
    w0 = np.array(W0)
    D = {"y": y, "mean": mean, "gm": gm, "gm_gap": float(np.hypot(*(gm - gm_newton))),
         "phi_gm": float(np.hypot(*mean_unit(y, gm))), "phi_mean": mean_unit(y, mean), "offset": off,
         "lam_gm": float(mean_distance(y, gm[None])[0]), "lam_mean": float(mean_distance(y, mean[None])[0]),
         "e_noise": float(np.hypot(*(y - mean).T).mean()), "d0": float(np.hypot(*(w0 - mean)))}
    # level sets of Lambda on the window of panel (a)
    gx = np.arange(WINDOW[0] - GRID_MARGIN, WINDOW[1] + GRID_MARGIN, 0.0125)   # wider than the window: the axes
    gy = np.arange(WINDOW[2] - GRID_MARGIN, WINDOW[3] + GRID_MARGIN, 0.0125)   # extend it to keep equal scales
    X, Y = np.meshgrid(gx, gy)
    D["grid"] = (gx, gy, mean_distance(y, np.stack([X.ravel(), Y.ravel()], axis=1)).reshape(X.shape) - D["lam_gm"])
    # batch-one runs
    no = [run_no_table(y, s) for s in SEEDS]
    avg = [window_average(p) for p in no]
    tb = [run_cooldown(y, s, True) for s in SEEDS]
    cd = [run_cooldown(y, s, False) for s in SEEDS]
    rec = np.arange(RECORD, PASSES * N + 1, RECORD)
    D["path_no"], D["path_avg"], D["path_tb"] = no[0], avg[0], tb[0]
    D["passes"] = rec / N
    D["no_gm"] = np.array([np.hypot(*(a[rec - 1] - gm).T) for a in avg])
    D["no_mean"] = np.array([np.hypot(*(a[rec - 1] - mean).T) for a in avg])
    D["tb_mean"] = np.array([np.hypot(*(p[rec] - mean).T) for p in tb])
    D["cd_mean"] = np.array([np.hypot(*(p[rec] - mean).T) for p in cd])
    D["cd_gm_end"] = np.array([np.hypot(*(p[-1] - gm)) for p in cd])
    D["raw_gm_end"] = np.array([np.hypot(*(p[-1] - gm)) for p in no])
    D["frozen"] = float(np.median(D["cd_mean"][:, -1]))          # the number of the label of panel (b)
    # The run with the table: with x_i = 1 the SAGA estimator W - phi_j + mean_i phi_i - ybar does not involve the
    # sampled y_j, and in exact arithmetic the iterate stays on the line through W_0 and W_* (see run_table_on_line).
    # In floating point the component across the line starts at rounding level and is amplified by the normalization
    # whenever the estimator is short. Per seed: (largest distance to the line during the first pass, pass at which
    # it first exceeds 1e-9, largest distance over the run); (largest overshoot beyond W_*, its pass, pass of the
    # first crossing of W_*); and the end of the exact one-dimensional iteration.
    u = (w0 - mean) / D["d0"]
    across = [np.abs((p - mean) @ np.array([-u[1], u[0]])) for p in tb]
    along = [(p - mean) @ u for p in tb]
    D["tb_line"] = [(float(a[:N + 1].max()), int(np.argmax(a > 1e-9)) / N, float(a.max())) for a in across]
    D["tb_over"] = [(float(-a.min()), int(a.argmin()) / N, int(np.argmax(a < 0.0)) / N) for a in along]
    D["tb_exact"] = np.array([run_table_on_line(D["d0"], s) for s in SEEDS])
    t = np.arange(PASSES * N)
    D["budget_cool"] = float(np.sum(LR0 * Q ** (t / N)))
    D["budget_no"] = float(np.sum(ETA0 * (1.0 + t / N) ** (-0.75)))
    D["lr_end"] = LR0 * Q ** PASSES
    D["eta_end"] = ETA0 * (1.0 + PASSES) ** (-0.75)
    # zeros of the mean field
    zeros = zeros_of_mean_field(y, mean, gm, monte_carlo=offsets)
    D["zeros"] = zeros
    D["rows"] = []
    D["offsets"] = offsets
    if not offsets:
        return D
    for b in BATCHES:
        z, se, how = zeros[b]
        row = {"b": b, "zero": z, "b_off": b * float(np.hypot(*(z - mean))), "se": se, "how": how}
        ends = averaged_batch_run(y, b, 50 + b)
        end = ends.mean(axis=0)
        u = (end - mean) / np.hypot(*(end - mean))
        row["run"] = b * float(np.hypot(*(end - mean)))
        row["run_se"] = b * float(np.std((ends - mean) @ u, ddof=1) / np.sqrt(CHAINS))
        sg = run_cooldown(y, 40 + b, True, b=b, passes=SAGA_PASSES, lr0=LR0 * b)
        row["saga"] = b * float(np.hypot(*(sg[-1] - mean)))
        row["saga_lr_end"] = LR0 * b * Q ** SAGA_PASSES
        D["rows"].append(row)
    short = run_cooldown(y, 40 + BATCHES[-1], True, b=BATCHES[-1], passes=SAGA_PASSES)   # lr_0 = LR0: path too short
    D["short"] = (float(np.hypot(*(short[-1] - mean))), LR0 / (1.0 - Q ** (BATCHES[-1] / N)))
    return D


def print_all(D):
    m, g = D["mean"], D["gm"]
    inside = ((D["y"][:, 0] > WINDOW[0]) & (D["y"][:, 0] < WINDOW[1]) & (D["y"][:, 1] > WINDOW[2])
              & (D["y"][:, 1] < WINDOW[3]))
    print(f"population: n = {N} (seed {POP_SEED}), mean W* = ({m[0]:+.4f}, {m[1]:+.4f}), geometric median W_dagger = "
          f"({g[0]:+.4f}, {g[1]:+.4f}); Weiszfeld - Newton {D['gm_gap']:.1e}; |mean field at W_dagger| = "
          f"{D['phi_gm']:.1e}; mean field at W* = ({D['phi_mean'][0]:+.4f}, {D['phi_mean'][1]:+.4f})")
    print(f"   offset |W_dagger - W*| = {D['offset']:.4f}  (Proposition 3(b): at most E|xi| = {D['e_noise']:.4f}); "
          f"F(W_dagger) - F* = {D['offset'] ** 2 / 2:.4f} (Theorem 2(c) with L_op = 1: equality); Lambda(W*) - "
          f"Lambda(W_dagger) = {D['lam_mean'] - D['lam_gm']:.4f}; {int(inside.sum())} of the {N} points are in the window "
          f"[{WINDOW[0]}, {WINDOW[1]}] x [{WINDOW[2]}, {WINDOW[3]}]")
    print(f"(a) level sets of Lambda - min Lambda: {', '.join(f'{v:g}' for v in LEVELS)}; W_0 = {W0}, |W_0 - W*| = "
          f"{D['d0']:.4f}; paths of seed {SEEDS[0]}: averaged iterate without a table ends {D['no_gm'][0, -1]:.4f} from "
          f"W_dagger, iterate with the table {D['tb_mean'][0, -1]:.1e} from W*")
    print(f"(b) {PASSES} passes, seeds {SEEDS}; eta_t = {ETA0} (1 + t/n)^(-3/4), lr_t = {LR0} * {Q}^(t/n); total path: "
          f"no table {D['budget_no']:.2f}, cooldown {D['budget_cool']:.2f} (= {D['budget_cool'] / D['d0']:.1f} x "
          f"|W_0 - W*|); last length: eta {D['eta_end']:.2e}, lr {D['lr_end']:.2e}")
    for key, text in (("no_gm", "no table, averaged iterate, to W_dagger"),
                      ("no_mean", "no table, averaged iterate, to W* (not drawn)"),
                      ("tb_mean", "SAGA table, to W*"), ("cd_mean", "cooldown without a table, to W*")):
        a = D[key]
        print(f"    {text}: at the end, per seed: " + ", ".join(f"{v:.3g}" for v in a[:, -1])
              + f"; median {np.median(a[:, -1]):.3g}")
    print("    cooldown without a table, distance to W_dagger at the end, per seed: "
          + ", ".join(f"{v:.3g}" for v in D["cd_gm_end"])
          + "; no table, last iterate (not averaged), to W_dagger: " + ", ".join(f"{v:.3g}" for v in D["raw_gm_end"]))
    print(f"    label of the cooldown without a table: frozen {D['frozen']:.2f} from W* (median of the seeds; "
          f"{100 * (D['cd_mean'][:, -1].min() / D['d0']) ** 2:.1f} % to {100 * (D['cd_mean'][:, -1].max() / D['d0']) ** 2:.1f} % "
          f"of the initial squared distance; |W_dagger - W*| = {D['offset']:.2f}: it stops next to W_dagger)")
    print("    run with the table: the SAGA estimator W - phi_j + mean_i phi_i - ybar does not involve the sampled "
          "y_j; in exact arithmetic the iterate stays on the line through W_0 and W*. Computed, per seed: distance to "
          "that line during the first pass at most " + ", ".join(f"{a:.0e}" for a, _, _ in D["tb_line"])
          + "; above 1e-9 from pass " + ", ".join(f"{k:.1f}" for _, k, _ in D["tb_line"])
          + " (rounding, amplified by the normalization); at most " + ", ".join(f"{a:.1e}" for _, _, a in D["tb_line"]))
    print("    first crossing of W* at pass " + ", ".join(f"{c:.2f}" for _, _, c in D["tb_over"])
          + "; largest overshoot beyond W*: " + ", ".join(f"{o:.2f} (pass {k:.1f})" for o, k, _ in D["tb_over"]))
    print("    the same run in exact arithmetic (sign iteration on the line): distance to W* at the end, per seed: "
          + ", ".join(f"{v:.3g}" for v in D["tb_exact"]) + f"; median {np.median(D['tb_exact']):.3g} (in the plane: "
          f"{np.median(D['tb_mean'][:, -1]):.3g}, whose digits depend on the rounding; last length lr {D['lr_end']:.2e})")
    if not D["offsets"]:
        return
    z2, se2 = D["zeros"]["mc2"]
    print(f"(c) check of the sampler, b = 2: Monte Carlo b x offset {2 * np.hypot(*(z2 - m)):.4f} (+- {se2:.4f}), "
          f"enumeration {D['rows'][1]['b_off']:.4f}")
    print("    b: b x |zero of the mean field - W*| (standard error; method); averaged runs (standard error); SAGA "
          "estimator (last length lr)")
    for r in D["rows"]:
        print(f"    b = {r['b']:2d}: {r['b_off']:.4f} (+- {r['se']:.4f}; {r['how']}; zero at ({r['zero'][0]:+.4f}, "
              f"{r['zero'][1]:+.4f})); runs {r['run']:.4f} (+- {r['run_se']:.4f}); SAGA {r['saga']:.1e} "
              f"({r['saga_lr_end']:.1e})")
    worst = max(abs(r["run"] / r["b_off"] - 1.0) for r in D["rows"])
    print(f"    largest relative gap between the averaged runs and the zeros: {worst:.3f}  ({CHAINS} chains of "
          f"{T_BATCH} iterations, average over the second half; SAGA: lr_t = {LR0} b * {Q}^(t b / n), "
          f"{SAGA_PASSES} passes)")
    print(f"    with lr_0 = {LR0} at b = {BATCHES[-1]} the SAGA run stops {D['short'][0]:.3f} from W*: its total path "
          f"is at most {D['short'][1]:.2f}, shorter than |W_0 - W*| = {D['d0']:.2f}")


# ----------------------------------------------------------------------------------------------------------------------
# figure (halo, path_point, panel_plane, fit_window and panel_distances are also used by make_figT11_polar_median.py)
# ----------------------------------------------------------------------------------------------------------------------
def _register(build):
    def reg(state, *artists):
        if build is not None:
            build.add(state, *artists)
        return artists[0] if len(artists) == 1 else artists
    return reg


def halo(ps, width=2.6):
    """Path effect of a label written over marks: a stroke in the surface colour around the glyphs, so that the level
    sets, the points of the cloud and the paths do not run through the text. (A helper that explain_style lacks.)"""
    from matplotlib import patheffects
    return [patheffects.withStroke(linewidth=width, foreground=ps.INK["surface"])]


def path_point(path, x):
    """The first point of a path whose abscissa is closest to x (anchor of a direct label)."""
    return path[int(np.argmin(np.abs(path[:, 0] - x)))]


def panel_plane(ax, D, ps, build=None, states=(1, 1), inset=True, labels=True, ms=5.0, lw=1.6):
    """The plane of W. states: the build states of (the run without a table, with the level sets and W_dagger; the
    run with the table); the cloud, W_0 and W_* are always visible. Equal scales: call fit_window once the figure is
    complete. labels = False leaves the direct labels to the caller."""
    ink = ps.INK
    NO, TB = ps.BLUE, ps.ORANGE
    reg = _register(build)
    s_no, s_tb = states
    y, mean, gm, w0 = D["y"], D["mean"], D["gm"], np.array(W0)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(False)
    ax.plot(y[:, 0], y[:, 1], ls="", marker="o", ms=0.44 * ms, mfc=ink["axis"], mec="none", zorder=1)
    gx, gy, lam = D["grid"]
    cs = ax.contour(gx, gy, lam, levels=LEVELS, colors=[ink["axis"]], linewidths=0.7, zorder=2)
    reg(s_no, *cs.collections)
    # the run without a table: the iterates (thin, light) and their window average (thick)
    reg(s_no, ax.plot(D["path_no"][:, 0], D["path_no"][:, 1], color=es.tint(NO, 0.40), lw=0.5, zorder=3))
    reg(s_no, ax.plot(D["path_avg"][:, 0], D["path_avg"][:, 1], color=NO, lw=lw, zorder=5))
    # the run with the table
    reg(s_tb, ax.plot(D["path_tb"][:, 0], D["path_tb"][:, 1], color=TB, lw=0.75 * lw, zorder=6))
    # where the two runs end: a ring in the colour of the run (around W_dagger and around W_*)
    reg(s_no, ax.plot(*D["path_avg"][-1], ls="", marker="o", ms=2.0 * ms, mfc="none", mec=NO, mew=1.3, zorder=7))
    reg(s_tb, ax.plot(*D["path_tb"][-1], ls="", marker="o", ms=2.0 * ms, mfc="none", mec=TB, mew=1.3, zorder=7))
    # the three points
    ax.plot(*w0, ls="", marker="o", ms=0.8 * ms, mfc=ink["secondary"], mec=ink["surface"], mew=0.8, zorder=8)
    ax.plot(*mean, ls="", marker="o", ms=ms, mfc=ink["primary"], mec=ink["surface"], mew=0.9, zorder=9)
    reg(s_no, ax.plot(*gm, ls="", marker="D", ms=0.84 * ms, mfc=ink["surface"], mec=ink["primary"], mew=1.3, zorder=9))
    ax.set_xlim(WINDOW[0], WINDOW[1])
    ax.set_ylim(WINDOW[2], WINDOW[3])
    ax.set_xlabel("$W_{11}$")
    ax.set_ylabel("$W_{21}$")
    if labels:
        kw = dict(textcoords="offset points", fontsize=es.small(), color=ink["secondary"], zorder=10, linespacing=1.15,
                  path_effects=halo(ps))
        ax.annotate("$W_0$", w0, xytext=(0, 5), ha="center", va="bottom", **kw)
        ax.annotate(r"$W_\ast$ = mean", mean, xytext=(-12, 0.9 * ms + 3.5), ha="center", va="bottom", **kw)
        reg(s_no, ax.annotate(r"$W^\dagger$ = geometric" + "\nmedian", gm, xytext=(1.0 * ms + 4, -(1.0 * ms + 5)),
                              ha="left", va="top", **kw))
        reg(s_no, ax.annotate("no table", path_point(D["path_avg"], 0.80), xytext=(4, -7), ha="left", va="top", **kw))
        reg(s_tb, ax.annotate("SAGA table", path_point(D["path_tb"][:40], 0.80), xytext=(0, 5), ha="center",
                              va="bottom", **kw))
    if inset and D["offsets"]:
        reg(s_no, *zoom_inset(ax, D, ps, ms))
    return ax


def zoom_inset(ax, D, ps, ms):
    """The segment from W_dagger to W_*, enlarged ZOOM times, with the zeros of the mean field for b = 2, 4, 8, 16
    (b = 1 is W_dagger), each labelled by its b. Returns the list of its artists (the inset axes and its header)."""
    ink = ps.INK
    mean, gm = D["mean"], D["gm"]
    c = 0.5 * (mean + gm)
    half_w, half_h = 0.105, 0.19
    shift = 0.030                                     # the segment on the left, the labels of b on its right
    x0, y0 = WINDOW[0] + 0.03, -0.52                  # lower left corner of the inset, in the coordinates of the plane
    axi = ax.inset_axes([x0, y0, 2 * half_w * ZOOM, 2 * half_h * ZOOM], transform=ax.transData)
    axi.set_xlim(c[0] - half_w + shift, c[0] + half_w + shift)
    axi.set_ylim(c[1] - half_h, c[1] + half_h)
    axi.set_xticks([])
    axi.set_yticks([])
    axi.grid(False)
    for s in axi.spines.values():
        s.set_visible(True)
        s.set_color(ink["axis"])
        s.set_linewidth(0.7)
    axi.plot([gm[0], mean[0]], [gm[1], mean[1]], color=ink["axis"], lw=0.7, zorder=1)
    axi.plot(*mean, ls="", marker="o", ms=0.9 * ms, mfc=ink["primary"], mec=ink["surface"], mew=0.8, zorder=5)
    axi.plot(*gm, ls="", marker="D", ms=0.76 * ms, mfc=ink["surface"], mec=ink["primary"], mew=1.2, zorder=5)
    kw = dict(textcoords="offset points", fontsize=es.small(), color=ink["secondary"], zorder=6, va="center")
    for b, dy in ((2, -0.5), (4, -1.5), (8, -1.5), (16, 1.5)):       # dy: the labels of 8 and 16 pushed apart
        z = D["zeros"][b][0]
        axi.plot(*z, ls="", marker="o", ms=0.62 * ms, mfc=ink["surface"], mec=ps.BLUE, mew=1.0, zorder=4)
        axi.annotate(f"{b}", z, xytext=(5.5, dy), ha="left", **kw)
    axi.annotate("$b = 1$", gm, xytext=(6, -0.5), ha="left", **kw)
    axi.annotate(r"$W_\ast$", mean, xytext=(-5.5, 1.0), ha="right", **kw)
    head = ax.annotate(f"zoom ×{ZOOM:g}", (x0, y0 + 2 * half_h * ZOOM), xytext=(0, 3), textcoords="offset points",
                       ha="left", va="bottom", fontsize=es.small(), color=ink["secondary"], zorder=10)
    return [axi, head]


def fit_window(fig, ax):
    """Give the plane the window WINDOW in x and the y range that keeps equal scales in the cell that the layout
    gave to the axes (centred on the middle of WINDOW in y). Call it once the figure is complete."""
    fig.canvas.draw()
    fig.canvas.draw()                                 # constrained layout settles in two passes
    box = ax.get_window_extent()
    half = 0.5 * (WINDOW[1] - WINDOW[0]) * box.height / box.width
    mid = 0.5 * (WINDOW[2] + WINDOW[3])
    ax.set_xlim(WINDOW[0], WINDOW[1])
    ax.set_ylim(mid - half, mid + half)
    return mid - half, mid + half


def panel_distances(ax, D, ps, build=None, states=(1, 1, 1), lw=1.4, control=None):
    """Distances against passes (median over the seeds, band = range): the averaged iterate without a table, to
    W_dagger; the iterate with the table and the iterate of the cooldown without a table, to W_*. states: the build
    states of (the run without a table, the run with the table, the cooldown without a table). control: (colour, line
    style) of the cooldown without a table; by default muted ink and a solid line (the paper figure: grey ink, as
    Figure 4(a) of the paper draws the runs without a table on the cooldown); the lecture figures draw every run
    without a table in blue and pass (ps.BLUE, a dash pattern). The direct labels are left to the caller. Returns the
    three lines, keyed "no", "table", "cooldown"."""
    ink = ps.INK
    NO, TB = ps.BLUE, ps.ORANGE
    CD, cd_ls = control if control is not None else (ink["muted"], "-")
    reg = _register(build)
    p = D["passes"]
    lines = {}
    for name, key, colour, ls, state, z in (("cooldown", "cd_mean", CD, cd_ls, states[2], 3),
                                            ("no", "no_gm", NO, "-", states[0], 4),
                                            ("table", "tb_mean", TB, "-", states[1], 5)):
        a = D[key]
        reg(state, ax.fill_between(p, a.min(axis=0), a.max(axis=0), color=colour, alpha=0.16, lw=0, zorder=2))
        lines[name], = ax.plot(p, np.median(a, axis=0), color=colour, lw=lw, ls=ls, zorder=z)
        reg(state, lines[name])
    ax.set_yscale("log")
    ax.set_xlim(0, PASSES)
    ax.set_xticks(np.arange(0, PASSES + 1, 20))
    ax.set_xlabel("passes")
    return lines


def below_no_table(D, frm=0.4):
    """A level just below the band of the run without a table over the passes after frm x PASSES (for its label)."""
    return float(D["no_gm"][:, D["passes"] > frm * PASSES].min()) / 1.5


def table_label_anchor(D, at=0.45, gap=2.0, below=4.0):
    """(pass, level) under the curve of the run with the table at `at` x PASSES: `gap` passes to its left and a
    factor `below` under its median there (for its label, set right-aligned with its top at that point)."""
    p = D["passes"]
    k = int(np.argmin(np.abs(p - at * PASSES)))
    return p[k] - gap, float(np.median(D["tb_mean"][:, k])) / below


def panel_offsets(ax, D, ps):
    ink = ps.INK
    rows = D["rows"]
    b = np.array([r["b"] for r in rows], dtype=float)
    zero = ax.errorbar(b, [r["b_off"] for r in rows], yerr=[2.0 * r["se"] for r in rows], color=ps.BLUE, lw=1.6,
                       elinewidth=0.9, capsize=0, zorder=3)
    runs = ax.errorbar(b, [r["run"] for r in rows], yerr=[2.0 * r["run_se"] for r in rows], color=ps.BLUE, ls="",
                       marker="o", ms=4.8, mec=ink["surface"], mew=0.8, elinewidth=0.9, capsize=0, zorder=4)
    saga, = ax.plot(b, [r["saga"] for r in rows], color=ps.ORANGE, ls="", marker="s", ms=4.6, mec=ink["surface"],
                    mew=0.8, clip_on=False, zorder=5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(b)
    ax.set_xticklabels([f"{int(v)}" for v in b])
    ax.minorticks_off()
    ax.set_ylim(0.0, 0.5)
    ax.set_xlabel("batch size $b$")
    ax.set_ylabel(r"$b\,\times$ distance of the limit to $W_\ast$")
    ps.legend(ax, handles=[zero, runs, saga], labels=["mean-field zero", "averaged runs", "SAGA table"],
              loc="center right", bbox_to_anchor=(1.0, 0.30), handlelength=1.3, handletextpad=0.5, borderpad=0.1,
              labelspacing=0.25, borderaxespad=0.1)


def draw(D):
    ps = es.setup("paper")
    ink = ps.INK
    fig = es.figure("paper", height=2.45)
    fig.get_layout_engine().set(w_pad=0.03, wspace=0.02)
    gs = fig.add_gridspec(1, 3, width_ratios=(1.72, 1.10, 0.80))
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, k]) for k in range(3))
    panel_plane(ax_a, D, ps)
    es.panel_title(ax_a, "(a) the plane of $W$")
    panel_distances(ax_b, D, ps)                      # the cooldown without a table in grey ink (see panel_distances)
    ax_b.set_ylim(1e-8, 40.0)
    ax_b.set_yticks([1e0, 1e-2, 1e-4, 1e-6, 1e-8])
    ax_b.minorticks_off()
    ax_b.set_ylabel(r"distance to $W_\ast$ or to $W^\dagger$")
    kw = dict(fontsize=es.small(), color=ink["secondary"], zorder=6, linespacing=1.15, ha="right")
    ax_b.text(PASSES - 1, 1.9 * D["frozen"], "no table, same cooldown:\n" + rf"frozen {D['frozen']:.2f} from $W_\ast$",
              va="bottom", **kw)
    ax_b.text(PASSES - 1, below_no_table(D), "no table\n" + r"(to $W^\dagger$)", va="top", **kw)
    ax_b.text(*table_label_anchor(D), "SAGA table,\ncooldown\n" + r"(to $W_\ast$)", va="top", **kw)
    es.panel_title(ax_b, "(b) two limits")
    panel_offsets(ax_c, D, ps)
    es.panel_title(ax_c, r"(c) $b$ × offset")
    lo, hi = fit_window(fig, ax_a)
    print(f"(a) window drawn (equal scales): [{WINDOW[0]}, {WINDOW[1]}] x [{lo:.3f}, {hi:.3f}]; inset: zoom x{ZOOM:g}")
    es.save(fig, NAME, "paper")


def main():
    D = data()
    print_all(D)
    draw(D)


if __name__ == "__main__":
    main()
