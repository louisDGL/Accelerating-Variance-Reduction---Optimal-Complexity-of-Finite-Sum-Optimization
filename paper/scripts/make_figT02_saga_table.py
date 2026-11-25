"""Figure T02 of the lecture (paper/talk/figures/figT02_saga_table): SGD and SAGA both step along an unbiased estimate
of the gradient, but the error of SAGA's estimate vanishes at the minimizer; with the same constant step SGD stalls at
a noise floor and SAGA converges linearly. Explanatory figure, not in the paper. It introduces the two properties of
the estimator named in the abstract and in Table 2 of paper/paper_saga_to_muon.tex (unbiased; error that vanishes at
the minimizer) and the algorithm of Section 2.1 ("SAGA").

The toy (an exact instance of Section 2.1, in the plane; nothing is drawn by hand):
    F(theta) = (1/n) sum_i f_i(theta),  f_i(theta) = log(1 + exp(-y_i x_i^T theta)) + (lam/2) |theta|^2,
    n = 50, d = 2: ridge logistic regression drawn by the generator of the project,
    vrlab.problems.make_logistic(50, 2, kappa = 20, alpha = 2, seed = 2)
(Gaussian features with covariance eigenvalues 1 and 1/4, rescaled so that max_i |x_i| = 1; noisy labels). Each f_i is
R^2-smooth with R^2 = max_i |x_i|^2 / 4 + lam and F is mu-strongly convex with mu = lam, kappa = R^2 / mu = 20: the
constants of Proposition 5.9 as the paper uses them. The minimizer eta_* is computed by the Newton solver of the
project (vrlab.solvers.solve_optimum; the script stops if |F'(eta_*)| > 1e-14).

Two algorithms from theta_0 = 0, with the SAME sampled indices (i uniform, numpy Generator seeded by the run) and the
SAME constant step gamma = 1/(4 R^2), the step of Proposition 5.9:
    SGD    theta <- theta - gamma f_i'(theta)                                   (omega_i = f_i'(theta));
    SAGA   theta <- theta - gamma omega_i,  omega_i = f_i'(theta) - z_i + (1/n) sum_j z_j,  then z_i <- f_i'(theta);
           the table starts at z_i = f_i'(theta_0).
The loops are written here because the path and the table are needed at every iteration; they are checked against
vrlab.algorithms.sgd and vrlab.algorithms.saga on this problem (same seeds; the script stops if the recorded squared
distances differ), and the exact error of SAGA's estimate is checked against the instrumentation of the project
(vrlab.instrumentation.saga_state_stats: Omega - |F'|^2).

Panels:
  (a) the plane: level sets of F, the minimizer eta_*, the two paths of the run of seed PATH_SEED (every iterate).
      The cloud of SGD is not centred at eta_* (the mean of its iterates over the second half of the run is printed,
      about (-0.06, +0.14) for a spread of about 0.4): this is what the run does, the figure does not explain it.
      Two causes are possible and neither is checked here: the stationary mean of constant-step SGD on a
      non-quadratic F is off the minimizer by O(gamma) (Dieuleveut, Durmus and Bach, Ann. Statist. 48, 2020), and a
      time average over one run of 1000 correlated iterates is itself noisy;
  (b) squared distance |theta_t - eta_*|^2 against passes (gradient evaluations / n), log scale: median over the
      SEEDS runs at each record and range over the runs (band);
  (c) the squared error of the estimate, E_i |omega_i - F'(theta_t)|^2, computed EXACTLY at each recorded state by
      enumerating the n indices (median and range over the same runs), and its value at the minimizer for SGD,
      (1/n) sum_i |f_i'(eta_*)|^2 (the gradient noise at eta_*). For SAGA it is 0 at theta_0 (the table is exact
      there: that record cannot be shown on a log axis), rises during the first pass and then decreases linearly.
Passes count gradient evaluations: SAGA pays n of them to fill its table (the convention of the project), so its
curves start at one pass and those of SGD at zero.

Status. Proved (classical): both estimates are unbiased, E_i omega_i = F'(theta) (checked to rounding at every
recorded state); SAGA converges linearly IN EXPECTATION with this step (Proposition 5.9 of Bach's book: the Lyapunov
function D + 3 gamma^2 M contracts in expectation; Section 2.1 of the paper); constant-step SGD keeps an expected squared
distance of order gamma times the gradient noise at the minimizer (classical). Measured on one instance (20 seeds):
the level of the floor of SGD and the speed of SAGA; no rate and no ratio is claimed or printed.

Progressive build (states s1-s3): s1 panel (a) with the path of SGD and the question; s2 adds the path of SAGA;
s3 adds panels (b) and (c).

Helper not in explain_style: none. The axes of panels (b) and (c) are registered in the build as artists; panel (a)
keeps equal scales with adjustable="datalim" so that it fills its cell, and its labels are left out of the layout.
The script prints every number it plots. Nothing is read from the results of the project.
Run:    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/scripts/make_figT02_saga_table.py
Output: paper/talk/figures/figT02_saga_table.pdf and .png, and the states _s1 ... _s3
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import explain_style as es  # noqa: E402
from vrlab import algorithms, instrumentation, problems, solvers  # noqa: E402

NAME = "figT02_saga_table"
N, DIM = 50, 2
KAPPA, ALPHA, DATA_SEED = 20.0, 2.0, 2    # arguments of vrlab.problems.make_logistic (the others: defaults)
PASSES = 40                  # iterations / n of each run (SAGA: plus the pass that fills the table)
RECORD = 5                   # the runs are recorded every RECORD iterations (10 records per pass)
SEEDS = tuple(range(20))     # sampling seeds of panels (b) and (c)
PATH_SEED = 0                # the run drawn in panel (a)
GRID = 241                   # points per side of the grid of the level sets
A_SHIFT = 0.2                # panel (a): the window is centred this far (in units of its height) right of the paths
N_LEVELS = 9                 # level sets F = F_* + (F(theta_0) - F_*) (k / 6)^2, k = 1 .. N_LEVELS


# ----------------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------------
def instance():
    prob = problems.make_logistic(N, DIM, KAPPA, alpha=ALPHA, seed=DATA_SEED)
    _, info = solvers.solve_optimum(prob)
    if info["grad_norm"] > 1e-14:
        raise SystemExit(f"the minimizer is not computed to rounding: |F'| = {info['grad_norm']:.1e}")
    return prob, info


def exact_error(prob, theta, z=None, zbar=None):
    """E_i |omega_i - F'(theta)|^2 by enumeration of the n indices, and the bias |mean_i omega_i - F'(theta)| (F'
    computed by prob.grad, not as the mean of the rows). z is None: SGD; otherwise SAGA with the table z and the
    running mean zbar that the algorithm uses."""
    G = prob.all_grads(theta)
    omega = G if z is None else G - z + zbar[None, :]
    Fg = prob.grad(theta)
    dev = omega - Fg[None, :]
    return float(np.einsum("ij,ij->", dev, dev) / prob.n), float(np.linalg.norm(omega.mean(axis=0) - Fg))


def run(prob, method, seed, gamma, keep_path=False):
    """SGD or SAGA with the constant step gamma, indices rng.integers(0, n, size = PASSES n) (the stream of
    vrlab.algorithms for these lengths). SAGA follows vrlab.algorithms.saga operation by operation (running mean of
    the table, recomputed every n iterations). Records every RECORD iterations: gradient evaluations, squared
    distance to eta_*, exact error and bias of the estimate at the recorded state."""
    n = prob.n
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=PASSES * n)
    theta = np.zeros(prob.d)
    saga = method == "SAGA"
    z = prob.all_grads(theta) if saga else None
    zbar = z.mean(0) if saga else None
    ge = n if saga else 0
    rec = {"ge": [], "D": [], "err": [], "bias": []}
    path = [theta.copy()] if keep_path else None

    def record():
        v = theta - prob.theta_star
        e, b = exact_error(prob, theta, z, zbar)
        rec["ge"].append(ge)
        rec["D"].append(float(v @ v))
        rec["err"].append(e)
        rec["bias"].append(b)

    record()
    for t, i in enumerate(idx, start=1):
        gi = prob.grad_i(theta, i)
        if saga:
            omega = gi + zbar - z[i]
            theta -= gamma * omega
            zbar += (gi - z[i]) / n
            z[i] = gi
            if t % n == 0:
                zbar = z.mean(0)
        else:
            theta -= gamma * gi
        ge += 1
        if keep_path:
            path.append(theta.copy())
        if t % RECORD == 0:
            record()
    out = {k: np.array(v, dtype=float) for k, v in rec.items()}
    out["passes"] = out.pop("ge") / n
    if keep_path:
        out["path"] = np.array(path)
    return out


def check_against_vrlab(prob, gamma, mine):
    """The loops above against vrlab.algorithms (squared distances at the records) and the exact error of SAGA's
    estimate against vrlab.instrumentation (seed PATH_SEED). Returns the largest differences; stops if they differ."""
    theta0 = np.zeros(prob.d)
    D0 = mine["SGD"][0]["D"][0]
    worst = {}
    for k, seed in enumerate(SEEDS):
        ref = {"SGD": algorithms.sgd(prob, theta0, gamma0=gamma, decay=False, seed=seed, passes=PASSES,
                                     record_every=RECORD, record_F=False),
               "SAGA": algorithms.saga(prob, theta0, gamma=gamma, seed=seed, passes=PASSES, record_every=RECORD,
                                       record_F=False)}
        for m in ("SGD", "SAGA"):
            a, b = mine[m][k]["D"], np.asarray(ref[m]["D"], dtype=float)
            if a.shape != b.shape or not np.allclose(np.asarray(ref[m]["grad_evals"]) / prob.n, mine[m][k]["passes"]):
                raise SystemExit(f"{m}, seed {seed}: the records do not fall at the same gradient counts as vrlab")
            diff = float(np.max(np.abs(a - b) / (np.abs(b) + 1e-30 * D0)))
            worst[m] = max(worst.get(m, 0.0), diff)
            if diff > 1e-6:
                raise SystemExit(f"{m}, seed {seed}: squared distances differ from vrlab.algorithms ({diff:.1e})")
    k = SEEDS.index(PATH_SEED)
    r = algorithms.saga(prob, theta0, gamma=gamma, seed=PATH_SEED, passes=PASSES, record_every=RECORD, record_F=False,
                        instrument_every=RECORD)
    ref_err = np.array([s["Omega"] - s["grad_norm2"] for s in r["instr"]])       # states t = 0, RECORD, ...
    my_err = mine["SAGA"][k]["err"][:len(ref_err)]
    big = ref_err > 1e-12 * ref_err.max()                # Omega - |F'|^2 cancels below that
    worst["instr"] = float(np.max(np.abs(my_err[big] / ref_err[big] - 1.0)))
    if worst["instr"] > 1e-6:
        raise SystemExit(f"exact error of SAGA's estimate differs from vrlab.instrumentation ({worst['instr']:.1e})")
    return worst


def data():
    prob, info = instance()
    gamma = 1.0 / (4.0 * prob.R2)
    runs = {m: [run(prob, m, s, gamma, keep_path=(s == PATH_SEED)) for s in SEEDS] for m in ("SGD", "SAGA")}
    worst = check_against_vrlab(prob, gamma, runs)
    bias = {m: max(float(r["bias"].max()) for r in runs[m]) for m in runs}
    grad_scale = float(np.linalg.norm(prob.grad(np.zeros(prob.d))))
    if max(bias.values()) > 1e-12 * grad_scale:
        raise SystemExit(f"an estimate is biased beyond rounding: {bias}")
    stats = {}
    for m in runs:
        D = np.array([r["D"] for r in runs[m]])
        E = np.array([r["err"] for r in runs[m]])
        stats[m] = {"passes": runs[m][0]["passes"], "D": D, "err": E,
                    "D_med": np.median(D, axis=0), "D_lo": D.min(axis=0), "D_hi": D.max(axis=0),
                    "err_med": np.median(E, axis=0), "err_lo": E.min(axis=0), "err_hi": E.max(axis=0)}
    k = SEEDS.index(PATH_SEED)
    paths = {m: runs[m][k]["path"] for m in runs}
    ts = prob.theta_star
    # window of panel (a): the vertical range contains theta_0, eta_* and the two paths; the horizontal range is set
    # by the shape of the panel (equal scales), the grid of the level sets is wide enough for any panel up to 3:2
    pts = np.vstack([paths["SGD"], paths["SAGA"], ts[None, :]])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pad = 0.09 * float(hi[1] - lo[1])
    ylim = (float(lo[1]) - 2.3 * pad, float(hi[1]) + pad)
    xmid = 0.5 * float(lo[0] + hi[0]) + A_SHIFT * (ylim[1] - ylim[0])     # the paths sit left of the centre
    half = 0.85 * (ylim[1] - ylim[0])
    xs = np.linspace(xmid - half, xmid + half, GRID)
    ys = np.linspace(ylim[0], ylim[1], GRID)
    XX, YY = np.meshgrid(xs, ys)
    FF = prob.F_batch(np.stack([XX.ravel(), YY.ravel()], axis=1)).reshape(XX.shape)
    F0 = prob.F(np.zeros(prob.d))
    levels = prob.F_star + (F0 - prob.F_star) * (np.arange(1, N_LEVELS + 1) / 6.0) ** 2
    return {"prob": prob, "info": info, "gamma": gamma, "stats": stats, "paths": paths, "worst": worst, "bias": bias,
            "noise_star": float(np.einsum("ij,ij->", prob.grads_star, prob.grads_star) / prob.n),
            "D0": float(ts @ ts), "window": (xmid, ylim), "grid": (xs, ys, FF), "levels": levels, "F0": F0,
            "hess_eigs": np.linalg.eigvalsh(prob.hessian(ts))}


def print_all(D):
    prob, st = D["prob"], D["stats"]
    ts = prob.theta_star
    print(f"instance: {prob.name}; n = {prob.n}, d = {prob.d}")
    print(f"   R^2 = {prob.R2:.6f}, mu = lam = {prob.mu:.6f}, kappa = R^2 / mu = {prob.kappa:.4f}; "
          f"gamma = 1/(4 R^2) = {D['gamma']:.4f}")
    print(f"   eta_* = ({ts[0]:+.5f}, {ts[1]:+.5f}), |F'(eta_*)| = {D['info']['grad_norm']:.1e}, "
          f"|theta_0 - eta_*|^2 = {D['D0']:.4f}; eigenvalues of F''(eta_*) / lam: "
          f"{D['hess_eigs'][0] / prob.lam:.3f}, {D['hess_eigs'][1] / prob.lam:.3f}")
    print(f"   gradient noise at eta_*: (1/n) sum_i |f_i'(eta_*)|^2 = {D['noise_star']:.5f}")
    w = D["worst"]
    print(f"checks: squared distances against vrlab.algorithms, largest relative difference over {len(SEEDS)} seeds: "
          f"SGD {w['SGD']:.1e}, SAGA {w['SAGA']:.1e}; exact error of SAGA's estimate against "
          f"vrlab.instrumentation (seed {PATH_SEED}): {w['instr']:.1e}")
    print(f"        bias |mean_i omega_i - F'(theta)|, largest over all records and seeds: SGD {D['bias']['SGD']:.1e}, "
          f"SAGA {D['bias']['SAGA']:.1e}")
    xmid, ylim = D["window"]
    print(f"(a) window: centred at theta_1 = {xmid:+.3f}, theta_2 from {ylim[0]:+.3f} to {ylim[1]:+.3f}, equal scales; "
          f"level sets F - F_* = "
          + ", ".join(f"{v - prob.F_star:.4f}" for v in D["levels"]) + f" (F(theta_0) - F_* = {D['F0'] - prob.F_star:.4f})")
    for m in ("SGD", "SAGA"):
        p = D["paths"][m]
        d2 = ((p - ts) ** 2).sum(axis=1)
        half_run = len(d2) // 2
        off = p[half_run:].mean(axis=0) - ts
        print(f"    path of {m} (seed {PATH_SEED}, {len(p) - 1} iterations): final squared distance {d2[-1]:.3e}; "
              f"over the second half: mean {d2[half_run:].mean():.3e}, max {d2[half_run:].max():.3e}, mean of the "
              f"iterates minus eta_* ({off[0]:+.2e}, {off[1]:+.2e})")
    for m in ("SGD", "SAGA"):
        s = st[m]
        p = s["passes"]
        sel = [int(np.argmin(np.abs(p - v))) for v in (p[0] + 0.1, p[0] + 1, 10, 20, 30, p[-1])]
        print(f"(b) {m}: |theta_t - eta_*|^2, median [min, max] over {len(SEEDS)} runs at passes "
              + "; ".join(f"{p[j]:.1f}: {s['D_med'][j]:.2e} [{s['D_lo'][j]:.1e}, {s['D_hi'][j]:.1e}]" for j in sel))
        print(f"(c) {m}: E_i|omega_i - F'|^2, median [min, max] at the same passes "
              + "; ".join(f"{p[j]:.1f}: {s['err_med'][j]:.2e} [{s['err_lo'][j]:.1e}, {s['err_hi'][j]:.1e}]"
                          for j in sel))
    s = st["SGD"]
    tail = s["passes"] >= 0.5 * PASSES
    print(f"    SGD over the second half of the runs: median of the squared distance {np.median(s['D'][:, tail]):.3e} "
          f"({np.median(s['D'][:, tail]) / D['D0']:.3f} of the initial one); median of the error of the estimate "
          f"{np.median(s['err'][:, tail]):.5f} (at eta_*: {D['noise_star']:.5f})")


# ----------------------------------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------------------------------
def draw(D):
    ps = es.setup("talk")
    ink = ps.INK
    COL = {"SGD": ps.ALGO["SGD"], "SAGA": ps.ALGO["SAGA"]}       # the colours of the two algorithms in the project
    prob, st = D["prob"], D["stats"]
    ts = prob.theta_star
    fig = es.figure("talk", height=2.6)
    fig.get_layout_engine().set(w_pad=0.03, wspace=0.03)
    gs = fig.add_gridspec(1, 3, width_ratios=(1.3, 1.0, 1.0))
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, k]) for k in range(3))
    b = es.Build()
    note_kw = dict(textcoords="offset points", fontsize=es.small(), color=ink["secondary"])

    # (a) the plane
    xmid, ylim = D["window"]
    xs, ys, FF = D["grid"]
    es.diagram_axes(ax_a, (xmid - 1.0, xmid + 1.0), ylim)
    ax_a.set_aspect("equal", adjustable="datalim")               # the panel fills its cell; theta_1 range follows
    ax_a.contour(xs, ys, FF, levels=D["levels"], colors=[ink["grid"]], linewidths=0.8, zorder=1)
    p = D["paths"]["SGD"]
    ax_a.plot(p[:, 0], p[:, 1], color=COL["SGD"], lw=0.5, zorder=3, label="SGD", solid_joinstyle="round")
    p = D["paths"]["SAGA"]
    b.add(2, ax_a.plot(p[:, 0], p[:, 1], color=COL["SAGA"], lw=1.3, zorder=4, label="SAGA", solid_joinstyle="round"))
    ax_a.plot([0.0], [0.0], ls="", marker="o", ms=5.0, mfc=ink["surface"], mec=ink["primary"], mew=1.2, zorder=6)
    ax_a.plot([ts[0]], [ts[1]], ls="", marker="o", ms=5.0, mfc=ink["primary"], mec=ink["surface"], mew=1.0, zorder=7)
    labels = [
        ax_a.annotate(r"start $\theta_0$", (0.0, 0.0), xytext=(0, -8), ha="center", va="top", **note_kw),
        ax_a.annotate(r"minimizer $\eta_\ast$", (ts[0], ts[1]), xytext=(30, 10), ha="left", va="center",
                      zorder=8,
                      arrowprops=dict(arrowstyle="-", color=ink["secondary"], lw=0.7, shrinkA=1.5, shrinkB=3.5),
                      **note_kw),
        ax_a.annotate("SGD: why does\nit stall?", (ts[0], ts[1]), xytext=(16, -40), ha="left", va="top",
                      linespacing=1.15, **note_kw)]
    for t in labels:
        t.set_in_layout(False)                                   # they lie inside the panel, which fills its cell
    es.panel_title(ax_a, "(a) in the plane")
    b.legend(ax_a, loc="lower right", handlelength=1.3, borderaxespad=0.0, labelspacing=0.25)

    # (b) squared distance, (c) error of the estimate
    for ax, key in ((ax_b, "D"), (ax_c, "err")):
        for m in ("SGD", "SAGA"):
            s = st[m]
            first = 1 if (m, key) == ("SAGA", "err") else 0          # SAGA's estimate is exact at theta_0 (error 0)
            x = s["passes"][first:]
            ax.fill_between(x, s[key + "_lo"][first:], s[key + "_hi"][first:], color=COL[m], alpha=0.18, lw=0,
                            zorder=2)
            ax.plot(x, s[key + "_med"][first:], color=COL[m], lw=1.6, zorder=4)
        ax.set_yscale("log")
        ax.set_xlim(0, PASSES + 1)
        ax.set_xticks([0, 10, 20, 30, 40])
        ax.set_ylim(1e-27, 1e5)             # room above the band of SGD in (b) for its label (the band dips to 1e-5)
        ax.set_yticks([1e-24, 1e-18, 1e-12, 1e-6, 1.0])
        ax.minorticks_off()
        ax.set_xlabel("passes")
    ax_c.plot([0.0, PASSES], [D["noise_star"]] * 2, color=ink["primary"], lw=0.9, ls=(0, (4.0, 2.0)), zorder=5)
    tail = st["SGD"]["passes"] >= 0.5 * PASSES
    ax_b.annotate("SGD: a floor", (PASSES, st["SGD"]["D_hi"][tail].max()), xytext=(0, 3), ha="right", va="bottom",
                  **note_kw)
    ax_b.annotate("SAGA (its table\ncosts the\nfirst pass)", (1.0, 1e-27), xytext=(3, 5), ha="left", va="bottom",
                  linespacing=1.15, **note_kw)
    ax_c.annotate("SGD: stays at\n" + r"the noise at $\eta_\ast$", (PASSES, D["noise_star"]), xytext=(0, -6),
                  ha="right", va="top", linespacing=1.15, **note_kw)
    ax_c.annotate("SAGA:\nit vanishes", (1.0, 1e-27), xytext=(3, 5), ha="left", va="bottom", linespacing=1.15,
                  **note_kw)
    es.panel_title(ax_b, r"(b) $\|\theta_t - \eta_\ast\|^2$")
    es.panel_title(ax_c, r"(c) $E_i\|\omega_i - F'\|^2$")
    b.add(3, ax_b, ax_c)
    es.save(fig, NAME, "talk", build=b,
            note=(rf"toy: ridge logistic, $n = {N}$, $d = {DIM}$, $\kappa = {KAPPA:g}$; step $1/(4R^2)$, same indices; "
                  rf"(a) one run, (b, c) median of {len(SEEDS)}"))


def main():
    D = data()
    print_all(D)
    draw(D)


if __name__ == "__main__":
    main()
