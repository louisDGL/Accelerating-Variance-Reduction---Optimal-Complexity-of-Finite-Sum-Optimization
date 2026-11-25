"""Figure T14 of the lecture (paper/talk/figures/figT14_estimator_error): with a SAGA table the error of the estimator
vanishes with the iterate; the error of the sampled gradient stays at its variance at the minimizer. Explanatory
figure, not in the paper. It explains the sentence of the introduction of paper/paper_saga_to_muon.tex ("an error that
vanishes at the minimizer"), Proposition 3(c), and why, in (M1) and (M5), only a finite-sum estimator lets fixed
lengths on a cooldown converge ("the cooldown outruns an estimator that does not vanish at W_*").

What is shown (log scale, against passes): the mean over the index j, uniform on the n indices, of
    ||omega_j - grad F(W_t)||_F^2
at the states (W_t, table) of ONE run of the recipe (4) of the paper with the SAGA table, for two estimators:
    orange   omega_j = zbar + lambda W_t + (r_j(W_t) - rtab_j) x_j^T      the SAGA estimator that drives the run;
    blue     omega_j = lambda W_t + r_j(W_t) x_j^T                        the batch-one gradient (no table), evaluated
                                                                         AT THE SAME STATES (it does not drive a run);
and, dashed, ||grad F(W_t)||_F^2 in the same units. Two panels, the same three curves:
  (a) passes 1 to 9, every state: while the table is stale (passes 1.2 to 6.9 of this run) the SAGA estimator has the
      LARGER error of the two;
  (b) passes 1 to 80: the error of the SAGA estimator decreases with the squared distance to W_*, the error of the
      batch-one gradient settles at its variance at W_* (the two estimators at every state, the gradient at the
      states of whole passes, so that its dashes stay readable).

Data. The run is a cached run of experiment M9, part C (muon/results/M9_summary.json, C/ls-cond/deep/"polar, no
momentum": least squares, n = 200, K x d = 8 x 40, kappa = 1e4, batch one, exact polar factor, lr_t = 0.3 * 0.8^(t/n),
seed 10; cache muon/results/M9_law.pkl). Its states are recomputed with the code of the project
(muon/lab/exact.collect_states) every 20 iterations (0.1 pass), and the script stops unless the relative squared
distance of the recomputed run equals the cached one at every cached record of the window. The means are exact: the n
estimators are enumerated, and compared with a closed form (recipe_data.estimator_error); the script stops if the two
disagree. At the first state (pass 1: the table has just been filled at W_0) the error of the SAGA estimator is
exactly 0 and is not drawn on the log axis.

Status. Measured exactly along one run (one seed, one instance, one data draw): no statistical statement. Proved, and
visible here: the SAGA estimator is exact for every index when the iterate and the table are at W_* (Proposition
3(c)), whereas the batch-one gradient keeps the variance of the f_j' at W_*. Not claimed: that the error of the
estimator is small next to the gradient (it stays about five times larger in Frobenius norm along this run: the
estimator vanishes, it is not accurate), nor why the run still decreases.

Progressive build (states s1-s3): s1 the batch-one gradient; s2 adds the SAGA estimator; s3 adds the gradient.

The script prints every number it plots.
Run:    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/scripts/make_figT14_estimator_error.py
Output: paper/talk/figures/figT14_estimator_error.pdf and .png, and the states _s1 ... _s3
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import explain_style as es  # noqa: E402
import recipe_data as rd  # noqa: E402

NAME = "figT14_estimator_error"
SEED = 10                   # the first seed of part C
PASSES = 80                 # window of the figure
EVERY = 20                  # iterations between two states (n / 10: the records of the cached run)
ZOOM = 9                    # last pass of panel (a)


# ----------------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------------
def data():
    m9, cache = rd.load()
    rd.check_paper(m9, cache)
    D = rd.estimator_error(m9, cache, label=rd.POLAR, seed=SEED, passes=PASSES, every=EVERY)
    worse = D["saga"] > D["sgd"]
    D["worse"] = (float(D["passes"][worse].min()), float(D["passes"][worse].max()))
    D["peak"] = (float(D["passes"][D["saga"].argmax()]), float(D["saga"].max()))
    tail = D["passes"] >= 0.5 * PASSES
    D["ratio_tail"] = D["saga"][tail] / D["grad2"][tail]
    if abs(D["sgd"][-1] / D["var_star"] - 1.0) > 1e-3:
        raise SystemExit("the error of the batch-one gradient does not end at its variance at the minimizer")
    return D


def print_all(D):
    print(f"run: {rd.INSTANCE}, '{D['label']}', seed {D['seed']}, cfg {D['cfg']}; n = {D['n']}, K x d = {D['K']} x "
          f"{D['d']}, lambda = {D['lam']:.4g}")
    print(f"recomputed run against the cache: {D['checked']} records compared, largest relative difference of the "
          f"squared distance {D['worst']:.1e}")
    print(f"states every {EVERY} iterations, passes {D['passes'][0]:g} to {D['passes'][-1]:g} ({len(D['passes'])} "
          "states); mean over the n indices of ||omega_j - grad F(W_t)||_F^2 (exact)")
    print("  passes     SAGA table   batch-one grad.   ||grad F||_F^2   rel. sq. distance")
    for k in list(range(0, 60, 10)) + list(range(60, len(D["passes"]), 50)):
        print(f"  {D['passes'][k]:6.1f}   {D['saga'][k]:12.4e}   {D['sgd'][k]:12.4e}   {D['grad2'][k]:12.4e}   "
              f"{D['rel'][k]:12.4e}")
    print(f"variance of the batch-one gradient at W_*: {D['var_star']:.4f}; its error at the last state: "
          f"{D['sgd'][-1]:.4f}; at the first state: {D['sgd'][0]:.4f}")
    print(f"the SAGA estimator has the larger error from pass {D['worse'][0]:g} to pass {D['worse'][1]:g} (stale "
          f"table); its largest error {D['peak'][1]:.3g} at pass {D['peak'][0]:g}")
    r = D["ratio_tail"]
    print(f"passes {0.5 * PASSES:g} to {PASSES}: error of the SAGA estimator / ||grad F||_F^2: median {np.median(r):.1f}, "
          f"range {r.min():.1f} to {r.max():.1f} (squared Frobenius norms; square root of the median "
          f"{np.sqrt(np.median(r)):.1f}); not printed on the figure")


# ----------------------------------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------------------------------
def draw(D):
    ps = es.setup("talk")
    ink = ps.INK
    TB, NO = ps.ORANGE, ps.BLUE                      # SAGA table: orange; no table: blue, as in the other figures of
    #                                                  the lecture (Figure 4(a) of the paper and figE5 draw it in grey ink)
    GRAD = dict(color=ink["primary"], lw=1.1, ls=(0, (4.0, 2.0)), zorder=2)
    fig = es.figure("talk", height=2.7)
    fig.get_layout_engine().set(w_pad=0.03, wspace=0.03)
    gs = fig.add_gridspec(1, 2, width_ratios=(1.0, 2.05))
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    b = es.Build()
    p = D["passes"]
    pos = D["saga"] > 0                              # the first state has an exact table: error 0, not on a log axis
    whole = np.abs(p - np.round(p)) < 1e-9           # the states at whole passes
    kw = dict(fontsize=es.small(), color=ink["secondary"], textcoords="offset points", linespacing=1.2)

    # (a) the first passes, every state
    z = p <= ZOOM
    b.add(1, ax_a.plot(p[z], D["sgd"][z], color=NO, lw=1.8, zorder=3))
    b.add(2, ax_a.plot(p[z & pos], D["saga"][z & pos], color=TB, lw=1.8, zorder=4))
    b.add(3, ax_a.plot(p[z], D["grad2"][z], **GRAD))
    k = int(np.argmin(np.abs(p - 5.6)))
    b.add(1, ax_a.annotate("no table", (2.9, 0.33), xytext=(0, 0), ha="left", va="center", **kw))
    b.add(2, ax_a.annotate("SAGA table,\nstale at first", (D["peak"][0], D["peak"][1]), xytext=(9, 5), ha="left",
                           va="center", **kw))
    b.add(3, ax_a.annotate(r"$\|\nabla F(W_t)\|_F^2$", (p[k], D["grad2"][k]), xytext=(-4, -7), ha="right",
                           va="top", **kw))
    ax_a.set_yscale("log")
    ax_a.set_xlim(1, ZOOM)
    ax_a.set_ylim(2e-3, 60.0)
    ax_a.set_yticks([1e-2, 1e-1, 1.0, 10.0])
    ax_a.minorticks_off()
    ax_a.set_xticks([1, 3, 5, 7, 9])
    ax_a.set_xlabel("passes")
    ax_a.set_ylabel(r"mean over $j$ of $\|\omega_j - \nabla F(W_t)\|_F^2$")
    es.panel_title(ax_a, "(a) first passes")

    # (b) the whole window: every state for the two estimators, the states at whole passes for the gradient
    b.add(1, ax_b.plot(p, D["sgd"], color=NO, lw=1.8, zorder=3))
    b.add(2, ax_b.plot(p[pos], D["saga"][pos], color=TB, lw=1.6, zorder=4))
    b.add(3, ax_b.plot(p[whole], D["grad2"][whole], **GRAD))
    b.add(1, ax_b.annotate("batch-one gradient (no table):\nstays at its variance at $W_\\ast$",
                           (PASSES, D["var_star"]), xytext=(-2, 5), ha="right", va="bottom", **kw))
    # in the wedge between the blue line and the orange curve, its lower left corner just above the orange curve
    # (Proposition 3(c): the estimator is exact when the iterate AND the table are at W_*)
    lab = ax_b.annotate("SAGA table: vanishes as $W_t$\nand the table approach $W_\\ast$", (PASSES, D["var_star"]),
                        xytext=(-2, -9), ha="right", va="top", **kw)
    b.add(2, lab)
    k = int(np.argmin(np.abs(p - 36.0)))
    b.add(3, ax_b.annotate(r"$\|\nabla F(W_t)\|_F^2$", (p[k], D["grad2"][k]), xytext=(-8, -8), ha="right",
                           va="top", **kw))
    ax_b.set_yscale("log")
    ax_b.set_xlim(0, PASSES)
    ax_b.set_ylim(1e-17, 3e3)                       # room above the blue line for its label
    ax_b.set_yticks([1e-16, 1e-12, 1e-8, 1e-4, 1.0])
    ax_b.minorticks_off()
    ax_b.set_xticks([1, 20, 40, 60, 80])
    ax_b.set_xlabel("passes")
    es.panel_title(ax_b, f"(b) {PASSES} passes")
    es.save(fig, NAME, "talk", build=b,
            note=(rf"least squares, $n = {D['n']}$, batch one, one run; both estimators evaluated at the states of the "
                  "table run"))


def main():
    D = data()
    print_all(D)
    draw(D)


if __name__ == "__main__":
    main()
