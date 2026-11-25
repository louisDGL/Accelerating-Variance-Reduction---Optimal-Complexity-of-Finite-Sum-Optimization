"""Figure E5 of the appendix of explanatory figures of the paper (paper/figures/figE5_recipe_table): why the recipe of
Section 3.3 of paper/paper_saga_to_muon.tex (fixed lengths on a geometric cooldown, equation (4)) needs the SAGA
table. Explanatory figure; it adds no result to the paper: it recuts the cached runs behind Figure 4(a) and (M5).
It is the paper version of the lecture figures figT14_estimator_error, figT15_schedule_law and figT16_same_schedule,
drawn with the same data functions (paper/scripts/recipe_data.py).

Instance of the three panels: least squares, conditioning-limited (n = 200, K x d = 8 x 40, kappa = 1e4), batch one;
cached runs of experiment M9, part C (muon/results/M9_summary.json, C/ls-cond; cache muon/results/M9_law.pkl), seeds
10..14, every seed drawn as a thin line in (b) and (c). Colours as in Figure 4(a): SAGA table in orange (exact polar
factor solid, ns5 dashed, ns5 with eps = 1e-30 dotted), no table in grey.

Panels.
  (a) error of the estimator   exact mean over the index j of ||omega_j - grad F(W_t)||_F^2 at the states of ONE
                               cached run with the table (exact polar factor, lr_t = 0.3 * 0.8^(t/n), seed 10; states
                               recomputed with muon/lab/exact.collect_states every 0.1 pass, passes 1 to 40, and
                               checked against the cached trace), for the SAGA estimator of the run and for the
                               batch-one gradient (no table) evaluated at the same states; dashed, ||grad F(W_t)||_F^2
                               at the states of whole passes. Enumeration of the n indices, checked against a closed
                               form. At pass 1 the table is exact (error 0, not on the log axis); from pass 1.2 to
                               pass 6.9 the stale table gives the SAGA estimator the larger error of the two.
  (b) schedule law             parametric plot of the relative squared distance against lr_t^2 (decreasing to the
                               right): table with the exact polar factor (q = 0.8, lr_0 = 0.3), with ns5 + Nesterov
                               0.9 at q = 0.8, lr_0 = 0.1 (eps = 1e-7 and eps = 1e-30); without a table, the minibatch
                               gradient and MARS-M at the schedule selected for Muon's recipe (q = 0.75, lr_0 = 0.3).
                               A guide of slope one (not a fit). The axis stops at lr_t^2 = 1e-40.
  (c) same schedule            relative squared distance against passes: ns5 + Nesterov 0.9 on lr_t = 0.3 * 0.75^(t/n)
                               with the SAGA table and with the minibatch gradient, two cached configurations that the
                               script compares key by key (they differ by the estimator only), the same five seeds.
                               One seed of five of the table runs leaves the schedule (the tuned point sits next to a
                               cliff in q); it is drawn.

Checks (the script stops if one fails): the numbers of (M5) (floors 2.3e-29, near 1e-18, 4.1e-30; without a table
8 % to 34 % at batch one over the two instances), the level of dist^2 / lr^2 of the summary, the recorded learning
rates against the closed form of the schedule, the recomputed run of (a) against the cache, the two ways of computing
the means of (a), the equality of the two configurations of (c) but for the estimator.

Status. Measured (five seeds, one instance, one data draw, batch one); (a) is measured exactly along one run. Nothing
is proved at these cooldowns: Proposition 5(b) certifies cooldowns 1e4 to 1e6 times slower, for the exact polar factor
without momentum; Proposition 5(c) explains why the same ratio distance / lr^2 is possible at every scale for a
scale-invariant map on least squares. The geometric form of the cooldown and the momentum 0.9 are choices of the paper.

The script prints every number it plots.
Run:    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/scripts/make_figE5_recipe_table.py
Output: paper/figures/figE5_recipe_table.pdf and .png
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import explain_style as es  # noqa: E402
import recipe_data as rd  # noqa: E402

NAME = "figE5_recipe_table"
A_SEED, A_PASSES, A_EVERY = 10, 40, 20        # panel (a): the run, its window, iterations between two states
B_XLIM, B_YLIM = (0.6, 1e-40), (1e-36, 30.0)  # panel (b): lr_t^2 decreasing to the right
B_GUIDE = (1e-11, 1e-20, 1e5)                 # guide of slope one: rel = c lr^2 between the two ends
C_PASSES, C_YLIM = 150, (1e-19, 30.0)         # panel (c)
C_LOST = 1e-12                                # a seed that stays above C_LOST has left the schedule


# ----------------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------------
def data():
    m9, cache = rd.load()
    law = rd.law_data(m9, cache)                                  # includes check_paper
    err = rd.estimator_error(m9, cache, label=rd.POLAR, seed=A_SEED, passes=A_PASSES, every=A_EVERY)
    worse = err["saga"] > err["sgd"]
    err["worse"] = (float(err["passes"][worse].min()), float(err["passes"][worse].max()))
    tb, no = rd.same_but_estimator(m9, cache)
    fin = np.array([r["final"] for r in tb["runs"]])
    on, lost = fin[fin < C_LOST], fin[fin >= C_LOST]
    if (len(on), len(lost)) != (4, 1):
        raise SystemExit(f"(c): expected four seeds on schedule and one lost, found {len(on)} and {len(lost)}")
    f = np.array([r["final"] for r in no["runs"]])
    return {"law": law, "err": err, "table": tb, "sgd": no, "on": on, "lost": float(lost[0]),
            "sgd_range": (float(f.min()), float(f.max()))}


def print_all(D):
    e = D["err"]
    print(f"(a) run '{e['label']}', seed {e['seed']}, cfg {e['cfg']}; recomputed against the cache: {e['checked']} "
          f"records, largest relative difference {e['worst']:.1e}")
    print("    passes     SAGA table   batch-one grad.   ||grad F||_F^2")
    for k in list(range(0, 60, 10)) + list(range(60, len(e["passes"]), 50)):
        print(f"    {e['passes'][k]:6.1f}   {e['saga'][k]:12.4e}   {e['sgd'][k]:12.4e}   {e['grad2'][k]:12.4e}")
    print(f"    variance of the batch-one gradient at W_*: {e['var_star']:.4f}; the SAGA estimator has the larger "
          f"error from pass {e['worse'][0]:g} to pass {e['worse'][1]:g}; its error is exactly 0 at pass 1")
    tail = e["passes"] >= 0.5 * A_PASSES
    r = e["saga"][tail] / e["grad2"][tail]
    print(f"    passes {0.5 * A_PASSES:g} to {A_PASSES}: error of the SAGA estimator / ||grad F||_F^2, median "
          f"{np.median(r):.1f} (square root {np.sqrt(np.median(r)):.1f})")
    print("(b)")
    rd.print_law(D["law"])
    x0, x1, c = B_GUIDE
    print(f"    guide of slope one: rel = {c:g} lr^2 for lr^2 from {x0:g} to {x1:g} (a guide, not a fit)")
    print(f"(c) table    {D['table']['cfg']}")
    print(f"    no table {D['sgd']['cfg']}")
    for k in ("table", "sgd"):
        print(f"    {k:5s} final per seed: " + ", ".join(f"{r['final']:.3g}" for r in D[k]["runs"]))
    print(f"    table: four seeds at {D['on'].min():.2e} to {D['on'].max():.2e}, one at {D['lost']:.2e}; no table: "
          f"{100 * D['sgd_range'][0]:.1f} % to {100 * D['sgd_range'][1]:.1f} % of the initial squared distance")


# ----------------------------------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------------------------------
def panel_a(ax, D, ps):
    ink = ps.INK
    e = D["err"]
    p = e["passes"]
    pos = e["saga"] > 0                              # exact table at the first state: error 0, not on a log axis
    whole = np.abs(p - np.round(p)) < 1e-9
    ax.plot(p, e["sgd"], color=ink["secondary"], lw=1.5, zorder=3)
    ax.plot(p[pos], e["saga"][pos], color=ps.ORANGE, lw=1.3, zorder=4)
    ax.plot(p[whole], e["grad2"][whole], color=ink["primary"], lw=0.9, ls=(0, (4.0, 2.0)), zorder=2)
    kw = dict(fontsize=es.small(), color=ink["secondary"], textcoords="offset points", linespacing=1.15)
    ax.annotate("batch-one gradient", (A_PASSES, e["var_star"]), xytext=(-1, -4), ha="right", va="top", **kw)
    ax.annotate("SAGA table", (A_PASSES, 2e-3), xytext=(-1, 0), ha="right", va="center", **kw)
    k = int(np.argmin(np.abs(p - 22.0)))
    ax.annotate(r"$\|\nabla F(W_t)\|_F^2$", (p[k], e["grad2"][k]), xytext=(-5, -5), ha="right", va="top", **kw)
    ax.set_yscale("log")
    ax.set_xlim(0, A_PASSES)
    ax.set_ylim(1e-10, 60.0)
    ax.set_yticks([1e-8, 1e-4, 1.0])
    ax.minorticks_off()
    ax.set_xticks([1, 10, 20, 30, 40])
    ax.set_xlabel("passes")
    ax.set_ylabel(r"mean over $j$ of $\|\omega_j - \nabla F(W_t)\|_F^2$")
    es.panel_title(ax, "(a) estimator error")


def panel_b(ax, D, ps):
    ink = ps.INK
    L = D["law"]
    TB, NO = ps.ORANGE, ink["secondary"]
    style = (("sgd", NO, "-", 3), ("mars", NO, rd.DASH, 3), ("polar", TB, "-", 4), ("ns5", TB, rd.DASH, 5),
             ("eps", TB, rd.DOT, 5))
    for k, col, ls, z in style:
        for r in L[k]["runs"]:
            ax.plot(r["lr"] ** 2, r["rel"], color=col, ls=ls, lw=0.8, zorder=z)
    x0, x1, c = B_GUIDE
    ax.plot([x0, x1], [c * x0, c * x1], color=ink["primary"], lw=0.8, zorder=2)
    kw = dict(fontsize=es.small(), color=ink["secondary"], textcoords="offset points", linespacing=1.15)
    xm = np.sqrt(x0 * x1)
    ax.annotate("slope one", (xm, c * xm), xytext=(3, 2), ha="left", va="bottom", **kw)
    xr = B_XLIM[1]
    ax.annotate("no table", (xr, L["no_range"][0]), xytext=(-1, -4), ha="right", va="top", **kw)
    ax.annotate("table, ns5", (xr, L["ns5"]["floor"]), xytext=(-1, 3), ha="right", va="bottom", **kw)
    ax.annotate("exact polar (solid),\n" r"ns5, $\epsilon = 10^{-30}$ (dotted)", (xr, L["eps"]["floor"]),
                xytext=(-1, -3), ha="right", va="top", **kw)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*B_XLIM)
    ax.set_ylim(*B_YLIM)
    ax.set_xticks([1.0, 1e-20, 1e-40])
    ax.set_yticks([1e-30, 1e-20, 1e-10, 1.0])
    ax.minorticks_off()
    ax.set_xlabel(r"$\mathrm{lr}_t^2$ (decreasing)")
    ax.set_ylabel(rd.DIST_LABEL)
    es.panel_title(ax, "(b) schedule law")


def panel_c(ax, D, ps):
    ink = ps.INK
    for k, col, ls, z in (("sgd", ink["secondary"], "-", 3), ("table", ps.ORANGE, rd.DASH, 4)):
        for r in D[k]["runs"]:
            m = r["passes"] <= C_PASSES + 2
            ax.plot(r["passes"][m], r["rel"][m], color=col, ls=ls, lw=0.8, zorder=z)
    kw = dict(fontsize=es.small(), color=ink["secondary"], textcoords="offset points", linespacing=1.15)
    ax.annotate("minibatch gradient\n(no table)", (C_PASSES, D["sgd_range"][0]), xytext=(-1, -4), ha="right", va="top", **kw)
    ax.annotate("1 seed of 5", (C_PASSES, D["lost"]), xytext=(-1, -3), ha="right", va="top", **kw)
    ax.annotate("SAGA table:\n4 seeds of 5", (C_PASSES, D["on"].max()), xytext=(-1, 3), ha="right", va="bottom",
                **kw)
    ax.set_yscale("log")
    ax.set_xlim(0, C_PASSES)
    ax.set_ylim(*C_YLIM)
    ax.set_yticks([1e-18, 1e-12, 1e-6, 1.0])
    ax.minorticks_off()
    ax.set_xticks([0, 50, 100, 150])
    ax.set_xlabel("passes")
    ax.set_ylabel(rd.DIST_LABEL)
    es.panel_title(ax, "(c) same schedule")


def draw(D):
    ps = es.setup("paper")
    fig = es.figure("paper", height=2.6)
    fig.get_layout_engine().set(w_pad=0.03, wspace=0.02)
    gs = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 1.0))
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, k]) for k in range(3))
    panel_a(ax_a, D, ps)
    panel_b(ax_b, D, ps)
    panel_c(ax_c, D, ps)
    es.save(fig, NAME, "paper")


def main():
    D = data()
    print_all(D)
    draw(D)


if __name__ == "__main__":
    main()
