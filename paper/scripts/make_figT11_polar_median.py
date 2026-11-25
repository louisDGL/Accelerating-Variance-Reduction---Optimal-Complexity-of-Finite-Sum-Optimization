"""Figure T11 of the lecture (paper/talk/figures/figT11_polar_median): in the plane, polar steps on sampled gradients
go to the geometric median of the data, not to their mean; a SAGA table brings the minimizer back. Explanatory
figure, not in the paper. It is the lecture version of panels (a) and (b) of figE3_polar_median (the planar companion
of figT10_sign_median) and explains Theorem 2(b),(c) and Proposition 3(c),(d) of paper/paper_saga_to_muon.tex, with
(M5) for the two runs on a geometric cooldown.

The toy and the runs are those of make_figE3_polar_median.py, whose functions compute and draw them (see its
docstring for the details; nothing is drawn by hand):
    W in R^{2 x 1} (K = 2, d = 1),  f_i(W) = |W - y_i|^2 / 2,  n = 200 points y_i of a skewed cloud (fixed seed).
The minimizer W_* of F is the mean of the y_i; the gradient noise ybar - y_i is additive and does not depend on W
(Theorem 2). For a K x 1 matrix the polar factor is the normalization and the nuclear norm the Euclidean norm (the
VECTOR case, not evidence about the spectral map): the limit of Theorem 2(b) is the geometric median W_dagger of the
y_i, the minimizer of Lambda(W) = (1/n) sum_i |W - y_i| (Proposition 3(d)).

Panels:
  (a) the plane of W: the points y_i (those in the window), the starting point W_0 and W_* (always visible); level
      sets of Lambda, the run without a table of one seed, W <- W - eta_t polar(g_i(W)), eta_t = ETA0 (1 + t/n)^(-3/4)
      (thin: the iterates; thick: the iterate averaged over the window (t/2, t]) and W_dagger; the run with the SAGA
      table on the cooldown lr_t = LR0 * Q^(t/n) (recipe (4) of the paper, exact polar factor, same indices). The
      path with the table is a straight segment through W_*: with x_i = 1 the SAGA estimator does not involve the
      sampled y_j and the run is the sign iteration of figT10 on the line through W_0 and W_* (it overshoots W_*
      during the first passes; see the docstring of make_figE3_polar_median.py, also for the role of the rounding);
  (b) distances against passes, median and range over the seeds: the averaged iterate without a table, to W_dagger;
      the iterate with the table, to W_*; the same cooldown on the sampled gradient without a table, to W_* (it stops
      next to W_dagger, 0.29 from W_*; its total path is 21 times |W_0 - W_*|). The legend gives the three length
      rules, the direct labels say which distance each curve is.
The window of panel (a) is narrower than in figE3 (no inset: the limits for minibatches are the subject of
figT12_offset_scaling).

Colours (the convention of the lecture figures): runs WITHOUT a table are blue whatever their length rule, runs WITH
the SAGA table are orange, grey is kept for the points of the cloud and the level sets. The two runs without a table
of panel (b) are therefore both blue: the one in the class of Theorem 2 is solid, the one on the cooldown is dashed
(DASHES), and each has its legend entry and its direct label. (The paper figure figE3 keeps the latter in grey ink, as
Figure 4(a) of the paper does.) The labels of panel (a) have a halo in the surface colour (E3.halo).

Status. Proved: the limit without a table (Theorem 2(b), Proposition 3(d)); that the minimizer is the only fixed
point with the table (Proposition 3(c)). Observed on this instance only (illustration): that the run with the table
reaches it on the cooldown q = 0.8 (a geometric cooldown is not in the class of Theorem 2), and where the cooldown
without a table freezes. The scope line says so ("proved: limit W_dagger (Thm 2), fixed point W_* (Prop. 3c); the
rest: observed"); the averaging window (t/2, t] is in the label of panel (b).

Progressive build (states s1-s4): s1 the cloud, W_0, the mean W_* and the two definitions (the question: where do
polar steps go?); s2 the level sets of Lambda, the run without a table and the geometric median; s3 the run with the
SAGA table; s4 panel (b).

The script prints every number it plots (through print_all of make_figE3_polar_median.py). Nothing is read from the
results of the project.
Run:    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/scripts/make_figT11_polar_median.py   (about 15 s)
Output: paper/talk/figures/figT11_polar_median.pdf and .png, and the states _s1 ... _s4
"""
from __future__ import annotations

import importlib.util
import os
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import explain_style as es  # noqa: E402

NAME = "figT11_polar_median"
WINDOW = (-0.80, 1.44, -0.78, 0.70)      # panel (a): x_min, x_max, y_min, y_max (y is refitted to keep equal scales)
NO_TABLE_LABEL_Y = 0.20                  # panel (a): top of the label of the run without a table, under its path
DASHES = (0, (3.2, 1.6))                 # line style of the run without a table on the cooldown (panel (b))
Y_TOP = 1e5                              # top of the axis of panel (b)


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E3 = _load_module(os.path.join(HERE, "make_figE3_polar_median.py"), "make_figE3_polar_median")
E3.WINDOW = WINDOW                       # the functions of E3 read the window of the plane from their module


def data():
    return E3.data(offsets=False)        # the toy and the batch-one runs; no minibatch (not shown here)


def print_all(D):
    E3.print_all(D)


def draw(D):
    ps = es.setup("talk")
    ink = ps.INK
    fig = es.figure("talk", height=2.75)
    fig.get_layout_engine().set(w_pad=0.03, wspace=0.03)
    gs = fig.add_gridspec(1, 2, width_ratios=(1.17, 1.0))
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    b = es.Build()
    ms = 6.4

    # (a) the plane; every label has a halo in the surface colour: the level sets and the points do not cross it
    E3.panel_plane(ax_a, D, ps, build=b, states=(2, 3), inset=False, labels=False, ms=ms, lw=2.0)
    ax_a.set_xticks([-0.5, 0.0, 0.5, 1.0])
    ax_a.set_yticks([-0.5, 0.0, 0.5])
    es.panel_title(ax_a, "(a) the plane of $W$")
    mean, gm, w0 = D["mean"], D["gm"], np.array(E3.W0)
    kw = dict(textcoords="offset points", fontsize=es.small(), color=ink["secondary"], zorder=10, linespacing=1.2,
              path_effects=E3.halo(ps, 3.0))
    corner = dict(xycoords="axes fraction", **kw)
    ax_a.annotate("$W_0$", w0, xytext=(0, 6), ha="center", va="bottom", **kw)
    ax_a.annotate(r"$W_\ast$ = mean", mean, xytext=(-14, 0.9 * ms + 4), ha="center", va="bottom", **kw)
    ax_a.annotate(rf"$f_i(W) = |W - y_i|^2/2$, $n = {E3.N}$", (1.0, 0.0), xytext=(-4, 5), ha="right", va="bottom",
                  **corner)
    ax_a.annotate(r"polar$(g) = g/|g|$", (1.0, 0.0), xytext=(-4, 17.5), ha="right", va="bottom", **corner)
    b.add(2, ax_a.annotate("level sets of\n" + r"$\Lambda = \mathrm{mean}_i\,|W - y_i|$", (0.0, 1.0), xytext=(4, -4),
                           ha="left", va="top", **corner))
    b.add(2, ax_a.annotate(r"$W^\dagger$ = geometric" + "\nmedian", gm, xytext=(0, -(1.0 * ms + 6)), ha="center",
                           va="top", **kw))
    b.add(2, ax_a.annotate("no table:\niterates (thin),\ntheir mean (thick)", (1.0, NO_TABLE_LABEL_Y),
                           xycoords=("axes fraction", "data"), xytext=(-4, 0), ha="right", va="top", **kw))
    b.add(3, ax_a.annotate("SAGA table", E3.path_point(D["path_tb"][:40], 0.78), xytext=(0, 6), ha="center",
                           va="bottom", **kw))

    # (b) distances; every run without a table is blue: the one on the cooldown is told apart by its dashes
    lines = E3.panel_distances(ax_b, D, ps, lw=1.6, control=(ps.BLUE, DASHES))
    ax_b.set_ylim(1e-8, Y_TOP)                           # the decades above the curves hold the legend
    ax_b.set_yticks([1e0, 1e-2, 1e-4, 1e-6, 1e-8])
    ax_b.minorticks_off()
    ax_b.set_ylabel("distance")
    es.panel_title(ax_b, "(b) two limits")
    ps.legend(ax_b, handles=[lines["no"], lines["table"], lines["cooldown"]],
              labels=[rf"no table, $\eta_t = {E3.ETA0:g}\,(1 + t/n)^{{-3/4}}$",
                      rf"SAGA table, $\mathrm{{lr}}_t = {E3.LR0:g}\cdot{E3.Q:g}^{{\,t/n}}$",
                      r"no table, the same $\mathrm{lr}_t$"],
              loc="upper right", handlelength=1.5, handletextpad=0.5, borderpad=0.1, labelspacing=0.25,
              borderaxespad=0.1)
    P = E3.PASSES
    kt = dict(fontsize=es.small(), color=ink["secondary"], zorder=6, linespacing=1.15, ha="right")
    ax_b.text(P - 1, 1.6 * D["frozen"], rf"frozen {D['frozen']:.2f} from $W_\ast$", va="bottom", **kt)
    ax_b.text(P - 1, E3.below_no_table(D), "mean over $(t/2, t]$,\n" + r"to $W^\dagger$", va="top", **kt)
    ax_b.text(*E3.table_label_anchor(D, at=0.42, gap=1.0, below=3.5), r"to $W_\ast$", va="top", **kt)
    b.add(4, ax_b)

    pad = min(0.12, 0.2 / fig.get_figheight())           # the room that es.save reserves for the scope line: the
    fig.get_layout_engine().set(rect=(0.0, pad, 1.0, 1.0 - pad))   # window is fitted to the final cell of the axes
    lo, hi = E3.fit_window(fig, ax_a)
    print(f"(a) window drawn (equal scales): [{WINDOW[0]}, {WINDOW[1]}] x [{lo:.3f}, {hi:.3f}]")
    print(f"(b) legend: no table, eta_t = {E3.ETA0:g} (1 + t/n)^(-3/4); SAGA table, lr_t = {E3.LR0:g} * {E3.Q:g}^(t/n); "
          f"no table, the same lr_t; label: frozen {D['frozen']:.2f} from W*")
    es.save(fig, NAME, "talk", build=b,
            note=(rf"toy; (b): median, range of {len(E3.SEEDS)} runs; proved: limit $W^\dagger$ (Thm 2), fixed point "
                  rf"$W_\ast$ (Prop. 3c); rest: observed"))


def main():
    D = data()
    print_all(D)
    draw(D)


if __name__ == "__main__":
    main()
