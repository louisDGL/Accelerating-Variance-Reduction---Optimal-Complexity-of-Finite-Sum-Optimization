"""Figure T15 of the lecture (paper/talk/figures/figT15_schedule_law): with a SAGA table the squared distance stays
slaved to the square of the learning rate over more than twenty orders of magnitude, down to a floor; without a table
it stops moving while the schedule keeps shrinking. Explanatory figure, not in the paper: it reads Figure 4(a) of
paper/paper_saga_to_muon.tex against the schedule instead of time, and explains (M5) in its corrected wording and
Proposition 5(c). Its companion figT15b_schedule_band (make_figT15b_schedule_band.py) shows the ratio of the two axes
of this figure against passes.

Data: the cached runs of experiment M9, part C (muon/results/M9_summary.json, C/ls-cond; cache muon/results/M9_law.pkl,
read through recipe_data.law_data): least squares, n = 200, K x d = 8 x 40, kappa = 1e4, batch one, 300 passes without
stop rule, the five seeds 10..14 of each configuration, every seed drawn as a thin line (no median):
    SAGA table, exact polar factor, no momentum, lr_t = 0.3 * 0.8^(t/n)                 deep/"polar, no momentum"
    SAGA table, ns5 + Nesterov 0.9 (Muon's map and momentum), lr_t = 0.1 * 0.8^(t/n)    deep/"ns5 + Nesterov at q =
                                                                                         0.8, lr_0 = 0.1" (eps = 1e-7)
    the same with eps = 1e-30 in ns5                                                    deep/"ns5 + Nesterov, eps =
                                                                                         1e-30, at q = 0.8, lr_0 = 0.1"
    no table, minibatch gradient, ns5 + Nesterov 0.9, lr_t = 0.3 * 0.75^(t/n)           controls/"no table (minibatch
                                                                                         gradient), ns5 + Nesterov, b=1"
    no table, MARS-M (its own momentum 0.95, not tuned), the same learning rates        controls/"MARS-M, b=1"
The two configurations without a table run at the schedule selected for Muon's recipe with the table (q = 0.75,
lr_0 = 0.3), not at the schedule of the table runs drawn here (q = 0.8): on these axes the speed of the schedule does
not appear, only the pairs (learning rate, distance) do. The geometric form of the cooldown is a choice of the paper
(Muon's reference runs cool linearly), as is the momentum 0.9.

One panel, a parametric plot: x = lr_t^2 (log scale, DECREASING to the right, so that a run goes from left to right),
y = ||W_t - W_*||_F^2 / ||W_*||_F^2 (log scale; W_0 = 0, so y is also the fraction of the initial squared distance).
lr_t is the recorded length of the last step (trace 'eta', checked against lr_0 q^((k-1)/period) in recipe_data); at
the first record, where no step has been taken, lr_0. A thin ink guide of slope one (distance^2 proportional to
lr_t^2; a guide, not a fit). The floors printed are the medians over the seeds of the lowest distance reached. The
axis stops at lr_t^2 = 1e-40; the runs go on (to 1e-59 and beyond) without leaving their floors.

Checks against the paper, (M5): floors 2.3e-29 (exact polar factor, least squares), near 1e-18 (ns5 at q = 0.8,
lr_0 = 0.1), 4.1e-30 (eps = 1e-30); without a table, 8 % to 34 % of the initial squared distance at batch one over
the two instances of part C; the level of dist^2 / lr^2 of the summary. The script stops if one is not reproduced.

Status. Measured (five seeds, one instance, one data draw, batch one). Per seed, the ratio distance / lr_t^2 stays
within a factor 3 of its level over 29 decades of lr_t^2 for the exact polar factor, 24 to 27 with eps = 1e-30 and 13
to 15 with the default eps = 1e-7 of ns5 (printed by the script). Proposition 5(c) (proved, least squares): for a map
o with o(sM) = o(M) the state divided by the learning rate is a time-homogeneous Markov chain, which is why the same
ratio is possible at every scale; the constant eps of ns5 breaks that invariance, hence its floor near 1e-18; the
floor of the exact polar factor is the rounding floor. That the chain stays bounded at q = 0.8 is observed, not proved
(Proposition 5(b) certifies cooldowns 1e4 to 1e6 times slower, exact polar factor, no momentum).

Progressive build (states s1-s3): s1 the guide of slope one and the runs with the table and the exact polar factor;
s2 adds the two ns5 configurations and their floors; s3 adds the runs without a table.

The script prints every number it plots.
Run:    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/scripts/make_figT15_schedule_law.py
Output: paper/talk/figures/figT15_schedule_law.pdf and .png, and the states _s1 ... _s3
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import explain_style as es  # noqa: E402
import recipe_data as rd  # noqa: E402

NAME = "figT15_schedule_law"
X_LIM = (0.6, 1e-40)         # lr_t^2, decreasing to the right
Y_LIM = (1e-34, 30.0)
GUIDE = (1e-11, 1e-20, 1e5)  # the guide of slope one: rel = c lr^2 for lr^2 between the two ends


# ----------------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------------
def data():
    m9, cache = rd.load()
    return rd.law_data(m9, cache)


def print_all(D):
    rd.print_law(D)
    x0, x1, c = GUIDE
    print(f"guide of slope one: rel = {c:g} lr^2 for lr^2 from {x0:g} to {x1:g} (a guide, not a fit)")
    for k in ("polar", "ns5", "eps", "sgd", "mars"):
        end = [r["lr"][-1] ** 2 for r in D[k]["runs"]]
        print(f"  {k}: lr^2 from {D[k]['lr0'] ** 2:g} to {min(end):.1e} (axis cut at {X_LIM[1]:g})")


# ----------------------------------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------------------------------
def draw(D):
    ps = es.setup("talk")
    ink = ps.INK
    TB, NO = ps.ORANGE, ps.BLUE                      # SAGA table: orange; no table: blue, as in the other figures of
    #                                                  the lecture (Figure 4(a) of the paper and figE5 draw it in grey ink)
    series = (   # key, colour, line style, zorder, build state, legend label
        ("polar", TB, "-", 4, 1, "table, exact polar factor"),
        ("ns5", TB, rd.DASH, 5, 2, "table, ns5 (Muon's map)\n+ Nesterov 0.9"),
        ("eps", TB, rd.DOT, 5, 2, r"the same, $\epsilon = 10^{-30}$ in ns5"),
        ("sgd", NO, "-", 3, 3, "no table ($q = 0.75$), minibatch gradient"),
        ("mars", NO, rd.DASH, 3, 3, "no table ($q = 0.75$), MARS-M"),
    )
    fig = es.figure("talk", height=2.7)
    ax = fig.add_subplot(111)
    b = es.Build()
    kw = dict(fontsize=es.small(), color=ink["secondary"], textcoords="offset points", linespacing=1.2)

    x0, x1, c = GUIDE
    b.add(1, ax.plot([x0, x1], [c * x0, c * x1], color=ink["primary"], lw=0.9, zorder=2))
    xm = np.sqrt(x0 * x1)
    b.add(1, ax.annotate(r"slope one: distance$^2 \propto \mathrm{lr}_t^2$", (xm, c * xm), xytext=(5, 3),
                         ha="left", va="bottom", **kw))
    for k, col, ls, z, state, label in series:
        for i, r in enumerate(D[k]["runs"]):
            b.add(state, ax.plot(r["lr"] ** 2, r["rel"], color=col, ls=ls, lw=0.9, zorder=z,
                                 label=label if i == 0 else None))
    xr = X_LIM[1]
    b.add(1, ax.annotate(f"rounding: {rd.tex(D['polar']['floor'])}", (xr, D["polar"]["floor"]),
                         xytext=(-2, 4), ha="right", va="bottom", **kw))
    b.add(2, ax.annotate(rf"floor of ns5 ($\epsilon = 10^{{-7}}$), {rd.tex(D['ns5']['floor'])}",
                         (xr, D["ns5"]["floor"]), xytext=(-2, 4), ha="right", va="bottom", **kw))
    b.add(2, ax.annotate(rf"$\epsilon = 10^{{-30}}$: {rd.tex(D['eps']['floor'])}", (xr, D["eps"]["floor"]),
                         xytext=(-2, -4), ha="right", va="top", **kw))
    lo, hi = D["no_range"]
    b.add(3, ax.annotate(f"no table: stops at {100 * lo:.0f} % to {100 * hi:.0f} % of the initial squared distance",
                         (xr, lo), xytext=(-2, -5), ha="right", va="top", **kw))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)
    ax.set_xticks([1.0, 1e-10, 1e-20, 1e-30, 1e-40])
    ax.set_yticks([1e-30, 1e-20, 1e-10, 1.0])
    ax.minorticks_off()
    ax.set_xlabel(r"$\mathrm{lr}_t^2$  (decreasing: a run moves to the right)")
    ax.set_ylabel(rd.DIST_LABEL)
    b.legend(ax, loc="lower left", handlelength=2.3, borderaxespad=0.3, labelspacing=0.3)
    es.save(fig, NAME, "talk", build=b,
            note=("least squares, $n = 200$, batch one, 5 seeds each; geometric cooldown (paper's choice), "
                  "table: $q = 0.8$"))


def main():
    D = data()
    print_all(D)
    draw(D)


if __name__ == "__main__":
    main()
