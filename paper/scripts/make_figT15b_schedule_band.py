"""Figure T15b of the lecture (paper/talk/figures/figT15b_schedule_band): "the distance follows the schedule" means
that the ratio distance^2 / lr_t^2 stays in a band; it leaves the band for good when the run meets its floor.
Explanatory figure, not in the paper; companion of figT15_schedule_law (the ratio of the two axes of that figure,
against passes). It explains the sentence of (M5) of paper/paper_saga_to_muon.tex, "with the table the squared
distance follows C lr_t^2 (the ratio varies by a factor 3.2 to 5.7 along a run) until it meets a floor", and
Proposition 5(c).

Data: the cached runs of experiment M9, part C, with the SAGA table (muon/results/M9_summary.json, C/ls-cond/deep;
cache muon/results/M9_law.pkl, read through recipe_data.law_data): least squares, n = 200, K x d = 8 x 40,
kappa = 1e4, batch one, the five seeds 10..14 of each configuration, every seed drawn as a thin line:
    exact polar factor, no momentum, lr_t = 0.3 * 0.8^(t/n)                      "polar, no momentum"
    ns5 + Nesterov 0.9 (Muon's map and momentum), lr_t = 0.1 * 0.8^(t/n)         "ns5 + Nesterov at q = 0.8,
                                                                                  lr_0 = 0.1" (eps = 1e-7)
    the same with eps = 1e-30 in ns5                                             "ns5 + Nesterov, eps = 1e-30, at
                                                                                  q = 0.8, lr_0 = 0.1"
One panel: against passes (0 to 180; the initial full gradient of the table counts for one pass), the ratio
    ||W_t - W_*||_F^2 / (||W_*||_F^2 lr_t^2)                                     (log scale),
lr_t being the recorded length of the last step. The geometric form of the cooldown is a choice of the paper, as is
the momentum 0.9.

Numbers (printed, not on the figure): the level of the ratio (median over the records with a relative squared
distance between 1e-12 and 1e-6, median over the seeds) is 69.7 for the exact polar factor and 7.9 and 7.7 for the two
ns5 configurations, i.e. 97644, 11035 and 10782 in absolute units, the values of the summary (checked). Over that
window a seed varies by a factor 1.3 to 2.1; over the second half of the run up to the point where the ratio passes
3 times the level (the reading of the paper, which includes the departure) by a factor 3.2 to 4.1 (medians of the
summary on this instance; 3.2 to 5.7 over the two instances and the fixed-point configurations in the paper).

Status. Measured (five seeds, one instance, one data draw, batch one). Proposition 5(c) (proved, least squares, a map
with o(sM) = o(M)): the state divided by the learning rate is a time-homogeneous Markov chain, so a stationary band
is possible at every scale; that the chain stays bounded at q = 0.8, and with momentum, is observed, not proved. The
departures are the floors: rounding for the exact polar factor and for ns5 with eps = 1e-30, the constant eps = 1e-7
of ns5 (which breaks the scale invariance) for Muon's map.

Progressive build (states s1-s2): s1 the exact polar factor; s2 adds the two ns5 configurations.

The script prints every number it plots.
Run:    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/scripts/make_figT15b_schedule_band.py
Output: paper/talk/figures/figT15b_schedule_band.pdf and .png, and the states _s1, _s2
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import explain_style as es  # noqa: E402
import recipe_data as rd  # noqa: E402

NAME = "figT15b_schedule_band"
PASSES = 180                # window of the figure
Y_LIM = (0.4, 3e5)
LEAVE = 10.0                # a run "has left the band" at the first record whose ratio exceeds LEAVE x its level


# ----------------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------------
def data():
    m9, cache = rd.load()
    D = rd.law_data(m9, cache)
    for k in ("polar", "ns5", "eps"):
        s = D[k]
        leave = []
        for r in s["runs"]:
            p, x = rd.ratio(r)
            late = (p > 20) & (x > LEAVE * s["level_rel"])
            leave.append(float(p[late][0]))
        s["leave"] = leave
        if max(leave) > PASSES:
            raise SystemExit(f"'{s['label']}' leaves its band after the window of the figure")
    return D


def print_all(D):
    rd.print_law(D)
    for k in ("polar", "ns5", "eps"):
        s = D[k]
        print(f"{k}: first pass with a ratio above {LEAVE:g} x level, per seed: "
              + ", ".join(f"{v:.1f}" for v in s["leave"]))
        for r in s["runs"][:1]:
            p, x = rd.ratio(r)
            pick = [int(np.argmin(np.abs(p - v))) for v in (5, 20, 40, 60, 80, 100, 120, 140, 160, 180)]
            print(f"      seed {r['seed']}, ratio at passes " + ", ".join(f"{p[i]:.1f}: {x[i]:.3g}" for i in pick))


# ----------------------------------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------------------------------
def draw(D):
    ps = es.setup("talk")
    ink = ps.INK
    TB = ps.ORANGE                                   # SAGA table: orange (exact polar solid, ns5 dashed / dotted)
    series = (("polar", "-", 4, 1), ("ns5", rd.DASH, 5, 2), ("eps", rd.DOT, 5, 2))
    fig = es.figure("talk", height=2.7)
    ax = fig.add_subplot(111)
    b = es.Build()
    kw = dict(fontsize=es.small(), color=ink["secondary"], textcoords="offset points", linespacing=1.2)
    for k, ls, z, state in series:
        for r in D[k]["runs"]:
            p, x = rd.ratio(r)
            m = p <= PASSES + 2
            b.add(state, ax.plot(p[m], x[m], color=TB, ls=ls, lw=1.0, zorder=z))
    b.add(1, ax.annotate("table, exact polar factor", (50, D["polar"]["level_rel"]), xytext=(0, 9), ha="center",
                         va="bottom", **kw))
    b.add(2, ax.annotate("table, ns5 (Muon's map) + Nesterov 0.9; dotted: "
                         r"$\epsilon = 10^{-30}$ in ns5", (4, D["ns5"]["level_rel"]),
                         xytext=(0, -11), ha="left", va="top", **kw))
    b.add(1, ax.annotate("rounding\nfloor", (np.median(D["polar"]["leave"]), 3e3),
                         xytext=(-9, 0), ha="right", va="center", **kw))
    b.add(2, ax.annotate("floor of ns5\n" r"($\epsilon = 10^{-7}$)", (np.median(D["ns5"]["leave"]), 3e3),
                         xytext=(-2, 0), ha="right", va="center", **kw))
    ax.set_yscale("log")
    ax.set_xlim(0, PASSES)
    ax.set_ylim(*Y_LIM)
    ax.set_yticks([1.0, 1e1, 1e2, 1e3, 1e4, 1e5])
    ax.minorticks_off()
    ax.set_xticks([1, 30, 60, 90, 120, 150, 180])
    ax.set_xlabel("passes")
    ax.set_ylabel(r"$\|W_t - W_\ast\|_F^2\ /\ (\|W_\ast\|_F^2\,\mathrm{lr}_t^2)$")
    es.save(fig, NAME, "talk", build=b,
            note=("least squares, $n = 200$, batch one, 5 seeds each; geometric cooldown (the paper's "
                  "choice), $q = 0.8$"))


def main():
    D = data()
    print_all(D)
    draw(D)


if __name__ == "__main__":
    main()
