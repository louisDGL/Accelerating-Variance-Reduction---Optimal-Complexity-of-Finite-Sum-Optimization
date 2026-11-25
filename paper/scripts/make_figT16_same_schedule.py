"""Figure T16 of the lecture (paper/talk/figures/figT16_same_schedule): same map, same momentum, same schedule, same
seeds; only the estimator differs. With the SAGA table four seeds of five end near 4e-18 and one stalls at 2.6e-6;
with the minibatch gradient all five stop at 8 to 9 % of the initial squared distance. Explanatory figure, not in the
paper. It explains (M5) of paper/paper_saga_to_muon.tex in its corrected wording ("without a table the schedule
selected for Muon's recipe freezes the iterate ... at batch one") and the sentence of the abstract ("where minibatch
and momentum-based estimators freeze"): of the cached runs, this pair makes "the same schedule" literally true.

Data: cached runs of experiment M9, part C (muon/results/M9_summary.json, C/ls-cond; cache muon/results/M9_law.pkl,
read through recipe_data): least squares, n = 200, K x d = 8 x 40, kappa = 1e4, batch one, seeds 10..14, 300 passes
without stop rule (the first 150 are drawn; nothing moves afterwards), every seed drawn as a thin line:
    orange, dashed   deep/"ns5 + Nesterov (Muon's recipe)": SAGA table, ns5, Nesterov momentum 0.9,
                     lr_t = 0.3 * 0.75^(t/n) (t: iterations), the point selected by the tuning of the paper;
    blue, solid      controls/"no table (minibatch gradient), ns5 + Nesterov, b=1": the SAME configuration with the
                     batch-one gradient in the place of the SAGA estimator. The script compares the two
                     configurations key by key and stops if anything but the estimator differs
                     (recipe_data.same_but_estimator);
    light blue       controls/"MARS-M, b=1" (dashed) and "STORM-Muon, b=1" (dotted): momentum-based estimators with
                     ns5 on the same learning rates per gradient evaluation, their own momentum (0.95), not tuned:
                     context, not a controlled comparison.
Passes are per-sample gradient evaluations / n; the run with the table starts at pass 1 (the initial full gradient
that fills the table), so that at equal passes it has taken n fewer steps on the same schedule.

Status. Measured: five seeds, one instance, one data draw, batch one. The tuned point of the recipe sits next to a
cliff in q (paper, (M5)): one seed of five (seed 14) leaves the schedule and stalls; it is drawn. Not shown because
not in the repository: runs without a table at slower cooldowns (the audit reports that the freeze persists there,
one seed), and the softmax instance of part C, where at its tuned point only one seed of five goes below 1e-12
(summary: on_schedule = 1). The geometric form of the
cooldown and the momentum 0.9 are choices of the paper. Nothing is proved at this cooldown (Proposition 5(b)
certifies cooldowns 1e4 to 1e6 times slower, for the exact polar factor without momentum).

Progressive build (states s1-s2): s1 the runs without a table; s2 adds the runs with the SAGA table.

The script prints every number it plots (the five final values of each series).
Run:    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/scripts/make_figT16_same_schedule.py
Output: paper/talk/figures/figT16_same_schedule.pdf and .png, and the states _s1, _s2
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import explain_style as es  # noqa: E402
import recipe_data as rd  # noqa: E402

NAME = "figT16_same_schedule"
PASSES = 150                # window of the figure
Y_LIM = (1e-19, 60.0)
LOST = 1e-12                # a seed whose lowest relative squared distance stays above LOST has left the schedule


# ----------------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------------
def data():
    m9, cache = rd.load()
    rd.check_paper(m9, cache)
    tb, no = rd.same_but_estimator(m9, cache)
    D = {"table": tb, "sgd": no,
         "mars": rd.series(m9, cache, "controls", rd.MARS), "storm": rd.series(m9, cache, "controls", rd.STORM)}
    fin = np.array([r["final"] for r in tb["runs"]])
    on = fin < LOST
    D["on"] = fin[on]
    D["lost"] = [(r["seed"], r["final"]) for r in tb["runs"] if r["final"] >= LOST]
    if (int(on.sum()), len(D["lost"])) != (4, 1):
        raise SystemExit(f"expected four seeds on schedule and one lost, found {int(on.sum())} and {len(D['lost'])}")
    for k in ("sgd", "mars", "storm"):
        s = D[k]
        if (s["lr0"], s["q"]) != (tb["lr0"], tb["q"]):
            raise SystemExit(f"'{s['label']}' is not on the learning rates of the table runs")
        f = np.array([r["final"] for r in s["runs"]])
        s["range"] = (float(f.min()), float(f.max()))
        # frozen: the last record of the window equals the final value (300 passes) to 1e-6
        k150 = [int(np.argmin(np.abs(r["passes"] - PASSES))) for r in s["runs"]]
        s["drift"] = max(abs(r["rel"][i] / r["final"] - 1.0) for r, i in zip(s["runs"], k150))
    other = [r["final"] for k in ("mars", "storm") for r in D[k]["runs"]]
    D["other_range"] = (min(other), max(other))
    return D


def print_all(D):
    tb = D["table"]
    print(f"the two configurations differ by the estimator only: table {tb['cfg']}")
    print(f"                                                     no table {D['sgd']['cfg']}")
    for k in ("table", "sgd", "mars", "storm"):
        s = D[k]
        print(f"{k:6s} '{s['label']}', lr_0 = {s['lr0']:g}, q = {s['q']:g}: final relative squared distance, seeds "
              f"{[r['seed'] for r in s['runs']]}: " + ", ".join(f"{r['final']:.3g}" for r in s["runs"]))
    print(f"table: four seeds end at {D['on'].min():.2e} to {D['on'].max():.2e} (median {np.median(D['on']):.2e}); "
          f"lost: seed {D['lost'][0][0]} at {D['lost'][0][1]:.2e}")
    for k in ("sgd", "mars", "storm"):
        s = D[k]
        print(f"{k}: {100 * s['range'][0]:.1f} % to {100 * s['range'][1]:.1f} % of the initial squared distance; "
              f"largest relative change between pass {PASSES} and pass 300: {s['drift']:.1e}")
    print(f"MARS-M and STORM-Muon together: {100 * D['other_range'][0]:.1f} % to {100 * D['other_range'][1]:.1f} %")


# ----------------------------------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------------------------------
def draw(D):
    ps = es.setup("talk")
    ink = ps.INK
    TB, NO, LIGHT = ps.ORANGE, ps.BLUE, ps.BLUE_RAMP[3]   # SAGA table: orange (ns5: dashed); no table: blue, as in the
    #                                                       other figures of the lecture (figE5 and Figure 4(a): grey ink)
    series = (("mars", LIGHT, rd.DASH, 2, 1), ("storm", LIGHT, rd.DOT, 2, 1), ("sgd", NO, "-", 3, 1),
              ("table", TB, rd.DASH, 4, 2))
    fig = es.figure("talk", height=2.7)
    ax = fig.add_subplot(111)
    b = es.Build()
    kw = dict(fontsize=es.small(), color=ink["secondary"], textcoords="offset points", linespacing=1.2)
    for k, col, ls, z, state in series:
        for r in D[k]["runs"]:
            m = r["passes"] <= PASSES + 2
            b.add(state, ax.plot(r["passes"][m], r["rel"][m], color=col, ls=ls, lw=1.0, zorder=z))
    lo, hi = D["sgd"]["range"]
    olo, ohi = D["other_range"]
    b.add(1, ax.annotate(f"no table, minibatch gradient: the 5 seeds stop at {100 * lo:.0f} % to {100 * hi:.0f} %",
                         (PASSES, lo), xytext=(-2, -6), ha="right", va="top", **kw))
    b.add(1, ax.annotate("light blue: MARS-M, STORM-Muon (own momentum, not tuned), "
                         f"{100 * olo:.0f} % to {100 * ohi:.0f} %",
                         (PASSES, ohi), xytext=(-2, 5), ha="right", va="bottom", **kw))
    seed, lost = D["lost"][0]
    b.add(2, ax.annotate(f"one seed of five leaves the schedule: {rd.tex(lost)}", (PASSES, lost), xytext=(-2, -4),
                         ha="right", va="top", **kw))
    b.add(2, ax.annotate(f"SAGA table: four seeds of five end at\n{rd.tex(D['on'].min())} to "
                         f"{rd.tex(D['on'].max())}", (PASSES, D["on"].max()), xytext=(-2, 5), ha="right",
                         va="bottom", **kw))
    ax.annotate("table and minibatch runs:\nsame map (ns5), momentum\n(Nesterov 0.9), seeds, and\n"
                r"$\mathrm{lr}_t = 0.3 \times 0.75^{\,t/n}$", (3, 3e-19), xytext=(0, 0), ha="left",
                va="bottom", **kw)
    ax.set_yscale("log")
    ax.set_xlim(0, PASSES)
    ax.set_ylim(*Y_LIM)
    ax.set_yticks([1e-18, 1e-12, 1e-6, 1.0])
    ax.minorticks_off()
    ax.set_xticks([0, 30, 60, 90, 120, 150])
    ax.set_xlabel("passes")
    ax.set_ylabel(rd.DIST_LABEL)
    es.save(fig, NAME, "talk", build=b,
            note=("least squares, $n = 200$, batch one, 5 seeds; the table's tuned point is next to a cliff in $q$ "
                  "(one seed lost)"))


def main():
    D = data()
    print_all(D)
    draw(D)


if __name__ == "__main__":
    main()
