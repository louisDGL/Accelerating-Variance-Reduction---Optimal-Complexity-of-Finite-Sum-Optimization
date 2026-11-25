"""Figure T13 of the lecture (paper/talk/figures/figT13_two_floors): a polar step has two floors, one set by its
length and one by the noise of its estimator. Explanatory figure, not in the paper; it unfolds item (M1) and
Figure 3(a) of paper/paper_saga_to_muon.tex, which overlays curves of this design in one panel.

A 2 x 2 table of small panels with shared axes (relative squared distance ||W - W_*||_F^2 / ||W_*||_F^2, log scale,
against passes = per-sample gradient evaluations / n):
    columns = the estimator fed to the polar factor:  minibatch gradient, b = 32 (blue: no table)
                                                      SAGA table, batch one       (orange)
    rows    = the length rule of the step W+ = W - (length) polar(estimator):
              fixed length eta = 1e-2 (spectral norm of the step);
              nuclear-scaled length gamma ||estimator||_* (it vanishes with the estimator; gamma is fixed and not
              tuned: 0.5 / L_op for the minibatch gradient, 0.25 / R_op^2 for the SAGA estimator).

Data: experiment M3 (muon/experiments/exp_M3_floors.py), softmax memory-limited instance "soft-mem" (n = 1000,
d = 50, K = 10, kappa = 1e3, alpha = 1, data seed 0), algorithm seed 0; nothing is rerun. From
muon/results/M3_summary.json, results["soft-mem"][<key>]:
    ["trace"]       the run from W_0 = 0 (60 passes): thin line;
    ["trace_stat"]  the run started at W_* + 1e-7 ||W_*||_F P (relative squared distance 1e-14, table of the
                    estimator initialized there; 30 passes): wide light line of the same hue;
    ["reached"], ["stationary"]  the levels of the three algorithm seeds (median over the last sixth of the run from
                    W_0 = 0; median over the second half of the run from W_*), compared with muon/results/digest.md.
Keys: "minibatch b=32, fixed length 1e-2", "SAGA est., fixed length 1e-2", "minibatch b=32, nuclear-scaled",
"SAGA est., nuclear-scaled". A floor is a level that a run reaches from above and that a run started at the minimizer
climbs to; a method without floor stays below its starting level 1e-14. The script stops if a level it prints on the
figure disagrees with the digest, or if a cell does not show what its verdict says (tests in check()).

Status of the elements.
  measured (one instance, one seed drawn, three seeds for the levels): the four pairs of curves and the two levels
      written in the cells, 4.6e-7 (SAGA table, fixed length 1e-2: 3.8 eta^2 in absolute squared distance, see
      figT13b) and 1.2e-1 (minibatch gradient, nuclear-scaled length);
  proved: a step of fixed spectral length eta keeps max(||W_t - W_*||_op, ||W_{t+1} - W_*||_op) >= eta / 2, whatever
      the estimator (fact F2 of muon/notes_theory.md; Lemma 4 of Jawanpuria et al. for exact gradients): the "length
      floor" of the first row;
  not resolved by these runs: the level of the top-left cell. After 60 passes the run from W_0 = 0 is still going
      down (2.6e-1) and after 30 passes the run from W_* is still going up (1.5e-3): the floor lies between the two,
      at least 3000 times above the length floor of the SAGA table at the same length; the verdict "both floors"
      says that both obstructions are present, not which one sets the level.
The verdict "rounding precision" of the bottom-right cell is about the SAGA estimator (in M3 the loopless-SVRG
estimator is not run, and a momentum (STORM) buffer with the nuclear-scaled length keeps a floor: 4.4e-2 at b = 32).
In each cell: in bold where the run ends; below it, with a check mark, the floor that the cell removes. The check
mark says that an obstruction is absent, not that the level is lower: the noise floor of the bottom-left cell (0.12)
lies above the unresolved level of the top-left cell (the nuclear-scaled step of a minibatch gradient, with the
untuned gamma = 0.5 / L_op, takes long steps). The direct label "from W_0 = 0" sits under the thin line of the
first cell. Not drawn: the run from W_0 = 0 of the top-left cell makes n / b = 31.25 steps of spectral length 1e-2
per pass, a total path of 18.75 in 60 passes against ||W_*||_op = 13.19 (instance rebuilt with
muon/experiments/common.problem): it is slow, and the path condition does not stop it after pass 42.

Progressive build (states s1-s4): s1 the top-left cell (the polar step of a minibatch gradient with a fixed length:
Muon without momentum and with the exact polar factor in place of ns5); s2 the top-right cell (a table, fixed length); s3 the bottom-left cell (a length that
vanishes, no table); s4 the bottom-right cell.

Deviation from the specification of the figure: Muon itself (b = 32, lr 1e-2, momentum, ns5) is not drawn in the
top-left cell: on this axis its curve lies between the two blue curves (1.3e-2 at pass 60 from W_0 = 0, 5.7e-3
from W_*), and the cell is 2.3 x 0.9 in. Its levels are printed by this script and drawn in figT13b.

The script prints every number it plots.
Run:    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/scripts/make_figT13_two_floors.py
Output: paper/talk/figures/figT13_two_floors.pdf and .png, and the states _s1 ... _s4
"""
from __future__ import annotations

import json
import os
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
from matplotlib.transforms import ScaledTranslation  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import explain_style as es  # noqa: E402

NAME = "figT13_two_floors"
RESULTS = os.path.join(es.ROOT, "muon", "results")
INSTANCE = "soft-mem"
ETA = 1e-2                                     # the fixed length of the first row
START_STAT = 1e-14                             # relative squared distance of the starting point of "trace_stat"
Y_LO, Y_HI = 1e-24, 1e2                        # below 1e-24 the runs are at rounding level (as in Figure 3(a))
Y_TICKS = (1.0, 1e-12, 1e-24)
Y_GRID = (1.0, 1e-6, 1e-12, 1e-18, 1e-24)
Y_BOLD, Y_NOTE = 1e-14, 1e-20                    # heights of the two lines of text of a cell (data units)

# (row, column) -> key in M3_summary.json ({b}: batch of the instance), state of the build
CELLS = {
    (0, 0): ("minibatch b={b}, fixed length 1e-2", 1),
    (0, 1): ("SAGA est., fixed length 1e-2", 2),
    (1, 0): ("minibatch b={b}, nuclear-scaled", 3),
    (1, 1): ("SAGA est., nuclear-scaled", 4),
}
MUON_KEY = "Muon b={b}, lr 1e-2"               # printed, not drawn (see the docstring)
STORM_KEY = "STORM buffer b={b}, nuclear-scaled"   # printed only: the buffer tested with the nuclear-scaled length


# ----------------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------------
def load(name):
    with open(os.path.join(RESULTS, name + ".json"), encoding="utf-8") as f:
        return json.load(f)


def _trace(t):
    """Recorded passes and relative squared distances of a stored trace (finite, positive values)."""
    p = np.asarray(t["passes"], dtype=float)
    r = np.asarray([np.nan if v is None else v for v in t["rel"]], dtype=float)
    ok = np.isfinite(r) & (r > 0)
    return p[ok], r[ok]


def _series(block, key):
    e = block[key]
    p0, r0 = _trace(e["trace"])
    p1, r1 = _trace(e["trace_stat"])
    return {"key": key, "kind": e["kind"], "diverged": e["diverged"],
            "zero": (p0, r0), "star": (p1, r1),
            "reached": np.asarray(e["reached"], dtype=float), "stationary": np.asarray(e["stationary"], dtype=float)}


def data():
    m3 = load("M3_summary")
    inst = m3["instances"][INSTANCE]
    b = inst["batch"]
    block = m3["results"][INSTANCE]
    D = {"spec": inst["spec"], "budget": inst["passes"], "batch": b, "seeds": m3["seeds"], "cells": {}}
    for rc, (key, state) in CELLS.items():
        D["cells"][rc] = dict(_series(block, key.format(b=b)), state=state)
    D["muon"] = _series(block, MUON_KEY.format(b=b))
    D["storm"] = _series(block, STORM_KEY.format(b=b))
    return D


def _mmm(v, fmt="{:.1e}"):
    return f"{fmt.format(np.median(v))} [{fmt.format(v.min())}, {fmt.format(v.max())}]"


def _digest_rows():
    """Rows of the M3 table of the instance in muon/results/digest.md: {configuration: [cells]}."""
    rows, inside = {}, False
    with open(os.path.join(RESULTS, "digest.md"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("### "):
                inside = line[4:].startswith(INSTANCE + " (") and "60 passes" in line
            elif line.startswith("## "):
                inside = False
            elif inside and line.startswith("|"):
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                rows[cells[0]] = cells
    return rows


def check(D):
    """Stop if a level disagrees with the digest or if a cell does not show what its verdict says."""
    rows = _digest_rows()
    bad = []
    for s in list(D["cells"].values()) + [D["muon"], D["storm"]]:
        r = rows.get(s["key"])
        mine = (_mmm(s["reached"]), _mmm(s["stationary"]))
        if r is None or (r[1], r[2]) != mine:
            bad.append(f"{s['key']}: JSON {mine}, digest {None if r is None else (r[1], r[2])}")
        if s["diverged"]:
            bad.append(f"{s['key']}: {s['diverged']} diverged run(s)")
    print(f"digest check (muon/results/digest.md, M3 table of {INSTANCE}, columns 'reached from W_0 = 0' and "
          f"'stationary level'): {2 * (len(D['cells']) + 2)} cells compared, {len(bad)} disagreement(s)")
    if bad:
        raise SystemExit("the figure is not drawn:\n  " + "\n  ".join(bad))
    c = D["cells"]
    tests = {
        # a floor: the run from W_* climbs far above its start; no floor: it never exceeds it
        "top-left: the run from W_* leaves the minimizer": c[0, 0]["star"][1][-1] > 1e8 * START_STAT,
        "top-left: the run from W_0 = 0 ends above the run from W_*": c[0, 0]["zero"][1][-1] > c[0, 0]["star"][1][-1],
        "top-left: the run from W_* ends 1000 times above the level of the SAGA table at the same length":
            c[0, 0]["star"][1][-1] > 1e3 * np.median(c[0, 1]["stationary"]),
        "top-right: the two runs meet (levels within 5 %)":
            abs(np.median(c[0, 1]["reached"]) / np.median(c[0, 1]["stationary"]) - 1.0) < 0.05,
        "bottom-left: the two runs meet (levels within 5 %)":
            abs(np.median(c[1, 0]["reached"]) / np.median(c[1, 0]["stationary"]) - 1.0) < 0.05,
        "bottom-right: the run from W_* never exceeds its start": c[1, 1]["star"][1].max() <= START_STAT * (1 + 1e-9),
        "bottom-right: the run from W_0 = 0 goes below the axis": c[1, 1]["zero"][1].min() < Y_LO,
    }
    for name, ok in tests.items():
        print(f"    {'ok  ' if ok else 'FAIL'} {name}")
    if not all(tests.values()):
        raise SystemExit("the figure is not drawn: a cell does not show what its verdict says")


def print_all(D):
    s = D["spec"]
    print(f"instance {INSTANCE}: {s['loss']}, n = {s['n']}, d = {s['d']}, K = {s['K']}, kappa = {s['kappa']:g}, "
          f"alpha = {s['alpha']:g}, data seed {s['seed']}; minibatch methods use b = {D['batch']}; curves: algorithm "
          f"seed {D['seeds'][0]}; levels: seeds {D['seeds']}")
    names = {(0, 0): "top-left", (0, 1): "top-right", (1, 0): "bottom-left", (1, 1): "bottom-right"}
    for rc, c in list(D["cells"].items()) + [("muon", D["muon"]), ("storm", D["storm"])]:
        p0, r0 = c["zero"]
        p1, r1 = c["star"]
        print(f"- {names.get(rc, 'not drawn')}: '{c['key']}' [{c['kind']}]")
        below = np.nonzero(r0 < Y_LO)[0]
        print(f"    from W_0 = 0: {len(p0)} points, passes {p0[0]:g}-{p0[-1]:g}; value at the last point {r0[-1]:.2e}, "
              f"at pass {p0[len(p0) // 2]:.1f}: {r0[len(p0) // 2]:.2e}"
              + (f"; below {Y_LO:g} (bottom of the axis) from pass {p0[below[0]]:.1f}" if len(below) else ""))
        below = np.nonzero(r1 < Y_LO)[0]
        print(f"    from W_*:     {len(p1)} points, passes {p1[0]:g}-{p1[-1]:g}; start {r1[0]:.1e}, largest value "
              f"{r1.max():.2e}, last {r1[-1]:.2e}"
              + (f"; below {Y_LO:g} from pass {p1[below[0]]:.1f}" if len(below) else ""))
        print(f"    levels over {len(c['reached'])} seeds, median [min, max]: reached from W_0 = 0 {_mmm(c['reached'])}; "
              f"stationary (from W_*) {_mmm(c['stationary'])}")
    c = D["cells"]
    print(f"written on the figure: top-right level {np.median(c[0, 1]['stationary']):.1e} (stationary, median of "
          f"{len(c[0, 1]['stationary'])} seeds); bottom-left level {np.median(c[1, 0]['stationary']):.1e} (stationary; "
          f"reached from W_0 = 0: {np.median(c[1, 0]['reached']):.1e}; the paper quotes 1.2e-1)")
    print(f"ratio (top-left run from W_* at its last point) / (top-right stationary level) = "
          f"{c[0, 0]['star'][1][-1] / np.median(c[0, 1]['stationary']):.0f}")


# ----------------------------------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------------------------------
def _sci(v):
    """4.6e-07 -> mathtext 4.6 x 10^-7."""
    m, e = f"{v:.1e}".split("e")
    return rf"${m}\cdot10^{{{int(e)}}}$"


def draw(D):
    ps = es.setup("talk")
    ink = ps.INK
    hue = {0: ps.BLUE, 1: ps.ORANGE}              # column 0: no table (minibatch gradient); column 1: SAGA table
    fig = es.figure("talk", height=2.75)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.04, hspace=0.05)
    gs = fig.add_gridspec(2, 2)
    axes = {}
    for r in range(2):
        for c in range(2):
            axes[r, c] = fig.add_subplot(gs[r, c], sharex=axes.get((0, 0)), sharey=axes.get((0, 0)))
    b = es.Build()
    cells = D["cells"]

    for (r, c), ax in axes.items():
        cell = cells[r, c]
        st = cell["state"]
        p1, r1 = cell["star"]
        p0, r0 = cell["zero"]
        b.add(st, ax.plot(p1, r1, color=es.tint(hue[c], 0.42), lw=5.0, solid_capstyle="butt", zorder=2))
        b.add(st, ax.plot(p0, r0, color=hue[c], lw=1.6, zorder=3))
        ax.set_yscale("log")
        ax.grid(axis="y", visible=False)
        ax.hlines(Y_GRID, 0, D["budget"], color=ink["grid"], lw=0.8, zorder=1)
        ax.tick_params(labelleft=(c == 0), labelbottom=(r == 1))
    ax0 = axes[0, 0]
    ax0.set_xlim(0, D["budget"])
    ax0.set_xticks(np.arange(0, D["budget"] + 1, 20))
    ax0.set_ylim(Y_LO, Y_HI)
    ax0.set_yticks(Y_TICKS)
    ax0.minorticks_off()
    for c in range(2):
        axes[1, c].set_xlabel("passes")
    fig.supylabel(r"$\|W - W_\ast\|_F^2\ /\ \|W_\ast\|_F^2$", fontsize=es.MODES["talk"]["label"],
                  color=ink["secondary"])

    # headers of the table: columns = estimator (above), rows = length rule (strips on the right)
    axes[0, 0].set_title(f"minibatch gradient, $b = {D['batch']}$", loc="center")
    axes[0, 1].set_title("SAGA table, $b = 1$", loc="center")
    strip = dict(xycoords="axes fraction", rotation=270, ha="left", va="center", fontsize=es.small() + 0.5,
                 color=ink["primary"], fontweight="bold", linespacing=1.15, annotation_clip=False)
    axes[0, 1].annotate("fixed length\n" + r"$\eta = 10^{-2}$", (1.03, 0.5), **strip)
    axes[1, 1].annotate("length\n" + r"$\propto \|\omega\|_*$", (1.03, 0.5), **strip)

    # verdicts (ink): in bold where the run ends, below it what the cell removes (check mark); direct labels in the
    # first cell; a ring at the starting point of the runs started at the minimizer
    verdict = dict(ha="right", va="center", fontsize=es.small() + 0.5, color=ink["primary"], fontweight="bold",
                   zorder=6)
    note = dict(ha="right", va="center", fontsize=es.small(), color=ink["secondary"], zorder=6)
    x_r = D["budget"] - 1.5
    lvl_tr = float(np.median(cells[0, 1]["stationary"]))
    lvl_bl = float(np.median(cells[1, 0]["stationary"]))
    for (r, c), ax in axes.items():
        p1, r1 = cells[r, c]["star"]
        b.add(cells[r, c]["state"], ax.plot([p1[0]], [r1[0]], ls="", marker="o", ms=5.5, mfc=es.tint(hue[c], 0.42),
                                            mec=ink["surface"], mew=1.0, zorder=4, clip_on=False))

    def removed(ax, state, y, text):
        """A check mark and the name of what the cell removes, right-aligned at x_r."""
        t = ax.text(x_r, y, text, **note)
        fig.canvas.draw()                                   # the width of a text in points does not depend on the layout
        w = t.get_window_extent().width * 72.0 / fig.dpi
        m = es.mark(ax, x_r, y, True)
        m.set_transform(ax.transData + ScaledTranslation(-(w + 7.0) / 72.0, 0.0, fig.dpi_scale_trans))
        b.add(state, t, m)

    ax = axes[0, 0]
    p1, r1 = cells[0, 0]["star"]
    b.add(1, ax.text(x_r, Y_BOLD, "both floors", **verdict))
    b.add(1, ax.text(x_r, 3e-2, "from $W_0 = 0$", **dict(note, va="top")))      # under the thin line
    b.add(1, ax.text(x_r, Y_NOTE, "level between the two runs", **note))
    b.add(1, ax.annotate(r"started at $W_\ast$", (p1[0], r1[0]), xytext=(7, 0), textcoords="offset points",
                         **dict(note, ha="left")))
    b.add(2, axes[0, 1].text(x_r, Y_BOLD, "length floor: " + _sci(lvl_tr), **verdict))
    removed(axes[0, 1], 2, Y_NOTE, "noise floor removed")
    b.add(3, axes[1, 0].text(x_r, Y_BOLD, f"noise floor: {lvl_bl:.2f}", **verdict))
    removed(axes[1, 0], 3, Y_NOTE, "length floor removed")
    b.add(4, axes[1, 1].text(x_r, 1e-2, "rounding precision", **verdict))
    removed(axes[1, 1], 4, 1e-8, "both floors removed")

    s = D["spec"]
    es.save(fig, NAME, "talk", build=b,
            note=(rf"softmax, $n = {s['n']}$; curves: seed {D['seeds'][0]}, levels: median of {len(D['seeds'])} "
                  r"seeds; third way out: cooling a fixed length (M5)"))


def main():
    D = data()
    print_all(D)
    check(D)
    draw(D)


if __name__ == "__main__":
    main()
