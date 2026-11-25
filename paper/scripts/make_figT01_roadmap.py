"""Figure T01 of the lecture (paper/talk/figures/figT01_roadmap): the plan of the lecture as a grid, the four
ingredients behind the linear rate of SAGA and what becomes of each of them in Muon-type (polar) steps. Explanatory
figure, not in the paper. It is Table 2 of paper/paper_saga_to_muon.tex ("The properties behind the rate of SAGA, and
what becomes of them in Muon-type steps") in its CORRECTED wording (paper/audit/paper_saga_to_muon_corrige.tex), cut
down to one short sentence per cell.

The grid: four rows, the ingredients
    1  Geometry: the global constants mu, R^2 certify the rate        (blue tint: first half of the title of the paper)
    2  Estimator: unbiased                                            (orange tint: second half of the title)
    3  Estimator: its error vanishes at the minimizer                 (orange tint)
    4  Memory: a stale entry of the table keeps its error             (orange tint)
and three columns: "in SAGA", "through the polar map", "what follows". Each cell holds one sentence cut from the
corrected Table 2, and on its last line the status tag of the statement and its label in the paper:
    [p] proved, [m] measured exactly along runs, [e] experiment
(the tags of Section 2.4 of the paper; the fourth tag, [f] formal expansion, belongs to Proposition 4, the size of
the offset, which the grid does not show). In the column "through the polar map" a mark says what becomes of the
ingredient: a cross with the word "lost" (unbiasedness), a check with the word "kept" (vanishing error, memory), a
neutral dash for the geometry (a norm neither keeps nor loses the worst case: Theorem 1 says that none improves it).
A mark is a statement about the ingredient, not about the status of the sentence: the status is the tag (the check of
row 4 stands next to an [e] tag: the table works as before, what the sampling order changes is measured).

The cells, and the sentence of the corrected Table 2 behind each:
    row 1   the proof loses at the global constants [m] (C2): I2, I4 | no norm improves the logistic worst case
            [p] Theorem 1 | another norm buys trajectory properties [e] (M3), (M6)
    row 2   exact in a linear step, E omega = F'(theta) [p] | lost: the mean step is a mirror step [p] Theorem 2 |
            grad F -> -C, C a nuclear median of the noise [p] Thm. 2, sum eta_t^2 < infinity = sum eta_t
    row 3   removes the variance term: linear rate [p] Proposition 5.9 of Bach (2024) | kept: the minimizer is the
            only fixed point [p] Proposition 3(c) | rounding precision with a length that vanishes [e] (M1), (M5)
    row 4   in expectation, at most 1 nat per pass (2 under a permutation) [p] linear model, note B4 | kept, and the
            order of sampling matters [e] (M7) | reshuffled (a fresh permutation per pass) at b = 20, 50: 36-49 % fewer
            passes [e] (M7), memory-limited
Wording that the audit of the paper requires (paper/audit/AUDIT.md, findings G, H, I, J), and the qualifiers added by
the review of the figure:
  - the memory rates are rates of the EXPECTED squared error of LINEAR SAGA (common Hessian, n -> infinity; note B4):
    the cell says "in expectation" and "linear model";
  - the 36-49 % of passes saved by a fresh permutation are measured on the two memory-limited instances at batches of
    20 and 50: the cell says "b = 20, 50", "memory-limited" and carries the tag [e] (on the conditioning-limited control
    soft-cond of the same experiment the saving is 24 % at b = 20 and 1 % at b = 50: printed by the script, not shown);
  - Theorem 1 needs a loss curvature that vanishes along a sequence (logistic, softmax); on ridge least squares a
    metric does improve the condition number (paper, after Theorem 1): the cell says "the logistic worst case", as the
    title of Section 3.1 does;
  - the limit of Theorem 2 needs deterministic lengths with sum eta_t = infinity and sum eta_t^2 < infinity (and
    additive noise, no momentum); a geometric cooldown has sum eta_t < infinity and is NOT covered: the tag line of
    that cell states the two sums;
  - "the same exact anatomy on quadratics", "MISO at batch one and untuned steps" and "(B-)SAGA with minibatches" are
    NOT printed (no room for their qualifiers): the cells say only what Theorem 1, (M1), (M3), (M5), (M6) state.
The cell "rounding precision" is about the exact polar factor; with ns5 (eps = 1e-7) the floor of (M5) is near 1e-18.

Data. Everything is text of the paper except one number, "36-49 %": it is recomputed here from
muon/results/M9_summary.json (keys D/<instance>/cells/<b>/schemes/<iid|reshuffle>/fastest/median, instances soft-mem
and ls-mem, b = 20 and 50: passes to 1e-9 at the fastest cooldown that every seed follows, median of five seeds) and
checked against the passes quoted in (M7) of the paper (16.7-20.0 with independent batches, 9.1-12.2 with a fresh
permutation per pass; the same cells are in muon/results/digest.md). The script stops if they disagree. It also prints,
from the same keys, the conditioning-limited control soft-cond (not on the figure).
"At most 1 nat per pass, 2 under a permutation" are the proved rates of note B4 as quoted in (M7), not data.

Status of the elements: each cell carries the tag of its statement; nothing is measured or proved by this figure.

Progressive build (states s1-s3): s1 the row headers and the column "in SAGA"; s2 adds "through the polar map";
s3 adds "what follows". Four more files, figT01_roadmap_r1 ... _r4, show the complete grid with all rows but one
greyed (muted ink, tints removed, marks in grey): progress markers between the parts of the lecture.

Helper not in explain_style: `tag_line` (status tags and labels laid out on one line from measured text widths).

Run:    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/scripts/make_figT01_roadmap.py
Output: paper/talk/figures/figT01_roadmap.pdf and .png, the states _s1 ... _s3 and the variants _r1 ... _r4
"""
from __future__ import annotations

import json
import os
import sys

sys.dont_write_bytecode = True

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import explain_style as es  # noqa: E402

NAME = "figT01_roadmap"
SUMMARY = os.path.join(es.ROOT, "muon", "results", "M9_summary.json")
M7_INSTANCES = ("soft-mem", "ls-mem")          # the two memory-limited instances of (M7)
M7_BATCHES = ("20", "50")
CONTROL = "soft-cond"                           # the conditioning-limited control of M9 part D (printed, not shown)
PAPER_M7 = {"iid": (16.7, 20.0), "reshuffle": (9.1, 12.2), "saved": (36, 49)}    # (M7) of the paper, to be reproduced
NOTE = "[p] proved    [m] measured exactly along runs    [e] experiment    (Table 2 of the paper, condensed)"

# geometry of the grid, in inches (the axes fill the canvas above the scope line: data units are inches) --------------
W_IN, H_IN = es.TALK_W, es.TALK_H
NOTE_H = 0.20                                   # what es.save reserves for the scope line
X0 = 0.015                                      # left margin
COL_W = (1.08, 1.47, 1.62, 1.63)                # row header, "in SAGA", "through the polar map", "what follows"
COL_GAP = 0.02                                  # (the last column is the widest: its tag lines carry the hypotheses)
HEAD_H = 0.23                                   # band of the column headers
ROW_H, ROW_GAP = 0.545, 0.035
PAD = 0.06                                      # text inset in a cell
GUTTER = 0.16                                   # room of the kept / lost mark in the column "through the polar map"
LINE = 1.2                                      # line spacing of the cell texts


# ----------------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------------
def data():
    """Passes to 1e-9 on the memory-limited instances (M7) and the share of passes saved by a fresh permutation."""
    with open(SUMMARY, encoding="utf-8") as f:
        m9 = json.load(f)
    cells = {}
    for inst in M7_INSTANCES:
        for b in M7_BATCHES:
            sch = m9["D"][inst]["cells"][b]["schemes"]
            iid, per = sch["iid"]["fastest"]["median"], sch["reshuffle"]["fastest"]["median"]
            cells[(inst, b)] = {"iid": iid, "reshuffle": per, "saved": 100.0 * (1.0 - per / iid)}
    rng = {k: (min(c[k] for c in cells.values()), max(c[k] for c in cells.values()))
           for k in ("iid", "reshuffle", "saved")}
    saved = (int(round(rng["saved"][0])), int(round(rng["saved"][1])))
    if rng["iid"] != PAPER_M7["iid"] or rng["reshuffle"] != PAPER_M7["reshuffle"] or saved != PAPER_M7["saved"]:
        raise SystemExit(f"(M7) is not reproduced from {SUMMARY}: {rng} against {PAPER_M7}")
    control = {}                                # not shown: why the cell says "memory-limited"
    for b in M7_BATCHES:
        sch = m9["D"][CONTROL]["cells"][b]["schemes"]
        iid, per = sch["iid"]["fastest"]["median"], sch["reshuffle"]["fastest"]["median"]
        control[b] = {"iid": iid, "reshuffle": per, "saved": 100.0 * (1.0 - per / iid)}
    return {"cells": cells, "range": rng, "saved": saved, "control": control}


def rows(D, ps):
    """The four rows of the grid: (colour of the row, title, description, cell "in SAGA", cell "through the polar map",
    cell "what follows"). A cell is (mark, sentence, [(tag, label), ...]); mark: True kept, False lost, None none,
    "dash" neutral."""
    lo, hi = D["saved"]
    return [
        (ps.BLUE, "Geometry", "global $\\mu$, $R^2$\ncertify the rate",
         (None, "the proof loses at\nthe global constants", [("m", "(C2): I2, I4")]),
         ("dash", "no norm improves the\nlogistic worst case", [("p", "Theorem 1")]),
         (None, "another norm buys\ntrajectory properties", [("e", "(M3), (M6)")])),
        (ps.ORANGE, "Estimator", "unbiased",
         (None, "exact in a linear step:\n" + r"$\mathrm{E}\,\omega = F'(\theta)$", [("p", "by construction")]),
         (False, r"$\mathbf{lost}$: the mean step" + "\nis a mirror step", [("p", "Theorem 2")]),
         (None, r"$\nabla F\,\to\,{-C}$, $C$ a nuclear" + "\nmedian of the noise",
          [("p", r"Thm. 2, $\Sigma\eta_t^2<\infty=\Sigma\eta_t$")])),
        (ps.ORANGE, "Estimator", "error vanishes\nat the minimizer",
         (None, "removes the variance\nterm: linear rate", [("p", "Bach, Prop. 5.9")]),
         (True, r"$\mathbf{kept}$: the minimizer is" + "\nthe only fixed point", [("p", "Proposition 3(c)")]),
         (None, "rounding precision with\na length that vanishes", [("e", "(M1), (M5)")])),
        (ps.ORANGE, "Memory", "a stale entry\nkeeps its error",
         (None, "in expectation, ≤ 1 nat\nper pass (permuted ≤ 2)", [("p", "linear model, B4")]),
         (True, r"$\mathbf{kept}$, and the order" + "\nof sampling matters", [("e", "(M7)")]),
         (None, f"permuted at b = 20, 50:\n{lo}–{hi} % fewer passes", [("e", "(M7), memory-limited")])),
    ]


def print_all(D):
    print(f"(M7), from {os.path.relpath(SUMMARY, es.ROOT)}: passes to 1e-9 at the fastest cooldown, median of 5 seeds")
    for (inst, b), c in D["cells"].items():
        print(f"    {inst:8s} b = {b:>2s}: independent batches {c['iid']:.1f}, fresh permutation {c['reshuffle']:.1f}, "
              f"saved {c['saved']:.1f} %")
    r = D["range"]
    print(f"    ranges: independent {r['iid'][0]:.1f}-{r['iid'][1]:.1f}, permutation {r['reshuffle'][0]:.1f}-"
          f"{r['reshuffle'][1]:.1f}, saved {r['saved'][0]:.1f}-{r['saved'][1]:.1f} % -> printed "
          f"{D['saved'][0]}-{D['saved'][1]} % (paper, (M7): {PAPER_M7['saved'][0]}-{PAPER_M7['saved'][1]} %)")
    for b, c in D["control"].items():
        print(f"    not shown, control {CONTROL} (conditioning-limited) b = {b:>2s}: independent {c['iid']:.1f}, "
              f"permutation {c['reshuffle']:.1f}, saved {c['saved']:.1f} % (hence 'memory-limited' on the figure)")
    print("the other numbers of the figure (1 nat, 2 nats per pass; b = 20, 50) are quoted from (M7) and note B4")


# ----------------------------------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------------------------------
def text_width(fig, artist):
    """Width of a text artist in inches (= data units of the diagram)."""
    return artist.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi


def tag_line(fig, ax, x, y, items, colour, edge=None):
    """Status tags and their labels on one line starting at x, e.g. [e] (M1), (M5). edge: colour of the frame of the
    tags (None: the default of explain_style.status_tag). Returns the artists."""
    out = []
    fs = es.small()
    pad = 0.22 * fs / 72.0                                     # padding of the tag box (explain_style.status_tag)
    for tag, text in items:
        t = es.status_tag(ax, x + pad, y, f"[{tag}]", fontsize=fs - 0.5)
        t.set_ha("left")
        t.set_color(colour)
        if edge is not None:
            t.get_bbox_patch().set_edgecolor(edge)
        x += text_width(fig, t) + 2 * pad + 0.045
        lab = es.label(ax, x, y, text, colour=colour, ha="left")
        x += text_width(fig, lab) + 0.09
        out += [t, lab]
    return out


def draw(D, focus=None):
    """The complete grid; focus = 1..4 greys every other row (progress marker); focus = None draws the build."""
    ps = es.setup("talk")
    ink = ps.INK
    fig = es.figure("talk", height=H_IN)
    fig.get_layout_engine().set(w_pad=0.0, h_pad=0.0, wspace=0.0, hspace=0.0)
    ax = fig.add_subplot(111)
    es.diagram_axes(ax, (0.0, W_IN), (0.0, H_IN - NOTE_H))
    b = es.Build()
    top = H_IN - NOTE_H
    xs = [X0]
    for w in COL_W[:-1]:
        xs.append(xs[-1] + w + COL_GAP)                         # left edge of each column

    # column headers
    heads = ("ingredient", "in SAGA", "through the polar map", "what follows")
    for k, text in enumerate(heads):            # centred: left-aligned, "through the polar map" ran into "what follows"
        t = es.label(ax, xs[k] + 0.5 * COL_W[k], top - 0.5 * HEAD_H - 0.01, text, fontsize=es.small() + 0.5,
                     ha="center", weight="bold", colour=ink["secondary"])
        b.add(max(k, 1), t)

    for i, (colour, title, sub, *cells) in enumerate(rows(D, ps)):
        y_top = top - HEAD_H - i * (ROW_H + ROW_GAP)
        yc = y_top - 0.5 * ROW_H
        dim = focus is not None and focus != i + 1
        c_main = ink["muted"] if dim else ink["primary"]
        c_sub = ink["muted"] if dim else ink["secondary"]
        edge = ink["grid"] if dim else ink["axis"]
        # row header (state 1)
        patch, _ = es.box(ax, xs[0] + 0.5 * COL_W[0], yc, COL_W[0], ROW_H, colour=None if dim else colour,
                          edge=edge if dim else None, rounding=0.016, lw=0.8)
        t1 = ax.text(xs[0] + PAD, y_top - 0.065, title, ha="left", va="top", fontsize=es.small() + 1.5,
                     fontweight="bold", color=c_main, zorder=5)
        t2 = ax.text(xs[0] + PAD, y_top - 0.235, sub, ha="left", va="top", fontsize=es.small(), color=c_sub,
                     zorder=5, linespacing=LINE)
        b.add(1, patch, t1, t2)
        # the three cells (states 1, 2, 3)
        for k, (mark, sentence, tags) in enumerate(cells):
            state = k + 1
            x_left, w = xs[k + 1], COL_W[k + 1]
            patch, _ = es.box(ax, x_left + 0.5 * w, yc, w, ROW_H, edge=edge, rounding=0.016, lw=0.7)
            arts = [patch]
            x_text = x_left + PAD
            if k == 1:                                          # the column of the polar map: kept / lost / neutral
                y_mark = y_top - 0.062 - 0.5 * es.small() * LINE / 72.0
                if mark == "dash":
                    m = es.label(ax, x_left + PAD + 0.055, y_mark, "–", fontsize=es.small() + 3,
                                 colour=ink["muted"], weight="bold")
                else:
                    m = es.mark(ax, x_left + PAD + 0.055, y_mark, bool(mark))
                    if dim:
                        m.set_color(ink["muted"])
                arts.append(m)
                x_text += GUTTER
            arts.append(ax.text(x_text, y_top - 0.062, sentence, ha="left", va="top", fontsize=es.small(),
                                color=c_main, zorder=5, linespacing=LINE))
            arts += tag_line(fig, ax, x_text, y_top - ROW_H + 0.105, tags, c_sub, ink["grid"] if dim else None)
            b.add(state, *arts)

    for a in ax.get_children():                 # the texts must not push the axes: data units stay inches
        a.set_in_layout(False)
    if focus is None:
        es.save(fig, NAME, "talk", build=b, note=NOTE)
    else:
        es.save(fig, f"{NAME}_r{focus}", "talk", note=NOTE)


def main():
    D = data()
    print_all(D)
    draw(D)
    for k in (1, 2, 3, 4):
        draw(D, focus=k)


if __name__ == "__main__":
    main()
