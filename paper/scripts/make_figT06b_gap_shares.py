"""Figure T06b of the lecture (paper/talk/figures/figT06b_gap_shares): which inequalities of the proof of Proposition
5.9 carry the gap between the exact and the certified contraction exponents of SAGA, in the two orders of attribution
that were computed. On this instance (spectrum j^(-1), alpha = 1), for kappa >= 1e4 the two inequalities that put a
global constant in the place of a quantity of the iterate (I2 with R^2, I4 with mu) carry about nine tenths of the
logarithmic gap in either order (90-92 %); at kappa = 1e2 their share depends on the order (48 % or 0 %). Over the
three spectra of the experiment the range for kappa >= 1e4 is wider, 77-96 % and 59-96 % (the 59 % is alpha = 0,
kappa = 1e4, other order): a slide title must not drop "spectrum j^(-1)". Explanatory figure, not in the paper: it recuts the numbers behind
Figure 1(c) and finding (C2) of paper/paper_saga_to_muon.tex and adds the second order, which the paper does not
show (its audit does). It is the talk version of panel (b) of paper/figures/figE1_proof_chain, with kappa = 1e4 added.

What is drawn. Ridge logistic regression, n = 1000, d = 50, covariance spectrum j^(-1) (alpha = 1), SAGA with the
constants of Proposition 5.9, kappa = 1e2, 1e4 and 1e5. One horizontal stacked bar per kappa and per order: the shares
of the logarithmic gap, (c_{k-1} - c_k) / (c_exact - c_bound) with c = -log(rho) the window means of the contraction
exponents over passes 2-20 (mean over five sampling seeds, one data set). Orders:
    "I5 undone first": the order of the paper, undo I5, I4, I2, I3, I1 from the bound (order B of
                       theory/B3_prop59_tightness.tex; key "shares");
    "I5 undone last":  undo I4, I2, I3, I1, then I5 (order A of the note: the maximum of the two coefficients of the
                       bound is kept until the end; key "shares_orderA", vrlab.analysis.STEP_KEYS_A).
Segments in the fixed order I2, I4, I1, I3, I5 (colours: plotstyle.INEQ), so that the length of the first two is
I2 + I4, printed at the right of each bar. Left: kappa and the size of the gap, exact over certified exponent (key
"speedup"). The legend names I3 with its largest plotted share ("<= 0.2 %", checked: its segments are invisible), and
its free slot states the order of the four other inequalities, the same in both orders (I4, I2, I3, I1).

Source (read only): results/B3_summary.json, attribution["kappa=<k>,alpha=1"] (shares, shares_orderA, speedup,
speedup_late_20_100). Checks, the script stops if one fails: the shares of each order sum to one; speed-up and the
ten shares of the three configurations agree with results/digest.md (tables "Attribution of the log-gap" and
"Attribution in order A") at its printed precision; over the 15 configurations of the experiment, I2 + I4 for
kappa >= 1e4 is 77-96 % in the order of the paper ((C2)) and 59-96 % in the other order (audit of the paper).

Status. Measured exactly along runs (every conditional expectation by enumeration of the n indices). A share is a
fraction of a difference of logarithms of exponents, not a percentage of the speed-up; it depends on the order of
attribution and on the window (the ratios of the exponents on passes 20-100 are printed below).

Progressive build (states s1-s2): s1 the order of the paper; s2 adds the other order.

The script prints every number it shows.
Run:    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/scripts/make_figT06b_gap_shares.py
Output: paper/talk/figures/figT06b_gap_shares.pdf and .png, and the states _s1, _s2
"""
from __future__ import annotations

import json
import os
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import explain_style as es  # noqa: E402

NAME = "figT06b_gap_shares"
KAPPAS = [10.0, 100.0, 1000.0, 10000.0, 100000.0]
ALPHAS = [0.0, 1.0, 2.0]
ALPHA_FIG, KAPPAS_FIG = 1.0, (100.0, 10000.0, 100000.0)
INEQS = ["I1", "I2", "I3", "I4", "I5"]
STACK = ["I2", "I4", "I1", "I3", "I5"]            # segments, left to right
NAMES = {"I2": "I2 cocoercivity, with $R^2$", "I4": "I4 strong convexity, with $\\mu$", "I1": "I1 square norm", "I3": "I3 variance (≤ 0.2 %)",
         "I5": "I5 min of rates"}
ORDERS = (("shares", "I5 undone first", 1), ("shares_orderA", "I5 undone last", 2))   # key, label, build state
UNDO = {"shares": "I5, I4, I2, I3, I1", "shares_orderA": "I4, I2, I3, I1, I5"}
DIGEST_B = "### Attribution of the log-gap (passes 2-20) and window-averaged ratios"
DIGEST_A = "### Attribution in order A of the B3 note (undo I4, I2, I3, I1, then the min I5; passes 2-20)"
QUOTED = {"order of the paper": ("77", "96"), "other order": ("59", "96")}   # I2 + I4 (%), kappa >= 1e4: (C2), audit


def key(kappa, alpha):
    return f"kappa={kappa:g},alpha={alpha:g}"


# ----------------------------------------------------------------------------------------------------------------------
# data (read only)
# ----------------------------------------------------------------------------------------------------------------------
def digest_fmt(x):
    """Number format of experiments/make_digest.py."""
    if x == 0:
        return "0"
    if abs(x) >= 1e4 or abs(x) < 1e-3:
        return f"{x:.3g}"
    return f"{x:.4g}"


def digest_table(title):
    """Rows of the table that follows the heading `title` in results/digest.md: {config: {column: cell}}."""
    with open(os.path.join(es.ROOT, "results", "digest.md"), encoding="utf-8") as f:
        lines = f.read().splitlines()
    found = [k for k, line in enumerate(lines) if line.strip() == title]
    if len(found) != 1:
        raise SystemExit(f"results/digest.md: heading {title!r} found {len(found)} times; no figure drawn")
    header = [c.strip() for c in lines[found[0] + 1].strip().strip("|").split("|")]
    rows = {}
    for line in lines[found[0] + 3:]:
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows[cells[0]] = dict(zip(header[1:], cells[1:]))
    return rows


def data():
    with open(os.path.join(es.ROOT, "results", "B3_summary.json"), encoding="utf-8") as f:
        att = json.load(f)["attribution"]
    bad = []
    for k in KAPPAS:
        for a in ALPHAS:
            for name, _, _ in ORDERS:
                s = sum(att[key(k, a)][name][i] for i in INEQS)
                if abs(s - 1.0) > 1e-9:
                    bad.append(f"{key(k, a)}: the shares of {name} sum to {s!r}")
    dig_b, dig_a = digest_table(DIGEST_B), digest_table(DIGEST_A)
    col_b = {"I1": "share I1", "I2": "I2", "I3": "I3", "I4": "I4", "I5": "I5"}
    col_a = {"I4": "share I4", "I2": "I2", "I3": "I3", "I1": "I1", "I5": "I5 (min, undone last)"}
    count = 0
    for k in KAPPAS_FIG:
        c = key(k, ALPHA_FIG)
        pairs = [("speed-up", att[c]["speedup"], dig_b[c]["speed-up exact/bound"])]
        pairs += [(f"share {i}, order of the paper", att[c]["shares"][i], dig_b[c][col_b[i]]) for i in INEQS]
        pairs += [(f"share {i}, other order", att[c]["shares_orderA"][i], dig_a[c][col_a[i]]) for i in INEQS]
        count += len(pairs)
        bad += [f"{c} {n}: summary {v!r} -> {digest_fmt(v)}, digest {cell}" for n, v, cell in pairs
                if digest_fmt(v) != cell]

    def i24(c, name):
        return att[c][name]["I2"] + att[c][name]["I4"]
    high = [key(k, a) for k in KAPPAS if k >= 1e4 for a in ALPHAS]
    ranges = {}
    for label, name in (("order of the paper", "shares"), ("other order", "shares_orderA")):
        v = [100 * i24(c, name) for c in high]
        ranges[label] = (f"{min(v):.0f}", f"{max(v):.0f}")
        if ranges[label] != QUOTED[label]:
            bad.append(f"I2 + I4, kappa >= 1e4, {label}: recomputed {ranges[label][0]}-{ranges[label][1]} %, quoted "
                       f"{QUOTED[label][0]}-{QUOTED[label][1]} %")
    i3_max = max(att[key(k, ALPHA_FIG)][name]["I3"] for k in KAPPAS_FIG for name, _, _ in ORDERS)
    if i3_max > 0.002:
        bad.append(f"largest plotted share of I3 is {i3_max!r}, above the 0.2 % written in the legend")
    if bad:
        raise SystemExit("CHECK FAILED, no figure drawn:\n   " + "\n   ".join(bad))
    groups = []
    for k in KAPPAS_FIG:
        c = key(k, ALPHA_FIG)
        groups.append({"kappa": k, "speedup": att[c]["speedup"], "late": att[c]["speedup_late_20_100"],
                       "bars": [{"order": name, "label": label, "state": state,
                                 "shares": {i: att[c][name][i] for i in INEQS}, "i24": i24(c, name)}
                                for name, label, state in ORDERS]})
    return {"groups": groups, "ranges": ranges, "digest_count": count, "i3_max": i3_max}


def fmt_ratio(x):
    """The ratio of the exponents as it is printed on the figure."""
    return f"{x:.1f}" if x < 10 else f"{x:.0f}"


def print_all(D):
    print(f"alpha = {ALPHA_FIG:g}, n = 1000, d = 50, passes 2-20, mean over 5 sampling seeds (results/B3_summary.json)")
    for g in D["groups"]:
        print(f"kappa = {g['kappa']:g}: exact / certified exponent {g['speedup']:.4g} (shown: {fmt_ratio(g['speedup'])}; "
              f"passes 20-100: {g['late']:.4g})")
        for r in g["bars"]:
            print(f"    {r['label']:15s} (undo {UNDO[r['order']]}): "
                  + ", ".join(f"{i} {r['shares'][i]:.4f}" for i in STACK)
                  + f"; I2 + I4 = {r['i24']:.4f} (shown: {100 * r['i24']:.0f} %)")
    print(f"largest plotted share of I3: {D['i3_max']:.4f} (legend: <= 0.2 %)")
    print(f"checks: {D['digest_count']} numbers agree with results/digest.md at its printed precision; I2 + I4 for "
          f"kappa >= 1e4 over the three spectra: " + "; ".join(f"{n}: {lo}-{hi} %" for n, (lo, hi) in D["ranges"].items())
          + " (as quoted by the paper and by its audit)")


# ----------------------------------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------------------------------
AXES = (0.390, 0.240, 0.515, 0.610)                # the bars, in figure fractions: the layout is fixed by hand because
#                                                    the columns of text left and right of the bars are not tick labels
X_GROUP, X_ORDER, X_SUM = -0.725, -0.02, 1.025      # these columns, in units of the x axis


def draw(D):
    from matplotlib.patches import Patch
    ps = es.setup("talk")
    ink = ps.INK
    fig = es.figure("talk", height=2.72)
    fig.set_layout_engine("none")
    ax = fig.add_axes(AXES)
    b = es.Build()
    height, pitch, group_gap = 0.80, 1.0, 0.62
    px = 1.0 / 450.0                                 # about 2 px of surface between segments (axes about 3 in wide)
    y = 0.0
    tr = ax.get_yaxis_transform()                    # x in axes fraction, y in data units
    for g in D["groups"]:
        y_top = y
        for r in g["bars"]:
            left = 0.0
            for i in STACK:
                w = r["shares"][i]
                if w > 2 * px:
                    b.add(r["state"], ax.barh(y, w - px, left=left + px / 2, height=height, color=ps.INEQ[i], lw=0,
                                              zorder=3))
                elif w > 0:
                    b.add(r["state"], ax.barh(y, w, left=left, height=height, color=ps.INEQ[i], lw=0, zorder=3))
                if w >= 0.10:
                    b.add(r["state"], ax.text(left + w / 2, y, i, ha="center", va="center", fontsize=es.small(),
                                              color=ink["primary"], zorder=4))
                left += w
            b.add(r["state"], ax.text(X_ORDER, y, r["label"], transform=tr, ha="right", va="center",
                                      fontsize=es.small(), color=ink["secondary"], clip_on=False))
            b.add(r["state"], ax.text(X_SUM, y, f"{100 * r['i24']:.0f} %", transform=tr, ha="left", va="center",
                                      fontsize=es.small() + 1.0, color=ink["primary"], clip_on=False))
            y -= pitch
        mid = 0.5 * (y_top + y + pitch)
        ax.text(X_GROUP, mid, rf"$\kappa = 10^{{{int(round(np.log10(g['kappa'])))}}}$:" "\nexact exponent is\n"
                f"{fmt_ratio(g['speedup'])} × certified", transform=tr, ha="left", va="center",
                fontsize=es.small() + 0.5, color=ink["primary"], linespacing=1.15, clip_on=False)
        y -= group_gap
    y_low = y + group_gap + pitch
    ax.text(X_SUM, 0.5 * height + 0.12, "I2 + I4", transform=tr, ha="left", va="bottom", fontsize=es.small(),
            color=ink["secondary"], clip_on=False)
    ax.set_ylim(y_low - 0.5 * height - 0.12, 0.5 * height + 0.75)
    ax.set_yticks([])
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25 %", "50 %", "75 %", "100 %"])
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("share of the logarithmic gap")
    handles = [Patch(fc=ps.INEQ[i], ec="none") for i in STACK]
    leg = ps.legend(ax, handles=handles, labels=[NAMES[i] for i in STACK], loc="upper left", bbox_to_anchor=(0.018, 0.992),
              bbox_transform=fig.transFigure, ncol=3, handlelength=1.0, handleheight=1.0, columnspacing=1.6,
              borderaxespad=0.0, borderpad=0.0, labelspacing=0.3)
    # fixed layout: the scope line is drawn with es.scope_note (save(note=...) needs the constrained layout engine)
    # the order of the four other inequalities, in the free slot of the legend (third column, second row)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    x_slot = inv.transform(leg.get_patches()[STACK.index("I5")].get_window_extent(rend))[0, 0]
    y_slot = inv.transform(leg.get_texts()[STACK.index("I3")].get_window_extent(rend)).mean(0)[1]
    fig.text(x_slot, y_slot, "others undone as I4, I2, I3, I1", ha="left", va="center", fontsize=es.small(),
             color=ink["secondary"])
    es.scope_note(fig, r"ridge logistic regression, $n = 1000$, $d = 50$, spectrum $j^{-1}$; passes 2–20, 5 sampling "
                  "seeds, one data set")
    es.save(fig, NAME, "talk", build=b)


def main():
    D = data()
    print_all(D)
    draw(D)


if __name__ == "__main__":
    main()
