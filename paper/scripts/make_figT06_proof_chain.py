"""Figure T06 of the lecture (paper/talk/figures/figT06_proof_chain): the proof of Proposition 5.9 (linear rate of
SAGA) is an exact one-step identity in which five inequalities put worst-case values; at kappa = 1e5 the two that put
a global constant in the place of a quantity of the iterate carry nine tenths of the gap. Explanatory figure, not in
the paper: it lays out Section 2.2 and Eq. (1) of paper/paper_saga_to_muon.tex and quotes numbers of finding (C2)
(Figure 1(c)). It is the talk version of panel (a) of paper/figures/figE1_proof_chain.

What is drawn:
  left, DIAGRAM (no data). Top: the exact identity (1) of the paper, E_t[V_{t+1}] = c_D D + c_M delta M, V = D + delta M,
      c_D = 1 - mu_eff/(2 R^2) (1 - L_eff/R^2 (r_sq/4 + 3/8)), c_M = 1 - (1/n)(1 - (2/3) r_sq r_var) (textbook
      constants gamma = 1/(4 R^2), delta = 3 gamma^2; theory/B3_prop59_tightness.tex, Proposition "exact contraction
      in terms of the ratios"). Then one row per inequality, in the order in which the attribution applies them
      (I1, I3, I2, I4, I5: the reverse of the order in which the paper undoes them; the book states I2 before I3, and
      the two act on different terms): I1 square norm, r_sq -> 1; I3 variance <= second moment, r_var -> 1;
      I2 cocoercivity, L_eff -> R^2; I4 strong convexity, mu_eff -> mu; I5 minimum of two rates, the mean of c_D and
      c_M weighted by D and delta M -> their maximum. Bottom: the certified contraction 1 - min{1/(3n), 3 mu/(16 R^2)}.
      I6 (initialization) changes the constant and not the rate: it is left out.
  right, DATA. For ridge logistic regression with n = 1000, d = 50, kappa = 1e5, spectrum j^(-1): the ratio of the
      exact to the certified contraction exponent (window means over passes 2-20, mean over five sampling seeds, one
      data set), and the share of the logarithmic gap carried by each inequality when they are undone from the bound
      in the order of the paper, I5, I4, I2, I3, I1 (from the bottom of the list up): one bar per row, in the colour
      of the inequality (plotstyle.INEQ), with its value; below, I2 + I4 in that order and in the other order
      computed (undo I4, I2, I3, I1, then I5).

Source (read only): results/B3_summary.json, attribution["kappa=100000,alpha=1"]: "speedup", "shares" (order of
the paper), "shares_orderA" (other order). Checks, the script stops if one fails: the shares of each order sum to
one; the speed-up and the ten shares agree with results/digest.md (tables "Attribution of the log-gap" and
"Attribution in order A") at its printed precision.

Status. Proved: the identity and the chain (main report). Measured exactly along runs (conditional expectations by
enumeration of the n indices): the ratio of the exponents and the shares; a share is a fraction of a difference of
logarithms of exponents and depends on the order of attribution and on the window (on passes 20-100 the ratio is 72
instead of 100).

Progressive build (states s1-s3): s1 the identity, the certified rate and the size of the gap, with the exponent
defined on the figure as -log of the one-step contraction factor ("where is the gap?");
s2 the five inequalities; s3 the tags "global constant" on I2 and I4 and the measured shares.

The script prints every number it shows.
Run:    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/scripts/make_figT06_proof_chain.py
Output: paper/talk/figures/figT06_proof_chain.pdf and .png, and the states _s1 ... _s3
"""
from __future__ import annotations

import json
import os
import sys

sys.dont_write_bytecode = True

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import explain_style as es  # noqa: E402

NAME = "figT06_proof_chain"
KAPPA, ALPHA = 100000.0, 1.0
CONFIG = f"kappa={KAPPA:g},alpha={ALPHA:g}"
CHAIN = ["I1", "I3", "I2", "I4", "I5"]            # rows: order in which the attribution applies the inequalities
GLOBAL = ("I2", "I4")
DIGEST_B = "### Attribution of the log-gap (passes 2-20) and window-averaged ratios"
DIGEST_A = "### Attribution in order A of the B3 note (undo I4, I2, I3, I1, then the min I5; passes 2-20)"


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


def digest_row(title, config):
    """The row `config` of the table that follows the heading `title` in results/digest.md: {column: cell}."""
    with open(os.path.join(es.ROOT, "results", "digest.md"), encoding="utf-8") as f:
        lines = f.read().splitlines()
    found = [k for k, line in enumerate(lines) if line.strip() == title]
    if len(found) != 1:
        raise SystemExit(f"results/digest.md: heading {title!r} found {len(found)} times; no figure drawn")
    header = [c.strip() for c in lines[found[0] + 1].strip().strip("|").split("|")]
    for line in lines[found[0] + 3:]:
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[0] == config:
            return dict(zip(header[1:], cells[1:]))
    raise SystemExit(f"results/digest.md: no row {config!r} under {title!r}")


def data():
    with open(os.path.join(es.ROOT, "results", "B3_summary.json"), encoding="utf-8") as f:
        att = json.load(f)["attribution"][CONFIG]
    row_b, row_a = digest_row(DIGEST_B, CONFIG), digest_row(DIGEST_A, CONFIG)
    col_b = {"I1": "share I1", "I2": "I2", "I3": "I3", "I4": "I4", "I5": "I5"}
    col_a = {"I4": "share I4", "I2": "I2", "I3": "I3", "I1": "I1", "I5": "I5 (min, undone last)"}
    pairs = [("speed-up", att["speedup"], row_b["speed-up exact/bound"])]
    pairs += [(f"share {i}, order of the paper", att["shares"][i], row_b[col_b[i]]) for i in CHAIN]
    pairs += [(f"share {i}, other order", att["shares_orderA"][i], row_a[col_a[i]]) for i in CHAIN]
    bad = [f"{n}: summary {v!r} -> {digest_fmt(v)}, digest {cell}" for n, v, cell in pairs if digest_fmt(v) != cell]
    for name in ("shares", "shares_orderA"):
        s = sum(att[name][i] for i in CHAIN)
        if abs(s - 1.0) > 1e-9:
            bad.append(f"the shares of {name} sum to {s!r}")
    if bad:
        raise SystemExit("CHECK FAILED, no figure drawn:\n   " + "\n   ".join(bad))
    return {"shares": {i: att["shares"][i] for i in CHAIN}, "speedup": att["speedup"],
            "late": att["speedup_late_20_100"], "i24": att["shares"]["I2"] + att["shares"]["I4"],
            "i24_other": att["shares_orderA"]["I2"] + att["shares_orderA"]["I4"], "checked": len(pairs)}


def percent(x):
    """A share as it is printed on the figure."""
    return f"{100 * x:.0f} %" if x >= 0.005 else f"{100 * x:.2f} %"


def print_all(D):
    print(f"{CONFIG}, n = 1000, d = 50, passes 2-20, mean over 5 sampling seeds (results/B3_summary.json)")
    print(f"exact / certified contraction exponent: {D['speedup']:.4g} (passes 20-100: {D['late']:.4g})")
    print("shares of the logarithmic gap, order of the paper (undo I5, I4, I2, I3, I1): "
          + ", ".join(f"{i} {D['shares'][i]:.6f} (shown: {percent(D['shares'][i])})" for i in CHAIN))
    print(f"I2 + I4: {D['i24']:.4f} in the order of the paper, {D['i24_other']:.4f} in the other order (undo I4, I2, "
          f"I3, I1, I5)")
    print(f"checks: {D['checked']} numbers agree with results/digest.md at its printed precision")


# ----------------------------------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------------------------------
ROWS = [  # key, name, substitution, note
    ("I1", "square norm", r"$r_{\mathrm{sq}}\;\rightarrow\;1$", r"ratio $\leq 1$"),
    ("I3", "variance $\\leq$ 2nd moment", r"$r_{\mathrm{var}}\;\rightarrow\;1$", r"ratio $\leq 1$"),
    ("I2", "cocoercivity", r"$L_{\mathrm{eff}}\;\rightarrow\;R^2$", "global constant"),
    ("I4", "strong convexity", r"$\mu_{\mathrm{eff}}\;\rightarrow\;\mu$", "global constant"),
    ("I5", "min of two rates", r"mean $\rightarrow$ max of $c_D$, $c_M$", ""),
]
X_CHAIN = 71.0              # right end of the chain; the shares are drawn to its right
X_BAR, BAR_UNIT = 74.0, 0.29   # left end of the bars, and their length per percent


def draw(D):
    from matplotlib.patches import Rectangle
    ps = es.setup("talk")
    ink = ps.INK
    fig = es.figure("talk", height=2.72)
    ax = fig.add_subplot(111)
    es.diagram_axes(ax, (-0.5, 100.5), (-1.2, 100.8))
    b = es.Build()
    f_text, f_math, f_small = 9.0, 9.5, es.small()
    xc = X_CHAIN / 2.0

    # s1: the two ends of the proof and the size of the gap
    es.box(ax, xc, 89.5, X_CHAIN, 21.0, edge=ink["secondary"], lw=1.1, rounding=0.03)
    ax.text(2.2, 95.0, "exact identity", fontsize=f_text, fontweight="bold", color=ink["primary"], va="center")
    ax.text(22.0, 95.0, r"$\mathbb{E}_t[V_{t+1}] \;=\; c_D\,D \;+\; c_M\,\delta M$", fontsize=f_math + 0.5,
            color=ink["primary"], va="center")
    ax.text(X_CHAIN - 1.8, 95.0, r"$V = D + \delta M$", fontsize=f_math, color=ink["secondary"], va="center",
            ha="right")
    ax.text(2.2, 84.4, r"$c_D = 1-\frac{\mu_{\mathrm{eff}}}{2R^2}\left(1-\frac{L_{\mathrm{eff}}}{R^2}\left("
            r"\frac{r_{\mathrm{sq}}}{4}+\frac{3}{8}\right)\right)$,   $c_M = 1-\frac{1}{n}\left(1-\frac{2}{3}\,"
            r"r_{\mathrm{sq}}\,r_{\mathrm{var}}\right)$", fontsize=f_math + 0.5, color=ink["primary"], va="center")
    es.box(ax, xc, 9.5, X_CHAIN, 19.0, edge=ink["secondary"], lw=1.1, rounding=0.03)
    ax.text(2.2, 14.3, "certified (Proposition 5.9)", fontsize=f_text, fontweight="bold", color=ink["primary"],
            va="center")
    ax.text(2.2, 5.3, r"$\mathbb{E}_t[V_{t+1}] \;\leq\; \left(1-\min\left\{\frac{1}{3n},\,\frac{3\mu}{16R^2}\right\}"
            r"\right)V$", fontsize=f_math + 0.5, color=ink["primary"], va="center")
    es.arrow(ax, (5.0, 78.6), (5.0, 19.6), lw=1.2, mutation_scale=9.0, zorder=1)
    ax.text(X_BAR, 100.0, rf"at $\kappa = 10^5$, the exact" "\n" r"exponent, $-\log$(factor)," "\n"
            rf"is ${D['speedup']:.0f}\,\times$ the certified:" "\nwhere is the gap?", fontsize=f_text,
            color=ink["primary"], va="top", ha="left", linespacing=1.22)

    # s2: the five inequalities, one worst-case value each
    top, pitch, gap = 73.0, 9.4, 1.0
    b.add(2, ax.add_patch(Rectangle((3.0, top - 5 * pitch), 4.0, 5 * pitch, fc=ink["surface"], ec="none", zorder=2)))
    b.add(2, es.label(ax, 8.0, 76.0, "the proof: one worst-case value per inequality", fontsize=f_small, ha="left"))
    for k, (name, title, subst, note) in enumerate(ROWS):
        y = top - pitch * (k + 0.5)
        col = ps.INEQ[name]
        b.add(2, es.box(ax, xc, y, X_CHAIN, pitch - gap, colour=col, fill=ink["surface"], lw=1.1, rounding=0.02))
        b.add(2, ax.add_patch(Rectangle((1.5, y - 2.5), 2.3, 5.0, fc=col, ec="none", zorder=5)))
        b.add(2, ax.text(5.2, y, name, fontsize=f_text, fontweight="bold", color=ink["primary"], va="center", zorder=5))
        b.add(2, ax.text(9.2, y, title, fontsize=f_text, color=ink["primary"], va="center", zorder=5))
        b.add(2, ax.text(38.5, y, subst, fontsize=f_math, color=ink["primary"], va="center", zorder=5))
        if name in GLOBAL:      # s3: tint and tag
            b.add(3, es.box(ax, xc, y, X_CHAIN, pitch - gap, colour=col, fill=es.tint(col, 0.20), lw=1.1,
                            rounding=0.02, zorder=4))
            b.add(3, ax.text(X_CHAIN - 1.5, y, note, fontsize=f_small, fontweight="bold", color=ink["primary"],
                             va="center", ha="right", zorder=5))
        elif note:
            b.add(2, ax.text(X_CHAIN - 1.5, y, note, fontsize=f_small, color=ink["secondary"], va="center",
                             ha="right", zorder=5))
        # s3: the measured share of this inequality
        s = D["shares"][name]
        w = 100.0 * s * BAR_UNIT
        b.add(3, ax.add_patch(Rectangle((X_BAR, y - 3.0), w, 6.0, fc=col, ec="none", zorder=3)))
        b.add(3, ax.text(X_BAR + w + 1.0, y, percent(s), fontsize=f_text, color=ink["primary"], va="center",
                         ha="left", zorder=5))
    b.add(3, es.label(ax, 8.0, 22.6, "shares (right): undone one at a time, bottom-up",
                      fontsize=f_small, ha="left"))
    b.add(3, ax.text(X_BAR, 15.2, rf"I2 + I4 carry {100 * D['i24']:.0f} %", fontsize=f_text, fontweight="bold",
                     color=ink["primary"], va="center", ha="left"))
    b.add(3, ax.text(X_BAR, 6.0, "of the logarithmic gap\n" rf"({100 * D['i24_other']:.0f} % in the other order)",
                     fontsize=f_small, color=ink["secondary"], va="center", ha="left", linespacing=1.2))
    es.save(fig, NAME, "talk", build=b,
            note=(r"ridge logistic regression, $n = 1000$, $d = 50$, spectrum $j^{-1}$; passes 2–20, 5 sampling seeds, "
                  "one data set"))


def main():
    D = data()
    print_all(D)
    draw(D)


if __name__ == "__main__":
    main()
