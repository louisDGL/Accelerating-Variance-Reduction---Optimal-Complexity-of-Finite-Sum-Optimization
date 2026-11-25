"""Figure E1 of the appendix of explanatory figures of the paper (paper/figures/figE1_proof_chain): the proof of
Proposition 5.9 (linear rate of SAGA) as an exact one-step identity followed by five substitutions of worst-case
values, and which of them carry the gap between the exact and the certified contraction exponents. Explanatory figure;
it adds no result to the paper: it lays out Section 2.2 and Eq. (1) of paper/paper_saga_to_muon.tex and recuts the
numbers behind Figure 1(c) and finding (C2).

Panels:
  (a) DIAGRAM (no data). Top: the exact identity (1) of the paper, E_t[V_{t+1}] = c_D D + c_M delta M, with
      c_D = 1 - mu_eff/(2 R^2) (1 - L_eff/R^2 (r_sq/4 + 3/8)) and c_M = 1 - (1/n)(1 - (2/3) r_sq r_var)
      (Proposition "exact contraction in terms of the ratios" of theory/B3_prop59_tightness.tex, textbook constants
      gamma = 1/(4 R^2), delta = 3 gamma^2). Then one row per inequality, in the order in which the attribution
      applies them (I1, I3, I2, I4, I5: the reverse of the order in which the paper undoes them; the book states I2
      before I3, and the two act on different terms): I1 square norm, r_sq -> 1; I3 variance <= second moment,
      r_var -> 1; I2 cocoercivity, L_eff -> R^2; I4 strong convexity, mu_eff -> mu; I5 minimum of two rates, the
      mean of c_D and c_M weighted by D and delta M -> their maximum. Bottom: the certified contraction
      1 - min{1/(3n), 3 mu/(16 R^2)}, and, detached, I6 (initialization, V_0 <= (1 + 3n/16) D_0), which changes the
      constant and not the rate. The rows I2 and I4, which put a global constant in the place of a quantity of the
      iterate, are tinted. The colour key of each row is the colour of its segment in (b) (plotstyle.INEQ).
  (b) DATA. Shares of the logarithmic gap, (c_{k-1} - c_k) / (c_exact - c_bound) with c = -log(rho) the window means
      of the contraction exponents over passes 2-20 (mean over five sampling seeds, one data set), for the spectrum
      j^(-1) (alpha = 1) at kappa = 1e2 and kappa = 1e5, in the two orders of attribution that were computed:
        "I5 first": the order of the paper, undo I5, I4, I2, I3, I1 from the bound (order B of the theory note;
                    key "shares");
        "I5 last":  undo I4, I2, I3, I1, then I5 (order A of the note: the maximum of the two coefficients is kept
                    until the end; key "shares_orderA", vrlab.analysis.STEP_KEYS_A = I4, I2, I3, I1, I5).
      Segments in the fixed order I2, I4, I1, I3, I5, so that the length of the first two is I2 + I4, printed at the
      right of each bar. Group labels: the size of the gap, exact over certified exponent (key "speedup").

Source (read only): results/B3_summary.json, attribution["kappa=<k>,alpha=<a>"] (shares, shares_orderA, speedup,
speedup_late_20_100). Checks, the script stops if one fails: the shares of each order sum to one; speed-up and the
ten shares of the plotted configurations agree with results/digest.md (tables "Attribution of the log-gap" and
"Attribution in order A") at the printed precision; the ranges of the paper and of its audit are reproduced from the
15 configurations: I2 + I4 = 77-96 % for kappa >= 1e4 in the order of the paper and 59-96 % in the other order;
speed-up 3.2-3.7 for kappa <= 1e2 and 19-220 at kappa = 1e5 (passes 2-20), 11-187 at kappa = 1e5 on passes 20-100.

Status. Proved: the identity and the chain of panel (a) (main report). Measured exactly along runs (every conditional
expectation by enumeration of the n indices): the shares of panel (b). A share is a fraction of a difference of
logarithms of exponents and depends on the order of attribution and on the window.

The script prints every number it plots.
Run:    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/scripts/make_figE1_proof_chain.py
Output: paper/figures/figE1_proof_chain.pdf and .png
"""
from __future__ import annotations

import json
import os
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import explain_style as es  # noqa: E402

NAME = "figE1_proof_chain"
KAPPAS = [10.0, 100.0, 1000.0, 10000.0, 100000.0]
ALPHAS = [0.0, 1.0, 2.0]
ALPHA_B, KAPPAS_B = 1.0, (100.0, 100000.0)        # configurations of panel (b)
CHAIN = ["I1", "I3", "I2", "I4", "I5"]            # rows of (a): order in which the attribution applies them
STACK = ["I2", "I4", "I1", "I3", "I5"]            # segments of (b), left to right
ORDERS = (("shares", "I5 undone\nfirst"), ("shares_orderA", "I5 undone\nlast"))
UNDO = {"shares": "I5, I4, I2, I3, I1", "shares_orderA": "I4, I2, I3, I1, I5"}
DIGEST_B = "### Attribution of the log-gap (passes 2-20) and window-averaged ratios"
DIGEST_A = "### Attribution in order A of the B3 note (undo I4, I2, I3, I1, then the min I5; passes 2-20)"
PAPER = {"I2+I4, kappa >= 1e4, order of the paper (%)": ("77", "96"),      # (C2) of the paper
         "I2+I4, kappa >= 1e4, other order (%)": ("59", "96"),             # audit, table of section 3
         "speed-up, kappa <= 1e2": ("3.2", "3.7"), "speed-up, kappa = 1e5": ("19", "220"),      # (C1)
         "speed-up, kappa = 1e5, passes 20-100": ("11", "187")}            # audit


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
            for name, _ in ORDERS:
                s = sum(att[key(k, a)][name][i] for i in CHAIN)
                if abs(s - 1.0) > 1e-9:
                    bad.append(f"{key(k, a)}: the shares of {name} sum to {s!r}")
    # the plotted configurations against the digest
    dig_b, dig_a = digest_table(DIGEST_B), digest_table(DIGEST_A)
    col_b = {"I1": "share I1", "I2": "I2", "I3": "I3", "I4": "I4", "I5": "I5"}
    col_a = {"I4": "share I4", "I2": "I2", "I3": "I3", "I1": "I1", "I5": "I5 (min, undone last)"}
    count = 0
    for k in KAPPAS_B:
        c = key(k, ALPHA_B)
        pairs = [("speed-up", att[c]["speedup"], dig_b[c]["speed-up exact/bound"])]
        pairs += [(f"share {i}, order of the paper", att[c]["shares"][i], dig_b[c][col_b[i]]) for i in CHAIN]
        pairs += [(f"share {i}, other order", att[c]["shares_orderA"][i], dig_a[c][col_a[i]]) for i in CHAIN]
        count += len(pairs)
        bad += [f"{c} {n}: summary {v!r} -> {digest_fmt(v)}, digest {cell}" for n, v, cell in pairs
                if digest_fmt(v) != cell]
    # the ranges quoted by the paper and by its audit, from the 15 configurations
    def i24(c, name):
        return att[c][name]["I2"] + att[c][name]["I4"]
    high = [key(k, a) for k in KAPPAS if k >= 1e4 for a in ALPHAS]
    sp_low = [att[key(k, a)]["speedup"] for k in KAPPAS if k <= 1e2 for a in ALPHAS]
    sp_5 = [att[key(1e5, a)]["speedup"] for a in ALPHAS]
    late_5 = [att[key(1e5, a)]["speedup_late_20_100"] for a in ALPHAS]
    b_vals, a_vals = [100 * i24(c, "shares") for c in high], [100 * i24(c, "shares_orderA") for c in high]
    ranges = {"I2+I4, kappa >= 1e4, order of the paper (%)": (f"{min(b_vals):.0f}", f"{max(b_vals):.0f}"),
              "I2+I4, kappa >= 1e4, other order (%)": (f"{min(a_vals):.0f}", f"{max(a_vals):.0f}"),
              "speed-up, kappa <= 1e2": (f"{min(sp_low):.1f}", f"{max(sp_low):.1f}"),
              "speed-up, kappa = 1e5": (f"{min(sp_5):.0f}", f"{max(sp_5):.0f}"),
              "speed-up, kappa = 1e5, passes 20-100": (f"{min(late_5):.0f}", f"{max(late_5):.0f}")}
    bad += [f"{n}: recomputed {ranges[n][0]}-{ranges[n][1]}, quoted {PAPER[n][0]}-{PAPER[n][1]}" for n in PAPER
            if ranges[n] != PAPER[n]]
    if bad:
        raise SystemExit("CHECK FAILED, no figure drawn:\n   " + "\n   ".join(bad))
    rows = []
    for k in KAPPAS_B:
        c = key(k, ALPHA_B)
        for name, label in ORDERS:
            rows.append({"kappa": k, "order": name, "label": label, "shares": {i: att[c][name][i] for i in CHAIN},
                         "i24": i24(c, name), "speedup": att[c]["speedup"], "late": att[c]["speedup_late_20_100"]})
    return {"rows": rows, "ranges": ranges, "digest_count": count}


def print_all(D):
    print(f"(b) alpha = {ALPHA_B:g}, n = 1000, d = 50, passes 2-20, mean over 5 sampling seeds (results/B3_summary.json)")
    for r in D["rows"]:
        print(f"    kappa = {r['kappa']:<7g} {r['label'].replace(chr(10), ' '):15s} (undo {UNDO[r['order']]}): "
              + ", ".join(f"{i} {r['shares'][i]:.4f}" for i in STACK)
              + f"; I2 + I4 = {r['i24']:.4f}; exact / certified exponent {r['speedup']:.4g} "
              f"(passes 20-100: {r['late']:.4g})")
    print(f"checks: {D['digest_count']} numbers agree with results/digest.md at its printed precision; ranges over the "
          "15 configurations: " + "; ".join(f"{n}: {lo}-{hi}" for n, (lo, hi) in D["ranges"].items())
          + " (as quoted by the paper and its audit)")


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
GLOBAL = ("I2", "I4")


def panel_chain(ax, ps):
    from matplotlib.patches import Rectangle
    ink = ps.INK
    es.diagram_axes(ax, (-0.6, 100.6), (-1.2, 100.8))
    f_text, f_math, f_small = 8.0, 8.5, es.small()
    # the exact identity
    es.box(ax, 50.0, 88.5, 100.0, 23.0, edge=ink["secondary"], lw=1.0, rounding=0.025)
    ax.text(3.0, 94.3, "exact identity", fontsize=f_text, fontweight="bold", color=ink["primary"], va="center")
    ax.text(31.0, 94.3, r"$\mathbb{E}_t[V_{t+1}] \;=\; c_D\,D \;+\; c_M\,\delta M$", fontsize=f_math + 0.5,
            color=ink["primary"], va="center")
    ax.text(97.5, 94.3, r"$V = D + \delta M$", fontsize=f_math, color=ink["secondary"], va="center", ha="right")
    ax.text(3.0, 83.0, r"$c_D = 1-\frac{\mu_{\mathrm{eff}}}{2R^2}\left(1-\frac{L_{\mathrm{eff}}}{R^2}\left("
            r"\frac{r_{\mathrm{sq}}}{4}+\frac{3}{8}\right)\right)$,   $c_M = 1-\frac{1}{n}\left(1-\frac{2}{3}\,"
            r"r_{\mathrm{sq}}\,r_{\mathrm{var}}\right)$", fontsize=f_math + 0.5, color=ink["primary"], va="center")
    # the five substitutions
    top, pitch, gap = 70.3, 9.06, 0.9                # top lowered, pitch reduced: same bottom, room for the label
    es.arrow(ax, (8.0, 77.0), (8.0, top + 0.2), lw=1.1, mutation_scale=8.0)
    es.label(ax, 11.5, 73.8, "the proof: one worst-case value per inequality", fontsize=f_small, ha="left")
    for k, (name, title, subst, note) in enumerate(ROWS):
        y = top - pitch * (k + 0.5)
        col = ps.INEQ[name]
        es.box(ax, 50.0, y, 100.0, pitch - gap, colour=col, fill=es.tint(col, 0.20) if name in GLOBAL else ink["surface"],
               lw=1.0, rounding=0.018)
        ax.add_patch(Rectangle((2.0, y - 2.4), 3.1, 4.8, fc=col, ec="none", zorder=5))
        ax.text(7.0, y, name, fontsize=f_text, fontweight="bold", color=ink["primary"], va="center", zorder=5)
        ax.text(12.5, y, title, fontsize=f_text, color=ink["primary"], va="center", zorder=5)
        ax.text(55.0, y, subst, fontsize=f_math, color=ink["primary"], va="center", zorder=5)
        if note:
            ax.text(98.0, y, note, fontsize=f_small, color=ink["primary" if name in GLOBAL else "secondary"],
                    fontweight="bold" if name in GLOBAL else "normal", va="center", ha="right", zorder=5)
    bottom = top - pitch * len(ROWS)
    es.arrow(ax, (8.0, bottom - 0.2), (8.0, bottom - 5.6), lw=1.1, mutation_scale=8.0)
    # the certificate, and I6 apart
    es.box(ax, 30.5, 9.5, 61.0, 19.0, edge=ink["secondary"], lw=1.0, rounding=0.025)
    ax.text(3.0, 14.5, "certified (Proposition 5.9)", fontsize=f_text, fontweight="bold", color=ink["primary"],
            va="center")
    ax.text(3.0, 5.6, r"$\mathbb{E}_t[V_{t+1}] \;\leq\; \left(1-\min\left\{\frac{1}{3n},\,\frac{3\mu}{16R^2}\right\}"
            r"\right)V$", fontsize=f_math + 0.5, color=ink["primary"], va="center")
    col = ps.INEQ["I6"]
    es.box(ax, 82.0, 9.5, 36.0, 19.0, colour=col, fill=ink["surface"], lw=1.0, rounding=0.025)
    ax.add_patch(Rectangle((66.3, 12.1), 3.1, 4.8, fc=col, ec="none", zorder=5))
    ax.text(71.0, 14.5, "I6", fontsize=f_text, fontweight="bold", color=ink["primary"], va="center", zorder=5)
    ax.text(75.9, 14.5, "initialization", fontsize=f_text, color=ink["primary"], va="center", zorder=5)
    ax.text(66.3, 6.3, "changes the constant,\nnot the rate", fontsize=f_small, color=ink["secondary"], va="center",
            linespacing=1.15, zorder=5)
    es.panel_title(ax, "(a) the proof as substitutions")


HEAD_X = -0.47                                       # left end of the group labels of (b), in units of the x axis


def panel_shares(ax, D, ps):
    ink = ps.INK
    rows = D["rows"]
    ys = [4.55, 3.75, 1.95, 1.15]                    # two bars per kappa
    height = 0.66
    px = 1.0 / 150.0                                 # about 2 px of surface between segments (axes about 1.5 in wide)
    for y, r in zip(ys, rows):
        left = 0.0
        for i in STACK:
            w = r["shares"][i]
            if w > 2 * px:
                ax.barh(y, w - px, left=left + px / 2, height=height, color=ps.INEQ[i], lw=0, zorder=3)
            elif w > 0:
                ax.barh(y, w, left=left, height=height, color=ps.INEQ[i], lw=0, zorder=3)
            if w >= 0.12:
                ax.text(left + w / 2, y, i, ha="center", va="center", fontsize=es.small(), color=ink["primary"],
                        zorder=4)
            left += w
        ax.text(1.035, y, f"{100 * r['i24']:.0f} %", ha="left", va="center", fontsize=es.small() + 0.5,
                color=ink["primary"], clip_on=False)
    ax.text(1.035, ys[0] + 0.50, "I2 + I4", ha="left", va="bottom", fontsize=es.small(), color=ink["secondary"],
            clip_on=False)
    for j, k in enumerate(KAPPAS_B):
        r = rows[2 * j]
        ax.text(HEAD_X, ys[2 * j] + 0.50, rf"$\kappa = 10^{{{int(round(np.log10(k)))}}}$: exact exponent "
                rf"= {r['speedup']:.{1 if r['speedup'] < 10 else 0}f} $\times$ certified", ha="left", va="bottom", fontsize=es.small() + 0.5,
                color=ink["primary"], clip_on=False)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=es.small(), linespacing=1.05)
    ax.tick_params(axis="y", length=0, pad=3.0)
    ax.set_ylim(0.6, 5.55)
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(["0", "50 %", "100 %"])
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("share of the logarithmic gap\n(passes 2–20, spectrum $j^{-1}$)")
    es.panel_title(ax, "(b) who carries the gap")


def draw(D):
    ps = es.setup("paper")
    fig = es.figure("paper", height=2.85)
    fig.get_layout_engine().set(w_pad=0.03, wspace=0.03)
    gs = fig.add_gridspec(1, 2, width_ratios=(2.15, 1.0))
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    panel_chain(ax_a, ps)
    panel_shares(ax_b, D, ps)
    es.save(fig, NAME, "paper")


def main():
    D = data()
    print_all(D)
    draw(D)


if __name__ == "__main__":
    main()
