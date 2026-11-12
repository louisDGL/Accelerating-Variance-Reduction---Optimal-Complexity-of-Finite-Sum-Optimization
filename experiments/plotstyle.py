"""Matplotlib style for all figures: validated categorical palette (fixed slot order, colors follow entities
across figures), thin 2px lines, hairline solid grid, recessive axes, legends always present for >= 2 series.
Sizes target the report, which includes every figure at its text width (6.5 in): figures are drawn about FIG_W wide
(scale about 0.95 once included) and the smallest text is SMALL = 7.5 pt (legends, annotations), so every text element
prints at about 7 pt or more (mathtext sub- and superscripts, e.g. the exponents of log-axis tick labels, are 70 % of
their text); savefig prints the printed size of the smallest text of each figure as a check."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import FIGURES  # noqa: E402

TEXTWIDTH_IN = 6.5      # text width of the report (letter paper, 1 in margins); figures are included at this width
FIG_W = 6.7             # figsize width of a full-width figure (the saved, tight PDF is about 6.6-6.9 in wide)
SMALL = 7.5             # smallest font size: legends, in-panel annotations, colorbar labels
SUPTITLE = 9.5          # figure title

# categorical slots (validated order; light mode)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = SERIES
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
             "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
INK = dict(primary="#0b0b0b", secondary="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
           surface="#fcfcfb", page="#f9f9f7")

# entity -> color, fixed for the whole project
ALGO = {
    "SAGA": BLUE, "L-SVRG": ORANGE, "L-Katyusha": AQUA, "AGD": YELLOW, "GD": MAGENTA,
    "Nesterov+SVRG": GREEN, "L-Katyusha (no neg. momentum)": VIOLET, "SGD": RED,
}
INEQ = {"I1": BLUE, "I3": AQUA, "I2": ORANGE, "I4": YELLOW, "I5": MAGENTA, "I6": GREEN}
INEQ_LABEL = {"I1": "I1 square norm", "I2": "I2 cocoercivity", "I3": "I3 variance ≤ 2nd moment",
              "I4": "I4 strong convexity", "I5": "I5 min of rates", "I6": "I6 initialization"}
RATIO = {"r_sq": BLUE, "r_coco": ORANGE, "r_var": AQUA, "r_sc": YELLOW, "r_min": MAGENTA}
RATIO_LABEL = {"r_sq": "r_sq (I1 square norm)", "r_coco": "r_coco = L_eff/R² (I2)", "r_var": "r_var (I3)",
               "r_sc": "r_sc = μ/μ_eff (I4)", "r_min": "r_min (I5)"}
BOUNDS = {"empirical": BLUE, "prop59": RED, "before_I6": ORANGE, "mu_eff": AQUA, "mu_eff+L_eff": YELLOW,
          "exact": VIOLET, "realized_V": INK["muted"]}
BOUNDS_LABEL = {"empirical": "empirical E‖θₜ−η*‖² (mean over seeds)", "prop59": "Prop. 5.9 bound",
                "before_I6": "bound before step I6", "mu_eff": "trajectory bound (μ_eff)",
                "mu_eff+L_eff": "trajectory bound (μ_eff, L_eff)", "exact": "product of exact contractions",
                "realized_V": "realized Vₜ (seed 0)"}
PRECOND = {"I": BLUE, "diag": ORANGE, "stat": AQUA, "hess": YELLOW}
PRECOND_LABEL = {"I": "P = I (SAGA)", "diag": "P = diag(Σ̂) + λI", "stat": "P = Σ̂ + λI", "hess": "P = F''(η*)"}


def setup():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial"],
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "legend.fontsize": SMALL,
        "legend.handlelength": 1.6,
        "legend.columnspacing": 1.2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.titlesize": SUPTITLE,
        "figure.labelsize": 8.5,
        "lines.linewidth": 2.0,
        "lines.solid_joinstyle": "round",
        "lines.solid_capstyle": "round",
        "axes.grid": True,
        "grid.color": INK["grid"],
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "axes.edgecolor": INK["axis"],
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": INK["surface"],
        "figure.facecolor": INK["surface"],
        "savefig.facecolor": INK["surface"],
        "xtick.color": INK["muted"],
        "ytick.color": INK["muted"],
        "axes.labelcolor": INK["secondary"],
        "axes.titlecolor": INK["primary"],
        "text.color": INK["primary"],
        "legend.frameon": False,
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.titleweight": "bold",
    })


def new_fig(ncols=1, nrows=1, panel=(4.2, 3.2), sharex=False, sharey=False):
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel[0] * ncols, panel[1] * nrows), sharex=sharex, sharey=sharey,
                             squeeze=False, constrained_layout=True)
    return fig, axes


def figure(width=FIG_W, height=4.0):
    """Empty figure with constrained layout, for grids built with fig.add_gridspec / fig.subfigures."""
    return plt.figure(figsize=(width, height), constrained_layout=True)


def suptitle(fig, text):
    return fig.suptitle(text, fontsize=SUPTITLE)


def legend_below(fig, handles, labels, ncol=1, **kw):
    """One legend shared by all panels, below them (constrained layout keeps it clear of the axes labels)."""
    leg = fig.legend(handles, labels, loc="outside lower center", ncol=ncol, frameon=False, **kw)
    for t in leg.get_texts():
        t.set_color(INK["secondary"])
    return leg


def legend_cell(fig, spec, handles, labels, ncol=1, **kw):
    """Legend in its own cell of the grid (an axes without frame, e.g. a thin row below a row of panels): the legend
    hangs from the top of the cell and constrained layout reserves its height below the panels' axis labels."""
    ax = fig.add_subplot(spec)
    ax.axis("off")
    leg = ax.legend(handles, labels, loc="upper center", ncol=ncol, borderaxespad=0.0, **kw)
    for t in leg.get_texts():
        t.set_color(INK["secondary"])
    return ax, leg


def print_size(fig):
    """Size of the saved (tight) figure and of its smallest text once included at the report's text width."""
    from matplotlib.text import Text
    renderer = fig.canvas.get_renderer()
    bb = fig.get_tightbbox(renderer)
    pad = 2 * plt.rcParams["savefig.pad_inches"]
    w, h = bb.width + pad, bb.height + pad
    sizes = [t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    k = TEXTWIDTH_IN / w
    print(f"   {w:.2f} x {h:.2f} in; included at {TEXTWIDTH_IN:g} in: scale {k:.2f}, height {h * k:.2f} in, "
          f"smallest text {min(sizes):g} pt -> {min(sizes) * k:.1f} pt", flush=True)


_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def savefig(fig, name, formats=("png", "pdf")):
    os.makedirs(FIGURES, exist_ok=True)
    import sys
    if "--quick" in sys.argv:          # never overwrite the full-run figures with smoke-test output
        name = name + "_quick"
    for ext in formats:
        fig.savefig(os.path.join(FIGURES, f"{name}.{ext}"))
    try:
        print_size(fig)
    except Exception as e:  # pragma: no cover - diagnostic only
        print("   (size check failed:", e, ")")
    plt.close(fig)
    print(f"saved figures/{name}.{{{','.join(formats)}}}", flush=True)


def end_label(ax, x, y, text, dx=4, color=None):
    """Direct label at the end of a line, in secondary ink (text never wears the series color)."""
    i = int(np.argmax(np.isfinite(y)[::-1]))  # last finite point
    j = len(y) - 1 - i
    ax.annotate(text, (x[j], y[j]), xytext=(dx, 0), textcoords="offset points", va="center", ha="left",
                fontsize=SMALL, color=color or INK["secondary"], annotation_clip=False)


def band(ax, x, lo, hi, color):
    ax.fill_between(x, lo, hi, color=color, alpha=0.10, linewidth=0)


def legend(ax, **kw):
    kw.setdefault("loc", "best")
    leg = ax.legend(**kw)
    for t in leg.get_texts():
        t.set_color(INK["secondary"])
    return leg


def fmt_kappa(k):
    e = int(round(np.log10(k)))
    return f"κ = 10{str(e).translate(_SUP)}" if abs(np.log10(k) - e) < 1e-9 else f"κ = {k:g}"
