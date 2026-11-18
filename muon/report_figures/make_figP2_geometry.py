"""Figure P2 of the paper muon/paper_saga_to_muon.tex: the geometry of Muon-type steps (finding 1, Muon side).

  (a) certified and local constants   M7: for the six instances of the benchmark, the per-sample condition number
                                      R_P^2 / mu_P of SAGA in three fixed metrics: the identity (Euclidean), the
                                      one-shot polar metric read off grad F(W_0), and the feature-covariance metric.
                                      For the last two, the best value over the floors of the gains. Filled markers:
                                      global constants (the ones a worst-case proof certifies); open markers: local
                                      constants at the optimum (gauge subspace for softmax). For least squares the two
                                      coincide (constant Hessian).
  (b) tuned steps                     M1, the two non-square conditioning-limited instances: passes to the relative
                                      squared distance 1e-6, median and range over ten confirmation seeds (panel (c) of
                                      figR2_muon_vr, drawn by the same function).

The script only reads muon/results/M7_summary.json, M1_summary.json (and M3, M6 for the digest check of panel (b)).
It prints every plotted value of (a) and compares it with the tables 'Conditioning in the frozen polar metric' of
muon/results/digest.md; it stops before drawing if a value disagrees.

Run:    .venv/Scripts/python muon/report_figures/make_figP2_geometry.py
Output: muon/figures/figP2_geometry.png and .pdf
"""
from __future__ import annotations

import importlib.util
import math
import os
import re
import sys

import numpy as np

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(ROOT, "muon", "results")
DIGEST = os.path.join(RESULTS, "digest.md")
NAME = "figP2_geometry"


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R2 = _load_module(os.path.join(HERE, "make_figR2_muon_vr.py"), "make_figR2_muon_vr")

INSTANCES = (("soft-mem", "softmax\nmemory"), ("soft-cond", "softmax\ncond."), ("soft-sq", "softmax\nsquare"),
             ("ls-mem", "l. sq.\nmemory"), ("ls-cond", "l. sq.\ncond."), ("ls-orth", "l. sq.\nsquare"))
ANCHORS = (("euclid", "identity", "SECONDARY"),
           ("grad F(W_0)", "one-shot polar metric", "ORANGE"),
           ("feature covariance, (c_l Sigma_hat + lam I)^-1", "covariance metric", "BLUE"))
OFFSETS = (-0.22, 0.0, 0.22)


# ----------------------------------------------------------------------------------------------------------------------
# data of (a)
# ----------------------------------------------------------------------------------------------------------------------
def data_a(m7):
    out = []
    for key, label in INSTANCES:
        inst = m7["metric"][key]
        cells = {}
        cells["euclid"] = dict(glob=float(inst["kappa_euclid_global"]), loc=float(inst["kappa_euclid"]),
                               floor_glob=None, floor_loc=None)
        for anchor, _, _ in ANCHORS[1:]:
            rows = [r for r in inst["rows"] if r["anchor"] == anchor]
            if not rows:
                raise SystemExit(f"{key}: no rows for anchor {anchor!r}")
            g = min(rows, key=lambda r: r["kappa_P_global"])
            l = min(rows, key=lambda r: r["kappa_P"])
            cells[anchor] = dict(glob=float(g["kappa_P_global"]), loc=float(l["kappa_P"]),
                                 floor_glob=float(g["floor"]), floor_loc=float(l["floor"]), n_rows=len(rows))
        out.append(dict(key=key, label=label, loss=inst["loss"], cells=cells))
    return out


def print_a(A):
    print("(a) per-sample condition number R_P^2/mu_P: global (certified) / local (at the optimum); for the two fixed"
          " metrics the best value over the floors of the gains, with the floor that attains it")
    for row in A:
        parts = []
        for anchor, lab, _ in ANCHORS:
            c = row["cells"][anchor]
            fl = "" if c["floor_glob"] is None else f" (f={c['floor_glob']:g} / f={c['floor_loc']:g})"
            parts.append(f"{lab}: {c['glob']:.0f} / {c['loc']:.0f}{fl}")
        print(f"    {row['key']:9s} " + "; ".join(parts))
    soft = [r for r in A if r["loss"] == "softmax"]
    ls = [r for r in A if r["loss"] == "ls"]
    worst = min(min(r["cells"][a]["glob"] / r["cells"]["euclid"]["glob"] for a, _, _ in ANCHORS[1:]) for r in soft)
    print(f"    softmax: smallest ratio (best certified constant of a fixed metric) / (Euclidean certified constant)"
          f" = {worst:.4f} (Theorem 1 predicts >= 1)")
    for r in soft:
        best_loc = min(r["cells"][a]["loc"] for a, _, _ in ANCHORS[1:])
        print(f"    {r['key']}: best local constant {best_loc:.0f} against Euclidean global "
              f"{r['cells']['euclid']['glob']:.0f} (factor {r['cells']['euclid']['glob'] / best_loc:.1f})")
    for r in ls:
        best = min(r["cells"][a]["glob"] for a, _, _ in ANCHORS[1:])
        print(f"    {r['key']}: best certified constant of a fixed metric {best:.0f} against Euclidean "
              f"{r['cells']['euclid']['glob']:.0f} (factor {r['cells']['euclid']['glob'] / best:.1f})")
    return worst


def digest_tables():
    """Tables 'Conditioning in the frozen polar metric': instance -> {'euclid': (local, global), (anchor, kind):
    [values by floor]} with kind 'local', 'global' or 'local = global'."""
    text = open(DIGEST, encoding="utf-8").read()
    start = text.index("### Conditioning in the frozen polar metric")
    end = text.index("### Weight of the y step in loopless Katyusha")
    sec = text[start:end]
    out = {}
    for block in re.split(r"\n\*\*", sec)[1:]:
        key = block.split(".**", 1)[0].strip()
        m = re.search(r"R2 / mu_true = ([0-9.]+)(?: \(global: R2 / lam = ([0-9.]+)\))?", block)
        tab = {"euclid": (float(m.group(1)), float(m.group(2)) if m.group(2) else float(m.group(1)))}
        anchor = None
        for line in block.splitlines():
            if not line.startswith("| ") or line.startswith("| anchor") or line.startswith("|---"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cells[0]:
                anchor = cells[0]
            kind = cells[1]
            vals = [float(v) for v in cells[2:10]]
            tab[(anchor, kind)] = vals
        out[key] = tab
    return out


def check_a(A):
    tabs = digest_tables()
    res = []
    for row in A:
        tab = tabs[row["key"]]
        e = row["cells"]["euclid"]
        res.append((f"{row['key']} Euclidean local", round(e["loc"]), round(tab["euclid"][0])))
        res.append((f"{row['key']} Euclidean global", round(e["glob"]), round(tab["euclid"][1])))
        for anchor, lab, _ in ANCHORS[1:]:
            c = row["cells"][anchor]
            if row["loss"] == "ls":
                vals = tab[(anchor, "local = global")]
                res.append((f"{row['key']} {lab} best (local = global)", round(c["glob"]), round(min(vals))))
                res.append((f"{row['key']} {lab} best local", round(c["loc"]), round(min(vals))))
            else:
                res.append((f"{row['key']} {lab} best global", round(c["glob"]), round(min(tab[(anchor, 'global')]))))
                res.append((f"{row['key']} {lab} best local", round(c["loc"]), round(min(tab[(anchor, 'local')]))))
    bad = [r for r in res if r[1] != r[2]]
    print(f"digest check of (a) (tables 'Conditioning in the frozen polar metric'): {len(res)} values compared, "
          f"{len(bad)} disagreement(s)")
    for what, mine, theirs in bad:
        print(f"    DISAGREEMENT {what}: JSON {mine}, digest {theirs}")
    return not bad


# ----------------------------------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------------------------------
def panel_constants(ax, A, ps):
    INK = ps.INK
    colour = {"SECONDARY": INK["secondary"], "ORANGE": ps.ORANGE, "BLUE": ps.BLUE}
    drawn = []
    for i, row in enumerate(A):
        for (anchor, _, col), dx in zip(ANCHORS, OFFSETS):
            c = row["cells"][anchor]
            x = i + dx
            col_ = colour[col]
            same = abs(c["glob"] - c["loc"]) <= 1e-9 * c["glob"]     # least squares: constant Hessian
            if not same:
                ax.plot([x, x], [c["loc"], c["glob"]], color=col_, lw=0.9, zorder=2)
                drawn.append(([x, x], [c["loc"], c["glob"]]))
            ax.plot([x], [c["glob"]], "o", ms=5.0, color=col_, mec=INK["surface"], mew=0.8, zorder=3)
            drawn.append(([x], [c["glob"]]))
            if not same:
                ax.plot([x], [c["loc"]], "o", ms=4.6, mfc=INK["surface"], mec=col_, mew=1.3, zorder=4)
                drawn.append(([x], [c["loc"]]))
    n_soft = sum(1 for r in A if r["loss"] == "softmax")
    ax.axvline(n_soft - 0.5, color=INK["grid"], lw=0.8, zorder=1)
    ax.set_yscale("log")
    ax.set_ylim(15, 3e5)
    ax.set_xlim(-0.6, len(A) - 0.4)
    ax.set_xticks(range(len(A)))
    ax.set_xticklabels([r["label"] for r in A], fontsize=ps.SMALL, linespacing=0.95)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("per-sample condition number $R_P^2/\\mu_P$")
    title = ax.set_title("(a) certified and local constants", loc="left")
    plt = ps.plt
    handles = [plt.Line2D([], [], color=colour[c], lw=3.0, solid_capstyle="butt") for _, _, c in ANCHORS]
    labels = [lab for _, lab, _ in ANCHORS]
    ink = INK["secondary"]
    handles += [plt.Line2D([], [], marker="o", ls="", ms=5.0, color=ink, mec=INK["surface"], mew=0.8),
                plt.Line2D([], [], marker="o", ls="", ms=4.6, mfc=INK["surface"], mec=ink, mew=1.3)]
    labels += ["global (certified)", "local (at the optimum)"]
    leg = ps.legend(ax, handles=handles, labels=labels, loc="upper right", ncol=2, handlelength=0.9,
                    handletextpad=0.4, labelspacing=0.2, columnspacing=0.8, borderpad=0.1, borderaxespad=0.2)
    return title, leg, drawn


def layout_checks(fig, panels):
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    pt = fig.dpi / 72.0
    msgs = []
    for name, ax, title, leg, lines in panels:
        lb, ab = leg.get_window_extent(rend), ax.bbox
        hit = R2._inside(ax, lb, lines, pad=2.0 * pt)
        if hit:
            msgs.append(f"({name}) the legend is on the data ({hit} sampled points)")
        if lb.x0 < ab.x0 - 1 or lb.x1 > ab.x1 + 1 or lb.y0 < ab.y0 - 1 or lb.y1 > ab.y1 + 1:
            msgs.append(f"({name}) the legend leaves the axes")
    t0, t1 = panels[0][2].get_window_extent(rend), panels[1][2].get_window_extent(rend)
    if t0.x1 > t1.x0 - 12 * pt:
        msgs.append("the titles of (a) and (b) are too close")
    print("layout checks (legends inside the axes and away from the data, titles apart): "
          + ("ok" if not msgs else "; ".join(msgs)))
    return not msgs


def main():
    m7 = R2.load("M7_summary")
    A = data_a(m7)
    worst = print_a(A)
    if worst < 1.0 - 1e-9:
        raise SystemExit("a fixed metric beats the Euclidean certified constant on softmax: check the data")
    Aok = check_a(A)
    Ra, Rb, Rc = R2.data_a(R2.load("M3_summary")), R2.data_b(R2.load("M6_summary")), R2.data_c(R2.load("M1_summary"))
    R2.print_c(Rc)
    Rok = R2.report_digest(R2.check_digest(Ra, Rb, Rc))
    if not (Aok and Rok):
        raise SystemExit("the figure is not drawn: the numbers above disagree with the digest")
    ps = R2._plotstyle()
    fig = ps.figure(width=ps.FIG_W, height=2.45)
    fig.get_layout_engine().set(w_pad=0.03, wspace=0.0)
    gs = fig.add_gridspec(1, 2, width_ratios=(1.55, 1.0))
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    pa = ("a", ax_a) + panel_constants(ax_a, A, ps)
    tb, lb, drawn_b = R2.panel_c(ax_b, Rc, ps)
    tb.set_text("(b) tuned steps")
    layout_checks(fig, [pa, ("b", ax_b, tb, lb, drawn_b)])
    ps.savefig(fig, NAME)


if __name__ == "__main__":
    main()
