"""Figures of the Muon study, built from the JSON summaries in muon/results (style of the main project:
experiments/plotstyle.py; palette validated with the data-visualization checks, see the comments below).

  figM1_ranking_<pair>   passes to the target for every family (median and range over the confirmation seeds), one
                         panel per instance; colour = family group, marker shape = status (published / lab / new)
  figM1_curves           relative squared distance against passes for six reference families on the six instances
  figM3_floors           traces of the floor experiment (length rule x estimator)
  figM4_sensitivity      separation of two runs started 1e-12 apart
  figM5_batch            passes to the target against the batch size, Euclidean versus orthogonalized SAGA
  figM2_mechanism        one-step expected contraction (own states, SAGA states) against the realized contraction
  figM6_miso             joint SAGA-Muon against MISO and dual-free SDCA (traces)
  figM8_dataseeds        passes to the target of the key families on three draws of the data

Usage: python muon/experiments/make_figures.py [--quick] [M1] [M2] [M3] [M4] [M5] [M6] [M8]
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

import common as C

sys.path.insert(0, os.path.join(C.ROOT, "experiments"))
import plotstyle as ps  # noqa: E402  (imports `common`, i.e. this directory's module: figures go to muon/figures)

plt = ps.plt
INK = ps.INK
# Categorical slots (fixed order, colour follows the entity). Validated with the skill's validate_palette.py on the
# surface #fcfcfb: the 8 slots pass the adjacent-pair checks (lines), and the three hues used where any two marks can be
# neighbours (dot plots, batch curves: blue, orange, aqua) pass the all-pairs checks; the fourth group of the dot plots
# is drawn in the neutral ink instead of a fourth hue. Identity is never carried by colour alone (row labels, legends,
# marker shapes).
GROUP_COLOR = {"euclid": ps.BLUE, "spectral": ps.ORANGE, "full": ps.AQUA, "stream": INK["muted"]}
GROUP_LABEL = {"euclid": "Euclidean variance reduction", "spectral": "orthogonalized variance reduction",
               "full": "full gradient", "stream": "minibatch, no finite-sum memory"}
STATUS_MARKER = {"sota": "o", "lab": "s", "new": "D"}
STATUS_LABEL = {"sota": "published method", "lab": "in the lab before this study", "new": "new configuration"}
INSTANCE_TITLE = {
    "soft-mem": "softmax, memory-limited", "ls-mem": "least squares, memory-limited",
    "soft-cond": "softmax, conditioning-limited", "ls-cond": "least squares, conditioning-limited",
    "ls-orth": "least squares, flat planted spectrum (K = d)", "soft-sq": "softmax, square (K = d)",
}
CURVES = [("SAGA", ps.BLUE), ("SAGA-Muon (joint)", ps.ORANGE), ("soft SAGA-Muon", ps.AQUA), ("B-SAGA", ps.YELLOW),
          ("L-Katyusha", ps.MAGENTA), ("PolarGrad", ps.GREEN)]


NICE = (1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 1000)
DIST_LABEL = r"$\|W - W_\star\|_F^2\ /\ \|W_\star\|_F^2$"


def _fin(v):
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


def _rank_key(row, tk):
    """Order of the digest: families whose seeds all reach the target first (by median), then partial ones (by the
    share of seeds), then the others (by final distance)."""
    s = row["summary"][tk]
    if s["reached"] == s["runs"]:
        return (0, s["median"], 0.0)
    if s["reached"] > 0:
        return (1, -s["reached"] / s["runs"], s["median"] if _fin(s["median"]) else float("inf"))
    return (2, row["final_rel"]["median"], 0.0)


def _nice_ticks(lo, hi, max_ticks=6):
    """Round tick values inside [lo, hi] for a log axis, thinned to at most max_ticks."""
    t = [v for v in NICE if lo <= v <= hi]
    while len(t) > max_ticks:
        t = t[::2] if (len(t) - 1) % 2 == 0 else t[::2] + [t[-1]]
        if len(t) > max_ticks and len(t) > 2:
            t = t[:-2] + [t[-1]]
    return t


# ----------------------------------------------------------------------------------------------
# M1
# ----------------------------------------------------------------------------------------------
def fig_m1_ranking(m1, keys, name):
    keys = [k for k in keys if k in m1]
    if not keys:
        return
    nrows = max(len(m1[k]["rows"]) for k in keys)
    fig = ps.figure(width=ps.FIG_W, height=0.19 * nrows + 1.3)
    gs = fig.add_gridspec(2, len(keys), height_ratios=[1.0, 0.09])
    for col, key in enumerate(keys):
        block = m1[key]
        tol = block["tol"]
        tk = f"{tol:g}"
        rows = sorted(block["rows"], key=lambda r: _rank_key(r, tk))
        ax = fig.add_subplot(gs[0, col])
        reached = [r for r in rows if _fin(r["summary"][tk]["median"])]
        xmax = block["passes"]
        xs = [r["summary"][tk]["min"] for r in reached] or [1.0]
        lo = max(1.0, 0.8 * min(xs))
        miss_x = xmax * 1.55
        for i, r in enumerate(rows):
            y = len(rows) - 1 - i
            col_ = GROUP_COLOR[r["group"]]
            mk = STATUS_MARKER[r["status"]]
            s = r["summary"][tk]
            if _fin(s["median"]):
                hi = s["max"] if _fin(s["max"]) else xmax
                ax.plot([s["min"], hi], [y, y], color=col_, lw=0.9, solid_capstyle="round", zorder=2)
                if _fin(s.get("q1")) and s["runs"] > 1:               # interquartile range of the seeds
                    q3 = s["q3"] if _fin(s["q3"]) else xmax
                    ax.plot([s["q1"], q3], [y, y], color=col_, lw=2.4, solid_capstyle="round", zorder=2)
                if not _fin(s["max"]):                                # some seeds missed the target
                    ax.plot([hi, miss_x], [y, y], color=col_, lw=0.8, ls=(0, (1, 2)), zorder=2)
                ax.plot(s["median"], y, mk, ms=6.0, color=col_, mec=INK["surface"], mew=1.2, zorder=3)
            else:
                ax.plot(miss_x, y, mk, ms=6.0, mfc=INK["surface"], mec=col_, mew=1.3, zorder=3)
                ax.annotate(f"{r['final_rel']['median']:.0e}", (miss_x, y), xytext=(8, 0),
                            textcoords="offset points", va="center", ha="left", fontsize=ps.SMALL - 1,
                            color=INK["muted"], annotation_clip=False)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r["family"] for r in rows][::-1], fontsize=ps.SMALL)
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.set_xscale("log")
        ax.set_xlim(lo, miss_x * 1.35)
        ax.axvspan(xmax, miss_x * 1.35, color=INK["grid"], alpha=0.45, lw=0, zorder=0)
        ticks = _nice_ticks(lo, xmax, 5)
        ax.set_xticks(ticks + [miss_x])
        ax.set_xticklabels([str(t) for t in ticks] + ["not\nreached"])
        ax.minorticks_off()
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(f"passes to reach relative squared distance {tol:g}")
        ax.set_title(INSTANCE_TITLE.get(key, key), loc="left")
        ax.tick_params(axis="y", length=0)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=6.5, color=GROUP_COLOR[g], mec=INK["surface"], mew=1.2)
               for g in ("euclid", "spectral", "full", "stream")]
    handles += [plt.Line2D([], [], marker=STATUS_MARKER[s_], ls="", ms=6, color=INK["secondary"], mec=INK["surface"],
                           mew=1.2) for s_ in ("sota", "lab", "new")]
    labels = [GROUP_LABEL[g] for g in ("euclid", "spectral", "full", "stream")] + \
             [STATUS_LABEL[s_] for s_ in ("sota", "lab", "new")]
    ps.legend_cell(fig, gs[1, :], handles, labels, ncol=4)
    ps.savefig(fig, name)


def fig_m1_curves(m1, name="figM1_curves"):
    keys = [k for k in INSTANCE_TITLE if k in m1]
    if not keys:
        return
    ncol = 3 if len(keys) > 2 else len(keys)
    nrow = int(math.ceil(len(keys) / ncol))
    fig = ps.figure(width=ps.FIG_W, height=2.25 * nrow + 0.55)
    gs = fig.add_gridspec(nrow + 1, ncol, height_ratios=[1.0] * nrow + [0.1])
    for q, key in enumerate(keys):
        ax = fig.add_subplot(gs[q // ncol, q % ncol])
        block = m1[key]
        rows = {r["family"]: r for r in block["rows"]}
        for fam, col in CURVES:
            r = rows.get(fam)
            if not r or not r.get("trace"):
                continue
            p, rel = np.asarray(r["trace"]["passes"]), np.asarray(r["trace"]["rel"])
            ok = np.isfinite(rel) & (rel > 0)
            ax.plot(p[ok], rel[ok], color=col, lw=1.6, label=fam)
        ax.set_yscale("log")
        ax.axhline(block["tol"], color=INK["axis"], lw=0.8, zorder=0)
        ax.set_title(INSTANCE_TITLE.get(key, key), loc="left", fontsize=ps.SMALL + 0.5)
        ax.set_xlim(0, block["passes"])
        lo = max(block["tol"] * 1e-4, 1e-16)
        ax.set_ylim(lo, 10.0)
        if q // ncol == nrow - 1:
            ax.set_xlabel("passes")
        if q % ncol == 0:
            ax.set_ylabel(DIST_LABEL)
    handles = [plt.Line2D([], [], color=c, lw=2) for _, c in CURVES]
    ps.legend_cell(fig, gs[nrow, :], handles, [f for f, _ in CURVES], ncol=6)
    ps.savefig(fig, name)


# ----------------------------------------------------------------------------------------------
# M3, M4, M5, M2
# ----------------------------------------------------------------------------------------------
def fig_m3(m3, name="figM3_floors"):
    """Rows: instances. Left: runs from W_0 = 0. Right: runs started at relative squared distance 1e-14 from W_*
    (a method without floor stays there; a method with a floor climbs to it)."""
    res = m3["results"]
    keys = list(res)
    fig = ps.figure(width=ps.FIG_W, height=2.35 * len(keys) + 0.95)
    gs = fig.add_gridspec(len(keys) + 1, 2, height_ratios=[1.0] * len(keys) + [0.2])
    handles, labels = [], []
    for row, key in enumerate(keys):
        block = res[key]
        b = m3["instances"][key]["batch"]
        series = [("SAGA est., fixed length 1e-2", ps.BLUE, "SAGA estimator, fixed length 10⁻²"),
                  ("SAGA est., nuclear-scaled", ps.ORANGE, "SAGA estimator, nuclear-scaled"),
                  ("SAGA est., clipped (rho = 1e-2)", ps.AQUA, "SAGA estimator, clipped (ρ = 10⁻²)"),
                  ("SAGA est., bounded gain (eps = 0.04), fixed length 1e-2", ps.YELLOW,
                   "SAGA estimator, bounded gain, fixed length 10⁻²"),
                  (f"Muon b={b}, lr 1e-2", ps.MAGENTA, f"Muon (b = {b}), lr 10⁻²"),
                  (f"STORM-Muon b={b}, lr 1e-2", ps.GREEN, f"STORM-Muon (b = {b}), lr 10⁻²"),
                  (f"STORM buffer b={b}, nuclear-scaled", ps.VIOLET, f"STORM buffer (b = {b}), nuclear-scaled"),
                  (f"minibatch b={b}, nuclear-scaled", ps.RED, f"minibatch gradient (b = {b}), nuclear-scaled")]
        for colm, tkey in enumerate(("trace", "trace_stat")):
            ax = fig.add_subplot(gs[row, colm])
            hs, ls_ = [], []
            for lab, color, text in series:
                e = block.get(lab)
                if not e or not e.get(tkey):
                    continue
                p, rel = np.asarray(e[tkey]["passes"]), np.asarray(e[tkey]["rel"])
                ok = np.isfinite(rel) & (rel > 0)
                h, = ax.plot(p[ok], np.maximum(rel[ok], 1e-30), color=color, lw=1.5)
                hs.append(h)
                ls_.append(text)
            ax.set_yscale("log")
            ax.set_ylim(1e-24, 10)
            ax.set_yticks([1e-24, 1e-18, 1e-12, 1e-6, 1])
            ax.minorticks_off()
            ttl = "from W₀ = 0" if colm == 0 else "started 10⁻¹⁴ from W⋆"
            ax.set_title(f"{INSTANCE_TITLE.get(key, key)}: {ttl}", loc="left", fontsize=ps.SMALL + 0.5)
            if row == len(keys) - 1:
                ax.set_xlabel("passes")
            if colm == 0:
                ax.set_ylabel(DIST_LABEL)
            if len(hs) > len(handles):
                handles, labels = hs, ls_
    ps.legend_cell(fig, gs[len(keys), :], handles, labels, ncol=2)
    ps.savefig(fig, name)


def fig_m4(m4, name="figM4_sensitivity"):
    """Emphasis form: the exact-polar joint method at three step sizes (one hue, darker = larger step) against all
    the other configurations in neutral ink."""
    res = m4["results"]
    keys = list(res)
    labels = list(next(iter(res.values())).keys())
    ramp = {"joint, exact polar, c = 1/64": ps.BLUE_RAMP[3], "joint, exact polar, c = 1/16": ps.BLUE_RAMP[7],
            "joint, exact polar, c = 1/4": ps.BLUE_RAMP[11]}
    others = [lab for lab in labels if lab not in ramp]
    fig = ps.figure(width=ps.FIG_W, height=3.25)
    gs = fig.add_gridspec(2, len(keys), height_ratios=[1.0, 0.2])
    for col, key in enumerate(keys):
        ax = fig.add_subplot(gs[0, col])
        for lab in others + list(ramp):
            tr = res[key][lab]["trace"]
            if not tr:
                continue
            p, sep = np.asarray(tr["passes"]), np.asarray(tr["sep"])
            ok = np.isfinite(sep) & (sep > 0)
            if lab in ramp:
                ax.plot(p[ok], sep[ok], color=ramp[lab], lw=1.8, zorder=3)
            else:
                ax.plot(p[ok], sep[ok], color=INK["muted"], lw=0.9, alpha=0.8, zorder=2)
        ax.set_yscale("log")
        ax.set_ylim(1e-13, 10)
        ax.set_yticks([1e-12, 1e-9, 1e-6, 1e-3, 1])
        ax.minorticks_off()
        ax.set_xlabel("passes")
        if col == 0:
            ax.set_ylabel(r"$\|W_a - W_b\|_F\ /\ \|W_a - W_\star\|_F$")
        ax.set_title(INSTANCE_TITLE.get(key, key), loc="left")
    handles = [plt.Line2D([], [], color=c, lw=2) for c in ramp.values()] + \
              [plt.Line2D([], [], color=INK["muted"], lw=1.2)]
    texts = ["exact polar, joint, c = 1/64", "exact polar, joint, c = 1/16", "exact polar, joint, c = 1/4",
             "other configurations at c = 1/4: Euclidean SAGA; joint with Newton-Schulz,\nwith bounded gain or with "
             "damping; split"]
    ps.legend_cell(fig, gs[1, :], handles, texts, ncol=2)
    ps.savefig(fig, name)


def fig_m5(m5, name="figM5_batch"):
    keys = list(m5)
    kinds = [("SAGA", ps.BLUE, "o", "Euclidean SAGA"), ("SAGA-Muon", ps.ORANGE, "s", "SAGA-Muon (joint)"),
             ("B-SAGA", ps.AQUA, "D", "Euclidean B-SAGA (damped)")]
    ncol = min(len(keys), 4)
    fig = ps.figure(width=ps.FIG_W, height=2.9)
    gs = fig.add_gridspec(2, ncol, height_ratios=[1.0, 0.1])
    for col, key in enumerate(keys[:ncol]):
        block = m5[key]
        tk = f"{block['tol']:g}"
        ax = fig.add_subplot(gs[0, col])
        for kind, color, mk, _ in kinds:
            rows = sorted((r for r in block["rows"] if r["kind"] == kind), key=lambda r: r["batch"])
            b = np.array([r["batch"] for r in rows], dtype=float)
            med = np.array([r["summary"][tk]["median"] for r in rows])
            lo = np.array([r["summary"][tk]["min"] for r in rows])
            hi = np.array([r["summary"][tk]["max"] for r in rows])
            ok = np.isfinite(med)
            hi = np.where(np.isfinite(hi), hi, block["passes"])
            ax.fill_between(b[ok], lo[ok], hi[ok], color=color, alpha=0.10, lw=0)
            ax.plot(b[ok], med[ok], color=color, lw=1.6, marker=mk, ms=5.5, mec=INK["surface"], mew=1.0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([1, 4, 16, 64])
        ax.set_xticklabels(["1", "4", "16", "64"])
        ylo, yhi = ax.get_ylim()
        yt = _nice_ticks(ylo, yhi, 6)
        if yt:
            ax.set_yticks(yt)
            ax.set_yticklabels([str(v) for v in yt])
        ax.minorticks_off()
        ax.set_xlabel("batch size b")
        if col == 0:
            ax.set_ylabel(f"passes to {block['tol']:g}")
        rk = min(block["spec"]["K"] - (1 if block["spec"]["loss"] == "softmax" else 0), block["spec"]["d"])
        ax.set_title(f"{key} (rank {rk})", loc="left", fontsize=ps.SMALL + 0.5)
    handles = [plt.Line2D([], [], color=c, lw=2, marker=m, ms=5.5, mec=INK["surface"]) for _, c, m, _ in kinds]
    ps.legend_cell(fig, gs[1, :], handles, [t for *_, t in kinds], ncol=4)
    ps.savefig(fig, name)


def fig_m6(m6, name="figM6_miso"):
    """Traces of the joint SAGA-Muon runs against their Euclidean counterparts (MISO, dual-free SDCA): the wide light
    lines are the Euclidean methods, the thin dark lines the orthogonalized ones (secondary encoding: line weight)."""
    res = m6["results"]
    keys = list(res)
    series = [("SAGA (Euclidean), c = 1/4", ps.BLUE, 1.6, 1.0, "SAGA, c = 1/4"),
              ("MISO (delta = 1)", ps.ORANGE, 4.0, 0.45, "MISO"),
              ("joint SAGA-Muon, c = 1/4", ps.RED, 1.2, 1.0, "joint SAGA-Muon, c = 1/4"),
              ("dual-free SDCA (delta = sdca)", ps.AQUA, 4.0, 0.45, "dual-free SDCA"),
              ("joint SAGA-Muon, relaxed table, c = 1/4", ps.GREEN, 1.2, 1.0, "joint SAGA-Muon, relaxed table, c = 1/4")]
    ncol = 3
    nrow = int(math.ceil(len(keys) / ncol))
    fig = ps.figure(width=ps.FIG_W, height=2.2 * nrow + 0.6)
    gs = fig.add_gridspec(nrow + 1, ncol, height_ratios=[1.0] * nrow + [0.12])
    for q, key in enumerate(keys):
        ax = fig.add_subplot(gs[q // ncol, q % ncol])
        for lab, color, lw, alpha, _ in series:
            e = res[key]["rows"].get(lab)
            if not e or not e.get("trace"):
                continue
            p, rel = np.asarray(e["trace"]["passes"]), np.asarray(e["trace"]["rel"])
            ok = np.isfinite(rel) & (rel > 0) & (rel < 1e6)
            ax.plot(p[ok], rel[ok], color=color, lw=lw, alpha=alpha, solid_capstyle="round")
        ax.set_yscale("log")
        ax.set_ylim(1e-20, 1e3)
        ax.set_yticks([1e-18, 1e-12, 1e-6, 1])
        ax.minorticks_off()
        ax.set_xlim(0, m6["passes"])
        ax.set_title(key.replace("kappa/n", "κ/n").replace("ls,", "least squares,"), loc="left",
                     fontsize=ps.SMALL + 0.5)
        if q // ncol == nrow - 1:
            ax.set_xlabel("passes")
        if q % ncol == 0:
            ax.set_ylabel(DIST_LABEL)
    handles = [plt.Line2D([], [], color=c, lw=min(lw, 3.2), alpha=a) for _, c, lw, a, _ in series]
    ps.legend_cell(fig, gs[nrow, :], handles, [t for *_, t in series], ncol=3)
    ps.savefig(fig, name)


def fig_m8(m8, m1, name="figM8_dataseeds"):
    """Rows: families; one marker shape per draw of the data (median over the confirmation seeds, thin line =
    range); colour = group, as in the ranking figures."""
    keys = [k for k in m8 if m1 and k in m1]
    if not keys:
        return
    eu = ("SAGA", "B-SAGA", "IS-B-SAGA", "cov-metric SAGA")
    marks = {"0": "o", "1": "s", "2": "D", "3": "^"}
    ncol = 2 if len(keys) > 2 else len(keys)
    nrow = int(math.ceil(len(keys) / ncol))
    nfam = max(len(next(iter(m8[k].values()))["rows"]) for k in keys)
    fig = ps.figure(width=ps.FIG_W, height=(0.27 * nfam + 0.75) * nrow + 0.45)
    gs = fig.add_gridspec(nrow + 1, ncol, height_ratios=[1.0] * nrow + [0.1 / nrow + 0.03])
    seeds_all = []
    for q, key in enumerate(keys):
        col = q % ncol
        block = m8[key]
        seeds = ["0"] + sorted(block.keys(), key=int)
        seeds_all = seeds if len(seeds) > len(seeds_all) else seeds_all
        tk = f"{m1[key]['tol']:g}"
        fams = [r["family"] for r in block[seeds[1]]["rows"]]
        cols = {"0": {r["family"]: r for r in m1[key]["rows"] if r["family"] in fams}}
        for sd in seeds[1:]:
            cols[sd] = {r["family"]: r for r in block[sd]["rows"]}
        order = sorted(fams, key=lambda f: (f not in eu, fams.index(f)))
        ax = fig.add_subplot(gs[q // ncol, col])
        xmax = m1[key]["passes"]
        for i, fam in enumerate(order):
            y0 = len(order) - 1 - i
            color = GROUP_COLOR["euclid" if fam in eu else "spectral"]
            for q, sd in enumerate(seeds):
                r = cols[sd].get(fam)
                if r is None:
                    continue
                s_ = r["summary"][tk]
                y = y0 + 0.24 * (1 - q)
                if _fin(s_["median"]):
                    hi = s_["max"] if _fin(s_["max"]) else xmax
                    ax.plot([s_["min"], hi], [y, y], color=color, lw=0.9, zorder=2)
                    ax.plot(s_["median"], y, marks[sd], ms=5.0, color=color, mec=INK["surface"], mew=1.0, zorder=3)
                else:
                    ax.plot(xmax * 1.25, y, marks[sd], ms=5.0, mfc=INK["surface"], mec=color, mew=1.2, zorder=3)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order[::-1] if col == 0 else [""] * len(order), fontsize=ps.SMALL)
        ax.set_ylim(-0.7, len(order) - 0.3)
        ax.set_xscale("log")
        lo, hi = ax.get_xlim()
        ticks = _nice_ticks(max(lo, 1.0), min(hi, xmax), 6)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.minorticks_off()
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel(f"passes to reach relative squared distance {m1[key]['tol']:g}")
        ax.set_title(INSTANCE_TITLE.get(key, key), loc="left")
    handles = [plt.Line2D([], [], marker=marks[sd], ls="", ms=5.5, color=INK["secondary"], mec=INK["surface"])
               for sd in seeds_all]
    handles += [plt.Line2D([], [], marker="o", ls="", ms=5.5, color=GROUP_COLOR[g], mec=INK["surface"])
                for g in ("euclid", "spectral")]
    labels = [f"data seed {sd}" for sd in seeds_all] + ["Euclidean (plain or covariance metric)",
                                                        "orthogonalized or polar metric"]
    ps.legend_cell(fig, gs[nrow, :], handles, labels, ncol=len(handles))
    ps.savefig(fig, name)


def fig_m2(m2, m1, name="figM2_mechanism"):
    keys = [k for k in INSTANCE_TITLE if k in m2]
    if not keys:
        return
    fams = [f for f in ("SAGA", "B-SAGA", "SAGA-Muon (joint)", "damped SAGA-Muon", "mean-scaled SAGA-Muon",
                        "soft SAGA-Muon", "split SAGA-Muon", "tangent SAGA-Muon", "radial SAGA-Muon",
                        "polar-metric SAGA", "one-shot polar-metric SAGA", "cov-metric SAGA")]
    ncol = 3 if len(keys) > 2 else len(keys)
    nrow = int(math.ceil(len(keys) / ncol))
    fig = ps.figure(width=ps.FIG_W, height=2.2 * nrow + 0.6)
    gs = fig.add_gridspec(nrow + 1, ncol, height_ratios=[1.0] * nrow + [0.1])

    def med(recs, k, scale):
        v = [r[k] * scale for r in recs if k in r and np.isfinite(r[k])]
        return float(np.median(v)) if v else float("nan")

    for q, key in enumerate(keys):
        ax = fig.add_subplot(gs[q // ncol, q % ncol])
        n = m1[key]["spec"]["n"]
        present = [f for f in fams if f in m2[key]]
        for i, fam in enumerate(present):
            y = len(present) - 1 - i
            r = m2[key][fam]
            own = med(r["own"], "eff_dist2", n)
            cross = med(r["cross"], "eff_dist2", n)
            real = [x["rate_per_pass"] for x in r["realized"] if np.isfinite(x["rate_per_pass"])]
            real = float(np.median(real)) if real else float("nan")
            vals = [v for v in (own, cross, real) if np.isfinite(v)]
            if vals:
                ax.plot([min(vals), max(vals)], [y, y], color=INK["axis"], lw=1.0, zorder=1)
            ax.plot(cross, y, "D", ms=5.5, color=ps.ORANGE, mec=INK["surface"], mew=1.0, zorder=3)
            ax.plot(own, y, "o", ms=5.5, color=ps.BLUE, mec=INK["surface"], mew=1.0, zorder=3)
            ax.plot(real, y, "s", ms=5.0, color=ps.AQUA, mec=INK["surface"], mew=1.0, zorder=4)
        ax.set_yticks(range(len(present)))
        ax.set_yticklabels(present[::-1] if q % ncol == 0 else [""] * len(present), fontsize=ps.SMALL - 0.5)
        ax.set_xscale("symlog", linthresh=0.01)
        ax.axvline(0.0, color=INK["axis"], lw=0.8)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(INSTANCE_TITLE.get(key, key), loc="left", fontsize=ps.SMALL + 0.5)
        if q // ncol == nrow - 1:
            ax.set_xlabel(r"relative decrease of $\|W - W_\star\|_F^2$ per pass")
    handles = [plt.Line2D([], [], marker="D", ls="", ms=5.5, color=ps.ORANGE, mec=INK["surface"]),
               plt.Line2D([], [], marker="o", ls="", ms=5.5, color=ps.BLUE, mec=INK["surface"]),
               plt.Line2D([], [], marker="s", ls="", ms=5.0, color=ps.AQUA, mec=INK["surface"])]
    ps.legend_cell(fig, gs[nrow, :], handles,
                   ["one-step expectation at SAGA states", "one-step expectation at own states", "realized"], ncol=3)
    ps.savefig(fig, name)


def main():
    quick = "--quick" in sys.argv
    which = [a for a in sys.argv[1:] if not a.startswith("--")] or ["M1", "M2", "M3", "M4", "M5", "M6", "M8"]
    sfx = "_quick" if quick else ""
    ps.setup()
    m1 = C.load_json(f"M1{sfx}_summary")
    if "M1" in which and m1:
        fig_m1_ranking(m1, ["soft-mem", "ls-mem"], "figM1_ranking_memory")
        fig_m1_ranking(m1, ["soft-cond", "ls-cond"], "figM1_ranking_conditioning")
        fig_m1_ranking(m1, ["ls-orth", "soft-sq"], "figM1_ranking_square")
        fig_m1_curves(m1)
    m2 = C.load_json(f"M2{sfx}_summary")
    if "M2" in which and m2 and m1:
        fig_m2(m2, m1)
    m3 = C.load_json(f"M3{sfx}_summary")
    if "M3" in which and m3:
        fig_m3(m3)
    m4 = C.load_json(f"M4{sfx}_summary")
    if "M4" in which and m4:
        fig_m4(m4)
    m5 = C.load_json(f"M5{sfx}_summary")
    if "M5" in which and m5:
        fig_m5(m5)
    m6 = C.load_json(f"M6{sfx}_summary")
    if "M6" in which and m6:
        fig_m6(m6)
    m8 = C.load_json(f"M8{sfx}_summary")
    if "M8" in which and m8 and m1:
        fig_m8(m8, m1)


if __name__ == "__main__":
    main()
