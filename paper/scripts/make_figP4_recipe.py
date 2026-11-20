"""Figure 4 of the paper paper/paper_saga_to_muon.tex (file name figP4_recipe): Muon's own recipe, steps of fixed
length on a geometric cooldown of the learning rate, fed by a SAGA table (experiments M9, M5b and M9b). One row of
three panels:

  (a) the schedule law     M9 part C, `ls-cond`, batch one, 300 passes without stop rule, 5 seeds (median over the
                           seeds of the relative squared distance at each record): the exact polar factor with the SAGA
                           table at its selected point; Muon's recipe (ns5 + Nesterov 0.9) with the table at the fixed
                           point q = 0.8, lr_0 = 0.1, with the default eps = 1e-7 of ns5 and with eps = 1e-30; the
                           schedule selected for Muon's recipe on the minibatch gradient and on MARS-M, without a table.
  (b) batch size           M9 part A, the two non-square conditioning-limited instances, b = 1, 16, 32: passes to the
                           relative squared distance 1e-6 (median over ten confirmation seeds) of the fastest spectral
                           variant (polar or ns5, with or without Nesterov momentum), of the fastest control map
                           (Frobenius normalization or sign, with or without momentum) on the same cooldown, and of the
                           faster of SAGA and B-SAGA with constant steps on 2^(1/4) grids (at b = 1 from M5b, the same
                           design; part A did not rerun them at b = 1).
  (c) sampling order       M9 part D (the two memory-limited instances) and M9b (a sum with identical Hessians, the
                           model), exact polar factor without momentum: passes to 1e-9 at the fastest cooldown where
                           every seed reaches the target (median over 5 seeds), with independent batches (open marker)
                           and with a fresh permutation per pass (filled marker), side by side for each batch size.

The script reads muon/results/M9_summary.json, M5b_summary.json, M9b_summary.json and the cache M9_law.pkl of part C
of M9 (rebuilt by `python muon/experiments/exp_M9_schedule.py --parts C`, a few minutes; it stops if a run is missing).
It prints every plotted number.

Run:    python paper/scripts/make_figP4_recipe.py
Output: paper/figures/figP4_recipe.png and .pdf
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))                 # paper/scripts -> project root
RESULTS = os.path.join(ROOT, "muon", "results")
FIGURES = os.path.join(ROOT, "paper", "figures")
NAME = "figP4_recipe"
TOL_A, TOL_D = "1e-06", "1e-09"
DIST_LABEL = r"$\|W - W_\star\|_F^2\ /\ \|W_\star\|_F^2$"

A_INSTANCE = "ls-cond"
A_SERIES = (   # (kind, key in M9_summary.json C/<instance>/<kind>, legend label, colour slot, line style)
    ("deep", "polar, no momentum", "exact polar factor", "ORANGE", "-"),
    ("deep", "ns5 + Nesterov at q = 0.8, lr_0 = 0.1", "Muon's recipe (ns5)", "ORANGE", (0, (4.0, 1.8))),
    ("deep", "ns5 + Nesterov, eps = 1e-30, at q = 0.8, lr_0 = 0.1", "the same, eps = 1e-30", "ORANGE",
     (0, (1.0, 1.4))),
    ("controls", "no table (minibatch gradient), ns5 + Nesterov, b=1", "minibatch gradient", "MUTED", "-"),
    ("controls", "MARS-M, b=1", "MARS-M", "MUTED", (0, (4.0, 1.8))),
)
B_INSTANCES = (("soft-cond", "softmax"), ("ls-cond", "least squares"))
B_BATCHES = ("1", "16", "32")
D_INSTANCES = (("soft-mem", "softmax"), ("ls-mem", "least sq."), ("model", "model"))
D_BATCHES = ("1", "5", "20", "50")
D_SCHEMES = (("iid", "independent batches"), ("reshuffle", "fresh permutation per pass"))


def load(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as f:
        return json.load(f)


def task_key(task):
    """Key of a task in the caches of muon/experiments/common.py (common.task_key)."""
    return hashlib.sha1(json.dumps(task, sort_keys=True).encode()).hexdigest()


# ----------------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------------
def data_a(m9):
    """Median over the seeds of the relative squared distance, at the records of the runs of part C (the records of
    the seeds of one configuration fall at the same passes)."""
    e = m9["C"][A_INSTANCE]
    path = os.path.join(RESULTS, "M9_law.pkl")
    if not os.path.exists(path):
        raise SystemExit(f"{path} is missing: run `python muon/experiments/exp_M9_schedule.py --parts C` first")
    with open(path, "rb") as f:
        cache = pickle.load(f)
    out = []
    for kind, label, legend, colour, ls in A_SERIES:
        d = e[kind][label]
        runs = []
        for seed in e["seeds"]:
            task = {"spec": e["spec"], "method": d.get("method", "vr"), "cfg": d["cfg"], "passes": e["passes"],
                    "seed": seed, "stop_rel": 0.0}
            r = cache.get(task_key(task))
            if r is None:
                raise SystemExit(f"run missing from M9_law.pkl: {label}, seed {seed}")
            runs.append(r)
        p = np.asarray(runs[0]["trace"]["passes"], dtype=float)
        for r in runs[1:]:
            if not np.allclose(np.asarray(r["trace"]["passes"], dtype=float), p):
                raise SystemExit(f"{label}: the seeds are not recorded at the same passes")
        rel = np.array([r["trace"]["rel"] for r in runs], dtype=float)
        med = 10.0 ** np.median(np.log10(rel), axis=0)
        out.append({"kind": kind, "label": label, "legend": legend, "colour": colour, "ls": ls, "passes": p,
                    "median": med, "cfg": d["cfg"], "lowest": [float(r["min_rel"]) for r in runs],
                    "final": [float(r["final_rel"]) for r in runs]})
    return out


def data_b(m9, m5b):
    """Per instance and batch size: medians (and confirmation seeds that reach the target) of the three groups."""
    out = {}
    for inst, _ in B_INSTANCES:
        for b in B_BATCHES:
            cmp_ = m9["A"][inst]["cells"][b]["compare"]
            row = {"spectral": cmp_["spectral"], "control": cmp_["control"]}
            if cmp_.get("euclid"):
                row["euclid"] = cmp_["euclid"]
            else:                                   # b = 1: the references of M5b on 2^(1/4) grids (same design)
                rows = m5b["instances"][inst]["0"]["batches"][b]["fine"]["rows"]
                best = min(("SAGA", "B-SAGA"), key=lambda f: rows[f]["summary"][TOL_A]["median"])
                sm = rows[best]["summary"][TOL_A]
                row["euclid"] = {"family": f"{best} (M5b)", "variant": best, "median": sm["median"],
                                 "reached": sm["reached"], "runs": sm["runs"]}
            out[(inst, b)] = row
    return out


def data_d(m9, m9b):
    """Per instance (the model: M9b), batch size and scheme: passes to 1e-9 at the fastest cooldown, and rho_star."""
    out = {}
    for inst, _ in D_INSTANCES:
        block = m9b["D"] if inst == "model" else m9["D"][inst]
        for b in D_BATCHES:
            for scheme, _ in D_SCHEMES:
                sc = block["cells"][b]["schemes"][scheme]
                out[(inst, b, scheme)] = {"fastest": sc["fastest"], "rho_star": sc["rho_star"]}
    return out


def print_all(A, B, D):
    print(f"(a) {A_INSTANCE}, batch one, median over the seeds; lowest and final relative squared distance per seed")
    for s in A:
        cfg = {k: v for k, v in s["cfg"].items() if k in ("estimator", "orth", "nesterov", "lr_decay", "lr", "orth_kw",
                                                           "batch", "schedule")}
        print(f"  {s['legend']:34s} {json.dumps(cfg)}")
        print(f"      lowest: {', '.join(f'{v:.2g}' for v in s['lowest'])}; final: {', '.join(f'{v:.3g}' for v in s['final'])}")
    print(f"(b) passes to {TOL_A}: median (seeds reaching the target / seeds)")
    for (inst, b), row in B.items():
        txt = "; ".join(f"{g} {row[g]['variant']} {row[g]['median']:.1f} ({row[g]['reached']}/{row[g]['runs']})"
                        for g in ("spectral", "control", "euclid"))
        r1 = row["spectral"]["median"] / row["control"]["median"]
        r2 = row["spectral"]["median"] / row["euclid"]["median"]
        print(f"  {inst:9s} b = {b:>2s}: {txt}; spectral/control {r1:.2f}, spectral/euclid {r2:.2f}")
    print(f"(c) passes to {TOL_D} at the fastest rho: median [min, max]; rho_star")
    for (inst, b, scheme), v in D.items():
        f = v["fastest"]
        print(f"  {inst:9s} b = {b:>2s} {scheme:9s}: {f['median']:.1f} [{f['min']:.1f}, {f['max']:.1f}] at rho "
              f"{f['rho']:g}; rho_star {v['rho_star']}")


# ----------------------------------------------------------------------------------------------------------------------
# figure (style of the main project: experiments/plotstyle.py)
# ----------------------------------------------------------------------------------------------------------------------
def _plotstyle():
    sys.modules.setdefault("common", types.SimpleNamespace(FIGURES=FIGURES))   # plotstyle only needs common.FIGURES
    sys.path.insert(0, os.path.join(ROOT, "experiments"))
    import plotstyle as ps
    ps.setup()
    return ps


def _colour(ps, slot):
    return {"ORANGE": ps.ORANGE, "BLUE": ps.BLUE, "MUTED": ps.INK["secondary"]}[slot]


def panel_a(ax, A, ps):
    for s in A:
        ax.plot(s["passes"], s["median"], color=_colour(ps, s["colour"]), ls=s["ls"], lw=1.6,
                zorder=3 if s["kind"] == "deep" else 2)
    kw = dict(fontsize=ps.SMALL, color=ps.INK["secondary"], ha="right")
    ax.annotate("no table:\nminibatch (solid),\nMARS-M (dashed)", (296, 1.5e-3), va="top", **kw)
    ax.annotate("table, Muon's ns5", (296, 4e-18), va="bottom", **kw)
    ax.annotate("table, exact polar", (296, 8e-29), va="bottom", **kw)
    ax.annotate(r"table, ns5, $\epsilon = 10^{-30}$", (296, 1.4e-30), va="top", **kw)
    ax.set_yscale("log")
    ax.set_xlim(0, 300)
    ax.set_ylim(1e-34, 30.0)
    ax.set_yticks([1e-30, 1e-24, 1e-18, 1e-12, 1e-6, 1.0])
    ax.minorticks_off()
    ax.set_xticks([0, 100, 200, 300])
    ax.set_xlabel("passes")
    ax.set_ylabel(DIST_LABEL)
    ax.set_title("(a) schedule law, b = 1", loc="left")



def _group_axis(ax, groups, batches, gap, ps, xlabel, tick_size=None):
    """Categorical x positions: one block of batch sizes per group, the group names under the tick labels."""
    pos, ticks, labels = {}, [], []
    x = 0.0
    for gi, (g, _) in enumerate(groups):
        for b in batches:
            pos[(g, b)] = x
            ticks.append(x)
            labels.append(b)
            x += 1.0
        x += gap
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=tick_size)
    ax.grid(axis="x", visible=False)
    ax.set_xlim(-0.6, x - gap - 0.4)
    nb = len(batches)
    for gi, (g, name) in enumerate(groups):
        mid = 0.5 * (pos[(g, batches[0])] + pos[(g, batches[-1])])
        ax.annotate(name, (mid, 0), xycoords=("data", "axes fraction"), xytext=(0, -19), textcoords="offset points",
                    ha="center", va="top", fontsize=ps.SMALL, color=ps.INK["secondary"])
        if gi:
            sep = pos[(g, batches[0])] - 0.5 - 0.5 * gap
            ax.axvline(sep, color=ps.INK["grid"], lw=0.8, zorder=0)
    ax.set_xlabel(xlabel, labelpad=13)
    return pos, nb


def panel_b(ax, B, ps):
    pos, _ = _group_axis(ax, B_INSTANCES, B_BATCHES, 0.7, ps, "batch size b", tick_size=ps.SMALL)
    style = {"spectral": (ps.ORANGE, "o", "-", "spectral map"),
             "control": (ps.INK["secondary"], "D", (0, (4.0, 1.8)), "control map"),
             "euclid": (ps.BLUE, "s", "-", "SAGA, B-SAGA")}
    for g, (col, mk, ls, lab) in style.items():
        for i, (inst, _) in enumerate(B_INSTANCES):
            xs = [pos[(inst, b)] for b in B_BATCHES]
            ys = [B[(inst, b)][g]["median"] for b in B_BATCHES]
            ax.plot(xs, ys, color=col, ls=ls, lw=1.4, marker=mk, ms=4.6, mec=ps.INK["surface"], mew=0.9,
                    label=lab if i == 0 else None, zorder=3)
            for b, x, y in zip(B_BATCHES, xs, ys):
                r = B[(inst, b)][g]
                if r["reached"] < r["runs"]:
                    ax.annotate(f"{r['reached']}/{r['runs']}", (x, y), xytext=(0, -7), textcoords="offset points",
                                ha="center", va="top", fontsize=ps.SMALL, color=ps.INK["secondary"])
    ax.set_yscale("log")
    ax.set_ylim(15, 170)
    ax.set_yticks([20, 30, 50, 70, 100])
    ax.set_yticklabels(["20", "30", "50", "70", "100"])
    ax.minorticks_off()
    ax.set_ylabel(r"passes to $10^{-6}$")
    ax.set_title("(b) batch size", loc="left")
    ps.legend(ax, loc="upper left", handlelength=2.0, handletextpad=0.4, borderpad=0.1, labelspacing=0.25,
              frameon=False)


def panel_c(ax, D, ps):
    shape = {"soft-mem": "o", "ls-mem": "s", "model": "D"}
    offset = {"soft-mem": -0.25, "ls-mem": 0.0, "model": 0.25}
    for k, b in enumerate(D_BATCHES):
        for inst, _ in D_INSTANCES:
            x = k + offset[inst]
            col = ps.INK["secondary"] if inst == "model" else ps.ORANGE
            f_i, f_r = D[(inst, b, "iid")]["fastest"], D[(inst, b, "reshuffle")]["fastest"]
            ax.plot([x, x], [f_i["median"], f_r["median"]], color=col, lw=1.0, zorder=2)
            ax.plot(x, f_i["median"], ls="", marker=shape[inst], ms=4.6, mfc=ps.INK["surface"], mec=col, mew=1.1,
                    zorder=3)
            ax.plot(x, f_r["median"], ls="", marker=shape[inst], ms=4.6, mfc=col, mec=ps.INK["surface"], mew=0.7,
                    zorder=4)
    ax.set_xticks(range(len(D_BATCHES)))
    ax.set_xticklabels(D_BATCHES)
    ax.set_xlim(-0.55, len(D_BATCHES) - 0.45)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("batch size b")
    ax.set_ylim(0, 26)
    ax.set_yticks([0, 5, 10, 15, 20, 25])
    ax.set_ylabel(r"passes to $10^{-9}$")
    ax.set_title("(c) sampling order", loc="left")
    L2D = ps.plt.Line2D
    h = [L2D([], [], ls="", marker="o", ms=4.6, mfc=ps.ORANGE, mec=ps.INK["surface"]),
         L2D([], [], ls="", marker="s", ms=4.6, mfc=ps.ORANGE, mec=ps.INK["surface"]),
         L2D([], [], ls="", marker="D", ms=4.6, mfc=ps.INK["secondary"], mec=ps.INK["surface"]),
         L2D([], [], ls="", marker="o", ms=4.6, mfc=ps.INK["surface"], mec=ps.INK["secondary"], mew=1.1),
         L2D([], [], ls="", marker="o", ms=4.6, mfc=ps.INK["secondary"], mec=ps.INK["surface"])]
    ps.legend(ax, handles=h, labels=["softmax", "least squares", "model", "independent", "permutation"],
              loc="lower left", ncol=2, handletextpad=0.2, columnspacing=0.8, borderpad=0.1, labelspacing=0.2,
              frameon=False)


def main():
    m9, m5b, m9b = load("M9_summary.json"), load("M5b_summary.json"), load("M9b_summary.json")
    A, B, D = data_a(m9), data_b(m9, m5b), data_d(m9, m9b)
    print_all(A, B, D)
    ps = _plotstyle()
    fig = ps.figure(width=ps.FIG_W, height=2.7)
    fig.get_layout_engine().set(w_pad=0.03, wspace=0.0)
    gs = fig.add_gridspec(1, 3, width_ratios=(1.15, 1.0, 1.0))
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, k]) for k in range(3))
    panel_a(ax_a, A, ps)
    panel_b(ax_b, B, ps)
    panel_c(ax_c, D, ps)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W = fig.get_figwidth() * fig.dpi
    for ax in fig.axes:                                   # what extends beyond the figure (diagnostic)
        for art in [ax.title, ax.xaxis.label, ax.yaxis.label] + ax.texts + ax.get_xticklabels() + ax.get_yticklabels():
            if art.get_visible() and art.get_text():
                bb = art.get_window_extent(r)
                if bb.x0 < 0 or bb.x1 > W:
                    print(f"   outside the figure: {art.get_text()!r} x0 {bb.x0 / fig.dpi:.2f} x1 {bb.x1 / fig.dpi:.2f} in")
    ps.savefig(fig, NAME)


if __name__ == "__main__":
    main()
