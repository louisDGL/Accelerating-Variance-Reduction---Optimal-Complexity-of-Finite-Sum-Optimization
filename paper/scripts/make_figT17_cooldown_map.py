"""Figure T17 of the lecture, a backup slide (paper/talk/figures/figT17_cooldown_map): the two ways in which a
geometric cooldown of a fixed-length polar step fails, on the tuning grid of the recipe (4) of
paper/paper_saga_to_muon.tex. Explanatory figure, not in the paper; it shows what item (M5) calls "a cliff in q" and
what the limitations call "nothing explains the critical q", together with the path condition of the audit
(paper/audit/AUDIT.md, finding on (M5)): steps of spectral norm lr_t started at W_0 = 0 cannot reach W_* when
sum_t lr_t < ||W_*||_op, whatever the estimator.

Data: experiment M9, part B (muon/experiments/exp_M9_schedule.py), least-squares conditioning-limited instance
"ls-cond" (n = 200, d = 40, K = 8, kappa = 1e4, alpha = 2, data seed 0), family "fixed-length SAGA [polar] b=1":
    W_{t+1} = W_t - lr_t polar(omega_t),  omega_t the SAGA estimator (batch one, independent indices),
    lr_t = lr_0 q^(t / n),  W_0 = 0,  exact polar factor, no momentum,
on the grid q in {0.55, ..., 0.9} x lr_0 in {0.003, ..., 1} (48 points). Read from muon/results/M9_summary.json,
B["ls-cond"]: ["q_grid"], ["lr_grid"], ["norms"] (||W_*||_op and D0 = ||W_*||_F^2), ["passes"] = 300, ["tol"] = 1e-6,
["rows"][family]["tune"] (per point: "screen_rel", "score", "kept"), ["best"], ["fixed_point"]; and from the cache
muon/results/M9_ls-cond_tune.pkl (keyed by the sha1 of the task, as in paper/scripts/make_figP4_recipe.py) the
outcome of each tuning seed.

What is drawn.
  colour   "screen_rel": the relative squared distance ||W - W_*||_F^2 / ||W_*||_F^2 at the end of the SCREENING run
           of the protocol of muon/experiments/common.py (tune_and_confirm): one run per grid point, tuning seed 0,
           60 passes (20 % of the budget), stopped at 1e-9 (1e-3 times the target). It is not a converged value: at
           q = 0.85 and 0.9 a run that is on schedule has not arrived after 60 passes (the marks say whether it
           arrives). One neutral sequential ramp (inks of the style), dark = close to W_*.
  marks    on the 17 points that the protocol ran at the full budget of 300 passes ("kept": the best third of the
           screening and the grid neighbours of the best point), the number of tuning seeds that reach 1e-6 (seeds
           0, 1; seeds 0 to 4 for the three finalists): a dot when all do, a cross when none does, the fraction
           otherwise. The star is the selected point (q = 0.8, lr_0 = 0.3), the diamond the default point
           (q = 0.8, lr_0 = 0.1) that the paper calls "the fixed point". The geometric mean of the scores of seeds
           0 and 1 recomputed from the cache is compared with "score" of the summary (the script stops if they
           differ): a score is a tuning score (passes to 1e-6, or a penalized budget when the target is missed),
           not a pass count.
  dashed   the curve sum_t lr_t = lr_0 / (1 - q^(1/n)) = ||W_*||_op. Below it the run cannot reach W_*: every step
           has spectral norm exactly lr_t (exact polar factor), so ||W_t - W_*||_op >= ||W_*||_op - sum_t lr_t, hence
           ||W_t - W_*||_F^2 / ||W_*||_F^2 >= (||W_*||_op - sum_t lr_t)^2 / ||W_*||_F^2 (the script checks this bound
           on every screening value below the curve). The condition is necessary, not sufficient: the budget is in
           spectral norm and the steps do not all point to W_*.

Status. Measured: every colour and every mark (one instance, one family, tuning seeds). Proved: the path condition
(two lines above). Not explained: the location of the cliff between q = 0.75 and q = 0.8; Proposition 5 gives no
critical q, and no line is drawn for it.

Progressive build: s1 the screening map and the path curve; s2 the marks of the runs at full budget.

Reading against the paper. The sentence of (M5) "its tuned points sit next to a cliff in q and lose confirmation
seeds on three instances (7/10, 9/10, 9/10), which the fixed point q = 0.8, lr_0 = 0.1 avoids" is about the fastest
variants of the recipe and about the ten CONFIRMATION seeds. This figure is one family (exact polar factor, no
momentum) on one instance (ls-cond), where both the selected and the default point reach 1e-6 on 10/10
confirmation seeds (36.1 and 31.3 passes, digest); the "4 of 5" of the default point counts TUNING seeds 0-4 (seed 2
stalls at 1.7e-5), which is why the protocol selected (0.8, 0.3). The scope line names the family.

Deviation from the specification of the figure: panel (b) of the specification (the path budget of the no-table
controls) is a figure of its own, figT17b_path_budget, so that the key of this map fits on the slide.

The script prints every number it plots.
Run:    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/scripts/make_figT17_cooldown_map.py
Output: paper/talk/figures/figT17_cooldown_map.pdf and .png, and the states _s1, _s2
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import explain_style as es  # noqa: E402

NAME = "figT17_cooldown_map"
RESULTS = os.path.join(es.ROOT, "muon", "results")
INSTANCE = "ls-cond"
FAMILY = "fixed-length SAGA [polar] b=1"
BASE = dict(estimator="saga", batch=1, length="const", orth="svd")     # families.recipe_family("svd", False, 1, ...)
SCREEN_FRACTION, SCREEN_STOP, TUNE_STOP = 0.2, 1e-3, 1e-3              # muon/experiments/common.py
TUNE_SEEDS, REFINE_SEEDS = (0, 1), (2, 3, 4)
V_LO, V_HI = 1e-9, 1.0                                                 # colour scale (the screening stops at 1e-9)


# ----------------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------------
def task_key(task):
    """Key of a task in the caches of muon/experiments/common.py (common.task_key)."""
    return hashlib.sha1(json.dumps(task, sort_keys=True).encode()).hexdigest()


def score(res, tol, budget):
    """Tuning score of one run (common.score): passes to tol, or a penalized budget if the target is missed."""
    if res["diverged"] or not math.isfinite(res["final_rel"]):
        return float("inf")
    p = res["passes_to"][f"{tol:g}"]
    if math.isfinite(p):
        return p
    return budget * (1.0 + max(0.0, math.log10(max(res["min_rel"], 1e-300) / tol)))


def _digest_part_B():
    """Rows of the table of the instance in the section "M9 part B" of muon/results/digest.md."""
    rows, part, inside = {}, False, False
    with open(os.path.join(RESULTS, "digest.md"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                part, inside = line.startswith("### M9 part B"), False
            elif part and line.startswith("**"):
                inside = line.startswith(f"**{INSTANCE}**")
            elif part and inside and line.startswith("|"):
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                rows[cells[0]] = cells
    return rows


def data():
    with open(os.path.join(RESULTS, "M9_summary.json"), encoding="utf-8") as f:
        m9 = json.load(f)
    B = m9["B"][INSTANCE]
    row = B["rows"][FAMILY]
    if (row["orth"], row["nesterov"], row["batch"]) != ("svd", False, 1):
        raise SystemExit("the family is not the exact polar factor without momentum at batch one")
    with open(os.path.join(RESULTS, f"M9_{INSTANCE}_tune.pkl"), "rb") as f:
        cache = pickle.load(f)
    s, passes, tol = B["spec"], B["passes"], B["tol"]
    qs, lrs = list(B["q_grid"]), list(B["lr_grid"])
    w_op, D0 = B["norms"]["W_star_op"], B["norms"]["D0"]
    tune = {(t["point"]["lr_decay"], t["point"]["lr"]): t for t in row["tune"]}
    if len(tune) != len(qs) * len(lrs):
        raise SystemExit("the tuning rows do not cover the grid")
    n = s["n"]
    screen = np.empty((len(lrs), len(qs)))
    path = np.empty_like(screen)
    cells = {}
    for i, lr in enumerate(lrs):
        for j, q in enumerate(qs):
            t = tune[q, lr]
            screen[i, j] = t["screen_rel"]
            path[i, j] = lr / (1.0 - q ** (1.0 / n)) / w_op          # sum_t lr_t in units of ||W_*||_op
            if path[i, j] < 1.0 and screen[i, j] < (1.0 - path[i, j]) ** 2 * w_op ** 2 / D0:
                raise SystemExit(f"q = {q}, lr_0 = {lr}: the screening value is below the bound of the path condition")
            if not t["kept"]:
                continue
            runs = []
            for seed in TUNE_SEEDS + REFINE_SEEDS:
                task = {"spec": s, "method": "vr", "cfg": dict(BASE, **t["point"]), "passes": passes, "seed": seed,
                        "stop_rel": float(TUNE_STOP * tol)}
                r = cache.get(task_key(task))
                if r is None:
                    if seed in TUNE_SEEDS:
                        raise SystemExit(f"run missing from the cache: q = {q}, lr_0 = {lr}, seed {seed}")
                    continue
                runs.append({"seed": seed, "passes": r["passes_to"][f"{tol:g}"], "min_rel": r["min_rel"],
                             "score": score(r, tol, passes)})
            if t.get("score_refined") is None:                       # not a finalist: only the two tuning seeds count
                runs = [r for r in runs if r["seed"] in TUNE_SEEDS]
            s2 = float(np.exp(np.mean(np.log([r["score"] for r in runs if r["seed"] in TUNE_SEEDS]))))
            if not math.isclose(s2, t["score"], rel_tol=1e-9):
                raise SystemExit(f"q = {q}, lr_0 = {lr}: score {s2} from the cache, {t['score']} in the summary")
            if t.get("score_refined") is not None:
                sa = float(np.exp(np.mean(np.log([r["score"] for r in runs]))))
                if len(runs) != 5 or not math.isclose(sa, t["score_refined"], rel_tol=1e-9):
                    raise SystemExit(f"q = {q}, lr_0 = {lr}: refined score {sa} from the cache, "
                                     f"{t['score_refined']} in the summary")
            cells[q, lr] = {"runs": runs, "reached": sum(math.isfinite(r["passes"]) for r in runs), "n": len(runs),
                            "score": t["score"], "score_refined": t.get("score_refined")}
    best = (row["best"]["lr_decay"], row["best"]["lr"])
    fp = B["fixed_point"]
    fixed = (fp["point"]["lr_decay"], fp["point"]["lr"])
    conf, conf_fp = row["summary"][f"{tol:g}"], fp["rows"][FAMILY]["summary"][f"{tol:g}"]
    digest = _digest_part_B()
    for name, pt, c in (("fixed-length SAGA [polar]", best, conf), ("fixed-length SAGA [polar], fixed point", fixed,
                                                                    conf_fp)):
        mine = (f"q={pt[0]:g}, lr0={pt[1]:g}", f"{c['median']:.1f} [{c['min']:.1f}, {c['max']:.1f}]"
                + ("" if c["reached"] == c["runs"] else f" ({c['reached']}/{c['runs']})"))
        if digest.get(name, [None] * 3)[1:3] != list(mine):
            raise SystemExit(f"'{name}': {mine} in M9_summary.json, {digest.get(name)} in the digest")
    return {"spec": s, "passes": passes, "tol": tol, "qs": qs, "lrs": lrs, "w_op": w_op, "D0": D0, "screen": screen,
            "path": path, "cells": cells, "best": best, "fixed": fixed, "conf": conf, "conf_fp": conf_fp,
            "screen_passes": max(1, int(math.ceil(SCREEN_FRACTION * passes))), "screen_stop": SCREEN_STOP * tol,
            "n_kept": row["n_kept"], "n_points": row["n_points"]}


def print_all(D):
    s = D["spec"]
    qs, lrs = D["qs"], D["lrs"]
    print(f"instance {INSTANCE}: {s['loss']}, n = {s['n']}, d = {s['d']}, K = {s['K']}, kappa = {s['kappa']:g}, alpha = "
          f"{s['alpha']:g}, data seed {s['seed']}; family '{FAMILY}'; ||W_*||_op = {D['w_op']:.4f}, ||W_*||_F^2 = "
          f"{D['D0']:.2f}")
    print(f"colour: screening run, tuning seed {TUNE_SEEDS[0]}, {D['screen_passes']} passes, stopped at "
          f"{D['screen_stop']:g}: relative squared distance at its end (rows lr_0 from the largest, columns q)")
    print("    lr_0 \\ q " + " ".join(f"{q:>9g}" for q in qs))
    for i in range(len(lrs) - 1, -1, -1):
        print(f"    {lrs[i]:<8g} " + " ".join(f"{v:9.2e}" for v in D["screen"][i]))
    print("total path sum_t lr_t = lr_0 / (1 - q^(1/n)), in units of ||W_*||_op:")
    for i in range(len(lrs) - 1, -1, -1):
        print(f"    {lrs[i]:<8g} " + " ".join(f"{v:9.3f}" for v in D["path"][i]))
    print("dashed curve, lr_0 = ||W_*||_op (1 - q^(1/n)): " +
          ", ".join(f"q = {q:g}: {D['w_op'] * (1 - q ** (1 / s['n'])):.4f}" for q in qs))
    below = D["path"] < 1.0
    print(f"    {int(below.sum())} grid points below the curve; their screening values are in "
          f"[{D['screen'][below].min():.3f}, {D['screen'][below].max():.3f}]; smallest value of the bound "
          f"(1 - path)^2 ||W_*||_op^2 / ||W_*||_F^2 checked: "
          f"{((1 - D['path'][below]) ** 2).min() * D['w_op'] ** 2 / D['D0']:.2e}")
    print(f"marks: the {len(D['cells'])} points run at the full budget of {D['passes']} passes (summary: n_kept = "
          f"{D['n_kept']} of {D['n_points']}); tuning seeds reaching {D['tol']:g}:")
    for (q, lr), c in sorted(D["cells"].items()):
        tag = " [selected]" if (q, lr) == D["best"] else " [default point]" if (q, lr) == D["fixed"] else ""
        print(f"    q = {q:g}, lr_0 = {lr:g}: {c['reached']}/{c['n']}{tag}; per seed (passes to {D['tol']:g} / lowest "
              f"distance): " + ", ".join(f"{r['seed']}: {r['passes']:.1f} / {r['min_rel']:.1e}" for r in c["runs"])
              + f"; score {c['score']:.2f}"
              + (f", on five seeds {c['score_refined']:.2f}" if c["score_refined"] is not None else ""))
    print(f"confirmation seeds 10-19, passes to {D['tol']:g}: selected point {D['best']}: median {D['conf']['median']}, "
          f"{D['conf']['reached']}/{D['conf']['runs']}; default point {D['fixed']}: median {D['conf_fp']['median']}, "
          f"{D['conf_fp']['reached']}/{D['conf_fp']['runs']}")


# ----------------------------------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------------------------------
def _edges(v, log):
    v = np.log(v) if log else np.asarray(v, dtype=float)
    mid = 0.5 * (v[:-1] + v[1:])
    e = np.concatenate([[v[0] - (mid[0] - v[0])], mid, [v[-1] + (v[-1] - mid[-1])]])
    return np.exp(e) if log else e


def draw(D):
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    ps = es.setup("talk")
    ink = ps.INK
    plt = ps.plt
    qs, lrs = D["qs"], D["lrs"]
    fig = es.figure("talk", height=2.75)
    fig.get_layout_engine().set(w_pad=0.03, wspace=0.02)
    gs = fig.add_gridspec(1, 2, width_ratios=(1.42, 1.0))
    ax = fig.add_subplot(gs[0, 0])
    b = es.Build()

    # one neutral sequential ramp (ink on the surface), dark = close to W_*: in this set blue means "no table" and the
    # runs of the map have the SAGA table, so the magnitude is not drawn in a series hue
    cmap = LinearSegmentedColormap.from_list("ink_ramp", ["#1f1f1d", ink["secondary"], ink["muted"], ink["axis"],
                                                          ink["grid"], "#efeee9"])
    norm = LogNorm(vmin=V_LO, vmax=V_HI)
    xe, ye = _edges(qs, False), _edges(lrs, True)
    mesh = ax.pcolormesh(xe, ye, np.clip(D["screen"], V_LO, V_HI), cmap=cmap, norm=norm, edgecolors=ink["surface"],
                         linewidth=1.2, zorder=1)
    ax.set_yscale("log")
    ax.set_xlim(xe[0], xe[-1])
    ax.set_ylim(ye[0], ye[-1])
    ax.set_xticks(qs)
    ax.set_xticklabels([f"{q:g}" for q in qs])
    ax.set_yticks(lrs)
    ax.set_yticklabels([f"{v:g}" for v in lrs])
    ax.minorticks_off()
    ax.grid(False)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel("cooldown per pass, $q$")
    ax.set_ylabel(r"first length, $\mathrm{lr}_0$")

    # the path condition
    qq = np.linspace(xe[0], xe[-1], 200)
    ax.plot(qq, D["w_op"] * (1.0 - qq ** (1.0 / D["spec"]["n"])), color=ink["primary"], lw=1.3, ls=(0, (4.0, 2.2)),
            zorder=3)
    ax.text(qs[0] - 0.018, lrs[0], "path too short, whatever the estimator", ha="left", va="center",
            fontsize=es.small(), color=ink["primary"], zorder=6)

    # marks of the runs at full budget
    dot = dict(ls="", marker="o", ms=6.2, mfc=es.GOOD, mec=ink["surface"], mew=1.0, zorder=5)
    cross = dict(ls="", marker="X", ms=7.2, mfc=es.CRITICAL, mec=ink["surface"], mew=0.9, zorder=5)
    special = dict(ls="", mfc=ink["primary"], mec=ink["surface"], mew=1.0, zorder=5)
    pill = dict(ha="center", va="center", fontsize=es.small(), color=ink["primary"], zorder=6,
                bbox=dict(boxstyle="round,pad=0.16", fc=ink["surface"], ec=ink["axis"], lw=0.6))
    for (q, lr), c in D["cells"].items():
        if (q, lr) == D["best"]:
            b.add(2, ax.plot([q], [lr], marker="*", ms=11.5, **special))
        elif (q, lr) == D["fixed"]:
            b.add(2, ax.plot([q], [lr], marker="D", ms=6.4, **special))
        elif c["reached"] == c["n"]:
            b.add(2, ax.plot([q], [lr], **dot))
        elif c["reached"] == 0:
            b.add(2, ax.plot([q], [lr], **cross))
        else:
            b.add(2, ax.text(q, lr, f"{c['reached']}/{c['n']}", **pill))
    q_cliff = 0.5 * (qs[qs.index(D["best"][0]) - 1] + D["best"][0])
    b.add(2, ax.annotate("cliff", (q_cliff, ye[-1]), xytext=(0, 9), textcoords="offset points", ha="center",
                         va="bottom", fontsize=es.small(), color=ink["primary"], annotation_clip=False,
                         arrowprops=dict(arrowstyle="-|>", color=ink["primary"], lw=1.0, shrinkA=0.5, shrinkB=0.5,
                                         mutation_scale=8)))

    # key (ink), in its own cell: the colour scale, then the marks
    kx = fig.add_subplot(gs[0, 1])
    es.diagram_axes(kx)
    small = es.small()
    kx.text(0.0, 1.0, "colour: " + r"$\|W - W_\ast\|_F^2\,/\,\|W_\ast\|_F^2$" + "\n"
            + rf"after {D['screen_passes']} passes (one seed)", ha="left", va="top", fontsize=small,
            color=ink["secondary"], linespacing=1.25)
    cax = kx.inset_axes([0.04, 0.775, 0.84, 0.05])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.set_ticks([1.0, 1e-3, 1e-6, 1e-9])
    cb.set_ticklabels(["1", r"$10^{-3}$", r"$10^{-6}$", r"$10^{-9}$"])
    cb.ax.minorticks_off()
    cb.ax.tick_params(length=2.5, labelsize=small, pad=1.5)
    cb.ax.invert_xaxis()                                            # from the start (1) to the stop level (1e-9)
    y0, dy, xm, xt = 0.515, 0.094, 0.065, 0.16
    b.add(2, kx.text(0.0, y0 + 0.5 * dy + 0.02, f"{D['passes']} passes, tuning seeds:", ha="left", va="bottom",
                     fontsize=small, color=ink["secondary"]))
    best_c, fixed_c = D["cells"][D["best"]], D["cells"][D["fixed"]]
    mixed = sorted({(c["reached"], c["n"]) for k, c in D["cells"].items()
                    if 0 < c["reached"] < c["n"] and k != D["fixed"]})
    rows = [("dot", r"all reach $10^{-6}$"), ("cross", "none does"),
            ("pill", "some do" if len(mixed) != 1 else f"{mixed[0][0]} of {mixed[0][1]} does"),
            ("star", f"selected: {best_c['reached']} of {best_c['n']} do"),
            ("diamond", f"default point: {fixed_c['reached']} of {fixed_c['n']}")]
    for k, (kind, text) in enumerate(rows):
        y = y0 - k * dy
        if kind == "dot":
            h = kx.plot([xm], [y], **dot)
        elif kind == "cross":
            h = kx.plot([xm], [y], **cross)
        elif kind == "star":
            h = kx.plot([xm], [y], marker="*", ms=11.5, **special)
        elif kind == "diamond":
            h = kx.plot([xm], [y], marker="D", ms=6.4, **special)
        else:
            h = kx.text(xm, y, f"{mixed[0][0]}/{mixed[0][1]}" if len(mixed) == 1 else "k/m", **pill)
        b.add(2, h, kx.text(xt, y, text, ha="left", va="center", fontsize=small, color=ink["secondary"]))
    y = y0 - len(rows) * dy - 0.01
    kx.plot([0.0, 0.13], [y, y], color=ink["primary"], lw=1.3, ls=(0, (4.0, 2.2)))
    kx.text(xt, y, r"path $\Sigma_t\,\mathrm{lr}_t = \|W_\ast\|_{op}$", ha="left", va="center", fontsize=small,
            color=ink["secondary"])
    s = D["spec"]
    es.save(fig, NAME, "talk", build=b,
            note=(rf"M9 tuning, least squares, $n = {s['n']}$, SAGA table $b = 1$, exact polar; path: necessary only; "
                  r"cliff unexplained"))


def main():
    D = data()
    print_all(D)
    draw(D)


if __name__ == "__main__":
    main()
