"""Figure T22 of the lecture, a backup slide (paper/talk/figures/figT22_grid_cliff): the fast steps of the nuclear-scaled
polar step of SAGA ("joint SAGA-Muon") end at a cliff. One notch of a 2^(1/4) grid below the selected step costs a
factor 2.7 to 6.3 in passes at b = 4 and 16, so a grid of factor 2 can report the slow side; every ratio between
methods must be given with its tuning grid. Explanatory figure, not in the paper. It explains the corrected findings
(M3) and (M6) of paper/paper_saga_to_muon.tex (findings C, D, E of paper/audit/AUDIT.md) and the last sentences of
(M6): "on grids of factor 2 it had appeared 1.7 to 2.7 times slower at b = 16, its fast steps ending at a cliff".

Data: muon/results/M5b_summary.json (experiment M5b, muon/findings.md, Section 5.12), data seed 0, cell
instances/<instance>/0/batches/<b>:
  (a) `window`/<family>/<j>, j = -2..2: passes to the relative squared distance 1e-6 at the step selected on the
      2^(1/4) grid times 2^(j/8), for the nuclear-scaled SAGA-Muon (orange; step gamma = c / R2_op) and B-SAGA (blue;
      gamma = c / R2 at its selected damping), each around ITS OWN selected step; softmax 8 x 40 (`soft-cond`),
      b = 1, 4, 16. Marker: median of the ten confirmation seeds; bar: their range; an arrow at the budget (300
      passes) with "0/10" when no seed reaches the target. Ring (state s2): the step that the same protocol selects
      for SAGA-Muon on the powers of two (`factor2`/rows/SAGA-Muon/best), when it falls in the window.
  (b) the ratio of medians SAGA-Muon / faster of SAGA and B-SAGA at the same batch against b = 1, 4, 16, on softmax
      8 x 40 and least squares 8 x 40 (`ls-cond`): every method tuned on 2^(1/4) grids (`fine`/ratio, filled) and on
      grids of factor 2 (`m5`/ratio: experiment M5, the numbers that "had appeared"; open). On least squares at
      b = 1 the factor-2 ratio is undefined in the summary (the selected SAGA-Muon point misses one seed of ten).

Checks (the script stops if one fails): the window point j = 0 is the selected point of the fine grid; the fine
ratios are recomputed from the medians; the control of M5b on the powers of two (`factor2`/ratio) is within 3 % of
M5's ratio; the window medians of panel (a) and the ratios of panel (b) equal the tables of muon/results/digest.md
(section M5b); the numbers of the paper are reproduced: (M6) 0.74 and 0.75 at b = 4 and 16 on softmax, 0.74 and 0.55
on least squares, 0.91 and 1.04 at batch one, 1.7 and 2.7 on factor-2 grids at b = 16; (M3) corrected: SAGA at batch
one needs 69.5 / 98.7 passes on factor-2 grids and 49.8 / 83.5 on 2^(1/4) grids.

Status. Measured (experiment): everything; one draw of the data, ten confirmation seeds; exact polar factor, no
momentum; passes do not count the factorization of each polar step. The step window exists for SAGA-Muon and B-SAGA
only; b = 64 is fragile (Section 5.12) and is not shown. "(B-)SAGA" is the faster of SAGA and B-SAGA, not Euclidean
methods in general.

Progressive build (states s1-s3): s1 panel (a); s2 adds the steps selected on a factor-2 grid; s3 adds panel (b).

The script prints every number it plots.
Run:    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/scripts/make_figT22_grid_cliff.py
Output: paper/talk/figures/figT22_grid_cliff.pdf and .png, and the states _s1 ... _s3
"""
from __future__ import annotations

import json
import math
import os
import sys

sys.dont_write_bytecode = True

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import explain_style as es  # noqa: E402

NAME = "figT22_grid_cliff"
RESULTS = os.path.join(es.ROOT, "muon", "results")
TOL = "1e-06"
A_INSTANCE = "soft-cond"
B_INSTANCES = (("soft-cond", "softmax"), ("ls-cond", "least squares"))
BATCHES = ("1", "4", "16")
OFFSETS = ("-2", "-1", "0", "1", "2")
FAMILIES = (("SAGA-Muon", "SAGA-Muon"), ("B-SAGA", "B-SAGA"))
# muon/results/digest.md, section "M5b": step window of soft-cond (medians; None: not reached) and ratios
DIGEST_WINDOW = {("1", "SAGA-Muon"): (None, 185.7, 24.6, 26.4, 29.0), ("1", "B-SAGA"): (30.2, 28.3, 27.1, 27.1, 27.6),
                 ("4", "SAGA-Muon"): (121.8, 22.0, 21.2, 22.8, 24.1), ("4", "B-SAGA"): (30.6, 29.1, 28.6, 29.2, 31.4),
                 ("16", "SAGA-Muon"): (94.0, 67.4, 23.2, 22.0, 24.2), ("16", "B-SAGA"): (30.6, 30.2, 31.0, 34.2, 38.6)}
DIGEST_RATIO = {  # (fine grid, M5); the fine values at b = 4, 16 and b = 1 and M5's at b = 16 are quoted in (M6)
    ("soft-cond", "1"): (0.91, 0.86), ("soft-cond", "4"): (0.74, 0.69), ("soft-cond", "16"): (0.75, 2.71),
    ("ls-cond", "1"): (1.04, None), ("ls-cond", "4"): (0.74, 2.04), ("ls-cond", "16"): (0.55, 1.74)}
PAPER_M6_B16 = {"soft-cond": "2.7", "ls-cond": "1.7"}          # "1.7 to 2.7 times slower at b = 16" on factor-2 grids
PAPER_M3_SAGA = {"soft-cond": (69.5, 49.8), "ls-cond": (98.7, 83.5)}   # SAGA, b = 1: factor-2 grid, 2^(1/4) grid


def load(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as f:
        return json.load(f)


def _close(a, b, tol=0.051):
    """Equality of two medians up to the rounding of a table (None stands for "not reached")."""
    if b is None:
        return a is None or not math.isfinite(a)
    return a is not None and math.isfinite(a) and abs(a - b) <= tol


# ----------------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------------
def data():
    m5b = load("M5b_summary.json")
    out = {"window": {}, "ring": {}, "ratio": {}, "saga_b1": {}}
    for inst, _ in B_INSTANCES:
        e = m5b["instances"][inst]["0"]
        if float(e["tol"]) != float(TOL):
            raise SystemExit(f"{inst}: the target is {e['tol']}, not {TOL}")
        out["budget"] = e["passes"]
        for b in BATCHES:
            c = e["batches"][b]
            rows = c["fine"]["rows"]
            med = {f: rows[f]["summary"][TOL]["median"] for f in ("SAGA", "B-SAGA", "SAGA-Muon")}
            best = min(("SAGA", "B-SAGA"), key=lambda f: med[f])
            fine = med["SAGA-Muon"] / med[best]
            if abs(fine - c["fine"]["ratio"]) > 1e-9 or best != c["fine"]["best_euclid"]:
                raise SystemExit(f"{inst}, b = {b}: fine ratio {fine} against {c['fine']['ratio']} in the summary")
            m5, ctrl = c["m5"]["ratio"], c["factor2"]["ratio"]
            if (m5 is None) != (ctrl is None) or (m5 is not None and abs(ctrl / m5 - 1.0) > 0.03):
                raise SystemExit(f"{inst}, b = {b}: factor-2 control {ctrl} against M5 {m5}")
            dg = DIGEST_RATIO[(inst, b)]
            if abs(fine - dg[0]) > 0.0051 or (m5 is None) != (dg[1] is None) or (m5 and abs(m5 - dg[1]) > 0.0051):
                raise SystemExit(f"{inst}, b = {b}: ratios {fine:.3f}, {m5} against {dg} in muon/results/digest.md")
            out["ratio"][(inst, b)] = {"fine": fine, "m5": m5, "control": ctrl, "best": best, "medians": med,
                                       "reached": {f: rows[f]["summary"][TOL]["reached"] for f in med}}
            w = c["window"]
            if abs(w["SAGA-Muon"]["0"]["median"] - med["SAGA-Muon"]) > 1e-9 \
                    or abs(w["B-SAGA"]["0"]["median"] - med["B-SAGA"]) > 1e-9:
                raise SystemExit(f"{inst}, b = {b}: the centre of the window is not the selected point")
            for fam, _ in FAMILIES:
                out["window"][(inst, b, fam)] = [w[fam][j] for j in OFFSETS]
            c2 = c["factor2"]["rows"]["SAGA-Muon"]["best"]["gamma"]["c"]
            hit = [k for k, j in enumerate(OFFSETS) if abs(w["SAGA-Muon"][j]["c"] / c2 - 1.0) < 1e-9]
            out["ring"][(inst, b)] = hit[0] if hit else None
            out["ring_c"] = out.get("ring_c", {})
            out["ring_c"][(inst, b)] = (c2, w["SAGA-Muon"]["0"]["c"])
        if out["ratio"][(inst, "16")]["m5"] is None or f"{out['ratio'][(inst, '16')]['m5']:.1f}" != PAPER_M6_B16[inst]:
            raise SystemExit(f"{inst}: the factor-2 ratio at b = 16 is not the {PAPER_M6_B16[inst]} of the paper")
        s2 = e["batches"]["1"]["factor2"]["rows"]["SAGA"]["summary"][TOL]["median"]
        s4 = e["batches"]["1"]["fine"]["rows"]["SAGA"]["summary"][TOL]["median"]
        if abs(s2 - PAPER_M3_SAGA[inst][0]) > 0.051 or abs(s4 - PAPER_M3_SAGA[inst][1]) > 0.051:
            raise SystemExit(f"{inst}: SAGA at batch one needs {s2} / {s4} passes, the paper says {PAPER_M3_SAGA[inst]}")
        out["saga_b1"][inst] = (s2, s4)
    for (b, fam), ref in DIGEST_WINDOW.items():
        got = [v["median"] for v in out["window"][(A_INSTANCE, b, fam)]]
        if not all(_close(g, r) for g, r in zip(got, ref)):
            raise SystemExit(f"window of {A_INSTANCE}, b = {b}, {fam}: {got} against {ref} in muon/results/digest.md")
    out["cliff"] = {(inst, b): out["window"][(inst, b, "SAGA-Muon")][0]["over_selected"]
                    for inst, _ in B_INSTANCES for b in ("4", "16")}
    return out


def print_all(D):
    print(f"(a) step window, passes to {TOL} at the selected step times 2^(j/8), j = -2..2: median [min, max] "
          "(seeds reaching / seeds)")
    for inst, name in B_INSTANCES:
        print(f"  {inst} ({name})" + ("" if inst == A_INSTANCE else "  [printed, not drawn]"))
        for b in BATCHES:
            for fam, _ in FAMILIES:
                cells = "; ".join(f"{v['median']:.1f} [{v['min']:.1f}, {v['max']:.1f}] ({v['reached']}/{v['runs']})"
                                  for v in D["window"][(inst, b, fam)])
                print(f"    b = {b:>2s} {fam:9s}: {cells}")
            k = D["ring"][(inst, b)]
            c2, c4 = D["ring_c"][(inst, b)]
            where = f"offset j = {OFFSETS[k]} of the window" if k is not None else "outside the window"
            print(f"           SAGA-Muon step selected on the powers of two: c = {c2:g} = 2^({8 * math.log2(c2 / c4):+.0f}/8) "
                  f"x the step selected on the 2^(1/4) grid (c = {c4:.4g}): {where}")
    print("    passes one notch (2^(-1/4)) below the selected step over passes at the selected step, SAGA-Muon: "
          + ", ".join(f"{inst} b = {b}: {v:.2f}" for (inst, b), v in D["cliff"].items())
          + f"  (range {min(D['cliff'].values()):.1f} to {max(D['cliff'].values()):.1f})")
    print("(b) ratio SAGA-Muon / faster of SAGA and B-SAGA (medians, same batch): 2^(1/4) grids; factor-2 grids (M5); "
          "factor-2 control of M5b")
    for inst, name in B_INSTANCES:
        for b in BATCHES:
            r = D["ratio"][(inst, b)]
            f = lambda v: "undefined" if v is None else f"{v:.3f}"                    # noqa: E731
            m = r["medians"]
            print(f"  {inst:9s} b = {b:>2s}: {r['fine']:.3f}; {f(r['m5'])}; {f(r['control'])}   (2^(1/4) grids: SAGA "
                  f"{m['SAGA']:.1f}, B-SAGA {m['B-SAGA']:.1f}, SAGA-Muon {m['SAGA-Muon']:.1f}; faster: {r['best']})")
    print("(M3) corrected, SAGA at batch one, passes on the factor-2 grid / on the 2^(1/4) grid: "
          + "; ".join(f"{inst} {a:.2f} / {c:.2f}" for inst, (a, c) in D["saga_b1"].items()))


# ----------------------------------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------------------------------
def panel_window(ax, D, b, ps, build, first, labels):
    ink = ps.INK
    colour = {"SAGA-Muon": ps.ORANGE, "B-SAGA": ps.BLUE}
    marker = {"SAGA-Muon": "o", "B-SAGA": "s"}
    x = [int(j) for j in OFFSETS]
    budget = D["budget"]
    note = dict(textcoords="offset points", fontsize=es.small(), color=ink["secondary"])
    for fam, _ in reversed(FAMILIES):
        w, col = D["window"][(A_INSTANCE, b, fam)], colour[fam]
        ys = [v["median"] if math.isfinite(v["median"]) else budget for v in w]
        ax.plot(x, ys, color=col, lw=1.5, marker=marker[fam], ms=5.6 if fam == "SAGA-Muon" else 5.0, mfc=col,
                mec=ink["surface"], mew=0.9, zorder=4 if fam == "SAGA-Muon" else 3)
        for xx, yy, v in zip(x, ys, w):
            if v["reached"] and v["max"] > v["min"]:
                ax.plot([xx, xx], [v["min"], v["max"]], color=col, lw=1.0, zorder=2, solid_capstyle="butt")
            if not math.isfinite(v["median"]):
                ax.annotate("", (xx, yy * 1.45), xytext=(xx, yy * 1.04), arrowprops=dict(
                    arrowstyle="-|>", color=col, lw=1.1, mutation_scale=8, shrinkA=0, shrinkB=0))
                ax.annotate(f"{v['reached']}/{v['runs']}", (xx, yy), xytext=(7, 0), ha="left", va="center", **note)
            elif v["reached"] < v["runs"]:
                ax.annotate(f"{v['reached']}/{v['runs']}", (xx, yy), xytext=(0, 7), ha="center", va="bottom", **note)
    k = D["ring"][(A_INSTANCE, b)]
    if k is not None:                                      # the step selected on the powers of two
        v = D["window"][(A_INSTANCE, b, "SAGA-Muon")][k]
        build.add(2, ax.plot([x[k]], [v["median"]], ls="", marker="o", ms=11.0, mfc="none", mec=ink["primary"],
                             mew=1.1, zorder=6, clip_on=False))
    ax.set_yscale("log")
    ax.set_ylim(12.5, 470)
    ax.set_yticks([20, 30, 50, 100, 200, budget])
    ax.set_yticklabels(["20", "30", "50", "100", "200", str(budget)])
    ax.minorticks_off()
    ax.set_xlim(-2.75, 2.75)
    ax.set_xticks([-2, 0, 2])
    ax.set_xticklabels([f"{2.0 ** -0.25:.2f}", "1", f"{2.0 ** 0.25:.2f}"], fontsize=es.small())
    ax.set_xticks([-1, 1], minor=True)
    ax.grid(axis="x", visible=False)
    if first:
        ax.set_ylabel(r"passes to $10^{-6}$")
        es.panel_title(ax, rf"(a) $b = {b}$")
    else:
        ax.tick_params(labelleft=False)
        es.panel_title(ax, rf"$b = {b}$")
    cliff = D["cliff"].get((A_INSTANCE, b))
    if cliff is not None:                                  # passes one notch below over passes at the selected step
        ax.annotate(rf"$\times${cliff:.1f}", (x[0], D["window"][(A_INSTANCE, b, "SAGA-Muon")][0]["median"]),
                    xytext=(9, 3), ha="left", va="bottom", **note)
    if labels:                                             # direct labels where the two curves are apart
        wm, wb = D["window"][(A_INSTANCE, b, "SAGA-Muon")], D["window"][(A_INSTANCE, b, "B-SAGA")]
        ax.annotate("B-SAGA", (2, max(v["max"] for v in wb[2:])), xytext=(5, 5), ha="right", va="bottom", **note)
        ax.annotate("SAGA-Muon", (2, min(v["min"] for v in wm[1:])), xytext=(5, -5), ha="right", va="top", **note)


def panel_ratio(ax, D, ps, build):
    ink = ps.INK
    marker = {"soft-cond": "o", "ls-cond": "o"}       # circles for both groups: a square is B-SAGA in panel (a)
    gap, pos, ticks, labels, x = 0.8, {}, [], [], 0.0
    for inst, _ in B_INSTANCES:
        for b in BATCHES:
            pos[(inst, b)] = x
            ticks.append(x)
            labels.append(b)
            x += 1.0
        x += gap
    ax.axhline(1.0, color=ink["primary"], lw=1.0, zorder=1)
    for gi, (inst, name) in enumerate(B_INSTANCES):
        xs = [pos[(inst, b)] for b in BATCHES]
        fine = [D["ratio"][(inst, b)]["fine"] for b in BATCHES]
        pts = [(xx, D["ratio"][(inst, b)]["m5"]) for xx, b in zip(xs, BATCHES) if D["ratio"][(inst, b)]["m5"] is not None]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=ps.ORANGE, lw=1.3, ls=(0, (3.0, 1.6)),
                marker=marker[inst], ms=5.6, mfc=ink["surface"], mec=ps.ORANGE, mew=1.3, zorder=3,
                label="factor-2 grids" if gi == 0 else None)
        ax.plot(xs, fine, color=ps.ORANGE, lw=1.5, marker=marker[inst], ms=6.0, mfc=ps.ORANGE, mec=ink["surface"],
                mew=0.9, zorder=4, label=r"$2^{1/4}$ grids" if gi == 0 else None)
        note = dict(textcoords="offset points", fontsize=es.small(), color=ink["secondary"], ha="left", va="center")
        ax.annotate(f"{pts[-1][1]:.1f}", pts[-1], xytext=(6, 0), **note)
        ax.annotate(f"{fine[-1]:.2f}", (xs[-1], fine[-1]), xytext=(6, 0), **note)
        mid = 0.5 * (xs[0] + xs[-1])
        ax.annotate(name, (mid, 0), xycoords=("data", "axes fraction"), xytext=(0, -20), textcoords="offset points",
                    ha="center", va="top", fontsize=es.small(), color=ink["secondary"])
        if gi:
            ax.axvline(xs[0] - 0.5 - 0.5 * gap, color=ink["grid"], lw=0.8, zorder=0)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.6, x - gap + 0.25)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("batch size $b$", labelpad=14)
    ax.set_yscale("log")
    ax.set_ylim(0.215, 3.6)
    yt = [0.5, 0.7, 1.0, 2.0, 3.0]
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{v:g}" for v in yt])
    ax.minorticks_off()
    ax.set_ylabel("SAGA-Muon / (B-)SAGA")
    es.panel_title(ax, "(b) ratio of passes")
    build.legend(ax, loc="lower left", title="steps tuned on", title_fontsize=es.small(), alignment="left",
                 handlelength=2.1, handletextpad=0.5, borderaxespad=0.15, labelspacing=0.25)


def draw(D):
    ps = es.setup("talk")
    fig = es.figure("talk", height=2.75)
    fig.get_layout_engine().set(w_pad=0.025, wspace=0.0)
    gs = fig.add_gridspec(1, 4, width_ratios=(1.0, 1.0, 1.0, 1.6))
    b = es.Build()
    axes = []
    for k, bs in enumerate(BATCHES):
        ax = fig.add_subplot(gs[0, k], sharey=axes[0] if axes else None)
        panel_window(ax, D, bs, ps, b, first=k == 0, labels=bs == "4")
        axes.append(ax)
    axes[1].set_xlabel(r"step / step selected on the $2^{1/4}$ grid")
    if any(D["ring"][(A_INSTANCE, bs)] is not None for bs in BATCHES):
        ring = axes[1].annotate("ring: the step selected on a factor-2 grid", (0.5, 0.0), xycoords="axes fraction",
                                xytext=(0, -34), textcoords="offset points", ha="center", va="top",
                                fontsize=es.small(), color=ps.INK["secondary"])
        ring.set_in_layout(False)              # wider than its panel; panel (b) reserves the room below the axes
        b.add(2, ring)
    ax_b = fig.add_subplot(gs[0, 3])
    panel_ratio(ax_b, D, ps, b)
    b.add(3, ax_b)
    es.save(fig, NAME, "talk", build=b,
            note=(r"SAGA-Muon: nuclear-scaled polar step; (a) softmax 8 $\times$ 40; one data draw; median, range of "
                  "10 seeds"))


def main():
    D = data()
    print_all(D)
    draw(D)


if __name__ == "__main__":
    main()
