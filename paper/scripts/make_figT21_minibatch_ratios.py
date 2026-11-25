"""Figure T21 of the lecture (paper/talk/figures/figT21_minibatch_ratios): where the spectral map pays. With batches of
16 or 32 the fastest spectral variant of the recipe (4) of paper/paper_saga_to_muon.tex needs fewer passes than the
faster of SAGA and B-SAGA at the same batch (0.41-0.89) and than the fastest normalized or sign control on the stored
grid of cooldowns (0.58-0.93, a range resolved to about 20 %); at batch one it does not (0.92-1.06 of the control).
Explanatory figure, not in the paper. It explains finding (M6) and Figure 4(b), which shows two instances; this one
shows the four conditioning-limited instances cell by cell.

One panel, a dot plot. Rows: the four conditioning-limited instances of experiment M9, part A (`ls-cond` least squares
8 x 40, `soft-cond` softmax 8 x 40, `soft-sq` softmax 20 x 20, `ls-orth` least squares 20 x 20 with a planted matrix
of flat spectrum), each at b = 1, 16, 32. Horizontal axis (log): ratio of medians of passes to the relative squared
distance 1e-6 over the ten confirmation seeds,
    fastest spectral variant (exact polar factor or ns5, with or without Nesterov momentum 0.9)
    over a comparator, coloured by the comparator:
      grey diamond   the fastest control map on the same grid of cooldowns (Frobenius normalization or coordinatewise
                     sign of the SAGA estimator, with or without momentum; b = 1, 16, 32);
      blue square    the faster of SAGA and B-SAGA with constant steps on 2^(1/4) grids at the same batch (b = 16, 32;
                     part A did not run them at batch one, and the paper's statement at batch one is about the control).
A vertical line at 1; a band from 0.8 to 1.25 (within a factor 1.25 of 1 on either side; the axis stops at 1.13): the
recipes are tuned on q in steps of 0.1 (0.05 above 0.9) and on half-decades of lr_0, which resolves ratios to about
20 % (audit, finding E), so a ratio inside the band does not separate the two maps, whichever side of 1 it lies on. Open diamonds: the ratio to the control
when both maps are re-tuned on a four times finer grid, done by the audit of the paper for two cells (0.58 -> 0.69,
0.84 -> 0.93; paper/audit/AUDIT.md, finding D, and paper/audit/evidence/num-m6m7.verif.json; these two numbers are not
in the summaries of the repository: the script only checks that AUDIT.md states them). "k/10" next to a row: seeds of
the spectral variant that reach the target when some miss.

Data: muon/results/M9_summary.json, A/<instance>/cells/<b>/compare (`spectral`, `control`, `euclid`, `ratios`).
Checks (the script stops if one fails): the ratios are recomputed from the medians and compared with `ratios`; the
three ranges of the paper are reproduced to two decimals (0.58-0.93 and 0.41-0.89 at b = 16, 32; 0.92-1.06 at b = 1);
the eight ratios to the control at b = 16, 32 equal the table of muon/results/digest.md (section M9, part A).

Status. Measured (experiment): everything; one draw of the data per instance, ten confirmation seeds, each map at its
own tuned (q, lr_0), each constant-step method at its own tuned step. The comparators are (B-)SAGA and normalized or
sign steps, not "Euclidean methods" in general: the covariance metric, the best method at batch one on least squares,
was not run with minibatches. Passes do not count the factorization of each polar step.

Progressive build (states s1-s3): s1 the eight cells with b = 16, 32; s2 adds the band of the resolution and the two
cells re-tuned by the audit; s3 adds batch one.

The script prints every number it plots.
Run:    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/scripts/make_figT21_minibatch_ratios.py
Output: paper/talk/figures/figT21_minibatch_ratios.pdf and .png, and the states _s1 ... _s3
"""
from __future__ import annotations

import json
import os
import sys

sys.dont_write_bytecode = True

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import explain_style as es  # noqa: E402

NAME = "figT21_minibatch_ratios"
RESULTS = os.path.join(es.ROOT, "muon", "results")
AUDIT_MD = os.path.join(es.ROOT, "paper", "audit", "AUDIT.md")
TOL = "1e-06"
INSTANCES = (("ls-cond", "least squares\n" + r"8 $\times$ 40"), ("soft-cond", "softmax\n" + r"8 $\times$ 40"),
             ("soft-sq", "softmax\n" + r"20 $\times$ 20"), ("ls-orth", "least squares\n" + r"20 $\times$ 20"))
SHAPES = {"ls-cond": ("ls", 200, 8, 40), "soft-cond": ("softmax", 200, 8, 40), "soft-sq": ("softmax", 500, 20, 20),
          "ls-orth": ("ls", 500, 20, 20)}                     # loss, n, K, d: checked against the stored specifications
BATCHES = ("1", "16", "32")
BAND = (0.8, 1.25)                                             # ratios within about 20 % of 1, on both sides (log axis)
PAPER = {"control": ("0.58", "0.93"), "euclid": ("0.41", "0.89"), "control_b1": ("0.92", "1.06")}   # ranges of (M6)
# muon/results/digest.md, section "M9 part A", row "fastest spectral / fastest control" (b = 16, 32)
DIGEST = {("soft-cond", "16"): 0.80, ("soft-cond", "32"): 0.84, ("ls-cond", "16"): 0.71, ("ls-cond", "32"): 0.58,
          ("ls-orth", "16"): 0.93, ("ls-orth", "32"): 0.89, ("soft-sq", "16"): 0.78, ("soft-sq", "32"): 0.83}
# the two cells re-tuned on a four times finer grid by the audit (AUDIT.md, finding D): stored ratio -> refined ratio
AUDIT = {("ls-cond", "32"): ("0,58 → 0,69", 0.69), ("soft-cond", "32"): ("0,84 → 0,93", 0.93)}


def load(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------------
def data():
    m9 = load("M9_summary.json")
    audit_text = None                                    # without paper/audit/ the two refined cells are not drawn
    if os.path.exists(AUDIT_MD):
        with open(AUDIT_MD, encoding="utf-8") as f:
            audit_text = f.read()
    else:
        print("   (paper/audit/AUDIT.md is absent: the two cells re-tuned by the audit are not drawn)")
    out = {}
    for inst, _ in INSTANCES:
        e = m9["A"][inst]
        s = e["spec"]
        if (s["loss"], s["n"], s["K"], s["d"]) != SHAPES[inst] or float(e["tol"]) != float(TOL):
            raise SystemExit(f"{inst}: unexpected specification {s} or target {e['tol']}")
        for b in BATCHES:
            c = e["cells"][b]["compare"]
            row = {"spectral": c["spectral"], "control": c["control"], "euclid": c.get("euclid")}
            for key in ("control", "euclid"):
                if row[key] is None:
                    if b != "1" or key != "euclid":
                        raise SystemExit(f"{inst}, b = {b}: no {key} comparator in the summary")
                    continue
                if row[key]["reached"] < row[key]["runs"]:
                    raise SystemExit(f"{inst}, b = {b}: the {key} comparator misses seeds; the figure does not mark it")
                r = row["spectral"]["median"] / row[key]["median"]
                if abs(r - c["ratios"][f"spectral/{key}"]) > 1e-9:
                    raise SystemExit(f"{inst}, b = {b}: ratio to {key} {r} against {c['ratios']} in the summary")
                row[f"r_{key}"] = r
            if b != "1" and abs(row["r_control"] - DIGEST[(inst, b)]) > 0.0051:
                raise SystemExit(f"{inst}, b = {b}: ratio to the control {row['r_control']:.3f} against "
                                 f"{DIGEST[(inst, b)]} in muon/results/digest.md")
            if (inst, b) in AUDIT and audit_text is not None:
                text, value = AUDIT[(inst, b)]
                if text not in audit_text or f"{row['r_control']:.2f}".replace(".", ",") != text[:4]:
                    raise SystemExit(f"{inst}, b = {b}: paper/audit/AUDIT.md does not state '{text}'")
                row["r_audit"] = value
            out[(inst, b)] = row
    rng = lambda key, bs: tuple(f"{f(out[(i, b)][key] for i, _ in INSTANCES for b in bs):.2f}"   # noqa: E731
                                for f in (min, max))
    out["ranges"] = {"control": rng("r_control", ("16", "32")), "euclid": rng("r_euclid", ("16", "32")),
                     "control_b1": rng("r_control", ("1",))}
    for key, ref in PAPER.items():
        if out["ranges"][key] != ref:
            raise SystemExit(f"range {key}: {out['ranges'][key]} from the summary, {ref} in the paper")
    inside = lambda r: BAND[0] <= r <= BAND[1]                                             # noqa: E731
    out["in_band"] = sum(inside(out[(i, b)]["r_control"]) for i, _ in INSTANCES for b in ("16", "32"))
    out["in_band_b1"] = sum(inside(out[(i, "1")]["r_control"]) for i, _ in INSTANCES)
    return out


def print_all(D):
    print(f"ratios of medians of passes to {TOL} (10 confirmation seeds): fastest spectral variant / comparator")
    for inst, name in INSTANCES:
        loss, n, K, d = SHAPES[inst]
        print(f"  {inst} ({loss}, n = {n}, K x d = {K} x {d})")
        for b in BATCHES:
            r = D[(inst, b)]
            sp, ct, eu = r["spectral"], r["control"], r["euclid"]
            txt = (f"    b = {b:>2s}: spectral {sp['variant']} {sp['median']:.2f} ({sp['reached']}/{sp['runs']}); "
                   f"control {ct['variant']} {ct['median']:.2f} ({ct['reached']}/{ct['runs']}), ratio {r['r_control']:.3f}")
            if eu is not None:
                txt += (f"; (B-)SAGA: {eu['variant']} {eu['median']:.2f} ({eu['reached']}/{eu['runs']}), "
                        f"ratio {r['r_euclid']:.3f}")
            if "r_audit" in r:
                txt += f"; audit, finer grid: ratio to the control {r['r_audit']:.2f}"
            print(txt)
    g = D["ranges"]
    print(f"ranges (= the paper, (M6)): to the control at b = 16, 32: {g['control'][0]}-{g['control'][1]}; to the faster "
          f"of SAGA and B-SAGA: {g['euclid'][0]}-{g['euclid'][1]}; to the control at b = 1: {g['control_b1'][0]}-"
          f"{g['control_b1'][1]}")
    print(f"ratios to the control inside the band [{BAND[0]}, {BAND[1]}] (drawn up to the edge of the axis): "
          f"{D['in_band']} of 8 at b = 16, 32; {D['in_band_b1']} of 4 at b = 1")


# ----------------------------------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------------------------------
def draw(D):
    ps = es.setup("talk")
    ink = ps.INK
    grey, blue = ink["secondary"], ps.BLUE                  # the control maps; SAGA and B-SAGA
    fig = es.figure("talk", height=2.75)
    ax = fig.add_subplot(111)
    b = es.Build()

    # rows: one block of three batch sizes per instance, top to bottom
    gap, y, ypos, ticks, labels = 0.7, 0.0, {}, [], []
    for inst, name in INSTANCES:
        for bs in BATCHES:
            ypos[(inst, bs)] = y
            ticks.append(y)
            labels.append(rf"$b = {bs}$")
            y += 1.0
        ax.annotate(name, (0, y - 2.0), xycoords=("axes fraction", "data"), xytext=(-46, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=es.small(), color=ink["secondary"],
                    linespacing=1.2)
        y += gap
    y_lo, y_hi = y - gap - 0.35, -0.65
    ax.set_ylim(y_lo, y_hi)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=es.small())
    ax.tick_params(axis="y", length=0)
    ax.axvline(1.0, color=ink["primary"], lw=1.0, zorder=2)
    ax.annotate(r"$\leftarrow$ the spectral variant needs fewer passes", (0.012, ypos[(INSTANCES[0][0], "1")]),
                xycoords=("axes fraction", "data"), ha="left", va="center", fontsize=es.small(),
                color=ink["secondary"])

    # the ratios
    kw_c = dict(ls="", marker="D", ms=6.0, mfc=grey, mec=ink["surface"], mew=1.0, zorder=5)
    kw_e = dict(ls="", marker="s", ms=6.4, mfc=blue, mec=ink["surface"], mew=1.0, zorder=4)
    note = dict(textcoords="offset points", fontsize=es.small(), color=ink["secondary"], va="center")
    for bs in BATCHES:
        state = 3 if bs == "1" else 1
        first = bs == "16"
        for inst, _ in INSTANCES:
            r, yy = D[(inst, bs)], ypos[(inst, bs)]
            b.add(state, ax.plot([r["r_control"]], [yy], label="over the fastest control\n(normalized or sign step)"
                                 if first and inst == INSTANCES[0][0] else None, **kw_c))
            if "r_euclid" in r:
                b.add(state, ax.plot([r["r_euclid"]], [yy], label="over the faster of\nSAGA and B-SAGA"
                                     if first and inst == INSTANCES[0][0] else None, **kw_e))
            if r["spectral"]["reached"] < r["spectral"]["runs"]:
                b.add(state, ax.annotate(f"{r['spectral']['reached']}/{r['spectral']['runs']} seeds", (r["r_control"], yy),
                                         xytext=(-8, 0), ha="right", **note))
            if "r_audit" in r:
                b.add(2, ax.annotate("", (r["r_audit"], yy), xytext=(r["r_control"], yy), zorder=3, arrowprops=dict(
                    arrowstyle="-|>", color=grey, lw=1.1, mutation_scale=8, shrinkA=5, shrinkB=5)))
                b.add(2, ax.plot([r["r_audit"]], [yy], ls="", marker="D", ms=5.6, mfc=ink["surface"], mec=grey,
                                 mew=1.2, zorder=5, label="the same, both maps re-tuned on\na 4 times finer grid "
                                 "(2 cells)" if (inst, bs) == next(iter(AUDIT)) else None))

    # the band of the resolution (drawn last: last entry of the legend)
    b.add(2, ax.axvspan(*BAND, color=es.tint(ink["muted"], 0.22), lw=0, zorder=0,
                        label="within 20 % of 1: not resolved\nby the grid of cooldowns"))

    ax.set_xscale("log")
    ax.set_xlim(0.37, 1.13)
    xt = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{v:g}" for v in xt])
    ax.minorticks_off()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(r"passes to $10^{-6}$: fastest spectral variant / comparator")
    b.legend(ax, loc="upper left", bbox_to_anchor=(0.0, 0.655), handlelength=1.2, handletextpad=0.6,
             borderaxespad=0.3, labelspacing=0.55)             # anchored at its top: the entries do not move in the build
    es.save(fig, NAME, "talk", build=b,
            note="one data draw per instance, 10 seeds; each method at its own tuned point; factorizations not counted")


def main():
    D = data()
    print_all(D)
    draw(D)


if __name__ == "__main__":
    main()
