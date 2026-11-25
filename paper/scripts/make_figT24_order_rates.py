"""Figure T24 of the lecture (paper/talk/figures/figT24_order_rates): what the sampling order changes, in the linear
model of SAGA and in the fixed-length recipe of the paper. Explanatory figure, not in the paper; shown after figure
T24b. It explains item (M7) of paper/paper_saga_to_muon.tex as corrected by the audit (paper/audit/AUDIT.md, finding
J), the row "Memory" of Table 2 and Figure 4(c).

Panels:
  (a) linear SAGA on n = 1000 quadratics with ONE common Hessian (vrlab/order_model.py; one mode, g = gamma lambda,
      c = n g = gamma n lambda): nats lost per pass by the squared error, against c from 1/4 to 32.
        - independent sampling, expected squared error: order_model.iid_exponent(n, c), exact at this n (4 x 4
          recursion of the second moments); its limit n -> infinity is min(2c, 1) [proved];
        - a fresh permutation per pass, expected squared error: order_model.rr_ms_exponent(n, c), exact at this n
          (4 x 4 matrix of the second moments over one pass); its limit at fixed c is order_model.rr_exponent(c),
          which tends to 2 as c grows [proved];
        - independent sampling, single runs: median (marker) and range (bar) over 9 seeds of the exponent of one
          simulated run, fitted over the passes 30-150: results/B4_summary.json
          ["model"]["by_c"][c]["n=1000"]["iid_typical"] (median, min, max) [measured].
      "Two nats against one" compares the two EXPECTATIONS; the single independent runs sit above their expectation
      (1.3 to 1.6 nats for c >= 1) and above the permutation for c <= 2.
  (b) the fixed-length recipe (4) of the paper with a SAGA table, exact polar factor, no momentum (experiment M9 part
      D and its copy M9b on a sum with identical Hessians, the "model" of Figure 4(c), labelled "common Hessian"
      here because panel (a) is about another model of the same sum; n = 1000 everywhere): passes to the relative
      squared distance 1e-9 at the fastest cooldown that every seed follows to the target (median over 5 seeds, bar:
      range of the seeds), by batch size b = 1, 5, 20, 50, with independent batches (open markers) and with a fresh
      permutation per pass (filled markers). Read as paper/scripts/make_figP4_recipe.py (data_d) does:
      muon/results/M9_summary.json ["D"][instance]["cells"][b]["schemes"][scheme]["fastest"] and "rho_star",
      muon/results/M9b_summary.json ["D"]["cells"][b]["schemes"][scheme] [measured]. The bands mark b = 20 and 50,
      where a permutation saves 36-49 % of the passes on the two memory-limited instances while the sum with
      identical Hessians gains nothing (its fastest factor of lr^2 per pass followed by four of five seeds, rho_star,
      is 0.22 both ways, and its median passes go from 8.7 to 9.4 and from 7.0 to 9.1).
  No arrow goes from (a) to (b): (a) is about the expected squared error of linear SAGA with a constant step, (b)
  about polar steps of fixed length; the paper says that the first explains part of the second, and that why the
  instances lose more than the model under independent sampling is not established.

Colours follow the sampling order in both panels (independent: blue, fresh permutation: orange), as in figures T23
and T24b; in Figure 4(c) of the paper the same data are coloured by instance.

Checks (the script stops if one fails): the two exact curves against the values stored in results/B4_summary.json and
against results/digest.md (c = 1, 2); their distance to the limits n -> infinity (printed); the passes of the
instances at b = 20, 50 against the paper (16.7-20.0 and 9.1-12.2 passes, 36-49 %), one cell against
muon/results/digest.md, rho_star of the model at b = 20, 50.

Progressive build (states s1-s3): s1 panel (a), the two exact curves; s2 the single runs; s3 panel (b).

The script prints every number it plots.
Run:    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/scripts/make_figT24_order_rates.py
Output: paper/talk/figures/figT24_order_rates.pdf and .png, and the states _s1 ... _s3
"""
from __future__ import annotations

import json
import os
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import explain_style as es  # noqa: E402

sys.path.insert(0, es.ROOT)
from vrlab import order_model  # noqa: E402

NAME = "figT24_order_rates"
N = 1000
C_LO, C_HI, C_POINTS = 0.25, 32.0, 113                      # panel (a): grid of c (16 points per octave)
TOL_IID_LIMIT, TOL_RR_LIMIT = 0.03, 0.005                   # accepted distance to the limits n -> infinity at n = 1000
# results/digest.md, "B4 (M)": rows c = 1 and c = 2, n = 1000 (E e^2 exact i.i.d.; typical run median [min, max];
# reshuffled E e^2 exact)
DIGEST_A = {"1": (0.9995, (1.611, 1.525, 1.642), 1.188), "2": (0.9992, (1.579, 1.504, 1.68), 1.538)}
INSTANCES = (("soft-mem", "softmax"), ("ls-mem", "least\nsquares"), ("model", "common\nHessian"))
BATCHES = ("1", "5", "20", "50")
SCHEMES = (("iid", "independent batches"), ("reshuffle", "fresh permutation"))
EMPHASIS = ("20", "50")
PAPER_IID, PAPER_RR, PAPER_SAVED = (16.7, 20.0), (9.1, 12.2), (36.0, 49.0)   # (M7): instances at b = 20, 50
DIGEST_D = ("soft-mem", "20", "reshuffle", 0.1, 9.1)        # muon/results/digest.md, M9 part D: "0.1: 9.1 [9.1, 9.1]"
MODEL_RHO_STAR = 0.22                                        # (M7), corrected: the model at b = 20, 50, both orders


def load(*parts):
    with open(os.path.join(es.ROOT, *parts), encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------------
def data_a():
    summ = load("results", "B4_summary.json")["model"]
    c = np.exp(np.linspace(np.log(C_LO), np.log(C_HI), C_POINTS))
    iid = np.array([order_model.iid_exponent(N, x) for x in c])
    rr = np.array([order_model.rr_ms_exponent(N, x) for x in c])
    iid_lim = np.array([order_model.iid_exponent_limit(x) for x in c])
    rr_lim = np.array([order_model.rr_exponent(x) for x in c])
    dev_iid, dev_rr = np.abs(iid - iid_lim), np.abs(rr - rr_lim)
    if dev_iid.max() > TOL_IID_LIMIT or dev_rr.max() > TOL_RR_LIMIT:
        raise SystemExit(f"exact exponents far from their limits: {dev_iid.max():.4f}, {dev_rr.max():.4f}")
    runs, stored = [], 0.0
    for key, e in summ["by_c"].items():
        x = float(key)
        if not C_LO <= x <= C_HI:
            continue
        cell = e[f"n={N}"]
        t = cell["iid_typical"]
        if int(t["n"]) != int(summ["n_seeds"]):
            raise SystemExit(f"c = {key}: {t['n']} seeds in iid_typical")
        runs.append((x, t["median"], t["min"], t["max"], order_model.iid_exponent(N, x),
                     order_model.rr_ms_exponent(N, x)))
        stored = max(stored, abs(runs[-1][4] - cell["iid_expectation"]), abs(runs[-1][5] - cell["rr_mean_square"]))
        if key in DIGEST_A:
            d_iid, d_typ, d_rr = DIGEST_A[key]
            ok = (abs(runs[-1][4] - d_iid) < 5e-5 and abs(runs[-1][5] - d_rr) < 5e-4
                  and all(abs(a - b) < 6e-4 for a, b in zip((t["median"], t["min"], t["max"]), d_typ)))
            if not ok:
                raise SystemExit(f"c = {key}: the exponents differ from results/digest.md")
    if stored > 1e-9:
        raise SystemExit(f"exact exponents differ from results/B4_summary.json by {stored:.1e}")
    runs = np.array(sorted(runs))
    return {"c": c, "iid": iid, "rr": rr, "dev_iid": dev_iid, "dev_rr": dev_rr, "runs": runs, "stored": stored,
            "seeds": int(summ["n_seeds"]), "window": list(summ["window_passes"])}


def data_b():
    """Per instance (the model: M9b), batch size and scheme: passes to 1e-9 at the fastest cooldown, and rho_star
    (as data_d of make_figP4_recipe.py)."""
    m9, m9b = load("muon", "results", "M9_summary.json"), load("muon", "results", "M9b_summary.json")
    out = {}
    for inst, _ in INSTANCES:
        block = m9b["D"] if inst == "model" else m9["D"][inst]
        if str(block["tol"]) != "1e-09" or int(block["spec"]["n"]) != N:
            raise SystemExit(f"{inst}: unexpected target or n")
        for b in BATCHES:
            if len(block["cells"][b]["seeds"]) != 5:
                raise SystemExit(f"{inst}, b = {b}: not five seeds")
            for scheme, _ in SCHEMES:
                sc = block["cells"][b]["schemes"][scheme]
                out[(inst, b, scheme)] = {"fastest": sc["fastest"], "rho_star": sc["rho_star"]}
    real = [i for i, _ in INSTANCES if i != "model"]
    iid = [out[(i, b, "iid")]["fastest"]["median"] for i in real for b in EMPHASIS]
    rr = [out[(i, b, "reshuffle")]["fastest"]["median"] for i in real for b in EMPHASIS]
    saved = [100.0 * (1.0 - r / s) for r, s in zip(rr, iid)]
    if (min(iid), max(iid)) != PAPER_IID or (min(rr), max(rr)) != PAPER_RR:
        raise SystemExit(f"instances at b = 20, 50: {min(iid)}-{max(iid)} and {min(rr)}-{max(rr)} passes, not the "
                         f"values of (M7)")
    if round(min(saved)) != PAPER_SAVED[0] or round(max(saved)) != PAPER_SAVED[1]:
        raise SystemExit(f"passes saved at b = 20, 50: {min(saved):.1f}-{max(saved):.1f} %, not 36-49 %")
    inst, b, scheme, rho, med = DIGEST_D
    f = out[(inst, b, scheme)]["fastest"]
    if f["rho"] != rho or f["median"] != med:
        raise SystemExit("M9 part D differs from muon/results/digest.md")
    for b in EMPHASIS:
        for scheme, _ in SCHEMES:
            if out[("model", b, scheme)]["rho_star"] != MODEL_RHO_STAR:
                raise SystemExit(f"model, b = {b}, {scheme}: rho_star is not {MODEL_RHO_STAR}")
    return {"cells": out, "saved": saved, "iid_range": (min(iid), max(iid)), "rr_range": (min(rr), max(rr))}


def print_all(A, B):
    print(f"(a) linear SAGA, identical Hessians, n = {N}: exponents per pass of the squared error (nats)")
    sel = np.arange(0, C_POINTS, 16)
    print("    c                               : " + ", ".join(f"{A['c'][k]:7.3g}" for k in sel))
    print("    independent, E e^2 (exact)      : " + ", ".join(f"{A['iid'][k]:7.4f}" for k in sel))
    print("    fresh permutation, E e^2 (exact): " + ", ".join(f"{A['rr'][k]:7.4f}" for k in sel))
    print(f"    independent, single runs, {A['seeds']} seeds, fit over the passes {A['window'][0]}-{A['window'][1]} "
          f"(B4_summary.json): median [min, max]")
    for x, med, lo, hi, e_iid, e_rr in A["runs"]:
        print(f"      c = {x:5.2f}: {med:.3f} [{lo:.3f}, {hi:.3f}]   (expectations: independent {e_iid:.4f}, "
              f"permutation {e_rr:.4f})")
    k_i, k_r = int(A["dev_iid"].argmax()), int(A["dev_rr"].argmax())
    print(f"    checks: |exact at n = {N} - limit|: independent, min(2c, 1): at most {A['dev_iid'].max():.4f} (at c = "
          f"{A['c'][k_i]:.3g}); permutation, rr_exponent(c): at most {A['dev_rr'].max():.5f} (at c = {A['c'][k_r]:.3g}); "
          f"largest permutation exponent on the grid {A['rr'].max():.4f}; against B4_summary.json: {A['stored']:.1e}")
    print("(b) passes to 1e-09 at the fastest rho where every seed reaches it: median [min, max] at rho; rho_star")
    for (inst, b, scheme), v in B["cells"].items():
        f = v["fastest"]
        print(f"    {inst:9s} b = {b:>2s} {scheme:9s}: {f['median']:5.1f} [{f['min']:.1f}, {f['max']:.1f}] at rho "
              f"{f['rho']:g}; rho_star {v['rho_star']}")
    print(f"    instances at b = 20, 50: independent {B['iid_range'][0]}-{B['iid_range'][1]} passes, permutation "
          f"{B['rr_range'][0]}-{B['rr_range'][1]}; passes saved: " + ", ".join(f"{s:.1f} %" for s in B["saved"]))
    for b in EMPHASIS:
        i, r = B["cells"][("model", b, "iid")], B["cells"][("model", b, "reshuffle")]
        print(f"    model at b = {b}: {i['fastest']['median']} passes (independent) against {r['fastest']['median']} "
              f"(permutation); rho_star {i['rho_star']} and {r['rho_star']}")


# ----------------------------------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------------------------------
def panel_a(ax, A, ps, b):
    ink = ps.INK
    IID, RR = ps.BLUE, ps.ORANGE
    for y in (1.0, 2.0):
        ax.axhline(y, color=ink["secondary"], lw=0.8, zorder=1)
    ax.plot(A["c"], A["iid"], color=IID, lw=2.2, zorder=4, label="independent: expectation")
    ax.plot(A["c"], A["rr"], color=RR, lw=2.2, zorder=3, label="permutation: expectation")
    r = A["runs"]
    bars = ax.vlines(r[:, 0], r[:, 2], r[:, 3], color=IID, lw=1.3, zorder=5)
    pts = ax.plot(r[:, 0], r[:, 1], ls="", marker="o", ms=5.0, mfc=IID, mec=ink["surface"], mew=1.0, zorder=6,
                  label="independent: single runs")
    b.add(2, bars, pts)
    ax.set_xscale("log")
    ax.set_xlim(C_LO / 1.12, C_HI * 1.12)
    ticks = [0.25, 0.5, 1, 2, 4, 8, 16, 32]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["1/4", "1/2", "1", "2", "4", "8", "16", "32"])
    ax.minorticks_off()
    ax.set_ylim(0.0, 2.2)
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.set_yticklabels(["0", "", "1", "", "2"])
    ax.set_xlabel(r"step $c = \gamma n \lambda$")
    ax.set_ylabel("squared error:\nnats lost per pass", linespacing=1.1)
    es.panel_title(ax, "(a) linear SAGA")
    # anchored by its top-right corner: the entries do not move when the single runs come in (state 2)
    b.legend(ax, loc="upper right", bbox_to_anchor=(1.0, 0.36), handlelength=1.2, borderaxespad=0.15,
             handletextpad=0.4, labelspacing=0.25)


def panel_b(ax, B, ps):
    ink = ps.INK
    colour = {"iid": ps.BLUE, "reshuffle": ps.ORANGE}
    offset = {"iid": -0.2, "reshuffle": 0.2}           # wide enough that close medians (model, b = 5) do not hide
    gap = 0.8
    pos, x = {}, 0.0
    for inst, _ in INSTANCES:
        for bsz in BATCHES:
            pos[(inst, bsz)] = x
            x += 1.0
        x += gap
    for gi, (inst, name) in enumerate(INSTANCES):
        lo, hi = pos[(inst, EMPHASIS[0])] - 0.5, pos[(inst, EMPHASIS[-1])] + 0.5
        ax.axvspan(lo, hi, color=es.tint(ink["secondary"], 0.10), lw=0, zorder=0)
        for scheme, label in SCHEMES:
            xs = np.array([pos[(inst, bsz)] for bsz in BATCHES]) + offset[scheme]
            f = [B["cells"][(inst, bsz, scheme)]["fastest"] for bsz in BATCHES]
            ax.vlines(xs, [v["min"] for v in f], [v["max"] for v in f], color=colour[scheme], lw=1.3, zorder=3)
            filled = scheme == "reshuffle"
            ax.plot(xs, [v["median"] for v in f], color=colour[scheme], lw=1.3, marker="o", ms=5.5,
                    mfc=colour[scheme] if filled else ink["surface"], mec=ink["surface"] if filled else colour[scheme],
                    mew=1.0 if filled else 1.4, zorder=5 if filled else 4, label=label if gi == 0 else None)
        mid = 0.5 * (pos[(inst, BATCHES[0])] + pos[(inst, BATCHES[-1])])
        ax.annotate(name, (mid, 0), xycoords=("data", "axes fraction"), xytext=(0, -21), textcoords="offset points",
                    ha="center", va="top", fontsize=es.small(), color=ink["secondary"], linespacing=1.1)
    ticks = [pos[(inst, bsz)] for inst, _ in INSTANCES for bsz in BATCHES]
    ax.set_xticks(ticks)
    ax.set_xticklabels([bsz for _ in INSTANCES for bsz in BATCHES], fontsize=es.small())
    ax.set_xlim(-0.6, max(ticks) + 0.6)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("batch size $b$", labelpad=25)
    ax.set_ylim(0, 30)
    ax.set_yticks([0, 10, 20, 30])
    ax.set_ylabel(r"passes to $10^{-9}$")
    es.panel_title(ax, "(b) fixed-length recipe")
    ps.legend(ax, loc="lower left", handlelength=1.4, borderaxespad=0.15, handletextpad=0.4, labelspacing=0.25)
    real = [i for i, _ in INSTANCES if i != "model"]
    x_real = 0.5 * (pos[(real[0], EMPHASIS[0])] + pos[(real[-1], EMPHASIS[-1])])
    kw = dict(ha="center", va="top", fontsize=es.small(), color=ink["secondary"], linespacing=1.15)
    lo, hi = min(B["saved"]), max(B["saved"])
    ax.text(x_real, 29.6, f"$b = 20,\\ 50$: {lo:.0f}–{hi:.0f} %\nfewer passes", **kw)
    x_model = 0.5 * (pos[("model", BATCHES[0])] + pos[("model", BATCHES[-1])])
    ax.text(x_model, 29.6, "$b = 20,\\ 50$:\nno gain", **kw)


def draw(A, B):
    ps = es.setup("talk")
    fig = es.figure("talk", height=2.75)
    fig.get_layout_engine().set(w_pad=0.03, wspace=0.04)
    gs = fig.add_gridspec(1, 2, width_ratios=(0.94, 1.06))
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    b = es.Build()
    panel_a(ax_a, A, ps, b)
    panel_b(ax_b, B, ps)
    b.add(3, ax_b)
    es.save(fig, NAME, "talk", build=b,
            note=(f"(a) common Hessian, $n = {N}$, {A['seeds']} runs; (b) exact polar, 5 seeds; why instances are "
                  "slower: open"))


def main():
    A, B = data_a(), data_b()
    print_all(A, B)
    draw(A, B)


if __name__ == "__main__":
    main()
