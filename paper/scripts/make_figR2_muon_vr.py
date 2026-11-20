"""Figure R2 of the extended report (archived: archive/muon/report_spectral_muon_new.tex), and helper functions of
make_figP2_geometry.py and make_figP3_noise.py (figures of the paper): finite-sum variance reduction (SAGA and relatives)
inside Muon-type steps, i.e. steps along the polar factor of a matrix gradient estimate. One row of three panels:

  (a) two floors        M3, softmax memory-limited instance, runs from W_0 = 0 (60 passes, algorithm seed 0): relative
                        squared distance to the minimizer against passes for the polar step of the SAGA estimator with
                        a fixed length and with the nuclear-scaled length, for Muon (b = 32) and for the minibatch
                        gradient (b = 32) with the nuclear-scaled length;
  (b) SAGA-Muon / MISO  M6, softmax, kappa/n = 1 (the same instance), fixed untuned parameters, same sampled indices
                        (algorithm seed 0): Euclidean SAGA (c = 1/4), MISO and the joint SAGA-Muon step (c = 1/4);
  (c) tuned steps       M1, the two conditioning-limited instances: passes to the relative squared distance 1e-6,
                        median and range over the 10 confirmation seeds, for seven families.

The script only reads the JSON summaries of muon/results (nothing is rerun). It prints, panel by panel, every number
quoted from the figure and compares them with the tables of muon/results/digest.md; it stops before drawing if a number
disagrees with the digest.

Colours (palette of experiments/plotstyle.py). (a): the colours of figM3_floors; the two minibatch methods are drawn
with broken lines because their hues are close to the orange of the SAGA estimator. (c): the group colours of
figM1_ranking (Euclidean blue, orthogonalized orange). (b): the SAGA-Muon run is the orange run of (a) (same method,
same seed) and keeps its colour; SAGA and MISO are in the blue of their group, MISO as a wide translucent line under
the dashed SAGA-Muon line. figM6_miso draws MISO in orange and SAGA-Muon in red instead: --figM6-colours restores that.

Run:    python paper/scripts/make_figR2_muon_vr.py [--figM6-colours] [--no-digest-check]
Output: muon/figures/figR2_muon_vr.png and .pdf (figure of the archived report, not of the paper; the paper's
        scripts that import this module write to paper/figures)
"""
from __future__ import annotations

import json
import math
import os
import sys
import types

import numpy as np

for _stream in (sys.stdout, sys.stderr):                     # Windows consoles default to cp1252
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))                 # paper/scripts -> project root
RESULTS = os.path.join(ROOT, "muon", "results")
FIGURES = os.path.join(ROOT, "muon", "figures")                  # figR2 only; the paper's figures: PAPER_FIGURES
PAPER_FIGURES = os.path.join(ROOT, "paper", "figures")
DIGEST = os.path.join(RESULTS, "digest.md")
NAME = "figR2_muon_vr"

FIG_H = 2.28                               # inches (width: plotstyle.FIG_W); the saved, tight figure is 2.4 in high
WIDTHS = (1.70, 1.38, 1.45)                # relative widths of the three axes
Y_LO, Y_DATA_TOP = 1e-24, 10.0             # (a), (b): data range; below 1e-24 the runs are at rounding level (as in
Y_TOP = 3e16                               # figM3_floors); from Y_DATA_TOP to Y_TOP: room of the legends, no data
Y_TICKS = (1e-24, 1e-18, 1e-12, 1e-6, 1.0)
DIST_LABEL = r"$\|W - W_\star\|_F^2\ /\ \|W_\star\|_F^2$"

# (a) key in M3_summary.json ({b} = batch of the instance), legend label, colour of figM3_floors, line style
A_INSTANCE = "soft-mem"
A_SERIES = (
    ("SAGA est., fixed length 1e-2", "SAGA est., fixed length", "BLUE", "-"),
    ("SAGA est., nuclear-scaled", "SAGA est., nuclear-scaled", "ORANGE", "-"),
    ("Muon b={b}, lr 1e-2", "Muon, b = {b}", "MAGENTA", (0, (3.4, 1.7))),
    ("minibatch b={b}, nuclear-scaled", "minibatch, nuclear-scaled", "RED", (0, (1.2, 1.3))),
)
A_MARKS = (10, 30, 60)

# (b) key in M6_summary.json, legend label; line styles for the two colour schemes
B_INSTANCE = "softmax, kappa/n = 1"
B_SERIES = (
    ("SAGA (Euclidean), c = 1/4", "SAGA, c = 1/4"),
    ("MISO (delta = 1)", "MISO"),
    ("joint SAGA-Muon, c = 1/4", "SAGA-Muon, c = 1/4"),
)
B_WIDE = dict(lw=3.8, alpha=0.45, solid_capstyle="butt")
B_THIN = dict(lw=1.3, ls=(0, (4.0, 2.0)))
B_STYLE = {
    "figure": {"SAGA (Euclidean), c = 1/4": dict(color="BLUE", lw=1.7),
               "MISO (delta = 1)": dict(color="BLUE", **B_WIDE),
               "joint SAGA-Muon, c = 1/4": dict(color="ORANGE", **B_THIN)},
    "figM6": {"SAGA (Euclidean), c = 1/4": dict(color="BLUE", lw=1.7),
              "MISO (delta = 1)": dict(color="ORANGE", **B_WIDE),
              "joint SAGA-Muon, c = 1/4": dict(color="RED", **B_THIN)},
}

# (c) instances (key in M1_summary.json, legend label), families (name in M1_summary.json, row label), groups
C_INSTANCES = (("soft-cond", "softmax"), ("ls-cond", "least squares"))
C_FAMILIES = (
    ("SAGA", "SAGA"),
    ("B-SAGA", "B-SAGA"),
    ("IS-B-SAGA", "IS-B-SAGA"),
    ("cov-metric B-SAGA", "cov-metric B-SAGA"),
    ("SAGA-Muon (joint)", "SAGA-Muon (joint)"),
    ("soft SAGA-Muon", "soft SAGA-Muon"),
    ("one-shot polar-metric B-SAGA", "one-shot polar-\nmetric B-SAGA"),
)
C_GROUP = {"euclid": ("BLUE", "Euclidean"), "spectral": ("ORANGE", "orthogonalized")}   # as in figM1_ranking
C_XLIM_LO = 12.0
C_TICKS = (20, 30, 50, 100, 300)


# ----------------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------------
def load(name):
    with open(os.path.join(RESULTS, name + ".json"), encoding="utf-8") as f:
        return json.load(f)


def _trace(e):
    """Recorded passes and relative squared distances of a stored trace (finite, positive values)."""
    p = np.asarray(e["trace"]["passes"], dtype=float)
    r = np.asarray([np.nan if v is None else v for v in e["trace"]["rel"]], dtype=float)
    ok = np.isfinite(r) & (r > 0)
    return p[ok], r[ok]


def _first_below(p, r, level):
    k = np.nonzero(r <= level)[0]
    return float(p[k[0]]) if len(k) else float("inf")


def data_a(m3):
    inst = m3["instances"][A_INSTANCE]
    b = inst["batch"]
    block = m3["results"][A_INSTANCE]
    out = {"spec": inst["spec"], "budget": inst["passes"], "batch": b, "seeds": m3["seeds"], "series": []}
    for key, label, colour, ls in A_SERIES:
        key, label = key.format(b=b), label.format(b=b)
        e = block[key]
        p, r = _trace(e)
        raw = np.asarray(e["trace"]["rel"], dtype=float)
        tail = raw[-max(1, len(raw) // 6):]                   # the rule of exp_M3_floors.py: last sixth of the run
        at = [(float(p[k]), float(r[k])) for k in (int(np.argmin(np.abs(p - q))) for q in A_MARKS)]
        out["series"].append({
            "key": key, "label": label, "colour": colour, "ls": ls, "passes": p, "rel": r, "kind": e["kind"],
            "at": at, "last": (float(p[-1]), float(r[-1])), "max": (float(p[np.argmax(r)]), float(r.max())),
            "tail_from": float(e["trace"]["passes"][-len(tail)]), "level_seed0": float(np.median(tail)),
            "reached": np.asarray(e["reached"], dtype=float), "stationary": np.asarray(e["stationary"], dtype=float),
            "leaves_axis": _first_below(p, r, Y_LO), "diverged": e["diverged"]})
    return out


def data_b(m6):
    block = m6["results"][B_INSTANCE]
    tol = m6["tol"]
    out = {"spec": m6["instances"][B_INSTANCE], "budget": m6["passes"], "tol": tol, "seeds": m6["seeds"],
           "delta_sdca": block["delta_sdca"], "series": []}
    for key, label in B_SERIES:
        e = block["rows"][key]
        p, r = _trace(e)
        out["series"].append({
            "key": key, "label": label, "passes": p, "rel": r, "cross_trace": _first_below(p, r, tol),
            "cross": np.asarray(e["passes"], dtype=float), "final": np.asarray(e["final"], dtype=float),
            "last": (float(p[-1]), float(r[-1])), "max": (float(p[np.argmax(r)]), float(r.max())),
            "diverged": e["diverged"]})
    return out


def compare_b(B):
    """MISO against the joint SAGA-Muon step on their common recorded passes."""
    s = {x["key"]: x for x in B["series"]}
    m, j = s["MISO (delta = 1)"], s["joint SAGA-Muon, c = 1/4"]
    if not np.array_equal(m["passes"], j["passes"]):
        raise SystemExit("(b): the MISO and SAGA-Muon traces are not recorded at the same passes")
    p, rm, rj = m["passes"], m["rel"], j["rel"]
    fac = np.maximum(rm / rj, rj / rm)                            # ratio of the two traces, >= 1

    def worst(sel):
        k = np.nonzero(sel)[0]
        k = k[np.argmax(fac[k])]
        return {"factor": float(fac[k]), "pass": float(p[k]), "miso": float(rm[k]), "muon": float(rj[k]),
                "rel_diff": float(abs(rm[k] - rj[k]) / max(rm[k], rj[k])), "n": int(np.sum(sel)),
                "median_factor": float(np.median(fac[sel]))}

    above = np.nonzero(rj >= rm)[0]                               # recorded passes where SAGA-Muon is not ahead
    cm, cj = m["cross_trace"], j["cross_trace"]
    return {"all": worst(np.ones(len(p), dtype=bool)),
            "windows": [((lo, hi), worst((p > lo) & (p <= hi))) for lo, hi in ((0, 2), (2, 12), (12, 20), (20, 40))],
            "to_tol": worst((p > 2) & (p <= max(cm, cj))),
            "ahead_from": float(p[above[-1] + 1]) if len(above) and above[-1] + 1 < len(p) else float("nan"),
            "n": len(p)}


def same_run(A, B):
    """The orange curve of (a) and the SAGA-Muon curve of (b): largest relative difference on the passes recorded in
    both traces (0 means the same run)."""
    a = next(x for x in A["series"] if x["key"] == "SAGA est., nuclear-scaled")
    b = next(x for x in B["series"] if x["key"] == "joint SAGA-Muon, c = 1/4")
    pa = {round(float(q), 6): v for q, v in zip(a["passes"], a["rel"])}
    common = [(pa[round(float(q), 6)], v) for q, v in zip(b["passes"], b["rel"]) if round(float(q), 6) in pa]
    d = max(abs(u - v) / max(u, v) for u, v in common) if common else float("nan")
    return len(common), d


def data_c(m1):
    out = {"instances": [], "rows": []}
    for key, label in C_INSTANCES:
        block = m1[key]
        out["instances"].append({"key": key, "label": label, "tol": block["tol"], "budget": block["passes"],
                                 "spec": block["spec"], "seeds": block["protocol"]["confirm_seeds"]})
    for fam, row_label in C_FAMILIES:
        row = {"family": fam, "label": row_label, "cells": []}
        for inst in out["instances"]:
            block = m1[inst["key"]]
            r = next(x for x in block["rows"] if x["family"] == fam)
            s = r["summary"][f"{block['tol']:g}"]
            per_seed = np.asarray(r["confirm_passes"], dtype=float)
            if not (math.isclose(float(np.median(per_seed)), s["median"]) and float(per_seed.min()) == s["min"]
                    and int(np.isfinite(per_seed).sum()) == s["reached"] and len(per_seed) == s["runs"]):
                raise SystemExit(f"(c): summary and per-seed passes of '{fam}' on {inst['key']} disagree")
            row.setdefault("group", r["group"])
            if r["group"] != row["group"] or r["group"] not in C_GROUP:
                raise SystemExit(f"(c): unexpected group of '{fam}'")
            row["cells"].append({"median": s["median"], "min": s["min"], "max": s["max"], "reached": s["reached"],
                                 "runs": s["runs"], "per_seed": per_seed, "best": r["best"],
                                 "final_median": r["final_rel"]["median"], "final": r["confirm_final"]})
        out["rows"].append(row)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# printed numbers
# ----------------------------------------------------------------------------------------------------------------------
def _spec(s):
    return (f"{s['loss']}, n = {s['n']}, d = {s['d']}, K = {s['K']}, kappa = {s['kappa']:g}, alpha = {s['alpha']:g}, "
            f"data seed {s['seed']}")


def _mmm(v, fmt="{:.1e}"):
    return f"{fmt.format(np.median(v))} [{fmt.format(v.min())}, {fmt.format(v.max())}]"


def _fmt1(v):
    """Format of a number of passes in the digest."""
    return f"{v:.1f}" if (v is not None and isinstance(v, (int, float)) and math.isfinite(v)) else "inf"


def _point(pt):
    parts = []
    for k, v in pt.items():
        if isinstance(v, dict) and set(v) == {"over", "c"}:
            parts.append(f"{k}={v['c']:g}" + ("" if v["over"] == "one" else f"/{v['over']}"))
        else:
            parts.append(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}")
    return ", ".join(parts)


def print_a(A):
    print(f"(a) two floors. Source: M3_summary.json, results['{A_INSTANCE}'][<key>]['trace'] (runs from W_0 = 0, "
          f"{A['budget']} passes, algorithm seed {A['seeds'][0]}; the experiment has seeds {A['seeds']}).")
    print(f"    instance: {_spec(A['spec'])}; minibatch methods use b = {A['batch']}.")
    print("    y = ||W - W_*||_F^2 / ||W_*||_F^2, x = passes (per-sample gradient evaluations / n). Values at the "
          "recorded point nearest to the pass named.")
    for s in A["series"]:
        at = "; ".join(f"pass {p:.2f}: {v:.2e}" for p, v in s["at"])
        print(f"    - {s['label']}  [key '{s['key']}', {s['kind']}; {s['colour'].lower()}, "
              f"{'solid' if s['ls'] == '-' else 'broken'} line; {len(s['passes'])} points, first at pass "
              f"{s['passes'][0]:g}]")
        print(f"        {at}; largest value {s['max'][1]:.2f} at pass {s['max'][0]:g}")
        print(f"        final level (median over passes {s['tail_from']:.1f}-{s['last'][0]:.0f}): seed "
              f"{A['seeds'][0]}: {s['level_seed0']:.2e}; {len(s['reached'])} seeds, median [min, max]: "
              f"{_mmm(s['reached'])}")
        print(f"        stationary level (run started 1e-14 from W_*, not drawn), median [min, max]: "
              f"{_mmm(s['stationary'])}"
              + (f"; below {Y_LO:g} (bottom of the axis) from pass {s['leaves_axis']:.1f}"
                 if math.isfinite(s["leaves_axis"]) else ""))


def print_b(B, cmp_, shared):
    print(f"(b) SAGA-Muon and MISO. Source: M6_summary.json, results['{B_INSTANCE}']['rows'][<key>]['trace'] "
          f"(fixed untuned parameters, {B['budget']} passes, same sampled indices, algorithm seed {B['seeds'][0]}; "
          f"the experiment has seeds {B['seeds']}).")
    print(f"    instance: {_spec(B['spec'])} (the instance of (a)); same axes as (a).")
    for s in B["series"]:
        print(f"    - {s['label']}  [key '{s['key']}'; {len(s['passes'])} points, first at pass {s['passes'][0]:g}]")
        print(f"        passes to {B['tol']:g}: seed {B['seeds'][0]}: {s['cross'][0]:.1f} (first point of the drawn "
              f"trace below {B['tol']:g}: {s['cross_trace']:.1f}); {len(s['cross'])} seeds, median [min, max]: "
              f"{_mmm(s['cross'], '{:.1f}')}")
        print(f"        at pass {s['last'][0]:.0f}: {s['last'][1]:.2e} (seed {B['seeds'][0]}); final value, "
              f"{len(s['final'])} seeds, median: {np.median(s['final']):.1e}; largest value {s['max'][1]:.2f} at pass "
              f"{s['max'][0]:g}")
    w = cmp_["all"]
    print(f"    MISO against SAGA-Muon on the {cmp_['n']} common recorded passes (ratio of the two traces, >= 1):")
    print(f"        largest ratio over the run: {w['factor']:.1f} at pass {w['pass']:.1f} (MISO {w['miso']:.2e}, "
          f"SAGA-Muon {w['muon']:.2e}; relative difference |a - b| / max(a, b) = {w['rel_diff']:.3f})")
    for (lo, hi), w in cmp_["windows"]:
        print(f"        passes ({lo}, {hi}]: largest ratio {w['factor']:.2f} at pass {w['pass']:.1f} (MISO "
              f"{w['miso']:.2e}, SAGA-Muon {w['muon']:.2e}); median ratio {w['median_factor']:.2f}")
    w = cmp_["to_tol"]
    print(f"        from pass 2 to the crossing of {B['tol']:g}: largest ratio {w['factor']:.2f} at pass "
          f"{w['pass']:.1f}; median ratio {w['median_factor']:.2f}")
    print(f"        SAGA-Muon is below MISO at every recorded pass from {cmp_['ahead_from']:.1f} on.")
    print(f"    the SAGA-Muon run of (b) and the 'SAGA est., nuclear-scaled' run of (a): largest relative difference "
          f"on the {shared[0]} passes recorded in both traces = {shared[1]:.1e} (same run).")


def print_c(Cd):
    i0 = Cd["instances"][0]
    print(f"(c) tuned steps. Source: M1_summary.json, [<instance>]['rows'][family]['summary']['{i0['tol']:g}'] and "
          f"['confirm_passes'] (passes to the relative squared distance {i0['tol']:g}; selected step of each family "
          f"run on the confirmation seeds {i0['seeds'][0]}..{i0['seeds'][-1]}; budget {i0['budget']} passes).")
    for inst in Cd["instances"]:
        print(f"    {inst['key']} ({inst['label']}): {_spec(inst['spec'])}")
    for row in Cd["rows"]:
        print(f"    - {row['family']}  [group '{row['group']}': {C_GROUP[row['group']][1]}]")
        for inst, c in zip(Cd["instances"], row["cells"]):
            seeds = ", ".join(_fmt1(float(v)) for v in np.sort(c["per_seed"]))
            print(f"        {inst['key']:9s} median {c['median']:.2f} (digest format: {_fmt1(c['median'])}), min "
                  f"{_fmt1(c['min'])}, max {_fmt1(c['max'])}, reached {c['reached']}/{c['runs']}; sorted seeds: "
                  f"{seeds}; selected step: {_point(c['best'])}")
            if c["reached"] < c["runs"]:
                miss = [f"{v:.1e}" for v, q in zip(c["final"], c["per_seed"]) if not math.isfinite(q)]
                print(f"        {'':9s} -> {c['runs'] - c['reached']} seed(s) do not reach {inst['tol']:g} within "
                      f"{inst['budget']} passes (final relative squared distance {', '.join(miss)}): the whisker "
                      f"runs to the right edge of the axis ({inst['budget']} passes = the budget)")


# ----------------------------------------------------------------------------------------------------------------------
# comparison with the digest
# ----------------------------------------------------------------------------------------------------------------------
def read_digest(path):
    """Tables of the digest: {(section, subsection): {'columns': [...], 'rows': {first cell: [cells]}}}."""
    tables, h2, h3 = {}, "", ""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("## "):
                h2, h3 = line[3:].strip(), ""
            elif line.startswith("### "):
                h3 = line[4:].strip()
            elif line.startswith("|"):
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                t = tables.setdefault((h2, h3), {"columns": None, "rows": {}})
                if t["columns"] is None:
                    t["columns"] = cells
                elif not all(set(c) <= set("-: ") for c in cells):
                    t["rows"][cells[0]] = cells
    return tables


def _table(tables, h2_prefix, h3_prefix):
    hits = [t for (h2, h3), t in tables.items() if h2.startswith(h2_prefix) and h3.startswith(h3_prefix)]
    if len(hits) != 1:
        raise SystemExit(f"digest: {len(hits)} tables for section '{h2_prefix}' / '{h3_prefix}' (expected one)")
    return hits[0]


def check_digest(A, B, Cd):
    """Compare the plotted quantities with the digest (same formats as make_digest.py). Returns the list of
    (panel, quantity, value from the JSON, value in the digest, equal)."""
    tables = read_digest(DIGEST)
    res = []

    def add(panel, what, mine, table, row, column):
        cells = table["rows"].get(row)
        found = cells[table["columns"].index(column)] if cells is not None and column in table["columns"] else None
        res.append((panel, what, mine, found, mine == found))

    t = _table(tables, "M3.", f"{A_INSTANCE} (")
    for s in A["series"]:
        add("a", f"{s['key']}: reached from W_0 = 0", _mmm(s["reached"]), t, s["key"], "reached from W_0 = 0")
        add("a", f"{s['key']}: stationary level", _mmm(s["stationary"]), t, s["key"], "stationary level")
    t = _table(tables, "M6.", f"{B_INSTANCE} (")
    for s in B["series"]:
        add("b", f"{s['key']}: passes to target", _mmm(s["cross"], "{:.1f}"), t, s["key"], "passes to target")
        add("b", f"{s['key']}: final rel. sq. distance", f"{np.median(s['final']):.1e}", t, s["key"],
            "final rel. sq. distance")
    over = _table(tables, "M1.", "M1 overview")
    for inst_i, inst in enumerate(Cd["instances"]):
        t = _table(tables, "M1.", f"{inst['key']}:")
        for row in Cd["rows"]:
            c = row["cells"][inst_i]
            part = f" ({c['reached']}/{c['runs']})" if c["reached"] < c["runs"] else ""
            add("c", f"{row['family']}, {inst['key']}: overview (median)", _fmt1(c["median"]) + part, over,
                row["family"], inst["key"])
            add("c", f"{row['family']}, {inst['key']}: median [min, max]",
                f"{_fmt1(c['median'])} [{_fmt1(c['min'])}, {_fmt1(c['max'])}]" + part, t, row["family"],
                f"passes to {inst['tol']:g}")
            add("c", f"{row['family']}, {inst['key']}: hit", f"{c['reached']}/{c['runs']}", t, row["family"], "hit")
    return res


def report_digest(res):
    bad = [r for r in res if not r[4]]
    print(f"digest check (muon/results/digest.md): {len(res)} cells compared, {len(bad)} disagreement(s).")
    for panel, tables in (("a", "M3 table of soft-mem, columns 'reached from W_0 = 0' and 'stationary level'"),
                          ("b", "M6 table 'softmax, kappa/n = 1', columns 'passes to target' and 'final rel. sq. "
                                "distance'"),
                          ("c", "M1 overview table and M1 tables of soft-cond and ls-cond, columns 'passes to 1e-06' "
                                "and 'hit'")):
        rows = [r for r in res if r[0] == panel]
        print(f"    ({panel}) {tables}: {sum(r[4] for r in rows)}/{len(rows)} cells equal")
    for panel, what, mine, found, _ in bad:
        print(f"    DISAGREEMENT ({panel}) {what}: JSON gives '{mine}', digest has '{found}'")
    return not bad


# ----------------------------------------------------------------------------------------------------------------------
# figure (style of the main project: experiments/plotstyle.py)
# ----------------------------------------------------------------------------------------------------------------------
def _plotstyle():
    sys.modules.setdefault("common", types.SimpleNamespace(FIGURES=FIGURES))   # plotstyle only needs common.FIGURES
    sys.path.insert(0, os.path.join(ROOT, "experiments"))
    import plotstyle as ps
    ps.setup()
    return ps


LEGEND_AB = dict(loc="upper left", handlelength=1.6, handletextpad=0.45, labelspacing=0.22, borderpad=0.1,
                 borderaxespad=0.2)


def _distance_axes(ax, budget, step, ps):
    """Axes of (a) and (b): passes against the relative squared distance; the left spine and the vertical grid lines
    stop at Y_DATA_TOP, the range above holds the legend."""
    ticks = np.arange(0, budget + 1, step)
    ax.set_xlim(0, budget)
    ax.set_xticks(ticks)
    ax.grid(axis="x", visible=False)
    ax.vlines(ticks, Y_LO, Y_DATA_TOP, color=ps.INK["grid"], lw=0.8, zorder=1.4)
    ax.spines["left"].set_bounds(Y_LO, Y_DATA_TOP)
    ax.set_xlabel("passes")


def panel_a(ax, A, ps):
    for z, s in enumerate(reversed(A["series"])):               # the SAGA-estimator curves on top
        ax.plot(s["passes"], s["rel"], color=getattr(ps, s["colour"]), ls=s["ls"], lw=1.7, label=s["label"],
                zorder=2 + z)
    ax.set_yscale("log")
    ax.set_ylim(Y_LO, Y_TOP)
    ax.set_yticks(Y_TICKS)
    ax.minorticks_off()
    _distance_axes(ax, A["budget"], 20, ps)
    ax.set_ylabel(DIST_LABEL)
    title = ax.set_title("(a) two floors", loc="left")
    h, lab = ax.get_legend_handles_labels()
    leg = ps.legend(ax, handles=h[::-1], labels=lab[::-1], **LEGEND_AB)
    return title, leg, [(s["passes"], s["rel"]) for s in A["series"]]


def panel_b(ax, B, ps, scheme):
    style = B_STYLE[scheme]
    for z, s in enumerate(B["series"]):                         # SAGA, then MISO (wide), then SAGA-Muon on top
        kw = dict(style[s["key"]])
        kw["color"] = getattr(ps, kw["color"])
        ax.plot(s["passes"], s["rel"], label=s["label"], zorder=2 + z, **kw)
    _distance_axes(ax, B["budget"], 10, ps)
    ax.tick_params(labelleft=False)
    title = ax.set_title("(b) SAGA-Muon ≈ MISO", loc="left")
    leg = ps.legend(ax, **LEGEND_AB)
    return title, leg, [(s["passes"], s["rel"]) for s in B["series"]]


def panel_c(ax, Cd, ps):
    plt, INK = ps.plt, ps.INK
    n = len(Cd["rows"])
    budget = Cd["instances"][0]["budget"]
    offsets = (0.2, -0.2)                                       # softmax above, least squares below
    drawn = []
    for i, row in enumerate(Cd["rows"]):
        y0 = n - 1 - i
        col = getattr(ps, C_GROUP[row["group"]][0])
        for q, c in enumerate(row["cells"]):
            y = y0 + offsets[q]
            hi = c["max"] if math.isfinite(c["max"]) else budget
            ax.plot([c["min"], hi], [y, y], color=col, lw=0.9, solid_capstyle="butt", zorder=2)
            if not math.isfinite(c["max"]):                    # some seeds miss the target: the range leaves the axis
                ax.plot([budget], [y], marker=">", ms=4.0, ls="", color=col, mec=col, mew=0.0, clip_on=False, zorder=4)
            if q == 0:
                ax.plot([c["median"]], [y], "o", ms=5.0, color=col, mec=INK["surface"], mew=0.8, zorder=3)
            else:
                ax.plot([c["median"]], [y], "s", ms=4.3, mfc=INK["surface"], mec=col, mew=1.3, zorder=3)
            drawn.append(([c["min"], hi], [y, y]))
    groups = [row["group"] for row in Cd["rows"]]
    for i in range(1, n):                                       # hairline between the two groups
        if groups[i] != groups[i - 1]:
            ax.axhline(n - i - 0.5, color=INK["grid"], lw=0.8, zorder=1)
    ax.set_yticks(range(n))
    ax.set_yticklabels([row["label"] for row in Cd["rows"]][::-1], fontsize=ps.SMALL, linespacing=0.95)
    ax.set_ylim(-1.5, n - 0.35)                                 # room for the legend below the long whisker
    ax.set_xscale("log")
    ax.set_xlim(C_XLIM_LO, budget)
    ax.set_xticks(C_TICKS)
    ax.set_xticklabels([str(t) for t in C_TICKS])
    ax.minorticks_off()
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("passes to reach 10⁻⁶")
    title = ax.set_title("(c) tuned steps", loc="left")
    ink = INK["secondary"]
    handles = [plt.Line2D([], [], marker="o", ls="", ms=5.0, color=ink, mec=INK["surface"], mew=0.8),
               plt.Line2D([], [], marker="s", ls="", ms=4.3, mfc=INK["surface"], mec=ink, mew=1.3)]
    labels = [inst["label"] for inst in Cd["instances"]]
    for g in dict.fromkeys(groups):                             # colour = group: a swatch, not a marker
        handles.append(plt.Line2D([], [], color=getattr(ps, C_GROUP[g][0]), lw=3.0, solid_capstyle="butt"))
        labels.append(C_GROUP[g][1])
    leg = ps.legend(ax, handles=handles, labels=labels, loc="lower right", handlelength=0.7, handletextpad=0.4,
                    labelspacing=0.2, borderpad=0.1, borderaxespad=0.2)
    return title, leg, drawn


def _inside(ax, box, lines, pad):
    """Number of points of the drawn lines (sampled every 1.5 pixel along their segments) inside the box enlarged
    by pad, among those inside the axes."""
    x0, y0, x1, y1 = box.x0 - pad, box.y0 - pad, box.x1 + pad, box.y1 + pad
    ab = ax.bbox
    count = 0
    for x, y in lines:
        pts = ax.transData.transform(np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)]))
        seg = [pts[:1]]
        for a, b in zip(pts[:-1], pts[1:]):
            m = max(1, int(math.ceil(float(np.hypot(*(b - a))) / 1.5)))
            seg.append(a + (b - a) * np.linspace(0.0, 1.0, m + 1)[1:, None])
        q = np.vstack(seg)
        q = q[(q[:, 0] >= ab.x0) & (q[:, 0] <= ab.x1) & (q[:, 1] >= ab.y0) & (q[:, 1] <= ab.y1)]
        count += int(np.sum((q[:, 0] >= x0) & (q[:, 0] <= x1) & (q[:, 1] >= y0) & (q[:, 1] <= y1)))
    return count


def layout_checks(fig, panels):
    """Legends at least 3 pt away from the data and inside their axes (above the data range in (a) and (b)); titles
    clear of the title of the next panel and of the edge of the figure; tick labels apart."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    pt = fig.dpi / 72.0
    msgs = []
    titles = [p[2].get_window_extent(rend) for p in panels]
    for k, (name, ax, _, leg, lines) in enumerate(panels):
        lb, ab = leg.get_window_extent(rend), ax.bbox
        hit = _inside(ax, lb, lines, pad=3.0 * pt)
        if hit:
            msgs.append(f"({name}) the legend is on the data ({hit} sampled points)")
        if lb.x0 < ab.x0 - 1 or lb.x1 > ab.x1 + 1 or lb.y0 < ab.y0 - 1 or lb.y1 > ab.y1 + 1:
            msgs.append(f"({name}) the legend leaves the axes by "
                        f"{max(ab.x0 - lb.x0, lb.x1 - ab.x1, ab.y0 - lb.y0, lb.y1 - ab.y1) / pt:.1f} pt")
        if ax.get_yscale() == "log" and lb.y0 < ax.transData.transform((0.0, Y_DATA_TOP))[1] + 2.0 * pt:
            msgs.append(f"({name}) the legend enters the data range")
        limit = titles[k + 1].x0 - 18.0 * pt if k + 1 < len(panels) else fig.bbox.x1
        if titles[k].x1 > limit:
            msgs.append(f"({name}) the title is {(titles[k].x1 - limit) / pt:.0f} pt too long")
        for axis, j in ((ax.xaxis, 0), (ax.yaxis, 1)):
            boxes = [t.get_window_extent(rend) for t in axis.get_ticklabels() if t.get_visible() and t.get_text()]
            boxes.sort(key=lambda b_: (b_.x0, b_.y0)[j])
            for b0, b1 in zip(boxes[:-1], boxes[1:]):
                gap = (b1.x0 - b0.x1, b1.y0 - b0.y1)[j]
                if gap < 1.0 * pt:
                    msgs.append(f"({name}) two {'xy'[j]} tick labels are {gap / pt:.1f} pt apart")
    print("layout checks (legends away from the data and inside the axes, titles apart, tick labels apart): "
          + ("ok" if not msgs else "; ".join(msgs)))
    return not msgs


def draw(A, B, Cd, scheme):
    ps = _plotstyle()
    fig = ps.figure(width=ps.FIG_W, height=FIG_H)
    fig.get_layout_engine().set(w_pad=0.03, wspace=0.0)
    gs = fig.add_gridspec(1, 3, width_ratios=WIDTHS)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
    ax_c = fig.add_subplot(gs[0, 2])
    panels = [("a", ax_a) + panel_a(ax_a, A, ps), ("b", ax_b) + panel_b(ax_b, B, ps, scheme),
              ("c", ax_c) + panel_c(ax_c, Cd, ps)]
    layout_checks(fig, panels)
    ps.savefig(fig, NAME)


def main():
    scheme = "figM6" if "--figM6-colours" in sys.argv else "figure"
    A, B, Cd = data_a(load("M3_summary")), data_b(load("M6_summary")), data_c(load("M1_summary"))
    print_a(A)
    print_b(B, compare_b(B), same_run(A, B))
    print_c(Cd)
    if "--no-digest-check" not in sys.argv:
        if not report_digest(check_digest(A, B, Cd)):
            raise SystemExit("the figure is not drawn: the numbers above disagree with the digest")
    print(f"colours of (b): {'figM6_miso' if scheme == 'figM6' else 'groups, as in (c) (default)'}")
    draw(A, B, Cd, scheme)


if __name__ == "__main__":
    main()
