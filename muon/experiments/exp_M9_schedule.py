"""M9: Muon's own recipe with a SAGA table.

Muon as used takes steps of fixed length, W <- W - lr_t orth(D_t), with a learning rate that is cooled down. Fed by a
minibatch gradient (or by a momentum-based buffer) the iteration freezes on a finite sum: the estimator does not
vanish at W_* (experiment M3). Fed by the SAGA estimator and cooled geometrically,

    lr_t = lr_0 q^(t b / n)        (the learning rate loses the factor q per pass over the data),

the squared distance follows the schedule, ||W_t - W_*||_F^2 = C lr_t^2 with C bounded, as long as q stays above an
instance-dependent critical value: the geometric-step law of normalized steps, made usable on a finite sum by the
table. Two numbers are tuned, q and lr_0, and the fastest q sits next to a cliff: a run that falls off the schedule
freezes for good (it does not diverge). This script measures the recipe with the protocol of common.py and puts next
to it the controls that say what belongs to the spectral map and what does not.

  A  maps by batch size (instances soft-cond, ls-cond, ls-orth, soft-sq; b in {1, 16, 32}): the map of the step is
     the exact polar factor, five Newton-Schulz steps, or one of two CONTROLS that are not orthogonalizations
     (Frobenius normalization, coordinatewise sign), each with and without Nesterov momentum 0.9; grid of q times
     half-decades of lr_0 (families.recipe_families). Same-batch references with constant steps on 2^(1/4) grids:
     SAGA, damped SAGA (B-SAGA), nuclear-scaled joint SAGA-Muon (families.resolved_references).
  B  batch one on the six instances of M1: exact polar factor without momentum, and Muon's recipe (Newton-Schulz,
     Nesterov 0.9), on a finer grid of q (families.batch_one_recipes); the same two at the fixed point q = 0.8,
     lr_0 = 0.1 on the confirmation seeds; next to the medians of M1 (read from M1_summary.json, factor-2 grids).
  C  the law and its controls (two instances, batch one, schedules selected in B): runs without stop rule for the
     exact polar factor, for Muon's recipe and for Muon's recipe with eps = 1e-30 in ns5 (level and band of
     dist^2 / lr^2, depth down to which the law holds, lowest distance reached); Muon's recipe also at the fixed
     point, because its selected point sits on the cliff; the schedule selected for Muon's recipe without a table
     (minibatch gradient, MARS-M, STORM-Muon at b = 1 and 32): final distance.
  D  sampling order and memory of the table (soft-mem, ls-mem; soft-cond as the control where memory is not the
     limit): exact polar factor, no momentum, lr decayed by 1e-5 in total at the per-pass factor rho of lr^2; order
     in {iid, reshuffle, shuffle_once} and, for independent batches, m in {b, 3b} extra table entries refreshed per
     step (rho grid extended downwards for m = 3b); fraction of the seeds on schedule and passes to 1e-9, next to
     the memory factor of the table. b in {20, 50}; b in {5, 1} on request (--dbatches), where the order effect is
     not reliable.

Protocol: common.tune_and_confirm (screening on 20% of the budget, tuning seeds 0-4, confirmation seeds 10-19,
budgets and targets of M1). The cooldown has no span, so the recipe families are screened like the constant-step
references (see families.recipe_family). Parts C and D are designs at fixed parameters (no tuning): their seeds and
criteria are stored in the summary. Every number printed here is in muon/results/M9_summary.json.

Outputs: muon/results/M9_<key>_{screen,tune,confirm}.pkl (parts A, B), M9_law.pkl (C), M9_order.pkl (D) (caches),
muon/results/M9_summary.json (updated part by part: a part that is not run keeps its previous content).
Usage: python muon/experiments/exp_M9_schedule.py [--quick] [--parts A,B,C,D] [--only key1,key2] [--jobs N]
           [--batches 1,16,32] [--variants polar,polar+N,ns5,ns5+N,frob,frob+N,sign,sign+N] [--no-refs]
           [--dbatches 20,50] [--dseeds 5]

The summary in the repository was built by these invocations (C reads the selections of B; a rerun of each hits
the cache and recomputes nothing):
    --parts A --batches 16,32
    --parts B
    --parts C
    --parts D
    --parts A --batches 1 --no-refs
    --parts D --dbatches 5,1 --only soft-mem,ls-mem
One piece of the design has not been run: the b = 1 references of part A on 2^(1/4) grids (M1 has them on factor-2
grids; part B quotes its medians). summary['coverage'] lists what every cell holds; the bare command runs the whole
design. At b = 1 the exact polar factor without momentum and Muon's recipe are tuned twice, on the grid of part A
and on the finer grid of q of part B (quoted in the b = 1 cell of part A under 'part_B').
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np

import common as C
from families import COOLDOWNS, RECIPE_LABEL, batch_one_recipes, recipe_families, recipe_lr_grid, resolved_references

INSTANCES = {                                                    # instances, budgets and targets of M1
    "soft-cond": dict(spec=C.spec("softmax", 200, 40, 8, 1e4, 2.0), passes=300, tol=1e-6),
    "ls-cond": dict(spec=C.spec("ls", 200, 40, 8, 1e4, 2.0), passes=300, tol=1e-6),
    "ls-orth": dict(spec=C.spec("ls", 500, 20, 20, 1e5, 2.0, noise=0.1, planted="orthogonal"), passes=100, tol=1e-6),
    "soft-sq": dict(spec=C.spec("softmax", 500, 20, 20, 1e4, 2.0), passes=120, tol=1e-6),
    "soft-mem": dict(spec=C.spec("softmax", 1000, 50, 10, 1e3, 1.0), passes=40, tol=1e-9),
    "ls-mem": dict(spec=C.spec("ls", 1000, 50, 10, 1e3, 1.0), passes=40, tol=1e-9),
}
QUICK = {
    "ls-orth": dict(spec=C.spec("ls", 150, 8, 8, 1e4, 2.0, noise=0.1, planted="orthogonal"), passes=40, tol=1e-6),
    "soft-mem": dict(spec=C.spec("softmax", 150, 12, 5, 1e2, 1.0), passes=25, tol=1e-9),
}
MAPS_ON = ("soft-cond", "ls-cond", "ls-orth", "soft-sq")          # part A: the conditioning-limited instances
BATCHES = (1, 16, 32)
FIXED_POINT = dict(schedule="geometric", lr_decay=0.8, lr=0.1)   # part B: one default for the six instances
M1_FAMILIES = ("SAGA", "B-SAGA", "SAGA-Muon (joint)", "soft SAGA-Muon", "one-shot polar-metric B-SAGA",
               "cov-metric B-SAGA")
LAW_ON = ("soft-cond", "ls-cond")                                 # part C
LAW_SEEDS = (10, 11, 12, 13, 14)
LAW_WINDOW = (1e-12, 1e-6)       # relative squared distances between which the level C of dist^2 / lr^2 is measured
LAW_FACTORS = (3.0, 10.0)        # the law "holds" down to the last record whose dist^2 / lr^2 is at most this
                                 # many times its level (two readings: tight and loose)
NO_TABLE_BATCHES = (1, 32)
ORDER_ON = ("soft-mem", "ls-mem")                                 # part D; ORDER_CONTROL is not memory-limited
ORDER_CONTROL = "soft-cond"
ORDER_BATCHES = (20, 50)
ORDER_SEEDS = (0, 1, 2, 3, 4)
RHOS = (0.61, 0.50, 0.43, 0.368, 0.32, 0.27, 0.22, 0.18, 0.135, 0.10, 0.07, 0.05)   # factor of lr^2 per pass
RHOS_CONTROL = (0.90, 0.80, 0.70) + RHOS
RHOS_LOW = (0.035, 0.025, 0.018, 0.012, 0.008, 0.005)   # extension for m = 3b, whose memory factor is below 0.05
TOTAL_DECAY = 1e-5               # lr_T / lr_0 of the runs of part D
PATH_FACTOR = 4.0                # ... whose total step length sum_t lr_t is this many times ||W_*||_op
TRACK_FACTOR = 100.0             # a seed is on schedule if its final dist^2 / lr^2 is at most this many times the
TRACK_QUORUM = 0.8               # reference level; the threshold needs this fraction of the seeds on schedule
ORDER_TOL = 1e-9


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def norms(s):
    """||W_*||_F^2 (the normalization of the distances) and ||W_*||_op of an instance."""
    Ws = C.problem(s).W_star
    return {"D0": float(np.vdot(Ws, Ws)), "W_star_op": float(np.linalg.norm(Ws, 2))}


def ratio_trace(res, D0):
    """(passes, rel, ratio) along the recorded points of a fixed-length run, with
    ratio = ||W_t - W_*||_F^2 / eta_t^2 and eta_t the length of the last step taken (lr_{t-1} times the scale
    max(1, K/d)^(1/2), which is 1 on every instance of this study). The record of W_0 (no step yet) is dropped.
    None for a failed run."""
    tr = res.get("trace")
    if not tr or "eta" not in tr:
        return None
    p, rel, eta = (np.asarray(tr[k], dtype=float) for k in ("passes", "rel", "eta"))
    ok = np.isfinite(eta) & (eta > 0) & np.isfinite(rel)
    if not ok.any():
        return None
    return p[ok], rel[ok], rel[ok] * D0 / eta[ok] ** 2


def band(runs, D0):
    """dist^2 / lr^2 over the second half (in passes) of the runs that stayed on schedule down to their stop level
    (stopped == 'stop_dist2'): smallest and largest value over these runs, and the median over the runs of the
    width max / min of one run."""
    lo, hi, width = [], [], []
    for r in runs:
        t = ratio_trace(r, D0) if r.get("stopped") == "stop_dist2" else None
        if t is None:
            continue
        p, _, ratio = t
        x = ratio[p >= 0.5 * p[-1]]
        lo.append(float(x.min()))
        hi.append(float(x.max()))
        width.append(float(x.max() / x.min()))
    if not lo:
        return {"runs": 0}
    return {"runs": len(lo), "min": min(lo), "max": max(hi), "width_median": float(np.median(width))}


def passes_summary(res):
    """Summary of a list of runs in the format of common.tune_and_confirm."""
    out = {}
    for t in C.TOLS:
        v = np.array([r["passes_to"][f"{t:g}"] for r in res])
        vs = np.sort(v)
        out[f"{t:g}"] = {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()),
                         "q1": float(vs[(len(vs) - 1) // 4]), "q3": float(vs[(3 * (len(vs) - 1) + 3) // 4]),
                         "reached": int(np.isfinite(v).sum()), "runs": len(v)}
    return out


def stats(v):
    v = np.asarray(v, dtype=float)
    return {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}


def row_of(rec, tol, D0, meta):
    """Row of the summary for one record of tune_and_confirm."""
    tk = f"{tol:g}"
    row = {"family": rec["family"], **meta, "best": rec["best"], "edge": rec["edge"], "best_score": rec["best_score"],
           "summary": rec["summary"], "final_rel": rec["final_rel"], "stop_level": rec["stop_level"],
           "refined": rec["refined"], "finalists": rec["finalists"], "n_points": rec["n_points"],
           "n_kept": rec["n_kept"], "diverged": rec["diverged"],
           "confirm_passes": [c["passes_to"][tk] for c in rec["confirm"]],
           "confirm_final": [c["final_rel"] for c in rec["confirm"]],
           "tune": [{"point": t["point"], "score": t["score"], "score_refined": t.get("score_refined"),
                     "screen_rel": t["screen_rel"], "kept": t["kept"]} for t in rec["tune"]]}
    if meta.get("kind") == "recipe":
        row["dist2_over_lr2"] = band(rec["confirm"], D0)
    return row


def variant_key(orth, nesterov):
    return RECIPE_LABEL[orth] + ("+N" if nesterov else "")


def recipe_meta(fam):
    b = fam["base"]
    return {"kind": "recipe", "variant": variant_key(b["orth"], bool(b.get("nesterov"))), "orth": b["orth"],
            "nesterov": bool(b.get("nesterov")), "batch": int(b["batch"]), "group": fam["group"],
            "what": fam["what"]}


def reference_meta(fam):
    return {"kind": "reference", "variant": fam["name"].split(" b=")[0], "batch": int(fam["base"]["batch"]),
            "group": fam["group"], "what": fam["what"]}


def protocol():
    return {"tune_seeds": list(C.TUNE_SEEDS), "refine_seeds": list(C.REFINE_SEEDS), "refine_top": C.REFINE_TOP,
            "confirm_seeds": list(C.CONFIRM_SEEDS), "screen_fraction": C.SCREEN_FRACTION,
            "screen_keep": C.SCREEN_KEEP, "stops": [C.SCREEN_STOP, C.TUNE_STOP, C.CONFIRM_STOP],
            "screening": "the recipe families are screened like the constant-step references: the geometric "
                         "cooldown does not depend on the length of the run, so the screening run is a prefix of "
                         "the full run (the schedule is a one-valued grid parameter, see families.recipe_family)",
            "accounting": "passes = per-sample gradient evaluations / n, initial full gradient of the SAGA table "
                          "included (1 pass); b + m per iteration with m extra refreshes"}


def definitions():
    return {
        "recipe": "W <- W - lr_t map(D_t), D_t the SAGA estimator (joint placement; through Nesterov momentum 0.9 "
                  "when stated), lr_t = lr_0 q^(t b / n) (schedule 'geometric', lr_decay = q), W_0 = 0",
        "maps": "polar = exact polar factor (orth 'svd'); ns5 = five Newton-Schulz steps with eps = 1e-7; frob = "
                "M / ||M||_F and sign = coordinatewise sign (gauge-projected for softmax) are controls, not "
                "orthogonalizations. lr_0 is a length in the norm of the map: not comparable across maps",
        "summary": "passes to the tolerance over the confirmation seeds: median, min, max, quartiles, number of "
                   "seeds reaching it ('reached' of 'runs'); a median is infinite when half of the seeds miss",
        "stop_level": "confirmation runs that went on to CONFIRM_STOP x target (stopped == 'stop_dist2'): the runs "
                      "that stayed on schedule",
        "edge": "grid parameters whose selected value is the smallest ('-') or largest ('+') of its grid",
        "dist2_over_lr2": "||W_t - W_*||_F^2 / lr_t^2 over the second half (in passes) of the confirmation runs "
                          "that reached the stop level: min and max over these runs, median width max/min of a run",
        "compare": "per cell of part A: the fastest variant (smallest median, ties broken by the number of seeds "
                   "reaching the target) among the spectral maps (polar, ns5; with or without Nesterov), among "
                   "the controls (frob, sign; with or without Nesterov), among the controls without momentum, "
                   "among the Euclidean references (SAGA, B-SAGA), and their ratios of medians",
        "part_B": "b = 1 cell of part A: results of the two batch-one recipes of part B on the same instance (tuned "
                  "on the finer grid of q of part B, lr_0 up to 1), and the ratios of their medians to the fastest "
                  "control of the cell (above 1: the control needs fewer passes)",
        "law": f"part C, per run without stop rule: level = median of dist^2 / lr^2 over the records with "
               f"relative squared distance in [{LAW_WINDOW[0]:g}, {LAW_WINDOW[1]:g}]; depth_rel_x3 and "
               f"depth_rel_x10 = relative squared distance of the last record whose ratio is at most 3 (resp. 10) "
               f"times the level (after a floor or a freeze the ratio grows for good: the law holds down to that "
               f"distance within that factor; a run that stalls on its floor is still within the factor for a "
               f"few passes, so these two readings sit next to min_rel); depth_rel_in_range = the same with the "
               f"largest ratio recorded in the window in place of 3 x level (the distance from which the ratio "
               f"leaves the range it had on schedule: where the floor starts to show); band = min and max of the "
               f"ratio over the second half (in passes) "
               f"of the run up to the x3 record; on_schedule = the run goes below {LAW_WINDOW[0]:g}; min_rel = "
               f"lowest relative squared distance recorded (the floor of the run)",
        "order": f"part D: rho = factor of lr^2 per pass (q = rho^(1/2)); T = ceil(P n / b) iterations with "
                 f"q^P = {TOTAL_DECAY:g}; lr_0 = {PATH_FACTOR:g} ||W_*||_op (1 - q^(b/n)) / (1 - {TOTAL_DECAY:g}) "
                 f"(total step length {PATH_FACTOR:g} ||W_*||_op), rounded to 6 digits; C = final dist^2 / lr^2; "
                 f"a seed is on schedule if C <= {TRACK_FACTOR:g} x the median C of the same cell and scheme at "
                 f"rho_ref; rho_star = smallest rho of the grid with at least {TRACK_QUORUM:g} of the seeds on "
                 f"schedule; rho_star_monotone = the same with the condition at every larger rho of the grid "
                 f"too; fastest = the rho with the smallest median passes to {ORDER_TOL:g} among those where "
                 f"every seed reaches it (passes count the extra refreshes)",
        "memory_factor": "(1 - (b + m) / n)^(n / b): probability that a table entry is not refreshed during one "
                         "pass, with independent batches of size b and m extra refreshes per step",
    }


def link_batch_one(summary):
    """Batch one: the exact polar factor without momentum and Muon's recipe are tuned in part B (finer grid of q,
    lr_0 up to 1). Their results are quoted in the b = 1 cell of part A, next to the variants tuned there, with the
    ratios of their medians to the fastest control of that cell (above 1: the control needs fewer passes)."""
    for key, e in summary.get("A", {}).items():
        cell = e.get("cells", {}).get("1")
        sec_B = summary.get("B", {}).get(key, {})
        if cell is None or not sec_B.get("rows") or sec_B.get("spec") != e.get("spec"):
            continue
        tk = f"{e['tol']:g}"
        out = {"rows": {}, "ratios": {}}
        for name, r in sec_B["rows"].items():
            s = r["summary"][tk]
            out["rows"][r["variant"]] = {"family": name, **s, "best": r["best"], "edge": r["edge"]}
            for label in ("control", "control_no_momentum"):
                c = cell.get("compare", {}).get(label)
                if c is not None and math.isfinite(s["median"]) and math.isfinite(c["median"]):
                    out["ratios"][f"{r['variant']}/{label}"] = s["median"] / c["median"]
        cell["part_B"] = out


def coverage(summary):
    """What the summary holds, and which of the eight variants of part A are missing in each cell."""
    every = [variant_key(o, m) for o in RECIPE_LABEL for m in (False, True)]
    out = {"A": {}, "B": sorted(summary.get("B", {})), "C": sorted(summary.get("C", {})), "D": {}}
    for key, e in summary.get("A", {}).items():
        for b, cell in e.get("cells", {}).items():
            have = [r["variant"] for r in cell["rows"].values() if r["kind"] == "recipe"]
            out["A"].setdefault(key, {})[b] = {
                "variants": [v for v in every if v in have], "missing_variants": [v for v in every if v not in have],
                "references": sorted(r["variant"] for r in cell["rows"].values() if r["kind"] == "reference")}
    for key, e in summary.get("D", {}).items():
        out["D"][key] = sorted(e.get("cells", {}), key=int)
    return out


def save(tag, summary, quick):
    link_batch_one(summary)
    summary["coverage"] = coverage(summary)
    summary["experiment"] = "M9: Muon's own recipe (fixed-length step, geometric cooldown) with a SAGA table"
    summary["quick"] = bool(quick)
    summary["code"] = C.code_fingerprint()
    summary["protocol"] = protocol()
    summary["definitions"] = definitions()
    C.save_json(f"{tag}_summary", summary)


def fmt_row(row, tol):
    s = row["summary"][f"{tol:g}"]
    return (f"{C.fmt_passes(row['summary'], tol):>20s} {str(s['reached']) + '/' + str(s['runs']):>6s}  "
            f"{C.fmt_point({k: v for k, v in row['best'].items() if k != 'schedule'}):34s}"
            f"{','.join(row['edge'])}")


# ----------------------------------------------------------------------------------------------
# part A: maps by batch size
# ----------------------------------------------------------------------------------------------
def best_of(rows, tol, pred):
    """Fastest row among those selected by `pred`: smallest median, then most seeds reaching the target."""
    tk = f"{tol:g}"
    cand = [r for r in rows if pred(r)]
    if not cand:
        return None
    r = min(cand, key=lambda r: (r["summary"][tk]["median"], -r["summary"][tk]["reached"], r["family"]))
    return {"family": r["family"], "variant": r["variant"], "median": r["summary"][tk]["median"],
            "reached": r["summary"][tk]["reached"], "runs": r["summary"][tk]["runs"]}


def compare(rows, tol):
    rec = lambda r: r["kind"] == "recipe"                                                           # noqa: E731
    out = {"spectral": best_of(rows, tol, lambda r: rec(r) and r["orth"] in ("svd", "ns5")),
           "control": best_of(rows, tol, lambda r: rec(r) and r["orth"] in ("frob", "sign")),
           "control_no_momentum": best_of(rows, tol, lambda r: rec(r) and r["orth"] in ("frob", "sign")
                                          and not r["nesterov"]),
           "euclid": best_of(rows, tol, lambda r: r["kind"] == "reference" and r["group"] == "euclid"),
           "nuclear_scaled": best_of(rows, tol, lambda r: r["kind"] == "reference" and r["group"] == "spectral")}

    def ratio(a, b):
        if out[a] is None or out[b] is None or not math.isfinite(out[a]["median"]):
            return None
        return out[a]["median"] / out[b]["median"] if math.isfinite(out[b]["median"]) else 0.0
    out["ratios"] = {"spectral/control": ratio("spectral", "control"),
                     "spectral/control_no_momentum": ratio("spectral", "control_no_momentum"),
                     "spectral/euclid": ratio("spectral", "euclid"),
                     "spectral/nuclear_scaled": ratio("spectral", "nuclear_scaled")}
    out["note"] = "a ratio is 0 when the denominator never reaches the target (infinite median), None when the " \
                  "numerator does not or a term is missing"
    return out


def part_A(tag, summary, insts, batches, variants, refs, jobs, quick):
    sec = summary.setdefault("A", {})
    for key, inst in insts.items():
        s, passes, tol = inst["spec"], inst["passes"], inst["tol"]
        fams, meta = [], {}
        for b in batches:
            if b > s["n"] // 2:
                continue
            for f in recipe_families(b, quick):
                m = recipe_meta(f)
                if variants is None or m["variant"] in variants:
                    fams.append(f)
                    meta[f["name"]] = m
            for f in (resolved_references(b, quick) if refs else []):
                fams.append(f)
                meta[f["name"]] = reference_meta(f)
        if not fams:
            continue
        print(f"\n=== A, {key}: {C.spec_tag(s)}, budget {passes} passes, target {tol:g}, {len(fams)} families",
              flush=True)
        nm = norms(s)
        recs = C.tune_and_confirm([s], fams, passes, tol, f"{tag}_{key}", n_jobs=jobs)
        e = sec.setdefault(key, {"cells": {}})
        e.update(spec=s, passes=passes, tol=tol, norms=nm, code=C.code_fingerprint())
        for r in recs:
            m = meta[r["family"]]
            cell = e["cells"].setdefault(str(m["batch"]), {"rows": {}})
            cell["rows"][r["family"]] = row_of(r, tol, nm["D0"], m)
        for b, cell in e["cells"].items():
            cell["compare"] = compare(list(cell["rows"].values()), tol)
            cell["lr_grids"] = {RECIPE_LABEL[o]: recipe_lr_grid(o, int(b)) for o in RECIPE_LABEL}
        e["q_grid"] = [0.6, 0.8, 0.9] if quick else list(COOLDOWNS)
        save(tag, summary, quick)
        for b in sorted(e["cells"], key=int):
            cell = e["cells"][b]
            print(f"  b = {b}: passes to {tol:g}, median [min, max] over the confirmation seeds, seeds reaching it, "
                  f"selected point, edges")
            for row in sorted(cell["rows"].values(), key=lambda r: (r["kind"] != "recipe", r["family"])):
                print(f"    {row['family']:40s} {fmt_row(row, tol)}", flush=True)
            cmp_ = cell["compare"]["ratios"]
            print("    ratios of medians: " + ", ".join(f"{k} = " + ("n.a." if v is None else f"{v:.2f}")
                                                         for k, v in cmp_.items()))
            for v, r in cell.get("part_B", {}).get("rows", {}).items():
                print(f"    part B, {v:8s} {r['median']:.1f} [{r['min']:.1f}, {r['max']:.1f}] {r['reached']}/{r['runs']}"
                      f"  ({C.fmt_point({k: x for k, x in r['best'].items() if k != 'schedule'})})")
            if cell.get("part_B", {}).get("ratios"):
                print("    part B over the controls: "
                      + ", ".join(f"{k} = {v:.2f}" for k, v in cell["part_B"]["ratios"].items()))


# ----------------------------------------------------------------------------------------------
# part B: batch one on the six instances
# ----------------------------------------------------------------------------------------------
def part_B(tag, summary, insts, jobs, quick):
    sec = summary.setdefault("B", {})
    m1 = C.load_json("M1_quick_summary" if quick else "M1_summary") or {}
    for key, inst in insts.items():
        s, passes, tol = inst["spec"], inst["passes"], inst["tol"]
        fams = batch_one_recipes(quick)
        meta = {f["name"]: recipe_meta(f) for f in fams}
        print(f"\n=== B, {key}: {C.spec_tag(s)}, budget {passes} passes, target {tol:g}", flush=True)
        nm = norms(s)
        recs = C.tune_and_confirm([s], fams, passes, tol, f"{tag}_{key}", n_jobs=jobs)
        rows = {r["family"]: row_of(r, tol, nm["D0"], meta[r["family"]]) for r in recs}
        fixed = {}
        for f in fams:                                   # same tasks as a confirmation of the fixed point
            _, tasks = C.make_tasks(s, f, passes, C.CONFIRM_SEEDS, points=[FIXED_POINT],
                                    stop_rel=C.CONFIRM_STOP * tol)
            res = C.run_tasks(tasks, f"{tag}_{key}_confirm", jobs)
            fixed[f["name"]] = {"summary": passes_summary(res), "final_rel": stats([r["final_rel"] for r in res]),
                                "confirm_passes": [r["passes_to"][f"{tol:g}"] for r in res],
                                "stop_level": {"reached": int(sum(r.get("stopped") == "stop_dist2" for r in res)),
                                               "runs": len(res)},
                                "dist2_over_lr2": band(res, nm["D0"])}
        ref = {}
        for r in (m1.get(key, {}).get("rows", []) if m1.get(key, {}).get("spec") == s else []):
            if r["family"] in M1_FAMILIES:
                ref[r["family"]] = {**r["summary"][f"{tol:g}"], "best": r["best"], "edge": r["edge"]}
        sec[key] = {"spec": s, "passes": passes, "tol": tol, "norms": nm, "code": C.code_fingerprint(), "rows": rows,
                    "q_grid": fams[0]["grid"]["lr_decay"], "lr_grid": fams[0]["grid"]["lr"],
                    "fixed_point": {"point": FIXED_POINT, "seeds": list(C.CONFIRM_SEEDS), "rows": fixed},
                    "M1": {"source": "M1_summary.json (same instance, budget and target; factor-2 step grids)",
                           "rows": ref}}
        save(tag, summary, quick)
        print(f"  passes to {tol:g}, median [min, max] over the confirmation seeds, seeds reaching it")
        for name, row in rows.items():
            bd = row["dist2_over_lr2"]
            print(f"    {name:40s} {fmt_row(row, tol)}   dist2/lr2 "
                  + (f"[{bd['min']:.3g}, {bd['max']:.3g}] on {bd['runs']} runs" if bd["runs"] else "n.a."))
        for name, fx in fixed.items():
            sm = fx["summary"][f"{tol:g}"]
            print(f"    {name + ' at q=0.8, lr0=0.1':40s} {C.fmt_passes(fx['summary'], tol):>20s} "
                  f"{str(sm['reached']) + '/' + str(sm['runs']):>6s}")
        for name, r in ref.items():
            print(f"    M1 {name:37s} {r['median']:8.1f} [{r['min']:.1f}, {r['max']:.1f}] "
                  f"{str(r['reached']) + '/' + str(r['runs']):>6s}")


# ----------------------------------------------------------------------------------------------
# part C: the law and its controls
# ----------------------------------------------------------------------------------------------
def law_depth(res, D0):
    """Level, band and depth of the schedule law for one run without stop rule (see definitions()['law'])."""
    out = {"min_rel": res["min_rel"], "final_rel": res["final_rel"], "on_schedule": False}
    t = ratio_trace(res, D0)
    if t is None:
        return out
    p, rel, ratio = t
    win = (rel >= LAW_WINDOW[0]) & (rel <= LAW_WINDOW[1])
    if not win.any() or rel.min() > LAW_WINDOW[0]:
        return out
    level = float(np.median(ratio[win]))
    k0 = int(np.argmax(rel <= LAW_WINDOW[1]))
    out["depth_rel_in_range"] = float(rel[k0 + int(np.nonzero(ratio[k0:] <= ratio[win].max())[0][-1])])
    last = {}
    for fac in LAW_FACTORS:                 # last record within the factor (half of the window is below the level)
        last[fac] = k0 + int(np.nonzero(ratio[k0:] <= fac * level)[0][-1])
        out[f"depth_rel_x{fac:g}"] = float(rel[last[fac]])
    k1 = last[LAW_FACTORS[0]]
    x = ratio[(p >= 0.5 * p[k1]) & (p <= p[k1])]
    out.update(on_schedule=True, level=level, depth_passes=float(p[k1]), band_min=float(x.min()),
               band_max=float(x.max()))
    return out


def part_C(tag, summary, insts, jobs, quick):
    sec = summary.setdefault("C", {})
    fam_P, fam_M = batch_one_recipes(quick)
    for key, inst in insts.items():
        rows = summary.get("B", {}).get(key, {}).get("rows")
        if not rows or summary["B"][key].get("spec") != inst["spec"]:
            print(f"\n=== C, {key}: skipped, part B has not been run on this instance")
            continue
        s, passes = inst["spec"], inst["passes"]
        n = s["n"]
        nm = norms(s)
        pt_P, pt_M = rows[fam_P["name"]]["best"], rows[fam_M["name"]]["best"]
        q, lr0 = pt_M["lr_decay"], pt_M["lr"]
        deep = [("polar, no momentum", {**fam_P["base"], **pt_P}),
                ("ns5 + Nesterov (Muon's recipe)", {**fam_M["base"], **pt_M}),
                ("ns5 + Nesterov, eps = 1e-30", {**fam_M["base"], "orth_kw": {"eps": 1e-30}, **pt_M}),
                # the selected point of Muon's recipe sits on the cliff (some seeds fall off): also the fixed point
                ("ns5 + Nesterov at q = 0.8, lr_0 = 0.1", {**fam_M["base"], **FIXED_POINT}),
                ("ns5 + Nesterov, eps = 1e-30, at q = 0.8, lr_0 = 0.1",
                 {**fam_M["base"], "orth_kw": {"eps": 1e-30}, **FIXED_POINT})]
        controls = []
        for b in NO_TABLE_BATCHES:
            if b > n // 2:
                continue
            controls.append((f"no table (minibatch gradient), ns5 + Nesterov, b={b}", "vr",
                             {**fam_M["base"], "estimator": "sgd", "batch": b, **pt_M}))
            for name, kw in (("MARS-M", dict(g=0.025, clip=1.0)), ("STORM-Muon", dict(g=1.0))):
                controls.append((f"{name}, b={b}", "mvr",        # 2 b gradients per step: n / (2 b) steps per pass
                                 dict(batch=b, lr=lr0, beta=0.95, same_batch=True, orth="ns5",
                                      schedule=["geometric", q, n / (2.0 * b)], **kw)))
        tasks, index = [], []
        for label, cfg in deep:
            for seed in LAW_SEEDS:
                tasks.append({"spec": s, "method": "vr", "cfg": cfg, "passes": passes, "seed": seed, "stop_rel": 0.0})
                index.append(("deep", label))
        for label, method, cfg in controls:
            for seed in LAW_SEEDS:
                tasks.append({"spec": s, "method": method, "cfg": cfg, "passes": passes, "seed": seed,
                              "stop_rel": 0.0})
                index.append(("control", label))
        print(f"\n=== C, {key}: {C.spec_tag(s)}, {passes} passes without stop rule, seeds {list(LAW_SEEDS)}",
              flush=True)
        res = C.run_tasks(tasks, f"{tag}_law", jobs)
        out = {"spec": s, "passes": passes, "seeds": list(LAW_SEEDS), "norms": nm, "code": C.code_fingerprint(),
               "points": {"polar, no momentum": pt_P, "ns5 + Nesterov (Muon's recipe)": pt_M},
               "controls_schedule": pt_M, "deep": {}, "controls": {}}
        for label, cfg in deep:
            runs = [law_depth(r, nm["D0"]) for (kind, lb), r in zip(index, res) if kind == "deep" and lb == label]
            on = [r for r in runs if r["on_schedule"]]
            e = {"cfg": cfg, "runs": runs, "on_schedule": len(on), "min_rel": stats([r["min_rel"] for r in runs])}
            if on:
                e.update(depth_rel_in_range=stats([r["depth_rel_in_range"] for r in on]),
                         depth_rel_x3=stats([r["depth_rel_x3"] for r in on]),
                         depth_rel_x10=stats([r["depth_rel_x10"] for r in on]),
                         level=stats([r["level"] for r in on]),
                         band={"min": min(r["band_min"] for r in on), "max": max(r["band_max"] for r in on),
                               "width_median": float(np.median([r["band_max"] / r["band_min"] for r in on]))})
            out["deep"][label] = e
        for label, method, cfg in controls:
            v = [r["final_rel"] for (kind, lb), r in zip(index, res) if kind == "control" and lb == label]
            out["controls"][label] = {"method": method, "cfg": cfg, "final_rel": stats(v), "values": v}
        sec[key] = out
        save(tag, summary, quick)
        print(f"  {'configuration':52s} {'on sched.':>9s} {'level C':>10s} {'band of dist2/lr2':>22s} "
              f"{'in range to':>12s} {'x3 to':>9s} {'x10 to':>9s} {'lowest rel. dist2: median [min, max]':>38s}")
        for label, e in out["deep"].items():
            mr = e["min_rel"]
            low = '%.1e [%.1e, %.1e]' % (mr['median'], mr['min'], mr['max'])
            if e["on_schedule"]:
                bd = e["band"]
                print(f"  {label:52s} {str(e['on_schedule']) + '/' + str(len(e['runs'])):>9s} "
                      f"{e['level']['median']:10.3g} {'[%.3g, %.3g]' % (bd['min'], bd['max']):>22s} "
                      f"{e['depth_rel_in_range']['median']:12.1e} {e['depth_rel_x3']['median']:9.1e} "
                      f"{e['depth_rel_x10']['median']:9.1e} {low:>38s}")
            else:
                print(f"  {label:52s} {'0/' + str(len(e['runs'])):>9s} {'':10s} {'':22s} {'':12s} {'':9s} {'':9s} "
                      f"{low:>38s}")
        print(f"  without a table, schedule {C.fmt_point({k: v for k, v in pt_M.items() if k != 'schedule'})}: "
              f"final relative squared distance, median [min, max]")
        for label, e in out["controls"].items():
            f = e["final_rel"]
            print(f"  {label:52s} {f['median']:.3g} [{f['min']:.3g}, {f['max']:.3g}]", flush=True)


# ----------------------------------------------------------------------------------------------
# part D: sampling order and memory of the table
# ----------------------------------------------------------------------------------------------
def schemes(b):
    return [("iid", "iid", 0), ("reshuffle", "reshuffle", 0), ("shuffle_once", "shuffle_once", 0),
            (f"iid + {b} extra refreshes (m = b)", "iid", b), (f"iid + {3 * b} extra refreshes (m = 3b)", "iid", 3 * b)]


def order_task(s, b, rho, order, m, seed, w_op):
    """Task of part D and its schedule: exact polar factor, no momentum, T iterations with lr_T = 1e-5 lr_0."""
    n = s["n"]
    q = math.sqrt(rho)
    T = int(math.ceil(math.log(1.0 / TOTAL_DECAY) / math.log(1.0 / q) * n / b))
    lr0 = float(f"{PATH_FACTOR * w_op * (1.0 - q ** (b / n)) / (1.0 - TOTAL_DECAY):.6g}")
    cfg = dict(estimator="saga", batch=int(b), length="const", orth="svd", schedule="geometric", lr_decay=q, lr=lr0)
    if order != "iid":
        cfg["order"] = order
    if m:
        cfg["refresh_extra"] = int(m)
    passes = round((n + T * (b + m)) / n, 6)                     # budget = n + T (b + m) gradients exactly
    return {"spec": s, "method": "vr", "cfg": cfg, "passes": passes, "seed": int(seed), "stop_rel": 0.0}, \
        {"q": q, "lr0": lr0, "iters": T, "passes": passes}


def part_D(tag, summary, insts, batches, seeds, jobs, quick, control):
    sec = summary.setdefault("D", {})
    tk = f"{ORDER_TOL:g}"
    for key, inst in insts.items():
        s = inst["spec"]
        n = s["n"]
        nm = norms(s)
        rhos = (RHOS_CONTROL if key == control else RHOS) if not quick else (0.61, 0.368, 0.18, 0.05)
        low = () if quick else RHOS_LOW
        grid_of = lambda b, m: rhos + (low if m == 3 * b else ())                                   # noqa: E731
        cells = [(b, name, order, m) for b in batches if b + 3 * b <= n for (name, order, m) in schemes(b)]
        tasks, index = [], []
        for b, name, order, m in cells:
            for rho in grid_of(b, m):
                for seed in seeds:
                    t, info = order_task(s, b, rho, order, m, seed, nm["W_star_op"])
                    tasks.append(t)
                    index.append((b, name, rho, info))
        print(f"\n=== D, {key}: {C.spec_tag(s)}, b in {[b for b in batches if 4 * b <= n]}, seeds {list(seeds)}, "
              f"{len(tasks)} runs", flush=True)
        res = C.run_tasks(tasks, f"{tag}_order", jobs)
        e = sec.setdefault(key, {"cells": {}})
        e.update(spec=s, norms=nm, rho_grid=list(rhos), rho_grid_extension_m3b=list(low), rho_ref=rhos[0],
                 total_decay=TOTAL_DECAY, tol=ORDER_TOL, code=C.code_fingerprint(),
                 role="control (conditioning-limited)" if key == control else "memory-limited")
        for b, name, order, m in cells:
            by_rho = []
            for rho in grid_of(b, m):
                sel = [(info, r) for (bb, nn, rr, info), r in zip(index, res) if (bb, nn, rr) == (b, name, rho)]
                info = sel[0][0]
                Cs = []
                for _, r in sel:
                    t = ratio_trace(r, nm["D0"])
                    Cs.append(float(t[2][-1]) if t is not None else float("inf"))
                pt = [r["passes_to"][tk] for _, r in sel]
                by_rho.append({"rho": rho, **info, "C": Cs, "passes_to": pt,
                               "reached": int(np.isfinite(pt).sum()), "runs": len(pt)})
            ref = float(np.median(by_rho[0]["C"]))
            for x in by_rho:
                x["on_schedule"] = int(sum(c <= TRACK_FACTOR * ref for c in x["C"]))
            ok = [x["on_schedule"] >= TRACK_QUORUM * x["runs"] for x in by_rho]
            star = min((x["rho"] for x, o in zip(by_rho, ok) if o), default=None)
            mono = None
            for x, o in zip(by_rho, ok):                         # the grid is decreasing
                if not o:
                    break
                mono = x["rho"]
            full = [x for x in by_rho if x["reached"] == x["runs"]]
            fastest = None
            if full:
                x = min(full, key=lambda x: float(np.median(x["passes_to"])))
                fastest = {"rho": x["rho"], **stats(x["passes_to"])}
            cell = e["cells"].setdefault(str(b), {"schemes": {}})
            cell["seeds"] = list(seeds)
            cell["schemes"][name] = {"order": order, "refresh_extra": m,
                                     "memory_factor": (1.0 - (b + m) / n) ** (n / b) if order == "iid" else None,
                                     "exp_minus_1_plus_m_over_b": math.exp(-(1.0 + m / b)) if order == "iid" else None,
                                     "reference_C": ref, "rho_star": star, "rho_star_monotone": mono,
                                     "fastest": fastest, "by_rho": by_rho}
        save(tag, summary, quick)
        print(f"  {'b':>3s} {'scheme':34s} {'memory':>7s} {'rho_*':>6s} {'(mono)':>6s} {'C at rho_ref':>12s}   "
              f"fastest rho with every seed at {ORDER_TOL:g}: passes median [min, max]   seeds on schedule by rho")
        for b in sorted(e["cells"], key=int):
            for name, c in e["cells"][b]["schemes"].items():
                f = c["fastest"]
                print(f"  {b:>3s} {name:34s} "
                      f"{'' if c['memory_factor'] is None else format(c['memory_factor'], '.3f'):>7s} "
                      f"{'-' if c['rho_star'] is None else format(c['rho_star'], 'g'):>6s} "
                      f"{'-' if c['rho_star_monotone'] is None else format(c['rho_star_monotone'], 'g'):>6s} "
                      f"{c['reference_C']:12.3g}   "
                      + ("-" if f is None else f"rho {f['rho']:g}: {f['median']:.1f} [{f['min']:.1f}, {f['max']:.1f}]")
                      + "   " + " ".join(str(x["on_schedule"]) for x in c["by_rho"]), flush=True)


# ----------------------------------------------------------------------------------------------
def main():
    argv = sys.argv
    quick = "--quick" in argv

    def opt(name, default=None):
        return argv[argv.index(name) + 1] if name in argv else default
    only = opt("--only")
    only = only.split(",") if only else None
    jobs = int(opt("--jobs")) if opt("--jobs") else None
    parts = opt("--parts", "A,B,C,D").split(",")
    variants = opt("--variants")
    variants = variants.split(",") if variants else None
    refs = "--no-refs" not in argv
    all_insts = QUICK if quick else INSTANCES
    pick = lambda keys: {k: all_insts[k] for k in keys if k in all_insts and (only is None or k in only)}  # noqa: E731
    batches = tuple(int(x) for x in opt("--batches", "1,8" if quick else ",".join(map(str, BATCHES))).split(","))
    dbatches = tuple(int(x) for x in opt("--dbatches", "5" if quick else ",".join(map(str, ORDER_BATCHES))).split(","))
    dseeds = ORDER_SEEDS[:int(opt("--dseeds", "2" if quick else str(len(ORDER_SEEDS))))]
    tag = "M9_quick" if quick else "M9"
    summary = C.load_json(f"{tag}_summary") or {}
    t0 = time.perf_counter()
    for part in parts:
        if part == "A":
            part_A(tag, summary, pick(("ls-orth",) if quick else MAPS_ON), batches, variants, refs, jobs, quick)
        elif part == "B":
            part_B(tag, summary, pick(tuple(all_insts)), jobs, quick)
        elif part == "C":
            part_C(tag, summary, pick(("ls-orth",) if quick else LAW_ON), jobs, quick)
        elif part == "D":
            control = None if quick else ORDER_CONTROL
            part_D(tag, summary, pick(("soft-mem",) if quick else ORDER_ON + (ORDER_CONTROL,)), dbatches, dseeds,
                   jobs, quick, control)
        else:
            raise ValueError(f"unknown part {part!r}")
    print(f"\nM9 done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
