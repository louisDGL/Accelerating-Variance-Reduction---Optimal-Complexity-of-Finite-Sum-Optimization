"""M5b: experiment M5 (batch size, Euclidean versus orthogonalized SAGA) at resolved steps.

M5 tuned its three families on factor-2 step grids. On soft-cond and ls-cond the fast steps of the nuclear-scaled
polar step form a window narrower than a factor 2: two eighths of an octave below the selected step the method is
several times slower or misses the target on some seeds, and at b = 64 the steps one or two eighths above it lose
seeds or diverge (part `window` below; the numbers are in the summary). A factor-2 grid can step over this window, and
M5 then reports the slow regime that borders it from below. This script repeats the comparison of M5 (same instances,
budgets, target, batch sizes and protocol; the families of exp_M5_batch.py, which is not modified)
    SAGA b            Euclidean minibatch SAGA
    B-SAGA b          Euclidean minibatch SAGA with damped innovation
    SAGA-Muon b       joint placement, nuclear-scaled exact polar step of the minibatch SAGA estimator
on step grids of ratio 2^(1/4), and measures how much of the conclusion of M5 depends on the grid. Steps are indexed
in eighths of an octave around the step c5 that M5 selected for the same (instance, family, batch):
c(j) = c5 * 2^(j / 8), |j| <= 16 (a factor 4 on each side, wide enough that no selection should sit on an edge; the
tables flag the selections that do). For every (instance, data seed, batch), the parts are
    fine      the three families tuned and confirmed with the protocol of common.py on the grid j even (17 steps;
              B-SAGA: theta in M5's list plus 0.35, between 0.3 and 0.6, where the best damping sits at b = 64);
    factor2   control: the same protocol on the sub-grid j = 0 mod 8 (the powers of two, M5's theta list), which
              shares its runs with `fine` through the cache. It should reproduce M5: these tasks are M5's own tasks;
    shifted   the same protocol on the grid shifted by half a step (j odd, 16 steps): does the result depend on the
              grid containing one lucky point?
    window    the step selected on the fine grid times 2^(-1/4), 2^(-1/8), 2^(1/8), 2^(1/4), on the confirmation
              seeds (SAGA-Muon and B-SAGA): width of the fast window, number of seeds that reach the target.
The reported ratio is the one of the M5 table of make_digest.py: median passes of SAGA-Muon over the smallest median
of the Euclidean families that reach the target on every confirmation seed (none if SAGA-Muon or no Euclidean family
does). Since the selection of the Euclidean baseline is itself noisy, the summary also gives SAGA-Muon over the best
confirmed Euclidean point of the cell, whatever the grid or the window offset it comes from ('conservative'). One
iteration costs b gradients and, for SAGA-Muon, one factorization: n / b factorizations per pass.

Default plan: data seed 0, the three instances, b in (1, 4, 16, 64), all parts; data seeds 1 and 2 (other draws of the
data; grids centred on the same steps), soft-cond and ls-cond, b in (16, 64), parts fine and factor2. The options
select a part of the plan (or cells outside it); the summary keeps the cells already computed and records, in
'about' -> 'missing_from_default_plan', what a run of the default plan would still have to compute.

Outputs: muon/results/M5b_<key>[_s<data seed>]_{screen,tune,confirm,window}.pkl (caches, keyed by the task),
muon/results/M5b_summary.json (self-describing: 'about' explains every key), tables on stdout.
Usage: python muon/experiments/exp_M5b_batch_fine.py [--only key1,key2] [--batches 16,64] [--data-seeds 0,1]
       [--parts fine,factor2,window,shifted] [--jobs N] [--quick] [--tables]
(--tables: print the tables of the existing summary and run nothing. Do not run two copies at the same time: they
share the summary and the caches. Needs muon/results/M5_summary.json for the centres of the grids.)
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np

import common as C
from common import over
from exp_M5_batch import BATCHES, INSTANCES, QUICK

KINDS = ("SAGA", "B-SAGA", "SAGA-Muon")
EUCLID = ("SAGA", "B-SAGA")
UNIT = {"SAGA": "R2", "B-SAGA": "R2", "SAGA-Muon": "R2_op"}
BASE = {"SAGA": dict(estimator="saga", placement="euclid"), "B-SAGA": dict(estimator="saga", placement="euclid"),
        "SAGA-Muon": dict(estimator="saga")}                 # the bases of exp_M5_batch.families
THETAS_M5 = [0.6, 0.3, 0.2, 0.1, 0.05]                       # damping list of M5 (factor-2 control)
THETAS = [0.6, 0.35, 0.3, 0.2, 0.1, 0.05]                    # fine and shifted grids
HALF = 16                                                    # half-width of the grids, in eighths of an octave
GRIDS = {"fine": range(-HALF, HALF + 1, 2), "factor2": range(-HALF, HALF + 1, 8), "shifted": range(-HALF + 1, HALF, 2)}
OFFSETS = (-2, -1, 1, 2)                                     # window: selected step times 2^(j / 8)
WINDOW_KINDS = ("SAGA-Muon", "B-SAGA")
PARTS = ("fine", "factor2", "window", "shifted")
PLAN = {0: dict(instances=("soft-cond", "ls-cond", "ls-orth"), batches=BATCHES, parts=PARTS),
        1: dict(instances=("soft-cond", "ls-cond"), batches=(16, 64), parts=("fine", "factor2")),
        2: dict(instances=("soft-cond", "ls-cond"), batches=(16, 64), parts=("fine", "factor2"))}
QUICK_CENTRE = {1: {"SAGA": 2.0, "B-SAGA": 4.0, "SAGA-Muon": 0.0625},     # smoke test only: M5's steps on ls-orth
                4: {"SAGA": 8.0, "B-SAGA": 16.0, "SAGA-Muon": 0.5},
                16: {"SAGA": 16.0, "B-SAGA": 64.0, "SAGA-Muon": 4.0}}
BOOT = dict(resamples=4000, seed=12345, quantiles=(0.05, 0.95))


def step(kind, c5, j):
    """Step constant c5 * 2^(j / 8) in the unit of the family. For j = 0 mod 8 this is exactly a grid value of M5."""
    return over(UNIT[kind], c5 * 2.0 ** (j / 8.0))


def family(kind, b, c5, grid):
    """Family of M5 (same name, method and base) on one of the grids of this experiment, centred on c5."""
    g = {}
    if kind == "B-SAGA":
        g["theta"] = list(THETAS_M5 if grid == "factor2" else THETAS)
    g["gamma"] = [step(kind, c5, j) for j in GRIDS[grid]]
    return dict(name=f"{kind} b={b}", method="vr", base=dict(BASE[kind], batch=b), grid=g)


def cache_name(tag, key, sd):
    return f"{tag}_{key}" + ("" if sd == 0 else f"_s{sd}")


def stages():
    """(cache name, instance, families, budget, target) of every tune-then-confirm call of the default plan: the
    arguments of audit_cache.stage_tasks, which rebuilds the tasks of the caches <name>_{screen,tune,confirm}."""
    m5 = C.load_json("M5_summary")
    for sd, plan in PLAN.items():
        for key in plan["instances"]:
            inst = INSTANCES[key]
            for b in plan["batches"]:
                for grid in (p for p in GRIDS if p in plan["parts"]):
                    fams = [family(k, b, centre(m5, key, b)[k], grid) for k in KINDS]
                    yield cache_name("M5b", key, sd), dict(inst["spec"], seed=sd), fams, inst["passes"], inst["tol"]


def window_tasks():
    """(cache name, task) of every run that the part `window` of the default plan may have made: the confirmation
    runs of all the steps c5 * 2^(j / 8), |j| <= HALF + 2, of the two families (the selected step is not known in
    advance), in the cache <name>_window."""
    m5 = C.load_json("M5_summary")
    for sd, plan in PLAN.items():
        for key in (plan["instances"] if "window" in plan["parts"] else ()):
            inst = INSTANCES[key]
            s = dict(inst["spec"], seed=sd)
            for b in plan["batches"]:
                for kind in WINDOW_KINDS:
                    c5 = centre(m5, key, b)[kind]
                    fam = family(kind, b, c5, "fine")
                    fam["grid"]["gamma"] = [step(kind, c5, j) for j in range(-HALF - 2, HALF + 3)]
                    for t in C.make_tasks(s, fam, inst["passes"], C.CONFIRM_SEEDS,
                                          stop_rel=C.CONFIRM_STOP * inst["tol"])[1]:
                        yield cache_name("M5b", key, sd) + "_window", t


def runs_fingerprint():
    """SHA-1 of the code that determines the cached runs of this experiment: muon/lab/*.py and common.py, i.e.
    common.code_fingerprint() without families.py, which this script does not use."""
    import hashlib
    import os
    lab = os.path.join(C.MUON, "lab")
    h = hashlib.sha1()
    for path in sorted(os.path.join(lab, f) for f in os.listdir(lab) if f.endswith(".py")) + [
            os.path.join(C.HERE, "common.py")]:
        with open(path, "rb") as f:
            h.update(f.read().replace(b"\r\n", b"\n"))
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------
# M5 as reference, ratios
# ----------------------------------------------------------------------------------------------
def m5_rows(m5, key, b):
    return {r["kind"]: r for r in m5[key]["rows"] if r["batch"] == b}


def centre(m5, key, b):
    """Step constants selected by M5 for (instance, batch), in the units of UNIT: the centres of the grids."""
    rows = m5_rows(m5, key, b)
    return {k: float(rows[k]["best"]["gamma"]["c"]) for k in KINDS}


def ratio(stats):
    """SAGA-Muon against the best Euclidean family. stats = {kind: {'median', 'reached', 'runs', optional 'passes'}}.
    Returns {'ratio', 'best_euclid', 'boot'}: the ratio of medians (None unless both reach the target on every seed),
    the Euclidean family of the denominator, and the 5-95% interval of the ratio under a paired bootstrap of the
    confirmation seeds (the two methods draw the same batches for a given seed)."""
    full = [k for k in EUCLID if k in stats and stats[k]["reached"] == stats[k]["runs"]]
    out = {"ratio": None, "best_euclid": None, "boot": None}
    if not full:
        return out
    be = min(full, key=lambda k: stats[k]["median"])
    out["best_euclid"] = be
    mu = stats.get("SAGA-Muon")
    if mu is None or mu["reached"] < mu["runs"]:
        return out
    out["ratio"] = mu["median"] / stats[be]["median"]
    if "passes" in mu and "passes" in stats[be]:
        a, e = np.asarray(mu["passes"], dtype=float), np.asarray(stats[be]["passes"], dtype=float)
        idx = np.random.default_rng(BOOT["seed"]).integers(0, len(a), size=(BOOT["resamples"], len(a)))
        bs = np.median(a[idx], axis=1) / np.median(e[idx], axis=1)
        out["boot"] = [float(q) for q in np.quantile(bs, BOOT["quantiles"])]
    return out


def seeds_summary(res, tk):
    """Median, range and counts of the passes to the target over a list of confirmation runs."""
    v = np.array([r["passes_to"][tk] for r in res], dtype=float)
    return {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()),
            "reached": int(np.isfinite(v).sum()), "runs": len(v), "diverged": int(sum(r["diverged"] for r in res)),
            "passes": v.tolist()}


def comparison(recs, tol):
    """Rows of one tune-then-confirm call (one per family) and the ratio SAGA-Muon / best Euclidean."""
    tk = f"{tol:g}"
    rows, stats = {}, {}
    for r in recs:
        kind = r["family"].split(" b=")[0]
        passes = [c["passes_to"][tk] for c in r["confirm"]]
        rows[kind] = {"family": r["family"], "best": r["best"], "edge": r["edge"], "best_score": r["best_score"],
                      "finalists": r["finalists"], "n_points": r["n_points"], "n_kept": r["n_kept"],
                      "summary": r["summary"], "confirm_passes": passes, "final_rel": r["final_rel"],
                      "stop_level": r["stop_level"], "diverged": r["diverged"]}
        if kind == "SAGA-Muon":                              # the step window as the tuning seeds see it
            rows[kind]["tune_curve"] = {"c": [t["point"]["gamma"]["c"] for t in r["tune"]],
                                        "screen_rel": [t["screen_rel"] for t in r["tune"]],
                                        "score": [t["score"] if t["kept"] else None for t in r["tune"]]}
        stats[kind] = dict(r["summary"][tk], passes=passes)
    return dict(rows=rows, **ratio(stats))


def m5_reference(m5, key, b, tol):
    """The rows of M5 for (instance, batch) and its ratio, computed with the same rule."""
    tk = f"{tol:g}"
    rows = m5_rows(m5, key, b)
    out = {"rows": {k: {"best": rows[k]["best"], "edge": rows[k]["edge"], "summary": {tk: rows[k]["summary"][tk]}}
                    for k in KINDS}}
    out.update(ratio({k: rows[k]["summary"][tk] for k in KINDS}))
    out.pop("boot")                                          # the per-seed passes are not in M5_summary.json
    return out


def reproduces(m5ref, ctrl, tk):
    """Factor-2 control against M5: same selected point and same confirmation median, per family."""
    out = {}
    for k in KINDS:
        a, c = m5ref["rows"][k], ctrl["rows"][k]
        ma, mc = a["summary"][tk]["median"], c["summary"][tk]["median"]
        both_inf = not math.isfinite(ma) and not math.isfinite(mc)
        out[k] = {"same_point": a["best"] == c["best"], "m5_median": ma, "control_median": mc,
                  "rel_diff": 0.0 if both_inf else (abs(mc - ma) / ma if math.isfinite(ma) and math.isfinite(mc)
                                                    else float("inf"))}
    return out


def conservative(cell, tk):
    """Guard against an unlucky selection of the Euclidean baseline: the smallest median among all the Euclidean
    confirmations of the cell that reach the target on every seed (selected points of the three grids, window
    points), and SAGA-Muon of the fine and of the shifted grid over it (None under the rule of `ratio`)."""
    cands = []
    for grid in GRIDS:
        if grid in cell:
            cands += [(cell[grid]["rows"][k]["summary"][tk], f"{grid}/{k}") for k in EUCLID]
    for k in EUCLID:
        cands += [(e, f"window/{k}/{j}") for j, e in cell.get("window", {}).get(k, {}).items()]
    cands = [(sm["median"], src) for sm, src in cands if sm["reached"] == sm["runs"]]
    if not cands:
        return None
    med, src = min(cands)
    out = {"euclid_median": med, "euclid_from": src}
    for grid in ("fine", "shifted"):
        if grid in cell:
            mu = cell[grid]["rows"]["SAGA-Muon"]["summary"][tk]
            out[grid] = mu["median"] / med if mu["reached"] == mu["runs"] else None
    return out


# ----------------------------------------------------------------------------------------------
# one cell
# ----------------------------------------------------------------------------------------------
def run_cell(tag, key, inst, sd, b, parts, cen, jobs):
    """Run the requested parts of the cell (instance, data seed, batch); returns the dict stored in the summary."""
    s = dict(inst["spec"], seed=int(sd))
    passes, tol = inst["passes"], inst["tol"]
    tk = f"{tol:g}"
    name = cache_name(tag, key, sd)
    cell = {"batch": int(b), "factorizations_per_pass": s["n"] / b, "centre": cen, "code": C.code_fingerprint(),
            "code_runs": runs_fingerprint()}
    for grid in GRIDS:
        if grid in parts or (grid == "fine" and "window" in parts):
            recs = C.tune_and_confirm([s], [family(k, b, cen[k], grid) for k in KINDS], passes, tol, name, n_jobs=jobs)
            cell[grid] = comparison(recs, tol)
    if "window" in parts:
        fine = cell["fine"]
        ref = fine["rows"][fine["best_euclid"]]["summary"][tk]["median"] if fine["best_euclid"] else None
        tasks, plan = [], []
        for kind in WINDOW_KINDS:
            best = fine["rows"][kind]["best"]
            fam = family(kind, b, cen[kind], "fine")
            j0 = int(round(8 * math.log2(best["gamma"]["c"] / cen[kind])))
            assert step(kind, cen[kind], j0) == best["gamma"], (kind, best)
            for j in OFFSETS:
                pt = dict(best, gamma=step(kind, cen[kind], j0 + j))
                _, tks = C.make_tasks(s, fam, passes, C.CONFIRM_SEEDS, points=[pt], stop_rel=C.CONFIRM_STOP * tol)
                plan.append((kind, j, pt, len(tasks), len(tks)))
                tasks.extend(tks)
        res = C.run_tasks(tasks, name + "_window", jobs)
        win = {}
        for kind in WINDOW_KINDS:
            own = fine["rows"][kind]["summary"][tk]
            sel = {"c": fine["rows"][kind]["best"]["gamma"]["c"], "median": own["median"], "min": own["min"],
                   "max": own["max"], "reached": own["reached"], "runs": own["runs"],
                   "diverged": fine["rows"][kind]["diverged"], "passes": fine["rows"][kind]["confirm_passes"]}
            win[kind] = {"0": sel}
        for kind, j, pt, lo, m in plan:
            win[kind][str(j)] = dict(c=pt["gamma"]["c"], **seeds_summary(res[lo:lo + m], tk))
        for kind in WINDOW_KINDS:
            own = win[kind]["0"]["median"]
            for e in win[kind].values():
                ok = e["reached"] == e["runs"]
                e["over_selected"] = e["median"] / own if (ok and math.isfinite(own)) else None
                e["over_best_euclid"] = e["median"] / ref if (ok and ref) else None
        cell["window"] = win
    return cell


# ----------------------------------------------------------------------------------------------
# summary and tables
# ----------------------------------------------------------------------------------------------
def about(quick):
    return {
        "experiment": "M5b: experiment M5 (muon/experiments/exp_M5_batch.py, muon/results/M5_summary.json) at "
                      "resolved steps; written by muon/experiments/exp_M5b_batch_fine.py"
                      + (" --quick" if quick else ""),
        "families": {"SAGA": "Euclidean minibatch SAGA, gamma = c / R2",
                     "B-SAGA": "Euclidean minibatch SAGA with damped innovation theta, gamma = c / R2",
                     "SAGA-Muon": "joint placement, nuclear-scaled exact polar step (orth='svd') of the minibatch SAGA "
                                  "estimator, gamma = c / R2_op"},
        "metric": "passes (per-sample gradient evaluations / n, initial full gradient included) to a relative squared "
                  "Frobenius distance to the optimum below 'tol'; W_0 = 0; inf if not reached within 'passes'",
        "protocol": "common.tune_and_confirm: screening on 20% of the budget (tuning seed 0), tuning seeds 0-1, "
                    "finalists on tuning seeds 2-4, selected point on confirmation seeds 10-19",
        "seeds": {"tune": list(C.TUNE_SEEDS), "refine": list(C.REFINE_SEEDS), "confirm": list(C.CONFIRM_SEEDS)},
        "steps": "c(j) = c5 * 2^(j / 8), c5 = step constant selected by M5 for the same instance (data seed 0), "
                 "family and batch ('centre' of the cell)",
        "grids": {"fine": {"j": list(GRIDS["fine"]), "thetas_B-SAGA": THETAS},
                  "factor2": {"j": list(GRIDS["factor2"]), "thetas_B-SAGA": THETAS_M5},
                  "shifted": {"j": list(GRIDS["shifted"]), "thetas_B-SAGA": THETAS}},
        "layout": "instances -> <instance> -> <data seed> -> {spec, constants, passes (budget), tol, batches -> <b> -> "
                  "cell}",
        "cell": {
            "factorizations_per_pass": "n / b (SAGA-Muon: one factorization per iteration)",
            "centre": "c5 per family",
            "m5": "data seed 0 only: rows of M5_summary.json for this batch and M5's ratio",
            "fine, factor2, shifted": "one tune-then-confirm call each: rows -> <family> -> {best (selected point), "
                                      "edge (selected value on the edge of its grid), finalists, n_points, n_kept "
                                      "(points run at full budget), summary -> <tolerance> -> median/min/max/q1/q3/"
                                      "reached/runs over the confirmation seeds, confirm_passes (per seed, at tol), "
                                      "final_rel, stop_level, diverged, and for SAGA-Muon tune_curve (per grid step: "
                                      "relative squared distance after the screening run, tuning score on seeds 0-1 "
                                      "or null if not run at full budget)}, then ratio, best_euclid, boot",
            "ratio": "median passes of SAGA-Muon / smallest median among the Euclidean families (SAGA, B-SAGA) that "
                     "reach tol on every confirmation seed ('best_euclid'); null if SAGA-Muon does not reach tol on "
                     "every confirmation seed or no Euclidean family does",
            "boot": "5%-95% quantiles of the ratio under a paired bootstrap of the confirmation seeds (same resampled "
                    "seeds for both methods); algorithm-seed noise only, not the grid nor the data draw",
            "factor2_vs_m5": "data seed 0 only: per family, whether the factor-2 control selects M5's point, and M5's "
                             "and the control's confirmation medians (rel_diff = |control - M5| / M5)",
            "conservative": "euclid_median = smallest median among all the Euclidean confirmations of the cell that "
                            "reach tol on every seed (selected points of the fine, shifted and factor-2 grids, window "
                            "points; euclid_from names it as grid/family or window/family/j), then fine and shifted = "
                            "median of SAGA-Muon on that grid over euclid_median (null as for 'ratio'): the ratio when "
                            "the Euclidean baseline is credited with its best confirmed point",
            "window": "<family> -> <j> -> passes to tol on the confirmation seeds at the fine-grid selection times "
                      "2^(j / 8) ('0' = the selection itself): c, median, min, max, reached, runs, diverged, passes "
                      "(per seed), over_selected (median / median at the selected step) and over_best_euclid (median "
                      "/ median of the best Euclidean family of 'fine'); the last two are null unless every seed "
                      "reaches tol",
            "code": "common.code_fingerprint() when the cell was written (muon/lab/*.py, common.py, families.py)",
            "code_runs": "the same without families.py, which this experiment does not use (runs_fingerprint)",
        },
        "bootstrap": BOOT,
        "m5_code": None,
        "default_plan": {str(sd): {"instances": list(p["instances"]), "batches": list(p["batches"]),
                                   "parts": list(p["parts"])} for sd, p in PLAN.items()},
    }


def missing(summary):
    """Parts of the default plan that are not in the summary, as 'instance/data seed/b/part'."""
    out = []
    for sd, plan in PLAN.items():
        for key in plan["instances"]:
            node = summary.get("instances", {}).get(key, {}).get(str(sd), {}).get("batches", {})
            for b in plan["batches"]:
                out += [f"{key}/{sd}/{b}/{p}" for p in plan["parts"] if p not in node.get(str(b), {})]
    return out


def _f(v, fmt="{:.1f}"):
    return fmt.format(v) if (v is not None and math.isfinite(v)) else "inf"


def _r(v):
    return "-" if v is None else f"{v:.2f}"


def _cell(sm):
    """'median [min, max]' of the passes, with the number of seeds that reached the target when not all did."""
    if sm["reached"] == 0:
        return f"not reached (0/{sm['runs']})"
    txt = f"{_f(sm['median'])} [{_f(sm['min'])}, {_f(sm['max'])}]"
    return txt + (f" ({sm['reached']}/{sm['runs']})" if sm["reached"] < sm["runs"] else "")


def _row(row, tk):
    pt = row["best"]
    th = f"theta={pt['theta']:g}, " if "theta" in pt else ""
    return f"{_cell(row['summary'][tk])} ({th}c={pt['gamma']['c']:.4g})" + ("*" if row["edge"] else "")


def _ratio(cmp_):
    if cmp_ is None:
        return "not run"
    txt = _r(cmp_["ratio"])
    if cmp_.get("boot"):
        txt += f" [{cmp_['boot'][0]:.2f}, {cmp_['boot'][1]:.2f}]"
    return txt


def print_tables(summary):
    if not summary or "instances" not in summary:
        print("no summary yet")
        return
    for key, by_seed in summary["instances"].items():
        for sd, node in by_seed.items():
            tol = node["tol"]
            tk = f"{tol:g}"
            cells = sorted(node["batches"].values(), key=lambda c: c["batch"])
            print(f"\n=== {key}, data seed {sd}: {C.spec_tag(node['spec'])}, budget {node['passes']} passes, target "
                  f"{tol:g}, R2 = {node['constants']['R2']:.4g}, R2_op = {node['constants']['R2_op']:.4g}")
            for grid, title in (("fine", "2^(1/4) grid"), ("shifted", "2^(1/4) grid shifted by half a step"),
                                ("factor2", "factor-2 sub-grid (control: M5's grid)")):
                if not any(grid in c for c in cells):
                    continue
                print(f"  {title}: passes to {tol:g}, median [min, max] over the confirmation seeds (selected point; "
                      "* = on the edge of its grid)")
                print(f"  {'b':>4s} {'fact./pass':>10s}  " + "  ".join(f"{k:^46s}" for k in KINDS))
                for c in cells:
                    if grid in c:
                        print(f"  {c['batch']:4d} {c['factorizations_per_pass']:10.1f}  "
                              + "  ".join(f"{_row(c[grid]['rows'][k], tk):46s}" for k in KINDS), flush=True)
            print("  SAGA-Muon / best Euclidean (ratio of medians; '-' = one of them misses the target on a seed; "
                  "brackets: 5-95% paired bootstrap over the confirmation seeds). 'best Eucl.': fine / shifted grid "
                  "over the best confirmed Euclidean point of the cell")
            print(f"  {'b':>4s}  {'fine grid':>20s}  {'shifted grid':>20s}  {'best Eucl.':>11s}  "
                  f"{'factor-2 control':>20s}  {'M5':>6s}  control against M5 (selected points; largest relative "
                  "difference of the medians)")
            for c in cells:
                m5r, v, cons = c.get("m5"), c.get("factor2_vs_m5"), c.get("conservative") or {}
                chk = "" if v is None else (
                    ("same points" if all(e["same_point"] for e in v.values()) else "other points: "
                     + ", ".join(k for k, e in v.items() if not e["same_point"]))
                    + f"; {max(e['rel_diff'] for e in v.values()):.1%}")
                con = " / ".join(_r(cons[g]) for g in ("fine", "shifted") if g in cons)
                print(f"  {c['batch']:4d}  {_ratio(c.get('fine')):>20s}  {_ratio(c.get('shifted')):>20s}  {con:>11s}  "
                      f"{_ratio(c.get('factor2')):>20s}  {_r(m5r['ratio']) if m5r else '':>6s}  {chk}")
            if any("window" in c for c in cells):
                print("  step window: passes at the fine-grid selection times 2^(j/8), confirmation seeds; in braces "
                      "the median over the median of the best Euclidean family of the fine grid")
                print(f"  {'b':>4s}  {'family':9s}  " + "  ".join(f"{h:^30s}" for h in
                                                                ("x 2^(-1/4)", "x 2^(-1/8)", "selected", "x 2^(1/8)",
                                                                 "x 2^(1/4)")))
                for c in cells:
                    for kind in WINDOW_KINDS:
                        if "window" not in c:
                            continue
                        parts = []
                        for j in ("-2", "-1", "0", "1", "2"):
                            e = c["window"][kind][j]
                            txt = _cell(e) + (f" div {e['diverged']}" if e["diverged"] else "")
                            parts.append(f"{txt + ' {' + _r(e['over_best_euclid']) + '}':30s}")
                        print(f"  {c['batch']:4d}  {kind:9s}  " + "  ".join(parts))
    if summary.get("about", {}).get("missing_from_default_plan"):
        miss = summary["about"]["missing_from_default_plan"]
        print(f"\nnot computed yet ({len(miss)} parts of the default plan, instance/data seed/b/part): "
              + ", ".join(miss))


def main():
    quick = "--quick" in sys.argv
    tag = "M5b_quick" if quick else "M5b"
    if "--tables" in sys.argv:
        print_tables(C.load_json(f"{tag}_summary"))
        return
    opt = {"--only": None, "--batches": None, "--data-seeds": None, "--parts": None, "--jobs": None}
    for i, a in enumerate(sys.argv):
        if a in opt:
            opt[a] = sys.argv[i + 1].split(",")
    jobs = int(opt["--jobs"][0]) if opt["--jobs"] else None
    parts = tuple(opt["--parts"]) if opt["--parts"] else None
    if parts and any(p not in PARTS for p in parts):
        raise ValueError(f"unknown part in {parts}: choose among {PARTS}")
    insts = QUICK if quick else INSTANCES
    if opt["--only"] and any(k not in insts for k in opt["--only"]):
        raise ValueError(f"unknown instance in {opt['--only']}: choose among {list(insts)}")
    m5 = None if quick else C.load_json("M5_summary")
    if not quick and m5 is None:
        raise ValueError("muon/results/M5_summary.json is needed: the grids are centred on the steps selected by M5")
    seeds = [int(v) for v in opt["--data-seeds"]] if opt["--data-seeds"] else ([0] if quick else list(PLAN))
    t0 = time.perf_counter()
    for sd in seeds:
        plan = PLAN.get(sd, PLAN[1])
        keys = opt["--only"] or [k for k in (insts if quick else plan["instances"])]
        batches = [int(v) for v in opt["--batches"]] if opt["--batches"] else list(QUICK_CENTRE if quick
                                                                                 else plan["batches"])
        for key in keys:
            inst = insts[key]
            s = dict(inst["spec"], seed=int(sd))
            tol = inst["tol"]
            consts = C.constants(C.problem(s))               # built here once: the worker processes inherit it
            for b in batches:
                if b > s["n"] // 2:
                    continue
                cen = QUICK_CENTRE[b] if quick else centre(m5, key, b)
                t1 = time.perf_counter()
                print(f"\n--- {key}, data seed {sd}, b = {b}: parts {list(parts or plan['parts'])}", flush=True)
                cell = run_cell(tag, key, inst, sd, b, parts or plan["parts"], cen, jobs)
                if sd == 0 and not quick:
                    cell["m5"] = m5_reference(m5, key, b, tol)
                summary = C.load_json(f"{tag}_summary") or {}
                node = summary.setdefault("instances", {}).setdefault(key, {}).setdefault(str(sd), {"batches": {}})
                node.update(spec=s, constants=consts, passes=inst["passes"], tol=tol)
                merged = node["batches"].setdefault(str(b), {})
                merged.update(cell)
                if "m5" in merged and "factor2" in merged:
                    merged["factor2_vs_m5"] = reproduces(merged["m5"], merged["factor2"], f"{tol:g}")
                merged["conservative"] = conservative(merged, f"{tol:g}")
                summary["about"] = about(quick)
                summary["about"]["m5_code"] = None if quick else {k: m5[k]["code"] for k in m5}
                summary["about"]["missing_from_default_plan"] = [] if quick else missing(summary)
                C.save_json(f"{tag}_summary", summary)
                print(f"--- {key}, data seed {sd}, b = {b}: " + ", ".join(
                    f"{g} {_ratio(merged.get(g))}" for g in GRIDS if g in merged)
                    + (f", M5 {_r(merged['m5']['ratio'])}" if "m5" in merged else "")
                    + f"   [{time.perf_counter() - t1:.0f}s]", flush=True)
    print_tables(C.load_json(f"{tag}_summary"))
    print(f"\nM5b done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
