"""Digest of the Muon experiments: every number quoted in muon/findings.md comes from the tables written here
(muon/results/digest.md), which are generated from the JSON summaries of the experiment scripts.

Usage: python muon/experiments/make_digest.py
"""
from __future__ import annotations

import math
import os

import numpy as np

import common as C

TITLES = {"soft-mem": "softmax, memory-limited", "ls-mem": "least squares, memory-limited",
          "soft-cond": "softmax, conditioning-limited", "ls-cond": "least squares, conditioning-limited",
          "ls-orth": "least squares, flat planted spectrum (K = d)", "soft-sq": "softmax, square (K = d)"}
GROUP = {"euclid": "E", "spectral": "O", "full": "F", "stream": "S"}
STATUS = {"sota": "pub", "lab": "lab", "new": "new"}
CLASSES = ["Euclidean VR, uniform sampling", "Euclidean VR, importance sampling",
           "Euclidean VR, feature-covariance metric", "orthogonalized VR, uniform sampling",
           "orthogonalized VR, importance sampling", "polar-metric VR (linear in the estimator)", "full batch",
           "minibatch without memory"]
PLAIN = CLASSES[:2]                    # the reference of the ratio table: Euclidean, no preconditioner


def _f(v, fmt="{:.1f}"):
    return fmt.format(v) if (v is not None and isinstance(v, (int, float)) and math.isfinite(v)) else "inf"


def _cell(s):
    """'median [min, max]' of the passes to a tolerance, with the number of seeds that reached it when not all did."""
    if s["reached"] == 0:
        return "not reached"
    if s["runs"] == 1:
        return _f(s["median"])
    txt = f"{_f(s['median'])} [{_f(s['min'])}, {_f(s['max'])}]"
    if s["reached"] < s["runs"]:
        txt += f" ({s['reached']}/{s['runs']})"
    return txt


def _class(row):
    name, group = row["family"], row["group"]
    if group == "euclid":
        if name.startswith("cov-metric"):
            return CLASSES[2]
        return CLASSES[1] if name.startswith("IS-") else CLASSES[0]
    if group == "spectral":
        if "polar-metric" in name:
            return CLASSES[5]
        return CLASSES[4] if name.startswith("IS ") else CLASSES[3]
    return CLASSES[6] if group == "full" else CLASSES[7]


def _rank_key(row, tk):
    """Rows whose confirmation seeds all reach the target first (by median), then partial ones, then the others."""
    s = row["summary"][tk]
    if s["reached"] == s["runs"]:
        return (0, s["median"], 0.0)
    if s["reached"] > 0:
        return (1, -s["reached"] / s["runs"], s["median"])
    return (2, row["final_rel"]["median"], 0.0)


def _generic(name, group):
    if " b=" in name:
        return name.split(" b=")[0] + (" (b = r)" if group != "stream" else " (minibatch)")
    return name


def _hit(s):
    return f"{s['reached']}/{s['runs']}"


def m1_tables(m1, out):
    out.append("## M1. Head-to-head at tuned steps\n")
    any_block = next(iter(m1.values()))
    proto = any_block.get("protocol", {})
    n_conf = len(proto.get("confirm_seeds", [])) or "?"
    out.append("Passes (per-sample gradient evaluations / n) to bring the relative squared distance "
               "||W - W_*||_F^2 / ||W_0 - W_*||_F^2 below the tolerance: median [min, max] over the "
               f"{n_conf} confirmation seeds (1 run for deterministic methods); a first hitting time. 'hit' = "
               "confirmation seeds that reach the target; 'deep' = seeds that go on to the stop level, 1e-4 x "
               "target, within the budget (a method that touches the target and then stalls has hit > deep). The "
               "final relative squared distance is censored at the stop level for the runs that reach it. 'edge' "
               "lists the tuned parameters whose selected value is the smallest (-) or the largest (+) of its grid "
               "(natural boundaries included: theta towards 1, miso_delta = 1). Groups: E Euclidean variance "
               "reduction, O orthogonalized variance reduction, F full gradient, S minibatch without finite-sum "
               "memory. Status: pub = published method or minor variant, lab = in the lab before this study, new = "
               "configuration of this study. 'window' = number of grid points whose tuning score is within a factor "
               "2 of the best one / number of grid points run at full budget ('-' when the family does not reach "
               "the target in tuning). Rows: families whose seeds all hit first, by median.\n")
    if proto:
        out.append(f"Protocol: tuning seeds {proto['tune_seeds']}, refinement of the {proto['refine_top']} best points "
                   f"on seeds {proto['refine_seeds']}, confirmation seeds {proto['confirm_seeds'][0]}.."
                   f"{proto['confirm_seeds'][-1]}; screening at {proto['screen_fraction']:.0%} of the budget. "
                   "Accounting: SVRG-type methods (L-SVRG, PAGE, L-Katyusha and their orthogonalized versions) are "
                   "charged 2 gradients per iteration plus n per refresh; storing the residuals at the reference "
                   "point (the memory of SAGA) would make it 1 per iteration, i.e. about 2/3 of the passes shown.\n")
    codes = sorted({b.get("code", "?") for b in m1.values()})
    out.append("Code fingerprint(s) of the runs: " + ", ".join(f"`{c[:12]}`" for c in codes) + ".\n")
    for key, block in m1.items():
        info, tol = block["info"], block["tol"]
        tk = f"{tol:g}"
        out.append(f"### {key}: {TITLES.get(key, key)}\n")
        out.append(f"Instance `{C.spec_tag(block['spec'])}`: n = {info['n']}, d = {info['d']}, K = {info['K']}, "
                   f"lam = {info['lam']:.2e}, R2 = {info['R2']:.3f}, R2_op = {info['R2_op']:.3f}, "
                   f"L_F = {info['L_F']:.4f}, L_op = {info['L_op']:.4f}; Hessian at the optimum in "
                   f"[{info['mu_true']:.2e}, {info['L_true']:.2e}] (condition number {info['cond_true']:.0f}); "
                   f"R2 / mu_true = {info['R2_over_mu_true']:.0f} = {info['R2_over_mu_true'] / info['n']:.2f} n; "
                   f"L_op / L_F = {info['Lop_over_LF']:.2f}; nuclear rank of grad F(W_0) = {info['nrank_G0']:.2f}; "
                   f"stable rank of W_* = {info['srank_Wstar']:.2f}. One full-gradient step from W_0 removes "
                   f"{info['step0'].get('gd_ls', float('nan')):.1%} (gradient, exact line search) or "
                   f"{info['step0'].get('sharp_ls', float('nan')):.1%} (polar direction, exact line search) of the "
                   f"squared distance. Budget {block['passes']} passes, target {tol:g}.\n")
        out.append(f"| family | group | status | selected step | passes to 1e-3 | passes to {tol:g} | hit | deep | "
                   "final rel. sq. distance (median) | edge | window |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(block["rows"], key=lambda r_: _rank_key(r_, tk)):
            sc = [t["score"] for t in r["tune"] if t.get("kept", True) and t["score"] is not None]
            fin = [x for x in sc if isinstance(x, (int, float)) and math.isfinite(x)]
            n_run = r.get("n_kept") or len(sc)
            if fin and min(fin) <= block["passes"]:
                win = f"{sum(1 for x in fin if x <= 2.0 * min(fin))}/{n_run}"
            else:
                win = "-"
            sl = r.get("stop_level")
            out.append(f"| {r['family']} | {GROUP[r['group']]} | {STATUS[r['status']]} | {C.fmt_point(r['best'])} | "
                       f"{_cell(r['summary']['0.001'])} | {_cell(r['summary'][tk])} | {_hit(r['summary'][tk])} | "
                       f"{_hit(sl) if sl else ''} | {r['final_rel']['median']:.1e} | {', '.join(r['edge'])} | {win} |")
        out.append("")


def m1_overview(m1, out):
    """One table for the six instances: median passes to the target of each family (absolute numbers)."""
    keys = list(m1)
    out.append("### M1 overview: median passes to the target\n")
    out.append("Target: relative squared distance " + ", ".join(f"{m1[k]['tol']:g} ({k})" for k in keys)
               + ". '(k/N)' = number of confirmation seeds that reached the target when not all did; '-' = none did "
               "(final relative squared distance in parentheses). Minibatch rows use b = "
               + ", ".join(sorted({r["family"].split(" b=")[1] for k in keys for r in m1[k]["rows"]
                                   if " b=" in r["family"] and r["group"] != "stream"}, key=int))
               + " (the rank of the instance) and b = "
               + ", ".join(sorted({r["family"].split(" b=")[1] for k in keys for r in m1[k]["rows"]
                                   if " b=" in r["family"] and r["group"] == "stream"}, key=int))
               + " for the streaming methods.\n")
    order, meta = [], {}
    for k in keys:
        for r in m1[k]["rows"]:
            gname = _generic(r["family"], r["group"])
            if gname not in meta:
                meta[gname] = (r["group"], r["status"])
                order.append(gname)
    rank = {"euclid": 0, "full": 1, "spectral": 2, "stream": 3}
    srank = {"sota": 0, "lab": 1, "new": 2}
    order.sort(key=lambda g_: (rank[meta[g_][0]], srank[meta[g_][1]]))
    out.append("| family | group | status | " + " | ".join(keys) + " |")
    out.append("|---|---|---|" + "---|" * len(keys))
    for gname in order:
        cells = []
        for k in keys:
            tk = f"{m1[k]['tol']:g}"
            row = next((r for r in m1[k]["rows"] if _generic(r["family"], r["group"]) == gname), None)
            if row is None:
                cells.append("")
                continue
            s_ = row["summary"][tk]
            if s_["reached"] == 0:
                cells.append(f"- ({row['final_rel']['median']:.0e})")
            elif s_["reached"] < s_["runs"]:
                cells.append(f"{_f(s_['median'])} ({s_['reached']}/{s_['runs']})")
            else:
                cells.append(_f(s_["median"]))
        out.append(f"| {gname} | {GROUP[meta[gname][0]]} | {STATUS[meta[gname][1]]} | " + " | ".join(cells) + " |")
    out.append("")


def _best_in(rows, tk, pred):
    """Best row (all seeds hit; smallest median) among those satisfying pred; None if no such row."""
    cand = [r for r in rows if pred(r) and r["summary"][tk]["reached"] == r["summary"][tk]["runs"]]
    return min(cand, key=lambda r: r["summary"][tk]["median"]) if cand else None


def m1_bests(m1, out):
    keys = list(m1)
    out.append("### M1: best family of each class (families whose confirmation seeds all reach the target)\n")
    out.append("Median passes [min, max], family name below. 'new' = best configuration introduced in this study.\n")
    out.append("| class | " + " | ".join(keys) + " |")
    out.append("|---|" + "---|" * len(keys))
    preds = [(c, (lambda r, c=c: _class(r) == c)) for c in CLASSES[:7]]
    preds.append(("status new (any class)", lambda r: r["status"] == "new"))
    for label, pred in preds:
        cells = []
        for k in keys:
            tk = f"{m1[k]['tol']:g}"
            b = _best_in(m1[k]["rows"], tk, pred)
            cells.append("-" if b is None else f"{_cell(b['summary'][tk])}<br>{b['family']}")
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    out.append("")
    # ratios to the best Euclidean variance-reduced family
    out.append("### M1 ratios: median passes to the target relative to the best Euclidean variance-reduced family\n")
    out.append("Reference (1.00) = best family of the two plain Euclidean classes on the instance (uniform or "
               "importance sampling, no preconditioner; named in the last row). '*' = not all confirmation seeds "
               "reach the target (ratio of the medians, for information); inf = the median seed does not reach "
               "it.\n")
    fams = []
    for k in keys:
        for r in m1[k]["rows"]:
            g_ = _generic(r["family"], r["group"])
            if g_ not in fams:
                fams.append(g_)
    table, ref = {}, {}
    for k in keys:
        tk = f"{m1[k]['tol']:g}"
        b = _best_in(m1[k]["rows"], tk, lambda r: _class(r) in PLAIN)
        ref[k] = b
        best = b["summary"][tk]["median"] if b else float("nan")
        for r in m1[k]["rows"]:
            v, s_ = r["summary"][tk]["median"], r["summary"][tk]
            cell = _f(v / best, "{:.2f}") if (math.isfinite(v) and math.isfinite(best)) else "inf"
            if 0 < s_["reached"] < s_["runs"]:
                cell += "*"
            table.setdefault(_generic(r["family"], r["group"]), {})[k] = cell
    out.append("| family | " + " | ".join(keys) + " |")
    out.append("|---|" + "---|" * len(keys))
    for name in fams:
        out.append(f"| {name} | " + " | ".join(table[name].get(k, "") for k in keys) + " |")
    out.append("| *reference family* | " + " | ".join((ref[k]["family"] if ref[k] else "-") for k in keys) + " |")
    out.append("")


def m1_resolution(m1, out):
    """What difference between two families the protocol resolves: seed-to-seed spread of the confirmation runs
    (bootstrap standard error of the median) and gap between the finalists of the tuning."""
    out.append("### M1 resolution: seed-to-seed spread and tuning gaps\n")
    out.append("Stochastic families whose confirmation seeds all reach the target. 'SE of the median': bootstrap "
               "standard error (2000 resamples of the confirmation seeds) of the median passes to the target, "
               "relative to the median; a difference between two families has about 1.4 times this standard error. "
               "'range': (max - min) / median over the seeds. 'finalists': relative gap between the two best grid "
               "points of a family on the five tuning seeds (geometric mean of the passes).\n")
    out.append("| instance | families | SE of the median: typical, 90th percentile, largest | range: typical, largest | "
               "finalists: typical gap |")
    out.append("|---|---|---|---|---|")
    rng = np.random.default_rng(0)
    for key, block in m1.items():
        tk = f"{block['tol']:g}"
        se, spread, gap = [], [], []
        for r in block["rows"]:
            v = np.array(r.get("confirm_passes", []), dtype=float)
            if len(v) < 5 or not np.isfinite(v).all():
                continue
            med = float(np.median(v))
            boot = np.median(rng.choice(v, size=(2000, len(v))), axis=1)
            se.append(float(boot.std()) / med)
            spread.append(float(v.max() - v.min()) / med)
            sc = sorted(t["score_refined"] for t in r["tune"]
                        if t.get("score_refined") is not None and math.isfinite(t["score_refined"]))
            if len(sc) >= 2:
                gap.append(sc[1] / sc[0] - 1.0)
        if not se:
            continue
        out.append(f"| {key} | {len(se)} | {np.median(se):.1%}, {np.quantile(se, 0.9):.1%}, {np.max(se):.1%} | "
                   f"{np.median(spread):.0%}, {np.max(spread):.0%} | "
                   + (f"{np.median(gap):.1%}" if gap else "-") + " |")
    out.append("")


def m2_tables(m2, m1, out):
    out.append("## M2. Mechanism: exact one-step conditional expectations\n")
    out.append("Medians over the recorded states (passes listed in the experiment script). 'own' = states of the "
               "configuration's own trajectory, 'SAGA states' = states of the Euclidean SAGA trajectory. "
               "n x eff = expected relative decrease of the squared distance per pass (n iterations); 'realized' = "
               "relative decrease per pass measured on the run between recorded states.\n")

    def med(recs, k, scale=1.0):
        v = [r[k] * scale for r in recs if k in r and r[k] is not None and math.isfinite(r[k])]
        return float(np.median(v)) if v else float("nan")

    stale = []
    for key, block in m2.items():
        n = m1[key]["spec"]["n"]
        best = {r["family"]: r["best"] for r in m1[key]["rows"]}
        out.append(f"### {key}: {TITLES.get(key, key)} (n = {n})\n")
        out.append("| family | selected step | n x eff_dist2 (own) | realized | n x eff_dist2 (SAGA states) | cos_grad | "
                   "dual_align | noise_nuc | smin_ratio | nrank omega | nrank grad | base_ratio | surrogate_gap | "
                   "share of indices increasing F |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for fam, r in block.items():
            if fam in best and any(r["cfg"].get(k) != v for k, v in best[fam].items()):
                stale.append(f"{key} / {fam}")
            real = [x["rate_per_pass"] for x in r["realized"] if math.isfinite(x["rate_per_pass"])]
            cfg = {k: v for k, v in r["cfg"].items() if isinstance(v, dict) or k in ("theta", "eps_innov")}
            out.append(f"| {fam} | {C.fmt_point(cfg)} | {med(r['own'], 'eff_dist2', n):.3f} | "
                       f"{(np.median(real) if real else float('nan')):.3f} | {med(r['cross'], 'eff_dist2', n):.3f} | "
                       f"{med(r['own'], 'cos_grad'):.2f} | {med(r['own'], 'dual_align'):.2f} | "
                       f"{med(r['own'], 'noise_nuc'):.2f} | {med(r['own'], 'smin_ratio'):.1f} | "
                       f"{med(r['own'], 'nrank_omega'):.2f} | {med(r['own'], 'nrank_G'):.2f} | "
                       f"{med(r['own'], 'base_ratio'):.3f} | {med(r['own'], 'surrogate_gap'):.3f} | "
                       f"{med(r['own'], 'frac_increase_F'):.2f} |")
        if "SAGA" in block and block["SAGA"].get("yardstick"):
            y = block["SAGA"]["yardstick"]
            out.append("\nFull-gradient steps at the SAGA states, relative decrease of the squared distance per pass "
                       "(one full gradient): " + ", ".join(f"{k} {np.median([q[k] for q in y]):.3f}"
                                                           for k in ("gd", "gd_ls", "sharp", "sharp_ls") if k in y[0])
                       + ".")
        out.append("")
    if stale:
        out.append("**WARNING: M2 was computed with steps that differ from the current M1 selection for: "
                   + ", ".join(stale) + ". Rerun exp_M2_mechanism.py.**\n")
    else:
        out.append("(Consistency check: every M2 configuration is the one selected by the current M1 summary.)\n")


def m3_tables(m3, out):
    out.append("## M3. Error floors\n")
    out.append("Relative squared distance ||W - W_*||_F^2 / ||W_*||_F^2, median [min, max] over "
               f"{len(m3['seeds'])} seeds. 'reached' = median over the last sixth of a run from W_0 = 0; 'stationary' = "
               "median over the second half of a run started at relative squared distance 1e-14 from W_*.\n")
    for key, block in m3["results"].items():
        inst = m3["instances"][key]
        out.append(f"### {key} (`{C.spec_tag(inst['spec'])}`, {inst['passes']} passes from W_0 = 0, "
                   f"{max(4, inst['passes'] // 2)} passes near W_*)\n")
        out.append("| configuration | reached from W_0 = 0 | stationary level | diverged runs |")
        out.append("|---|---|---|---|")
        for label, e in block.items():
            f, g = np.array(e["reached"]), np.array(e["stationary"])
            out.append(f"| {label} | {np.median(f):.1e} [{f.min():.1e}, {f.max():.1e}] | "
                       f"{np.median(g):.1e} [{g.min():.1e}, {g.max():.1e}] | {e['diverged']} |")
        out.append("")


def m4_tables(m4, out):
    out.append("## M4. Sensitivity to a rounding-size perturbation\n")
    out.append(f"Relative separation ||W_a - W_b||_F / ||W_a - W_*||_F of two runs with the same sampled indices, "
               f"started {m4['delta']:g} apart (relative to ||W_*||_F); median over {len(m4['seeds'])} seeds.\n")
    for key, block in m4["results"].items():
        out.append(f"### {key} (`{C.spec_tag(m4['instances'][key]['spec'])}`)\n")
        marks = list(next(iter(block.values()))["at"].keys())
        out.append("| configuration | " + " | ".join(f"pass {m}" for m in marks) + " | max over the run | "
                   "first pass above 1e-3 |")
        out.append("|---|" + "---|" * (len(marks) + 2))
        for label, e in block.items():
            out.append(f"| {label} | " + " | ".join(f"{np.nanmedian(e['at'][m]):.1e}" for m in marks)
                       + f" | {np.nanmedian(e['max']):.1e} | {_f(float(np.median(e['first_1e-3'])))} |")
        out.append("")


def m5_tables(m5, out):
    out.append("## M5. Batch size\n")
    runs = next(iter(next(iter(m5.values()))["rows"]))["summary"]
    n_conf = next(iter(runs.values()))["runs"]
    out.append(f"Passes to the target, median [min, max] over the {n_conf} confirmation seeds ('(k/N)' when not all "
               "seeds reach it), each family tuned for each batch size (selected step in parentheses; * = on the edge "
               "of its grid). Last columns: SAGA-Muon relative to the best Euclidean family of the row (ratio of "
               "medians; '-' when one of them does not reach the target on every seed).\n")
    for key, block in m5.items():
        tol = block["tol"]
        tk = f"{tol:g}"
        out.append(f"### {key}: {TITLES.get(key, key)} (`{C.spec_tag(block['spec'])}`, target {tol:g}, "
                   f"budget {block['passes']} passes)\n")
        kinds = ["SAGA", "B-SAGA", "SAGA-Muon"]
        out.append("| b | " + " | ".join(kinds) + " | SAGA-Muon / best Euclidean | factorizations per pass |")
        out.append("|---|---|---|---|---|---|")
        for b in sorted({r["batch"] for r in block["rows"]}):
            cells, med, full = [], {}, {}
            for k in kinds:
                row = next((r for r in block["rows"] if r["batch"] == b and r["kind"] == k), None)
                if row is None:
                    cells.append("")
                    continue
                s_ = row["summary"][tk]
                med[k], full[k] = s_["median"], s_["reached"] == s_["runs"]
                cells.append(f"{_cell(s_)} ({C.fmt_point(row['best'])})" + ("*" if row["edge"] else ""))
            eu = [med[k] for k in ("SAGA", "B-SAGA") if full.get(k)]
            ratio = f"{med['SAGA-Muon'] / min(eu):.2f}" if (eu and full.get("SAGA-Muon")) else "-"
            out.append(f"| {b} | " + " | ".join(cells) + f" | {ratio} | {block['spec']['n'] / b:.0f} |")
        out.append("")


def m6_tables(m6, out):
    out.append("## M6. The joint step as an implicit MISO iteration\n")
    out.append(f"Fixed, untuned parameters; n = 1000, d = 50, K = 10, alpha = 1. Passes to {m6['tol']:g}: median "
               f"[min, max] over {len(m6['seeds'])} seeds within {m6['passes']} passes; final relative squared "
               "distance (median); 'base' = median over the run of ||m||_* / ||grad F||_* (size of the predictable "
               "part relative to the gradient; identically 0 for the MISO members).\n")

    def cell(v):
        v = np.array(v, dtype=float)
        if not np.isfinite(v).any():
            return "not reached"
        return f"{_f(float(np.median(v)))} [{_f(float(v.min()))}, {_f(float(v.max()))}]"

    for key, block in m6["results"].items():
        out.append(f"### {key} (delta_sdca = {block['delta_sdca']:.3f})\n")
        out.append("| configuration | passes to target | final rel. sq. distance | base | diverged runs |")
        out.append("|---|---|---|---|---|")
        for label, e in block["rows"].items():
            base = float(np.nanmedian(e["base"])) if e["base"] else float("nan")
            out.append(f"| {label} | {cell(e['passes'])} | {np.median(e['final']):.1e} | {base:.3f} | "
                       f"{e['diverged']} |")
        out.append("")


def m7_tables(m7, out):
    out.append("## M7. Checks: polar metric, Katyusha y step, per-sample constants\n")
    if m7.get("spread"):
        out.append("### Spread of the per-sample smoothness constants L_i = c ||x_i||^2 + lam\n")
        out.append("| instance | R2 / mean L_i | max L_i / min L_i | n p_i for p ~ L^(1/2): min, max | smallest n p_i for "
                   "p ~ L |")
        out.append("|---|---|---|---|---|")
        for key, sp in m7["spread"].items():
            out.append(f"| {key} | {sp['R2_over_mean_L']:.2f} | {sp['max_over_min_L']:.1f} | "
                       f"{sp['n_pmin_sqrt']:.2f}, {sp['n_pmax_sqrt']:.2f} | {sp['n_pmin_lin']:.3f} |")
        out.append("")
    floors = m7.get("floors", [])
    if m7.get("metric"):
        out.append("### Conditioning in the frozen polar metric\n")
        out.append("kappa_P = R2P / mu_P by floor f of the gains (R2P: certified per-sample smoothness in the metric). "
                   "*Local*: mu_P is the smallest Hessian eigenvalue of the reparametrized problem at the optimum (gauge "
                   "subspace for softmax). *Global*: mu_P is the global strong-convexity constant, the one the rate of "
                   "F11 needs; for least squares the two coincide (constant Hessian), for softmax it is "
                   "lam lambda_min(C). Last column: number of singular directions of the anchor below the floor, per "
                   "floor.\n")
    for key, res in m7.get("metric", {}).items():
        soft = res.get("loss", "ls") != "ls"
        out.append(f"**{key}.** Euclidean geometry: R2 / mu_true = {res['kappa_euclid']:.0f}"
                   + (f" (global: R2 / lam = {res['kappa_euclid_global']:.0f})" if soft else "")
                   + (f"; d_inf = max_i x_i^T H^-1 x_i = {res['d_inf']:.1f}" if res.get("d_inf") else "") + ".\n")
        out.append("| anchor | constant | " + " | ".join(f"f = {f:g}" for f in floors) + " | floored directions |")
        out.append("|---|---|" + "---|" * (len(floors) + 1))
        labels = []
        for r in res["rows"]:
            if r["anchor"] not in labels:
                labels.append(r["anchor"])
        for lab in labels:
            rr = [r for r in res["rows"] if r["anchor"] == lab]
            out.append(f"| {lab} | {'local' if soft else 'local = global'} | "
                       + " | ".join(f"{r['kappa_P']:.0f}" for r in rr) + " | "
                       + "/".join(str(r["floored"]) for r in rr) + f" of {rr[0]['rank']} |")
            if soft:
                out.append("| | global | " + " | ".join(f"{r['kappa_P_global']:.0f}" for r in rr) + " | |")
        out.append("")
    if m7.get("katyusha"):
        out.append("### Weight of the y step in loopless Katyusha\n")
        out.append("At the (L, mu) selected by M1 for L-Katyusha; passes to the target, median [min, max] over the "
                   "confirmation seeds.\n")
        labels = list(next(iter(m7["katyusha"].values()))["rows"].keys())
        keys = list(m7["katyusha"].keys())
        out.append("| y step | " + " | ".join(keys) + " |")
        out.append("|---|" + "---|" * len(keys))

        def cell(e):
            v = np.array(e["passes"], dtype=float)
            if np.isfinite(v).all():
                return f"{np.median(v):.1f} [{v.min():.1f}, {v.max():.1f}]"
            if e["diverged"] == len(v):
                return "diverges"
            return f"{int(np.isfinite(v).sum())}/{len(v)} reach it (final {np.median(e['final']):.0e})"
        for lab in labels:
            out.append(f"| {lab} | " + " | ".join(cell(m7["katyusha"][k]["rows"][lab]) for k in keys) + " |")
        out.append("| *(L, mu)* | " + " | ".join(C.fmt_point({"L": m7["katyusha"][k]["L"], "mu": m7["katyusha"][k]["mu"]})
                                               for k in keys) + " |")
        out.append("")


def m8_tables(m8, m1, out):
    out.append("## M8. Other draws of the data\n")
    out.append("Same sizes, spectrum and regularization as the M1 instance, other data seeds; every family tuned and "
               "confirmed again on each draw (seed 0 = the M1 instance, copied from the M1 summary). Median passes to "
               "the target [min, max] over the confirmation seeds.\n")
    for key, block in m8.items():
        seeds = sorted(block.keys(), key=int)
        any_ = block[seeds[0]]
        tk = f"{any_['tol']:g}"
        fams = [r["family"] for r in any_["rows"]]
        cols = {}
        if m1 and key in m1:
            cols["0"] = {r["family"]: r for r in m1[key]["rows"] if r["family"] in fams}
        for sd in seeds:
            cols[sd] = {r["family"]: r for r in block[sd]["rows"]}
        heads = []
        for sd in cols:
            info = (m1[key]["info"] if sd == "0" else block[sd]["info"])
            heads.append(f"data seed {sd} (R2/mu_true = {info['R2_over_mu_true'] / info['n']:.0f} n)")
        out.append(f"### {key}: {TITLES.get(key, key)} (target {any_['tol']:g}, budget {any_['passes']} passes)\n")
        out.append("| family | " + " | ".join(heads) + " |")
        out.append("|---|" + "---|" * len(cols))
        for fam in fams:
            cells = []
            for sd in cols:
                r = cols[sd].get(fam)
                cells.append("" if r is None else f"{_cell(r['summary'][tk])} ({C.fmt_point(r['best'])})")
            out.append(f"| {fam} | " + " | ".join(cells) + " |")
        # ratios (families whose confirmation seeds all reach the target)
        eu = [f for f in fams if f in ("SAGA", "B-SAGA", "IS-B-SAGA")]
        cov = [f for f in fams if f.startswith("cov-metric")]
        pm = [f for f in fams if "polar-metric" in f]
        sp = [f for f in fams if f not in eu + cov + pm]

        def best(col, names):
            v = [col[f]["summary"][tk]["median"] for f in names
                 if f in col and col[f]["summary"][tk]["reached"] == col[f]["summary"][tk]["runs"]]
            return min(v) if v else float("nan")
        lines = [("best orthogonalized / SAGA", sp, ["SAGA"]),
                 ("best orthogonalized / best Euclidean (IS included)", sp, eu)]
        if pm:
            lines.append(("best one-shot polar metric / best Euclidean (IS included)", pm, eu))
        if pm and cov:
            lines.append(("best one-shot polar metric / covariance metric", pm, cov))
        for label, num, den in lines:
            out.append(f"| *{label}* | " + " | ".join(_f(best(cols[sd], num) / best(cols[sd], den), "{:.2f}")
                                                      for sd in cols) + " |")
        out.append("")


def _ratio(v, fmt="{:.2f}"):
    """A ratio of medians of a summary: '-' when it is undefined (null in the JSON)."""
    return fmt.format(v) if isinstance(v, (int, float)) and math.isfinite(v) else "-"


def m5b_tables(m5b, out):
    """M5 at resolved steps (exp_M5b_batch_fine.py). Keys of M5b_summary.json are given with every table."""
    about, inst = m5b["about"], m5b["instances"]
    out.append("## M5b. Batch size at resolved steps\n")
    out.append("Experiment M5 again (same instances, budgets, target, batch sizes, families and protocol) on step "
               "grids of ratio 2^(1/4) instead of 2: c(j) = c5 * 2^(j / 8), c5 the step constant selected by M5, "
               f"j in {about['grids']['fine']['j'][0]}..{about['grids']['fine']['j'][-1]} even ('fine', "
               f"{len(about['grids']['fine']['j'])} steps), odd ('shifted', {len(about['grids']['shifted']['j'])} "
               "steps: the same grid moved by half a step) or a multiple of 8 ('factor-2 control': the powers of two, "
               "i.e. M5's own tasks, recomputed). The ratios of this section supersede those of the M5 tables above "
               "wherever the two differ. Source: `M5b_summary.json`, cell = instances/<instance>/<data seed>/"
               "batches/<b>; protocol: " + about["protocol"] + ".\n")
    out.append("Ratio = median passes of SAGA-Muon / smallest median of the Euclidean families (SAGA, B-SAGA) that "
               "reach the target on every confirmation seed ('-' when SAGA-Muon or no Euclidean family does); in "
               "brackets the 5%-95% quantiles of a paired bootstrap over the confirmation seeds (algorithm-seed noise "
               "only: not the grid, not the data draw). 'conservative' = SAGA-Muon of the fine / shifted grid over the "
               "best confirmed Euclidean point of the cell, whatever the grid or window offset it comes from. "
               "'M5' = the ratio of `M5_summary.json`.\n")
    for key, block in inst.items():
        if "0" not in block:
            continue
        b0 = block["0"]
        tk = f"{b0['tol']:g}"
        out.append(f"### {key}: {TITLES.get(key, key)} (`{C.spec_tag(b0['spec'])}`, target {b0['tol']:g}, "
                   f"budget {b0['passes']} passes)\n")
        out.append("Fine grid, data seed 0 (keys fine/rows/<family>/{summary, best}; * = selected value on the edge "
                   "of its grid).\n")
        out.append("| b | SAGA | B-SAGA | SAGA-Muon | factorizations per pass |")
        out.append("|---|---|---|---|---|")
        cells = sorted(b0["batches"].items(), key=lambda kv: int(kv[0]))
        for b, cell in cells:
            row = []
            for kind in ("SAGA", "B-SAGA", "SAGA-Muon"):
                r = cell["fine"]["rows"][kind]
                row.append(f"{_cell(r['summary'][tk])} ({C.fmt_point(r['best'])})" + ("*" if r["edge"] else ""))
            out.append(f"| {b} | " + " | ".join(row) + f" | {cell['factorizations_per_pass']:.3g} |")
        out.append("")
        out.append("Ratios, data seed 0 (keys fine/ratio and fine/boot, shifted/ratio, factor2/ratio, m5/ratio, "
                   "conservative/{fine, shifted}; 'same point' = number of families for which the factor-2 control "
                   "selects M5's point, key factor2_vs_m5).\n")
        out.append("| b | fine grid | shifted grid | factor-2 control | M5 | conservative: fine, shifted | control "
                   "selects M5's point |")
        out.append("|---|---|---|---|---|---|---|")
        for b, cell in cells:
            def with_boot(part):
                bt = part.get("boot")
                txt = _ratio(part.get("ratio"))
                if bt and txt != "-":
                    txt += f" [{bt[0]:.2f}, {bt[1]:.2f}]"
                return txt
            cons = cell.get("conservative", {})
            same = cell.get("factor2_vs_m5", {})
            out.append(f"| {b} | {with_boot(cell['fine'])} | {with_boot(cell['shifted'])} | "
                       f"{_ratio(cell['factor2'].get('ratio'))} | {_ratio(cell.get('m5', {}).get('ratio'))} | "
                       f"{_ratio(cons.get('fine'))}, {_ratio(cons.get('shifted'))} | "
                       f"{sum(1 for v in same.values() if v.get('same_point'))}/{len(same)} |")
        out.append("")
        offs = ("-2", "-1", "0", "1", "2")
        out.append("Step window, data seed 0: passes to the target at the step selected on the fine grid times "
                   "2^(j / 8), median over the confirmation seeds ('(k/N)' when not all reach it; key "
                   "window/<family>/<j>).\n")
        out.append("| b | family | " + " | ".join(f"j = {j}" for j in offs) + " |")
        out.append("|---|---|" + "---|" * len(offs))
        for b, cell in cells:
            for kind, w in cell.get("window", {}).items():
                vals = []
                for j in offs:
                    e = w.get(j)
                    if e is None:
                        vals.append("")
                    elif e["reached"] == 0:
                        vals.append("not reached" + (f" ({e['diverged']} diverged)" if e.get("diverged") else ""))
                    else:
                        vals.append(_f(e["median"]) + ("" if e["reached"] == e["runs"]
                                                       else f" ({e['reached']}/{e['runs']})"))
                out.append(f"| {b} | {kind} | " + " | ".join(vals) + " |")
        out.append("")
    others = [(key, sd, blk) for key, block in inst.items() for sd, blk in sorted(block.items()) if sd != "0"]
    if others:
        out.append("### M5b on other draws of the data\n")
        out.append("Grids centred on the steps of data seed 0; parts fine and factor-2 control only (keys as above, "
                   "with the data seed in the path).\n")
        out.append("| instance | data seed | b | SAGA | B-SAGA | SAGA-Muon | fine grid | factor-2 control | "
                   "conservative (fine) |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for key, sd, blk in others:
            tk = f"{blk['tol']:g}"
            for b, cell in sorted(blk["batches"].items(), key=lambda kv: int(kv[0])):
                rows = cell["fine"]["rows"]
                row = [f"{_cell(rows[k]['summary'][tk])} ({C.fmt_point(rows[k]['best'])})"
                       for k in ("SAGA", "B-SAGA", "SAGA-Muon")]
                out.append(f"| {key} | {sd} | {b} | " + " | ".join(row) + f" | {_ratio(cell['fine'].get('ratio'))} | "
                           f"{_ratio(cell['factor2'].get('ratio'))} | "
                           f"{_ratio(cell.get('conservative', {}).get('fine'))} |")
        out.append("")
    codes = sorted({cell.get("code_runs", "?") for block in inst.values() for blk in block.values()
                    for cell in blk["batches"].values()})
    out.append("Code fingerprint(s) of the runs (muon/lab and common.py, key code_runs): "
               + ", ".join(f"`{c[:12]}`" for c in codes) + ".\n")


M9_VARIANTS = ["polar", "polar+N", "ns5", "ns5+N", "frob", "frob+N", "sign", "sign+N"]     # row order of part A


def _recipe_point(pt):
    """Selected point of a recipe family: 'q=0.8, lr0=0.1'; the constant-step references keep fmt_point."""
    if "lr_decay" in pt:
        return f"q={pt['lr_decay']:g}, lr0={pt['lr']:g}"
    return C.fmt_point(pt)


def _range(e, fmt="{:.1e}"):
    return f"{fmt.format(e['median'])} [{fmt.format(e['min'])}, {fmt.format(e['max'])}]"


def m9_tables(m9, out):
    """Muon's own recipe with a SAGA table (exp_M9_schedule.py). Keys of M9_summary.json are given with every table."""
    d, proto = m9["definitions"], m9["protocol"]
    out.append("## M9. Muon's own recipe with a SAGA table\n")
    out.append("Recipe: " + d["recipe"] + ". Maps: " + d["maps"] + ". Protocol of M1 (tuning seeds "
               f"{proto['tune_seeds']} and {proto['refine_seeds']}, confirmation seeds {proto['confirm_seeds'][0]}.."
               f"{proto['confirm_seeds'][-1]}); screening: " + proto["screening"] + ". Accounting: "
               + proto["accounting"] + f". Source: `M9_summary.json` (code fingerprint `{m9['code'][:12]}`).\n")
    if m9.get("A"):
        out.append("### M9 part A: maps by batch size\n")
        out.append("Passes to the target, median [min, max] over the confirmation seeds ('(k/N)' when not all reach "
                   "it), selected (q, lr_0) or step in parentheses; * = selected value on the edge of its grid. "
                   "'+ Nesterov' = Nesterov momentum 0.9. frob and sign are controls, not orthogonalizations. The "
                   "three references use constant steps on 2^(1/4) grids at the same batch (not run at b = 1). Keys: "
                   "A/<instance>/cells/<b>/rows/<family>/{summary, best, edge}.\n")
        for key, block in m9["A"].items():
            tk = f"{block['tol']:g}"
            bs = sorted(block["cells"], key=int)
            out.append(f"**{key}** (`{C.spec_tag(block['spec'])}`, target {block['tol']:g}, budget "
                       f"{block['passes']} passes; q in {block['q_grid']}).\n")
            names = []
            for b in bs:
                for r in block["cells"][b]["rows"].values():
                    if (r["kind"], r["variant"]) not in names:
                        names.append((r["kind"], r["variant"]))
            names.sort(key=lambda kv: (kv[0] != "recipe", M9_VARIANTS.index(kv[1]) if kv[1] in M9_VARIANTS else 99))
            out.append("| family | " + " | ".join(f"b = {b}" for b in bs) + " |")
            out.append("|---|" + "---|" * len(bs))
            for kind, variant in names:
                cells = []
                for b in bs:
                    r = next((r_ for r_ in block["cells"][b]["rows"].values()
                              if (r_["kind"], r_["variant"]) == (kind, variant)), None)
                    cells.append("" if r is None else f"{_cell(r['summary'][tk])} ({_recipe_point(r['best'])})"
                                 + ("*" if r["edge"] else ""))
                label = f"fixed-length SAGA [{variant.replace('+N', ' + Nesterov')}]" if kind == "recipe" else variant
                out.append(f"| {label} | " + " | ".join(cells) + " |")
            out.append("")
        out.append("Ratios of medians per cell (key A/<instance>/cells/<b>/compare): fastest variant among the "
                   "spectral maps (polar, ns5; with or without Nesterov) over the fastest control (frob, sign; with "
                   "or without Nesterov), over the fastest control without momentum, over the faster of SAGA and "
                   "B-SAGA, and over the nuclear-scaled joint SAGA-Muon. The fastest variant is the smallest median, "
                   "even when its row loses seeds ('(k/N)' next to its name); 0.00 = the denominator never reaches "
                   "the target; '-' = not available.\n")
        out.append("| instance | b | fastest spectral | fastest control | / control | / control without momentum | "
                   "/ best of SAGA, B-SAGA | / nuclear-scaled SAGA-Muon |")
        out.append("|---|---|---|---|---|---|---|---|")
        for key, block in m9["A"].items():
            for b in sorted(block["cells"], key=int):
                cmp_ = block["cells"][b]["compare"]

                def who(e):
                    if not e:
                        return "-"
                    return e["variant"] + ("" if e["reached"] == e["runs"] else f" ({e['reached']}/{e['runs']})")
                rt = cmp_["ratios"]
                out.append(f"| {key} | {b} | {who(cmp_.get('spectral'))} | {who(cmp_.get('control'))} | "
                           + " | ".join(_ratio(rt.get(k)) for k in ("spectral/control", "spectral/control_no_momentum",
                                                                    "spectral/euclid", "spectral/nuclear_scaled"))
                           + " |")
        out.append("")
    if m9.get("B"):
        out.append("### M9 part B: batch one on the six instances\n")
        out.append("Exact polar factor without momentum, and Muon's recipe (ns5, Nesterov 0.9), tuned on a finer grid "
                   "of q; then both at the fixed point q = 0.8, lr_0 = 0.1 (no tuning) on the same confirmation "
                   "seeds. 'on schedule' = confirmation runs that go on to the stop level. dist^2 / lr^2: "
                   + d["dist2_over_lr2"] + ". M1: medians of `M1_summary.json` on the same instance (factor-2 step "
                   "grids). Keys: B/<instance>/rows/<family>, B/<instance>/fixed_point/rows/<family>, "
                   "B/<instance>/M1/rows.\n")
        for key, block in m9["B"].items():
            tk = f"{block['tol']:g}"
            out.append(f"**{key}** (target {block['tol']:g}, budget {block['passes']} passes; q in {block['q_grid']}, "
                       f"lr_0 in {block['lr_grid']}).\n")
            out.append("| configuration | selected point | passes to the target | on schedule | dist^2 / lr^2: "
                       "min, max (median width of a run) |")
            out.append("|---|---|---|---|---|")

            def line(label, pt, r, edge=()):
                dl = r.get("dist2_over_lr2") or {}
                band = (f"{dl['min']:.3g}, {dl['max']:.3g} ({dl['width_median']:.1f})" if dl.get("runs") else "-")
                out.append(f"| {label} | {_recipe_point(pt)}" + ("*" if edge else "") + f" | "
                           f"{_cell(r['summary'][tk])} | {_hit(r['stop_level'])} | {band} |")
            for name, r in block["rows"].items():
                line(name.replace(" b=1", ""), r["best"], r, r["edge"])
            fp = block["fixed_point"]
            for name, r in fp["rows"].items():
                line(name.replace(" b=1", "") + ", fixed point", fp["point"], r)
            out.append("")
            m1r = block.get("M1", {}).get("rows", {})
            if m1r:
                out.append("M1, same instance: " + "; ".join(
                    f"{n} {_f(v['median'])}" + ("" if v["reached"] == v["runs"] else f" ({v['reached']}/{v['runs']})")
                    for n, v in m1r.items()) + ".\n")
    if m9.get("C"):
        out.append("### M9 part C: the schedule law and its controls\n")
        out.append("Batch one. Definitions of the summary: " + d["law"] + ". Keys: C/<instance>/deep/<configuration>, "
                   "C/<instance>/controls/<configuration>/final_rel.\n")
        for key, block in m9["C"].items():
            out.append(f"**{key}** ({block['passes']} passes, seeds {block['seeds'][0]}..{block['seeds'][-1]}).\n")
            out.append("| configuration with a SAGA table | (q, lr_0) | on schedule | level of dist^2 / lr^2 | band | "
                       "law holds down to (ratio in its on-schedule range) | within 3x of the level | lowest relative "
                       "squared distance |")
            out.append("|---|---|---|---|---|---|---|---|")
            for label, e in block["deep"].items():
                band = f"{e['band']['min']:.3g}, {e['band']['max']:.3g}"
                out.append(f"| {label} | {_recipe_point(e['cfg'])} | {e['on_schedule']}/{len(e['runs'])} | "
                           f"{e['level']['median']:.3g} | {band} | {_range(e['depth_rel_in_range'])} | "
                           f"{_range(e['depth_rel_x3'])} | {_range(e['min_rel'])} |")
            out.append("")
            cs = block["controls_schedule"]
            out.append(f"Without a table, on the schedule {_recipe_point(cs)}: final relative squared distance, "
                       "median [min, max].\n")
            out.append("| configuration without a table | final relative squared distance |")
            out.append("|---|---|")
            for label, e in block["controls"].items():
                out.append(f"| {label} | {_range(e['final_rel'], '{:.3f}')} |")
            out.append("")
    if m9.get("D"):
        out.append("### M9 part D: sampling order and memory of the table\n")
        out.append("Exact polar factor, no momentum. Definitions of the summary: " + d["order"] + ". Memory factor: "
                   + d["memory_factor"]
                   + ". 'on schedule by rho' lists, from the largest rho of the grid to the smallest, the number of "
                   "seeds on schedule. Keys: D/<instance>/cells/<b>/schemes/<scheme>.\n")
        for key, block in m9["D"].items():
            out.append(f"**{key}** ({block.get('role', '')}; target {block['tol']:g}; rho grid {block['rho_grid']}, "
                       f"extended by {block.get('rho_grid_extension_m3b', [])} for m = 3b; reference rho "
                       f"{block['rho_ref']}).\n")
            out.append("| b | scheme | memory factor | exp(-(1 + m/b)) | rho_star | rho_star (monotone) | fastest rho: "
                       "passes to the target | reference level C | on schedule by rho |")
            out.append("|---|---|---|---|---|---|---|---|---|")
            for b in sorted(block["cells"], key=int):
                cell = block["cells"][b]
                n_seeds = len(cell["seeds"])
                for name, sc in cell["schemes"].items():
                    f_ = sc.get("fastest") or {}
                    fast = (f"{f_['rho']:g}: {_f(f_['median'])} [{_f(f_['min'])}, {_f(f_['max'])}]"
                            if f_.get("rho") is not None else "-")
                    ons = " ".join(str(e["on_schedule"]) for e in sc["by_rho"])
                    out.append(f"| {b} | {name} | {_ratio(sc.get('memory_factor'), '{:.3f}')} | "
                               f"{_ratio(sc.get('exp_minus_1_plus_m_over_b'), '{:.3f}')} | "
                               f"{_ratio(sc.get('rho_star'), '{:g}')} | "
                               f"{_ratio(sc.get('rho_star_monotone'), '{:g}')} | "
                               f"{fast} | {sc['reference_C']:.3g} | {ons} (of {n_seeds}) |")
            out.append("")


def m9b_tables(m9b, out):
    """Part D of M9 on a sum with identical Hessians (exp_M9b_order_model.py). Keys of M9b_summary.json are given
    with every table."""
    d = m9b["definitions"]
    g = lambda v: "-" if v is None else format(v, "g")                                              # noqa: E731
    out.append("## M9b. The sampling-order part of M9 on a sum with identical Hessians\n")
    out.append("Model: " + d["model"] + f". Source: `M9b_summary.json` (code fingerprint `{m9b['code'][:12]}`).\n")
    if m9b.get("D"):
        blk = m9b["D"]
        s = blk["spec"]
        out.append("### M9b part D: the design of part D of M9 on the model instance\n")
        out.append(f"Instance: n = {s['n']}, K = {s['K']}, lam = {s['lam']:g}. Definitions: " + d["D"] + "; "
                   + d["D_vs_M9"] + f". Step scale {blk['step_scale']:.4g}; rho grid {blk['rho_grid']}, extended by "
                   f"{blk['rho_grid_extension_m3b']} for m = 3b; reference rho {blk['rho_ref']}. 'on schedule by rho' "
                   "lists, from the largest rho of the grid to the smallest, the number of seeds on schedule. A "
                   "threshold equal to the smallest rho of its grid is an upper bound. Keys: "
                   "D/cells/<b>/schemes/<scheme>, D_vs_M9/<b>/<scheme>.\n")
        out.append("| b | scheme | memory factor | rho_star, model | rho_star (monotone), model | rho_star, soft-mem | "
                   "rho_star, ls-mem | median passes to the target at the fastest rho: model / soft-mem / ls-mem | "
                   "reference level C, model | on schedule by rho, model |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for b in sorted(blk["cells"], key=int):
            cell = blk["cells"][b]
            for name, sc in cell["schemes"].items():
                cmp_ = m9b.get("D_vs_M9", {}).get(b, {}).get(name, {})
                edge = " (edge)" if sc["rho_star"] == sc["by_rho"][-1]["rho"] else ""
                ons = " ".join(str(e["on_schedule"]) for e in sc["by_rho"])
                fast = cmp_.get("passes_at_fastest_rho") or {}
                fp = " / ".join(_f(fast.get(k)) if fast.get(k) is not None else "-"
                                for k in ("model", "soft-mem", "ls-mem"))
                out.append(f"| {b} | {name} | {_ratio(sc.get('memory_factor'), '{:.3f}')} | {g(sc['rho_star'])}{edge} | "
                           f"{g(sc['rho_star_monotone'])} | {g(cmp_.get('soft-mem'))} | {g(cmp_.get('ls-mem'))} | "
                           f"{fp} | {sc['reference_C']:.3g} | {ons} (of {len(cell['seeds'])}) |")
        out.append("")
    if m9b.get("L"):
        sec = m9b["L"]
        out.append("### M9b part L: long horizons at batch one\n")
        out.append("Definitions: " + d["L"] + ". Linear SAGA: " + d["linear_saga"] + ". Keys: "
                   "L/cells/n=<n>/orders/<order>/{level_half, level_end}, L/cells/n=<n>/linear_saga.\n")
        h, e = sec["passes"] // 2, sec["passes"]
        out.append(f"| n | order | x_star at {h} passes | x_star at {e} passes | within 10 x the reference level, "
                   f"{h} / {e} passes | linear SAGA on the model: typical run, independent sampling (expectation) | "
                   "linear SAGA: reshuffling, mean square |")
        out.append("|---|---|---|---|---|---|---|")
        for key, cell in sec["cells"].items():
            ls = cell.get("linear_saga")
            for order, o in cell["orders"].items():
                lin_i = (f"{ls['typical_iid']:.3f} at c = {ls['at_c']:g} ({ls['expectation_iid_at_that_c']:.3f})"
                         if ls and order == "iid" else "")
                lin_r = f"{ls['reshuffled_mean_square_best']:.3f}" if ls and order == "reshuffle" else ""
                out.append(f"| {key[2:]} | {o['label']} | {g(o['level_half']['x_star'])} | {g(o['level_end']['x_star'])} "
                           f"| {g(o['level_half']['x_star_clean'])} / {g(o['level_end']['x_star_clean'])} | {lin_i} | "
                           f"{lin_r} |")
        out.append("")
        xs = sec["x_grid"]
        for key, cell in sec["cells"].items():
            out.append(f"n = {key[2:]}: runs on schedule (of {sec['runs']}) and median level over the reference "
                       f"level at {e} passes, by x = log(1/rho).\n")
            out.append("| x | " + " | ".join(f"{x:.2f}" for x in xs) + " |")
            out.append("|---|" + "---|" * len(xs))
            for order, o in cell["orders"].items():
                out.append(f"| {o['label']}: on schedule | " + " | ".join(str(v) for v in o["level_end"]["on_schedule"])
                           + " |")
                out.append(f"| {o['label']}: level / reference | "
                           + " | ".join(f"{v:.2g}" for v in o["level_end"]["level_over_reference"]) + " |")
            out.append("")


def audit_table(aud, out):
    out.append("## Cache audit\n")
    out.append(f"`audit_cache.py`: {aud['checked']} cached runs drawn at random (seed {aud['seed']}) from "
               f"{len(aud['files'])} cache files were recomputed with the current code "
               f"(fingerprint `{aud['code'][:12]}`): {aud['identical']} reproduced exactly (passes to every "
               f"tolerance and final distance), {aud['mismatch']} differ.\n")
    if aud.get("details"):
        out.append("| cache | method | configuration | cached -> recomputed |")
        out.append("|---|---|---|---|")
        for d in aud["details"][:20]:
            out.append(f"| {d['file']} | {d['method']} | {d['cfg']} | {d['what']} |")
        out.append("")


def main():
    out = ["# Digest of the Muon experiments\n",
           "Generated by `muon/experiments/make_digest.py` from `muon/results/*_summary.json`. Protocol: "
           "`muon/experiments/common.py`.\n"]
    m1 = C.load_json("M1_summary")
    if m1:
        m1_tables(m1, out)
        m1_overview(m1, out)
        m1_bests(m1, out)
        m1_resolution(m1, out)
    m2 = C.load_json("M2_summary")
    if m2 and m1:
        m2_tables(m2, m1, out)
    m3 = C.load_json("M3_summary")
    if m3:
        m3_tables(m3, out)
    m4 = C.load_json("M4_summary")
    if m4:
        m4_tables(m4, out)
    m5 = C.load_json("M5_summary")
    if m5:
        m5_tables(m5, out)
    m5b = C.load_json("M5b_summary")
    if m5b:
        m5b_tables(m5b, out)
    m6 = C.load_json("M6_summary")
    if m6:
        m6_tables(m6, out)
    m7 = C.load_json("M7_summary")
    if m7:
        m7_tables(m7, out)
    m8 = C.load_json("M8_summary")
    if m8:
        m8_tables(m8, m1, out)
    m9 = C.load_json("M9_summary")
    if m9:
        m9_tables(m9, out)
    m9b = C.load_json("M9b_summary")
    if m9b:
        m9b_tables(m9b, out)
    aud = C.load_json("audit_summary")
    if aud:
        audit_table(aud, out)
    path = os.path.join(C.RESULTS, "digest.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"wrote {path} ({len(out)} lines)")


if __name__ == "__main__":
    main()
