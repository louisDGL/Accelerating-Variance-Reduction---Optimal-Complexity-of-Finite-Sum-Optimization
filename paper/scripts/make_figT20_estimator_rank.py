"""Figure T20 of the lecture (paper/talk/figures/figT20_estimator_rank): along runs of the recipe (4) of
paper/paper_saga_to_muon.tex, W <- W - lr_t polar(omega_t), lr_t = lr_0 q^(t b / n), the SAGA estimator omega = m + c is
close to rank one at batch one and has a nuclear rank of two to three with 16 or 32 samples per batch: the batch sizes
at which the Frobenius-normalized control needs momentum and the exact polar factor does not. Explanatory figure, not in
the paper. It explains finding (M6), the mechanism it borrows from (M2) ("the estimator being close to rank one"), the
rank-one identity F1 of muon/notes_theory.md (the polar factor of a rank-one matrix is its normalization) and
Figure 4(b).

Panels:
  (a) NEW MEASUREMENT with the project code (muon/lab), on the least-squares conditioning-limited instance `ls-cond`
      (n = 200, K x d = 8 x 40, the specification stored in M9_summary.json): the singular values s_k / s_1, k = 1..8,
      of the SAGA estimator omega = m + c for b = 1, 16, 32. For each batch size the recipe with the exact polar factor
      and no momentum is rerun at its tuned point (M9_summary.json, A/ls-cond/cells/<b>/rows/"fixed-length SAGA [polar]
      b=<b>"/best) on the ten confirmation seeds 10..19; at the states reached after PASSES passes (the initial full
      gradient of the table counts for one pass) the estimator is formed for every index (b = 1: all n indices) or for
      N_BATCH sampled batches (b > 1), with VRMuon.pieces, and its singular values are computed with
      lab.polar.singular_values. Marker: median over seeds, states and batches; bar: 10 % to 90 %. The legend gives
      the median nuclear rank ||omega||_*^2 / ||omega||_F^2 (1 for a rank-one matrix, 8 for a multiple of a
      partial isometry).
  (b) REPOSITORY DATA (M9_summary.json, part A): passes to the relative squared distance 1e-6 against b in {1, 16, 32}
      on the two non-square conditioning-limited instances (softmax 8 x 40 `soft-cond`, least squares 8 x 40
      `ls-cond`), for three maps on the geometric cooldown, each at its own tuned (q, lr_0): the exact polar factor
      without momentum (orange), the Frobenius normalization without momentum (grey, open, dashed) and with
      Nesterov momentum 0.9 (grey, filled). Marker: median of the ten confirmation seeds (A/<instance>/cells/<b>/rows/<family>/
      confirm_passes, checked against .../summary/1e-06); bar: range of the seeds that reach the target; "k/10": seeds
      that reach it when some miss; an arrow at the budget (300 passes) when no seed reaches it; "*": the tuned point
      is at the edge of its grid (Frobenius without momentum on least squares: q = 0.95, the largest, at b = 16 and
      32; also the largest lr_0 at b = 32), so the passes are an upper bound for that map.

Checks (the script stops if one fails):
  - the rerun trajectories are those of the repository: the passes to 1e-6 of each rerun seed equal
    `confirm_passes` of M9_summary.json (30 runs of panel (a)), and the first three states of seed 10 are those of
    lab.exact.collect_states;
  - at batch one the innovation c has rank one (s_2(c) <= 1e-12 s_1(c) for every index and state);
  - the medians of panel (b) equal the summaries of the JSON and the table of muon/results/digest.md (section M9,
    part A), and the ratios polar / Frobenius at batch one equal those of the JSON (1.22 and 1.24; the paper quotes
    1.16-1.36 over four instances).

Status. Measured: everything. Panel (a) is one draw of the data, ten algorithm seeds, the tuned point of one map;
it is a diagnostic of the estimator, not of the step. Panel (b): one draw of the data per instance, ten confirmation
seeds, grids of q in steps of 0.1 (0.05 above 0.9) and half-decades of lr_0; passes do not count the factorization of
each polar step. The two panels are juxtaposed: nothing here proves that the rank explains the passes. The same
diagnostic on the softmax instance (gauge basis, 7 singular values, two seeds) is printed, not drawn.

Progressive build (states s1-s3): s1 panel (a) for b = 1; s2 adds b = 16 and 32; s3 adds panel (b).

The script prints every number it plots.
Run:    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/scripts/make_figT20_estimator_rank.py
Output: paper/talk/figures/figT20_estimator_rank.pdf and .png, and the states _s1 ... _s3
"""
from __future__ import annotations

import copy
import json
import math
import os
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import explain_style as es  # noqa: E402

sys.path.insert(0, os.path.join(es.ROOT, "muon"))
from lab import exact, maps, optimum, polar, problems  # noqa: E402
from lab.vrmuon import VRMuon, _draws  # noqa: E402

NAME = "figT20_estimator_rank"
RESULTS = os.path.join(es.ROOT, "muon", "results")
TOL = "1e-06"
BATCHES = (1, 16, 32)
A_INSTANCE = "ls-cond"                 # panel (a)
CHECK_INSTANCE = "soft-cond"           # the same diagnostic, printed only
CHECK_SEEDS = (10, 11)
PASSES = (2, 5, 10, 15, 20, 25)        # states of panel (a): after that many passes (initial full gradient included)
N_BATCH = 300                          # sampled batches per state for b > 1 (b = 1: the n indices)
B_INSTANCES = (("soft-cond", "softmax"), ("ls-cond", "least squares"))
B_FAMILIES = (                         # (key, family name without the batch size, legend)
    ("polar", "fixed-length SAGA [polar]", "polar, no momentum"),
    ("frob", "fixed-length SAGA [frob]", "Frobenius, no momentum"),
    ("frob+N", "fixed-length SAGA [frob + Nesterov]", "Frobenius + Nesterov"),
)
# muon/results/digest.md, section "M9 part A" (medians of passes to 1e-6; None: not reached), for the check
DIGEST = {("soft-cond", "polar"): (28.0, 22.4, 25.7), ("soft-cond", "frob"): (22.9, 48.3, 75.0),
          ("soft-cond", "frob+N"): (32.0, 27.8, 30.5), ("ls-cond", "polar"): (36.1, 27.7, 30.8),
          ("ls-cond", "frob"): (29.0, 144.3, None), ("ls-cond", "frob+N"): (50.5, 38.9, 53.2)}
B1_RATIO = {"soft-cond": 1.22, "ls-cond": 1.24}        # polar / Frobenius at batch one (digest; paper: 1.16-1.36)


def load(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------------------------------------------------
# panel (a): the spectrum of the estimator along reruns of the recipe
# ----------------------------------------------------------------------------------------------------------------------
def make_problem(spec):
    """The instance of a specification, with its minimizer (as muon/experiments/common.problem, without its cache)."""
    kw = {k: v for k, v in spec.items() if k != "loss"}
    prob = (problems.make_softmax if spec["loss"] == "softmax" else problems.make_matrix_ls)(**kw)
    optimum.solve_optimum(prob)
    return prob


def rerun(prob, cfg, seed, at, tol, budget_passes):
    """The run vrmuon.vr_muon(prob, 0, ., seed, **cfg), through the public stages of VRMuon (the loop of
    lab.exact.collect_states) with the records of the experiments (every max(1, n // (10 b)) iterations). Returns the
    states (t, W, state) BEFORE iteration t for t in `at`, and the passes at the first record with relative squared
    distance <= tol (inf if the budget ends first). The run stops once the states are collected and the target met."""
    n, b = prob.n, int(cfg["batch"])
    n_it = (budget_passes * n - n) // b
    algo = VRMuon(prob, n_iters=n_it, **cfg)
    W = np.zeros((prob.K, prob.d))
    st = algo.init_state(W)
    rng = np.random.default_rng(seed)
    d0 = float(np.vdot(prob.W_star, prob.W_star))
    rec = max(1, n // (10 * b))
    at = sorted(int(a) for a in at)
    states, reached, ge, t = [], math.inf, algo.init_cost, 0
    for B, coin in _draws(rng, n, b, n_it, algo.needs_coin, algo.probs, algo.order):
        if t in at:
            states.append((t, W.copy(), copy.deepcopy(st)))
        m, c, aux = algo.pieces(W, st, B, coin)
        S, info = algo.step(W, m, c, st, t)
        ge += algo.commit(W, st, B, coin, aux, info)
        W = algo.advance(W, S, st)
        t += 1
        if t % rec == 0 and not math.isfinite(reached):
            if float(np.linalg.norm(W - prob.W_star) ** 2) / d0 <= tol:
                reached = ge / n
        if (math.isfinite(reached) and t > at[-1]) or ge >= budget_passes * n:
            break
    return algo, states, reached


def spectra(prob, algo, states, seed):
    """s_k(omega) for the estimator omega = m + c at each state: every index (b = 1) or N_BATCH uniform batches
    without replacement (b > 1). Softmax: singular values in the gauge basis. Returns (array (N, r), largest
    s_2(c)/s_1(c) at batch one or None)."""
    n, b = prob.n, algo.batch
    basis = maps.gauge_basis(prob)
    rng = np.random.default_rng((2026, b, seed))
    out, rank_one = [], 0.0 if b == 1 else None
    for _, W, st in states:
        batches = range(n) if b == 1 else [rng.choice(n, size=b, replace=False) for _ in range(N_BATCH)]
        for B in batches:
            m, c, _ = algo.pieces(W, st, B, 1.0)
            om = m + c
            out.append(polar.singular_values(om if basis is None else basis.T @ om))
            if b == 1:
                sc = polar.singular_values(c)
                if sc[0] > 0:
                    rank_one = max(rank_one, float(sc[1] / sc[0]))
    return np.array(out), rank_one


def diagnostic(m9, inst, seeds, check_states=True):
    """Per batch size: pooled spectra of the estimator over seeds, states and batches, with the checks of the rerun."""
    e = m9["A"][inst]
    prob = make_problem(e["spec"])
    n = prob.n
    out = {"n": n, "K": prob.K, "d": prob.d, "cells": {}}
    for b in BATCHES:
        row = e["cells"][str(b)]["rows"][f"fixed-length SAGA [polar] b={b}"]
        if row["orth"] != "svd" or row["nesterov"] or row["edge"]:
            raise SystemExit(f"{inst}, b = {b}: the polar row is not the one expected ({row['what']}, edge {row['edge']})")
        cfg = dict(estimator="saga", batch=b, length="const", orth="svd", **row["best"])
        at = [int(round((p - 1) * n / b)) for p in PASSES]
        confirm = dict(zip(m9["protocol"]["confirm_seeds"], row["confirm_passes"]))
        S, per_seed, per_state, reached, rank_one = [], [], [[] for _ in PASSES], [], 0.0
        for seed in seeds:
            algo, states, hit = rerun(prob, cfg, seed, at, float(TOL), e["passes"])
            ref = confirm[seed]
            if not (hit == ref or abs(hit - ref) <= 1e-9 * max(1.0, abs(ref))):
                raise SystemExit(f"{inst}, b = {b}, seed {seed}: the rerun reaches {TOL} at {hit} passes, "
                                 f"M9_summary.json says {ref}")
            reached.append(hit)
            if len(states) != len(at):
                raise SystemExit(f"{inst}, b = {b}, seed {seed}: {len(states)} states collected, {len(at)} expected")
            if check_states and seed == seeds[0]:
                for (t, W, _), (t2, W2, _) in zip(states, exact.collect_states(      # the first three states
                        prob, np.zeros((prob.K, prob.d)), at[2], at[:3], seed=seed, **cfg)):
                    if t != t2 or not np.array_equal(W, W2):
                        raise SystemExit(f"{inst}, b = {b}: the rerun differs from lab.exact.collect_states at t = {t}")
            for k, state in enumerate(states):
                s, r1 = spectra(prob, algo, [state], seed)
                per_state[k].append(s)
                if r1 is not None:
                    rank_one = max(rank_one, r1)
            per_seed.append(np.concatenate([ps_[-1] for ps_ in per_state]))
            S.append(per_seed[-1])
        if b == 1 and rank_one > 1e-12:
            raise SystemExit(f"{inst}: the innovation at batch one is not of rank one (s_2/s_1 up to {rank_one:.1e})")
        S = np.concatenate(S)
        R = S / S[:, :1]
        nr = S.sum(axis=1) ** 2 / (S ** 2).sum(axis=1)
        ratio2 = lambda s: float(np.median(s[:, 1] / s[:, 0]))                       # noqa: E731
        out["cells"][b] = {
            "cfg": row["best"], "N": len(S), "median": np.median(R, axis=0), "p10": np.percentile(R, 10, axis=0),
            "p90": np.percentile(R, 90, axis=0), "nrank": [float(np.percentile(nr, p)) for p in (50, 10, 90)],
            "s2_by_seed": [ratio2(s) for s in per_seed],
            "s2_by_state": [ratio2(np.concatenate(s)) for s in per_state],
            "reached": reached, "rank_one_c": rank_one if b == 1 else None}
    return out


# ----------------------------------------------------------------------------------------------------------------------
# panel (b): passes of three maps (repository data)
# ----------------------------------------------------------------------------------------------------------------------
def data_b(m9):
    out = {}
    for inst, _ in B_INSTANCES:
        e = m9["A"][inst]
        if float(e["tol"]) != float(TOL):
            raise SystemExit(f"{inst}: the target of part A is {e['tol']}, not {TOL}")
        for j, b in enumerate(BATCHES):
            for key, fam, _ in B_FAMILIES:
                row = e["cells"][str(b)]["rows"][f"{fam} b={b}"]
                p = np.array(row["confirm_passes"], dtype=float)
                ok = np.isfinite(p)
                med = float(np.median(p))
                sm = row["summary"][TOL]
                if not (sm["reached"] == int(ok.sum()) and sm["runs"] == len(p)
                        and (med == sm["median"] or abs(med - sm["median"]) < 1e-9)):
                    raise SystemExit(f"{inst}, {fam} b={b}: confirm_passes disagrees with the summary")
                dg = DIGEST[(inst, key)][j]
                if (dg is None) != (not math.isfinite(med)) or (dg is not None and abs(med - dg) > 0.051):
                    raise SystemExit(f"{inst}, {fam} b={b}: median {med} against {dg} in muon/results/digest.md")
                out[(inst, b, key)] = {
                    "median": med, "min": float(p[ok].min()) if ok.any() else math.inf,
                    "max": float(p[ok].max()) if ok.any() else math.inf, "reached": int(ok.sum()), "runs": len(p),
                    "edge": bool(row["edge"]), "edge_list": list(row["edge"]), "q": row["best"]["lr_decay"],
                    "lr0": row["best"]["lr"]}
        r = out[(inst, 1, "polar")]["median"] / out[(inst, 1, "frob")]["median"]
        ref = e["cells"]["1"]["part_B"]["ratios"]["polar/control"]
        if abs(r - ref) > 1e-9 or abs(r - B1_RATIO[inst]) > 0.0051:
            raise SystemExit(f"{inst}: polar / Frobenius at batch one is {r:.3f} (JSON {ref:.3f}, digest {B1_RATIO[inst]})")
        out[(inst, "ratio_b1")] = r
        out[(inst, "budget")] = e["passes"]
        out[(inst, "q_grid")] = e["q_grid"]
    return out


def data():
    m9 = load("M9_summary.json")
    seeds = tuple(m9["protocol"]["confirm_seeds"])
    return {"A": diagnostic(m9, A_INSTANCE, seeds), "check": diagnostic(m9, CHECK_INSTANCE, CHECK_SEEDS, False),
            "B": data_b(m9), "seeds": seeds}


def _print_diag(D, inst, seeds):
    print(f"  {inst}: n = {D['n']}, K x d = {D['K']} x {D['d']}; seeds {seeds[0]}..{seeds[-1]}; states after "
          f"{', '.join(str(p) for p in PASSES)} passes")
    for b, c in D["cells"].items():
        f = lambda v: " ".join(f"{x:.3f}" for x in v)                                # noqa: E731
        print(f"    b = {b:2d} at q = {c['cfg']['lr_decay']}, lr_0 = {c['cfg']['lr']}: {c['N']} estimators; "
              f"passes to {TOL} of the reruns (= M9_summary.json): {', '.join(f'{v:g}' for v in c['reached'])}")
        print(f"        s_k/s_1 median: {f(c['median'])}")
        print(f"                 10 % : {f(c['p10'])}")
        print(f"                 90 % : {f(c['p90'])}")
        print(f"        median of s_2/s_1: {c['median'][1]:.3f}; per seed {min(c['s2_by_seed']):.3f} to "
              f"{max(c['s2_by_seed']):.3f}; per state {f(c['s2_by_state'])}")
        print(f"        nuclear rank: median {c['nrank'][0]:.2f}, 10 % {c['nrank'][1]:.2f}, 90 % {c['nrank'][2]:.2f}"
              + (f"; innovation: s_2(c)/s_1(c) <= {c['rank_one_c']:.1e}" if b == 1 else ""))


def print_all(D):
    print("(a) singular values of the SAGA estimator along reruns of the recipe (exact polar factor, no momentum)")
    _print_diag(D["A"], A_INSTANCE, D["seeds"])
    print("    the same diagnostic on the other instance (printed, not drawn):")
    _print_diag(D["check"], CHECK_INSTANCE, CHECK_SEEDS)
    print(f"(b) passes to {TOL}: median [min, max of the seeds that reach it] (seeds reaching / seeds), tuned (q, lr_0); "
          "* = tuned point on the edge of its grid (lr_decay+: q = 0.95, the largest; lr+: the largest lr_0)")
    for inst, name in B_INSTANCES:
        print(f"  {inst} ({name}), budget {D['B'][(inst, 'budget')]} passes, q grid {D['B'][(inst, 'q_grid')]}; "
              f"polar / Frobenius at b = 1: {D['B'][(inst, 'ratio_b1')]:.2f}")
        for key, _, legend in B_FAMILIES:
            cells = []
            for b in BATCHES:
                r = D["B"][(inst, b, key)]
                cells.append(f"b = {b}: {r['median']:.1f} [{r['min']:.1f}, {r['max']:.1f}] ({r['reached']}/{r['runs']}) "
                             f"({r['q']}, {r['lr0']})" + (f"* (edge: {', '.join(r['edge_list'])})" if r['edge'] else ""))
            print(f"    {key:7s} " + "; ".join(cells))


# ----------------------------------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------------------------------
def panel_a(ax, A, ps, build):
    ink = ps.INK
    shades = {1: es.tint(ps.ORANGE, 0.50), 16: es.tint(ps.ORANGE, 0.75), 32: ps.ORANGE}   # one hue, ordered in b
    marker = {1: "o", 16: "s", 32: "D"}
    dodge = {1: -0.2, 16: 0.0, 32: 0.2}
    k = np.arange(1, A["K"] + 1)
    for b in BATCHES:
        c, col, x = A["cells"][b], shades[b], k + dodge[b]
        state = 1 if b == 1 else 2
        build.add(state, ax.vlines(x[1:], c["p10"][1:], c["p90"][1:], color=col, lw=1.6, zorder=3))
        build.add(state, ax.plot(x, c["median"], color=col, lw=1.1, zorder=3))
        build.add(state, ax.plot(x, c["median"], ls="", marker=marker[b], ms=5.4 if b != 32 else 4.8, mfc=col,
                                 mec=ink["surface"], mew=0.9, zorder=4,
                                 label=rf"$b = {b}$, nuclear rank {c['nrank'][0]:.1f}"))
    ax.set_xlim(0.45, A["K"] + 0.55)
    ax.set_xticks(k)
    ax.set_ylim(0.0, 1.06)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.grid(axis="x", visible=False)
    ax.set_xlabel(r"index $k$ of the singular value")
    ax.set_ylabel(r"$s_k(\omega)\ /\ s_1(\omega)$")
    es.panel_title(ax, r"(a) singular values of $\omega$")
    build.legend(ax, loc="upper right", handlelength=1.0, handletextpad=0.4, borderaxespad=0.1, labelspacing=0.3)
    ax.annotate(r"nuclear rank: $\|\omega\|_*^2\,/\,\|\omega\|_F^2$" + "\nmarks: median\nbars: 10-90 %",
                (0.985, 0.57), xycoords="axes fraction", ha="right", va="top", fontsize=es.small(),
                color=ink["secondary"], linespacing=1.3)


def _groups(ax, ps, gap=0.8):
    """Categorical x positions: one block of batch sizes per instance, the instance names under the tick labels."""
    pos, ticks, labels, x = {}, [], [], 0.0
    for inst, _ in B_INSTANCES:
        for b in BATCHES:
            pos[(inst, b)] = x
            ticks.append(x)
            labels.append(str(b))
            x += 1.0
        x += gap
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.grid(axis="x", visible=False)
    ax.set_xlim(-0.6, x - gap - 0.4)
    for gi, (inst, name) in enumerate(B_INSTANCES):
        mid = 0.5 * (pos[(inst, BATCHES[0])] + pos[(inst, BATCHES[-1])])
        ax.annotate(name, (mid, 0), xycoords=("data", "axes fraction"), xytext=(0, -20), textcoords="offset points",
                    ha="center", va="top", fontsize=es.small(), color=ps.INK["secondary"])
        if gi:
            ax.axvline(pos[(inst, BATCHES[0])] - 0.5 - 0.5 * gap, color=ps.INK["grid"], lw=0.8, zorder=0)
    ax.set_xlabel("batch size $b$", labelpad=14)
    return pos


def panel_b(ax, B, ps, build):
    ink = ps.INK
    grey = ink["secondary"]
    style = {"polar": dict(color=ps.ORANGE, marker="o", ls="-", mfc=ps.ORANGE, mec=ink["surface"], mew=0.9, ms=6.0),
             "frob": dict(color=grey, marker="D", ls=(0, (3.0, 1.6)), mfc=ink["surface"], mec=grey, mew=1.2, ms=4.8),
             "frob+N": dict(color=grey, marker="D", ls="-", mfc=grey, mec=ink["surface"], mew=0.9, ms=5.4)}
    pos = _groups(ax, ps)
    budget = B[(B_INSTANCES[0][0], "budget")]
    note = dict(textcoords="offset points", fontsize=es.small(), color=grey)
    for key, _, legend in B_FAMILIES:
        st = style[key]
        for gi, (inst, _) in enumerate(B_INSTANCES):
            xs = [pos[(inst, b)] for b in BATCHES]
            rows = [B[(inst, b, key)] for b in BATCHES]
            ys = [r["median"] if math.isfinite(r["median"]) else budget for r in rows]
            ax.plot(xs, ys, lw=1.4, zorder=3 if key != "polar" else 4, label=legend if gi == 0 else None, **st)
            for x, y, r in zip(xs, ys, rows):
                if r["reached"] and r["max"] > r["min"]:
                    ax.plot([x, x], [r["min"], r["max"]], color=st["color"], lw=1.0, zorder=2)
                if not math.isfinite(r["median"]):                     # no seed reaches the target within the budget
                    ax.annotate("", (x, y * 1.42), xytext=(x, y * 1.04), arrowprops=dict(
                        arrowstyle="-|>", color=st["color"], lw=1.1, mutation_scale=8, shrinkA=0, shrinkB=0))
                    ax.annotate(f"{r['reached']}/{r['runs']}" + ("*" if r["edge"] else ""), (x, y), xytext=(-7, 1),
                                ha="right", va="center", **note)
                elif r["reached"] < r["runs"]:
                    ax.annotate(f"{r['reached']}/{r['runs']}", (x, r["min"]), xytext=(0, -3), ha="center", va="top",
                                **note)
                elif r["edge"]:
                    ax.annotate("*", (x, y), xytext=(-6, -1), ha="right", va="center", **note)
    ax.set_yscale("log")
    ax.set_ylim(13.5, 620)
    ax.set_yticks([20, 30, 50, 100, 200, budget])
    ax.set_yticklabels(["20", "30", "50", "100", "200", str(budget)])
    ax.minorticks_off()
    ax.set_ylabel("passes")
    es.panel_title(ax, r"(b) passes to $10^{-6}$")
    build.legend(ax, loc="upper left", handlelength=1.7, handletextpad=0.4, borderaxespad=0.1, labelspacing=0.25)
    ax.annotate(r"* edge of its tuning grid", (0.015, 0.02), xycoords="axes fraction", ha="left", va="bottom",
                fontsize=es.small(), color=grey)


def draw(D):
    ps = es.setup("talk")
    fig = es.figure("talk", height=2.75)
    fig.get_layout_engine().set(w_pad=0.03, wspace=0.03)
    gs = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.18))
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    b = es.Build()
    panel_a(ax_a, D["A"], ps, b)
    panel_b(ax_b, D["B"], ps, b)
    b.add(3, ax_b)
    A = D["A"]
    es.save(fig, NAME, "talk", build=b,
            note=(rf"(a) new diagnostic, least squares 8 $\times$ 40, {len(D['seeds'])} runs per $b$; (b) "
                  rf"{len(D['seeds'])} seeds, each map tuned; not causal"))


def main():
    D = data()
    print_all(D)
    draw(D)


if __name__ == "__main__":
    main()
