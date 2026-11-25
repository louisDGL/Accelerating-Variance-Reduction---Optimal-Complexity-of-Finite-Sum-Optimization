"""Figure T25 of the lecture (paper/talk/figures/figT25_proofs_and_runs): how far the runs are from what the proofs
certify, row by row. Explanatory figure, not in the paper (closing or backup slide). It explains Section 4.1 of
paper/paper_saga_to_muon.tex ("our convergence proofs hold far from the steps that are used"), the paragraph after
Proposition 5 and findings (C1), (C4), in the wording corrected by the audit (paper/audit/AUDIT.md).

Message: each convergence statement of the paper is proved in a regime that is at a measured distance from the runs;
the figure gives that distance, one row per quantity, on one logarithmic axis "what is run / what is certified".

Rows, sources and status (every number is recomputed here and compared with the paper or the audit; the script stops
if one of them is not reproduced):

 SAGA and Proposition 5.9 of Bach's book (blue; ridge logistic regression, n = 1000, d = 50)
  1. exact / certified contraction exponent along runs, passes 2-20, five sampling seeds, 15 problems
     (kappa = 10 ... 1e5, alpha = 0, 1, 2; one data set each): results/B3_summary.json, attribution[*]["speedup"],
     smallest and largest; checked against results/digest.md and (C1) of the paper, l. 287-288 ("3.2-3.7 ... 19-220").
     MEASURED.
  2. observed exponent at the textbook step / best certificate within the proof (step and Lyapunov weight optimized),
     kappa = 1e3, alpha = 1: results/GammaDelta_summary.json, empirical_rates_per_pass["0.25"] (mean of three seeds,
     slope of the log squared distance over passes 2-40) over n * optimal_within_proof["rate"], the latter recomputed
     with vrlab.bounds.saga_optimal_params; the corrected (C4): "4.4 times the best certificate"
     (paper/audit/paper_saga_to_muon_corrige.tex, l. 320; AUDIT.md, finding A). MEASURED / certificate COMPUTED.
  3. a step at which SAGA converges / largest step the proof admits, same problem: the proof needs gamma R^2 < 1/2
     (checked on vrlab.bounds.saga_rate and on certified_rates_per_pass of the same JSON); SAGA converges at
     gamma R^2 = 2 (ratio 4) and stalls at gamma R^2 = 4 (ratio 8): AUDIT.md, finding A (l. 45-56), and a rerun here
     with vrlab.algorithms.saga (seed 0: 60 passes at gamma R^2 = 2, 100 passes at gamma R^2 = 4). The figure does
     not say that 2 is the largest converging step (nothing between 2 and 4 is run here). MEASURED, one instance.

 Polar steps fed by a SAGA table (orange; the two non-square conditioning-limited instances soft-cond and ls-cond,
 n = 200, K x d = 8 x 40, specifications read in muon/results/M1_summary.json, problems built with muon/lab)
  4. cooldown used / cooldown certified by Proposition 5(b), in log(1/q) per pass, batch one: q = 0.8 (the schedule of
     the exact polar factor with the table: muon/results/M9_summary.json, C[*]["points"]["polar, no momentum"]) against
     q_cert = rho^(n/2), rho = 1 - mu_op / L_eff, L_eff = L_op + 10 A L_max, with the age bound A = n (fixed cyclic
     order) or 2n - 1 (fresh permutation per pass); mu_op = lam (softmax; Theorem 1) or lam + lambda_min of the
     covariance (least squares). Paper, l. 578-579 ("3.3e4 to 8.9e4 times slower, in log(1/q), than q = 0.8"),
     qualified "for permutation orders" by the audit (AUDIT.md, section 4; evidence/math-schedule.md, S1). COMPUTED
     from PROVED constants.
  5. the same for independent sampling, the order of the runs of (M5): the proposition then holds with probability
     1 - delta up to a horizon T, A = ceil(n log(nT/delta)); here T = 300 passes (the length of the runs of (M5)) and
     delta = 0.05, as in paper/audit/evidence/math-schedule.md, S1 ("6.3e5 (ls-cond) a 8.6e5 (soft-cond)";
     checks/math-schedule_benchmark_numbers.out). This is the audit's computation, not a number of the original
     paper (the corrected paper says "1e4 to 1e6"): lighter bar, words "not in the paper". COMPUTED from PROVED constants.
  6. tuned step / largest step with a Lyapunov function, nuclear-scaled polar step with a SAGA table, batch one:
     gamma <= 1/(14 n L_max) (muon/notes_theory.md, F15, l. 740-790) against the tuned step of the family
     "SAGA-Muon (joint)" in muon/results/M1_summary.json (factor-2 step grids, single tuned points). Paper, l. 649-650
     ("350 to 1400 times below the tuned steps"). The script also prints the ratio on the six instances of M1
     (not drawn; the audit, evidence/num-m5.md, m5, reports 217 to 3493, and 416 / 1176 on finer grids). MEASURED
     step / PROVED bound, two instances.

Not on the figure: Theorem 1 (not a convergence proof), Theorem 2 and Proposition 4 (no rate is certified), and the
nuclear-scaled polar step with exact gradients on least squares (Eq. (2); optional row of the specification, left
out). The rows compare different quantities (exponents, steps, cooldown rates): they share an axis, not a meaning.
All the certificates are sufficient conditions: outside them nothing is proved and nothing is refuted.

Colours: Euclidean SAGA in blue, polar steps with a table in orange; the status mark (cross, with the word "stalls")
in the reserved colour. Progressive build (states s1, s2): s1 the three rows on SAGA; s2 adds the three rows on polar
steps.

Run:    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/scripts/make_figT25_proofs_and_runs.py   (about 15 s)
Output: paper/talk/figures/figT25_proofs_and_runs.pdf and .png, and the states _s1, _s2
"""
from __future__ import annotations

import json
import math
import os
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import explain_style as es  # noqa: E402

ROOT = es.ROOT
sys.path.insert(0, ROOT)                              # vrlab
sys.path.insert(0, os.path.join(ROOT, "muon"))        # lab (muon/lab)

NAME = "figT25_proofs_and_runs"
INSTANCES = ("soft-cond", "ls-cond")                  # the two non-square conditioning-limited instances
T_PASSES, DELTA = 300, 0.05                           # horizon and failure probability of row 5 (the audit's choice)
STEP_OK, STEP_STALL = 2.0, 4.0                        # gamma R^2 of the two reruns of row 3
PASSES_OK, PASSES_STALL = 60, 100
NOTE = "each row: a different ratio; sufficient conditions: outside them nothing is proved and nothing is refuted"


def _json(*parts):
    with open(os.path.join(ROOT, *parts)) as f:
        return json.load(f)


def _expect(name, value, target, rel=0.02):
    """Stop if a number the figure relies on is not the one of the paper or of the audit (to the digits they print)."""
    if not (abs(value / target - 1.0) <= rel):
        raise SystemExit(f"{name}: computed {value:.6g}, expected {target:g} (paper / audit)")


# ----------------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------------
def rows_saga():
    """Rows 1-3: SAGA on ridge logistic regression against Proposition 5.9."""
    from vrlab import algorithms, bounds, problems, solvers
    # row 1
    b3 = _json("results", "B3_summary.json")["attribution"]
    speed = {k: float(v["speedup"]) for k, v in b3.items()}
    if len(speed) != 15:
        raise SystemExit(f"B3_summary.json: {len(speed)} problems, 15 expected")
    lo_key, hi_key = min(speed, key=speed.get), max(speed, key=speed.get)
    _expect("smallest exact / certified exponent", speed[lo_key], 3.2)
    _expect("largest exact / certified exponent", speed[hi_key], 220.0)
    digest, in_table = None, False
    with open(os.path.join(ROOT, "results", "digest.md")) as f:
        for line in f:
            in_table = in_table or line.startswith("| config | speed-up exact/bound |")
            if in_table and line.startswith(f"| {hi_key} |"):
                digest = float(line.split("|")[2])
                break
    if digest is None or abs(digest / speed[hi_key] - 1.0) > 1e-3:
        raise SystemExit(f"results/digest.md does not give the speed-up of {hi_key}: {digest} / {speed[hi_key]}")
    # row 2
    gd = _json("results", "GammaDelta_summary.json")
    P = gd["problem"]
    n, R2, mu = int(P["n"]), float(P["R2"]), float(P["mu"])
    opt = bounds.saga_optimal_params(n, R2, mu)
    if abs(opt["rate"] / gd["optimal_within_proof"]["rate"] - 1.0) > 1e-9:
        raise SystemExit("the best certificate within the proof is not the one of GammaDelta_summary.json")
    observed = float(gd["empirical_rates_per_pass"]["0.25"])
    best_cert = n * float(opt["rate"])
    _expect("observed exponent / best certificate", observed / best_cert, 4.4)
    # row 3: the steps the proof admits (c_D > 0 and c_M > 0 force gamma R^2 < 1/2), then the two reruns
    cert = {float(k): float(v) for k, v in gd["certified_rates_per_pass"].items()}
    if not (all(v > 0 for g, v in cert.items() if g < 0.5) and all(v == 0 for g, v in cert.items() if g >= 0.5)):
        raise SystemExit("certified_rates_per_pass: the admissible steps are not gamma R^2 < 1/2")
    weights = np.exp(np.linspace(np.log(2.0 + 1e-9), np.log(1e6), 2000))          # Delta = c gamma^2, c > 2
    def admissible(g):
        return any(bounds.saga_rate(n, R2, mu, g / R2, c * (g / R2) ** 2) is not None for c in weights)
    if not admissible(0.4999) or admissible(0.5):
        raise SystemExit("vrlab.bounds.saga_rate: the largest admissible step is not gamma R^2 = 1/2")
    largest_proved = 0.5
    prob = problems.make_logistic(n, int(P["d"]), 1e3, alpha=1.0, seed=0)
    solvers.solve_optimum(prob)
    if abs(prob.R2 / R2 - 1.0) > 1e-12 or abs(prob.summary()["L_F"] / P["L_F"] - 1.0) > 1e-9:
        raise SystemExit("the logistic problem rebuilt here is not the one of GammaDelta_summary.json")
    rel = {}
    for g, passes in ((STEP_OK, PASSES_OK), (STEP_STALL, PASSES_STALL)):
        r = algorithms.saga(prob, np.zeros(prob.d), gamma=g / prob.R2, passes=passes, seed=0, record_every=n,
                            record_F=False)
        D = np.asarray(r["D"], dtype=float)
        rel[g] = D / D[0]                             # one record per pass, record 0 = the start
    if not rel[STEP_OK][-1] < 1e-12:
        raise SystemExit(f"SAGA does not converge at gamma R^2 = {STEP_OK:g}: {rel[STEP_OK][-1]:.2e}")
    if not (rel[STEP_STALL][-1] > 1e-2 and rel[STEP_STALL][-1] > 0.5 * rel[STEP_STALL][40]):
        raise SystemExit(f"SAGA does not stall at gamma R^2 = {STEP_STALL:g}: {rel[STEP_STALL][[40, -1]]}")
    return {"speed": speed, "lo_key": lo_key, "hi_key": hi_key, "digest": digest,
            "observed": observed, "best_cert": best_cert, "ratio_cert": observed / best_cert,
            "largest_proved": largest_proved, "rel": rel,
            "step_ok": STEP_OK / largest_proved, "step_stall": STEP_STALL / largest_proved,
            "stored_rate": {g: float(gd["empirical_rates_per_pass"][str(g)]) for g in (STEP_OK, STEP_STALL)}}


def rows_polar():
    """Rows 4-6: polar steps fed by a SAGA table against Proposition 5(b) and the Lyapunov function F15."""
    from lab import problems
    m1 = _json("muon", "results", "M1_summary.json")
    m9 = _json("muon", "results", "M9_summary.json")["C"]
    out = {"inst": {}, "all_six": {}}
    for name, entry in m1.items():
        s = entry["spec"]
        mk = problems.make_softmax if s["loss"] == "softmax" else problems.make_matrix_ls
        prob = mk(**{k: v for k, v in s.items() if k != "loss"})
        n = prob.n
        L_max = float((prob.c_loss * prob.sqnorms).max())             # L_i = c_l |x_i|^2 (loss part, no ridge)
        L_op = float(prob.L_op)
        mu_op = float(prob.lam + (np.min(prob.cov_eigs) if s["loss"] == "ls" else 0.0))
        # row 6: tuned step of the nuclear-scaled joint SAGA-Muon against 1 / (14 n L_max)
        row = [r for r in entry["rows"] if r.get("family") == "SAGA-Muon (joint)"]
        if len(row) != 1 or row[0]["best"]["gamma"]["over"] != "R2_op":
            raise SystemExit(f"M1_summary.json, {name}: no tuned step c / R2_op for 'SAGA-Muon (joint)'")
        gamma_tuned = float(row[0]["best"]["gamma"]["c"]) / float(prob.R2_op)
        gamma_cert = 1.0 / (14.0 * n * L_max)
        if not (gamma_cert * L_op <= 1.0):
            raise SystemExit(f"{name}: the hypothesis gamma L_op <= 1 of F15 fails at 1 / (14 n L_max)")
        out["all_six"][name] = gamma_tuned / gamma_cert
        if name not in INSTANCES:
            continue
        # rows 4-5: certified cooldown per pass at batch one, q_cert = rho^(n/2)
        q_used = float(m9[name]["points"]["polar, no momentum"]["lr_decay"])
        if q_used != 0.8 or int(m9[name]["passes"]) != T_PASSES or not (mu_op <= L_max):
            raise SystemExit(f"{name}: q = {q_used}, passes = {m9[name]['passes']} (0.8 and {T_PASSES} expected)")
        ages = {"cyclic": n, "permutation": 2 * n - 1,
                "independent": math.ceil(n * math.log(n * (T_PASSES * n) / DELTA))}
        cool = {}
        for order, A in ages.items():
            L_eff = L_op + 10.0 * A * L_max
            log_q_cert = 0.5 * n * math.log1p(-mu_op / L_eff)
            cool[order] = math.log(q_used) / log_q_cert
        out["inst"][name] = {"n": n, "L_max": L_max, "L_op": L_op, "mu_op": mu_op, "ages": ages, "cool": cool,
                             "q_used": q_used, "step": gamma_tuned / gamma_cert,
                             "gamma_cert_R2op": gamma_cert * float(prob.R2_op),
                             "gamma_tuned_R2op": float(row[0]["best"]["gamma"]["c"])}
    perm = [out["inst"][i]["cool"][o] for i in INSTANCES for o in ("cyclic", "permutation")]
    indep = [out["inst"][i]["cool"]["independent"] for i in INSTANCES]
    steps = [out["inst"][i]["step"] for i in INSTANCES]
    out.update({"perm": (min(perm), max(perm)), "indep": (min(indep), max(indep)), "steps": (min(steps), max(steps))})
    _expect("certified cooldown, permutation orders, smallest", out["perm"][0], 3.3e4)
    _expect("certified cooldown, permutation orders, largest", out["perm"][1], 8.9e4)
    _expect("certified cooldown, independent sampling, smallest", out["indep"][0], 6.3e5)
    _expect("certified cooldown, independent sampling, largest", out["indep"][1], 8.6e5)
    _expect("tuned step / Lyapunov step, smallest", out["steps"][0], 350.0)
    _expect("tuned step / Lyapunov step, largest", out["steps"][1], 1400.0)
    return out


def data():
    return {"saga": rows_saga(), "polar": rows_polar()}


def print_all(D):
    S, P = D["saga"], D["polar"]
    print("row 1  exact / certified exponent, passes 2-20 (results/B3_summary.json, attribution[*]['speedup']):")
    print("       " + ", ".join(f"{k}: {v:.3g}" for k, v in S["speed"].items()))
    print(f"       smallest {S['speed'][S['lo_key']]:.4f} ({S['lo_key']}), largest {S['speed'][S['hi_key']]:.4f} "
          f"({S['hi_key']}; results/digest.md: {S['digest']:g}); drawn: 3.2 to 220")
    print(f"row 2  observed exponent per pass at gamma R^2 = 1/4: {S['observed']:.4f}; best certificate of the proof "
          f"per pass: {S['best_cert']:.4f}; ratio {S['ratio_cert']:.3f} (drawn: 4.4)")
    print(f"row 3  the proof admits gamma R^2 < {S['largest_proved']:g}; rerun (seed 0) at gamma R^2 = {STEP_OK:g}: "
          f"squared distance / initial = {S['rel'][STEP_OK][-1]:.2e} after {PASSES_OK} passes (converges; ratio of "
          f"steps {S['step_ok']:g}); at gamma R^2 = {STEP_STALL:g}: {S['rel'][STEP_STALL][40]:.3f} after 40 passes, "
          f"{S['rel'][STEP_STALL][-1]:.3f} after {PASSES_STALL} (stalls; ratio of steps {S['step_stall']:g}); slopes "
          f"stored in GammaDelta_summary.json (passes 2-40, three seeds): {S['stored_rate'][STEP_OK]:.3f} and "
          f"{S['stored_rate'][STEP_STALL]:.3f} per pass (the latter is the end of a transient: AUDIT.md, finding A)")
    for name in INSTANCES:
        I = P["inst"][name]
        print(f"rows 4-5  {name}: n = {I['n']}, L_max = {I['L_max']:.4g}, L_op = {I['L_op']:.4g}, mu_op = "
              f"{I['mu_op']:.4g}, q used = {I['q_used']}; log(1/q) / log(1/q_cert): "
              + ", ".join(f"{o} (A = {I['ages'][o]}) {I['cool'][o]:.3g}" for o in I["cool"]))
    print(f"row 4  permutation orders (cyclic, fresh permutation), two instances: {P['perm'][0]:.3g} to "
          f"{P['perm'][1]:.3g} (paper: 3.3e4 to 8.9e4)")
    print(f"row 5  independent sampling, T = {T_PASSES} passes, delta = {DELTA}: {P['indep'][0]:.3g} to "
          f"{P['indep'][1]:.3g} (audit: 6.3e5 to 8.6e5)")
    for name in INSTANCES:
        I = P["inst"][name]
        print(f"row 6  {name}: 1/(14 n L_max) = {I['gamma_cert_R2op']:.4g} / R2_op, tuned step "
              f"{I['gamma_tuned_R2op']:g} / R2_op, ratio {I['step']:.0f}")
    print(f"       drawn: {P['steps'][0]:.0f} to {P['steps'][1]:.0f} (paper: 350 to 1400); on the six instances of M1 "
          f"(not drawn): " + ", ".join(f"{k} {v:.0f}" for k, v in P["all_six"].items()))


# ----------------------------------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------------------------------
def sci(x):
    """2 significant digits, as a mathtext power of ten (3.3e4 -> 3.3 . 10^4)."""
    e = int(math.floor(math.log10(x)))
    return rf"${x / 10 ** e:.1f}\cdot10^{{{e}}}$"


def draw(D):
    ps = es.setup("talk")
    ink = ps.INK
    S, P = D["saga"], D["polar"]
    EU, PO = ps.BLUE, ps.ORANGE                       # Euclidean SAGA: blue; polar steps with a table: orange
    fig = es.figure("talk", height=2.75)
    ax = fig.add_subplot(111)
    Y = [6.0, 5.0, 4.0, 2.7, 1.7, 0.7]                # the six rows, top to bottom (a wider gap between the groups)
    val = dict(fontsize=es.small(), color=ink["secondary"], va="center", textcoords="offset points")
    BAR, DOT = dict(lw=6.0, solid_capstyle="butt", zorder=3), dict(ls="", marker="o", ms=7.5, mew=1.0, zorder=4)

    b = es.Build()                                    # state 1: SAGA (rows 1-3); state 2: polar steps (rows 4-6)

    def bar(y, lo, hi, colour):
        return ax.plot([lo, hi], [y, y], color=colour, **BAR)

    # rows 1-3: SAGA against Proposition 5.9
    lo, hi = S["speed"][S["lo_key"]], S["speed"][S["hi_key"]]
    bar(Y[0], lo, hi, EU)
    ax.annotate(f"{lo:.1f} to {hi:.0f}", (hi, Y[0]), xytext=(6, 0), ha="left", **val)
    ax.plot([S["ratio_cert"]], [Y[1]], mfc=EU, mec=ink["surface"], **DOT)
    ax.annotate(f"{S['ratio_cert']:.1f}", (S["ratio_cert"], Y[1]), xytext=(8, 0), ha="left", **val)
    ax.plot([S["step_ok"]], [Y[2]], mfc=EU, mec=ink["surface"], **DOT)
    es.mark(ax, S["step_stall"], Y[2] - 0.03, False, size=es.small() + 2)
    ax.annotate(f"{S['step_ok']:g} converges, {S['step_stall']:g} stalls", (S["step_stall"], Y[2]), xytext=(9, 0),
                ha="left", **val)
    # rows 4-6: polar steps fed by a SAGA table
    b.add(2, bar(Y[3], *P["perm"], PO),
          ax.annotate(sci(P["perm"][0]), (P["perm"][0], Y[3]), xytext=(-6, 0), ha="right", **val),
          ax.annotate(sci(P["perm"][1]), (P["perm"][1], Y[3]), xytext=(6, 0), ha="left", **val),
          bar(Y[4], *P["indep"], es.tint(PO, 0.45)),          # the audit's computation: lighter
          ax.annotate(f"{sci(P['indep'][0])} to {sci(P['indep'][1])}\n(not in the paper)", (P["indep"][0], Y[4]),
                      xytext=(-6, 0), ha="right", linespacing=1.15, **val),
          bar(Y[5], *P["steps"], PO),
          ax.annotate(f"{P['steps'][0]:.0f} to {round(P['steps'][1], -2):.0f}", (P["steps"][1], Y[5]),
                      xytext=(6, 0), ha="left", **val))

    # row labels: what is compared (first line), where it is measured (second line)
    labels = [
        ("exact / certified exponent", "15 logistic problems, passes 2\u201320"),
        ("observed / best certified exponent", r"$\kappa = 10^3$, run at $\gamma R^2 = 1/4$; proof at its best"),
        ("converging step / largest proved step", r"$\kappa = 10^3$, one run per step"),
        ("cooldown rate $\\log(1/q)$: used / certified", "Proposition 5(b), permutation orders"),
        ("the same for independent sampling", rf"as in the runs; {T_PASSES} passes, $\delta = {DELTA}$"),
        ("tuned step / Lyapunov-certified step", "nuclear-scaled polar step, batch one"),
    ]
    tr = ax.get_yaxis_transform()
    for k, (y, (what, where)) in enumerate(zip(Y, labels)):
        two = [ax.text(-0.025, y + 0.235, what, transform=tr, ha="right", va="center", fontsize=es.small(),
                       color=ink["primary"]),
               ax.text(-0.025, y - 0.235, where, transform=tr, ha="right", va="center", fontsize=es.small(),
                       color=ink["muted"])]
        if k >= 3:
            b.add(2, *two)
    # the two groups, named in the empty part of their rows
    head = dict(fontsize=es.small(), color=ink["primary"], fontweight="bold", va="center", linespacing=1.2)
    # "of Bach": not to be read as part of Proposition 5 of the paper (row 4)
    ax.text(5.5e6, Y[1], "SAGA and its proof\n(Proposition 5.9 of Bach)", ha="right", **head)
    b.add(2, ax.text(1.5, 0.5 * (Y[3] + Y[4]), "polar steps fed by\na SAGA table\n(two instances)", ha="left", **head),
          ax.axhline(0.5 * (Y[2] + Y[3]), color=ink["axis"], lw=0.8, zorder=1))
    ax.axvline(1.0, color=ink["primary"], lw=1.0, zorder=1)

    ax.set_xscale("log")
    ax.set_xlim(0.7, 7e6)
    ax.set_xticks([10.0 ** k for k in range(7)])
    ax.set_xticklabels(["1", "10"] + [rf"$10^{k}$" for k in range(2, 7)])
    ax.minorticks_off()
    ax.set_ylim(0.1, 6.6)
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("what is run / what is certified")
    es.save(fig, NAME, "talk", build=b, note=NOTE)


def main():
    D = data()
    print_all(D)
    draw(D)


if __name__ == "__main__":
    main()
