"""Figure T07 of the lecture (paper/talk/figures/figT07_two_identities): along runs of SAGA the exact one-step
contraction is the certified one times a bounded slack (at most 8/3) times the curvature that the run meets
(mu_eff / mu, bounded only by a condition number). Explanatory figure, not in the paper. It explains Eq. (1) and the
findings (C1)-(C3) of paper/paper_saga_to_muon.tex; its companion for the polar step on quadratics (Eq. (2)) is
figT07b_polar_identity.

What is drawn. SAGA with the constants of Proposition 5.9 of Bach's book (gamma = 1/(4 R^2), Delta = 3 gamma^2,
z_i = f_i'(theta_0), theta_0 = 0) on l2-regularized logistic regression, n = 1000, d = 50 (vrlab.problems.make_logistic,
data seed 0: one data set per spectrum j^(-alpha)), Lyapunov function V = D + Delta M. Eq. (1) of the paper:
    E_t[V_{t+1}] = c_D D + c_M Delta M,
    c_D = 1 - (mu_eff / (2 R^2)) (1 - (L_eff / R^2)(r_sq / 4 + 3/8)),    c_M = 1 - (1/n)(1 - (2/3) r_sq r_var),
and the certified factor is rho_bound = max{1 - 3 mu / (16 R^2), 1 - 1/(3n)} (worst-case values r_sq = r_var = 1,
L_eff = R^2, mu_eff = mu). Dividing the contraction gap of the parameter term by its certified value,
    (1 - c_D) / (3 mu / (16 R^2)) = B_t x (mu_eff / mu),      B_t = (1 - (L_eff / R^2)(r_sq / 4 + 3/8)) / (3/8) <= 8/3
(B_t is the joint slack of I1, square norm, and I2, cocoercivity: r_sq >= 0 and L_eff >= 0 give the bound 8/3;
mu_eff / mu is the slack of I4, strong convexity, bounded only by a condition number).

Panels:
  (a) kappa = 1e4, alpha = 1, where the parameter term sets the certificate (rho_bound = 1 - 3 mu/(16 R^2)) and
      carries more than 92 % of V, passes 0.1-100 on a log scale (the state at pass 0 is printed, not drawn), exact
      conditional expectations at every tenth of a pass (vrlab.algorithms.saga with instrument_every, enumeration of
      the n indices): the ratio (1 - rho_exact) / (1 - rho_bound) of the exact to the certified
      contraction gap of V (ink); B_t and its bound 8/3 (colour of I2); mu_eff / mu and the level
      lambda_min(F''(eta_*)) / mu that it reaches (colour of I4); the product B_t mu_eff / mu (dashed, grey). The
      product is not the ink curve: rho_exact = Dshare c_D + (1 - Dshare) c_M also weighs the memory term
      (Dshare = D / V); the script prints both. Median over the sampling seeds 0-4 and range (the range is thinner
      than the lines: the five runs differ by at most a few percent).
  (b) the 15 problems of (C1) (kappa = 10 ... 1e5, alpha = 0, 1, 2): the measured ratio of the exact to the certified
      contraction exponent (window means over passes 2-20, mean over 5 seeds: results/B3_summary.json,
      attribution.<problem>.speedup) against the product of the two factors at the geometric means over the same
      window (mu_eff_geomean, L_eff_geomean, r_sq_geomean; problems.<problem>.mu, R2). Filled: kappa >= 1e4, where the
      product reproduces 91-96 % of the ratio (the script stops if this sentence of (C2) is not reproduced); open:
      kappa <= 1e3, where the memory term can bind ("open: memory term can bind" on the figure), so that the
      product, which is the factor of the parameter term alone, need not match: the certificate is the memory rate
      1/(3n) for kappa <= 1e2 (where the paper says that the memory term is active), and at kappa = 1e3 the exact
      contraction gap is 0.96e-3 per iteration for alpha = 0, 1, near the ceiling 1/n of the memory term (product /
      measured 2.94 and 1.15), but 0.63e-3 for alpha = 2, where the product matches (0.99): hence "can bind", not
      "is active", for the open markers as a group (c_exact_per_iter of results/B3_summary.json).

Checks (the script stops if one fails): identity (1), rho_exact = Dshare c_D + (1 - Dshare) c_M, at each of the 5000
instrumented states (relative error of the contraction gap below 1e-9); B_t <= 8/3 and mu_eff >= mu at each state; the
five runs reproduce results/B3_summary.json for this problem (constants, speed-up over passes 2-20 recomputed with
vrlab.analysis.attribution, geometric means of mu_eff, L_eff, r_sq, mean final squared distance); the 15 speed-ups of
the summary agree with results/digest.md; "2.45-2.62", "reaches 81" and "91-96 %" of (C2) are reproduced.

Status. Proved: Eq. (1) and the bound 8/3 on B_t (main report, Proposition "exact contraction in terms of the
ratios"). Measured exactly along runs (conditional expectations by enumeration, no sampling error): every curve of
(a). Measured (one data set per alpha, five sampling seeds, window passes 2-20): the points of (b).

Progressive build (states s1-s4): s1 panel (a) with the ratio exact / certified; s2 adds the bounded slack; s3 adds
the curvature ratio and the product; s4 adds panel (b).

The script prints every number it plots. Runs: 5 runs of 100 passes with exact instrumentation, 2 worker processes,
about 25 s.
Run:    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/scripts/make_figT07_two_identities.py
Output: paper/talk/figures/figT07_two_identities.pdf and .png, and the states _s1 ... _s4
"""
from __future__ import annotations

import json
import multiprocessing
import os
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import explain_style as es  # noqa: E402

if es.ROOT not in sys.path:
    sys.path.insert(0, es.ROOT)

NAME = "figT07_two_identities"
N, DIM = 1000, 50
KAPPA_A, ALPHA_A = 1.0e4, 1.0            # problem of panel (a)
PASSES = 100
SEEDS = (0, 1, 2, 3, 4)
WORKERS = 2
INSTRUMENT_EVERY = N // 10                # exact conditional expectations every tenth of a pass
WINDOW = (2, 20)                          # passes of the window of (C1)-(C2), panel (b)
V_MIN = 1e-18                             # floor of the attribution windows of the project
KAPPAS = (10.0, 100.0, 1000.0, 10000.0, 100000.0)
ALPHAS = (0.0, 1.0, 2.0)
SLACK_BOUND = 8.0 / 3.0
FIELDS = ("t", "rho_exact", "rho_bound", "rho_D", "rho_M", "mu_eff", "L_eff", "r_sq", "r_var", "Dshare", "V")
DIGEST_TABLE = "### Attribution of the log-gap (passes 2-20) and window-averaged ratios"
PASSES_PRINT = (0, 0.1, 1, 2, 5, 10, 20, 50, 99.9)
X_LO, X_HI = 0.1, 100.0                   # passes shown in panel (a), log scale (the state at pass 0 is printed)


def key(kappa, alpha):
    return f"kappa={kappa:g},alpha={alpha:g}"


# ----------------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------------
def problem_a():
    """The logistic problem of panel (a) with its minimizer, built as experiments/common.logistic_problem does."""
    from vrlab import problems, solvers
    prob = problems.make_logistic(N, DIM, kappa=KAPPA_A, alpha=ALPHA_A, seed=0)
    solvers.solve_optimum(prob)
    return prob


def run_one(seed):
    """One instrumented run of SAGA, with the arguments of experiments/exp_B3_tightness.py (nothing is cached)."""
    from vrlab import algorithms, analysis
    prob = problem_a()
    r = algorithms.saga(prob, np.zeros(DIM), passes=PASSES, seed=seed, record_every=max(1, N // 20),
                        instrument_every=INSTRUMENT_EVERY, stop_D=1e-26)
    att = analysis.attribution(r["instr"], r["gamma"], r["Delta"], r["R2"], r["mu"], N, t_min=WINDOW[0] * N,
                               t_max=WINDOW[1] * N, v_min=V_MIN)
    out = {k: np.array([q[k] for q in r["instr"]], dtype=np.float64) for k in FIELDS}
    out.update(seed=seed, R2=float(r["R2"]), mu=float(r["mu"]), gamma=float(r["gamma"]), Delta=float(r["Delta"]),
               final_D=float(r["D"][-1]), stopped=r["stopped"], speedup=float(att["speedup_total"]),
               hess_min=float(np.linalg.eigvalsh(prob.hessian(prob.theta_star))[0]))
    return out


def factors(r):
    """Eq. (1) at every instrumented state of a run: c_D, c_M, the ratio of contraction gaps and its two factors."""
    R2, mu = r["R2"], r["mu"]
    c_D = 1.0 - (r["mu_eff"] / (2.0 * R2)) * (1.0 - (r["L_eff"] / R2) * (r["r_sq"] / 4.0 + 0.375))
    c_M = 1.0 - (1.0 / N) * (1.0 - (2.0 / 3.0) * r["r_sq"] * r["r_var"])
    rho = r["Dshare"] * c_D + (1.0 - r["Dshare"]) * c_M
    slack = (1.0 - (r["L_eff"] / R2) * (r["r_sq"] / 4.0 + 0.375)) / 0.375
    curv = r["mu_eff"] / mu
    cert = 1.0 - r["rho_bound"]
    return {"identity_err": np.abs(rho - r["rho_exact"]) / (1.0 - r["rho_exact"]),
            "ratio": (1.0 - r["rho_exact"]) / cert, "slack": slack, "curv": curv, "product": slack * curv,
            "param_term": r["Dshare"] * (1.0 - c_D) / cert, "memory_term": (1.0 - r["Dshare"]) * (1.0 - c_M) / cert}


def load_summary():
    with open(os.path.join(es.ROOT, "results", "B3_summary.json"), encoding="utf-8") as f:
        return json.load(f)


def digest_fmt(x):
    """Number format of experiments/make_digest.py."""
    if x == 0:
        return "0"
    if abs(x) >= 1e4 or abs(x) < 1e-3:
        return f"{x:.3g}"
    return f"{x:.4g}"


def digest_speedups():
    """Column "speed-up exact/bound" of the B3 attribution table of results/digest.md: {problem: cell}."""
    with open(os.path.join(es.ROOT, "results", "digest.md"), encoding="utf-8") as f:
        lines = f.read().splitlines()
    found = [k for k, line in enumerate(lines) if line.strip() == DIGEST_TABLE]
    if len(found) != 1:
        raise SystemExit(f"results/digest.md: heading {DIGEST_TABLE!r} found {len(found)} times")
    header = [c.strip() for c in lines[found[0] + 1].strip().strip("|").split("|")]
    col = header.index("speed-up exact/bound")
    rows = {}
    for line in lines[found[0] + 3:]:
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows[cells[0]] = cells[col]
    return rows


def data_runs(summary):
    """Panel (a): the five runs, Eq. (1) at every state, and the checks against the summary of the project."""
    with multiprocessing.get_context("fork").Pool(WORKERS) as pool:
        runs = pool.map(run_one, SEEDS)
    k_a = key(KAPPA_A, ALPHA_A)
    ref_p, ref_a = summary["problems"][k_a], summary["attribution"][k_a]
    r0 = runs[0]
    if not (np.isclose(r0["R2"], ref_p["R2"], rtol=1e-12) and np.isclose(r0["mu"], ref_p["mu"], rtol=1e-12)
            and np.isclose(r0["hess_min"], ref_p["hess_star_min"], rtol=1e-6)):
        raise SystemExit("panel (a): the constants of the rebuilt problem differ from results/B3_summary.json")
    t = r0["t"]
    fac = [factors(r) for r in runs]
    for r, f in zip(runs, fac):
        if r["stopped"] is not None or not np.array_equal(r["t"], t) or r["V"].min() < V_MIN:
            raise SystemExit(f"panel (a): seed {r['seed']} stopped early, or its states are below the floor V = 1e-18")
        if not np.all(r["rho_bound"] == r["rho_D"]) or not r["rho_D"][0] > r["rho_M"][0]:
            raise SystemExit("panel (a): the certificate is not the parameter term 1 - 3 mu/(16 R^2)")
        if f["identity_err"].max() > 1e-9:
            raise SystemExit(f"panel (a): identity (1) fails at seed {r['seed']}: {f['identity_err'].max():.2e}")
        if f["slack"].max() > SLACK_BOUND or f["curv"].min() < 1.0:
            raise SystemExit("panel (a): the slack exceeds 8/3 or mu_eff < mu at an instrumented state")
    # the five runs against the summary: speed-up, geometric means over the window, final squared distance
    sel = (t >= WINDOW[0] * N) & (t <= WINDOW[1] * N)

    def geomean(name):
        return float(np.exp(np.mean([np.log(r[name][sel]) for r in runs])))
    pairs = [("speed-up", np.mean([r["speedup"] for r in runs]), ref_a["speedup"]),
             ("mu_eff geomean", geomean("mu_eff"), ref_a["mu_eff_geomean"]),
             ("L_eff geomean", geomean("L_eff"), ref_a["L_eff_geomean"]),
             ("r_sq geomean", geomean("r_sq"), ref_a["r_sq_geomean"]),
             ("final E D_t", np.mean([r["final_D"] for r in runs]), ref_a["final_D"])]
    bad = [f"{name}: rerun {v!r}, summary {w!r}" for name, v, w in pairs if not np.isclose(v, w, rtol=1e-8, atol=0.0)]
    if bad:
        raise SystemExit("panel (a): results/B3_summary.json is not reproduced: " + "; ".join(bad))
    curves = {}
    for name in ("ratio", "slack", "curv", "product", "param_term", "memory_term"):
        X = np.array([f[name] for f in fac])
        curves[name] = {"med": np.median(X, axis=0), "lo": X.min(axis=0), "hi": X.max(axis=0)}
    return {"passes": t / N, "curves": curves, "R2": r0["R2"], "mu": r0["mu"], "gamma": r0["gamma"],
            "rho_bound": float(r0["rho_bound"][0]), "rho_M": float(r0["rho_M"][0]),
            "hess_ratio": r0["hess_min"] / r0["mu"], "identity_err": max(float(f["identity_err"].max()) for f in fac),
            "states": sum(len(r["t"]) for r in runs), "checked": pairs, "speedups": [r["speedup"] for r in runs],
            "Dshare": (min(float(r["Dshare"].min()) for r in runs), max(float(r["Dshare"].max()) for r in runs))}


def data_problems(summary):
    """Panel (b): the 15 problems of (C1), from the summary of the project, with the checks of (C2)."""
    digest = digest_speedups()
    rows = []
    for a in ALPHAS:
        for k in KAPPAS:
            att, prob = summary["attribution"][key(k, a)], summary["problems"][key(k, a)]
            if digest_fmt(att["speedup"]) != digest[key(k, a)]:
                raise SystemExit(f"{key(k, a)}: speed-up {att['speedup']!r} of the summary is not that of "
                                 f"results/digest.md ({digest[key(k, a)]})")
            slack = (1.0 - (att["L_eff_geomean"] / prob["R2"]) * (att["r_sq_geomean"] / 4.0 + 0.375)) / 0.375
            curv = att["mu_eff_geomean"] / prob["mu"]
            rows.append({"kappa": k, "alpha": a, "slack": slack, "curv": curv, "product": slack * curv,
                         "speedup": att["speedup"], "c_exact": att["c_exact_per_iter"], "memory_certificate":1.0 / (3 * N) < 3.0 * prob["mu"] / (16 * prob["R2"])})
    big = [r for r in rows if r["kappa"] >= 1e4]
    share = [r["product"] / r["speedup"] for r in big]
    slacks = [r["slack"] for r in big]
    # the three numbers of (C2) for kappa >= 1e4: "2.45-2.62", "reaches 81", "91-96 %"
    ok = (f"{min(slacks):.2f}" == "2.45" and f"{max(slacks):.2f}" == "2.62" and round(max(r["curv"] for r in big)) == 81
          and round(100 * min(share)) == 91 and round(100 * max(share)) == 96)
    if not ok:
        raise SystemExit(f"(C2) is not reproduced: slack {min(slacks):.3f}-{max(slacks):.3f}, largest curvature ratio "
                         f"{max(r['curv'] for r in big):.2f}, product / speed-up {min(share):.3f}-{max(share):.3f}")
    return {"rows": rows, "share": (min(share), max(share)), "slack": (min(slacks), max(slacks)),
            "curv_max": max(r["curv"] for r in big), "speedup": (min(r["speedup"] for r in rows),
                                                              max(r["speedup"] for r in rows))}


def data():
    summary = load_summary()
    return {"a": data_runs(summary), "b": data_problems(summary)}


def print_all(D):
    A, B = D["a"], D["b"]
    c, p = A["curves"], A["passes"]
    print(f"(a) kappa = {KAPPA_A:g}, alpha = {ALPHA_A:g}, n = {N}, d = {DIM}; R^2 = {A['R2']:.6g}, mu = {A['mu']:.6g}, "
          f"gamma = 1/(4 R^2) = {A['gamma']:.6g}; certified gap 1 - rho_bound = {1 - A['rho_bound']:.6g} per iteration "
          f"(parameter term 3 mu/(16 R^2); the memory term would give {1 - A['rho_M']:.6g})")
    print(f"    {A['states']} instrumented states ({len(SEEDS)} seeds x {len(p)}); identity (1): largest relative error "
          f"of the contraction gap {A['identity_err']:.1e}; D / V between {A['Dshare'][0]:.3f} and {A['Dshare'][1]:.3f}")
    print("    reproduced from results/B3_summary.json: " + "; ".join(f"{n_} {v:.10g} (summary {w:.10g})"
                                                                     for n_, v, w in A["checked"]))
    print("    speed-up over passes 2-20 per seed: " + ", ".join(f"{v:.3f}" for v in A["speedups"]))
    print(f"    bound of the slack 8/3 = {SLACK_BOUND:.4f}; lambda_min(F''(eta_*)) / mu = {A['hess_ratio']:.4f}")
    print("    passes | exact / certified gap [min, max over seeds] | slack B_t | mu_eff / mu | product | "
          "Dshare (1 - c_D) / cert + (1 - Dshare)(1 - c_M) / cert")
    for q in PASSES_PRINT:
        j = int(np.argmin(np.abs(p - q)))
        print(f"    {p[j]:6.1f} | {c['ratio']['med'][j]:7.3f} [{c['ratio']['lo'][j]:.3f}, {c['ratio']['hi'][j]:.3f}] | "
              f"{c['slack']['med'][j]:.4f} | {c['curv']['med'][j]:7.3f} | {c['product']['med'][j]:7.3f} | "
              f"{c['param_term']['med'][j]:.3f} + {c['memory_term']['med'][j]:.3f}")
    print(f"    over the run (medians): exact / certified {c['ratio']['med'].min():.2f}-{c['ratio']['med'].max():.2f}; "
          f"slack {c['slack']['med'].min():.3f}-{c['slack']['med'].max():.3f}; mu_eff / mu "
          f"{c['curv']['med'].min():.3f}-{c['curv']['med'].max():.3f}; product / (exact / certified) "
          f"{(c['product']['med'] / c['ratio']['med']).min():.3f}-{(c['product']['med'] / c['ratio']['med']).max():.3f}; "
          f"largest max / min over seeds of exact / certified {(c['ratio']['hi'] / c['ratio']['lo']).max():.3f}")
    print(f"(b) 15 problems, window passes {WINDOW[0]}-{WINDOW[1]}, results/B3_summary.json (speed-ups checked against "
          f"results/digest.md)")
    print("    kappa   alpha | slack | mu_eff / mu | product | exact / certified exponent | product / measured | "
          "exact gap per iteration (memory ceiling 1/n = 1e-3) | marker")
    for r in B["rows"]:
        print(f"    {r['kappa']:7g} {r['alpha']:5g} | {r['slack']:.3f} | {r['curv']:11.3f} | {r['product']:7.2f} | "
              f"{r['speedup']:26.2f} | {r['product'] / r['speedup']:18.3f} | {r['c_exact']:.2e} | "
              f"{'filled' if r['kappa'] >= 1e4 else 'open'}"
              f"{' (certificate = memory rate)' if r['memory_certificate'] else ''}")
    print(f"    kappa >= 1e4: slack {B['slack'][0]:.3f}-{B['slack'][1]:.3f}, mu_eff / mu up to {B['curv_max']:.2f}, "
          f"product / measured {B['share'][0]:.3f}-{B['share'][1]:.3f} (the paper: 2.45-2.62, 81, 91-96 %); measured "
          f"ratio over the 15 problems {B['speedup'][0]:.2f}-{B['speedup'][1]:.1f} (the paper: 3.2-220)")


# ----------------------------------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------------------------------
def panel_a(ax, A, ps, b):
    ink = ps.INK
    p, c = A["passes"], A["curves"]
    SL, CU = ps.INEQ["I2"], ps.INEQ["I4"]            # bounded slack (I1, I2): colour of I2; curvature met: colour of I4
    dash = (0, (4.0, 2.2))
    note = dict(textcoords="offset points", ha="right", va="bottom", fontsize=es.small(), color=ink["secondary"])
    ax.axhline(1.0, color=ink["axis"], lw=0.9, zorder=1)
    ax.annotate("certificate", (X_LO, 1.0), xytext=(4, 2.5), **dict(note, ha="left"))
    # s1: the ratio of the exact to the certified contraction gap (drawn first: first entry of the legend)
    ax.fill_between(p, c["ratio"]["lo"], c["ratio"]["hi"], color=ink["primary"], alpha=0.15, lw=0, zorder=3)
    ax.plot(p, c["ratio"]["med"], color=ink["primary"], lw=2.4, zorder=5, label="exact / certified contraction")
    # s2: the bounded slack and its bound
    b.add(2, ax.axhline(SLACK_BOUND, color=ink["secondary"], lw=0.9, ls=dash, zorder=2))
    b.add(2, ax.fill_between(p, c["slack"]["lo"], c["slack"]["hi"], color=SL, alpha=0.18, lw=0, zorder=2))
    b.add(2, ax.plot(p, c["slack"]["med"], color=SL, lw=1.9, zorder=4, label="slack of I1, I2 (bounded)"))
    b.add(2, ax.annotate("bound 8/3", (X_HI, SLACK_BOUND), xytext=(-3, 2.5), **note))
    # s3: the curvature ratio, its level at the minimizer, and the product
    b.add(3, ax.axhline(A["hess_ratio"], color=ink["secondary"], lw=0.9, ls=dash, zorder=2))
    b.add(3, ax.fill_between(p, c["curv"]["lo"], c["curv"]["hi"], color=CU, alpha=0.18, lw=0, zorder=2))
    b.add(3, ax.plot(p, c["curv"]["med"], color=CU, lw=1.9, zorder=4, label=r"curvature met, $\mu_{\mathrm{eff}}/\mu$"))
    b.add(3, ax.annotate(r"$\lambda_{\min}(F''(\eta_\ast))/\mu$", (X_LO, A["hess_ratio"]), xytext=(4, 2.5),
                         **dict(note, ha="left")))
    b.add(3, ax.plot(p, c["product"]["med"], color=ink["muted"], lw=1.4, ls=(0, (2.2, 1.5)), zorder=6,
                     label="product of the two"))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(X_LO, X_HI)
    ax.set_ylim(0.8, 400.0)
    ax.set_xticks([0.1, 1, 10, 100])
    ax.set_xticklabels(["0.1", "1", "10", "100"])
    ax.set_yticks([1, 3, 10, 30, 100])
    ax.set_yticklabels(["1", "3", "10", "30", "100"])
    ax.minorticks_off()
    ax.set_xlabel("passes")
    ax.set_ylabel("factor over the certificate")
    es.panel_title(ax, rf"(a) SAGA runs, $\kappa = 10^{{{int(round(np.log10(KAPPA_A)))}}}$, $\alpha = {ALPHA_A:g}$")
    b.legend(ax, loc="upper right", handlelength=1.5, borderaxespad=0.15, labelspacing=0.28)


def panel_b(ax, Bd, ps, b):
    ink = ps.INK
    lim = (2.0, 400.0)
    ax.plot(lim, lim, color=ink["primary"], lw=0.9, zorder=2)
    for ai, a in enumerate(ALPHAS):
        col = ps.BLUE_RAMP[4 + 3 * ai]                   # the blue shades of Figure 1(b) of the paper
        rows = [r for r in Bd["rows"] if r["alpha"] == a]
        op = [r for r in rows if r["kappa"] < 1e4]
        fi = [r for r in rows if r["kappa"] >= 1e4]
        ax.plot([r["product"] for r in op], [r["speedup"] for r in op], ls="", marker="o", ms=6.0, mfc=ink["surface"],
                mec=col, mew=1.5, zorder=4)
        ax.plot([r["product"] for r in fi], [r["speedup"] for r in fi], ls="", marker="o", ms=7.0, mfc=col,
                mec=ink["surface"], mew=1.0, zorder=5, label=rf"$\alpha = {a:g}$")
    L2D = ps.plt.Line2D
    ax.add_line(L2D([], [], ls="", marker="o", ms=7.0, mfc=ink["muted"], mec=ink["surface"], mew=1.0,
                    label=r"$\kappa \geq 10^4$"))
    ax.add_line(L2D([], [], ls="", marker="o", ms=6.0, mfc=ink["surface"], mec=ink["muted"], mew=1.5,
                    label=r"$\kappa \leq 10^3$"))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(lim[0], 2500.0)
    ticks = [3, 10, 30, 100, 300]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(v) for v in ticks])
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(v) for v in ticks])
    ax.minorticks_off()
    ax.set_xlabel("slack $\\times$ curvature met")
    ax.set_ylabel("exact / certified, passes 2\u201320")
    kw = dict(fontsize=es.small(), color=ink["secondary"], linespacing=1.15)
    ax.text(200.0, 330.0, "equal", ha="right", va="center", **kw)
    ax.text(330.0, 2.35, "open: memory\nterm can bind", ha="right", va="bottom", **kw)
    es.panel_title(ax, "(b) 15 problems")
    ps.legend(ax, loc="upper left", handlelength=1.0, handletextpad=0.3, borderaxespad=0.15, labelspacing=0.28)
    b.add(4, ax)


def draw(D):
    ps = es.setup("talk")
    fig = es.figure("talk", height=2.75)
    fig.get_layout_engine().set(w_pad=0.03, wspace=0.03)
    gs = fig.add_gridspec(1, 2, width_ratios=(1.32, 1.0))
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    b = es.Build()
    panel_a(ax_a, D["a"], ps, b)
    panel_b(ax_b, D["b"], ps, b)
    es.save(fig, NAME, "talk", build=b,
            note=(rf"logistic, $n = {N}$, $d = {DIM}$, textbook step; one data set per $\alpha$, "
                  rf"{len(SEEDS)} seeds; (a): parameter term dominates"))


def main():
    D = data()
    print_all(D)
    draw(D)


if __name__ == "__main__":
    main()
