"""Figure T05 of the lecture (paper/talk/figures/figT05_bound_vs_run): the hook of Finding 1. On one ridge logistic
regression with kappa = 1e4, the squared distance of SAGA to the minimizer falls by more than eleven orders of
magnitude in 100 passes, while the bound of Proposition 5.9 of Bach's book falls by the factor 0.15. Explanatory
figure, not in the paper: it is panel (a) of Figure 1 of paper/paper_saga_to_muon.tex revealed in three states, with
the vertical axis divided by the initial squared distance. It illustrates (C1) and (C3) and adds no claim.

Data. results/B3.pkl (the stored runs behind Figure 1) is absent from this machine, so the five runs are recomputed
here with the code of the project, exactly as experiments/exp_B3_tightness.py defines them (nothing is written):
    prob  = vrlab.problems.make_logistic(1000, 50, kappa=1e4, alpha=1.0, seed=0); vrlab.solvers.solve_optimum(prob)
            (l2-regularized logistic regression, n = 1000, d = 50, covariance spectrum j^(-1), one data set)
    runs  = vrlab.algorithms.saga(prob, zeros(50), passes=100, seed=s, record_every=50, instrument_every=100,
            stop_D=1e-26) for the sampling seeds s = 0..4: SAGA with the constants of Proposition 5.9
            (gamma = 1/(4 R^2), Lyapunov weight 3 gamma^2), table initialized at theta_0 = 0
and the three curves as paper/scripts/make_figR1_classical.py (data_curves) computes them:
    bound       vrlab.bounds.saga_bound(t, n, R^2, mu, D0) = (1 - min{1/(3n), 3 mu/(16 R^2)})^t (1 + 3n/16) D0, with
                the constant 1 + 3n/16 obtained in the proof (the statement of the proposition rounds it to 1 + n/4);
                checked here against this closed form;
    SAGA        mean over the five seeds of ||theta_t - eta_*||^2 (vrlab.analysis.mean_over_seeds), and its range;
    a posteriori  vrlab.analysis.trajectory_bounds(run of seed 0, which=("mu_eff+L_eff",), v_min=1e-18): V_0 times the
                product of the factors 1 - min{1/(3n), 2 gamma c mu_eff}, c = 1 - gamma L - 3 gamma L / 2,
                L = min(L_eff, R^2), with mu_eff and L_eff measured along that run every 100 iterations.
Every curve is divided by D0 = ||theta_0 - eta_*||^2 (the same for all seeds).

Checks (the script stops if one fails): the constants of the problem and the final mean squared distance against
results/B3_summary.json (problems and attribution, key "kappa=10000,alpha=1": R2, mu, n, final_D; the conventions
are the same: mean over the five seeds of the last recorded D), the final value against the cell "final E D_t" of
results/digest.md, and the two numbers of the message against results/findings.md (11.4 orders of magnitude;
factor e^(-1.875) = 0.15 for the bound).

Status. Proved: the bound (Proposition 5.9). Measured: the five runs. A-posteriori reading, not a guarantee: the
yellow curve (it uses quantities measured along one run; (C3) of the paper). The figure is one instance; for
kappa <= 1e2 the ratio of the exponents is 3.2-3.7 only ((C1)).

Progressive build (states s1-s3): s1 the bound of Proposition 5.9; s2 adds SAGA (mean of five runs, range as a band:
it is thinner than the line); s3 adds the a-posteriori bound.

The script prints every number it plots (for the curves: the values at passes 0, 20, 50 and 100).
Run:    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/scripts/make_figT05_bound_vs_run.py   (about 35 s)
Output: paper/talk/figures/figT05_bound_vs_run.pdf and .png, and the states _s1 ... _s3
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import explain_style as es  # noqa: E402

if es.ROOT not in sys.path:
    sys.path.insert(0, es.ROOT)

NAME = "figT05_bound_vs_run"
N, DIM, KAPPA, ALPHA, DATA_SEED = 1000, 50, 1e4, 1.0, 0
SEEDS = (0, 1, 2, 3, 4)                 # sampling seeds
PASSES = 100
RECORD_EVERY, INSTRUMENT_EVERY = 50, 100          # as experiments/exp_B3_tightness.py (n // 20, n // 10)
V_MIN = 1e-18                           # floor of the trajectory products, as in Figure 1
WORKERS = 2
KEY = f"kappa={KAPPA:g},alpha={ALPHA:g}"
PASSES_PRINT = (0, 20, 50, 100)
DIGEST_TABLE = "### Attribution of the log-gap (passes 2-20) and window-averaged ratios"
FINDINGS_DECADES, FINDINGS_FACTOR = "11.4", "0.15"   # results/findings.md, lines 51-53


# ----------------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------------
def problem():
    from vrlab import problems, solvers
    prob = problems.make_logistic(N, DIM, kappa=KAPPA, alpha=ALPHA, seed=DATA_SEED)
    solvers.solve_optimum(prob)
    return prob


def one_run(seed):
    """One SAGA run with the constants of Proposition 5.9 (the problem is rebuilt in the worker: deterministic)."""
    from vrlab import algorithms
    r = algorithms.saga(problem(), np.zeros(DIM), passes=PASSES, seed=seed, record_every=RECORD_EVERY,
                        instrument_every=INSTRUMENT_EVERY, stop_D=1e-26)
    r.pop("theta", None)
    return r


def digest_fmt(x):
    """Number format of experiments/make_digest.py."""
    if x == 0:
        return "0"
    if abs(x) >= 1e4 or abs(x) < 1e-3:
        return f"{x:.3g}"
    return f"{x:.4g}"


def digest_cell(config, column):
    """One cell of the table DIGEST_TABLE of results/digest.md."""
    with open(os.path.join(es.ROOT, "results", "digest.md"), encoding="utf-8") as f:
        lines = f.read().splitlines()
    found = [k for k, line in enumerate(lines) if line.strip() == DIGEST_TABLE]
    if len(found) != 1:
        raise SystemExit(f"results/digest.md: heading {DIGEST_TABLE!r} found {len(found)} times")
    header = [c.strip() for c in lines[found[0] + 1].strip().strip("|").split("|")]
    for line in lines[found[0] + 3:]:
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[0] == config:
            return dict(zip(header, cells))[column]
    raise SystemExit(f"results/digest.md: no row {config!r}")


def data():
    from vrlab import analysis, bounds
    with mp.get_context("fork").Pool(WORKERS) as pool:
        runs = pool.map(one_run, SEEDS)
    it, mean, allD = analysis.mean_over_seeds(runs, "D")
    r0 = runs[0]
    if not (np.array_equal(it, np.asarray(r0["iters"])) and int(it[-1]) == PASSES * N):
        raise SystemExit("recording grids differ or a run is shorter than 100 passes")
    if not all(r["D0"] == r0["D0"] and r["stopped"] is None for r in runs):
        raise SystemExit("the runs do not share D0 or one stopped early")
    R2, mu, D0 = r0["R2"], r0["mu"], r0["D0"]
    bound = bounds.saga_bound(it, N, R2, mu, D0)
    rate_mem, rate_par = 1.0 / (3 * N), 3 * mu / (16 * R2)
    rho = 1.0 - min(rate_mem, rate_par)
    closed = rho ** it.astype(float) * (1 + 3 * N / 16) * D0
    if not np.allclose(bound, closed, rtol=1e-9, atol=0.0):
        raise SystemExit("vrlab.bounds.saga_bound differs from the closed form of Proposition 5.9")
    post = analysis.trajectory_bounds(r0, which=("mu_eff+L_eff",), v_min=V_MIN)["mu_eff+L_eff"]
    if not np.all(np.isfinite(post)):
        raise SystemExit("a-posteriori bound cut by the V < 1e-18 floor")
    D = {"passes": it / N, "bound": bound / D0, "mean": mean / D0, "lo": allD.min(0) / D0, "hi": allD.max(0) / D0,
         "post": post / D0, "R2": R2, "mu": mu, "D0": D0, "gamma": r0["gamma"], "Delta": r0["Delta"],
         "rho": rho, "active": "memory" if rate_mem < rate_par else "parameter", "final_mean": float(mean[-1]),
         "final_each": [float(r["D"][-1]) for r in runs], "V0": float(r0["V"][0]),
         "factor_bound": float(bound[-1] / bound[0]), "decades": float(np.log10(mean[0] / mean[-1])),
         "factors": len(r0["instr"])}
    check(D)
    return D


def check(D):
    """Against results/B3_summary.json, results/digest.md and results/findings.md; stops on a disagreement."""
    with open(os.path.join(es.ROOT, "results", "B3_summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    prob, att = summary["problems"][KEY], summary["attribution"][KEY]
    bad = []
    if not (prob["n"] == N and np.isclose(prob["R2"], D["R2"], rtol=1e-12) and np.isclose(prob["mu"], D["mu"], rtol=1e-12)):
        bad.append(f"constants: recomputed R2 {D['R2']!r}, mu {D['mu']!r}; summary {prob['R2']!r}, {prob['mu']!r}")
    D["final_rel_dev"] = abs(D["final_mean"] / att["final_D"] - 1.0)
    if D["final_rel_dev"] > 1e-6:
        bad.append(f"final mean D: recomputed {D['final_mean']!r}, B3_summary.json {att['final_D']!r}")
    cell = digest_cell(KEY, "final E D_t")
    if digest_fmt(D["final_mean"]) != cell:
        bad.append(f"final mean D: recomputed {digest_fmt(D['final_mean'])}, digest {cell}")
    if f"{D['decades']:.1f}" != FINDINGS_DECADES:
        bad.append(f"decades over 100 passes: recomputed {D['decades']:.3f}, findings.md {FINDINGS_DECADES}")
    if f"{D['factor_bound']:.2f}" != FINDINGS_FACTOR or abs(D["factor_bound"] / np.exp(-1.875) - 1.0) > 1e-3:
        bad.append(f"factor of the bound over 100 passes: recomputed {D['factor_bound']:.5f}, findings.md "
                   f"e^-1.875 = {np.exp(-1.875):.5f}")
    if bad:
        raise SystemExit("CHECK FAILED, no figure drawn:\n   " + "\n   ".join(bad))
    D["summary_final"], D["digest_final"] = att["final_D"], cell


def print_all(D):
    p = D["passes"]
    print(f"instance: ridge logistic regression, n = {N}, d = {DIM}, kappa = {KAPPA:g}, spectrum j^-{ALPHA:g}, data "
          f"seed {DATA_SEED}; R^2 = {D['R2']:.6g}, mu = {D['mu']:.6g}; SAGA, gamma = 1/(4 R^2) = {D['gamma']:.6g}, "
          f"sampling seeds {list(SEEDS)}; recorded every {RECORD_EVERY} iterations ({len(p)} points per curve)")
    print(f"D0 = ||theta_0 - eta_*||^2 = {D['D0']:.6g} (all curves are divided by D0); V0 / D0 = {D['V0'] / D['D0']:.4f}")
    print(f"bound: rho = 1 - min{{1/(3n), 3 mu/(16 R^2)}} = {D['rho']:.8f} ({D['active']} term active); constant "
          f"1 + 3n/16 = {1 + 3 * N / 16:g}; factor over 100 passes rho^(100 n) = {D['factor_bound']:.5f} "
          f"(e^-1.875 = {np.exp(-1.875):.5f})")
    print(f"a-posteriori bound: {D['factors']} factors measured along seed {SEEDS[0]} every {INSTRUMENT_EVERY} iterations")
    print("passes | bound / D0 | a-posteriori bound / D0 (seed 0) | SAGA mean / D0 [min, max over the 5 seeds]")
    for q in PASSES_PRINT:
        j = int(np.argmin(np.abs(p - q)))
        print(f"{q:6d} | {D['bound'][j]:10.4f} | {D['post'][j]:32.4e} | {D['mean'][j]:.4e} "
              f"[{D['lo'][j]:.4e}, {D['hi'][j]:.4e}]")
    print(f"SAGA over 100 passes: mean / D0 = {D['mean'][-1]:.3e}, i.e. {D['decades']:.2f} orders of magnitude; "
          f"largest max/min over the seeds along the run = {np.max(D['hi'] / D['lo']):.4f} (the band is thinner than "
          f"the line); a-posteriori bound / mean between {np.min(D['post'] / D['mean']):.3g} and "
          f"{np.max(D['post'] / D['mean']):.3g}")
    print(f"decrease per pass over the 100 passes (log of first/last value, / 100): bound "
          f"{-np.log(D['factor_bound']) / 100:.5f}, a-posteriori bound {np.log(D['post'][0] / D['post'][-1]) / 100:.4f}, "
          f"SAGA mean {np.log(D['mean'][0] / D['mean'][-1]) / 100:.4f}")
    print(f"checks: final mean D = {D['final_mean']:.6e} (per seed: " + ", ".join(f"{v:.3e}" for v in D["final_each"])
          + f"); results/B3_summary.json final_D = {D['summary_final']:.6e} (relative deviation "
          f"{D['final_rel_dev']:.1e}); results/digest.md 'final E D_t' = {D['digest_final']}; results/findings.md: "
          f"{FINDINGS_DECADES} orders of magnitude, factor {FINDINGS_FACTOR} for the bound: reproduced")


# ----------------------------------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------------------------------
def draw(D):
    ps = es.setup("talk")
    ink = ps.INK
    BOUND, RUN, POST = ps.RED, ps.BLUE, ps.YELLOW          # as in Figure 1(a) of the paper
    fig = es.figure("talk", height=2.72)
    ax = fig.add_subplot(111)
    b = es.Build()
    p = D["passes"]

    ax.plot(p, D["bound"], color=BOUND, lw=2.4, zorder=3)
    b.add(2, ax.fill_between(p, D["lo"], D["hi"], color=RUN, alpha=0.18, lw=0, zorder=4))
    b.add(2, ax.plot(p, D["mean"], color=RUN, lw=2.4, zorder=6))
    b.add(3, ax.plot(p, D["post"], color=POST, lw=2.2, zorder=5))

    ax.set_yscale("log")
    ax.set_xlim(0, 100)
    ax.set_ylim(1e-13, 3e4)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([1e-12, 1e-8, 1e-4, 1e0, 1e4])
    ax.minorticks_off()
    ax.set_xlabel("passes over the data, $t/n$")
    ax.set_ylabel(r"$\|\theta_t-\eta_\ast\|^2\,/\,\|\theta_0-\eta_\ast\|^2$")

    def at(curve, q):
        return float(curve[int(np.argmin(np.abs(p - q)))])

    kw = dict(fontsize=es.small() + 0.5, color=ink["primary"], linespacing=1.25, zorder=8)
    leader = dict(arrowstyle="-", color=ink["muted"], lw=0.8, shrinkA=3.0, shrinkB=2.5)
    ax.text(99.0, at(D["bound"], 99.0) / 2.6, rf"bound of Proposition 5.9: $\times\,{D['factor_bound']:.2f}$ in 100 passes",
            ha="right", va="top", **kw)
    b.add(2, ax.annotate("SAGA, mean of 5 runs:\n" rf"$\times\,10^{{-{D['decades']:.1f}}}$ in 100 passes",
                         xy=(42.0, at(D["mean"], 42.0)), xytext=(35.0, 3e-10), ha="right", va="center",
                         arrowprops=dict(leader, relpos=(1.0, 0.75)), **kw))
    ax.text(1.5, at(D["bound"], 1.5) * 2.0, r"starts at the constant $1 + 3n/16$ of the proof", ha="left",
            va="bottom", fontsize=es.small(), color=ink["secondary"], zorder=8)
    b.add(3, ax.annotate("a-posteriori bound, with " r"$\mu_{\mathrm{eff}}$, $L_{\mathrm{eff}}$"
                         "\nmeasured along one run:\nexplains, does not certify",
                         xy=(55.0, at(D["post"], 55.0)), xytext=(99.0, 1.2e-3), ha="right", va="center",
                         arrowprops=dict(leader, relpos=(0.0, 0.25)), **kw))
    es.save(fig, NAME, "talk", build=b,
            note=(rf"ridge logistic regression, $n = {N}$, $d = {DIM}$, $\kappa = 10^{{{int(np.log10(KAPPA))}}}$, spectrum "
                  rf"$j^{{-1}}$; step $1/(4R^2)$; {len(SEEDS)} seeds, one data set"))


def main():
    D = data()
    print_all(D)
    draw(D)


if __name__ == "__main__":
    main()
