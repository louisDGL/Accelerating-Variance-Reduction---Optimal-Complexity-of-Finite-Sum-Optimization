"""Hypothesis A4: on the hard instance of the finite-sum lower bound (block-separable Nesterov chains),
the number of component-gradient evaluations needed to reach a fixed accuracy grows like kappa^1 for SAGA
(and L-SVRG) and like kappa^(1/2) for L-Katyusha in the regime kappa >> n; AGD scales like n sqrt(kappa),
GD like n kappa. We also include the "naive" accelerations (hypothesis A2).
Gradient counts follow vrlab.algorithms: L-SVRG, L-Katyusha and Nesterov+SVRG are charged 3 per iteration in
expectation (2 + n p), about 3/2 of the stored-reference convention of the A1 note's bounds; slopes are unaffected.
Usage: python experiments/exp_A4_hard_instance.py [--quick]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import plotstyle as ps  # noqa: E402
from vrlab import algorithms, analysis, problems, solvers  # noqa: E402

QUICK = "--quick" in sys.argv
# make_hard_instance needs c > 0, i.e. kappa < n lambda_max(T)/lambda_min(T) ~ n (2(m+1)/pi)^2 (n = 16: kappa < 4.3e5
# for m = 256, < 2.7e4 for m = 64). This is only the positivity of c. The lower bound is certified only while the chain
# is not exhausted at the target accuracy: q^(2m) <= eps with q = (sqrt(Q)-1)/(sqrt(Q)+1), Q = 1 + 4c/mu_b, i.e.
# m >= log(1/eps)/(2 log(1/q)), for which m >= (sqrt(Q)/4) log(1/eps) suffices (the dimension condition of the A1 note).
# With n = 16, m = 256, eps = 1e-8 this holds up to kappa ~ 4.4e4 (q^(2m) = 1.2e-10 at kappa = 3e4, 1.2e-5 at 1e5), so
# kappa = 1e5 lies outside the certified window; m = 64 at kappa = 1e4 (A2, A3) would need m >= 144 (181 at eps = 1e-10).
N, M = (16, 64) if QUICK else (16, 256)
KAPPAS = [1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5] if not QUICK else [1e2, 1e3, 1e4]
SEEDS = list(range(1 if QUICK else 3))
EPS = 1e-8                     # target: ||theta - eta_*||^2 <= EPS * ||theta_0 - eta_*||^2
MAX_TIME = 120 if QUICK else float(os.environ.get("VRLAB_MAX_TIME", 600))
GRID_BETAS = [0.8, 0.95, "kappa"]
GRID_GAMMAS = [1 / 6, 1 / 20]
GRID_TIME = 20 if QUICK else 45
_grid_cache = {}
TAG = "A4_quick" if QUICK else "A4"
INSTANCES = ["hard", "random"] if not QUICK else ["hard"]
METHODS = ["SAGA", "L-SVRG", "L-Katyusha", "L-Katyusha (no neg. momentum)", "Nesterov+SVRG", "AGD", "GD"]


def make_problem(kind, kappa, seed):
    if kind == "hard":
        prob = problems.make_hard_instance(N, M, kappa, seed=seed, jitter=0.0)
    else:
        prob = problems.make_diag_quadratic(N, M, kappa, seed=seed)
    solvers.solve_optimum(prob)
    return prob


def run_method(name, prob, theta0, seed, D0):
    n, R2, mu = prob.n, prob.R2, prob.mu
    kappa = prob.kappa
    log = np.log(1.0 / EPS)
    stop = EPS * D0
    rec = max(1, n // 4)
    if name == "SAGA":
        cap = int(1.5 * (3 * n + 16 * kappa / 3) * (log + np.log(1 + n / 4))) + 10 * n
        return algorithms.saga(prob, theta0, n_iters=cap, seed=seed, record_every=rec, record_F=False, stop_D=stop, max_time=MAX_TIME)
    if name == "L-SVRG":
        cap = int(1.5 * (3 * n + 16 * kappa / 3) * (log + np.log(1 + n / 4))) + 10 * n
        return algorithms.lsvrg(prob, theta0, gamma=1.0 / (6 * R2), n_iters=cap, seed=seed, record_every=rec, record_F=False, stop_D=stop, max_time=MAX_TIME)
    if name == "L-Katyusha":
        cap = int(40 * (n + np.sqrt(n * kappa)) * log) + 10 * n
        return algorithms.lkatyusha(prob, theta0, n_iters=cap, seed=seed, record_every=rec, record_F=False, stop_D=stop, max_time=MAX_TIME)
    if name == "L-Katyusha (no neg. momentum)":
        cap = int(1.5 * (3 * n + 16 * kappa / 3) * log) + 10 * n
        return algorithms.lkatyusha(prob, theta0, n_iters=cap, seed=seed, theta2=0.0, record_every=rec, record_F=False, stop_D=stop, max_time=MAX_TIME, name=name)
    if name == "Nesterov+SVRG":
        # fair comparison: the naive scheme gets the best (beta, gamma) of a 6-setting grid; L-Katyusha keeps the paper's
        # untuned parameters. The selection is cached in memory under key = (prob.name, prob.kappa) and made on the first
        # seed run with that key. On (H) the key is the same for every seed: the setting is selected on seed 0 and reused
        # for the other seeds. On (R) prob.kappa = R2/mu depends on the seed's random eigenbases and differs in the last
        # digits across seeds, so the grid is re-run and re-selected on every seed's own instance. The final A4 results
        # were produced this way (keying on the nominal kappa would select on seed 0 on both instances). The cache is not
        # persisted: after a restart the grid is re-run on the first seed that runs.
        q = np.sqrt(kappa)
        cap = int(1.5 * (3 * n + 16 * kappa / 3) * log) + 10 * n
        key = (prob.name, kappa)
        if key not in _grid_cache:
            best = None
            for b in GRID_BETAS:
                beta = (q - 1) / (q + 1) if b == "kappa" else b
                for gR2 in GRID_GAMMAS:
                    r = algorithms.nesterov_svrg(prob, theta0, gamma=gR2 / R2, beta=beta, n_iters=cap, seed=seed, record_every=rec,
                                                 record_F=False, stop_D=stop, max_time=GRID_TIME)
                    g = algorithms.grads_to_accuracy(r, EPS, key="D", relative=True)
                    print(f"      grid beta={beta:.4f} gammaR2={gR2:.3g}: {g:.3g} ({r['stopped']})", flush=True)
                    if np.isfinite(g) and (best is None or g < best[0]):
                        best = (g, beta, gR2)
            _grid_cache[key] = None if best is None else {"beta": best[1], "gammaR2": best[2]}
        gb = _grid_cache[key]
        if gb is None:
            r = algorithms.nesterov_svrg(prob, theta0, gamma=1 / (6 * R2), beta=(q - 1) / (q + 1), n_iters=10, seed=seed, record_every=1, record_F=False)
            r["grid_best"] = None
            r["D"] = np.array([r["D"][0], np.inf])
            r["grad_evals"] = np.array([0.0, 1.0])
            return r
        r = algorithms.nesterov_svrg(prob, theta0, gamma=gb["gammaR2"] / R2, beta=gb["beta"], n_iters=cap, seed=seed, record_every=rec,
                                     record_F=False, stop_D=stop, max_time=MAX_TIME)
        r["grid_best"] = gb
        return r
    if name == "AGD":
        kF = prob.L_F / mu
        cap = int(10 * np.sqrt(kF) * log) + 100
        return algorithms.agd(prob, theta0, n_iters=cap, record_every=1, record_F=False, stop_D=stop, max_time=MAX_TIME)
    if name == "GD":
        kF = prob.L_F / mu
        cap = int(1.5 * kF * log / 2) + 100
        return algorithms.gd(prob, theta0, n_iters=cap, record_every=1, record_F=False, stop_D=stop, max_time=MAX_TIME)
    raise ValueError(name)


def run_all():
    """Incremental: results are saved after every (instance, kappa) and reloaded on restart."""
    res = common.load_result(TAG)
    if res is None:
        res = {"config": dict(n=N, m=M, kappas=KAPPAS, seeds=SEEDS, eps=EPS, methods=METHODS), "grads": {}, "status": {}, "grid": {}}
    for kind in INSTANCES:
        for kappa in KAPPAS:
            if all((kind, kappa, seed, name) in res["grads"] for seed in SEEDS for name in METHODS):
                continue
            for seed in SEEDS:
                prob = make_problem(kind, kappa, seed)
                theta0 = np.zeros(prob.d)
                D0 = float(np.sum((theta0 - prob.theta_star) ** 2))
                for name in METHODS:
                    if (kind, kappa, seed, name) in res["grads"]:
                        continue
                    with common.Timer(f"{kind} kappa={kappa:g} seed={seed} {name}"):
                        r = run_method(name, prob, theta0, seed, D0)
                    g = algorithms.grads_to_accuracy(r, EPS, key="D", relative=True)
                    res["grads"][(kind, kappa, seed, name)] = g
                    res["status"][(kind, kappa, seed, name)] = r["stopped"]
                    if "grid_best" in r:
                        res["grid"][(kind, kappa, seed)] = r["grid_best"]
                    print(f"   -> grads to eps: {g:.3g}  (stopped: {r['stopped']}, final D/D0 = {r['D'][-1] / D0:.2e})", flush=True)
            common.save_result(TAG, res)
    common.save_result(TAG, res)
    return res


def plot(res):
    kinds = [k for k in INSTANCES]
    fig = ps.figure(ps.FIG_W, 5.0)
    gs = fig.add_gridspec(2, len(kinds), height_ratios=[1.0, 0.02])   # the panels, then a row holding each legend
    axes = [fig.add_subplot(gs[0, j]) for j in range(len(kinds))]
    summary = {}
    for j, (ax, kind) in enumerate(zip(axes, kinds)):
        for name in METHODS:
            vals, lo, hi = [], [], []
            for kappa in KAPPAS:
                g = [res["grads"].get((kind, kappa, s, name), np.nan) for s in SEEDS]
                g = [v for v in g if np.isfinite(v)]
                vals.append(np.exp(np.mean(np.log(g))) if g else np.nan)
                lo.append(min(g) if g else np.nan)
                hi.append(max(g) if g else np.nan)
            vals, lo, hi = np.array(vals), np.array(lo), np.array(hi)
            # least-squares slope of the per-kappa geometric means over kappa >= 10 n, with its 95% t-interval
            # (K - 2 degrees of freedom: the seeds are not used as independent replicates; GD and AGD give identical
            # counts across seeds on (H), whose instance is fixed, while on (R) each seed draws its own instance)
            fit = analysis.loglog_slope(KAPPAS, vals, xmin=10 * N)
            slope = fit["slope"]
            summary[(kind, name)] = {"grads": vals.tolist(), "grads_min": lo.tolist(), "grads_max": hi.tolist(), "slope_kappa_ge_10n": slope,
                                     "slope_ci95": [fit["ci_low"], fit["ci_high"]], "slope_n_points": fit["n_points"]}
            lab = "Nesterov+SVRG (best of a (β, γ) grid)" if name == "Nesterov+SVRG" else name
            pm = f" ± {fit['half_width']:.2f}" if np.isfinite(fit["half_width"]) else ""
            sl = f"slope {slope:.2f}{pm}" if np.isfinite(slope) else "diverges for κ ≥ 10³"
            yerr = np.vstack([np.maximum(vals - lo, 0.0), np.maximum(hi - vals, 0.0)])
            full = f"{lab} ({sl})"
            ax.errorbar(KAPPAS, vals, yerr=yerr, marker="o", markersize=5, color=ps.ALGO[name],
                        label=full if len(full) <= 34 else f"{lab}\n({sl})", capsize=2, linewidth=2.0, elinewidth=1.0)
            ax.set_xscale("log")
            ax.set_yscale("log")
        # reference slopes
        kk = np.array([10 * N, KAPPAS[-1]])
        y0 = np.nanmin([summary[(kind, m)]["grads"][0] for m in METHODS]) * 0.3
        ax.loglog(kk, y0 * (kk / kk[0]) ** 1.0, color=ps.INK["muted"], linewidth=1.0)
        ax.loglog(kk, y0 * (kk / kk[0]) ** 0.5, color=ps.INK["muted"], linewidth=1.0)
        ax.annotate(r"∝ $\kappa$", (kk[-1], y0 * (kk[-1] / kk[0])), xytext=(5, 0), textcoords="offset points", fontsize=ps.SMALL,
                    color=ps.INK["secondary"], va="center")
        ax.annotate(r"∝ $\kappa^{1/2}$", (kk[-1], y0 * (kk[-1] / kk[0]) ** 0.5), xytext=(5, 0), textcoords="offset points",
                    fontsize=ps.SMALL, color=ps.INK["secondary"], va="center")
        ax.axvline(N, color=ps.INK["axis"], linewidth=0.8)
        ax.annotate("κ = n", (N, ax.get_ylim()[0]), xytext=(2, 4), textcoords="offset points", fontsize=ps.SMALL, color=ps.INK["secondary"])
        ax.set_xlabel("κ = R²/μ  (n = %d, block size m = %d)" % (N, M))
        if j == 0:
            ax.set_ylabel("component gradients to reach\n‖θ−η*‖² ≤ 10⁻⁸ ‖θ₀−η*‖²")
        ax.set_title("hard instance: Nesterov chains" if kind == "hard" else "random-spectrum block quadratics")
        h, l = ax.get_legend_handles_labels()               # legend below the panel (the slopes differ between panels)
        ps.legend_cell(fig, gs[1, j], h, l, ncol=1)
    ps.suptitle(fig, r"Figure A4 — gradient complexity versus condition number: $\kappa^{1}$ (SAGA) against $\kappa^{1/2}$ (L-Katyusha)")
    ps.savefig(fig, "figA4_hard_instance_exponents")
    return summary


def main():
    ps.setup()
    res = run_all()
    summary = plot(res)
    common.save_json(TAG + "_summary", {f"{k[0]}|{k[1]}": v for k, v in summary.items()})
    print("done A4")


if __name__ == "__main__":
    main()
