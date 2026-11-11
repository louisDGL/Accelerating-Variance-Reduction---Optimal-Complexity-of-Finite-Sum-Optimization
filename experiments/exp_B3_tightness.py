"""Hypothesis B3 (and the three key figures of the proposal):
  Fig. 1  empirical convergence of SAGA vs the Proposition 5.9 bound (and trajectory-dependent bounds),
  Fig. 2  tightness ratio of each proof inequality along training,
  Fig. 3  which proof step is the bottleneck as the conditioning / anisotropy varies.
Setting: l2-regularized logistic regression, n = 1000, d = 50, kappa in {1e1..1e4}, covariance spectrum
j^{-alpha} with alpha in {0, 1, 2}; SAGA with the textbook step gamma = 1/(4 R^2); 5 seeds.
Usage: python experiments/exp_B3_tightness.py [--quick]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import plotstyle as ps  # noqa: E402
from vrlab import algorithms, analysis, bounds  # noqa: E402

QUICK = "--quick" in sys.argv
N, D = (300, 20) if QUICK else (1000, 50)
KAPPAS = [10.0, 100.0, 1000.0, 10000.0, 100000.0] if not QUICK else [10.0, 1000.0]
ALPHAS = [0.0, 1.0, 2.0] if not QUICK else [0.0, 2.0]
SEEDS = list(range(2 if QUICK else 5))
PASSES = 30 if QUICK else 100
TAG = "B3_quick" if QUICK else "B3"


def run_all():
    res = common.load_result(TAG)
    if res is not None:
        print("loaded cached", TAG)
        return res
    res = {"config": dict(n=N, d=D, kappas=KAPPAS, alphas=ALPHAS, seeds=SEEDS, passes=PASSES), "runs": {}, "problems": {}}
    for kappa in KAPPAS:
        for alpha in ALPHAS:
            prob = common.logistic_problem(N, D, kappa, alpha, seed=0)
            res["problems"][(kappa, alpha)] = prob.summary()
            theta0 = np.zeros(D)
            runs = []
            with common.Timer(f"SAGA kappa={kappa:g} alpha={alpha:g}"):
                for s in SEEDS:
                    r = algorithms.saga(prob, theta0, passes=PASSES, seed=s, record_every=max(1, N // 20),
                                        instrument_every=max(1, N // 10), stop_D=1e-26)
                    runs.append(common.strip_run(r))
            res["runs"][(kappa, alpha)] = runs
            print("   final D:", [f"{r['D'][-1]:.1e}" for r in runs], "r_init", f"{runs[0]['r_init']:.3f}", flush=True)
    common.save_result(TAG, res)
    return res


def fig1(res):
    kap = [k for k in KAPPAS if k >= 100] or KAPPAS
    alpha = 1.0 if 1.0 in ALPHAS else ALPHAS[-1]
    fig, axes = ps.new_fig(len(kap), 1, panel=(4.3, 3.4), sharey=False)
    for ax, kappa in zip(axes[0], kap):
        runs = res["runs"][(kappa, alpha)]
        it, meanD, allD = analysis.mean_over_seeds(runs, "D")
        passes = it / N
        r0 = runs[0]
        ps.band(ax, passes, allD.min(0), allD.max(0), ps.BOUNDS["empirical"])
        ax.semilogy(passes, bounds.saga_bound(it, N, r0["R2"], r0["mu"], r0["D0"]), color=ps.BOUNDS["prop59"], label=ps.BOUNDS_LABEL["prop59"])
        ax.semilogy(passes, bounds.saga_bound(it, N, r0["R2"], r0["mu"], r0["D0"], S0=r0["S0"]), color=ps.BOUNDS["before_I6"], label=ps.BOUNDS_LABEL["before_I6"])
        tb = analysis.trajectory_bounds(r0)
        g = np.asarray(r0["iters"]) / N
        for key in ["mu_eff", "mu_eff+L_eff", "exact"]:
            ax.semilogy(g, tb[key], color=ps.BOUNDS[key], label=ps.BOUNDS_LABEL[key], linewidth=1.6)
        ax.semilogy(np.asarray(r0["iters"]) / N, r0["V"], color=ps.BOUNDS["realized_V"], linewidth=1.2, label=ps.BOUNDS_LABEL["realized_V"])
        ax.semilogy(passes, meanD, color=ps.BOUNDS["empirical"], label=ps.BOUNDS_LABEL["empirical"], linewidth=2.4, zorder=10)
        ax.set_title(f"{ps.fmt_kappa(kappa)}, α = {alpha:g}, n = {N}")
        ax.set_xlabel("effective passes (t / n)")
        lo = max(1e-28, meanD.min() / 10)
        ax.set_ylim(lo, max(meanD.max(), r0["D0"] * (1 + N / 4)) * 3)
    axes[0][0].set_ylabel("squared distance to η*  /  Lyapunov value")
    ps.legend(axes[0][-1], loc="upper right", fontsize=7)
    fig.suptitle("Figure 1 — SAGA: empirical convergence versus the Proposition 5.9 bound and trajectory-dependent bounds", fontsize=10)
    ps.savefig(fig, "fig1_convergence_vs_bound")


def _ratio_curves(runs, key):
    T = min(len(r["instr"]) for r in runs)
    t = np.array([q["t"] for q in runs[0]["instr"][:T]])
    X = np.array([[q[key] for q in r["instr"][:T]] for r in runs])
    return t, X


def fig2(res):
    kappa = 1000.0 if 1000.0 in KAPPAS else KAPPAS[-1]
    alpha = 1.0 if 1.0 in ALPHAS else ALPHAS[-1]
    runs = res["runs"][(kappa, alpha)]
    fig, axes = ps.new_fig(2, 1, panel=(5.0, 3.6))
    ax = axes[0][0]
    tV, XV = _ratio_curves(runs, "V")
    floor = (XV < 1e-18)                      # below this the state is dominated by the precision of eta_* (rounding noise)
    for key in ["r_sq", "r_coco", "r_var", "r_sc", "r_min"]:
        t, X = _ratio_curves(runs, key)
        X = np.where(floor, np.nan, X)
        m = np.exp(np.nanmean(np.log(np.clip(X, 1e-300, None)), 0))
        ax.semilogy(t / N, m, color=ps.RATIO[key], label=ps.RATIO_LABEL[key])
        ps.band(ax, t / N, np.nanmin(X, 0), np.nanmax(X, 0), ps.RATIO[key])
    ax.plot([0], [runs[0]["r_init"]], marker="o", markersize=6, color=ps.INEQ["I6"], linestyle="none", label="r_init (I6), at t = 0")
    ax.set_ylim(1e-3, 1.5)
    ax.set_xlabel("effective passes (t / n)")
    ax.set_ylabel("tightness ratio  (LHS / RHS of the inequality)")
    ax.set_title(f"Tightness of each proof inequality ({ps.fmt_kappa(kappa)}, α = {alpha:g})")
    ps.legend(ax, loc="lower left", fontsize=7)
    # right panel: chain of contraction exponents per iteration
    ax = axes[0][1]
    r0 = runs[0]
    att = analysis.attribution(r0["instr"], r0["gamma"], r0["Delta"], r0["R2"], r0["mu"], N)
    names = ["exact", "I1", "I3", "I2", "I4", "I5"]
    colors = [ps.INK["primary"], ps.INEQ["I1"], ps.INEQ["I3"], ps.INEQ["I2"], ps.INEQ["I4"], ps.INEQ["I5"]]
    labels = ["exact one-step contraction", "after I1", "after I1, I3", "after I1, I3, I2", "after I1–I4", "Prop. 5.9 (after I5)"]
    for j, (nm, c, lb) in enumerate(zip(names, colors, labels)):
        ax.semilogy(att["t"] / N, att["c"][:, j] * N, color=c, label=lb, linewidth=2.0 if j in (0, 5) else 1.5)
    ax.set_xlabel("effective passes (t / n)")
    ax.set_ylabel("contraction exponent per pass  (−n·log ρ)")
    ax.set_title("From the exact contraction to the bound, one inequality at a time")
    ps.legend(ax, loc="upper right", fontsize=7)
    fig.suptitle("Figure 2 — where the proof of Proposition 5.9 loses: tightness ratios and the attribution chain", fontsize=10)
    ps.savefig(fig, "fig2_tightness_ratios")


def fig3(res, window=(2, 20)):
    """Share of the (log) theory-practice gap attributed to each inequality, averaged over passes in `window`."""
    summary = {}
    fig, axes = ps.new_fig(len(ALPHAS) + 1, 1, panel=(4.0, 3.4), sharey=False)
    for ai, alpha in enumerate(ALPHAS):
        ax = axes[0][ai]
        shares = {k: [] for k in ["I1", "I3", "I2", "I4", "I5"]}
        spread = {k: [] for k in ["I1", "I3", "I2", "I4", "I5"]}
        speed = []
        for kappa in KAPPAS:
            runs = res["runs"][(kappa, alpha)]
            sh = []
            sp = []
            for r in runs:
                att = analysis.attribution(r["instr"], r["gamma"], r["Delta"], r["R2"], r["mu"], N,
                                           t_min=window[0] * N, t_max=window[1] * N, v_min=1e-18)
                sh.append(att["shares"])
                sp.append(att["speedup_total"])
            sh = np.array(sh)
            shm = np.nanmean(sh, 0)
            for j, k in enumerate(["I1", "I3", "I2", "I4", "I5"]):
                shares[k].append(shm[j])
                spread[k].append((shm[j] - np.nanmin(sh[:, j]), np.nanmax(sh[:, j]) - shm[j]))
            speed.append(float(np.nanmean(sp)))
            summary[(kappa, alpha)] = {"shares": dict(zip(["I1", "I3", "I2", "I4", "I5"], map(float, shm))), "speedup": speed[-1],
                                       "shares_min_over_seeds": dict(zip(["I1", "I3", "I2", "I4", "I5"], map(float, np.nanmin(sh, 0)))),
                                       "shares_max_over_seeds": dict(zip(["I1", "I3", "I2", "I4", "I5"], map(float, np.nanmax(sh, 0))))}
        for k in ["I1", "I3", "I2", "I4", "I5"]:
            yerr = np.array(spread[k]).T
            ax.errorbar(KAPPAS, shares[k], yerr=yerr, marker="o", markersize=5, color=ps.INEQ[k], label=ps.INEQ_LABEL[k], capsize=2, linewidth=2.0, elinewidth=1.0)
            ax.set_xscale("log")
        ax.set_title(f"α = {alpha:g}  (cov. condition number d^α = {D ** alpha:g})")
        ax.set_xlabel("κ = R²/μ")
        ax.set_ylim(-0.05, 1.05)
        if ai == 0:
            ax.set_ylabel(f"share of the log-gap (passes {window[0]}–{window[1]})")
    ax = axes[0][-1]
    for ai, alpha in enumerate(ALPHAS):
        sp = [summary[(k, alpha)]["speedup"] for k in KAPPAS]
        ax.loglog(KAPPAS, sp, marker="o", markersize=5, color=ps.BLUE_RAMP[4 + 3 * ai], label=f"α = {alpha:g}")
    ax.set_xlabel("κ = R²/μ")
    ax.set_ylabel("speed-up: exact contraction / bound")
    ax.set_title("Total theory–practice gap")
    ps.legend(ax, loc="upper left", fontsize=7)
    ps.legend(axes[0][0], loc="upper left", fontsize=7)
    fig.suptitle("Figure 3 — which inequality is the bottleneck of Proposition 5.9, across conditioning and anisotropy", fontsize=10)
    ps.savefig(fig, "fig3_bottleneck_by_geometry")
    return summary


def main():
    ps.setup()
    res = run_all()
    fig1(res)
    fig2(res)
    summary = fig3(res)
    out = {"problems": {f"kappa={k:g},alpha={a:g}": v for (k, a), v in res["problems"].items()},
           "attribution": {f"kappa={k:g},alpha={a:g}": v for (k, a), v in summary.items()}}
    # per-config averages of ratios over the window
    for (kappa, alpha), runs in res["runs"].items():
        key = f"kappa={kappa:g},alpha={alpha:g}"
        d = {}
        tV, XV = _ratio_curves(runs, "V")
        for rk in ["r_sq", "r_coco", "r_var", "r_sc", "r_min", "mu_eff", "L_eff", "rho_exact", "rho_bound"]:
            t, X = _ratio_curves(runs, rk)
            sel = (t >= 2 * N) & (t <= 20 * N)
            ok = (XV[:, sel] >= 1e-18)
            vals = np.log(np.clip(X[:, sel], 1e-300, None))[ok]
            d[rk + "_geomean"] = float(np.exp(vals.mean())) if vals.size else float("nan")
        d["r_init"] = float(np.mean([r["r_init"] for r in runs]))
        d["final_D"] = float(np.mean([r["D"][-1] for r in runs]))
        # attribution over a late window and leave-one-out exponents (per-iteration, averaged over seeds)
        late = [analysis.attribution(r["instr"], r["gamma"], r["Delta"], r["R2"], r["mu"], N, t_min=20 * N, t_max=PASSES * N, v_min=1e-18) for r in runs if len(r["instr"]) > 0]
        late = [a for a in late if np.all(np.isfinite(a["shares"])) and len(a["t"]) >= 10]
        if late:
            d["shares_late_20_100"] = dict(zip(["I1", "I3", "I2", "I4", "I5"], map(float, np.mean([a["shares"] for a in late], 0))))
            d["speedup_late_20_100"] = float(np.mean([a["speedup_total"] for a in late]))
        early = [analysis.attribution(r["instr"], r["gamma"], r["Delta"], r["R2"], r["mu"], N, t_min=2 * N, t_max=20 * N) for r in runs]
        d["loo_exponents_per_iter"] = {k: float(np.mean([a["loo_exponents"][k] for a in early])) for k in ["I1", "I3", "I2", "I4", "I5"]}
        orderA = [analysis.attribution_A(r["instr"], r["gamma"], r["Delta"], r["R2"], r["mu"], N, t_min=2 * N, t_max=20 * N, v_min=1e-18) for r in runs]
        d["shares_orderA"] = dict(zip(analysis.STEP_KEYS_A, map(float, np.nanmean([a["shares"] for a in orderA], 0))))
        d["c_exact_per_iter"] = float(np.mean([a["c_exact"] for a in early]))
        d["c_bound_per_iter"] = float(np.mean([a["c_bound"] for a in early]))
        out["attribution"][key].update(d)
    common.save_json(TAG + "_summary", out)
    print("done B3")


if __name__ == "__main__":
    main()
