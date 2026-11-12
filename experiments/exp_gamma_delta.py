"""Optimising the constants (gamma, Delta) inside the proof of Proposition 5.9 versus what SAGA tolerates in practice.
Left: the best contraction exponent per pass the proof can certify as a function of the step gamma R^2 (Delta optimised),
for several kappa; the textbook choice gamma R^2 = 1/4 is marked. Right: empirical contraction exponent per pass of SAGA
on logistic regression as a function of gamma R^2 (the proof's certificate on the same axis).
Usage: python experiments/exp_gamma_delta.py [--quick]
"""
import os
import sys

import matplotlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import plotstyle as ps  # noqa: E402
from vrlab import algorithms, bounds  # noqa: E402

QUICK = "--quick" in sys.argv
N, D = (300, 20) if QUICK else (1000, 50)
KAPPA, ALPHA = 1e3, 1.0
GAMMA_R2 = [1 / 8, 1 / 4, 1 / 3, 1 / 2, 1.0, 2.0, 4.0]
SEEDS = list(range(1 if QUICK else 3))
PASSES = 20 if QUICK else 40
TAG = "GD_quick" if QUICK else "GammaDelta"


def best_rate_for_gamma(n, R2, mu, gamma):
    """max over Delta of 1 - rho(gamma, Delta) within the proof (None if no admissible Delta)."""
    cs = np.exp(np.linspace(np.log(2.0001), np.log(1e6), 4000))
    best = 0.0
    for c in cs:
        rho = bounds.saga_rate(n, R2, mu, gamma, c * gamma ** 2)
        if rho is not None and 1 - rho > best:
            best = 1 - rho
    return best


def empirical_rate(run, window=(2, PASSES)):
    it = np.asarray(run["iters"], dtype=float)
    y = np.asarray(run["D"], dtype=float)
    sel = (it >= window[0] * N) & (it <= window[1] * N) & (y > 1e-25) & np.isfinite(y)
    if sel.sum() < 3 or not np.all(np.isfinite(y)):
        return np.nan
    return -np.polyfit(it[sel], np.log(y[sel]), 1)[0]


def main():
    ps.setup()
    res = common.load_result(TAG)
    if res is None:
        prob = common.logistic_problem(N, D, KAPPA, ALPHA, seed=0)
        res = {"summary": prob.summary(), "emp": {}, "R2": prob.R2, "mu": prob.mu}
        theta0 = np.zeros(D)
        for g in GAMMA_R2:
            gamma = g / prob.R2
            rates = []
            with common.Timer(f"gamma R2 = {g:.3g}"):
                for s in SEEDS:
                    r = algorithms.saga(prob, theta0, gamma=gamma, passes=PASSES, seed=s, record_every=max(1, N // 10), record_F=False, stop_D=1e-26)
                    rates.append(empirical_rate(r))
            res["emp"][g] = rates
            print("   rates per iteration:", rates, flush=True)
        common.save_result(TAG, res)
    R2, mu = res["R2"], res["mu"]
    fig = ps.figure(ps.FIG_W, 3.6)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.02])  # the two panels, then a row holding the legend of each
    ax = fig.add_subplot(gs[0, 0])
    gr = np.exp(np.linspace(np.log(0.003), np.log(0.49), 60))
    for j, kappa in enumerate([1e2, 1e3, 1e4, 1e5]):
        mu_k = R2 / kappa
        y = [best_rate_for_gamma(N, R2, mu_k, g / R2) * N for g in gr]
        ax.loglog(gr, y, color=ps.BLUE_RAMP[3 + 3 * j], label=f"{ps.fmt_kappa(kappa)} (n = {N})")
        tb = (1 - bounds.saga_rate(N, R2, mu_k)) * N
        ax.plot([0.25], [tb], marker="o", markersize=6, color=ps.BLUE_RAMP[3 + 3 * j])
    ax.axvline(0.25, color=ps.INK["axis"], linewidth=0.8)
    # label at the foot of the textbook line, on its left: the space under the curves is empty there
    ax.annotate("textbook γ = 1/(4R²)", (0.25, ax.get_ylim()[0]), xytext=(-3, 4), textcoords="offset points", ha="right",
                fontsize=ps.SMALL, color=ps.INK["secondary"])
    ax.set_xlabel("step size γ R²")
    ax.set_ylabel("best certified contraction\nexponent per pass\n(max over Δ of −n·log ρ)")
    ax.set_title("what the proof structure can certify\n(Δ optimised)")
    h, l = ax.get_legend_handles_labels()
    ps.legend_cell(fig, gs[1, 0], h, l, ncol=2)
    ax = fig.add_subplot(gs[0, 1])
    emp = [np.nanmean(res["emp"][g]) * N for g in GAMMA_R2]
    cert = [best_rate_for_gamma(N, R2, mu, g / R2) * N for g in GAMMA_R2]
    # the certificate is 0 where no Delta is admissible, i.e. for gamma R^2 >= 1/2 (Delta > 2 gamma^2 and
    # 1 - gamma R^2 - R^2 Delta / (2 gamma) > 0 force gamma R^2 < 1/2): not drawn on the log axis (a zero would draw a
    # vertical drop from the last positive value), and the region is shaded instead
    no_delta = 0.5
    cert_plot = [c if c > 0 else np.nan for c in cert]
    ax.plot(GAMMA_R2, emp, marker="o", markersize=5, color=ps.BLUE, label="empirical SAGA (slope of log‖θ−η*‖²)")
    ax.plot(GAMMA_R2, cert_plot, marker="o", markersize=5, color=ps.RED, label="best certificate of the proof (Δ optimised)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(GAMMA_R2)
    ax.set_xticklabels(["1/8", "1/4", "1/3", "1/2", "1", "2", "4"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    xl = ax.get_xlim()
    ax.axvspan(no_delta, 10 * xl[1], color=ps.INK["grid"], alpha=0.5, zorder=0, linewidth=0)
    ax.set_xlim(xl)
    ax.annotate("no admissible Δ", (no_delta, ax.get_ylim()[0]), xytext=(4, 4), textcoords="offset points", ha="left",
                va="bottom", fontsize=ps.SMALL, color=ps.INK["secondary"])
    ax.set_xlabel("step size γ R²")
    ax.set_ylabel("contraction exponent per pass")
    ax.set_title(f"logistic regression\n{ps.fmt_kappa(KAPPA)}, α = {ALPHA:g}, n = {N}")
    h, l = ax.get_legend_handles_labels()
    ps.legend_cell(fig, gs[1, 1], h, l, ncol=1)
    ps.suptitle(fig, "Figure C — at κ = 10³, optimising (γ, Δ) inside the proof cannot close the gap:\n"
                "certified vs empirical progress per pass")
    ps.savefig(fig, "figC_gamma_delta")
    opt = bounds.saga_optimal_params(N, R2, mu)
    common.save_json(TAG + "_summary", {"optimal_within_proof": opt, "empirical_rates_per_pass": {str(g): float(np.nanmean(res["emp"][g]) * N) for g in GAMMA_R2},
                                        "certified_rates_per_pass": {str(g): float(c) for g, c in zip(GAMMA_R2, cert)}, "problem": res["summary"]})
    print("done gamma/Delta; optimal within proof:", opt)


if __name__ == "__main__":
    main()
