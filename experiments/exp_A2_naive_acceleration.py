"""Hypothesis A2 / A3: naive acceleration (Nesterov momentum + SVRG estimator, or L-Katyusha without the
negative-momentum term theta_2 w) versus loopless Katyusha, on the hard instance and on logistic regression.
Top row: trajectories of the methods (naive scheme with its best grid setting). Bottom row: gradient evaluations
to reach ||theta - eta_*||^2 <= 1e-6 ||theta_0 - eta_*||^2 for the naive scheme over a (beta, gamma) grid.
Usage: python experiments/exp_A2_naive_acceleration.py [--quick]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import plotstyle as ps  # noqa: E402
from vrlab import algorithms  # noqa: E402

QUICK = "--quick" in sys.argv
TAG = "A2_quick" if QUICK else "A2"
PASSES = 40 if QUICK else 200
EPS = 1e-6
BETAS = [0.5, 0.8, 0.9, 0.95, "kappa"]
GAMMAS = [1 / 6, 1 / 20, 1 / 60]


def problems_list():
    if QUICK:
        return [("hard instance (n=16, m=64, κ=10⁴)", common.hard_problem(16, 64, 1e4)),
                ("logistic (n=300, d=20, κ=10³, α=1)", common.logistic_problem(300, 20, 1e3, 1.0))]
    return [("hard instance (n=16, m=64, κ=10⁴)", common.hard_problem(16, 64, 1e4)),
            ("logistic (n=1000, d=50, κ=10⁴, α=1)", common.logistic_problem(1000, 50, 1e4, 1.0))]


def run_all():
    res = common.load_result(TAG)
    if res is not None:
        print("loaded cached", TAG)
        return res
    res = {"problems": [], "runs": {}, "grid": {}}
    for label, prob in problems_list():
        n = prob.n
        theta0 = np.zeros(prob.d)
        rec = max(1, n // 10)
        R2, mu = prob.R2, prob.mu
        q = np.sqrt(prob.kappa)
        beta_k = (q - 1) / (q + 1)
        # budget: logistic problems get PASSES passes; the hard instance (tiny n) gets the SAGA worst-case budget
        passes = PASSES if n >= 100 else int(1.5 * (3 * n + 16 * prob.kappa / 3) * np.log(1e8) / n)
        stop = 1e-10 * float(np.sum((theta0 - prob.theta_star) ** 2))
        runs = {}
        with common.Timer(label):
            runs["SAGA"] = algorithms.saga(prob, theta0, passes=passes, seed=0, record_every=rec, record_F=False, stop_D=stop)
            runs["L-SVRG"] = algorithms.lsvrg(prob, theta0, gamma=1 / (6 * R2), passes=passes, seed=0, record_every=rec, record_F=False, stop_D=stop)
            runs["L-Katyusha"] = algorithms.lkatyusha(prob, theta0, passes=passes, seed=0, record_every=rec, record_F=False, stop_D=stop)
            runs["L-Katyusha (no neg. momentum)"] = algorithms.lkatyusha(prob, theta0, passes=passes, seed=0, theta2=0.0, record_every=rec, record_F=False, stop_D=stop, name="L-Katyusha (no neg. momentum)")
            runs["AGD"] = algorithms.agd(prob, theta0, n_iters=int(passes), record_every=1, record_F=False, stop_D=stop)
            grid = {}
            best = None
            for b in BETAS:
                beta = beta_k if b == "kappa" else b
                for g in GAMMAS:
                    r = algorithms.nesterov_svrg(prob, theta0, gamma=g / R2, beta=beta, passes=passes, seed=0, record_every=rec, record_F=False, stop_D=stop, max_time=120)
                    ge = algorithms.grads_to_accuracy(r, EPS, key="D", relative=True)
                    grid[(b, g)] = {"grads": ge, "stopped": r["stopped"], "beta": beta, "final_rel_D": float(r["D"][-1] / r["D"][0])}
                    if np.isfinite(ge) and (best is None or ge < best[0]):
                        best = (ge, b, g, r)
            if best is not None:
                runs["Nesterov+SVRG"] = best[3]
                runs["Nesterov+SVRG"]["grid_best"] = {"beta": best[1], "gammaR2": best[2]}
            res["grid"][label] = grid
        res["problems"].append((label, prob.summary(), beta_k))
        res["runs"][label] = {k: common.strip_run(v) for k, v in runs.items()}
        for k, v in runs.items():
            print(f"   {k:40s} final D/D0 = {v['D'][-1] / v['D'][0]:.2e}  grads = {v['grad_evals'][-1]:.0f}  stopped={v['stopped']}", flush=True)
        for k, v in grid.items():
            print(f"   naive grid beta={k[0]} gammaR2={k[1]:.3g}: grads-to-eps = {v['grads']:.3g} ({v['stopped']})", flush=True)
    common.save_result(TAG, res)
    return res


def plot(res):
    labels = [p[0] for p in res["problems"]]
    fig, axes = ps.new_fig(len(labels), 2, panel=(5.2, 3.8))
    for j, label in enumerate(labels):
        runs = res["runs"][label]
        n = res["problems"][j][1]["n"]
        ax = axes[0][j]
        for name in ["SAGA", "L-SVRG", "L-Katyusha", "L-Katyusha (no neg. momentum)", "Nesterov+SVRG", "AGD"]:
            if name not in runs:
                continue
            r = runs[name]
            lab = name
            if name == "Nesterov+SVRG":
                gb = r.get("grid_best") or {}
                lab = f"Nesterov+SVRG, best of grid (β={gb.get('beta')}, γR²={gb.get('gammaR2', 0):.3g})"
            ax.semilogy(np.asarray(r["grad_evals"]) / n, np.asarray(r["D"]) / r["D"][0], color=ps.ALGO[name], label=lab)
        ax.set_title(label)
        ax.set_ylabel("‖θ−η*‖² / ‖θ₀−η*‖²")
        ax.set_xlabel("component gradient evaluations / n")
        ax.set_ylim(1e-18, 10)
        ps.legend(ax, loc="lower left", fontsize=7)
        # grid heatmap (sequential blue ramp: darker = more gradients)
        ax = axes[1][j]
        grid = res["grid"][label]
        beta_k = res["problems"][j][2]
        Mx = np.full((len(BETAS), len(GAMMAS)), np.nan)
        for bi, b in enumerate(BETAS):
            for gi, g in enumerate(GAMMAS):
                Mx[bi, gi] = grid[(b, g)]["grads"]
        ref = runs["L-Katyusha"]
        ref_g = algorithms.grads_to_accuracy(ref, EPS, key="D", relative=True)
        from matplotlib.colors import LinearSegmentedColormap, LogNorm
        cmap = LinearSegmentedColormap.from_list("blue_ramp", ps.BLUE_RAMP)
        finite = Mx[np.isfinite(Mx)]
        vmin, vmax = (finite.min(), finite.max()) if finite.size else (1, 10)
        im = ax.imshow(np.where(np.isfinite(Mx), Mx, np.nan), cmap=cmap, norm=LogNorm(vmin=vmin, vmax=max(vmax, vmin * 1.01)), aspect="auto")
        ax.set_xticks(range(len(GAMMAS)))
        ax.set_xticklabels([f"γR² = {g:.3g}" for g in GAMMAS], fontsize=7)
        ax.set_yticks(range(len(BETAS)))
        ax.set_yticklabels([f"β = {beta_k:.3f} (√κ rule)" if b == "kappa" else f"β = {b}" for b in BETAS], fontsize=7)
        for bi in range(len(BETAS)):
            for gi in range(len(GAMMAS)):
                v = Mx[bi, gi]
                txt = "diverged / not reached" if not np.isfinite(v) else f"{v / n:.0f} n"
                ax.text(gi, bi, txt, ha="center", va="center", fontsize=7, color=ps.INK["primary"] if (not np.isfinite(v) or v < np.sqrt(vmin * vmax)) else "#ffffff")
        ax.grid(False)
        ax.set_title(f"naive scheme, (β, γ) grid: gradients to 10⁻⁶\n(L-Katyusha with the paper's parameters: {ref_g / n:.0f} n)", fontsize=9)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("gradient evaluations", fontsize=7)
    fig.suptitle("Figure A2 — plugging the SVRG estimator into Nesterov momentum versus the corrected (negative-momentum) coupling of L-Katyusha", fontsize=10)
    ps.savefig(fig, "figA2_naive_acceleration")


def main():
    ps.setup()
    res = run_all()
    plot(res)
    summary = {label: {"runs": {k: {"final_rel_D": float(v["D"][-1] / v["D"][0]), "grads": float(v["grad_evals"][-1]), "stopped": v["stopped"],
                                    "grid_best": v.get("grid_best")} for k, v in res["runs"][label].items()},
                       "grid": {f"beta={k[0]},gammaR2={k[1]:.3g}": v for k, v in res["grid"][label].items()}} for label in res["runs"]}
    common.save_json(TAG + "_summary", summary)
    print("done A2")


if __name__ == "__main__":
    main()
