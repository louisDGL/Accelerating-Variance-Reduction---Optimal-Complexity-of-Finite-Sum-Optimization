"""Validation: reproduce the setting of Figure 5.4 of Bach's book (Section 5.4.4, "Experiments"):
l2-regularized logistic regression, d = 40, Gaussian inputs, y = sign(linear + Gaussian noise), mu = R^2/n with
R^2 = max_i ||x_i||^2, n in {1000, 10000}; GD (step 1/R^2) and SAGA (step 1/(4R^2)) with the steps of the book's
analysis, and SGD with the decreasing step 1/(R^2 + mu t), our choice (the book's text gives 1/(mu t) in Prop. 5.8,
and the code of its Figure 5.4 uses 1/(R^2 sqrt(t+1) + mu (t+1))); training objective log10(F(theta_t) - F_*) versus
effective passes. The book reports SAGA reaching machine precision in about 25 (n = 1000) and 10 (n = 10000) passes.
Usage: python experiments/exp_validation_book.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import plotstyle as ps  # noqa: E402
from vrlab import algorithms, problems, solvers  # noqa: E402

TAG = "validation_book"


def main():
    ps.setup()
    res = common.load_result(TAG)
    if res is None:
        res = {}
        for n in [1000, 10000]:
            prob = problems.make_logistic_book(n, d=40, seed=0)
            solvers.solve_optimum(prob)
            R2f = prob.meta["R2_feat"]                        # the book's R^2 (bounded features)
            theta0 = np.zeros(prob.d)
            rec = max(1, n // 10)
            with common.Timer(f"book n={n}  R2_feat={R2f:.2f} mu={prob.mu:.3g} (per-sample L_max = {prob.R2:.3f})"):
                runs = {
                    "SGD": algorithms.sgd(prob, theta0, passes=50, gamma0=1.0 / R2f, decay=True, seed=0, record_every=rec),
                    "SAGA": algorithms.saga(prob, theta0, gamma=1.0 / (4 * R2f), passes=50, seed=0, record_every=rec, stop_D=1e-30),
                    "GD": algorithms.gd(prob, theta0, n_iters=50, step=1.0 / R2f, record_every=1),
                }
            res[n] = {"summary": prob.summary(), "runs": {k: common.strip_run(v, keep_instr=False) for k, v in runs.items()}}
            for k, v in runs.items():
                print(f"   {k:5s}: F-F* after 50 passes = {v['Fgap'][-1]:.2e}", flush=True)
        common.save_result(TAG, res)
    fig, axes = ps.new_fig(2, 1, panel=(3.35, 2.6))         # the two panels of the published Figure 0, at the text width
    for ax, n in zip(axes[0], [1000, 10000]):
        for k in ["SGD", "SAGA", "GD"]:
            r = res[n]["runs"][k]
            ax.plot(np.asarray(r["grad_evals"]) / n, np.log10(np.maximum(np.asarray(r["Fgap"]), 1e-16)), color=ps.ALGO[k], label=k)
        ax.set_title(f"Training objective — n = {n:,}")
        ax.set_xlabel("effective passes")
        ax.set_ylabel("log₁₀[F(θₜ) − F(η*)]")
        ax.set_ylim(-16.5, 0.5)
        ps.legend(ax, loc="center right")                    # clear of the SGD and GD plateaus near the top
    # the book's constant R_x^2 = max_i ||x_i||^2 (the report's R^2 = max_i L_i is about R_x^2 / 4 for logistic losses)
    ps.suptitle(fig, "Validation — the setting of Figure 5.4 of Bach (2024): d = 40, Gaussian inputs,\n"
                r"μ = $R_x^2/n$ with $R_x^2 = \mathrm{max}_i\,\|x_i\|^2$; steps 1/$R_x^2$ (GD) and 1/(4$R_x^2$) (SAGA)" "\n"
                r"of the book's analysis, our SGD step 1/($R_x^2$ + μt)")
    ps.savefig(fig, "fig0_validation_book")
    common.save_json(TAG + "_summary", {str(n): {"problem": v["summary"], "final_Fgap": {k: float(r["Fgap"][-1]) for k, r in v["runs"].items()},
                                                "passes_to_1e-8": {k: float(algorithms.grads_to_accuracy(r, 1e-8, key="Fgap", relative=False) / n)
                                                                   for k, r in v["runs"].items()}} for n, v in res.items()})
    print("done validation")


if __name__ == "__main__":
    main()
