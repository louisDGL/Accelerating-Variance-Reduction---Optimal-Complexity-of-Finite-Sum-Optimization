"""Hypothesis A3: the Lyapunov proof of loopless Katyusha (Kovalev, Horvath, Richtarik, ALT 2020), instrumented
exactly as Proposition 5.9 is: at instrumented iterations we compute the exact conditional expectation
E[Psi_{k+1} | state] (enumerating the n indices), compare the exact one-step contraction with the factor
max{c_Z, c_Y, c_W} = max{1/(1+eta sigma), 1 - theta1(1 - theta2), 1 - p theta1/(1 + theta1)} certified by Lemma 10
(Lemma 4.5 and Theorem 4.6 in the arXiv version 1901.08689 are Lemma 10 and Theorem 11 of the ALT version; the
closed-form rate rho of Theorem 11 satisfies rho <= 1 - max{c_Z, c_Y, c_W}), and measure the tightness of
the estimator-variance bound E||g - F'(x)||^2 <= 2L D_F(w, x) (the inequality that the negative momentum makes absorbable).
Usage: python experiments/exp_A3_lkatyusha_lyapunov.py [--quick]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import plotstyle as ps  # noqa: E402
from vrlab import algorithms  # noqa: E402

QUICK = "--quick" in sys.argv
TAG = "A3_quick" if QUICK else "A3"
PASSES = 20 if QUICK else 60
SEEDS = list(range(1 if QUICK else 3))
PSI_FLOOR = 1e-10      # below this, Psi is dominated by the rounding error of F(y) - F_* (amplified by the W coefficient)


def _valid_prefix(q):
    """Instrumentation records before the Lyapunov value reaches the rounding floor."""
    out = []
    for x in q:
        if not (x["Psi"] >= PSI_FLOOR):
            break
        out.append(x)
    return out


def _sign_and_contraction_stats(runs, q):
    """Sign-aware statistics of the cancelled inner product and componentwise form of Lemma 10, pooled over the seeds on
    the valid prefixes. The A3 note's q_+(t) equals q(t) when <F'(x_t), x_t - w_t> > 0 (c(t) > 0: the removed term has the
    sign that a bound without the anchor would have to absorb) and 0 otherwise. Lemma 10 componentwise:
    E[Psi_{t+1} | state] <= c_Z Z_t + c_Y Y_t + c_W W_t; rho = min{sigma/(6 theta1), theta1/(2n)} is the closed-form rate."""
    if not all(k in x for x in q for k in ("q_cancel", "inner_xw", "c_share")):   # caches older than these fields
        return {}
    r0 = runs[0]
    sigma = r0["mu"] / r0["L"]
    th1, th2, eta, p, n = r0["theta1"], r0["theta2"], r0["eta"], r0["p"], r0["n"]
    c_z, c_y, c_w = 1.0 / (1.0 + eta * sigma), 1.0 - th1 * (1.0 - th2), 1.0 - p * th1 / (1.0 + th1)
    rho = min(sigma / (6.0 * th1), th1 / (2.0 * n))
    qc = np.array([x["q_cancel"] for x in q], dtype=float)
    ip = np.array([x["inner_xw"] for x in q], dtype=float)
    c = np.array([x["c_share"] for x in q], dtype=float)
    pos, big = ip > 0, qc >= 1
    cdef = big & np.isfinite(c)
    return {"inner_xw_frac_positive": float(np.mean(pos)),
            "q_plus_frac_ge_1": float(np.mean(pos & big)),
            "q_cancel_frac_ge_1_given_inner_positive": float(np.mean(big[pos])) if pos.any() else None,
            "q_cancel_median_given_inner_positive": float(np.median(qc[pos])) if pos.any() else None,
            "c_share_frac_negative_given_q_ge_1": float(np.mean(c[cdef] < 0)) if cdef.any() else None,
            "contraction_componentwise_max": float(max((x["EPsi_next"] - c_z * x["Z"] - c_y * x["Y"] - c_w * x["W"]) / x["Psi"] for x in q)),
            "rho_theorem11": float(rho),
            "speedup_exact_over_rho": float(np.mean([np.log(x["rho_exact"]) for x in q]) / np.log(1.0 - rho))}


def problems_list():
    if QUICK:
        return [("hard instance (n=16, m=64, κ=10⁴)", common.hard_problem(16, 64, 1e4)),
                ("logistic (n=300, d=20, κ=10³, α=1)", common.logistic_problem(300, 20, 1e3, 1.0))]
    return [("hard instance (n=16, m=64, κ=10⁴)", common.hard_problem(16, 64, 1e4)),
            ("logistic (n=1000, d=50, κ=10³, α=1)", common.logistic_problem(1000, 50, 1e3, 1.0)),
            ("logistic (n=1000, d=50, κ=10⁴, α=1)", common.logistic_problem(1000, 50, 1e4, 1.0))]


def run_all():
    res = common.load_result(TAG)
    if res is not None:
        print("loaded cached", TAG)
        return res
    res = {"problems": [], "runs": {}}
    for label, prob in problems_list():
        n = prob.n
        theta0 = np.zeros(prob.d)
        runs = []
        with common.Timer(label):
            for s in SEEDS:
                r = algorithms.lkatyusha(prob, theta0, passes=PASSES, seed=s, record_every=max(1, n // 10),
                                         instrument_every=max(1, n // 10), stop_D=1e-26)
                runs.append(common.strip_run(r))
        res["problems"].append((label, prob.summary()))
        res["runs"][label] = runs
        q = _valid_prefix(runs[0]["instr"])
        print(f"   rho_lemma = {q[0]['rho_lemma']:.6f}; exact rho (geomean over run) = {np.exp(np.mean([np.log(x['rho_exact']) for x in q])):.6f}; "
              f"r_var geomean = {np.exp(np.nanmean([np.log(x['r_var']) for x in q if np.isfinite(x['r_var']) and x['r_var'] > 0])):.3f}", flush=True)
    common.save_result(TAG, res)
    return res


def _panel_title(label):
    """'logistic (n=1000, d=50, κ=10³, α=1)' -> 'logistic, κ=10³' over '(n=1000, d=50, α=1)': fits a third of the
    text width."""
    name, _, rest = label.partition(" (")
    params = [p.strip() for p in rest.rstrip(")").split(",")]
    kap = [p for p in params if p.startswith("κ")]
    other = [p for p in params if p and not p.startswith("κ")]
    return f"{name}, {', '.join(kap)}\n({', '.join(other)})" if kap and other else label


def plot(res):
    labels = [p[0] for p in res["problems"]]
    nc = len(labels)
    fig = ps.figure(ps.FIG_W, 6.6)                          # fits one page with its caption
    # three rows of panels (one column per problem), each followed by a thin row holding the legend of that row
    gs = fig.add_gridspec(6, nc, height_ratios=[1.0, 0.02, 1.0, 0.02, 1.0, 0.02])
    axes = [[None] * nc for _ in range(3)]
    for j in range(nc):
        axes[0][j] = fig.add_subplot(gs[0, j])
        axes[1][j] = fig.add_subplot(gs[2, j], sharex=axes[0][j])
        axes[2][j] = fig.add_subplot(gs[4, j], sharex=axes[0][j])
        for r in (0, 1):
            axes[r][j].tick_params(labelbottom=False)
    for j, label in enumerate(labels):
        runs = res["runs"][label]
        n = res["problems"][j][1]["n"]
        r0 = runs[0]
        q = _valid_prefix(r0["instr"])
        t = np.array([x["t"] for x in q])
        Psi = np.array([x["Psi"] for x in q])
        rho_lemma = q[0]["rho_lemma"]
        # row 1: Lyapunov function
        ax = axes[0][j]
        T = min(len(_valid_prefix(r["instr"])) for r in runs)
        Pall = np.array([[x["Psi"] for x in r["instr"][:T]] for r in runs])
        ax.semilogy(t[:T] / n, Pall.mean(0), color=ps.BOUNDS["empirical"], label="E[Ψₖ] (mean over seeds)", linewidth=2.4, zorder=10)
        ax.semilogy(t / n, Psi[0] * rho_lemma ** t, color=ps.BOUNDS["prop59"], label=r"Lemma 10: $\max\{c_Z, c_Y, c_W\}^k\,\Psi_0$")
        cum = np.concatenate([[0.0], np.cumsum(np.log(np.clip([x["rho_exact"] for x in q[:-1]], 1e-300, 1)) * np.diff(t))])
        ax.semilogy(t / n, Psi[0] * np.exp(cum), color=ps.BOUNDS["exact"], label="product of exact contractions", linewidth=1.6)
        ax.semilogy(t / n, Psi, color=ps.BOUNDS["realized_V"], linewidth=1.2, label="realized Ψₖ (seed 0)")
        ax.set_title(_panel_title(label))
        if j == 0:
            ax.set_ylabel("Lyapunov\nΨₖ = Zₖ + Yₖ + Wₖ")
        # row 2: contraction exponents
        ax = axes[1][j]
        ex = -np.log(np.clip([x["rho_exact"] for x in q], 1e-300, 1)) * n
        ax.semilogy(t / n, ex, color=ps.INK["primary"], label="exact one-step contraction")
        ax.axhline(-np.log(rho_lemma) * n, color=ps.INEQ["I5"], label="Lemma 10 (paper parameters)")
        if j == 0:
            ax.set_ylabel("contraction exponent\nper pass (−n·log ρ)")
        # row 3: variance-bound tightness and Lyapunov composition
        ax = axes[2][j]
        rv = np.array([x["r_var"] for x in q], dtype=float)
        rv[~np.isfinite(rv)] = np.nan
        ax.semilogy(t / n, rv, color=ps.RATIO["r_var"], label="r_var: E‖g−F′(x)‖² / (2L·D_F(w,x))")
        for key, col, lab in [("Z", ps.ORANGE, "Z/Ψ"), ("Y", ps.YELLOW, "Y/Ψ"), ("W", ps.MAGENTA, "W/Ψ")]:
            ax.semilogy(t / n, np.clip([x[key] / x["Psi"] for x in q], 1e-6, None), color=col, label=lab, linewidth=1.4)
        ax.set_ylim(1e-4, 2)
        ax.set_xlabel("effective passes (k / n)")
        if j == 0:
            ax.set_ylabel("tightness ratio /\nshare of Ψ")
    for r, ncol in zip(range(3), (2, 2, 4)):                 # one legend per row, below its panels
        h, l = axes[r][0].get_legend_handles_labels()
        ps.legend_cell(fig, gs[2 * r + 1, :], h, l, ncol=ncol)
    ps.suptitle(fig, "Figure A3 — L-Katyusha: exact one-step contraction of the Lyapunov function versus Lemma 10,\n"
                "and tightness of the variance bound")
    ps.savefig(fig, "figA3_lkatyusha_lyapunov")


def main():
    ps.setup()
    res = run_all()
    plot(res)
    summary = {}
    for label, runs in res["runs"].items():
        q = [x for r in runs for x in _valid_prefix(r["instr"])]
        summary[label] = {"rho_lemma": q[0]["rho_lemma"], "theta1": runs[0]["theta1"], "eta": runs[0]["eta"], "p": runs[0]["p"],
                          "rho_exact_geomean": float(np.exp(np.mean([np.log(x["rho_exact"]) for x in q]))),
                          "speedup_exact_over_lemma": float(np.mean([np.log(x["rho_exact"]) for x in q]) / np.log(q[0]["rho_lemma"])),
                          "r_var_geomean": float(np.exp(np.mean([np.log(x["r_var"]) for x in q if np.isfinite(x["r_var"]) and x["r_var"] > 0]))),
                          "r_var_max": float(np.nanmax([x["r_var"] for x in q])),
                          "r_cen_geomean": float(np.exp(np.nanmean([np.log(x["r_cen"]) for x in q if np.isfinite(x.get("r_cen", np.nan)) and x["r_cen"] > 0]))) if any(np.isfinite(x.get("r_cen", np.nan)) for x in q) else None,
                          "r_coco_lk_geomean": float(np.exp(np.nanmean([np.log(x["r_coco_lk"]) for x in q if np.isfinite(x.get("r_coco_lk", np.nan)) and x["r_coco_lk"] > 0]))) if any(np.isfinite(x.get("r_coco_lk", np.nan)) for x in q) else None,
                          "c_share_median": float(np.nanmedian([x["c_share"] for x in q if np.isfinite(x.get("c_share", np.nan))])) if any(np.isfinite(x.get("c_share", np.nan)) for x in q) else None,
                          "c_share_frac_negative": float(np.mean([x["c_share"] < 0 for x in q if np.isfinite(x.get("c_share", np.nan))])) if any(np.isfinite(x.get("c_share", np.nan)) for x in q) else None,
                          "c_share_frac_above_1": float(np.mean([x["c_share"] > 1 for x in q if np.isfinite(x.get("c_share", np.nan))])) if any(np.isfinite(x.get("c_share", np.nan)) for x in q) else None,
                          "q_cancel_median": float(np.nanmedian([x["q_cancel"] for x in q if "q_cancel" in x])) if all("q_cancel" in x for x in q) else None,
                          "q_cancel_frac_ge_1": float(np.mean([x["q_cancel"] >= 1 for x in q if "q_cancel" in x])) if all("q_cancel" in x for x in q) else None,
                          "max_violation_rho_exact_minus_rho_lemma": float(max(x["rho_exact"] - x["rho_lemma"] for x in q)),
                          "final_rel_D": float(np.mean([r["D"][-1] / r["D"][0] for r in runs]))}
        summary[label].update(_sign_and_contraction_stats(runs, q))
    common.save_json(TAG + "_summary", summary)
    print("done A3")


if __name__ == "__main__":
    main()
