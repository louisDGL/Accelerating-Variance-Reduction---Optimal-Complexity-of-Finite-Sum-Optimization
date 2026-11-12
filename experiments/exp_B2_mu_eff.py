"""Hypothesis B2: trajectory-dependent constants. Along SAGA and GD trajectories we measure
  mu_eff(theta) = (theta - eta_*)^T F'(theta) / ||theta - eta_*||^2        (>= mu, -> Rayleigh quotient of F''(eta_*))
  L_eff(theta)  = (1/n) sum_i ||f'_i(theta) - f'_i(eta_*)||^2 / ((theta - eta_*)^T F'(theta))   (<= R^2)
and the spectral decomposition of the error theta - eta_* in the eigenbasis of F''(eta_*).
The runs are cached in results/B2.pkl: when it exists, the script only redraws the figure and rewrites
results/B2_summary.json, which also stores the numbers quoted in Remark B2-P2-outcome of the B2 note and in Section
exp-B2 of the report (function remark_stats).
Usage: python experiments/exp_B2_mu_eff.py [--quick]
"""
import os
import sys

import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import plotstyle as ps  # noqa: E402
from vrlab import algorithms  # noqa: E402

QUICK = "--quick" in sys.argv
N, D = (300, 20) if QUICK else (1000, 50)
KAPPAS = [1e2, 1e3, 1e4] if not QUICK else [1e2, 1e4]
ALPHAS = [0.0, 1.0, 2.0] if not QUICK else [1.0]
PASSES = 20 if QUICK else 60
GD_ITERS = 100 if QUICK else 400
TAG = "B2_quick" if QUICK else "B2"
FLOOR = 1e-18          # states with ||theta - eta_*||^2 below this are discarded (precision of eta_*)


def curvature_stats(prob, theta, U, k_low=(1, 5), w=None):
    """mu_eff, L_eff, spectral profile pi_k = (q_k^T v)^2/||v||^2 of the error in the eigenbasis U of F''(eta_*):
    energy on the k lowest eigenvectors, effective number of directions d_eff = 1/sum pi_k^2 (B2 note, (P1)-(P2)),
    and the quadratic proxy sum_k lambda_k pi_k of mu_eff."""
    v = theta - prob.theta_star
    D = float(v @ v)
    G = float(v @ prob.grad(theta))
    S = float(np.sum((prob.all_grads(theta) - prob.grads_star) ** 2))
    c = U.T @ v
    pi = c ** 2 / D
    out = {"D": D, "mu_eff": G / D, "L_eff": S / (prob.n * G), "d_eff_dirs": float(1.0 / np.sum(pi ** 2))}
    if w is not None:
        out["mu_eff_quadratic_proxy"] = float(np.sum(w * pi))
    for k in k_low:
        out[f"energy_low{k}"] = float(np.sum(pi[:k]))
    return out


def run_all():
    res = common.load_result(TAG)
    if res is not None:
        print("loaded cached", TAG)
        return res
    res = {"config": dict(n=N, d=D, kappas=KAPPAS, alphas=ALPHAS, passes=PASSES, gd_iters=GD_ITERS), "saga": {}, "gd": {}, "spectra": {}}
    for alpha in ALPHAS:
        for kappa in KAPPAS:
            prob = common.logistic_problem(N, D, kappa, alpha, seed=0)
            H = prob.hessian(prob.theta_star)
            w, U = np.linalg.eigh(H)
            Hsq = prob.hessian_i_sq_mean(prob.theta_star)
            gen = eigh(Hsq, H, eigvals_only=True)          # limits of L_eff by direction
            res["spectra"][(alpha, kappa)] = {"hess_eigs": w, "mu": prob.mu, "R2": prob.R2, "L_eff_limit_min": float(gen[0]),
                                              "L_eff_limit_max": float(gen[-1]), "summary": prob.summary()}
            theta0 = np.zeros(D)
            with common.Timer(f"alpha={alpha:g} kappa={kappa:g}"):
                r = algorithms.saga(prob, theta0, passes=PASSES, seed=0, record_every=max(1, N // 20), instrument_every=max(1, N // 10),
                                    snapshot_theta=True, stop_D=1e-26)
                KEYS = ["mu_eff", "L_eff", "energy_low1", "energy_low5", "D", "d_eff_dirs", "mu_eff_quadratic_proxy"]
                sag = {"t": []}
                sag.update({k: [] for k in KEYS})
                for q in r["instr"]:
                    cs = curvature_stats(prob, q["theta"], U, w=w)
                    sag["t"].append(q["t"])
                    for k in KEYS:
                        sag[k].append(cs[k])
                res["saga"][(alpha, kappa)] = {k: np.array(v) for k, v in sag.items()}
                # gradient descent with step 1/L_F, one full gradient per iteration
                theta = theta0.copy()
                gdd = {"t": []}
                gdd.update({k: [] for k in KEYS})
                for it in range(GD_ITERS + 1):
                    cs = curvature_stats(prob, theta, U, w=w)
                    gdd["t"].append(it)
                    for k in KEYS:
                        gdd[k].append(cs[k])
                    if cs["D"] < 1e-26:
                        break
                    theta -= prob.grad(theta) / prob.L_F
                res["gd"][(alpha, kappa)] = {k: np.array(v) for k, v in gdd.items()}
            s = res["saga"][(alpha, kappa)]
            g = res["gd"][(alpha, kappa)]
            print(f"   SAGA: mu_eff/mu {s['mu_eff'][0] / prob.mu:.2f} -> {s['mu_eff'][-1] / prob.mu:.2f}; L_eff/R2 {s['L_eff'][0] / prob.R2:.3f} -> {s['L_eff'][-1] / prob.R2:.3f}; "
                  f"GD: mu_eff/mu -> {g['mu_eff'][-1] / prob.mu:.2f}; lmin(H*)/mu = {w[0] / prob.mu:.2f}; L_eff limits [{gen[0] / prob.R2:.3f}, {gen[-1] / prob.R2:.3f}] R2", flush=True)
    common.save_result(TAG, res)
    return res


def _log_ylim(ax, values, pad=0.14):
    """Log y-limits covering `values` with a margin of `pad` (fraction of the axis) at both ends, where the labels of
    the horizontal reference lines sit (above the top line, below the bottom one)."""
    v = np.concatenate([np.ravel(np.asarray(x, dtype=float)) for x in values])
    v = v[np.isfinite(v) & (v > 0)]
    lo, hi = np.log10(v.min()), np.log10(v.max())
    m = pad * (hi - lo) / (1 - 2 * pad)
    ax.set_ylim(10 ** (lo - m), 10 ** (hi + m))


def plot(res):
    alpha = 1.0 if 1.0 in ALPHAS else ALPHAS[0]
    fig, axes = ps.new_fig(len(KAPPAS), 3, panel=(2.25, 1.8), sharex="col")
    lab = dict(fontsize=ps.SMALL, color=ps.INK["secondary"], textcoords="offset points")
    for j, kappa in enumerate(KAPPAS):
        sp = res["spectra"][(alpha, kappa)]
        s = res["saga"][(alpha, kappa)]
        g = res["gd"][(alpha, kappa)]
        s = {k: np.asarray(v)[np.asarray(s["D"]) >= FLOOR] for k, v in s.items()}
        g = {k: np.asarray(v)[np.asarray(g["D"]) >= FLOOR] for k, v in g.items()}
        mu, R2 = sp["mu"], sp["R2"]
        ax = axes[0][j]
        ax.semilogy(s["t"] / N, s["mu_eff"] / mu, color=ps.ALGO["SAGA"], label="SAGA (x: passes)")
        ax.semilogy(g["t"], g["mu_eff"] / mu, color=ps.ALGO["GD"], label="GD (x: iterations)")
        lo_line, hi_line = sp["hess_eigs"][0] / mu, sp["hess_eigs"][-1] / mu
        ax.axhline(lo_line, color=ps.INK["muted"], linewidth=1.0)
        ax.axhline(hi_line, color=ps.INK["muted"], linewidth=1.0)
        _log_ylim(ax, [s["mu_eff"] / mu, g["mu_eff"] / mu, [lo_line, hi_line]])
        ax.annotate("λmax(F''(η*))/μ", (0, hi_line), xytext=(2, 2), va="bottom", **lab)
        ax.annotate("λmin(F''(η*))/μ", (0, lo_line), xytext=(2, -2), va="top", **lab)
        ax.set_title(f"{ps.fmt_kappa(kappa)}, α = {alpha:g}")
        if j == 0:
            ax.set_ylabel("μ_eff(θₜ) / μ")
        ax = axes[1][j]
        ax.semilogy(s["t"] / N, s["L_eff"] / R2, color=ps.ALGO["SAGA"], label="SAGA")
        ax.semilogy(g["t"], g["L_eff"] / R2, color=ps.ALGO["GD"], label="GD")
        lo_line, hi_line = sp["L_eff_limit_min"] / R2, sp["L_eff_limit_max"] / R2
        ax.axhline(lo_line, color=ps.INK["muted"], linewidth=1.0)
        ax.axhline(hi_line, color=ps.INK["muted"], linewidth=1.0)
        _log_ylim(ax, [s["L_eff"] / R2, g["L_eff"] / R2, [lo_line, hi_line]])
        # limits of L_eff at eta_* over directions: largest (top line) and smallest (bottom line)
        ax.annotate("largest limit at η*", (0, hi_line), xytext=(2, 2), va="bottom", **lab)
        ax.annotate("smallest limit at η*", (0, lo_line), xytext=(2, -2), va="top", **lab)
        if j == 0:
            ax.set_ylabel("L_eff(θₜ) / R²")
        ax = axes[2][j]
        ax.plot(s["t"] / N, s["energy_low1"], color=ps.ALGO["SAGA"], label="SAGA")
        ax.plot(g["t"], g["energy_low1"], color=ps.ALGO["GD"], label="GD")
        ax.set_ylim(-0.02, 1.02)
        if j == 0:
            ax.set_ylabel("share of ‖θₜ−η*‖²\non the slowest\neigenvector of F''(η*)")
        ax.set_xlabel("effective passes (SAGA) /\niterations (GD)")
    h, l = axes[0][0].get_legend_handles_labels()           # one legend for all panels, below them
    ps.legend_below(fig, h, l, ncol=len(l))
    ps.suptitle(fig, "Figure B2 — effective curvature μ_eff, effective cocoercivity L_eff and error direction\n"
                "along SAGA and GD trajectories")
    ps.savefig(fig, "figB2_effective_curvature")


def _first_time(t, mask):
    """First recorded time at which `mask` holds (None if it never does)."""
    idx = np.flatnonzero(mask)
    return t[idx[0]].item() if len(idx) else None


def _value_at(t, x, t0):
    """x at the recorded time t0 (None if no state was recorded at t0 or if it was discarded below FLOOR)."""
    idx = np.flatnonzero(t == t0)
    return float(x[idx[0]]) if len(idx) else None


def remark_stats(sp, s_all, g_all, config):
    """Numbers quoted in Remark B2-P2-outcome of the B2 note and in Section exp-B2 of the report, besides the start,
    geometric-mean and end values of summarize(), computed from the cached trajectories.
    s_all, g_all: every recorded state of the SAGA and GD runs. All statistics use the states with
    ||theta_t - eta_*||^2 >= FLOOR, except saga_first_pass_below_floor and the keys ending in _all_states; "end" is the
    last such state. Times are in passes t/n for SAGA and in iterations for GD; "start" is theta_0 = 0, common to SAGA
    and GD. energy_low1 = pi_min is the share of ||theta_t - eta_*||^2 on the eigenvector q_min of H_* = F''(eta_*) with
    the smallest eigenvalue, energy_low5 the share on the five lowest eigenvectors, d_eff_dirs = 1/sum_k pi_k^2.
    saga_last5_*: statistics over the kept SAGA states of the last five recorded passes before the end
    (t >= t_end - 5n; geometric mean, min and max of mu_eff, mean/min/max of d_eff, max of pi_min, mean of energy_low5),
    since mu_eff, d_eff and pi_min fluctuate from one state to the next. saga_decay_rate_second_half_per_pass: minus the
    least-squares slope of log ||theta_t - eta_*||^2 against passes over the kept states with t >= t_end / 2, compared
    with the decay 2 n gamma lmin(H_*) = n lmin(H_*) / (2 R^2) per pass of the slowest mode of the mean dynamics at the
    textbook step gamma = 1/(4 R^2) of the cached runs."""
    n = config["n"]
    s = {k: np.asarray(v)[np.asarray(s_all["D"]) >= FLOOR] for k, v in s_all.items()}
    g = {k: np.asarray(v)[np.asarray(g_all["D"]) >= FLOOR] for k, v in g_all.items()}
    w = np.asarray(sp["hess_eigs"])                         # eigenvalues of H_*, ascending
    lmin, mu, R2 = float(w[0]), sp["mu"], sp["R2"]
    Rx2 = sp["summary"]["max_sqnorm"]                       # R_x^2 = max_i ||x_i||^2
    ts_all, tg_all = np.asarray(s_all["t"]), np.asarray(g_all["t"])
    ts, tg = s["t"], g["t"]
    last10 = ts >= ts[-1] - 10 * n                          # SAGA states of the last 10 passes before the end
    last5 = ts >= ts[-1] - 5 * n                            # SAGA states of the last 5 passes before the end
    half = ts >= ts[-1] / 2                                 # second half of the kept SAGA states
    deff = "d_eff_dirs" in s                                # d_eff_dirs and the quadratic proxy are absent from old caches
    proxy = "mu_eff_quadratic_proxy" in s
    me5 = s["mu_eff"][last5]
    decay = -float(np.polyfit(ts[half] / n, np.log(s["D"][half]), 1)[0])
    mode = n * lmin / (2 * R2)                              # 2 n gamma lmin(H_*) with gamma = 1/(4 R^2)
    return {
        # bottom of the spectrum of H_*; criterion 9 n lmin(H_*) / (16 R^2) of the explanation in the remark
        "lambda2_over_lmin": float(w[1] / w[0]),
        "n_eigs_within_10pct_of_lmin": int(np.sum(w <= 1.1 * lmin)),
        "crit_9n_lmin_over_16R2": float(9 * n * lmin / (16 * R2)),
        "crit_9n_mu_over_16R2": float(9 * n * mu / (16 * R2)),
        # initial state theta_0 = 0
        "energy_low1_start": float(s["energy_low1"][0]),
        "d_eff_dirs_start": float(s["d_eff_dirs"][0]) if deff else None,
        "rayleigh_H_start_over_mu": float(s["mu_eff_quadratic_proxy"][0] / mu) if proxy else None,
        "Rx_dist_start": float(np.sqrt(Rx2 * s["D"][0])),
        # SAGA
        "saga_end_pass": float(ts[-1] / n),
        "saga_first_pass_below_floor": _first_time(ts_all / n, np.asarray(s_all["D"]) < FLOOR),
        "saga_D_end": float(s["D"][-1]),
        "saga_Rx_dist_end": float(np.sqrt(Rx2 * s["D"][-1])),
        "saga_first_pass_Rx_dist_le_1": _first_time(ts / n, Rx2 * s["D"] <= 1.0),   # entry into R_x ||theta_t - eta_*|| <= 1
        "saga_energy_low5_end": float(s["energy_low5"][-1]),
        "saga_energy_low1_min": float(np.min(s["energy_low1"])),
        "saga_energy_low1_argmin_pass": float(ts[np.argmin(s["energy_low1"])] / n),
        "saga_first_pass_energy_low1_ge_0.99": _first_time(ts / n, s["energy_low1"] >= 0.99),
        "saga_energy_low1_pass10": _value_at(ts, s["energy_low1"], 10 * n),
        "saga_d_eff_dirs_pass10": _value_at(ts, s["d_eff_dirs"], 10 * n) if deff else None,
        "saga_mu_eff_end_over_lmin": float(s["mu_eff"][-1] / lmin),
        "saga_mu_eff_over_mu_min": float(np.min(s["mu_eff"]) / mu),
        "saga_mu_eff_over_mu_min_all_states": float(np.min(s_all["mu_eff"]) / mu),
        "saga_mu_eff_quadratic_proxy_end_rel_diff": float(s["mu_eff_quadratic_proxy"][-1] / s["mu_eff"][-1] - 1) if proxy else None,
        "saga_mu_eff_over_mu_10_passes_before_end": _value_at(ts, s["mu_eff"] / mu, ts[-1] - 10 * n),
        "saga_mu_eff_last10_passes_nonincreasing_fraction": float(np.mean(np.diff(s["mu_eff"][last10]) <= 0)),
        "R2_over_saga_L_eff_geomean": float(np.exp(-np.mean(np.log(s["L_eff"] / R2)))),
        # SAGA, last five recorded passes before the end (kept states)
        "saga_last5_n_states": int(np.sum(last5)),
        "saga_last5_mu_eff_over_mu_geomean": float(np.exp(np.mean(np.log(me5 / mu)))),
        "saga_last5_mu_eff_over_mu_min": float(np.min(me5) / mu),
        "saga_last5_mu_eff_over_mu_max": float(np.max(me5) / mu),
        "saga_last5_mu_eff_over_lmin_geomean": float(np.exp(np.mean(np.log(me5 / lmin)))),
        "saga_last5_mu_eff_over_lmin_min": float(np.min(me5) / lmin),
        "saga_last5_mu_eff_over_lmin_max": float(np.max(me5) / lmin),
        "saga_last5_d_eff_dirs_mean": float(np.mean(s["d_eff_dirs"][last5])) if deff else None,
        "saga_last5_d_eff_dirs_min": float(np.min(s["d_eff_dirs"][last5])) if deff else None,
        "saga_last5_d_eff_dirs_max": float(np.max(s["d_eff_dirs"][last5])) if deff else None,
        "saga_last5_energy_low1_max": float(np.max(s["energy_low1"][last5])),
        "saga_last5_energy_low5_mean": float(np.mean(s["energy_low5"][last5])),
        # SAGA, decay of ||theta_t - eta_*||^2 per pass over the second half of the kept states vs the slowest mode
        "saga_decay_rate_second_half_per_pass": decay,
        "saga_slowest_mode_rate_per_pass": float(mode),
        "saga_decay_rate_over_slowest_mode": float(decay / mode),
        # gradient descent
        "gd_end_iter": int(tg[-1]),
        "gd_D_end": float(g["D"][-1]),
        "gd_energy_low5_end": float(g["energy_low5"][-1]),
        "gd_first_iter_energy_low1_ge_0.99": _first_time(tg, g["energy_low1"] >= 0.99),
        "gd_mu_eff_end_over_lmin": float(g["mu_eff"][-1] / lmin),
        "gd_mu_eff_over_mu_min": float(np.min(g["mu_eff"]) / mu),
        "gd_mu_eff_over_mu_min_all_states": float(np.min(g_all["mu_eff"]) / mu),
        # steps of GD on which mu_eff increases (kept states): count, largest relative increase, largest ||theta - eta_*||^2
        # after such a step (rounding-level increases near eta_* show up as tiny relative sizes at tiny distances)
        "gd_mu_eff_increasing_steps": int(np.sum(np.diff(g["mu_eff"]) > 0)),
        "gd_mu_eff_increase_max_rel": (float(np.max((np.diff(g["mu_eff"]) / g["mu_eff"][:-1])[np.diff(g["mu_eff"]) > 0]))
                                       if np.any(np.diff(g["mu_eff"]) > 0) else None),
        "gd_D_max_after_increase": (float(np.max(g["D"][1:][np.diff(g["mu_eff"]) > 0])) if np.any(np.diff(g["mu_eff"]) > 0) else None),
        # protocol, read from the cache
        "protocol": {"saga_passes": config["passes"], "saga_record_every_iters": int(np.min(np.diff(ts_all))),
                     "gd_max_iters": config["gd_iters"], "gd_record_every_iters": int(np.min(np.diff(tg_all))),
                     "floor_D": FLOOR},
    }


def summarize(res):
    summary = {}
    for (alpha, kappa), sp in res["spectra"].items():
        s_all, g_all = res["saga"][(alpha, kappa)], res["gd"][(alpha, kappa)]
        # keep only the states above the precision floor of eta_* (below ~1e-18 the curvature ratios are rounding noise)
        s = {k: np.asarray(v)[np.asarray(s_all["D"]) >= FLOOR] for k, v in s_all.items()}
        g = {k: np.asarray(v)[np.asarray(g_all["D"]) >= FLOOR] for k, v in g_all.items()}
        mu, R2 = sp["mu"], sp["R2"]
        key = f"alpha={alpha:g},kappa={kappa:g}"
        summary[key] = {"lmin_H_over_mu": float(sp["hess_eigs"][0] / mu), "lmax_H_over_mu": float(sp["hess_eigs"][-1] / mu),
                        "saga_mu_eff_over_mu_start": float(s["mu_eff"][0] / mu), "saga_mu_eff_over_mu_end": float(s["mu_eff"][-1] / mu),
                        "saga_mu_eff_over_mu_geomean": float(np.exp(np.mean(np.log(s["mu_eff"] / mu)))),
                        "gd_mu_eff_over_mu_end": float(g["mu_eff"][-1] / mu),
                        "saga_L_eff_over_R2_geomean": float(np.exp(np.mean(np.log(s["L_eff"] / R2)))), "gd_L_eff_over_R2_end": float(g["L_eff"][-1] / R2),
                        "L_eff_limit_min_over_R2": float(sp["L_eff_limit_min"] / R2), "L_eff_limit_max_over_R2": float(sp["L_eff_limit_max"] / R2),
                        "saga_energy_low1_end": float(s["energy_low1"][-1]), "gd_energy_low1_end": float(g["energy_low1"][-1]),
                        "saga_d_eff_dirs_end": float(s["d_eff_dirs"][-1]) if "d_eff_dirs" in s else None,
                        "saga_d_eff_dirs_geomean": float(np.exp(np.mean(np.log(s["d_eff_dirs"])))) if "d_eff_dirs" in s else None,
                        "gd_d_eff_dirs_end": float(g["d_eff_dirs"][-1]) if "d_eff_dirs" in g else None,
                        "saga_mu_eff_quadratic_proxy_end_over_mu": float(s["mu_eff_quadratic_proxy"][-1] / mu) if "mu_eff_quadratic_proxy" in s else None,
                        "saga_mu_eff_monotone_decreasing_fraction": float(np.mean(np.diff(s["mu_eff"]) <= 0)),
                        "gd_mu_eff_monotone_decreasing_fraction": float(np.mean(np.diff(g["mu_eff"]) <= 0)),
                        **remark_stats(sp, s_all, g_all, res["config"]),
                        "problem": sp["summary"]}
    return summary


def main():
    ps.setup()
    res = run_all()
    plot(res)
    common.save_json(TAG + "_summary", summarize(res))
    print("done B2")


if __name__ == "__main__":
    main()
