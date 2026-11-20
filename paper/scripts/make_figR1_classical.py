"""Figure 1 of the paper (paper/paper_saga_to_muon.tex; first drawn for the archived short reports): what the
main project measured on SAGA in the
classical (Euclidean) setting. Built from results that already exist; nothing is rerun.

  figR1_classical  l2-regularized logistic regression, n = 1000, d = 50, SAGA with the constants of Proposition 5.9 of
                   Bach's book (gamma = 1/(4 R^2), Delta = 3 gamma^2), 5 sampling seeds, one data set per alpha:
                   (a) kappa = 1e4, alpha = 1, against the effective passes t/n: the bound of Proposition 5.9 (with
                       the constant 1 + 3n/16 of its proof), the trajectory bound with (mu_eff, L_eff) along seed 0,
                       and the mean over the seeds of ||theta_t - eta*||^2 (colours of fig1 of the main report);
                   (b) speed-up (exact contraction exponent over certified exponent, window means over passes 2-20,
                       mean over the seeds) against kappa, for the covariance spectra j^(-alpha), alpha = 0, 1, 2
                       (blue shades of the bottom-right panel of fig3);
                   (c) alpha = 1: share of the logarithmic gap carried by each inequality I1-I5 of the proof, against
                       kappa; mean over the seeds, error bars min-max over the seeds (colours of fig3). Attribution
                       order: order B of the B3 note, as in fig3 (the inequalities are undone from the bound in the
                       order I5, I4, I2, I3, I1, i.e. the intermediate rates of the proof are applied cumulatively
                       from the exact rate in proof order I1, I3, I2, I4, I5).

Sources (read only):
  results/B3_summary.json   panels (b), (c): attribution.<config>.speedup, .shares, .shares_min_over_seeds,
                            .shares_max_over_seeds; panel (a): attribution.<config>.final_D and problems.<config>
                            (consistency with the pickle)
  results/B3.pkl            panel (a): runs[(1e4, 1.0)], curves computed as in fig1 of experiments/exp_B3_tightness.py
                            (vrlab.analysis.mean_over_seeds, vrlab.bounds.saga_bound, vrlab.analysis.trajectory_bounds);
                            not plotted: the summary numbers of (b) and (c) are recomputed from the stored runs
                            (vrlab.analysis.attribution, as in fig3), which checks the summary against the runs and
                            gives the range over the seeds of the speed-up, printed with panel (b)
  results/digest.md         cross-check (B3 table "Attribution of the log-gap"): speed-up and five shares of the 15
                            configurations, final E D_t of panel (a).
The script stops before plotting if a number disagrees with the digest or if the summary is not reproduced from the
stored runs.
Style: experiments/plotstyle.py. Every plotted number is printed, panel by panel (for the curves of (a): the values at
passes 0, 20, 50 and 100).
Run: python paper/scripts/make_figR1_classical.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import types

sys.dont_write_bytecode = True          # importing project modules must not create or refresh any __pycache__ file

import numpy as np  # noqa: E402

for _stream in (sys.stdout, sys.stderr):   # Windows consoles default to cp1252
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))                 # paper/scripts -> project root
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "paper", "figures")
NAME = "figR1_classical"

KAPPAS = [10.0, 100.0, 1000.0, 10000.0, 100000.0]
ALPHAS = [0.0, 1.0, 2.0]
INEQS = ["I1", "I2", "I3", "I4", "I5"]
KAPPA_A, ALPHA_A = 10000.0, 1.0         # configuration of panel (a)
ALPHA_C = 1.0                           # spectrum of panel (c)
PASSES_PRINT = (0, 20, 50, 100)
V_MIN = 1e-18                           # floor of the trajectory products, as in fig1
DIGEST_TABLE = "### Attribution of the log-gap (passes 2-20) and window-averaged ratios"


def key(kappa, alpha):
    return f"kappa={kappa:g},alpha={alpha:g}"


# ----------------------------------------------------------------------------------------------------------------------
# data (read only)
# ----------------------------------------------------------------------------------------------------------------------
def load_summary():
    with open(os.path.join(RESULTS, "B3_summary.json"), encoding="utf-8") as f:
        return json.load(f)


def load_runs():
    """The stored SAGA runs (results/B3.pkl, written by experiments/exp_B3_tightness.py)."""
    with open(os.path.join(RESULTS, "B3.pkl"), "rb") as f:
        return pickle.load(f)


def _vrlab():
    """Post-processing functions of the main project (the ones that drew fig1 and fig3)."""
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from vrlab import analysis, bounds
    return analysis, bounds


def data_curves(res):
    """Curves of panel (a), computed from the stored runs exactly as fig1 of experiments/exp_B3_tightness.py does."""
    analysis, bounds = _vrlab()
    n = res["config"]["n"]
    runs = res["runs"][(KAPPA_A, ALPHA_A)]
    it, meanD, allD = analysis.mean_over_seeds(runs, "D")
    r0 = runs[0]
    assert np.array_equal(it, np.asarray(r0["iters"])), "recording grids differ"
    assert all(r["D0"] == r0["D0"] and r["stopped"] is None for r in runs), "runs do not share D0 or stopped early"
    prop59 = bounds.saga_bound(it, n, r0["R2"], r0["mu"], r0["D0"])
    traj = analysis.trajectory_bounds(r0, which=("mu_eff+L_eff",), v_min=V_MIN)["mu_eff+L_eff"]
    assert np.all(np.isfinite(traj)), "trajectory bound cut by the V < 1e-18 floor"
    # closed form: (1 - min{1/(3n), 3 mu/(16 R^2)})^t (1 + 3n/16) D0, with the constant obtained in the proof (the
    # statement of the proposition rounds it up to 1 + n/4)
    rate_mem, rate_par = 1.0 / (3 * n), 3 * r0["mu"] / (16 * r0["R2"])
    rho = 1.0 - min(rate_mem, rate_par)
    closed = rho ** it.astype(float) * (1 + 3 * n / 16) * r0["D0"]
    assert np.allclose(prop59, closed, rtol=1e-9, atol=0.0), "saga_bound differs from the closed form of Prop. 5.9"
    return {"n": n, "d": res["config"]["d"], "seeds": [r["seed"] for r in runs], "passes": it / n,
            "prop59": prop59, "traj": traj, "mean": meanD, "min": allD.min(0), "max": allD.max(0),
            "R2": r0["R2"], "mu": r0["mu"], "gamma": r0["gamma"], "Delta": r0["Delta"], "D0": r0["D0"],
            "V0": float(r0["V"][0]), "rho_bound": rho, "active": "memory" if rate_mem < rate_par else "parameter",
            "record_every": int(it[1] - it[0]), "factors": len(r0["instr"]),
            "factor_every": int(r0["instr"][1]["t"] - r0["instr"][0]["t"])}


def data_speedup(summary):
    return {a: [summary["attribution"][key(k, a)]["speedup"] for k in KAPPAS] for a in ALPHAS}


def data_shares(summary):
    out = {}
    for i in INEQS:
        att = [summary["attribution"][key(k, ALPHA_C)] for k in KAPPAS]
        out[i] = {"mean": np.array([v["shares"][i] for v in att]),
                  "min": np.array([v["shares_min_over_seeds"][i] for v in att]),
                  "max": np.array([v["shares_max_over_seeds"][i] for v in att])}
    return out


def seed_statistics(res, summary):
    """Not plotted. Attribution of every stored run over passes 2-20, recomputed with the function that produced
    B3_summary.json (vrlab.analysis.attribution, as in fig3 of experiments/exp_B3_tightness.py). Returns the range
    over the seeds of the speed-up, which the summary does not keep, and the list of the summary numbers (speed-up;
    mean, min and max over the seeds of the five shares) that the recomputation does not reproduce."""
    analysis, _ = _vrlab()
    n = res["config"]["n"]
    order = analysis.STEP_KEYS[1:]                          # I1, I3, I2, I4, I5
    ranges, bad, count = {}, [], 0
    for a in ALPHAS:
        ranges[a] = []
        for k in KAPPAS:
            atts = [analysis.attribution(r["instr"], r["gamma"], r["Delta"], r["R2"], r["mu"], n, t_min=2 * n,
                                         t_max=20 * n, v_min=V_MIN) for r in res["runs"][(k, a)]]
            sp = np.array([x["speedup_total"] for x in atts])
            sh = np.array([x["shares"] for x in atts])
            ref = summary["attribution"][key(k, a)]
            ranges[a].append((float(sp.min()), float(sp.max())))
            pairs = [("speed-up", sp.mean(), ref["speedup"])]
            for j, i in enumerate(order):
                pairs += [(f"share {i} (mean)", sh[:, j].mean(), ref["shares"][i]),
                          (f"share {i} (min)", sh[:, j].min(), ref["shares_min_over_seeds"][i]),
                          (f"share {i} (max)", sh[:, j].max(), ref["shares_max_over_seeds"][i])]
            count += len(pairs)
            bad += [f"{key(k, a)} {name}: B3.pkl {v!r}, B3_summary.json {w!r}" for name, v, w in pairs
                    if not np.isclose(v, w, rtol=1e-9, atol=1e-12)]
    return ranges, bad, count


# ----------------------------------------------------------------------------------------------------------------------
# cross-check against results/digest.md
# ----------------------------------------------------------------------------------------------------------------------
def digest_fmt(x):
    """Number format of experiments/make_digest.py."""
    if x == 0:
        return "0"
    if abs(x) >= 1e4 or abs(x) < 1e-3:
        return f"{x:.3g}"
    return f"{x:.4g}"


def read_digest_table(title):
    """Rows of the table that follows the heading `title` in results/digest.md: {config: {column: cell}}."""
    with open(os.path.join(RESULTS, "digest.md"), encoding="utf-8") as f:
        lines = f.read().splitlines()
    found = [k for k, line in enumerate(lines) if line.strip() == title]
    if len(found) != 1:
        sys.exit(f"results/digest.md: heading {title!r} found {len(found)} times; no figure drawn")
    i = found[0]
    header = [c.strip() for c in lines[i + 1].strip().strip("|").split("|")]
    rows = {}
    for line in lines[i + 3:]:
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows[cells[0]] = dict(zip(header[1:], cells[1:]))
    return rows


def check_digest(summary, curves):
    """Every plotted number of (b) and (c), and the end point of the empirical curve of (a), against the digest.
    Returns the list of disagreements (empty if all agree at the printed precision of the digest)."""
    rows = read_digest_table(DIGEST_TABLE)
    col = {"I1": "share I1", "I2": "I2", "I3": "I3", "I4": "I4", "I5": "I5"}
    bad, count = [], 0
    for a in ALPHAS:
        for k in KAPPAS:
            att, row = summary["attribution"][key(k, a)], rows[key(k, a)]
            pairs = [("speed-up", att["speedup"], row["speed-up exact/bound"])]
            pairs += [(f"share {i}", att["shares"][i], row[col[i]]) for i in INEQS]
            for name, value, cell in pairs:
                count += 1
                if digest_fmt(value) != cell:
                    bad.append(f"{key(k, a)} {name}: summary {value!r} -> {digest_fmt(value)}, digest {cell}")
    k_a = key(KAPPA_A, ALPHA_A)
    final = float(curves["mean"][-1])
    count += 1
    if digest_fmt(final) != rows[k_a]["final E D_t"]:
        bad.append(f"{k_a} final E D_t: B3.pkl {final!r} -> {digest_fmt(final)}, digest {rows[k_a]['final E D_t']}")
    if not np.isclose(final, summary["attribution"][k_a]["final_D"], rtol=1e-12, atol=0.0):
        bad.append(f"{k_a} final E D_t: B3.pkl {final!r}, B3_summary.json {summary['attribution'][k_a]['final_D']!r}")
    prob = summary["problems"][k_a]
    if not (prob["R2"] == curves["R2"] and prob["mu"] == curves["mu"] and prob["n"] == curves["n"]):
        bad.append(f"{k_a}: constants of B3.pkl and of B3_summary.json differ")
    return bad, count


# ----------------------------------------------------------------------------------------------------------------------
# printed numbers
# ----------------------------------------------------------------------------------------------------------------------
def print_numbers(curves, speed, shares, seed_range):
    c = curves
    p = c["passes"]
    print(f"(a) kappa = {KAPPA_A:g}, alpha = {ALPHA_A:g}, n = {c['n']}, d = {c['d']}; SAGA, gamma = 1/(4 R^2) = "
          f"{c['gamma']:.6g}, Delta = 3 gamma^2 = {c['Delta']:.6g}; seeds {c['seeds']}; recorded every "
          f"{c['record_every']} iterations ({len(p)} points per curve, passes {p[0]:g} to {p[-1]:g})")
    print(f"    R^2 = {c['R2']:.6g}, mu = {c['mu']:.6g}, kappa = R^2/mu = {c['R2'] / c['mu']:.6g}; "
          f"D0 = ||theta_0 - eta*||^2 = {c['D0']:.6g} (all seeds); V0 = D0 + Delta M0 = {c['V0']:.6g}")
    print(f"    rho_bound = 1 - min{{1/(3n), 3 mu/(16 R^2)}} = {c['rho_bound']:.8f} ({c['active']} term active), "
          f"rho_bound^(100 n) = {c['rho_bound'] ** (100 * c['n']):.5f}; constant 1 + 3n/16 = {1 + 3 * c['n'] / 16:g}")
    print(f"    trajectory bound: V0 times the product of the factors 1 - min{{1/(3n), 2 gamma c mu_eff}}, "
          f"c = 1 - gamma L - L Delta/(2 gamma), L = min(L_eff, R^2), along seed {c['seeds'][0]}; {c['factors']} "
          f"factors, measured every {c['factor_every']} iterations and held constant in between")
    print("    passes | Prop. 5.9 bound | trajectory bound (mu_eff, L_eff), seed 0 | mean over seeds of "
          "||theta_t - eta*||^2 [min, max over seeds]")
    for q in PASSES_PRINT:
        j = int(np.argmin(np.abs(p - q)))
        assert abs(p[j] - q) < 1e-12
        print(f"    {q:6d} | {c['prop59'][j]:15.4e} | {c['traj'][j]:40.4e} | {c['mean'][j]:.4e} "
              f"[{c['min'][j]:.4e}, {c['max'][j]:.4e}]")
    print(f"    at 100 passes: bound / D0 = {c['prop59'][-1] / c['D0']:.3g}; mean / D0 = {c['mean'][-1] / c['D0']:.3e} "
          f"({-np.log10(c['mean'][-1] / c['D0']):.2f} decades); trajectory bound / mean = "
          f"{c['traj'][-1] / c['mean'][-1]:.3g}; bound / mean = {c['prop59'][-1] / c['mean'][-1]:.3e}")
    print(f"    along the run: trajectory bound / mean between {np.min(c['traj'] / c['mean']):.3g} and "
          f"{np.max(c['traj'] / c['mean']):.3g}; largest max/min over seeds = {np.max(c['max'] / c['min']):.4f} "
          f"(band not drawn); decrease per pass over the 100 passes (log of first/last value, / 100): bound "
          f"{np.log(c['prop59'][0] / c['prop59'][-1]) / 100:.5f}, trajectory bound "
          f"{np.log(c['traj'][0] / c['traj'][-1]) / 100:.4f}, mean {np.log(c['mean'][0] / c['mean'][-1]) / 100:.4f}")
    print("(b) speed-up = (-log rho_exact) / (-log rho_bound): ratio of the window means over passes 2-20 (records "
          "with V_t >= 1e-18), mean over 5 seeds")
    print("    alpha | " + " | ".join(f"kappa={k:<7g}" for k in KAPPAS))
    for a in ALPHAS:
        print(f"    {a:5g} | " + " | ".join(f"{v:13.4g}" for v in speed[a]))
    print("    not plotted: [min, max] of the speed-up over the 5 seeds (recomputed from results/B3.pkl)")
    for a in ALPHAS:
        print(f"    {a:5g} | " + " | ".join(f"{f'[{lo:.4g}, {hi:.4g}]':>13s}" for lo, hi in seed_range[a]))
    print(f"(c) alpha = {ALPHA_C:g}: share of the log-gap per inequality, order B (undo I5, I4, I2, I3, I1 from the "
          "bound = proof order I1, I3, I2, I4, I5 applied cumulatively from the exact rate), passes 2-20, mean over "
          "5 seeds [min, max over seeds]")
    for j, k in enumerate(KAPPAS):
        cells = ", ".join(f"{i} {shares[i]['mean'][j]:.4f} [{shares[i]['min'][j]:.4f}, {shares[i]['max'][j]:.4f}]"
                          for i in INEQS)
        tot = sum(shares[i]["mean"][j] for i in INEQS)
        print(f"    kappa={k:<7g}: {cells}; I2+I4 = {shares['I2']['mean'][j] + shares['I4']['mean'][j]:.4f}; "
              f"sum = {tot:.4f}")


# ----------------------------------------------------------------------------------------------------------------------
# figure (style of the main project: experiments/plotstyle.py)
# ----------------------------------------------------------------------------------------------------------------------
def _plotstyle():
    sys.modules.setdefault("common", types.SimpleNamespace(FIGURES=FIGURES))   # plotstyle only needs common.FIGURES
    sys.path.insert(0, os.path.join(ROOT, "experiments"))
    import plotstyle as ps
    ps.setup()
    return ps


# Layout: three panels on the text width. The legends set the widths of the axes: the legend of (a) is 1.43 in wide,
# the two-column legend of (c) 2.29 in, and (b) takes what is left; compact legends and pads keep each legend inside
# its axes (check_layout reports it if a change of label or font breaks this).
WIDTHS = [1.50, 1.20, 2.32]             # width ratios of the axes (a), (b), (c)
LEGEND = dict(handlelength=1.1, handletextpad=0.5, borderpad=0.2, borderaxespad=0.3)
YPAD, YTICK_PAD = 2.0, 2.5              # pads of the y-labels and of the y tick labels, in points (defaults 4, 3.5)


def panel_curves(ax, c, ps):
    """(a) SAGA against its certificates, kappa = 1e4, alpha = 1 (colours of fig1)."""
    p = c["passes"]
    ax.plot(p, c["prop59"], color=ps.BOUNDS["prop59"], lw=1.7, label="Prop. 5.9 bound")
    ax.plot(p, c["traj"], color=ps.BOUNDS["mu_eff+L_eff"], lw=1.7, label="bound with μ_eff, L_eff")
    ax.plot(p, c["mean"], color=ps.BOUNDS["empirical"], lw=1.8, zorder=10, label="SAGA (mean, 5 seeds)")
    ax.set_yscale("log")
    ax.set_xlim(-2, 102)
    ax.set_ylim(1e-16, 3e6)                                # room for the legend below the curves
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([1e-8, 1e-4, 1e0, 1e4])
    ax.minorticks_off()
    ax.set_xlabel("effective passes t / n")
    ax.set_ylabel("squared distance to η*", labelpad=YPAD)
    ax.set_title("(a) SAGA and its bounds", loc="left")
    ax.text(0.97, 0.80, f"{ps.fmt_kappa(KAPPA_A)}, α = {ALPHA_A:g}", transform=ax.transAxes, ha="right", va="center",
            fontsize=ps.SMALL, color=ps.INK["secondary"])      # the instance, between the bound and the curves
    ps.legend(ax, loc="lower left", **LEGEND)


def panel_speedup(ax, speed, ps):
    """(b) total gap between the exact and the certified exponents (blue shades of fig3)."""
    for ai, a in enumerate(ALPHAS):
        ax.plot(KAPPAS, speed[a], color=ps.BLUE_RAMP[4 + 3 * ai], lw=1.7, marker="o", ms=3.6, label=f"α = {a:g}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(KAPPAS)
    ax.set_yticks([3, 10, 30, 100, 300]); ax.set_yticklabels(["3", "10", "30", "100", "300"])
    ax.minorticks_off()
    ax.set_xlim(10 ** 0.8, 10 ** 5.2)
    ax.set_ylim(2.3, 420)
    ax.set_xlabel("κ = R²/μ")
    ax.set_ylabel("exact / certified rate", labelpad=YPAD)
    ax.set_title("(b) size of the gap", loc="left")
    ps.legend(ax, loc="upper left", **LEGEND)


def panel_shares(ax, shares, ps):
    """(c) attribution of the gap, alpha = 1 (colours of fig3, order B)."""
    short = {"I1": "I1 square norm", "I2": "I2 cocoercivity", "I3": "I3 variance", "I4": "I4 strong convexity",
             "I5": "I5 min of rates"}
    handles = {}
    for i in ["I3", "I5", "I1", "I4", "I2"]:               # drawn from the smallest to the largest at large kappa
        s = shares[i]
        cont = ax.errorbar(KAPPAS, s["mean"], yerr=[s["mean"] - s["min"], s["max"] - s["mean"]], color=ps.INEQ[i],
                           lw=1.7, marker="o", ms=3.6, capsize=1.5, elinewidth=0.9, capthick=0.9)
        handles[i] = cont.lines[0]
    ax.set_xscale("log")
    ax.set_xticks(KAPPAS)
    ax.minorticks_off()
    ax.set_xlim(10 ** 0.8, 10 ** 5.2)
    ax.set_ylim(-0.04, 1.0)                                # room for the legend above the curves
    ax.set_yticks([0.0, 0.2, 0.4, 0.6])
    ax.set_xlabel("κ = R²/μ")
    ax.set_ylabel("share of the log-gap", labelpad=YPAD)
    ax.set_title(f"(c) share of the gap, α = {ALPHA_C:g}", loc="left")
    ps.legend(ax, handles=[handles[i] for i in INEQS], labels=[short[i] for i in INEQS], loc="upper center", ncol=2,
              columnspacing=0.6, **LEGEND)


def build_figure(ps, curves, speed, shares):
    fig = ps.figure(width=ps.FIG_W, height=2.3)
    fig.get_layout_engine().set(wspace=0.0)                # the y-labels already separate the panels
    gs = fig.add_gridspec(1, 3, width_ratios=WIDTHS)
    panel_curves(fig.add_subplot(gs[0, 0]), curves, ps)
    panel_speedup(fig.add_subplot(gs[0, 1]), speed, ps)
    panel_shares(fig.add_subplot(gs[0, 2]), shares, ps)
    for ax in fig.axes:
        ax.tick_params(axis="y", pad=YTICK_PAD)
    return fig


def check_layout(fig):
    """The widths are tight (see WIDTHS): report a legend that sticks out of its axes or crosses a curve."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    notes = []
    for ax in fig.axes:
        name = ax.get_title(loc="left")[:3]
        box, leg = ax.get_window_extent(r), ax.get_legend().get_window_extent(r)
        if not (box.x0 <= leg.x0 and leg.x1 <= box.x1 and box.y0 <= leg.y0 and leg.y1 <= box.y1):
            notes.append(f"{name} legend sticks out of its axes")
        if any(line.get_transform().transform_path(line.get_path()).intersects_bbox(leg, filled=False)
               for line in ax.lines):
            notes.append(f"{name} legend crosses a curve")
    print("layout check: " + ("; ".join(notes) + " (WARNING)" if notes else
                              "the three legends are inside their axes and cross no curve"))


def make_figure(curves, speed, shares):
    ps = _plotstyle()
    fig = build_figure(ps, curves, speed, shares)
    check_layout(fig)
    ps.savefig(fig, NAME)


if __name__ == "__main__":
    summary, res = load_summary(), load_runs()
    curves = data_curves(res)
    speed, shares = data_speedup(summary), data_shares(summary)
    seed_range, bad_runs, count_runs = seed_statistics(res, summary)
    print_numbers(curves, speed, shares, seed_range)
    bad, count = check_digest(summary, curves)
    if bad:
        print(f"DIGEST CHECK FAILED ({len(bad)} of {count} numbers disagree with results/digest.md); no figure drawn:")
    if bad_runs:
        print(f"SUMMARY CHECK FAILED ({len(bad_runs)} of {count_runs} numbers of results/B3_summary.json are not "
              f"reproduced from results/B3.pkl); no figure drawn:")
    if bad or bad_runs:
        for line in bad + bad_runs:
            print("   ", line)
        sys.exit(1)
    print(f"digest check: the {count} numbers compared with results/digest.md (speed-up and five shares of the 15 "
          f"configurations, final E D_t of panel (a)) agree at the printed precision of the digest")
    print(f"summary check: the {count_runs} numbers of results/B3_summary.json compared with their recomputation from "
          f"results/B3.pkl (speed-up; mean, min and max over the seeds of the five shares; 15 configurations) agree")
    make_figure(curves, speed, shares)
