"""Supplement to A4: the two other instances discussed in the A1 note.
 (1) 'shared': the instance of the note itself (shared regularizer, geometric chain zeta = zeta_Q), on which the
     nominal kappa and the effective kappa_eff = R^2/lambda_min(F'') differ once the chain is exhausted
     (kappa >> n m^2): SAGA's cost should scale with kappa_eff (Corollary 'SAGA sandwich'), and the lower-bound window
     for the exponent 1/2 is n << kappa << kappa_max(m, eps) = n (16 m^2 / log^2(1/eps) - 1) + 1.
 (2) 'aligned': all components equal to one chain (L_F = R^2), where the four orders kappa (SAGA), sqrt(n kappa)
     (L-Katyusha), n kappa (GD) and n sqrt(kappa) (AGD) separate.
Usage: python experiments/exp_A4b_instances.py [--quick]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import plotstyle as ps  # noqa: E402
from vrlab import algorithms, problems, solvers  # noqa: E402

QUICK = "--quick" in sys.argv
N, M = (16, 64) if QUICK else (16, 256)
M_ALIGNED = 128 if QUICK else 2048      # single chain: needs m >= sqrt(kappa)/4 log(1/eps), i.e. 1448 for kappa = 1e5
KAPPAS_SHARED = [1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6] if not QUICK else [1e2, 1e3, 1e4, 1e5]
KAPPAS_ALIGNED = [1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5] if not QUICK else [1e2, 1e3, 1e4]
SEEDS = list(range(1 if QUICK else 2))
EPS = 1e-8
MAX_TIME = 120 if QUICK else float(os.environ.get("VRLAB_MAX_TIME", 600))
TAG = "A4b_quick" if QUICK else "A4b"
METHODS = ["SAGA", "L-Katyusha", "AGD", "GD"]


def run_method(name, prob, theta0, seed, D0):
    n, R2, mu = prob.n, prob.R2, prob.mu
    kappa = prob.kappa
    log = np.log(1.0 / EPS)
    stop = EPS * D0
    rec = max(1, n // 4)
    if name == "SAGA":
        cap = int(1.5 * (3 * n + 16 * kappa / 3) * (log + np.log(1 + n / 4))) + 10 * n
        return algorithms.saga(prob, theta0, n_iters=cap, seed=seed, record_every=rec, record_F=False, stop_D=stop, max_time=MAX_TIME)
    if name == "L-Katyusha":
        cap = int(40 * (n + np.sqrt(n * kappa)) * log) + 10 * n
        return algorithms.lkatyusha(prob, theta0, n_iters=cap, seed=seed, record_every=rec, record_F=False, stop_D=stop, max_time=MAX_TIME)
    if name == "AGD":
        kF = prob.L_F / mu
        return algorithms.agd(prob, theta0, n_iters=int(10 * np.sqrt(kF) * log) + 100, record_every=1, record_F=False, stop_D=stop, max_time=MAX_TIME)
    if name == "GD":
        kF = prob.L_F / mu
        return algorithms.gd(prob, theta0, n_iters=int(1.5 * kF * log / 2) + 100, record_every=1, record_F=False, stop_D=stop, max_time=MAX_TIME)
    raise ValueError(name)


def run_all():
    res = common.load_result(TAG)
    if res is None:
        res = {"config": dict(n=N, m=M, eps=EPS, seeds=SEEDS), "grads": {}, "status": {}, "constants": {}}
    for kind, kappas in [("shared", KAPPAS_SHARED), ("aligned", KAPPAS_ALIGNED)]:
        for kappa in kappas:
            if all((kind, kappa, s, name) in res["grads"] for s in SEEDS for name in METHODS):
                continue
            prob = problems.make_hard_instance_shared(N, M, kappa) if kind == "shared" else problems.make_aligned_instance(N, M_ALIGNED, kappa)
            solvers.solve_optimum(prob)
            kmax = N * (16 * M ** 2 / np.log(1 / EPS) ** 2 - 1) + 1 if kind == "shared" else (4 * M_ALIGNED / np.log(1 / EPS)) ** 2
            res["constants"][(kind, kappa)] = {"kappa_eff": prob.kappa, "R2": prob.R2, "mu_eff": prob.mu, "L_F": prob.L_F, "kappa_F": prob.L_F / prob.mu,
                                               "kappa_max_window": kmax, "m": prob.d if kind == "aligned" else M, "meta": prob.meta}
            theta0 = np.zeros(prob.d)
            D0 = float(np.sum((theta0 - prob.theta_star) ** 2))
            for seed in SEEDS:
                for name in METHODS:
                    if (kind, kappa, seed, name) in res["grads"]:
                        continue
                    with common.Timer(f"{kind} kappa={kappa:g} (kappa_eff={prob.kappa:.3g}) seed={seed} {name}"):
                        r = run_method(name, prob, theta0, seed, D0)
                    g = algorithms.grads_to_accuracy(r, EPS, key="D", relative=True)
                    res["grads"][(kind, kappa, seed, name)] = g
                    res["status"][(kind, kappa, seed, name)] = r["stopped"]
                    print(f"   -> grads to eps: {g:.3g} ({r['stopped']})", flush=True)
            common.save_result(TAG, res)
    common.save_result(TAG, res)
    return res


def fit_slope(x, y, xmin):
    xs, ys = [], []
    for a, b in zip(x, y):
        if a >= xmin and np.isfinite(b):
            xs.append(np.log(a))
            ys.append(np.log(b))
    return float(np.polyfit(xs, ys, 1)[0]) if len(xs) >= 2 else np.nan


def plot(res):
    fig, axes = ps.new_fig(2, 2, panel=(5.2, 4.0))
    summary = {}
    # row 1: shared instance vs nominal kappa and vs kappa_eff; row 2: aligned instance, same two x-axes
    panels = [("shared", KAPPAS_SHARED, "nominal"), ("shared", KAPPAS_SHARED, "eff"), ("aligned", KAPPAS_ALIGNED, "nominal"), ("aligned", KAPPAS_ALIGNED, "eff")]
    for ai, (kind, kappas, xkey) in enumerate(panels):
        ax = axes[ai // 2][ai % 2]
        xs = [kappa if xkey == "nominal" else res["constants"][(kind, kappa)]["kappa_eff"] for kappa in kappas]
        for name in METHODS:
            vals = []
            for kappa in kappas:
                g = [res["grads"].get((kind, kappa, s, name), np.nan) for s in SEEDS]
                g = [v for v in g if np.isfinite(v)]
                vals.append(np.exp(np.mean(np.log(g))) if g else np.nan)
            slope = fit_slope(xs, vals, xmin=10 * N)
            summary[f"{kind}|{xkey}|{name}"] = {"x": xs, "grads": vals, "slope_x_ge_10n": slope}
            ax.loglog(xs, vals, marker="o", markersize=5, color=ps.ALGO[name], label=f"{name} (slope {slope:.2f})")
        kmax = res["constants"][(kind, kappas[0])]["kappa_max_window"]
        mm = res["constants"][(kind, kappas[0])]["m"]
        if xkey == "nominal":
            ax.axvline(kmax, color=ps.INK["axis"], linewidth=0.8)
            ax.annotate("κ_max(m, ε)", (kmax, ax.get_ylim()[0]), xytext=(2, 4), textcoords="offset points", fontsize=7.5, color=ps.INK["secondary"])
        ax.axvline(N, color=ps.INK["axis"], linewidth=0.8)
        ax.set_xlabel("nominal κ" if xkey == "nominal" else "κ_eff = R² / λmin(F'')")
        ax.set_ylabel("component gradients to reach 10⁻⁸ relative error")
        ax.set_title({"shared": f"A1-note instance (shared regularizer, ζ = ζ_Q), m = {mm}", "aligned": f"aligned instance (all components equal, L_F = R²), m = {mm}"}[kind]
                     + ("" if xkey == "nominal" else " — against κ_eff"), fontsize=9)
        ps.legend(ax, loc="upper left", fontsize=7)
    fig.suptitle(f"Figure A4b — the other instances of the A1 note (n = {N}, ε = 10⁻⁸): saturation of the chain and the four orders κ, √(nκ), nκ, n√κ", fontsize=10)
    ps.savefig(fig, "figA4b_instances")
    return summary


def main():
    ps.setup()
    res = run_all()
    summary = plot(res)
    summary["constants"] = {f"{k[0]}|kappa={k[1]:g}": v for k, v in res["constants"].items()}
    common.save_json(TAG + "_summary", summary)
    print("done A4b")


if __name__ == "__main__":
    main()
