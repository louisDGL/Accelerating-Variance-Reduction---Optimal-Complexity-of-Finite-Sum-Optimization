"""Figure T06c of the lecture (paper/talk/figures/figT06c_curvature_far): where the global strong-convexity constant
of ridge logistic regression lives. Along every generic ray the Hessian tends to lambda times the identity, so the
global constant is exactly mu = lambda and is approached only at infinity, while the iterates of SAGA stay in a ball
where the curvature is tens of times larger. Explanatory figure, not in the paper; it illustrates (C3) ("the global
strong-convexity constant of ridge logistic regression is exactly lambda; near the optimum mu_eff is at least about
the smallest eigenvalue of the Hessian at eta_*"), the unbounded factor mu_eff / mu of (C2) and the hypothesis of
Theorem 1 (W_t = t W_1 for almost every W_1) of paper/paper_saga_to_muon.tex. It adds no claim.

The toy ("Toy A" of the storyboard; the same instance as figT09c_nfl_bands), an exact instance of the setting of
Section 2.1 of the paper with d = 2:
    F(theta) = (1/n) sum_i log(1 + exp(-y_i x_i.theta)) + (lam/2) |theta|^2,   n = 100,
    generated with numpy.random.default_rng(0), in this order: X = (standard_normal((100, 2)) * sqrt([1, 0.05])) R^T
    with R the rotation by 30 degrees; X /= max_i |x_i|; w = standard_normal(2) rescaled so that mean((X w)^2) = 4;
    y_i = +1 where random(100) < sigmoid(x_i.w), else -1;   lam = (1/4) / (kappa - 1) with kappa = 1e4,
so that R^2 = max_i |x_i|^2 / 4 + lam = kappa lam (the per-sample smoothness constant of Proposition 5.9) and
mu = lam. Its Hessian is F''(theta) = (1/n) sum_i s_i x_i x_i^T + lam I, s_i = sigma'(y_i x_i.theta) in (0, 1/4].

Panels (they share the vertical axis: a curvature divided by lambda):
  (a) the two eigenvalues of F''(t u) / lam against the distance t = |theta| from the origin, on the ray through the
      minimizer (u = eta_* / |eta_*|, thick, the band between them filled) and on N_RAYS other rays (thin; unit
      directions every 15 degrees from RAY_OFFSET_DEG degrees, the whole-degree offset that keeps every x_i farthest
      from orthogonal to the rays: a ray orthogonal to some x_i is not generic, the curvature of that sample does
      not vanish along it). Two levels: lam_max(F''(0)) / lam = L_F / lam (every s_i is largest at theta = 0) and
      1 (the global constant mu = lam). The shaded strip is the ball |theta| <= max_t |theta_t| that contains every
      iterate of the five runs of (b); the dot is the minimizer (smallest eigenvalue of F''(eta_*) / lam).
  (b) mu_eff(theta_t) / mu along SAGA, mu_eff(theta) = <theta - eta_*, F'(theta)> / |theta - eta_*|^2 (Section 2.2
      of the paper), for five sampling seeds (one highlighted), against passes (n iterations); the levels 1 (the
      constant the proof uses) and lam_min(F''(eta_*)) / lam, and the ratio of the two (double arrow). The runs are those of the project code:
      vrlab.algorithms.saga(prob, zeros(2), passes=PASSES, seed=s, instrument_every=1, snapshot_theta=True), i.e.
      SAGA with the constants of Proposition 5.9 (gamma = 1/(4 R^2), Lyapunov weight 3 gamma^2), table initialized
      at theta_0 = 0, uniform sampling; mu_eff is evaluated before every iteration.
      The label "exact / certified exponent" is vrlab.analysis.attribution(...)["speedup_total"] on passes 2-20, the
      quantity of (C1) of the paper (mean over the window of -log of the exact conditional contraction of the
      Lyapunov function, computed by enumerating the n indices, over -log of the certified rate), per seed: it is a
      number of THIS TOY (n = 100, d = 2), not a result of the paper (3.2-220 there, n = 1000, d = 50).

Checks (the script stops if one fails): R^2 = kappa lam; the Hessian of vrlab against the formula above written
independently; mu_eff of the instrumentation against its definition recomputed from the stored iterates; every ray
drawn reaches lam I (to 1e-9) at t = 1e7; RAY_OFFSET_DEG is the offset selected by the stated rule; the deterministic
values of the storyboard (eta_*, eigenvalues at 0 and at eta_*, values on the ray) to their rounding.

Status. Proved: mu = lam exactly, and the limit along almost every ray ((C3), proof of Theorem 1). Exact evaluations
on a stated instance: panel (a). Measured exactly along runs of an illustrative toy: panel (b); mu_eff explains the
speed of a run, it certifies nothing (local constants certify no global rate).

Progressive build (states s1-s3): s1 the two levels and the ray through the minimizer; s2 adds the other rays;
s3 adds the ball of the iterates and panel (b).

The script prints every number it plots. Nothing is read from the results of the project.
Run:    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/scripts/make_figT06c_curvature_far.py   (about 12 s)
Output: paper/talk/figures/figT06c_curvature_far.pdf and .png, and the states _s1 ... _s3
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import explain_style as es  # noqa: E402

if es.ROOT not in sys.path:
    sys.path.insert(0, es.ROOT)

NAME = "figT06c_curvature_far"
N, KAPPA, DATA_SEED, ROT_DEG = 100, 1e4, 0, 30.0
SEEDS = (0, 1, 2, 3, 4)                  # sampling seeds of the SAGA runs
HIGHLIGHT = 0                            # the run drawn thick
PASSES = 20                              # passes of the runs (the window of the paper ends there)
WINDOW = (2, 20)                         # passes of the "exact / certified exponent" (the window of the paper)
T_GRID = np.logspace(-1.0, 4.0, 401)     # distances t along the rays
N_RAYS, RAY_STEP_DEG, RAY_OFFSET_DEG = 12, 15, 10
T_FAR = 1e7                              # where every ray drawn must have reached lam I
# deterministic values of the storyboard (prototype of 2026-11-25), checked to their rounding
STORYBOARD = {"eta_star": (-4.839, -1.326), "eig0": (79.2, 1669.7), "eig_star": (48.07, 414.9), "norm_star": 5.02,
              "ray_min": {5.0: 48.0, 30.0: 5.9, 100.0: 1.1}}


# ----------------------------------------------------------------------------------------------------------------------
# the toy
# ----------------------------------------------------------------------------------------------------------------------
def unit(deg):
    a = np.deg2rad(deg)
    return np.array([np.cos(a), np.sin(a)])


def instance():
    """Toy A: features, labels and ridge parameter (see the docstring; the order of the draws matters)."""
    rng = np.random.default_rng(DATA_SEED)
    a = np.deg2rad(ROT_DEG)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    X = (rng.standard_normal((N, 2)) * np.sqrt([1.0, 0.05])) @ R.T
    X /= np.linalg.norm(X, axis=1).max()
    w = rng.standard_normal(2)
    w *= 2.0 / np.sqrt(np.mean((X @ w) ** 2))
    y = np.where(rng.random(N) < 1.0 / (1.0 + np.exp(-(X @ w))), 1.0, -1.0)
    lam = 0.25 / (KAPPA - 1.0)
    return X, y, lam


def hessian(X, lam, theta):
    """F''(theta), written independently of vrlab: sigma'(z) = e^{-|z|} / (1 + e^{-|z|})^2 (no overflow)."""
    e = np.exp(-np.abs(X @ theta))                                  # sigma' is even: the labels do not matter
    s = e / (1.0 + e) ** 2
    return (X * s[:, None]).T @ X / X.shape[0] + lam * np.eye(2)


def ray_eigs(X, lam, u, ts):
    """Eigenvalues (ascending) of F''(t u) / lam for each t."""
    return np.array([np.linalg.eigvalsh(hessian(X, lam, t * u)) for t in ts]) / lam


def close(a, b, digits):
    """a equals b once rounded to the given number of decimals (b is a rounded storyboard value)."""
    return abs(a - b) <= 0.5 * 10.0 ** (-digits) + 1e-12


def data():
    from vrlab import algorithms, analysis, problems, solvers
    X, y, lam = instance()
    prob = problems.LogisticProblem(X, y, lam)
    if abs(prob.R2 / (KAPPA * lam) - 1.0) > 1e-12 or prob.mu != lam:
        raise SystemExit(f"R^2 = {prob.R2} is not kappa lam = {KAPPA * lam}")
    eta, info = solvers.newton_solve(prob, tol=1e-16)
    gnorm = float(np.linalg.norm(prob.grad(eta)))
    if gnorm > 1e-15:
        raise SystemExit(f"the minimizer is not converged: |F'| = {gnorm:.1e}")
    prob.set_optimum(eta)
    for theta in (np.zeros(2), eta, 3.0 * unit(70.0), 200.0 * unit(160.0)):      # two ways of writing the Hessian
        if np.abs(prob.hessian(theta) - hessian(X, lam, theta)).max() > 1e-13 * prob.R2:
            raise SystemExit("the Hessian of vrlab and the formula of the docstring disagree")
    eig0 = np.linalg.eigvalsh(hessian(X, lam, np.zeros(2))) / lam
    eig_star = np.linalg.eigvalsh(hessian(X, lam, eta)) / lam
    norm_star = float(np.linalg.norm(eta))
    u_star = eta / norm_star

    # rays: the stated rule for the offset, the curves, the limit
    def margin(off):
        return min(float(np.abs(X @ unit(off + RAY_STEP_DEG * k)).min()) for k in range(N_RAYS))
    margins = {off: margin(off) for off in range(RAY_STEP_DEG)}
    if max(margins, key=margins.get) != RAY_OFFSET_DEG:
        raise SystemExit("RAY_OFFSET_DEG is not the offset that keeps the x_i farthest from orthogonal to the rays")
    ray_deg = [RAY_OFFSET_DEG + RAY_STEP_DEG * k for k in range(N_RAYS)]
    rays = [ray_eigs(X, lam, unit(deg), T_GRID) for deg in ray_deg]
    ray_star = ray_eigs(X, lam, u_star, T_GRID)
    far = [ray_eigs(X, lam, u, [T_FAR])[0] for u in [unit(deg) for deg in ray_deg] + [u_star]]
    if max(abs(e - 1.0).max() for e in far) > 1e-9:
        raise SystemExit(f"a ray has not reached lam I at t = {T_FAR:g}")
    star_margin = float(np.abs(X @ u_star).min())

    # storyboard values
    sb = STORYBOARD
    ok = (all(close(eta[k], sb["eta_star"][k], 3) for k in (0, 1)) and close(eig0[0], sb["eig0"][0], 1)
          and close(eig0[1], sb["eig0"][1], 1) and close(eig_star[0], sb["eig_star"][0], 2)
          and close(eig_star[1], sb["eig_star"][1], 1) and close(norm_star, sb["norm_star"], 2))
    ray_min = {t: float(ray_eigs(X, lam, u_star, [t])[0][0]) for t in sb["ray_min"]}
    ok = ok and close(ray_min[5.0], 48.0, 0) and close(ray_min[30.0], 5.9, 1) and close(ray_min[100.0], 1.1, 1)
    if not ok:
        raise SystemExit(f"a deterministic value of the storyboard is not reproduced: eta_* {eta}, eig0 {eig0}, "
                         f"eig_star {eig_star}, |eta_*| {norm_star}, ray {ray_min}")

    # SAGA runs (project code), mu_eff before every iteration
    runs = []
    for s in SEEDS:
        r = algorithms.saga(prob, np.zeros(2), passes=PASSES, seed=s, record_every=N, instrument_every=1,
                            snapshot_theta=True, record_F=False)
        instr = r["instr"]
        t = np.array([q["t"] for q in instr])
        th = np.array([q["theta"] for q in instr])
        mu_eff = np.array([q["mu_eff"] for q in instr]) / lam
        v = th - eta
        g = np.array([prob.grad(x) for x in th[::97]])                      # definition, on a subsample of iterates
        direct = np.einsum("ij,ij->i", v[::97], g) / np.einsum("ij,ij->i", v[::97], v[::97]) / lam
        if np.abs(direct / mu_eff[::97] - 1.0).max() > 1e-9:
            raise SystemExit("mu_eff of the instrumentation disagrees with its definition")
        att = analysis.attribution(instr, r["gamma"], r["Delta"], r["R2"], r["mu"], N, t_min=WINDOW[0] * N,
                                   t_max=WINDOW[1] * N, v_min=1e-18)
        win = (t >= WINDOW[0] * N) & (t <= WINDOW[1] * N)
        L_eff = np.array([q["L_eff"] for q in instr]) / prob.R2
        runs.append({"seed": s, "passes": t / N, "mu_eff": mu_eff, "norm_max": float(np.linalg.norm(th, axis=1).max()),
                     "speedup": float(att["speedup_total"]), "gm": float(np.exp(np.log(mu_eff[win]).mean())),
                     "win_min": float(mu_eff[win].min()), "win_max": float(mu_eff[win].max()),
                     "L_eff_gm": float(np.exp(np.log(L_eff[win]).mean())), "last": float(mu_eff[-1]),
                     "D_rel": float(r["D"][-1] / r["D"][0]), "gamma": r["gamma"]})
    return {"X": X, "lam": lam, "R2": prob.R2, "eta": eta, "gnorm": gnorm, "eig0": eig0, "eig_star": eig_star,
            "norm_star": norm_star, "ray_deg": ray_deg, "rays": rays, "ray_star": ray_star, "margins": margins,
            "star_margin": star_margin, "star_deg": float(np.degrees(np.arctan2(u_star[1], u_star[0])) % 360.0),
            "ray_min": ray_min, "runs": runs, "radius": max(r["norm_max"] for r in runs),
            "mu_eff_0": float(runs[0]["mu_eff"][0])}


def print_all(D):
    lam = D["lam"]
    print(f"toy: n = {N}, d = 2, lam = mu = {lam:.6e}, R^2 = {D['R2']:.6f} = {D['R2'] / lam:.1f} lam (kappa = "
          f"{KAPPA:g}); SAGA step gamma = 1/(4 R^2) = {D['runs'][0]['gamma']:.6f}")
    print(f"minimizer eta_* = ({D['eta'][0]:+.4f}, {D['eta'][1]:+.4f}), |eta_*| = {D['norm_star']:.4f}, direction "
          f"{D['star_deg']:.2f} deg, |F'(eta_*)| = {D['gnorm']:.1e}")
    print(f"eigenvalues of F''(0) / lam: {D['eig0'][0]:.2f}, {D['eig0'][1]:.2f} (the upper level, L_F / lam); "
          f"of F''(eta_*) / lam: {D['eig_star'][0]:.3f}, {D['eig_star'][1]:.2f}")
    print(f"(a) rays at {D['ray_deg']} deg: smallest |x_i . u| = {D['margins'][RAY_OFFSET_DEG]:.5f} (offset "
          f"{RAY_OFFSET_DEG} deg, the best whole-degree offset; offset 0 would give {D['margins'][0]:.5f}); ray "
          f"through eta_*: {D['star_margin']:.5f}")
    print("    smallest eigenvalue / lam on the ray through eta_*: "
          + ", ".join(f"{v:.3f} at t = {t:g}" for t, v in D["ray_min"].items()))
    for name, e in [("eta_*", D["ray_star"])] + [(f"{deg:3d} deg", e) for deg, e in zip(D["ray_deg"], D["rays"])]:
        k = {t: int(np.argmin(np.abs(T_GRID - t))) for t in (0.1, 10.0, 100.0, 1000.0, 1e4)}
        print(f"    ray {name:>7s}: (lam_min, lam_max) / lam = "
              + "; ".join(f"({e[i, 0]:.4g}, {e[i, 1]:.4g}) at {t:g}" for t, i in k.items()))
    print(f"    ball of the iterates (five runs, {PASSES} passes): max_t |theta_t| = {D['radius']:.4f} "
          f"(per seed: " + ", ".join(f"{r['norm_max']:.3f}" for r in D["runs"]) + ")")
    print(f"(b) mu_eff(theta_0) / mu = {D['mu_eff_0']:.2f}; per seed, on passes {WINDOW[0]}-{WINDOW[1]}: geometric "
          f"mean, min, max of mu_eff / mu; exact / certified exponent; geometric mean of L_eff / R^2; mu_eff / mu "
          f"and relative squared distance after {PASSES} passes")
    for r in D["runs"]:
        print(f"    seed {r['seed']}: {r['gm']:.2f}, {r['win_min']:.2f}, {r['win_max']:.1f}; {r['speedup']:.1f}; "
              f"{r['L_eff_gm']:.4f}; {r['last']:.3f}, {r['D_rel']:.2e}")
    sp = [r["speedup"] for r in D["runs"]]
    print(f"    exact / certified exponent on passes {WINDOW[0]}-{WINDOW[1]}: {min(sp):.0f}-{max(sp):.0f} over the "
          f"five seeds (the storyboard quoted 141-158 for five other index streams; see the report)")


# ----------------------------------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------------------------------
def draw(D):
    ps = es.setup("talk")
    ink = ps.INK
    RUN = ps.BLUE                                  # the SAGA runs, and the ball that contains their iterates
    fig = es.figure("talk", height=2.75)
    fig.get_layout_engine().set(w_pad=0.03, wspace=0.03)
    gs = fig.add_gridspec(1, 2, width_ratios=(1.35, 1.0))
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
    b = es.Build()
    top, floor_star = D["eig0"][1], D["eig_star"][0]
    note_kw = dict(fontsize=es.small(), color=ink["secondary"], textcoords="offset points")

    # (a) curvature along rays ---------------------------------------------------------------------------------------
    ax_a.axhline(1.0, color=ink["primary"], lw=1.0, zorder=2)
    b.add(3, ax_b.axhline(1.0, color=ink["primary"], lw=1.0, zorder=2))
    ax_a.axhline(top, color=ink["primary"], lw=1.0, ls=(0, (4.0, 2.0)), zorder=2)
    b.add(3, ax_a.axvspan(T_GRID[0], D["radius"], color=es.tint(RUN, 0.16), lw=0, zorder=0))
    for e in D["rays"]:
        for j in (0, 1):
            b.add(2, ax_a.plot(T_GRID, e[:, j], color=ink["muted"], lw=0.7, zorder=3))
    e = D["ray_star"]
    ax_a.fill_between(T_GRID, e[:, 0], e[:, 1], color=ink["muted"], alpha=0.20, lw=0, zorder=1)
    for j in (0, 1):
        ax_a.plot(T_GRID, e[:, j], color=ink["primary"], lw=2.0, zorder=5)
    ax_a.plot([D["norm_star"]], [floor_star], ls="", marker="o", ms=6.5, mfc=ink["primary"], mec=ink["surface"],
              mew=1.2, zorder=7)
    ax_a.annotate(r"minimizer $\eta_\ast$", (D["norm_star"], floor_star), xytext=(-7, -5), ha="right", va="top",
                  **note_kw)
    ax_a.annotate("on the ray\n" + r"through $\eta_\ast$", (T_GRID[0], 360.0), xytext=(4, 0), ha="left", va="center",
                  linespacing=1.2, **note_kw)
    b.add(2, ax_a.annotate(f"{N_RAYS} other rays", (330.0, 20.0), xytext=(1900.0, 150.0), textcoords="data",
                           ha="center", va="bottom", fontsize=es.small(), color=ink["secondary"],
                           arrowprops=dict(arrowstyle="-", color=ink["muted"], lw=0.7, shrinkA=1.0, shrinkB=2.0)))
    ax_a.annotate(rf"largest curvature: {top:.0f}, reached at $\theta = 0$", (T_GRID[-1], top), xytext=(0, 3),
                  ha="right", va="bottom", **note_kw)
    ax_a.annotate(r"smallest: $\mu = \lambda$, only as $|\theta| \to \infty$", (T_GRID[-1], 1.0), xytext=(0, -4),
                  ha="right", va="top", **note_kw)
    b.add(3, ax_a.annotate("where SAGA goes:\n" + rf"$|\theta_t| \leq {D['radius']:.2f}$", (T_GRID[0], 1.0),
                           xytext=(4, 5), ha="left", va="bottom", linespacing=1.2, **note_kw))
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlim(T_GRID[0], T_GRID[-1])
    ax_a.set_ylim(0.28, 6000.0)
    ax_a.set_yticks([1, 10, 100, 1000])
    ax_a.set_yticklabels(["1", "10", "100", "1000"])
    ax_a.set_xticks([0.1, 1, 10, 100, 1000, 10000])
    ax_a.set_xticklabels(["0.1", "1", "10", "100", r"$10^3$", r"$10^4$"])
    ax_a.minorticks_off()
    ax_a.set_xlabel(r"distance $|\theta|$ from the origin, along a ray")
    ax_a.set_ylabel(r"curvature / $\lambda$")
    es.panel_title(ax_a, r"(a) eigenvalues of $F''(\theta)$")

    # (b) the curvature met by SAGA ------------------------------------------------------------------------------------
    b.add(3, ax_b.axhline(floor_star, color=ink["primary"], lw=1.0, ls=(0, (1.2, 1.6)), zorder=2))
    for r in D["runs"]:
        if r["seed"] != HIGHLIGHT:
            b.add(3, ax_b.plot(r["passes"], r["mu_eff"], color=RUN, lw=0.7, alpha=0.55, zorder=3))
    r = D["runs"][SEEDS.index(HIGHLIGHT)]
    b.add(3, ax_b.plot(r["passes"], r["mu_eff"], color=RUN, lw=1.6, zorder=4))
    sp = [x["speedup"] for x in D["runs"]]
    b.add(3, ax_b.annotate(r"$\mu_{\mathrm{eff}}(\theta_t)\,/\,\mu$ along five runs", (PASSES, 700.0), xytext=(-2, 0),
                           ha="right", va="center", **note_kw))
    b.add(3, ax_b.annotate(r"smallest curvature at $\eta_\ast$", (PASSES, floor_star), xytext=(-2, -4),
                           ha="right", va="top", **note_kw))
    x_arrow = 0.07 * PASSES
    b.add(3, es.arrow(ax_b, (x_arrow, 1.0), (x_arrow, floor_star), colour=ink["secondary"], lw=1.0, style="<|-|>",
                      mutation_scale=8.0, zorder=5))
    b.add(3, ax_b.annotate(rf"$\times\,{floor_star:.0f}$", (x_arrow, np.sqrt(floor_star)), xytext=(5, 0), ha="left",
                           va="center", fontsize=es.small() + 1.0, color=ink["primary"],
                           textcoords="offset points"))
    b.add(3, ax_b.annotate("toy: exact / certified\n" + f"exponent = {min(sp):.0f}–{max(sp):.0f}\n"
                           + f"(passes {WINDOW[0]}–{WINDOW[1]})", (PASSES, 1.0), xytext=(-2, 5), ha="right",
                           va="bottom", linespacing=1.2, **note_kw))
    b.add(3, ax_b.annotate(r"$\mu$: the constant of the proof", (PASSES, 1.0), xytext=(-2, -4), ha="right",
                           va="top", **note_kw))
    ax_b.set_xlim(0, PASSES)
    ax_b.set_xticks(np.arange(0, PASSES + 1, 5))
    ax_b.set_xlabel("passes")
    ax_b.tick_params(labelleft=False)
    es.panel_title(ax_b, "(b) along SAGA runs")
    es.save(fig, NAME, "talk", build=b,
            note=(rf"toy ($d = 2$, $n = {N}$, $R^2/\lambda = 10^4$), exact evaluations, {N_RAYS + 1} generic rays; "
                  rf"SAGA: $\gamma = 1/(4R^2)$, $\theta_0 = 0$, {len(SEEDS)} seeds"))


def main():
    D = data()
    print_all(D)
    draw(D)


if __name__ == "__main__":
    main()
