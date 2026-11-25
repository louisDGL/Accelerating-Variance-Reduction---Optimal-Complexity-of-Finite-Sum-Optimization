"""Figure T19 of the lecture (paper/talk/figures/figT19_one_step_map): the one inequality behind Proposition 5(b) of
paper/paper_saga_to_muon.tex and behind the length floor of item (M1), drawn as a map of the line. Explanatory
figure, not in the paper; nothing is measured, nothing is read from the results of the project.

The statement (proof sketch of Proposition 5(b); facts F13 (a)-(c) of muon/notes_theory.md). For polar steps
W_{t+1} = W_t - eta_t polar(omega_t) fed by a SAGA table whose entries are at most A iterations old, with
D_t = F(W_t) - F_* (a gap in function values, not a squared distance),
    D_{t+1} <= D_t - eta_t (2 mu_op D_t)^(1/2) + (L_eff / 2) eta_t^2,        L_eff = L_op + 10 A L_max :
the table enters through L_eff only. In the scale-free variable u_t = D_t / eta_t^2, with eta_{t+1} = rho^(1/2) eta_t,
    u_{t+1} <= g_rho(u_t),      g_rho(u) = (u - (2 mu_op u)^(1/2) + L_eff / 2) / rho,
a convex function of u, so that u_t <= C implies u_{t+1} <= max(g_rho(0), g_rho(C)).

The toy: the exact graphs of g_rho for the illustrative constants mu_op = 1, L_eff = 4 (kappa_eff = 4). They satisfy
the conditions of the induction (F13 (b)-(c): L_eff >= 2 mu_op, 1 - mu_op / L_eff <= rho < 1, and L_eff at least the
staleness constant L_op + 4 (theta L_max + L_mean) sum_{k <= A} rho^(-k/2)); they are NOT values of the closed form
L_eff = L_op + 10 A L_max of Proposition 5(b), which is at least 11 mu_op (A >= 1, L_max >= mu_op). On the two
conditioning-limited instances of the paper the closed form gives kappa_eff of order 1e7 (F13, "What F13 does not
say" (1): a certified length factor 1 - 2.5e-6 and 1 - 3.4e-6 per pass at n/b = 200, i.e. kappa_eff about 4e7 and
3e7 for the reshuffled order), millions of times the toy's: the graphs are then within a hair of the diagonal and
nothing could be seen.
  (a) constant length (rho = 1). g_1(u) <= u iff u >= u_fix = L_eff^2 / (8 mu_op) (= 2), g_1(0) = L_eff / 2 (= 2), and
      the minimum of g_1 is (L_eff - mu_op) / 2 at u = mu_op / 2. Every interval [0, C] with
      C >= max(L_eff^2 / (8 mu_op), L_eff / 2) is sent into itself (F13 (b)): D <= C eta^2 is invariant, the upper
      bound stalls at order eta^2 and certifies no convergence (label "bound stalls: D <= 2 eta^2"; it is an upper
      bound, not the level of a run). The staircase iterates the bound, u <- g_1(u), from u_0 = 11: it decreases to
      u_fix (valid because every step of it stays above u_fix >= g_1(0), where g_1 is increasing). In the other
      direction max(D_t, D_{t+1}) >= mu_op eta^2 / 8 (fact F2, u >= 1/8 every other step): together, a floor of
      order eta^2; the lower bound is not drawn. The diagonal u_{t+1} = u_t is labelled in panel (a).
  (b) geometric cooldown. Dividing by rho lifts the graph of (a) (thin line: rho = 1). For
      rho = 1 - mu_op / L_eff (= 3/4) it touches the diagonal at the single point C = L_eff^2 / (2 mu_op) (= 8), and
      g_rho(0) = L_eff / (2 rho) <= C: the interval [0, C] is sent into itself, i.e. D_0 <= C eta_0^2 gives
      D_t <= C eta_t^2 = C eta_0^2 rho^t, which is Proposition 5(b) (eta_0 = (2 mu_op D_0)^(1/2) / L_eff); the label
      of the figure states the hypothesis D_0 <= 8 eta_0^2. (The contact is a tangency for every constants:
      g_rho(u) - u = (sqrt(mu_op / L_eff) sqrt(u) - sqrt(L_eff / 2))^2 / rho at rho = 1 - mu_op / L_eff.) For a
      faster cooldown rho' < 1 - mu_op / L_eff (drawn: rho' = 0.6) the graph is above the diagonal everywhere:
      g(C) > C for every C, no interval [0, C] is invariant, the bound on D_t / eta_t^2 grows without limit: the
      inequality certifies nothing (it does not say that the run fails: the proposition gives no critical q).

Every drawn point (u_fix, C, the values at 0) is computed from the closed forms above AND by brute force on a grid or
by root finding; the script stops if they disagree, and it checks numerically that [0, C] is sent into itself for
rho = 1 - mu_op / L_eff and for no C when rho = rho' (print_all lists the checks).

Status. Proved: everything drawn (the map of the proof and its invariant intervals). Illustration: the constants.
Not on the figure: any run; measured values of D_t / eta_t^2; the speed of the experiments (the certified cooldown
is 3.3e4 to 8.9e4 times slower than q = 0.8 on the instances of the paper for the permutation orders, and of the
order of 1e6 for independent sampling: audit, finding on Proposition 5).

Colours: the map of the polar step fed by a table in orange (thin and light for rho = 1 in panel (b)), the cooldown
that the inequality does not certify in grey, the diagonal and the staircase in ink; the two status marks say
"certified" / "not certified" and nothing about what a run does.

Progressive build (states s1-s3): s1 panel (a) without the staircase; s2 the staircase; s3 panel (b) and its labels
(to the right of the panel: the certified ratio, and where the table enters).

Run:    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/scripts/make_figT19_one_step_map.py
Output: paper/talk/figures/figT19_one_step_map.pdf and .png, and the states _s1 ... _s3
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import explain_style as es  # noqa: E402

NAME = "figT19_one_step_map"
MU, L = 1.0, 4.0                    # illustrative constants mu_op, L_eff
RHO_FAST = 0.6                      # a cooldown faster than the certified one, rho' < 1 - mu_op / L_eff
U0, N_STEPS = 11.0, 9               # the staircase of panel (a)
U_MAX = 12.0                        # window of both panels
GRID = np.linspace(0.0, 200.0, 2000001)


# ----------------------------------------------------------------------------------------------------------------------
# the map
# ----------------------------------------------------------------------------------------------------------------------
def g(u, rho=1.0):
    """Bound on u_{t+1} = D_{t+1} / eta_{t+1}^2 given u_t = D_t / eta_t^2, for eta_{t+1}^2 = rho eta_t^2."""
    u = np.asarray(u, dtype=float)
    return (u - np.sqrt(2.0 * MU * u) + 0.5 * L) / rho


def data():
    rho = 1.0 - MU / L
    D = {"rho": rho, "u_fix": L ** 2 / (8.0 * MU), "g0": 0.5 * L, "u_min": 0.5 * MU, "g_min": 0.5 * (L - MU),
         "C": L ** 2 / (2.0 * MU), "g0_rho": 0.5 * L / rho}
    checks = []

    def check(name, ok, detail):
        checks.append((name, bool(ok), detail))

    # (a) rho = 1
    root = brentq(lambda u: float(g(u) - u), 0.6, 50.0)
    check("(a) fixed point: closed form L^2/(8 mu) = root of g(u) = u", abs(root - D["u_fix"]) < 1e-10,
          f"{D['u_fix']:.6f} / {root:.6f}")
    k = int(np.argmin(g(GRID)))
    check("(a) minimum of g: (L - mu)/2 at mu/2", abs(GRID[k] - D["u_min"]) < 1e-3 and abs(g(GRID[k]) - D["g_min"]) < 1e-9,
          f"{D['g_min']:.4f} at {D['u_min']:.4f} / {float(g(GRID[k])):.4f} at {GRID[k]:.4f}")
    above = GRID[g(GRID) > GRID + 1e-12]
    check("(a) g(u) <= u exactly for u >= u_fix", abs(above.max() - D["u_fix"]) < 1e-3, f"g(u) > u up to u = {above.max():.4f}")
    C_a = max(D["u_fix"], D["g0"])
    inside = GRID[GRID <= C_a]
    check("(a) [0, C] is sent into itself for C = max(L^2/(8 mu), L/2)", g(inside).max() <= C_a + 1e-12,
          f"C = {C_a:.4f}, largest image {g(inside).max():.6f}, smallest {g(inside).min():.6f}")
    inside = GRID[GRID <= 0.9 * C_a]
    check("(a) ... and not for C 10 % smaller", g(inside).max() > 0.9 * C_a, f"largest image {g(inside).max():.4f} > "
          f"{0.9 * C_a:.4f}")
    D["C_a"] = C_a
    stairs = [U0]
    for _ in range(N_STEPS):
        stairs.append(float(g(stairs[-1])))
    D["stairs"] = np.array(stairs)
    check("(a) the staircase decreases to u_fix", np.all(np.diff(stairs) < 0) and stairs[-1] > D["u_fix"],
          ", ".join(f"{v:.3f}" for v in stairs))

    # (b) rho = 1 - mu / L
    gap = g(GRID, rho) - GRID
    k = int(np.argmin(gap))
    check("(b) rho = 1 - mu/L: the graph touches the diagonal at C = L^2/(2 mu) only",
          abs(GRID[k] - D["C"]) < 1e-2 and abs(gap[k]) < 1e-9 and gap.min() > -1e-12,
          f"smallest g(u) - u = {gap[k]:.2e} at u = {GRID[k]:.4f}; closed form C = {D['C']:.4f}")
    inside = GRID[GRID <= D["C"]]
    check("(b) rho = 1 - mu/L: [0, C] is sent into itself", g(inside, rho).max() <= D["C"] + 1e-9
          and abs(float(g(0.0, rho)) - D["g0_rho"]) < 1e-12 and D["g0_rho"] <= D["C"],
          f"largest image {g(inside, rho).max():.6f} (at u = {inside[np.argmax(g(inside, rho))]:.4f}), g(0) = "
          f"{D['g0_rho']:.4f} = L/(2 rho)")
    eta0, D0 = 1.0, D["C"]                                   # the recursion of the proposition, worst case, 60 steps
    Dt, eta, worst = D0, eta0, 0.0
    for t in range(60):
        Dt = Dt - eta * np.sqrt(2.0 * MU * Dt) + 0.5 * L * eta ** 2
        eta *= np.sqrt(rho)
        worst = max(worst, Dt / (D0 * rho ** (t + 1)))
    check("(b) the recursion started at D_0 = C eta_0^2 stays at D_t <= D_0 rho^t", worst <= 1.0 + 1e-9,
          f"largest D_t / (D_0 rho^t) over 60 steps = {worst:.12f}")
    # (b) rho' < 1 - mu / L
    gap = g(GRID, RHO_FAST) - GRID
    k = int(np.argmin(gap))
    D["gap_fast"] = (float(gap[k]), float(GRID[k]))
    check(f"(b) rho' = {RHO_FAST}: the graph is above the diagonal (no C with g(C) <= C)", RHO_FAST < rho and gap.min() > 0,
          f"smallest g(u) - u = {gap[k]:.4f} at u = {GRID[k]:.4f}; closed form "
          f"L/(2 rho') - mu/(2 rho' (1 - rho')) = {0.5 * L / RHO_FAST - 0.5 * MU / (RHO_FAST * (1 - RHO_FAST)):.4f}")
    check(f"(b) rho' = {RHO_FAST}: the image of C exceeds C", float(g(D["C"], RHO_FAST)) > D["C"],
          f"g(C) = {float(g(D['C'], RHO_FAST)):.4f} > C = {D['C']:.4f}")
    u, seq = 0.0, []
    for _ in range(40):
        u = float(g(u, RHO_FAST))
        seq.append(u)
    check(f"(b) rho' = {RHO_FAST}: the iterated bound grows without limit", np.all(np.diff(seq) > 0) and seq[-1] > 1e3,
          f"after 10, 20, 40 steps from u = 0: {seq[9]:.1f}, {seq[19]:.1f}, {seq[39]:.3g}")
    D["checks"] = checks
    return D


def print_all(D):
    print(f"illustrative constants: mu_op = {MU:g}, L_eff = {L:g} (kappa_eff = {L / MU:g})")
    print(f"(a) rho = 1: g(0) = L/2 = {D['g0']:g}; fixed point u_fix = L^2/(8 mu) = {D['u_fix']:g}; minimum "
          f"(L - mu)/2 = {D['g_min']:g} at u = mu/2 = {D['u_min']:g}; invariant interval drawn: [0, {D['C_a']:g}]")
    print(f"    staircase from u_0 = {U0:g}: " + ", ".join(f"{v:.3f}" for v in D["stairs"]))
    print(f"(b) rho = 1 - mu/L = {D['rho']:g}: C = L^2/(2 mu) = {D['C']:g}; g(0) = L/(2 rho) = {D['g0_rho']:.4f}; "
          f"rho' = {RHO_FAST:g}: g(0) = {0.5 * L / RHO_FAST:.4f}, g(C) = {float(g(D['C'], RHO_FAST)):.4f}, smallest "
          f"distance to the diagonal {D['gap_fast'][0]:.4f} at u = {D['gap_fast'][1]:.3f}")
    print("checks (closed form / brute force):")
    for name, ok, detail in D["checks"]:
        print(f"    {'ok  ' if ok else 'FAIL'} {name}: {detail}")
    if not all(ok for _, ok, _ in D["checks"]):
        raise SystemExit("the figure is not drawn: a closed form and its numerical check disagree")


# ----------------------------------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------------------------------
def draw(D):
    from matplotlib.patches import Rectangle
    ps = es.setup("talk")
    ink = ps.INK
    MAP = ps.ORANGE                                   # the polar step fed by a table
    fig = es.figure("talk", height=2.75)
    fig.get_layout_engine().set(w_pad=0.03, wspace=0.06)
    gs = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 0.74))
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], sharex=ax_a, sharey=ax_a)
    ax_t = fig.add_subplot(gs[0, 2])                  # room for the labels of panel (b)
    es.diagram_axes(ax_t)
    b = es.Build()
    b.add(3, ax_b)
    u = np.linspace(0.0, U_MAX, 1201)
    small = es.small()
    note = dict(fontsize=small, color=ink["secondary"], zorder=7, linespacing=1.3)

    def frame(ax):
        ax.plot([0, U_MAX], [0, U_MAX], color=ink["muted"], lw=1.0, zorder=2)
        ax.set_xlim(0, U_MAX)
        ax.set_ylim(0, U_MAX)
        ax.set_aspect("equal", adjustable="box")
        ax.set_axisbelow(True)
        ax.set_xticks([0, 4, 8, 12])
        ax.set_yticks([0, 4, 8, 12])
        ax.set_xlabel(r"$u_t = D_t\,/\,\eta_t^2$")

    def box(ax, c):
        """The invariant interval [0, c] as a square: the graph over [0, c] stays inside it."""
        ax.add_patch(Rectangle((0, 0), c, c, fc=es.tint(MAP, 0.16), ec=MAP, lw=0.9, ls=(0, (3.0, 2.0)), zorder=1))

    # (a) constant length
    frame(ax_a)
    box(ax_a, D["C_a"])
    ax_a.plot(u, g(u), color=MAP, lw=2.0, zorder=4)
    ax_a.plot([D["u_fix"]], [D["u_fix"]], ls="", marker="o", ms=6.5, mfc=MAP, mec=ink["surface"], mew=1.2, zorder=6)
    ax_a.set_ylabel(r"bound on $u_{t+1}$")
    es.panel_title(ax_a, "(a) constant length")
    ax_a.text(0.4, 11.5, rf"toy: $\mu_{{op}} = {MU:g}$, $L_{{\mathrm{{eff}}}} = {L:g}$", ha="left", va="top", **note)
    ax_a.text(11.8, 2.8, r"$u - \sqrt{2\mu_{op}u} + L_{\mathrm{eff}}/2$", ha="right", va="center", **note)
    ax_a.text(D["C_a"] + 0.55, 0.95, rf"bound stalls: $D \leq {D['C_a']:g}\,\eta^2$", ha="left", va="center", **note)
    ax_a.text(6.9, 7.5, r"$u_{t+1} = u_t$", rotation=45, rotation_mode="anchor", ha="center", va="bottom", **note)
    st = D["stairs"]
    xs, ys = [st[0]], [st[0]]
    for k in range(len(st) - 1):
        xs += [st[k], st[k + 1]]
        ys += [st[k + 1], st[k + 1]]
    b.add(2, ax_a.plot(xs, ys, color=ink["primary"], lw=1.0, zorder=5))
    b.add(2, ax_a.plot([st[0]], [st[0]], ls="", marker="o", ms=4.5, mfc=ink["primary"], mec=ink["surface"], mew=0.8,
                       zorder=6))
    b.add(2, ax_a.annotate("$u_0$", (st[0], st[0]), xytext=(-5, 4), textcoords="offset points", ha="right",
                           va="bottom", **note))

    # (b) geometric cooldown
    frame(ax_b)
    box(ax_b, D["C"])
    ax_b.plot(u, g(u), color=es.tint(MAP, 0.45), lw=1.2, zorder=3)
    keep = g(u, RHO_FAST) <= U_MAX
    ax_b.plot(u[keep], g(u[keep], RHO_FAST), color=ink["muted"], lw=2.0, zorder=4)
    ax_b.plot(u, g(u, D["rho"]), color=MAP, lw=2.0, zorder=5)
    ax_b.plot([D["C"]], [D["C"]], ls="", marker="o", ms=6.5, mfc=MAP, mec=ink["surface"], mew=1.2, zorder=6)
    ax_b.tick_params(labelleft=False)
    es.panel_title(ax_b, r"(b) cooldown $\eta_{t+1}^2 = \rho\,\eta_t^2$")
    es.mark(ax_b, 1.85, 11.0, False)                  # next to the grey graph of rho' (it leaves the panel at u = 9.6)
    ax_b.text(2.55, 11.55, rf"$\rho' = {RHO_FAST:g}$:" + "\nnot certified", ha="left", va="top", **note)
    outside = []
    outside.append(ax_b.annotate(r"$\rho = 1$, as in (a)", (U_MAX, float(g(U_MAX))), xytext=(7, 0),
                                 textcoords="offset points", ha="left", va="center", annotation_clip=False, **note))
    outside.append(ax_b.text(
        1.075, 0.42, r"$\rho = 1 - \mu_{op}/L_{\mathrm{eff}}$:" + "\n"
        + rf"$D_0 \leq {D['C']:g}\,\eta_0^2\ \Rightarrow$" + "\n"
        + rf"$D_t \leq {D['C']:g}\,\eta_t^2 = {D['C']:g}\,\eta_0^2\rho^t$", transform=ax_b.transAxes, ha="left",
        va="center", clip_on=False, **note))
    c = D["C"] / U_MAX                                # leader from the point of contact to its label
    outside.append(ax_b.plot([c + 0.03, 1.055], [c - 0.018, 0.555], color=ink["muted"], lw=0.7, transform=ax_b.transAxes,
                             clip_on=False, zorder=6)[0])
    outside.append(ax_b.text(1.075, 0.64, "certified", transform=ax_b.transAxes, ha="left", va="center",
                             clip_on=False, **note))
    fig.canvas.draw()                                 # the width of a text in points does not depend on the layout
    w = outside[-1].get_window_extent().width * 72.0 / fig.dpi
    outside.append(ax_b.annotate("✓", (1.075, 0.64), xycoords="axes fraction", xytext=(w + 9.0, 0.0),
                                 textcoords="offset points", ha="center", va="center", fontsize=small + 3,
                                 color=es.GOOD, fontweight="bold", annotation_clip=False, zorder=7))
    outside.append(ax_b.text(1.075, 0.06, "the table enters\nthrough " + r"$L_{\mathrm{eff}}$ only:" + "\n"
                             + r"$L_{\mathrm{eff}} = L_{op} + 10\,A\,L_{\max}$" + "\n" + r"($A$: oldest entry's age)",
                             transform=ax_b.transAxes, ha="left", va="center", clip_on=False, **note))
    for t in outside:                                 # the labels live in the empty third column of the grid
        t.set_in_layout(False)
    es.save(fig, NAME, "talk", build=b,
            note=(r"Prop. 5(b): exact polar, no momentum; toy constants; the runs cool "
                  r"$10^4$–$10^6\times$ faster than certified"))


def main():
    D = data()
    print_all(D)
    draw(D)


if __name__ == "__main__":
    main()
