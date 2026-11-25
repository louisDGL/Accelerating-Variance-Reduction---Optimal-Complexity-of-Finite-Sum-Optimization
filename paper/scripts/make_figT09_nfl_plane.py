"""Figure T09 of the lecture (paper/talk/figures/figT09_nfl_plane): Theorem 1 ("no free lunch, for every norm") of
paper/paper_saga_to_muon.tex in the plane. Explanatory figure, not in the paper; it is the talk version of panels (a)
and (b) of paper/figures/figE2_nfl_plane (the dot plot of the condition numbers is the companion figure
figT09b_nfl_norms).

Message: no norm improves the worst-case condition number of a ridge logistic model: in one and the same direction the
curvature already varies by kappa_F between two points of the space, and a norm can only reweight directions.

The toy is the one of make_figE2_nfl_plane.py, imported from that script (its data(), ellipse(), constants), an exact
instance of the hypotheses of Theorem 1 with K = 1 and d = 2 (the parameter W is a row vector of the plane):
    F(W) = (1/n) sum_i log(1 + exp(-y_i W x_i)) + (lam/2) |W|^2,   n = 40,  lam = 0.02,
with features whose second-moment matrix has the eigenvalues 32 lam and 5 lam, so that H0 = D^2F(0) has the eigenvalues
9 lam and 2.25 lam (eigenvectors u_1, u_2), D^2F(t W_1) -> lam I along the ray, mu_F = lam and kappa_F = 9.

Panels:
  (a) curvature in a fixed direction, D^2F(t W_1)[u, u] / lam, along the ray t W_1 (W_1 at 11 degrees), for u = u_1
      and u = u_2: logistic loss (solid) and the quadratic loss l(z) = z^2/8, whose Hessian is H0 everywhere (dashed).
      In the direction u_1 alone the curvature of the logistic model goes from kappa_F lam to lam.
  (b) the plane of the directions Delta, unit 1/sqrt(lam), drawn in the orthonormal basis (u_2, u_1) of the
      eigenvectors of H0 (u_1 upwards; a rotation of the picture of figure E2, so that the whole outer copy of the
      matched ball fits a wide panel): the set E_0 = {D^2F(0)[Delta, Delta] <= 1} (grey; the
      curvature is largest at W = 0 in every direction) and the disc E_inf = {lam |Delta|^2 <= 1} (thick circle; the
      floor). For a norm N with unit ball B: L_N = 1/r_in^2 with r_in B the largest copy of B inside E_0, and
      mu_N = 1/r_out^2 with r_out B the smallest copy of B that contains E_inf, so kappa_N = (r_out/r_in)^2. On the
      line of u_1 the boundary of E_0 is at distance 1/sqrt(kappa_F) and the circle at distance 1: an inner copy
      cannot pass the first point, an outer copy must reach the second, whatever the ball. Drawn for the Euclidean
      ball and for the ball matched to H0 (N(Delta)^2 = H0[Delta, Delta]): both need the magnification
      sqrt(kappa_F) = 3. The script checks these four radii against L_N and mu_N of the imported data (closed forms,
      themselves checked there by brute force) and stops if they disagree.

Status. Proved: kappa_N >= kappa_F for every norm (Theorem 1). The figure shows why on one instance and two norms; the
numbers are those of the toy, computed, nothing is drawn by hand. Not claimed: that a change of norm is useless (it
changes local constants and trajectories, findings (C5) and (M4) of the paper); the figure says "norm", not "geometry"
(a Bregman geometry escapes the theorem), and K = 1, so the spectral norm of W is its Euclidean norm.

Progressive build (states s1-s4): s1 panel (a), logistic loss; s2 adds the quadratic loss with the same Hessian at 0;
s3 adds panel (b) with the two sets and the two points on the line of u_1; s4 adds the two balls, each with its copy
inside the grey set and its copy around the circle.

The script prints every number it plots. Nothing is read from the results of the project.
Run:    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/scripts/make_figT09_nfl_plane.py
Output: paper/talk/figures/figT09_nfl_plane.pdf and .png, and the states _s1 ... _s4
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import explain_style as es  # noqa: E402
import make_figE2_nfl_plane as E2  # noqa: E402  (the toy and its checks; importing it draws nothing)

NAME = "figT09_nfl_plane"
LAM, T_GRID = E2.LAM, E2.T_GRID
NOTE = (r"exact toy: ridge logistic regression, $K = 1$, $d = 2$, $n = 40$ (row vector: spectral norm = Euclidean "
        "norm)")


# ----------------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------------
def data():
    D = E2.data()                                        # the toy, with closed forms checked by brute force
    missing = {"H0", "eigs", "u1", "u2", "kappa_F", "curv", "rows"} - set(D)
    if missing:
        raise SystemExit(f"make_figE2_nfl_plane.data() no longer returns {sorted(missing)}")
    root = float(np.sqrt(D["kappa_F"]))
    u1 = D["u1"] * np.sign(D["u1"][0])
    H0n = D["H0"] / LAM                                  # unit of panel (b): 1/sqrt(lam)
    rows = {r["key"]: r for r in D["rows"]}
    # the copies r_in B and r_out B of the unit ball B of a norm, r_in = L_N^(-1/2) and r_out = mu_N^(-1/2), from L_N and
    # mu_N of the imported data. In the picture (unit 1/sqrt(lam)) the Euclidean unit ball is the disc of radius
    # sqrt(lam), and the unit ball of the matched norm, {H0[Delta, Delta] <= 1}, is E_0 itself.
    radii = {}
    for key, unit in (("l2", np.sqrt(LAM)), ("H0", 1.0)):
        radii[key] = (float(unit / np.sqrt(rows[key]["L"])), float(unit / np.sqrt(rows[key]["mu"])))
    # what the picture draws: Euclidean copies of radii 1/sqrt(kappa_F) and 1; matched copies E_0 and sqrt(kappa_F) E_0
    drawn = {"l2": (1.0 / root, 1.0), "H0": (1.0, root)}
    for key in drawn:
        if max(abs(radii[key][0] / drawn[key][0] - 1.0), abs(radii[key][1] / drawn[key][1] - 1.0)) > 1e-9:
            raise SystemExit(f"{key}: the copies drawn in panel (b) disagree with L_N and mu_N: {radii[key]}")
        if abs((radii[key][1] / radii[key][0]) ** 2 / rows[key]["kappa"] - 1.0) > 1e-9:
            raise SystemExit(f"{key}: (r_out / r_in)^2 is not kappa_N")
    # the two pinned points: where the line of u_1 leaves E_0 and where it leaves the disc
    t_in = 1.0 / np.sqrt(float(u1 @ H0n @ u1))
    if abs(t_in * root - 1.0) > 1e-9:
        raise SystemExit("the boundary of E_0 on the line of u_1 is not at 1/sqrt(kappa_F)")
    # the quadratic loss with the Hessian H0 everywhere: constant curvature in each direction
    quad_level = {"u1": float(D["eigs"][1] / LAM), "u2": float(D["eigs"][0] / LAM)}
    # panel (b) is drawn in the orthonormal basis (u_2, u_1) of the eigenvectors of H0: coordinates = Delta @ basis
    u2 = D["u2"] * np.sign(np.linalg.det(np.stack([D["u2"], u1], axis=1)))        # (u_2, u_1) direct: a rotation
    basis = np.stack([u2, u1], axis=1)
    if abs(np.linalg.det(basis) - 1.0) > 1e-12 or np.abs(basis.T @ H0n @ basis - np.diag(np.diag(basis.T @ H0n @ basis))).max() > 1e-9:
        raise SystemExit("the basis (u_2, u_1) is not a rotation that diagonalizes H0")
    D.update({"root": root, "H0n": H0n, "radii": radii, "t_in": float(t_in), "quad_level": quad_level,
              "basis": basis, "p_in": t_in * u1, "p_out": u1})
    return D


def print_all(D):
    E2.print_all(D)
    c = D["curv"]
    print(f"[T09] (a) logistic, curvature / lam at t = {T_GRID[0]:g} and t = {T_GRID[-1]:g}: u1 {c['u1'][0]:.4f} -> "
          f"{c['u1'][-1]:.6f}, u2 {c['u2'][0]:.4f} -> {c['u2'][-1]:.6f}; at W = 0: {D['quad_level']['u1']:.4f}, "
          f"{D['quad_level']['u2']:.4f} (the levels of the quadratic loss); ratio in the direction u1: "
          f"{D['quad_level']['u1']:.4f} = kappa_F")
    print(f"[T09] (b) unit 1/sqrt(lam): semi-axes of E_0 {1 / np.sqrt(D['quad_level']['u1']):.4f} (along u1), "
          f"{1 / np.sqrt(D['quad_level']['u2']):.4f} (along u2); radius of the floor disc 1")
    print(f"[T09]     points on the line of u1: {D['t_in']:.4f} and 1; ratio {1 / D['t_in']:.4f} = sqrt(kappa_F)")
    for key, name in (("l2", "Euclidean ball"), ("H0", "ball matched to D^2F(0) (unit: E_0)")):
        r = D["radii"][key]
        print(f"[T09]     {name}: copy inside E_0 x {r[0]:.4f}, copy around the disc x {r[1]:.4f}, "
              f"kappa_N = {(r[1] / r[0]) ** 2:.4f}")


# ----------------------------------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------------------------------
DASH = (0, (3.2, 1.8))


def panel_a(ax, D, ps, b):
    ink = ps.INK
    for k in ("u1", "u2"):
        ax.plot(T_GRID, D["curv"][k], color=ink["primary"], lw=2.2, zorder=3,
                label="logistic loss" if k == "u1" else None)
        b.add(2, ax.plot(T_GRID, np.full_like(T_GRID, D["quad_level"][k]), color=ink["muted"], lw=1.6,
                         ls=(0, (4.0, 2.0)), zorder=2, label="quadratic loss" if k == "u1" else None))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(T_GRID[0], T_GRID[-1])
    ax.set_ylim(0.78, 60.0)
    ax.set_yticks([1, 2.25, 9])
    ax.set_yticklabels(["1", "2.25", r"$\kappa_F = 9$"])
    ax.minorticks_off()
    ax.set_xticks([0.1, 1, 10, 100])
    ax.set_xticklabels(["0.1", "1", "10", "100"])
    ax.set_xlabel(r"position $t$ on the ray $tW_1$")
    ax.set_ylabel(r"curvature $D^2F(tW_1)[u,u]\,/\,\lambda$")
    kw = dict(fontsize=es.small(), color=ink["secondary"], textcoords="offset points")
    ax.annotate(r"$u = u_1$", (0.11, D["quad_level"]["u1"]), xytext=(0, 3), ha="left", va="bottom", **kw)
    ax.annotate(r"$u = u_2$", (0.11, D["quad_level"]["u2"]), xytext=(0, 3), ha="left", va="bottom", **kw)
    ax.annotate(r"floor $\lambda$", (92.0, 1.0), xytext=(0, 3), ha="right", va="bottom", **kw)
    b.legend(ax, loc="upper right", handlelength=1.7, borderaxespad=0.1, labelspacing=0.25)
    es.panel_title(ax, "(a) along a ray")


def panel_b(ax, D, ps, b):
    ink = ps.INK
    EU, MA = ink["secondary"], ps.BLUE                  # Euclidean norm: secondary ink; norm matched to a Hessian: blue
    root, H0n = D["root"], D["H0n"]
    B = D["basis"]                                      # columns u_2, u_1: the picture has u_2 across and u_1 upwards
    XL, YT, YB = 2.40, 1.13, -2.30
    es.diagram_axes(ax, (-XL, XL), (YB, YT), equal=True)
    ax.set_anchor("N")
    ang = np.linspace(0.0, 2.0 * np.pi, 721)
    circle = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    e0 = E2.ellipse(H0n) @ B
    big = E2.ellipse(H0n, root) @ B
    # s3: the two sets and the two points on the line of u_1
    b.add(3, ax.fill(e0[:, 0], e0[:, 1], color=ink["grid"], lw=0, zorder=1))
    b.add(3, ax.plot(circle[:, 0], circle[:, 1], color=ink["axis"], lw=4.2, zorder=1, solid_capstyle="round"))
    p_in, p_out = D["p_in"] @ B, D["p_out"] @ B
    b.add(3, ax.plot([0.0, p_out[0]], [0.0, p_out[1]], color=ink["primary"], lw=1.0, zorder=6))
    b.add(3, ax.plot([0.0], [0.0], ls="", marker="o", ms=3.0, color=ink["primary"], zorder=7))
    for p in (p_in, p_out):
        b.add(3, ax.plot([p[0]], [p[1]], ls="", marker="o", ms=6.5, mfc=ink["primary"], mec=ink["surface"], mew=1.0,
                         zorder=8))
    mid = 0.5 * (p_in + p_out)
    b.add(3, ax.text(mid[0] + 0.09, mid[1] - 0.06, rf"$\times\,{root:.0f}$" "\n" r"$= \sqrt{\kappa_F}$",
                     fontsize=es.small(), color=ink["primary"], ha="left", va="center", linespacing=1.05, zorder=9))
    b.add(3, ax.text(mid[0] - 0.09, mid[1], r"$u_1$", fontsize=es.small(), color=ink["secondary"], ha="right",
                     va="center", zorder=9))
    # s4: the two balls, each with its largest copy inside the grey set and its smallest copy around the circle
    b.add(4, ax.plot(circle[:, 0] / root, circle[:, 1] / root, color=EU, lw=1.6, zorder=4))
    b.add(4, ax.plot(circle[:, 0], circle[:, 1], color=EU, lw=1.6, ls=DASH, zorder=4))
    b.add(4, ax.plot(e0[:, 0], e0[:, 1], color=MA, lw=1.6, zorder=5))
    b.add(4, ax.plot(big[:, 0], big[:, 1], color=MA, lw=1.6, ls=DASH, zorder=5))

    # the key, drawn by hand below the picture (same axes) so that each row appears with its state
    HW, GAP = 0.44, 0.11                                 # width of a handle, space between a handle and its text
    txt = dict(fontsize=es.small(), color=ink["secondary"], ha="left", va="center", zorder=6)

    def entry(state, x, y, text, handle):
        arts = [ax.text(x + HW + GAP, y, text, **txt)]
        xm = x + 0.5 * HW
        if handle == "patch":
            arts += ax.fill([x, x + HW, x + HW, x], [y - 0.1, y - 0.1, y + 0.1, y + 0.1], color=ink["grid"], lw=0)
        elif handle == "ring":
            arts += ax.plot([x + 0.04, x + HW - 0.04], [y, y], color=ink["axis"], lw=4.2, solid_capstyle="round")
        else:                                            # a pair solid / dashed in the colour of the ball
            arts += ax.plot([x, xm - 0.04], [y, y], color=handle, lw=1.6, solid_capstyle="butt")
            arts += ax.plot([xm + 0.04, x + HW], [y, y], color=handle, lw=1.6, ls=(0, (2.0, 1.2)))
        b.add(state, *arts)

    x0, y1, dy = -XL + 0.03, -1.37, 0.26
    entry(3, x0, y1, r"largest curvature $\leq 1$", "patch")
    entry(3, x0 + 2.75, y1, r"floor $\lambda\,|\Delta|^2 \leq 1$", "ring")
    entry(4, x0, y1 - dy, "Euclidean ball", EU)
    entry(4, x0 + 2.05, y1 - dy, r"ball matched to $D^2F(0)$", MA)
    b.add(4, ax.text(x0, y1 - 2 * dy, "solid: largest copy inside the grey set", **txt))
    b.add(4, ax.text(x0, y1 - 3 * dy, "dashed: smallest copy around the circle", **txt))
    b.add(3, es.panel_title(ax, "(b) the plane of the directions \u0394"))     # plain text: the two titles align


def draw(D):
    ps = es.setup("talk")
    fig = es.figure("talk", height=2.75)
    fig.get_layout_engine().set(w_pad=0.03, wspace=0.0)
    # two subfigures: the diagram of panel (b) has no tick labels and uses the whole height of its half
    sub_a, sub_b = fig.subfigures(1, 2, width_ratios=(2.5, 3.4))
    ax_a, ax_b = sub_a.add_subplot(111), sub_b.add_subplot(111)
    b = es.Build()
    panel_a(ax_a, D, ps, b)
    panel_b(ax_b, D, ps, b)
    es.save(fig, NAME, "talk", build=b, note=NOTE)


def main():
    D = data()
    print_all(D)
    draw(D)


if __name__ == "__main__":
    main()
