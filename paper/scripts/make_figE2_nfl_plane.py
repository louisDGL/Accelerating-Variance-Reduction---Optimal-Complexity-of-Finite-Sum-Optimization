"""Figure E2 of the appendix of explanatory figures of the paper (paper/figures/figE2_nfl_plane): Theorem 1 ("no free
lunch, for every norm") in the plane. Explanatory figure; it adds no result to the paper.

The toy is an exact instance of the hypotheses of Theorem 1 of paper/paper_saga_to_muon.tex, with K = 1 and d = 2
(the parameter W is a row vector of the plane, so that every norm on R^{K x d} can be drawn):
    F(W) = (1/n) sum_i log(1 + exp(-y_i W x_i)) + (lam/2) |W|^2,   n = 40,  lam = 0.02,
with features x_i whose second-moment matrix (1/n) sum_i x_i x_i^T has the eigenvalues 32 lam and 5 lam (drawn once
with a fixed seed, then whitened and coloured so that this holds exactly). Then
    D^2 F(W) = (1/n) sum_i s_i(W) x_i x_i^T + lam I,   s_i = sigma'(y_i W x_i) in (0, 1/4],
is largest at W = 0 in every direction (s_i = 1/4 there): H0 = D^2 F(0) has the eigenvalues 9 lam and 2.25 lam, and
D^2 F(t W_1) -> lam I as t -> infinity for a generic W_1. Hence L_F = 9 lam, mu_F = lam and kappa_F = 9.

Panels:
  (a) curvature in a fixed direction, D^2 F(t W_1)[u, u] / lam, along the ray t W_1 (W_1 at RAY_DEG degrees), for the
      two eigenvectors u_1 (largest) and u_2 (smallest) of H0: logistic loss (solid) and, for comparison, the
      quadratic loss l(z) = z^2 / 8, whose Hessian is H0 everywhere (dashed). In the direction u_1 alone the curvature
      of the logistic model varies by the factor kappa_F, which no norm can change: for every norm N and direction u,
      L_N >= D^2F(W)[u,u] / N(u)^2 and mu_N <= D^2F(W')[u,u] / N(u)^2, so kappa_N >= the ratio of two curvatures in
      the same direction. Taking u = u_1 gives kappa_N >= 9 lam / lam = kappa_F, the consequence kappa_N >= kappa_F
      of Theorem 1 by a per-direction route (the proof in the paper computes mu_N = lam / c_N^2 with the direction
      that maximizes N / |.|, and L_N >= L_F / c_N^2).
  (b) the same argument in the plane of the directions Delta (unit: 1/sqrt(lam)): the set E_0 = {H0[Delta,Delta] <= 1}
      (where the largest curvature is at most 1) and the set E_inf = {lam |Delta|^2 <= 1} (where the floor is at most
      1). For a norm N with unit ball B: L_N = 1/r_in^2 with r_in B the largest copy of B inside E_0 (the supremum of
      the curvature is attained at W = 0), and mu_N = 1/r_out^2 with r_out B the smallest copy of B that contains
      E_inf (Theorem 1: mu_N = lam / c_N^2), so kappa_N = (r_out / r_in)^2. Drawn for the Euclidean ball and for the
      ball matched to H0 (N(Delta)^2 = H0[Delta,Delta]): on the line of u_1 the inner copy cannot pass the end of
      the short axis of E_0 and the outer copy must reach the circle; the two points are at distances in the ratio
      sqrt(kappa_F) = 3, so r_out / r_in >= 3 whatever the ball. The two balls drawn pass exactly through both points
      (kappa_N = kappa_F); a ball with kappa_N > kappa_F stops short of the inner point or goes beyond the outer one.
  (c) condition numbers of F in six norms (Euclidean, which for K = 1 is also the spectral and the nuclear norm,
      l_infinity, l_1, matched to H0, matched to the Hessian at the minimizer, and the best of a scan of 10,801
      ellipsoidal norms, distinct up to a factor): global, L_N / mu_N (filled; all >= kappa_F), and at the
      minimizer W_*, max / min of D^2F(W_*)[Delta,Delta] over the unit sphere of N (open; a norm can improve it).
      The open marks of l_infinity and l_1 coincide exactly: in the plane, the condition number of a fixed H in l_1
      equals that of H^{-1} in l_infinity (dual norms), and H^{-1} is H rotated by 90 degrees up to a factor.

Every number is computed in two ways and the script stops if they disagree: L_N and mu_N in closed form (the formulas
above) and by brute force (maximum and minimum of the curvature over the unit sphere of N, sampled at SPHERE points,
and over W = 0 and the points of the ray; the minimum also over the limit lam I). That the supremum over W is attained
at W = 0 is checked separately on a grid of 142 x 142 points W of the plane (D^2F(0) - D^2F(W) positive
semidefinite, D^2F(W) >= lam I). The script prints every number it plots.

Status. Proved: kappa_N >= kappa_F for every norm (Theorem 1); the figure checks it on one instance and five norms
plus a scan, and shows why. The numbers are those of this toy and are not quoted in the text of the paper.

Run:    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/scripts/make_figE2_nfl_plane.py
Output: paper/figures/figE2_nfl_plane.pdf and .png
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")   # fixed CreationDate: the PDF is byte-identical from run to run

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import explain_style as es  # noqa: E402

NAME = "figE2_nfl_plane"
N_PTS, LAM, SEED = 40, 0.02, 3
MOMENT_EIGS = (32.0 * LAM, 5.0 * LAM)        # eigenvalues of (1/n) sum x x^T: H0 = moment/4 + lam I has (9, 2.25) lam
MOMENT_DEG = 25.0                            # orientation of the top eigenvector of the second-moment matrix
RAY_DEG = 11.0                               # direction W_1 of the ray of panel (a): among the whole degrees, the one
#                                              that keeps every x_i farthest from orthogonal to W_1 (see print_all)
T_GRID = np.logspace(-1.0, 2.0, 241)         # positions t along the ray
SPHERE = 40001                               # points of the unit sphere of a norm (brute-force check)
SCAN_ANGLES = np.arange(0.0, 180.0, 1.0)     # scan of the ellipsoidal norms N(D)^2 = D^T P D, P = Rot diag(1, a) Rot^T:
SCAN_LOG_ASPECT = np.linspace(0.05, 3.0, 60)  # a = 10^0.05 .. 10^3, plus P = I. Kappa_P does not change when P is scaled,
#                                              and (phi, a) and (phi + 90, 1/a) give the same norm up to a factor: this
#                                              grid lists each of the norms with a in [1e-3, 1e3] on a 1-degree, 0.05-decade
#                                              grid exactly once (180 x 60 + 1 = 10,801 norms)
W_GRID = np.concatenate([np.linspace(-3.0, 3.0, 61), np.linspace(-40.0, 40.0, 81)])   # points W of the plane (check)


# ----------------------------------------------------------------------------------------------------------------------
# the toy
# ----------------------------------------------------------------------------------------------------------------------
def rot(deg):
    t = np.deg2rad(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def instance():
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((N_PTS, 2))
    X -= X.mean(axis=0)
    X = X @ np.linalg.inv(np.linalg.cholesky(X.T @ X / N_PTS)).T          # (1/n) X^T X = I exactly
    R = rot(MOMENT_DEG)
    X = X @ (R @ np.diag(np.sqrt(MOMENT_EIGS)) @ R.T)                     # (1/n) X^T X = R diag(eigs) R^T
    y = rng.choice([-1.0, 1.0], size=N_PTS)
    return X * y[:, None]                                                 # rows y_i x_i


def hessian(Xs, w):
    z = Xs @ w
    s = 0.25 / np.cosh(0.5 * z) ** 2                                      # sigma'(z), without overflow
    return (Xs * s[:, None]).T @ Xs / N_PTS + LAM * np.eye(2)


def minimizer(Xs):
    w = np.zeros(2)
    for _ in range(60):
        z = Xs @ w
        g = -(Xs * (0.5 * (1.0 - np.tanh(0.5 * z)))[:, None]).mean(axis=0) + LAM * w
        w = w - np.linalg.solve(hessian(Xs, w), g)
    z = Xs @ w
    g = -(Xs * (0.5 * (1.0 - np.tanh(0.5 * z)))[:, None]).mean(axis=0) + LAM * w
    return w, float(np.linalg.norm(g))


def quad(H, D):
    return np.einsum("ij,jk,ik->i", D, H, D)


def unit_sphere(norm):
    """Points of the unit sphere of a norm, and the constant c_N^2 = max of N^2 over the Euclidean unit sphere."""
    ang = np.linspace(0.0, 2.0 * np.pi, SPHERE)
    U = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    nu = norm(U)
    return U / nu[:, None], float(nu.max() ** 2)


def norms(H0, Hs):
    def pnorm(P):
        return lambda D: np.sqrt(quad(P, D))
    return [   # key, row label, colour slot, norm, P (None unless ellipsoidal)
        ("l2", "Euclidean", "INK", lambda D: np.linalg.norm(D, axis=1), np.eye(2)),
        ("linf", r"$\ell_\infty$ (sign descent)", "ORANGE", lambda D: np.abs(D).max(axis=1), None),
        ("l1", r"$\ell_1$", "AQUA", lambda D: np.abs(D).sum(axis=1), None),
        ("H0", r"matched to $D^2F(0)$", "BLUE", pnorm(H0), H0),
        ("Hs", r"matched to $D^2F(W_\ast)$", "BLUE", pnorm(Hs), Hs),
    ]


def closed_form(key, P, H0):
    """(L_N, mu_N) from the formulas of the docstring."""
    if key == "linf":
        corners = np.array([[1.0, 1.0], [1.0, -1.0]])
        return float(quad(H0, corners).max()), LAM / 1.0
    if key == "l1":
        return float(max(H0[0, 0], H0[1, 1])), LAM / 2.0
    Li = np.linalg.inv(np.linalg.cholesky(P))
    return float(np.linalg.eigvalsh(Li @ H0 @ Li.T)[-1]), LAM / float(np.linalg.eigvalsh(P)[-1])


def data():
    Xs = instance()
    M = Xs.T @ Xs / N_PTS
    H0 = hessian(Xs, np.zeros(2))
    ev, V = np.linalg.eigh(H0)
    u1, u2 = V[:, 1], V[:, 0]                                            # largest, smallest curvature at 0
    kappa_F = ev[1] / LAM
    w1 = rot(RAY_DEG)[:, 0]
    smallest = {deg: float(np.abs(Xs @ rot(deg)[:, 0]).min()) for deg in range(180)}
    if max(smallest, key=smallest.get) != int(RAY_DEG):
        raise SystemExit("RAY_DEG is not the direction that keeps the x_i farthest from orthogonal to the ray")
    ray = [hessian(Xs, t * w1) for t in T_GRID]
    # the supremum of the curvature in every direction is attained at W = 0: D^2F(0) - D^2F(W) is positive semidefinite
    # at every point W of a grid of the plane (s_i(W) <= 1/4), and D^2F(W) >= lam I there
    grid = np.array([[a, b] for a in W_GRID for b in W_GRID])
    gap = min(float(np.linalg.eigvalsh(H0 - hessian(Xs, w))[0]) for w in grid)
    floor = min(float(np.linalg.eigvalsh(hessian(Xs, w))[0]) for w in grid) / LAM
    if gap < -1e-12 or floor < 1.0 - 1e-12:
        raise SystemExit(f"the curvature exceeds D^2F(0) ({gap}) or falls below the floor ({floor}) somewhere")
    curv = {"u1": np.array([u1 @ H @ u1 for H in ray]) / LAM, "u2": np.array([u2 @ H @ u2 for H in ray]) / LAM}
    ws, gnorm = minimizer(Xs)
    Hs = hessian(Xs, ws)
    rows = []
    for key, label, colour, norm, P in norms(H0, Hs):
        S, c2 = unit_sphere(norm)
        L_cf, mu_cf = closed_form(key, P, H0)
        L_bf = max(float(quad(H, S).max()) for H in [H0] + ray)           # sup over the sphere and the sampled points
        mu_bf = min(float(quad(H, S).min()) for H in ray + [LAM * np.eye(2)])   # the last one: the limit t -> infinity
        if abs(L_bf / L_cf - 1.0) > 1e-6 or abs(mu_bf / mu_cf - 1.0) > 1e-6 or abs(LAM / c2 / mu_cf - 1.0) > 1e-6:
            raise SystemExit(f"{key}: closed form and brute force disagree: L {L_cf} / {L_bf}, mu {mu_cf} / {mu_bf}")
        qs = quad(Hs, S)
        rows.append({"key": key, "label": label, "colour": colour, "L": L_cf, "mu": mu_cf, "kappa": L_cf / mu_cf,
                     "local": float(qs.max() / qs.min())})
    best, scan_size = closed_form("P", np.eye(2), H0), 1
    best = best[0] / best[1]
    for phi in SCAN_ANGLES:
        R = rot(phi)
        for la in SCAN_LOG_ASPECT:
            P = R @ np.diag([1.0, 10.0 ** la]) @ R.T
            L, mu = closed_form("P", P, H0)
            best, scan_size = min(best, L / mu), scan_size + 1
    return {"Xs": Xs, "moment_eigs": np.linalg.eigvalsh(M), "H0": H0, "eigs": ev, "u1": u1, "u2": u2,
            "kappa_F": float(kappa_F), "curv": curv, "ws": ws, "gnorm": gnorm, "Hs": Hs,
            "Hs_eigs": np.linalg.eigvalsh(Hs), "rows": rows, "scan_best": float(best),
            "scan_size": scan_size, "grid_size": len(grid), "grid_gap": gap, "grid_floor": floor, "ray_margin": smallest[int(RAY_DEG)],
            "limit": [float(np.linalg.eigvalsh(ray[-1])[k] / LAM) for k in (0, 1)]}


def print_all(D):
    print(f"instance: n = {N_PTS}, lam = {LAM}, eigenvalues of the second-moment matrix / lam: "
          f"{D['moment_eigs'][1] / LAM:.4f}, {D['moment_eigs'][0] / LAM:.4f}")
    print(f"H0 = D^2F(0): eigenvalues / lam {D['eigs'][1] / LAM:.4f}, {D['eigs'][0] / LAM:.4f}; "
          f"kappa_F = L_F / lam = {D['kappa_F']:.4f}, sqrt = {np.sqrt(D['kappa_F']):.4f}")
    print(f"grid of {D['grid_size']} points W of the plane: smallest eigenvalue of D^2F(0) - D^2F(W) = "
          f"{D['grid_gap']:.2e} (>= 0: the supremum is at W = 0), smallest eigenvalue of D^2F(W) / lam = "
          f"{D['grid_floor']:.6f} (>= 1)")
    print(f"(a) ray at {RAY_DEG:g} deg: smallest |x_i . W_1| = {D['ray_margin']:.4f} (no x_i is orthogonal to it)")
    print(f"    curvature / lam along the ray, t = {T_GRID[0]:g} .. {T_GRID[-1]:g}: "
          f"u1 {D['curv']['u1'][0]:.3f} -> {D['curv']['u1'][-1]:.6f}; u2 {D['curv']['u2'][0]:.3f} -> "
          f"{D['curv']['u2'][-1]:.6f}; eigenvalues of D^2F / lam at the last point: {D['limit'][0]:.6f}, "
          f"{D['limit'][1]:.6f}")
    print(f"minimizer W* = ({D['ws'][0]:+.4f}, {D['ws'][1]:+.4f}), |grad| = {D['gnorm']:.1e}; eigenvalues of "
          f"D^2F(W*) / lam: {D['Hs_eigs'][1] / LAM:.4f}, {D['Hs_eigs'][0] / LAM:.4f}")
    print("(c) norm: L_N / lam, mu_N / lam, global kappa_N (ratio to kappa_F), condition number at W*")
    for r in D["rows"]:
        print(f"    {r['key']:5s} {r['L'] / LAM:9.4f} {r['mu'] / LAM:8.4f} {r['kappa']:8.3f} "
              f"({r['kappa'] / D['kappa_F']:.3f}) {r['local']:8.3f}")
    print(f"    scan of {D['scan_size']} ellipsoidal norms: smallest global kappa_P = {D['scan_best']:.6f} "
          f"({D['scan_best'] / D['kappa_F']:.6f} kappa_F)")


# ----------------------------------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------------------------------
def ellipse(H, scale=1.0, m=721):
    """Boundary of scale * {Delta: Delta^T H Delta <= 1}."""
    ev, V = np.linalg.eigh(H)
    ang = np.linspace(0.0, 2.0 * np.pi, m)
    return scale * (np.stack([np.cos(ang), np.sin(ang)], axis=1) / np.sqrt(ev)) @ V.T


def panel_a(ax, D, ps):
    ink = ps.INK
    quad_level = {"u1": D["eigs"][1] / LAM, "u2": D["eigs"][0] / LAM}
    for k in ("u1", "u2"):
        ax.plot(T_GRID, np.full_like(T_GRID, quad_level[k]), color=ink["muted"], lw=1.3, ls=(0, (4.0, 2.0)), zorder=2,
                label="quadratic loss" if k == "u1" else None)
        ax.plot(T_GRID, D["curv"][k], color=ink["primary"], lw=1.8, zorder=3,
                label="logistic loss" if k == "u1" else None)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(T_GRID[0], T_GRID[-1])
    ax.set_ylim(0.8, 40.0)
    ax.set_yticks([1, 2.25, 9])
    ax.set_yticklabels(["1", "2.25", r"$\kappa_F = 9$"])
    ax.minorticks_off()
    ax.set_xticks([0.1, 1, 10, 100])
    ax.set_xticklabels(["0.1", "1", "10", "100"])
    ax.set_xlabel(r"position $t$ on the ray $tW_1$")
    ax.set_ylabel(r"curvature $D^2F(tW_1)[u,u]\,/\,\lambda$")
    kw = dict(fontsize=es.small(), color=ink["secondary"], va="bottom", textcoords="offset points")
    # labels on the left, where the logistic curves still sit on their quadratic levels (on the right the u_1 curve
    # crosses the level of u_2)
    ax.annotate(r"$u = u_1$", (0.11, quad_level["u1"]), xytext=(0, 2), ha="left", **kw)
    ax.annotate(r"$u = u_2$", (0.11, quad_level["u2"]), xytext=(0, 2), ha="left", **kw)
    ax.annotate(r"floor $\lambda$", (0.11, 1.0), xytext=(0, 2), ha="left", **kw)
    ps.legend(ax, loc="upper right", handlelength=1.7, borderaxespad=0.1, labelspacing=0.25)
    es.panel_title(ax, "(a) along a ray")


def panel_b(ax, D, ps):
    ink = ps.INK
    H0n = D["H0"] / LAM                                # unit of the picture: 1/sqrt(lam) (the floor circle has radius 1):
    #                                                    E_0 = {Delta^T H0n Delta <= 1} in that unit
    root = np.sqrt(D["kappa_F"])
    lim = 1.30
    es.diagram_axes(ax, (-lim, lim), (-lim, lim), equal=True)
    ang = np.linspace(0.0, 2.0 * np.pi, 721)
    circle = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    e0 = ellipse(H0n)
    # the two reference sets
    ax.fill(e0[:, 0], e0[:, 1], color=ink["grid"], lw=0, zorder=1)
    ax.plot(circle[:, 0], circle[:, 1], color=ink["axis"], lw=3.2, zorder=1, solid_capstyle="round")
    # Euclidean ball: inner copy of radius 1/sqrt(kappa_F), outer copy = the floor circle
    EU, MA = ink["secondary"], ps.BLUE
    ax.plot(circle[:, 0] / root, circle[:, 1] / root, color=EU, lw=1.3, zorder=4)
    ax.plot(circle[:, 0], circle[:, 1], color=EU, lw=1.3, ls=(0, (3.2, 1.8)), zorder=4)
    # ball matched to H0: inner copy = E_0, outer copy = sqrt(kappa_F) E_0
    ax.plot(e0[:, 0], e0[:, 1], color=MA, lw=1.3, zorder=5)
    big = ellipse(H0n, root)
    ax.plot(big[:, 0], big[:, 1], color=MA, lw=1.3, ls=(0, (3.2, 1.8)), zorder=5, clip_on=True)
    # the line of u_1 and the two pinned points
    u1 = D["u1"] * np.sign(D["u1"][0])
    p_in, p_out = u1 / root, u1
    ax.plot([0.0, p_out[0]], [0.0, p_out[1]], color=ink["primary"], lw=0.9, zorder=6)
    ax.plot([0.0], [0.0], ls="", marker="o", ms=2.6, color=ink["primary"], zorder=7)
    for p in (p_in, p_out):
        ax.plot([p[0]], [p[1]], ls="", marker="o", ms=5.0, mfc=ink["primary"], mec=ink["surface"], mew=0.9, zorder=8)
    mid = 0.5 * (p_in + p_out)
    nrm = np.array([-u1[1], u1[0]])
    ax.text(*(mid + 0.15 * nrm), rf"$\times\,{root:.0f}$", fontsize=es.small(), color=ink["primary"], ha="center",
            va="center", rotation=np.degrees(np.arctan2(u1[1], u1[0])), zorder=9)
    ax.text(*(1.17 * p_out), r"$u_1$", fontsize=es.small(), color=ink["secondary"], ha="center", va="center",
            zorder=9)
    L2D = ps.plt.Line2D
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.patches import Patch
    dash = (0, (3.2, 1.8))
    handles = [Patch(fc=ink["grid"], ec="none"), L2D([], [], color=ink["axis"], lw=3.2),
               (L2D([], [], color=EU, lw=1.3), L2D([], [], color=EU, lw=1.3, ls=dash)),
               (L2D([], [], color=MA, lw=1.3), L2D([], [], color=MA, lw=1.3, ls=dash)),
               L2D([], [], ls="", marker="o", ms=5.0, mfc=ink["primary"], mec=ink["surface"], mew=0.9)]
    labels = [r"curvature at $W = 0$ is $\leq 1$", r"floor $\lambda\,|\Delta|^2$ is $\leq 1$",
              "Euclidean ball: inside, around", "matched ball: inside, around",
              rf"pinned on $u_1$: ratio $\sqrt{{\kappa_F}} = {root:.0f}$"]
    leg = ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=1, frameon=False,
                    handlelength=2.4, labelspacing=0.2, borderaxespad=0.0, handletextpad=0.5,
                    handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)})
    for t in leg.get_texts():
        t.set_color(ink["secondary"])
    ax.set_anchor("N")
    es.panel_title(ax, "(b) why no ball helps")


def panel_c(ax, D, ps):
    ink = ps.INK
    colour = {"INK": ink["secondary"], "ORANGE": ps.ORANGE, "AQUA": ps.AQUA, "BLUE": ps.BLUE}
    rows = D["rows"]
    n = len(rows)
    ax.axvline(D["kappa_F"], color=ink["primary"], lw=0.9, zorder=1)
    for k, r in enumerate(rows):
        y = n - k
        col = colour[r["colour"]]
        ax.plot([r["local"], r["kappa"]], [y, y], color=col, lw=1.0, zorder=2)
        ax.plot([r["local"]], [y], ls="", marker="o", ms=5.2, mfc=ink["surface"], mec=col, mew=1.2, zorder=3)
        ax.plot([r["kappa"]], [y], ls="", marker="o", ms=5.6, mfc=col, mec=ink["surface"], mew=0.9, zorder=4)
    ax.plot([D["scan_best"]], [0], ls="", marker="o", ms=5.6, mfc=ps.BLUE, mec=ink["surface"], mew=0.9, zorder=4)
    ax.set_yticks(list(range(n, 0, -1)) + [0])
    ax.set_yticklabels([r["label"] for r in rows] + [f"best of {D['scan_size']:,}\nellipsoidal norms"],
                       fontsize=es.small())
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.75, n + 1.9)
    ax.set_xscale("log")
    ax.set_xlim(0.8, 30.0)
    ax.set_xticks([1, 3, 9])
    ax.set_xticklabels(["1", "3", r"$\kappa_F = 9$"])
    ax.minorticks_off()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(r"condition number $\kappa_N$ of $F$")
    L2D = ps.plt.Line2D
    handles = [L2D([], [], ls="", marker="o", ms=5.6, mfc=ink["secondary"], mec=ink["surface"], mew=0.9),
               L2D([], [], ls="", marker="o", ms=5.2, mfc=ink["surface"], mec=ink["secondary"], mew=1.2)]
    ps.legend(ax, handles=handles, labels=["global (certified)", r"at the minimizer $W_\ast$"], loc="upper right",
              handletextpad=0.2, borderaxespad=0.1, labelspacing=0.25, borderpad=0.1)
    es.panel_title(ax, "(c) five norms and a scan")


def draw(D):
    ps = es.setup("paper")
    fig = es.figure("paper", height=2.75)
    fig.get_layout_engine().set(w_pad=0.03, wspace=0.02)
    gs = fig.add_gridspec(1, 3, width_ratios=(1.55, 2.05, 2.75))
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, k]) for k in range(3))
    panel_a(ax_a, D, ps)
    panel_b(ax_b, D, ps)
    panel_c(ax_c, D, ps)
    es.save(fig, NAME, "paper")


def main():
    D = data()
    print_all(D)
    draw(D)


if __name__ == "__main__":
    main()
