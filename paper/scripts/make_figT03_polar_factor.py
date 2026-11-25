"""Figure T03 of the lecture (paper/talk/figures/figT03_polar_factor): the polar factor keeps the singular directions
of a matrix and sets every singular value to one; a step along it has a fixed length whatever the size of the
gradient, it is the steepest-descent direction for the spectral norm, and Muon approximates it by five Newton-Schulz
steps. Explanatory figure, not in the paper. It introduces Section 2.1 ("Matrix problems") of
paper/paper_saga_to_muon.tex: polar(omega) = U V^T for omega = U diag(s) V^T, the polar step W - eta polar(omega),
and the map of Muon named in the introduction.

Everything is computed; nothing is drawn by hand.
  (a) what the map does to a matrix. G = U diag(S1, S2) V^T, a 2 x 2 matrix with U and V the rotations of angles
      U_DEG and V_DEG. The image of the unit circle by G is an ellipse with half-axes s_1 u_1 and s_2 u_2; by
      polar(G) = U V^T (muon/lab/polar.py, polar_svd) it is the unit circle: the same rotation, every stretch set to
      one; by G / |G|_F (the normalized matrix, the control map "frob" of the paper) it is the same ellipse, shrunk.
      The dots are the images of the right singular vectors v_1, v_2 by G and by polar(G): on the rays of u_1, u_2.
  (b) what the maps do to a singular value (logarithmic axes). One 8 x 40 matrix omega = diag(2^-(k-1)) Z, Z with
      i.i.d. standard Gaussian entries (seed OMEGA_SEED): the rows are scaled so that the singular values cover two
      decades. Abscissa: x = s / |omega|_F for a singular value s of the input; ordinate: the singular value of the
      output in the same pair of singular vectors: 1 for the exact polar factor, x for the normalized matrix, p^5(x)
      for ns5 with p(x) = 3.4445 x - 4.7750 x^3 + 2.0315 x^5 (Muon's five Newton-Schulz steps;
      polar.ns_scalar_map). The dots are the eight singular values of omega. Those of ns5(omega) are computed twice:
      as p^5(x_k) and as the singular values of the matrix polar.ns5(omega) (normalization constant eps = 1e-7, the
      default); the script stops if the two sets differ. ns5 does NOT return U V^T: p^5 is linear near zero (slope
      3.4445^5 = 484.9) and then oscillates in a band around one, [0.68, 1.20] for every x above about 0.0015 (the
      band and its start are computed, printed and drawn as a tint).
  (c) steepest descent for the spectral norm. For the matrix G of (a), steps Delta = U diag(d_1, d_2) V^T in the
      singular basis of G: <G, Delta> = s_1 d_1 + s_2 d_2. The unit balls of the Frobenius norm (disc), of the
      spectral norm (square) and of the nuclear norm (diamond) in the plane (d_1, d_2), and the maximizer of
      <G, Delta> on each: G / |G|_F (value |G|_F), polar(G) at the corner (1, 1) (value |G|_* = s_1 + s_2), the top
      singular pair u_1 v_1^T (value |G|_op = s_1). The steepest-descent step of length eta for a norm is -eta times
      its maximizer (the polar step W - eta polar(G) for the spectral norm), as the third line under the balls says.
      The axes d_1, d_2 are drawn over the fills of the balls. The restriction to this plane loses nothing for the maximizers:
      by von Neumann's trace inequality <G, Delta> <= sum_k s_k(G) s_k(Delta), with equality when Delta has the
      singular vectors of G, and the three norms depend on the singular values of Delta only. Checked by brute force:
      over N_RANDOM random 2 x 2 matrices scaled to the unit sphere of each norm, <G, Delta> never exceeds the dual
      norm and comes within a few percent of it; the three maximizers reach it to rounding (the script stops
      otherwise).

Status. Proved (linear algebra, classical): everything in (a) and (c), and the scalar maps of (b) (Section 2.1 of the
paper; muon/notes_theory.md and the docstring of muon/lab/polar.py). The numbers are those of these two matrices. The
figure says nothing about convergence speed: the paper shows that no norm repairs the logistic worst case (Theorem 1)
and that at batch one Euclidean devices reproduce the gain of orthogonalization.

Colours: the polar factor in orange (exact: solid; ns5: dashed), the normalized matrix (a control map) in grey, the
matrix itself in ink; in (c) the Frobenius ball in grey, the spectral ball in orange, the nuclear ball in aqua.

Progressive build (states s1-s3): s1 panel (a); s2 adds (b); s3 adds (c).

Helpers not in explain_style (written here): axes_in and window (the three panels are placed by hand, in inches, and
the two diagrams fill their axes with equal scales), leader (a label with a hairline to its mark).

The script prints every number it plots. Nothing is read from the results of the project.
Run:    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/scripts/make_figT03_polar_factor.py
Output: paper/talk/figures/figT03_polar_factor.pdf and .png, and the states _s1 ... _s3
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "muon"))
import explain_style as es  # noqa: E402
from lab import polar  # noqa: E402

NAME = "figT03_polar_factor"
S1, S2 = 1.6, 0.8                    # singular values of the 2 x 2 matrix G
U_DEG, V_DEG = 20.0, -25.0           # G = rot(U_DEG) diag(S1, S2) rot(V_DEG)^T
OMEGA_SHAPE, OMEGA_SEED = (8, 40), 0  # panel (b): omega = diag(ROW_DECAY^(k-1)) Z, Z standard Gaussian
ROW_DECAY = 0.5
X_LO = 1e-4                          # left end of panel (b)
N_RANDOM = 200000                    # random matrices of the brute-force check of panel (c)
CHECK_SEED = 1


# ----------------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------------
def rot(deg):
    t = np.deg2rad(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def norms(D):
    """Frobenius, spectral and nuclear norms of a stack of 2 x 2 matrices (N x 2 x 2)."""
    s = np.linalg.svd(D, compute_uv=False)
    return {"fro": np.sqrt((s ** 2).sum(axis=-1)), "op": s.max(axis=-1), "nuc": s.sum(axis=-1)}


def data():
    # (a) the 2 x 2 matrix
    U, V = rot(U_DEG), rot(V_DEG)
    G = U @ np.diag([S1, S2]) @ V.T
    s = polar.singular_values(G)
    if not np.allclose(s, [S1, S2], rtol=1e-12):
        raise SystemExit(f"singular values of G: {s}")
    P = polar.polar_svd(G)
    if not np.allclose(P, U @ V.T, atol=1e-12):
        raise SystemExit("polar_svd(G) is not U V^T")
    fro = float(np.linalg.norm(G))
    Gn = polar.frob_normalize(G)
    maps = {"G": G, "polar": P, "frob": Gn}
    images = {k: np.stack([M @ V[:, 0], M @ V[:, 1]]) for k, M in maps.items()}        # images of v_1, v_2
    ns5_G = polar.orth_singular_values(G, "ns5")

    # (b) the Gaussian matrix
    Z = np.random.default_rng(OMEGA_SEED).standard_normal(OMEGA_SHAPE)
    omega = (ROW_DECAY ** np.arange(OMEGA_SHAPE[0]))[:, None] * Z
    sv = polar.singular_values(omega)
    nrm = float(np.linalg.norm(omega))
    x = sv / nrm
    sv_polar = polar.singular_values(polar.polar_svd(omega))
    sv_frob = polar.singular_values(polar.frob_normalize(omega))
    ns_scalar = polar.ns_scalar_map(sv / (nrm + 1e-7))                 # eps = 1e-7: the default of polar.ns5
    ns_matrix = polar.orth_singular_values(omega, "ns5")
    if not np.allclose(np.sort(ns_scalar), np.sort(ns_matrix), rtol=1e-9):
        raise SystemExit("ns5: the scalar map and the singular values of the matrix ns5(omega) disagree")
    if not np.allclose(sv_polar, 1.0, atol=1e-12) or not np.allclose(sv_frob, x, rtol=1e-12):
        raise SystemExit("singular values of the polar factor or of the normalized matrix are not as stated")
    grid = np.logspace(np.log10(X_LO), 0.0, 40001)
    curve = polar.ns_scalar_map(grid)
    peak = int(np.argmax(np.diff(curve) < 0))                           # end of the increasing ramp of p^5
    lo = float(curve[peak:].min())                                      # floor of the band after the ramp
    first = int(np.argmax(curve >= lo))                                 # the ramp enters the band here, for good
    if first > peak or curve[first:].min() < lo or np.any(np.diff(curve[:peak + 1]) <= 0):
        raise SystemExit("the band of p^5 is not as stated")
    band = (lo, float(curve.max()), float(grid[first]))
    coeffs = polar.NS5_COEFFS
    if abs(curve[0] / grid[0] / float(coeffs[0]) ** 5 - 1.0) > 1e-2:
        raise SystemExit("p^5 is not linear with slope a^5 at the left end of the panel")

    # (c) maximizers of <G, Delta> on the three unit balls, and the brute-force check
    rng = np.random.default_rng(CHECK_SEED)
    R = rng.standard_normal((N_RANDOM, 2, 2))
    nr = norms(R)
    dual = {"fro": fro, "op": S1 + S2, "nuc": S1}                       # dual norms of G: |G|_F, |G|_*, |G|_op
    best = {"fro": Gn, "op": P, "nuc": np.outer(U[:, 0], V[:, 0])}
    coords = {"fro": np.array([S1, S2]) / fro, "op": np.array([1.0, 1.0]), "nuc": np.array([1.0, 0.0])}
    check = {}
    for k in ("fro", "op", "nuc"):
        val = np.einsum("ij,nij->n", G, R) / nr[k]                      # <G, Delta> on the unit sphere of the norm
        own = float(norms(best[k][None])[k][0])
        reached = float(np.vdot(G, best[k]))
        in_plane = U.T @ best[k] @ V                                    # coordinates in the singular basis of G
        if (val.max() > dual[k] * (1 + 1e-12) or abs(own - 1.0) > 1e-12 or abs(reached / dual[k] - 1.0) > 1e-12
                or not np.allclose(in_plane, np.diag(coords[k]), atol=1e-12)):
            raise SystemExit(f"panel (c), {k}: the maximizer or the dual norm is not as stated")
        check[k] = {"random_max": float(val.max()), "dual": dual[k], "reached": reached}
    return {"U": U, "V": V, "G": G, "fro": fro, "maps": maps, "images": images, "ns5_G": ns5_G,
            "omega_sv": sv, "omega_fro": nrm, "x": x, "ns_dots": polar.ns_scalar_map(x), "ns_matrix": ns_matrix,
            "grid": grid, "curve": curve, "band": band,
            "slope0": float(coeffs[0]) ** 5, "coeffs": coeffs, "coords": coords, "check": check}


def print_all(D):
    G = D["G"]
    print(f"(a) G = rot({U_DEG:g} deg) diag({S1:g}, {S2:g}) rot({V_DEG:g} deg)^T = [[{G[0, 0]:+.4f}, {G[0, 1]:+.4f}], "
          f"[{G[1, 0]:+.4f}, {G[1, 1]:+.4f}]]; |G|_F = {D['fro']:.4f}, |G|_* = {S1 + S2:.4f}, |G|_op = {S1:.4f}")
    for k, name in (("G", "G"), ("polar", "polar(G)"), ("frob", "G/|G|_F")):
        im = D["images"][k]
        print(f"    {name:9s}: singular values {', '.join(f'{v:.4f}' for v in polar.singular_values(D['maps'][k]))}; "
              f"images of v_1, v_2: ({im[0, 0]:+.4f}, {im[0, 1]:+.4f}), ({im[1, 0]:+.4f}, {im[1, 1]:+.4f})")
    print(f"    (not drawn) ns5(G): singular values {', '.join(f'{v:.4f}' for v in D['ns5_G'])}")
    print(f"(b) omega = diag({ROW_DECAY}^(k-1)) Z, Z {OMEGA_SHAPE[0]} x {OMEGA_SHAPE[1]} standard Gaussian, seed "
          f"{OMEGA_SEED}; |omega|_F = {D['omega_fro']:.4f}")
    print("    singular values s_k:        " + ", ".join(f"{v:.3f}" for v in D["omega_sv"]))
    print("    x_k = s_k / |omega|_F:      " + ", ".join(f"{v:.4f}" for v in D["x"]))
    print("    polar factor:               1 for every k")
    print("    ns5, p^5(x_k):              " + ", ".join(f"{v:.4f}" for v in D["ns_dots"]))
    print("    ns5, matrix (sorted):       " + ", ".join(f"{v:.4f}" for v in D["ns_matrix"]))
    a, b, c = D["coeffs"]
    print(f"    p(x) = {a:.4f} x {b:+.4f} x^3 {c:+.4f} x^5; slope of p^5 at 0: a^5 = {D['slope0']:.1f} (p^5(x)/x at "
          f"x = {X_LO:g}: {D['curve'][0] / D['grid'][0]:.1f}); p^5(1) = {D['curve'][-1]:.4f}")
    print(f"    band of p^5: [{D['band'][0]:.4f}, {D['band'][1]:.4f}] for every x in [{D['band'][2]:.5f}, 1] (largest "
          f"value at x = {D['grid'][int(np.argmax(D['curve']))]:.5f})")
    print(f"(c) maximizers of <G, Delta> = s_1 d_1 + s_2 d_2 on the unit balls, coordinates (d_1, d_2); brute force over "
          f"{N_RANDOM} random matrices on each unit sphere")
    for k, name in (("fro", "Frobenius"), ("op", "spectral"), ("nuc", "nuclear")):
        c_ = D["check"][k]
        print(f"    {name:9s}: maximizer ({D['coords'][k][0]:.4f}, {D['coords'][k][1]:.4f}), value {c_['reached']:.6f} "
              f"= dual norm {c_['dual']:.6f}; largest value over the random matrices {c_['random_max']:.6f}")


# ----------------------------------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------------------------------
def axes_in(fig, left, bottom, width, height):
    """Axes at a position given in inches. The three panels are placed by hand (with two equal-aspect diagrams
    constrained layout leaves a third of the canvas empty); axes made by add_axes are not moved by the layout
    engine of the figure, and the scope line keeps the 0.2 in that explain_style.save reserves at the bottom."""
    W, H = fig.get_figwidth(), fig.get_figheight()
    return fig.add_axes([left / W, bottom / H, width / W, height / H])


def window(ax, x0, x1, y1):
    """Data window of a diagram panel with equal scales that fills its axes: x from x0 to x1, top at y1."""
    bb = ax.get_position()
    fig = ax.figure
    w, h = bb.width * fig.get_figwidth(), bb.height * fig.get_figheight()
    es.diagram_axes(ax, (x0, x1), (y1 - (x1 - x0) * h / w, y1), equal=True)


def leader(ax, text, at, start, target, ps, **kw):
    """A label in ink at `at` and a hairline from `start` (next to the label) to `target` (on the mark)."""
    ax.plot([start[0], target[0]], [start[1], target[1]], color=ps.INK["secondary"], lw=0.7, zorder=6,
            solid_capstyle="round")
    return ax.text(at[0], at[1], text, fontsize=es.small(), color=ps.INK["secondary"], zorder=8, **kw)


def panel_a(ax, D, ps):
    ink = ps.INK
    window(ax, -1.85, 1.85, 1.62)
    ang = np.linspace(0.0, 2.0 * np.pi, 721)
    circle = np.stack([np.cos(ang), np.sin(ang)])                       # 2 x m
    out = {k: M @ circle for k, M in D["maps"].items()}
    u1, u2 = D["U"][:, 0], D["U"][:, 1]
    for u, r in ((u1, S1), (u2, 1.0)):                                  # the rays of u_1 and u_2
        ax.plot([0.0, r * u[0]], [0.0, r * u[1]], color=ink["muted"], lw=0.8, zorder=2)
    ax.fill(out["G"][0], out["G"][1], color=es.tint(ink["muted"], 0.14), lw=0, zorder=1)
    ax.plot(out["G"][0], out["G"][1], color=ink["primary"], lw=1.2, zorder=3)
    ax.plot(out["frob"][0], out["frob"][1], color=ink["secondary"], lw=1.3, ls=(0, (3.5, 1.8)), zorder=4)
    ax.plot(out["polar"][0], out["polar"][1], color=ps.ORANGE, lw=2.0, zorder=5)
    for k, col in (("G", ink["primary"]), ("polar", ps.ORANGE)):
        im = D["images"][k]
        ax.plot(im[:, 0], im[:, 1], ls="", marker="o", ms=5.0, mfc=col, mec=ink["surface"], mew=0.9, zorder=7)
    kw = dict(fontsize=es.small(), color=ink["secondary"], zorder=8, ha="center", va="center")
    ax.text(*(S1 * u1 + 0.56 * u2 + 0.09 * u1), r"$s_1u_1$", **kw)
    ax.text(*(1.0 * u1 - 0.27 * u2 + 0.10 * u1), r"$u_1$", **kw)
    ax.text(*(0.66 * u2 + 0.40 * u1), r"$s_2u_2$", **kw)
    ax.text(*(1.27 * u2), r"$u_2$", **kw)
    # the three outlines
    leader(ax, r"$G$", (-1.83, -1.24), (-1.62, -1.18), D["maps"]["G"] @ circle[:, 420], ps, ha="left", va="center")
    leader(ax, r"$G\,/\,\|G\|_F$", (1.85, -1.24), (1.02, -1.08), D["maps"]["frob"] @ circle[:, 590], ps, ha="right",
           va="center")
    leader(ax, r"$\mathrm{polar}(G) = UV^{\!\top}$", (-0.08, -1.74), (-0.08, -1.58),
           (np.cos(np.deg2rad(-95.0)), np.sin(np.deg2rad(-95.0))), ps, ha="center", va="center")
    es.panel_title(ax, "(a) a circle, mapped")


def panel_b(ax, D, ps):
    ink = ps.INK
    x = D["x"]
    lo, hi, start = D["band"]
    ax.fill_between([start, 1.0], [lo, lo], [hi, hi], color=es.tint(ps.ORANGE, 0.16), lw=0, zorder=1)
    ax.plot(D["grid"], D["grid"], color=ink["secondary"], lw=1.4, zorder=3)
    ax.plot(D["grid"], D["curve"], color=ps.ORANGE, lw=1.3, ls=(0, (3.5, 1.6)), zorder=4)
    ax.plot([X_LO, 1.0], [1.0, 1.0], color=ps.ORANGE, lw=2.0, zorder=5)
    mk = dict(ls="", marker="o", ms=5.0, zorder=7)
    ax.plot(x, x, mfc=ink["secondary"], mec=ink["surface"], mew=0.9, **mk)
    ax.plot(x, np.ones_like(x), mfc=ps.ORANGE, mec=ink["surface"], mew=0.9, **{**mk, "ms": 4.4})
    ax.plot(x, D["ns_dots"], mfc=ink["surface"], mec=ps.ORANGE, mew=1.3, **{**mk, "zorder": 8})
    ax.set_xscale("log")
    ax.set_xlim(X_LO, 1.0)
    ax.set_ylim(0.0, 1.48)
    ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", "0.5", "1"])
    ax.minorticks_off()
    ax.set_xlabel(r"singular value of $\omega$, over $\|\omega\|_F$")       # the dots: those of omega
    ax.set_ylabel("singular value out")
    kw = dict(fontsize=es.small(), color=ink["secondary"], zorder=9, linespacing=1.15)
    ax.text(1.25e-4, 1.26, r"$\mathrm{polar}(\omega)$: exactly 1", ha="left", va="bottom", **kw)
    ax.text(2.6e-3, 0.46, "ns5 (Muon):\n" + f"{lo:.2f} to {hi:.2f}", ha="left", va="center", **kw)
    ax.text(2.2e-3, 0.17, r"$\omega\,/\,\|\omega\|_F$", ha="left", va="center", **kw)
    es.panel_title(ax, "(b) singular values")


def panel_c(ax, D, ps):
    ink = ps.INK
    window(ax, -1.18, 3.05, 1.58)
    ax.plot([-1.16, 1.16], [0, 0], color=ink["axis"], lw=0.8, zorder=2)      # the axes d_1, d_2: over the fills of
    ax.plot([0, 0], [-1.16, 1.16], color=ink["axis"], lw=0.8, zorder=2)      # the balls, under their outlines
    ang = np.linspace(0.0, 2.0 * np.pi, 361)
    shapes = {"fro": (np.cos(ang), np.sin(ang), ink["secondary"]),
              "op": (np.array([1, -1, -1, 1, 1.0]), np.array([1, 1, -1, -1, 1.0]), ps.ORANGE),
              "nuc": (np.array([1, 0, -1, 0, 1.0]), np.array([0, 1, 0, -1, 0.0]), ps.AQUA)}
    for k in ("op", "fro", "nuc"):
        xs, ys, col = shapes[k]
        ax.fill(xs, ys, color=es.tint(col, 0.07), lw=0, zorder=1)
    for k in ("op", "fro", "nuc"):
        xs, ys, col = shapes[k]
        ax.plot(xs, ys, color=col, lw=1.6, zorder=3, solid_joinstyle="miter")
    g = np.array([S1, S2]) / D["fro"]
    es.arrow(ax, (0.0, 0.0), tuple(0.66 * g), colour=ink["primary"], lw=1.3, zorder=5, mutation_scale=9.0)
    t = np.array([-g[1], g[0]])                                          # level lines of <G, .> through the maximizers
    kw = dict(fontsize=es.small(), color=ink["secondary"], zorder=8)
    rows = (("op", "spectral:\n" + r"$\mathrm{polar}(G)$", 1.14),
            ("fro", "Frobenius:\n" + r"$G\,/\,\|G\|_F$", 0.22),
            ("nuc", "nuclear:\n" + r"$u_1v_1^{\!\top}$", -0.70))
    for k, text, y in rows:
        c = D["coords"][k]
        a, b_ = c - 0.30 * t, c + 0.30 * t
        ax.plot([a[0], b_[0]], [a[1], b_[1]], color=ink["primary"], lw=0.8, zorder=4)
        ax.plot([c[0]], [c[1]], ls="", marker="o", ms=5.6, mfc=shapes[k][2], mec=ink["surface"], mew=0.9, zorder=7)
        d = np.array([1.46, y + 0.16]) - c
        leader(ax, text, (1.52, y), (1.46, y + 0.16), c + 0.14 * d / np.linalg.norm(d), ps, ha="left", va="center",
               linespacing=1.15)
    ax.text(0.20, 0.40, r"$G$", ha="center", va="center", **kw)
    ax.text(0.50, -0.06, r"$d_1$", ha="center", va="top", **kw)
    ax.text(-0.07, 0.56, r"$d_2$", ha="right", va="center", **kw)
    ax.text(-1.0, -1.30, r"$\Delta = U\,\mathrm{diag}(d_1, d_2)\,V^{\!\top}$", ha="left", va="center", **kw)
    ax.text(-1.0, -1.63, r"dots: largest $\langle G, \Delta\rangle$ on a ball", ha="left", va="center", **kw)
    ax.text(-1.0, -1.96, r"the step: $-\eta$ times a dot", ha="left", va="center", **kw)
    es.panel_title(ax, "(c) steepest step")


def draw(D):
    import warnings
    warnings.filterwarnings("ignore", message="There are no gridspecs with layoutgrids")   # axes placed by hand
    ps = es.setup("talk")
    fig = es.figure("talk", height=2.6)
    top = 2.28                                                      # top of the three panels, in inches
    ax_a = axes_in(fig, 0.05, top - 1.72, 1.62, 1.72)
    ax_b = axes_in(fig, 2.27, 0.74, 1.66, top - 0.74)
    ax_c = axes_in(fig, 4.02, top - 1.60, 1.86, 1.60)
    panel_a(ax_a, D, ps)
    panel_b(ax_b, D, ps)
    panel_c(ax_c, D, ps)
    b = es.Build()
    b.add(2, ax_b)
    b.add(3, ax_c)
    a, b5, c = D["coeffs"]
    es.save(fig, NAME, "talk", build=b,
            note=(rf"$G$: $s = ({S1:g}, {S2:g})$; $\omega$: ${OMEGA_SHAPE[0]} \times {OMEGA_SHAPE[1]}$, rows scaled; "
                  rf"ns5: Muon's 5 Newton-Schulz steps (${a:.4f}$, ${b5:.4f}$, ${c:.4f}$)"))


def main():
    D = data()
    print_all(D)
    draw(D)


if __name__ == "__main__":
    main()
