"""Figure T09b of the lecture (paper/talk/figures/figT09b_nfl_norms): the condition numbers behind Theorem 1 ("no free
lunch, for every norm") of paper/paper_saga_to_muon.tex, norm by norm, on the toy of figure T09. Explanatory figure,
not in the paper; it is the talk version of panel (c) of paper/figures/figE2_nfl_plane, with one more element (the
quadratic loss, for which the hypothesis of the theorem fails).

Message: on the ridge logistic model no norm has a global condition number below kappa_F; a norm can improve the
condition number at the minimizer, and it can improve the global one when the loss curvature does not vanish
(quadratic loss: a matched norm gives exactly 1).

The toy is the one of make_figE2_nfl_plane.py, imported from that script (its data(), norms(), unit_sphere(), quad()):
    F(W) = (1/n) sum_i log(1 + exp(-y_i W x_i)) + (lam/2) |W|^2,   K = 1, d = 2, n = 40, lam = 0.02,
H0 = D^2F(0) with the eigenvalues 9 lam and 2.25 lam, mu_F = lam, kappa_F = 9. The quadratic loss is l(z) = z^2/8:
    F_q(W) = (1/n) sum_i (W x_i)^2 / 8 + (lam/2) |W|^2,   D^2F_q = H0 everywhere (checked).

One panel, one row per norm N on the plane (Euclidean, l_infinity, l_1, matched to D^2F(0), matched to D^2F(W_*), and
the best of the scan of ellipsoidal norms of the imported script: 10,801 norms when this was written; the
number shown is the one the imported script returns):
  filled disc   global condition number of F, kappa_N = L_N / mu_N (certified: these are the constants a proof uses);
                all are >= kappa_F (Theorem 1);
  open disc     condition number at the minimizer W_*: max / min of D^2F(W_*)[Delta, Delta] over the unit sphere of N;
  diamond       (two rows below) global condition number of the quadratic model F_q, whose Hessian is H0 everywhere:
                max / min of H0[Delta, Delta] over the unit sphere of N, in the Euclidean norm and in the norm matched
                to H0. The Euclidean value is the quadratic model's own kappa_F (lambda_max / lambda_min of H0 = 4),
                and is labelled so ("its kappa_F = 4"): Theorem 1 compares kappa_N(F) with kappa_F of the same F, so the
                matched 1 is to be read against 4, not against the kappa_F = 9 of the logistic model on the axis.
Values written next to a mark: only those that are not on a labelled tick.

Every number is computed in two ways and the script stops if they disagree: the global constants of F in the imported
script (closed form and brute force over the unit sphere of N and the points of a ray), the condition numbers of a
fixed Hessian (at W_*, and H0 for the quadratic loss) here, in closed form (cond_closed) and by brute force over the
unit sphere of N.

Status. Proved: kappa_N >= kappa_F for every norm (Theorem 1), here checked on five norms and a scan; exact
computation on one toy for the rest. Not claimed: that a change of norm is useless (it changes local constants and
trajectories: findings (C5) and (M4) of the paper; "the gain of a norm is a property of the trajectory"). The toy has
K = 1: its spectral norm is the Euclidean norm, and the figure says "norm", not "geometry".

Progressive build (states s1-s3): s1 the global condition numbers and kappa_F; s2 adds the condition numbers at the
minimizer; s3 adds the quadratic loss.

The script prints every number it plots. Nothing is read from the results of the project.
Run:    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/scripts/make_figT09b_nfl_norms.py
Output: paper/talk/figures/figT09b_nfl_norms.pdf and .png, and the states _s1 ... _s3
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import explain_style as es  # noqa: E402
import make_figE2_nfl_plane as E2  # noqa: E402  (the toy and its checks; importing it draws nothing)

NAME = "figT09b_nfl_norms"
LAM = E2.LAM
QUAD_ROWS = ("l2", "H0")                     # norms shown for the quadratic loss
NOTE = (r"exact toy: ridge logistic regression, $K = 1$, $d = 2$, $n = 40$; quadratic loss: its Hessian is $D^2F(0)$ "
        "everywhere")


# ----------------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------------
def cond_closed(H, key, P):
    """max / min of H[Delta, Delta] over the unit sphere of the norm, in closed form (H symmetric positive definite).
    The minimum is 1 / max of N^2 over the ellipsoid {H <= 1}: for l_infinity the largest (H^-1)_jj, for l_1 the
    largest s^T H^-1 s over the sign vectors s; the maximum of a convex function over a polygon is at a vertex."""
    Hi = np.linalg.inv(H)
    signs = np.array([[1.0, 1.0], [1.0, -1.0]])
    if key == "linf":
        return float(E2.quad(H, signs).max() * np.diag(Hi).max())
    if key == "l1":
        return float(np.diag(H).max() * E2.quad(Hi, signs).max())
    Li = np.linalg.inv(np.linalg.cholesky(P))
    ev = np.linalg.eigvalsh(Li @ H @ Li.T)
    return float(ev[-1] / ev[0])


def data():
    D = E2.data()                                        # global constants: closed forms checked by brute force
    missing = {"H0", "Hs", "Xs", "kappa_F", "rows", "scan_best", "scan_size"} - set(D)
    if missing:
        raise SystemExit(f"make_figE2_nfl_plane.data() no longer returns {sorted(missing)}")
    H0, Hs, Xs = D["H0"], D["Hs"], D["Xs"]
    Hq = Xs.T @ Xs / E2.N_PTS / 4.0 + LAM * np.eye(2)    # Hessian of the quadratic model F_q (l'' = 1/4), any W
    if np.abs(Hq - H0).max() > 1e-12 * np.abs(H0).max():
        raise SystemExit("the Hessian of the quadratic model is not D^2F(0)")
    rows = {r["key"]: r for r in D["rows"]}
    for key, _, _, norm, P in E2.norms(H0, Hs):
        S, _ = E2.unit_sphere(norm)
        r = rows[key]
        for name, H in (("local", Hs), ("quad", Hq)):
            q = E2.quad(H, S)
            brute, closed = float(q.max() / q.min()), cond_closed(H, key, P)
            if abs(brute / closed - 1.0) > 1e-6:
                raise SystemExit(f"{key}, {name}: closed form {closed} and brute force {brute} disagree")
            r[name + "_closed"] = closed
        if abs(r["local_closed"] / r["local"] - 1.0) > 1e-6:
            raise SystemExit(f"{key}: the condition number at the minimizer disagrees with the imported one")
        if r["kappa"] < D["kappa_F"] * (1.0 - 1e-9):
            raise SystemExit(f"{key}: kappa_N < kappa_F, against Theorem 1")
    if D["scan_best"] < D["kappa_F"] * (1.0 - 1e-9):
        raise SystemExit("scan: a norm with kappa_N < kappa_F, against Theorem 1")
    ev = np.linalg.eigvalsh(H0)
    if abs(rows["l2"]["quad_closed"] / (ev[-1] / ev[0]) - 1.0) > 1e-9 or abs(rows["H0"]["quad_closed"] - 1.0) > 1e-9:
        raise SystemExit("quadratic loss: Euclidean and matched condition numbers are not lambda_max/lambda_min and 1")
    return D


def print_all(D):
    E2.print_all(D)
    print("[T09b] norm: global kappa_N of F (filled), at W* (open), global for the quadratic loss (diamond if drawn)")
    for r in D["rows"]:
        drawn = "drawn" if r["key"] in QUAD_ROWS else "not drawn"
        print(f"[T09b]   {r['key']:5s} {r['kappa']:8.4f} {r['local_closed']:8.4f} {r['quad_closed']:8.4f} ({drawn})")
    ev = np.linalg.eigvalsh(D["H0"])
    print(f"[T09b]   quadratic model: its own kappa_F = lambda_max / lambda_min of D^2F(0) = {ev[-1] / ev[0]:.4f} "
          f"(label 'its kappa_F = 4'); matched norm {next(r for r in D['rows'] if r['key'] == 'H0')['quad_closed']:.4f}")
    print(f"[T09b]   scan of {D['scan_size']} ellipsoidal norms: smallest global kappa_N = {D['scan_best']:.6f}; "
          f"kappa_F = {D['kappa_F']:.4f}")


# ----------------------------------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------------------------------
def draw(D):
    ps = es.setup("talk")
    ink = ps.INK
    colour = {"INK": ink["secondary"], "ORANGE": ps.ORANGE, "AQUA": ps.AQUA, "BLUE": ps.BLUE}
    fig = es.figure("talk", height=2.75)
    fig.get_layout_engine().set(w_pad=0.05, wspace=0.0)
    gs = fig.add_gridspec(1, 2, width_ratios=(4.05, 1.85))
    ax = fig.add_subplot(gs[0, 0])
    ax_n = fig.add_subplot(gs[0, 1], sharey=ax)          # the key and the notes, row by row
    b = es.Build()
    rows = D["rows"]
    kF = D["kappa_F"]
    names = {"l2": "Euclidean", "linf": r"$\ell_\infty$ (sign descent)", "l1": r"$\ell_1$",
             "H0": r"matched to $D^2F(0)$", "Hs": r"matched to $D^2F(W_\ast)$"}
    GLOB = dict(ls="", marker="o", ms=7.5, mew=1.0)      # global condition number: filled disc
    LOC = dict(ls="", marker="o", ms=7.0, mew=1.5)       # at the minimizer: open disc
    QUAD = dict(ls="", marker="D", ms=6.8, mew=1.0)      # quadratic loss, global: diamond
    value = dict(textcoords="offset points", ha="left", va="center", fontsize=es.small(), color=ink["secondary"])

    # logistic model: rows y = 8 ... 3 (top to bottom); quadratic model: rows y = 1, 0
    ax.axvline(kF, color=ink["primary"], lw=1.0, zorder=1)
    yt, yl = [], []
    for k, r in enumerate(rows):
        y, col = 8 - k, colour[r["colour"]]
        yt.append(y)
        yl.append(names[r["key"]])
        b.add(2, ax.plot([r["local"], r["kappa"]], [y, y], color=col, lw=1.2, zorder=2))
        b.add(2, ax.plot([r["local"]], [y], mfc=ink["surface"], mec=col, zorder=3, **LOC))
        ax.plot([r["kappa"]], [y], mfc=col, mec=ink["surface"], zorder=4, **GLOB)
        if r["kappa"] > 1.01 * kF:                       # written only when the mark is not on the tick kappa_F
            ax.annotate(f"{r['kappa']:.1f}", (r["kappa"], y), xytext=(7, 0), **value)
    ax.plot([D["scan_best"]], [3], mfc=ps.BLUE, mec=ink["surface"], zorder=4, **GLOB)
    yt.append(3)
    yl.append(f"best of {D['scan_size']:,} ellipsoidal norms")
    quad = {r["key"]: r for r in rows}
    # the Euclidean value of the quadratic model is its own kappa_F (lambda_max / lambda_min of H0, checked in data()):
    # say so, so that its matched 1 is compared with 4, not with the kappa_F = 9 of the logistic model on the axis
    # (written on the left of the mark: on the right it would cross the line kappa_F = 9)
    for y, key, text, fmt, dx in ((1, "l2", "Euclidean", r"its $\kappa_F$ = {:.0f}", -9),
                                  (0, "H0", "matched to its Hessian", "{:.0f}", 8)):
        r = quad[key]
        b.add(3, ax.plot([r["quad_closed"]], [y], mfc=colour[r["colour"]], mec=ink["surface"], zorder=4, **QUAD))
        b.add(3, ax.annotate(fmt.format(r["quad_closed"]), (r["quad_closed"], y), xytext=(dx, 0),
                             **dict(value, ha="right" if dx < 0 else "left")))
        yt.append(y)
        yl.append(text)
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=es.small(), color=ink["secondary"])
    for lab in ax.get_yticklabels()[-2:]:
        b.add(3, lab)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.75, 8.75)
    ax.set_xscale("log")
    ax.set_xlim(0.82, 34.0)
    ax.set_xticks([1, 3, 9])
    ax.set_xticklabels(["1", "3", r"$\kappa_F = 9$"])
    ax.minorticks_off()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("condition number in the norm $N$")
    b.add(3, ax.axhline(2.0, color=ink["axis"], lw=0.8, zorder=1))          # between the two models

    # the key, in the rows of the right column: mark shapes in neutral ink, each with what it shows
    es.diagram_axes(ax_n, (0.0, 1.0), ax.get_ylim())
    head = dict(fontsize=es.small(), color=ink["primary"], ha="left", va="center", fontweight="bold")
    note = dict(fontsize=es.small(), color=ink["secondary"], ha="left", va="center")
    XM, XT = 0.10, 0.19                                 # mark, text
    ax_n.text(0.0, 8.0, "logistic loss", **head)
    ax_n.plot([XM], [7.0], mfc=ink["secondary"], mec=ink["surface"], **GLOB)
    ax_n.text(XT, 7.0, "global constants:", **note)
    ax_n.text(XT, 6.25, r"$\kappa_N \geq \kappa_F$ for every norm", **note)
    b.add(2, ax_n.plot([XM], [5.0], mfc=ink["surface"], mec=ink["secondary"], **LOC))
    b.add(2, ax_n.text(XT, 5.0, r"at the minimizer $W_\ast$:", **note))
    b.add(2, ax_n.text(XT, 4.25, "a norm can help; its gain is", **note))
    b.add(2, ax_n.text(XT, 3.5, "a property of the trajectory", **note))
    b.add(3, ax_n.text(0.0, 1.0, "quadratic loss", **head))
    b.add(3, ax_n.plot([XM], [0.0], mfc=ink["secondary"], mec=ink["surface"], **QUAD))
    b.add(3, ax_n.text(XT, 0.0, "global: matched norm gives 1", **note))
    es.save(fig, NAME, "talk", build=b, note=NOTE)


def main():
    D = data()
    print_all(D)
    draw(D)


if __name__ == "__main__":
    main()
