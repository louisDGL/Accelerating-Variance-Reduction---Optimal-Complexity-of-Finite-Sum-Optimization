"""Figure T04 of the lecture (paper/talk/figures/figT04_recipe_pipeline): three steps, one skeleton,
estimator -> map -> step. Explanatory figure, not in the paper (its paper version, the complete diagram in paper mode,
is figE4_recipe_pipeline, drawn by make_figE4_recipe_pipeline.py with the drawing function of this script).

It explains Section 2.1 (SAGA; the polar step and its length rules), Section 2.3 (the estimators) and Eq. (4) of
paper/paper_saga_to_muon.tex, in the corrected wording of paper/audit/paper_saga_to_muon_corrige.tex.

Three lanes, one per method, each with three stations (the formulas are those of the paper):
  SAGA (Section 2.1, vector notation)
      estimator  omega = f_i'(theta) - z_i + (1/n) sum_j z_j,  table z_i <- f_i'(theta), i uniform
      map        identity
      step       theta+ = theta - gamma omega, constant gamma
  Muon (as published by Jordan et al., 2024; the baseline `muon_sgd` of muon/lab/algorithms.py)
      estimator  Nesterov momentum of the minibatch gradient g_t:  m_t = beta m_{t-1} + g_t,  omega_t = g_t + beta m_t;
                 no table
      map        ns5: five Newton-Schulz steps (normalization constant eps = 1e-7), an approximation of the polar factor
      step       W_{t+1} = W_t - lr_t ns5(omega_t): a step of fixed length, the learning rate lr_t is cooled down
                 (no schedule formula is written: Muon's reference runs cool linearly). Not drawn: the shape factor
                 max(1, K/d)^(1/2) of the public code (1 for the K <= d matrices of the paper), its weight decay, and
                 its EMA form of the buffer, (1 - beta) times the one written (a factor that ns5 removes up to eps).
  The recipe of Section 3.3, Eq. (4)
      estimator  SAGA estimator on a batch B_t of b indices (Proposition 5 with theta = 1):
                 omega_t = zbar + lambda W_t + (1/b) sum_{j in B_t} (grad g_j(W_t) - z_j),  table z_j <- grad g_j(W_t)
      map        the exact polar factor U V^T, or ns5
      step       W_{t+1} = W_t - lr_t o(omega_t),  lr_t = lr_0 q^(tb/n): a geometric cooldown, which is a choice of the
                 paper (as is the momentum 0.9 of the runs "with momentum"; Muon: 0.95)
Colours: the station that holds a table is tinted, blue for Euclidean SAGA, orange when the table feeds a polar step;
the stations of Muon (no table) and the maps and steps are neutral. The arrows are the operations of one iteration:
the estimator goes through the map, the result is the direction of the step.
Under the lanes: the two other length rules of the paper for the polar step (nuclear-scaled, Section 2.1; clipped,
Section 2.3), and an inset that shows what the map does.

The inset (the only plot; computed here, nothing is drawn by hand): one K x d = 8 x 40 matrix omega with i.i.d.
standard Gaussian entries (seed SEED), omega = U diag(s) V^T. For each singular direction k = 1..8, three bars on a
common scale: s_k / s_1 before the map (grey); u_k^T polar(omega) v_k after the exact polar factor (solid orange; all
equal to 1); and u_k^T ns5(omega) v_k after Muon's Newton-Schulz map (open orange bars: a dashed outline, the mark of
ns5 in the other figures, is unreadable on bars this thin), with `polar_svd` and `ns5` of muon/lab/polar.py. ns5 is
U diag(p^5(s_k / (|omega|_F + eps))) V^T with the quintic p of the public Muon code: its singular values are computed
in two ways (the matrix routine projected on the singular vectors, and the scalar map `ns_scalar_map`), and the script
stops if they disagree or if ns5(omega) is not diagonal in the singular basis of omega. They are not 1: on this draw
they lie between the two numbers printed in the inset (in general in about [0.68, 1.13] for normalized singular values
in [0.01, 1], also printed by the script; values up to about 1.20 occur for normalized singular values below 0.01).

Status. The lanes are definitions (of SAGA, of Muon, of Eq. (4)); the inset is an illustration on one draw of an exact
property of the two maps. No claim of the paper is proved or measured by this figure.

Progressive build (talk; states s1-s3): s1 the lane of SAGA; s2 adds Muon; s3 adds the recipe, the other length rules
and the inset. The scope line follows the build: none in s1, "Muon: as published (Jordan et al. 2024)" from s2, and
"; recipe: also run with momentum 0.9 (our choice)" from s3 (two texts registered in the build; es.save is given a
blank note, which only reserves the room of the line).

Run:    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/scripts/make_figT04_recipe_pipeline.py
Output: paper/talk/figures/figT04_recipe_pipeline.pdf and .png, and the states _s1 ... _s3
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import explain_style as es  # noqa: E402

sys.path.insert(0, os.path.join(es.ROOT, "muon"))
from lab import polar  # noqa: E402  (muon/lab/polar.py: polar_svd, ns5, ns_scalar_map; importing it writes nothing)

NAME = "figT04_recipe_pipeline"
K, D_COLS, SEED = 8, 40, 0            # the matrix of the inset: K x d standard Gaussian
NS_EPS = 1e-7                         # normalization constant of ns5 (the default of the public Muon code)
NOTE_MUON = "Muon: as published (Jordan et al. 2024)"                    # scope line, from state s2
NOTE_RECIPE = "; recipe: also run with momentum 0.9 (our choice)"          # its end, from state s3

# geometry of the diagram, in inches (the axes fill the canvas: data units are inches) --------------------------------
GEOM = {
    "talk": dict(W=es.TALK_W, H=es.TALK_H - 0.20, x0=0.015, w_lab=0.74, w_est=2.22, w_map=1.14, w_step=1.50,
                 gap=0.125, before_gap=0.25, strip=0.46, head=0.20, lanes=(0.50, 0.50, 0.56), lane_gap=0.045,
                 band=0.62, fs_name=1.0),
    "paper": dict(W=es.PAPER_W, H=2.55, x0=0.02, w_lab=0.80, w_est=2.70, w_map=1.30, w_step=1.50,
                  gap=0.17, before_gap=0.30, strip=0.52, head=0.20, lanes=(0.47, 0.47, 0.54), lane_gap=0.06,
                  band=0.66, fs_name=1.0),
}


# ----------------------------------------------------------------------------------------------------------------------
# data: what the two maps do to the singular values of one Gaussian matrix
# ----------------------------------------------------------------------------------------------------------------------
def data():
    rng = np.random.default_rng(SEED)
    M = rng.standard_normal((K, D_COLS))
    U, s, Vt = polar.svd_thin(M)
    P = polar.polar_svd(M)
    N5 = polar.ns5(M, eps=NS_EPS)
    in_basis = {"polar": U.T @ P @ Vt.T, "ns5": U.T @ N5 @ Vt.T}       # the maps in the singular basis of M
    for name, B in in_basis.items():
        off = float(np.abs(B - np.diag(np.diag(B))).max())
        if off > 1e-9:
            raise SystemExit(f"{name}(omega) is not diagonal in the singular basis of omega (off-diagonal {off:.1e})")
    after_polar = np.diag(in_basis["polar"]).copy()
    after_ns5 = np.diag(in_basis["ns5"]).copy()
    scalar = polar.ns_scalar_map(s / (np.linalg.norm(M) + NS_EPS))       # second way: the scalar quintic, five times
    sv_sorted = polar.singular_values(N5)                                # third way: singular values of ns5(M)
    if np.abs(after_polar - 1.0).max() > 1e-12:
        raise SystemExit("the singular values of the polar factor are not 1")
    if np.abs(after_ns5 - scalar).max() > 1e-9 or np.abs(np.sort(after_ns5)[::-1] - sv_sorted).max() > 1e-9:
        raise SystemExit("ns5: the matrix routine and the scalar map disagree")
    grid = np.linspace(0.01, 1.0, 99001)
    band = polar.ns_scalar_map(grid)
    return {"s": s, "before": s / s[0], "polar": after_polar, "ns5": after_ns5,
            "ns5_range": (float(after_ns5.min()), float(after_ns5.max())),
            "band": (float(band.min()), float(band.max())),
            "check": float(np.abs(after_ns5 - scalar).max())}


def print_all(D):
    f = lambda v: ", ".join(f"{x:.3f}" for x in v)                       # noqa: E731
    print(f"inset: one {K} x {D_COLS} matrix with i.i.d. N(0, 1) entries (numpy default_rng({SEED}))")
    print(f"    singular values s_k:                 {f(D['s'])}")
    print(f"    before the map, s_k / s_1:           {f(D['before'])}")
    print(f"    after the exact polar factor:        {f(D['polar'])}")
    print(f"    after ns5 (eps = {NS_EPS:g}), direction k: {f(D['ns5'])}")
    print(f"    range after ns5 on this draw: {D['ns5_range'][0]:.4f} to {D['ns5_range'][1]:.4f} -> printed "
          f"{D['ns5_range'][0]:.2f} to {D['ns5_range'][1]:.2f}; matrix routine against scalar map: "
          f"largest gap {D['check']:.1e}")
    print(f"    for comparison, image of [0.01, 1] under the five quintic steps: [{D['band'][0]:.3f}, {D['band'][1]:.3f}]")


# ----------------------------------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------------------------------
def lane_texts(D):
    """(name, sub-label, colour slot of the estimator station, estimator, map, step) for the three lanes."""
    return [
        ("SAGA", r"vectors $\theta$", "BLUE",
         r"$\omega = f_i'(\theta) - z_i + \frac{1}{n}{\sum}_j\,z_j$" + "\n"
         + r"table: $z_i \leftarrow f_i'(\theta)$,  $i$ uniform",
         "identity\n" + r"$o(\omega) = \omega$",
         r"$\theta^+ = \theta - \gamma\,\omega$" + "\n" + r"constant $\gamma$"),
        ("Muon", r"matrices $W$", None,
         r"$m_t = \beta\,m_{t-1} + \hat g_t$,   $\omega_t = \hat g_t + \beta\,m_t$" + "\n"
         + r"$\hat g_t$: minibatch gradient;  no table",
         r"$\mathrm{ns5}(\omega_t) \approx UV^{\!\top}$" + "\nNewton–Schulz",
         r"$W_{t+1} = W_t - \mathrm{lr}_t\,o(\omega_t)$" + "\n" + r"$\mathrm{lr}_t$ cooled down"),
        ("Recipe", "Eq. (4)", "ORANGE",
         r"$\omega_t = \bar z + \lambda W_t + \frac{1}{b}{\sum}_{j\in B_t}(\nabla g_j(W_t) - z_j)$" + "\n"
         + r"table: $z_j \leftarrow \nabla g_j(W_t)$,  $j \in B_t$",
         r"$\mathrm{polar}(\omega_t) = UV^{\!\top}$" + "\n" + r"or $\mathrm{ns5}(\omega_t)$",
         r"$W_{t+1} = W_t - \mathrm{lr}_t\,o(\omega_t)$" + "\n" + r"$\mathrm{lr}_t = \mathrm{lr}_0\,q^{\,tb/n}$"
         + "\ngeometric (our choice)"),
    ]


def draw_diagram(ax, D, ps, g, build=None):
    """The complete diagram on an axes whose data units are inches (window g["W"] x g["H"]). build: an es.Build in
    which the artists are registered by state (lane 1, lane 2, lane 3 with the band), or None."""
    ink = ps.INK
    fs = es.small()
    add = (lambda state, *a: build.add(state, *a)) if build is not None else (lambda state, *a: None)
    top = g["H"]
    x_lab = g["x0"]
    x_est = x_lab + g["w_lab"]
    x_map = x_est + g["w_est"] + g["gap"]
    x_step = x_map + g["w_map"] + g["gap"]
    cols = ((x_est, g["w_est"]), (x_map, g["w_map"]), (x_step, g["w_step"]))

    # column headers (with lane 1)
    for (x, w), text in zip(cols, (r"estimator $\omega$", r"map $o(\omega)$", "step")):
        add(1, es.label(ax, x + 0.5 * w, top - 0.5 * g["head"] - 0.01, text, fontsize=fs + 0.5, weight="bold",
                        colour=ink["secondary"]))

    y = top - g["head"]
    for i, (name, sub, slot, est, mp, step) in enumerate(lane_texts(D)):
        state = i + 1
        h = g["lanes"][i]
        yc = y - 0.5 * h
        colour = getattr(ps, slot) if slot else None
        arts = [ax.text(x_lab + 0.02, yc + 0.075, name, ha="left", va="center", fontsize=fs + g["fs_name"],
                        fontweight="bold", color=ink["primary"], zorder=5),
                ax.text(x_lab + 0.02, yc - 0.085, sub, ha="left", va="center", fontsize=fs, color=ink["secondary"],
                        zorder=5)]
        for (x, w), text, c in zip(cols, (est, mp, step), (colour, None, None)):
            arts += list(es.box(ax, x + 0.5 * w, yc, w, h, text, colour=c, rounding=0.016, lw=0.8, linespacing=1.2))
        for (xa, wa), (xb, _) in zip(cols[:-1], cols[1:]):                # the estimator through the map, then the step
            arts.append(es.arrow(ax, (xa + wa + 0.012, yc), (xb - 0.012, yc), lw=1.1, mutation_scale=8.0))
        add(state, *arts)
        y -= h + g["lane_gap"]

    # the band under the lanes (state 3): what the map does, and the other length rules
    y_band_top = y + g["lane_gap"] - 0.07
    base = y_band_top - g["band"] + 0.19                                  # level 0 of the bars
    unit = (g["band"] - 0.25) / 1.22                                      # height of the value 1
    mid = base + 0.5 * unit + 0.03
    arts = []
    strip_w, strip_gap = g["strip"], 0.11
    x_polar = x_map + 0.5 * (g["w_map"] - 2 * strip_w - strip_gap)       # the two "after" strips: under the map column
    x_before = x_map - g["gap"] - strip_w - g["before_gap"]
    strips = (("before", x_before, D["before"], r"$s_k\,/\,s_1$"),       # omega = U diag(s) V^T (Section 2.1)
              ("polar", x_polar, D["polar"], r"polar($\omega$)"),
              ("ns5", x_polar + strip_w + strip_gap, D["ns5"], r"ns5($\omega$)"))
    pitch = strip_w / K
    for key, x_left, values, text in strips:
        xk = x_left + pitch * (np.arange(K) + 0.5)
        if key == "before":                                               # the estimator: context, grey
            bars = ax.bar(xk, unit * values, width=0.62 * pitch, bottom=base, color=ink["muted"], lw=0, zorder=3)
        elif key == "polar":                                              # exact polar factor: solid orange
            bars = ax.bar(xk, unit * values, width=0.62 * pitch, bottom=base, color=ps.ORANGE, lw=0, zorder=3)
        else:                                                             # ns5: open bars (a dashed outline is
            #                                                               unreadable on bars this thin)
            bars = ax.bar(xk, unit * values, width=0.50 * pitch, bottom=base, color=es.tint(ps.ORANGE, 0.25),
                          edgecolor=ps.ORANGE, lw=0.7, zorder=3)
        arts += list(bars)
        arts.append(es.label(ax, x_left + 0.5 * strip_w, base - 0.09, text))
    x_a, x_b = strips[0][1], strips[2][1] + strip_w
    arts += ax.plot([x_a - 0.03, x_b + 0.03], [base, base], color=ink["axis"], lw=0.8, zorder=2)
    arts += ax.plot([x_a - 0.03, x_b + 0.03], [base + unit, base + unit], color=ink["primary"], lw=0.7, zorder=4)
    arts.append(es.label(ax, x_a - 0.06, base + unit, "1", ha="right", colour=ink["muted"]))
    arts.append(es.arrow(ax, (x_before + strip_w + 0.04, base + 0.5 * unit), (x_polar - 0.04, base + 0.5 * unit),
                         lw=1.1, mutation_scale=8.0))                     # the map, applied to omega
    arts.append(es.label(ax, 0.5 * (x_before + strip_w + x_polar), base + 0.5 * unit + 0.085, "$o$"))
    lo, hi = D["ns5_range"]
    arts.append(ax.text(x_lab + 0.02, mid,
                        f"one {K} " + r"$\times$" + f" {D_COLS} Gaussian matrix " + r"$\omega$:" + "\n"
                        + "singular values 1 after polar,\n" + f"{lo:.2f} to {hi:.2f} after ns5",
                        ha="left", va="center", fontsize=fs, color=ink["secondary"], linespacing=1.2, zorder=5))
    arts.append(ax.text(x_step - 0.03, mid,
                        r"other lengths than $\mathrm{lr}_t$:" + "\n"
                        + r"$\gamma\|\omega\|_*$ (nuclear-scaled)," + "\n"
                        + r"$\min\{\rho,\,\gamma\|\omega\|_*\}$ (clipped)",
                        ha="left", va="center", fontsize=fs, color=ink["secondary"], linespacing=1.2, zorder=5))
    add(3, *arts)


def draw(D, mode="talk", name=NAME):
    ps = es.setup(mode)
    g = GEOM[mode]
    fig = es.figure(mode, height=es.TALK_H if mode == "talk" else g["H"])
    fig.get_layout_engine().set(w_pad=0.0, h_pad=0.0, wspace=0.0, hspace=0.0)
    ax = fig.add_subplot(111)
    es.diagram_axes(ax, (0.0, g["W"]), (0.0, g["H"]))
    build = es.Build() if mode == "talk" else None
    draw_diagram(ax, D, ps, g, build)
    for a in ax.get_children():                 # the texts must not push the axes: data units stay inches
        a.set_in_layout(False)
    if mode == "talk":
        # the scope line follows the build: nothing in s1 (SAGA, Section 2.1), Muon's half from s2, the recipe's from s3
        # (es.save draws one fixed note: it is given a blank one, which only reserves the room of the line)
        fig.canvas.draw()
        first = build.add(2, es.scope_note(fig, NOTE_MUON))
        x1 = first.get_window_extent(fig.canvas.get_renderer()).x1 / fig.bbox.width
        build.add(3, es.scope_note(fig, NOTE_RECIPE, x=x1))
        es.save(fig, name, "talk", build=build, note=" ")
    else:
        es.save(fig, name, "paper")


def main():
    D = data()
    print_all(D)
    draw(D, "talk", NAME)


if __name__ == "__main__":
    main()
