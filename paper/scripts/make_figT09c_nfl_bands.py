"""Figure T09c of the lecture (paper/talk/figures/figT09c_nfl_bands): Theorem 1 ("no free lunch, for every norm") of
paper/paper_saga_to_muon.tex as bands over all directions of the plane. In each direction the curvature of the
objective ranges over a band; a norm slides the band of each direction up or down (on a logarithmic axis) and cannot
change its height, so the condition number in any norm is at least the tallest band. Explanatory figure, not in the
paper; it is one instance of the theorem, not its proof, and adds no claim.

The toy ("Toy A" of the storyboard, the instance of figT06c_curvature_far), an exact instance of the hypotheses of
Theorem 1 with K = 1 and d = 2 (the parameter W is a row vector of the plane):
    F(W) = (1/n) sum_i log(1 + exp(-y_i W x_i)) + (lam/2) |W|^2,   n = 100,
    generated with numpy.random.default_rng(0), in this order: X = (standard_normal((100, 2)) * sqrt([1, 0.05])) R^T
    with R the rotation by 30 degrees; X /= max_i |x_i|; w = standard_normal(2) rescaled so that mean((X w)^2) = 4;
    y_i = +1 where random(100) < sigmoid(x_i.w), else -1;   lam = (1/4) / (10^4 - 1),
and its quadratic twin: the same features and lam with the loss l(z) = z^2 / 8 in place of log(1 + e^{-z}), a ridge
least-squares model whose Hessian is H0 = D^2F(0) at every point (the case where the hypothesis of Theorem 1 fails).
Since K = 1 the spectral norm of W is its Euclidean norm: no curve of this figure is "spectral".

For the direction Delta = e(phi) = (cos phi, sin phi), phi in [0, 180] degrees:
    upper(phi) = sup_W D^2F(W)[e, e] = e^T H0 e =: h0(phi)  for both models (logistic: attained at W = 0, where every
                 sample has its largest curvature 1/4);
    lower(phi) = inf_W D^2F(W)[e, e] = lam for the logistic model (the limit along almost every ray t W_1: the
                 hypothesis of Theorem 1), and h0(phi) for the twin (constant Hessian: a band of height one).
A norm N enters through n(phi) = N(e(phi))^2 only: L_N = max_phi upper / n, mu_N = min_phi lower / n,
kappa_N = L_N / mu_N. Every panel plots, against phi, upper / (n mu_N) and lower / (n mu_N) on a logarithmic axis:
the lowest point of the lower edges is 1 and the highest point of the upper edges reads kappa_N. The height of the
band at phi is upper(phi) / lower(phi) whatever the norm, hence kappa_N >= max_phi upper / lower = kappa_F for the
logistic model (Theorem 1), whereas all the bands of the twin have height one and the matched norm levels them.

Panels (one vertical axis):
  (a) the quadratic twin: Euclidean norm (n = 1; kappa_N = lam_max(H0) / lam_min(H0)) and the norm matched to the
      Hessian, N(Delta)^2 = D^2F(0)[Delta, Delta] (n = h0; kappa_N = 1);
  (b) the logistic model in the same two norms, one sub-panel each: Euclidean (lower edge flat at 1, upper edge
      h0(phi) / lam) and matched (upper edge flat at kappa_F, lower edge max h0 / h0(phi)). Double arrows: the height
      of the band in the top eigendirection of H0 (kappa_F) and in the bottom one, the same in the two norms. Thin
      dashed line: the curvature at the minimizer W_*, D^2F(W_*)[e, e] / (n mu_N); its max / min is the condition
      number of the Hessian at W_* in that norm (printed, not written on the figure): a norm can improve it, and
      local constants certify no global rate. The three curves of a band are labelled by where they live: the
      upper edge at W = 0, the dashed line at W_* (marked "local"), the lower edge in the limit |W| -> infinity.
      The sentence "local constants certify no global rate" is not on the figure: the slide or the speaker says it.

Two ways for every constant (the script stops if they disagree): L_N and mu_N from the closed forms above evaluated
on a grid of N_PHI angles, from eigenvalue formulas (ellipsoidal norms) or vertex formulas (l_1, l_infinity), and by
brute force over W (curvature at W = 0 and at the points t u of N_RAYS generic rays, t from 1e-2 to 1e7). Numerical
checks printed and not drawn (the storyboard put them in a third panel, left out for room; the sibling figure
figT09_nfl_plane has its own dot plot): kappa_N for l_1, l_infinity and the norm matched to D^2F(W_*), a scan of
SCAN_METRICS random quadratic metrics and of SCAN_POLY random polyhedral norms (all >= kappa_F), and the per-sample
ratios L_N(f_i) / mu_N(F) of the last statement of the theorem for the sample of largest norm.

Status. Proved: Theorem 1 (kappa_N >= kappa_F for every norm). Exact closed forms on the toy: the band edges and
the values of kappa_N. Numerical checks: the scans. Illustrative: the choice of the instance and of the norms.

Progressive build (states s1-s4): s1 the twin in the Euclidean norm; s2 adds the matched norm; s3 the logistic
model in the Euclidean norm; s4 the logistic model in the matched norm.

The script prints every number it plots. Nothing is read from the results of the project.
Run:    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/scripts/make_figT09c_nfl_bands.py   (about 10 s)
Output: paper/talk/figures/figT09c_nfl_bands.pdf and .png, and the states _s1 ... _s4
"""
from __future__ import annotations

import itertools
import os
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import explain_style as es  # noqa: E402

NAME = "figT09c_nfl_bands"
N, KAPPA, DATA_SEED, ROT_DEG = 100, 1e4, 0, 30.0
N_PHI = 7201                               # angles of the directions, 0 to 180 degrees (step 0.025 degree)
N_RAYS, RAY_STEP_DEG, RAY_OFFSET_DEG = 12, 15, 10    # generic rays of the brute-force check (as in figT06c)
RADII = np.logspace(-2.0, 7.0, 109)        # distances along these rays
SCAN_SEED, SCAN_METRICS, SCAN_POLY = 1, 20000, 5000
# values of the storyboard (prototype of 2026-11-25), checked to their rounding (the scans are reported, not checked:
# their random streams are not specified there)
STORYBOARD = {"eig0": (79.2, 1669.7), "top_deg": 31.7, "kappa_twin": 21.1,
              "kappa": {"l2": 1670.0, "H0": 1670.0, "Hs": 1678.0, "l1": 2461.0, "linf": 3171.0},
              "local": {"l2": 8.6, "H0": 2.5, "Hs": 1.0, "l1": 12.3, "linf": 12.3},
              "sample": {"l2": 10000.0, "H0": 14215.0, "Hs": 12066.0, "linf": 17284.0}}


# ----------------------------------------------------------------------------------------------------------------------
# the toy
# ----------------------------------------------------------------------------------------------------------------------
def rot(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def instance():
    """Toy A: features, labels and ridge parameter (see the docstring; the order of the draws matters)."""
    rng = np.random.default_rng(DATA_SEED)
    X = (rng.standard_normal((N, 2)) * np.sqrt([1.0, 0.05])) @ rot(ROT_DEG).T
    X /= np.linalg.norm(X, axis=1).max()
    w = rng.standard_normal(2)
    w *= 2.0 / np.sqrt(np.mean((X @ w) ** 2))
    y = np.where(rng.random(N) < 1.0 / (1.0 + np.exp(-(X @ w))), 1.0, -1.0)
    return X, y, 0.25 / (KAPPA - 1.0)


def hessian(X, lam, w):
    """D^2F(w) of the logistic model; sigma'(z) = e^{-|z|} / (1 + e^{-|z|})^2 (even in z: the labels do not matter)."""
    e = np.exp(-np.abs(X @ w))
    s = e / (1.0 + e) ** 2
    return (X * s[:, None]).T @ X / X.shape[0] + lam * np.eye(2)


def minimizer(X, y, lam):
    """Newton's method with step halving on |grad|; returns (W_*, |grad F(W_*)|)."""
    Xs = X * y[:, None]

    def grad(w):
        return -(Xs * (0.5 * (1.0 - np.tanh(0.5 * (Xs @ w))))[:, None]).mean(axis=0) + lam * w
    w = np.zeros(2)
    g = grad(w)
    for _ in range(100):
        step = np.linalg.solve(hessian(X, lam, w), g)
        t = 1.0
        while t > 1e-8 and np.linalg.norm(grad(w - t * step)) >= np.linalg.norm(g):
            t *= 0.5
        if t <= 1e-8:
            break
        w = w - t * step
        g = grad(w)
    return w, float(np.linalg.norm(g))


def qform(H, c, s):
    """e^T H e for the directions e = (c, s)."""
    return H[0, 0] * c * c + 2.0 * H[0, 1] * c * s + H[1, 1] * s * s


def gen_eigs(H, P):
    """Eigenvalues (ascending) of P^{-1/2} H P^{-1/2}: the extreme values of Delta^T H Delta / Delta^T P Delta."""
    Li = np.linalg.inv(np.linalg.cholesky(P))
    return np.linalg.eigvalsh(Li @ H @ Li.T)


def close(a, b, digits):
    """a equals b once rounded to the given number of decimals (b is a rounded storyboard value)."""
    return abs(a - b) <= 0.5 * 10.0 ** (-digits) + 1e-9


def data():
    X, y, lam = instance()
    H0 = hessian(X, lam, np.zeros(2))
    ev0, V0 = np.linalg.eigh(H0)
    ws, gnorm = minimizer(X, y, lam)
    if gnorm > 1e-15:
        raise SystemExit(f"the minimizer is not converged: |grad F| = {gnorm:.1e}")
    Hs = hessian(X, lam, ws)
    evs = np.linalg.eigvalsh(Hs)
    kappa_F = float(ev0[1] / lam)
    top_deg = float(np.degrees(np.arctan2(V0[1, 1], V0[0, 1])) % 180.0)
    bot_deg = (top_deg + 90.0) % 180.0

    phi = np.linspace(0.0, 180.0, N_PHI)
    c, s = np.cos(np.deg2rad(phi)), np.sin(np.deg2rad(phi))
    h0, hs = qform(H0, c, s), qform(Hs, c, s)

    # brute force over W: curvature in every direction at W = 0 and along generic rays
    pts = [np.zeros(2)]
    margins = {off: min(float(np.abs(X @ rot(off + RAY_STEP_DEG * k)[:, 0]).min()) for k in range(N_RAYS))
               for off in range(RAY_STEP_DEG)}
    if max(margins, key=margins.get) != RAY_OFFSET_DEG:
        raise SystemExit("RAY_OFFSET_DEG is not the offset that keeps the x_i farthest from orthogonal to the rays")
    for k in range(N_RAYS):
        u = rot(RAY_OFFSET_DEG + RAY_STEP_DEG * k)[:, 0]
        pts += [t * u for t in RADII]
    Q = np.array([qform(hessian(X, lam, w), c, s) for w in pts])          # (points, angles)
    upper_bf, lower_bf = Q.max(axis=0), Q.min(axis=0)
    if np.abs(upper_bf / h0 - 1.0).max() > 1e-12:
        raise SystemExit("sup_W of the curvature in a direction is not its value at W = 0")
    if (lower_bf / lam - 1.0).min() < -1e-12 or (lower_bf / lam - 1.0).max() > 1e-9:
        raise SystemExit("inf_W of the curvature in a direction is not lam on the sampled rays")

    # norms: n(phi) = N(e(phi))^2 and the closed forms of L_N, mu_N (logistic), mu_N (twin), c_N^2
    corners = np.array([[1.0, 1.0], [1.0, -1.0]])
    vertices = np.eye(2)

    def ellipsoidal(P):
        g = gen_eigs(H0, P)
        return qform(P, c, s), float(g[1]), lam / float(np.linalg.eigvalsh(P)[1]), float(g[0])
    table = {
        "l2": ("Euclidean",) + ellipsoidal(np.eye(2)),
        "H0": ("matched to D^2F(0)",) + ellipsoidal(H0),
        "Hs": ("matched to D^2F(W*)",) + ellipsoidal(Hs),
        "l1": ("l_1", (np.abs(c) + np.abs(s)) ** 2, float(max(v @ H0 @ v for v in vertices)), lam / 2.0, None),
        "linf": ("l_infinity", np.maximum(np.abs(c), np.abs(s)) ** 2, float(max(v @ H0 @ v for v in corners)),
                 lam / 1.0, None),
    }
    i_max = int(np.argmax(np.linalg.norm(X, axis=1)))
    li = 0.25 * (X[i_max, 0] * c + X[i_max, 1] * s) ** 2 + lam            # sup_W of the curvature of f_i in e(phi)
    norms = {}
    for key, (label, n, L_cf, mu_cf, mu_twin_cf) in table.items():
        L_grid, mu_grid = float((h0 / n).max()), float(lam / n.max())
        L_bf, mu_bf = float((Q / n).max()), float((Q / n).min())
        for a, b_, what in ((L_grid, L_cf, "L_N grid / closed form"), (mu_grid, mu_cf, "mu_N grid / closed form"),
                            (L_bf, L_cf, "L_N brute force / closed form"),
                            (mu_bf, mu_cf, "mu_N brute force / closed form")):
            if abs(a / b_ - 1.0) > 1e-6:
                raise SystemExit(f"{key}: {what} disagree: {a} / {b_}")
        mu_twin = float((h0 / n).min())
        if mu_twin_cf is not None and abs(mu_twin / mu_twin_cf - 1.0) > 1e-6:
            raise SystemExit(f"{key}: mu_N of the twin, grid / closed form disagree")
        norms[key] = {"label": label, "n": n, "L": L_cf, "mu": mu_cf, "kappa": L_cf / mu_cf,
                      "mu_twin": mu_twin, "kappa_twin": L_cf / mu_twin,
                      "local": float((hs / n).max() / (hs / n).min()),
                      "sample": float((li / n).max() / mu_cf)}
        if norms[key]["kappa"] < kappa_F * (1.0 - 1e-12):
            raise SystemExit(f"{key}: kappa_N < kappa_F, which contradicts Theorem 1")

    # scans (numerical checks, not drawn): random quadratic metrics, random polyhedral norms; exact formulas
    rng = np.random.default_rng(SCAN_SEED)
    ang, logs = rng.uniform(0.0, np.pi, SCAN_METRICS), rng.uniform(-3.0, 3.0, SCAN_METRICS)
    Rs = np.stack([np.stack([np.cos(ang), -np.sin(ang)], axis=-1), np.stack([np.sin(ang), np.cos(ang)], axis=-1)],
                  axis=-2)                                                # P = R diag(1, 10^l) R^T, one R per metric
    dinv = np.stack([np.ones(SCAN_METRICS), 10.0 ** (-0.5 * logs)], axis=-1)
    M = np.einsum("mji,jk,mkl->mil", Rs, H0, Rs) * dinv[:, :, None] * dinv[:, None, :]   # same spectrum as P^-1/2 H0 P^-1/2
    best_metric = float((np.linalg.eigvalsh(M)[:, 1] * np.maximum(1.0, 10.0 ** logs)).min() / lam)
    P = rot(np.degrees(ang[0])) @ np.diag([1.0, 10.0 ** logs[0]]) @ rot(np.degrees(ang[0])).T
    if abs(np.linalg.eigvalsh(M[0])[1] / gen_eigs(H0, P)[1] - 1.0) > 1e-9:
        raise SystemExit("the vectorized scan of the quadratic metrics disagrees with the direct formula")
    best_poly = np.inf
    for _ in range(SCAN_POLY):
        A = rng.standard_normal((int(rng.integers(2, 7)), 2))             # N(x) = max_k |<a_k, x>|, 2 to 6 rows
        L = 0.0                                                           # max of x^T H0 x over the vertices of N <= 1
        for i, j in itertools.combinations(range(A.shape[0]), 2):
            for sj in (1.0, -1.0):
                v = np.linalg.solve(A[[i, j]], np.array([1.0, sj]))
                if np.abs(A @ v).max() <= 1.0 + 1e-9:
                    L = max(L, float(v @ H0 @ v))
        best_poly = min(best_poly, L * float((A ** 2).sum(axis=1).max()) / lam)   # c_N^2 = max_k |a_k|^2
    if min(best_metric, best_poly) < kappa_F * (1.0 - 1e-9):
        raise SystemExit("a scanned norm has kappa_N < kappa_F, which contradicts Theorem 1")

    # storyboard values
    sb = STORYBOARD
    ok = (close(ev0[0] / lam, sb["eig0"][0], 1) and close(ev0[1] / lam, sb["eig0"][1], 1)
          and close(top_deg, sb["top_deg"], 1) and close(norms["l2"]["kappa_twin"], sb["kappa_twin"], 1)
          and all(close(norms[k]["kappa"], v, 0) for k, v in sb["kappa"].items())
          and all(close(norms[k]["local"], v, 1) for k, v in sb["local"].items())
          and all(close(norms[k]["sample"], v, 0) for k, v in sb["sample"].items()))
    if not ok:
        raise SystemExit("a value of the storyboard is not reproduced: "
                         + repr({k: (v["kappa"], v["local"], v["sample"]) for k, v in norms.items()}))

    def at(arr, deg):
        return float(arr[int(np.argmin(np.abs(phi - deg)))])
    curves = {}
    for key in ("l2", "H0"):
        v = norms[key]
        curves[key] = {"twin": h0 / v["n"] / v["mu_twin"], "upper": h0 / v["n"] / v["mu"],
                       "lower": lam / v["n"] / v["mu"], "local": hs / v["n"] / v["mu"]}
        curves[key]["arrows"] = {deg: (at(curves[key]["lower"], deg), at(curves[key]["upper"], deg))
                                 for deg in (top_deg, bot_deg)}
    return {"lam": lam, "ev0": ev0 / lam, "evs": evs / lam, "ws": ws, "gnorm": gnorm, "kappa_F": kappa_F,
            "top_deg": top_deg, "bot_deg": bot_deg, "phi": phi, "norms": norms, "curves": curves,
            "best_metric": float(best_metric), "best_poly": float(best_poly), "n_points": len(pts),
            "i_max": i_max, "margin": margins[RAY_OFFSET_DEG]}


def print_all(D):
    kF = D["kappa_F"]
    print(f"toy: n = {N}, K = 1, d = 2, lam = {D['lam']:.6e} (R^2 / lam = {KAPPA:g}); eigenvalues of D^2F(0) / lam: "
          f"{D['ev0'][0]:.2f}, {D['ev0'][1]:.2f}; kappa_F = {kF:.2f}; top eigendirection at {D['top_deg']:.2f} deg, "
          f"bottom one at {D['bot_deg']:.2f} deg")
    print(f"minimizer W* = ({D['ws'][0]:+.4f}, {D['ws'][1]:+.4f}), |grad| = {D['gnorm']:.1e}; eigenvalues of "
          f"D^2F(W*) / lam: {D['evs'][0]:.3f}, {D['evs'][1]:.2f}")
    print(f"brute force: {D['n_points']} points W (0 and {N_RAYS} rays, smallest |x_i . u| = {D['margin']:.5f}), "
          f"{N_PHI} directions: sup = curvature at 0, inf = lam, L_N and mu_N equal to the closed forms (checked)")
    print("norm: L_N / lam, mu_N / lam, kappa_N (ratio to kappa_F), condition number of D^2F(W*) in N, "
          "per-sample ratio L_N(f_i) / mu_N(F) for the sample of largest norm; twin: kappa_N")
    for key, v in D["norms"].items():
        print(f"    {v['label']:20s} {v['L'] / D['lam']:9.2f} {v['mu'] / D['lam']:7.4f} {v['kappa']:9.2f} "
              f"({v['kappa'] / kF:.4f}) {v['local']:7.3f} {v['sample']:9.1f}; twin {v['kappa_twin']:.3f}")
    for key in ("l2", "H0"):
        cv = D["curves"][key]
        print(f"drawn, {D['norms'][key]['label']}: twin curve {cv['twin'].min():.3f} to {cv['twin'].max():.3f}; "
              f"logistic lower edge {cv['lower'].min():.3f} to {cv['lower'].max():.3f}, upper edge "
              f"{cv['upper'].min():.2f} to {cv['upper'].max():.2f}; curvature at W* {cv['local'].min():.2f} to "
              f"{cv['local'].max():.2f}; band at " + "; ".join(f"{deg:.1f} deg: {lo:.3f} to {hi:.2f} (x {hi / lo:.2f})"
                                                              for deg, (lo, hi) in cv["arrows"].items()))
    print(f"scans (not drawn): smallest kappa_N / kappa_F over {SCAN_METRICS} random quadratic metrics "
          f"{D['best_metric'] / kF:.6f}, over {SCAN_POLY} random polyhedral norms {D['best_poly'] / kF:.4f} "
          f"(the storyboard quoted 1.0000 and 1.057 for its own random streams)")


# ----------------------------------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------------------------------
def draw(D):
    ps = es.setup("talk")
    ink = ps.INK
    COL = {"l2": ink["secondary"], "H0": ps.BLUE}          # Euclidean norm: ink; norm matched to a Hessian: blue
    NAME_OF = {"l2": "Euclidean", "H0": "matched"}
    fig = es.figure("talk", height=2.75)
    fig.get_layout_engine().set(w_pad=0.03, wspace=0.03)
    gs = fig.add_gridspec(1, 3, width_ratios=(0.86, 1.0, 1.0))
    ax_a = fig.add_subplot(gs[0, 0])
    ax_e = fig.add_subplot(gs[0, 1], sharey=ax_a)
    ax_m = fig.add_subplot(gs[0, 2], sharey=ax_a)
    b = es.Build()
    phi, kF = D["phi"], D["kappa_F"]
    bot_deg = D["bot_deg"]
    kw = dict(fontsize=es.small(), color=ink["secondary"], textcoords="offset points")

    # (a) the quadratic twin: one curvature per direction
    for state, key in ((1, "l2"), (2, "H0")):
        b.add(state, ax_a.plot(phi, D["curves"][key]["twin"], color=COL[key], lw=2.0, zorder=3 + state))
    k_twin = D["norms"]["l2"]["kappa_twin"]
    b.add(1, ax_a.annotate(rf"Euclidean: $\kappa_N = {k_twin:.1f}$", (0.0, k_twin), xytext=(3, 9), ha="left",
                           va="bottom", **kw))
    b.add(2, ax_a.annotate(rf"matched: $\kappa_N = {D['norms']['H0']['kappa_twin']:.0f}$", (180.0, 1.0),
                           xytext=(-3, -5), ha="right", va="top", **kw))
    ax_a.text(0.5, 0.80, "constant Hessian:\none curvature\nper direction", transform=ax_a.transAxes, ha="center",
              va="center", fontsize=es.small(), color=ink["secondary"], linespacing=1.2)
    es.panel_title(ax_a, "(a) least squares")

    # (b) the logistic model, one sub-panel per norm
    for state, key, ax in ((3, "l2", ax_e), (4, "H0", ax_m)):
        cv = D["curves"][key]
        b.add(state, ax.fill_between(phi, cv["lower"], cv["upper"], color=es.tint(COL[key], 0.16), lw=0, zorder=1))
        for edge in ("lower", "upper"):
            b.add(state, ax.plot(phi, cv[edge], color=COL[key], lw=2.0, zorder=4))
        b.add(state, ax.plot(phi, cv["local"], color=ink["primary"], lw=1.0, ls=(0, (4.0, 2.0)), zorder=3))
        for deg, (lo, hi) in cv["arrows"].items():
            b.add(state, es.arrow(ax, (deg, lo), (deg, hi), colour=ink["primary"], lw=1.0, style="<|-|>",
                                  mutation_scale=7.0, zorder=5))
            dy = -7 if deg == D["top_deg"] else 0          # keep the top label clear of the "at W_*" label above it
            b.add(state, ax.annotate(rf"$\times\,{hi / lo:.0f}$", (deg, np.sqrt(lo * hi)), xytext=(4, dy), ha="left",
                                     va="center", fontsize=es.small(), color=ink["primary"],
                                     textcoords="offset points", zorder=6))
        b.add(state, ax.annotate(rf"{NAME_OF[key]}: $\kappa_N = {D['norms'][key]['kappa']:.0f}$", (180.0, kF),
                                 xytext=(-3, 6), ha="right", va="bottom", **kw))
        ax.tick_params(labelleft=False)
    # where each curve of the band lives: the upper edge at W = 0, the dashed line at the minimizer, the lower edge
    # in the limit along rays
    b.add(3, ax_e.annotate(r"at $W = 0$", (bot_deg, 900.0), xytext=(0, 0), ha="center", va="center", **kw))
    b.add(3, ax_e.annotate("at $W_\\ast$\n(local)", (62.0, 125.0), xytext=(0, 0), ha="center", va="center",
                           linespacing=1.1, **kw))
    b.add(3, ax_e.annotate(r"floor $\lambda$, as $|W| \to \infty$", (180.0, 1.0), xytext=(-3, -5), ha="right",
                           va="top", **kw))
    b.add(4, ax_m.annotate("at $W_\\ast$\n(local)", (70.0, 190.0), xytext=(0, 0), ha="center", va="center",
                           linespacing=1.1, **kw))
    b.add(4, ax_m.annotate(r"$|W| \to \infty$", (bot_deg, 3.2), xytext=(0, 0), ha="center", va="center", **kw))
    es.panel_title(ax_e, "(b) logistic: the same two norms")
    for ax in (ax_a, ax_e, ax_m):
        ax.set_xlim(0.0, 180.0)
        ax.set_xticks([0, 90, 180])
        for tick, ha in zip(ax.get_xticklabels(), ("left", "center", "right")):   # keep 180 and the next 0 apart
            tick.set_ha(ha)
    ax_e.set_xlabel(r"direction of $\Delta$ (angle in degrees)")
    ax_a.set_yscale("log")
    ax_a.set_ylim(0.3, 6500.0)
    ax_a.set_yticks([1, 10, 100, 1000])
    ax_a.set_yticklabels(["1", "10", "100", "1000"])
    ax_a.minorticks_off()
    ax_a.set_ylabel("curvature / $N(\\Delta)^2$\n(unit: $\\mu_N$)")
    es.save(fig, NAME, "talk", build=b,
            note=(rf"toy: ridge logistic and its least-squares twin, $K = 1$, $d = 2$, $n = {N}$; matched: "
                  r"$N(\Delta)^2 = D^2F(0)[\Delta,\Delta]$; exact"))


def main():
    D = data()
    print_all(D)
    draw(D)


if __name__ == "__main__":
    main()
