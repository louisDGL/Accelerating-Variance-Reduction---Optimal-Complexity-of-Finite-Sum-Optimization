"""Figure T12 of the lecture (paper/talk/figures/figT12_offset_scaling): the limit of polar steps on minibatch
gradients is off the minimizer by an offset of order 1/b. Explanatory figure, not in the paper. It explains
Proposition 4 and Eq. (3) of paper/paper_saga_to_muon.tex, the paragraph "Numerical checks" and Figure 3(c).

Each panel shows b times the offset, i.e. b times the nuclear norm of the limit gradient G_b^dagger (a zero of the
mean field G -> E polar(G + xi_b), minus a nuclear median of the noise xi_b of a batch of b), against the batch size b.
The three panels have three different noise laws: their levels are not comparable, only the fact that each one
levels off (offset of order 1/b).

  (a) K = d = 1 (the polar factor is the sign, a nuclear median is a median): xi_b = the mean of b i.i.d. Exp(1) - 1.
      The median of xi_b is (median of Gamma(b, 1)) / b - 1, so b |G_b^dagger| = b - median of Gamma(b, 1), exact,
      for b = 1 ... 256; it tends to kappa_3 / (6 sigma^2) = 1/3 (sigma^2 = 1, kappa_3 = 2). Classical (the median of
      a mean, Cornish-Fisher), not a contribution of the paper. Computed in two ways (scipy.stats.gamma(b).median()
      and the root of the regularized incomplete Gamma function; b = 1: 1 - log 2).
  (b) 2 x 3 matrices with i.i.d. entries Exp(1) - 1: xi_b has the entries Gamma(b, 1)/b - 1 exactly.
      Line: the prediction of Eq. (3), b |G_b^dagger|_* -> (kappa_3 / (6 sigma^2)) (a_{2,3} / c_{2,3}) sqrt(6), with
      the constants a_{2,3} = -E[polar(Z)_11 (Z_11^3 - 3 Z_11)], c_{2,3} = E|Z|_* / 6 by the two-dimensional
      quadrature of muon/theory_checks/check_offset_first_order.py (two orders, and a Monte Carlo check; the script
      stops unless the limit is the 0.464 of the paper). The expansion is formal (one-term Edgeworth).
      Dots: the zero of the mean field computed here for b = 2 ... 64 from sampled matrices xi_b. The law of xi_b
      is invariant under permutations of the rows and of the columns and the polar factor commutes with them, so the
      set of minimizers of the convex function psi is convex and invariant, and contains a multiple g of the all-ones
      matrix J; g is the root of g -> mean over the sample of <polar(g J + xi_b), J>, increasing in g (Brent's
      method; the inner product is computed from five statistics of each sampled matrix, see polar_sum), and
      |G_b^dagger|_* = sqrt(6) |g|. REPS independent samples per b, of 20,000 b matrices each (at least 50,000), which
      keeps the standard error of b |G_b^dagger|_* near 0.002 for every b; the dot is their mean, the bar is +- 2
      standard errors (from the dispersion of the replicates; smaller than the dot at the scale of the panel). The
      zeros are within 3 % of the limit (printed), below it by about 2 to 3 % at b = 4 and 2: the remainder o(1/b).
      One root of the full 6-dimensional mean field (scipy root, b = B_CHECK) is compared with the root on the line
      (the script stops if their directions or their components along J disagree; the nuclear norm of the full root
      is larger by the sampling noise across the line, which is why the roots on the line are used). The polar
      factor of a 2 x 3 matrix M is computed as (M M^T)^(-1/2) M with the closed form of the square root of a 2 x 2
      matrix (checked against the SVD).
  (c) the finite sum of Figure 3(b)-(c) of the paper (unregularized least squares with an intercept and skewed
      residuals, n = 2000, K = 2, d = 3), read from muon/theory_checks/paper_figures_data.json (key noise_offsets):
      zeros of the mean field (line and small dots; bars: +- 2 x the half-difference of the two replicates stored,
      i.e. two standard errors estimated on one degree of freedom; b = 1 is exact), averaged runs with decreasing
      lengths (open circles; 64 chains, no error is stored), SAGA estimator with the nuclear-scaled length (squares,
      at rounding level). The gradients of a sample are of rank one: the noise does not have i.i.d. entries and
      formula (3) does not apply; nothing but the 1/b scaling is claimed.
      Checked against the numbers of the paper ("Numerical checks": 0.170, 0.17 to a plateau near 0.3, runs within
      5 % of the zeros, SAGA at rounding level); the script stops if the cache disagrees.

Status. (a) exact (classical). (b) the line is a formal expansion, the dots are a simulation with its Monte Carlo
error. (c) measured (repository data). The panel titles and the notes inside the panels say so.

Colours: the zeros of the mean field and the runs on minibatch gradients (no table) are blue, the SAGA table is
orange, the reference lines are in ink.

Progressive build (states s1-s3): one panel per state.

The script prints every number it plots. Run from the project root:
Run:    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/scripts/make_figT12_offset_scaling.py   (about 45 s)
Output: paper/talk/figures/figT12_offset_scaling.pdf and .png, and the states _s1 ... _s3
"""
from __future__ import annotations

import json
import os
import sys
from math import log, pi, sqrt

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
from scipy.optimize import brentq, root  # noqa: E402
from scipy.special import gamma as gamma_fn, gammainc  # noqa: E402
from scipy.stats import gamma as gamma_law  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import explain_style as es  # noqa: E402

NAME = "figT12_offset_scaling"
CACHE = os.path.join(es.ROOT, "muon", "theory_checks", "paper_figures_data.json")
B_SCALAR = (1, 2, 4, 8, 16, 32, 64, 128, 256)     # panel (a)
B_MATRIX = (2, 4, 8, 16, 32, 64)                  # panel (b)
REPS = 16                                         # panel (b): independent samples per b
SEED = 20261125
B_CHECK, S_CHECK = 8, 400_000                     # panel (b): the 6-dimensional root compared with the root on the line
S_CONST = 2_000_000                               # Monte Carlo check of the constants a, c
PAPER_LIMIT = 0.464                               # "Numerical checks" of the paper: b |G_b|_* -> 0.464


def sample_size(b):
    """Matrices per replicate: grows with b so that the standard error of b |G_b|_* stays about the same."""
    return max(50_000, 20_000 * b)


# ----------------------------------------------------------------------------------------------------------------------
# (a) the scalar case
# ----------------------------------------------------------------------------------------------------------------------
def scalar_rows():
    rows = []
    for b in B_SCALAR:
        med = float(gamma_law(b).median())
        med2 = brentq(lambda m: gammainc(b, m) - 0.5, 1e-9, 2.0 * b + 1.0, xtol=1e-13, rtol=1e-14)
        if abs(med - med2) > 1e-9 or (b == 1 and abs(med - log(2.0)) > 1e-12):
            raise SystemExit(f"(a) b = {b}: the two computations of the median of Gamma(b, 1) disagree")
        rows.append({"b": b, "b_off": b - med})
    return rows


# ----------------------------------------------------------------------------------------------------------------------
# (b) 2 x 3 matrices with i.i.d. entries
# ----------------------------------------------------------------------------------------------------------------------
def _gram(M):
    r1, r2 = M[:, 0], M[:, 1]
    a11, a22, a12 = np.einsum("si,si->s", r1, r1), np.einsum("si,si->s", r2, r2), np.einsum("si,si->s", r1, r2)
    s = np.sqrt(a11 * a22 - a12 * a12)                # sqrt(det(M M^T)); sqrt(A) = (A + s I) / t, t = sqrt(tr A + 2 s)
    f = np.sqrt(a11 + a22 + 2.0 * s) / ((a11 + s) * (a22 + s) - a12 * a12)
    return r1, r2, f * (a22 + s), f * (a11 + s), f * a12   # (M M^T)^(-1/2) = [[p, -q], [-q, r]]


def polar23(M):
    """Polar factor (M M^T)^(-1/2) M of a stack (S, 2, 3) of matrices of rank two."""
    r1, r2, p, r, q = _gram(M)
    out = np.empty_like(M)
    out[:, 0] = p[:, None] * r1 - q[:, None] * r2
    out[:, 1] = r[:, None] * r2 - q[:, None] * r1
    return out


def line_stats(xi):
    """Five numbers per matrix of the stack, from which <polar(xi + g J), J> follows for every g: the squared norms
    and the inner product of the two rows, and the two row sums."""
    r1, r2 = xi[:, 0], xi[:, 1]
    return (np.einsum("si,si->s", r1, r1), np.einsum("si,si->s", r2, r2), np.einsum("si,si->s", r1, r2),
            r1.sum(axis=1), r2.sum(axis=1))


def polar_sum(stats, g):
    """<polar(xi + g J), J> for each matrix xi of a stack with the statistics `stats` (J = all-ones 2 x 3 matrix):
    the Gram matrix of xi + g J is that of xi plus g (row sums) + 3 g^2, and <(M M^T)^(-1/2) M, J> is
    (p - q) (sum of row 1) + (r - q) (sum of row 2) in the notation of _gram."""
    n11, n22, n12, s1, s2 = stats
    g2 = 3.0 * g * g
    a11, a22, a12 = n11 + 2.0 * g * s1 + g2, n22 + 2.0 * g * s2 + g2, n12 + g * (s1 + s2) + g2
    s = np.sqrt(np.maximum(a11 * a22 - a12 * a12, 0.0))
    f = np.sqrt(a11 + a22 + 2.0 * s) / ((a11 + s) * (a22 + s) - a12 * a12)
    return f * ((a22 + s - a12) * (s1 + 3.0 * g) + (a11 + s - a12) * (s2 + 3.0 * g))


def quadrature(nq):
    """a_{2,3} and c_{2,3} by Gauss-Legendre quadrature (the formulas of check_offset_first_order.py): the rows of Z
    are rho_i w_i, |Z|_* = sqrt(rho_1^2 + rho_2^2 + 2 rho_1 rho_2 sqrt(1 - t^2)), t = <w_1, w_2> uniform on [-1, 1];
    a = -E[|Z|_* He_4(Z_11)] and E[He_4(rho u) | rho] = rho^4/5 - 2 rho^2 + 3; the radial integral is a Gamma
    function. Returns a, c and the total mass (which must be 1)."""
    def radial(k):                                    # int_0^infinity R^k exp(-R^2/2) dR
        return 2.0 ** ((k - 1) / 2.0) * gamma_fn((k + 1) / 2.0)
    g, w = np.polynomial.legendre.leggauss(nq)
    al, wal = (g + 1.0) * pi / 4.0, w * pi / 4.0
    ph, wph = g * pi / 2.0, w * pi / 2.0
    A, P = np.meshgrid(al, ph, indexing="ij")
    W2 = wal[:, None] * wph[None, :] * np.cos(P) / 2.0 * (2.0 / pi) * np.cos(A) ** 2 * np.sin(A) ** 2
    h = np.sqrt(1.0 + np.sin(2.0 * A) * np.cos(P))
    a = -np.sum(W2 * h * (radial(10) * np.cos(A) ** 4 / 5.0 - 2.0 * radial(8) * np.cos(A) ** 2 + 3.0 * radial(6)))
    c = np.sum(W2 * h * radial(6)) / 6.0
    return float(a), float(c), float(np.sum(W2 * radial(5)))


def constants():
    a40, c40, m40 = quadrature(40)
    a, c, mass = quadrature(80)
    if abs(a - a40) > 1e-9 or abs(c - c40) > 1e-9 or abs(mass - 1.0) > 1e-9 or abs(m40 - 1.0) > 1e-9:
        raise SystemExit("(b) the quadrature of a and c has not converged")
    rng = np.random.default_rng(SEED)
    Z = rng.standard_normal((S_CONST, 2, 3))
    U = polar23(Z)
    if np.abs(U[:1000] - np.einsum("sij,sjk->sik", *np.linalg.svd(Z[:1000], full_matrices=False)[::2])).max() > 1e-9:
        raise SystemExit("(b) the closed form of the polar factor disagrees with the SVD")
    v = -(U * (Z ** 3 - 3.0 * Z)).mean(axis=(1, 2))               # the six entries are exchangeable
    nuc = np.einsum("sij,sij->s", U, Z) / 6.0                     # |Z|_* = <polar(Z), Z>
    a_mc, a_se = float(v.mean()), float(v.std(ddof=1) / sqrt(S_CONST))
    c_mc, c_se = float(nuc.mean()), float(nuc.std(ddof=1) / sqrt(S_CONST))
    if abs(a_mc - a) > 4.0 * a_se or abs(c_mc - c) > 4.0 * c_se:
        raise SystemExit(f"(b) Monte Carlo and quadrature disagree: a {a_mc} / {a}, c {c_mc} / {c}")
    limit = (2.0 / 6.0) * (a / c) * sqrt(6.0)                     # kappa_3 / (6 sigma^2) = 1/3 for Exp(1) - 1
    if round(limit, 3) != PAPER_LIMIT:
        raise SystemExit(f"(b) the limit {limit:.5f} is not the {PAPER_LIMIT} of the paper")
    return {"a": a, "c": c, "limit": limit, "a_mc": a_mc, "a_se": a_se, "c_mc": c_mc, "c_se": c_se}


def root_on_line(xi, guess):
    """g with mean <polar(g J + xi), J> = 0 (the zero of the mean field of the sample, on the invariant line)."""
    stats = line_stats(xi)
    return brentq(lambda g: float(polar_sum(stats, g).mean()), -guess, 4.0 * guess, xtol=1e-13, rtol=1e-11)


def matrix_rows(C):
    rows = []
    for b in B_MATRIX:
        guess = C["limit"] / sqrt(6.0) / b
        vals = []
        for r in range(REPS):
            rng = np.random.default_rng([SEED, b, r])
            xi = rng.gamma(b, 1.0, size=(sample_size(b), 2, 3)) / b - 1.0
            vals.append(b * sqrt(6.0) * root_on_line(xi, guess))
        vals = np.array(vals)
        rows.append({"b": b, "b_off": float(vals.mean()), "se": float(vals.std(ddof=1) / sqrt(REPS)),
                     "n": REPS * sample_size(b)})
    return rows


def full_root_check(C):
    """One sample, b = B_CHECK: the root of the 6-dimensional mean field against the root on the line."""
    b = B_CHECK
    xi = np.random.default_rng([SEED, b, 999]).gamma(b, 1.0, size=(S_CHECK, 2, 3)) / b - 1.0
    guess = C["limit"] / sqrt(6.0) / b
    sub = xi[:2000] + guess                                       # the short formula against the polar factor itself
    if np.abs(polar_sum(line_stats(xi[:2000]), guess) - polar23(sub).sum(axis=(1, 2))).max() > 1e-9:
        raise SystemExit("(b) the formula for <polar(xi + g J), J> disagrees with the polar factor")
    sol = root(lambda g: polar23(xi + g.reshape(2, 3)).mean(axis=0).ravel(), np.full(6, guess), method="hybr",
               tol=1e-12)
    G = sol.x.reshape(2, 3)
    out = {"b": b, "full": float(b * np.linalg.svd(G, compute_uv=False).sum()),
           "proj": float(b * G.sum() / sqrt(6.0)),                # b x its component along J / |J|_F
           "line": float(b * sqrt(6.0) * root_on_line(xi, guess)),
           "cosine": float(G.sum() / (np.linalg.norm(G) * sqrt(6.0))), "residual": float(np.abs(sol.fun).max())}
    if (not sol.success or out["cosine"] < 0.999 or abs(out["proj"] - out["line"]) > 0.005
            or abs(out["full"] - out["line"]) > 0.02):
        raise SystemExit(f"(b) the full root and the root on the line disagree: {out}")
    return out


# ----------------------------------------------------------------------------------------------------------------------
# (c) the finite sum of Figure 3 of the paper
# ----------------------------------------------------------------------------------------------------------------------
def finite_sum_rows():
    with open(CACHE) as f:
        cache = json.load(f)
    rows, tr = cache["noise_offsets"]["rows"], cache["noise_trace"]
    out = [{"b": int(r["b"]), "b_off": r["b"] * r["root_mean"], "se": r["b"] * r["root_std"], "run": r["b"] * r["run"],
            "saga": r["b"] * r["saga"]} for r in rows]
    worst = max(abs(r["run"] / r["b_off"] - 1.0) for r in out)
    checks = {"n": tr["n"], "dist": tr["dist_lad_ls"], "worst": worst}
    ok = (tr["n"] == 2000 and [r["b"] for r in out] == [1, 2, 4, 8, 16, 32]
          and round(tr["dist_lad_ls"], 3) == 0.170                # "|W_LAD - W_LS|_F = 0.170"
          and round(out[0]["b_off"], 2) == 0.17                   # "rising from 0.17 ..."
          and abs(out[-1]["b_off"] - 0.3) < 0.02                  # "... to a plateau near 0.3"
          and worst < 0.05                                        # "the runs end within 5 % of the zero"
          and max(r["saga"] for r in out) < 1e-12)                # "ends at rounding level"
    if not ok:
        raise SystemExit(f"(c) the cache disagrees with the numbers of the paper: {checks}, {out}")
    return out, checks


def data():
    C = constants()
    return {"scalar": scalar_rows(), "const": C, "matrix": matrix_rows(C), "check": full_root_check(C),
            "sum": finite_sum_rows()}


def print_all(D):
    print("(a) K = d = 1, noise = mean of b draws of Exp(1) - 1: b x |median| = b - median of Gamma(b, 1); limit 1/3")
    print("    " + ", ".join(f"b = {r['b']}: {r['b_off']:.4f}" for r in D["scalar"]))
    C = D["const"]
    print(f"(b) constants (quadrature, 80 nodes): a = {C['a']:.6f}, c = {C['c']:.6f}, a/c = {C['a'] / C['c']:.5f}; "
          f"Monte Carlo ({S_CONST:,} Gaussian matrices): a = {C['a_mc']:.5f} (+- {C['a_se']:.5f}), c = {C['c_mc']:.5f} "
          f"(+- {C['c_se']:.5f}); Eq. (3): b |G_b|_* -> (1/3) (a/c) sqrt(6) = {C['limit']:.5f} (paper: {PAPER_LIMIT})")
    print("    zero of the mean field on the line of the all-ones matrix, b x nuclear norm (standard error; matrices):")
    for r in D["matrix"]:
        print(f"    b = {r['b']:2d}: {r['b_off']:.4f} (+- {r['se']:.4f}; {REPS} x {r['n'] // REPS:,}); "
              f"{(r['b_off'] - C['limit']) / r['se']:+.1f} standard errors from the limit")
    k = D["check"]
    worst = max(abs(r["b_off"] - C["limit"]) for r in D["matrix"]) / C["limit"]
    print(f"    largest gap between these zeros and the limit of Eq. (3): {100 * worst:.1f} % (standard errors "
          f"{100 * min(r['se'] for r in D['matrix']) / C['limit']:.1f} to "
          f"{100 * max(r['se'] for r in D['matrix']) / C['limit']:.1f} %)")
    print(f"    check, b = {k['b']}, one sample of {S_CHECK:,}: root on the line {k['line']:.4f}; full 6-dimensional root: "
          f"b x its component along the all-ones matrix {k['proj']:.4f}, b x its nuclear norm {k['full']:.4f} (the sampling "
          f"noise across the line adds to a nuclear norm), cosine with the all-ones matrix {k['cosine']:.5f}, residual "
          f"{k['residual']:.1e}")
    rows, chk = D["sum"]
    print(f"(c) finite sum of Figure 3 (n = {chk['n']}, |W_LAD - W_LS|_F = {chk['dist']:.4f}), b x nuclear norm of the "
          f"gradient at the limit point: zero of the mean field (half-difference of two replicates), runs, SAGA")
    for r in rows:
        print(f"    b = {r['b']:2d}: {r['b_off']:.4f} (+- {r['se']:.4f}), runs {r['run']:.4f}, SAGA {r['saga']:.1e}")
    print(f"    largest relative gap between the runs and the zeros: {chk['worst']:.3f}")


# ----------------------------------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------------------------------
def draw(D):
    ps = es.setup("talk")
    ink = ps.INK
    fig = es.figure("talk", height=2.6)
    fig.get_layout_engine().set(w_pad=0.03, wspace=0.03)
    ax_a, ax_b, ax_c = fig.subplots(1, 3)
    build = es.Build()
    ZERO = dict(color=ps.BLUE, lw=1.6, marker="o", ms=4.4, mec=ink["surface"], mew=0.8, elinewidth=1.1, capsize=0,
                zorder=4)
    ref = dict(color=ink["primary"], lw=1.0, zorder=2)
    txt = dict(fontsize=es.small(), color=ink["secondary"], linespacing=1.15, zorder=6)

    def axis(ax, bs, ticks, top):
        ax.set_xscale("log", base=2)
        ax.set_xlim(bs[0] / 1.35, bs[-1] * 1.35)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_ylim(0.0, top)
        ax.set_xlabel("batch size $b$")

    # (a) exact
    rows = D["scalar"]
    b = [r["b"] for r in rows]
    ax_a.axhline(1.0 / 3.0, **ref)
    ax_a.errorbar(b, [r["b_off"] for r in rows], **ZERO)
    axis(ax_a, b, [1, 4, 16, 64, 256], 0.5)
    ax_a.set_ylabel(r"$b\,\times$ offset $\|G_b^\dagger\|_*$")
    ax_a.text(b[-1], 1.0 / 3.0 + 0.012, r"$\kappa_3/(6\sigma^2) = 1/3$", ha="right", va="bottom", **txt)
    ax_a.text(0.05, 0.06, "classical: median of\na mean of $b$ draws", transform=ax_a.transAxes, ha="left",
              va="bottom", **txt)
    es.panel_title(ax_a, r"(a) $1\times 1$: exact")

    # (b) formal expansion, checked by simulation
    rows, C = D["matrix"], D["const"]
    b = [r["b"] for r in rows]
    ax_b.axhline(C["limit"], **ref)
    ax_b.errorbar(b, [r["b_off"] for r in rows], yerr=[2.0 * r["se"] for r in rows], **ZERO)
    axis(ax_b, b, [2, 4, 8, 16, 32, 64], 0.7)
    ax_b.text(b[-1], C["limit"] + 0.017, f"Eq. (3): {C['limit']:.3f}", ha="right", va="bottom", **txt)
    ax_b.text(0.05, 0.06, "i.i.d. entries: expansion,\nchecked by simulation", transform=ax_b.transAxes, ha="left",
              va="bottom", **txt)
    es.panel_title(ax_b, r"(b) $2\times 3$: formal")
    build.add(2, ax_b)

    # (c) the finite sum of the paper
    rows = D["sum"][0]
    b = [r["b"] for r in rows]
    zero = ax_c.errorbar(b, [r["b_off"] for r in rows], yerr=[2.0 * r["se"] for r in rows], **ZERO)
    runs, = ax_c.plot(b, [r["run"] for r in rows], ls="", marker="o", ms=7.4, mfc="none", mec=ps.BLUE, mew=1.2, zorder=5)
    saga, = ax_c.plot(b, [r["saga"] for r in rows], ls="", marker="s", ms=5.6, color=ps.ORANGE, mec=ink["surface"],
                      mew=0.8, clip_on=False, zorder=6)
    axis(ax_c, b, [1, 2, 4, 8, 16, 32], 0.5)
    ax_c.text(0.05, 0.97, f"measured, $n = {D['sum'][1]['n']}$;\nrank-one gradients:\nEq. (3) does not apply",
              transform=ax_c.transAxes, ha="left", va="top", **txt)
    ps.legend(ax_c, handles=[zero, runs, saga], labels=["mean-field zero", "averaged runs", "SAGA table"], loc="lower right",
              bbox_to_anchor=(1.0, 0.045), handlelength=1.3, handletextpad=0.5, borderpad=0.1, labelspacing=0.2,
              borderaxespad=0.1)
    es.panel_title(ax_c, "(c) a finite sum")
    build.add(3, ax_c)

    es.save(fig, NAME, "talk", build=build,
            note="Three noise laws: levels not comparable. (a), (b): entries Exp(1) − 1. Bars: ± 2 s.e. ((c): 2 "
                 "replicates)")


def main():
    D = data()
    print_all(D)
    draw(D)


if __name__ == "__main__":
    main()
