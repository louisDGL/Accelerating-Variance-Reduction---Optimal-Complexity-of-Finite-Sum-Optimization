"""Figure T10 of the lecture (paper/talk/figures/figT10_sign_median): a sign step on a sampled gradient computes a
median; a SAGA table brings the mean back. Explanatory figure, not in the paper. It is the case K = d = 1 of
Theorem 2 and of Proposition 3(c)-(d) of paper/paper_saga_to_muon.tex, where the polar factor of a number is its sign,
the nuclear norm is the absolute value and a nuclear median is a median.

The toy (an exact instance of the statements, nothing is drawn by hand):
    F(w) = (1/n) sum_i f_i(w),  f_i(w) = (w - a_i)^2 / 2,  n = 201,
    a_i = the quantiles of Exp(1) at (i - 1/2)/n, centred: a skewed population with mean exactly 0.
The minimizer of F is the mean of the a_i (w* = 0). The gradient noise xi_i = f_i'(w) - F'(w) = -a_i is additive and
does not depend on w, as Theorem 2 assumes; mu_op = L_op = 1.

Panels (they share the vertical axis w):
  (a) the population: histogram of the a_i (those below W_HI; the 11 values above it, up to 5.0, are counted in a
      note at the top of the panel: they are the tail that pulls the mean above the median), with their mean and
      their median;
  (b) the mean step of the batch-one sign step at w, Phi(w) = (1/n) sum_i sign(w - a_i) (exact: enumeration of the n
      indices; the step itself is -eta Phi(w)). It is the mean field E polar(G + xi) of the paper at G = F'(w); it
      vanishes at the median of the a_i (exactly, n being odd), not at their mean, where it equals Phi(0) = 53/201 > 0
      (the mean step at the minimizer points down, towards the median);
  (c) two runs from w_0 = W0, against passes (n iterations):
        - "no table": w <- w - eta_t sign(f_i'(w)), i uniform, eta_t = ETA0 (1 + t/n)^(-3/4) (deterministic lengths
          with sum eta_t = infinity, sum eta_t^2 < infinity: Theorem 2(b), Proposition 3(d)); it converges to the
          median (the minimizer of the least-absolute-deviations objective (1/n) sum_i |w - a_i|);
        - "SAGA table": w <- w - lr_t sign(omega_t), omega_t = f_i'(w) - z_i + mean_j z_j, z_i <- f_i'(w),
          z_i = f_i'(w_0) at the start, lr_t = LR0 * Q^(t/n) (the recipe (4) of the paper in dimension one, exact
          polar factor, no momentum); in these runs it settles at the mean (measured: |w| <= 5.5e-3 after 20 passes),
          which is its only fixed point: at the minimizer with the table at the minimizer the estimator is 0 for
          every index (Proposition 3(c), which says nothing about attraction).
      The two runs use the same index sequence per seed but different length rules (each the one its statement or
      recipe uses); the legend says which. Passes are t/n; the initial full gradient of the table is not counted.
      Median over SEEDS seeds of the iterate at each recorded iteration, and the range over the seeds as a band.

Status. Exact: panel (a) and the curve of panel (b). Proved: the limit of the run without a table (Theorem 2(b) with
Med(xi) = {mean - median}, i.e. G_t = w_t -> median; also Proposition 3(d) with x_i = 1, y_i = a_i, whose
least-distances objective is (1/n) sum_i |w - a_i|; it is the classical convergence of the stochastic subgradient
method) and the fixed point of the run with a table (Proposition 3(c)). Measured, illustration only: that the run with
a table reaches that fixed point at this cooldown (q = 0.8); Proposition 5 certifies the mechanism at much slower
cooldowns and for the exact polar factor of matrices. The 20-pass runs without a table are still a random walk of step
eta_t (about 0.01 at the end) around the median: their limit is proved, not shown. The scope line on the figure says
which parts are proved and which are an illustration.

Progressive build (states s1-s4): s1 panel (a); s2 adds (b); s3 adds the run without a table; s4 adds the run with
the SAGA table.

The script prints every number it plots. Nothing is read from the results of the project.
Run:    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/scripts/make_figT10_sign_median.py
Output: paper/talk/figures/figT10_sign_median.pdf and .png, and the states _s1 ... _s4
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import explain_style as es  # noqa: E402

NAME = "figT10_sign_median"
N = 201                     # odd: the median is unique
W0 = 1.5                    # starting point of the runs
PASSES = 20
SEEDS = (0, 1, 2, 3, 4)
ETA0 = 0.1                  # run without a table: eta_t = ETA0 (1 + t/n)^(-3/4)
LR0, Q = 0.15, 0.8          # run with the table: lr_t = LR0 * Q^(t/n)
W_LO, W_HI = -1.1, 1.9      # window of the shared axis
RECORD = 10                 # the runs are recorded every RECORD iterations


# ----------------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------------
def population():
    u = (np.arange(N) + 0.5) / N
    a = -np.log1p(-u)
    return a - a.mean()


def mean_field(a, w):
    """Phi(w) = (1/n) sum_i sign(w - a_i) for an array of points w."""
    w = np.atleast_1d(w)
    return np.sign(w[:, None] - a[None, :]).mean(axis=1)


def run_no_table(a, seed):
    """Batch-one sign steps on the sampled gradient, decreasing deterministic lengths."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, N, size=PASSES * N)
    w, out = W0, [W0]
    for t in range(PASSES * N):
        w -= ETA0 * (1.0 + t / N) ** (-0.75) * np.sign(w - a[idx[t]])
        if (t + 1) % RECORD == 0:
            out.append(w)
    return np.array(out)


def run_table(a, seed):
    """Sign steps on the SAGA estimator, geometric cooldown of the length (the same indices as run_no_table)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, N, size=PASSES * N)
    w = W0
    z = np.full(N, W0) - a                      # z_i = f_i'(w_0)
    zbar = float(z.mean())
    out = [W0]
    for t in range(PASSES * N):
        i = idx[t]
        g = w - a[i]
        omega = g - z[i] + zbar
        zbar += (g - z[i]) / N
        z[i] = g
        w -= LR0 * Q ** (t / N) * np.sign(omega)
        if (t + 1) % RECORD == 0:
            out.append(w)
        if (t + 1) % N == 0:
            zbar = float(z.mean())              # recomputed once per pass, as the SAGA loops of the project do
    return np.array(out)


def data():
    a = population()
    mean, med = float(a.mean()), float(np.median(a))
    # closed form: the middle index has u = 1/2, so the median is ln 2 minus the mean of the uncentred quantiles
    med_closed = np.log(2.0) - float((-np.log1p(-(np.arange(N) + 0.5) / N)).mean())
    if abs(med - med_closed) > 1e-12 or abs(mean) > 1e-12:
        raise SystemExit(f"median {med} != closed form {med_closed}, or mean {mean} != 0")
    grid = np.linspace(W_LO, W_HI, 801)
    runs_no = np.array([run_no_table(a, s) for s in SEEDS])
    runs_tb = np.array([run_table(a, s) for s in SEEDS])
    passes = np.arange(runs_no.shape[1]) * RECORD / N
    tail = passes > 0.5 * PASSES
    return {"a": a, "mean": mean, "median": med, "grid": grid, "phi": mean_field(a, grid),
            "phi_at_mean": float(mean_field(a, np.array([mean]))[0]), "passes": passes,
            "no": runs_no, "tb": runs_tb,
            "no_tail": runs_no[:, tail].mean(axis=1), "tb_final": runs_tb[:, -1]}


def print_all(D):
    a = D["a"]
    print(f"population: n = {N}, mean {D['mean']:+.2e}, median {D['median']:+.4f}, variance {a.var():.4f}, "
          f"third central moment {((a - a.mean()) ** 3).mean():.4f}; {int((a > W_HI).sum())} values above {W_HI} "
          f"(max {a.max():.3f}) are outside panel (a)")
    print(f"(a) note: {int((a > W_HI).sum())} more up to {a.max():.1f}")
    print(f"(b) Phi(mean) = {D['phi_at_mean']:+.4f} = {int(round(D['phi_at_mean'] * N))}/{N}; Phi changes sign at the "
          f"median {D['median']:+.4f}, where it is {float(mean_field(a, np.array([D['median']]))[0]):+.1f}")
    print(f"(c) no table, eta_t = {ETA0} (1 + t/n)^(-3/4): mean of w over the last half of the run, per seed: "
          + ", ".join(f"{v:+.4f}" for v in D["no_tail"]) + f"  (median of the a_i: {D['median']:+.4f})")
    print(f"    SAGA table, lr_t = {LR0} * {Q}^(t/n): w after {PASSES} passes, per seed: "
          + ", ".join(f"{v:+.2e}" for v in D["tb_final"]) + f"  (mean of the a_i: 0; lr at the end "
          f"{LR0 * Q ** PASSES:.2e}; eta at the end {ETA0 * (1.0 + PASSES) ** (-0.75):.2e})")
    print(f"    legend: no table, eta_t = {ETA0:g} (1 + t/n)^(-3/4); SAGA table, lr_t = {LR0:g} * {Q:g}^(t/n)")


# ----------------------------------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------------------------------
def draw(D):
    ps = es.setup("talk")
    ink = ps.INK
    NO, TB = ps.BLUE, ps.ORANGE                   # no table (sampled gradient): blue; SAGA table: orange
    fig = es.figure("talk", height=2.6)
    fig.get_layout_engine().set(w_pad=0.03, wspace=0.03)
    gs = fig.add_gridspec(1, 3, width_ratios=(0.62, 0.92, 1.70))
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
    ax_c = fig.add_subplot(gs[0, 2], sharey=ax_a)
    b = es.Build()

    def references(ax, state=None):
        """The mean (the minimizer) and the median of the population, across the panel."""
        lines = [ax.axhline(D["mean"], color=ink["primary"], lw=1.0, zorder=1),
                 ax.axhline(D["median"], color=ink["primary"], lw=1.0, ls=(0, (4.0, 2.0)), zorder=1)]
        if state is not None:
            b.add(state, *lines)

    # (a) the population
    ax_a.hist(D["a"], bins=np.linspace(W_LO, W_HI, 31), orientation="horizontal", color=ink["axis"],
              edgecolor=ink["surface"], linewidth=0.6, zorder=2)
    references(ax_a)
    ax_a.set_ylim(W_LO, W_HI)
    n_hi = int((D["a"] > W_HI).sum())
    ax_a.annotate(f"+{n_hi} above,\nup to {D['a'].max():.1f}", (1.0, 1.0), xycoords="axes fraction",
                  xytext=(-2, -2), textcoords="offset points", ha="right", va="top", fontsize=es.small(),
                  color=ink["secondary"], linespacing=1.1, zorder=6,
                  bbox=dict(boxstyle="square,pad=0.15", fc=ink["surface"], ec="none"))
    ax_a.set_ylabel("$w$")
    ax_a.set_xlabel("count")
    ax_a.grid(axis="y", visible=False)
    es.panel_title(ax_a, r"(a) the $a_i$")

    # (b) the mean step of the batch-one sign step
    references(ax_b, 2)
    b.add(2, ax_b.axvline(0.0, color=ink["axis"], lw=0.8, zorder=1))
    b.add(2, ax_b.plot(D["phi"], D["grid"], color=NO, lw=1.8, zorder=3))
    b.add(2, ax_b.plot([0.0], [D["median"]], ls="", marker="o", ms=6.5, mfc=NO, mec=ink["surface"], mew=1.2,
                       zorder=5))
    b.add(2, ax_b.annotate("zero at\nthe median", (0.0, D["median"]), xytext=(8, -6), textcoords="offset points",
                           va="top", ha="left", fontsize=es.small(), color=ink["secondary"], linespacing=1.15))
    ax_b.set_xlim(-1.08, 1.08)
    ax_b.set_xticks([-1, 0, 1])
    ax_b.set_xlabel(r"mean of $\mathrm{sign}(w - a_i)$")
    ax_b.tick_params(labelleft=False)
    ax_b.grid(axis="y", visible=False)
    es.panel_title(ax_b, "(b) mean step")

    # (c) two runs
    references(ax_c)
    p = D["passes"]
    labels = (rf"no table, $\eta_t = {ETA0:g}\,(1 + t/n)^{{-3/4}}$",
              rf"SAGA table, $\mathrm{{lr}}_t = {LR0:g}\cdot{Q:g}^{{\,t/n}}$")
    for state, runs, colour, text in ((3, D["no"], NO, labels[0]), (4, D["tb"], TB, labels[1])):
        b.add(state, ax_c.fill_between(p, runs.min(axis=0), runs.max(axis=0), color=colour, alpha=0.18, lw=0,
                                       zorder=2))
        b.add(state, ax_c.plot(p, np.median(runs, axis=0), color=colour, lw=1.5, zorder=4, label=text))
    ax_c.set_xlim(0, PASSES)
    ax_c.set_xticks([0, 5, 10, 15, 20])
    ax_c.set_xlabel("passes")
    ax_c.tick_params(labelleft=False)
    ax_c.grid(axis="y", visible=False)
    es.panel_title(ax_c, "(c) where the sign step goes")
    b.legend(ax_c, loc="upper right", handlelength=1.4, borderaxespad=0.2)
    kw = dict(textcoords="offset points", fontsize=es.small(), color=ink["secondary"], ha="right")
    ax_c.annotate("mean of the $a_i$ = minimizer", (PASSES, D["mean"]), xytext=(-2, 5), va="bottom", **kw)
    ax_c.annotate("median of the $a_i$", (PASSES, D["median"]), xytext=(-2, -9), va="top", **kw)
    es.save(fig, NAME, "talk", build=b,
            note=(rf"toy: $f_i = (w - a_i)^2/2$, $n = {N}$; median, range of {len(SEEDS)} runs; proved: the end "
                  rf"points only"))


def main():
    D = data()
    print_all(D)
    draw(D)


if __name__ == "__main__":
    main()
