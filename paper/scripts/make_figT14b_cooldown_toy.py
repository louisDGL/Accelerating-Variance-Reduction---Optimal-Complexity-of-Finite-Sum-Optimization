"""Figure T14b of the lecture (paper/talk/figures/figT14b_cooldown_toy): steps of fixed length on a geometric cooldown,
on a toy in the plane. With a SAGA table the distance to the minimizer follows the current length down to the rounding
floor; the same schedule, with the same indices, fed by single-sample gradients freezes. Explanatory figure, not in
the paper. It illustrates (M5) and Proposition 5(c) of paper/paper_saga_to_muon.tex (Section 3.3, recipe (4)); the
certified schedule of Proposition 5(b) on the same toy is figT19b_certified_cooldown.

The toy (an exact instance of recipe (4) and of the hypotheses of Proposition 5; nothing is drawn by hand):
    F(W) = (1/n) sum_i f_i(W),   f_i(W) = |W - y_i|^2 / 2,   W in R^{2 x 1},   n = 200,
    y_i = +a_i and -a_i,  a = numpy.random.default_rng(0).standard_normal((100, 2)).
In the notation (GLM) of the paper: K = 2, d = 1, x_i = 1, least-squares loss, lambda = 0 (g_i = f_i). A 2 x 1 matrix
has one singular value: its spectral, nuclear and Frobenius norms are its Euclidean norm and its polar factor is
polar(M) = M / |M| (checked against the SVD). Hence mu_op = L_op = L_max = 1. The points are symmetric about 0, so the
mean and the geometric median of the y_i are both 0 = W_*: the mean of the single-sample polar steps vanishes at W_*
(no nuclear-median offset, Proposition 3); only the effect of the cooldown is seen.

The iteration, from W_0 = (3, 3)/sqrt(2), batch one, index j_t uniform in {1..n} (independent draws; one index
sequence per seed, shared by the estimators):
    W_{t+1} = W_t - lr_t polar(omega_t),     lr_t = LR0 * Q^(t/n)      (Q = 0.8, LR0 = 0.1: the same for all),
with one of three estimators
    exact gradient          omega_t = W_t - W_*                                  (control: is the path long enough?)
    single-sample gradient  omega_t = W_t - y_j                                  (no table)
    SAGA table              omega_t = zbar + (W_t - y_j) - z_j,  then z_j <- W_t - y_j;   z_i = W_0 - y_i at t = 0
(zbar = mean of the table, updated at every step and recomputed once per pass, as in muon/lab/vrmuon.py).
"Passes" are t/n; the full gradient at W_0 that fills the table is not counted.

Panels:
  (a) |W_t - W_*|^2 / |W_0 - W_*|^2 against passes: the guide lr_t^2 / |W_0 - W_*|^2; the exact gradient (largest value
      over each pass); the single-sample gradient and the SAGA table at the end of each pass (median and range of the
      seeds);
  (b) |W_t - W_*| / lr_t at the end of each pass (median and range of the seeds), with and without the table.

Checks done by the script (it stops if one fails):
  - polar(M) = M/|M| against the SVD; the mean of the y_i and the mean single-sample polar step at W_* vanish;
  - the first passes of the SAGA and single-sample loops against the project's implementation of the recipe
    (muon/lab/vrmuon.vr_muon on lab.problems.MatrixLSProblem(X = ones, Y = y, lam = 0); there the length of a step
    is lr * max(1, K/d)^(1/2), hence lr = LR0/sqrt(2));
  - the exact-gradient run against its scalar form r_{t+1} = |r_t - lr_t| (r = |W - W_*|), and the bound
    r_{t+1} <= lr_t from the first step with r_t <= lr_t on (proved by induction: if r_t <= lr_{t-1} then
    |r_t - lr_t| <= max(lr_t, lr_{t-1} - lr_t) <= lr_t, because lr_{t-1} <= 2 lr_t);
  - sum_t lr_t = LR0 / (1 - Q^(1/n)) against the sum of the schedule.

Two facts about this toy that the figure does not show and a reader should know (both printed):
  1. With the table the y_i cancel in omega_t = (W_t - phi_j) + mean_i phi_i (phi_i: the iterate stored in entry i):
     the loss Hessians are identical. In exact arithmetic the run then stays on the line through W_0 and W_*, where
     the polar factor is a sign. In double precision the rounding of z_i = phi_i - y_i takes the run off that line
     after a few passes (printed) and the run is then planar. The script also runs the exact one-dimensional
     reduction (table of positions on the line, no data): it follows the schedule in the same way and has no floor.
  2. With the exact gradient the distance r = |W - W_*| follows r <- |r - lr_t|: once r <= lr_t the iterate jumps
     across W_* at every step and stays within lr_t of it, which is all that is proved for this control. In double
     precision one of these planar jumps lands exactly on W_* (a rounding event: lr_t/|W_t| rounds to 1; polar(0) = 0,
     so the iterate stays there); the scalar recursion computed in double precision never reaches 0 (printed). The
     grey curve ends at that pass, which the figure states with the word "rounding".

Status. Proved: Proposition 5(c) (least squares, scale-invariant map: (W_t - W_*, table errors)/lr_t is a
time-homogeneous Markov chain under i.i.d. indices): in exact arithmetic no level of the distance plays a special role,
so a floor at a fixed level can only come from rounding (the proposition does not say that the chain stays bounded);
the bound of the exact-gradient control. Measured on the toy (an illustration, not a result of the paper): that the chain stays
bounded at Q = 0.8, i.e. that the distance follows lr_t; the level of the freeze without a table, which depends on the
schedule and on the problem (the paper measures 8 % to 34 % of the initial squared distance at batch one on its
instances, at another schedule); the rounding floor, which depends on the implementation of the table mean.
Proposition 5(b) certifies the law only at slower cooldowns (figT19b). The geometric form of the cooldown is the
choice of the paper (Muon's reference runs cool linearly).

Progressive build (states s1-s4): s1 the guide and the exact gradient; s2 adds the single-sample gradient; s3 adds the
SAGA table; s4 adds panel (b).

The script prints every number it plots. Nothing is read from the results of the project.
Run:    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/scripts/make_figT14b_cooldown_toy.py
Output: paper/talk/figures/figT14b_cooldown_toy.pdf and .png, and the states _s1 ... _s4
"""
from __future__ import annotations

import math
import os
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import explain_style as es  # noqa: E402

NAME = "figT14b_cooldown_toy"
N_HALF, POP_SEED = 100, 0                   # the points are +a_i and -a_i, a = default_rng(POP_SEED).standard_normal
N = 2 * N_HALF
W0 = np.array([3.0, 3.0]) / math.sqrt(2.0)  # starting point, at distance 3 of W_* = 0
Q, LR0 = 0.8, 0.1                           # lr_t = LR0 * Q^(t/n): the same schedule for the three estimators
PASSES = 200
SEEDS = (0, 1, 2, 3, 4, 5)                  # one index sequence per seed, shared by the estimators
LAB_PASSES = 3                              # cross-check against muon/lab on the first passes (seed SEEDS[0])
FROZEN_TOL = 0.01                           # "frozen": the squared distance stays within 1 % of its final value
FOLLOW = (10, 150)                          # passes over which the ratio distance / length is summarized


# ----------------------------------------------------------------------------------------------------------------------
# the toy
# ----------------------------------------------------------------------------------------------------------------------
def population():
    a = np.random.default_rng(POP_SEED).standard_normal((N_HALF, 2))
    return np.concatenate([a, -a])


def schedule(passes):
    return LR0 * Q ** (np.arange(passes * N) / N)


def indices(seed, passes):
    return np.random.default_rng(seed).integers(0, N, size=passes * N)


def run(Y, estimator, idx, lrs):
    """Polar steps of length lrs[t] on the given estimator ('exact', 'single', 'saga'). Returns the distance to
    W_* = 0 at the end of each pass, its largest value over each pass, the share of that distance that is off the
    line through W_0 and W_* (end of each pass), the iteration at which the iterate is exactly W_* for the first time
    (or None), and the iterates at the end of the first passes."""
    W = W0.copy()
    Z = W[None, :] - Y                                   # SAGA table: z_i = grad f_i(W_0)
    zbar = Z.sum(axis=0) / N
    end, top, big, landed, first = [float(np.linalg.norm(W))], [float(np.linalg.norm(W))], 0.0, None, []
    off = [0.0]
    for t in range(len(lrs)):
        j = idx[t]
        if estimator == "exact":
            omega = W
        elif estimator == "single":
            omega = W - Y[j]
        else:
            g = W - Y[j]
            c = g - Z[j]
            omega = zbar + c
            zbar = zbar + c / N
            Z[j] = g
        nrm = math.sqrt(omega[0] * omega[0] + omega[1] * omega[1])
        if nrm > 0.0:                                    # polar(0) = 0
            W = W - (lrs[t] / nrm) * omega
        dist = math.sqrt(W[0] * W[0] + W[1] * W[1])
        if dist == 0.0 and landed is None:
            landed = t + 1
        big = max(big, dist)
        if (t + 1) % N == 0:
            end.append(dist)
            top.append(big)
            off.append(abs(W[0] - W[1]) / math.sqrt(2.0) / dist if dist > 0.0 else 0.0)   # W_0 is on the diagonal
            big = 0.0
            if estimator == "saga":
                zbar = Z.sum(axis=0) / N                 # recomputed once per pass
            if len(first) < LAB_PASSES:
                first.append(W.copy())
    return {"end": np.array(end), "top": np.array(top), "off": np.array(off), "landed": landed, "first": first}


def run_line(idx, lrs):
    """The exact one-dimensional reduction of the SAGA run: s = coordinate of W on the line through W_0 and W_*,
    table of the stored positions phi_i, omega = s - phi_j + mean(phi), s <- s - lr sign(omega). No data enters."""
    s = float(np.linalg.norm(W0))
    phi = [s] * N
    pbar = s
    out = [abs(s)]
    for t in range(len(lrs)):
        j = int(idx[t])
        omega = s - phi[j] + pbar
        pbar += (s - phi[j]) / N
        phi[j] = s
        if omega > 0.0:
            s -= float(lrs[t])
        elif omega < 0.0:
            s += float(lrs[t])
        if (t + 1) % N == 0:
            out.append(abs(s))
            pbar = math.fsum(phi) / N
    return np.array(out)


# ----------------------------------------------------------------------------------------------------------------------
# checks
# ----------------------------------------------------------------------------------------------------------------------
def check_polar():
    """polar(M) = M/|M| for a 2 x 1 matrix, against the SVD; the three norms coincide."""
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(200):
        M = rng.standard_normal((2, 1)) * 10.0 ** rng.uniform(-12, 2)
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        worst = max(worst, float(np.abs(U @ Vt - M / np.linalg.norm(M)).max()),
                    abs(s.sum() / np.linalg.norm(M) - 1.0), abs(s.max() / np.linalg.norm(M) - 1.0))
    if worst > 1e-12:
        raise SystemExit(f"polar(M) = M/|M| fails against the SVD: {worst:.1e}")
    return worst


def check_exact(D):
    """The planar exact-gradient run against r_{t+1} = |r_t - lr_t|, and the bound r_{t+1} <= lr_t."""
    lrs = D["lrs"]
    r, reached, top, big, worst_bound = float(np.linalg.norm(W0)), None, [float(np.linalg.norm(W0))], 0.0, 0.0
    zero = None                                          # first iteration at which the scalar recursion is exactly 0
    for t in range(len(lrs)):
        if reached is None and r <= lrs[t]:
            reached = t
        r = abs(r - float(lrs[t]))
        if r == 0.0 and zero is None:
            zero = t + 1
        if reached is not None:
            worst_bound = max(worst_bound, r / float(lrs[t]))
        big = max(big, r)
        if (t + 1) % N == 0:
            top.append(big)
            big = 0.0
    top = np.array(top)
    planar = D["exact"]["top"]
    live = (planar > 0.0) & (top > 0.0)
    dev = float(np.abs(planar[live] / top[live] - 1.0).max())
    if worst_bound > 1.0 + 1e-12:
        raise SystemExit(f"exact gradient: r_(t+1) <= lr_t fails ({worst_bound})")
    if dev > 1e-3:
        raise SystemExit(f"exact gradient: the planar run and the scalar recursion disagree ({dev:.1e})")
    if D["exact"]["landed"] is not None and zero == D["exact"]["landed"]:
        raise SystemExit("the scalar recursion lands on W_* at the same iteration: revise the word 'rounding'")
    return {"reached": reached, "bound": worst_bound, "dev": dev, "passes_compared": int(live.sum()) - 1,
            "scalar_zero": zero}


def check_lab(Y, D):
    """The first passes of the SAGA and single-sample loops against muon/lab (the code of the paper's experiments)."""
    sys.path.insert(0, os.path.join(es.ROOT, "muon"))
    from lab.problems import MatrixLSProblem
    from lab.vrmuon import vr_muon
    prob = MatrixLSProblem(np.ones((N, 1)), Y, 0.0)
    prob.set_optimum(np.zeros((2, 1)))
    if prob.L_op != 1.0 or float(prob.L_i.max()) != 1.0 or abs(prob.cov_eigs[0] - 1.0) > 0:
        raise SystemExit("the constants of the toy in muon/lab are not L_op = L_max = mu_op = 1")
    out = {}
    for key, est in (("saga", "saga"), ("single", "sgd")):
        worst = []
        for p in range(1, LAB_PASSES + 1):
            res = vr_muon(prob, W0.reshape(2, 1), p * N, seed=SEEDS[0], record_every=N, estimator=est, batch=1,
                          placement="joint", length="const", orth="svd", lr=LR0 / math.sqrt(2.0),
                          schedule="geometric", lr_decay=Q, gauge=None)
            worst.append(float(np.abs(res["W"].ravel() - D[key][0]["first"][p - 1]).max()))
        out[key] = worst
        if max(worst) > 1e-9:
            raise SystemExit(f"{key}: the loop of the script and muon/lab disagree on the first passes: {worst}")
    return out


# ----------------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------------
def frozen_from(sq):
    """First pass from which every run stays within FROZEN_TOL (relative) of its final squared distance."""
    ok = np.all(np.abs(sq / sq[:, -1:] - 1.0) <= FROZEN_TOL, axis=0)
    bad = np.nonzero(~ok)[0]
    return int(bad[-1] + 1) if len(bad) else 0


def data():
    Y = population()
    lrs = schedule(PASSES)
    d0 = float(np.linalg.norm(W0))
    D = {"Y": Y, "lrs": lrs, "d0": d0, "passes": np.arange(PASSES + 1), "polar_check": check_polar()}
    D["mean"] = float(np.linalg.norm(Y.mean(axis=0)))
    D["mean_step_at_min"] = float(np.linalg.norm((-Y / np.linalg.norm(Y, axis=1)[:, None]).mean(axis=0)))
    if D["mean"] > 1e-15 or D["mean_step_at_min"] > 1e-15:
        raise SystemExit("the population is not symmetric: W_* = 0 or the zero of the mean single-sample step fails")
    D["total_sum"] = float(lrs.sum() + lrs[-1] * Q ** (1.0 / N) / (1.0 - Q ** (1.0 / N)))    # computed + geometric tail
    D["total_closed"] = LR0 / (1.0 - Q ** (1.0 / N))
    D["total_approx"] = LR0 * N / math.log(1.0 / Q)
    if abs(D["total_sum"] / D["total_closed"] - 1.0) > 1e-10:
        raise SystemExit("the total length of the schedule disagrees with its closed form")
    idx = {s: indices(s, PASSES) for s in SEEDS}
    D["exact"] = run(Y, "exact", idx[SEEDS[0]], lrs)
    D["single"] = [run(Y, "single", idx[s], lrs) for s in SEEDS]
    D["saga"] = [run(Y, "saga", idx[s], lrs) for s in SEEDS]
    D["line"] = np.array([run_line(idx[s], lrs) for s in SEEDS])
    D["exact_check"] = check_exact(D)
    D["lab_check"] = check_lab(Y, D)
    D["lr_end"] = LR0 * Q ** D["passes"]                                 # length of the step that follows pass p
    for key in ("single", "saga"):
        end = np.array([r["end"] for r in D[key]])
        D[key + "_sq"] = (end / d0) ** 2                                  # squared distance / initial, per seed
        D[key + "_ratio"] = end / D["lr_end"][None, :]                    # distance / current length
    D["leaves_line"] = [int(np.argmax(r["off"] > 0.01)) for r in D["saga"]]   # first pass with 1 % off the line
    if min(D["leaves_line"]) == 0:
        raise SystemExit("a run with the table never leaves the line through W_0 and W_*: revise the docstring")
    D["frozen_from"] = frozen_from(D["single_sq"])
    D["frozen_level"] = float(np.median(D["single_sq"][:, -1]))
    D["floor"] = float(np.median(D["saga_sq"][:, -1]))
    # the table follows the schedule until the pass where the median distance exceeds 3 lr_t for good (the floor)
    med_ratio = np.median(D["saga_ratio"], axis=0)
    above = np.nonzero(med_ratio[FOLLOW[0]:] <= 3.0)[0]
    D["floor_pass"] = int(FOLLOW[0] + above[-1] + 1)
    lo, hi = FOLLOW
    D["follow"] = {k: (float(v[:, lo:hi + 1].min()), float(v[:, lo:hi + 1].max()),
                       float(np.median(v[:, lo:hi + 1], axis=0).min()), float(np.median(v[:, lo:hi + 1], axis=0).max()))
                   for k, v in (("saga", D["saga_ratio"]), ("line", D["line"] / D["lr_end"][None, :]))}
    if D["follow"]["saga"][1] > 10.0 or D["follow"]["line"][1] > 10.0:
        raise SystemExit("with the table the distance does not stay within 10 lr_t on the passes summarized")
    if not (D["frozen_level"] > 1e-6 and D["single_sq"][:, -1].min() > 1e6 * D["saga_sq"][:, -1].max()):
        raise SystemExit("the single-sample runs do not freeze far above the runs with the table")
    return D


def print_all(D):
    lo, hi = FOLLOW
    print(f"toy: n = {N} points +-a_i in the plane, |mean| = {D['mean']:.1e}, |mean single-sample polar step at W_*| = "
          f"{D['mean_step_at_min']:.1e}; polar(M) = M/|M| against the SVD: {D['polar_check']:.1e}")
    print(f"schedule: lr_t = {LR0} * {Q}^(t/n); total length sum_t lr_t = {D['total_closed']:.3f} (sum: "
          f"{D['total_sum']:.3f}; lr_0 n / ln(1/q) = {D['total_approx']:.3f}) for a distance |W_0 - W_*| = {D['d0']:.4f}")
    e, c = D["exact"], D["exact_check"]
    print(f"exact gradient: r <= lr_t first at iteration {c['reached']}; afterwards max r_(t+1)/lr_t = {c['bound']:.6f} "
          f"(bound 1); planar run against the scalar recursion over {c['passes_compared']} passes: {c['dev']:.1e}; "
          + (f"the iterate is exactly W_* from iteration {e['landed']} (pass {e['landed'] / N:.2f})"
             if e["landed"] else "the iterate never lands exactly on W_*")
          + ("; the scalar recursion never reaches 0 (rounding event of the planar run)" if c["scalar_zero"] is None
             else f"; the scalar recursion reaches 0 at iteration {c['scalar_zero']}"))
    print("    largest squared distance over a pass / initial, passes 1, 2, 10, 50, 90: "
          + ", ".join(f"{(e['top'][p] / D['d0']) ** 2:.2e}" for p in (1, 2, 10, 50, 90))
          + "; guide lr^2/|W_0|^2 at the same passes: "
          + ", ".join(f"{(D['lr_end'][p] / D['d0']) ** 2:.2e}" for p in (1, 2, 10, 50, 90)))
    print(f"cross-check with muon/lab (seed {SEEDS[0]}), max |W - W_lab| after 1..{LAB_PASSES} passes: SAGA "
          + ", ".join(f"{v:.1e}" for v in D["lab_check"]["saga"]) + "; single sample "
          + ", ".join(f"{v:.1e}" for v in D["lab_check"]["single"]))
    s = D["single_sq"]
    print("(a) single-sample gradient, squared distance / initial at the end, per seed: "
          + ", ".join(f"{v:.2e}" for v in s[:, -1]) + f"; median {D['frozen_level']:.2e}; |W| per seed: "
          + ", ".join(f"{v:.4f}" for v in np.sqrt(s[:, -1]) * D["d0"])
          + f"; within {100 * FROZEN_TOL:g} % of the final value from pass {D['frozen_from']} on (every seed)")
    g = D["saga_sq"]
    print("    SAGA table, squared distance / initial (median [min, max] of the seeds) at passes "
          "10, 50, 100, 150, 160, 170, 180, 200: "
          + "; ".join(f"{np.median(g[:, p]):.1e} [{g[:, p].min():.1e}, {g[:, p].max():.1e}]"
                      for p in (10, 50, 100, 150, 160, 170, 180, 200)))
    print(f"    rounding floor: median {D['floor']:.2e} at pass {PASSES} (|W - W_*| = "
          + ", ".join(f"{v:.2e}" for v in np.sqrt(g[:, -1]) * D["d0"]) + f"); the median distance exceeds 3 lr_t for "
          f"good from pass {D['floor_pass']} on")
    for key, text in (("saga", "planar runs"), ("line", "one-dimensional reduction")):
        f = D["follow"][key]
        print(f"(b) SAGA table, {text}: |W - W_*|/lr_t at the end of passes {lo}..{hi}: all seeds in "
              f"[{f[0]:.3f}, {f[1]:.3f}], medians over the seeds in [{f[2]:.3f}, {f[3]:.3f}]")
    print("    planar runs with the table: first pass at which 1 % of the distance is off the line through W_0 and "
          "W_* (rounding), per seed: " + ", ".join(str(v) for v in D["leaves_line"]))
    print(f"    one-dimensional reduction at pass {PASSES}: |s|/lr = "
          + ", ".join(f"{v:.2f}" for v in D["line"][:, -1] / D["lr_end"][-1]) + " (no floor: no data, no rounding)")
    r = D["single_ratio"]
    print("    single sample: median |W - W_*|/lr_t at passes 10, 20, 40, 60: "
          + ", ".join(f"{np.median(r[:, p]):.3g}" for p in (10, 20, 40, 60))
          + f" (grows by 1/q = {1 / Q:.2f} per pass once frozen)")


# ----------------------------------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------------------------------
def sci(x, digits=1):
    """Mathtext of a number in scientific notation, e.g. 6.7e-5 -> 6.7 x 10^-5."""
    m, e = f"{x:.{digits}e}".split("e")
    return rf"{m}\cdot10^{{{int(e)}}}"


def draw(D):
    ps = es.setup("talk")
    ink = ps.INK
    NO, TB, EX = ps.BLUE, ps.ORANGE, ink["axis"]          # no table: blue; SAGA table: orange; exact gradient: grey
    fig = es.figure("talk", height=2.75)
    fig.get_layout_engine().set(w_pad=0.03, wspace=0.04)
    gs = fig.add_gridspec(1, 2, width_ratios=(1.5, 1.0))
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    b = es.Build()
    p = D["passes"]
    note = dict(fontsize=es.small(), color=ink["secondary"], linespacing=1.2)

    # (a) squared distance, relative to the initial one -------------------------------------------------------------
    # plotted in the order of the legend rows (top to bottom); the legend grows upwards with the states
    for state, key, colour, text in ((3, "saga_sq", TB, "SAGA table"), (2, "single_sq", NO, "single sample")):
        v = D[key]
        b.add(state, ax_a.fill_between(p, v.min(axis=0), v.max(axis=0), color=colour, alpha=0.22, lw=0, zorder=3))
        b.add(state, ax_a.plot(p, np.median(v, axis=0), color=colour, lw=1.9, zorder=4, label=text))
    e = D["exact"]
    last = PASSES if e["landed"] is None else (e["landed"] - 1) // N       # last pass with a positive maximum
    ax_a.plot(p[:last + 1], (e["top"][:last + 1] / D["d0"]) ** 2, color=EX, lw=4.2, zorder=2, solid_capstyle="butt",
              label="exact gradient")
    ax_a.plot(p, (D["lr_end"] / D["d0"]) ** 2, color=ink["primary"], lw=1.0, ls=(0, (4.0, 2.4)), zorder=6,
              label=r"$\mathrm{lr}_t^2\,/\,\|W_0-W_\ast\|^2$")
    ax_a.set_yscale("log")
    ax_a.set_ylim(1e-37, 30.0)
    ax_a.set_yticks([1.0, 1e-10, 1e-20, 1e-30])
    ax_a.minorticks_off()
    ax_a.set_xlim(0, PASSES)
    ax_a.set_xticks([0, 50, 100, 150, 200])
    ax_a.set_xlabel("passes")
    ax_a.set_ylabel(r"$\|W_t-W_\ast\|^2\,/\,\|W_0-W_\ast\|^2$")
    es.panel_title(ax_a, "(a) one schedule, three estimators")
    b.legend(ax_a, loc="lower left", handlelength=1.5, borderaxespad=0.2, labelspacing=0.3)
    right = PASSES - 4
    ax_a.text(right, 3e-10, f"total length of the steps: {D['total_closed']:.0f}\n"
              f"distance to cover: {D['d0']:.0f}", ha="right", va="center", **note)
    if e["landed"] is not None:                          # the text sits next to the end of the grey curve
        ax_a.text(right, 2e-18, "exact gradient: lands\n" + rf"on $W_\ast$ at pass {e['landed'] / N:.0f}" + "\n(rounding)",
                  ha="right", va="center", **note)
    b.add(2, ax_a.text(right, D["frozen_level"] * 12.0, rf"no table: frozen at ${sci(D['frozen_level'])}$",
                       ha="right", va="bottom", **note))
    b.add(3, ax_a.text(right, D["floor"] * 3e6, "rounding floor\n" + rf"${sci(D['floor'])}$", ha="right",
                       va="center", **note))

    # (b) distance / current length -------------------------------------------------------------------------------------
    ax_b.axhline(1.0, color=ink["axis"], lw=0.9, zorder=1)
    for key, colour, text in (("single_ratio", NO, "single sample"), ("saga_ratio", TB, "SAGA table")):
        v = D[key]
        ax_b.fill_between(p, v.min(axis=0), v.max(axis=0), color=colour, alpha=0.22, lw=0, zorder=3)
        ax_b.plot(p, np.median(v, axis=0), color=colour, lw=1.5, zorder=4, label=text)
    ax_b.set_yscale("log")
    ax_b.set_ylim(2e-3, 1e5)
    ax_b.set_yticks([1e-2, 1.0, 1e2, 1e4])
    ax_b.minorticks_off()
    ax_b.set_xlim(0, PASSES)
    ax_b.set_xticks([0, 100, 200])
    ax_b.set_xlabel("passes")
    ax_b.set_ylabel(r"$\|W_t-W_\ast\|\,/\,\mathrm{lr}_t$")
    es.panel_title(ax_b, "(b) distance / length")
    f = D["follow"]["saga"]
    ax_b.text(66, 2.2e4, "no table:\n" + r"$\times\,1/q$ per pass", ha="left", va="top", **note)
    ax_b.text(92, 25.0, f"SAGA table:\n{f[0]:.2f} to {f[1]:.1f}\n(passes {FOLLOW[0]}\u2013{FOLLOW[1]})", ha="center",
              va="center", **note)
    x_f = D["floor_pass"] + 15                           # a point of the rise that follows the floor
    y_f = float(np.median(D["saga_ratio"][:, x_f]))
    ax_b.text(PASSES - 4, 2.6e-3, "rounding floor", ha="right", va="bottom", **note)
    ax_b.plot([PASSES - 17, x_f + 0.6], [1.1e-2, 0.45 * y_f], color=ink["secondary"], lw=0.8, zorder=5,
              solid_capstyle="round")
    b.add(4, ax_b)
    es.save(fig, NAME, "talk", build=b,
            note=(rf"toy: $f_i(W)=\|W-y_i\|^2/2$, {N} planar points $\pm a_i$, batch one, "
                  rf"$\mathrm{{lr}}_t={LR0}\cdot{Q}^{{t/n}}$; median and range, {len(SEEDS)} seeds"))


def main():
    D = data()
    print_all(D)
    draw(D)


if __name__ == "__main__":
    main()
