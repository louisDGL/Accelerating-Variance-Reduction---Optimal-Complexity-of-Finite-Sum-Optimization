"""Figure T19b of the lecture (paper/talk/figures/figT19b_certified_cooldown): Proposition 5(b) of
paper/paper_saga_to_muon.tex in action. On the toy of figT14b_cooldown_toy, polar steps of the prescribed lengths, fed
by a SAGA table refreshed in the order of a fresh permutation per pass, stay under the proved bound at every
iteration; the bound is far from tight, and the cooldown it certifies is far slower than the one that works.
Explanatory figure, not in the paper. It was split from the three-panel figure of the storyboard (its panel (c)).

The toy (an exact instance of the hypotheses of Proposition 5; the same as in make_figT14b_cooldown_toy.py):
    F(W) = (1/n) sum_i f_i(W),   f_i(W) = |W - y_i|^2 / 2,   W in R^{2 x 1},   n = 200,
    y_i = +a_i and -a_i,  a = numpy.random.default_rng(0).standard_normal((100, 2)),   W_* = 0,   W_0 = (3, 3)/sqrt(2).
In the notation (GLM) of the paper: K = 2, d = 1, x_i = 1, least squares (c_l = 1), lambda = 0, so g_i = f_i,
L_i = L_max = Lbar = 1; for a 2 x 1 matrix the spectral and nuclear norms are the Euclidean norm and polar(M) = M/|M|,
so mu_op = L_op = 1 (and mu_op <= L_max, as the statement asks). D_t = F(W_t) - F_* = |W_t|^2 / 2 (checked against the
direct evaluation of F), D_0 = 4.5.

The statement, Proposition 5(b), with theta = 1 (SAGA), b = 1 and the table initialised at W_0 (phi_i = W_0):
    every entry at most A iterations old,   L_eff = L_op + 10 A L_max,   rho = 1 - mu_op / L_eff,
    eta_t = eta_0 rho^(t/2),   eta_0 = (2 mu_op D_0)^(1/2) / L_eff      ==>      D_t <= D_0 rho^t   for all t,
and A = 2n/b - 1 for a fresh permutation per pass (surely). Here A = 399, L_eff = 3991, eta_0 = 3/3991.

The runs (SEEDS: one sequence of permutations per seed):
    omega_t = zbar + (W_t - y_j) - z_j,   z_j <- W_t - y_j,   W_{t+1} = W_t - eta_t polar(omega_t),
with j_t read in a fresh numpy permutation per pass, on the certified schedule (PASSES passes) and, for comparison, on
the schedule of figT14b (lr_t = 0.1 * 0.8^(t/n), same permutations), which no statement certifies on this toy: its
factor per iteration, 0.8^(2/n) for lr^2, is below 1 - mu_op / L_eff even for the smallest admissible constant of
the theory notes (F13(c): L_eff >= L_op + 4 (L_max + Lbar) A); the script checks it.

Panels:
  (a) the constants of the statement on the toy (a text panel; the largest age observed appears with the runs);
  (b) the largest value of D_t / D_0 over each pass, against passes (t/n): the bound rho^t (dashed; also its largest
      value over each pass, rho^(n(p-1)+1) for pass p, so that the two curves compare like with like), the two runs on
      the certified schedule, the two runs on the fast schedule. The runs end on the rounding floor of the table
      (D/D_0 of a few 1e-34), below which nothing is compared.

Checks done by the script at EVERY iteration of the certified runs (it stops if one fails):
  - the bound D_t <= D_0 rho^t, compared while D_t / D_0 > FLOOR_GUARD (above the rounding floor of the run);
  - the recursion of the proof, D_{t+1} <= D_t - eta_t (2 mu_op D_t)^(1/2) + (L_eff / 2) eta_t^2 (same guard);
  - the age of the entry that is replaced, t - (iteration of its last replacement) <= A; an entry is oldest just
    before it is replaced, so this is the largest age of the run;
and, every CHECK_A iterations, Proposition 5(a): |omega - grad F(W)|_* <= (L_max + Lbar) max_i |W - phi_i|_op.

Status. Proved: the bound (Proposition 5(b); proof redone by the audit of the paper). The figure shows one instance
of it, checked numerically. Measured on the toy (illustration): the distance between the runs and the bound, and
the fact that the faster schedule is followed too. The ratios quoted for the instances of the paper (cooldown
3.3e4 to 8.9e4 times slower than q = 0.8, in log(1/q), for the permutation orders) are read in the paper (the script
stops if that sentence is not found in the source and in its corrected version, paper/audit/
paper_saga_to_muon_corrige.tex); they are not recomputed here. They cover both permutation orders on the two
non-square conditioning-limited instances (theory notes F13: cyclic 3.3e4 and 4.5e4, fresh permutation 6.5e4 and
8.9e4). Not covered by the proposition:
momentum, Newton-Schulz in place of the exact polar factor, independent sampling for all t (ages are then bounded
only with high probability over a finite horizon), and any other eta_0.

Progressive build (states s1-s3): s1 the constants and the bound; s2 the runs on the certified schedule; s3 the
runs on the fast schedule and the comparison of the cooldowns.

The script prints every number it plots.
Run:    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/scripts/make_figT19b_certified_cooldown.py
Output: paper/talk/figures/figT19b_certified_cooldown.pdf and .png, and the states _s1 ... _s3
"""
from __future__ import annotations

import math
import os
import re
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import explain_style as es  # noqa: E402

NAME = "figT19b_certified_cooldown"
N_HALF, POP_SEED = 100, 0                   # the toy of figT14b_cooldown_toy
N = 2 * N_HALF
W0 = np.array([3.0, 3.0]) / math.sqrt(2.0)
MU_OP = L_OP = L_MAX = L_BAR = 1.0          # constants of the toy (see the docstring)
SEEDS = (0, 1)
PASSES = 1400                               # certified schedule
Q_FAST, LR0_FAST, PASSES_FAST = 0.8, 0.1, 200    # the schedule of figT14b
FLOOR_GUARD = 1e-30                         # the inequalities are compared while D_t / D_0 is above this value
CHECK_A = 50                                # Proposition 5(a) is checked every CHECK_A iterations
PAPER = os.path.join(es.ROOT, "paper", "paper_saga_to_muon.tex")
PAPER_CORRECTED = os.path.join(es.ROOT, "paper", "audit", "paper_saga_to_muon_corrige.tex")
PAPER_RATIOS = (3.3e4, 8.9e4)               # read in the paper (Section 3.3), see paper_sentence()


# ----------------------------------------------------------------------------------------------------------------------
# the toy and the constants of Proposition 5(b)
# ----------------------------------------------------------------------------------------------------------------------
def population():
    a = np.random.default_rng(POP_SEED).standard_normal((N_HALF, 2))
    return np.concatenate([a, -a])


def constants():
    A = 2 * N - 1                                        # fresh permutation per pass, b = 1
    L_eff = L_OP + 10.0 * A * L_MAX
    rho = 1.0 - MU_OP / L_eff
    D0 = 0.5 * float(W0 @ W0)
    eta0 = math.sqrt(2.0 * MU_OP * D0) / L_eff
    return {"A": A, "L_eff": L_eff, "rho": rho, "D0": D0, "eta0": eta0,
            "per_pass_length": rho ** (N / 2.0), "per_pass_D": rho ** N,
            "slower": math.log(1.0 / Q_FAST) / (0.5 * N * math.log(1.0 / rho)),
            # the fast schedule against the weakest condition of the theory notes (F13(c)): 1 - rho <= mu_op / L_eff
            "fast_one_minus_rho": 1.0 - Q_FAST ** (2.0 / N),
            "largest_certifiable": MU_OP / (L_OP + 4.0 * (L_MAX + L_BAR) * A),
            "total_length": eta0 / (1.0 - math.sqrt(rho))}


def paper_sentence():
    """The sentence of the paper that compares the certified cooldown with q = 0.8 on its instances."""
    found = None
    for path in (PAPER, PAPER_CORRECTED):                # the source and its corrected version (audit) must agree
        if path == PAPER_CORRECTED and not os.path.exists(path):
            print("   (paper/audit/ is absent: the sentence is checked in the source of the paper only)")
            continue
        with open(path, encoding="utf-8") as fh:
            tex = " ".join(fh.read().split())
        m = re.search(r"\$3\.3\\cdot10\^4\$ to \$8\.9\\cdot10\^4\$ times slower, in \$\\log\(1/q\)\$, than \$q=0\.8\$",
                      tex)
        if m is None:
            raise SystemExit(f"the sentence 'its cooldown is 3.3e4 to 8.9e4 times slower ... than q = 0.8' is not in "
                             f"{os.path.relpath(path, es.ROOT)} any more: revise PAPER_RATIOS and the note of the figure")
        found = m.group(0)
    return found


def run(Y, seed, lengths, C=None):
    """Polar steps of the given lengths on the SAGA estimator, indices in a fresh permutation per pass. Returns the
    largest D_t / D_0 over each pass and, with the constants C of the certified schedule, the largest
    D_t / (D_0 rho^t) over each pass and the checks."""
    rng = np.random.default_rng(seed)
    W = W0.copy()
    Z = W[None, :] - Y                                   # table of gradients, z_i = grad g_i(W_0)
    Phi = np.repeat(W[None, :], N, axis=0)               # the iterates at which the entries were computed
    zbar = Z.sum(axis=0) / N
    last = np.zeros(N, dtype=np.int64)                   # iteration of the last replacement of each entry
    D0 = 0.5 * float(W @ W)
    D = D0
    top, top_ratio = [1.0], [1.0]
    chk = {"bound": 0.0, "bound_pass": 0.0, "recursion": -np.inf, "age": 0, "prop_a": 0.0, "compared": 0}
    rho = C["rho"] if C is not None else 1.0
    passes = len(lengths) // N
    t = 0
    for _ in range(passes):
        perm = rng.permutation(N)
        big, big_ratio = 0.0, 0.0
        for j in perm:
            eta = lengths[t]
            g = W - Y[j]
            c = g - Z[j]
            omega = zbar + c
            if C is not None:
                chk["age"] = max(chk["age"], t - int(last[j]))
                if t % CHECK_A == 0 and D / D0 > FLOOR_GUARD:
                    lhs = float(np.linalg.norm(omega - W))                      # grad F(W) = W - W_*
                    rhs = (L_MAX + L_BAR) * float(np.sqrt(((W[None, :] - Phi) ** 2).sum(axis=1)).max())
                    if lhs > rhs * (1.0 + 1e-9) + 1e-15:
                        raise SystemExit(f"Proposition 5(a) fails at iteration {t}: {lhs} > {rhs}")
                    if rhs > 0.0:
                        chk["prop_a"] = max(chk["prop_a"], lhs / rhs)
            zbar = zbar + c / N
            Z[j] = g
            Phi[j] = W
            last[j] = t
            nrm = math.sqrt(omega[0] * omega[0] + omega[1] * omega[1])
            if nrm > 0.0:
                W = W - (eta / nrm) * omega
            D_new = 0.5 * (W[0] * W[0] + W[1] * W[1])
            t += 1
            ratio = D_new / (D0 * rho ** t)
            big, big_ratio = max(big, D_new / D0), max(big_ratio, ratio)
            if C is not None and D_new / D0 > FLOOR_GUARD:
                chk["compared"] += 1
                chk["bound"] = max(chk["bound"], ratio)
                rec = D_new - (D - eta * math.sqrt(2.0 * MU_OP * D) + 0.5 * C["L_eff"] * eta * eta)
                chk["recursion"] = max(chk["recursion"], rec / D)
                if ratio > 1.0 + 1e-12 or rec > 1e-12 * D:
                    raise SystemExit(f"Proposition 5(b) fails at iteration {t}: D_t/(D_0 rho^t) = {ratio}, "
                                     f"recursion slack {rec}")
            D = D_new
        top.append(big)
        top_ratio.append(big_ratio)
        if C is not None and D / D0 > FLOOR_GUARD:
            chk["bound_pass"] = max(chk["bound_pass"], D / (D0 * C["rho"] ** t))
        zbar = Z.sum(axis=0) / N                         # recomputed once per pass
    if C is not None:
        chk["age"] = max(chk["age"], int((t - last).max()))                    # entries not replaced since
        if chk["age"] > C["A"]:
            raise SystemExit(f"an entry is {chk['age']} iterations old: more than A = {C['A']}")
    return np.array(top), np.array(top_ratio), chk


def data():
    Y = population()
    C = constants()
    # D = |W - W_*|^2 / 2: against the direct evaluation of F at W_0 and at W_*
    F = lambda W: float(0.5 * ((W[None, :] - Y) ** 2).sum(axis=1).mean())      # noqa: E731
    if abs((F(W0) - F(np.zeros(2))) / C["D0"] - 1.0) > 1e-12 or float(np.linalg.norm(Y.mean(axis=0))) > 1e-15:
        raise SystemExit("D_0 = |W_0|^2 / 2 disagrees with F(W_0) - F(W_*)")
    if not (MU_OP <= L_MAX and C["A"] == 399 and C["L_eff"] == 3991.0):
        raise SystemExit("the constants of Proposition 5(b) on the toy are not A = 399, L_eff = 3991")
    if not C["fast_one_minus_rho"] > C["largest_certifiable"]:
        raise SystemExit("the fast schedule would be certified: revise the label 'no certificate'")
    t = np.arange(PASSES * N)
    cert = C["eta0"] * C["rho"] ** (t / 2.0)
    fast = LR0_FAST * Q_FAST ** (np.arange(PASSES_FAST * N) / N)
    D = {"C": C, "paper": paper_sentence(), "passes": np.arange(PASSES + 1), "passes_fast": np.arange(PASSES_FAST + 1)}
    # the largest value of the bound over pass p (iterations n(p-1)+1 .. np), as for the runs; 1 at p = 0
    D["bound"] = C["rho"] ** np.maximum(N * (D["passes"].astype(float) - 1.0) + 1.0, 0.0)
    D["cert"], D["ratio"], D["chk"], D["fast"] = [], [], [], []
    for s in SEEDS:
        top, top_ratio, chk = run(Y, s, cert, C)
        D["cert"].append(top)
        D["ratio"].append(top_ratio)
        D["chk"].append(chk)
        D["fast"].append(run(Y, s, fast)[0])
    D["cert"], D["ratio"], D["fast"] = np.array(D["cert"]), np.array(D["ratio"]), np.array(D["fast"])
    # the plotted curves themselves: every per-pass maximum of a run is under the per-pass maximum of the bound
    if np.any(D["cert"] > D["bound"][None, :] * (1.0 + 1e-12)):
        raise SystemExit("a plotted run is above the plotted bound")
    # the stretch on which the runs follow the schedule: after the approach, before the rounding floor
    D["floor"] = float(np.median(D["cert"][:, -1]))
    follow = (D["passes"] >= 100) & np.all(D["cert"] > 1e4 * D["floor"], axis=0)
    D["follow_passes"] = (int(D["passes"][follow][0]), int(D["passes"][follow][-1]))
    r = D["ratio"][:, follow]
    D["gap"] = (float(np.median(r)), float(r.min()), float(r.max()))
    D["fast_floor"] = float(np.median(D["fast"][:, -1]))
    D["fast_pass_1e-20"] = [int(np.argmax(v < 1e-20)) for v in D["fast"]]
    D["cert_pass_1e-20"] = [int(np.argmax(v < 1e-20)) for v in D["cert"]]
    if not (D["gap"][2] < 1e-3 and max(D["fast_pass_1e-20"]) < min(D["cert_pass_1e-20"])):
        raise SystemExit("the runs are not far below the bound, or the fast schedule is not faster: revise the labels")
    return D


def print_all(D):
    C = D["C"]
    print(f"toy: n = {N}, mu_op = L_op = L_max = {MU_OP:g}, D_0 = {C['D0']:.4f}")
    print(f"Proposition 5(b): A = 2n - 1 = {C['A']}, L_eff = L_op + 10 A L_max = {C['L_eff']:.0f}, rho = 1 - 1/"
          f"{C['L_eff']:.0f} per iteration, eta_0 = (2 mu_op D_0)^(1/2)/L_eff = {C['eta0']:.4e}; per pass: length x "
          f"{C['per_pass_length']:.4f}, bound on D x {C['per_pass_D']:.4f}; total length sum_t eta_t = "
          f"{C['total_length']:.3f} for a distance of {math.sqrt(2 * C['D0']):.0f}")
    print(f"cooldown of the length: certified {C['per_pass_length']:.4f} per pass against q = {Q_FAST}: "
          f"log(1/q) / log(1/certified) = {C['slower']:.2f}; in the paper, on its two instances: {PAPER_RATIOS[0]:.1e} "
          f"to {PAPER_RATIOS[1]:.1e} (sentence found in the source: '{D['paper']}')")
    print(f"fast schedule: 1 - rho = 1 - q^(2/n) = {C['fast_one_minus_rho']:.2e} per iteration, above the largest "
          f"value that F13(c) can certify, mu_op / (L_op + 4 (L_max + Lbar) A) = {C['largest_certifiable']:.2e}: "
          f"no certificate")
    for s, chk, v in zip(SEEDS, D["chk"], D["cert"]):
        print(f"certified schedule, seed {s}: over {chk['compared']} iterations with D_t/D_0 > {FLOOR_GUARD:g}: "
              f"max D_t/(D_0 rho^t) = {chk['bound']:.5f} (bound 1; at the ends of the passes: {chk['bound_pass']:.3f}), "
              f"max of the excess over the recursion of the proof / D_t = {chk['recursion']:.2e} (must be <= 0), "
              f"largest age = {chk['age']} (A = {C['A']}), Proposition 5(a): max lhs/rhs = {chk['prop_a']:.3f}; "
              f"largest D/D_0 over passes 50, 200, 600, 1000, {PASSES}: "
              + ", ".join(f"{v[p]:.1e}" for p in (50, 200, 600, 1000, PASSES)))
    print(f"bound D_0 rho^t / D_0, largest value over passes 50, 200, 600, 1000, {PASSES}: "
          + ", ".join(f"{D['bound'][p]:.1e}" for p in (50, 200, 600, 1000, PASSES)))
    g = D["gap"]
    print(f"largest D_t/(D_0 rho^t) over each of the passes {D['follow_passes'][0]}..{D['follow_passes'][1]} (both seeds): median "
          f"{g[0]:.2e}, range [{g[1]:.1e}, {g[2]:.1e}]; rounding floor of the runs: D/D_0 = {D['floor']:.2e}; "
          f"first pass with D/D_0 < 1e-20: {D['cert_pass_1e-20']}")
    print(f"fast schedule lr_t = {LR0_FAST} * {Q_FAST}^(t/n), same permutations: largest D/D_0 over passes 20, 50, 100, 150, "
          f"{PASSES_FAST}: " + "; ".join(", ".join(f"{v[p]:.1e}" for p in (20, 50, 100, 150, PASSES_FAST))
                                         for v in D["fast"])
          + f"; floor {D['fast_floor']:.2e}; first pass with D/D_0 < 1e-20: {D['fast_pass_1e-20']}")


# ----------------------------------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------------------------------
def sci(x, digits=1):
    m, e = f"{x:.{digits}e}".split("e")
    return rf"{m}\cdot10^{{{int(e)}}}"


def draw(D):
    ps = es.setup("talk")
    ink = ps.INK
    TB = ps.ORANGE                                       # SAGA table inside polar steps
    C = D["C"]
    fig = es.figure("talk", height=2.75)
    fig.get_layout_engine().set(w_pad=0.03, wspace=0.02)
    gs = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.62))
    ax_t = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[0, 1])
    b = es.Build()
    note = dict(fontsize=es.small(), color=ink["secondary"], linespacing=1.25)
    age = max(c["age"] for c in D["chk"])

    # (a) the constants of the statement on the toy -----------------------------------------------------------------
    es.diagram_axes(ax_t)
    es.box(ax_t, 0.5, 0.5, 1.0, 1.0, "", rounding=0.03)
    lines = [   # (text, ink, state)
        (r"$D_t = F(W_t) - F_\ast$", "primary", 1),
        (r"$\mu_{\mathrm{op}} = L_{\mathrm{op}} = L_{\max} = 1$", "primary", 1),
        ("fresh permutation per pass:", "secondary", 1),
        (rf"ages $\leq A = 2n-1 = {C['A']}$", "primary", 1),
        (f"(largest age observed: {age})", "secondary", 2),
        (rf"$L_{{\mathrm{{eff}}}} = L_{{\mathrm{{op}}}} + 10\,A\,L_{{\max}} = {C['L_eff']:.0f}$", "primary", 1),
        (r"$\rho = 1 - \mu_{\mathrm{op}}/L_{\mathrm{eff}}$", "primary", 1),
        (r"$\eta_t = \eta_0\,\rho^{t/2}$", "primary", 1),
        (r"$\eta_0 = (2\mu_{\mathrm{op}}D_0)^{1/2}/L_{\mathrm{eff}}$", "primary", 1),
        (r"$\Rightarrow\ \ D_t \leq D_0\,\rho^{t}$ for all $t$", "primary", 1),
    ]
    for (text, tone, state), y in zip(lines, np.linspace(0.925, 0.075, len(lines))):
        t = ax_t.text(0.06, y, text, ha="left", va="center", fontsize=es.small(), color=ink[tone], zorder=5)
        if state > 1:
            b.add(state, t)
    es.panel_title(ax_t, "(a) Proposition 5(b), toy")

    # (b) D_t / D_0 against passes (largest value over each pass) ----------------------------------------------------
    p = D["passes"]
    ax.plot(p, D["bound"], color=ink["primary"], lw=1.2, ls=(0, (4.0, 2.4)), zorder=5)
    for v in D["cert"]:
        b.add(2, ax.plot(p, v, color=TB, lw=1.7, zorder=4))
    for v in D["fast"]:
        b.add(3, ax.plot(D["passes_fast"], v, color=TB, lw=1.1, zorder=3))
    ax.set_yscale("log")
    ax.set_ylim(1e-37, 30.0)
    ax.set_yticks([1.0, 1e-10, 1e-20, 1e-30])
    ax.minorticks_off()
    ax.set_xlim(0, PASSES)
    ax.set_xticks([0, 400, 800, 1200])
    ax.set_xlabel("passes")
    ax.set_ylabel(r"$D_t\,/\,D_0$ (max per pass)")
    es.panel_title(ax, "(b) the proved bound and the runs")
    ax.text(PASSES - 25, 1e-17, r"bound $\rho^{t}$ (proved)", ha="right", va="center", **note)
    # the gap between the runs and the bound, at a pass in the middle of the stretch where the runs follow
    x_g = 450
    y_run, y_bnd = float(np.sqrt(D["cert"][0, x_g] * D["cert"][1, x_g])), float(D["bound"][x_g])
    b.add(2, es.arrow(ax, (x_g, y_run * 8.0), (x_g, y_bnd / 8.0), style="<|-|>", lw=0.9, mutation_scale=7.0))
    b.add(2, ax.text(365, 2e-28, "certified lengths:\n" + rf"about ${sci(D['gap'][0], 0)}$ of the bound", ha="left",
                     va="top", **note))
    b.add(2, ax.text(PASSES - 25, 4e-36, "rounding floor", ha="right", va="center", **note))
    b.add(3, ax.text(160, 3e-21, rf"$\mathrm{{lr}}_t = {LR0_FAST}\cdot{Q_FAST}^{{t/n}}$:" + "\nno certificate",
                     ha="left", va="top", **note))
    b.add(3, ax.text(PASSES - 25, 3.0, rf"certified cooldown: $\times\,{C['per_pass_length']:.3f}$ per pass," + "\n"
                     + rf"here ${C['slower']:.1f}$ times slower than $q={Q_FAST}$" + "\n"
                     + "(instances of the paper, permutation\n"
                     + rf"orders: ${sci(PAPER_RATIOS[0])}$ to ${sci(PAPER_RATIOS[1])}$ times)",
                     ha="right", va="top", **note))
    es.save(fig, NAME, "talk", build=b,
            note=(rf"toy of fig. T14b ($n={N}$, batch one), SAGA table, fresh permutation per pass, exact polar; "
                  rf"{len(SEEDS)} seeds"))


def main():
    D = data()
    print_all(D)
    draw(D)


if __name__ == "__main__":
    main()
