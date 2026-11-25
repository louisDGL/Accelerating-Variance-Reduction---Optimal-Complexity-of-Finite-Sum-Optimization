"""Figure T08 of the lecture (paper/talk/figures/figT08_worst_case): on its worst case the certificate of SAGA has the
right order, and on the logistic instance of the paper SAGA converges far beyond the steps that the proof admits.
Explanatory figure, not in the paper. It explains finding (C4) of paper/paper_saga_to_muon.tex as corrected by the
audit (paper/audit/AUDIT.md, findings A and B): the lower bound needs the starting point theta_0 = (r, r); SAGA
converges at gamma R^2 = 2, four times the largest step the proof admits, and stalls at gamma R^2 = 4.

Panel (a), an exact toy (an instance of Proposition "lower bound for SAGA with any constant step" of
theory/B3_prop59_tightness.tex; nothing is drawn by hand):
    f_i(theta) = (mu theta_1^2 + R^2 theta_2^2) / 2 for all i <= n,   n = 100,   R^2 = 1,   mu = 1 / kappa,
    theta_0 = (r, r),   table z_i = f_i'(theta_0),   SAGA with a constant step gamma, index uniform.
  The minimizer is 0 and each coordinate follows the scalar model of vrlab/order_model.py with g = gamma times its
  curvature, whose second moments (E e^2, E mean psi^2, E psibar^2, E e psibar) follow the exact linear recursion
  iid_moment_matrix(n, g) from (1, 1, 1, 1) e_0^2. Hence E|theta_t|^2 / |theta_0|^2 is the mean of the first moments of
  the two coordinates, and the number of iterations N(kappa, gamma) needed for E|theta_t|^2 <= eps |theta_0|^2
  (eps = 1e-6) is found with matrix powers (doubling, then bisection: N is the t with value(t - 1) > eps >= value(t);
  it is the first such t for kappa <= 1e3, checked by stepping the recursion). Against kappa = 10 ... 1e5:
    - the lower bound (kappa - 1)/4 log(1/(2 eps)), valid for every constant step (dashed);
    - N at the textbook step gamma R^2 = 1/4, and the smallest N over the grid gamma R^2 in GRID;
    - the certificate of Proposition 5.9 (textbook step): the smallest t with
      (1 - min{1/(3n), 3 mu/(16 R^2)})^t (1 + 3n/16) <= eps (vrlab.bounds.saga_bound).
  The legend gives the value of each curve divided by kappa at kappa = 1e5: these constants are those of this toy
  (n = 100, eps = 1e-6), not results of the paper.

Panel (b), data of the project: contraction exponent per pass of SAGA against the step gamma R^2 on l2-regularized
logistic regression, kappa = 1e3, alpha = 1, n = 1000, d = 50 (results/GammaDelta_summary.json):
    - observed (empirical_rates_per_pass): minus the slope of log |theta_t - eta_*|^2 over passes 2-40, mean of 3
      seeds, at gamma R^2 = 1/8, 1/4, 1/3, 1/2, 1, 2;
    - certified, Lyapunov weight Delta optimised at each step (certified_rates_per_pass at three steps; the curve is
      n max_c min{2 gamma mu (1 - gamma R^2 (1 + c/2)), (1 - 2/c)/n}, Corollary "parametrization Delta = c gamma^2" of
      the B3 note, computed here and checked against the three stored values and against optimal_within_proof);
      no Delta is admissible for gamma R^2 >= 1/2: the admissible steps are shaded;
    - the stored value at gamma R^2 = 4 (0.027) is NOT drawn as a rate: it is a slope fitted on a transient. The
      script reruns SAGA (vrlab.algorithms.saga, seeds 0-2) at gamma R^2 = 2 for 100 passes and at gamma R^2 = 4 for
      200 passes: it reproduces the two stored slopes, finds convergence to the rounding floor at 2 and a stall at 4
      (squared distance at a few percent of its initial value, no slope after pass 100), and marks 4 with a cross.
  The arrow "x 4.4" goes from the best certificate (optimal_within_proof.rate, attained at gamma R^2 = 0.219; it is
  1.6 % above the curve at 1/4) to the observed exponent at the textbook step, as in (C4).

Checks (the script stops if one fails): the recursion of (a) against a direct simulation of SAGA on the toy
(kappa = SIM_KAPPA, SIM_RUNS runs, at SIM_TIMES: the slow coordinate at every time, the stiff one while its second
moment is not carried by rare runs, within 4 standard errors); lower bound <= N <= certificate at every kappa;
the summary of (b) against results/digest.md; the certified curve against the stored values; the reruns against the
stored slopes; "4.4" and "17 %" of (C4).

Status. Proved: the lower bound, for theta_0 = (r, r) and the expected squared error; the certificate. Exact
computation on the toy: the two counts of SAGA. Measured (one data set, three sampling seeds): the observed exponents
of (b); convergence at gamma R^2 = 2 and the stall at 4 are the audit's finding A, rerun here on three seeds.
The step axes of (a) and (b) are not the same statement: on the toy (n = 100) the second moments of the stiff
coordinate grow at gamma R^2 = 2 (spectral radius of its moment matrix above 1, printed), where the logistic runs
converge.

Progressive build (states s1-s3): s1 panel (a) with the certificate and the lower bound; s2 adds the exact counts of
SAGA; s3 adds panel (b).

The script prints every number it plots. Reruns of (b): 6 runs on 2 worker processes; about 45 s in all.
Run:    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/scripts/make_figT08_worst_case.py
Output: paper/talk/figures/figT08_worst_case.pdf and .png, and the states _s1 ... _s3
"""
from __future__ import annotations

import json
import multiprocessing
import os
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import explain_style as es  # noqa: E402

if es.ROOT not in sys.path:
    sys.path.insert(0, es.ROOT)

NAME = "figT08_worst_case"
# panel (a): the toy
N_TOY, EPS = 100, 1e-6
KAPPAS = np.logspace(1.0, 5.0, 33)
GRID = (0.125, 0.25, 0.5, 0.75, 1.0)          # steps gamma R^2 of "the best step of a small grid"
TEXTBOOK = 0.25
BRUTE_KAPPAS = (10.0, 100.0, 1000.0)          # the count is checked by stepping the recursion for these kappa
SIM_KAPPA, SIM_RUNS, SIM_SEED = 100.0, 20000, 0
SIM_TIMES = (20, 50, 100, 200, 500, 1000, 2000)
SIM_MIN_SLOTS = 100                           # see data_toy: where the stiff coordinate is compared with the simulation
# panel (b): the logistic instance of the project
N, DIM, KAPPA_B, ALPHA_B = 1000, 50, 1.0e3, 1.0
STEPS_B = (0.125, 0.25, 1.0 / 3.0, 0.5, 1.0, 2.0, 4.0)
ADMISSIBLE = 0.5                               # the proof admits gamma R^2 < 1/2
RERUNS = ((4.0, 200), (2.0, 100))              # (gamma R^2, passes) of the reruns, seeds RERUN_SEEDS
RERUN_SEEDS = (0, 1, 2)
FIT = (2, 40)                                  # passes of the slope fits of the stored exponents
WORKERS = 2
DIGEST_HEADING = "## (gamma, Delta) inside the proof (n = 1000, kappa = 1e3, alpha = 1)"


# ----------------------------------------------------------------------------------------------------------------------
# panel (a): exact counts on the toy
# ----------------------------------------------------------------------------------------------------------------------
ONES = np.ones(4)


def _value(P_slow, P_stiff):
    """E|theta_t|^2 / |theta_0|^2 from the t-th powers of the two moment matrices (uniform memory, theta_0 = (r, r))."""
    return 0.5 * ((P_slow @ ONES)[0] + (P_stiff @ ONES)[0])


def toy_matrices(kappa, step):
    """Moment matrices of the two coordinates: g = gamma mu = step / kappa (slow) and g = gamma R^2 = step (stiff)."""
    from vrlab import order_model
    return order_model.iid_moment_matrix(N_TOY, step / kappa), order_model.iid_moment_matrix(N_TOY, step)


def toy_count(kappa, step, t_max=2 ** 36):
    """The t with value(t - 1) > EPS >= value(t), by doubling and bisection on matrix powers (inf if the second
    moments do not contract or if t_max is passed)."""
    T1, T2 = toy_matrices(kappa, step)
    if max(np.abs(np.linalg.eigvals(T1)).max(), np.abs(np.linalg.eigvals(T2)).max()) >= 1.0:
        return np.inf
    P1, P2 = [T1], [T2]
    while _value(P1[-1], P2[-1]) > EPS:
        if 2 ** len(P1) > t_max:
            return np.inf
        P1.append(P1[-1] @ P1[-1])
        P2.append(P2[-1] @ P2[-1])
    A1, A2, t = np.eye(4), np.eye(4), 0                     # largest t with value(t) > EPS
    for j in range(len(P1) - 1, -1, -1):
        B1, B2 = A1 @ P1[j], A2 @ P2[j]
        if _value(B1, B2) > EPS:
            A1, A2, t = B1, B2, t + 2 ** j
    return t + 1


def toy_count_brute(kappa, step, t_max=200000):
    """First t with value(t) <= EPS, by stepping the recursion (None beyond t_max)."""
    T1, T2 = toy_matrices(kappa, step)
    v1, v2 = ONES.copy(), ONES.copy()
    for t in range(1, t_max + 1):
        v1, v2 = T1 @ v1, T2 @ v2
        if 0.5 * (v1[0] + v2[0]) <= EPS:
            return t
    return None


def toy_certificate(kappa):
    """Smallest t at which the bound of Proposition 5.9 (textbook step, constant 1 + 3n/16) is below EPS |theta_0|^2."""
    from vrlab import bounds
    R2, mu, D0 = 1.0, 1.0 / kappa, 2.0
    rho = bounds.saga_rate(N_TOY, R2, mu)
    t = int(np.ceil(np.log((1.0 + 3.0 * N_TOY / 16.0) / EPS) / -np.log(rho)))
    while bounds.saga_bound(t, N_TOY, R2, mu, D0) > EPS * D0:          # guards against rounding at the boundary
        t += 1
    while t > 0 and bounds.saga_bound(t - 1, N_TOY, R2, mu, D0) <= EPS * D0:
        t -= 1
    return t


def toy_simulation(kappa, step):
    """Direct simulation of SAGA on the toy (SIM_RUNS independent runs, vectorized): mean and standard error of
    theta_1^2 / r^2 and theta_2^2 / r^2 at the iterations SIM_TIMES."""
    rng = np.random.default_rng(SIM_SEED)
    Q = np.array([1.0 / kappa, 1.0])                        # f_i'(theta) = Q theta (diagonal), R^2 = 1
    gamma = step
    S = SIM_RUNS
    theta = np.ones((S, 2))                                 # theta_0 = (r, r), r = 1
    z = np.tile(Q, (S, N_TOY, 1))                           # table z_i = f_i'(theta_0)
    zbar = z.mean(axis=1)
    rows = np.arange(S)
    out = {}
    for t in range(1, max(SIM_TIMES) + 1):
        i = rng.integers(0, N_TOY, size=S)
        g = theta * Q
        zi = z[rows, i]
        theta = theta - gamma * (g - zi + zbar)
        zbar = zbar + (g - zi) / N_TOY
        z[rows, i] = g
        if t % N_TOY == 0:
            zbar = z.mean(axis=1)                           # recomputed once per pass, as the SAGA loops of the project
        if t in SIM_TIMES:
            sq = theta ** 2
            out[t] = (sq.mean(axis=0), sq.std(axis=0, ddof=1) / np.sqrt(S))
    return out


def data_toy():
    lower = (KAPPAS - 1.0) / 4.0 * np.log(1.0 / (2.0 * EPS))
    counts = np.array([[toy_count(k, s) for s in GRID] for k in KAPPAS])
    textbook = counts[:, GRID.index(TEXTBOOK)]
    best = counts.min(axis=1)
    best_step = np.array([GRID[j] for j in counts.argmin(axis=1)])
    cert = np.array([toy_certificate(k) for k in KAPPAS], dtype=float)
    if not (np.all(np.isfinite(textbook)) and np.all(lower <= best) and np.all(textbook <= cert)):
        raise SystemExit("toy: lower bound <= exact count <= certificate fails at some kappa")
    brute = []
    for k in BRUTE_KAPPAS:
        for s in GRID:
            c = toy_count(k, s)
            if np.isfinite(c) and c <= 200000:
                cb = toy_count_brute(k, s)
                brute.append((k, s, int(c), cb))
                if cb != c:
                    raise SystemExit(f"toy: count by matrix powers {c} differs from the stepped recursion {cb} "
                                     f"(kappa = {k:g}, gamma R^2 = {s:g})")
    # the recursion against a direct simulation of SAGA
    from vrlab import order_model
    sim = toy_simulation(SIM_KAPPA, TEXTBOOK)
    mom = [order_model.iid_moments(N_TOY, g, max(SIM_TIMES))[:, 0] for g in (TEXTBOOK / SIM_KAPPA, TEXTBOOK)]
    rows, worst = [], 0.0
    for t in SIM_TIMES:
        mean, se = sim[t]
        exact = np.array([mom[0][t], mom[1][t]])
        zs = (mean - exact) / se
        # E theta_2^2 is carried by the runs that still hold a slot of the initial table (vrlab/order_model.py, (R1)):
        # the stiff coordinate is tested only while the SIM_RUNS runs hold at least SIM_MIN_SLOTS such slots in all
        tested = SIM_RUNS * N_TOY * (1.0 - 1.0 / N_TOY) ** t >= SIM_MIN_SLOTS
        worst = max(worst, float(abs(zs[0])), float(abs(zs[1])) if tested else 0.0)
        rows.append((t, exact, mean, se, zs, tested))
    if worst > 4.0:
        raise SystemExit(f"toy: the recursion and the simulation of SAGA differ by {worst:.1f} standard errors")
    radius = {s: float(np.abs(np.linalg.eigvals(order_model.iid_moment_matrix(N_TOY, s))).max()) for s in (1.0, 2.0)}
    if not (radius[1.0] < 1.0 < radius[2.0]):
        raise SystemExit(f"toy: unexpected spectral radii of the stiff coordinate: {radius}")
    big = -1                                                # kappa = 1e5: the constants of the legend
    return {"lower": lower, "textbook": textbook, "best": best, "best_step": best_step, "cert": cert, "counts": counts,
            "brute": brute, "sim": rows, "sim_worst": worst, "radius": radius,
            "per_kappa": {"cert": cert[big] / KAPPAS[big], "textbook": textbook[big] / KAPPAS[big],
                          "best": best[big] / KAPPAS[big], "lower": lower[big] / KAPPAS[big]}}


# ----------------------------------------------------------------------------------------------------------------------
# panel (b): the logistic instance
# ----------------------------------------------------------------------------------------------------------------------
def certified_per_pass(step, n, kappa):
    """n max over c > 2 of min{2 gamma mu (1 - gamma R^2 (1 + c/2)), (1 - 2/c)/n} at gamma R^2 = step (0 if no Delta
    = c gamma^2 is admissible): the first term decreases with c and the second increases, so the maximum is where
    they cross."""
    if step >= ADMISSIBLE:
        return 0.0
    def diff(c):
        return 2.0 * (step / kappa) * (1.0 - step * (1.0 + 0.5 * c)) - (1.0 - 2.0 / c) / n
    c = brentq(diff, 2.0, 2.0 / step - 2.0, xtol=1e-14, rtol=1e-14)
    return float(n * (1.0 - 2.0 / c) / n)


def rerun(task):
    """One run of SAGA on the logistic instance at the step gamma R^2 (the call of experiments/exp_gamma_delta.py,
    with more passes; nothing is cached). Returns the recorded passes and the relative squared distance."""
    step, passes, seed = task
    from vrlab import algorithms, problems, solvers
    prob = problems.make_logistic(N, DIM, kappa=KAPPA_B, alpha=ALPHA_B, seed=0)
    solvers.solve_optimum(prob)
    r = algorithms.saga(prob, np.zeros(DIM), gamma=step / prob.R2, passes=passes, seed=seed,
                        record_every=max(1, N // 10), record_F=False, stop_D=1e-26)
    it, D = np.asarray(r["iters"], dtype=float), np.asarray(r["D"], dtype=float)
    return {"step": step, "seed": seed, "passes": it / N, "D": D, "D0": float(r["D0"]), "R2": float(prob.R2),
            "mu": float(prob.mu), "stopped": r["stopped"]}


def slope(run, lo, hi):
    """Minus the slope per pass of log |theta - eta_*|^2 over the passes [lo, hi] (the fit of exp_gamma_delta.py)."""
    p, D = run["passes"], run["D"]
    sel = (p >= lo) & (p <= hi) & (D > 1e-25) & np.isfinite(D)
    return float(-np.polyfit(p[sel], np.log(D[sel]), 1)[0])


def digest_rows():
    """Table of the (gamma, Delta) section of results/digest.md: {gamma R^2 as printed: (observed, certified)}."""
    with open(os.path.join(es.ROOT, "results", "digest.md"), encoding="utf-8") as f:
        lines = f.read().splitlines()
    found = [j for j, line in enumerate(lines) if line.strip() == DIGEST_HEADING]
    if len(found) != 1:
        raise SystemExit(f"results/digest.md: heading {DIGEST_HEADING!r} found {len(found)} times")
    rows = {}
    for line in lines[found[0] + 1:]:
        if line.startswith("## "):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and len(cells) == 3 and cells[0][:1].isdigit():
            rows[cells[0]] = (cells[1], cells[2])
    return rows


def digest_fmt(x):
    """Number format of experiments/make_digest.py."""
    if x == 0:
        return "0"
    if abs(x) >= 1e4 or abs(x) < 1e-3:
        return f"{x:.3g}"
    return f"{x:.4g}"


def data_logistic():
    with open(os.path.join(es.ROOT, "results", "GammaDelta_summary.json"), encoding="utf-8") as f:
        S = json.load(f)
    obs = {float(k): v for k, v in S["empirical_rates_per_pass"].items()}
    cert = {float(k): v for k, v in S["certified_rates_per_pass"].items()}
    prob, opt = S["problem"], S["optimal_within_proof"]
    if sorted(obs) != sorted(STEPS_B) or prob["n"] != N or not np.isclose(prob["kappa"], KAPPA_B):
        raise SystemExit("GammaDelta_summary.json: unexpected steps or problem")
    dig = digest_rows()
    for k in S["empirical_rates_per_pass"]:
        if dig.get(k) != (digest_fmt(obs[float(k)]), digest_fmt(cert[float(k)])):
            raise SystemExit(f"gamma R^2 = {k}: the summary ({obs[float(k)]}, {cert[float(k)]}) is not the row of "
                             f"results/digest.md {dig.get(k)}")
    # the certified curve (Delta optimised at each step) against the stored values and the stored optimum
    kappa = prob["R2"] / prob["mu"]
    for s in STEPS_B:
        mine = certified_per_pass(s, N, kappa)
        if not np.isclose(mine, cert[s], rtol=3e-3, atol=1e-12):        # the stored values come from a grid in c
            raise SystemExit(f"certified exponent at gamma R^2 = {s:g}: {mine} here, {cert[s]} stored")
    best_step = opt["gamma"] * prob["R2"]
    best_cert = opt["rate"] * N
    closed = 2.0 * N / (4.0 * kappa + N + np.sqrt(N ** 2 + 16.0 * kappa ** 2))   # Proposition "optimal constants"
    if not (np.isclose(best_cert, closed, rtol=2e-3) and np.isclose(certified_per_pass(best_step, N, kappa), best_cert,
                                                                   rtol=2e-3)):
        raise SystemExit("the stored optimum within the proof is not reproduced")
    xs = np.exp(np.linspace(np.log(0.08), np.log(ADMISSIBLE), 400))
    curve = np.array([certified_per_pass(x, N, kappa) for x in xs])
    ratio = obs[TEXTBOOK] / best_cert
    gain = opt["gain_over_textbook"]
    if f"{ratio:.1f}" != "4.4" or round(100 * (gain - 1.0)) != 17:
        raise SystemExit(f"(C4) is not reproduced: observed / best certificate = {ratio:.3f}, gain {gain:.4f}")
    # reruns at gamma R^2 = 4 and 2
    tasks = [(s, p, seed) for s, p in RERUNS for seed in RERUN_SEEDS]
    with multiprocessing.get_context("fork").Pool(WORKERS) as pool:
        runs = pool.map(rerun, tasks, chunksize=1)
    rr = {}
    for s, passes in RERUNS:
        mine = [r for r in runs if r["step"] == s]
        if not all(np.isclose(r["R2"], prob["R2"], rtol=1e-12) and np.isclose(r["mu"], prob["mu"], rtol=1e-12)
                   for r in mine):
            raise SystemExit("the rebuilt logistic problem differs from GammaDelta_summary.json")
        fit = float(np.mean([slope(r, *FIT) for r in mine]))
        if not np.isclose(fit, obs[s], rtol=1e-6):
            raise SystemExit(f"gamma R^2 = {s:g}: slope over passes {FIT[0]}-{FIT[1]} {fit!r}, stored {obs[s]!r}")
        rr[s] = {"fit": fit, "passes": passes,
                 "final": [float(r["D"][-1] / r["D0"]) for r in mine],
                 "at40": [float(r["D"][np.argmin(np.abs(r["passes"] - 40))] / r["D0"]) for r in mine],
                 "late": [slope(r, 100, passes) for r in mine] if passes > 100 else None,
                 "median_late": [float(np.median(r["D"][r["passes"] >= 100] / r["D0"])) for r in mine]
                 if passes > 100 else None,
                 "last_pass": [float(r["passes"][-1]) for r in mine]}
    stall, conv = rr[4.0], rr[2.0]
    if not (min(stall["final"]) > 1e-3 and max(abs(v) for v in stall["late"]) < 0.01):
        raise SystemExit(f"gamma R^2 = 4: no stall in the reruns: {stall}")
    if not max(conv["final"]) < 1e-20:
        raise SystemExit(f"gamma R^2 = 2: no convergence in the reruns: {conv}")
    return {"obs": obs, "cert": cert, "xs": xs, "curve": curve, "best_step": float(best_step),
            "best_cert": float(best_cert), "ratio": float(ratio), "gain": float(gain), "kappa": float(kappa),
            "textbook_cert": float(opt["rate_textbook"] * N), "reruns": rr}


def data():
    return {"a": data_toy(), "b": data_logistic()}


def print_all(D):
    A, B = D["a"], D["b"]
    print(f"(a) toy: n = {N_TOY} identical quadratics (mu theta_1^2 + R^2 theta_2^2)/2, theta_0 = (r, r), eps = {EPS:g}; "
          f"iterations to E|theta_t|^2 <= eps |theta_0|^2")
    print("    kappa | lower bound | SAGA, gamma R^2 = 1/4 | SAGA, best of the grid [step] | certificate | "
          "ratios: 1/4 over lower, certificate over 1/4, certificate over lower")
    for j, k in enumerate(KAPPAS):
        if j % 4 == 0:
            print(f"    {k:8.0f} | {A['lower'][j]:11.1f} | {A['textbook'][j]:21.0f} | {A['best'][j]:20.0f} "
                  f"[{A['best_step'][j]:g}] | {A['cert'][j]:11.0f} | {A['textbook'][j] / A['lower'][j]:.2f}, "
                  f"{A['cert'][j] / A['textbook'][j]:.2f}, {A['cert'][j] / A['lower'][j]:.1f}")
    print(f"    grid of steps gamma R^2: {GRID}; counts at kappa = {KAPPAS[-1]:g}: "
          + ", ".join(f"{s:g}: {c:.0f}" for s, c in zip(GRID, A["counts"][-1])))
    print("    spectral radius of the moment matrix of the stiff coordinate: "
          + ", ".join(f"{v:.6f} at gamma R^2 = {s:g}" for s, v in A["radius"].items())
          + " (above 1: its second moments grow; the logistic runs of (b) converge at gamma R^2 = 2)")
    pk = A["per_kappa"]
    print(f"    count / kappa at kappa = {KAPPAS[-1]:g} (legend): certificate {pk['cert']:.2f}, gamma R^2 = 1/4 "
          f"{pk['textbook']:.2f}, best of the grid {pk['best']:.2f}, lower bound {pk['lower']:.2f}")
    print("    counts checked by stepping the recursion (kappa, gamma R^2, count): "
          + "; ".join(f"({k:g}, {s:g}, {c})" for k, s, c, _ in A["brute"]))
    print(f"    recursion against {SIM_RUNS} simulated runs of SAGA (kappa = {SIM_KAPPA:g}, gamma R^2 = 1/4): "
          f"largest deviation {A['sim_worst']:.2f} standard errors")
    for t, exact, mean, se, zs, tested in A["sim"]:
        print(f"      t = {t:5d}: E theta_1^2 exact {exact[0]:.6e}, simulated {mean[0]:.6e} +- {se[0]:.1e}; "
              f"E theta_2^2 exact {exact[1]:.3e}, simulated {mean[1]:.3e} +- {se[1]:.1e}"
              + ("" if tested else "  (stiff coordinate not tested: carried by rare runs)"))
    print(f"(b) logistic regression, kappa = {B['kappa']:g}, alpha = {ALPHA_B:g}, n = {N}, d = {DIM} "
          f"(results/GammaDelta_summary.json, checked against results/digest.md)")
    print("    gamma R^2 | observed exponent per pass (passes 2-40, 3 seeds) | certified (Delta optimised): stored, "
          "curve")
    for s in STEPS_B:
        drawn = "" if s < 4.0 else "   <- not drawn as a rate (transient slope); marked as a stall"
        print(f"    {s:9.4f} | {B['obs'][s]:.4f} | {B['cert'][s]:.4f}, {certified_per_pass(s, N, B['kappa']):.4f}{drawn}")
    print(f"    best certificate {B['best_cert']:.4f} per pass at gamma R^2 = {B['best_step']:.4f} (textbook constants: "
          f"{B['textbook_cert']:.4f}; gain {B['gain']:.3f}); observed at 1/4 over best certificate = {B['ratio']:.3f}")
    for s, _ in RERUNS:
        r = B["reruns"][s]
        print(f"    rerun at gamma R^2 = {s:g}, seeds {RERUN_SEEDS}, {r['passes']} passes: slope over passes 2-40 "
              f"{r['fit']:.6f} (stored {B['obs'][s]:.6f}); |theta - eta_*|^2 / initial at pass 40: "
              + ", ".join(f"{v:.2e}" for v in r["at40"]) + "; at the end (pass "
              + ", ".join(f"{v:g}" for v in r["last_pass"]) + "): " + ", ".join(f"{v:.2e}" for v in r["final"])
              + ("" if r["late"] is None else "; slope over passes 100-" + f"{r['passes']}: "
                 + ", ".join(f"{v:+.4f}" for v in r["late"]) + "; median over passes >= 100: "
                 + ", ".join(f"{v:.3f}" for v in r["median_late"])))


# ----------------------------------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------------------------------
def panel_a(ax, A, ps, b):
    ink = ps.INK
    pk = A["per_kappa"]
    LIGHT = ps.BLUE_RAMP[3]                              # SAGA at another step: a lighter shade of the blue of SAGA
    ax.plot(KAPPAS, A["cert"], color=ps.RED, lw=2.0, zorder=4,
            label=rf"certificate ($\approx {pk['cert']:.0f}\,\kappa$)")
    b.add(2, ax.plot(KAPPAS, A["textbook"], color=ps.BLUE, lw=2.0, zorder=5,
                     label=rf"SAGA, $\gamma R^2 = 1/4$ ($\approx {pk['textbook']:.0f}\,\kappa$)"))
    b.add(2, ax.plot(KAPPAS, A["best"], color=LIGHT, lw=2.0, zorder=3,
                     label=rf"SAGA, best of {len(GRID)} steps ($\approx {pk['best']:.1f}\,\kappa$)"))
    ax.plot(KAPPAS, A["lower"], color=ink["primary"], lw=1.5, ls=(0, (4.0, 2.2)), zorder=6,
            label=rf"lower bound, any constant $\gamma$ ($\approx {pk['lower']:.1f}\,\kappa$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(KAPPAS[0], KAPPAS[-1])
    ax.set_ylim(10.0, 2e11)
    ax.set_xticks([1e1, 1e2, 1e3, 1e4, 1e5])
    ax.set_yticks([1e2, 1e4, 1e6])
    ax.minorticks_off()
    ax.set_xlabel(r"$\kappa = R^2/\mu$")
    ax.set_ylabel(r"iterations to $\varepsilon = 10^{-6}$")
    ax.text(11.5, 1.3e4, "memory-limited\n" r"($\kappa \,\lesssim\, n$)", ha="left", va="bottom", fontsize=es.small(),
            color=ink["secondary"], linespacing=1.15)
    es.panel_title(ax, "(a) worst case, exact")
    b.legend(ax, loc="upper left", handlelength=1.5, borderaxespad=0.15, labelspacing=0.28)


def panel_b(ax, B, ps, b):
    ink = ps.INK
    steps = [s for s in STEPS_B if s < 4.0]
    ax.axvspan(0.05, ADMISSIBLE, color=es.tint(ps.RED, 0.09), lw=0, zorder=0)
    ax.axhline(0.0, color=ink["axis"], lw=0.9, zorder=1)
    ax.plot(B["xs"], B["curve"], color=ps.RED, lw=2.0, zorder=3, label=r"certified, best $\Delta$")
    ax.plot(steps, [B["obs"][s] for s in steps], color=ps.BLUE, lw=1.8, marker="o", ms=6.5, mec=ink["surface"],
            mew=1.2, zorder=5, label="observed")
    kw = dict(fontsize=es.small(), color=ink["secondary"], linespacing=1.15)
    # observed at the textbook step against the best certificate
    ax.annotate("", (TEXTBOOK, B["obs"][TEXTBOOK] - 0.05), xytext=(TEXTBOOK, B["best_cert"] + 0.015),
                arrowprops=dict(arrowstyle="<->", color=ink["secondary"], lw=0.9, shrinkA=0.0, shrinkB=0.0,
                                mutation_scale=7.0), zorder=4)
    ax.text(TEXTBOOK * 1.07, 0.5 * (B["obs"][TEXTBOOK] + B["best_cert"]), rf"$\times\,{B['ratio']:.1f}$", ha="left",
            va="center", **kw)
    ax.text(0.085, 1.56, "the proof\nadmits\n" r"$\gamma R^2 < 1/2$", ha="left", va="top", **kw)
    # the two largest steps
    ax.annotate("converges", (2.0, B["obs"][2.0]), xytext=(0, -9), textcoords="offset points", ha="center",
                va="top", **kw)
    es.mark(ax, 4.0, 0.0, False)
    ax.annotate("stalls", (4.0, 0.0), xytext=(0, 9), textcoords="offset points", ha="center", va="bottom", **kw)
    ax.set_xscale("log")
    ax.set_xlim(0.08, 5.6)
    ax.set_ylim(-0.1, 1.6)
    ax.set_xticks(list(STEPS_B))
    ax.set_xticklabels(["1/8", "1/4", "", "1/2", "1", "2", "4"])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.minorticks_off()
    ax.set_xlabel(r"step $\gamma R^2$")
    ax.set_ylabel("exponent per pass")
    es.panel_title(ax, r"(b) logistic, $\kappa = 10^3$")
    ps.legend(ax, loc="upper right", handlelength=1.5, borderaxespad=0.15, labelspacing=0.28)
    b.add(3, ax)


def draw(D):
    ps = es.setup("talk")
    fig = es.figure("talk", height=2.75)
    fig.get_layout_engine().set(w_pad=0.03, wspace=0.03)
    gs = fig.add_gridspec(1, 2, width_ratios=(1.15, 1.0))
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    b = es.Build()
    panel_a(ax_a, D["a"], ps, b)
    panel_b(ax_b, D["b"], ps, b)
    es.save(fig, NAME, "talk", build=b,
            note=(rf"(a) {N_TOY} identical quadratics, $\theta_0 = (r, r)$, exact $E|\theta_t|^2$; (b) $n = {N}$, 3 seeds; "
                  rf"steps 2 and 4 rerun here"))


def main():
    D = data()
    print_all(D)
    draw(D)


if __name__ == "__main__":
    main()
