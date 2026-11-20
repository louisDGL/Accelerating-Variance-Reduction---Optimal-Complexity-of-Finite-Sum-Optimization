"""Hypothesis B4: sampling order and the memory term of SAGA. The ceiling of one nat per pass on the expected
squared error (memory slots that are never refreshed) belongs to sampling WITH replacement, not to the table.
 (M) solvable model with identical Hessians (vrlab.order_model): exponents per pass against c = gamma n lambda for
     i.i.d. indices (exact expectation, and typical runs: median of simulated runs), a fixed order (limit, companion
     matrix at finite n, critical step c_*) and random reshuffling (fluid limit, exact first moments, simulation).
 (L) logistic regression: SAGA with i.i.d., reshuffled, shuffled-once and cyclic indices over a grid of steps;
     exponent per pass of ||theta - eta_*||^2, passes to 1e-10, status of every run; the law (R3) of the model at
     c = gamma n lambda_min(F''(eta_*)) against the reshuffled runs; fixed orders against gamma n lambda_max.
 (S) hard instance (S) of the A1 note: gradient counts at which the seed mean of the relative squared error falls
     below eps, against the lower bounds of vrlab.bounds for i.i.d. span methods (Corollary cor:sweep) and for any
     index sequence (Proposition prop:anyseq); how far a seed mean is from the expectation under i.i.d. sampling.
Exponents are per pass (n iterations), for SQUARED errors, in nats. Every definition used below is written in the
"definitions" entry of results/B4_summary.json.
Usage: python experiments/exp_B4_sampling_order.py [--quick]        (VRLAB_WORKERS=k: k worker processes, default 1)
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import plotstyle as ps  # noqa: E402
from vrlab import algorithms, bounds, order_model, problems, solvers  # noqa: E402

QUICK = "--quick" in sys.argv
WORKERS = max(1, int(os.environ.get("VRLAB_WORKERS", "1")))
TAG = "B4_quick" if QUICK else "B4"
ORDER_LABEL = {"iid": "i.i.d.", "rr": "reshuffled", "so": "shuffled once", "cyclic": "cyclic"}
ORDER_COLOR = {"iid": ps.BLUE, "rr": ps.ORANGE, "so": ps.AQUA, "cyclic": ps.YELLOW}

# (M) model: simulated runs of M_PASSES passes; exponents fitted on the passes >= M_BURN (all memory slots refreshed)
M_NS = [100, 400] if QUICK else [100, 1000, 10000]
M_CS = [0.5, 2.0, 8.0, 32.0] if QUICK else [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
M_SEEDS = list(range(3 if QUICK else 9))
M_PASSES, M_BURN = (80, 30) if QUICK else (150, 30)
M_COMPANION_NS = [100] if QUICK else [100, 1000]          # companion matrix of the fixed order: O(n^3) per value of c
M_COMPANION_CS = list(M_CS) if QUICK else sorted(set(M_CS + [1.5, 3.0, 6.0, 12.0, 20.0, 24.0]))
M_ILLUS = dict(n=400 if QUICK else 1000, c=2.0, passes=40 if QUICK else 60)   # expectation against single runs

# (L) logistic regression, theta_0 = 0, one data draw per (kappa, alpha)
L_N, L_D = (300, 20) if QUICK else (1000, 50)
L_KAPPAS = [30.0, 300.0] if QUICK else [10.0, 1e2, 1e3, 1e4]
L_ALPHAS = [1.0] if QUICK else [0.0, 1.0, 2.0]
# gamma R^2; the steps below 1/16 are there for the i.i.d. order, whose best step is gamma n lambda_min of order 1 to 2
L_STEPS = [1 / 4, 1 / 2, 1.0] if QUICK else [1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0, 2.0]
L_SEEDS = list(range(2 if QUICK else 5))
L_PASSES = 80 if QUICK else 300
L_REC = L_N // 10                 # one record every n/10 iterations
TEXTBOOK = 1 / 4                  # gamma R^2 of Prop. 5.9
FLOOR = 1e-18                     # rounding floor on ||theta - eta_*||^2 used in B2 and B3
TARGET = 1e-10                    # "converged": ||theta_t - eta_*||^2 <= TARGET ||theta_0 - eta_*||^2
DIVERGED = 1e12                   # a run whose error exceeds DIVERGED times its initial value is frozen
MIN_FIT = 4                       # fewest pass-end records in a fitting window
CHECK_PASSES = 30                 # length of the comparison of the batched loop with vrlab.algorithms.saga

# (S) instance (S) of the A1 note, theta_0 = 0; kappa/n -> number of passes
S_N, S_M = (32, 12) if QUICK else (256, 12)
S_RATIOS = {0.1: 40, 1.0: 80} if QUICK else {0.1: 60, 1 / 3: 80, 1.0: 140, 4.0: 200}
S_STEPS = [1 / 4, 1 / 2, 1.0] if QUICK else [1 / 8, 1 / 4, 1 / 2, 1.0, 2.0, 4.0]        # gamma R^2
S_SEEDS = list(range(4 if QUICK else 32))
S_EPS = [1e-8, 1e-16, 1e-24]
S_ORDERS = ("iid", "rr")
S_REC = max(1, S_N // 32)
S_BOOT = 100 if QUICK else 500    # bootstrap resamples of the seeds for the counts of the seed mean
FLOOR_CMP = 1e-25                 # comparison at equal counts with the i.i.d. floor: only where the floor exceeds this

CONFIG = dict(M_NS=M_NS, M_CS=M_CS, M_SEEDS=M_SEEDS, M_PASSES=M_PASSES, M_BURN=M_BURN, M_COMPANION_NS=M_COMPANION_NS,
              M_COMPANION_CS=M_COMPANION_CS, M_ILLUS=M_ILLUS, L_N=L_N, L_D=L_D, L_KAPPAS=L_KAPPAS, L_ALPHAS=L_ALPHAS,
              L_STEPS=L_STEPS, L_SEEDS=L_SEEDS, L_PASSES=L_PASSES, L_REC=L_REC, FLOOR=FLOOR, TARGET=TARGET,
              DIVERGED=DIVERGED, MIN_FIT=MIN_FIT, S_N=S_N, S_M=S_M, S_RATIOS=S_RATIOS, S_STEPS=S_STEPS, S_SEEDS=S_SEEDS,
              S_EPS=S_EPS, S_REC=S_REC, S_BOOT=S_BOOT, CHECK_PASSES=CHECK_PASSES)

DEFINITIONS = {
    "exponent": "contraction exponent per pass (n iterations) of a SQUARED error, in nats: r means a decay like "
                "exp(-r t/n). Negative: growth.",
    "c": "gamma n lambda, the step in units of the model (vrlab.order_model): lambda an eigenvalue of the common Hessian "
         "in the model; on logistic regression c_min and c_max use lambda_min and lambda_max of F''(eta_*).",
    "gammaR2": "step size gamma times R^2 = prob.R2, the largest smoothness constant of the f_i (logistic regression: "
               "max_i (||x_i||^2/4 + lambda); instance (S): mu + c lambda_max(A_m), just below the nominal mu kappa); the "
               "textbook step is 1/4.",
    "model.iid_expectation": "exponent of E e_t^2 for i.i.d. indices: -n log(spectral radius) of the exact 4x4 moment "
                             "recursion (order_model.iid_exponent); its limit n -> infinity is min(2c, 1).",
    "model.typical": f"median over {len(M_SEEDS)} seeds of the exponent of ONE simulated run "
                     f"(order_model.simulate_passes, {M_PASSES} passes): minus the least-squares slope of the "
                     f"pass-end values of log max(e^2, max_j psi_j^2) (squared size of the state) over the passes "
                     f"{M_BURN}..{M_PASSES}; 'e2' is the same fit on log e^2. min and max are over the seeds.",
    "model.rr_limit": "order_model.rr_exponent(c): -2 log(spectral radius) of the fluid pass matrix (R3).",
    "model.rr_first_moment": "the same with the exact first-moment pass matrix at finite n (an upper bound on the "
                             "exponent of E e^2).",
    "model.cyclic_limit": "order_model.cyclic_exponent_limit(c): -2 Re of the rightmost root of the limiting "
                          "characteristic function (R2); negative beyond c_*.",
    "model.cyclic_companion": "order_model.cyclic_exponent(n, c): companion matrix of the recurrence at finite n.",
    "model.cyclic_simulated": "exponent of the (deterministic) simulated cyclic run, same window as model.typical.",
    "model.cyclic_threshold": "c at which the companion radius at finite n equals 1 (brentq, xtol 1e-4).",
    "logistic.status": f"converged: ||theta_t - eta_*||^2 <= {TARGET:g} ||theta_0 - eta_*||^2 at some record within "
                       f"{L_PASSES} passes; diverged: the error exceeded {DIVERGED:g} times its initial value (the run "
                       f"is then frozen: 'exploded') or ended above its initial value ('ended_above'); stalled: "
                       f"neither. A stalled run is 'growing' when its fitted exponent is negative (the error "
                       f"increases over the second half of the run) and slow otherwise (it still converges, too "
                       f"slowly for the budget). status_counts gives the three statuses and, under 'detail', the "
                       f"numbers of exploded, ended_above, stalled_growing and stalled_slow runs.",
    "logistic.exponent": f"minus the least-squares slope of log ||theta_t - eta_*||^2 against the pass number over the "
                         f"pass-end records p with p_end/2 <= p <= p_end, p_end being the last pass that ends before "
                         f"the first record below the floor {FLOOR:g} (the second-half window of B2); needs at least "
                         f"{MIN_FIT} records; not defined for diverged runs.",
    "logistic.passes": f"passes (iterations / n, the initial full gradient not counted) to the first record with "
                       f"relative squared distance <= {TARGET:g}, log-linear interpolation between records "
                       f"(algorithms.grads_to_accuracy); records every n/10 iterations. This is a FIRST passage: under "
                       f"reshuffling the error is not monotone within a pass (it dips after each pass boundary).",
    "logistic.passes_stay": f"passes from which the relative squared distance stays <= {TARGET:g} until the end of the "
                            f"run (log-linear interpolation after the last record above the target); equal to "
                            f"'passes' for a monotone run.",
    "logistic.gain": "passes_iid_fastest, passes_rr_fastest: median first-passage passes at the step of the grid that "
                     "minimises them, per order; passes_iid_stay, passes_rr_stay: median passes_stay at the step that "
                     "minimises it; saving_first = 1 - rr/iid on the first, saving_stay on the second; "
                     "rr_best_over_iid_best_exponent: ratio of the median exponents at the best steps.",
    "logistic.best": "for one order: the step of the grid with the largest median exponent among the steps at which "
                     "every run converged ('fastest': the smallest median number of passes).",
    "logistic.law_R3": "order_model.rr_exponent(c_min) with c_min = gamma n lambda_min(F''(eta_*)); rel_err = "
                       "measured median reshuffled exponent / law - 1, at the steps where every reshuffled run converged.",
    "logistic.fixed_order_failure": f"a shuffled-once or cyclic run FAILS at a step if it diverged, or if it did not "
                                    f"reach {TARGET:g} within the {L_PASSES} passes of the budget AND its exponent is "
                                    f"below half the median exponent of the i.i.d. runs at the same step (so the count "
                                    f"depends on the budget). SLOW (fixed_orders.*.n_slow_or_failing) is the wider "
                                    f"notion: failing, or converged with an exponent below half the i.i.d. median. "
                                    f"Steps at which an i.i.d. run diverged are left out. For each notion: number of "
                                    f"(problem, step) pairs concerned, smallest c_max = gamma n lambda_max(F''(eta_*)) "
                                    f"with one, largest c_max without, and how many lie below c_* of the model.",
    "logistic.law_R2": "min of order_model.cyclic_exponent_limit at c_min and at c_max (the limiting fixed-order exponent "
                       "of the model increases up to c = 2.148 and decreases beyond: over the spectrum of F''(eta_*) it is "
                       "smallest at one of the two ends); compared with the measured exponents by difference, in nats per "
                       "pass, because it vanishes at c_*.",
    "logistic.check": f"largest relative difference, over the records of the first {CHECK_PASSES} passes at which the "
                      f"squared error is above the floor {FLOOR:g}, between the batched loop of this script and "
                      f"vrlab.algorithms.saga (seed {L_SEEDS[0]}, textbook step, the four orders).",
    "hard.count": "gradient evaluations (n for the initial full gradient, then 1 per iteration), in units of n, at "
                  "which the mean over the seeds of ||theta_t - eta_*||^2 / ||eta_*||^2 FIRST falls below eps "
                  f"(records every {S_REC} iterations, log-linear interpolation); null if not reached in the budget. "
                  "Under reshuffling the seed mean is a sawtooth within each pass (a dip shortly after each pass "
                  "boundary), so three counts are given: 'count' (first passage), 'stay_count' (count from which the "
                  "mean stays <= eps until the end of the run) and 'pass_end_count' (first pass END, i.e. count "
                  "n (p + 1), at which the mean is <= eps, the convention of the floors of Proposition B4-floors). "
                  f"'boot' is the 2.5%, 50%, 97.5% quantiles of 'count' over {S_BOOT} bootstrap resamples of the seeds "
                  "(seed variability of the count of the mean; it does not estimate the distance to an expectation "
                  "carried by rare runs).",
    "hard.below_iid_floor_from": "for the reshuffled order at its best step: smallest count from which the seed mean "
                                 "is below the floor of inequality (expected-sweep) of the A1 note at the SAME count, "
                                 "at every later record at which that floor exceeds 1e-25 (comparison at equal "
                                 "counts, free of the choice of eps; below 1e-25 the runs meet the rounding floor of "
                                 "double precision); below_iid_floor_until is the last count of that range; "
                                 "smallest_mean_over_floor the smallest ratio mean / floor over it; null if the mean is "
                                 "above the floor at the last record of the range.",
    "hard.typical_count": "median over the seeds of the same count for single runs.",
    "hard.bound_iid": "bounds.lower_bound_iid_span(..., initial_full_gradient=True) / n: Corollary cor:sweep of the A1 "
                      "note, total count of any span method with i.i.d. queries after one full gradient, for the "
                      "EXPECTED error; null when the dimension condition q^(2m) <= eps fails.",
    "hard.bound_any": "bounds.lower_bound_any_sequence(...) / n: Proposition prop:anyseq, any index sequence, pathwise.",
    "hard.best": "step of the grid with the smallest first-passage count (per order and per eps); its stay_count, "
                 "pass_end_count and boot are those of that step; 'best_stay' is the step with the smallest "
                 "stay_count.",
    "hard.worst_seed_share": "largest single-seed error divided by the sum over the seeds, at the first record at "
                             "which the seed mean is <= eps (1/S for identical runs).",
    "model.rr_mean_square": "order_model.rr_ms_exponent(n, c): exact exponent of E e^2 for the reshuffled order at "
                            "finite n, -log(spectral radius) of the exact 4x4 second-moment pass matrix.",
    "model.rr_over_iid_typical": "rr_simulated.median / iid_typical.median at the same n and c.",
    "model.c_star, omega_star": "order_model.cyclic_critical(): critical step of a fixed order in the limit n -> "
                                "infinity and the frequency of the root on the imaginary axis.",
    "model.cyclic_best": "order_model.cyclic_best(): c and value of the largest limiting fixed-order exponent.",
    "model.n=<n>": "rr_max_abs_dev_from_limit: largest |rr_simulated.median - rr_limit| over the values of c; "
                   "rr_max_abs_dev_from_mean_square: the same against rr_mean_square (exact at that n); "
                   "rr_max_seed_spread, iid_max_seed_spread: largest (max - min over the seeds) of the simulated "
                   "exponents over the values of c.",
    "model.expectation_vs_runs": "at the passes listed: exact_E_e2 (4x4 recursion), median, max and mean over the "
                                 "seeds of e^2 of single i.i.d. runs; all_slots_refreshed_pass: pass at which the last "
                                 "slot of the initial memory is refreshed (median, min, max over the seeds); "
                                 "slow_amplitude_times_n: n * order_model.iid_slow_amplitude; plateau_limit: "
                                 "order_model.iid_plateau = 2 c^3 / (2c - 1)^2.",
    "logistic.at_grid_edge": "true when the selected step is the smallest or the largest of the grid.",
    "logistic.law_R3.<group>": "over the (problem, step) pairs of the group at which every reshuffled run converged: "
                               "n_pairs, max_abs_rel_err with the pair where it occurs (worst), median_rel_err, "
                               "n_within_10pct.",
    "logistic.fixed_orders.law_R2": "differences measured - law over the pairs with a measured exponent: numbers of "
                                    "pairs, largest and median absolute difference, median difference, per order.",
    "hard.count_over_bound": "count_over_bound_iid, count_over_bound_any: count / bound; the same with stay_ and "
                             "pass_end_ prefixes; rr_best_over_iid_best: ratio of the first-passage counts of the two "
                             "orders at their best steps.",
    "hard.min_block": "bounds.hard_instance_min_block: smallest block length m with q^(2m) <= eps (dimension condition).",
    "hard.min_ratio_to_any_sequence_floor": "smallest ratio, over the records, of the seed mean to the pathwise floor "
                                            "of Proposition prop:anyseq (must be >= 1).",
    "hard.caveat": "at the count where the i.i.d. seed mean reaches eps: expectation_floor is the exact expectation of "
                   "the pathwise tail bound (1/n) sum_j (q^(2 min(1 + tau_j, m)) - q^(2m)) / (1 - q^(2m)), tau_j the "
                   "number of queries to component j (Lemma lem:support (ii) and Proposition prop:tail of the A1 "
                   "note), a certified lower bound on the EXPECTED relative error; pathwise_floor_seed_mean is the "
                   "mean of the same bound over the index streams of the seeds actually run; worst_seed_share: see "
                   "hard.worst_seed_share; expected_unsampled is the expected number of (seed, component) pairs never "
                   "queried, S n (1 - 1/n)^t; realized_unsampled their number in the streams actually run. (The seed "
                   "mean at that count is eps by construction.)",
}


def fk(x):
    """Key of a float in the summary."""
    return f"{x:.4g}"


def clean(o):
    """Summary values for JSON: non-finite floats become null (strict JSON has no NaN)."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


# ----------------------------------------------------------------------------------------------
# (M) model
# ----------------------------------------------------------------------------------------------
def model_task(task):
    kind = task[0]
    if kind == "sim":
        _, n, c, order, seed = task
        sim = order_model.simulate_passes(n, c / n, order, passes=M_PASSES, seed=seed)
        return (order_model.fit_exponent(sim["log_size2"], burn=M_BURN),
                order_model.fit_exponent(sim["log_e2"], burn=M_BURN))
    if kind == "companion":
        _, n, c = task
        return order_model.cyclic_exponent(n, c)
    if kind == "threshold":
        _, n = task
        return float(brentq(lambda c: order_model.cyclic_radius(n, c / n) - 1.0, 8.0, 30.0, xtol=1e-4))
    raise ValueError(kind)


def run_model(res, pmap):
    mod = res["model"]
    tasks = [("sim", n, c, order, s) for n in M_NS for c in M_CS for order in ("iid", "rr") for s in M_SEEDS]
    tasks += [("sim", n, c, "cyclic", 0) for n in M_NS for c in M_CS]         # deterministic: one run
    tasks += [("companion", n, c) for n in M_COMPANION_NS for c in M_COMPANION_CS]
    tasks += [("threshold", n) for n in M_COMPANION_NS]
    for n in sorted(set(M_NS + M_COMPANION_NS)):                               # one cache write per n
        todo = [t for t in tasks if t[1] == n and t not in mod["tasks"]]
        if not todo:
            continue
        with common.Timer(f"model n={n}: {len(todo)} tasks"):
            for t, r in zip(todo, pmap(model_task, todo)):
                mod["tasks"][t] = r
        common.save_result(TAG, res)
    if "illus" not in mod:
        n, c, P = M_ILLUS["n"], M_ILLUS["c"], M_ILLUS["passes"]
        Tn = np.linalg.matrix_power(order_model.iid_moment_matrix(n, c / n), n)      # one pass of the exact recursion
        mom, exact = np.ones(4), [1.0]
        for _ in range(P):
            mom = Tn @ mom
            exact.append(float(mom[0]))
        runs = np.array([order_model.simulate(n, c / n, "iid", passes=P, seed=s)["e"][::n] ** 2 for s in M_SEEDS])
        # pass at which the last slot of the initial memory is refreshed (coupon collector), per seed
        last = []
        for s in M_SEEDS:
            idx = np.concatenate(list(algorithms._index_stream(np.random.default_rng(s), n, P * n)))
            first = np.full(n, -1)
            first[idx[::-1]] = np.arange(idx.size)[::-1]                              # first occurrence of each index
            last.append(float(first.max() + 1) / n if first.min() >= 0 else np.nan)
        mod["illus"] = {"n": n, "c": c, "exact_E_e2": np.array(exact), "runs_e2": runs, "all_refreshed_pass": last}
        common.save_result(TAG, res)


# ----------------------------------------------------------------------------------------------
# (L) logistic regression: batched SAGA (all orders, seeds and steps of one problem advance together)
# ----------------------------------------------------------------------------------------------
def saga_batch(prob, gammas, streams, col, passes, rec):
    """SAGA on a LogisticProblem for B runs at once, from theta_0 = 0 with z_i = f'_i(theta_0): run b uses the step
    gammas[b] and the index stream streams[col[b]]. Same update and same order of operations as
    vrlab.algorithms.saga with P = None (direction f'_i(theta) + zbar - z_i, zbar updated incrementally and
    recomputed from the table every n iterations), written for arrays (B, d); main() compares the two on every
    problem. A run whose squared error exceeds DIVERGED times its initial value, or is not finite, is frozen and
    recorded as inf from then on. Returns the squared errors ||theta - eta_*||^2, array (passes n / rec + 1, B)."""
    X, yv, lam, n, d = prob.X, prob.y, prob.lam, prob.n, prob.d
    B = len(gammas)
    theta = np.zeros((B, d))
    z = np.repeat(prob.all_grads(np.zeros(d))[None], B, axis=0)             # (B, n, d)
    zbar = z.mean(1)
    star = prob.theta_star
    D0 = float(star @ star)
    gam = np.array(gammas, dtype=np.float64)[:, None]
    ar = np.arange(B)
    dead = np.zeros(B, dtype=bool)
    D = np.empty((passes * (n // rec) + 1, B))
    D[0] = D0
    k = 0
    with np.errstate(all="ignore"):
        for p in range(passes):
            idx = np.ascontiguousarray(streams[:, p * n:(p + 1) * n][col].T)        # (n, B)
            for t in range(n):
                i = idx[t]
                xi = X[i]
                yi = yv[i]
                marg = yi * np.einsum("bd,bd->b", xi, theta)
                s = np.exp(-np.logaddexp(0.0, marg))                                  # sigma(-margin)
                gi = (-(yi * s))[:, None] * xi + lam * theta
                zi = z[ar, i]
                theta -= gam * (gi + zbar - zi)
                zbar += (gi - zi) / n
                z[ar, i] = gi
                if (t + 1) % rec == 0:
                    k += 1
                    diff = theta - star
                    Dk = np.einsum("bd,bd->b", diff, diff)
                    bad = (~np.isfinite(Dk) | (Dk > DIVERGED * D0)) & ~dead
                    if bad.any():
                        dead |= bad
                        gam[bad] = 0.0
                        theta[bad] = 0.0
                        zbar[bad] = 0.0
                        z[bad] = 0.0
                    Dk[dead] = np.inf
                    D[k] = Dk
            zbar = z.mean(1)
    return D


def last_passage(x, y, level):
    """Abscissa from which y stays <= level until the end of the series (log-linear interpolation after the last
    value above the level); 0 if y never exceeds it, nan if the last value is above it."""
    y = np.asarray(y, dtype=np.float64)
    above = np.where(~(y <= level))[0]
    if not above.size:
        return float(x[0])
    j = int(above[-1])
    if j == y.size - 1:
        return float("nan")
    if not (np.isfinite(y[j]) and y[j + 1] > 0):
        return float(x[j + 1])
    w = (np.log(y[j]) - np.log(level)) / (np.log(y[j]) - np.log(y[j + 1]))
    return float(x[j] + w * (x[j + 1] - x[j]))


def run_stats(D, n, rec):
    """Status, exponent and passes to TARGET of one run from its records (see DEFINITIONS)."""
    D = np.asarray(D, dtype=np.float64)
    out = {"final_rel": float(D[-1] / D[0]), "exponent": np.nan, "passes": np.nan, "passes_stay": np.nan,
           "end_pass": np.nan, "n_fit": 0, "detail": None}
    if not np.all(np.isfinite(D)) or D[-1] >= D[0]:
        out["status"] = "diverged"
        out["detail"] = "exploded" if not np.all(np.isfinite(D)) else "ended_above"
        return out
    per = n // rec
    out["passes"] = float(algorithms.grads_to_accuracy({"grad_evals": np.arange(D.size) / per, "D": D}, TARGET))
    out["status"] = "converged" if np.isfinite(out["passes"]) else "stalled"
    out["passes_stay"] = last_passage(np.arange(D.size) / per, D / D[0], TARGET)
    below = np.where(D < FLOOR)[0]
    k_end = int(below[0]) - 1 if below.size else D.size - 1
    p_end = k_end // per
    ps_ = np.arange(int(np.ceil(p_end / 2)), p_end + 1)
    out["end_pass"], out["n_fit"] = float(p_end), int(ps_.size)
    if ps_.size >= MIN_FIT:
        out["exponent"] = float(-np.polyfit(ps_.astype(float), np.log(D[ps_ * per]), 1)[0])
    if out["status"] == "stalled":
        out["detail"] = "stalled_growing" if out["exponent"] < 0 else "stalled_slow"
    return out


def logistic_task(task):
    kappa, alpha = task
    prob = common.logistic_problem(L_N, L_D, kappa, alpha, seed=0)
    n, d = prob.n, prob.d
    info = prob.summary()
    info.update({"grad_norm_at_optimum": prob.opt_info["grad_norm"], "D0": float(prob.theta_star @ prob.theta_star),
                 "kappa_over_n": kappa / n, "alpha": alpha,
                 "n_lmin_over_R2": n * info["hess_star_min"] / prob.R2, "n_lmax_over_R2": n * info["hess_star_max"] / prob.R2})
    stream_keys = [(o, s) for o in ("iid", "rr", "so") for s in L_SEEDS] + [("cyclic", 0)]   # cyclic ignores the seed
    T = L_PASSES * n
    streams = np.stack([np.concatenate(list(algorithms._index_stream(np.random.default_rng(s), n, T, sampling=o)))
                        for o, s in stream_keys]).astype(np.int32)
    runs = [(o, s, g) for g in L_STEPS for (o, s) in stream_keys]
    col = np.array([stream_keys.index((o, s)) for o, s, g in runs])
    D = saga_batch(prob, [g / prob.R2 for o, s, g in runs], streams, col, L_PASSES, L_REC)
    dev = 0.0
    for o in ("iid", "rr", "so", "cyclic"):          # the batched loop against the reference implementation
        ref = algorithms.saga(prob, np.zeros(d), gamma=TEXTBOOK / prob.R2, passes=CHECK_PASSES, seed=L_SEEDS[0],
                              record_every=L_REC, record_F=False, sampling=o)["D"]
        b = runs.index((o, 0 if o == "cyclic" else L_SEEDS[0], TEXTBOOK))
        above = ref > FLOOR                             # below the rounding floor relative differences mean nothing
        dev = max(dev, float(np.max(np.abs(D[:ref.size, b][above] / ref[above] - 1.0))))
    out_runs = []
    for b, (o, s, g) in enumerate(runs):
        st = run_stats(D[:, b], n, L_REC)
        st.update({"order": o, "seed": s, "gammaR2": g})
        out_runs.append(st)
    return {"problem": info, "runs": out_runs, "check_max_rel_dev": dev}


def run_logistic(res, pmap):
    todo = [(kappa, alpha) for kappa in L_KAPPAS for alpha in L_ALPHAS if (kappa, alpha) not in res["logistic"]]
    if not todo:
        return
    with common.Timer(f"logistic: {len(todo)} problems, {len(L_STEPS) * (3 * len(L_SEEDS) + 1)} runs each"):
        for task, out in zip(todo, pmap(logistic_task, todo)):
            res["logistic"][task] = out
            print(f"   kappa={task[0]:g} alpha={task[1]:g}: batched loop vs vrlab.algorithms.saga, max relative "
                  f"difference {out['check_max_rel_dev']:.1e}", flush=True)
            common.save_result(TAG, res)


# ----------------------------------------------------------------------------------------------
# (S) hard instance of the A1 note
# ----------------------------------------------------------------------------------------------
_HARD = {}


def hard_problem(kappa):
    if kappa not in _HARD:
        prob = problems.make_hard_instance_shared(S_N, S_M, kappa)
        solvers.solve_optimum(prob)
        _HARD[kappa] = prob
    return _HARD[kappa]


def hard_task(task):
    ratio, order, g, seed = task
    prob = hard_problem(ratio * S_N)
    with np.errstate(all="ignore"):
        r = algorithms.saga(prob, np.zeros(prob.d), gamma=g / prob.R2, passes=S_RATIOS[ratio], seed=seed,
                            record_every=S_REC, record_F=False, sampling=order)
    return r["D"] / r["D"][0]


def crossing(grads, y, eps):
    """First gradient count at which y <= eps (log-linear interpolation between records; nan if never)."""
    with np.errstate(all="ignore"):
        return float(algorithms.grads_to_accuracy({"grad_evals": grads, "D": np.asarray(y)}, eps, relative=False))


def run_hard(res, pmap):
    for ratio in S_RATIOS:
        tasks = [(ratio, order, g, s) for order in S_ORDERS for g in S_STEPS for s in S_SEEDS
                 if (ratio, order, g) not in res["hard"]]
        if not tasks:
            continue
        with common.Timer(f"instance (S) kappa/n={ratio:.3g}: {len(tasks)} runs of {S_RATIOS[ratio]} passes"):
            rels = list(pmap(hard_task, tasks))
        for order in S_ORDERS:
            for g in S_STEPS:
                rel = np.array([r for t, r in zip(tasks, rels) if t[1] == order and t[2] == g])
                if not rel.size:
                    continue
                grads = S_N + S_REC * np.arange(rel.shape[1])
                # seed variability of the count of the seed mean: resample the seeds with replacement
                brng = np.random.default_rng([20261119, int(round(1e4 * ratio)), S_ORDERS.index(order), int(round(1e4 * g))])
                with np.errstate(all="ignore"):
                    boot = np.array([[crossing(grads, rel[ix].mean(0), eps) for eps in S_EPS]
                                     for ix in brng.integers(0, rel.shape[0], size=(S_BOOT, rel.shape[0]))])
                    res["hard"][(ratio, order, g)] = {
                        "mean": rel.mean(0), "max": rel.max(0), "median": np.median(rel, axis=0),
                        "seed_counts": np.array([[crossing(grads, r, eps) for eps in S_EPS] for r in rel]),
                        "boot_counts": np.array([[np.quantile(b, q) if np.all(np.isfinite(b)) else np.nan
                                                  for q in (0.025, 0.5, 0.975)] for b in boot.T]),
                        "n_not_finite": int(np.sum(~np.all(np.isfinite(rel), axis=1)))}
        common.save_result(TAG, res)


# ----------------------------------------------------------------------------------------------
# cache
# ----------------------------------------------------------------------------------------------
def run_all():
    res = common.load_result(TAG)
    if res is None:
        res = {"config": CONFIG, "model": {"tasks": {}}, "logistic": {}, "hard": {}}
    elif res["config"] != CONFIG:
        raise RuntimeError(f"results/{TAG}.pkl was computed with another configuration: delete it to recompute")
    else:
        print("loaded cached", TAG)
    pool = Pool(WORKERS) if WORKERS > 1 else None
    # ordered and lazy: the results are written to the cache as they arrive
    pmap = (lambda f, tasks: pool.imap(f, tasks, chunksize=1)) if pool else (lambda f, tasks: map(f, tasks))
    try:
        run_model(res, pmap)
        run_logistic(res, pmap)
        run_hard(res, pmap)
    finally:
        if pool:
            pool.close()
            pool.join()
    return res


# ----------------------------------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------------------------------
def spread(values):
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if not v.size:
        return {"median": None, "min": None, "max": None, "n": 0}
    return {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()), "n": int(v.size)}


def analyse_model(res):
    T = res["model"]["tasks"]
    omega, c_star = order_model.cyclic_critical()
    c_best, e_best = order_model.cyclic_best()
    out = {"window_passes": [M_BURN, M_PASSES], "n_seeds": len(M_SEEDS), "omega_star": omega, "c_star": c_star,
           "cyclic_best": {"c": c_best, "exponent": e_best},
           "cyclic_threshold": {f"n={n}": T[("threshold", n)] for n in M_COMPANION_NS}, "by_c": {}}
    for c in M_CS:
        row = {"iid_expectation_limit": order_model.iid_exponent_limit(c), "rr_limit": order_model.rr_exponent(c),
               "cyclic_limit": order_model.cyclic_exponent_limit(c)}
        for n in M_NS:
            iid = [T[("sim", n, c, "iid", s)] for s in M_SEEDS]
            rr = [T[("sim", n, c, "rr", s)] for s in M_SEEDS]
            row[f"n={n}"] = {
                "iid_expectation": order_model.iid_exponent(n, c),
                "iid_typical": spread([x[0] for x in iid]), "iid_typical_e2": spread([x[1] for x in iid]),
                "rr_first_moment": order_model.rr_exponent(c, n=n),
                "rr_mean_square": order_model.rr_ms_exponent(n, c),
                "rr_simulated": spread([x[0] for x in rr]), "rr_simulated_e2": spread([x[1] for x in rr]),
                "cyclic_simulated": T[("sim", n, c, "cyclic", 0)][0],
                "cyclic_companion": T.get(("companion", n, c))}
            r = row[f"n={n}"]
            r["rr_over_iid_typical"] = r["rr_simulated"]["median"] / r["iid_typical"]["median"]
        out["by_c"][fk(c)] = row
    out["cyclic_companion"] = {f"n={n}": {fk(c): T[("companion", n, c)] for c in M_COMPANION_CS} for n in M_COMPANION_NS}
    # agreement of the simulated reshuffled runs with the fluid law, and range of the typical/expected i.i.d. ratio
    for n in M_NS:
        rows = [out["by_c"][fk(c)] for c in M_CS]
        out[f"n={n}"] = {
            "rr_max_abs_dev_from_limit": max(abs(r[f"n={n}"]["rr_simulated"]["median"] - r["rr_limit"]) for r in rows),
            "rr_max_abs_dev_from_mean_square": max(abs(r[f"n={n}"]["rr_simulated"]["median"] - r[f"n={n}"]["rr_mean_square"])
                                                   for r in rows),
            "rr_max_seed_spread": max(r[f"n={n}"]["rr_simulated"]["max"] - r[f"n={n}"]["rr_simulated"]["min"] for r in rows),
            "iid_max_seed_spread": max(r[f"n={n}"]["iid_typical"]["max"] - r[f"n={n}"]["iid_typical"]["min"] for r in rows)}
    ill = res["model"]["illus"]
    runs = ill["runs_e2"]
    marks = [p for p in (10, 20, 30, 40, 50, 60) if p <= M_ILLUS["passes"]]
    out["expectation_vs_runs"] = {
        "n": ill["n"], "c": ill["c"], "n_seeds": int(runs.shape[0]), "passes": marks,
        "exact_E_e2": [float(ill["exact_E_e2"][p]) for p in marks],
        "median_run_e2": [float(np.median(runs[:, p])) for p in marks],
        "max_run_e2": [float(runs[:, p].max()) for p in marks],
        "seed_mean_e2": [float(runs[:, p].mean()) for p in marks],
        "all_slots_refreshed_pass": spread(ill["all_refreshed_pass"]),
        "exact_exponent": order_model.iid_exponent(ill["n"], ill["c"]),
        "slow_amplitude_times_n": ill["n"] * order_model.iid_slow_amplitude(ill["n"], ill["c"]),
        "plateau_limit": order_model.iid_plateau(ill["c"])}
    return out


def analyse_logistic(res, model):
    out = {"problems": {}, "configs": {}, "headline": {}, "law_R3": {"by_problem": {}}, "fixed_orders": {"pairs": []},
           "check_max_rel_dev": max(e["check_max_rel_dev"] for e in res["logistic"].values()),
           "status_counts": {o: {"converged": 0, "stalled": 0, "diverged": 0,
                                 "detail": {"exploded": 0, "ended_above": 0, "stalled_growing": 0, "stalled_slow": 0}}
                             for o in ORDER_LABEL}}
    law_rows = []
    for kappa in L_KAPPAS:
        for alpha in L_ALPHAS:
            entry = res["logistic"][(kappa, alpha)]
            key = f"kappa={kappa:g}|alpha={alpha:g}"
            P = entry["problem"]
            out["problems"][key] = {k: P[k] for k in ("n", "d", "kappa", "alpha", "kappa_over_n", "lam", "R2", "D0",
                                                       "hess_star_min", "hess_star_max", "mu_local_over_mu",
                                                       "n_lmin_over_R2", "n_lmax_over_R2", "grad_norm_at_optimum")}
            cfg = {}
            for o in ORDER_LABEL:
                for g in L_STEPS:
                    rs = [r for r in entry["runs"] if r["order"] == o and r["gammaR2"] == g]
                    st = {s: sum(r["status"] == s for r in rs) for s in ("converged", "stalled", "diverged")}
                    for s, k in st.items():
                        out["status_counts"][o][s] += k
                    for r in rs:
                        if r["detail"]:
                            out["status_counts"][o]["detail"][r["detail"]] += 1
                    cfg[(o, g)] = {"status": st, "n_runs": len(rs), "all_converged": st["converged"] == len(rs),
                                   "exponent": spread([r["exponent"] for r in rs]),
                                   "passes": spread([r["passes"] for r in rs]),
                                   "passes_stay": spread([r["passes_stay"] for r in rs]),
                                   "c_min": g * P["n_lmin_over_R2"], "c_max": g * P["n_lmax_over_R2"]}
                    out["configs"][f"{key}|{o}|gammaR2={fk(g)}"] = cfg[(o, g)]
            head = {"iid_textbook": dict(cfg[("iid", TEXTBOOK)], gammaR2=TEXTBOOK)}
            for o in ORDER_LABEL:
                ok = [g for g in L_STEPS if cfg[(o, g)]["all_converged"] and cfg[(o, g)]["exponent"]["median"] is not None]
                if ok:
                    gb = max(ok, key=lambda g: cfg[(o, g)]["exponent"]["median"])
                    gf = min(ok, key=lambda g: cfg[(o, g)]["passes"]["median"])
                    gs = min(ok, key=lambda g: cfg[(o, g)]["passes_stay"]["median"])
                    head[f"{o}_best"] = dict(cfg[(o, gb)], gammaR2=gb, at_grid_edge=gb in (L_STEPS[0], L_STEPS[-1]))
                    head[f"{o}_fastest"] = {"gammaR2": gf, "passes": cfg[(o, gf)]["passes"],
                                            "passes_stay": cfg[(o, gf)]["passes_stay"]}
                    head[f"{o}_fastest_stay"] = {"gammaR2": gs, "passes_stay": cfg[(o, gs)]["passes_stay"]}
                else:
                    head[f"{o}_best"] = head[f"{o}_fastest"] = head[f"{o}_fastest_stay"] = None
            if head["rr_best"] and head["iid_best"]:
                tb = head["iid_textbook"]
                head["gain"] = {
                    "rr_best_over_iid_best_exponent": head["rr_best"]["exponent"]["median"] / head["iid_best"]["exponent"]["median"],
                    "rr_best_over_iid_textbook_exponent": (head["rr_best"]["exponent"]["median"] / tb["exponent"]["median"]
                                                           if tb["exponent"]["median"] else None),
                    "passes_iid_textbook": tb["passes"]["median"], "passes_iid_fastest": head["iid_fastest"]["passes"]["median"],
                    "passes_rr_fastest": head["rr_fastest"]["passes"]["median"],
                    "passes_iid_stay": head["iid_fastest_stay"]["passes_stay"]["median"],
                    "passes_rr_stay": head["rr_fastest_stay"]["passes_stay"]["median"]}
                gn = head["gain"]
                gn["saving_first"] = 1.0 - gn["passes_rr_fastest"] / gn["passes_iid_fastest"]
                gn["saving_stay"] = 1.0 - gn["passes_rr_stay"] / gn["passes_iid_stay"]
            out["headline"][key] = head
            # the law (R3) at c_min against the reshuffled runs
            law = {}
            for g in L_STEPS:
                c = cfg[("rr", g)]
                if c["all_converged"] and c["exponent"]["median"] is not None:
                    pred = order_model.rr_exponent(c["c_min"])
                    law[fk(g)] = {"c_min": c["c_min"], "law": pred, "measured": c["exponent"]["median"],
                                  "rel_err": c["exponent"]["median"] / pred - 1.0}
                    law_rows.append((key, g, law[fk(g)]["rel_err"]))
                else:
                    law[fk(g)] = None
            out["law_R3"]["by_problem"][key] = law
            # fixed orders against c_max = gamma n lambda_max
            for g in L_STEPS:
                iid_runs = [r for r in entry["runs"] if r["order"] == "iid" and r["gammaR2"] == g]
                if any(r["status"] == "diverged" for r in iid_runs):
                    continue
                ref = cfg[("iid", g)]["exponent"]["median"]

                def fails(r):
                    if r["status"] == "diverged":
                        return True
                    return r["status"] != "converged" and not (ref is not None and r["exponent"] >= 0.5 * ref)

                def slow(r):
                    return fails(r) or not (ref is not None and r["exponent"] >= 0.5 * ref)
                so = [r for r in entry["runs"] if r["order"] == "so" and r["gammaR2"] == g]
                cy = [r for r in entry["runs"] if r["order"] == "cyclic" and r["gammaR2"] == g]
                c_lo, c_hi = cfg[("so", g)]["c_min"], cfg[("so", g)]["c_max"]
                out["fixed_orders"]["pairs"].append({
                    "problem": key, "gammaR2": g, "c_max": c_hi, "c_min": c_lo,
                    "so_failures": sum(fails(r) for r in so), "so_runs": len(so), "cyclic_fails": bool(fails(cy[0])),
                    "so_slow": sum(slow(r) for r in so), "cyclic_slow": bool(slow(cy[0])),
                    "so_exponent_median": cfg[("so", g)]["exponent"]["median"],
                    "cyclic_exponent": cfg[("cyclic", g)]["exponent"]["median"], "iid_exponent_median": ref,
                    "law_R2": min(order_model.cyclic_exponent_limit(c_lo), order_model.cyclic_exponent_limit(c_hi))})
    lw = out["law_R3"]
    for name, sel in (("gammaR2_le_half", lambda g: g <= 0.5), ("gammaR2_ge_1", lambda g: g >= 1.0)):
        rows = [r for r in law_rows if sel(r[1])]
        if rows:
            worst = max(rows, key=lambda r: abs(r[2]))
            lw[name] = {"n_pairs": len(rows), "max_abs_rel_err": abs(worst[2]), "worst": {"problem": worst[0], "gammaR2": worst[1], "rel_err": worst[2]},
                        "median_rel_err": float(np.median([r[2] for r in rows])),
                        "n_within_10pct": sum(abs(r[2]) <= 0.10 for r in rows)}
        else:
            lw[name] = None
    lw["by_step"] = {fk(g): spread([r[2] for r in law_rows if r[1] == g]) for g in L_STEPS}
    fo = out["fixed_orders"]
    fo["pairs"].sort(key=lambda p: p["c_max"])
    fo["c_star"] = model["c_star"]
    diffs = {o: [p[k] - p["law_R2"] for p in fo["pairs"] if p[k] is not None]
             for o, k in (("so", "so_exponent_median"), ("cyclic", "cyclic_exponent"))}
    fo["law_R2"] = {"n_pairs": len(fo["pairs"])}
    for o, v in diffs.items():
        fo["law_R2"].update({f"n_measured_{o}": len(v), f"max_abs_diff_{o}": float(np.max(np.abs(v))) if v else None,
                             f"median_abs_diff_{o}": float(np.median(np.abs(v))) if v else None,
                             f"median_diff_{o}": float(np.median(v)) if v else None})
    fo["cyclic_threshold_at_n"] = model["cyclic_threshold"].get(f"n={L_N}")
    for name, bad, slw in (("so", lambda p: p["so_failures"] > 0, lambda p: p["so_slow"] > 0),
                           ("cyclic", lambda p: p["cyclic_fails"], lambda p: p["cyclic_slow"])):
        ok_c = [p["c_max"] for p in fo["pairs"] if not bad(p)]
        ko_c = [p["c_max"] for p in fo["pairs"] if bad(p)]
        ok_s = [p["c_max"] for p in fo["pairs"] if not slw(p)]
        ko_s = [p["c_max"] for p in fo["pairs"] if slw(p)]
        fo[name] = {"n_pairs": len(fo["pairs"]), "n_failing": len(ko_c),
                    "largest_c_max_without_failure": max(ok_c) if ok_c else None,
                    "smallest_c_max_with_failure": min(ko_c) if ko_c else None,
                    "separated": bool(ok_c and ko_c and max(ok_c) < min(ko_c)),
                    "n_ok_above_c_star": sum(c > model["c_star"] for c in ok_c),
                    "n_failing_below_c_star": sum(c <= model["c_star"] for c in ko_c),
                    "n_slow_or_failing": len(ko_s),
                    "largest_c_max_not_slow": max(ok_s) if ok_s else None,
                    "smallest_c_max_slow": min(ko_s) if ko_s else None,
                    "n_slow_below_c_star": sum(c <= model["c_star"] for c in ko_s)}
    return out


def pathwise_floor(counts, q, m):
    """Tail bound of the A1 note for one run after an initial full gradient: counts (n,) queries per component."""
    tail = q ** (2 * m)
    return float(np.mean((q ** (2.0 * np.minimum(1 + counts, m)) - tail) / (1.0 - tail)))


def expected_floor(t, n, q, m):
    """Exact expectation of pathwise_floor after t i.i.d. queries: tau ~ Binomial(t, 1/n) for every component."""
    k = np.arange(0, m - 1)
    tail = q ** (2 * m)
    return float(np.sum(binom.pmf(k, t, 1.0 / n) * (q ** (2.0 * (1 + k)) - tail)) / (1.0 - tail))


def analyse_hard(res):
    n, m, S = S_N, S_M, len(S_SEEDS)
    out = {}
    for ratio, passes in S_RATIOS.items():
        kappa = ratio * n
        Q, q = bounds.hard_instance_constants(n, kappa)
        prob = hard_problem(kappa)
        Hev = np.linalg.eigvalsh(prob.A[0] + prob.ridge * np.eye(m))               # the blocks of F'' are equal
        e = {"kappa": kappa, "Q": Q, "q": q, "q_pow_2m": q ** (2 * m), "passes": passes, "R2": prob.R2,
             "n_lmin_over_R2": n * float(Hev[0]) / prob.R2, "n_lmax_over_R2": n * float(Hev[-1]) / prob.R2,
             "by_eps": {}, "min_ratio_to_any_sequence_floor": {}, "n_not_finite": {}}
        grads = None
        for order in S_ORDERS:
            for g in S_STEPS:
                c = res["hard"][(ratio, order, g)]
                grads = n + S_REC * np.arange(c["mean"].size)
                fl = bounds.hard_instance_error_floor(grads, n, kappa, m, sampling="any")
                with np.errstate(all="ignore"):
                    ratio_floor = np.where(fl > 0, c["mean"] / np.where(fl > 0, fl, 1.0), np.inf)
                e["min_ratio_to_any_sequence_floor"][f"{order}|gammaR2={fk(g)}"] = float(np.nanmin(ratio_floor))
                e["n_not_finite"][f"{order}|gammaR2={fk(g)}"] = c["n_not_finite"]
        for ie, eps in enumerate(S_EPS):
            b = {"min_block": bounds.hard_instance_min_block(n, kappa, eps)}
            b["dimension_condition"] = bool(m >= b["min_block"])
            for name, f in (("bound_iid", lambda: bounds.lower_bound_iid_span(n, kappa, eps, m=m, initial_full_gradient=True)),
                            ("bound_any", lambda: bounds.lower_bound_any_sequence(n, kappa, eps, m=m))):
                try:
                    b[name] = f() / n
                except ValueError:
                    b[name] = None
            pass_end = np.arange(0, grads.size, n // S_REC)                     # records at the pass ends, count n (p + 1)
            for order in S_ORDERS:
                means = {g: res["hard"][(ratio, order, g)]["mean"] for g in S_STEPS}
                cnt = {g: crossing(grads, means[g], eps) / n for g in S_STEPS}
                with np.errstate(all="ignore"):
                    stay = {g: last_passage(grads, means[g], eps) / n for g in S_STEPS}
                pend = {}
                for g in S_STEPS:
                    ok = np.where(means[g][pass_end] <= eps)[0]
                    pend[g] = float(grads[pass_end[ok[0]]]) / n if ok.size else np.nan
                typ = {g: float(np.median(res["hard"][(ratio, order, g)]["seed_counts"][:, ie])) / n for g in S_STEPS}
                opt = lambda d: {fk(g): (v if np.isfinite(v) else None) for g, v in d.items()}
                b[order] = {"count_by_step": opt(cnt), "stay_count_by_step": opt(stay), "pass_end_count_by_step": opt(pend),
                            "typical_count_by_step": opt(typ),
                            "textbook": cnt[TEXTBOOK] if np.isfinite(cnt[TEXTBOOK]) else None}
                reached = [g for g in S_STEPS if np.isfinite(cnt[g])]
                if reached:
                    gb = min(reached, key=lambda g: cnt[g])
                    c = res["hard"][(ratio, order, gb)]
                    k = int(np.argmax(c["mean"] <= eps))                           # first record below eps
                    bq = c["boot_counts"][ie] / n
                    b[order]["best"] = {"gammaR2": gb, "count": cnt[gb], "typical_count": typ[gb] if np.isfinite(typ[gb]) else None,
                                        "stay_count": stay[gb] if np.isfinite(stay[gb]) else None,
                                        "pass_end_count": pend[gb] if np.isfinite(pend[gb]) else None,
                                        "boot": [float(x) if np.isfinite(x) else None for x in bq],
                                        "at_grid_edge": gb in (S_STEPS[0], S_STEPS[-1]),
                                        "worst_seed_share": float(c["max"][k] / (S * c["mean"][k]))}
                    for name in ("bound_iid", "bound_any"):
                        for pre, val in (("count", cnt[gb]), ("stay_count", stay[gb]), ("pass_end_count", pend[gb])):
                            b[order]["best"][f"{pre}_over_{name}"] = val / b[name] if (b[name] and np.isfinite(val)) else None
                    stays = [g for g in S_STEPS if np.isfinite(stay[g])]
                    if stays:
                        gs = min(stays, key=lambda g: stay[g])
                        b[order]["best_stay"] = {"gammaR2": gs, "stay_count": stay[gs],
                                                 "at_grid_edge": gs in (S_STEPS[0], S_STEPS[-1]),
                                                 "stay_count_over_bound_iid": stay[gs] / b["bound_iid"] if b["bound_iid"] else None}
                    else:
                        b[order]["best_stay"] = None
                else:
                    b[order]["best"] = b[order]["best_stay"] = None
            if b["iid"]["best"] and b["rr"]["best"]:
                b["rr_best_over_iid_best"] = b["rr"]["best"]["count"] / b["iid"]["best"]["count"]
                lo, hi = b["rr"]["best"]["boot"], b["iid"]["best"]["boot"]
                if None not in (lo[0], lo[2], hi[0], hi[2]):
                    b["rr_best_over_iid_best_boot_range"] = [lo[0] / hi[2], lo[2] / hi[0]]
            # seed mean against expectation under i.i.d. sampling, at the count where the seed mean reaches eps
            if b["iid"]["best"]:
                t = int(round((b["iid"]["best"]["count"] - 1.0) * n))                # i.i.d. queries after the full gradient
                floors, unsampled = [], 0
                for s in S_SEEDS:
                    idx = np.concatenate(list(algorithms._index_stream(np.random.default_rng(s), n, t)))
                    counts = np.bincount(idx, minlength=n)
                    floors.append(pathwise_floor(counts, q, m))
                    unsampled += int(np.sum(counts == 0))
                ef = expected_floor(t, n, q, m)
                b["caveat_iid"] = {"queries_after_full_gradient": t, "seed_mean": eps, "expectation_floor": ef,
                                   "expectation_floor_over_eps": ef / eps,
                                   "pathwise_floor_seed_mean": float(np.mean(floors)),
                                   "pathwise_floor_seed_max": float(np.max(floors)),
                                   "expected_unsampled": S * n * (1.0 - 1.0 / n) ** t, "realized_unsampled": unsampled,
                                   "worst_seed_share": b["iid"]["best"]["worst_seed_share"],
                                   "seed_mean_below_certified_expectation": bool(ef > eps)}
            e["by_eps"][f"{eps:g}"] = b
        best_rr = e["by_eps"][f"{S_EPS[0]:g}"]["rr"]["best"]
        if best_rr:
            mean = res["hard"][(ratio, "rr", best_rr["gammaR2"])]["mean"]
            fl = bounds.hard_instance_error_floor(grads - n, n, kappa, m, initial_full_gradient=True)
            pos = fl > FLOOR_CMP                         # beyond, the runs sit on the rounding floor of double precision
            above = np.where(pos & ~(mean <= fl))[0]
            j = int(above[-1]) + 1 if above.size else 0
            e["below_iid_floor"] = {"gammaR2": best_rr["gammaR2"],
                                    "below_iid_floor_from": float(grads[j]) / n if j < grads.size and pos[j] else None,
                                    "below_iid_floor_until": float(grads[np.where(pos)[0][-1]]) / n,
                                    "smallest_mean_over_floor": float(np.min(mean[pos][j:] / fl[pos][j:])) if j < int(pos.sum()) else None}
        out[f"kappa/n={fk(ratio)}"] = e
    return out


# ----------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------
def plot(res, summary):
    mod, logi, hard = summary["model"], summary["logistic"], summary["hard_instance"]
    fig = ps.figure(ps.FIG_W, 7.6)
    gs = fig.add_gridspec(4, 2, height_ratios=[1.0, 0.02, 1.0, 0.02])      # two rows of panels, a legend row under each
    sec, mk = ps.INK["secondary"], dict(markersize=5.5, markeredgecolor=ps.INK["surface"], markeredgewidth=0.8)

    # (a) model: exponents against c
    ax = fig.add_subplot(gs[0, 0])
    cg = np.exp(np.linspace(np.log(0.2), np.log(40.0), 100))
    n_ref = M_NS[1]
    ax.plot(cg, [order_model.rr_exponent(c) for c in cg], color=ORDER_COLOR["rr"], label="reshuffled: fluid law (R3)")
    ax.plot(M_CS, [mod["by_c"][fk(c)][f"n={n_ref}"]["rr_simulated"]["median"] for c in M_CS], linestyle="none", marker="o",
            color=ORDER_COLOR["rr"], label=f"reshuffled: simulated runs, n = {n_ref}", **mk)
    ax.plot(cg, [order_model.iid_exponent_limit(c) for c in cg], color=ORDER_COLOR["iid"], linestyle=(0, (4, 2)),
            label="i.i.d.: expectation E e², min(2c, 1)")
    for n, marker in zip(M_NS, ["s", "o", "^"]):
        ax.plot(M_CS, [mod["by_c"][fk(c)][f"n={n}"]["iid_typical"]["median"] for c in M_CS], linestyle="none", marker=marker,
                color=ORDER_COLOR["iid"], label=f"i.i.d.: typical run (median), n = {n}", **mk)
    cyc = np.array([order_model.cyclic_exponent_limit(c) for c in cg])
    ax.plot(cg, cyc, color=ORDER_COLOR["so"], label="fixed order: limit (R2)")
    ax.axvline(mod["c_star"], color=ps.INK["axis"], linewidth=0.8)
    ax.axhline(0.0, color=ps.INK["axis"], linewidth=0.8)
    ax.set_xscale("log")
    ax.set_ylim(-0.15, 2.15)
    ax.annotate(f"c* = {mod['c_star']:.2f}", (mod["c_star"], 2.15), xytext=(-3, -3), textcoords="offset points", ha="right",
                va="top", fontsize=ps.SMALL, color=sec)
    ax.set_xlabel("c = γ n λ")
    ax.set_ylabel("exponent per pass of the squared error")
    ax.set_title("identical-Hessian model")
    h, l = ax.get_legend_handles_labels()
    ps.legend_cell(fig, gs[1, 0], h, l, ncol=1)

    # (b) logistic regression: measured exponents against c_min, with the law (R3)
    ax = fig.add_subplot(gs[0, 1])
    pts = {("rr", True): [], ("rr", False): [], ("iid", True): [], ("iid", False): []}
    for key, cfgk in logi["configs"].items():
        prob_key, o, gk = key.rsplit("|", 2)
        if o in ("rr", "iid") and cfgk["all_converged"] and cfgk["exponent"]["median"] is not None:
            pts[(o, float(gk.split("=")[1]) <= 0.5)].append((cfgk["c_min"], cfgk["exponent"]["median"]))
    xs = [p[0] for v in pts.values() for p in v]
    cl = np.exp(np.linspace(np.log(min(xs) / 1.5), np.log(max(xs) * 1.5), 120))
    ax.plot(cl, [order_model.rr_exponent(c) for c in cl], color=ORDER_COLOR["rr"], label="law (R3) at c = γ n λmin(F″(η*))")
    ax.plot(cl, [order_model.iid_exponent_limit(c) for c in cl], color=ORDER_COLOR["iid"], linestyle=(0, (4, 2)),
            label="model, i.i.d. expectation: min(2c, 1)")
    ax.plot(M_CS, [mod["by_c"][fk(c)][f"n={n_ref}"]["iid_typical"]["median"] for c in M_CS], color=ORDER_COLOR["iid"],
            linewidth=1.0, label=f"model, i.i.d. typical run (n = {n_ref})")
    for o in ("iid", "rr"):
        for small, fill, lab in ((True, ORDER_COLOR[o], "γR² ≤ 1/2"), (False, ps.INK["surface"], "γR² ≥ 1")):
            v = pts[(o, small)]
            if v:
                ax.plot([p[0] for p in v], [p[1] for p in v], linestyle="none", marker="o", markersize=5.5 if small else 5,
                        markerfacecolor=fill, markeredgecolor=ps.INK["surface"] if small else ORDER_COLOR[o],
                        markeredgewidth=0.8 if small else 1.3, color=ORDER_COLOR[o], label=f"{ORDER_LABEL[o]} runs, {lab}")
    ax.set_xscale("log")
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("c = γ n λmin(F″(η*))")
    ax.set_ylabel("measured exponent per pass\n(median over seeds)")
    ax.set_title(f"logistic regression, n = {L_N}")
    h, l = ax.get_legend_handles_labels()
    ps.legend_cell(fig, gs[1, 1], h, l, ncol=1)

    # (c) logistic regression: best exponent of each order, problem by problem
    ax = fig.add_subplot(gs[2, 0])
    keys = list(logi["headline"])
    ypos = np.arange(len(keys))[::-1]
    series = [("iid_textbook", "iid", "i.i.d., textbook step γR² = 1/4", False), ("iid_best", "iid", "i.i.d., best step", True),
              ("rr_best", "rr", "reshuffled, best step", True), ("so_best", "so", "shuffled once, best step", True),
              ("cyclic_best", "cyclic", "cyclic, best step", True)]
    for name, o, lab, filled in series:
        xv = [(logi["headline"][k][name] or {}).get("exponent", {}).get("median") for k in keys]
        xv = [np.nan if v is None else v for v in xv]
        ax.plot(xv, ypos, linestyle="none", marker="o" if o in ("iid", "rr") else ("D" if o == "so" else "^"),
                markersize=5.5 if filled else 5, markerfacecolor=ORDER_COLOR[o] if filled else ps.INK["surface"],
                markeredgecolor=ps.INK["surface"] if filled else ORDER_COLOR[o], markeredgewidth=0.8 if filled else 1.3,
                color=ORDER_COLOR[o], label=lab)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"κ/n = {logi['problems'][k]['kappa_over_n']:g}, α = {logi['problems'][k]['alpha']:g}" for k in keys],
                       fontsize=ps.SMALL)
    ax.set_xlim(left=0.0)
    ax.set_ylim(-0.6, len(keys) - 0.4)
    ax.set_xlabel("exponent per pass (median over seeds)")
    ax.set_title("logistic regression: best step per order")
    h, l = ax.get_legend_handles_labels()
    ps.legend_cell(fig, gs[3, 0], h, l, ncol=1)

    # (d) instance (S): seed means against the two floors of the A1 note
    ax = fig.add_subplot(gs[2, 1])
    ratio = list(S_RATIOS)[0]
    e = hard[f"kappa/n={fk(ratio)}"]
    kappa = ratio * S_N
    eps0 = f"{S_EPS[0]:g}"
    xmax = 0.0
    for o in S_ORDERS:
        best = e["by_eps"][eps0][o]["best"]
        if not best:
            continue
        c = res["hard"][(ratio, o, best["gammaR2"])]
        x = (S_N + S_REC * np.arange(c["mean"].size)) / S_N
        ax.semilogy(x, c["mean"], color=ORDER_COLOR[o],
                    label=f"SAGA, {ORDER_LABEL[o]}, γR² = {best['gammaR2']:g} (best at ε = {S_EPS[0]:g}): mean of {len(S_SEEDS)} seeds")
        xmax = max(xmax, x[-1])
    x = np.linspace(1.0, xmax, 600)
    fl_iid = bounds.hard_instance_error_floor((x - 1.0) * S_N, S_N, kappa, S_M, initial_full_gradient=True)
    fl_any = bounds.hard_instance_error_floor(x * S_N, S_N, kappa, S_M, sampling="any")
    ax.semilogy(x, np.where(fl_iid > 0, fl_iid, np.nan), color=ps.INK["secondary"], linestyle=(0, (4, 2)), linewidth=1.4,
                label="floor on the expectation, i.i.d. span methods (expected-sweep)")
    ax.semilogy(x, np.where(fl_any > 0, fl_any, np.nan), color=ps.INK["muted"], linestyle=(0, (1, 2)), linewidth=1.4,
                label="lower bound, any index sequence")
    for eps in S_EPS:
        ax.axhline(eps, color=ps.INK["axis"], linewidth=0.8)
    be = e["by_eps"]
    xb = [(be[f"{eps:g}"]["bound_iid"], eps) for eps in S_EPS if be[f"{eps:g}"]["bound_iid"]]
    ax.plot([v[0] for v in xb], [v[1] for v in xb], linestyle="none", marker="D", markersize=5, color=ps.INK["secondary"],
            markeredgecolor=ps.INK["surface"], markeredgewidth=0.8, label="count of Corollary cor:sweep (table)")
    rb = [(be[f"{eps:g}"]["rr"]["best"], eps) for eps in S_EPS if be[f"{eps:g}"]["rr"]["best"]]
    ax.plot([v[0]["count"] for v in rb], [v[1] for v in rb], linestyle="none", marker="o", markersize=5,
            markerfacecolor=ps.INK["surface"], markeredgecolor=ORDER_COLOR["rr"], markeredgewidth=1.3,
            label="reshuffled: first passage below ε (best step per ε)")
    rs = [(v[0]["stay_count"], v[1]) for v in rb if v[0]["stay_count"]]
    ax.plot([v[0] for v in rs], [v[1] for v in rs], linestyle="none", marker="o", markersize=5, color=ORDER_COLOR["rr"],
            markeredgecolor=ps.INK["surface"], markeredgewidth=0.8, label="reshuffled: stays below ε from there on")
    ax.set_ylim(min(S_EPS) * 1e-3, 3.0)
    ax.set_xlim(0.0, xmax)
    ax.set_xlabel("gradient evaluations / n")
    ax.set_ylabel("‖θ − η*‖² / ‖η*‖²")
    ax.set_title(f"instance (S), κ/n = {ratio:g}, n = {S_N}")
    h, l = ax.get_legend_handles_labels()
    ps.legend_cell(fig, gs[3, 1], h, l, ncol=1)
    ps.suptitle(fig, "Figure B4 — sampling order and the memory term of SAGA: the ceiling of one nat per pass\n"
                "belongs to the expectation under sampling with replacement")
    ps.savefig(fig, "figB4_sampling_order")


def main():
    ps.setup()
    res = run_all()
    model = analyse_model(res)
    summary = {"definitions": DEFINITIONS, "config": {k: (v if not isinstance(v, dict) else {fk(a) if isinstance(a, float) else str(a): b for a, b in v.items()})
                                                      for k, v in CONFIG.items()},
               "model": model, "logistic": analyse_logistic(res, model), "hard_instance": analyse_hard(res)}
    plot(res, summary)
    common.save_json(TAG + "_summary", clean(summary))
    print("done B4")


if __name__ == "__main__":
    main()
