"""Attribution of the theory/practice gap of Proposition 5.9 to each inequality of its proof.

For a SAGA state with statistics (D, M, G, S, B, Omega), the proof upper-bounds E[V_t] = E[D_t] + Delta E[M_t]
by applying, in order, I1 (square norm), I3 (variance <= second moment), I2 (cocoercivity, both occurrences),
I4 (strong convexity), I5 (max of the two rates). Applying the inequalities cumulatively to the exact
measured quantities gives a monotone chain of one-step rates
    rho_exact <= rho_I1 <= rho_I3 <= rho_I2 <= rho_I4 <= rho_I5 = rho_bound,
and the differences (1 - rho_k) - (1 - rho_{k+1}) (or the log-ratios) attribute the gap to each step: this is order B
of the B3 note (cumulative_chain, attribution). The attribution is path dependent; order A is cumulative_chain_A,
attribution_A. leave_one_out gives single (non-nested) substitutions.
"""
from __future__ import annotations

import numpy as np

from .instrumentation import rate_coefficients

STEP_NAMES = ["exact", "I1 square norm", "I3 variance", "I2 cocoercivity", "I4 strong convexity", "I5 min"]
STEP_KEYS = ["exact", "I1", "I3", "I2", "I4", "I5"]


def cumulative_chain(q, gamma, Delta, R2, mu, n):
    """Chain of one-step rates obtained by applying the proof inequalities cumulatively (proof order)."""
    D, M, G, S, B, Om = q["D"], q["M"], q["G"], q["S"], q["B"], q["Omega"]
    V = D + Delta * M
    cD, rho_D, rho_M = rate_coefficients(gamma, Delta, R2, mu, n)
    base = D - 2 * gamma * G + Delta * ((1 - 1 / n) * M + S / n)        # common part
    EV_exact = base + gamma ** 2 * Om
    EV_I1 = base + gamma ** 2 * (2 * S / n + 2 * B)
    EV_I3 = base + gamma ** 2 * (2 * S / n + 2 * M / n)
    EV_I2 = D - 2 * gamma * cD * G + rho_M * Delta * M                    # S/n -> R^2 G in both occurrences
    EV_I4 = rho_D * D + rho_M * Delta * M
    EV_I5 = max(rho_D, rho_M) * V
    rates = np.array([EV_exact, EV_I1, EV_I3, EV_I2, EV_I4, EV_I5]) / V
    return rates


def cumulative_chain_A(q, gamma, Delta, R2, mu, n):
    """Order A of the B3 note: undo I4, I2, I3, I1 while keeping the max structure of the bound, and undo the
    min (I5) last. Returns [rho_bound, rho_A1 (I4 undone), rho_A2 (+I2), rho_A3 (+I3), rho_A4 (+I1), rho_exact]."""
    D, M, G, S, B, Om = q["D"], q["M"], q["G"], q["S"], q["B"], q["Omega"]
    V = D + Delta * M
    cD, rho_D, rho_M = rate_coefficients(gamma, Delta, R2, mu, n)
    eps = 1e-300
    r_sq = Om / max(2 * S / n + 2 * B, eps)
    rho_bound = max(rho_D, rho_M)
    rA1 = max(1 - 2 * gamma * cD * G / max(D, eps), rho_M)
    cD2 = 1 - 2 * gamma * G / max(D, eps) + (2 * gamma ** 2 + Delta) * S / (n * max(D, eps))
    rA2 = max(cD2, rho_M)
    cM3 = 1 - 1 / n + 2 * gamma ** 2 * B / max(Delta * M, eps)
    rA3 = max(cD2, cM3)
    cD4 = 1 - 2 * gamma * G / max(D, eps) + 2 * gamma ** 2 * r_sq * S / (n * max(D, eps)) + Delta * S / (n * max(D, eps))
    cM4 = 1 - 1 / n + 2 * gamma ** 2 * r_sq * B / max(Delta * M, eps)
    rA4 = max(cD4, cM4)
    rho_exact = q["EV_next"] / max(V, eps)
    return np.array([rho_bound, rA1, rA2, rA3, rA4, rho_exact])


STEP_KEYS_A = ["I4", "I2", "I3", "I1", "I5"]


def attribution_A(instr, gamma, Delta, R2, mu, n, t_min=None, t_max=None, v_min=1e-20):
    """Window-averaged shares for order A (I4, I2, I3, I1, I5) of the B3 note."""
    recs = [q for q in instr if (t_min is None or q["t"] >= t_min) and (t_max is None or q["t"] <= t_max)
            and (v_min is None or q["V"] >= v_min)]
    if not recs:
        return {"shares": np.full(5, np.nan), "step_keys": STEP_KEYS_A}
    chains = np.array([cumulative_chain_A(q, gamma, Delta, R2, mu, n) for q in recs])
    c = -np.log(np.clip(chains, 1e-300, None))
    c_mean = c.mean(0)                      # [bound, A1, A2, A3, A4, exact] increasing
    total = c_mean[-1] - c_mean[0]
    shares = np.diff(c_mean) / total if total > 0 else np.full(5, np.nan)
    return {"shares": shares, "step_keys": STEP_KEYS_A, "c_mean": c_mean}


def leave_one_out(q, gamma, Delta, R2, mu, n):
    """Single-substitution ("leave-one-out") one-step rates EV/V, V = D + Delta M, for the state statistics q.

    EV is an upper bound on E[V_t | state] in which ONE inequality is not applied (its quantity is kept measured);
    none of the variants applies I5 except 'bound'. With c_D = 1 - gamma R^2 - R^2 Delta/(2 gamma) and rho_D, rho_M
    as in rate_coefficients, the code computes exactly:
      'I1'    (1 - 2 gamma mu) D + gamma^2 Omega + Delta ((1 - 1/n) M + R^2 G):
              Omega measured (I1, and hence I3, not applied); I2 applied only in the memory recursion, with G measured
              there; I4 applied to the term -2 gamma G alone (not, as in the proof, to the net coefficient
              -2 gamma c_D G of G). This hybrid is NOT the proof with I1 undone, which would give
              (1 - (2 gamma - Delta R^2) mu) D + gamma^2 Omega + Delta (1 - 1/n) M.
      'I3'    (1 - 2 gamma c_D mu) D + 2 gamma^2 B + Delta (1 - 1/n) M: B measured; I1, I2 (both occurrences) and I4
              (net coefficient) applied.
      'I2'    (1 - 2 gamma mu) D + (2 gamma^2 + Delta) S/n + (2 gamma^2/n + Delta (1 - 1/n)) M: S measured in both
              occurrences; I1 and I3 applied; I4 applied to -2 gamma G alone.
      'I4'    D - 2 gamma c_D G + rho_M Delta M: G measured; I1, I3, I2 applied (= rho_I2 of cumulative_chain).
      'I5'    rho_D D + rho_M Delta M: I1-I4 applied, the max not taken (= rho_I4 of cumulative_chain).
      'bound' max(rho_D, rho_M) = rho_bound (no division by V).
    Every variant is >= rho_exact (each applies only valid inequalities to the exact E[V_t | state]). The single
    substitutions are not nested:
    'I3', 'I4', 'I5' are <= rho_bound, but 'I1' and 'I2' may exceed rho_bound and even 1: 'I1' when Delta R^2 G (the
    bound kept for Delta S/n) outweighs the decrease 2 gamma mu D (for the textbook constants, when mu_eff > 8 mu/3
    and M is small), 'I2' when L_eff mu_eff >> mu R^2. Such values are not contraction rates (they give the negative
    exponents -log(rate) of the 'loo' columns of results/digest.md)."""
    D, M, G, S, B, Om = q["D"], q["M"], q["G"], q["S"], q["B"], q["Omega"]
    V = D + Delta * M
    cD, rho_D, rho_M = rate_coefficients(gamma, Delta, R2, mu, n)
    out = {}
    # 'I1': Omega measured; G -> mu D in -2 gamma G only, Delta S/n -> Delta R^2 G with G measured (see the docstring)
    EV = D - 2 * gamma * mu * D + gamma ** 2 * Om + Delta * ((1 - 1 / n) * M + R2 * G)
    out["I1"] = EV / V
    # 'I3': B measured; I1, I2 (both occurrences), I4 on the net coefficient of G
    EV = D * (1 - 2 * gamma * cD * mu) + 2 * gamma ** 2 * B + Delta * (1 - 1 / n) * M
    out["I3"] = EV / V
    # 'I2': S measured (both occurrences); I1, I3; G -> mu D in -2 gamma G
    EV = D - 2 * gamma * G + 2 * gamma ** 2 * S / n + 2 * gamma ** 2 * M / n + Delta * ((1 - 1 / n) * M + S / n)
    EV = EV + 2 * gamma * G - 2 * gamma * mu * D
    out["I2"] = EV / V
    # 'I4': G measured (mu -> mu_eff); I1, I3, I2
    EV = D - 2 * gamma * cD * G + rho_M * Delta * M
    out["I4"] = EV / V
    # 'I5': rho_D D + rho_M Delta M
    out["I5"] = (rho_D * D + rho_M * Delta * M) / V
    out["bound"] = max(rho_D, rho_M)
    return out


def attribution(instr, gamma, Delta, R2, mu, n, t_min=None, t_max=None, v_min=1e-20):
    """Aggregate attribution over the instrumentation records of one run (optionally a window of iterations).
    Returns per-record chains and the window-averaged contraction exponents c_k = -log(rho_k) per iteration,
    the share of each inequality (c_{k-1} - c_k) / (c_exact - c_bound) and the log-gap factors.
    Records with V < v_min are discarded (the exact contraction is dominated by rounding noise at small V; the
    experiments pass v_min = 1e-18, the floor set by the precision of eta_*).
    This is order B of the B3 note: the inequalities are undone from the bound in the order I5, I4, I2, I3, I1, i.e.
    the intermediate rates of the proof applied cumulatively from the exact rate (cumulative_chain). The attribution
    is path dependent; order A is computed by attribution_A and gives different shares."""
    recs = [q for q in instr if (t_min is None or q["t"] >= t_min) and (t_max is None or q["t"] <= t_max)
            and (v_min is None or q["V"] >= v_min)]
    if not recs:
        nan5 = np.full(5, np.nan)
        return {"t": np.array([]), "chains": np.zeros((0, 6)), "c": np.zeros((0, 6)), "c_mean": np.full(6, np.nan), "shares": nan5,
                "step_keys": STEP_KEYS[1:], "speedup_total": np.nan, "loo_exponents": {k: np.nan for k in ["I1", "I3", "I2", "I4", "I5"]},
                "c_bound": np.nan, "c_exact": np.nan}
    chains = np.array([cumulative_chain(q, gamma, Delta, R2, mu, n) for q in recs])   # (T, 6)
    ts = np.array([q["t"] for q in recs])
    c = -np.log(np.clip(chains, 1e-300, None))         # contraction exponents per iteration
    c_mean = c.mean(0)
    total = c_mean[0] - c_mean[-1]
    shares = -(np.diff(c_mean)) / total if total > 0 else np.full(5, np.nan)
    loo = {k: np.mean([-np.log(max(leave_one_out(q, gamma, Delta, R2, mu, n)[k], 1e-300)) for q in recs])
           for k in ["I1", "I3", "I2", "I4", "I5"]}
    return {"t": ts, "chains": chains, "c": c, "c_mean": c_mean, "shares": shares, "step_keys": STEP_KEYS[1:],
            "speedup_total": c_mean[0] / c_mean[-1] if c_mean[-1] > 0 else np.nan,
            "loo_exponents": loo, "c_bound": c_mean[-1], "c_exact": c_mean[0]}


def trajectory_bounds(run, which=("mu_eff", "mu_eff+L_eff", "exact"), v_min=None):
    """Predicted curves on the recording grid from the per-step instrumented rates, for the run's (gamma, Delta)
    (Proposition prop:B3-traj of the B3 note, inequality ineq:B3-traj-general):
    product over iterations of (1 - min{1 - rho_M, 2 gamma c_D mu_eff}) with c_D = 1 - gamma R^2 - R^2 Delta/(2 gamma)
    and rho_M = 1 - 1/n + 2 gamma^2/(n Delta) (a-posteriori bound with mu_eff; for the textbook constants this is
    (1 - min{1/(3n), 3 mu_eff/(16R^2)})), the same with min{L_eff, R^2} replacing R^2 inside c_D (gamma fixed), and the
    product of exact conditional contractions. Rates are held piecewise constant from each instrumentation point to the
    next (including the record at t = 0, taken with the fresh memory z_i = f'_i(theta_0)).
    If v_min is given, the curves are cut (NaN) after the first instrumentation record with V < v_min: below that
    level the conditional expectations are dominated by rounding and the products are no longer meaningful."""
    instr = run["instr"]
    if not instr:
        return {}
    bad = [q["t"] for q in instr if v_min is not None and q["V"] < v_min]
    t_stop = bad[0] if bad else np.inf
    n, R2, mu, gamma, Delta = run["n"], run["R2"], run["mu"], run["gamma"], run["Delta"]
    cD, _, rho_M = rate_coefficients(gamma, Delta, R2, mu, n)
    ts = np.array([q["t"] for q in instr])
    V0 = run["V"][0]
    grid = np.asarray(run["iters"], dtype=np.float64)
    out = {}
    rate_funcs = {}
    if "mu_eff" in which:
        rate_funcs["mu_eff"] = np.array([max(rho_M, 1 - 2 * gamma * cD * q["mu_eff"]) for q in instr])
    if "mu_eff+L_eff" in which:
        def rr(q):
            Le = min(q["L_eff"], R2)
            cD_eff = 1 - gamma * Le - Le * Delta / (2 * gamma)
            return max(rho_M, 1 - 2 * gamma * cD_eff * q["mu_eff"])
        rate_funcs["mu_eff+L_eff"] = np.array([rr(q) for q in instr])
    if "exact" in which:
        rate_funcs["exact"] = np.array([q["rho_exact"] for q in instr])
    for name, rates in rate_funcs.items():
        logr = np.log(np.clip(rates, 1e-300, 1.0))
        # cumulative log-contraction at instrumentation times (piecewise constant rate between points)
        seg = np.diff(np.append(ts, grid[-1]))
        cum = np.concatenate([[0.0], np.cumsum(logr * seg)])        # at ts[0], ts[1], ..., end
        knots = np.append(ts, grid[-1])
        vals = np.interp(grid, knots, cum)
        curve = V0 * np.exp(vals)
        curve[grid > t_stop] = np.nan
        out[name] = curve
    return out


def loglog_slope(x, y, xmin=None, xmax=None, level=0.95):
    """Least-squares fit of log y = a + b log x over the points with xmin <= x <= xmax and finite y > 0.
    Returns the slope b, its two-sided `level` t-interval b +/- t_{(1+level)/2, N-2} se(b) with
    se(b)^2 = RSS / ((N - 2) sum (log x - mean)^2), the half-width and the number of points N.
    Pass ONE point per x (e.g. the geometric mean over seeds): repeated seeds are not independent replicates when the
    method or the instance is deterministic (identical counts), and would inflate the degrees of freedom. With one
    point per x the interval measures the departure of the curve from a single power law over the window (the
    seed-to-seed variation is small in our runs): a descriptive interval, not a sampling one. With balanced seeds the
    slope is the same as the slope fitted to all (x, seed) pairs; only the interval changes."""
    from scipy import stats
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if xmin is not None:
        keep &= x >= xmin
    if xmax is not None:
        keep &= x <= xmax
    lx, ly = np.log(x[keep]), np.log(y[keep])
    out = {"slope": np.nan, "ci_low": np.nan, "ci_high": np.nan, "half_width": np.nan, "n_points": int(lx.size),
           "n_x": int(np.unique(lx).size), "level": level}
    if np.unique(lx).size < 2:
        return out
    b, a = np.polyfit(lx, ly, 1)
    out["slope"] = float(b)
    if lx.size > 2:
        resid = ly - (a + b * lx)
        se = np.sqrt((resid @ resid) / (lx.size - 2) / np.sum((lx - lx.mean()) ** 2))
        h = float(stats.t.ppf(0.5 + level / 2, lx.size - 2) * se)
        out.update(ci_low=float(b - h), ci_high=float(b + h), half_width=h)
    return out


def mean_over_seeds(runs, key="D"):
    """Average a recorded quantity over runs with identical recording grids (E over seeds)."""
    L = min(len(r[key]) for r in runs)
    X = np.array([np.asarray(r[key][:L]) for r in runs])
    return np.asarray(runs[0]["iters"][:L]), X.mean(0), X
