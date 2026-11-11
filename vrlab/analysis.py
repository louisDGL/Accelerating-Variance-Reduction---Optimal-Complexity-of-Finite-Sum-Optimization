"""Attribution of the theory/practice gap of Proposition 5.9 to each inequality of its proof.

For a SAGA state with statistics (D, M, G, S, B, Omega), the proof upper-bounds E[V_t] = E[D_t] + Delta E[M_t]
by applying, in order, I1 (square norm), I3 (variance <= second moment), I2 (cocoercivity, both occurrences),
I4 (strong convexity), I5 (max of the two rates). Replacing the inequalities one at a time by the exact
measured quantities gives a monotone chain of one-step rates
    rho_exact <= rho_I1 <= rho_I3 <= rho_I2 <= rho_I4 <= rho_I5 = rho_bound,
and the differences (1 - rho_k) - (1 - rho_{k+1}) (or the log-ratios) attribute the gap to each step.
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
    """One-step rate when a single inequality is replaced by the exact quantity, the others kept worst case."""
    D, M, G, S, B, Om = q["D"], q["M"], q["G"], q["S"], q["B"], q["Omega"]
    V = D + Delta * M
    cD, rho_D, rho_M = rate_coefficients(gamma, Delta, R2, mu, n)
    out = {}
    # I1 exact: keep Omega measured; everything else worst case (cocoercivity only in the M recursion, strong convexity)
    EV = D - 2 * gamma * G + gamma ** 2 * Om + Delta * ((1 - 1 / n) * M + R2 * G)
    EV = EV + 2 * gamma * G - 2 * gamma * mu * D - Delta * R2 * G + Delta * R2 * mu * D   # G -> mu D in the negative terms
    # (the term -2 gamma G is bounded using G >= mu D; the +Delta R2 G term would need an upper bound on G:
    #  we keep G measured there since the proof itself never needs it after cocoercivity+strong convexity)
    EV = D - 2 * gamma * mu * D + gamma ** 2 * Om + Delta * ((1 - 1 / n) * M + R2 * G)
    out["I1"] = EV / V
    # I3 exact: B measured; I1, I2, I4 worst case
    EV = D - 2 * gamma * cD * G + 2 * gamma ** 2 * B + Delta * (1 - 1 / n) * M
    EV = D * (1 - 2 * gamma * cD * mu) + 2 * gamma ** 2 * B + Delta * (1 - 1 / n) * M
    out["I3"] = EV / V
    # I2 exact: S measured (both occurrences), I1, I3, I4 worst case
    EV = D - 2 * gamma * G + 2 * gamma ** 2 * S / n + 2 * gamma ** 2 * M / n + Delta * ((1 - 1 / n) * M + S / n)
    EV = EV + 2 * gamma * G - 2 * gamma * mu * D
    out["I2"] = EV / V
    # I4 exact: mu -> mu_eff (G measured), I1-I3 worst case
    EV = D - 2 * gamma * cD * G + rho_M * Delta * M
    out["I4"] = EV / V
    # I5 exact: rho_D D + rho_M Delta M
    out["I5"] = (rho_D * D + rho_M * Delta * M) / V
    out["bound"] = max(rho_D, rho_M)
    return out


def attribution(instr, gamma, Delta, R2, mu, n, t_min=None, t_max=None, v_min=1e-20):
    """Aggregate attribution over the instrumentation records of one run (optionally a window of iterations).
    Returns per-record chains and the window-averaged contraction exponents c_k = -log(rho_k) per iteration,
    the share of each inequality (c_{k-1} - c_k) / (c_exact - c_bound) and the log-gap factors.
    Records with V < v_min are discarded (below ~1e-20 the exact contraction is dominated by rounding noise).
    Note: the proof is a *nested* chain of substitutions, so the cumulative attribution is the same whether the
    inequalities are relaxed in proof order or in reverse order (the intermediate rates coincide)."""
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


def trajectory_bounds(run, which=("mu_eff", "mu_eff+L_eff", "exact")):
    """Predicted curves on the recording grid from the per-step instrumented rates:
    product over iterations of (1 - min{1/(3n), 3 mu_eff/(16R^2)}) (a-posteriori bound with mu_eff),
    the same with L_eff replacing R^2 inside the step-dependent coefficient (gamma fixed), and the product of
    exact conditional contractions. Rates are held piecewise constant between instrumentation points."""
    instr = run["instr"]
    if not instr:
        return {}
    n, R2, mu, gamma, Delta = run["n"], run["R2"], run["mu"], run["gamma"], run["Delta"]
    ts = np.array([q["t"] for q in instr])
    V0 = run["V"][0]
    grid = np.asarray(run["iters"], dtype=np.float64)
    out = {}
    rate_funcs = {}
    if "mu_eff" in which:
        rate_funcs["mu_eff"] = np.array([max(1 - 1 / (3 * n), 1 - 2 * gamma * (1 - gamma * R2 - R2 * Delta / (2 * gamma)) * q["mu_eff"]) for q in instr])
    if "mu_eff+L_eff" in which:
        def rr(q):
            Le = min(q["L_eff"], R2)
            cD = 1 - gamma * Le - Le * Delta / (2 * gamma)
            return max(1 - 1 / (3 * n), 1 - 2 * gamma * cD * q["mu_eff"])
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
        out[name] = V0 * np.exp(vals)
    return out


def mean_over_seeds(runs, key="D"):
    """Average a recorded quantity over runs with identical recording grids (E over seeds)."""
    L = min(len(r[key]) for r in runs)
    X = np.array([np.asarray(r[key][:L]) for r in runs])
    return np.asarray(runs[0]["iters"][:L]), X.mean(0), X
