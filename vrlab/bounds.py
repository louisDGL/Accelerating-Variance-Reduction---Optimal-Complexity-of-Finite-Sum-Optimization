"""Theoretical rates and bounds."""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------------------------
# Proposition 5.9 (SAGA) and its general (gamma, Delta) version
# ----------------------------------------------------------------------------------------------
def saga_textbook_params(R2):
    """gamma = 1/(4R^2), Delta = 3 gamma^2 (Bach, Prop. 5.9)."""
    gamma = 1.0 / (4.0 * R2)
    return gamma, 3.0 * gamma ** 2


def saga_rate(n, R2, mu, gamma=None, Delta=None):
    """One-step contraction factor rho of the Lyapunov function in the proof of Prop. 5.9, for general
    (gamma, Delta): rho = max(1 - 2 gamma mu (1 - gamma R^2 - R^2 Delta/(2 gamma)), 1 - 1/n + 2 gamma^2/(n Delta)).
    Returns rho (and None if the parameters violate the positivity conditions)."""
    if gamma is None or Delta is None:
        gamma, Delta = saga_textbook_params(R2)
    cD = 1.0 - gamma * R2 - R2 * Delta / (2 * gamma)
    cM = 1.0 / n - 2 * gamma ** 2 / (n * Delta)
    if cD <= 0 or cM <= 0:
        return None
    return max(1.0 - 2 * gamma * mu * cD, 1.0 - cM)


def saga_bound(t, n, R2, mu, D0, gamma=None, Delta=None, S0=None):
    """Prop. 5.9 bound on E||theta_t - eta_*||^2 as a function of the iteration count t (array ok).
    With the textbook constants and z_i^(0) = f'_i(theta_0): (1 - min{1/(3n), 3mu/(16R^2)})^t (1 + 3n/16) D0, the
    constant obtained in the proof (the statement (5.30) rounds it up to 1 + n/4); in general rho^t (1 + Delta n R^4) D0
    with rho = saga_rate(n, R2, mu, gamma, Delta).
    If S0 = sum_i ||f'_i(theta_0) - f'_i(eta_*)||^2 is given, the sharper factor (D0 + Delta S0) is used
    (this is the bound before the Lipschitz step I6)."""
    if gamma is None or Delta is None:
        gamma, Delta = saga_textbook_params(R2)
    rho = saga_rate(n, R2, mu, gamma, Delta)
    t = np.asarray(t, dtype=np.float64)
    if S0 is None:
        V0 = D0 + Delta * n * R2 ** 2 * D0
    else:
        V0 = D0 + Delta * S0
    return rho ** t * V0


def saga_optimal_params(n, R2, mu, c_grid=None, gamma_grid=None):
    """Best (gamma, Delta) achievable *within the proof structure* of Prop. 5.9: maximise 1 - rho over
    Delta = c gamma^2 (c > 2) and gamma. Closed form for fixed c: gamma*(c) = 1/(R^2 (2 + c)) and the
    rates are mu/(R^2 (2 + c)) and (1 - 2/c)/n; the best c balances the two. Returns a dict."""
    if c_grid is None:
        c_grid = np.exp(np.linspace(np.log(2.001), np.log(1e4), 2000))
    best = None
    for c in c_grid:
        gamma = 1.0 / (R2 * (2.0 + c))
        rateD = mu / (R2 * (2.0 + c))
        rateM = (1.0 - 2.0 / c) / n
        rate = min(rateD, rateM)
        if best is None or rate > best["rate"]:
            best = {"c": float(c), "gamma": float(gamma), "Delta": float(c * gamma ** 2), "rate": float(rate),
                    "rateD": float(rateD), "rateM": float(rateM)}
    gamma_tb, Delta_tb = saga_textbook_params(R2)
    rho_tb = saga_rate(n, R2, mu, gamma_tb, Delta_tb)
    best["rate_textbook"] = 1.0 - rho_tb
    best["gain_over_textbook"] = best["rate"] / (1.0 - rho_tb)
    return best


# ----------------------------------------------------------------------------------------------
# full-gradient methods (rates per iteration on ||theta - eta_*||^2)
# ----------------------------------------------------------------------------------------------
def gd_rate(L, mu):
    """GD with step 1/L on an L-smooth mu-strongly convex F: ||theta_t - eta_*||^2 <= (1 - mu/L)^{2t} ... we
    return the per-iteration contraction of the squared distance, (1 - mu/L)^2 (Bach, Sec. 5.2.3 gives
    exp(-2 t mu / L) with step 1/L)."""
    return (1.0 - mu / L) ** 2


def agd_rate(L, mu):
    """Nesterov acceleration for strongly convex functions: function values contract like (1 - sqrt(mu/L))^t."""
    return 1.0 - np.sqrt(mu / L)


# ----------------------------------------------------------------------------------------------
# L-Katyusha (Kovalev, Horvath, Richtarik, ALT 2020): parameters for uniform sampling
# ----------------------------------------------------------------------------------------------
def lkatyusha_params(n, L, mu, p=None, theta2=0.5):
    """Default parameters of L-Katyusha (Algorithm 2 of Kovalev, Horvath, Richtarik, ALT 2020) with uniform sampling:
    sigma = mu/L, theta_2 = 1/2, theta_1 = min{sqrt(2 sigma n / 3), 1/2}, eta = theta_2 / ((1 + theta_2) theta_1), p = 1/n.
    (L is the smoothness constant of each f_i, i.e. R^2 in the book's notation.)"""
    p = 1.0 / n if p is None else p
    sigma = mu / L
    theta1 = min(np.sqrt(2.0 * sigma * n / 3.0), 0.5)
    eta = theta2 / ((1.0 + theta2) * theta1)
    return {"sigma": sigma, "theta1": theta1, "theta2": theta2, "eta": eta, "p": p, "L": L, "mu": mu}


def lkatyusha_rate(n, L, mu, p=None, theta2=0.5, theta1=None):
    """Per-iteration contraction factor max{c_Z, c_Y, c_W} of the L-Katyusha Lyapunov function Psi^k = Z^k + Y^k + W^k
    (Kovalev, Horvath, Richtarik, ALT 2020, Lemma 10 and Theorem 11; Lemma 4.5 and Theorem 4.6 in the arXiv version
    1901.08689): Lemma 10 gives E[Psi^{k+1}] <= c_Z Z^k + c_Y Y^k + c_W W^k, hence
        E[Psi^{k+1}] <= max{ 1/(1 + eta sigma), 1 - theta1 (1 - theta2), 1 - p theta1/(1 + theta1) } Psi^k,
    with eta = theta2/((1 + theta2) theta1), sigma = mu/L. Theorem 11 (theta1 = min{sqrt(2 sigma n/3), 1/2},
    theta2 = 1/2, p = 1/n) bounds this factor by 1 - min{sigma/(6 theta1), theta1/(2n)}, which gives
    O((n + sqrt(n L/mu)) log(1/eps)) iterations; the returned value is the factor itself, not this closed form."""
    prm = lkatyusha_params(n, L, mu, p, theta2)
    theta1 = prm["theta1"] if theta1 is None else theta1
    eta = theta2 / ((1.0 + theta2) * theta1)
    es = eta * prm["sigma"]
    return max(1.0 / (1.0 + es), 1.0 - theta1 * (1.0 - theta2), 1.0 - prm["p"] * theta1 / (1.0 + theta1))


def complexity_table(n, kappa, eps=1e-6):
    """Order-of-magnitude gradient-evaluation counts to reach accuracy eps (constants dropped)."""
    log = np.log(1.0 / eps)
    return {
        "GD": n * kappa * log,
        "AGD": n * np.sqrt(kappa) * log,
        "SAGA": (n + kappa) * log,
        "L-Katyusha": (n + np.sqrt(n * kappa)) * log,
        "lower bound": (n + np.sqrt(n * kappa)) * log,
    }
