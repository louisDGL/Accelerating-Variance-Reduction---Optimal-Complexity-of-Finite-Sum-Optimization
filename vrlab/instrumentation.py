"""Exact conditional expectations of all quantities in the proof of Proposition 5.9 (Bach, Sec. 5.4.4)
and the tightness ratio of each inequality, for a SAGA state (theta_{t-1}, z^{(t-1)}).

Quantities (all conditional on the state, expectation over i(t) uniform in {1..n}; dual/primal norms
in a quadratic metric P when a preconditioner is used, Euclidean otherwise):
  D     = ||theta - eta_*||_P^2
  M     = sum_i ||z_i - f'_i(eta_*)||_{P^-1}^2                       (gradient-memory error)
  G     = (theta - eta_*)^T F'(theta)                                 (metric-free)
  S     = sum_i ||f'_i(theta) - f'_i(eta_*)||_{P^-1}^2
  B     = E_i ||f'_i(eta_*) - z_i + zbar||_{P^-1}^2 = M/n - ||zbar||_{P^-1}^2   (variance term)
  Omega = E_i ||f'_i(theta) + zbar - z_i||_{P^-1}^2                   (second moment of SAGA direction)
Exact one-step identities (Delta = Lyapunov weight):
  E[D_t] = D - 2 gamma G + gamma^2 Omega
  E[M_t] = (1 - 1/n) M + S/n
  E[V_t] = E[D_t] + Delta E[M_t]
Proof inequalities and tightness ratios in (0, 1]:
  I1 square norm           Omega <= 2 S/n + 2 B          r_sq   = Omega / (2S/n + 2B)
  I2 cocoercivity (x2)     S/n   <= R^2 G                r_coco = (S/n) / (R^2 G) = L_eff / R^2
  I3 variance <= 2nd mom.  B     <= M/n                  r_var  = B / (M/n)
  I4 strong convexity      G     >= mu D                 r_sc   = mu D / G = mu / mu_eff
  I5 min of the two rates  rho_D D + rho_M Delta M <= max(rho_D, rho_M) (D + Delta M)
  I6 initialization        S_0 <= n R^4 D_0              r_init = S_0 / (n R^4 D_0)
"""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------------------------
# quadratic metric P (preconditioner)
# ----------------------------------------------------------------------------------------------
class Metric:
    """Quadratic metric ||v||_P^2 = v^T P v with dual norm ||g||_{P^-1}^2. P may be None (identity),
    a 1-d array (diagonal) or a dense SPD matrix."""

    def __init__(self, P=None, d=None):
        self.P = P
        if P is None:
            self.kind = "identity"
        elif np.ndim(P) == 1:
            self.kind = "diag"
            self.P = np.asarray(P, dtype=np.float64)
            assert np.all(self.P > 0)
            self.Pinv = 1.0 / self.P
        else:
            self.kind = "dense"
            self.P = np.asarray(P, dtype=np.float64)
            self.chol = np.linalg.cholesky(self.P)                 # P = L L^T
            self.Pinv = np.linalg.inv(self.P)
            self.Pinv = 0.5 * (self.Pinv + self.Pinv.T)
            self.eigs = np.linalg.eigvalsh(self.P)

    # primal norm ||v||_P^2
    def norm2(self, v):
        if self.kind == "identity":
            return float(v @ v)
        if self.kind == "diag":
            return float(v @ (self.P * v))
        return float(v @ (self.P @ v))

    # dual norm ||g||_{P^-1}^2
    def dual_norm2(self, g):
        if self.kind == "identity":
            return float(g @ g)
        if self.kind == "diag":
            return float(g @ (self.Pinv * g))
        return float(g @ (self.Pinv @ g))

    def dual_norm2_rows(self, Gm):
        """Row-wise dual norms of an (n x d) matrix of gradients."""
        if self.kind == "identity":
            return np.einsum("ij,ij->i", Gm, Gm)
        if self.kind == "diag":
            return np.einsum("ij,ij->i", Gm, Gm * self.Pinv[None, :])
        return np.einsum("ij,ij->i", Gm, Gm @ self.Pinv)

    def apply_inv(self, g):
        if self.kind == "identity":
            return g
        if self.kind == "diag":
            return self.Pinv * g
        return self.Pinv @ g

    def sqrt_apply(self, v):
        """P^{1/2} v (symmetric square root), used for the change of variables u = P^{1/2} theta."""
        if self.kind == "identity":
            return v
        if self.kind == "diag":
            return np.sqrt(self.P) * v
        w, U = np.linalg.eigh(self.P)
        return U @ (np.sqrt(w) * (U.T @ v))

    def constants(self, prob, theta_ref=None):
        """Metric constants (R_P^2, mu_P) for a logistic problem: R_P^2 = max_i ||x_i||^2_{P^-1}/4 + lam lmax(P^-1),
        mu_P = lam * lmin(P^-1) (global); if theta_ref is given also the local mu_P at theta_ref."""
        X = prob.X
        if self.kind == "identity":
            lev = np.einsum("ij,ij->i", X, X)
            lmax_inv, lmin_inv = 1.0, 1.0
        elif self.kind == "diag":
            lev = np.einsum("ij,ij->i", X, X * self.Pinv[None, :])
            lmax_inv, lmin_inv = float(self.Pinv.max()), float(self.Pinv.min())
        else:
            lev = np.einsum("ij,ij->i", X, X @ self.Pinv)
            lmax_inv, lmin_inv = 1.0 / float(self.eigs[0]), 1.0 / float(self.eigs[-1])
        R2P = float(lev.max() / 4.0 + prob.lam * lmax_inv)
        muP = float(prob.lam * lmin_inv)
        out = {"R2_P": R2P, "mu_P": muP, "kappa_P": R2P / muP, "max_lev": float(lev.max()), "mean_lev": float(lev.mean())}
        if theta_ref is not None:
            H = prob.hessian(theta_ref)
            Pm = np.eye(prob.d) if self.kind == "identity" else (np.diag(self.P) if self.kind == "diag" else self.P)
            # generalized eigenvalues of (H, P)
            from scipy.linalg import eigh
            ev = eigh(H, Pm, eigvals_only=True)
            out.update({"mu_P_local": float(ev[0]), "L_P_local": float(ev[-1]), "kappa_P_local": float(ev[-1] / ev[0])})
        return out


# ----------------------------------------------------------------------------------------------
# state statistics
# ----------------------------------------------------------------------------------------------
def saga_state_stats(prob, theta, z, zbar, gamma, Delta, R2, mu, metric=None, Gall=None):
    """All proof quantities for the state (theta, z) (before the step), exactly (O(n d))."""
    metric = metric or Metric(None)
    n = prob.n
    gs = prob.grads_star
    v = theta - prob.theta_star
    if Gall is None:
        Gall = prob.all_grads(theta)
    Fgrad = Gall.mean(0)
    D = metric.norm2(v)
    G = float(v @ Fgrad)
    S = float(metric.dual_norm2_rows(Gall - gs).sum())
    M = float(metric.dual_norm2_rows(z - gs).sum())
    B = float(metric.dual_norm2_rows(gs - z + zbar[None, :]).mean())
    Omega = float(metric.dual_norm2_rows(Gall + zbar[None, :] - z).mean())
    V = D + Delta * M
    ED = D - 2 * gamma * G + gamma ** 2 * Omega
    EM = (1 - 1 / n) * M + S / n
    EV = ED + Delta * EM
    out = dict(D=D, M=M, G=G, S=S, B=B, Omega=Omega, V=V, ED_next=ED, EM_next=EM, EV_next=EV,
               grad_norm2=metric.dual_norm2(Fgrad), zbar_norm2=metric.dual_norm2(zbar))
    out.update(tightness_ratios(out, gamma, Delta, R2, mu, n))
    return out


def rate_coefficients(gamma, Delta, R2, mu, n):
    """Coefficients of the worst-case recursion E[V_t] <= rho_D D + rho_M Delta M for general (gamma, Delta)."""
    cD = 1.0 - gamma * R2 - R2 * Delta / (2 * gamma)        # = 3/8 for the textbook constants
    rho_D = 1.0 - 2 * gamma * cD * mu                        # = 1 - 3 mu / (16 R^2)
    rho_M = 1.0 - 1.0 / n + 2 * gamma ** 2 / (n * Delta)     # = 1 - 1/(3n)
    return cD, rho_D, rho_M


def tightness_ratios(q, gamma, Delta, R2, mu, n):
    """Tightness ratios (LHS/RHS in (0,1]) of the inequalities I1-I5 for the state statistics q."""
    D, M, G, S, B, Om = q["D"], q["M"], q["G"], q["S"], q["B"], q["Omega"]
    eps = 1e-300
    cD, rho_D, rho_M = rate_coefficients(gamma, Delta, R2, mu, n)
    rho_bound = max(rho_D, rho_M)
    V = D + Delta * M
    r = {
        "r_sq": Om / max(2 * S / n + 2 * B, eps),
        "r_coco": (S / n) / max(R2 * G, eps),
        "r_var": B / max(M / n, eps),
        "r_sc": mu * D / max(G, eps),
        "r_min": (rho_D * D + rho_M * Delta * M) / max(rho_bound * V, eps),
        "mu_eff": G / max(D, eps),
        "L_eff": (S / n) / max(G, eps),
        "rho_exact": q["EV_next"] / max(V, eps),
        "rho_bound": rho_bound,
        "rho_D": rho_D, "rho_M": rho_M,
        "Dshare": D / max(V, eps),
    }
    return r


def lkatyusha_state_stats(prob, y, z, w, theta1, theta2, eta, L, mu, p, Gw=None, gw=None):
    """Exact conditional expectations for one step of L-Katyusha (Kovalev, Horvath, Richtarik 2020) from the
    state (y, z, w): realized Lyapunov Psi = Z + Y + W with
        Z = L(1 + eta sigma)/(2 eta) ||z - eta_*||^2,  Y = (F(y) - F_*)/theta1,  W = theta2 (1 + theta1)/(p theta1) (F(w) - F_*),
    the exact E[Psi_{k+1} | state] (enumerating the n possible indices for the non-linear Y term; O(n^2 d)),
    the per-step contraction rho_exact = E[Psi_{k+1}]/Psi versus the Lemma 4.5 factor, and the tightness of the
    estimator variance bound  E||g - F'(x)||^2 <= 2 L [F(w) - F(x) - <F'(x), w - x>]  (ratio in (0,1])."""
    n = prob.n
    sigma = mu / L
    es = eta * sigma
    xs, Fs = prob.theta_star, prob.F_star
    x = theta1 * z + theta2 * w + (1.0 - theta1 - theta2) * y
    Gx = prob.all_grads(x)
    Fx_grad = Gx.mean(0)
    if Gw is None:
        Gw = prob.all_grads(w)
        gw = Gw.mean(0)
    Gmat = Gx - Gw + gw[None, :]                                   # all possible estimators g^(i)
    var_g = float(np.einsum("ij,ij->i", Gmat - Fx_grad[None, :], Gmat - Fx_grad[None, :]).mean())
    Fx, Fw, Fy = prob.F(x), prob.F(w), prob.F(y)
    bregman = Fw - Fx - float(Fx_grad @ (w - x))                   # D_F(w, x) >= 0
    # A3 note, Table "what to instrument": centering and co-coercivity factors of the variance bound, and the
    # Bregman-gap cancellation shares c(t), 1 - c(t), and q(t) = |theta2/theta1 <F'(x), x - w>| / (rho Psi)
    second_moment = float(np.einsum("ij,ij->i", Gx - Gw, Gx - Gw).mean())       # (1/n) sum_i ||f'_i(x) - f'_i(w)||^2
    inner_xw = float(Fx_grad @ (x - w))                                         # <F'(x), x - w>
    cZ = L * (1.0 + es) / (2.0 * eta)
    Z = cZ * float(np.sum((z - xs) ** 2))
    Y = (Fy - Fs) / theta1
    W = theta2 * (1.0 + theta1) / (p * theta1) * (Fw - Fs)
    Psi = Z + Y + W
    Znext = (es * x[None, :] + z[None, :] - (eta / L) * Gmat) / (1.0 + es)       # n x d
    Ynext = x[None, :] + theta1 * (Znext - z[None, :])
    EZ = cZ * float(np.einsum("ij,ij->i", Znext - xs[None, :], Znext - xs[None, :]).mean())
    EY = float((prob.F_batch(Ynext) - Fs).mean()) / theta1
    EW = (1.0 - p) * W + theta2 * (1.0 + theta1) / theta1 * (Fy - Fs)
    EPsi = EZ + EY + EW
    rho_lemma = max(1.0 / (1.0 + es), 1.0 - theta1 * (1.0 - theta2), 1.0 - p * theta1 / (1.0 + theta1))
    eps = 1e-300
    defined = bregman > 1e-14 * max(abs(Fx), 1e-300) + 1e-300                   # undefined when w = x
    r_var = var_g / (2.0 * L * bregman) if defined else np.nan
    r_cen = var_g / second_moment if (defined and second_moment > 0) else np.nan   # centering: 1 - ||F'(x)-F'(w)||^2 / second moment
    r_coco_lk = second_moment / (2.0 * L * bregman) if defined else np.nan          # co-coercivity (Bregman form) of the per-sample gradients
    c_share = inner_xw / bregman if defined else np.nan                             # share of D_F(w,x) removed by the anchor (signed)
    one_minus_rho = 1.0 - rho_lemma
    q_cancel = abs(theta2 / theta1 * inner_xw) / max(one_minus_rho * Psi, eps)      # cancelled term over the certified decrease
    return dict(Z=Z, Y=Y, W=W, Psi=Psi, EZ_next=EZ, EY_next=EY, EW_next=EW, EPsi_next=EPsi,
                rho_exact=EPsi / max(Psi, eps), rho_lemma=rho_lemma,
                var_g=var_g, bregman=bregman, r_var=r_var, r_cen=r_cen, r_coco_lk=r_coco_lk, c_share=c_share, q_cancel=q_cancel,
                second_moment=second_moment, inner_xw=inner_xw,
                Fgap_y=Fy - Fs, Fgap_w=Fw - Fs, Fgap_x=Fx - Fs, Dz=float(np.sum((z - xs) ** 2)))


def init_ratio(prob, theta0, R2, metric=None):
    """r_init = sum_i ||f'_i(theta_0) - f'_i(eta_*)||^2 / (n R^4 ||theta_0 - eta_*||^2) (inequality I6)."""
    metric = metric or Metric(None)
    G0 = prob.all_grads(theta0)
    S0 = float(metric.dual_norm2_rows(G0 - prob.grads_star).sum())
    D0 = metric.norm2(theta0 - prob.theta_star)
    return S0 / (prob.n * R2 ** 2 * D0), S0, D0
