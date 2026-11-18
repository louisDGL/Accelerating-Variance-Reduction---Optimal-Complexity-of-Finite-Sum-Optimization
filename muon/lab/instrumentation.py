"""Trajectory-dependent (effective) constants of the matrix problems, in the Euclidean and the spectral geometries.

`instrument(prob, W, G, O, eta)` evaluates, at a point W with E = W - W_*, a direction source G (an estimator of the
gradient; default the full gradient grad F(W)) and an update direction O (default orth(G)), the following quantities
(all exact, O(n K d) plus O(n K^2 d) for the per-sample nuclear norms; NaN when a denominator vanishes):

  s            = <grad F(W), E>  (= <grad F(W) - grad F(W_*), E>, the secant of vrlab's "G"); s >= lam ||E||_F^2
  dist2_F      = ||E||_F^2,  dist2_op = ||E||_op^2,  srank_E = ||E||_F^2 / ||E||_op^2 in [1, r]  (stable rank)
  mu_eff_F     = s / ||E||_F^2      Euclidean effective strong convexity (vrlab's mu_eff), >= mu = lam
  mu_eff_op    = s / ||E||_op^2     spectral effective strong convexity, = srank_E mu_eff_F >= mu_op = lam
  nrank_G      = ||G||_*^2 / ||G||_F^2 in [1, rank G]   (nuclear rank of G)
  L_eff_F      = (1/n) sum_i ||g_i'(W) - g_i'(W_*)||_F^2 / s     (loss parts g_i only). The differences are rank one,
                 (r_i - r_i^*) x_i^T, with norm ||r_i - r_i^*|| ||x_i||; co-coercivity gives L_eff_F <= max_i c_l||x_i||^2
  L_eff_F_reg  = (1/n) sum_i ||f_i'(W) - f_i'(W_*)||_F^2 / s     (f_i = g_i + (lam/2)||.||_F^2: vrlab's L_eff); <= R2
  L_eff_op     = (1/n) sum_i ||g_i'(W) - g_i'(W_*)||_*^2 / s     = L_eff_F (rank one: nuclear = Frobenius norm)
  L_eff_op_reg = (1/n) sum_i ||f_i'(W) - f_i'(W_*)||_*^2 / s     <= R2_op (co-coercivity in the pair (||.||_op, ||.||_*))
  curv_O       = <O, hess F(W) O>;  curv_O_F = curv_O / ||O||_F^2;  curv_O_op = curv_O / ||O||_op^2
  curv_G       = <G, hess F(W) G> / ||G||_F^2
  pair         = <grad F(W), O> (pairing with the TRUE gradient);  pair_G = <G, O> (with the estimator)
  nuc_grad     = ||grad F(W)||_*;  align_dual = pair / nuc_grad (in [-1, 1] when ||O||_op <= 1; 1 iff O acts as
                 polar(grad F) on its range)
  cos_G        = <G, grad F(W)> / (||G||_F ||grad F(W)||_F)
  eta_ls       = pair / curv_O      minimizer of the quadratic model of F along W - eta O
and, if a step eta is given (W+ = W - eta O):
  decrease       = F(W) - F(W+)     exact (Bregman form of problems.F_diff, accurate for tiny steps)
  decrease_model = eta pair - eta^2 curv_O / 2          (equals `decrease` for LS)
  decrease_bound = eta pair - eta^2 L ||O||_op^2 / 2    (L = prob.L_op by default: the decrease guaranteed by
                                                           op-norm smoothness)
"""
from __future__ import annotations

import numpy as np

from .polar import get_orth, singular_values


def _ratio(a, b):
    return float(a) / float(b) if b > 0 else float("nan")


def instrument(prob, W, G=None, O=None, eta=None, orth="svd", orth_kw=None, L=None, per_sample=True, chunk=2048):
    """Effective constants at W (see the module docstring). per_sample=False skips the per-sample nuclear norms
    (L_eff_op_reg, the only O(n K^2 d) part). Returns a dict of floats."""
    if prob.W_star is None:
        raise ValueError("set the optimum first (lab.optimum.solve_optimum(prob))")
    W = np.asarray(W, dtype=np.float64)
    E = W - prob.W_star
    gF = prob.grad(W)
    G = gF if G is None else np.asarray(G, dtype=np.float64)
    if O is None:
        O = get_orth(orth, **(orth_kw or {}))(G)
    s = float(np.vdot(gF, E))
    sE = singular_values(E)
    dF2 = float((sE ** 2).sum())
    dop2 = float(sE[0] ** 2) if sE.size else 0.0
    sG = singular_values(G)
    sO = singular_values(O)
    sgF = singular_values(gF)
    froG2 = float((sG ** 2).sum())
    froO2 = float((sO ** 2).sum())
    opO2 = float(sO[0] ** 2) if sO.size else 0.0
    nuc_grad = float(sgF.sum())

    # per-sample gradient differences
    dR = prob.loss_residuals(W) - prob.R_star                  # n x K, rows r_i - r_i^*
    a = np.einsum("ij,ij->i", dR, dR) * prob.sqnorms          # ||g_i'(W) - g_i'(W_*)||_F^2
    b = np.einsum("ij,ij->i", dR, prob.X @ E.T)               # <r_i - r_i^*, E x_i>
    lam = prob.lam
    S_loss = float(a.mean())
    S_reg = float((a + 2.0 * lam * b).mean() + lam ** 2 * dF2)
    out = {
        "s": s, "dist2_F": dF2, "dist2_op": dop2, "srank_E": _ratio(dF2, dop2),
        "mu_eff_F": _ratio(s, dF2), "mu_eff_op": _ratio(s, dop2),
        "nrank_G": _ratio(sG.sum() ** 2, froG2),
        "L_eff_F": _ratio(S_loss, s), "L_eff_F_reg": _ratio(S_reg, s), "L_eff_op": _ratio(S_loss, s),
        "S_loss": S_loss, "S_reg": S_reg,
        "grad_norm_F": float(np.sqrt((sgF ** 2).sum())), "nuc_grad": nuc_grad,
        "G_norm_F": float(np.sqrt(froG2)), "nuc_G": float(sG.sum()),
    }
    if per_sample:
        nuc2 = np.empty(prob.n)
        for lo in range(0, prob.n, chunk):
            hi = min(prob.n, lo + chunk)
            stack = dR[lo:hi, :, None] * prob.X[lo:hi, None, :] + lam * E[None, :, :]
            nuc2[lo:hi] = np.linalg.svd(stack, compute_uv=False).sum(axis=1) ** 2
        out["S_op_reg"] = float(nuc2.mean())
        out["L_eff_op_reg"] = _ratio(out["S_op_reg"], s)
    curv_O = float(np.vdot(O, prob.hvp(W, O)))
    curv_G = float(np.vdot(G, prob.hvp(W, G)))
    pair = float(np.vdot(gF, O))
    out.update({
        "curv_O": curv_O, "curv_O_F": _ratio(curv_O, froO2), "curv_O_op": _ratio(curv_O, opO2),
        "curv_G": _ratio(curv_G, froG2),
        "pair": pair, "pair_G": float(np.vdot(G, O)), "align_dual": _ratio(pair, nuc_grad),
        "cos_G": _ratio(float(np.vdot(G, gF)), np.sqrt(froG2) * out["grad_norm_F"]),
        "eta_ls": _ratio(pair, curv_O), "O_op": float(np.sqrt(opO2)), "O_fro2": froO2,
    })
    if eta is not None:
        eta = float(eta)
        Lb = prob.L_op if L is None else float(L)
        out.update({
            "eta": eta,
            "decrease": -prob.F_diff(W - eta * O, W, G0=gF),
            "decrease_model": eta * pair - 0.5 * eta ** 2 * curv_O,
            "decrease_bound": eta * pair - 0.5 * eta ** 2 * Lb * opO2,
        })
    return out
