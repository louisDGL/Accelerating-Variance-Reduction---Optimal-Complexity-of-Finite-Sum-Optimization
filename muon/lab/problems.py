"""Matrix-parameter finite sums with exact constants, for Muon-type methods.

Problem: W in R^{K x d} (K outputs or classes, d features), samples x_i in R^d (rows of X, n x d),
    F(W) = (1/n) sum_i g_i(W) + (lam/2) ||W||_F^2,   g_i(W) = l(W x_i; y_i),
and, as in the book (and vrlab), f_i = g_i + (lam/2)||.||_F^2 so that F = (1/n) sum_i f_i.
Losses (z = W x_i in R^K):
  softmax cross-entropy  l(z; y) = logsumexp(z) - z_y,  y in {0..K-1};  grad l = p - e_y, p = softmax(z);
                         hess l = diag(p) - p p^T  <=  I/2   (c_l = 1/2)
  multi-output LS        l(z; y) = ||z - y||^2 / 2,  y in R^K;           grad l = z - y;   hess l = I_K  (c_l = 1)
The loss residual is r_i = grad l(W x_i; y_i) in R^K, so g_i'(W) = r_i x_i^T (rank one) and
grad F(W) = (1/n) R^T X + lam W with R the n x K matrix of the r_i.
Vectorization: row-major, vec(W)[k d + j] = W[k, j]; then hess g_i = hess l(z_i) kron x_i x_i^T (K x K outer factor).
Inner product <A, B> = tr(A^T B); norms ||.||_F, ||.||_op (largest singular value), ||.||_* (nuclear, dual of ||.||_op).

Why c_l = 1/2 for softmax: u^T (diag(p) - p p^T) u = Var_{k~p}(u_k) <= (max_k u_k - min_k u_k)^2 / 4 (Popoviciu)
<= (u_a^2 + u_b^2) / 2 <= ||u||^2 / 2, with equality in the limit p = (1/2, 1/2, 0, ...), u = (1, -1, 0, ...).

Constants (Sigma = X^T X / n with eigenvalues lambda_1 >= ... >= lambda_d; r = min(K, d)):
  L_i    = c_l ||x_i||^2 + lam            Euclidean smoothness of f_i (hess f_i <= c_l ||x_i||^2 I + lam I)
  R2     = max_i L_i,  mu = lam,  kappa = R2 / mu   (the generators set lam so that kappa is the requested value)
  L_F    = c_l lambda_1 + lam             Euclidean smoothness of F (upper bound; exact for LS)
  L_op_i = c_l ||x_i||^2 + lam r          smoothness of f_i with respect to ||.||_op
  R2_op  = max_i L_op_i
  L_op   = c_l (lambda_1 + ... + lambda_r) + lam r   smoothness of F with respect to ||.||_op (upper bound; exact for LS)
  mu_op  = lam                            strong convexity of F with respect to ||.||_op (global)
A C^2 convex function is L-smooth with respect to a norm ||.|| iff D : hess f(W) : D <= L ||D||^2 for all W, D; for
||.||_op this is the guarantee f(W - eta O) <= f(W) - eta <grad f, O> + (L/2) eta^2 ||O||_op^2 used by Muon's steps.
Proofs, for ||D||_op <= 1:
  * loss part of f_i: hess l(z_i)[D x_i, D x_i] <= c_l ||D x_i||^2 <= c_l ||x_i||^2, since ||D x|| <= ||D||_op ||x||;
  * loss part of F: (1/n) sum_i c_l ||D x_i||^2 = c_l tr(D Sigma D^T) = c_l tr(Sigma D^T D) <= c_l (lambda_1 + ... +
    lambda_r), because 0 <= D^T D <= I has rank <= r (Ky Fan's maximum principle / von Neumann's trace inequality);
  * regularizer: lam ||D||_F^2 = lam sum_k s_k(D)^2 <= lam rank(D) ||D||_op^2, i.e. its spectral smoothness constant is
    lam sup ||D||_F^2 / ||D||_op^2 = lam r (attained by any partial isometry of rank r);
  * for LS both bounds are attained together by D whose rows (K <= d) or columns (K > d) are the top eigenvectors of
    Sigma, so L_op is exact; mu_op: D : hess F : D >= lam ||D||_F^2 >= lam ||D||_op^2 (for softmax mu = mu_op = lam is
    attained along D = 1 v^T, since (diag(p) - p p^T) 1 = 0; for LS the exact value is lambda_d + lam).
Softmax redundancy: 1^T grad F(W) = lam 1^T W, hence 1^T W_* = 0, and every method of `algorithms` started from W0
with 1^T W0 = 0 stays in that subspace (all per-sample gradients and their polar factors have columns orthogonal to 1),
where the gradients have rank <= K - 1.

Accuracy: `F_diff` / `Fgap` evaluate F(W1) - F(W0) through an exact Bregman identity whose rounding error is
O(eps ||W1 - W0||) instead of O(eps |F|), so F - F_* stays meaningful down to ~1e-30 (the naive difference of two
values of F has a floor of about 1e-16 |F_*|).
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

try:
    from vrlab.problems import covariance_spectrum
except ImportError:  # the project root (parent of muon/) is not on sys.path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from vrlab.problems import covariance_spectrum


# ----------------------------------------------------------------------------------------------
# numerics helpers
# ----------------------------------------------------------------------------------------------
def softmax_rows(Z):
    """Row-wise softmax of an (m x K) array (stable)."""
    E = np.exp(Z - Z.max(axis=1, keepdims=True))
    E /= E.sum(axis=1, keepdims=True)
    return E


def logsumexp_rows(Z):
    """Row-wise log(sum_k exp(z_k)) of an (m x K) array (stable)."""
    m = Z.max(axis=1)
    return m + np.log(np.exp(Z - m[:, None]).sum(axis=1))


_EM1MX_COEF = [1.0 / math.factorial(k) for k in range(2, 18)]       # 1/k!, k = 2..17
_EM1MX_SMALL = 0.25


def _em1mx_terms(m):
    """Smallest k_max in [2, 17] with 2.5 m^(k_max - 1) / (k_max + 1)! < 1e-17: for |x| <= m <= 1/4 the series
    truncated after x^k_max / k_max! then has relative error below 1e-17 (e^x - 1 - x >= 0.9 x^2 / 2 there)."""
    for k in range(2, 18):
        if 2.5 * m ** (k - 1) / math.factorial(k + 1) < 1e-17:
            return k
    return 17


def em1mx(x):
    """exp(x) - 1 - x without cancellation: Taylor series sum_{k=2}^{k_max} x^k / k! for |x| < 1/4 (k_max adapted to
    max |x|, relative truncation error below 1e-17), expm1(x) - x otherwise (relative error below 16 eps there)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.zeros_like(x)
    ax = np.abs(x)
    m = float(ax.max())
    if m < _EM1MX_SMALL:
        coef = _EM1MX_COEF[: _em1mx_terms(m) - 1]
        acc = np.full_like(x, coef[-1])
        for c in coef[-2::-1]:
            acc *= x
            acc += c
        acc *= x
        acc *= x
        return acc
    with np.errstate(over="ignore"):
        out = np.expm1(x) - x
    small = ax < _EM1MX_SMALL
    if np.any(small):
        out[small] = em1mx(x[small])
    return out


# ----------------------------------------------------------------------------------------------
# base class
# ----------------------------------------------------------------------------------------------
class MatrixFiniteSum:
    """Interface shared by the matrix problems (see the module docstring for the constants).

    Attributes: n, d, K, lam, X, y, r = min(K, d), sqnorms (||x_i||^2), Sigma_hat, cov_eigs (ascending), c_loss,
    L_i, R2, mu, L_F, L_op_i, R2_op, L_op, mu_op; after `set_optimum`: W_star, F_star, Z_star (= X W_*^T),
    R_star (loss residuals at W_*), grad_star, grad_star_norm, opt_info.
    """

    loss = "abstract"
    c_loss = None

    def __init__(self, X, y, K, lam, name="matrix-finite-sum"):
        X = np.ascontiguousarray(X, dtype=np.float64)
        self.X = X
        self.n, self.d = X.shape
        self.K = int(K)
        self.lam = float(lam)
        self.y = y
        self.name = name
        self.r = min(self.K, self.d)
        c = float(self.c_loss)
        self.sqnorms = np.einsum("ij,ij->i", X, X)
        self.L_i = c * self.sqnorms + self.lam
        self.R2 = float(self.L_i.max())
        self.mu = self.lam
        self.Sigma_hat = X.T @ X / self.n
        self.cov_eigs = np.linalg.eigvalsh(self.Sigma_hat)
        self.L_F = float(c * self.cov_eigs[-1] + self.lam)
        self.L_op_i = c * self.sqnorms + self.lam * self.r
        self.R2_op = float(self.L_op_i.max())
        self.L_op = float(c * np.sort(self.cov_eigs)[::-1][: self.r].sum() + self.lam * self.r)
        self.mu_op = self.lam
        self.W_star = None
        self.F_star = None
        self.Z_star = None
        self.P_star = None
        self.R_star = None
        self.grad_star = None
        self.grad_star_norm = None
        self.opt_info = None
        self.meta = {}

    # --- loss-specific pieces (row-wise), implemented by the subclasses ----------------------------
    def _loss_rows(self, Z, idx=None):
        """Per-sample losses l(z_i; y_i) for the rows of Z (labels y[idx], or all labels if idx is None)."""
        raise NotImplementedError

    def _residual_rows(self, Z, idx=None):
        """Per-sample residuals r_i = grad l(z_i; y_i) for the rows of Z."""
        raise NotImplementedError

    def residual_i(self, z, i):
        """Residual r_i = grad l(z; y_i) for one sample (z = W x_i in R^K); the fast path of the stochastic loops."""
        raise NotImplementedError

    def residual_diff_i(self, z, delta, i):
        """r_i(z) - r_i(z - delta) for one sample, accurate when delta is small (no cancellation)."""
        raise NotImplementedError

    def _hess_rows(self, Z, U):
        """Rows hess l(z_i) u_i for the rows z_i of Z and u_i of U."""
        raise NotImplementedError

    def _bregman_rows(self, Delta, Z0=None, P0=None):
        """Per-sample Bregman divergences D_l(z0_i + Delta_i, z0_i) = l(z0 + Delta) - l(z0) - <grad l(z0), Delta>
        (labels cancel), computed without cancellation."""
        raise NotImplementedError

    def hessian(self, W):
        """Dense Hessian of F at W, (K d) x (K d) in the row-major vec convention (small sizes only)."""
        raise NotImplementedError

    # --- objective and derivatives ----------------------------------------------------------------
    @property
    def kappa(self):
        return self.R2 / self.mu

    @property
    def kappa_F(self):
        return self.L_F / self.mu

    @property
    def kappa_op(self):
        return self.L_op / self.mu_op

    def outputs(self, W):
        """Z = X W^T (n x K): row i is z_i = W x_i."""
        return self.X @ np.asarray(W, dtype=np.float64).T

    def remove_gauge(self, M):
        """Projection that removes the directions along which every loss is invariant (none here; see SoftmaxProblem)."""
        return np.asarray(M, dtype=np.float64)

    def F(self, W):
        W = np.asarray(W, dtype=np.float64)
        return float(self._loss_rows(self.outputs(W)).mean() + 0.5 * self.lam * np.vdot(W, W))

    def f_i(self, W, i):
        """f_i(W) = l(W x_i; y_i) + (lam/2) ||W||_F^2."""
        W = np.asarray(W, dtype=np.float64)
        z = np.dot(W, self.X[i])[None, :]
        return float(self._loss_rows(z, np.array([i]))[0] + 0.5 * self.lam * np.vdot(W, W))

    def F_minibatch(self, W, idx):
        """(1/|idx|) sum_{i in idx} f_i(W) (indices counted with multiplicity)."""
        W = np.asarray(W, dtype=np.float64)
        idx = np.asarray(idx)
        Z = self.X[idx] @ W.T
        return float(self._loss_rows(Z, idx).mean() + 0.5 * self.lam * np.vdot(W, W))

    def loss_residuals(self, W):
        """n x K matrix R of the loss residuals r_i = grad l(W x_i; y_i)."""
        return self._residual_rows(self.outputs(W))

    def grad(self, W):
        """grad F(W) = (1/n) R^T X + lam W."""
        W = np.asarray(W, dtype=np.float64)
        R = self._residual_rows(self.outputs(W))
        return R.T @ self.X / self.n + self.lam * W

    def grad_i(self, W, i):
        """f_i'(W) = r_i x_i^T + lam W."""
        W = np.asarray(W, dtype=np.float64)
        xi = self.X[i]
        return np.outer(self.residual_i(np.dot(W, xi), i), xi) + self.lam * W

    def grad_batch(self, W, idx):
        """(1/|idx|) sum_{i in idx} f_i'(W), lam W included (indices counted with multiplicity)."""
        W = np.asarray(W, dtype=np.float64)
        idx = np.asarray(idx)
        Xb = self.X[idx]
        Rb = self._residual_rows(Xb @ W.T, idx)
        return Rb.T @ Xb / len(idx) + self.lam * W

    def hvp(self, W, D):
        """Exact Hessian-vector product hess F(W)[D] = (1/n) sum_i (hess l(z_i) D x_i) x_i^T + lam D."""
        W = np.asarray(W, dtype=np.float64)
        D = np.asarray(D, dtype=np.float64)
        V = self._hess_rows(self.outputs(W), self.X @ D.T)
        return V.T @ self.X / self.n + self.lam * D

    # --- accurate differences of F ----------------------------------------------------------------
    def F_diff(self, W1, W0, Z0=None, G0=None, P0=None):
        """F(W1) - F(W0) through the exact identity
            F(W1) - F(W0) = (1/n) sum_i D_l(z0_i + Delta_i, z0_i) + <grad F(W0), Delta W> + (lam/2) ||Delta W||_F^2,
        Delta W = W1 - W0, Delta_i = Delta W x_i (computed as X Delta W^T, never as a difference of outputs), D_l >= 0
        the Bregman divergence of the loss. Optional caches at W0: Z0 = X W0^T, G0 = grad F(W0), P0 = softmax(Z0)."""
        W0 = np.asarray(W0, dtype=np.float64)
        Dlt = np.asarray(W1, dtype=np.float64) - W0
        Delta = self.X @ Dlt.T
        if G0 is None:
            G0 = self.grad(W0)
        if Z0 is None and P0 is None and self.loss == "softmax":
            Z0 = self.outputs(W0)
        breg = self._bregman_rows(Delta, Z0, P0)
        return float(breg.mean() + np.vdot(G0, Dlt) + 0.5 * self.lam * np.vdot(Dlt, Dlt))

    def Fgap(self, W):
        """F(W) - F(W_*) via `F_diff` (accurate down to O(eps ||W - W_*||_F))."""
        if self.W_star is None:
            raise ValueError("set the optimum first (lab.optimum.solve_optimum(prob))")
        return self.F_diff(W, self.W_star, Z0=self.Z_star, G0=self.grad_star, P0=self.P_star)

    # --- optimum and summaries --------------------------------------------------------------------
    def set_optimum(self, W_star, info=None):
        """Store W_* and the derived quantities used by the traces and the instrumentation."""
        W = np.array(W_star, dtype=np.float64, copy=True)
        if W.shape != (self.K, self.d):
            raise ValueError(f"W_star must have shape {(self.K, self.d)}")
        self.W_star = W
        self.Z_star = self.outputs(W)
        self.P_star = softmax_rows(self.Z_star) if self.loss == "softmax" else None
        self.R_star = self._residual_rows(self.Z_star)
        self.grad_star = self.R_star.T @ self.X / self.n + self.lam * W
        self.grad_star_norm = float(np.linalg.norm(self.grad_star))
        self.F_star = self.F(W)
        self.opt_info = dict(info) if info else {}
        return self

    def summary(self):
        s = {"name": self.name, "loss": self.loss, "n": self.n, "d": self.d, "K": self.K, "lam": self.lam,
             "R2": self.R2, "mu": self.mu, "kappa": self.kappa, "L_F": self.L_F, "kappa_F": self.kappa_F,
             "R2_op": self.R2_op, "L_op": self.L_op, "mu_op": self.mu_op, "kappa_op": self.kappa_op,
             "cov_cond": float(self.cov_eigs[-1] / max(self.cov_eigs[0], 1e-300)),
             "max_sqnorm": float(self.sqnorms.max()), "mean_sqnorm": float(self.sqnorms.mean())}
        if self.W_star is not None:
            sv = np.linalg.svd(self.W_star, compute_uv=False)
            s.update({"F_star": self.F_star, "grad_star_norm": self.grad_star_norm,
                      "W_star_fro": float(np.sqrt((sv ** 2).sum())), "W_star_op": float(sv[0]),
                      "W_star_srank": float((sv ** 2).sum() / sv[0] ** 2),
                      "W_star_sv_ratio": float(sv[-1] / sv[0])})
            s.update({k: v for k, v in (self.opt_info or {}).items() if np.isscalar(v)})
        return s


# ----------------------------------------------------------------------------------------------
# softmax cross-entropy
# ----------------------------------------------------------------------------------------------
class SoftmaxProblem(MatrixFiniteSum):
    """l(z; y) = logsumexp(z) - z_y, y in {0..K-1} (c_l = 1/2)."""

    loss = "softmax"
    c_loss = 0.5

    def __init__(self, X, y, K, lam, name="softmax"):
        y = np.ascontiguousarray(np.asarray(y), dtype=np.int64)
        if y.ndim != 1 or y.shape[0] != np.shape(X)[0] or y.min() < 0 or y.max() >= K:
            raise ValueError("labels must be integers in {0..K-1}, one per row of X")
        super().__init__(X, y, K, lam, name)

    def _labels(self, idx):
        return self.y if idx is None else self.y[np.asarray(idx)]

    def remove_gauge(self, M):
        """M - (1/K) 1 1^T M (centering over the K rows). Adding 1 v^T to W leaves every softmax loss unchanged, so the
        loss part of every gradient satisfies 1^T g_i'(W) = 0, and the whole gradient too when 1^T W = 0. Near W_*,
        the computed gradient violates this by an ABSOLUTE rounding error (about 1e-17, from (1/n) R^T X), which is
        large relative to a small gradient: s_K / s_1 reaches 1e-10..1e-4, far above the rtol of polar_svd, and the
        exact polar factor turns it into a spurious unit singular direction 1 v^T / sqrt(K). After this projection
        s_K / s_1 is at the rounding level of M itself. Use orth=lambda M: polar_svd(prob.remove_gauge(M)) (valid
        when 1^T W_0 = 0, e.g. W_0 = 0)."""
        M = np.asarray(M, dtype=np.float64)
        return M - M.mean(axis=0, keepdims=True)

    def _loss_rows(self, Z, idx=None):
        y = self._labels(idx)
        return logsumexp_rows(Z) - Z[np.arange(Z.shape[0]), y]

    def _residual_rows(self, Z, idx=None):
        y = self._labels(idx)
        P = softmax_rows(Z)
        P[np.arange(Z.shape[0]), y] -= 1.0
        return P

    def residual_i(self, z, i):
        e = z - z.max()
        np.exp(e, out=e)
        e /= e.sum()
        e[self.y[i]] -= 1.0
        return e

    def residual_diff_i(self, z, delta, i):
        """softmax(z) - softmax(z - delta) (labels cancel). With p = softmax(z) and A = sum_l p_l exp(-delta_l):
        p_k(z - delta) = p_k exp(-delta_k) / A, hence
            p_k(z) - p_k(z - delta) = p_k exp(-delta_k) sum_l p_l expm1(delta_k - delta_l) / A,
        which is accurate to O(eps |delta|) instead of O(eps p_k)."""
        p = np.exp(z - z.max())
        p /= p.sum()
        e = np.exp(-delta)
        A = float(np.dot(p, e))
        return p * e * np.dot(np.expm1(delta[:, None] - delta[None, :]), p) / A

    def _hess_rows(self, Z, U):
        P = softmax_rows(Z)
        PU = P * U
        return PU - P * PU.sum(axis=1, keepdims=True)

    def _bregman_rows(self, Delta, Z0=None, P0=None):
        """D_lse(z0 + Delta, z0) = log(sum_k p0_k exp(Delta_k)) - <p0, Delta>, p0 = softmax(z0). With
        T = Delta - <p0, Delta> (so that sum_k p0_k T_k = 0 up to rounding):
            D = log1p( sum_k p0_k (exp(T_k) - 1 - T_k) ),
        a sum of nonnegative terms (no cancellation; the rounding of <p0, T> enters only at second order). Rows where
        exp overflows fall back to the direct formula."""
        if P0 is None:
            P0 = softmax_rows(Z0)
        T = Delta - (P0 * Delta).sum(axis=1, keepdims=True)
        S = (P0 * em1mx(T)).sum(axis=1)
        out = np.log1p(S)
        bad = ~np.isfinite(out)
        if np.any(bad):
            if Z0 is None:
                raise FloatingPointError("overflow in the Bregman divergence; pass Z0 for the direct formula")
            Zb, Db, Pb = Z0[bad], Delta[bad], P0[bad]
            out[bad] = logsumexp_rows(Zb + Db) - logsumexp_rows(Zb) - (Pb * Db).sum(axis=1)
        return out

    def hessian(self, W):
        n, d, K, X = self.n, self.d, self.K, self.X
        P = softmax_rows(self.outputs(W))
        H = np.empty((K * d, K * d))
        for k in range(K):
            for l in range(k, K):
                w = P[:, k] * (1.0 - P[:, k]) if k == l else -P[:, k] * P[:, l]
                blk = (X.T * w) @ X / n                  # (1/n) sum_i S_i[k, l] x_i x_i^T, symmetric
                H[k * d:(k + 1) * d, l * d:(l + 1) * d] = blk
                if l != k:
                    H[l * d:(l + 1) * d, k * d:(k + 1) * d] = blk
        H[np.diag_indices_from(H)] += self.lam
        return H


# ----------------------------------------------------------------------------------------------
# multi-output least squares
# ----------------------------------------------------------------------------------------------
class MatrixLSProblem(MatrixFiniteSum):
    """l(z; y) = ||z - y||^2 / 2, y in R^K (c_l = 1); F is quadratic with Hessian I_K kron Sigma + lam I."""

    loss = "ls"
    c_loss = 1.0

    def __init__(self, X, Y, lam, name="matrix-ls"):
        Y = np.ascontiguousarray(Y, dtype=np.float64)
        if Y.ndim != 2 or Y.shape[0] != np.shape(X)[0]:
            raise ValueError("Y must be an n x K array")
        super().__init__(X, Y, Y.shape[1], lam, name)
        self.Y = self.y

    def _targets(self, idx):
        return self.Y if idx is None else self.Y[np.asarray(idx)]

    def _loss_rows(self, Z, idx=None):
        Rz = Z - self._targets(idx)
        return 0.5 * np.einsum("ij,ij->i", Rz, Rz)

    def _residual_rows(self, Z, idx=None):
        return Z - self._targets(idx)

    def residual_i(self, z, i):
        return z - self.Y[i]

    def residual_diff_i(self, z, delta, i):
        return np.array(delta, dtype=np.float64, copy=True)

    def _hess_rows(self, Z, U):
        return U

    def _bregman_rows(self, Delta, Z0=None, P0=None):
        return 0.5 * np.einsum("ij,ij->i", Delta, Delta)

    def hessian(self, W=None):
        H = np.kron(np.eye(self.K), self.Sigma_hat)
        H[np.diag_indices_from(H)] += self.lam
        return H


# ----------------------------------------------------------------------------------------------
# generators
# ----------------------------------------------------------------------------------------------
def make_features(n, d, alpha=0.0, rng=None, rotate=True, R=1.0, spectrum="power"):
    """Features exactly as vrlab.problems.make_logistic (same random draws in the same order): x_i = Sigma^{1/2} g_i,
    g_i ~ N(0, I_d), spectrum `covariance_spectrum(d, alpha, spectrum)` (j^-alpha for 'power'), Haar rotation (QR of a
    Gaussian matrix) if `rotate`, then a global rescaling so that max_i ||x_i|| = R."""
    ev = covariance_spectrum(d, alpha, spectrum)
    G = rng.standard_normal((n, d))
    X = G * np.sqrt(ev)[None, :]
    if rotate:
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        X = X @ Q.T
    X *= R / np.sqrt(np.einsum("ij,ij->i", X, X).max())
    return X


def make_softmax(n, d, K, kappa, alpha=0.0, seed=0, rotate=True, snr=2.0, R=1.0, spectrum="power", name=None):
    """Synthetic l2-regularized softmax regression with controlled constants.

    * features as vrlab make_logistic (`make_features`; for the same (n, d, alpha, seed, rotate, R, spectrum) X is
      bitwise the X of vrlab.problems.make_logistic);
    * planted W0 = c G0, G0 ~ N(0, 1)^{K x d}, c such that the logits X W0^T have empirical standard deviation `snr`;
      labels y_i ~ softmax(W0 x_i), sampled by inverse CDF (informative, not separable: the labels are random given x);
    * lam = (max_i ||x_i||^2 / 2) / (kappa - 1), so that kappa = R2 / mu exactly (R2 = max_i ||x_i||^2/2 + lam)."""
    if not kappa > 1:
        raise ValueError("kappa must be > 1")
    rng = np.random.default_rng(seed)
    X = make_features(n, d, alpha, rng, rotate, R, spectrum)
    W0 = rng.standard_normal((K, d))
    logits = X @ W0.T
    c = snr / logits.std()
    W0 *= c
    P = softmax_rows(X @ W0.T)
    u = rng.random(n)
    y = np.minimum((np.cumsum(P, axis=1) < u[:, None]).sum(axis=1), K - 1)
    lam = 0.5 * float(np.einsum("ij,ij->i", X, X).max()) / (kappa - 1.0)
    prob = SoftmaxProblem(X, y, K, lam, name=name or
                          f"softmax(n={n},d={d},K={K},kappa={kappa:g},alpha={alpha:g},seed={seed})")
    prob.meta = dict(n=n, d=d, K=K, kappa=kappa, alpha=alpha, seed=seed, rotate=rotate, snr=snr, R=R,
                     spectrum=spectrum, planted=W0, label_counts=np.bincount(y, minlength=K))
    return prob


def make_matrix_ls(n, d, K, kappa, alpha=0.0, seed=0, rotate=True, noise=0.5, snr=1.0, R=1.0, spectrum="power",
                   name=None, planted="gaussian"):
    """Synthetic l2-regularized multi-output least squares with controlled constants.

    * features as vrlab make_logistic (`make_features`);
    * planted W0 = c G0, G0 ~ N(0, 1)^{K x d}, c such that X W0^T has empirical standard deviation `snr`;
      targets y_i = W0 x_i + noise * eps_i, eps_i ~ N(0, I_K);
    * lam = max_i ||x_i||^2 / (kappa - 1), so that kappa = R2 / mu exactly (R2 = max_i ||x_i||^2 + lam).
    planted='orthogonal' replaces G0 by its polar factor (same random draws): a partial isometry, all of whose
    singular values are equal (an orthogonal matrix when K = d). For K = d, noise = 0 and lam -> 0 the gradient at
    W = 0 is then -c G0 Sigma_hat, whose polar factor is -G0 = polar(W - W_*) whatever the spectrum of Sigma_hat (a
    product of an orthogonal and a symmetric positive definite matrix is its own polar decomposition): the
    orthogonalized direction points at the minimizer, the situation where Muon-type steps do not see the condition
    number of the features. An explicit K x d array is also accepted."""
    if not kappa > 1:
        raise ValueError("kappa must be > 1")
    rng = np.random.default_rng(seed)
    X = make_features(n, d, alpha, rng, rotate, R, spectrum)
    W0 = rng.standard_normal((K, d))
    if isinstance(planted, str):
        if planted == "orthogonal":
            U0, _, Vt0 = np.linalg.svd(W0, full_matrices=False)
            W0 = U0 @ Vt0
        elif planted != "gaussian":
            raise ValueError(f"unknown planted {planted!r}")
    else:
        W0 = np.array(planted, dtype=np.float64, copy=True)
        if W0.shape != (K, d):
            raise ValueError(f"planted must have shape {(K, d)}")
    S = X @ W0.T
    W0 *= snr / S.std()
    Y = X @ W0.T + noise * rng.standard_normal((n, K))
    lam = float(np.einsum("ij,ij->i", X, X).max()) / (kappa - 1.0)
    prob = MatrixLSProblem(X, Y, lam, name=name or
                           f"matrix_ls(n={n},d={d},K={K},kappa={kappa:g},alpha={alpha:g},seed={seed})")
    prob.meta = dict(n=n, d=d, K=K, kappa=kappa, alpha=alpha, seed=seed, rotate=rotate, noise=noise, snr=snr, R=R,
                     spectrum=spectrum, planted=W0)
    return prob
