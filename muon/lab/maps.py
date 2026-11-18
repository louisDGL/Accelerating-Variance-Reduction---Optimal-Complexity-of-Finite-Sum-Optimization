"""Spectral maps of Muon-type steps and their Frechet derivatives.

Notation: M in R^{K x d}, thin SVD M = U diag(s) V^T (s_1 >= ... >= s_r >= 0, r = min(K, d)), <A, B> = tr(A^T B).
A scalar function f on [0, inf) defines the generalized matrix function f(M) = U diag(f(s)) V^T. The maps used here:

  polar(M)          f = 1                          U V^T, the spectral-norm LMO / "msign" (polar.polar_svd)
  sharp(M)          ||M||_* U V^T                  dual map of the spectral norm: the gradient of h(M) = ||M||_*^2 / 2 where
                                                   h is differentiable; W - gamma sharp(G) is the dualized steepest-descent
                                                   (PolarGrad) step. On RANK-ONE matrices sharp is the identity:
                                                   sharp(a b^T) = ||a|| ||b|| (a/||a||)(b/||b||)^T = a b^T.
  soft_polar(M, e)  f(s) = s / sqrt(s^2 + e^2)     = M (M^T M + e^2 I)^{-1/2}: bounded gain 1/e at 0, tends to polar(M) for
                                                   s >> e and to M / e for s << e
  power_map(M, a)   f(s) = s^a, a in [0, 1]        a = 0 polar (on the nonzero singular values), a = 1 identity

Frechet derivative of a generalized matrix function (Noferini 2017, Thm 2.2, wide full-rank case K <= d written here;
the tall case follows by transposition, f(M^T) = f(M)^T). With V_perp an orthonormal basis of the orthogonal complement
of the row space, A = U^T E V (K x K) and B = U^T E V_perp:

    D f(M)[E] = U ( Phi_sym o sym(A) + Phi_skew o skew(A) ) V^T + U diag(f(s) / s) B V_perp^T,
    sym(A) = (A + A^T)/2,  skew(A) = (A - A^T)/2,
    Phi_sym[i, j]  = (f(s_i) - f(s_j)) / (s_i - s_j)    (f'(s_i) when s_i = s_j),
    Phi_skew[i, j] = (f(s_i) + f(s_j)) / (s_i + s_j),

"o" the entrywise product; the last term equals U diag(f(s) / s) U^T E (I - V V^T). For the polar factor (f = 1):

    D polar(M)[E] = U ( (A - A^T) / (s_i + s_j) ) V^T + U diag(1 / s) U^T E (I - V V^T),

so ||D polar(M)|| is of order 1 / s_min(M): the polar map amplifies a perturbation of the smallest singular directions by
the inverse of the smallest singular value. For the sharp map,

    D sharp(M)[E] = <polar(M), E> polar(M) + ||M||_* D polar(M)[E],        <polar(M), E> = tr(A).

Where it exists, D sharp(M) is the Hessian of the convex function h, hence a symmetric positive semidefinite operator.

Rank deficiency. polar and sharp are not differentiable where a singular value vanishes. The derivative routines require
numerical full rank (s_r > rtol s_1) and raise numpy.linalg.LinAlgError otherwise. For softmax problems every gradient
has its columns orthogonal to the all-ones vector (rank <= K - 1): pass `basis=Q`, a K x K' matrix with orthonormal
columns spanning the subspace that contains the column space of M and E (problems.SoftmaxProblem: the orthogonal
complement of 1); all computations are then done on Q^T M (K' x d), i.e. f_Q(M) = Q f(Q^T M), which equals
polar_svd(remove_gauge(M)) for the polar factor.
"""
from __future__ import annotations

import numpy as np

from .polar import svd_thin


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def gauge_basis(prob):
    """Orthonormal basis Q (K x K') of the subspace of R^K that contains the columns of every gradient of `prob`:
    None for least squares (the whole space), the orthogonal complement of the all-ones vector for softmax (Q has
    K - 1 columns; valid for iterates with 1^T W = 0, e.g. started at W_0 = 0)."""
    if getattr(prob, "loss", None) != "softmax":
        return None
    K = prob.K
    # Helmert-type basis: QR of the centering projector's first K - 1 columns
    C = np.eye(K) - np.full((K, K), 1.0 / K)
    Q, _ = np.linalg.qr(C[:, : K - 1])
    return Q


def _reduce(M, basis):
    return M if basis is None else basis.T @ M


def _expand(M, basis):
    return M if basis is None else basis @ M


def _svd_wide(M):
    """Thin SVD of M returned for the wide orientation: (U, s, Vt, transposed) with U square (r x r)."""
    tr = M.shape[0] > M.shape[1]
    U, s, Vt = svd_thin(M.T if tr else M)
    return U, s, Vt, tr


def _check_full_rank(s, rtol, what):
    if s.size == 0 or not s[-1] > rtol * s[0]:
        raise np.linalg.LinAlgError(f"{what}: the matrix is numerically rank deficient "
                                    f"(s_min / s_max = {s[-1] / s[0] if s.size and s[0] > 0 else float('nan'):.2e}); "
                                    "pass `basis` (see maps.gauge_basis) or use a bounded-gain map")


# ----------------------------------------------------------------------------------------------
# maps
# ----------------------------------------------------------------------------------------------
def _nan_like(M):
    return np.full(np.shape(M), np.nan)


def polar(M, basis=None, rtol=1e-15):
    """Polar factor U V^T restricted to the singular values s_k > rtol s_1 (as polar.polar_svd), computed on
    basis^T M when a basis is given. A non-finite M gives an all-NaN matrix (as polar.polar_svd)."""
    M = np.asarray(M, dtype=np.float64)
    if not np.isfinite(M).all():
        return _nan_like(M)
    R = _reduce(M, basis)
    U, s, Vt = svd_thin(R)
    k = int(np.count_nonzero(s > rtol * (s[0] if s.size else 0.0)))
    return _expand(np.dot(U[:, :k], Vt[:k]), basis)


def sharp(M, basis=None, rtol=1e-15, return_parts=False):
    """Dual map ||M||_* polar(M). With return_parts=True returns (sharp, polar, nuclear norm). A non-finite M gives
    all-NaN outputs."""
    M = np.asarray(M, dtype=np.float64)
    if not np.isfinite(M).all():
        return (_nan_like(M), _nan_like(M), float("nan")) if return_parts else _nan_like(M)
    R = _reduce(M, basis)
    U, s, Vt = svd_thin(R)
    k = int(np.count_nonzero(s > rtol * (s[0] if s.size else 0.0)))
    O = _expand(np.dot(U[:, :k], Vt[:k]), basis)
    nuc = float(s.sum())
    return (nuc * O, O, nuc) if return_parts else nuc * O


def soft_polar(M, eps, basis=None):
    """Bounded-gain orthogonalization U diag(s / sqrt(s^2 + eps^2)) V^T = M (M^T M + eps^2 I)^{-1/2} (eps > 0)."""
    if not eps > 0:
        raise ValueError("soft_polar needs eps > 0 (use polar for eps = 0)")
    M = np.asarray(M, dtype=np.float64)
    if not np.isfinite(M).all():
        return _nan_like(M)
    R = _reduce(M, basis)
    U, s, Vt = svd_thin(R)
    return _expand(np.dot(U * (s / np.sqrt(s * s + eps * eps)), Vt), basis)


def power_map(M, a, basis=None, rtol=1e-15):
    """U diag(s^a) V^T on the singular values s_k > rtol s_1 (a = 0: polar factor; a = 1: M)."""
    R = _reduce(np.asarray(M, dtype=np.float64), basis)
    U, s, Vt = svd_thin(R)
    k = int(np.count_nonzero(s > rtol * (s[0] if s.size else 0.0)))
    return _expand(np.dot(U[:, :k] * s[:k] ** a, Vt[:k]), basis)


# ----------------------------------------------------------------------------------------------
# Frechet derivatives
# ----------------------------------------------------------------------------------------------
class SpectralDerivative:
    """Frechet derivative E -> D f(M)[E] of a generalized matrix function at a fixed full-rank M, with the SVD of M
    computed once (apply it to many directions E). `f` and `df` are vectorized scalar functions (f(s), f'(s));
    kind='polar' and kind='sharp' are built in; kind='soft' needs eps.

        D = SpectralDerivative(M, kind='sharp', basis=Q);  D(E)  ->  K x d matrix;   D.value  ->  f(M)

    Also exposes U, s, Vt (wide orientation of the reduced matrix), nuc = ||M||_* and polar = U V^T over all the
    r singular directions: the polar factor of M when M has full rank (always the case for kind 'polar' / 'sharp',
    which refuse rank-deficient matrices); for 'soft' / 'custom' at a rank-deficient M it differs from maps.polar,
    which keeps the nonzero singular values only.
    """

    def __init__(self, M, kind="polar", eps=None, f=None, df=None, basis=None, rtol=1e-12):
        M = np.asarray(M, dtype=np.float64)
        self.basis = basis
        R = _reduce(M, basis)
        self.U, self.s, self.Vt, self.tr = _svd_wide(R)
        s = self.s
        self.kind = kind
        if kind in ("polar", "sharp"):
            _check_full_rank(s, rtol, f"D {kind}")
            fs, dfs = np.ones_like(s), np.zeros_like(s)
        elif kind == "soft":
            if eps is None or not eps > 0:
                raise ValueError("kind='soft' needs eps > 0")
            den = np.sqrt(s * s + eps * eps)
            fs, dfs = s / den, eps * eps / den ** 3
        elif kind == "custom":
            fs, dfs = np.asarray(f(s), dtype=np.float64), np.asarray(df(s), dtype=np.float64)
        else:
            raise ValueError(f"unknown kind {kind!r}")
        self.fs = fs
        si, sj = s[:, None], s[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            dsum = si + sj
            self.phi_skew = np.where(dsum > 0, (fs[:, None] + fs[None, :]) / np.where(dsum > 0, dsum, 1.0),
                                     0.5 * (dfs[:, None] + dfs[None, :]))       # limit when both values vanish
            if kind == "soft":
                # cancellation-free divided difference of f(s) = s / A(s), A = sqrt(s^2 + eps^2):
                # (f(a) - f(b)) / (a - b) = eps^2 (a + b) / (A B (a B + b A)) = eps^2 / (A B avg), with
                # avg = (a B + b A) / (a + b) a weighted mean of A and B (= A when a = b, giving f'(a) = eps^2 / A^3)
                Ai, Aj = den[:, None], den[None, :]
                w = np.where(dsum > 0, si / np.where(dsum > 0, dsum, 1.0), 0.5)
                self.phi_sym = eps * eps / (Ai * Aj * (w * Aj + (1.0 - w) * Ai))
                self.f_over_s = 1.0 / den
            else:
                # a subtraction f(s_i) - f(s_j) loses about 1e-16 / gap: below a relative gap of 1e-5 the mean of the
                # derivatives (error of order gap^2) is more accurate. Exact for polar and sharp (f constant).
                diff = si - sj
                close = np.abs(diff) <= 1e-5 * np.maximum(si, sj)
                self.phi_sym = np.where(close, 0.5 * (dfs[:, None] + dfs[None, :]),
                                        (fs[:, None] - fs[None, :]) / np.where(close, 1.0, diff))
                self.f_over_s = np.where(s > 0, fs / np.where(s > 0, s, 1.0), dfs)
        self.nuc = float(s.sum())
        self.square = self.Vt.shape[0] == self.Vt.shape[1]           # no complement of the row space
        P = np.dot(self.U, self.Vt)                                  # polar factor of the (wide) reduced matrix
        self._P_wide = P
        self.polar = _expand(P.T if self.tr else P, basis)
        Fv = np.dot(self.U * fs, self.Vt)
        if kind == "sharp":
            self.value = self.nuc * self.polar
        else:
            self.value = _expand(Fv.T if self.tr else Fv, basis)

    def __call__(self, E):
        E = _reduce(np.asarray(E, dtype=np.float64), self.basis)
        if self.tr:
            E = E.T
        U, Vt = self.U, self.Vt
        UtE = np.dot(U.T, E)                                         # r x d
        A = np.dot(UtE, Vt.T)                                        # r x r
        core = self.phi_sym * (0.5 * (A + A.T)) + self.phi_skew * (0.5 * (A - A.T))
        out = np.dot(np.dot(U, core), Vt)
        if not self.square:                                          # for a square factor the term is exactly zero
            comp = UtE - np.dot(A, Vt)                               # U^T E (I - V V^T)   (and its rounding residue
            out += np.dot(U * self.f_over_s, comp)                   # would be amplified by 1 / s_min)
        if self.kind == "sharp":
            out *= self.nuc
            out += float(np.trace(A)) * self._P_wide
        if self.tr:
            out = out.T
        return _expand(out, self.basis)


def polar_frechet(M, E, basis=None, rtol=1e-12):
    """D polar(M)[E] for a numerically full-rank M (after reduction by `basis`)."""
    return SpectralDerivative(M, "polar", basis=basis, rtol=rtol)(E)


def sharp_frechet(M, E, basis=None, rtol=1e-12):
    """D sharp(M)[E] = <polar(M), E> polar(M) + ||M||_* D polar(M)[E]."""
    return SpectralDerivative(M, "sharp", basis=basis, rtol=rtol)(E)


def soft_polar_frechet(M, E, eps, basis=None):
    """D soft_polar(M, eps)[E] (defined everywhere for eps > 0)."""
    return SpectralDerivative(M, "soft", eps=eps, basis=basis)(E)
