"""Orthogonalization maps of Muon-type methods, for M in R^{K x d}.

Notation: M = U diag(s) V^T is the thin SVD (s_1 >= s_2 >= ... >= 0), r = min(K, d).
  polar factor   polar(M) = U_+ V_+^T, restricted to the singular values s_k > rtol * s_1 (rank-deficient safe);
                 it is the maximiser of <M, O> over ||O||_op <= 1 (the "dualized" gradient of Bernstein & Newhouse),
                 and <M, polar(M)> = ||M||_* (nuclear norm, dual of the spectral norm ||.||_op).
  ns5            Keller Jordan's quintic Newton-Schulz iteration of the public Muon code; it acts on each normalized
                 singular value x = s_k / (||M||_F + eps) through the odd polynomial p(x) = a x + b x^3 + c x^5,
                 (a, b, c) = (3.4445, -4.7750, 2.0315). These coefficients maximise the slope at 0 and do NOT converge
                 to 1: p(1) = 0.701, and after 5 steps every x >= 0.01 lands in about [0.68, 1.20] (see `ns_scalar_map`
                 and the tests). It therefore returns U diag(p^5(x)) V^T, not U V^T.
                 The normalization constant eps (default 1e-7, the value of the public code; `orth_kw={'eps': ...}`
                 in the loops) makes the map depend on the SCALE of M: ns5(c M) = ns5(M) only while ||M||_F >> eps.
                 Every x carries the factor ||M||_F / (||M||_F + eps): when ||M||_F is no longer large compared with
                 eps the output shrinks with M (to first order it is 3.4445^5 M / eps once ||M||_F << eps / 3.4445^5;
                 tests/test_lab.py, test_ns5_scale_dependence). Two consequences for fixed-length steps
                 W <- W - lr_t ns5(omega_t) fed by a variance-reduced estimator, whose omega_t vanishes at the
                 minimizer: (i) once the estimator has come down to the size of eps the step no longer has the length
                 lr_t: it tends to the linear map (3.4445^5 lr_t / eps) omega_t, whose gain decays with the schedule,
                 and the run freezes. Under a geometric cooldown this sets a floor near 1e-18 relative squared
                 distance on the benchmark instances of the study (experiment M9, part C: 8e-19 to 5e-18), where the
                 exact polar factor goes on to the rounding level; the level depends on the instance (8e-27 to 2e-25
                 against 1e-32 on the
                 small well-conditioned instance of tests/test_vrmuon.py, test_ns5_floor), and a smaller eps (1e-30)
                 removes it; (ii) the exact scale covariance of the iteration on least squares (fact F14 of
                 notes_theory.md) holds for a scale-invariant orthogonalization only, not with the default eps.
                 The default is kept: it is the published routine.
  ns_cubic       classical cubic Newton-Schulz X <- 1.5 X - 0.5 X X^T X after Frobenius scaling (all x in (0, 1]);
                 it converges to polar(M) on the singular values that are not zero, quadratically once x is near 1,
                 but only by a factor <= 1.5 per step while x is small, so values s_k / ||M||_F ~ 1e-16 (rounding noise
                 of an exactly rank-deficient M) reach O(1) after about 90 steps.
CONTROL MAPS (`CONTROL_MAPS`; not orthogonalizations). Two other maps of fixed length are available under a name, so
that an experiment can replace the polar factor by them in the same loop and keep its tasks JSON-serializable:
  frob           M / ||M||_F (`frob_normalize`): the direction of normalized gradient descent, the LMO of the Frobenius
                 ball. The singular vectors AND the relative singular values of M are kept (nothing is equalized);
                 ||O||_F = 1, ||O||_op <= 1 with equality iff M has rank one, <M, O> = ||M||_F.
  sign           coordinatewise sign (`sign_map`): the LMO of the entrywise sup-norm ball (signSGD), <M, O> = sum|M_kj|,
                 ||O||_F = (number of nonzero entries)^(1/2), i.e. (K d)^(1/2) generically against r^(1/2) for the polar
                 factor: learning rates are not comparable across maps and must be tuned per map. The columns of
                 sign(M) do not sum to zero when those of M do: for softmax the output must be projected back on
                 the gauge subspace (center=True: O - (1/K) 1 1^T O, which is SoftmaxProblem.remove_gauge); the
                 projected map is no longer an LMO and its entries are not +-1.
                 vrmuon.VRMuon and the gauge wrapper of experiments/common.py set center=True by themselves when the
                 gauge is active; the other loops (algorithms.py, mvr.py) apply the map as given: pass
                 orth_kw={'center': True} there for softmax.
They are controls for the question "is the effect specific to the orthogonalization?", not Muon variants.

bf16 emulation: every intermediate result is rounded to bfloat16 (8 significant bits) by rounding the float64 value to
float32 and then the float32 mantissa to 8 bits with round-to-nearest-even (bit masking). Products are formed in float64
and rounded once, which mimics bf16 tensor-core matmuls with fp32 accumulation. The double rounding
float64 -> float32 -> bf16 can differ from a direct rounding in rare tie cases.

The SVD is LAPACK dgesdd called through scipy on the F-contiguous view M.T of a C-contiguous M (no copy and about half
the overhead of np.linalg.svd on small matrices); np.linalg.svd is the fallback.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg.lapack import dgesdd

NS5_COEFFS = (3.4445, -4.7750, 2.0315)
CUBIC_COEFFS = (1.5, -0.5, 0.0)

_BF16_MASK = np.uint32(0xFFFF0000)
_BF16_HALF = np.uint32(0x7FFF)
_ONE32 = np.uint32(1)


# ----------------------------------------------------------------------------------------------
# SVD helpers
# ----------------------------------------------------------------------------------------------
def svd_thin(M):
    """Thin SVD M = U diag(s) Vt, s in decreasing order (U: K x r, s: r, Vt: r x d, r = min(K, d))."""
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2:
        raise ValueError("svd_thin expects a 2-d array")
    if M.size and np.isfinite(M).all():
        if M.flags.c_contiguous:
            # M.T is F-contiguous: M^T = u diag(s) vt, hence M = vt^T diag(s) u^T
            u, s, vt, info = dgesdd(M.T, compute_uv=1, full_matrices=0)
            if info == 0:
                return vt.T, s, u.T
        else:
            u, s, vt, info = dgesdd(np.asfortranarray(M), compute_uv=1, full_matrices=0)
            if info == 0:
                return u, s, vt
    return np.linalg.svd(M, full_matrices=False)


def singular_values(M):
    """Singular values of M in decreasing order."""
    M = np.asarray(M, dtype=np.float64)
    if M.size and np.isfinite(M).all():
        if M.flags.c_contiguous:
            _, s, _, info = dgesdd(M.T, compute_uv=0, full_matrices=0)
        else:
            _, s, _, info = dgesdd(np.asfortranarray(M), compute_uv=0, full_matrices=0)
        if info == 0:
            return s
    return np.linalg.svd(M, compute_uv=False)


def spectral_norm(M):
    """||M||_op = largest singular value."""
    s = singular_values(M)
    return float(s[0]) if s.size else 0.0


def nuclear_norm(M):
    """||M||_* = sum of the singular values (dual norm of ||.||_op)."""
    return float(singular_values(M).sum())


# ----------------------------------------------------------------------------------------------
# polar factor
# ----------------------------------------------------------------------------------------------
def polar_svd(M, rtol=1e-15, return_sv=False):
    """Exact polar factor U_+ V_+^T of the thin SVD, where only the singular values s_k > rtol * s_1 are kept
    (the others are treated as zero, so an exactly rank-deficient M whose zero singular values are perturbed by
    rounding RELATIVE to M, about eps s_1, does not get spurious unit singular values). This relative threshold cannot
    detect zero singular values perturbed by an absolute rounding error larger than eps s_1, as for a small matrix
    computed as a sum of large terms (softmax gradients near W_*: see SoftmaxProblem.remove_gauge); the exact polar
    factor maps such directions to unit singular values. M = 0 gives O = 0. A non-finite M gives an all-NaN O.
    With return_sv=True, returns (O, s) with s the singular values of M (so that ||M||_* = s.sum() is free)."""
    M = np.asarray(M, dtype=np.float64)
    if not np.isfinite(M).all():
        O = np.full(M.shape, np.nan)
        return (O, np.full(min(M.shape), np.nan)) if return_sv else O
    U, s, Vt = svd_thin(M)
    smax = float(s[0]) if s.size else 0.0
    k = int(np.count_nonzero(s > rtol * smax))
    O = np.dot(U[:, :k], Vt[:k])
    return (O, s) if return_sv else O


# ----------------------------------------------------------------------------------------------
# bfloat16 emulation
# ----------------------------------------------------------------------------------------------
def bf16_round(A):
    """Round to the nearest bfloat16 value (8 significant bits, float32 exponent range), returned as float64:
    float64 -> float32 (round to nearest), then round-to-nearest-even of the float32 mantissa at bit 16 by bit masking.
    NaN stays NaN; values beyond the bf16 range overflow to +-inf."""
    a32 = np.array(A, dtype=np.float32, copy=True, ndmin=1)
    bits = a32.view(np.uint32)
    bits += ((bits >> 16) & _ONE32) + _BF16_HALF
    bits &= _BF16_MASK
    out = bits.view(np.float32).astype(np.float64)
    nan = np.isnan(np.asarray(A, dtype=np.float64))
    if np.any(nan):
        out = np.where(np.reshape(nan, out.shape), np.nan, out)
    return out.reshape(np.shape(A)) if np.ndim(A) else float(out[0])


# ----------------------------------------------------------------------------------------------
# Newton-Schulz iterations
# ----------------------------------------------------------------------------------------------
def ns5(M, steps=5, coeffs=NS5_COEFFS, eps=1e-7, bf16=False):
    """Keller Jordan's zeropower_via_newtonschulz5 (public Muon code), in float64 or with bf16 emulation:
        X = M (bf16 cast); transpose if rows > cols; X = X / (||X||_F + eps);
        repeat `steps` times: A = X X^T; B = b A + (c A) A; X = a X + B X;  transpose back.
    (`c * A @ A` in the PyTorch code parses as (c A) A; the same order is used here.) In exact arithmetic the output
    is U diag(p^steps(s / (||M||_F + eps))) V^T with p(x) = a x + b x^3 + c x^5 (see `ns_scalar_map`).
    Because of eps the map is not scale invariant: ns5(c M) = ns5(M) only for ||M||_F >> eps, and the output shrinks
    with M below that (M = 0 gives 0). For fixed-length steps on a vanishing estimator this sets a floor near 1e-18
    relative squared distance on the benchmark instances and breaks the exact scale covariance of the iteration
    (module docstring); eps = 1e-30 removes both, eps = 0 is exactly scale invariant but divides by zero at M = 0."""
    a, b, c = (float(v) for v in coeffs)
    M = np.asarray(M, dtype=np.float64)
    if bf16:
        return _ns_bf16(M, steps, a, b, c, eps)
    tr = M.shape[0] > M.shape[1]
    X = M.T if tr else M
    X = X / (np.sqrt(np.vdot(X, X)) + eps)
    for _ in range(steps):
        A = np.dot(X, X.T)
        B = np.dot(c * A, A)
        B += b * A                     # = b A + (c A) A (IEEE addition is commutative)
        Y = np.dot(B, X)
        X = a * X
        X += Y                         # = a X + B X
    return X.T if tr else X


def _ns_bf16(M, steps, a, b, c, eps):
    """Polynomial Newton-Schulz with every intermediate rounded to bf16 (see the module docstring)."""
    r = bf16_round
    X = r(M)
    tr = X.shape[0] > X.shape[1]
    if tr:
        X = X.T
    nrm = r(np.sqrt(np.vdot(X, X)))
    X = r(X / r(nrm + eps))
    for _ in range(steps):
        A = r(np.dot(X, X.T))
        B = r(r(b * A) + r(np.dot(r(c * A), A)))
        X = r(r(a * X) + r(np.dot(B, X)))
    return X.T if tr else X


def ns_cubic(M, steps=30, bf16=False):
    """Classical cubic Newton-Schulz iteration X <- 1.5 X - 0.5 (X X^T) X after Frobenius scaling X = M / ||M||_F
    (M = 0 gives 0). Converges to polar(M) on the nonzero singular values (see the module docstring for the speed)."""
    M = np.asarray(M, dtype=np.float64)
    nrm = float(np.sqrt(np.vdot(M, M)))
    if nrm == 0.0:
        return np.zeros_like(M)
    if bf16:
        return _ns_bf16(M, steps, 1.5, -0.5, 0.0, 0.0)
    tr = M.shape[0] > M.shape[1]
    X = (M.T if tr else M) / nrm
    for _ in range(steps):
        A = np.dot(X, X.T)
        Y = np.dot(A, X)
        X = 1.5 * X
        X -= 0.5 * Y
    return X.T if tr else X


def ns_scalar_map(x, steps=5, coeffs=NS5_COEFFS):
    """Iterate the scalar odd polynomial x <- a x + b x^3 + c x^5 (`steps` times) on an array of normalized singular
    values. In exact arithmetic the singular values of ns5(M) are ns_scalar_map(s / (||M||_F + eps)) (and those of
    ns_cubic(M) are ns_scalar_map(s / ||M||_F, steps, CUBIC_COEFFS)); used to compute the output band."""
    a, b, c = (float(v) for v in coeffs)
    x = np.array(x, dtype=np.float64, copy=True)
    for _ in range(steps):
        x2 = x * x
        x = x * (a + x2 * (b + c * x2))
    return x


# ----------------------------------------------------------------------------------------------
# control maps of fixed length (not orthogonalizations)
# ----------------------------------------------------------------------------------------------
def frob_normalize(M):
    """M / ||M||_F: the unit-Frobenius-norm direction of M (normalized gradient descent). A CONTROL, not an
    orthogonalization: the relative singular values of M are unchanged. M = 0 gives 0; a non-finite M, or a norm
    that overflows, gives an all-NaN result (as polar_svd)."""
    M = np.asarray(M, dtype=np.float64)
    nrm = float(np.sqrt(np.vdot(M, M)))
    if nrm == 0.0:
        return np.zeros_like(M)
    if not np.isfinite(nrm):
        return np.full(M.shape, np.nan)
    return M / nrm


def sign_map(M, center=False):
    """Coordinatewise sign of M (0 where M is 0, NaN where it is NaN): the direction of signSGD. A CONTROL, not an
    orthogonalization. With center=True the result is centered over the rows, O - (1/K) 1 1^T O: the projection
    on the softmax gauge subspace (SoftmaxProblem.remove_gauge), without which the iterates of a softmax problem
    leave the subspace 1^T W = 0 (see the module docstring)."""
    O = np.sign(np.asarray(M, dtype=np.float64))
    if center:
        O = O - O.mean(axis=0, keepdims=True)
    return O


# ----------------------------------------------------------------------------------------------
# dispatcher
# ----------------------------------------------------------------------------------------------
ORTH_METHODS = ("svd", "ns5", "ns_cubic")
CONTROL_MAPS = ("frob", "sign")          # fixed-length maps that are NOT orthogonalizations (controls)


def get_orth(method="svd", **kw):
    """Return the map M -> O for method in {'svd', 'ns5', 'ns_cubic'} (keyword arguments forwarded: rtol for 'svd';
    steps, coeffs, eps, bf16 for 'ns5'; steps, bf16 for 'ns_cubic'), for one of the control maps {'frob', 'sign'}
    (not orthogonalizations, see the module docstring; center for 'sign'), or `method` itself if it is callable."""
    if callable(method):
        return method if not kw else (lambda M: method(M, **kw))
    if method == "frob":
        if kw:
            raise TypeError(f"unexpected arguments for 'frob': {sorted(kw)}")
        return frob_normalize
    if method == "sign":
        center = bool(kw.pop("center", False))
        if kw:
            raise TypeError(f"unexpected arguments for 'sign': {sorted(kw)}")
        return (lambda M: sign_map(M, center=True)) if center else sign_map
    if method == "svd":
        rtol = kw.pop("rtol", 1e-15)
        if kw:
            raise TypeError(f"unexpected arguments for 'svd': {sorted(kw)}")
        return lambda M: polar_svd(M, rtol=rtol)
    if method == "ns5":
        return lambda M: ns5(M, **kw)
    if method == "ns_cubic":
        return lambda M: ns_cubic(M, **kw)
    raise ValueError(f"unknown orthogonalization method {method!r}; expected one of {ORTH_METHODS}, a control map "
                     f"{CONTROL_MAPS} or a callable")


def orth(M, method="svd", **kw):
    """Orthogonalize M with method in {'svd', 'ns5', 'ns_cubic'} (or a callable; the control maps 'frob' and 'sign'
    of `get_orth` are accepted too, although they do not orthogonalize)."""
    return get_orth(method, **kw)(M)


def orth_singular_values(M, method="ns5", **kw):
    """Singular values (decreasing) of the output orth(M, method)."""
    return singular_values(orth(M, method, **kw))
