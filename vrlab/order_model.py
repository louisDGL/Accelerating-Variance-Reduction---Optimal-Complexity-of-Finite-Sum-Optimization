"""Solvable model of the sampling order in SAGA (hypothesis B4): finite sums with identical Hessians.

Model. f_i(theta) = 1/2 theta^T H theta - b_i^T theta with ONE symmetric positive definite H for all i
(vrlab.problems.IdenticalHessianProblem, built by `identical_hessian_problem`). Write e = theta - eta_* and
psi_j = phi_j - eta_*, where phi_j is the point at which the memory z_j = f'_j(phi_j) was computed. Then the SAGA
direction is f'_i(theta) - z_i + zbar = H (e - psi_i + psibar), psibar = mean_j psi_j: the b_i have disappeared and

    e+ = e - gamma H (e - psi_i + psibar),      then  psi_i <- e   (the error BEFORE the step).

In the eigenbasis of H every mode with eigenvalue lam follows the scalar recursion

    e+ = e - g (e - psi_i + psibar),  psi_i <- e,      g = gamma lam,      c = n g = gamma n lam,

driven by the same index sequence for all modes. The memory starts at psi_j = e_0 (the code initialises
z_i = f'_i(theta_0)). Everything below concerns this scalar recursion; `simulate` runs it for the four orders of
vrlab.algorithms.saga with the same index streams, so that saga() on an identical-Hessian problem reproduces it mode
by mode (tests/test_vrlab.py, archived locally in _archive/).

Status of the statements (each was tested in tests/test_vrlab.py, now archived locally in _archive/, against an
independent computation):

(R1) i.i.d. index, exact at every n [proved: linear algebra; tested by enumerating all index sequences].
     (A, B, C, P) = (E e^2, E mean_j psi_j^2, E psibar^2, E e psibar) follows the linear recursion of
     `iid_moment_matrix`. As n -> infinity at fixed c its four modes decay at 2c, 1, 2 and c + 1 nats per pass, so
     E e_t^2 decays at min(2c, 1) nats per pass: the memory ceiling e^-1 per pass is attained IN EXPECTATION for
     c >= 1/2, and for c > 1/2, after a transient, E e_t^2 ~ (2 c^3 / ((2c - 1)^2 n)) exp(-t/n) e_0^2
     (`iid_plateau`). This is a statement about the expectation: it is carried by the rare runs in which a slot of
     the initial memory survived until shortly before t (the expected number of never-refreshed slots after t
     iterations is n (1 - 1/n)^t, about n exp(-t/n)); once n exp(-t/n) < 1 a typical run has no such slot and decays
     faster than its expectation. Expectation and typical behaviour must be kept apart: the typical exponent is a
     Lyapunov exponent, measured by simulation; `iid_typical_heuristic` is a heuristic for it [numerical].
     At FIXED g = gamma lam (the regime of a stiff mode at a step that does not shrink with n) the same matrix gives
     lim_n iid_exponent(n, n g) = 2 (1 - g) / (2 - g) for 0 < g < 1 and divergence of E e_t^2 for g > 1
     [proved: perturbation of a semi-simple eigenvalue; `iid_exponent_fixed_g`].

(R2) fixed order (cyclic; any fixed permutation gives the same sequence e_t in this model, the memory being uniform
     at the start), exact at every n [proved; tested against the simulator]:
         e_{t+1} = (1 - g) e_t - (g/n) sum_{s=1}^{n-1} e_{t-s} + (g - g/n) e_{t-n},     e_k := e_0 for k <= 0.
     With roots x = exp(s/n) the characteristic equation tends, as n -> infinity, to
         chi_c(s) = s^2 e^s + c (s + 1)(e^s - 1) = 0        (s = 0 is a spurious root),
     the characteristic equation of the delay equation e'(u) = -c e(u) + c e(u-1) - c int_{u-1}^u e [the convergence
     of the growth rate is proved in the B4 note (Hurwitz's theorem); tested here against the companion matrix at
     large n]. A pair of roots sits on the imaginary axis at s = i omega iff omega^2 = 2c and
     tan(omega/2) = -1/omega [proved: real and imaginary parts]; the smallest positive solution gives
     (omega_*, c_*) = `cyclic_critical()`: fixed-order SAGA is unstable in the model for gamma n lam > c_* and n
     large enough [proved in the note: argument principle; at finite n the threshold is `cyclic_radius` = 1].

(R3) fresh permutation at every pass (random reshuffling). Conditionally on the memory at the start of a pass, of
     which only the mean m matters, the FIRST moments follow an exact recursion at every n [proved; tested by
     averaging over all permutations]:
         e_{k+1} = (1 - g) e_k + g m - (g/n)(S_k + (n - k) m),   S_{k+1} = S_k + e_k,   S_0 = 0,
     and the pass maps (e_0, m) to (e_n, S_n / n) (`rr_pass_matrix`). Fluid limit n -> infinity at fixed c, with
     I(u) the running integral of the error over the pass: I'' + c I' + c I = c m u, I(0) = 0, I'(0) = e_0, and the
     pass maps (e_0, m) to (I'(1), I(1)) by the 2x2 matrix `rr_fluid_matrix` (closed form) [proved]. The SECOND
     moments close too: over one reshuffled pass (A, B, C, P) is mapped by an exact 4x4 matrix at every n
     (`rr_moment_matrix`) [proved: a linear system driven by a sample without replacement; tested by averaging
     over all permutations], so that the exponent of E e^2 is `rr_ms_exponent` = -log(spectral radius). Two regimes
     as n -> infinity: at fixed c it tends to -2 log(spectral radius of the fluid matrix) = `rr_exponent(c)`, which
     tends to 2 as c -> infinity (twice the ceiling of (R1)); at fixed g = gamma lam it tends to
     min(2, log((2 - g) / g)) (`rr_exponent_fixed_g`): the dispersion of the memory is multiplied by g / (2 - g) per
     pass, which is the slower mode for g > 2 / (1 + e^2) = 0.238, and E e^2 diverges for g > 1 although the mean
     still converges [proved: the deviation from the mean path is sum_l w_{k-l} y_l with
     w_j = g (1 - g)^j + O(1/n) (closed form, tested), driven by the centred memory read without replacement].

Beyond identical Hessians [heuristic]: `effective_curvature` gives the constant L~ that plays the role of lam in
the factor g / (2 - g) when the Hessians H_i differ (L~ = lam_max(H) when they are equal).

Other extreme [exact, 2x2 maps]: `orth_*` treats sums of n components acting on orthogonal directions, where the
order hardly matters and every order needs gamma R^2 below about 2/3.

Exponents are per pass (n iterations) and for SQUARED errors: an exponent r means a decay like exp(-r t / n).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from . import problems, solvers
from .algorithms import SAMPLINGS, _check_sampling, _index_stream


_BRENT_RTOL = 4.0 * np.finfo(np.float64).eps          # smallest relative tolerance accepted by scipy's brentq


def _check_n_g(n, g):
    if int(n) != n or n < 1:
        raise ValueError("n must be a positive integer")
    if not g > 0:
        raise ValueError("g = gamma * lambda (or c = n g) must be positive")


# ----------------------------------------------------------------------------------------------
# the scalar model and its simulator
# ----------------------------------------------------------------------------------------------
def simulate_indices(n, g, indices, e0=1.0, psi0=None):
    """Run e+ = e - g (e - psi_i + psibar), psi_i <- e along the given index sequence.
    psi0: initial memory (n,), default e0 everywhere (z_i = f'_i(theta_0)). psibar is updated incrementally and
    recomputed from the memory every n iterations, as zbar in vrlab.algorithms.saga.
    Returns {"e": (len(indices) + 1,) errors e_0..e_T, "psi": final memory, "psibar": its mean}."""
    _check_n_g(n, g)
    idx = np.asarray(indices, dtype=np.int64)
    if idx.ndim != 1 or (idx.size and (idx.min() < 0 or idx.max() >= n)):
        raise ValueError("indices must be a 1-d sequence of integers in {0..n-1}")
    psi = np.full(n, float(e0)) if psi0 is None else np.array(psi0, dtype=np.float64).copy()
    if psi.shape != (n,):
        raise ValueError("psi0 must have shape (n,)")
    e = float(e0)
    pbar = float(psi.mean())
    out = np.empty(idx.size + 1)
    out[0] = e
    for t, i in enumerate(idx.tolist(), start=1):
        e_new = e - g * (e - psi[i] + pbar)
        pbar += (e - psi[i]) / n
        psi[i] = e
        e = e_new
        out[t] = e
        if t % n == 0:
            pbar = float(psi.mean())
    return {"e": out, "psi": psi, "psibar": pbar}


def simulate(n, g, order="iid", n_iters=None, passes=None, seed=0, e0=1.0, psi0=None):
    """Scalar model along the index stream of vrlab.algorithms.saga(..., seed=seed, sampling=order): the same seed
    gives the same indices as the SAGA run. Give n_iters or passes (n iterations each).
    Returns the dict of `simulate_indices` plus "indices", "order", "n", "g"."""
    _check_n_g(n, g)
    _check_sampling(order)
    if (n_iters is None) == (passes is None):
        raise ValueError("give exactly one of n_iters and passes")
    n_iters = int(passes * n) if n_iters is None else int(n_iters)
    chunks = list(_index_stream(np.random.default_rng(seed), n, n_iters, sampling=order))
    idx = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int64)
    out = simulate_indices(n, g, idx, e0=e0, psi0=psi0)
    out.update({"indices": idx, "order": order, "n": n, "g": g})
    return out


def simulate_passes(n, g, order="iid", passes=100, seed=0):
    """Long runs without underflow: the state (e, psi) is rescaled after every pass (the recursion is linear) and the
    logarithm of the scale is accumulated. Starts from e_0 = 1 and the uniform memory psi_j = e_0.
    Returns {"log_e2": log e^2 at the end of each pass (passes + 1 values, the first is 0; -inf if e = 0),
             "log_size2": log max(e^2, max_j psi_j^2), the squared size of the whole state, which does not vanish
             when the error oscillates through zero}."""
    _check_n_g(n, g)
    _check_sampling(order)
    if int(passes) != passes or passes < 0:
        raise ValueError("passes must be a non-negative integer (the state is rescaled at pass ends)")
    # the whole index stream of saga(..., seed=seed, sampling=order), then cut into passes of n indices
    idx = np.concatenate(list(_index_stream(np.random.default_rng(seed), n, int(passes) * n, sampling=order))
                         or [np.zeros(0, dtype=np.int64)])
    e, psi, logscale = 1.0, np.ones(n), 0.0
    log_e2, log_size2 = [0.0], [0.0]
    for start in range(0, len(idx), n):
        r = simulate_indices(n, g, idx[start:start + n], e0=e, psi0=psi)
        e, psi = float(r["e"][-1]), r["psi"]
        size = max(abs(e), float(np.abs(psi).max()))
        if size == 0.0 or not np.isfinite(size):
            raise FloatingPointError("the state of the scalar model vanished or overflowed within one pass")
        e, psi, logscale = e / size, psi / size, logscale + np.log(size)
        with np.errstate(divide="ignore"):
            log_e2.append(2.0 * (logscale + np.log(abs(e))))
        log_size2.append(2.0 * logscale)                           # the rescaled state has size 1
    return {"log_e2": np.array(log_e2), "log_size2": np.array(log_size2), "order": order, "n": n, "g": g}


def fit_exponent(log_values, burn=0):
    """Least-squares exponent (nats per pass) of a per-pass series of logarithms: minus the slope over the passes
    >= burn (non-finite values dropped)."""
    y = np.asarray(log_values, dtype=np.float64)
    x = np.arange(y.size, dtype=np.float64)
    sel = (x >= burn) & np.isfinite(y)
    if sel.sum() < 2:
        raise ValueError("fewer than two finite values after `burn`")
    return float(-np.polyfit(x[sel], y[sel], 1)[0])


# ----------------------------------------------------------------------------------------------
# (R1) i.i.d. sampling: exact recursion of the second moments
# ----------------------------------------------------------------------------------------------
def iid_moment_matrix(n, g):
    """4x4 matrix T with (A, B, C, P)_{t+1} = T (A, B, C, P)_t for a uniform index independent of the state, where
    A = E e^2, B = E mean_j psi_j^2, C = E psibar^2, P = E e psibar:
        A+ = (1-g)^2 A + g^2 B - g^2 C
        B+ = A/n + (1 - 1/n) B
        C+ = A/n^2 + B/n^2 + (1 - 2/n) C + (2/n - 2/n^2) P
        P+ = ((1-g)/n) A - (g/n) B + (g/n) C + (1-g)(1 - 1/n) P.
    The uniform memory psi_j = e_0 is the vector (1, 1, 1, 1) e_0^2."""
    _check_n_g(n, g)
    T = np.zeros((4, 4))
    T[0] = [(1.0 - g) ** 2, g * g, -g * g, 0.0]
    T[1] = [1.0 / n, 1.0 - 1.0 / n, 0.0, 0.0]
    T[2] = [1.0 / n ** 2, 1.0 / n ** 2, 1.0 - 2.0 / n, 2.0 / n - 2.0 / n ** 2]
    T[3] = [(1.0 - g) / n, -g / n, g / n, (1.0 - g) * (1.0 - 1.0 / n)]
    return T


def iid_moments(n, g, n_iters, init=(1.0, 1.0, 1.0, 1.0)):
    """Exact moments (A, B, C, P)_t for t = 0..n_iters, array (n_iters + 1, 4); `init` defaults to the uniform memory
    with e_0 = 1, so that column 0 is E e_t^2 / e_0^2."""
    T = iid_moment_matrix(n, g)
    out = np.empty((int(n_iters) + 1, 4))
    out[0] = init
    for t in range(int(n_iters)):
        out[t + 1] = T @ out[t]
    return out


def iid_exponent(n, c):
    """Exponent per pass of E e_t^2 at finite n: -n log(spectral radius of iid_moment_matrix(n, c/n))."""
    ev = np.linalg.eigvals(iid_moment_matrix(n, c / n))
    return float(-n * np.log(np.abs(ev).max()))


def iid_limit_modes(c):
    """Limits as n -> infinity of the exponents per pass of the four modes of (R1): (2c, 1, 2, c + 1), the
    eigenvalues of minus the generator lim n (T - I) (triangular in the order A, B, P, C)."""
    return (2.0 * c, 1.0, 2.0, c + 1.0)


def iid_exponent_limit(c):
    """lim_n iid_exponent(n, c) = min(2c, 1): the memory ceiling of one nat per pass binds for c >= 1/2."""
    return float(min(2.0 * c, 1.0))


def iid_plateau(c):
    """For c > 1/2, lim_n of n times the amplitude of the slow mode in E e_t^2 / e_0^2 (uniform memory):
    E e_t^2 ~ (iid_plateau(c) / n) exp(-t/n) e_0^2 once exp(-(2c - 1) t/n) is negligible, with
    iid_plateau(c) = 2 c^3 / (2c - 1)^2. (Leading order: A = exp(-2c u), B' = A - B gives the slow part
    B ~ (2c/(2c-1)) exp(-u), and A' = -2c A + (c^2/n)(B - C) gives A ~ (c^2/((2c-1) n)) B.)"""
    if not c > 0.5:
        raise ValueError("the slow mode is the memory mode only for c > 1/2")
    return float(2.0 * c ** 3 / (2.0 * c - 1.0) ** 2)


def iid_slow_amplitude(n, c):
    """Finite-n counterpart of iid_plateau(c) / n: the coefficient a with E e_t^2 / e_0^2 ~ a rho^t for large t,
    rho the dominant eigenvalue of iid_moment_matrix(n, c/n) (component of the vector (1, 1, 1, 1) on the dominant
    eigenvector, read on the coordinate A). Requires a real simple dominant eigenvalue (true away from c = 1/2).
    For c < 1/2 the dominant mode is the gradient mode exp(-2c t/n), not the memory mode, and the value returned is
    the amplitude of that mode (close to 1), which iid_plateau does not describe."""
    T = iid_moment_matrix(n, c / n)
    w, V = np.linalg.eig(T)
    k = int(np.argmax(np.abs(w)))
    coef = np.linalg.solve(V, np.ones(4))
    a = coef[k] * V[0, k]
    if abs(a.imag) > 1e-9 * max(1.0, abs(a.real)):
        raise ValueError("the dominant eigenvalue is not real: no single slow mode (c too close to 1/2 ?)")
    return float(a.real)


# ----------------------------------------------------------------------------------------------
# (R2) fixed order: linear recurrence of order n + 1, limiting characteristic function, critical step
# ----------------------------------------------------------------------------------------------
def cyclic_coefficients(n, g):
    """Coefficients (a_0, .., a_n) of e_{t+1} = sum_{s=0}^{n} a_s e_{t-s} for a fixed order:
    a_0 = 1 - g, a_s = -g/n (1 <= s <= n-1), a_n = g - g/n (for n = 1: a_0 = 1 - g, a_1 = 0, plain gradient steps)."""
    _check_n_g(n, g)
    a = np.full(n + 1, -g / n)
    a[0] = 1.0 - g
    a[n] = g - g / n
    return a


def cyclic_recurrence(n, g, n_iters, e0=1.0):
    """e_0..e_T from the recurrence of (R2), with the history e_k = e_0 for k <= 0 (uniform memory)."""
    a = cyclic_coefficients(n, g)
    hist = np.full(n + 1, float(e0))                 # (e_t, e_{t-1}, .., e_{t-n})
    out = np.empty(int(n_iters) + 1)
    out[0] = e0
    for t in range(int(n_iters)):
        new = float(a @ hist)
        hist[1:] = hist[:-1]
        hist[0] = new
        out[t + 1] = new
    return out


def cyclic_companion(n, g):
    """Companion matrix ((n+1) x (n+1)) of the recurrence of (R2), acting on (e_t, e_{t-1}, .., e_{t-n})."""
    a = cyclic_coefficients(n, g)
    A = np.zeros((n + 1, n + 1))
    A[0] = a
    A[np.arange(1, n + 1), np.arange(0, n)] = 1.0
    return A


def cyclic_radius(n, g):
    """Spectral radius of cyclic_companion(n, g): fixed-order SAGA is stable in the model iff it is < 1."""
    return float(np.abs(np.linalg.eigvals(cyclic_companion(n, g))).max())


def cyclic_exponent(n, c):
    """Exponent per pass of the squared error for a fixed order at finite n: -2 n log cyclic_radius(n, c/n)
    (negative: divergence)."""
    return float(-2.0 * n * np.log(cyclic_radius(n, c / n)))


def cyclic_char(s, c):
    """Limiting characteristic function chi_c(s) = s^2 e^s + c (s + 1)(e^s - 1) (s complex; s = 0 is spurious)."""
    s = np.asarray(s, dtype=np.complex128)
    return s * s * np.exp(s) + c * (s + 1.0) * np.expm1(s)


def cyclic_char_prime(s, c):
    """d chi_c / ds = (s^2 + 2s) e^s + c (e^s - 1) + c (s + 1) e^s."""
    s = np.asarray(s, dtype=np.complex128)
    es = np.exp(s)
    return (s * s + 2.0 * s) * es + c * np.expm1(s) + c * (s + 1.0) * es


def cyclic_real_root(c):
    """The unique real root of chi_c other than 0; it lies in (-1, 0) (e^{-s} chi_c(s) is convex there, vanishes at 0
    with slope c > 0 and equals 1 at s = -1; chi_c > 0 on the rest of the real axis)."""
    if not c > 0:
        raise ValueError("c must be positive")
    f = lambda s: s * s * np.exp(s) + c * (s + 1.0) * np.expm1(s)
    return float(brentq(f, -1.0, -1e-6 * min(1.0, c), xtol=1e-15, rtol=_BRENT_RTOL, maxiter=200))


def cyclic_roots(c, re_min=-6.0, re_max=3.0, im_max=None):
    """Roots of chi_c in the rectangle [re_min, re_max] x [0, im_max] (one of each conjugate pair, the spurious root 0
    excluded), by Newton's method from a grid of starting points, sorted by decreasing real part. im_max defaults to
    max(40, 3 sqrt(2c) + 15) (a root on the imaginary axis has omega^2 = 2c). The count is not certified by this
    function; tests compare the rightmost root with the companion matrix at large n."""
    if not c > 0:
        raise ValueError("c must be positive")
    im_max = max(40.0, 3.0 * np.sqrt(2.0 * c) + 15.0) if im_max is None else float(im_max)
    re = np.arange(re_min, re_max + 1e-9, 0.5)
    im = np.arange(0.0, im_max + 1e-9, 0.5)
    s = (re[:, None] + 1j * im[None, :]).ravel()
    with np.errstate(all="ignore"):
        for _ in range(100):
            fp = cyclic_char_prime(s, c)
            step = cyclic_char(s, c) / np.where(fp == 0, 1.0, fp)
            s = s - step
        res = np.abs(cyclic_char(s, c)) / (np.abs(s) ** 2 * np.abs(np.exp(s)) + c * np.abs(s + 1.0) + 1e-300)
    ok = np.isfinite(s.real) & np.isfinite(s.imag) & (res < 1e-12) & (np.abs(s) > 1e-6)
    s = s[ok]
    s = np.where(s.imag < 0, np.conj(s), s)
    s = s[(s.real >= re_min) & (s.real <= re_max) & (s.imag <= im_max)]
    roots = []
    for z in s[np.argsort(-s.real, kind="stable")]:
        z = complex(z.real, 0.0 if abs(z.imag) < 1e-9 else z.imag)
        if all(abs(z - r) > 1e-7 for r in roots):
            roots.append(z)
    return roots


def cyclic_rightmost_root(c):
    """Root of chi_c with the largest real part (imaginary part >= 0), s = 0 excluded: the squared error of
    fixed-order SAGA behaves like exp(2 Re(s) t/n) in the limit n -> infinity."""
    roots = cyclic_roots(c) + [complex(cyclic_real_root(c), 0.0)]
    return max(roots, key=lambda z: z.real)


def cyclic_exponent_limit(c):
    """lim_n cyclic_exponent(n, c) = -2 Re(cyclic_rightmost_root(c)) (negative for c > c_*)."""
    return float(-2.0 * cyclic_rightmost_root(c).real)


def cyclic_critical():
    """(omega_*, c_*): smallest positive omega with tan(omega/2) = -1/omega, i.e. the root of
    omega sin(omega/2) + cos(omega/2) in (pi, 2 pi) (positive at pi, negative at 2 pi; no solution in (0, pi], where
    tan(omega/2) > 0), and c_* = omega_*^2 / 2. chi_{c_*}(i omega_*) = 0; for c > c_* the fixed order is unstable."""
    h = lambda w: w * np.sin(0.5 * w) + np.cos(0.5 * w)
    omega = float(brentq(h, np.pi, 2.0 * np.pi, xtol=1e-15, rtol=_BRENT_RTOL, maxiter=200))
    return omega, 0.5 * omega * omega


def cyclic_best(c_lo=0.2, c_hi=None):
    """(c, exponent) maximising cyclic_exponent_limit over [c_lo, c_hi] (c_hi defaults to c_*): the best step and
    the best squared-error exponent per pass of a fixed order in the limit n -> infinity. The maximum sits where the
    real root and a complex pair have the same real part (a kink), located here to about 1e-8 in c."""
    c_hi = cyclic_critical()[1] if c_hi is None else float(c_hi)
    res = minimize_scalar(lambda c: -cyclic_exponent_limit(c), bounds=(c_lo, c_hi), method="bounded",
                          options={"xatol": 1e-9, "maxiter": 500})
    return float(res.x), float(-res.fun)


# ----------------------------------------------------------------------------------------------
# (R3) random reshuffling: exact first-moment pass map, fluid limit
# ----------------------------------------------------------------------------------------------
def rr_pass_matrix(n, g):
    """Exact 2x2 matrix of the conditional first moments over one reshuffled pass: given the error e_0 and the memory
    at the start of the pass, with mean m, E[(e_n, m')] = M (e_0, m), where m' = S_n / n is the mean of the memory at
    the end of the pass (it holds e_0, .., e_{n-1}). Recursion (expectation over the uniform permutation):
        e_{k+1} = (1 - g) e_k + g m - (g/n)(S_k + (n - k) m) = (1 - g) e_k - (g/n) S_k + (g k/n) m,   S_{k+1} = S_k + e_k.
    Rows: (e_n, m'); columns: (e_0, m)."""
    _check_n_g(n, g)
    cols = []
    for e, m in ((1.0, 0.0), (0.0, 1.0)):
        S = 0.0
        for k in range(n):
            e_new = (1.0 - g) * e - (g / n) * S + (g * k / n) * m
            S += e
            e = e_new
        cols.append((e, S / n))
    return np.array(cols).T


def _damped_pair(c):
    """(exp(-c/2) cosh(delta), exp(-c/2) sinh(delta)/delta) with delta^2 = c^2/4 - c (delta imaginary for c < 4),
    without overflow for large c nor cancellation near c = 4."""
    a = 0.5 * c
    disc = a * a - c
    if disc >= 0:
        d = np.sqrt(disc)
        ep, em = np.exp(-c / (a + d)), np.exp(-a - d)            # -a + d = -c / (a + d): no cancellation for large c
        ch = 0.5 * (ep + em)
        if d > 1.0:
            sh = 0.5 * (ep - em) / d
        else:
            sh = np.exp(-a) * (np.sinh(d) / d if d > 1e-4 else 1.0 + d * d / 6.0)
    else:
        w = np.sqrt(-disc)
        ch = np.exp(-a) * np.cos(w)
        sh = np.exp(-a) * (np.sin(w) / w if w > 1e-4 else 1.0 - w * w / 6.0)
    return float(ch), float(sh)


def rr_fluid_matrix(c):
    """Fluid limit (n -> infinity, c = n g fixed) of rr_pass_matrix, in closed form. With I(u) = int_0^u e over the
    pass, I'' + c I' + c I = c m u, I(0) = 0, I'(0) = e_0; a particular solution is m (u - 1), and with h_1, h_0 the
    solutions of the homogeneous equation such that (h_1, h_1')(0) = (0, 1), (h_0, h_0')(0) = (1, 0),
        I = m (u - 1) + m h_0 + (e_0 - m) h_1,   h_1(u) = e^{-cu/2} sinh(delta u)/delta,
        h_0(u) = e^{-cu/2} (cosh(delta u) + (c/2) sinh(delta u)/delta),   delta^2 = c^2/4 - c   (h_0' = -c h_1).
    The pass maps (e_0, m) to (I'(1), I(1)):
        [[h_1'(1), 1 + h_0'(1) - h_1'(1)], [h_1(1), h_0(1) - h_1(1)]].
    As c -> infinity the matrix tends to [[0, 1 - 1/e], [0, 1/e]]."""
    if not c > 0:
        raise ValueError("c must be positive")
    ch, sh = _damped_pair(c)
    a = 0.5 * c
    h1, dh1 = sh, ch - a * sh
    h0, dh0 = ch + a * sh, -c * sh
    return np.array([[dh1, 1.0 + dh0 - dh1], [h1, h0 - h1]])


def rr_exponent(c, n=None):
    """Squared-error exponent per pass of reshuffled SAGA in the model: -2 log(spectral radius) of the fluid matrix
    (n=None: the limit n -> infinity), or of the exact first-moment matrix rr_pass_matrix(n, c/n), which at finite n
    is an upper bound on the exponent of E e^2 (Jensen) and its limit as the fluctuations vanish."""
    M = rr_fluid_matrix(c) if n is None else rr_pass_matrix(n, c / n)
    return float(-2.0 * np.log(np.abs(np.linalg.eigvals(M)).max()))


# ----------------------------------------------------------------------------------------------
# (R3) random reshuffling: exact second moments over a pass, and the two regimes n -> infinity
# ----------------------------------------------------------------------------------------------
def rr_moment_matrix(n, g):
    """Exact 4x4 matrix of the second moments over ONE reshuffled pass (n >= 2): with (A, B, C, P) =
    (E e^2, E mean_j psi_j^2, E psibar^2, E e psibar) taken at pass ends, (A, B, C, P)' = R (A, B, C, P), for a uniform
    permutation independent of the state (the counterpart of `iid_moment_matrix`, which advances one iteration).

    Derivation. Condition on the state (e, psi) at the start of the pass, m = mean psi, v = mean psi^2 - m^2. With
    y_k = psi_{pi(k)} - m (a sample without replacement of a centred population of variance v: E y_k^2 = v,
    E y_k y_l = -v / (n - 1)), S_k = e_0 + .. + e_{k-1} and Y_k = y_1 + .. + y_k,
        e_{k+1} = (1 - g) e_k - (g/n) S_k + (g k/n) m + g y_{k+1} + (g/n) Y_k.
    The mean follows the recursion of `rr_pass_matrix`; the deviation z = (e - E e, S - E S, Y) follows
    z_{k+1} = F z_k + h y_{k+1} with E[y_{k+1} | past] = -Y_k / (n - k), so that its covariance obeys an exact linear
    recursion, proportional to v. At the end of the pass the memory holds e_0, .., e_{n-1}. Hence
        A' = (E e_n)^2 + a v,  C' = (E S_n / n)^2 + c v,  P' = E e_n E S_n / n + p v,  B' = mean_k (E e_k)^2 + b v,
    with constants a, b, c, p depending on (n, g) only, and e^2 -> A, e m -> P, m^2 -> C, v -> B - C.
    Cost O(n). Tested against the average over all permutations."""
    _check_n_g(n, g)
    if n < 2:
        raise ValueError("n >= 2 is needed (the covariance of a sample without replacement has the factor 1/(n-1))")
    # means: e_k = al_k e + be_k m, accumulated sums for B'
    al, be, Sa, Sb = 1.0, 0.0, 0.0, 0.0
    bAA = bAP = bCC = 0.0
    # covariance of z = (de, dS, Y) for v = 1
    F = np.array([[1.0 - g, -g / n, g / n], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    h = np.array([g, 0.0, 1.0])
    Sig = np.zeros((3, 3))
    b_acc = 0.0
    for k in range(n):
        bAA += al * al
        bAP += al * be
        bCC += be * be
        b_acc += Sig[0, 0]
        al_new = (1.0 - g) * al - (g / n) * Sa
        be_new = (1.0 - g) * be - (g / n) * Sb + g * k / n
        Sa += al
        Sb += be
        al, be = al_new, be_new
        cross = -(F @ Sig[:, 2]) / (n - k)                 # F E[z_k y_{k+1}],  E[z_k y_{k+1}] = -E[z_k Y_k] / (n - k)
        Sig = F @ Sig @ F.T + np.outer(cross, h) + np.outer(h, cross) + np.outer(h, h)
    M11, M12, M21, M22 = al, be, Sa / n, Sb / n
    a_n, c_n, p_n, b_n = Sig[0, 0], Sig[1, 1] / n ** 2, Sig[0, 1] / n, b_acc / n
    R = np.zeros((4, 4))                                    # rows and columns in the order (A, B, C, P)
    R[0] = [M11 * M11, a_n, M12 * M12 - a_n, 2.0 * M11 * M12]
    R[1] = [bAA / n, b_n, bCC / n - b_n, 2.0 * bAP / n]
    R[2] = [M21 * M21, c_n, M22 * M22 - c_n, 2.0 * M21 * M22]
    R[3] = [M11 * M21, p_n, M12 * M22 - p_n, M11 * M22 + M12 * M21]
    return R


def rr_ms_exponent(n, c):
    """Exact exponent per pass of E e^2 for reshuffled SAGA in the model at finite n: -log(spectral radius of
    rr_moment_matrix(n, c/n)) (negative: divergence of the mean square). Compare with rr_exponent(c, n=n), the exponent
    of the squared MEAN (never smaller, by Jensen's inequality)."""
    return float(-np.log(np.abs(np.linalg.eigvals(rr_moment_matrix(n, c / n))).max()))


def fluctuation_factor(g):
    """g / (2 - g): the factor by which one refresh of the whole memory multiplies its dispersion when the error is
    slaved to the memory (stationary variance of e+ = (1 - g) e + g xi, Var xi = v, is g v / (2 - g)). Limit
    n -> infinity at fixed g of the eigenvalue of rr_moment_matrix carried by B - C; for i.i.d. indices the dispersion
    then decays at 1 - g/(2 - g) nats per pass."""
    if not 0 < g < 2:
        raise ValueError("defined for 0 < g < 2")
    return float(g / (2.0 - g))


def rr_exponent_fixed_g(g):
    """lim_{n -> infinity} rr_ms_exponent(n, n g) at FIXED g = gamma lam, 0 < g < 2: min(2, log((2 - g) / g)).
    The first branch is the squared mean of the memory (e^-2 per pass), the second the dispersion of the memory
    (`fluctuation_factor`); they cross at g = 2 / (1 + e^2) = 0.2384, and the mean square diverges for g > 1."""
    return float(min(2.0, -np.log(fluctuation_factor(g))))


def iid_exponent_fixed_g(g):
    """lim_{n -> infinity} iid_exponent(n, n g) at FIXED g = gamma lam, 0 < g < 2: 2 (1 - g) / (2 - g), i.e.
    1 - fluctuation_factor(g). Positive iff g < 1: E e_t^2 diverges for g > 1. (Perturbation of the double eigenvalue
    1 of lim_n iid_moment_matrix(n, g): the reduced matrix on (B, C) is [[g/(2-g) - 1, -g/(2-g)], [0, -2]] / n.)"""
    return float(1.0 - fluctuation_factor(g))


def iid_typical_heuristic(n, c):
    """HEURISTIC for the exponent of a typical run under i.i.d. sampling when the error is slaved to the memory
    (c >> 1, c << n): min(2c, 2 - delta) with delta the solution in (0, 2) of delta log(n/c) = 2 log(2/delta).
    Argument (not a proof): a slot of age a passes holds an error exp(r a / 2) times the current one, so the stored
    values have a tail of index 2/r; the error is the centred mean of the n/c slots sampled during its response time,
    of relative size (n/c)^(r/2 - 1) times 2/(2 - r) (the largest sampled value), and r is the value at which this is
    1. Compared with simulations in experiments/exp_B4b_refinements.py."""
    _check_n_g(n, c)
    ell = np.log(n / c)
    if not ell > 0:
        raise ValueError("needs c < n")
    delta = brentq(lambda d: d * ell - 2.0 * np.log(2.0 / d), 1e-12, 2.0, xtol=1e-14, rtol=_BRENT_RTOL, maxiter=200)
    return float(min(2.0 * c, 2.0 - delta))


# ----------------------------------------------------------------------------------------------
# beyond identical Hessians: the effective curvature of the fluctuation mode (heuristic)
# ----------------------------------------------------------------------------------------------
def _top_positive_eigenvalue(apply_K, Hbar, tol=1e-13, maxiter=20000):
    """Top eigenvalue of Xi -> Lyap^{-1} K(Xi) on symmetric matrices, by power iteration from the identity. Both K
    (Xi -> mean_i H_i Xi H_i) and Lyap^{-1} (the solution of Hbar S + S Hbar = Y) map positive semi-definite matrices
    to positive semi-definite matrices, so the spectral radius is an eigenvalue with a semi-definite eigenvector."""
    w, V = np.linalg.eigh(Hbar)
    if not w[0] > 0:
        raise ValueError("the mean Hessian must be positive definite")
    denom = w[:, None] + w[None, :]
    Xi = np.eye(Hbar.shape[0])
    lam = 0.0
    for _ in range(maxiter):
        Y = apply_K(Xi)
        S = V @ ((V.T @ Y @ V) / denom) @ V.T
        S = 0.5 * (S + S.T)
        new = float(np.linalg.norm(S))                    # Xi has unit Frobenius norm
        Xi = S / new
        if abs(new - lam) <= tol * new:
            return new
        lam = new
    raise RuntimeError("power iteration did not converge")


def effective_curvature(hessians, method="power"):
    """L~ = 2 lambda_max(Lyap^{-1} K) for per-sample Hessians H_i (array (n, d, d)), where K(Xi) = mean_i H_i Xi H_i
    and Lyap(Xi) = Hbar Xi + Xi Hbar (Hbar = mean_i H_i) act on symmetric matrices. Motivation (heuristic, see the B4
    note): when the error is slaved to the memory, a dispersion Xi of the stored points produces kicks of covariance
    gamma^2 K(Xi), the stationary covariance Sigma of the error solves
    Sigma = mean_i (I - gamma H_i) Sigma (I - gamma H_i) + gamma^2 K(Xi), and the refreshed memory inherits Sigma:
    the dispersion is multiplied by the top eigenvalue of Xi -> Sigma(Xi), which equals g / (2 - g) with
    g = gamma L~ exactly (if K x = (L~/2) Lyap x then Sigma(x) = (g/2)/(1 - g/2) x). For identical Hessians
    L~ = lambda_max(H). method='power': power iteration on symmetric matrices, O(n d^3) per iteration;
    method='dense': eigenvalues of the d^2 x d^2 matrix (small d only; used by the tests as a cross-check)."""
    Hs = np.asarray(hessians, dtype=np.float64)
    if Hs.ndim != 3 or Hs.shape[1] != Hs.shape[2]:
        raise ValueError("hessians must have shape (n, d, d)")
    n, d, _ = Hs.shape
    Hbar = Hs.mean(axis=0)
    if method == "dense":
        flat = Hs.reshape(n, d * d)
        G = flat.T @ flat / n                                      # G[(a,b),(c,e)] = mean_i H_i[a,b] H_i[c,e]
        K = G.reshape(d, d, d, d).transpose(0, 2, 1, 3).reshape(d * d, d * d)   # mean_i kron(H_i, H_i)
        I = np.eye(d)
        ev = np.linalg.eigvals(np.linalg.solve(np.kron(Hbar, I) + np.kron(I, Hbar), K))
        return float(2.0 * np.max(ev.real))
    if method != "power":
        raise ValueError("method must be 'power' or 'dense'")
    return 2.0 * _top_positive_eigenvalue(lambda Xi: np.einsum("iab,bc,icd->ad", Hs, Xi, Hs, optimize=True) / n, Hbar)


def effective_curvature_glm(X, h, lam):
    """effective_curvature for H_i = h_i x_i x_i^T + lam I (generalised linear models at their optimum: X (n, d) the
    features, h (n,) the second derivatives of the losses, lam the ridge), without forming the n Hessians:
    K(Xi) = mean_i h_i^2 (x_i^T Xi x_i) x_i x_i^T + lam (H_0 Xi + Xi H_0) + lam^2 Xi, H_0 = mean_i h_i x_i x_i^T."""
    X = np.asarray(X, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    n, d = X.shape
    H0 = (X * h[:, None]).T @ X / n

    def apply_K(Xi):
        w = h * h * np.einsum("ia,ab,ib->i", X, Xi, X, optimize=True)
        return (X * w[:, None]).T @ X / n + lam * (H0 @ Xi + Xi @ H0) + lam * lam * Xi

    return 2.0 * _top_positive_eigenvalue(apply_K, H0 + lam * np.eye(d))


# ----------------------------------------------------------------------------------------------
# incremental aggregated gradient in the same model (direction zbar alone), for comparison with (R2)
# ----------------------------------------------------------------------------------------------
def iag_radius(n, g):
    """Spectral radius of the recurrence e_{t+1} = e_t - (g/n) (e_t + e_{t-1} + .. + e_{t-n+1}) (IAG with a fixed
    order on an identical-Hessian sum: the direction is the table mean, every slot holding the error at its last
    visit)."""
    _check_n_g(n, g)
    A = np.zeros((n, n))
    A[0] = -g / n
    A[0, 0] += 1.0
    A[np.arange(1, n), np.arange(0, n - 1)] = 1.0
    return float(np.abs(np.linalg.eigvals(A)).max())


def iag_critical(n=None):
    """Critical value of c = n g for the recurrence of `iag_radius`: 2 n^2 sin^2(pi / (2n)) at finite n (roots on the
    unit circle of (x - 1) x^(n-1) (x - 1) + (g/n)(x^n - 1) need sin(n theta) = 0 and g/n = 2 sin^2(theta/2), first
    at theta = pi/n), tending to pi^2 / 2 (n=None), the stability boundary of the delay equation
    e'(u) = -c int_{u-1}^u e."""
    if n is None:
        return float(np.pi ** 2 / 2.0)
    if int(n) != n or n < 2:
        raise ValueError("n must be an integer >= 2")
    return float(2.0 * n * n * np.sin(np.pi / (2.0 * n)) ** 2)


# ----------------------------------------------------------------------------------------------
# the other extreme: n components acting on orthogonal directions (d = n)
# ----------------------------------------------------------------------------------------------
def orth_cyclic_matrix(c, n=None):
    """Sum of n components f_i(theta) = (n lam / 2)(theta_i - theta*_i)^2 on orthogonal directions (per-sample
    curvature R^2 = n lam, F'' = lam I), c = gamma n lam = gamma R^2. The coordinate i with its own memory slot,
    (e, psi), evolves alone: a step on another index gives e <- e - (c/n) psi, a step on its own index gives
    e <- (1 - c) e + c (1 - 1/n) psi, psi <- e (old). With a fixed order one pass is the 2x2 map returned here,
    [[1 - c, c^2 (n-1)/n], [1, -c (n-1)/n]] (n=None: the limit [[1 - c, c^2], [1, -c]], trace 1 - 2c, determinant -c),
    stable iff c < 2/3 in the limit."""
    if not c > 0:
        raise ValueError("c must be positive")
    f = 1.0 if n is None else (n - 1.0) / n
    return np.array([[1.0 - c, c * c * f], [1.0, -c * f]])


def orth_exponent(c, order="cyclic", n=1000, passes=20000, seed=0, burn=200):
    """Squared-error exponent per pass of one coordinate of the orthogonal model: exact for the cyclic order
    (-2 log of the spectral radius of orth_cyclic_matrix(c, n)); for 'rr' (one visit per pass at a uniform position)
    and 'iid' (Binomial(n, 1/n) visits per pass at uniform positions) the Lyapunov exponent of the product of the
    random 2x2 pass maps, estimated on `passes` passes after `burn` (a typical-run exponent)."""
    if order == "cyclic":
        return float(-2.0 * np.log(np.abs(np.linalg.eigvals(orth_cyclic_matrix(c, n))).max()))
    if order not in ("rr", "iid"):
        raise ValueError("order must be 'cyclic', 'rr' or 'iid'")
    rng = np.random.default_rng(seed)
    e, psi, logscale, log_at_burn = 1.0, 1.0, 0.0, 0.0
    for p in range(int(passes)):
        if order == "rr":
            visits = [int(rng.integers(n))]
        else:
            visits = sorted(rng.integers(0, n, size=rng.binomial(n, 1.0 / n)).tolist())
        pos = 0
        for v in visits:
            e = e - (c / n) * (v - pos) * psi
            e, psi = (1.0 - c) * e + c * (1.0 - 1.0 / n) * psi, e
            pos = v + 1
        e = e - (c / n) * (n - pos) * psi
        size = float(np.hypot(e, psi))
        if size == 0.0 or not np.isfinite(size):
            raise FloatingPointError("the state of the orthogonal model vanished or overflowed")
        e, psi, logscale = e / size, psi / size, logscale + np.log(size)
        if p + 1 == burn:
            log_at_burn = logscale
    return float(-2.0 * (logscale - log_at_burn) / (int(passes) - burn))


# ----------------------------------------------------------------------------------------------
# identical-Hessian finite sums on which vrlab.algorithms.saga can be run
# ----------------------------------------------------------------------------------------------
def identical_hessian_problem(n, eigs, seed=0, rotate=True, b_scale=1.0, name=None):
    """Finite sum f_i(theta) = 1/2 theta^T H theta - b_i^T theta with H = Q diag(eigs) Q^T (Q a random rotation, or
    the identity with rotate=False) and b_i = H eta + b_scale * xi_i, eta and xi_i standard Gaussian. The optimum is
    set (eta_* = H^{-1} mean_i b_i, exact up to rounding), so that vrlab.algorithms.saga can be run on it;
    prob.modes(theta) gives the coordinates of theta - eta_* in the eigenbasis of H (ascending eigenvalues
    prob.eigvals), each of which follows the scalar recursion of this module with g = gamma * eigenvalue."""
    eigs = np.asarray(eigs, dtype=np.float64)
    if eigs.ndim != 1 or eigs.size == 0 or not np.all(eigs > 0):
        raise ValueError("eigs must be a non-empty 1-d array of positive eigenvalues")
    if int(n) != n or n < 1:
        raise ValueError("n must be a positive integer")
    d = eigs.size
    rng = np.random.default_rng(seed)
    if rotate:
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        H = (Q * eigs) @ Q.T
        H = 0.5 * (H + H.T)
    else:
        H = np.diag(eigs)
    eta = rng.standard_normal(d)
    b = (H @ eta)[None, :] + b_scale * rng.standard_normal((n, d))
    prob = problems.IdenticalHessianProblem(H, b, name=name or f"identical_hessian(n={n},d={d},seed={seed})")
    solvers.solve_optimum(prob)                       # exact solve; sets prob.opt_info like the other constructions
    prob.meta = dict(n=int(n), d=d, seed=seed, rotate=bool(rotate), b_scale=b_scale, eigs=eigs.copy())
    return prob


__all__ = ["SAMPLINGS", "simulate_indices", "simulate", "simulate_passes", "fit_exponent",
           "iid_moment_matrix", "iid_moments", "iid_exponent", "iid_limit_modes", "iid_exponent_limit", "iid_plateau",
           "iid_slow_amplitude", "cyclic_coefficients", "cyclic_recurrence", "cyclic_companion", "cyclic_radius",
           "cyclic_exponent", "cyclic_char", "cyclic_char_prime", "cyclic_real_root", "cyclic_roots",
           "cyclic_rightmost_root", "cyclic_exponent_limit", "cyclic_critical", "cyclic_best", "rr_pass_matrix",
           "rr_fluid_matrix", "rr_exponent", "rr_moment_matrix", "rr_ms_exponent", "fluctuation_factor",
           "rr_exponent_fixed_g", "iid_exponent_fixed_g", "iid_typical_heuristic", "effective_curvature",
           "effective_curvature_glm", "iag_radius", "iag_critical", "orth_cyclic_matrix", "orth_exponent",
           "identical_hessian_problem"]
