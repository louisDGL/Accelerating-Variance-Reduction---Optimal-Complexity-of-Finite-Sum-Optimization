"""Unit tests for muon/lab/maps.py (spectral maps and their Frechet derivatives). Run with
`python -m pytest muon/tests/test_maps.py -q` or `python muon/tests/test_maps.py`."""
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))                       # muon/  (package `lab`)
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))      # project root (package `vrlab`, read-only)

from lab import maps, polar, problems, optimum  # noqa: E402

SHAPES = [(4, 9), (9, 4), (6, 6), (1, 5), (5, 1)]


def _rand_sv(rng, shape, cond=30.0):
    """Random matrix with singular values geometrically spread over [1/cond, 1] times a random scale."""
    K, d = shape
    r = min(K, d)
    U, _ = np.linalg.qr(rng.standard_normal((K, r)))
    V, _ = np.linalg.qr(rng.standard_normal((d, r)))
    s = np.geomspace(1.0, 1.0 / cond, r) * rng.uniform(0.3, 3.0)
    return (U * s) @ V.T


def _fd(fun, M, E, h):
    """Fourth-order central finite difference of t -> fun(M + t E) at t = 0."""
    return (-fun(M + 2 * h * E) + 8 * fun(M + h * E) - 8 * fun(M - h * E) + fun(M - 2 * h * E)) / (12 * h)


def test_maps_values():
    rng = np.random.default_rng(0)
    for shape in SHAPES:
        M = _rand_sv(rng, shape)
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        assert np.allclose(maps.polar(M), U @ Vt, atol=1e-13)
        assert np.allclose(maps.polar(M), polar.polar_svd(M), atol=1e-14)
        S, O, nuc = maps.sharp(M, return_parts=True)
        assert abs(nuc - s.sum()) < 1e-13 * s.sum() and np.allclose(S, s.sum() * (U @ Vt), atol=1e-12)
        assert abs(np.vdot(M, S) - s.sum() ** 2) < 1e-12 * s.sum() ** 2          # <M, sharp(M)> = ||M||_*^2
        for eps in (1e-3, 0.5, 40.0):
            ref = (U * (s / np.sqrt(s ** 2 + eps ** 2))) @ Vt
            assert np.allclose(maps.soft_polar(M, eps), ref, atol=1e-13)
            if shape[0] <= shape[1]:                                              # M (M^T M + eps^2)^{-1/2}
                w, Q = np.linalg.eigh(M @ M.T + eps ** 2 * np.eye(shape[0]))
                assert np.allclose(maps.soft_polar(M, eps), (Q / np.sqrt(w)) @ Q.T @ M, atol=1e-12)
        assert np.allclose(maps.soft_polar(M, 1e-9), U @ Vt, atol=1e-7)           # eps -> 0: polar
        assert np.allclose(maps.soft_polar(M, 1e9) * 1e9, M, atol=1e-7)           # eps -> inf: M / eps
        assert np.allclose(maps.power_map(M, 1.0), M, atol=1e-13)
        assert np.allclose(maps.power_map(M, 0.0), U @ Vt, atol=1e-13)
        assert np.allclose(maps.power_map(M, 0.5), (U * np.sqrt(s)) @ Vt, atol=1e-13)
    # rank-one matrices: sharp is the identity, polar is the normalized outer product
    a, b = rng.standard_normal(5), rng.standard_normal(8)
    M = np.outer(a, b)
    assert np.allclose(maps.sharp(M), M, atol=1e-13)
    assert np.allclose(maps.polar(M), M / (np.linalg.norm(a) * np.linalg.norm(b)), atol=1e-13)


def test_frechet_finite_differences():
    rng = np.random.default_rng(1)
    worst = {}
    for shape in SHAPES:
        for trial in range(4):
            M = _rand_sv(rng, shape, cond=10.0 ** rng.uniform(0.3, 2.0))
            smin = np.linalg.svd(M, compute_uv=False)[-1]
            E = rng.standard_normal(shape)
            E /= np.linalg.norm(E)
            h = 1e-3 * smin
            cases = [
                ("polar", maps.polar, lambda E_: maps.polar_frechet(M, E_)),
                ("sharp", maps.sharp, lambda E_: maps.sharp_frechet(M, E_)),
            ]
            for eps in (0.05 * smin, 2.0 * smin, 30.0):
                cases.append((f"soft", lambda X, e=eps: maps.soft_polar(X, e),
                              lambda E_, e=eps: maps.soft_polar_frechet(M, E_, e)))
            cases.append(("power", lambda X: maps.power_map(X, 0.3),
                          maps.SpectralDerivative(M, "custom", f=lambda s: s ** 0.3, df=lambda s: 0.3 * s ** -0.7)))
            for name, fun, dfun in cases:
                num = _fd(fun, M, E, h)
                ana = dfun(E)
                err = np.linalg.norm(num - ana) / max(np.linalg.norm(ana), 1e-300)
                worst[name] = max(worst.get(name, 0.0), err)
                assert err < 1e-6, (name, shape, err)
    print("\n  Frechet derivatives vs 4th-order finite differences, worst relative errors: "
          + ", ".join(f"{k} {v:.1e}" for k, v in worst.items()))


def test_frechet_structure():
    """D polar and D sharp: linearity, known special cases, D sharp symmetric PSD (Hessian of ||.||_*^2 / 2)."""
    rng = np.random.default_rng(2)
    for shape in [(4, 9), (9, 4), (6, 6)]:
        M = _rand_sv(rng, shape)
        Dp = maps.SpectralDerivative(M, "polar")
        Ds = maps.SpectralDerivative(M, "sharp")
        assert np.allclose(Dp.value, maps.polar(M), atol=1e-13) and np.allclose(Ds.value, maps.sharp(M), atol=1e-12)
        assert np.allclose(Dp.polar, maps.polar(M), atol=1e-13)
        E1, E2 = rng.standard_normal(shape), rng.standard_normal(shape)
        assert np.allclose(Dp(2.0 * E1 - 3.0 * E2), 2.0 * Dp(E1) - 3.0 * Dp(E2), atol=1e-11)
        # polar is degree-0 homogeneous: D polar(M)[M] = 0; sharp is degree-1 homogeneous: D sharp(M)[M] = sharp(M)
        assert np.linalg.norm(Dp(M)) < 1e-12
        assert np.allclose(Ds(M), maps.sharp(M), atol=1e-11)
        # symmetry and positive semidefiniteness of D sharp as an operator on matrices
        assert abs(np.vdot(E1, Ds(E2)) - np.vdot(E2, Ds(E1))) < 1e-10 * (1 + abs(np.vdot(E1, Ds(E2))))
        K, d = shape
        H = np.empty((K * d, K * d))
        for c in range(K * d):
            B = np.zeros(K * d)
            B[c] = 1.0
            H[:, c] = Ds(B.reshape(K, d)).ravel()
        assert np.allclose(H, H.T, atol=1e-10)
        ev = np.linalg.eigvalsh(0.5 * (H + H.T))
        assert ev[0] > -1e-10 * ev[-1]
        # operator norm of D polar is of order 1 / s_min
        s = np.linalg.svd(M, compute_uv=False)
        Hp = np.empty((K * d, K * d))
        for c in range(K * d):
            B = np.zeros(K * d)
            B[c] = 1.0
            Hp[:, c] = Dp(B.reshape(K, d)).ravel()
        gain = np.linalg.norm(Hp, 2)
        if K != d:
            assert abs(gain - 1.0 / s[-1]) < 1e-8 / s[-1]               # rectangular: exactly 1 / s_min
        else:
            assert abs(gain - 2.0 / (s[-1] + s[-2])) < 1e-8 / s[-1]      # square: 2 / (s_r + s_{r-1})
    # rank deficiency is refused by polar / sharp and accepted by the bounded-gain map
    M = rng.standard_normal((5, 2)) @ rng.standard_normal((2, 8))
    for fn in (maps.polar_frechet, maps.sharp_frechet):
        try:
            fn(M, rng.standard_normal((5, 8)))
            raise AssertionError("expected LinAlgError")
        except np.linalg.LinAlgError:
            pass
    E = rng.standard_normal((5, 8))
    num = _fd(lambda X: maps.soft_polar(X, 0.7), M, E, 1e-4)
    assert np.linalg.norm(num - maps.soft_polar_frechet(M, E, 0.7)) < 1e-7 * np.linalg.norm(num)


def test_frechet_accuracy_corner_cases():
    """Two rounding corner cases of the derivative: (a) square matrix with a tiny singular value: the gains are
    2 / (s_i + s_j), the complement term (exactly zero) must not bring a 1 / s_min error; (b) bounded-gain map with
    nearly equal singular values: the divided differences are evaluated without cancellation."""
    from decimal import Decimal, getcontext
    getcontext().prec = 60
    rng = np.random.default_rng(7)
    U, _ = np.linalg.qr(rng.standard_normal((5, 5)))
    V, _ = np.linalg.qr(rng.standard_normal((5, 5)))
    for s_min in (1e-4, 1e-8, 1e-10):
        s = np.array([1.0, 0.8, 0.6, 0.4, s_min])
        M = (U * s) @ V.T
        E = rng.standard_normal((5, 5))
        A = U.T @ E @ V
        ref = U @ ((A - A.T) / (s[:, None] + s[None, :])) @ V.T          # built from the known factors
        err = np.linalg.norm(maps.polar_frechet(M, E) - ref) / np.linalg.norm(ref)
        assert err < 1e-12, (s_min, err)
        ref_sharp = float(np.trace(A)) * (U @ V.T) + s.sum() * ref
        err = np.linalg.norm(maps.sharp_frechet(M, E) - ref_sharp) / np.linalg.norm(ref_sharp)
        assert err < 1e-12, (s_min, err)

    def dd(a, b, eps):                                                    # divided difference in 60-digit arithmetic
        a, b, e = Decimal(a), Decimal(b), Decimal(eps)
        fa, fb = a / (a * a + e * e).sqrt(), b / (b * b + e * e).sqrt()
        return float((fa - fb) / (a - b))

    worst = 0.0
    for eps in (1e-6, 1e-3, 1.0):
        for delta in (2e-9, 1e-7, 1e-4, 1e-2):
            s = np.array([1.0, 1.0 - delta, 0.3])
            SD = maps.SpectralDerivative(np.diag(s), "soft", eps=eps)
            order = np.argsort(-SD.s)
            assert np.array_equal(order, [0, 1, 2])
            for i, j in ((0, 1), (0, 2), (1, 2)):
                ref = dd(float(SD.s[i]), float(SD.s[j]), eps)
                rel = abs(SD.phi_sym[i, j] - ref) / abs(ref)
                worst = max(worst, rel)
                assert rel < 1e-12, (eps, delta, i, j, rel)
            for i in range(3):                                            # diagonal: f'(s) = eps^2 / (s^2 + eps^2)^(3/2)
                a = float(SD.s[i])
                assert abs(SD.phi_sym[i, i] - eps ** 2 / (a * a + eps ** 2) ** 1.5) <= 1e-13 * SD.phi_sym[i, i]
    SD0 = maps.SpectralDerivative(np.zeros((3, 4)), "soft", eps=0.5)       # M = 0: f'(0) = 1 / eps everywhere
    assert np.allclose(SD0.phi_sym, 2.0) and np.allclose(SD0(np.ones((3, 4))), 2.0 * np.ones((3, 4)))
    print(f"\n  soft divided differences vs 60-digit reference: worst relative error {worst:.1e}")


def test_gauge_basis_softmax():
    """Softmax gradients have rank <= K - 1; with the gauge basis the maps and derivatives are those of Q^T M."""
    prob = problems.make_softmax(200, 10, 4, kappa=100.0, alpha=1.0, seed=2)
    optimum.solve_optimum(prob)
    Q = maps.gauge_basis(prob)
    K = prob.K
    assert Q.shape == (K, K - 1) and np.allclose(Q.T @ Q, np.eye(K - 1), atol=1e-14)
    assert np.abs(np.ones(K) @ Q).max() < 1e-14
    assert maps.gauge_basis(problems.make_matrix_ls(50, 6, 3, kappa=10.0)) is None
    rng = np.random.default_rng(3)
    W = 0.3 * rng.standard_normal((K, prob.d))
    W -= W.mean(axis=0, keepdims=True)
    G = prob.grad(W)
    assert np.abs(np.ones(K) @ G).max() < 1e-14
    assert np.allclose(maps.polar(G, basis=Q), polar.polar_svd(prob.remove_gauge(G)), atol=1e-12)
    S, O, nuc = maps.sharp(G, basis=Q, return_parts=True)
    assert abs(nuc - polar.nuclear_norm(G)) < 1e-12 * nuc
    j = 7
    E = prob.grad_i(W, j) - prob.lam * W                       # loss part of a per-sample gradient: rank one, in the gauge
    assert np.abs(np.ones(K) @ E).max() < 1e-14
    h = 1e-3 * np.linalg.svd(Q.T @ G, compute_uv=False)[-1]
    for name, fun, dfun in [("polar", lambda X: maps.polar(X, basis=Q), lambda E_: maps.polar_frechet(G, E_, basis=Q)),
                            ("sharp", lambda X: maps.sharp(X, basis=Q), lambda E_: maps.sharp_frechet(G, E_, basis=Q)),
                            ("soft", lambda X: maps.soft_polar(X, 0.01, basis=Q),
                             lambda E_: maps.soft_polar_frechet(G, E_, 0.01, basis=Q))]:
        num, ana = _fd(fun, G, E, h), dfun(E)
        assert np.linalg.norm(num - ana) < 1e-6 * max(np.linalg.norm(ana), 1e-300), name
        assert np.abs(np.ones(K) @ ana).max() < 1e-12 * max(1.0, np.abs(ana).max())     # stays in the gauge
    # without the basis the full matrix is rank deficient and the derivative is refused
    try:
        maps.polar_frechet(G, E)
        raise AssertionError("expected LinAlgError")
    except np.linalg.LinAlgError:
        pass


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items(), key=lambda kv: kv[1].__code__.co_firstlineno
                                        if callable(kv[1]) and hasattr(kv[1], "__code__") else 0)
             if k.startswith("test_") and callable(v)]
    failed = []
    for name, fn in tests:
        t0 = time.perf_counter()
        try:
            fn()
            print(f"PASS {name} ({time.perf_counter() - t0:.1f}s)")
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            import traceback
            traceback.print_exc()
            print(f"FAIL {name}: {exc!r}")
    print(f"\n{len(tests) - len(failed)} passed, {len(failed)} failed")
    sys.exit(1 if failed else 0)
