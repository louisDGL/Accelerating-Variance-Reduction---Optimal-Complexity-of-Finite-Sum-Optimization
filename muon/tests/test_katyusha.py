"""Unit tests for muon/lab/katyusha.py (loopless Katyusha on matrix finite sums, with the spectral y step).
Run with `python -m pytest muon/tests/test_katyusha.py -q -s` or `python muon/tests/test_katyusha.py`."""
import functools
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))                       # muon/  (package `lab`)
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))      # project root (package `vrlab`, read-only)

from lab import problems, optimum, maps, katyusha as kat, vrmuon as vm  # noqa: E402


@functools.lru_cache(maxsize=None)
def _prob(loss, n=100, d=8, K=4, kappa=2000.0, alpha=1.0, seed=0):
    mk = problems.make_softmax if loss == "softmax" else problems.make_matrix_ls
    prob = mk(n, d, K, kappa=kappa, alpha=alpha, seed=seed)
    optimum.solve_optimum(prob)
    return prob


def _literal(prob, W0, n_iters, seed, ystep="coupling", gamma_y=None):
    """Line-by-line transcription of the recursion with dense per-sample gradients (prob.grad_i)."""
    n = prob.n
    L, mu = prob.R2, prob.mu
    sigma = mu / L
    theta2 = 0.5
    theta1 = min(np.sqrt(2 * sigma * n / 3), 0.5)
    eta = theta2 / ((1 + theta2) * theta1)
    p = 1.0 / n
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=n_iters)
    coins = rng.random(n_iters)
    y, z, w = W0.copy(), W0.copy(), W0.copy()
    gw = prob.grad(w)
    Q = maps.gauge_basis(prob)
    for j, c in zip(idx, coins):
        x = theta1 * z + theta2 * w + (1 - theta1 - theta2) * y
        g = prob.grad_i(x, j) - prob.grad_i(w, j) + gw
        z_new = (eta * sigma * x + z - (eta / L) * g) / (1 + eta * sigma)
        y_old = y
        if ystep == "coupling":
            y = x + theta1 * (z_new - z)
        elif ystep == "sharp":
            y = x - gamma_y * maps.sharp(g, basis=Q)
        else:
            y = x - gamma_y * g
        z = z_new
        if c < p:
            w = y_old.copy()
            gw = prob.grad(w)
    return y


def test_matches_literal_recursion():
    for loss in ("softmax", "ls"):
        prob = _prob(loss)
        W0 = np.zeros((prob.K, prob.d))
        for ystep, gy in [("coupling", None), ("euclid", 1.0 / (3 * prob.R2)), ("sharp", 1.0 / (9 * prob.R2_op))]:
            T = 60 if ystep == "sharp" else 500               # the sharp step is sensitive to rounding (see test_vrmuon)
            r = kat.lkatyusha(prob, W0, T, ystep=ystep, seed=4)
            y = _literal(prob, W0, T, 4, ystep, gy)
            assert np.linalg.norm(r["W"] - y) <= 1e-9 * np.linalg.norm(y), (loss, ystep)
            assert r["grad_evals"][-1] == prob.n + 2 * T + prob.n * r["refreshes"]
        prm = kat.lkatyusha_params(prob.n, prob.R2, prob.mu)
        r = kat.lkatyusha(prob, W0, 10)
        assert r["theta1"] == prm["theta1"] and r["eta"] == prm["eta"] and r["p"] == 1.0 / prob.n
        assert abs(prm["theta1"] - min(np.sqrt(2 * prob.mu / prob.R2 * prob.n / 3), 0.5)) < 1e-15


def test_converges_and_accelerates():
    """On an ill-conditioned instance (kappa = 2000 >> n = 100) L-Katyusha needs fewer gradients than L-SVRG, and the
    three y steps converge."""
    for loss in ("softmax", "ls"):
        prob = _prob(loss)
        n = prob.n
        W0 = np.zeros((prob.K, prob.d))
        d0 = float(np.vdot(prob.W_star, prob.W_star))
        budget = 600 * n
        out = {}
        for ystep in ("coupling", "euclid", "sharp"):
            r = kat.lkatyusha(prob, W0, 10 ** 9, ystep=ystep, max_grad_evals=budget, stop_dist2=1e-16 * d0)
            out[ystep] = (r["grad_evals"][-1] / n, r["dist2_F"][-1] / d0)
            assert r["dist2_F"][-1] <= 1e-8 * d0, (loss, ystep, r["dist2_F"][-1] / d0)
        s = vm.vr_muon(prob, W0, 10 ** 9, estimator="lsvrg", placement="euclid", max_grad_evals=budget,
                       stop_dist2=1e-16 * d0)
        print(f"\n  {loss}: passes to 1e-16 (or final relative dist2): L-SVRG {s['grad_evals'][-1] / n:.0f} "
              f"({s['dist2_F'][-1] / d0:.1e}); L-Katyusha " +
              ", ".join(f"{k} {v[0]:.0f} ({v[1]:.1e})" for k, v in out.items()))
    r = kat.lkatyusha(prob, W0, 10 ** 9, max_grad_evals=5 * n)
    assert 5 * n <= r["grad_evals"][-1] < 6 * n + 2
    for bad in [dict(ystep="other"), dict(theta1=0.8, theta2=0.5)]:
        try:
            kat.lkatyusha(prob, W0, 5, **bad)
            raise AssertionError("expected ValueError")
        except ValueError:
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
