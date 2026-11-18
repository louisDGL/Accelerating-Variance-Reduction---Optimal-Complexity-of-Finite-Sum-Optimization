"""Unit tests for muon/lab/mvr.py (momentum-based variance reduction for Muon: Nesterov / MARS-M / STORM family).
Run with `python -m pytest muon/tests/test_mvr.py -q -s` or `python muon/tests/test_mvr.py`."""
import functools
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))                       # muon/  (package `lab`)
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))      # project root (package `vrlab`, read-only)

from lab import problems, optimum, polar, algorithms as alg, mvr  # noqa: E402


@functools.lru_cache(maxsize=None)
def _prob(loss, n=100, d=8, K=4, kappa=20.0, alpha=1.0, seed=0):
    mk = problems.make_softmax if loss == "softmax" else problems.make_matrix_ls
    prob = mk(n, d, K, kappa=kappa, alpha=alpha, seed=seed)
    optimum.solve_optimum(prob)
    return prob


def test_storm_identity():
    """g = 1 with the same batch is STORM: M_t = G_t + beta (M_{t-1} - Gt_{t-1}) (algorithms.storm_muon, a = 1 - beta)."""
    for loss in ("softmax", "ls"):
        prob = _prob(loss)
        W0 = np.zeros((prob.K, prob.d))
        a = alg.storm_muon(prob, W0, 400, batch=10, lr=0.02, a=0.2, seed=3, record_every=1)
        b = mvr.mvr_muon(prob, W0, 400, batch=10, lr=0.02, beta=0.8, g=1.0, same_batch=True, seed=3, gauge=False,
                         record_every=1)
        assert np.array_equal(a["grad_evals"], b["grad_evals"])
        # the two recursions are algebraically identical; they differ by rounding, which ns5 amplifies on the softmax
        # gauge direction (the gradient is rank deficient there and ns5 has slope 3.4445^5 at 0), so the agreement is
        # tight on the first iterations and for least squares, and only approximate later for softmax
        assert np.allclose(a["dist2_F"][:40], b["dist2_F"][:40], rtol=1e-10), loss
        assert np.allclose(a["dist2_F"], b["dist2_F"], rtol=1e-10 if loss == "ls" else 1e-2), loss
        assert b["grad_evals"][-1] == 10 + 2 * 10 * 399


def test_family_members_and_accounting():
    prob = _prob("softmax")
    W0 = np.zeros((prob.K, prob.d))
    d0 = float(np.vdot(prob.W_star, prob.W_star))
    runs = {
        "heavy ball (g=0)": mvr.mvr_muon(prob, W0, 600, batch=10, lr=0.02, g=0.0, same_batch=False),
        "nesterov (g=1-beta, previous batch)": mvr.mvr_muon(prob, W0, 600, batch=10, lr=0.02, g=0.05,
                                                            same_batch=False),
        "mars-m (g=0.025, same batch, clip 1)": mvr.mvr_muon(prob, W0, 600, batch=10, lr=0.02, g=0.025, clip=1.0),
        "storm cosine svd": mvr.mvr_muon(prob, W0, 600, batch=10, lr=0.05, g=1.0, orth="svd", schedule="cosine"),
        "sharp mvr": mvr.mvr_muon(prob, W0, 600, batch=10, length="sharp", gamma=1.0 / prob.L_op, g=1.0, orth="svd"),
    }
    for name, r in runs.items():
        assert np.all(np.isfinite(r["dist2_F"])) and r["dist2_F"][-1] < 0.1 * d0, (name, r["dist2_F"][-1])
        per = 2 if r["same_batch"] else 1
        assert r["grad_evals"][-1] == 10 + per * 10 * 599, name
    r = mvr.mvr_muon(prob, W0, 10 ** 6, batch=10, lr=0.02, g=1.0, max_grad_evals=3000)
    assert 3000 <= r["grad_evals"][-1] < 3020
    # the previous-batch correction with g = 1 - beta gives Nesterov's direction after the transient of the buffer
    # initialization: compare the buffers of the two recursions directly on a common gradient sequence
    rng = np.random.default_rng(0)
    beta = 0.9
    Gs = [rng.standard_normal((3, 5)) for _ in range(200)]
    B = np.zeros((3, 5))
    M = None
    for t, G in enumerate(Gs):
        B = beta * B + (1 - beta) * G                         # EMA of the gradients
        U = (1 - beta) * G + beta * B                         # Nesterov direction of Muon
        M = G.copy() if M is None else beta * M + (1 - beta) * G + beta * (1 - beta) * (G - Gs[t - 1])
    assert np.linalg.norm(M - U) < 1e-8 * np.linalg.norm(U)
    for bad in [dict(length="const"), dict(length="sharp"), dict(length="other", lr=0.1)]:
        try:
            mvr.mvr_muon(prob, W0, 5, batch=4, **bad)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_mvr_floor_on_finite_sum():
    """With constant parameters the MVR buffer keeps a noise floor on a finite sum, even with the nuclear-scaled step
    and the exact polar factor, whereas the SAGA estimator with the same step rule converges to rounding."""
    from lab import vrmuon as vm
    prob = _prob("ls")
    n = prob.n
    W0 = np.zeros((prob.K, prob.d))
    d0 = float(np.vdot(prob.W_star, prob.W_star))
    r = mvr.mvr_muon(prob, W0, 60 * n, batch=1, length="sharp", gamma=0.25 / prob.R2_op, g=1.0, beta=0.9, orth="svd")
    tail = r["dist2_F"][len(r["dist2_F"]) // 2:] / d0
    s = vm.vr_muon(prob, W0, 60 * n, estimator="saga")
    print(f"\n  LS n=100: STORM buffer + sharp step: dist2/dist2_0 over the last 30 passes in [{tail.min():.1e}, "
          f"{tail.max():.1e}]; SAGA estimator + sharp step: {s['dist2_F'][-1] / d0:.1e}")
    assert tail.min() > 1e-12 and s["dist2_F"][-1] / d0 < 1e-20


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
