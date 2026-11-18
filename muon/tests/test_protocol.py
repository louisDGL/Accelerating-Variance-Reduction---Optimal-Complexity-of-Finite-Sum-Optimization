"""Unit tests for the experiment protocol (muon/experiments/common.py), the importance-sampling option of
vrmuon.py and the full-batch baselines added for the head-to-head comparison (accelerated gradient, L-BFGS).
Run with `python -m pytest muon/tests/test_protocol.py -q -s` or `python muon/tests/test_protocol.py`."""
import functools
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_MUON = os.path.dirname(_HERE)
sys.path.insert(0, _MUON)                                        # muon/  (package `lab`)
sys.path.insert(0, os.path.dirname(_MUON))                       # project root (package `vrlab`, read-only)
sys.path.insert(0, os.path.join(_MUON, "experiments"))           # common.py, families.py

from lab import problems, optimum, algorithms as alg, vrmuon as vm  # noqa: E402
import common as C  # noqa: E402
import families as F  # noqa: E402


@functools.lru_cache(maxsize=None)
def _prob(loss, n=60, d=6, K=4, kappa=30.0, alpha=1.0, seed=3):
    mk = problems.make_softmax if loss == "softmax" else problems.make_matrix_ls
    prob = mk(n, d, K, kappa=kappa, alpha=alpha, seed=seed)
    optimum.solve_optimum(prob)
    return prob


def test_importance_sampling():
    """p_i ~ L_i^a with the innovation weighted by 1 / (n p_i): the estimator stays unbiased, the table is updated
    with the unweighted gradient, the indices follow the distribution, and the methods converge."""
    for loss in ("softmax", "ls"):
        prob = _prob(loss)
        n = prob.n
        W0 = np.zeros((prob.K, prob.d))
        for est in ("saga", "lsvrg"):
            for a in (0.5, 1.0):
                algo = vm.VRMuon(prob, estimator=est, placement="euclid", sampling=a)
                assert abs(algo.probs.sum() - 1.0) < 1e-14 and np.allclose(algo.probs * algo.iw, 1.0 / n)
                assert np.allclose(algo.probs / algo.probs[0], (prob.L_i / prob.L_i[0]) ** a)
                # a warm state reached with the importance-sampled run itself
                W = W0.copy()
                st = algo.init_state(W)
                rng = np.random.default_rng(1)
                for t, (j, coin) in enumerate(vm._draws(rng, n, 1, 45, algo.needs_coin, algo.probs)):
                    m, c, aux = algo.pieces(W, st, int(j), coin)
                    S, info = algo.step(W, m, c, st, t)
                    algo.commit(W, st, int(j), coin, aux, info)
                    W = algo.advance(W, S, st)
                G = prob.grad(W)
                acc = np.zeros_like(W)
                for j in range(n):
                    m, c, _ = algo.pieces(W, st, j)
                    acc += algo.probs[j] * (m + c)
                assert np.linalg.norm(acc - G) <= 1e-13 * max(1.0, np.linalg.norm(G)), (loss, est, a)
                if est == "saga":                                  # the table mean stays the mean of the table
                    assert np.linalg.norm(st.zbar - st.rtab.T @ prob.X / n) < 1e-13
        r = vm.vr_muon(prob, W0, 80 * n, estimator="saga", placement="euclid", sampling=0.5, seed=2)
        assert r["dist2_F"][-1] < 1e-18 * r["dist2_F"][0], loss
        r = vm.vr_muon(prob, W0, 80 * n, estimator="saga", sampling=0.5, seed=2)       # inside the polar step
        assert r["dist2_F"][-1] < 1e-18 * r["dist2_F"][0], loss
        assert r["sampling"] == 0.5
    prob = _prob("ls")
    algo = vm.VRMuon(prob, estimator="saga", placement="euclid", sampling=1.0)
    idx = np.array([j for j, _ in vm._draws(np.random.default_rng(0), prob.n, 1, 200000, False, algo.probs)])
    freq = np.bincount(idx, minlength=prob.n) / len(idx)
    assert np.abs(freq - algo.probs).max() < 5 * np.sqrt(algo.probs.max() / len(idx))
    for bad in (dict(estimator="saga", batch=2, sampling=0.5), dict(estimator="page", sampling=0.5),
                dict(estimator="saga", sampling=-1.0)):
        try:
            vm.VRMuon(prob, **bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")
    assert vm.VRMuon(prob, estimator="saga", sampling="uniform").probs is None


def test_full_batch_baselines():
    """Accelerated gradient and L-BFGS converge to W_* and are charged one pass per gradient evaluation."""
    for loss in ("softmax", "ls"):
        prob = _prob(loss)
        n = prob.n
        W0 = np.zeros((prob.K, prob.d))
        D0 = float(np.vdot(prob.W_star, prob.W_star))
        r = alg.agd(prob, W0, 400)
        assert r["dist2_F"][-1] < 1e-18 * D0 and r["grad_evals"][-1] == 400 * n and 0 < r["beta"] < 1
        g = alg.gd(prob, W0, 400)
        k = 60
        assert r["dist2_F"][k] < g["dist2_F"][k]                         # acceleration is visible
        r = alg.agd(prob, W0, 400, stop_dist2=1e-10 * D0)
        assert r["stopped"] == "stop_dist2" and r["dist2_F"][-1] <= 1e-10 * D0
        for mem in (5, 20):
            r = alg.lbfgs(prob, W0, 200, memory=mem)
            assert r["dist2_F"][-1] < 1e-12 * D0, (loss, mem, r["dist2_F"][-1] / D0)   # limited by the precision of F
            assert r["evals"] <= 200 and r["grad_evals"][-1] <= r["evals"] * n
            assert np.all(np.diff(r["grad_evals"]) >= n)                 # at least one evaluation per iteration
        r = alg.lbfgs(prob, W0, 200, stop_dist2=1e-8 * D0)
        assert r["stopped"] == "stop_dist2" and r["dist2_F"][-1] <= 1e-8 * D0 and r["evals"] < 200
        assert np.allclose(np.vdot(r["W"] - prob.W_star, r["W"] - prob.W_star), r["dist2_F"][-1])
        r = alg.lbfgs(prob, W0, 7)
        assert r["evals"] <= 8 and r["stopped"] is None
        # the experiment runner charges them in passes
        s = C.spec(loss, 60, 6, 4, 30.0, 1.0, seed=3)
        out = C.run_one({"spec": s, "method": "lbfgs", "cfg": {"memory": 10}, "passes": 60, "seed": 0,
                         "stop_rel": 1e-12})
        assert out["stopped"] == "stop_dist2" and out["passes_used"] <= 60 and math.isfinite(out["passes_to"]["1e-09"])
        out = C.run_one({"spec": s, "method": "agd", "cfg": {"L": C.times("L_F", 1.0), "mu": C.times("lam", 1.0)},
                         "passes": 400, "seed": 0, "stop_rel": 1e-12})
        assert out["stopped"] == "stop_dist2" and math.isfinite(out["passes_to"]["1e-09"])


def test_protocol_helpers():
    grid = {"a": [1.0, 2.0, 4.0], "b": [C.over("R2", 8.0), C.over("R2", 4.0)]}
    pts = C.grid_points(grid)
    assert len(pts) == 6 and pts[0] == {"a": 1.0, "b": C.over("R2", 8.0)} and pts[1]["b"] == C.over("R2", 4.0)
    # neighbours in the product order (last parameter fastest)
    assert sorted(C._neighbours(grid, 0)) == [1, 2] and sorted(C._neighbours(grid, 3)) == [1, 2, 5]
    assert C._neighbours({}, 0) == [] and sorted(C._neighbours({"a": [1, 2, 3]}, 1)) == [0, 2]
    # edge flags are by value, whatever the order of the list
    assert C._edge(pts, 0, grid) == ["a-", "b+"] and C._edge(pts, 5, grid) == ["a+", "b-"] and C._edge(pts, 2, grid) == ["b+"]
    g2 = {"miso_delta": [1.0, 0.5, "sdca"], "theta": [0.6, 0.3, 0.1]}
    p2 = C.grid_points(g2)
    assert C._edge(p2, 0, g2) == ["theta+"] and C._edge(p2, 8, g2) == ["theta-"]
    # scores
    res = {"diverged": False, "final_rel": 1e-3, "min_rel": 1e-3, "passes_to": {"1e-06": float("inf")}}
    assert abs(C.score(res, 1e-6, 100) - 400.0) < 1e-9
    assert C.score({"diverged": True, "final_rel": 1.0, "min_rel": 1.0, "passes_to": {}}, 1e-6, 100) == float("inf")
    assert abs(C.geo_mean([2.0, 8.0]) - 4.0) < 1e-12 and C.geo_mean([1.0, float("inf")]) == float("inf")
    # MISO expansion, including the dual-free-SDCA fraction
    prob = _prob("ls")
    sd = prob.lam * prob.n / (prob.R2 + prob.lam * prob.n)
    for val, ref in ((0.25, 0.25), ("sdca", sd), ("sdca/2", sd / 2), ("2sdca", min(1.0, 2 * sd))):
        cfg = C.resolve({"estimator": "saga", "placement": "euclid", "miso_delta": val}, prob)
        assert abs(cfg["mem_relax"] - ref) < 1e-15 and abs(cfg["theta"] - ref / prob.n) < 1e-15
        assert cfg["gamma"] == 1.0 / prob.lam and "miso_delta" not in cfg
    assert len(C.code_fingerprint()) == 40
    assert C.fmt_point({"miso_delta": "sdca", "gamma": C.over("R2", 0.5), "theta": 0.1}) == \
        "miso_delta=sdca, gamma=0.5/R2, theta=0.1"


def test_catalogue_is_valid():
    """Every grid point of every family is accepted by its method (a ValueError would be cached as a failed run)."""
    prob = _prob("softmax")
    names = set()
    fams = F.catalogue(prob.n) + F.minibatch_families(4) + F.streaming_families(8)
    for fam in fams:
        assert fam["name"] not in names and fam["group"] in ("euclid", "full", "spectral", "stream")
        assert fam["status"] in ("sota", "lab", "new")
        names.add(fam["name"])
        for pt in C.grid_points(fam["grid"]):
            cfg = dict(fam["base"])
            cfg.update(pt)
            cfg = C.resolve(cfg, prob)
            if fam["method"] == "vr":
                sched = cfg.get("schedule", "const") != "const"
                vm.VRMuon(prob, n_iters=100 if sched else None, **cfg)
            for v in cfg.values():
                assert not isinstance(v, dict), (fam["name"], cfg)
    assert {"SAGA", "B-SAGA", "IS-B-SAGA", "soft SAGA-Muon", "AGD", "L-BFGS", "PolarGrad"} <= names


def test_tune_and_confirm_small():
    """End-to-end protocol on a tiny instance (no cache file is kept): screening, neighbours of the best point,
    refinement on more seeds, confirmation on fresh seeds."""
    s = C.spec("ls", 60, 6, 4, 30.0, 1.0, seed=3)
    fams = [dict(name="SAGA", method="vr", base=dict(estimator="saga", placement="euclid"),
                 grid=dict(gamma=[C.over("R2", 2.0 ** k) for k in range(-6, 4)])),
            dict(name="GD", method="gd", base={}, grid=dict(step=[C.over("L_F", c) for c in (0.5, 1.0, 1.9)]))]
    name = "_test_protocol_tmp"
    old = os.environ.get("MUON_JOBS")
    os.environ["MUON_JOBS"] = "1"
    try:
        recs = C.tune_and_confirm([s], fams, 40, 1e-6, name, confirm_seeds=(10, 11, 12))
    finally:
        if old is None:
            os.environ.pop("MUON_JOBS", None)
        else:
            os.environ["MUON_JOBS"] = old
        for stage in ("screen", "tune", "confirm"):
            path = os.path.join(C.RESULTS, f"{name}_{stage}.pkl")
            if os.path.exists(path):
                os.remove(path)
    saga, gd = recs
    assert saga["refined"] and len(saga["finalists"]) == 3 and saga["summary"]["1e-06"]["runs"] == 3
    assert saga["summary"]["1e-06"]["reached"] == 3 and saga["stop_level"]["reached"] == 3
    best = saga["best"]["gamma"]["c"]
    kept = [t for t in saga["tune"] if t["kept"]]
    assert 4 <= len(kept) < 10 and any(t.get("score_refined") is not None for t in kept)
    # the selected point is a local minimum of the 2-seed score among its evaluated grid neighbours, and its
    # neighbours were evaluated unless they diverged in the screening
    cs = [t["point"]["gamma"]["c"] for t in saga["tune"]]
    i = cs.index(best)
    top2 = min(kept, key=lambda t: t["score"])["point"]["gamma"]["c"]
    for j in (cs.index(top2) - 1, cs.index(top2) + 1):
        if 0 <= j < len(cs):
            assert saga["tune"][j]["kept"] or not math.isfinite(saga["tune"][j]["screen_rel"])
    assert 0.125 <= best <= 8.0 and i >= 0
    assert not gd["refined"] and gd["summary"]["1e-06"]["runs"] == 1 and gd["n_kept"] == 3


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
