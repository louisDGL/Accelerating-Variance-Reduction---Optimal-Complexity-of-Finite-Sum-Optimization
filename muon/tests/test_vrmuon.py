"""Unit tests for muon/lab/vrmuon.py (generalized variance-reduced Muon) and muon/lab/exact.py (exact one-step
conditional expectations). Run with `python -m pytest muon/tests/test_vrmuon.py -q -s` or
`python muon/tests/test_vrmuon.py`."""
import copy
import functools
import itertools
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))                       # muon/  (package `lab`)
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))      # project root (package `vrlab`, read-only)

from lab import problems, optimum, polar, maps, exact, algorithms as alg, vrmuon as vm  # noqa: E402


@functools.lru_cache(maxsize=None)
def _prob(loss, n=60, d=6, K=4, kappa=30.0, alpha=1.0, seed=3):
    mk = problems.make_softmax if loss == "softmax" else problems.make_matrix_ls
    prob = mk(n, d, K, kappa=kappa, alpha=alpha, seed=seed)
    optimum.solve_optimum(prob)
    return prob


def _warm_state(prob, cfg, iters=37, seed=5):
    """A generic state (iterate and memory) reached after a few iterations of the configuration `cfg`."""
    W0 = np.zeros((prob.K, prob.d))
    t, W, st = exact.collect_states(prob, W0, iters, [iters], seed=seed, **cfg)[-1]
    return W, st


def _state_arrays(st):
    return {k: np.array(v, copy=True) for k, v in vars(st).items() if isinstance(v, np.ndarray)}


# ----------------------------------------------------------------------------------------------
# the generalized loop reproduces the existing methods
# ----------------------------------------------------------------------------------------------
def test_reproduces_existing_methods():
    for loss in ("ls", "softmax"):
        prob = _prob(loss, n=200, d=10, K=4, kappa=50.0, seed=1)
        W0 = np.zeros((prob.K, prob.d))
        T = 20 * prob.n

        def close(a, b, rtol, head=None):
            assert np.array_equal(a["grad_evals"], b["grad_evals"]) and np.array_equal(a["iters"], b["iters"])
            x, y = a["dist2_F"], b["dist2_F"]
            live = x > 1e-20 * x[0]
            if head is not None:
                live &= np.arange(len(x)) <= head
            assert np.allclose(x[live], y[live], rtol=rtol, atol=0.0), (loss, np.abs(x[live] / y[live] - 1).max())

        # Euclidean methods and smooth orthogonalization (ns5): the two implementations differ only by rounding
        close(alg.saga(prob, W0, T), vm.vr_muon(prob, W0, T, estimator="saga", placement="euclid"), 1e-4)
        a, b = alg.lsvrg(prob, W0, T), vm.vr_muon(prob, W0, T, estimator="lsvrg", placement="euclid")
        close(a, b, 1e-4)
        assert a["refreshes"] == b["refreshes"]
        close(alg.saga_muon(prob, W0, T, orth="ns5"),
              vm.vr_muon(prob, W0, T, estimator="saga", orth="ns5", gauge=None), 1e-2)
        # exact polar factor: the trajectories are sensitive to rounding (see test_polar_sensitivity), so only the first
        # iterations are compared tightly
        for kw_old, kw_new in [
            (dict(), dict()),
            (dict(rule="const", lr=0.01), dict(length="const", lr=0.01)),
            (dict(beta=0.5), dict(beta=0.5)),
            (dict(beta=0.5, nesterov=True), dict(beta=0.5, nesterov=True)),
        ]:
            a = alg.saga_muon(prob, W0, 40, record_every=1, **kw_old)
            b = vm.vr_muon(prob, W0, 40, estimator="saga", gauge=None, record_every=1, **kw_new)
            close(a, b, 1e-9)
            if "rule" not in kw_old:
                assert np.allclose(a["eta"][1:], b["eta"][1:], rtol=1e-8)
        a = alg.lsvrg_muon(prob, W0, 40, record_every=1)
        b = vm.vr_muon(prob, W0, 40, estimator="lsvrg", gauge=None, record_every=1)
        close(a, b, 1e-9)
        # default steps
        assert vm.VRMuon(prob).gamma == 1.0 / (4 * prob.R2_op)
        assert vm.VRMuon(prob, estimator="lsvrg").gamma == 1.0 / (6 * prob.R2_op)
        assert vm.VRMuon(prob, placement="euclid").gamma == 1.0 / (4 * prob.R2)


def test_polar_sensitivity():
    """With the exact polar factor, a perturbation of W_0 of size 1e-14 is amplified by many orders of magnitude within
    a few passes (the polar map is discontinuous where a singular value vanishes), whereas Euclidean SAGA keeps it at
    the rounding level. Individual trajectories of the exact-polar methods are therefore not reproducible to rounding;
    only their statistics are."""
    prob = _prob("ls", n=200, d=10, K=4, kappa=50.0, seed=1)
    W0 = np.zeros((prob.K, prob.d))
    P = 1e-14 * np.random.default_rng(0).standard_normal(W0.shape)
    T = 3000
    out = {}
    for name, cfg in [("saga", dict(estimator="saga", placement="euclid")), ("saga_muon", dict(estimator="saga"))]:
        a = vm.vr_muon(prob, W0, T, record_every=T, **cfg)
        b = vm.vr_muon(prob, W0 + P, T, record_every=T, **cfg)
        out[name] = np.linalg.norm(a["W"] - b["W"]) / np.sqrt(a["dist2_F"][-1])
    print(f"\n  relative separation after {T} iterations of two runs started 1e-14 apart: "
          f"SAGA {out['saga']:.1e}, SAGA-Muon (exact polar, sharp) {out['saga_muon']:.1e}")
    assert out["saga"] < 1e-8 and out["saga_muon"] > 1e-6


# ----------------------------------------------------------------------------------------------
# estimators: exact enumeration
# ----------------------------------------------------------------------------------------------
def test_estimators_unbiased():
    for loss in ("softmax", "ls"):
        prob = _prob(loss)
        n = prob.n
        for est in ("saga", "lsvrg", "sgd"):
            W, st = _warm_state(prob, dict(estimator=est, placement="euclid"))
            algo = vm.VRMuon(prob, estimator=est, placement="euclid")
            G = prob.grad(W)
            acc = np.zeros_like(W)
            for j in range(n):
                m, c, _ = algo.pieces(W, st, j)
                acc += m + c
            assert np.linalg.norm(acc / n - G) <= 1e-13 * max(1.0, np.linalg.norm(G)), (loss, est)
        # minibatches: all subsets of size 2 of a tiny problem
        small = _prob(loss, n=12, d=3, K=3, kappa=20.0, alpha=0.0, seed=4)
        for est in ("saga", "lsvrg", "sgd"):
            W, st = _warm_state(small, dict(estimator=est, placement="euclid", batch=2), iters=9)
            algo = vm.VRMuon(small, estimator=est, placement="euclid", batch=2)
            G = small.grad(W)
            acc, cnt = np.zeros_like(W), 0
            for B in itertools.combinations(range(small.n), 2):
                m, c, _ = algo.pieces(W, st, np.array(B))
                acc += m + c
                cnt += 1
            assert np.linalg.norm(acc / cnt - G) <= 1e-13 * max(1.0, np.linalg.norm(G)), (loss, est, "b=2")
        # b = n: the SAGA and L-SVRG estimators are the full gradient
        for est in ("saga", "lsvrg"):
            W, st = _warm_state(small, dict(estimator=est, placement="euclid", batch=small.n), iters=3)
            algo = vm.VRMuon(small, estimator=est, placement="euclid", batch=small.n)
            m, c, _ = algo.pieces(W, st, np.arange(small.n))
            assert np.linalg.norm(m + c - small.grad(W)) < 1e-13


def test_sag_limit_and_memory():
    """theta = 1/n with the SAGA memory is the SAG direction: the mean of the table AFTER the update of entry j."""
    prob = _prob("softmax")
    n = prob.n
    W, st = _warm_state(prob, dict(estimator="saga", placement="euclid"))
    algo = vm.VRMuon(prob, estimator="saga", placement="euclid", theta=1.0 / n)
    for j in (0, 17, n - 1):
        m, c, aux = algo.pieces(W, st, j)
        st2 = copy.deepcopy(st)
        S, info = algo.step(W, m, c, st2)
        algo.commit(W, st2, j, 1.0, aux, info)
        sag = st2.rtab.T @ prob.X / n + prob.lam * W
        assert np.linalg.norm(m + c / n - sag) < 1e-14
        assert np.allclose(S, algo.gamma * sag, atol=1e-15)
        assert np.linalg.norm(st2.zbar - st2.rtab.T @ prob.X / n) < 1e-14       # incremental mean stays exact
        assert np.allclose(st2.rtab[j], prob.loss_residuals(W)[j], atol=1e-15)


def test_step_is_pure():
    prob = _prob("ls")
    for cfg in [dict(estimator="saga"), dict(estimator="saga", placement="split"),
                dict(estimator="saga", placement="tangent"), dict(estimator="lsvrg", placement="split", lazy=True),
                dict(estimator="lsvrg", placement="tangent", lazy=True), dict(estimator="page"),
                dict(estimator="saga", beta=0.5), dict(estimator="saga", beta_post=0.5),
                dict(estimator="saga", placement="radial"), dict(estimator="saga", placement="metric"),
                dict(estimator="lsvrg", placement="metric")]:
        W, st = _warm_state(prob, cfg, iters=23)
        algo = vm.VRMuon(prob, **cfg)
        if cfg.get("lazy") or cfg.get("placement") == "metric":                 # caches filled on demand by `step`:
            st.anchor = None                                                    # start from an unfilled state
            st.metric = None
        before = _state_arrays(st)
        W_before = W.copy()
        m, c, aux = algo.pieces(W, st, 7)
        S1, _ = algo.step(W, m, c, st)
        S2, _ = algo.step(W, m, c, st)
        assert np.array_equal(S1, S2) and np.array_equal(W, W_before), cfg
        after = _state_arrays(st)
        assert before.keys() == after.keys() and all(np.array_equal(before[k], after[k]) for k in before), cfg


# ----------------------------------------------------------------------------------------------
# placements: exact conditional expectations of the step
# ----------------------------------------------------------------------------------------------
def test_split_expectation():
    """E_j[S_j] = T(m) + gamma_c theta (grad F(W) - m) for the split placement (exact enumeration)."""
    for loss in ("softmax", "ls"):
        prob = _prob(loss)
        W, st = _warm_state(prob, dict(estimator="saga", placement="euclid"))
        G = prob.grad(W)
        for theta in (1.0, 0.3):
            for length, kw in [("sharp", {}), ("clip", dict(rho=1e-3)), ("const", dict(lr=0.01))]:
                algo = vm.VRMuon(prob, estimator="saga", placement="split", theta=theta, length=length,
                                 gamma=0.7, gamma_c=0.4, **kw)
                S, m, C = exact.enumerate_steps(algo, W, st, return_pieces=True)
                T_m, eta, pair = algo._map(m, 0)
                ref = T_m + 0.4 * theta * (G - m)
                assert np.linalg.norm(S.mean(axis=0) - ref) <= 1e-12 * np.linalg.norm(ref), (loss, theta, length)
                nuc = polar.nuclear_norm(m)
                want = {"sharp": 0.7 * nuc, "clip": min(1e-3, 0.7 * nuc), "const": 0.01}[length]
                assert abs(eta - want) <= 1e-12 * want and abs(polar.spectral_norm(T_m) - want) <= 1e-10 * want


def test_tangent_expectation_and_order():
    """Tangent placement: E_j[S_j] = T(m) + DT(m)[grad F(W) - m] exactly, and this equals T(grad F(W)) up to second
    order in the staleness grad F(W) - m (the split and joint placements are only first-order accurate)."""
    for loss in ("softmax", "ls"):
        prob = _prob(loss)
        n = prob.n
        W, _ = _warm_state(prob, dict(estimator="saga", placement="euclid"))
        G = prob.grad(W)
        rng = np.random.default_rng(8)
        Dir = rng.standard_normal(W.shape)
        Dir -= Dir.mean(axis=0, keepdims=True)
        errs = {"tangent": [], "split": []}
        for h in (1e-2, 1e-3):
            Wm = W + h * Dir                                         # memory taken at a point at distance h from W
            for pl in ("tangent", "split"):
                algo = vm.VRMuon(prob, estimator="saga", placement=pl, gamma=0.5, gamma_c=0.5)
                st = algo.init_state(Wm)
                S, m, C = exact.enumerate_steps(algo, W, st, return_pieces=True)
                assert np.linalg.norm(C.mean(axis=0) - (G - m)) < 1e-13
                T_G = 0.5 * maps.sharp(G, basis=algo.basis)
                if pl == "tangent":
                    SD = maps.SpectralDerivative(m, "sharp", basis=algo.basis)
                    ref = 0.5 * (SD.value + SD(G - m))
                    assert np.linalg.norm(S.mean(axis=0) - ref) <= 1e-10 * np.linalg.norm(ref), loss
                    assert algo.fallbacks == 0
                errs[pl].append(np.linalg.norm(S.mean(axis=0) - T_G) / np.linalg.norm(T_G))
        assert errs["tangent"][1] < 0.02 * errs["tangent"][0], (loss, errs)          # second order: about 1/100
        assert errs["split"][1] > 0.05 * errs["split"][0], (loss, errs)              # first order: about 1/10
        assert errs["tangent"][1] < 1e-2 * errs["split"][1]


def test_lazy_and_page():
    for loss in ("softmax", "ls"):
        prob = _prob(loss)
        n = prob.n
        # lazy split: E_j[S_j] = T(grad F(w)) + gamma_c (grad F(W) - grad F(w))
        cfg = dict(estimator="lsvrg", placement="split", lazy=True, gamma=0.6, gamma_c=0.3, p=0.02)
        W, st = _warm_state(prob, cfg, iters=45)
        assert st.refreshes >= 0 and np.linalg.norm(W - st.w) > 0
        algo = vm.VRMuon(prob, **cfg)
        S = exact.enumerate_steps(algo, W, st)
        Gw = prob.grad(st.w)
        ref = 0.6 * maps.sharp(Gw, basis=algo.basis) + 0.3 * (prob.grad(W) - Gw)
        assert np.linalg.norm(S.mean(axis=0) - ref) <= 1e-11 * np.linalg.norm(ref), loss
        # lazy tangent: T(grad F(w)) + DT(grad F(w))[grad F(W) - grad F(w)]
        cfg = dict(estimator="lsvrg", placement="tangent", lazy=True, gamma=0.6, p=0.02)
        algo = vm.VRMuon(prob, **cfg)                              # same state: the anchor cache is keyed
        S = exact.enumerate_steps(algo, W, st)
        SD = maps.SpectralDerivative(Gw, "sharp", basis=algo.basis)
        ref = 0.6 * (SD.value + SD(prob.grad(W) - Gw))
        assert np.linalg.norm(S.mean(axis=0) - ref) <= 1e-10 * np.linalg.norm(ref), loss
        # PAGE: the recursion telescopes, v_t = v_0 + sum of the innovations; with p = 1 it is the exact gradient
        W0 = np.zeros((prob.K, prob.d))
        algo = vm.VRMuon(prob, estimator="page", placement="euclid", p=0.0)
        Wt = W0.copy()
        st = algo.init_state(Wt)
        rng = np.random.default_rng(3)
        Ws, total = [Wt.copy()], np.zeros_like(Wt)
        js = rng.integers(0, n, 25)
        for t, j in enumerate(js):
            m, c, aux = algo.pieces(Wt, st, int(j), 1.0)
            S, info = algo.step(Wt, m, c, st, t)
            cost = algo.commit(Wt, st, int(j), 1.0, aux, info)
            assert cost == (0 if t == 0 else 2)
            if t > 0:                                           # innovation = f_j'(W_t) - f_j'(W_{t-1}) (loss parts)
                d_ref = prob.grad_i(Ws[-1], int(j)) - prob.grad_i(Ws[-2], int(j)) - prob.lam * (Ws[-1] - Ws[-2])
                assert np.allclose(c, d_ref, atol=1e-13)
                total += c
            Wt = Wt - S
            Ws.append(Wt.copy())
        assert np.allclose(st.v, prob.grad(W0) - prob.lam * W0 + total, atol=1e-13)
        a = vm.vr_muon(prob, W0, 30, estimator="page", p=1.0, record_every=1)
        b = alg.muon_full(prob, W0, 30, rule="sharp", L=6 * prob.R2_op, record_every=1,
                          orth=(lambda M: polar.polar_svd(prob.remove_gauge(M))))
        assert np.allclose(a["dist2_F"], b["dist2_F"], rtol=1e-8)
        assert a["grad_evals"][-1] == n + 29 * n                 # the first iteration reuses the initial gradient
    # batched accurate residual differences
    prob = _prob("softmax")
    rng = np.random.default_rng(4)
    Z = rng.standard_normal((5, prob.K))
    for scale in (1e-1, 1e-9):
        Dl = scale * rng.standard_normal((5, prob.K))
        rows = vm.residual_diff_rows(prob, Z, Dl)
        for i in range(5):
            assert np.allclose(rows[i], prob.residual_diff_i(Z[i], Dl[i], 0), rtol=1e-12, atol=1e-300)


def test_full_batch_limit():
    """With b = n the SAGA estimator is the full gradient: the joint sharp method is deterministic PolarGrad."""
    for loss in ("softmax", "ls"):
        prob = _prob(loss)
        W0 = np.zeros((prob.K, prob.d))
        a = vm.vr_muon(prob, W0, 25, estimator="saga", batch=prob.n, record_every=1)
        b = alg.muon_full(prob, W0, 25, rule="sharp", L=4 * prob.R2_op, record_every=1,
                          orth=(lambda M: polar.polar_svd(prob.remove_gauge(M))))
        assert np.allclose(a["dist2_F"], b["dist2_F"], rtol=1e-7), loss
        assert a["grad_evals"][-1] == prob.n + 25 * prob.n


# ----------------------------------------------------------------------------------------------
# step maps
# ----------------------------------------------------------------------------------------------
def test_step_maps():
    prob = _prob("ls")
    W, st = _warm_state(prob, dict(estimator="saga", placement="euclid"))
    j = 11
    base = vm.VRMuon(prob, estimator="saga")
    m, c, _ = base.pieces(W, st, j)
    omega = m + c
    nuc = polar.nuclear_norm(omega)
    O = polar.polar_svd(omega)
    # sharp, const, clip (both regimes)
    S, info = base.step(W, m, c, st)
    assert np.allclose(S, base.gamma * nuc * O, atol=1e-14) and abs(info["pair"] - nuc) < 1e-12 * nuc
    S, info = vm.VRMuon(prob, estimator="saga", length="const", lr=0.03).step(W, m, c, st)
    assert np.allclose(S, 0.03 * O, atol=1e-14)
    for rho in (1e-6, 1e6):
        S, info = vm.VRMuon(prob, estimator="saga", length="clip", rho=rho).step(W, m, c, st)
        assert np.allclose(S, min(rho, base.gamma * nuc) * O, atol=1e-14)
    # pairing with the predictable part
    S, info = vm.VRMuon(prob, estimator="saga", pair_with="mean").step(W, m, c, st)
    assert np.allclose(S, base.gamma * max(np.vdot(m, O), 0.0) * O, atol=1e-14)
    # damping
    S, _ = vm.VRMuon(prob, estimator="saga", theta=0.2).step(W, m, c, st)
    assert np.allclose(S, base.gamma * maps.sharp(m + 0.2 * c), atol=1e-13)
    # soft orthogonalization: absolute eps, eps -> 0 (polar) and eps -> infinity (Euclidean direction)
    S, _ = vm.VRMuon(prob, estimator="saga", orth="soft", eps=0.37, length="const", lr=0.03).step(W, m, c, st)
    assert np.allclose(S, 0.03 * maps.soft_polar(omega, 0.37), atol=1e-14)
    S, _ = vm.VRMuon(prob, estimator="saga", orth="soft", eps=1e-12).step(W, m, c, st)
    assert np.allclose(S, base.gamma * nuc * O, rtol=1e-6, atol=1e-12)
    S, _ = vm.VRMuon(prob, estimator="saga", orth="soft", eps=1e9, length="const", lr=1e9 * 0.5).step(W, m, c, st)
    assert np.allclose(S, 0.5 * omega, rtol=1e-6, atol=1e-12)
    # eps relative to the innovation of the step
    S, _ = vm.VRMuon(prob, estimator="saga", orth="soft", eps_innov=2.0).step(W, m, c, st)
    Os = maps.soft_polar(omega, 2.0 * np.linalg.norm(c))
    assert np.allclose(S, base.gamma * np.vdot(omega, Os) * Os, atol=1e-13)
    # momentum before the map: the direction is the polar factor of the buffer, the length pairs with omega
    algo = vm.VRMuon(prob, estimator="saga", beta=0.6)
    st.buf = np.random.default_rng(1).standard_normal(W.shape)
    S, info = algo.step(W, m, c, st)
    Ob = polar.polar_svd(0.6 * st.buf + omega)
    assert np.allclose(S, algo.gamma * np.vdot(omega, Ob) * Ob, atol=1e-13)
    assert np.allclose(info["buf"], 0.6 * st.buf + omega)
    # EMA after the map
    algo = vm.VRMuon(prob, estimator="saga", beta_post=0.8)
    st.Sbar = np.random.default_rng(2).standard_normal(W.shape)
    S, info = algo.step(W, m, c, st)
    assert np.allclose(S, 0.8 * st.Sbar + 0.2 * base.gamma * nuc * O, atol=1e-13)
    # invalid combinations
    for bad in [dict(placement="split", beta=0.5), dict(length="const"), dict(length="clip"),
                dict(orth="soft"), dict(orth="soft", eps=1.0, eps_innov=1.0), dict(lazy=True),
                dict(placement="tangent", orth="ns5"), dict(estimator="svrg")]:
        try:
            vm.VRMuon(prob, **bad)
            raise AssertionError(f"expected ValueError for {bad}")
        except ValueError:
            pass


def test_accounting_and_stops():
    prob = _prob("softmax")
    n = prob.n
    W0 = np.zeros((prob.K, prob.d))
    T = 3 * n
    r = vm.vr_muon(prob, W0, T, estimator="saga")
    assert r["grad_evals"][-1] == n + T and r["zbar_err"] < 1e-13
    r = vm.vr_muon(prob, W0, T, estimator="saga", batch=5)
    assert r["grad_evals"][-1] == n + 5 * T
    r = vm.vr_muon(prob, W0, T, estimator="lsvrg", batch=3, placement="euclid")
    assert r["grad_evals"][-1] == n + 6 * T + n * r["refreshes"] and r["refreshes"] > 0
    r = vm.vr_muon(prob, W0, T, estimator="page", batch=2, placement="euclid")
    assert r["grad_evals"][-1] == n + 4 * (T - 1 - r["refreshes"]) + n * r["refreshes"]
    r = vm.vr_muon(prob, W0, T, estimator="sgd", batch=4, placement="euclid", gamma=0.1)
    assert r["grad_evals"][-1] == 4 * T and r["grad_evals"][0] == 0
    r = vm.vr_muon(prob, W0, 10 ** 6, estimator="saga", max_grad_evals=5 * n)
    assert r["grad_evals"][-1] == 5 * n and r["iters"][-1] == 4 * n
    r = vm.vr_muon(prob, W0, 10 ** 6, estimator="saga", placement="euclid", stop_dist2=1e-6)
    assert r["stopped"] == "stop_dist2" and r["dist2_F"][-1] <= 1e-6
    assert exact.expected_cost(vm.VRMuon(prob, estimator="lsvrg")) == 3.0
    assert exact.expected_cost(vm.VRMuon(prob, estimator="saga", batch=4)) == 4.0


# ----------------------------------------------------------------------------------------------
# exact one-step statistics
# ----------------------------------------------------------------------------------------------
def test_exact_stats():
    for loss in ("softmax", "ls"):
        prob = _prob(loss)
        n = prob.n
        W, st = _warm_state(prob, dict(estimator="saga", placement="euclid"))
        E = W - prob.W_star
        G = prob.grad(W)
        # Euclidean SAGA: closed forms. E[S] = gamma grad F; dec_dist2 = 2 gamma <G, E> - gamma^2 E||omega||^2
        algo = vm.VRMuon(prob, estimator="saga", placement="euclid")
        s = exact.one_step_stats(prob, algo, W, st)
        S = exact.enumerate_steps(algo, W, st)
        g = algo.gamma
        om2 = np.mean([np.linalg.norm(S[j] / g) ** 2 for j in range(n)])
        assert abs(s["dec_dist2"] - (2 * g * np.vdot(G, E) - g * g * om2)) <= 1e-12 * abs(s["dec_dist2"])
        assert abs(s["cos_grad"] - 1.0) < 1e-12 and abs(s["bias_F"]) < 1e-12 and s["cost"] == 1.0
        assert abs(s["lin"] - g * np.vdot(G, G)) <= 1e-12 * s["lin"]
        # dec_F against the plain evaluation of F
        direct = np.mean([prob.F(W) - prob.F(W - S[j]) for j in range(n)])
        assert abs(s["dec_F"] - direct) <= 1e-9 * abs(direct) + 1e-15
        assert abs(s["eff_F"] - s["dec_F"] / prob.Fgap(W)) < 1e-12 * abs(s["eff_F"])
        # damped estimator: bias_F = (1 - theta) * staleness
        algo = vm.VRMuon(prob, estimator="saga", placement="euclid", theta=0.25)
        s2 = exact.one_step_stats(prob, algo, W, st)
        assert abs(s2["bias_F"] - 0.75 * s2["stale_F"]) < 1e-12 and s2["stale_F"] > 0
        # mapped configurations run and give finite numbers; the joint sharp step is biased
        for cfg in [dict(), dict(placement="split"), dict(placement="tangent"), dict(theta=0.1),
                    dict(pair_with="mean"), dict(length="clip", rho=1e-2), dict(orth="soft", eps_innov=1.0)]:
            s3 = exact.one_step_stats(prob, vm.VRMuon(prob, estimator="saga", **cfg), W, st)
            assert all(np.isfinite(v) for v in s3.values()), (cfg, s3)
            assert 1.0 - 1e-9 <= s3["nrank_G"] <= prob.r + 1e-9 and -1 - 1e-9 <= s3["dual_align"] <= 1 + 1e-9
        ref = exact.reference_steps(prob, W, basis=maps.gauge_basis(prob))
        for k in ("gd", "sharp", "gd_ls", "sharp_ls"):
            assert ref[k]["rel_dec_F"] > 0 and ref[k]["eff_F"] == ref[k]["rel_dec_F"] / n
        if loss == "ls":                                        # exact line search beats the fixed steps on a quadratic
            assert ref["gd_ls"]["rel_dec_F"] >= ref["gd"]["rel_dec_F"] * (1 - 1e-12)
            assert ref["sharp_ls"]["rel_dec_F"] >= ref["sharp"]["rel_dec_F"] * (1 - 1e-12)
    # collect_states returns independent copies at the requested times
    prob = _prob("ls")
    W0 = np.zeros((prob.K, prob.d))
    states = exact.collect_states(prob, W0, 50, [0, 10, 50], seed=1, estimator="saga", placement="euclid")
    assert [s[0] for s in states] == [0, 10, 50]
    r = vm.vr_muon(prob, W0, 50, seed=1, estimator="saga", placement="euclid", record_every=10)
    assert np.allclose(np.vdot(states[1][1] - prob.W_star, states[1][1] - prob.W_star), r["dist2_F"][1], rtol=1e-12)
    assert not np.shares_memory(states[1][2].rtab, states[2][2].rtab)


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items(), key=lambda kv: kv[1].__code__.co_firstlineno
                                        if callable(kv[1]) and hasattr(kv[1], "__code__") else 0)
             if k.startswith("test_") and callable(v)]
    failed = []
    t_all = time.perf_counter()
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
    print(f"\n{len(tests) - len(failed)} passed, {len(failed)} failed in {time.perf_counter() - t_all:.1f}s")
    sys.exit(1 if failed else 0)
