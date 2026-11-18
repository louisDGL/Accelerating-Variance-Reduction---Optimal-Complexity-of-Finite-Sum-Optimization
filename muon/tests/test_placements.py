"""Unit tests for the linear-in-the-estimator placements of muon/lab/vrmuon.py (radial, polar metric), the anchor
PolarGrad step, the relaxed memory and the MISO member of the family. Run with
`python -m pytest muon/tests/test_placements.py -q -s` or `python muon/tests/test_placements.py`."""
import copy
import functools
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


def _warm(prob, cfg, iters=37, seed=5):
    W0 = np.zeros((prob.K, prob.d))
    return exact.collect_states(prob, W0, iters, [iters], seed=seed, **cfg)[-1][1:]


def _metric_matrix(prob, metric):
    V, gains, r, g_P, R2P = metric
    return V @ np.diag(gains) @ V.T + r * (np.eye(prob.d) - V @ V.T)


def test_radial_expectation():
    """Radial placement: E_j[S] = gamma_c grad F + gamma <grad F, O_m> O_m (exact enumeration)."""
    for loss in ("softmax", "ls"):
        prob = _prob(loss)
        W, st = _warm(prob, dict(estimator="saga", placement="euclid"))
        G = prob.grad(W)
        algo = vm.VRMuon(prob, estimator="saga", placement="radial", gamma=0.7, gamma_c=0.3)
        S, m, C = exact.enumerate_steps(algo, W, st, return_pieces=True)
        O = maps.polar(m, basis=algo.basis)
        ref = 0.3 * G + 0.7 * float(np.vdot(G, O)) * O
        assert np.linalg.norm(S.mean(0) - ref) <= 1e-12 * np.linalg.norm(ref), loss
        mm, c, _ = algo.pieces(W, st, 9)
        S9, info = algo.step(W, mm, c, st)
        assert np.allclose(S9, 0.3 * (mm + c) + 0.7 * float(np.vdot(mm + c, O)) * O, atol=1e-14)
        assert abs(info["pair"] - float(np.vdot(mm + c, O))) < 1e-13


def test_polar_metric():
    """Metric placement: equal to the nuclear-scaled polar step at the anchor, linear in the estimator elsewhere
    (E_j[S] = gamma_P grad F C), with a certified per-sample smoothness in the metric."""
    for loss in ("softmax", "ls"):
        prob = _prob(loss)
        n = prob.n
        W, st = _warm(prob, dict(estimator="saga", placement="euclid"))
        G = prob.grad(W)
        # fresh memory: m = grad F(W), the step is gamma_P sharp(grad F)
        algo = vm.VRMuon(prob, estimator="saga", placement="metric", c_metric=0.25, metric_floor=1e-12)
        st0 = algo.init_state(W)
        m0, c0, _ = algo.pieces(W, st0, 3)
        assert np.linalg.norm(c0) < 1e-14 and np.allclose(m0, G, atol=1e-14)
        S0, _ = algo.step(W, m0, c0, st0)
        met0 = algo._metric(st0, m0)
        g_P, R2P = met0[3], met0[4]
        assert np.allclose(S0, g_P * maps.sharp(G, basis=algo.basis), atol=1e-12 * np.abs(S0).max()), loss
        assert abs(g_P - 0.25 / R2P) < 1e-18
        Cmat = _metric_matrix(prob, met0)
        ev = np.linalg.eigvalsh(Cmat)
        assert ev[0] > 0
        quad = np.einsum("id,de,ie->i", prob.X, Cmat, prob.X)
        assert abs(R2P - (prob.c_loss * quad.max() + prob.lam * ev[-1])) < 1e-10 * R2P
        # stale memory: linear in the estimator
        algo = vm.VRMuon(prob, estimator="saga", placement="metric", c_metric=0.25)
        st2 = copy.deepcopy(st)
        S = exact.enumerate_steps(algo, W, st2)
        assert st2.metric is None                                 # the enumeration leaves the state untouched
        met2 = algo._metric(st2, algo.pieces(W, st2, 0)[0])
        Cmat = _metric_matrix(prob, met2)
        g_P, R2P = met2[3], met2[4]
        assert np.linalg.norm(S.mean(0) - g_P * G @ Cmat) <= 1e-12 * np.linalg.norm(G @ Cmat), loss
        # per-sample smoothness in the metric: ||(f_i'(A) - f_i'(B)) C^{1/2}||_F <= R2P ||(A - B) C^{-1/2}||_F
        w_, Q_ = np.linalg.eigh(Cmat)
        Ch, Cmh = (Q_ * np.sqrt(w_)) @ Q_.T, (Q_ / np.sqrt(w_)) @ Q_.T
        rng = np.random.default_rng(1)
        for _ in range(20):
            A, B = rng.standard_normal(W.shape), rng.standard_normal(W.shape)
            i = int(rng.integers(n))
            lhs = np.linalg.norm((prob.grad_i(A, i) - prob.grad_i(B, i)) @ Ch)
            assert lhs <= R2P * np.linalg.norm((A - B) @ Cmh) * (1 + 1e-10), (loss, lhs)
        # L-SVRG: the anchor is the exact gradient at the reference point
        algo = vm.VRMuon(prob, estimator="lsvrg", placement="metric", metric_floor=1e-12)
        stl = algo.init_state(W)
        ml, cl, _ = algo.pieces(W, stl, 5)
        Sl, _ = algo.step(W, ml, cl, stl)
        assert np.allclose(Sl, algo._metric(stl, ml)[3] * maps.sharp(G, basis=algo.basis),
                           atol=1e-12 * np.abs(Sl).max())
    # orthogonal planted least squares without noise: the metric at W_0 is a multiple of the inverse covariance
    p = problems.make_matrix_ls(300, 8, 8, kappa=1e4, alpha=2.0, seed=2, noise=0.0, planted="orthogonal")
    optimum.solve_optimum(p)
    algo = vm.VRMuon(p, estimator="lsvrg", placement="metric", metric_floor=1e-12)
    st = algo.init_state(np.zeros((8, 8)))
    V, gains, r, g_P, R2P = algo._metric(st, None)
    ratio = (V @ np.diag(gains) @ V.T) @ p.Sigma_hat
    assert np.allclose(ratio, ratio[0, 0] * np.eye(8), atol=1e-8 * abs(ratio[0, 0]))


def test_covariance_metric():
    """metric_source='cov': the metric is built from the model Hessian c_l Sigma_hat + lam I on the feature side (the
    Newton metric for least squares without floor). It is linear in the estimator, with a certified per-sample
    smoothness, and it is fixed: neither the SAGA period nor an L-SVRG refresh invalidates it."""
    for loss in ("softmax", "ls"):
        prob = _prob(loss)
        d = prob.d
        H = prob.c_loss * prob.Sigma_hat + prob.lam * np.eye(d)
        h = np.linalg.eigvalsh(H)[::-1]
        cov = dict(placement="metric", metric_source="cov", metric_floor=0.0)
        algo = vm.VRMuon(prob, estimator="saga", c_metric=0.25, **cov)
        W, st = _warm(prob, dict(estimator="saga", placement="euclid"))
        G = prob.grad(W)
        st2 = copy.deepcopy(st)
        m, c, _ = algo.pieces(W, st2, 4)
        met = algo._metric(st2, m)
        Cmat = _metric_matrix(prob, met)
        g_P, R2P = met[3], met[4]
        assert np.allclose(Cmat, np.trace(H) * np.linalg.inv(H), rtol=1e-9, atol=0)
        quad = np.einsum("id,de,ie->i", prob.X, Cmat, prob.X)
        assert abs(R2P - (prob.c_loss * quad.max() + prob.lam * np.linalg.eigvalsh(Cmat)[-1])) < 1e-10 * R2P
        assert abs(g_P - 0.25 / R2P) < 1e-18
        S4, _ = algo.step(W, m, c, st2)
        ref = g_P * (m + c) @ Cmat
        assert np.linalg.norm(S4 - ref) <= 1e-11 * np.linalg.norm(ref), loss
        S = exact.enumerate_steps(algo, W, st)
        assert np.linalg.norm(S.mean(0) - g_P * G @ Cmat) <= 1e-11 * np.linalg.norm(G @ Cmat), loss
        # a floor caps the gains
        fl = vm.VRMuon(prob, estimator="saga", placement="metric", metric_source="cov", metric_floor=0.3)
        gains = fl.polar_metric(svd=fl.covariance_factors())[1]
        assert np.allclose(gains, h.sum() / np.maximum(h, 0.3 * h[0]), rtol=1e-10, atol=0)
        # fixed metric: still there after three passes (SAGA period, L-SVRG refreshes)
        for est in ("saga", "lsvrg"):
            Wt, stt = _warm(prob, dict(estimator=est, **cov), iters=3 * prob.n)
            assert stt.metric is not None and stt.metric["source"] == "cov"
            assert np.allclose(stt.metric["svd"][0], h, rtol=1e-10, atol=0)
            assert est == "saga" or stt.refreshes > 0
        # a polar-metric configuration evaluated at a state that holds the covariance metric uses its own anchor
        pol = vm.VRMuon(prob, estimator="saga", placement="metric", metric_floor=0.1)
        assert st2.metric["source"] == "cov"
        assert np.allclose(exact.enumerate_steps(pol, W, st2), exact.enumerate_steps(pol, W, st), rtol=1e-12, atol=0)
        # convergence
        r = vm.vr_muon(prob, np.zeros((prob.K, prob.d)), 300 * prob.n, estimator="saga", **cov)
        assert r["dist2_F"][-1] < 1e-16 * r["dist2_F"][0], (loss, r["dist2_F"][-1] / r["dist2_F"][0])
        assert r["metric_source"] == "cov" and "cov-metric" in r["name"]
    # least squares: in the covariance metric without floor the Hessian is tr(H) Id (Newton metric)
    prob = _prob("ls")
    H = prob.c_loss * prob.Sigma_hat + prob.lam * np.eye(prob.d)
    algo = vm.VRMuon(prob, estimator="saga", placement="metric", metric_source="cov", metric_floor=0.0)
    Cmat = _metric_matrix(prob, algo.polar_metric(svd=algo.covariance_factors()))
    w_, Q_ = np.linalg.eigh(Cmat)
    Ch = (Q_ * np.sqrt(w_)) @ Q_.T
    assert np.allclose(Ch @ H @ Ch, np.trace(H) * np.eye(prob.d), rtol=0, atol=1e-9 * np.trace(H))


def test_state_caches_are_keyed():
    """The caches that `step` keeps in the state (frozen metric, lazy anchor) are keyed by the configuration: a
    second configuration evaluated at the same state gets its own quantities, and the exact enumeration does not
    modify the state."""
    for loss in ("softmax", "ls"):
        prob = _prob(loss)
        # metric: state of a run with c_metric = 0.25, evaluated with c_metric = 1 and with another floor
        cfg = dict(estimator="saga", placement="metric", c_metric=0.25, metric_floor=0.1)
        W, st = _warm(prob, cfg)
        a1 = vm.VRMuon(prob, **cfg)
        a2 = vm.VRMuon(prob, estimator="saga", placement="metric", c_metric=1.0, metric_floor=0.1)
        a3 = vm.VRMuon(prob, estimator="saga", placement="metric", c_metric=0.25, metric_floor=1.0)
        S1, S2, S3 = (exact.enumerate_steps(a, W, st) for a in (a1, a2, a3))
        assert np.allclose(S2, 4.0 * S1, rtol=1e-12, atol=0)
        fresh = copy.deepcopy(st)
        fresh.metric = {"svd": fresh.metric["svd"], "gauged": fresh.metric["gauged"]}   # same anchor, no derived cache
        assert np.allclose(exact.enumerate_steps(a3, W, fresh), S3, rtol=1e-13, atol=0)
        assert np.linalg.norm(S3 - S1) > 1e-3 * np.linalg.norm(S1)                 # another floor, another step
        # a state without metric: the enumeration computes one on a copy and leaves the state as it was
        Wn, stn = _warm(prob, dict(estimator="saga", placement="euclid"))
        assert stn.metric is None
        exact.enumerate_steps(a1, Wn, stn)
        assert stn.metric is None
        # lazy anchor: state of a lazy split run, evaluated with another step and with the tangent placement
        g1, g2 = 0.5 / prob.R2_op, 0.05 / prob.R2_op
        cfg = dict(estimator="lsvrg", placement="split", lazy=True, gamma=g1)
        W, st = _warm(prob, cfg)
        b2 = vm.VRMuon(prob, estimator="lsvrg", placement="split", lazy=True, gamma=g2)
        G_w = st.gbar + prob.lam * st.w
        ref = g2 * maps.sharp(G_w, basis=b2.basis) + b2.gamma_c * (prob.grad(W) - G_w)
        Sb = exact.enumerate_steps(b2, W, st).mean(0)
        assert np.linalg.norm(Sb - ref) <= 1e-12 * np.linalg.norm(ref), loss
        b3 = vm.VRMuon(prob, estimator="lsvrg", placement="tangent", lazy=True, gamma=g2)
        St = exact.enumerate_steps(b3, W, st).mean(0)
        SD = maps.SpectralDerivative(G_w, "sharp", basis=b3.basis)
        ref = g2 * (SD.value + SD(prob.grad(W) - G_w))
        assert np.linalg.norm(St - ref) <= 1e-10 * np.linalg.norm(ref), loss


def test_lazy_tangent_fallback():
    """Lazy tangent placement at a rank-deficient reference gradient (softmax without the gauge reduction): the step
    is the lazy split step T(m_w) + gamma_c (lam (W - w) + c), with one factorization and one counted fallback per
    reference period."""
    prob = _prob("softmax")
    W0 = np.zeros((prob.K, prob.d))
    g, gc = 0.3 / prob.R2_op, 0.2 / prob.R2
    cfg = dict(estimator="lsvrg", gauge=None, gamma=g, gamma_c=gc)
    W, st = _warm(prob, dict(placement="split", lazy=True, **cfg), iters=30)
    assert np.linalg.norm(W - st.w) > 0
    tan = vm.VRMuon(prob, placement="tangent", lazy=True, **cfg)
    spl = vm.VRMuon(prob, placement="split", lazy=True, **cfg)
    for j in (0, 7, 23):
        m, c, _ = tan.pieces(W, st, j)
        S_t, _ = tan.step(W, m, c, copy.deepcopy(st))
        S_s, _ = spl.step(W, m, c, copy.deepcopy(st))
        assert np.linalg.norm(S_t - S_s) <= 1e-13 * np.linalg.norm(S_s)
    assert tan.fallbacks == 3                                       # three fresh states, one fallback each
    st2 = copy.deepcopy(st)
    for j in range(5):                                              # same state: one fallback for the whole period
        m, c, _ = tan.pieces(W, st2, j)
        tan.step(W, m, c, st2)
    assert tan.fallbacks == 4
    r = vm.vr_muon(prob, W0, 200, placement="tangent", lazy=True, seed=1, **cfg)
    assert r["fallbacks"] == r["refreshes"] + 1 and np.isfinite(r["dist2_F"]).all()


def test_argument_validation():
    """Options that a placement would silently ignore are refused; the softmax gauge of W_0 is checked."""
    prob = _prob("softmax")
    bad = [dict(placement="split", orth="soft", eps_innov=1.0),          # eps_innov: joint placement only
           dict(placement="euclid", beta=0.9), dict(placement="euclid", nesterov=True),
           dict(orth="svd", eps=0.1), dict(orth="svd", eps_innov=0.1), dict(orth="soft", eps=0.0),
           dict(placement="radial", orth="ns5"), dict(placement="radial", length="const", lr=1e-3),
           dict(placement="radial", pair_with="mean"), dict(placement="metric", gamma=1.0),
           dict(placement="metric", length="clip", rho=1.0), dict(placement="euclid", orth="ns5"),
           dict(schedule="cosine", n_iters=100), dict(warmup=0.5, n_iters=100),
           dict(schedule="no-such-schedule", n_iters=100),
           dict(estimator="lsvrg", placement="split", lazy=True, length="const", lr=1e-2, schedule="cosine",
                n_iters=100),
           dict(beta=1.0), dict(beta_post=1.0), dict(resync_every=0), dict(placement="metric", metric_every=0),
           dict(metric_source="cov"), dict(placement="metric", metric_source="no-such-source"),
           dict(placement="metric", metric_source="cov", metric_every=5),
           dict(placement="metric", metric_floor=1.5)]
    for kw in bad:
        try:
            vm.VRMuon(prob, **kw)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kw}")
    for kw in [dict(length="const", lr=1e-2, schedule="cosine", n_iters=100), dict(orth="soft", eps_innov=1.0),
               dict(placement="split", orth="soft", eps=0.1), dict(placement="radial", gamma=1.0, gamma_c=0.1),
               dict(placement="metric", c_metric=0.5, metric_floor=0.3)]:
        vm.VRMuon(prob, **kw)
    # gauge: a W_0 whose columns do not sum to zero would stall the mapped placements
    W_bad = 0.3 * np.random.default_rng(0).standard_normal((prob.K, prob.d))
    for kw in (dict(), dict(placement="split"), dict(estimator="lsvrg", placement="split", lazy=True)):
        try:
            vm.vr_muon(prob, W_bad, 10, **kw)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError (gauge) for {kw}")
    for kw in (dict(placement="euclid"), dict(gauge=None), dict(placement="radial")):
        r = vm.vr_muon(prob, W_bad, 300 * prob.n, **kw)
        assert r["dist2_F"][-1] < 1e-18 * r["dist2_F"][0], kw
    r = vm.vr_muon(prob, prob.remove_gauge(W_bad), 300 * prob.n)
    assert r["dist2_F"][-1] < 1e-18 * r["dist2_F"][0]


def test_divergence_is_reported():
    """A run whose iterate or estimator becomes non-finite ends with stopped = 'diverged' instead of raising."""
    for loss in ("softmax", "ls"):
        prob = _prob(loss)
        W0 = np.zeros((prob.K, prob.d))
        for cfg in [dict(estimator="page", gamma=1e4 / prob.R2_op), dict(estimator="saga", gamma=1e4 / prob.R2_op),
                    dict(estimator="saga", placement="euclid", gamma=1e3 / prob.R2),
                    dict(estimator="lsvrg", placement="tangent", gamma=1e4 / prob.R2_op),
                    dict(estimator="saga", placement="metric", c_metric=1e4)]:
            for rec in (None, 1):
                r = vm.vr_muon(prob, W0, 400 * prob.n, record_every=rec, **cfg)
                assert r["stopped"] == "diverged", (loss, cfg, rec, r["stopped"])
    assert np.isnan(maps.polar(np.array([[1.0, np.inf], [0.0, 1.0]]))).all()
    S, O, nuc = maps.sharp(np.array([[1.0, np.nan], [0.0, 1.0]]), return_parts=True)
    assert np.isnan(S).all() and np.isnan(O).all() and np.isnan(nuc)
    assert np.isnan(maps.soft_polar(np.array([[np.inf, 0.0]]), 0.1)).all()


def test_anchor_step():
    """Anchor PolarGrad step: certified decrease of F at the anchor, same accounting as L-SVRG."""
    for loss in ("softmax", "ls"):
        prob = _prob(loss)
        n = prob.n
        W0 = np.zeros((prob.K, prob.d))
        algo = vm.VRMuon(prob, estimator="lsvrg", placement="euclid", anchor_step=1.0)
        sta = algo.init_state(W0)
        A0 = algo.anchor_displacement(sta)
        G0 = prob.grad(W0)
        nuc = polar.nuclear_norm(prob.remove_gauge(G0))
        assert np.allclose(A0, (nuc / prob.L_op) * maps.polar(G0, basis=algo.basis), atol=1e-13)
        dec = -prob.F_diff(W0 - A0, W0)
        assert dec >= nuc ** 2 / (2 * prob.L_op) * (1 - 1e-10), (loss, dec)
        T = 6 * n
        a = vm.vr_muon(prob, W0, T, estimator="lsvrg", placement="euclid", anchor_step=0.0, seed=2)
        b = alg.lsvrg(prob, W0, T, seed=2)
        assert np.allclose(a["dist2_F"], b["dist2_F"], rtol=1e-6)
        c = vm.vr_muon(prob, W0, T, estimator="lsvrg", placement="euclid", anchor_step=1.0, seed=2)
        assert np.array_equal(c["grad_evals"], a["grad_evals"]) and c["refreshes"] == a["refreshes"] > 0
        assert c["dist2_F"][-1] < 1e-2 * c["dist2_F"][0]
        sts = exact.collect_states(prob, W0, T, [T], seed=2, estimator="lsvrg", placement="euclid", anchor_step=1.0)
        assert np.allclose(sts[-1][1], c["W"], atol=1e-13)
    for bad in [dict(estimator="saga", anchor_step=1.0), dict(estimator="sgd", placement="metric"),
                dict(estimator="saga", placement="radial", beta=0.5), dict(estimator="lsvrg", mem_relax=0.5),
                dict(estimator="saga", mem_relax=0.0)]:
        try:
            vm.VRMuon(prob, **bad)
            raise AssertionError(f"expected ValueError for {bad}")
        except ValueError:
            pass


def test_relaxed_memory_and_miso():
    """Relaxed table: the estimator stays unbiased and m_+ = m - lam S + delta c / n. The Euclidean member with
    theta = delta / n and gamma = 1 / lam keeps the predictable part at zero (MISO / dual-free SDCA)."""
    for loss in ("softmax", "ls"):
        prob = _prob(loss)
        n = prob.n
        W0 = np.zeros((prob.K, prob.d))
        algo = vm.VRMuon(prob, estimator="saga", mem_relax=0.3)
        Wr, st = _warm(prob, dict(estimator="saga", mem_relax=0.3))
        acc = np.zeros_like(Wr)
        for j in range(n):
            mm, c, _ = algo.pieces(Wr, st, j)
            acc += mm + c
        assert np.linalg.norm(acc / n - prob.grad(Wr)) < 1e-13
        mm, c, aux = algo.pieces(Wr, st, 4)
        S, info = algo.step(Wr, mm, c, st)
        st3 = copy.deepcopy(st)
        algo.commit(Wr, st3, 4, 1.0, aux, info)
        m_new = st3.zbar + prob.lam * (Wr - S)
        assert np.linalg.norm(m_new - (mm - prob.lam * S + 0.3 * c / n)) < 1e-13
        assert np.linalg.norm(st3.zbar - st3.rtab.T @ prob.X / n) < 1e-13
        assert abs(vm.VRMuon(prob, estimator="saga", mem_relax="sdca").mem_relax
                   - prob.lam * n / (prob.R2 + prob.lam * n)) < 1e-15
        for delta in (1.0, 0.4):
            algo = vm.VRMuon(prob, estimator="saga", placement="euclid", theta=delta / n, gamma=1.0 / prob.lam,
                             mem_relax=delta)
            Wm = W0.copy()
            stm = algo.init_state(Wm)
            rng = np.random.default_rng(0)
            for t in range(30):
                j = int(rng.integers(n))
                mm, c, aux = algo.pieces(Wm, stm, j)
                S, info = algo.step(Wm, mm, c, stm, t)
                algo.commit(Wm, stm, j, 1.0, aux, info)
                Wm = algo.advance(Wm, S, stm)
                assert np.linalg.norm(stm.zbar + prob.lam * Wm) < 1e-12 * max(1.0, np.linalg.norm(stm.zbar))
        r = vm.vr_muon(prob, W0, 60 * n, estimator="saga", placement="euclid", theta=1.0 / n, gamma=1.0 / prob.lam)
        assert r["dist2_F"][-1] < 1e-20 * r["dist2_F"][0]                 # kappa = 30 < n = 60: MISO converges


def test_new_placements_converge():
    for loss in ("softmax", "ls"):
        prob = _prob(loss)
        W0 = np.zeros((prob.K, prob.d))
        for cfg, tol in [(dict(estimator="saga", placement="radial"), 1e-20),
                         (dict(estimator="saga", placement="metric"), 1e-6),
                         (dict(estimator="lsvrg", placement="metric"), 1e-3),
                         (dict(estimator="saga", mem_relax="sdca"), 1e-20),
                         (dict(estimator="lsvrg", placement="euclid", anchor_step=1.0), 1e-12)]:
            r = vm.vr_muon(prob, W0, 60 * prob.n, **cfg)
            assert r["dist2_F"][-1] < tol * r["dist2_F"][0], (loss, cfg, r["dist2_F"][-1] / r["dist2_F"][0])


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
