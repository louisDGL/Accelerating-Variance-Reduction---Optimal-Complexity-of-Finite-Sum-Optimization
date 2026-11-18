"""Numerical checks of the structural facts of muon/notes_theory.md (F1-F11) that are statements about the methods
rather than about the code: inequalities and identities verified by exact enumeration of the sampled index on small
problems. The identities that define the code paths are tested in test_maps.py, test_vrmuon.py and test_placements.py.
Run with `python -m pytest muon/tests/test_facts.py -q -s` or `python muon/tests/test_facts.py`."""
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

from lab import problems, optimum, maps, exact, vrmuon as vm  # noqa: E402


@functools.lru_cache(maxsize=None)
def _prob(loss, n=60, d=6, K=4, kappa=30.0, alpha=1.0, seed=3):
    mk = problems.make_softmax if loss == "softmax" else problems.make_matrix_ls
    prob = mk(n, d, K, kappa=kappa, alpha=alpha, seed=seed)
    optimum.solve_optimum(prob)
    return prob


def _warm(prob, cfg, iters=37, seed=5):
    W0 = np.zeros((prob.K, prob.d))
    return exact.collect_states(prob, W0, iters, [iters], seed=seed, **cfg)[-1][1:]


def _nuc(M):
    return float(np.linalg.svd(M, compute_uv=False).sum())


def _op(M):
    return float(np.linalg.svd(M, compute_uv=False)[0])


def _fro2(M):
    return float(np.vdot(M, M))


STATES = [dict(estimator="saga", placement="euclid"), dict(estimator="saga"),
          dict(estimator="lsvrg", placement="euclid")]


def test_F1_dual_norm_error_at_batch_one():
    """F1 (ii): E||c_j||_*^2 = E||c_j||_F^2 and E||omega_j - grad F||_*^2 <= 4 E||omega_j - grad F||_F^2
    <= 4 E||c_j||_F^2 for single-index SAGA / L-SVRG estimators. F1 (i): with the regularizer inside the estimator,
    the memoryless single-index estimator is rank one only at W = 0."""
    worst = 0.0
    for loss in ("softmax", "ls"):
        prob = _prob(loss)
        for cfg in STATES:
            for iters in (5, 37, 150):
                W, st = _warm(prob, cfg, iters=iters)
                algo = vm.VRMuon(prob, **cfg)
                _, m, C = exact.enumerate_steps(algo, W, st, return_pieces=True)
                G = prob.grad(W)
                nuc2_c = np.mean([_nuc(c) ** 2 for c in C])
                fro2_c = np.mean([_fro2(c) for c in C])
                assert abs(nuc2_c - fro2_c) <= 1e-12 * fro2_c                      # rank-one innovations
                err_nuc2 = np.mean([_nuc(m + c - G) ** 2 for c in C])
                err_fro2 = np.mean([_fro2(m + c - G) for c in C])
                assert err_nuc2 <= 4.0 * err_fro2 * (1 + 1e-12), (loss, cfg, iters)
                assert err_fro2 <= fro2_c * (1 + 1e-12), (loss, cfg, iters)
                worst = max(worst, err_nuc2 / err_fro2)
        # (i) memoryless estimator: m = lam W is part of the orthogonalized matrix
        g = 0.1 / prob.R2
        sgd_j = vm.VRMuon(prob, estimator="sgd", gamma=g)
        sgd_e = vm.VRMuon(prob, estimator="sgd", placement="euclid", gamma=g)
        for W, same in ((np.zeros((prob.K, prob.d)), True), (_warm(prob, STATES[0])[0], False)):
            st = sgd_j.init_state(W)
            m, c, _ = sgd_j.pieces(W, st, 11)
            Sj, _ = sgd_j.step(W, m, c, st)
            Se, _ = sgd_e.step(W, m, c, st)
            rel = np.linalg.norm(Sj - Se) / np.linalg.norm(Se)
            assert (rel < 1e-13) if same else (rel > 1e-2), (loss, same, rel)
    print(f"\n  F1(ii): largest ratio E||err||_*^2 / E||err||_F^2 = {worst:.2f} (bound 4)")


def test_F2_fixed_length_floor():
    """F2: with a step of fixed spectral length eta, max(||W_t - W_*||_op, ||W_{t+1} - W_*||_op) >= eta ||O||_op / 2,
    and in Frobenius norm >= eta sqrt(rank O) / 2 for the exact polar factor, whatever the estimator."""
    levels = []
    for loss in ("softmax", "ls"):
        prob = _prob(loss)
        rank = prob.K - 1 if loss == "softmax" else min(prob.K, prob.d)
        for orth, est in (("svd", "saga"), ("svd", "lsvrg"), ("ns5", "saga")):
            lr = 3e-3
            algo = vm.VRMuon(prob, estimator=est, length="const", lr=lr, orth=orth)
            W = np.zeros((prob.K, prob.d))
            st = algo.init_state(W)
            rng = np.random.default_rng(0)
            d_op, d_F, s_op, s_F = [], [], [], []
            for t in range(1500):
                j, coin = int(rng.integers(prob.n)), float(rng.random())
                m, c, aux = algo.pieces(W, st, j, coin)
                S, info = algo.step(W, m, c, st, t)
                algo.commit(W, st, j, coin, aux, info)
                E = W - prob.W_star
                d_op.append(_op(E))
                d_F.append(np.sqrt(_fro2(E)))
                s_op.append(_op(S))
                s_F.append(np.sqrt(_fro2(S)))
                W = algo.advance(W, S, st)
            E = W - prob.W_star
            d_op.append(_op(E))
            d_F.append(np.sqrt(_fro2(E)))
            d_op, d_F, s_op, s_F = map(np.array, (d_op, d_F, s_op, s_F))
            assert np.all(np.maximum(d_op[:-1], d_op[1:]) >= 0.5 * s_op * (1 - 1e-9)), (loss, orth)
            assert np.all(np.maximum(d_F[:-1], d_F[1:]) >= 0.5 * s_F * (1 - 1e-9)), (loss, orth)
            if orth == "svd":                                  # exact polar factor: ||S||_op = eta, ||S||_F^2 = rank eta^2
                assert np.allclose(s_op, lr, rtol=1e-10) and np.allclose(s_F ** 2, rank * lr ** 2, rtol=1e-10)
                tail = d_F[-300:] ** 2                             # the iterates do not converge below the floor
                assert tail.max() >= rank * lr ** 2 / 4, (loss, tail.max())
                levels.append(f"{loss}/{est}: median dist2 = {np.median(tail) / lr ** 2:.2f} lr^2 "
                              f"(bound on every other iterate {rank / 4:.2f} lr^2)")
            else:                                              # Newton-Schulz: only the largest singular value matters
                assert 0.6 * lr < s_op.min() and s_op.max() < 1.3 * lr
    print("\n  F2, median squared distance over the last 300 of 1500 iterations: " + "; ".join(levels))


def _mem_terms(prob, W):
    dR = prob.loss_residuals(W) - prob.R_star
    return float((np.einsum("ij,ij->i", dR, dR) * prob.sqnorms).mean())


def test_F3_memory_recursion():
    """F3: E[M_+ | past] = (1 - b/n) M + (b/n) S exactly, for every step rule (enumeration of all batches); with a
    relaxed table the expectation lies between the two bounds of the notes."""
    for loss in ("softmax", "ls"):
        for b, n in ((1, 60), (2, 18)):
            prob = _prob(loss, n=n)
            for cfg in (dict(placement="euclid"), dict(placement="joint"), dict(placement="split"),
                        dict(placement="joint", length="const", lr=1e-2)):
                for delta in (1.0, 0.4):
                    full = dict(estimator="saga", batch=b, mem_relax=delta, **cfg)
                    W, st = _warm(prob, full, iters=25)
                    algo = vm.VRMuon(prob, **full)
                    M, S_t = algo.memory_error(st), _mem_terms(prob, W)
                    batches = ([int(j) for j in range(n)] if b == 1
                               else [np.array(B) for B in itertools.combinations(range(n), b)])
                    acc = 0.0
                    for B in batches:
                        st2 = copy.deepcopy(st)
                        m, c, aux = algo.pieces(W, st2, B)
                        _, info = algo.step(W, m, c, st2)
                        algo.commit(W, st2, B, 1.0, aux, info)
                        acc += algo.memory_error(st2)
                    EM = acc / len(batches)
                    q = b / n
                    if delta == 1.0:
                        assert abs(EM - ((1 - q) * M + q * S_t)) <= 1e-12 * M, (loss, b, cfg)
                        assert EM >= (1 - q) * M * (1 - 1e-12)
                    else:
                        assert EM <= ((1 - delta * q) * M + delta * q * S_t) * (1 + 1e-12), (loss, b, cfg)
                        low = (1 - q * delta * (2 - delta)) * M - 2 * q * delta * (1 - delta) * np.sqrt(M * S_t)
                        assert EM >= low * (1 - 1e-12) - 1e-15, (loss, b, cfg)


def test_F4_surrogate():
    """F4: the predictable part of SAGA is the gradient of h(W) = (1/n) sum_i [g_i(phi_i) + <z_i, W - phi_i>] +
    (lam/2) ||W||_F^2, an isotropic quadratic lower bound of F whose minimizer is -zbar / lam."""
    for loss in ("softmax", "ls"):
        prob = _prob(loss)
        n, lam = prob.n, prob.lam
        algo = vm.VRMuon(prob, estimator="saga")
        W = np.zeros((prob.K, prob.d))
        st = algo.init_state(W)
        phi = [W.copy() for _ in range(n)]                               # point where each entry was last refreshed
        rng = np.random.default_rng(1)
        for t in range(90):
            j = int(rng.integers(n))
            m, c, aux = algo.pieces(W, st, j)
            S, info = algo.step(W, m, c, st, t)
            algo.commit(W, st, j, 1.0, aux, info)
            phi[j] = W.copy()
            W = algo.advance(W, S, st)
        Z = [np.outer(st.rtab[i], prob.X[i]) for i in range(n)]            # stored loss gradients z_i
        g_phi = [prob.f_i(phi[i], i) - 0.5 * lam * _fro2(phi[i]) for i in range(n)]
        for i in (0, 17, 41):                                              # the table holds the gradients at phi_i
            assert np.allclose(Z[i], prob.grad_i(phi[i], i) - lam * phi[i], atol=1e-13)

        def h(V):
            return float(np.mean([g_phi[i] + np.vdot(Z[i], V - phi[i]) for i in range(n)]) + 0.5 * lam * _fro2(V))

        m_now = st.zbar + lam * W
        assert np.allclose(m_now, np.mean(Z, axis=0) + lam * W, atol=1e-13)   # grad h(W) = zbar + lam W = m
        W_B = -st.zbar / lam
        pts = [W, W_B, prob.W_star] + [W + s * rng.standard_normal(W.shape) for s in (1e-2, 1.0, 10.0)]
        for V in pts:
            assert h(V) <= prob.F(V) + 1e-12 * max(1.0, abs(prob.F(V))), loss   # lower bound (convexity of g_i)
            # isotropic quadratic: h(V) = h(W_B) + (lam/2) ||V - W_B||_F^2
            assert abs(h(V) - h(W_B) - 0.5 * lam * _fro2(V - W_B)) <= 1e-10 * max(1.0, abs(h(V)))
        assert abs(np.linalg.norm(W - W_B) - np.linalg.norm(m_now) / lam) <= 1e-10 * np.linalg.norm(W - W_B)


def test_F5_pathwise_descent():
    """F5: F(W - gamma ||omega||_* polar(omega)) <= F(W) - gamma ||omega||_* [(1 - gamma L_op / 2) ||omega||_* -
    ||omega - grad F||_*] for every sampled index (stochastic estimators, stale memory)."""
    viol, n_pos, total = 0, 0, 0
    for loss in ("softmax", "ls"):
        prob = _prob(loss)
        for cfg in STATES:
            W, st = _warm(prob, cfg, iters=60)
            G = prob.grad(W)
            for c_gamma in (0.05, 0.5, 1.0, 3.0):
                gamma = c_gamma / prob.L_op
                algo = vm.VRMuon(prob, estimator=cfg["estimator"], gamma=gamma)
                S, m, C = exact.enumerate_steps(algo, W, st, return_pieces=True)
                for j in range(prob.n):
                    om = m + C[j]
                    nu, err = _nuc(om), _nuc(om - G)
                    bracket = (1 - gamma * prob.L_op / 2) * nu - err
                    dF = prob.F_diff(W - S[j], W, G0=G)
                    total += 1
                    n_pos += bracket > 0
                    if dF > -gamma * nu * bracket + 1e-12 * max(1.0, abs(gamma * nu * bracket)):
                        viol += 1
    assert viol == 0, viol
    print(f"\n  F5: 0 violations in {total} steps; the guaranteed decrease is positive for {n_pos} of them")


def test_F7_worst_direction():
    """F7 (consequence): D polar(M) vanishes on a singular pair u_k v_k^T and has gain 1/s_K on u_K v^T with v
    orthogonal to the rows of M (K < d); for K = d the largest gain is 2 / (s_K + s_{K-1})."""
    rng = np.random.default_rng(5)
    U, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    V, _ = np.linalg.qr(rng.standard_normal((7, 7)))
    s = np.array([1.0, 0.7, 0.4, 0.05])
    M = (U * s) @ V[:, :4].T
    Dp = maps.SpectralDerivative(M, "polar")
    for k in range(4):
        assert np.linalg.norm(Dp(np.outer(U[:, k], V[:, k]))) < 1e-12
    E = np.outer(U[:, 3], V[:, 5])                                    # rotation of the smallest pair out of the rows
    assert np.allclose(Dp(E), E / s[3], atol=1e-10)
    Q4, _ = np.linalg.qr(V[:4, :4])                                   # square case
    Ms = (U * s) @ Q4.T
    Dq = maps.SpectralDerivative(Ms, "polar")
    E = np.outer(U[:, 3], Q4[:, 2]) - np.outer(U[:, 2], Q4[:, 3])
    assert np.allclose(Dq(E), 2.0 / (s[2] + s[3]) * E, atol=1e-10)


def test_F9_radial_descent():
    """F9: for a predictable O with ||O||_op <= 1, q_j = <omega_j, O> is unbiased for a = <grad F, O> and
    E_j F(W - gamma q_j O) <= F(W) - gamma a^2 + (L_op / 2) gamma^2 (a^2 + v) (exact enumeration)."""
    rng = np.random.default_rng(2)
    for loss in ("softmax", "ls"):
        prob = _prob(loss)
        for cfg in STATES:
            W, st = _warm(prob, cfg, iters=60)
            G = prob.grad(W)
            algo = vm.VRMuon(prob, **cfg)
            _, m, C = exact.enumerate_steps(algo, W, st, return_pieces=True)
            basis = maps.gauge_basis(prob)
            R = prob.remove_gauge(rng.standard_normal(W.shape))
            for O in (maps.polar(m, basis=basis), maps.polar(G, basis=basis), 0.6 * R / _op(R)):
                q = np.array([float(np.vdot(m + C[j], O)) for j in range(prob.n)])
                a, v = float(np.vdot(G, O)), float(q.var())
                assert abs(q.mean() - a) <= 1e-12 * max(1.0, abs(a))
                for c_gamma in (0.05, 0.5, 1.0, 2.0, 5.0):
                    gamma = c_gamma / prob.L_op
                    EdF = np.mean([prob.F_diff(W - gamma * q[j] * O, W, G0=G) for j in range(prob.n)])
                    bound = -gamma * a * a + 0.5 * prob.L_op * gamma ** 2 * (a * a + v)
                    assert EdF <= bound + 1e-12 * max(1.0, abs(bound)), (loss, cfg, c_gamma)


def test_F10_newton_metric():
    """F10: least squares with K = d and W - W_* = s Q, Q orthogonal: the polar metric of grad F(W) without floor is
    C = tr(H) H^{-1}, the Hessian of the reparametrized objective is tr(H) Id, and
    R2P / mu_P = max_i x_i^T H^{-1} x_i + lam / lambda_min(H). With metric_floor = 1 the metric is Euclidean."""
    K = d = 6
    prob = problems.make_matrix_ls(80, d, K, kappa=300.0, alpha=1.5, seed=4)
    optimum.solve_optimum(prob)
    Q, _ = np.linalg.qr(np.random.default_rng(3).standard_normal((K, d)))
    W = prob.W_star + 0.7 * Q
    H = prob.c_loss * prob.Sigma_hat + prob.lam * np.eye(d)
    G = prob.grad(W)
    assert np.allclose(G, 0.7 * Q @ H, atol=1e-12)
    algo = vm.VRMuon(prob, estimator="lsvrg", placement="metric", metric_floor=1e-14, c_metric=1.0)
    V, gains, r, g_P, R2P = algo.polar_metric(G)
    C = V @ np.diag(gains) @ V.T + r * (np.eye(d) - V @ V.T)
    assert np.allclose(C, np.trace(H) * np.linalg.inv(H), rtol=1e-9, atol=0)
    w, P = np.linalg.eigh(C)
    Ch = (P * np.sqrt(w)) @ P.T
    assert np.allclose(Ch @ H @ Ch, np.trace(H) * np.eye(d), rtol=0, atol=1e-9 * np.trace(H))
    Hinv = np.linalg.inv(H)
    d_inf = prob.c_loss * float(np.einsum("id,de,ie->i", prob.X, Hinv, prob.X).max())
    kappa_P = R2P / np.trace(H)
    assert abs(kappa_P - (d_inf + prob.lam / np.linalg.eigvalsh(H)[0])) <= 1e-9 * kappa_P
    assert d_inf <= kappa_P <= d_inf + 1.0
    # floor = 1: all gains equal, the metric is a multiple of the identity (Euclidean SAGA / L-SVRG)
    flat = vm.VRMuon(prob, estimator="lsvrg", placement="metric", metric_floor=1.0, c_metric=1.0)
    V1, gains1, r1, g1, R2P1 = flat.polar_metric(G)
    assert np.allclose(gains1, gains1[0]) and abs(g1 * gains1[0] - 1.0 / prob.R2) <= 1e-12 / prob.R2
    # an active floor destroys the Newton property: the conditioning in the metric degrades towards the Euclidean one
    kap = []
    for floor in (1e-14, 0.1, 0.5, 1.0):
        a = vm.VRMuon(prob, estimator="lsvrg", placement="metric", metric_floor=floor, c_metric=1.0)
        Vf, gf, rf, _, R2Pf = a.polar_metric(G)
        Cf = Vf @ np.diag(gf) @ Vf.T + rf * (np.eye(d) - Vf @ Vf.T)
        wf, Pf = np.linalg.eigh(Cf)
        Chf = (Pf * np.sqrt(wf)) @ Pf.T
        kap.append(R2Pf / np.linalg.eigvalsh(Chf @ H @ Chf)[0])
    assert all(kap[i] <= kap[i + 1] * (1 + 1e-9) for i in range(3)) and kap[0] < 0.5 * kap[3], kap
    assert abs(kap[3] - prob.R2 / np.linalg.eigvalsh(H)[0]) <= 1e-9 * kap[3]
    print("\n  F10: kappa_P for floors 0 / 0.1 / 0.5 / 1: " + " / ".join(f"{k:.1f}" for k in kap)
          + f" (Euclidean {prob.R2 / np.linalg.eigvalsh(H)[0]:.1f}, d_inf {d_inf:.1f})")


def test_F3_one_step_variance():
    """F3 (iterate): E_j ||W_+ - W_*||_F^2 >= Var_j(S_j), and when the innovation enters the step linearly (Euclidean
    and split placements) Var_j(S_j) >= (gamma_c theta)^2 (sqrt(V) - sqrt(S))_+^2 with
    V = M - ||zbar - grad G(W_*)||_F^2 (exact enumeration of the index)."""
    n_active, total = 0, 0
    for loss in ("softmax", "ls"):
        prob = _prob(loss)
        gradG_star = -prob.lam * prob.W_star                       # (1/n) sum_i g_i'(W_*) = -lam W_*
        for placement in ("euclid", "split"):
            for theta in (1.0, 0.3):
                for iters in (3, 37, 150):
                    cfg = dict(estimator="saga", placement=placement, theta=theta)
                    W, st = _warm(prob, cfg, iters=iters)
                    algo = vm.VRMuon(prob, **cfg)
                    S = np.asarray(exact.enumerate_steps(algo, W, st))
                    var_S = float(((S - S.mean(axis=0)) ** 2).sum() / prob.n)
                    e_dist = float(((W - S - prob.W_star) ** 2).sum() / prob.n)
                    assert e_dist >= var_S * (1 - 1e-12), (loss, placement, theta, iters)
                    M, S_t = algo.memory_error(st), _mem_terms(prob, W)
                    V = M - _fro2(st.zbar - gradG_star)
                    assert V >= -1e-12 * M
                    g_lin = algo.gamma if placement == "euclid" else algo.gamma_c
                    low = (g_lin * theta) ** 2 * max(np.sqrt(max(V, 0.0)) - np.sqrt(S_t), 0.0) ** 2
                    assert var_S >= low * (1 - 1e-10), (loss, placement, theta, iters, var_S, low)
                    n_active += low > 0
                    total += 1
    assert n_active > 0
    print(f"\n  F3 (iterate): bound active (V > S) at {n_active} of {total} states")


def test_F9_pairing_bounds():
    """F9 (what governs the progress): for O = polar(m) and a = <grad F, O>, |a| <= ||grad F||_* and
    a >= ||m||_* - ||grad F - m||_*; with a stale memory a can be negative (the lower bound is then negative too)."""
    rng = np.random.default_rng(4)
    n_neg, total = 0, 0
    for loss in ("softmax", "ls"):
        prob = _prob(loss)
        basis = maps.gauge_basis(prob)
        cases = []
        for cfg in STATES:
            for iters in (2, 10, 37, 150, 600):
                W, st = _warm(prob, cfg, iters=iters)
                _, m, _ = exact.enumerate_steps(vm.VRMuon(prob, **cfg), W, st, return_pieces=True)
                cases.append((W, m))
        # stale SAGA memory: table of an early state, iterate moved close to the optimum
        _, st = _warm(prob, STATES[0], iters=5)
        for _ in range(20):
            W = prob.W_star + 1e-2 * prob.remove_gauge(rng.standard_normal((prob.K, prob.d)))
            cases.append((W, st.zbar + prob.lam * W))
        for W, m in cases:
            G = prob.grad(W)
            O = maps.polar(m, basis=basis)
            a = float(np.vdot(G, O))
            nG, nm, nerr = _nuc(G), _nuc(m), _nuc(G - m)
            assert abs(a) <= nG * (1 + 1e-12)
            assert a >= nm - nerr - 1e-12 * (nm + nerr), (loss, a, nm, nerr)
            n_neg += a < 0
            total += 1
    assert n_neg > 0                                                # the pairing is not sign-definite
    print(f"\n  F9: a = <grad F, polar(m)> negative at {n_neg} of {total} states (stale memories included)")


ONE_SHOT = dict(estimator="saga", placement="metric", metric_every=10 ** 9)


def _metric_matrix(algo, D):
    """C, gamma_P, R2P of the polar metric of the anchor D, from the lab's factors."""
    V, gains, r, g_P, R2P = algo.polar_metric(D)
    d = algo.prob.d
    C = V @ np.diag(gains) @ V.T + r * (np.eye(d) - V @ V.T)
    return 0.5 * (C + C.T), g_P, R2P


def _sym_roots(C):
    w, Q = np.linalg.eigh(C)
    return (Q * np.sqrt(w)) @ Q.T, (Q / np.sqrt(w)) @ Q.T, w


def _drive(algo, W, idx):
    st = algo.init_state(W)
    for t, j in enumerate(idx):
        m, c, aux = algo.pieces(W, st, int(j))
        S, info = algo.step(W, m, c, st, t)
        algo.commit(W, st, int(j), 1.0, aux, info)
        W = algo.advance(W, S, st)
    return W, st


def test_F11_one_shot_equivalence():
    """F11: SAGA preconditioned by the polar metric of grad F(W_0), computed once and never refreshed, is for the
    whole run SAGA in the variables What = W C^(-1/2) on Fhat(What) = F(What C^(1/2)), in the form used by the lab:
    features C^(1/2) x_i, table of loss residuals (the same table), exact gradient lam What C of the regularizer
    (lam/2) ||What C^(1/2)||_F^2. It is not the lab's Euclidean SAGA run on the transformed features, whose
    regularizer is (lam/2) ||What||_F^2 (another objective, another minimizer)."""
    for loss, K, d in (("softmax", 4, 6), ("ls", 4, 6), ("ls", 5, 5)):
        prob = _prob(loss, K=K, d=d)
        n, lam = prob.n, prob.lam
        W0 = np.zeros((K, d))
        idx = np.random.default_rng(7).integers(n, size=3 * n)
        for floor in (1e-14, 0.1, 0.5):
            algo = vm.VRMuon(prob, metric_floor=floor, c_metric=0.25, **ONE_SHOT)
            C, g_P, R2P = _metric_matrix(algo, prob.grad(W0))
            Ch, Cmh, _ = _sym_roots(C)
            assert abs(g_P * R2P - 0.25) <= 1e-12
            # lab iteration and explicit SAGA in the variables What, same indices
            W, st = W0.copy(), algo.init_state(W0)
            Xh = prob.X @ Ch
            Wh = W0 @ Cmh
            rtab = prob.loss_residuals(W0).copy()
            zbar = rtab.T @ Xh / n
            worst = 0.0
            for t, j in enumerate(idx):
                j = int(j)
                m, c, aux = algo.pieces(W, st, j)
                S, info = algo.step(W, m, c, st, t)
                algo.commit(W, st, j, 1.0, aux, info)
                W = algo.advance(W, S, st)
                rj = prob.residual_i(Wh @ Xh[j], j)
                ch = np.outer(rj - rtab[j], Xh[j])
                Wh = Wh - g_P * (ch + zbar + lam * (Wh @ C))        # exact gradient of the non-isotropic regularizer
                zbar = zbar + ch / n
                rtab[j] = rj
                worst = max(worst, np.linalg.norm(W - Wh @ Ch) / np.linalg.norm(W))
            assert worst <= 1e-10, (loss, K, d, floor, worst)
            assert np.abs(rtab - st.rtab).max() <= 1e-10 * max(1.0, np.abs(rtab).max())
            if loss == "softmax":                                   # the gauge subspace is invariant
                assert np.abs(W.sum(axis=0)).max() <= 1e-12
            if loss == "ls" and floor == 0.1:                       # control: Euclidean SAGA on the transformed features
                ph = problems.MatrixLSProblem(Xh, prob.Y, lam)
                optimum.solve_optimum(ph)
                Wn, _ = _drive(vm.VRMuon(ph, estimator="saga", placement="euclid", gamma=g_P), W0 @ Cmh, idx)
                assert np.linalg.norm(Wn @ Ch - W) > 1e-3 * np.linalg.norm(W)
                assert np.linalg.norm(ph.W_star @ Ch - prob.W_star) > 1e-3 * np.linalg.norm(prob.W_star)


def test_F11_lyapunov():
    """F11: with gamma = c / R2P and V = ||(W - W_*) C^(-1/2)||_F^2 + k gamma^2 sum_i ||(z_i - g_i'(W_*)) C^(1/2)||_F^2,
    E_j[V_+] <= (1 - min{2 gamma mu_P (1 - gamma R2P (1 + k/2)), (1 - 2/k) / n}) V at every state (states of the run,
    arbitrary iterates, stale tables, tables that are not loss gradients). mu_P is the global strong-convexity
    constant of Fhat: lambda_min(C^(1/2) H C^(1/2)) for least squares, lam lambda_min(C) for softmax. The case
    c = 1/4, k = 3 is the rate 1 - min{1/(3n), 3 mu_P / (16 R2P)} of Proposition 5.9 in the metric. Exact
    enumeration of the n next steps."""
    rng = np.random.default_rng(0)
    margins = []
    for loss, K, d in (("softmax", 4, 6), ("ls", 4, 6), ("ls", 5, 5)):
        prob = _prob(loss, K=K, d=d)
        n, lam, X = prob.n, prob.lam, prob.X
        W0 = np.zeros((K, d))
        G0 = prob.grad(W0)
        for floor in (1e-14, 0.1, 0.5):
            for c_metric, k in ((0.25, 3.0), (0.1, 4.0), (0.45, 2.2)):
                cfg = dict(metric_floor=floor, c_metric=c_metric, **ONE_SHOT)
                algo = vm.VRMuon(prob, **cfg)
                C, g_P, R2P = _metric_matrix(algo, G0)
                Ch, Cmh, wC = _sym_roots(C)
                q = np.einsum("id,de,ie->i", X, C, X)
                if loss == "ls":
                    H = prob.c_loss * prob.Sigma_hat + lam * np.eye(d)
                    mu_P = float(np.linalg.eigvalsh(Ch @ H @ Ch)[0])
                else:
                    mu_P = lam * float(wC[0])
                rho = 1.0 - min(2 * g_P * mu_P * (1 - g_P * R2P * (1 + k / 2)), (1 - 2 / k) / n)
                if (c_metric, k) == (0.25, 3.0):
                    assert abs(rho - (1 - min(1 / (3 * n), 3 * mu_P / (16 * R2P)))) <= 1e-14
                states = [(W, st) for _, W, st in exact.collect_states(prob, W0, 4 * n, [0, n // 2, 4 * n], seed=3,
                                                                       **cfg)]
                for i_state in range(4):                            # arbitrary iterates and tables
                    scale = (1e-2, 1.0, 10.0, 1.0)[i_state]
                    Wr = prob.W_star + scale * prob.remove_gauge(rng.standard_normal((K, d)))
                    st = algo.init_state(Wr)
                    if i_state >= 1:                                # stale table: residuals at scattered points
                        Phi = prob.W_star[None] + rng.standard_normal((n, K, d))
                        st.rtab = np.array([prob.residual_i(Phi[i] @ X[i], i) for i in range(n)])
                        if i_state == 3:                            # not even gradients of the losses
                            st.rtab = st.rtab + 0.5 * rng.standard_normal((n, K))
                        st.zbar = st.rtab.T @ X / n
                    st.metric = {"svd": algo.anchor_svd(G0), "gauged": algo.basis is not None}
                    states.append((Wr, st))
                for i_state, (W, st) in enumerate(states):
                    S = np.asarray(exact.enumerate_steps(algo, W, st))
                    dR = st.rtab - prob.R_star
                    dN = prob.loss_residuals(W) - prob.R_star
                    e_tab = np.einsum("ik,ik->i", dR, dR) * q
                    e_new = np.einsum("ik,ik->i", dN, dN) * q
                    Mg = float(e_tab.sum())
                    dist = _fro2((W - prob.W_star) @ Cmh)
                    V = dist + k * g_P ** 2 * Mg
                    EV = float(np.mean([_fro2((W - S[j] - prob.W_star) @ Cmh)
                                        + k * g_P ** 2 * (Mg - e_tab[j] + e_new[j]) for j in range(n)]))
                    assert EV <= rho * V * (1 + 1e-12), (loss, K, d, floor, c_metric, i_state, EV / V, rho)
                    margins.append(EV / V - rho)
                    if i_state == 0 and c_metric == 0.25:           # fresh table at W_0: V_0 <= (1 + 3n/16) ||.||^2
                        assert V <= (1 + 3 * n / 16) * dist * (1 + 1e-12)
    print(f"\n  F11: E[V+]/V - rho between {min(margins):.2e} and {max(margins):.2e} over {len(margins)} states")


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
