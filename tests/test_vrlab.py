"""Unit tests for vrlab. Run with `python -m pytest tests -q` or `python tests/test_vrlab.py`."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from vrlab import problems, solvers, algorithms, instrumentation, bounds, analysis  # noqa: E402


def _small_logistic(n=40, d=6, kappa=50.0, alpha=1.0, seed=3):
    prob = problems.make_logistic(n, d, kappa=kappa, alpha=alpha, seed=seed)
    solvers.solve_optimum(prob)
    return prob


def test_logistic_derivatives():
    prob = problems.make_logistic(50, 7, kappa=30.0, alpha=0.5, seed=1)
    rng = np.random.default_rng(0)
    th = rng.standard_normal(prob.d)
    g = prob.grad(th)
    G = prob.all_grads(th)
    assert np.allclose(G.mean(0), g, atol=1e-12)
    for i in [0, 5, 49]:
        assert np.allclose(prob.grad_i(th, i), G[i], atol=1e-12)
    # finite differences
    eps = 1e-6
    for j in range(prob.d):
        e = np.zeros(prob.d)
        e[j] = eps
        fd = (prob.F(th + e) - prob.F(th - e)) / (2 * eps)
        assert abs(fd - g[j]) < 1e-7 * max(1.0, abs(g[j]))
        fdH = (prob.grad(th + e) - prob.grad(th - e)) / (2 * eps)
        assert np.allclose(fdH, prob.hessian(th)[j], atol=1e-6)
    # per-sample smoothness: ||f'_i(a) - f'_i(b)|| <= L_i ||a - b||
    a, b = rng.standard_normal(prob.d), rng.standard_normal(prob.d)
    for i in range(prob.n):
        assert np.linalg.norm(prob.grad_i(a, i) - prob.grad_i(b, i)) <= prob.L_i[i] * np.linalg.norm(a - b) * (1 + 1e-12)
    assert abs(prob.kappa - 30.0) < 1e-9


def test_newton_no_stall_near_optimum():
    """Regression test: on the B3 problem kappa = 10, alpha = 1 (n = 1000, d = 50, data seed 0), the Armijo test on F
    is decided by rounding near the optimum; the solver used to stall at ||F'(eta_*)|| = 1.3e-12 after 26-40
    iterations. It must now reach the tolerance 1e-15 in a few iterations."""
    prob = problems.make_logistic(1000, 50, kappa=10.0, alpha=1.0, seed=0)
    theta, info = solvers.newton_solve(prob)
    assert info["grad_norm"] <= 1e-15 and info["iters"] <= 10, info
    assert np.linalg.norm(prob.grad(theta)) <= 1e-15


def test_newton_optimum():
    prob = _small_logistic()
    assert prob.opt_info["grad_norm"] < 1e-12
    # eta_* is a minimizer: F(eta_* + h) >= F(eta_*)
    rng = np.random.default_rng(1)
    for _ in range(5):
        h = 1e-3 * rng.standard_normal(prob.d)
        assert prob.F(prob.theta_star + h) >= prob.F_star - 1e-15
    # strong convexity sanity: lambda_min(H*) >= mu
    assert np.linalg.eigvalsh(prob.hessian(prob.theta_star))[0] >= prob.mu * (1 - 1e-10)


def _check_identity(prob, P=None):
    """E[V_t] from saga_state_stats must equal the average over all n possible indices of the realized V_t."""
    rng = np.random.default_rng(7)
    n, d = prob.n, prob.d
    metric = instrumentation.Metric(P, d)
    theta = prob.theta_star + 0.5 * rng.standard_normal(d)
    z = prob.all_grads(theta + 0.3 * rng.standard_normal(d)) + 0.1 * rng.standard_normal((n, d))
    zbar = z.mean(0)
    R2 = prob.R2 if P is None else metric.constants(prob)["R2_P"]
    mu = prob.mu if P is None else metric.constants(prob)["mu_P"]
    gamma, Delta = bounds.saga_textbook_params(R2)
    st = instrumentation.saga_state_stats(prob, theta, z, zbar, gamma, Delta, R2, mu, metric)
    Ds, Ms = [], []
    gs = prob.grads_star
    for i in range(n):
        gi = prob.grad_i(theta, i)
        omega = gi + zbar - z[i]
        th_new = theta - gamma * metric.apply_inv(omega)
        z_new = z.copy()
        z_new[i] = gi
        Ds.append(metric.norm2(th_new - prob.theta_star))
        Ms.append(metric.dual_norm2_rows(z_new - gs).sum())
    Ds, Ms = np.array(Ds), np.array(Ms)
    assert abs(Ds.mean() - st["ED_next"]) < 1e-10 * max(1, st["D"])
    assert abs(Ms.mean() - st["EM_next"]) < 1e-10 * max(1, st["M"])
    assert abs(Ds.mean() + Delta * Ms.mean() - st["EV_next"]) < 1e-10 * max(1, st["V"])
    # B = M/n - ||zbar||^2 (requires sum_i f'_i(eta_*) = 0, true up to Newton precision)
    assert abs(st["B"] - (st["M"] / n - st["zbar_norm2"])) < 1e-8 * max(1, st["M"] / n)
    for k in ["r_sq", "r_coco", "r_var", "r_sc", "r_min"]:
        assert 0 < st[k] <= 1 + 1e-12, (k, st[k])
    assert st["mu_eff"] >= mu * (1 - 1e-10)
    assert st["L_eff"] <= R2 * (1 + 1e-10)
    # cumulative chain is monotone and ends at the worst-case bound
    chain = analysis.cumulative_chain(st, gamma, Delta, R2, mu, n)
    assert np.all(np.diff(chain) >= -1e-12), chain
    assert abs(chain[-1] - bounds.saga_rate(n, R2, mu, gamma, Delta)) < 1e-12
    assert abs(chain[0] - st["rho_exact"]) < 1e-12
    loo = analysis.leave_one_out(st, gamma, Delta, R2, mu, n)
    for k in ["I1", "I3", "I2", "I4", "I5"]:
        # each leave-one-out rate applies only valid inequalities, hence upper-bounds the exact contraction; 'I1' and
        # 'I2' need not stay below the certified bound (single substitutions, not nested; see analysis.leave_one_out)
        assert chain[0] - 1e-12 <= loo[k], (k, loo[k], chain[0])
    for k in ["I3", "I4", "I5"]:
        assert loo[k] <= loo["bound"] + 1e-12, (k, loo[k], loo["bound"])
    assert loo["I4"] == chain[3] and loo["I5"] == chain[4]       # = rho_I2 and rho_I4 of the cumulative chain
    # order A of the B3 note is monotone and connects the bound to the exact rate
    chainA = analysis.cumulative_chain_A(st, gamma, Delta, R2, mu, n)
    assert np.all(np.diff(chainA) <= 1e-12), chainA
    assert abs(chainA[0] - bounds.saga_rate(n, R2, mu, gamma, Delta)) < 1e-12 and abs(chainA[-1] - st["rho_exact"]) < 1e-12


def test_saga_exact_identity_euclidean():
    _check_identity(_small_logistic())


def test_saga_exact_identity_metric():
    prob = _small_logistic()
    rng = np.random.default_rng(11)
    A = rng.standard_normal((prob.d, prob.d))
    P = A @ A.T + 0.5 * np.eye(prob.d)
    _check_identity(prob, P=P)
    _check_identity(prob, P=np.diag(P).copy())


def test_preconditioned_saga_converges():
    """SAGA in a diagonal and in a dense quadratic metric P (theta <- theta - gamma P^{-1} omega), with the P-constants
    (R_P^2, mu_P) of Metric.constants and the textbook step 1/(4 R_P^2), converges to eta_* on a small anisotropic
    logistic problem; the recorded D_P is the P-norm of the final error."""
    prob = _small_logistic(n=60, d=6, kappa=40.0, alpha=2.0)
    theta0 = np.zeros(prob.d)
    P_dense = prob.Sigma_hat + prob.lam * np.eye(prob.d)
    for P, kind in [(np.diag(P_dense).copy(), "diag"), (P_dense, "dense")]:
        metric = instrumentation.Metric(P, prob.d)
        cst = metric.constants(prob)
        assert cst["kappa_P"] >= 1.0
        r = algorithms.saga(prob, theta0, P=P, R2=cst["R2_P"], mu=cst["mu_P"], passes=60, seed=0, record_every=10)
        assert r["metric"] == kind and abs(r["gamma"] - 1.0 / (4 * cst["R2_P"])) < 1e-15
        assert np.all(np.isfinite(r["D"]))
        assert r["D"][-1] < 1e-10 * r["D"][0], (kind, r["D"][-1] / r["D"][0])
        assert r["D_P"][-1] == metric.norm2(r["theta"] - prob.theta_star)


def test_saga_matches_bound_in_expectation():
    prob = _small_logistic(n=60, d=5, kappa=40.0)
    theta0 = np.zeros(prob.d)
    runs = [algorithms.saga(prob, theta0, passes=20, seed=s, record_every=6, instrument_every=30) for s in range(6)]
    it, meanD, _ = analysis.mean_over_seeds(runs, key="D")
    bnd = bounds.saga_bound(it, prob.n, prob.R2, prob.mu, runs[0]["D0"])
    assert np.all(meanD <= bnd * 1.0001)
    # sharper bound (before I6) also holds in expectation
    bnd2 = bounds.saga_bound(it, prob.n, prob.R2, prob.mu, runs[0]["D0"], S0=runs[0]["S0"])
    assert np.all(meanD <= bnd2 * 1.0001)
    assert runs[0]["r_init"] <= 1.0
    # the recorded incremental M equals the exact M at instrumentation points
    r = runs[0]
    for q in r["instr"][1:]:
        k = np.where(r["iters"] == q["t"])[0]
        if len(k):
            assert abs(r["M"][k[0]] - q["M"]) < 1e-8 * max(1.0, q["M"])
    assert meanD[-1] < 1e-6 * meanD[0]


def test_hard_instance_and_exact_minimizer():
    prob = problems.make_hard_instance(n=8, m=20, kappa=500.0)
    assert abs(prob.kappa - 500.0) < 1e-8
    solvers.solve_optimum(prob)
    assert np.linalg.norm(prob.grad(prob.theta_star)) < 1e-10
    rng = np.random.default_rng(2)
    th = rng.standard_normal(prob.d)
    assert np.allclose(prob.all_grads(th).mean(0), prob.grad(th), atol=1e-12)
    eps = 1e-6
    e = np.zeros(prob.d)
    e[5] = eps
    assert abs((prob.F(th + e) - prob.F(th - e)) / (2 * eps) - prob.grad(th)[5]) < 1e-6
    # smoothness of f_i equals n * lambda_max(A_i) (quadratic)
    assert abs(prob.R2 - prob.n * np.linalg.eigvalsh(prob.A[0])[-1]) < 1e-9


def test_shared_regularizer_instance_matches_A1_note():
    """The instance of the A1 note: with zeta = zeta_Q the minimizer is the geometric chain x*_j = q^j on every block,
    each f_i is (mu kappa)-smooth up to lambda_max(A_m)/4, F is mu-strongly convex, and per-block Q = 1 + (kappa-1)/n."""
    n, m, kappa, mu = 5, 30, 400.0, 1.0
    prob = problems.make_hard_instance_shared(n, m, kappa, mu=mu)
    solvers.solve_optimum(prob)
    q = prob.meta["q"]
    xs = q ** np.arange(1, m + 1)
    assert np.allclose(prob.theta_star.reshape(n, m), xs[None, :], atol=1e-10)
    assert np.linalg.norm(prob.grad(prob.theta_star)) < 1e-10
    assert prob.R2 <= mu * kappa + 1e-9 and prob.R2 > 0.9 * mu * kappa
    assert prob.mu >= mu - 1e-12
    rng = np.random.default_rng(3)
    th = rng.standard_normal(prob.d)
    G = prob.all_grads(th)
    assert np.allclose(G.mean(0), prob.grad(th), atol=1e-10)
    assert np.allclose(prob.grad_i(th, 2), G[2], atol=1e-12)
    eps = 1e-6
    e = np.zeros(prob.d)
    e[7] = eps
    assert abs((prob.F(th + e) - prob.F(th - e)) / (2 * eps) - prob.grad(th)[7]) < 1e-6
    # F(eta_*) = -c q / 2 (Proposition "closed-form minimizer" of the note)
    assert abs(prob.F_star - (-prob.meta["c"] * q / 2)) < 1e-9 * max(1.0, abs(prob.F_star))
    # per-component gradient Lipschitz constant <= R2
    a, b = rng.standard_normal(prob.d), rng.standard_normal(prob.d)
    for i in range(n):
        assert np.linalg.norm(prob.grad_i(a, i) - prob.grad_i(b, i)) <= prob.R2 * np.linalg.norm(a - b) * (1 + 1e-12)
    # exact identity test of the SAGA instrumentation also holds on this instance
    _check_identity(prob)


def test_aligned_instance():
    prob = problems.make_aligned_instance(4, 20, 30.0)
    solvers.solve_optimum(prob)
    assert np.linalg.norm(prob.grad(prob.theta_star)) < 1e-10
    assert abs(prob.L_F - prob.R2) < 1e-12          # aligned: L_F = R^2
    rng = np.random.default_rng(4)
    th = rng.standard_normal(prob.d)
    assert np.allclose(prob.all_grads(th).mean(0), prob.grad(th), atol=1e-12)
    assert np.allclose(prob.F_batch(np.vstack([th, 2 * th])), [prob.F(th), prob.F(2 * th)])
    # O(m) chain products agree with the dense matrix, and the gradient with finite differences
    assert np.allclose(prob.grad(th), prob.A @ th - np.eye(prob.d)[0] * prob.c, atol=1e-10)
    eps = 1e-6
    e = np.zeros(prob.d)
    e[3] = eps
    assert abs((prob.F(th + e) - prob.F(th - e)) / (2 * eps) - prob.grad(th)[3]) < 1e-6
    # geometric minimizer x*_j = q^j for zeta = zeta_Q (single chain, Q = kappa)
    q = (np.sqrt(30.0) - 1) / (np.sqrt(30.0) + 1)
    assert np.allclose(prob.theta_star, q ** np.arange(1, prob.d + 1), atol=1e-10)
    # Prop. 5.9 budget: (3n + 16 kappa/3) log((1 + n/4)/eps) iterations suffice for E D_t <= eps D_0
    budget = int((3 * prob.n + 16 * prob.kappa / 3) * np.log((1 + prob.n / 4) / 1e-6)) + 1
    r = algorithms.saga(prob, np.zeros(prob.d), n_iters=budget, seed=0, record_every=4, record_F=False)
    assert r["D"][-1] < 1e-6 * r["D"][0]


def test_stochastic_methods_converge():
    prob = _small_logistic(n=80, d=6, kappa=60.0, alpha=0.5)
    theta0 = np.zeros(prob.d)
    D0 = float(np.sum((theta0 - prob.theta_star) ** 2))
    r = algorithms.lsvrg(prob, theta0, gamma=1.0 / (6 * prob.R2), passes=40, seed=0, record_every=8)
    assert r["D"][-1] < 1e-6 * D0
    r = algorithms.lkatyusha(prob, theta0, passes=40, seed=0, record_every=8)
    assert r["D"][-1] < 1e-6 * D0
    q = np.sqrt(prob.R2 / prob.mu)
    r = algorithms.nesterov_svrg(prob, theta0, gamma=1.0 / (6 * prob.R2), beta=(q - 1) / (q + 1), passes=40, seed=0, record_every=8)
    assert np.isfinite(r["D"][-1])
    r = algorithms.sgd(prob, theta0, passes=5, seed=0)
    assert np.isfinite(r["D"][-1])


def test_lkatyusha_without_negative_momentum_well_conditioned():
    """L-Katyusha with theta_2 = 0 (no anchor, default eta = 1/(3 theta_1)) stays finite and converges on
    well-conditioned problems (kappa = 5 and 20); it diverges on the hard instance for kappa >= 1e3 (A2, A4)."""
    for kappa in (5.0, 20.0):
        prob = _small_logistic(n=80, d=6, kappa=kappa, alpha=0.5)
        r = algorithms.lkatyusha(prob, np.zeros(prob.d), theta2=0.0, passes=30, seed=0, record_every=8)
        assert r["theta2"] == 0.0 and r["stopped"] is None
        assert np.all(np.isfinite(r["D"]))
        assert r["D"][-1] < 1e-10 * r["D"][0], (kappa, r["D"][-1] / r["D"][0])


def test_naive_nesterov_svrg_converges():
    """The naive scheme (Nesterov momentum beta = (sqrt(kappa)-1)/(sqrt(kappa)+1) with the loopless-SVRG estimator,
    step 1/(6R^2)) converges on an easy instance: ||theta - eta_*||^2 decreases by more than ten orders of magnitude."""
    prob = _small_logistic(n=80, d=6, kappa=60.0, alpha=0.5)
    q = np.sqrt(prob.kappa)
    for seed in (0, 1):
        r = algorithms.nesterov_svrg(prob, np.zeros(prob.d), gamma=1.0 / (6 * prob.R2), beta=(q - 1) / (q + 1), passes=40,
                                     seed=seed, record_every=8)
        assert r["stopped"] is None and np.all(np.isfinite(r["D"]))
        assert r["D"][-1] < 1e-10 * r["D"][0], (seed, r["D"][-1] / r["D"][0])
        # accounting: n initially, 2 per iteration, n per refresh of the reference point (3 per iteration in expectation)
        assert (r["grad_evals"][-1] - prob.n - 2 * 40 * prob.n) % prob.n == 0


def test_sgd_decreases():
    """SGD with the decreasing step gamma_t = 1/(R^2 + mu t): the error decreases (O(1/t) regime, no linear rate)."""
    prob = _small_logistic(n=80, d=6, kappa=60.0, alpha=0.5)
    r = algorithms.sgd(prob, np.zeros(prob.d), passes=20, seed=0, record_every=8)
    D, Fgap = r["D"], r["Fgap"]
    k = len(D) // 4
    assert np.all(np.isfinite(D)) and r["gamma0"] == 1.0 / prob.R2
    assert D[-1] < 0.1 * D[0] and Fgap[-1] < 0.1 * Fgap[0]
    assert D[-k:].mean() < 0.5 * D[:k].mean()                 # last quarter of the run against the first quarter


def test_lkatyusha_exact_identity_and_lemma():
    """E[Psi_{k+1} | state] from lkatyusha_state_stats equals the average over all indices of the realized
    Psi_{k+1}; the variance-bound ratio lies in (0,1]; the contraction max{c_Z, c_Y, c_W} given by Lemma 10 of
    Kovalev et al. (ALT 2020; Lemma 4.5 in the arXiv version 1901.08689) holds (rho_exact <= rho_lemma)."""
    prob = _small_logistic(n=50, d=6, kappa=80.0)
    rng = np.random.default_rng(5)
    n, d = prob.n, prob.d
    prm = bounds.lkatyusha_params(n, prob.R2, prob.mu)
    th1, th2, eta, p, L, mu = prm["theta1"], prm["theta2"], prm["eta"], prm["p"], prob.R2, prob.mu
    es = eta * prm["sigma"]
    y = prob.theta_star + 0.7 * rng.standard_normal(d)
    z = prob.theta_star + 0.5 * rng.standard_normal(d)
    w = prob.theta_star + 0.9 * rng.standard_normal(d)
    st = instrumentation.lkatyusha_state_stats(prob, y, z, w, th1, th2, eta, L, mu, p)
    assert 0 < st["r_var"] <= 1 + 1e-12
    assert st["rho_exact"] <= st["rho_lemma"] + 1e-12
    x = th1 * z + th2 * w + (1 - th1 - th2) * y
    Gw = prob.all_grads(w)
    gw = Gw.mean(0)
    cZ = L * (1 + es) / (2 * eta)
    vals = []
    for i in range(n):
        g = prob.grad_i(x, i) - Gw[i] + gw
        z1 = (es * x + z - (eta / L) * g) / (1 + es)
        y1 = x + th1 * (z1 - z)
        Z1 = cZ * np.sum((z1 - prob.theta_star) ** 2)
        Y1 = (prob.F(y1) - prob.F_star) / th1
        W1_new = th2 * (1 + th1) / (p * th1) * (prob.F(y) - prob.F_star)     # w <- y with prob p
        W1_old = th2 * (1 + th1) / (p * th1) * (prob.F(w) - prob.F_star)
        vals.append(Z1 + Y1 + p * W1_new + (1 - p) * W1_old)
    assert abs(np.mean(vals) - st["EPsi_next"]) < 1e-9 * max(1.0, st["Psi"])
    # the instrumented run returns records
    r = algorithms.lkatyusha(prob, np.zeros(d), passes=4, seed=0, record_every=10, instrument_every=25)
    assert len(r["instr"]) > 0 and all(q["rho_exact"] <= q["rho_lemma"] + 1e-12 for q in r["instr"])


def test_full_gradient_methods():
    prob = _small_logistic(n=80, d=6, kappa=60.0, alpha=0.5)
    theta0 = np.zeros(prob.d)
    D0 = float(np.sum((theta0 - prob.theta_star) ** 2))
    g = algorithms.gd(prob, theta0, n_iters=2000)
    a = algorithms.agd(prob, theta0, n_iters=2000)
    assert g["D"][-1] < 1e-8 * D0 and a["D"][-1] < 1e-8 * D0
    assert algorithms.grads_to_accuracy(a, 1e-6) < algorithms.grads_to_accuracy(g, 1e-6)


def test_bounds_helpers():
    best = bounds.saga_optimal_params(n=1000, R2=1.0, mu=1e-3)
    assert best["gain_over_textbook"] >= 1.0
    assert bounds.saga_rate(1000, 1.0, 1e-3) == 1 - min(1 / 3000, 3e-3 / 16)
    rho = bounds.lkatyusha_rate(1000, 1.0, 1e-3)
    assert 0 < rho < 1
    prob = _small_logistic()
    r = algorithms.saga(prob, np.zeros(prob.d), passes=10, seed=0, record_every=4, instrument_every=20)
    tb = analysis.trajectory_bounds(r)
    for k, v in tb.items():
        assert np.all(np.isfinite(v)) and v[-1] < v[0]
    att = analysis.attribution(r["instr"], r["gamma"], r["Delta"], r["R2"], r["mu"], r["n"])
    assert abs(np.nansum(att["shares"]) - 1) < 1e-8
    assert att["speedup_total"] >= 1.0


def test_trajectory_bounds_general_constants():
    """For (gamma, Delta) other than the textbook pair, trajectory_bounds uses the factor of inequality
    ineq:B3-traj-general of the B3 note, max{rho_M, 1 - 2 gamma c_D mu_eff} with rho_M = 1 - 1/n + 2 gamma^2/(n Delta)
    (not the textbook 1 - 1/(3n)), and these a-posteriori factors dominate the exact conditional contractions."""
    prob = _small_logistic()
    n, R2, mu = prob.n, prob.R2, prob.mu
    gamma = 1.0 / (4 * R2)
    Delta = 2.2 * gamma ** 2                                  # admissible (2 gamma^2 < Delta, c_D > 0), not textbook
    cD, rho_D, rho_M = instrumentation.rate_coefficients(gamma, Delta, R2, mu, n)
    assert cD > 0 and rho_M > 1 - 1 / (3 * n) and rho_D < rho_M  # the memory factor binds for every mu_eff >= mu
    r = algorithms.saga(prob, np.zeros(prob.d), gamma=gamma, Delta=Delta, passes=10, seed=0, record_every=4, instrument_every=4)
    assert r["Delta"] == Delta
    for q in r["instr"]:
        Le = min(q["L_eff"], R2)
        factor = max(rho_M, 1 - 2 * gamma * (1 - gamma * Le - Le * Delta / (2 * gamma)) * q["mu_eff"])
        assert q["rho_exact"] <= factor + 1e-12
    tb = analysis.trajectory_bounds(r)
    grid = np.asarray(r["iters"], dtype=np.float64)
    assert np.allclose(tb["mu_eff"], r["V"][0] * rho_M ** grid, rtol=1e-9, atol=0)
    assert np.all(tb["mu_eff"] >= tb["mu_eff+L_eff"] * (1 - 1e-12))
    assert np.all(tb["mu_eff+L_eff"] >= tb["exact"] * (1 - 1e-9))


def test_loglog_slope_interval():
    x = np.repeat([1e2, 1e3, 1e4, 1e5], 3)
    f = analysis.loglog_slope(x, 5.0 * x ** 0.5)
    assert abs(f["slope"] - 0.5) < 1e-12 and f["half_width"] < 1e-8 and f["n_points"] == 12
    rng = np.random.default_rng(0)
    y = 5.0 * x ** 0.5 * np.exp(0.1 * rng.standard_normal(x.size))
    y[0] = np.nan                                    # missing points are dropped
    g = analysis.loglog_slope(x, y, xmin=1e3)
    assert g["n_points"] == 9 and g["n_x"] == 3 and g["ci_low"] < g["slope"] < g["ci_high"]
    # balanced replicates: same slope as the fit to the per-x geometric means
    xm = np.array([1e3, 1e4, 1e5])
    gm = np.exp(np.log(y[3:]).reshape(3, 3).mean(1))
    assert abs(g["slope"] - np.polyfit(np.log(xm), np.log(gm), 1)[0]) < 1e-12


if __name__ == "__main__":
    import inspect
    fns = [f for name, f in sorted(globals().items()) if name.startswith("test_") and inspect.isfunction(f)]
    for f in fns:
        f()
        print("ok", f.__name__)
    print("all tests passed")
