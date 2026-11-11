"""Finite-sum problems F(theta) = (1/n) sum_i f_i(theta) with exact constants.

Notation follows Bach, *Learning Theory from First Principles*, Sec. 5.4.4:
  * each f_i is R^2-smooth  (R2 = max_i smoothness constant of f_i),
  * F is mu-strongly convex (mu = global strong-convexity constant),
  * kappa = R^2 / mu.
For l2-regularized logistic regression f_i(theta) = log(1 + exp(-y_i x_i^T theta)) + (lam/2)||theta||^2,
the per-sample smoothness constant is ||x_i||^2 / 4 + lam (the logistic second derivative is <= 1/4),
and the global strong-convexity constant of F is exactly lam.
"""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------------------------
# numerics helpers
# ----------------------------------------------------------------------------------------------
def sigmoid(u):
    """Numerically stable logistic function sigma(u) = 1/(1+exp(-u)), vectorised."""
    u = np.asarray(u, dtype=np.float64)
    out = np.empty_like(u)
    pos = u >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-u[pos]))
    eu = np.exp(u[~pos])
    out[~pos] = eu / (1.0 + eu)
    return out


def logistic_loss(u):
    """log(1 + exp(-u)), stable."""
    return np.logaddexp(0.0, -np.asarray(u, dtype=np.float64))


def _sigmoid_scalar(u):
    if u >= 0:
        return 1.0 / (1.0 + np.exp(-u))
    eu = np.exp(u)
    return eu / (1.0 + eu)


# ----------------------------------------------------------------------------------------------
# base class
# ----------------------------------------------------------------------------------------------
class FiniteSumProblem:
    """Interface shared by all problems. Attributes: n, d, R2, mu, kappa, L_F (smoothness of F),
    theta_star / F_star / grads_star (filled by `set_optimum`)."""

    n: int
    d: int
    R2: float
    mu: float
    L_F: float

    def F(self, theta):
        raise NotImplementedError

    def grad(self, theta):
        raise NotImplementedError

    def grad_i(self, theta, i):
        raise NotImplementedError

    def all_grads(self, theta):
        raise NotImplementedError

    def hessian(self, theta):
        raise NotImplementedError

    @property
    def kappa(self):
        return self.R2 / self.mu

    def set_optimum(self, theta_star):
        self.theta_star = np.asarray(theta_star, dtype=np.float64).copy()
        self.F_star = self.F(self.theta_star)
        self.grads_star = self.all_grads(self.theta_star)       # f'_i(eta_*), n x d
        self.grad_star_norm = float(np.linalg.norm(self.grads_star.mean(0)))
        return self


# ----------------------------------------------------------------------------------------------
# l2-regularized logistic regression
# ----------------------------------------------------------------------------------------------
class LogisticProblem(FiniteSumProblem):
    """f_i(theta) = log(1 + exp(-y_i x_i^T theta)) + (lam/2) ||theta||^2, labels y_i in {-1, +1}."""

    def __init__(self, X, y, lam, name="logistic"):
        self.X = np.ascontiguousarray(X, dtype=np.float64)
        self.y = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
        assert set(np.unique(self.y)) <= {-1.0, 1.0}
        self.lam = float(lam)
        self.n, self.d = self.X.shape
        self.sqnorms = np.einsum("ij,ij->i", self.X, self.X)
        self.L_i = self.sqnorms / 4.0 + self.lam        # per-sample smoothness constants
        self.R2 = float(self.L_i.max())                 # "R^2" of Prop. 5.9
        self.mu = self.lam                              # global strong convexity of F
        self.Sigma_hat = self.X.T @ self.X / self.n     # empirical second-moment matrix
        self.cov_eigs = np.linalg.eigvalsh(self.Sigma_hat)
        self.L_F = float(self.cov_eigs[-1] / 4.0 + self.lam)   # smoothness constant of F (global bound)
        self.name = name
        self.theta_star = None
        self.F_star = None
        self.grads_star = None

    # --- objective / derivatives -----------------------------------------------------------
    def margins(self, theta):
        return self.y * (self.X @ theta)

    def F(self, theta):
        m = self.margins(theta)
        return float(np.mean(logistic_loss(m)) + 0.5 * self.lam * float(theta @ theta))

    def f_i(self, theta, i):
        m = self.y[i] * float(self.X[i] @ theta)
        return float(np.logaddexp(0.0, -m) + 0.5 * self.lam * float(theta @ theta))

    def F_batch(self, Theta):
        """F evaluated at every row of Theta (k x d) -> (k,)."""
        Theta = np.atleast_2d(Theta)
        M = self.y[:, None] * (self.X @ Theta.T)                 # n x k margins
        return logistic_loss(M).mean(0) + 0.5 * self.lam * np.einsum("kj,kj->k", Theta, Theta)

    def grad(self, theta):
        m = self.margins(theta)
        s = sigmoid(-m)                                  # sigma(-m) = 1 - sigma(m)
        return self.X.T @ (-self.y * s) / self.n + self.lam * theta

    def grad_i(self, theta, i):
        xi = self.X[i]
        m = self.y[i] * float(xi @ theta)
        s = _sigmoid_scalar(-m)
        return (-self.y[i] * s) * xi + self.lam * theta

    def all_grads(self, theta):
        m = self.margins(theta)
        s = sigmoid(-m)
        return (-(self.y * s))[:, None] * self.X + self.lam * theta[None, :]

    def sigma_prime(self, theta):
        """Second derivative of the logistic loss at each sample: sigma(m) sigma(-m) <= 1/4."""
        m = self.margins(theta)
        return sigmoid(m) * sigmoid(-m)

    def hessian(self, theta):
        w = self.sigma_prime(theta)
        return (self.X.T * w) @ self.X / self.n + self.lam * np.eye(self.d)

    def hessian_i_sq_mean(self, theta):
        """(1/n) sum_i f''_i(theta)^2  (used for the limit of L_eff)."""
        w = self.sigma_prime(theta)
        H = np.zeros((self.d, self.d))
        I = np.eye(self.d)
        for i in range(self.n):
            Hi = w[i] * np.outer(self.X[i], self.X[i]) + self.lam * I
            H += Hi @ Hi
        return H / self.n

    # --- geometry summaries -----------------------------------------------------------------
    def leverage_scores(self, lam=None):
        """||x_i||^2_{(Sigma_hat + lam I)^{-1}} = n * (ridge leverage score of x_i)."""
        lam = self.lam if lam is None else lam
        P = self.Sigma_hat + lam * np.eye(self.d)
        sol = np.linalg.solve(P, self.X.T)                 # d x n
        return np.einsum("ij,ji->i", self.X, sol)

    def d_eff(self, lam=None):
        lam = self.lam if lam is None else lam
        return float(np.sum(self.cov_eigs / (self.cov_eigs + lam)))

    def summary(self):
        s = {
            "n": self.n, "d": self.d, "lam": self.lam, "R2": self.R2, "mu": self.mu,
            "kappa": self.kappa, "L_F": self.L_F, "kappa_F": self.L_F / self.mu,
            "cov_cond": float(self.cov_eigs[-1] / max(self.cov_eigs[0], 1e-300)),
            "max_sqnorm": float(self.sqnorms.max()), "mean_sqnorm": float(self.sqnorms.mean()),
            "d_eff": self.d_eff(),
        }
        if self.theta_star is not None:
            H = self.hessian(self.theta_star)
            ev = np.linalg.eigvalsh(H)
            sp = self.sigma_prime(self.theta_star)
            s.update({
                "F_star": self.F_star, "theta_star_norm": float(np.linalg.norm(self.theta_star)),
                "hess_star_min": float(ev[0]), "hess_star_max": float(ev[-1]),
                "kappa_local": float(ev[-1] / ev[0]), "mu_local_over_mu": float(ev[0] / self.mu),
                "sigma_prime_min": float(sp.min()), "sigma_prime_mean": float(sp.mean()),
                "sigma_prime_max": float(sp.max()), "grad_star_norm": self.grad_star_norm,
            })
        return s


def covariance_spectrum(d, alpha, kind="power"):
    """Eigenvalues of the feature covariance.
    kind='power'    : lambda_j = j^{-alpha}  (condition number d^alpha; alpha = 0 is isotropic)
    kind='geometric': lambda_j = alpha^{-(j-1)/(d-1)}  (condition number alpha >= 1)
    """
    j = np.arange(1, d + 1, dtype=np.float64)
    if kind == "power":
        return j ** (-float(alpha))
    if kind == "geometric":
        cond = float(alpha)
        return cond ** (-(j - 1) / (d - 1))
    raise ValueError(kind)


def make_logistic(n, d, kappa, alpha=0.0, seed=0, snr=2.0, noise=1.0, R=1.0, rotate=True,
                  spectrum="power", name=None):
    """Synthetic l2-regularized logistic regression with controlled constants.

    * features x_i = Sigma^{1/2} g_i, g_i ~ N(0, I_d), Sigma with spectrum `covariance_spectrum(d, alpha)`
      (alpha = 0: isotropic; larger alpha: more anisotropic), randomly rotated, then globally rescaled so
      that max_i ||x_i|| = R;
    * labels y_i = sign(x_i^T w + noise * eps_i), with w scaled so that std(X w) = snr;
    * regularization lam chosen so that kappa = R2 / mu exactly, with R2 = max_i ||x_i||^2/4 + lam
      and mu = lam, i.e. lam = (R^2/4) / (kappa - 1).
    """
    assert kappa > 1
    rng = np.random.default_rng(seed)
    ev = covariance_spectrum(d, alpha, spectrum)
    G = rng.standard_normal((n, d))
    X = G * np.sqrt(ev)[None, :]
    if rotate:
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        X = X @ Q.T
    X *= R / np.sqrt(np.einsum("ij,ij->i", X, X).max())
    w = rng.standard_normal(d)
    logits = X @ w
    logits *= snr / logits.std()
    y = np.sign(logits + noise * rng.standard_normal(n))
    y[y == 0] = 1.0
    lam = (R ** 2 / 4.0) / (kappa - 1.0)
    prob = LogisticProblem(X, y, lam, name=name or f"logistic(n={n},d={d},kappa={kappa:g},alpha={alpha:g},seed={seed})")
    prob.meta = dict(n=n, d=d, kappa=kappa, alpha=alpha, seed=seed, snr=snr, noise=noise, R=R, spectrum=spectrum)
    return prob


def make_logistic_book(n, d=40, seed=0, noise=1.0):
    """The setting of Bach's Figure 5.4: Gaussian inputs, y = sign(linear + Gaussian noise), mu = R^2/n
    (with R^2 = max_i ||x_i||^2, the book's bounded-features constant)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    w = rng.standard_normal(d) / np.sqrt(d)
    y = np.sign(X @ w + noise * rng.standard_normal(n))
    y[y == 0] = 1.0
    R2_feat = float(np.einsum("ij,ij->i", X, X).max())
    lam = R2_feat / n
    prob = LogisticProblem(X, y, lam, name=f"book(n={n},d={d},seed={seed})")
    prob.meta = dict(n=n, d=d, seed=seed, noise=noise, R2_feat=R2_feat)
    return prob


# ----------------------------------------------------------------------------------------------
# block-separable quadratic finite sums (hard instance of the lower bound)
# ----------------------------------------------------------------------------------------------
class BlockQuadraticProblem(FiniteSumProblem):
    """f_i(theta) = n * [ 1/2 theta_{B_i}^T A_i theta_{B_i} - b_i^T theta_{B_i} ], blocks B_i disjoint of size m.

    F(theta) = sum_i [ 1/2 theta_{B_i}^T A_i theta_{B_i} - b_i^T theta_{B_i} ] is block separable;
    each f_i is (n * lambda_max(A_i))-smooth, F is (min_i lambda_min(A_i))-strongly convex,
    and the minimizer is eta_* = concat_i A_i^{-1} b_i (exact).
    """

    def __init__(self, A_blocks, b_blocks, name="blockquad", ridge=0.0):
        """`ridge` adds the shared regularizer (ridge/2)||theta||^2 to EVERY f_i (the normalisation of the A1 note
        and of Agarwal & Bottou), so that F = sum_i[...] + (ridge/2)||theta||^2."""
        self.A = np.ascontiguousarray(np.asarray(A_blocks, dtype=np.float64))   # n x m x m
        self.b = np.ascontiguousarray(np.asarray(b_blocks, dtype=np.float64))   # n x m
        self.n, self.m, _ = self.A.shape
        self.d = self.n * self.m
        self.ridge = float(ridge)
        eigs = np.array([np.linalg.eigvalsh(Ai) for Ai in self.A])
        self.block_lmax = eigs[:, -1]
        self.block_lmin = eigs[:, 0]
        self.L_i = self.n * self.block_lmax + self.ridge
        self.R2 = float(self.L_i.max())
        self.mu = float(self.block_lmin.min() + self.ridge)
        self.L_F = float(self.block_lmax.max() + self.ridge)
        self.name = name
        self.theta_star = None
        self.F_star = None
        self.grads_star = None

    def _blocks(self, theta):
        return theta.reshape(self.n, self.m)

    def F(self, theta):
        T = self._blocks(theta)
        AT = np.einsum("ijk,ik->ij", self.A, T)
        return float(0.5 * np.einsum("ij,ij->", T, AT) - np.einsum("ij,ij->", self.b, T) + 0.5 * self.ridge * float(theta @ theta))

    def F_batch(self, Theta):
        Theta = np.atleast_2d(Theta)
        return np.array([self.F(th) for th in Theta])

    def grad(self, theta):
        T = self._blocks(theta)
        return (np.einsum("ijk,ik->ij", self.A, T) - self.b).reshape(-1) + self.ridge * theta

    def grad_i(self, theta, i):
        g = self.ridge * theta if self.ridge else np.zeros(self.d)
        sl = slice(i * self.m, (i + 1) * self.m)
        g[sl] += self.n * (self.A[i] @ theta[sl] - self.b[i])
        return g

    def all_grads(self, theta):
        T = self._blocks(theta)
        G = np.zeros((self.n, self.d)) + (self.ridge * theta)[None, :]
        blocks = self.n * (np.einsum("ijk,ik->ij", self.A, T) - self.b)
        for i in range(self.n):
            G[i, i * self.m:(i + 1) * self.m] += blocks[i]
        return G

    def hessian(self, theta=None):
        H = self.ridge * np.eye(self.d)
        for i in range(self.n):
            sl = slice(i * self.m, (i + 1) * self.m)
            H[sl, sl] += self.A[i]
        return H

    def exact_minimizer(self):
        I = np.eye(self.m)
        return np.concatenate([np.linalg.solve(self.A[i] + self.ridge * I, self.b[i]) for i in range(self.n)])

    def summary(self):
        return {"n": self.n, "m": self.m, "d": self.d, "R2": self.R2, "mu": self.mu, "kappa": self.kappa,
                "L_F": self.L_F, "kappa_F": self.L_F / self.mu}


def nesterov_chain_matrix(m):
    """Tridiagonal matrix T with 2 on the diagonal and -1 off-diagonal (eigenvalues in (0, 4))."""
    return 2.0 * np.eye(m) - np.eye(m, k=1) - np.eye(m, k=-1)


def make_hard_instance(n, m, kappa, seed=0, scale=1.0, jitter=0.0):
    """Block-separable Nesterov-chain quadratic: the hard instance of the finite-sum lower bound.

    Block function h(x) = c (1/2 x^T T x - e_1^T x) + (mu_b/2) ||x||^2 (Nesterov's chain), so that
    A_i = c T + mu_b I has lambda_max(A_i) < 4c + mu_b and lambda_min(A_i) > mu_b.
    With f_i = n h_i, the book's constants are R^2 = n lambda_max(A_i) and mu = min_i lambda_min(A_i);
    c is solved numerically so that kappa = R^2 / mu holds exactly (requires kappa > n).
    `jitter` optionally perturbs b_i to randomise the minimizer across blocks.
    """
    rng = np.random.default_rng(seed)
    T = nesterov_chain_matrix(m)
    evT = np.linalg.eigvalsh(T)
    mu_b = 1.0 * scale
    # lambda_max(A) = c evT_max + mu_b ; lambda_min(A) = c evT_min + mu_b ; kappa = n lambda_max / lambda_min
    #   -> c (n evT_max - kappa evT_min) = mu_b (kappa - n)
    denom = n * evT[-1] - kappa * evT[0]
    assert kappa > n and denom > 0, "need kappa > n and m large enough (kappa * lambda_min(T) < n * lambda_max(T))"
    c = mu_b * (kappa - n) / denom
    A = c * T + mu_b * np.eye(m)
    A_blocks = np.repeat(A[None], n, axis=0)
    e1 = np.zeros(m)
    e1[0] = c                                   # minus the gradient of -c e_1^T x
    b_blocks = np.repeat(e1[None], n, axis=0)
    if jitter > 0:
        b_blocks = b_blocks + jitter * c * rng.standard_normal((n, m))
    prob = BlockQuadraticProblem(A_blocks, b_blocks, name=f"hard(n={n},m={m},kappa={kappa:g})")
    prob.meta = dict(n=n, m=m, kappa=kappa, seed=seed, c=c, mu_b=mu_b, L_h=4 * c + mu_b)
    return prob


def chain_matrix(m, zeta=2.0):
    """A_m(zeta): tridiagonal with 2 on the diagonal except the last entry zeta, -1 off-diagonal (A1 note)."""
    A = nesterov_chain_matrix(m)
    A[-1, -1] = zeta
    return A


def make_hard_instance_shared(n, m, kappa, mu=1.0, zeta="Q"):
    """The hard instance of the A1 note (shared regularizer, as in Agarwal & Bottou 2015):
        f_i(theta) = c phi(theta^{(i)}) + (mu/2)||theta||^2,  phi(x) = 1/2 <A_m(zeta) x, x> - x_1,  c = mu (kappa - 1)/4,
    so that each f_i is (mu kappa)-smooth (up to the factor lambda_max(A_m)/4 < 1), F is mu-strongly convex, the
    per-block condition number is Q = 1 + (kappa - 1)/n, and with zeta = zeta_Q = (sqrt(Q)+3)/(sqrt(Q)+1) the minimizer
    is the geometric chain x*_j = q^j, q = (sqrt(Q)-1)/(sqrt(Q)+1), repeated on every block. Any kappa > 1 is allowed."""
    Q = 1.0 + (kappa - 1.0) / n
    if zeta == "Q":
        zeta = (np.sqrt(Q) + 3.0) / (np.sqrt(Q) + 1.0)
    c = mu * (kappa - 1.0) / 4.0
    A = chain_matrix(m, zeta)
    e1 = np.zeros(m)
    e1[0] = 1.0
    A_blocks = np.repeat((c / n * A)[None], n, axis=0)        # f_i = n [1/2 x^T (cA/n) x - (c/n) e_1^T x] = c phi(x)
    b_blocks = np.repeat((c / n * e1)[None], n, axis=0)
    prob = BlockQuadraticProblem(A_blocks, b_blocks, name=f"hard_shared(n={n},m={m},kappa={kappa:g})", ridge=mu)
    q = (np.sqrt(Q) - 1.0) / (np.sqrt(Q) + 1.0)
    prob.meta = dict(n=n, m=m, kappa=kappa, Q=Q, q=q, zeta=zeta, c=c, mu_reg=mu,
                     m_required=0.25 * np.sqrt(Q) * np.log(1e8))   # dimension requirement of the A1 note at eps = 1e-8
    return prob


class SharedQuadraticProblem(FiniteSumProblem):
    """f_i(theta) = 1/2 theta^T A_i theta - b_i^T theta on the SAME coordinates (no block structure).
    Used for the 'aligned' instance of the A1 note (all components equal to one chain): L_F = R^2."""

    def __init__(self, A_list, b_list, name="sharedquad"):
        self.A = np.ascontiguousarray(np.asarray(A_list, dtype=np.float64))     # n x d x d
        self.b = np.ascontiguousarray(np.asarray(b_list, dtype=np.float64))     # n x d
        self.n, self.d, _ = self.A.shape
        self.Abar = self.A.mean(0)
        self.bbar = self.b.mean(0)
        self.L_i = np.array([np.linalg.eigvalsh(Ai)[-1] for Ai in self.A])
        self.R2 = float(self.L_i.max())
        ev = np.linalg.eigvalsh(self.Abar)
        self.mu = float(ev[0])
        self.L_F = float(ev[-1])
        self.name = name
        self.theta_star = None
        self.F_star = None
        self.grads_star = None

    def F(self, theta):
        return float(0.5 * theta @ (self.Abar @ theta) - self.bbar @ theta)

    def F_batch(self, Theta):
        Theta = np.atleast_2d(Theta)
        return 0.5 * np.einsum("kj,kj->k", Theta, Theta @ self.Abar) - Theta @ self.bbar

    def grad(self, theta):
        return self.Abar @ theta - self.bbar

    def grad_i(self, theta, i):
        return self.A[i] @ theta - self.b[i]

    def all_grads(self, theta):
        return np.einsum("ijk,k->ij", self.A, theta) - self.b

    def hessian(self, theta=None):
        return self.Abar.copy()

    def exact_minimizer(self):
        return np.linalg.solve(self.Abar, self.bbar)

    def summary(self):
        return {"n": self.n, "d": self.d, "R2": self.R2, "mu": self.mu, "kappa": self.kappa, "L_F": self.L_F, "kappa_F": self.L_F / self.mu}


class AlignedChainProblem(FiniteSumProblem):
    """Aligned instance of the A1 note: all n components equal f(theta) = c phi(theta) + (mu/2)||theta||^2 with
    phi(x) = 1/2 <A_m(zeta) x, x> - x_1 on one chain in R^m, so that L_F = R^2 = c lambda_max(A_m) + mu ~ mu kappa.
    Products with the tridiagonal chain are computed in O(m) (no dense matrix in the inner loop)."""

    def __init__(self, n, m, c, mu, zeta, name="aligned"):
        self.n, self.d, self.c, self.mu_reg, self.zeta = n, m, float(c), float(mu), float(zeta)
        self.A = self.c * chain_matrix(m, zeta) + self.mu_reg * np.eye(m)     # dense copy for the Hessian / solves
        ev = np.linalg.eigvalsh(self.A)
        self.R2 = float(ev[-1])
        self.mu = float(ev[0])
        self.L_F = self.R2
        self.L_i = np.full(n, self.R2)
        self.name = name
        self.theta_star = None
        self.F_star = None
        self.grads_star = None

    def _chain_apply(self, x):
        """A_m(zeta) x in O(m)."""
        y = 2.0 * x
        y[:-1] -= x[1:]
        y[1:] -= x[:-1]
        y[-1] += (self.zeta - 2.0) * x[-1]
        return y

    def _grad_single(self, theta):
        g = self.c * self._chain_apply(theta) + self.mu_reg * theta
        g[0] -= self.c
        return g

    def F(self, theta):
        return float(0.5 * self.c * theta @ self._chain_apply(theta) - self.c * theta[0] + 0.5 * self.mu_reg * theta @ theta)

    def F_batch(self, Theta):
        Theta = np.atleast_2d(Theta)
        return np.array([self.F(th) for th in Theta])

    def grad(self, theta):
        return self._grad_single(theta)

    def grad_i(self, theta, i):
        return self._grad_single(theta)

    def all_grads(self, theta):
        return np.repeat(self._grad_single(theta)[None, :], self.n, axis=0)

    def hessian(self, theta=None):
        return self.A.copy()

    def exact_minimizer(self):
        e1 = np.zeros(self.d)
        e1[0] = self.c
        return np.linalg.solve(self.A, e1)

    def summary(self):
        return {"n": self.n, "d": self.d, "R2": self.R2, "mu": self.mu, "kappa": self.kappa, "L_F": self.L_F, "kappa_F": self.L_F / self.mu}


def make_aligned_instance(n, m, kappa, mu=1.0, zeta="Q"):
    """Aligned variant of the A1 note: all n components equal c phi(theta) + (mu/2)||theta||^2 on one chain in R^m,
    so that L_F = R^2 = mu kappa (up to lambda_max(A_m)/4): GD costs n kappa, AGD n sqrt(kappa), SAGA kappa.
    The dimension condition of the note for the single chain is m >= sqrt(kappa)/4 log(1/eps)."""
    Q = kappa
    if zeta == "Q":
        zeta = (np.sqrt(Q) + 3.0) / (np.sqrt(Q) + 1.0)
    c = mu * (kappa - 1.0) / 4.0
    prob = AlignedChainProblem(n, m, c, mu, zeta, name=f"aligned(n={n},m={m},kappa={kappa:g})")
    prob.meta = dict(n=n, m=m, kappa=kappa, zeta=zeta, c=c, mu_reg=mu, m_required=0.25 * np.sqrt(kappa) * np.log(1e8))
    return prob


def make_diag_quadratic(n, m, kappa, seed=0):
    """Easy comparison instance: block quadratics with eigenvalues log-spaced in [mu_b, L_h], random bases."""
    rng = np.random.default_rng(seed)
    mu_b = 1.0
    L_h = kappa * mu_b / n
    assert L_h > mu_b
    A_blocks, b_blocks = [], []
    for i in range(n):
        ev = np.exp(rng.uniform(np.log(mu_b), np.log(L_h), size=m))
        ev[0], ev[1] = mu_b, L_h
        Q, _ = np.linalg.qr(rng.standard_normal((m, m)))
        A_blocks.append((Q * ev) @ Q.T)
        b_blocks.append(rng.standard_normal(m))
    prob = BlockQuadraticProblem(np.array(A_blocks), np.array(b_blocks), name=f"diagquad(n={n},m={m},kappa={kappa:g})")
    prob.meta = dict(n=n, m=m, kappa=kappa, seed=seed)
    return prob
