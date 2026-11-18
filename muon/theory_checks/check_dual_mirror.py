"""Numerical checks for Part 2 of notes_spectral_theory: polar steps as mirror descent on psi(G) = ||G||_* in gradient
space with mirror map F^* (the conjugate of F).

  P2.1  D_{F*}(0, grad F(W)) = F(W) - F_*   (Bregman divergence of the conjugate at the dual solution 0);
        dual sharpness ||grad F(W)||_*^2 >= 2 mu_op (F(W) - F_*).
  P2.2  exact representation of one nuclear-scaled step on a quadratic:
        D+/D = 1 - (mu_PL / L)(2 - J / L),  mu_PL = ||G||_*^2/(2D), J = <O, H O>;  J in [mu_op, L_op].
  P2.3  flow identities along small polar steps (ridge softmax, non-quadratic): dF/dt = -||G||_*, d||G||_*/dt = -J.
  P2.4  geometric schedule eta_t = eta_0 rho^{t/2}, rho = 1 - 1/kappa_op, eta_0 = sqrt(2 mu_op Dbar_0)/L_op
        gives F(W_t) - F_* <= Dbar_0 rho^t  (least squares with exact constants; ridge softmax with global bounds).
  P2.5  constant length: floor between mu_op eta^2/8 (every other iterate) and (kappa_op/8 + 1/2) L_op eta^2.
Run: .venv/Scripts/python muon/theory_checks/check_dual_mirror.py
"""
import numpy as np

rng = np.random.default_rng(20261116)


def polar(M, tol=1e-13):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    keep = s > tol * max(s[0], 1e-300)
    return U[:, keep] @ Vt[keep]


def nuc(M):
    return np.linalg.svd(M, compute_uv=False).sum()


# ---------------------------------------------------------------- ridge softmax (non-quadratic, strongly convex)
class Softmax:
    def __init__(self, n, d, K, lam, rng):
        X = rng.standard_normal((n, d)) * np.arange(1, d + 1.0) ** (-0.5)
        X /= np.max(np.linalg.norm(X, axis=1))
        W0 = rng.standard_normal((K, d)) * 3
        Z = X @ W0.T
        P = np.exp(Z - Z.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
        self.y = np.array([rng.choice(K, p=p) for p in P])
        self.X, self.n, self.d, self.K, self.lam = X, n, d, K, lam
        self.Yoh = np.eye(K)[self.y]
        S = X.T @ X / n
        ev = np.sort(np.linalg.eigvalsh(S))[::-1]
        r = min(K, d)
        self.L_op = 0.5 * ev[:r].sum() + lam * r          # global upper bound (Part 1)
        self.mu_op = lam                                   # global constant (Part 1)

    def probs(self, W):
        Z = self.X @ W.T
        P = np.exp(Z - Z.max(1, keepdims=True))
        return P / P.sum(1, keepdims=True)

    def F(self, W):
        Z = self.X @ W.T
        m = Z.max(1)
        return float(np.mean(m + np.log(np.exp(Z - m[:, None]).sum(1)) - Z[np.arange(self.n), self.y]) + 0.5 * self.lam * np.sum(W * W))

    def grad(self, W):
        return (self.probs(W) - self.Yoh).T @ self.X / self.n + self.lam * W

    def hvp(self, W, D):
        P = self.probs(W)
        A = self.X @ D.T                                   # n x K
        SA = P * A - P * np.sum(P * A, 1, keepdims=True)
        return SA.T @ self.X / self.n + self.lam * D

    def solve(self):
        W = np.zeros((self.K, self.d))
        for _ in range(100):                               # Newton with CG on the Hessian-vector product
            g = self.grad(W)
            if np.linalg.norm(g) < 1e-14:
                break
            D = np.zeros_like(W); r = -g.copy(); p = r.copy(); rs = np.sum(r * r)
            for _ in range(500):
                Hp = self.hvp(W, p)
                a = rs / np.sum(p * Hp)
                D += a * p; r -= a * Hp
                rs_new = np.sum(r * r)
                if rs_new < 1e-30:
                    break
                p = r + (rs_new / rs) * p; rs = rs_new
            t = 1.0
            while self.F(W + t * D) > self.F(W) + 1e-4 * t * np.sum(g * D) and t > 1e-8:
                t *= 0.5
            W = W + t * D
        return W


prob = Softmax(n=300, d=12, K=5, lam=1e-2, rng=rng)
Wst = prob.solve()
Fst = prob.F(Wst)
print(f"softmax: ||grad F(W*)||_F = {np.linalg.norm(prob.grad(Wst)):.1e}, L_op bound = {prob.L_op:.4f}, mu_op = {prob.mu_op:.4f}, kappa_op bound = {prob.L_op / prob.mu_op:.1f}")

print("== P2.1 dual sharpness ||G||_*^2 >= 2 mu_op (F - F*) and smoothness ||G||_*^2 <= 2 L_op (F - F*) at random points (softmax) ==")
ok = True
for _ in range(2000):
    W = Wst + rng.standard_normal(Wst.shape) * 10 ** rng.uniform(-3, 1)
    g2 = nuc(prob.grad(W)) ** 2
    gap = prob.F(W) - Fst
    ok &= g2 >= 2 * prob.mu_op * gap * (1 - 1e-9) and g2 <= 2 * prob.L_op * gap * (1 + 1e-9) + 1e-18
print("  holds at 2000 random points:", ok)

print("== P2.3 flow identities along small polar steps (softmax) ==")
W = np.zeros_like(Wst); eta = 1e-5
errs = []
for it in range(4000):
    G = prob.grad(W); O = polar(G)
    J = np.sum(O * prob.hvp(W, O))
    Wn = W - eta * O
    dF = (prob.F(Wn) - prob.F(W)) / eta
    dpsi = (nuc(prob.grad(Wn)) - nuc(G)) / eta
    if it % 1000 == 0:
        errs.append((dF + nuc(G), dpsi + J, J))
    W = Wn
print("  (dF/dt + ||G||_*, d||G||_*/dt + J, J) at 4 times:", [(f"{a:+.1e}", f"{b:+.1e}", f"{c:.4f}") for a, b, c in errs],
      f"with J in [mu_op, L_op] = [{prob.mu_op}, {prob.L_op:.4f}]")

print("== P2.4 geometric schedule on softmax with the global constants ==")
kap = prob.L_op / prob.mu_op
rho = 1 - 1 / kap
W = np.zeros_like(Wst)
D0 = prob.F(W) - Fst
eta0 = np.sqrt(2 * prob.mu_op * D0) / prob.L_op
ok = True; t = 0
Dt = D0
while Dt > 1e-13 * D0 and t < 200000:
    W = W - eta0 * rho ** (t / 2) * polar(prob.grad(W)); t += 1
    Dt = prob.F(W) - Fst
    ok &= Dt <= D0 * rho ** t * (1 + 1e-9) + 1e-15
print(f"  F(W_t) - F* <= D_0 rho^t for all t: {ok}; reached {Dt / D0:.1e} after {t} steps (bound {rho ** t:.1e}; rho = {rho:.5f})")
# same budget with the nuclear-scaled (Polyak) step
W = np.zeros_like(Wst); t2 = 0; Dt2 = D0
while Dt2 > 1e-13 * D0 and t2 < 200000:
    G = prob.grad(W)
    W = W - (nuc(G) / prob.L_op) * polar(G); t2 += 1
    Dt2 = prob.F(W) - Fst
print(f"  nuclear-scaled step with L_op: {t2} steps to 1e-13 (certified per-step factor {rho:.5f}; observed mean factor {(Dt2 / D0) ** (1 / t2):.5f})")

# ---------------------------------------------------------------- quadratic with exact constants
print("== P2.2 / P2.4 / P2.5 on a one-sided quadratic with exact constants ==")
K, d = 6, 5
lam = np.arange(1, d + 1.0) ** (-1.5)
V, _ = np.linalg.qr(rng.standard_normal((d, d)))
Sigma = (V * lam) @ V.T
L_op, mu_op = lam.sum(), lam.min()
kap = L_op / mu_op; rho = 1 - 1 / kap
E0 = rng.standard_normal((K, d))
Fq = lambda E: 0.5 * np.trace(E @ Sigma @ E.T)
# P2.2 representation
E = E0.copy(); ok = True
for _ in range(50):
    G = E @ Sigma; O = polar(G); J = np.trace(O @ Sigma @ O.T); D = Fq(E)
    muPL = nuc(G) ** 2 / (2 * D)
    En = E - (nuc(G) / L_op) * O
    ok &= abs(Fq(En) / D - (1 - (muPL / L_op) * (2 - J / L_op))) < 1e-10
    ok &= mu_op * (1 - 1e-12) <= J <= L_op * (1 + 1e-12) and muPL >= mu_op * (1 - 1e-12)
    E = En
print("  D+/D = 1 - (mu_PL/L)(2 - J/L), J in [mu_op, L_op], mu_PL >= mu_op over 50 steps:", ok)
# P2.4 geometric schedule
E = E0.copy(); D0 = Fq(E); eta0 = np.sqrt(2 * mu_op * D0) / L_op; ok = True
for t in range(3000):
    E = E - eta0 * rho ** (t / 2) * polar(E @ Sigma)
    ok &= Fq(E) <= D0 * rho ** (t + 1) * (1 + 1e-9)
print(f"  geometric schedule: F_t <= D_0 rho^t for 3000 steps: {ok} (kappa_op = {kap:.1f}; final ratio to the bound {Fq(E) / (D0 * rho ** 3000):.2e})")
# P2.5 constant length floor
for eta in [1e-1, 1e-2]:
    E = E0.copy()
    for t in range(40000):
        E = E - eta * polar(E @ Sigma)
    tail = []
    for t in range(2000):
        E = E - eta * polar(E @ Sigma); tail.append(Fq(E))
    tail = np.array(tail)
    pairmax = np.maximum(tail[:-1], tail[1:])
    print(f"  constant length eta = {eta}: stationary F in [{tail.min():.3e}, {tail.max():.3e}]; min over t of max(F_t, F_t+1) = {pairmax.min():.3e} "
          f">= mu_op eta^2/8 = {mu_op * eta ** 2 / 8:.3e}; upper bound (kappa/8 + 1/2) L eta^2 = {(kap / 8 + 0.5) * L_op * eta ** 2:.3e}")
