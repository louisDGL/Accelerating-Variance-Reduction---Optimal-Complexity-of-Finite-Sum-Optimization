"""Audit math-saga, controle 1 : identite (1) du papier (l. 217-221) par enumeration exacte des n indices.

Regression logistique ridge, n = 20, d = 5.  Implementation independante de vrlab/ :
  LHS = E_t[V_{t+1}] obtenu en executant reellement le pas SAGA pour chacun des n indices et en moyennant V ;
  RHS = c_D D + c_M delta M avec mu_eff, L_eff, r_sq, r_var calcules depuis leurs definitions (l. 213-216).
Controle aussi : I1-I6, bornes mu <= mu_eff <= L_eff <= R^2, valeurs pire cas, facteur 8/3 (c_D) et facteur 3 (c_M),
Prop. 5.9 (E_t V_{t+1} <= rho V_t), et (C3) : inf lambda_min(Hessienne) = lambda, mu_eff pres de l'optimum.

Lancer depuis la racine du depot :
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 paper/audit/checks/math-saga_check1_identity.py
"""
import numpy as np
from scipy.special import expit


def sigmoid(u):
    return expit(u)


class Logistic:
    def __init__(self, n, d, kappa, seed):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, d)) * (np.arange(1, d + 1) ** -0.5)[None, :]
        X /= np.sqrt((X ** 2).sum(1).max())            # max_i ||x_i||^2 = 1
        self.X, self.y = X, rng.choice([-1.0, 1.0], n)
        self.n, self.d = n, d
        # kappa = R^2 / mu with R^2 = max_i ||x_i||^2/4 + lam, mu = lam
        self.lam = 0.25 / (kappa - 1.0)
        self.R2 = 0.25 * (X ** 2).sum(1).max() + self.lam
        self.mu = self.lam

    def grads(self, th):                               # n x d matrix of f_i'(theta)
        m = self.y * (self.X @ th)
        return (-self.y * sigmoid(-m))[:, None] * self.X + self.lam * th[None, :]

    def hess(self, th):
        m = self.y * (self.X @ th)
        w = sigmoid(m) * sigmoid(-m)
        return (self.X * w[:, None]).T @ self.X / self.n + self.lam * np.eye(self.d)

    def solve(self):
        th = np.zeros(self.d)
        for _ in range(200):
            g = self.grads(th).mean(0)
            th = th - np.linalg.solve(self.hess(th), g)
        return th


def V_of(prob, th, z, ths, gs, delta):
    return float(((th - ths) ** 2).sum() + delta * ((z - gs) ** 2).sum())


def one_state(prob, th, z, ths, gs, gamma, delta):
    n = prob.n
    Gall = prob.grads(th)
    zbar = z.mean(0)
    # ---- LHS : enumeration brute, on execute le pas pour chaque indice
    lhs = 0.0
    for i in range(n):
        omega = Gall[i] - z[i] + zbar
        th2 = th - gamma * omega
        z2 = z.copy()
        z2[i] = Gall[i]
        lhs += V_of(prob, th2, z2, ths, gs, delta) / n
    # ---- RHS : formule (1) du papier, a partir des definitions
    D = float(((th - ths) ** 2).sum())
    M = float(((z - gs) ** 2).sum())
    Fp = Gall.mean(0)
    G = float((th - ths) @ Fp)
    S = float(((Gall - gs) ** 2).sum())
    mu_eff = G / D
    L_eff = (S / n) / G
    a = Gall - gs
    b = gs - z + zbar[None, :]
    Omega = float(((a + b) ** 2).sum(1).mean())
    B = float((b ** 2).sum(1).mean())
    r_sq = Omega / (2 * S / n + 2 * B)
    r_var = B / (M / n)
    R2, mu = prob.R2, prob.mu
    cD = 1 - mu_eff / (2 * R2) * (1 - L_eff / R2 * (r_sq / 4 + 3 / 8))
    cM = 1 - (1 - 2 / 3 * r_sq * r_var) / n
    rhs = cD * D + cM * delta * M
    rho = 1 - min(1 / (3 * n), 3 * mu / (16 * R2))
    checks = dict(
        I1=Omega <= 2 * S / n + 2 * B + 1e-15, I2=S / n <= R2 * G * (1 + 1e-12), I3=B <= M / n * (1 + 1e-12),
        I4=G >= mu * D * (1 - 1e-12), prop59=lhs <= rho * (D + delta * M) * (1 + 1e-12),
        order=(mu <= mu_eff * (1 + 1e-12)) and (mu_eff <= L_eff * (1 + 1e-12)) and (L_eff <= R2 * (1 + 1e-12)),
        r01=(0 <= r_sq <= 1 + 1e-12) and (0 <= r_var <= 1 + 1e-12),
        cD_le=cD <= 1 - 3 * mu / (16 * R2) + 1e-15, cM_le=cM <= 1 - 1 / (3 * n) + 1e-15)
    return dict(lhs=lhs, rhs=rhs, relerr=abs(lhs - rhs) / lhs, D=D, M=M, mu_eff=mu_eff, L_eff=L_eff, r_sq=r_sq,
                r_var=r_var, cD=cD, cM=cM, checks=checks,
                fac_D=(1 - L_eff / R2 * (r_sq / 4 + 3 / 8)) / (3 / 8),      # contribution de I1+I2 dans c_D, <= 8/3
                fac_M=3 * (1 - 2 / 3 * r_sq * r_var),                        # contribution de I1+I3 dans c_M, <= 3
                fac_M_I1only=3 * (1 - 2 / 3 * r_sq))                         # I1 seule (r_var = 1) dans c_M


def main():
    n, d = 20, 5
    print("=== Controle 1 : identite (1), n=%d d=%d, enumeration exacte ===" % (n, d))
    worst = 0.0
    allok = True
    for kappa in (5.0, 50.0, 500.0, 5000.0):
        for seed in (0, 1, 2):
            prob = Logistic(n, d, kappa, seed)
            ths = prob.solve()
            gs = prob.grads(ths)
            assert np.linalg.norm(gs.mean(0)) < 1e-13
            R2, mu = prob.R2, prob.mu
            gamma = 1 / (4 * R2)
            delta = 3 * gamma ** 2
            rng = np.random.default_rng(100 + seed)
            th = 3.0 * rng.standard_normal(d)
            th0 = th.copy()
            z = prob.grads(th)                          # z_i^(0) = f_i'(theta_0)
            # I6 a t = 0
            M0 = float(((z - gs) ** 2).sum())
            D0 = float(((th0 - ths) ** 2).sum())
            i6 = M0 <= n * R2 ** 2 * D0
            T = 40 * n
            recs = []
            for t in range(T):
                if t in (0, 1, 2, 3, 5, 10, n, 2 * n, 5 * n, 10 * n, 20 * n, 39 * n):
                    recs.append((t, one_state(prob, th, z, ths, gs, gamma, delta)))
                i = rng.integers(n)
                gi = prob.grads(th)[i]
                omega = gi - z[i] + z.mean(0)
                th = th - gamma * omega
                z[i] = gi
            # l'identite est controlee sur les etats non encore converges a la precision machine (D > 1e-16),
            # ou LHS et RHS ne sont pas domines par l'arrondi ; les inegalites le sont sur tous les etats
            mx = max(r["relerr"] for _, r in recs if r["D"] > 1e-16)
            nst = sum(1 for _, r in recs if r["D"] > 1e-16)
            worst = max(worst, mx)
            ok = all(all(r["checks"].values()) for _, r in recs) and i6
            allok &= ok
            fD = max(r["fac_D"] for _, r in recs)
            fM = max(r["fac_M"] for _, r in recs)
            fM1 = max(r["fac_M_I1only"] for _, r in recs)
            r0 = recs[0][1]
            print("kappa=%6.0f seed=%d  (%2d etats) max relerr |LHS-RHS|/LHS = %.2e   inegalites/ordre OK=%s   "
                  "max fac_D=%.3f (<=8/3=2.667)  max fac_M=%.3f (<=3)  [t=0: r_sq=%.3f r_var=%.3f fac_M=%.3f fac_M(I1 seule)=%.3f]"
                  % (kappa, seed, nst, mx, ok, fD, fM, r0["r_sq"], r0["r_var"], r0["fac_M"], r0["fac_M_I1only"]))
            # valeurs pire cas
            cD_w = 1 - mu / (2 * R2) * (1 - 1.0 * (1 / 4 + 3 / 8))
            cM_w = 1 - (1 - 2 / 3) / n
            assert abs(cD_w - (1 - 3 * mu / (16 * R2))) < 1e-15 and abs(cM_w - (1 - 1 / (3 * n))) < 1e-15
    print("pire erreur relative sur l'identite (1) : %.2e ; toutes les inegalites I1-I6, l'ordre mu<=mu_eff<=L_eff<=R^2 "
          "et Prop. 5.9 verifies : %s" % (worst, allok))

    print("\n=== (C3) constante globale de forte convexite = lambda ; mu_eff pres de l'optimum ===")
    prob = Logistic(n, d, 500.0, 0)
    ths = prob.solve()
    Hs = prob.hess(ths)
    ev, U = np.linalg.eigh(Hs)
    rng = np.random.default_rng(7)
    th1 = rng.standard_normal(d)
    print("lambda = %.6e ; lambda_min(H(eta*)) = %.6e ; min_i |x_i.th1| = %.3e" % (prob.lam, ev[0], np.abs(prob.X @ th1).min()))
    for tt in (1, 10, 100, 1000, 10000):
        print("  t=%6d  lambda_min(Hess(t*th1))/lambda - 1 = %.3e" % (tt, np.linalg.eigvalsh(prob.hess(tt * th1))[0] / prob.lam - 1))
    for eps in (1.0, 1e-1, 1e-2, 1e-3):
        vals = []
        for _ in range(2000):
            u = rng.standard_normal(d)
            u *= eps / np.linalg.norm(u)
            th = ths + u
            vals.append(float(u @ prob.grads(th).mean(0)) / eps ** 2)
        u = eps * U[:, 0]
        mflat = min(float(u @ prob.grads(ths + u).mean(0)), float((-u) @ prob.grads(ths - u).mean(0))) / eps ** 2
        print("  ||theta-eta*||=%.0e : min mu_eff (2000 directions aleatoires)/lmin(H*) = %.6f ; dans la direction la plus plate (+/-) = %.6f"
              % (eps, min(vals) / ev[0], mflat / ev[0]))


if __name__ == "__main__":
    main()
