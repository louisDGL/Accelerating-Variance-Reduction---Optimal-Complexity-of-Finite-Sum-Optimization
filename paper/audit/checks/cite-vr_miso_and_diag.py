"""cite-vr checks.
(1) The step theta - gamma (m + vartheta c), m = zbar + lam theta, c = grad g_j(theta) - z_j, with vartheta = 1/n and
    gamma = 1/lam, equals the MISO update with the lower-bound surrogates of Mairal (2015, eq. (3.8), mu = lam) on
    f_i = g_i + lam/2 ||.||^2 (ridge logistic components): checked on random runs.
(2) ((kappa-1)/(kappa+1))^2 < e^{-1} iff kappa < (sqrt(e)+1)/(sqrt(e)-1) (threshold of line 609).
Run from the project root: python3 paper/audit/checks/cite-vr_miso_and_diag.py
"""
import numpy as np

rng = np.random.RandomState(0)
n, d, lam = 20, 5, 0.3
X = rng.randn(n, d)
y = np.sign(rng.randn(n))


def grad_g(i, th):  # logistic loss of sample i
    z = y[i] * X[i] @ th
    return -y[i] * X[i] / (1.0 + np.exp(z))


def grad_f(i, th):
    return grad_g(i, th) + lam * th


worst = 0.0
for run in range(5):
    kap = rng.randn(n, d)                      # MISO anchors kappa_i (arbitrary start)
    th = np.mean(kap - np.array([grad_f(i, kap[i]) for i in range(n)]) / lam, axis=0)  # argmin of mean surrogate
    z = np.array([grad_g(i, kap[i]) for i in range(n)])  # SAGA-type table of grad g_i at the anchors
    for t in range(200):
        j = rng.randint(n)
        # MISO: refresh anchor j at theta, new theta = argmin (1/n) sum_i d_i, d_i(u) = f_i(k_i) + <grad f_i(k_i), u-k_i> + lam/2 |u-k_i|^2
        kap[j] = th.copy()
        th_miso = np.mean(kap - np.array([grad_f(i, kap[i]) for i in range(n)]) / lam, axis=0)
        # paper's form
        m = z.mean(axis=0) + lam * th
        c = grad_g(j, th) - z[j]
        th_paper = th - (1.0 / lam) * (m + c / n)
        z[j] = grad_g(j, th)
        worst = max(worst, np.max(np.abs(th_miso - th_paper)))
        th = th_miso
print("(1) max |MISO - paper form| over 1000 steps:", worst)

thr = (np.sqrt(np.e) + 1) / (np.sqrt(np.e) - 1)
ks = np.linspace(1.01, 10, 100000)
ok = ((ks - 1) / (ks + 1)) ** 2 < np.exp(-1)
print("(2) threshold (sqrt e + 1)/(sqrt e - 1) =", thr, "; largest kappa on grid with rho^2 < 1/e:", ks[ok].max())
