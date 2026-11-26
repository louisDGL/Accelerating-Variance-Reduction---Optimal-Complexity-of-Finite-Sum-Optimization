"""Constants of the isotropic law.  q_{K,d} = E[sum_{i!=j} s_i s_j/(s_i+s_j)] / E[sum_i s_i]  (Z standard Gaussian K x d).
a1 = 3[(K-1)(d-1) - 2(K+d+1) q]/D,  a2 = 6[(Kd+2) q + (K-1)(d-1)]/D,  D = (K-1)(d-1)(K+2)(d+2),
a1 + a2 = 3(3+2q)/((K+2)(d+2))   (rank-one / i.i.d.-entry constant, = a_{K,d}/c_{K,d} of the paper).
Checks: Stein identity m3 = (K+d) m1 - E sum_{i!=j} s_i s_j/(s_i+s_j);  bounds 0 <= q <= (min(K,d)-1)/2;
sign of a1; Marchenko-Pastur limit kappa(gamma) = lim q/K, theta(gamma) = 6 kappa = lim d*(a1+a2).
"""
import numpy as np, json
from scipy import integrate
rng = np.random.default_rng(7)

def mc(K, d, S, chunk):
    acc = np.zeros(3); sq = np.zeros(3); n = 0
    for _ in range(S // chunk):
        Z = rng.standard_normal((chunk, K, d))
        s = np.linalg.svd(Z, compute_uv=False)
        m1 = s.sum(1); m3 = (s ** 3).sum(1)
        P = s[:, :, None] * s[:, None, :] / (s[:, :, None] + s[:, None, :])
        h = P.sum((1, 2)) - np.trace(P, axis1=1, axis2=2)
        X = np.stack([m1, m3, h], 1)
        acc += X.sum(0); sq += (X ** 2).sum(0); n += chunk
    mean = acc / n
    # delta-method SE of q = h/m1 from per-chunk is skipped; report crude SE
    return mean, np.sqrt((sq / n - mean ** 2) / n)

def consts(K, d, q):
    D = (K - 1) * (d - 1) * (K + 2) * (d + 2)
    a1 = 3 * ((K - 1) * (d - 1) - 2 * (K + d + 1) * q) / D
    a2 = 6 * ((K * d + 2) * q + (K - 1) * (d - 1)) / D
    return a1, a2

rows = []
print(" K   d      q      (K-1)/2  thr_a1    a1        a2      a1+a2   d*(a1+a2)  K*a1+a2  stein_resid")
for (K, d, S) in [(2, 2, 4e6), (2, 3, 4e6), (2, 8, 2e6), (2, 64, 4e5), (3, 3, 2e6), (3, 5, 2e6), (4, 4, 2e6), (4, 16, 1e6),
                  (8, 8, 1e6), (8, 32, 4e5), (16, 16, 4e5), (16, 64, 1e5), (32, 32, 1e5), (64, 64, 3e4), (32, 128, 3e4)]:
    S = int(S); mean, se = mc(K, d, S, min(S, 100000))
    m1, m3, h = mean; q = h / m1
    a1, a2 = consts(K, d, q)
    thr = (K - 1) * (d - 1) / (2 * (K + d + 1))
    print(f"{K:3d} {d:3d}  {q:8.4f} {(K-1)/2:7.2f} {thr:8.4f} {a1:9.5f} {a2:9.5f} {a1+a2:8.5f} {d*(a1+a2):8.4f} {K*a1+a2:9.5f}"
          f"  {(m3 - (K+d)*m1 + h)/m1:+.1e}")
    rows.append(dict(K=K, d=d, q=q, a1=a1, a2=a2, m1=m1, m3=m3))

# Marchenko-Pastur limit: singular values x = s/sqrt(d), K/d = gamma <= 1
def kappa(gamma):
    lo, hi = (1 - np.sqrt(gamma)), (1 + np.sqrt(gamma))
    dens = lambda x: np.sqrt(max((hi ** 2 - x * x) * (x * x - lo ** 2), 0.0)) / (np.pi * gamma * x)
    norm = integrate.quad(dens, lo, hi, limit=200)[0]
    m1 = integrate.quad(lambda x: x * dens(x), lo, hi, limit=200)[0]
    m3 = integrate.quad(lambda x: x ** 3 * dens(x), lo, hi, limit=200)[0]
    num = integrate.dblquad(lambda y, x: x * y / (x + y) * dens(x) * dens(y), lo, hi, lambda x: lo, lambda x: hi,
                            epsabs=1e-9, epsrel=1e-9)[0]
    return norm, m1, m3, num / m1

print("\nMarchenko-Pastur limit: gamma, normalisation, kappa(gamma)=lim q/K, theta=6 kappa, [6((1+g)mu1-mu3)/(g mu1)], a1-sign margin kappa-1/(2(1+g))")
for g in [1.0, 0.75, 0.5, 0.25, 0.1, 0.01]:
    nrm, m1, m3, kap = kappa(g)
    print(f"  {g:5.2f}  {nrm:.6f}  {kap:.6f}  {6*kap:.5f}  {6*((1+g)*m1-m3)/(g*m1):.5f}  {kap-1/(2*(1+g)):+.5f}")
print("  gamma=1 exact: kappa=2/5, theta=12/5")
json.dump(rows, open('../../../results/bias_law/checks/c02_constants.json', 'w'), indent=1)
