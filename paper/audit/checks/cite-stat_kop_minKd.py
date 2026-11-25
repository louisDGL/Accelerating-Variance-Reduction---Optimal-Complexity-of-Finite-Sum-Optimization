"""cite-stat: check of the statement attributed to sidford2018coordinate (paper line 354): kappa_op >= min(K, d).

Sidford-Tian (arXiv:1808.01278v2, Appendix A.1, Lemma A.1) only prove the ell_inf *range* bound
(a 1-strongly convex function in ell_inf on [-1,1]^n has range >= n/2).  The matrix statement used in the paper
follows from an averaging argument, checked here numerically:

  for a PSD quadratic form H on R^{K x d} (Hessian), with mu_op <= <D,HD>/||D||_op^2 <= L_op,
  (i)  rank-one D = u v^T, u, v uniform on spheres: ||D||_op = 1 and E <D,HD> = tr(H)/(Kd)  => mu_op <= tr(H)/(Kd)
  (ii) Haar partial isometry Q (all singular values 1): ||Q||_op = 1 and E <Q,HQ> = min(K,d) tr(H)/(Kd)
       => L_op >= min(K,d) tr(H)/(Kd)
  hence kappa_op = L_op/mu_op >= min(K,d), with equality for H = I (Frobenius quadratic).

Run from the project root:  OPENBLAS_NUM_THREADS=1 python3 paper/audit/checks/cite-stat_kop_minKd.py
"""
import numpy as np

rng = np.random.default_rng(0)


def haar_partial_isometry(K, d):
    A = rng.standard_normal((K, d))
    U, _, Vt = np.linalg.svd(A, full_matrices=False)
    return U @ Vt


def check(K, d, nmc=20000):
    p = K * d
    B = rng.standard_normal((p, p))
    H = B @ B.T / p + 0.1 * np.eye(p)          # generic PSD Hessian on vec(D), row-major vec
    trH = np.trace(H)
    r1, iso = [], []
    for _ in range(nmc):
        u = rng.standard_normal(K); u /= np.linalg.norm(u)
        v = rng.standard_normal(d); v /= np.linalg.norm(v)
        D = np.outer(u, v).ravel()
        r1.append(D @ H @ D)
        Q = haar_partial_isometry(K, d).ravel()
        iso.append(Q @ H @ Q)
    r1, iso = np.array(r1), np.array(iso)
    # crude certified brackets from samples: mu_op <= min over samples of rank-one, L_op >= max over isometries
    print(f"K={K} d={d}: E<uv,H uv>={r1.mean():.4f} (pred tr/Kd={trH/p:.4f}, se {r1.std()/np.sqrt(nmc):.4f}); "
          f"E<Q,HQ>={iso.mean():.4f} (pred min(K,d)tr/Kd={min(K,d)*trH/p:.4f}, se {iso.std()/np.sqrt(nmc):.4f}); "
          f"ratio of means={iso.mean()/r1.mean():.3f} vs min(K,d)={min(K,d)}")
    # equality case H = I: <D,D> = ||D||_F^2, mu_op = 1 (rank one), L_op = min(K,d) (isometry)
    Q = haar_partial_isometry(K, d)
    print(f"   H=I: ||Q||_F^2/||Q||_op^2 = {np.sum(Q**2)/np.linalg.norm(Q, 2)**2:.6f} = min(K,d) -> kappa_op = min(K,d)")


for K, d in [(2, 3), (3, 3), (4, 6), (5, 2)]:
    check(K, d)
