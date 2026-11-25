"""Audit cite-muon: control of the sentence (paper lines 313-314)
  "On least squares the curvature is constant and the gap is about a factor 4 on the ray that attracts
   generic runs [shulgin2026quadratic]".
Setting of Shulgin et al. (arXiv:2610.30280, Theorem 3, Eq. (15)-(17), Corollary 5), transposed to the paper's
notation: F(W) = 1/2 tr(E Sigma E^T), E = W - W_*, K x d with K >= d, gradient G = E Sigma, nuclear-scaled polar step
with L = L_op = tr(Sigma):  E+ = E - ||G||_* polar(G) / tr(Sigma).
Certified factor (Jawanpuria et al. Theorem 3 at u = 1): 1 - lambda_d / tr(Sigma), exponent ~ lambda_d / tr(Sigma).
Two-cluster ray of the flattest direction (Shulgin et al. Corollary 5, w = lambda_d / tr Sigma): F contracts by
(1 - 2w)^2 per step, exponent ~ 4 w.
We measure, from generic (Gaussian) starts: (i) the exponent per step in the tail, in units of w; (ii) the mean
exponent over the head of the run (until F/F_0 = 1e-2, 1e-4, 1e-6, 1e-8), in units of w; (iii) the ratio at step 0.
The step is positively homogeneous, so E is renormalized at each step (no rounding floor).
Run from the project root:  OPENBLAS_NUM_THREADS=1 python3 paper/audit/checks/cite-muon_ray_factor4.py
"""
import numpy as np


def run(K, d, alpha, seed, T):
    rng = np.random.default_rng(seed)
    lam = np.arange(1, d + 1, dtype=float) ** (-alpha)      # Sigma = diag(lam) (generic E makes it non-commuting)
    tr = lam.sum()
    w = lam[-1] / tr
    E = rng.standard_normal((K, d))
    logF = np.empty(T + 1)
    acc = 0.0                                               # log of the accumulated normalization
    cos_last = None
    for t in range(T + 1):
        F = 0.5 * np.sum(E * E * lam)
        logF[t] = np.log(F) + 2 * acc
        if t == T:
            # alignment of the top right singular vector of E with the flattest eigenvector e_d
            _, s, Vt = np.linalg.svd(E, full_matrices=False)
            cos_last = abs(Vt[0, -1]); share = s[0] ** 2 / np.sum(s ** 2)
            break
        G = E * lam
        U, s, Vt = np.linalg.svd(G, full_matrices=False)
        E = E - (s.sum() / tr) * (U @ Vt)
        nrm = np.linalg.norm(E)
        E /= nrm; acc += np.log(nrm)
    return lam, w, logF, cos_last, share


def main():
    T = 6000
    print("exponent per step of F - F_*, in units of the certified exponent w = lambda_d / tr(Sigma)")
    print("(ray of Shulgin et al., Cor. 5: -2 log(1-2w)/w ~ 4;  certified: -log(1-w)/w ~ 1)")
    for (K, d, alpha) in [(20, 20, 1.0), (20, 20, 2.0), (40, 20, 1.0), (100, 100, 1.0)]:
        TT = T if d <= 20 else 3000
        for seed in range(3 if d <= 20 else 1):
            lam, w, logF, cos_last, share = run(K, d, alpha, seed, TT)
            tail = -(logF[-1] - logF[-TT // 4 - 1]) / (TT // 4)
            ray = -2 * np.log(1 - 2 * w)
            head = []
            for target in (1e-2, 1e-4, 1e-6, 1e-8):
                idx = np.nonzero(logF - logF[0] <= np.log(target))[0]
                head.append(np.nan if len(idx) == 0 else -np.log(target) / idx[0] / w)
            first = -(logF[1] - logF[0]) / w
            print(f"K={K:3d} d={d:3d} spectrum k^-{alpha:g} seed={seed}: w={w:.3e}  tail/w={tail / w:5.2f} "
                  f"(ray {ray / w:4.2f})  cos(top right sv, e_d)={cos_last:.3f} s1^2 share={share:.3f} | "
                  f"mean exponent/w until F/F0=1e-2,1e-4,1e-6,1e-8: "
                  + ", ".join(f"{h:6.1f}" for h in head) + f" | first step: {first:7.1f}")


if __name__ == "__main__":
    main()
