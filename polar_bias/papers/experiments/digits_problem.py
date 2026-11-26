"""Real-data instance for prediction (5): ridge-regularised multi-output least squares on sklearn load_digits.
features: first d = 16 principal components of the standardised pixels (centred, not whitened).
targets (default, mode='proj'): K = 4 fixed random orthonormal projections (numpy default_rng(0), QR) of the centred one-hot labels.
Why not the one-hot labels themselves: with all ten classes the residuals sum to zero across classes for every sample, so the
covariance of the per-sample gradients is singular (16-dimensional kernel); dropping the last class (mode='drop', K = 9) gives
a nondegenerate but badly conditioned covariance (eigenvalue ratio 1.3e-4) and g*' Sigma^{-1} g* ~ 90, i.e. the first-order
regime b >> 90 is out of reach of b <= 128.  F(W) = (1/2n) sum |W x_i - y_i|^2 + (lam/2)|W|_F^2."""
import numpy as np
from common import *

LAM = 1e-3


def problem(K=4, d=16, mode='proj'):
    from sklearn.datasets import load_digits
    D = load_digits()
    P = D.data.astype(float); P = P - P.mean(0); sd = P.std(0); P = P[:, sd > 0] / sd[sd > 0]
    U, s, Vt = np.linalg.svd(P, full_matrices=False)
    X = U[:, :d] * s[:d]; X = X - X.mean(0)                      # PCA scores (not whitened)
    Y = np.eye(10)[D.target]; Y = Y - Y.mean(0)
    Yfull = Y
    Y = Y[:, :K] if mode == 'drop' else Y @ np.linalg.qr(np.random.default_rng(0).standard_normal((10, K)))[0]
    n = len(X); Sx = X.T @ X / n; Hx = Sx + LAM * np.eye(d)       # Hessian = I_K (x) Hx
    Ws = np.linalg.solve(Hx, X.T @ Y / n).T                        # K x d minimiser
    return dict(X=X, Y=Y, Yfull=Yfull, n=n, K=K, d=d, Hx=Hx, Ws=Ws)


def F(p, W):
    R = p['X'] @ W.T - p['Y']
    return 0.5 * (R ** 2).sum() / p['n'] + 0.5 * LAM * (W ** 2).sum()


def per_sample_grads(p, Y=None):
    Y = p['Y'] if Y is None else Y
    Ws = np.linalg.solve(p['Hx'], p['X'].T @ Y / p['n']).T
    R = p['X'] @ Ws.T - Y
    g = R[:, :, None] * p['X'][:, None, :] + LAM * Ws[None]
    return g.reshape(p['n'], -1)


def gstar_mc(V, K, d, S, seed, A=None, ch=100_000):
    """g* = -(1/6) J^{-1} T[K3] as in checks/lib.general_prediction, for the EMPIRICAL law of the rows of V (n,N):
    <K3, y^{(x)3}> = mean_i (v_i.y)^3 is evaluated without forming K3.  A: optional (N,N) control variate, T uses
    O - A y in place of O (E[y He_3] = 0, so any fixed A keeps T unbiased)."""
    n, N = V.shape
    rng = np.random.default_rng(seed)
    Sig = V.T @ V / n; L = np.linalg.cholesky(Sig); Si = np.linalg.inv(Sig)
    lin = 3 * (V * ((V @ Si) * V).sum(1, keepdims=True)).mean(0)
    J = np.zeros((N, N)); T = np.zeros(N)
    for _ in range(S // ch):
        Z = rng.standard_normal((ch, N)) @ L.T
        Yv = Z @ Si
        U, s, Vt = np.linalg.svd(Z.reshape(-1, K, d), full_matrices=False)
        O = (U @ Vt).reshape(-1, N)
        he = ((Yv @ V.T) ** 3).mean(1) - Yv @ lin
        J += O.T @ Yv
        T += (O - Yv @ A.T if A is not None else O).T @ he
    J /= S; T /= S
    return (-np.linalg.solve(J, T) / 6).reshape(K, d), J, T


if __name__ == '__main__':
    # prediction g* for the default instance: pilot J for the control variate, then 8 independent Monte-Carlo evaluations
    import sys, time
    from multiprocessing import Pool
    p = problem(); K, d, n = p['K'], p['d'], p['n']
    V = per_sample_grads(p); Sig = V.T @ V / n; ev = np.linalg.eigvalsh(Sig)
    _, J0, _ = gstar_mc(V, K, d, 400_000, 100)
    A = J0 @ Sig
    S = int(float(sys.argv[1])) if len(sys.argv) > 1 else 2_000_000
    with Pool(4) as pool:
        gs = np.array([r[0] for r in pool.starmap(gstar_mc, [(V, K, d, S, s, A) for s in range(8)])])
    np.savez(os.path.join(DATA, 'digits_gstar.npz'), gs=gs, S=S, cov_eigs=ev)
    nuc = [np.linalg.svd(g, compute_uv=False).sum() for g in gs]
    print('Cov eig min/max', ev[0], ev[-1], ev[0] / ev[-1], '|mean grad|', np.abs(V.mean(0)).max())
    print('nuc', np.mean(nuc), np.std(nuc, ddof=1), ' fro', [round(float(np.linalg.norm(g)), 4) for g in gs])
