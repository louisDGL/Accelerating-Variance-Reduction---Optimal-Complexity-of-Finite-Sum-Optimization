"""Shared tools: polar factor, zeros of mean fields by common random numbers, first-order predictions."""
import numpy as np
from scipy.optimize import root


def polar(M):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    return U @ Vt


def nuc(M):
    return np.linalg.svd(M, compute_uv=False).sum()


def field_zero(xi, x0=None, direction=polar):
    """Zero of G -> mean_s direction(G + xi_s) (common random numbers); returns (G, residual norm)."""
    S, K, d = xi.shape
    f = lambda g: direction(g.reshape(K, d)[None] + xi).mean(0).reshape(-1)
    sol = root(f, np.zeros(K * d) if x0 is None else x0.reshape(-1), method='hybr', tol=1e-13)
    return sol.x.reshape(K, d), np.linalg.norm(f(sol.x))


def field_zero_se(xi, G, direction=polar, J=None):
    """Monte-Carlo standard error (Frobenius) of the zero: J^{-1} * se(mean field), J ~ sqrt(b) c Id if not given."""
    S, K, d = xi.shape
    P = direction(G[None] + xi).reshape(S, -1)
    return np.sqrt(P.var(0).sum() / S)


def q_const(K, d, S=2_000_000, seed=0):
    rng = np.random.default_rng(seed)
    h = m1 = 0.0
    ch = 100_000
    for _ in range(S // ch):
        s = np.linalg.svd(rng.standard_normal((ch, K, d)), compute_uv=False)
        P = s[:, :, None] * s[:, None, :] / (s[:, :, None] + s[:, None, :])
        h += P.sum() - np.trace(P, axis1=1, axis2=2).sum(); m1 += s.sum()
    return h / m1, m1 / S


def a_consts(K, d, q):
    if min(K, d) == 1:
        return None
    D = (K - 1) * (d - 1) * (K + 2) * (d + 2)
    return (3 * ((K - 1) * (d - 1) - 2 * (K + d + 1) * q) / D, 6 * ((K * d + 2) * q + (K - 1) * (d - 1)) / D)


def contractions(K3, K, d):
    """A[K3]_{ij} = sum_kl K3_{ij,kl,kl};  B[K3]_{ij} = sum_kl K3_{kj,kl,il};  K3 of shape (K,d,K,d,K,d)."""
    return np.einsum('ijklkl->ij', K3), np.einsum('kjklil->ij', K3)


def third_cumulant(samples):
    """Empirical third cumulant tensor (K,d,K,d,K,d) of centred samples (S,K,d)."""
    S, K, d = samples.shape
    X = samples.reshape(S, -1); X = X - X.mean(0)
    T = np.zeros((K * d,) * 3)
    for c in np.array_split(X, max(1, S // 50000)):
        T += np.einsum('na,nb,nc->abc', c, c, c)
    return (T / S).reshape(K, d, K, d, K, d)


def general_prediction(Sigma, K3, K, d, S=4_000_000, seed=1, direction=polar):
    """g* = -(1/6) J^{-1} T[K3] with J = E[o(Z) (Sigma^{-1}Z)^T], T = E[o(Z) <K3, He_3^Sigma(Z)>], Z ~ N(0,Sigma);
    Sigma (N,N) on vec (row-major), K3 (N,N,N). Returns g* (K,d) so that the first-order zero is g*/b."""
    rng = np.random.default_rng(seed)
    N = K * d
    L = np.linalg.cholesky(Sigma); Si = np.linalg.inv(Sigma)
    K3 = K3.reshape(N, N, N)
    lin = 3 * np.einsum('abc,bc->a', K3, Si)
    J = np.zeros((N, N)); T = np.zeros(N); ch = 200_000
    for _ in range(S // ch):
        Z = rng.standard_normal((ch, N)) @ L.T
        Z = np.concatenate([Z, -Z])                       # antithetic (o odd, He_3 odd)
        Y = Z @ Si
        O = direction(Z.reshape(-1, K, d)).reshape(-1, N)
        he = np.einsum('abc,na,nb,nc->n', K3, Y, Y, Y, optimize=True) - Y @ lin
        J += O.T @ Y; T += O.T @ he
    J /= 2 * S; T /= 2 * S
    return (-np.linalg.solve(J, T) / 6).reshape(K, d), J, T
