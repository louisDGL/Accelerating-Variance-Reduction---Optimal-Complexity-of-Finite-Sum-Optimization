"""Two-constant structure of the first-order bias operator for isotropic noise.
T[K3]_{ij} = E[polar(Z)_{ij} <He_3(Z), K3>] = alpha * A[K3] + beta * B[K3],
A[K3]_{ij} = sum_{kl} K3_{ij,kl,kl},  B[K3]_{ij} = sum_{kl} K3_{kj,kl,il}.
(1) Gram matrix of the invariant tensors (exact, by einsum) and its polynomial form.
(2) Monte Carlo of the full tensor T for small sizes versus alpha*A + beta*B.
"""
import numpy as np, itertools, sys
rng = np.random.default_rng(1)

def inv_tensors(K, d):
    # index order: (i,j, a,b, c,dd, e,f)
    I_K, I_d = np.eye(K), np.eye(d)
    slots = [(2, 3), (4, 5), (6, 7)]
    A = np.zeros((K, d) * 4); B = np.zeros((K, d) * 4)
    letters = 'ijabcdef'
    nA = nB = 0
    for s in range(3):
        o = [t for t in range(3) if t != s]
        # output = slot s ; slots o contracted
        sub = f"{letters[0]}{letters[slots[s][0]]},{letters[1]}{letters[slots[s][1]]},{letters[slots[o[0]][0]]}{letters[slots[o[1]][0]]},{letters[slots[o[0]][1]]}{letters[slots[o[1]][1]]}->ijabcdef"
        A += np.einsum(sub, I_K, I_d, I_K, I_d); nA += 1
    for R in itertools.combinations(range(3), 2):
        for C in itertools.combinations(range(3), 2):
            if R == C: continue
            r_out = [t for t in range(3) if t not in R][0]; c_out = [t for t in range(3) if t not in C][0]
            sub = (f"{letters[slots[R[0]][0]]}{letters[slots[R[1]][0]]},{letters[slots[C[0]][1]]}{letters[slots[C[1]][1]]},"
                   f"{letters[0]}{letters[slots[r_out][0]]},{letters[1]}{letters[slots[c_out][1]]}->ijabcdef")
            B += np.einsum(sub, I_K, I_d, I_K, I_d); nB += 1
    return A / nA, B / nB

def gram(K, d):
    A, B = inv_tensors(K, d)
    return (A * A).sum(), (A * B).sum(), (B * B).sum()

print("Gram <A,A>, <A,B>, <B,B> (symmetrized tensors) and candidates")
for K, d in [(1, 1), (1, 3), (2, 2), (2, 3), (3, 3), (2, 4), (3, 4)]:
    gAA, gAB, gBB = gram(K, d)
    N = K * d
    print(K, d, gAA, gAB, gBB, "| N(N+2)/3 =", N * (N + 2) / 3, " N(K+d+1)/3 =", N * (K + d + 1) / 3,
          " cand BB =", N * (N + K + d + 3 + (K + d + 1) ** 2 - (K+d+1) - 0) / 6)

def mc_T(K, d, S=4_000_000, chunk=200_000):
    T = np.zeros((K, d) * 4); m1 = m3 = 0.0
    I = np.eye(K * d).reshape(K, d, K, d)
    for _ in range(S // chunk):
        Z = rng.standard_normal((chunk, K, d))
        U, s, Vt = np.linalg.svd(Z, full_matrices=False)
        P = U @ Vt
        m1 += s.sum(); m3 += (s ** 3).sum()
        T += np.einsum('nij,nab,ncd,nef->ijabcdef', P, Z, Z, Z, optimize=True)
        PZ = np.einsum('nij,nab->ijab', P, Z)
        T -= np.einsum('ijab,cdef->ijabcdef', PZ, I) + np.einsum('ijcd,abef->ijabcdef', PZ, I) + np.einsum('ijef,abcd->ijabcdef', PZ, I)
    return T / S, m1 / S, m3 / S

if len(sys.argv) > 1:
    for K, d in [(2, 2), (2, 3)]:
        T, m1, m3 = mc_T(K, d)
        A, B = inv_tensors(K, d)
        gAA, gAB, gBB = (A * A).sum(), (A * B).sum(), (B * B).sum()
        cA, cB = (A * T).sum(), (B * T).sum()
        al, be = np.linalg.solve([[gAA, gAB], [gAB, gBB]], [cA, cB])
        res = T - al * A - be * B
        print(f"K={K} d={d}: alpha={al:.5f} beta={be:.5f}; |T|={np.linalg.norm(T):.4f} residual={np.linalg.norm(res):.4f}",
              f"| cA={cA:.4f} vs -m1={-m1:.4f}; cB={cB:.4f} vs m3-(K+d+1)m1={m3 - (K + d + 1) * m1:.4f}")
