import mpmath as mp, sys
from wish import Wish, target
mp.mp.dps = 80
def gauss(mom, n):
    # mom[0..2n-1] moments -> nodes, weights via Hankel solve
    H = mp.matrix(n, n); b = mp.matrix(n, 1)
    for i in range(n):
        for j in range(n): H[i, j] = mom[i+j]
        b[i] = -mom[i+n]
    c = mp.lu_solve(H, b)
    nodes = mp.polyroots([1]+[c[n-1-i] for i in range(n)], maxsteps=500, extraprec=400)
    V = mp.matrix(n, n)
    for i in range(n):
        for j in range(n): V[i, j] = nodes[j]**i
    w = mp.lu_solve(V, mp.matrix(mom[:n]))
    return nodes, w
def cstar(K, d): return mp.mpf(2*K*K+2*d*d+3*K*d+3*K+3*d-1)/(2*(K+d+1))
if __name__ == "__main__":
    pairs = [(K, d) for K in range(2, 13) for d in range(K, 13)] + [(2,20),(2,40),(3,30),(4,30),(8,30),(16,16),(16,40),(20,20)]
    for (K, d) in pairs:
        w = Wish(K, d)
        q = w.q(); t = target(K, d); c = cstar(K, d)
        nu = [w.Etr(k+1)/(K*d) for k in range(8)]
        out = []
        for n in (2, 3, 4):
            x, wt = gauss(nu, n)
            L = sum(wt[i]*(c-x[i])/mp.sqrt(x[i]) for i in range(n))
            out.append(mp.nstr(mp.re(L), 4))
        true = (c*w.D(1) - w.D(3))/(K*d)
        # concavity of r on grid
        ps = [mp.mpf(i)/4 for i in range(0, 41)]
        rs = [w.r(p) for p in ps]
        dd = max(rs[i-1]+rs[i+1]-2*rs[i] for i in range(1, len(rs)-1))
        print(K, d, "q", mp.nstr(q, 8), "tgt", mp.nstr(t, 6), "ratio", mp.nstr(q/t, 5),
              "(K-1)/2-q *d", mp.nstr(((K-1)/mp.mpf(2)-q)*d, 6),
              "true", mp.nstr(true, 4), "Gauss2,3,4", out, "max 2nd diff r", mp.nstr(dd, 3))
        sys.stdout.flush()
