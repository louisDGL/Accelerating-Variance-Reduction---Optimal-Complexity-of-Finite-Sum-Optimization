import mpmath as mp, sys, pickle, os
from wish import Wish
from scan import cstar
mp.mp.dps = 60
def moms(K, d, kmax=10):
    f = f"mom_{K}_{d}.pkl"
    w = Wish(K, d)
    return [w.Etr(k) for k in range(kmax+1)], w
def LN(K, d, E, N, theta):
    c = cstar(K, d); m = K+d+1; x0 = theta*m
    nu = [E[j+1]/(K*d) for j in range(N+1)]
    tot = 0
    for k in range(N+1):
        # E y^k, y = x/x0 - 1
        Ey = sum(mp.binomial(k, j)*nu[j]/x0**j*(-1)**(k-j) for j in range(k+1))
        tot += (c*mp.binomial(-0.5, k) - x0*mp.binomial(0.5, k))*Ey
    return tot/mp.sqrt(x0)
if __name__ == "__main__":
    pairs = [(2,2),(2,3),(2,6),(3,3),(3,4),(3,5),(3,8),(3,20),(4,4),(4,5),(4,8),(4,30),(5,5),(6,6),(6,12),(8,8),(8,30),(12,12),(16,16),(16,40),(24,24)]
    for (K, d) in pairs:
        E, w = moms(K, d)
        true = (cstar(K, d)*w.D(1)-w.D(3))/(K*d)
        row = []
        for N in (3, 5, 7, 9):
            best = max(((LN(K, d, E, N, mp.mpf(t)/20), t/20) for t in range(8, 41)), key=lambda z: z[0])
            row.append((mp.nstr(LN(K, d, E, N, 1), 3), mp.nstr(best[0], 3), best[1]))
        print(K, d, "true", mp.nstr(true, 3), row); sys.stdout.flush()
