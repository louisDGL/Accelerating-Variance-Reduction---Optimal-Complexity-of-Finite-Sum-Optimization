"""Exact (high-precision) evaluation of D_p = E tr W^{p/2}, W real Wishart K x K, d dof,
via the de Bruijn Pfaffian formula:  E sum f(l_i) = (1/2) tr(M^{-1} M')."""
import mpmath as mp
mp.mp.dps = 60
from functools import lru_cache

def J(A, B):
    # int int_{0<x<y} x^{A-1} y^{B-1} e^{-(x+y)/2}
    return mp.mpf(2)**(A+B)*mp.gamma(A+B)*mp.betainc(A, B, 0, mp.mpf(1)/2)

class Wish:
    def __init__(self, K, d):
        self.K, self.d = K, d
        self.a = mp.mpf(d-K-1)/2
        self.n = K if K % 2 == 0 else K+1
        self.M = self._mat(None)
        self.Minv = mp.inverse(self.M)
        self.cache = {}
    def I(self, al, be):
        a1 = self.a+1
        return J(al+a1, be+a1) - J(be+a1, al+a1)
    def mom1(self, al):
        return mp.mpf(2)**(al+self.a+1)*mp.gamma(al+self.a+1)
    def _mat(self, s):
        K, n = self.K, self.n
        M = mp.zeros(n, n)
        for i in range(K):
            for j in range(i+1, K):
                v = self.I(i, j) if s is None else self.I(i+s, j)+self.I(i, j+s)
                M[i, j] = v; M[j, i] = -v
            if s is not None:
                M[i, i] = 0  # I(i+s,i)+I(i,i+s)=0
            if n > K:
                v = self.mom1(i) if s is None else self.mom1(i+s)
                M[i, K] = v; M[K, i] = -v
        return M
    def Etr(self, s):
        """E tr W^s"""
        s = mp.mpf(s)
        if s in self.cache: return self.cache[s]
        if s == 0: return mp.mpf(self.K)
        Mp = self._mat(s)
        t = mp.mpf(0)
        n = self.n
        for i in range(n):
            for j in range(n):
                t += self.Minv[i, j]*Mp[j, i]
        self.cache[s] = t/2
        return t/2
    def D(self, p): return self.Etr(mp.mpf(p)/2)
    def r(self, p): return self.D(mp.mpf(p)+2)/self.D(p)
    def q(self): return self.K+self.d-self.r(1)

def target(K, d): return mp.mpf((K-1)*(d-1))/(2*(K+d+1))

if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(0)
    for (K, d) in [(2,2),(2,5),(3,3),(3,6),(4,4),(5,9),(6,7)]:
        w = Wish(K, d)
        e1, e2, e3 = w.Etr(1), w.Etr(2), w.Etr(3)
        ex = (K*d, K*d*(K+d+1), K*d*(K*K+d*d+3*K*d+3*K+3*d+4))
        m4 = K*d*(K**3+d**3+6*K*K*d+6*K*d*d+6*K*K+6*d*d+17*K*d+21*K+21*d+20)
        Z = rng.standard_normal((200000, K, d))
        s = np.linalg.svd(Z, compute_uv=False)
        D1 = s.sum(1).mean(); D3 = (s**3).sum(1).mean()
        ss = s[:, :, None]*s[:, None, :]/(s[:, :, None]+s[:, None, :])
        qq = (ss.sum((1,2)) - (s/2).sum(1)).mean()/D1
        print(K, d, [mp.nstr(x-y, 5) for x, y in zip((e1,e2,e3), ex)], mp.nstr(w.Etr(4)-m4, 5),
              "D1", mp.nstr(w.D(1), 8), D1, "D3", mp.nstr(w.D(3), 8), D3,
              "q", mp.nstr(w.q(), 10), qq, "target", mp.nstr(target(K, d), 6))
    print("q22", mp.nstr(Wish(2,2).q(), 20), mp.nstr((3*mp.sqrt(2)-2)/7, 20))
