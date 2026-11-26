import sympy as sp, mpmath as mp, numpy as np, pickle, itertools
K,d=sp.symbols('K d',positive=True)
m=K+d+1
cs=K+d-(K-1)*(d-1)/(2*m)
r2=m; r4=(K**2+d**2+3*K*d+3*K+3*d+4)/m
print("3r(2)-r(4)==2c*:",sp.simplify(3*r2-r4-2*cs)==0)
v=K*d+K+d+3
print("m - v/(2m)==c*:",sp.simplify(m-v/(2*m)-cs)==0, " var check:",sp.simplify((K**2+d**2+3*K*d+3*K+3*d+4)-m**2-v)==0)
print("r(4)-4-(K+d-1)-(K-1)(d-1)/m==0:",sp.simplify(r4-4-(K+d-1)-(K-1)*(d-1)/m)==0)
# independent Wick (different algorithm: explicit matchings + union-find over index cycles), k<=6, compare with results pickle
def wick(k):
    n=2*k
    # factor t: Z[row[t], col[t]]; tr (Z Z^T)^k = sum Z[i0,j0]Z[i1,j0]Z[i1,j1]Z[i2,j1]...Z[i0,j_{k-1}]
    row=[((t+1)//2)%k for t in range(n)]; col=[t//2 for t in range(n)]
    from collections import Counter
    cnt=Counter()
    def matchings(lst):
        if not lst: yield []; return
        a=lst[0]
        for i in range(1,len(lst)):
            for rest in matchings(lst[1:i]+lst[i+1:]): yield [(a,lst[i])]+rest
    def ncomp(pairs,lab):
        par=list(range(k))
        def f(x):
            while par[x]!=x: x=par[x]
            return x
        for a,b in pairs:
            ra,rb=f(lab[a]),f(lab[b])
            if ra!=rb: par[ra]=rb
        return len({f(x) for x in range(k)})
    for mt in matchings(list(range(n))): cnt[(ncomp(mt,row),ncomp(mt,col))]+=1
    return sp.expand(sum(c*K**a*d**b for (a,b),c in cnt.items()))
stored=pickle.load(open('../results/wick_moments.pkl','rb'))
Ks,ds=sp.symbols('K d')
for k in range(1,7):
    print("k=%d my Wick == stored:"%k, sp.expand(wick(k)-sp.sympify(stored[k]).subs({Ks:K,ds:d}))==0)
# K=2 exact check of all moments k<=8 for d=2..6 through the (rho,gamma) representation
mp.mp.dps=30
def D_K2(p,dd):
    # s1,s2 = rho(cos(phi/2) +- sin(phi/2))/2, rho^2 = 2 chi^2_{2d}; gamma=cos(phi) density (d-1)gamma^{d-2}
    Erho=mp.mpf(2)**(p/2)*mp.mpf(2)**(p/2)*mp.gamma(dd+p/2)/mp.gamma(dd)
    f=lambda g:(dd-1)*g**(dd-2)*(((mp.sqrt((1+g)/2)+mp.sqrt((1-g)/2))/2)**p+((mp.sqrt((1+g)/2)-mp.sqrt((1-g)/2))/2)**p)
    return Erho*mp.quad(f,[0,1])
ok=True
for dd in range(2,7):
    for k in range(1,9):
        ex=sp.sympify(stored[k]).subs({Ks:2,ds:dd}); ok&= abs(D_K2(2*k,dd)/ex-1)<1e-15
print("K=2, d=2..6, k<=8: density formula reproduces Wick polynomials:",ok)
# q_{2,d}, bound, and D3=(K+d-q)D1
for dd in [2,3,4,10,50]:
    q=mp.quad(lambda g:(dd-1)*g**(dd-2)*g/mp.sqrt(1+g),[0,1])/mp.quad(lambda g:(dd-1)*g**(dd-2)*mp.sqrt(1+g),[0,1])
    print(" d=%d q_2d=%.6f bound (d-1)/(2(d+3))=%.6f  Stein check D3/D1-(2+d-q)=%.2e"%(dd,q,(dd-1)/(2*(dd+3)),D_K2(3,dd)/D_K2(1,dd)-(2+dd-q)))
print(" q_22 closed form (3sqrt2-2)/7 =",(3*mp.sqrt(2)-2)/7, "; polar g* 2x2 =",3*(3+2*(3*mp.sqrt(2)-2)/7)/16*2/12)
q23=mp.quad(lambda g:2*g*g/mp.sqrt(1+g),[0,1])/mp.quad(lambda g:2*g*mp.sqrt(1+g),[0,1])
print(" polar g* 2x3 =",3*(3+2*q23)/20*2/12)
# Remark (iii): r(p)-p -> K+d-1 ? K=2
for dd in [2,3,5]:
    print(" K=2 d=%d: r(p)-p at p=4,8,16,64,256,1024:"%dd,[float(D_K2(p+2,dd)/D_K2(p,dd)-p) for p in [4,8,16,64,256,1024]]," limit claimed",dd+1)
# concavity sanity on [0,4] second differences, and location of convexity
for dd in [2,3]:
    ps=np.arange(0.5,40,0.5); r=[float(D_K2(p+2,dd)/D_K2(p,dd)) for p in ps]
    sd=np.diff(r,2); print(" K=2 d=%d: first p where second difference of r >0:"%dd, ps[1:-1][sd>0][:1])
# MC check K=3
rng=np.random.default_rng(1)
for (KK,dd) in [(3,3),(3,21),(4,4)]:
    n=2_000_000; Z=rng.standard_normal((n,KK,dd)); s=np.linalg.svd(Z,compute_uv=False)
    D1=s.sum(1); D3=(s**3).sum(1); c=KK+dd-(KK-1)*(dd-1)/(2*(KK+dd+1))
    x=c*D1-D3; print(" MC (K,d)=(%d,%d): (c*D1-D3)/(Kd) = %.4f +- %.4f"%(KK,dd,x.mean()/(KK*dd),x.std()/np.sqrt(n)/(KK*dd)))
