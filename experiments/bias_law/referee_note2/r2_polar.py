import numpy as np
from scipy import integrate
rng=np.random.default_rng(0)
def polar(M):
    U,s,Vt=np.linalg.svd(M,full_matrices=False); return U@Vt
# integral representation
M=rng.standard_normal((2,3))
val=np.zeros((2,3))
for i in range(2):
    for j in range(3):
        f=lambda tau:(np.linalg.solve(M@M.T+tau**2*np.eye(2),M))[i,j]
        val[i,j]=2/np.pi*integrate.quad(f,0,np.inf,limit=500)[0]
print("integral rep error:",np.abs(val-polar(M)).max())
# derivative bounds |D^k polar| sigma_K^k, k=1,2,3 by finite differences in float128-ish (use mpmath-free: scaled steps)
def dk(M,H,k,h):
    if k==1: return (polar(M+h*H)-polar(M-h*H))/(2*h)
    if k==2: return (polar(M+h*H)-2*polar(M)+polar(M-h*H))/h**2
    if k==3: return (polar(M+2*h*H)-2*polar(M+h*H)+2*polar(M-h*H)-polar(M-2*h*H))/(2*h**3)
for (K,d) in [(2,3),(3,5),(2,2),(3,3)]:
    for sig in [1e-1,1e-2,1e-3]:
        mx=[0,0,0]
        for trial in range(300):
            U,_=np.linalg.qr(rng.standard_normal((K,K))); V,_=np.linalg.qr(rng.standard_normal((d,d)))
            s=np.sort(np.abs(rng.standard_normal(K)))[::-1]+0.5; s[-1]=sig
            if K==d and K>=2 and trial%2==0: s[-2]=3*sig  # two small singular values
            if K<d and trial%2==0 and K>=2: s[-2]=3*sig
            M=U@np.diag(s)@V[:K,:]
            for _ in range(10):
                H=rng.standard_normal((K,d)); H/=np.linalg.norm(H)
                if _<3:  # adversarial: directions in the small singular subspace
                    H=np.outer(U[:,-1],V[rng.integers(K-1,d),:])+0.3*H; H/=np.linalg.norm(H)
                for k in (1,2,3):
                    h=sig*(1e-4 if k==1 else 2e-3 if k==2 else 1e-2)
                    mx[k-1]=max(mx[k-1],np.linalg.norm(dk(M,H,k,h))*sig**k)
        print("K=%d d=%d sigma_K=%.0e  max |D^k polar| sigma_K^k (k=1,2,3): %.3f %.3f %.3f"%(K,d,sig,*mx))
print("(for K=d the bound must FAIL only through sigma_{K-1}; the K=d rows use dist to V = sigma_K, expect bounded k=1 only if smooth off V: polar is smooth off V with |D^k| ~ (s_K+s_{K-1})^-k)")
# 2x2 formulas
err=[0,0,0,0]
for _ in range(2000):
    M=rng.standard_normal((2,2)); p,q,r,s=M.ravel()
    yp=np.array([p+s,q-r])/np.sqrt(2); ym=np.array([p-s,q+r])/np.sqrt(2)
    err[0]=max(err[0],abs(np.linalg.det(M)-(yp@yp-ym@ym)/2))
    nuc=np.linalg.svd(M,compute_uv=False).sum()
    err[1]=max(err[1],abs(nuc-np.sqrt(2)*max(np.linalg.norm(yp),np.linalg.norm(ym))))
    err[2]=max(err[2],abs(nuc-np.sqrt((M**2).sum()+2*abs(np.linalg.det(M)))))
    P=polar(M); Pp=np.array([P[0,0]+P[1,1],P[0,1]-P[1,0]])/np.sqrt(2); Pm=np.array([P[0,0]-P[1,1],P[0,1]+P[1,0]])/np.sqrt(2)
    if yp@yp>ym@ym: e=max(np.abs(Pp-np.sqrt(2)*yp/np.linalg.norm(yp)).max(),np.abs(Pm).max())
    else: e=max(np.abs(Pm-np.sqrt(2)*ym/np.linalg.norm(ym)).max(),np.abs(Pp).max())
    err[3]=max(err[3],e)
print("2x2: det, nuc(max), nuc(sqrt), polar errors:",err)
# volume exponent: P(sigma_K <= r) for uniform in unit ball of R^{Kxd}
for (K,d) in [(2,3),(2,4),(3,5),(2,2)]:
    X=rng.standard_normal((400000,K,d)); X/=np.linalg.norm(X.reshape(len(X),-1),axis=1)[:,None,None]
    X*=rng.random(len(X))[:,None,None]**(1/(K*d))
    sK=np.linalg.svd(X,compute_uv=False)[:,-1]
    r1,r2=0.02,0.08
    print("K=%d d=%d empirical exponent of vol{sigma_K<=r}: %.2f (claimed d-K+1=%d)"%(K,d,np.log((sK<=r2).mean()/(sK<=r1).mean())/np.log(r2/r1),d-K+1))
