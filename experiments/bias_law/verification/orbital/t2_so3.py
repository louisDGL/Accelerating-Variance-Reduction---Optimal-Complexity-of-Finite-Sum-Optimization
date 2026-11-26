"""Item 1 numerics. I(u;e,y)=E exp(i u <e,U y V^T>), U,V Haar on SO(3).
By bi-invariance e,y may be taken signed-diagonal: phase = sum_ij tau_i sigma_j U_ij V_ij, tau,sigma in S^2.
(a) exact quadrature for e=y=I/sqrt3 (sharpness paragraph); (b) exact rank-one; (c) MC sup search."""
import numpy as np, mpmath as mp
from scipy.special import sici
rng=np.random.default_rng(1)
def haar(n):
    q=rng.normal(size=(n,4)); q/=np.linalg.norm(q,axis=1,keepdims=True)
    a,b,c,d=q.T
    R=np.empty((n,3,3))
    R[:,0,0]=a*a+b*b-c*c-d*d; R[:,0,1]=2*(b*c-a*d); R[:,0,2]=2*(b*d+a*c)
    R[:,1,0]=2*(b*c+a*d); R[:,1,1]=a*a-b*b+c*c-d*d; R[:,1,2]=2*(c*d-a*b)
    R[:,2,0]=2*(b*d-a*c); R[:,2,1]=2*(c*d+a*b); R[:,2,2]=a*a-b*b-c*c+d*d
    return R,a
# (a) trace law: tr U = 4 s^2 - 1 with density (2/pi) sqrt(1-s^2)
R,a=haar(2_000_000)
tr=np.trace(R,axis1=1,axis2=2)
print("max|tr U-(4a^2-1)| =",np.abs(tr-(4*a*a-1)).max())
h,edges=np.histogram(a,bins=20,range=(-1,1),density=True); mid=(edges[1:]+edges[:-1])/2
print("max dev of density of s from (2/pi)sqrt(1-s^2):",np.abs(h-2/np.pi*np.sqrt(1-mid**2)).max())
mp.mp.dps=30
print("u   |E e^{iu trU}|  exact/ (1/sqrt(pi u))   arg-check: exact*sqrt(pi u)*e^{iu-i pi/4}")
for u in [10,40,160,640,2560]:
    f=lambda s: 2/mp.pi*mp.sqrt(1-s*s)*mp.exp(1j*u*(4*s*s-1))
    val=mp.quadosc(lambda s:0,[0,1]) if False else mp.quad(f,mp.linspace(-1,1,int(8*u/3)+50))
    print(u, float(abs(val)), float(abs(val)*mp.sqrt(mp.pi*u)), complex(val*mp.sqrt(mp.pi*u)*mp.exp(1j*u-1j*mp.pi/4)))
# (c) MC sup search
def sph(n):
    v=rng.normal(size=(n,3)); return v/np.linalg.norm(v,axis=1,keepdims=True)
def special():
    c=[]
    base=[[1,0,0],[1,1,0],[1,1,1],[1,1,-1],[1,-1,0],[2,1,0],[2,1,1],[2,2,1],[3,1,1],[1,1,.5],[1,1,.2],[1,.5,.5],[1,.2,.2],[1,.5,0],[1,.2,0],[1,1,-.5],[1,-.5,-.5]]
    for v in base:
        v=np.array(v,float); c.append(v/np.linalg.norm(v))
    return np.array(c)
sp=special(); ns=len(sp)
P=3000
tau=np.vstack([np.repeat(sp,ns,axis=0),sph(P)]); sig=np.vstack([np.tile(sp,(ns,1)),sph(P)])
# also permuted pairings of special ones
perm=sp[:,[1,2,0]]; tau=np.vstack([tau,np.repeat(sp,ns,axis=0)]); sig=np.vstack([sig,np.tile(perm,(ns,1))])
K=(tau[:,:,None]*sig[:,None,:]).reshape(len(tau),9)        # coefficient of U_ij V_ij
n1=200_000
U,_=haar(n1); V,_=haar(n1); W=(U*V).reshape(n1,9)
U2,_=haar(2_000_000); V2,_=haar(2_000_000); W2=(U2*V2).reshape(-1,9)
def I(Wm,Kv,u,chunk=200):
    out=np.empty(len(Kv))
    for s in range(0,len(Kv),chunk):
        ph=u*(Wm@Kv[s:s+chunk].T)
        out[s:s+chunk]=np.abs(np.exp(1j*ph).mean(axis=0))
    return out
print("\nMC sup search: n_search=%d (noise %.4f), top-5 re-evaluated on fresh n=2e6 (noise %.4f)"%(n1,n1**-.5,(2e6)**-.5))
print("u  sup_fresh  sup*sqrt(u)  identity:sqrt(3/(pi u))  rank1:Si(u)/u   argmax tau, sigma")
res=[]
for u in [5,10,20,40,80,160,320]:
    v=I(W,K,u); top=np.argsort(v)[-8:]
    # local refinement around the top candidates
    cand=[K[top]]; tt=[tau[top]]; ss=[sig[top]]
    for t in top:
        for scale in (.15,.05):
            a_=tau[t]+scale*rng.normal(size=(40,3)); b_=sig[t]+scale*rng.normal(size=(40,3))
            a_/=np.linalg.norm(a_,axis=1,keepdims=True); b_/=np.linalg.norm(b_,axis=1,keepdims=True)
            tt.append(a_); ss.append(b_); cand.append((a_[:,:,None]*b_[:,None,:]).reshape(-1,9))
    cand=np.vstack(cand); tt=np.vstack(tt); ss=np.vstack(ss)
    v1=I(W,cand,u); best=np.argsort(v1)[-5:]
    v2=I(W2,cand[best],u); j=np.argmax(v2)
    si=sici(u)[0]/u
    res.append((u,v2[j]))
    print(u, "%.4f"%v2[j], "%.3f"%(v2[j]*np.sqrt(u)), "%.4f"%np.sqrt(3/(np.pi*u)), "%.4f"%si, np.round(tt[best[j]],3), np.round(ss[best[j]],3))
lu=np.log([r[0] for r in res][2:]); lv=np.log([r[1] for r in res][2:])
print("fitted decay exponent of sup for u>=20:",np.polyfit(lu,lv,1)[0])
