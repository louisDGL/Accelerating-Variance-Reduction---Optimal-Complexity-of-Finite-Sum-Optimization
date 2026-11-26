"""Lemma cube test: zeta in {-1,2} w.p. 2/3,1/3 (lattice, span 3). X=S_b/sqrt b.
sup_{t0} int_{t0-a}^{t0+a} |phi(t/sqrt b)|^b dt, a=c*sqrt(b). Should be bounded in b for fixed c."""
import numpy as np
def absphi(s):
    return np.abs(2/3*np.exp(-1j*s)+1/3*np.exp(2j*s))
for c in [0.25, 1.0, 3.0]:
    row=[]
    for b in [4,16,64,256,1024,4096,16384,65536]:
        # in s=t/sqrt b: window half-length c, integrand sqrt(b)*|phi(s)|^b; period 2pi/3
        P=2*np.pi/3
        n=400001
        s=np.linspace(0,P,n); ds=s[1]-s[0]
        f=np.sqrt(b)*np.exp(b*np.log(np.maximum(absphi(s),1e-300)))
        # cumulative integral over periodic extension
        F=np.concatenate([[0],np.cumsum((f[1:]+f[:-1])/2)*ds]); tot=F[-1]
        def I(x): # integral from 0 to x
            k=np.floor(x/P); r=x-k*P
            return k*tot+np.interp(r,s,F)
        t0=np.linspace(0,P,2001)
        row.append(np.max(I(t0+c)-I(t0-c)))
    print("c=%.2f"%c," ".join("%.4f"%v for v in row))
# 2D lattice: independent coordinates -> product, so bounded too. Also a weighted (momentum) case:
print("momentum weights beta=0.99,b=1: window half-length c/eps_w")
beta=0.99; w=(1-beta)*beta**np.arange(2500); rho2=np.sum(w**2); eps=w.max()/np.sqrt(rho2)
for c in [0.25,1.0]:
    a=c/eps
    t=np.linspace(-a,a,4001); best=0
    for t0 in np.linspace(0,2*np.pi/3*np.sqrt(rho2)/w.max()*3,120):
        tt=t+t0
        lg=np.zeros_like(tt)
        for wk in w[w>1e-12*w.max()][:2500]:
            lg+=np.log(np.maximum(absphi(wk*tt/np.sqrt(rho2)),1e-300))
        best=max(best,np.trapezoid(np.exp(lg),tt))
    print("c=%.2f sup integral=%.4f"%(c,best))
