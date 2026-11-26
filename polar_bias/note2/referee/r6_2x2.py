"""2x2 polar, entries -b+3Bin(b,1/3). Field f(m)=<E o(m 11^T+S_b),11^T>, o in subdifferential of nuclear norm.
e01 sets the contribution of matrices with det(MM^T)=0 to 0.  But atoms L with L(1,-1)^T=0 or (1,-1)L=0 (equal columns / equal rows)
stay rank one for ALL m; there m -> ||L+m11^T||_* is smooth with derivative 2*y+_1/|y+| != 0.  Compare roots with/without them."""
import numpy as np, sys
from scipy.stats import binom
def axis(b,T=5.5):
    mu,sd=b/3,np.sqrt(b*2/9); lo,hi=max(0,int(np.floor(mu-T*sd))),min(b,int(np.ceil(mu+T*sd)))
    k=np.arange(lo,hi+1); return -b+3.0*k, binom.pmf(k,b,1/3)
def field(b,m,include=True):
    x,w=axis(b); B,C,D=np.meshgrid(x,x,x,indexing='ij'); W3=np.einsum('j,k,l->jkl',w,w,w)
    tot=0.0; pers_mass=0.0; pers=0.0
    for a,wa in zip(x,w):
        yp1=(a+D+2*m); yp2=(B-C); ym1=(a-D); ym2=(B+C+2*m)
        n1=np.hypot(yp1,yp2); n2=np.hypot(ym1,ym2)
        persistent=((a+D)==(B+C))&((a*D)==(B*C))
        big=n1>n2
        with np.errstate(invalid='ignore',divide='ignore'):
            v=np.where(big,2*yp1/n1,2*ym2/n2)
            vp=np.where(n1>0,2*yp1/n1,0.0)
        v=np.where(persistent,0.0,v)
        tot+=wa*np.sum(W3*v); pers+=wa*np.sum(W3[persistent]*vp[persistent]); pers_mass+=wa*W3[persistent].sum()
    return (tot+pers if include else tot), pers, pers_mass
def root(b,include):
    lo,hi=-0.5,0.8
    for _ in range(34):
        mid=(lo+hi)/2
        if field(b,mid,include)[0]>0: hi=mid
        else: lo=mid
    return (lo+hi)/2
g=3*(3+2*(3*np.sqrt(2)-2)/7)/16*2/12
for b in [int(s) for s in sys.argv[1].split(',')]:
    r0=root(b,False); r1=root(b,True); _,P,pm=field(b,r0,True)
    print("b=%4d mod3=%d  root dropping rank-one-for-all-m atoms (as e01): %+.6f | correct subgradient: %+.6f | sqrt(b)*err: %+.4f -> %+.4f | persistent mass %.2e, their field contribution at e01 root %+.2e"%(b,b%3,r0,r1,np.sqrt(b)*(r0-g),np.sqrt(b)*(r1-g),pm,P),flush=True)
