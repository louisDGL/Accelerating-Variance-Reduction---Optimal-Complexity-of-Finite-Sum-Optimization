"""Independent exact computation, Euclid R^2, entries -b+3Bin(b,1/3): root m of E o(m(1,1)+S_b).(1,1)=0; compare with prediction."""
import numpy as np
from scipy.stats import binom
from scipy.optimize import brentq
pred={0:-1.0119,1:0.3326,2:-0.1775}
def mb(b):
    sd=np.sqrt(2*b/9); k=np.arange(max(0,int(b/3-12*sd)),min(b,int(b/3+12*sd))+1)
    p=binom.pmf(k,b,1/3); x=-b+3.0*k
    X,Y=np.meshgrid(x,x,indexing='ij'); W=np.outer(p,p)
    def f(m):
        r=np.hypot(X+m,Y+m); r[r==0]=np.inf
        return np.sum(W*(X+Y+2*m)/r)
    # f is nondecreasing; kink possible at m=0 (class 0)
    lo,hi=-1e-9,1e-9
    if f(lo)<0<f(hi): return 0.0
    return brentq(f,-0.5,0.6,xtol=1e-13)
for b in [1026,4098,16386, 4096,16384, 2048,8192,32768]:
    m=mb(b); print("b=%6d mod3=%d  b*G=%.6f  sqrt(b)*err=%+.4f  pred %+.4f"%(b,b%3,m,np.sqrt(b)*(m-0.125),pred[b%3]))
