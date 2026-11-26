"""Third opinion at b=130 (2x2): SVD-based polar on every atom, chunked. On singular atoms use the rank-one part u1 v1^T
(exact directional derivative along 11^T for atoms singular for all m)."""
import numpy as np
from scipy.stats import binom
b=130; mu,sd=b/3,np.sqrt(b*2/9); k=np.arange(int(np.floor(mu-5.5*sd)),int(np.ceil(mu+5.5*sd))+1); x=-b+3.0*k; w=binom.pmf(k,b,1/3)
B,C,D=np.meshgrid(x,x,x,indexing='ij'); W3=np.einsum('j,k,l->jkl',w,w,w).ravel()
for m in [0.1341,0.1386]:
    tot=0.0
    for a,wa in zip(x,w):
        M=np.empty((B.size,2,2)); M[:,0,0]=a+m; M[:,0,1]=B.ravel()+m; M[:,1,0]=C.ravel()+m; M[:,1,1]=D.ravel()+m
        U,s,Vt=np.linalg.svd(M)
        sing=s[:,1]<1e-9*s[:,0]
        full=(U@Vt).sum((1,2)); top=U[:,:,0].sum(1)*Vt[:,0,:].sum(1)
        tot+=wa*np.dot(W3,np.where(sing,top,full))
    print("b=130 m=%.4f SVD field %+.6e"%(m,tot),flush=True)
