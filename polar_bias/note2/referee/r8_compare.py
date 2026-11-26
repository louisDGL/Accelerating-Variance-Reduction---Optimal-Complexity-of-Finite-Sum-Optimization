import sys, numpy as np, importlib.util
sys.argv=['x']
spec=importlib.util.spec_from_file_location('e01','../checks/e01_exact_lattice.py'); e01=importlib.util.module_from_spec(spec); spec.loader.exec_module(e01)
src=open('r6_2x2.py').read().split("g=3*(3+2*")[0]; exec(src)
for b in [17,130]:
    for m in [0.03,0.0588,0.1341,0.1386]:
        t=e01.field_sum((2,2),b,m,5.5); mine=field(b,m,True)
        print("b=%d m=%.4f  e01 field=%+.6e | mine: without persistent atoms %+.6e, with %+.6e (persistent part %+.3e)"%(b,m,t,mine[0]-mine[1],mine[0],mine[1]))
# independent third opinion: SVD-based polar on all atoms, b=17
b=17; x,w=axis(b); 
import itertools
for m in [0.03,0.0588]:
    A,B,C,D=np.meshgrid(x,x,x,x,indexing='ij'); W=np.einsum('i,j,k,l->ijkl',w,w,w,w)
    M=np.stack([np.stack([A+m,B+m],-1),np.stack([C+m,D+m],-1)],-2).reshape(-1,2,2)
    U,s,Vt=np.linalg.svd(M); P=U@Vt
    # on (numerically) singular matrices keep only the top singular pair (minimal-norm subgradient; exact along 11^T for persistent atoms)
    sing=s[:,1]<1e-9*np.maximum(s[:,0],1e-300)
    top=np.einsum('ni,nj->nij',U[:,:,0],Vt[:,0,:])
    val=np.where(sing,top.sum((1,2)),P.sum((1,2)))
    val=np.where(s[:,0]==0,0,val)
    print("b=17 m=%.4f SVD-based field (rank-one part on singular atoms): %+.6e"%(m,np.sum(W.ravel()*val)))
