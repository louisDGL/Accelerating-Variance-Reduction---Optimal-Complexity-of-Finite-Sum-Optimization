"""Item 6. 3x3, iid entries in {-1,2} w.p. 2/3,1/3. S_b = -b + 3 Bin(b,1/3) entrywise.
h_b(m) = E || m 11^T + S_b ||_*  (restriction to 11^T is exact: the law of S_b is invariant under row and
column permutations, H is convex, so averaging any minimiser over S_3 x S_3 gives a minimiser m 11^T;
only non-uniqueness of the full argmin set is not probed).  Theorem predicts argmin -> g_* = (3+2q_33)/50.
Sample-average minimiser with a mean-zero linear control variate, batches for standard errors.
Also the exact one-sided derivatives (subdifferential) at the lattice points m = b mod 3."""
import numpy as np, sys, time
from multiprocessing import Pool
from scipy.optimize import minimize_scalar
ONE=np.ones((3,3))
def nuc(M): return np.linalg.svd(M,compute_uv=False).sum(axis=1)
def onesided(S,m):
    """per-sample base derivative and kink half-jump of t->||(m+t)11^T+S||_* at t=0"""
    U,s,Vt=np.linalg.svd(m*ONE+S)
    a=U.sum(axis=1); c=Vt.sum(axis=2)            # a_i=<u_i,1>, c_i=<v_i,1>
    null=s<1e-7*np.maximum(s[:,:1],1)
    base=np.where(null,0,a*c).sum(axis=1)
    jump=np.sqrt(np.where(null,a*a,0).sum(axis=1)*np.where(null,c*c,0).sum(axis=1))
    return base,jump
def run(args):
    b,seed,nb,bs=args
    rng=np.random.default_rng(seed)
    mins=[]; d0=[]; psing=[]; grid=np.array([-0.3,-0.15,0.,0.05,0.1,0.15,0.2,0.3,0.45]); gd=[]
    lat=[m for m in (-2,-1,0,1,2) if (m-b)%3==0]; latd={m:[] for m in lat}
    for _ in range(nb):
        S=(-b+3*rng.binomial(b,1/3,size=(bs,3,3))).astype(float)
        T=S.sum(axis=(1,2))                       # <S,11^T>, exact mean 0, var 18 b
        base,jump=onesided(S,0.0)
        beta=np.cov(base,T)[0,1]/(18*b)
        f=lambda m: nuc(m*ONE+S).mean()-beta*m*T.mean()
        r=minimize_scalar(f,bounds=(-4,4),method='bounded',options={'xatol':2e-4})
        mins.append(r.x)
        Si=np.rint(S).astype(np.int64)
        det=(Si[:,0,0]*(Si[:,1,1]*Si[:,2,2]-Si[:,1,2]*Si[:,2,1])-Si[:,0,1]*(Si[:,1,0]*Si[:,2,2]-Si[:,1,2]*Si[:,2,0])+Si[:,0,2]*(Si[:,1,0]*Si[:,2,1]-Si[:,1,1]*Si[:,2,0]))
        psing.append((det==0).mean())
        for m in lat:
            bb,jj=onesided(S,float(m)); cv=beta*T.mean()
            latd[m].append((bb.mean()-cv-jj.mean(), bb.mean()-cv+jj.mean()))
        sub=S[:bs//4]; Tm=T[:bs//4].mean()
        gd.append([onesided(sub,g+1e-9)[0].mean()-beta*Tm for g in grid])
    mins=np.array(mins)
    out=dict(b=b,n=nb*bs,m=mins.mean(),se=mins.std(ddof=1)/np.sqrt(nb),med=np.median(mins),psing=np.mean(psing),
             lat={m:(np.mean(v,axis=0),np.std(v,axis=0,ddof=1)/np.sqrt(nb)) for m,v in latd.items()},
             grid=grid,gd=np.mean(gd,axis=0),gse=np.std(gd,axis=0,ddof=1)/np.sqrt(nb),mins=mins)
    return out
if __name__=="__main__":
    rng=np.random.default_rng(7)
    # q_{3,3}
    acc=[]; 
    for _ in range(20):
        s=np.linalg.svd(rng.normal(size=(500_000,3,3)),compute_uv=False)
        num=sum(2*s[:,i]*s[:,j]/(s[:,i]+s[:,j]) for i in range(3) for j in range(i+1,3))
        acc.append((num.mean(),s.sum(axis=1).mean(),(s**3).sum(axis=1).mean()))
    acc=np.array(acc); q=acc[:,0]/acc[:,1]
    qm=acc[:,0].mean()/acc[:,1].mean(); qse=q.std(ddof=1)/np.sqrt(len(q))
    print("q_33 = %.5f +- %.5f ; identity check m3/m1 = %.5f vs 6-q = %.5f"%(qm,qse,acc[:,2].mean()/acc[:,1].mean(),6-qm))
    gstar=(3+2*qm)/50; print("g_* = (3+2q)/50 = %.5f   (m1/9 = c_33 = %.4f)"%(gstar,acc[:,1].mean()/9),flush=True)
    bs_list=[12,13,14,30,31,32,99,100,101,300,301,302,999,1000,1001]
    nb=int(sys.argv[1]) if len(sys.argv)>1 else 16; bs=int(sys.argv[2]) if len(sys.argv)>2 else 400_000
    t=time.time()
    with Pool(14) as p: res=p.map(run,[(b,1000+b,nb,bs) for b in bs_list])
    print("time %.0fs, n=%d per b"%(time.time()-t,nb*bs))
    print("\n b  bmod3   argmin m (mean of batch minimisers +- se)   ratio to g_*    P(det S_b=0)")
    for r in res:
        print("%5d  %d   %.4f +- %.4f   %.3f +- %.3f   %.5f"%(r['b'],r['b']%3,r['m'],r['se'],r['m']/gstar,r['se']/gstar,r['psing']))
    print("\nsubdifferential of h_b at lattice points m = b mod 3 (control-variated; locked iff interval contains 0)")
    for r in res:
        for m,(v,se) in r['lat'].items():
            print("  b=%5d m=%+d : [%.5f, %.5f] (se %.5f)  %s"%(r['b'],m,v[0],v[1],se.max(),"LOCKED" if v[0]<=0<=v[1] else ""))
    print("\nh_b'(m) on grid (smooth part, CV), scaled by sqrt(2b):")
    print("   m:    "+"  ".join("%+.2f"%g for g in res[0]['grid']))
    for r in res:
        print("b=%5d "%r['b']+"  ".join("%+.3f"%(x*np.sqrt(2*r['b'])) for x in r['gd'])+"   (se %.3f)"%(r['gse'].max()*np.sqrt(2*r['b'])))
