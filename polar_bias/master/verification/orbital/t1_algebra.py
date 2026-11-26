"""Item 1 algebra: span of first/second-order derivatives of g -> g.M at the identity
for SO(k)xSO(l) acting on R^{k x l} by M -> U M V^T, at nonzero M of every rank.
Derivative vectors: A M, M B (order 1); A A' M, M B B', A M B (order 2). A,B skew."""
import numpy as np, itertools
rng = np.random.default_rng(0)
def skew_basis(n):
    out=[]
    for i in range(n):
        for j in range(i+1,n):
            A=np.zeros((n,n)); A[i,j]=1; A[j,i]=-1; out.append(A)
    return out
def span_rank(vecs, tol=1e-9):
    if not vecs: return 0
    return np.linalg.matrix_rank(np.array([v.ravel() for v in vecs]), tol=tol)
def deriv_span(M, order2=True, mixed=True):
    k,l=M.shape; SA=skew_basis(k); SB=skew_basis(l)
    v=[A@M for A in SA]+[M@B for B in SB]
    if order2:
        v+=[A@A2@M for A in SA for A2 in SA]+[M@B@B2 for B in SB for B2 in SB]
        if mixed: v+=[A@M@B for A in SA for B in SB]
    return span_rank(v)
# products of two skew matrices span gl(s) iff s>=3
for s in range(2,7):
    S=skew_basis(s); print("size",s,"span{AA'} dim",span_rank([A@B for A in S for B in S]),"of",s*s)
print("k l rank | dim(order1) dim(order<=2 no mixed) dim(order<=2) | N")
for k,l in [(2,2),(3,3),(3,4),(4,4),(5,3),(3,2),(4,2),(2,5),(3,1),(5,5)]:
    for r in range(1,min(k,l)+1):
        worst=[10**9]*3
        for trial in range(20):
            U=np.linalg.qr(rng.normal(size=(k,k)))[0]; V=np.linalg.qr(rng.normal(size=(l,l)))[0]
            s=np.zeros((k,l)); 
            sv = rng.uniform(.5,2,size=r) if trial%2 else np.ones(r)   # include equal singular values
            for i in range(r): s[i,i]=sv[i]
            M=U@s@V.T if trial>1 else s
            d=(deriv_span(M,False),deriv_span(M,True,False),deriv_span(M))
            worst=[min(a,b) for a,b in zip(worst,d)]
        print(k,l,r,"|",*worst,"|",k*l, "" if worst[2]==k*l else "  <-- NOT FULL")
# explicit 2x2 failure: y = I/sqrt2, e = diag(1,-1)/sqrt2 : phase identically 0
e=np.diag([1.,-1.])/np.sqrt(2); y=np.eye(2)/np.sqrt(2)
mx=0
for _ in range(10000):
    a,b=rng.uniform(0,2*np.pi,2)
    R=lambda t: np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
    mx=max(mx,abs(np.sum(e*(R(a)@y@R(b).T))))
print("2x2: max |<e,U y V^T>| over 1e4 random (U,V), e=diag(1,-1)/sqrt2, y=I/sqrt2:",mx," => orbital integral == 1 for all u")
# irreducible chain: V_j dims for SO(3)xSO(3) on 3x3 at rank-1 M, and for SO(3) on R^3
M=np.zeros((3,3)); M[0,0]=1
print("3x3 rank-1 M: dim V_1 =",deriv_span(M,False)," dim V_2 =",deriv_span(M))
