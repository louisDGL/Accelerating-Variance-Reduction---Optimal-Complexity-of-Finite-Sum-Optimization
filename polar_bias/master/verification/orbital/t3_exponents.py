"""Items 2-3: exponent bookkeeping (sympy), lattice-sum scaling, convex sandwich, example constant."""
import sympy as sp, numpy as np
N,k,e,d,R=sp.symbols('N kappa epsilon delta R',positive=True)
q=N/(N-k)
# sum_j: R^{N+1} (R|j|/eps)^{-2-k}  -> R^{N-1-k} eps^{2+k};  sum_{|j|<eps/delta}|j|^{-2-k} ~ (eps/delta)^{N-2-k}
expo_eps=sp.simplify((2+k)+(N-2-k)); print("eps exponent:",expo_eps,"(= N)")
# eps^N = delta^{N/q}; total delta exponent
print("delta exponent:",sp.simplify(N/q-(N-2-k)),"(= 2)")
print("2q<3 <=> N>3kappa:",sp.simplify(3-2*q),"  at kappa=1/2:",sp.simplify((3-2*q).subs(k,sp.Rational(1,2))))
print("(1+q)/2 - 1 - 1/(4N-2) at kappa=1/2:",sp.simplify(((1+q)/2-1-1/(4*N-2)).subs(k,sp.Rational(1,2))))
print("irreducible case kappa=1/N: (1+q)/2 - 1 =",sp.simplify(((1+q)/2-1).subs(k,1/N)))
# dilation: F_R(z)=R F_1(z/R) => hat F_R(t)=R^{N+1} hat F_1(Rt)   (homogeneity degree 1 + Jacobian R^N)
# lattice sum scaling in N=3,4 with kappa=1/2, D=N+1
for Nn in (3,4):
    kap=.5; D=Nn+1; L=60
    g=np.stack(np.meshgrid(*[np.arange(-L,L+1)]*Nn,indexing='ij'),-1).reshape(-1,Nn).astype(float)
    r=np.linalg.norm(g,axis=1); r=r[r>0]
    print("N=%d  T, sum/T^{N-2-kappa}:"%Nn,[(T,round(float(np.sum(r**(-2-kap)*(1+r/T)**(-D))/T**(Nn-2-kap)),3)) for T in (2,4,8,16)])
# convex sandwich 2 nu*h_d - nu*h_2d <= nu <= nu*h_d for nuclear norm 3x3, h uniform on Frobenius ball (even, unit mass)
rng=np.random.default_rng(0); nuc=lambda M: np.linalg.svd(M,compute_uv=False).sum(axis=-1)
u=rng.normal(size=(200000,3,3)); u/=np.linalg.norm(u,axis=(1,2),keepdims=True); u*=rng.uniform(size=(200000,1,1))**(1/9)
u=np.concatenate([u,-u])       # antithetic => exactly even empirical kernel, inequalities hold samplewise
worst=[1e9,1e9]
for _ in range(40):
    z=rng.normal(size=(3,3))*rng.choice([0,.01,.1,1]); 
    if rng.random()<.5: z[2]=z[0]+z[1]          # singular point
    dl=.05; a=nuc(z-dl*u).mean(); b2=nuc(z-2*dl*u).mean(); n0=nuc(z)
    worst=[min(worst[0],n0-(2*a-b2)),min(worst[1],a-n0)]
print("sandwich slacks (must be >=0): nu-(2nu*h-nu*h2) min %.2e ; nu*h-nu min %.2e"%tuple(worst))
# example constant: (a1+a2) kappa3/(6 sigma^2) with K=d=k
qq,kk=sp.symbols('q k'); a12=3*(3+2*qq)/(kk+2)**2
print("g* entry:",sp.simplify(a12*2/(6*2)),"  vs (3+2q)/(2(k+2)^2)")
# sign-descent values 0,1,-1: nearest point of b+3Z to 1/6
print("sign descent b mod 3 = 0,1,2 ->",[min([b+3*j for j in range(-3,3)],key=lambda x:abs(x-1/6)) for b in (0,1,2)])
