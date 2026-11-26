import numpy as np
from scipy.special import hyp2f1
rng = np.random.default_rng(0)

print("== (A) 1D momentum, coupled dynamics vs -median(N_beta) ==")
def med_stationary(beta, S=4_000_000, noise=lambda n: rng.exponential(size=n)-1):
    N = np.zeros(S)
    for _ in range(int(np.ceil(np.log(1e-9)/np.log(beta))) if beta>0 else 1):
        N = beta*N + (1-beta)*noise(S)
    return np.median(N), 1.2533*N.std()/np.sqrt(S)
for beta, theta in [(0.5,0.75),(0.9,0.75),(0.9,1.0),(0.9,0.55)]:
    med, se = med_stationary(beta)
    R, T = 2000, 200_000
    w = np.full(R, 0.3); M = np.zeros(R)
    eta0, t0 = (0.5, 10) if theta<1 else (3.0, 50)
    tail = np.zeros(R); cnt = 0
    for t in range(1, T+1):
        M = beta*M + (1-beta)*(w + rng.exponential(size=R)-1)
        w = w - eta0*(t+t0)**(-theta)*np.sign(M)
        if t > T*0.9: tail += w; cnt += 1
    print(f"beta={beta} theta={theta}: -median(N_beta)={-med:.5f}+-{se:.5f}; final mean G_T={w.mean():.5f} (sd over runs {w.std():.5f}); tail-avg mean={ (tail/cnt).mean():.5f}+-{(tail/cnt).std()/np.sqrt(R):.5f}")

print("== (B) Li-Sun bound ||polar A - polar B||_F <= 2||A-B||_F/(s_r(A)+s_r(B)) ==")
def polar(M):
    U,s,Vt = np.linalg.svd(M, full_matrices=False); return U@Vt
worst = 0
for K,d in [(2,2),(2,5),(3,3),(4,7)]:
    for _ in range(20000):
        A = rng.standard_normal((K,d))*rng.choice([1,.1,.01])
        if rng.random()<.5:  # near singular
            U,s,Vt = np.linalg.svd(A, full_matrices=False); s[-1]*=10**rng.uniform(-4,0); A=(U*s)@Vt
        B = A + rng.standard_normal((K,d))*10**rng.uniform(-4,0)
        sa = np.linalg.svd(A,compute_uv=False)[-1]; sb=np.linalg.svd(B,compute_uv=False)[-1]
        ratio = np.linalg.norm(polar(A)-polar(B))*(sa+sb)/(2*np.linalg.norm(A-B))
        worst = max(worst, ratio)
print("max ratio LHS/RHS over 80000 random pairs:", worst)

print("== (C) filter bias via regrouping, 1D, exp noise, signed weights ==")
# N_w = sum_k w_k xi_b^{(k)}, xi_b = Gamma(b)/b - 1 ; prediction -median*b -> (rho3/rho2)*kappa3/(6 sigma^2) = (rho3/rho2)/3
S = 4_000_000
for w in [(1.0,), (1.0,-0.5), (0.6,0.3,0.1), (1.0,-1.0), (0.5,-0.9)]:
    w = np.array(w); r2=(w**2).sum(); r3=(w**3).sum()
    for b in [8, 32, 128]:
        N = sum(wk*(rng.gamma(b, size=S)/b-1) for wk in w)
        med = np.median(N); se = 1.2533*N.std()/np.sqrt(S)
        print(f"w={w} b={b}: -b*median={-b*med:.4f}+-{b*se:.4f}  predicted (rho3/rho2)/3={r3/r2/3:.4f}")

print("== (D) small ball with non-product covariance, 3x3 and 2x4 ==")
for K,d in [(3,3),(2,4)]:
    N=K*d; A=rng.standard_normal((N,N)); Sig=A@A.T/N+0.05*np.eye(N); L=np.linalg.cholesky(Sig)
    lam=np.linalg.eigvalsh(Sig)[0]
    bound=lambda t: K*2*np.sqrt(K)*t/np.sqrt(2*np.pi*lam)
    for yname,y in [("0",np.zeros((K,d))),("rank1", 3*np.outer(np.ones(K),np.ones(d))),("rand",2*rng.standard_normal((K,d)))]:
        Z=(rng.standard_normal((2_000_000,N))@L.T).reshape(-1,K,d)+y
        s=np.linalg.svd(Z,compute_uv=False)[:,-1]
        print(K,d,yname,"lam_min=%.3f"%lam," ".join(f"t={t}: P/t={np.mean(s<=t)/t:.3f} (bound {bound(t)/t:.2f})" for t in [0.3,0.1,0.03,0.01]))

print("== (E) theta(gamma)=6 kappa(gamma) ==")
mu=lambda s,g: hyp2f1(-s,1-s,2,g)
kap=lambda g: ((1+g)*mu(.5,g)-mu(1.5,g))/(g*mu(.5,g))
gs=np.linspace(1e-4,1,2001); th=6*kap(gs)
print("theta(1)=",6*kap(1.0)," theta(1/4)=",6*kap(.25)," theta(1e-4)=",th[0]," monotone decreasing on grid:",bool(np.all(np.diff(th)<0))," min=",th.min())
print("a1<0 in limit: min over grid of kappa-1/(2(1+g)) =",(kap(gs)-1/(2*(1+gs))).min())
# check hypergeometric form against MP integral for s=1/2,3/2 at gamma=.3
from scipy.integrate import quad
g=.3;a,b_=(1-np.sqrt(g))**2,(1+np.sqrt(g))**2
for s in (.5,1.5):
    I=quad(lambda x: x**s*np.sqrt((b_-x)*(x-a))/(2*np.pi*g*x),a,b_)[0]; print("mu_",s,I,mu(s,g))
