"""Exact medians of lattice sums vs Theorem 'sign'. b*G_b = -(medians of S_b)."""
import numpy as np
from fractions import Fraction as Fr
from scipy.stats import binom
import mpmath as mp
def pmf_conv(vals, probs, b):
    """exact-ish pmf of S_b on integer offsets by repeated squaring convolution (float64); vals integers"""
    lo=min(vals); base=np.zeros(max(vals)-lo+1); 
    for v,p in zip(vals,probs): base[v-lo]+=p
    res=np.array([1.0]); off=0; cur=base; curoff=lo; n=b
    while n:
        if n&1: res=np.convolve(res,cur); off+=curoff
        cur=np.convolve(cur,cur); curoff*=2; n>>=1
    return off,res
def medians(off,pm,tol=1e-12):
    c=np.cumsum(pm); k=np.arange(len(pm))+off
    # lattice medians: P(S<m)<=1/2<=P(S<=m)
    lt=np.concatenate([[0],c[:-1]])
    return [(int(kk), l, cc) for kk,l,cc in zip(k,lt,c) if l<=0.5+tol and cc>=0.5-tol]
print("== {-1,2} w.p. 2/3,1/3: g*=1/6, expect b*G_b = 0,+1,-1 for b mod 3 = 0,1,2")
bad=[]
for b in range(1,3003):
    # S_b = -b+3 Bin(b,1/3)
    k=np.arange(b+1); c=binom.cdf(k,b,1/3); lt=np.concatenate([[0],c[:-1]])
    med=[-b+3*kk for kk,l,cc in zip(k,lt,c) if l<=0.5 and cc>=0.5]
    exp={0:0,1:-1,2:1}[b%3]
    if med!=[exp]: bad.append((b,med))
print(" violations for b in 1..3002:",bad[:20])
print("== {-1,0,2}, P(-1)=2p,P(2)=p: h=1,a=0,g*=1/6 -> predicted G_b={0} for large b")
for p in [0.05,0.2,1/3-0.01]:
    bad=[]
    for b in list(range(1,400))+[1000,3000]:
        off,pm=pmf_conv([-1,0,2],[2*p,1-3*p,p],b); m=[x[0] for x in medians(off,pm)]
        if m!=[0]: bad.append((b,m))
    print(" p=%.3f  b with median set != {0}:"%p,bad[:12],"... last bad b:",bad[-1][0] if bad else None)
print("== tie case {-1,0,4}, P(-1)=4p,P(4)=p: h=1,a=0,g*=kappa3/(6 sigma^2)=1/2 exactly (q=1, g* in h/2+hZ)")
for p in [0.05,0.15]:
    out=[]
    for b in [10,30,100,101,300,301,1000,1001,3000,3001]:
        off,pm=pmf_conv([-1,0,4],[4*p,1-5*p,p],b); out.append((b,[-x[0] for x in medians(off,pm)]))
    print(" p=%.2f  (b, b*G_b):"%p,out)
print("== irrational a/h: zeta in {-s,1-s}, P(1-s)=s, s=1/sqrt2: h=1,a=-s; S_b=Bin(b,s)-bs; g*=(1-2s)/6")
s=1/np.sqrt(2); g=(1-2*s)/6
viol=0; offs=[]; worst=0
for b in range(200,20001):
    k0=int(np.floor(b*s)); ks=np.arange(k0-3,k0+5)
    c=binom.cdf(ks,b,s); l=binom.cdf(ks-1,b,s)
    med=[kk-b*s for kk,ll,cc in zip(ks,l,c) if ll<=0.5 and cc>=0.5]
    assert len(med)==1
    bG=-med[0]; offs.append(bG-g)
    # predicted: point of -(ba+Z)=bs-Z nearest to g
    pred=b*s-np.round(b*s-g)
    d=abs(abs(pred-g)-0.5)   # margin delta
    if abs(pred-bG)>1e-9:
        viol+=1; worst=max(worst,d)
offs=np.array(offs)
print(" b in 200..20000: #b where median != nearest point: %d (largest margin |dist-1/2| among them: %.4f)"%(viol,worst))
print(" range of b*G_b-g*: [%.4f, %.4f]; histogram (10 bins on [-.5,.5]):"%(offs.min(),offs.max()),np.histogram(offs,bins=10,range=(-.5,.5))[0])
print("== rational a/h with q=2: zeta in {-1/2,1/2}+... skewed: zeta in {-1,1,3}-mean? use {-1,1} asym impossible (mean 0) -> take {-3,-1,1,3}/.. P chosen mean 0")
vals=[-1,1,5]; # span 2, a=1 (odd), mean zero needs -p1+p2+5p3=0
p3=0.05; p2=0.3; p1=p2+5*p3; rest=1-p1-p2-p3
assert abs(rest)<1e-12 or True
vals=[-1,1,5,-3]; 
# solve: probabilities (p1,p2,p3,p4) mean zero
p=np.array([0.5,0.3,0.1,0.1]); # mean=-.5+.3+.5-.3=0
m1=np.dot(vals,p); s2=np.dot(np.array(vals)**2,p); k3=np.dot(np.array(vals)**3,p); g=k3/(6*s2)
print(" law",vals,p,"mean",m1,"g*=%.4f, h=2,a=1 -> q=2; Lambda_b=-(b+2Z)"%g)
for b in [50,51,200,201,1000,1001,2000,2001]:
    off,pm=pmf_conv(vals,list(p),b); 
    med=[-x[0] for x in medians(off,pm)]
    lat=np.array([x for x in range(-9,10) if (x-b)%2==0]); pred=lat[np.argmin(abs(lat-g))]
    print("  b=%d b*G_b=%s predicted %d"%(b,med,pred))
