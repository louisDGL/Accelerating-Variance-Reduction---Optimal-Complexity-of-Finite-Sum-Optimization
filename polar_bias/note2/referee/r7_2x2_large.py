"""Same field as r6 (import), large b. mode zero: one-sided fields at m=0 (locking test); mode root: corrected root."""
import sys, numpy as np
sys.argv, args = [sys.argv[0], ''], sys.argv[1:]
src=open('r6_2x2.py').read().split("g=3*(3+2*")[0]; exec(src)
g=3*(3+2*(3*np.sqrt(2)-2)/7)/16*2/12
mode=args[0]
for b in [int(s) for s in args[1].split(',')]:
    if mode=='zero':
        fm=field(b,-1e-9,True); fp=field(b,1e-9,True)
        print("b=%d: e01-type field (rank-one-for-all-m atoms dropped) f(0-)=%+.3e f(0+)=%+.3e | corrected f(0-)=%+.3e f(0+)=%+.3e  => 0 is a minimizer: e01-type %s, corrected %s"%(b,fm[0]-fm[1],fp[0]-fp[1],fm[0],fp[0],(fm[0]-fm[1])<=0<=(fp[0]-fp[1]),fm[0]<=0<=fp[0]),flush=True)
    else:
        lo,hi=-0.02,0.3
        for _ in range(21):
            mid=(lo+hi)/2
            if field(b,mid,True)[0]>0: hi=mid
            else: lo=mid
        r=(lo+hi)/2
        print("b=%d mod3=%d corrected root b*G=%+.6f  sqrt(b)*err=%+.4f"%(b,b%3,r,np.sqrt(b)*(r-g)),flush=True)
