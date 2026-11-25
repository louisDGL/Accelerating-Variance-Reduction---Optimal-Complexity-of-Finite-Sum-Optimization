"""Audit num-f1: recompute every (C1)/(C2) number of the paper from results/B3_summary.json (read only).
Run from the project root: OPENBLAS_NUM_THREADS=1 python3 paper/audit/checks/num-f1_b3_summary.py
"""
import json

s = json.load(open("results/B3_summary.json"))
a, p = s["attribution"], s["problems"]
K = [10, 100, 1000, 10000, 100000]
A = [0, 1, 2]
key = lambda k, al: f"kappa={k:g},alpha={al:g}"


def rng(vals):
    return f"[{min(vals):.4f}, {max(vals):.4f}]"


print("== (C1) speed-up = mean over seeds of (window mean of -log rho_exact)/(window mean of -log rho_bound), passes 2-20")
for al in A:
    print(f"  alpha={al}: " + ", ".join(f"{a[key(k, al)]['speedup']:.4f}" for k in K))
lo = [a[key(k, al)]["speedup"] for k in (10, 100) for al in A]
hi = [a[key(100000, al)]["speedup"] for al in A]
al_ = [v["speedup"] for v in a.values()]
print("  kappa<=1e2 :", rng(lo), "  paper: 3.2-3.7")
print("  kappa=1e5  :", rng(hi), "  paper: 19-220")
print("  all 15     :", rng(al_), "  paper (abstract, Table 1): 3.2-220")
# consistency: speedup vs ratio of seed-mean exponents stored separately
for kk, v in a.items():
    assert abs(v["c_exact_per_iter"] / v["c_bound_per_iter"] / v["speedup"] - 1) < 1e-3, kk

print("== (C2) shares, order B (undo I5, I4, I2, I3, I1), mean over 5 seeds")
big = [a[key(k, al)] for k in (10000, 100000) for al in A]
small = [a[key(k, al)] for k in (10, 100) for al in A]
print("  kappa>=1e4 I2+I4 :", rng([v["shares"]["I2"] + v["shares"]["I4"] for v in big]), " paper: 77-96 %")
print("  kappa<=1e2 I1    :", rng([v["shares"]["I1"] for v in small]), " paper: 39-44 %")
print("  kappa<=1e2 I2    :", rng([v["shares"]["I2"] for v in small]), " paper: 32-37 %")
print("  I3 mean, all 15  :", rng([v["shares"]["I3"] for v in a.values()]), " paper: at most 0.6 %")
print("  I3 max over seeds, all 15:", max(v["shares_max_over_seeds"]["I3"] for v in a.values()))
print("  I3 late window 20-100, all 15:", rng([v["shares_late_20_100"]["I3"] for v in a.values() if "shares_late_20_100" in v]))
print("  memory term active (1/(3n) < 3 mu/(16 R2))?")
for k in K:
    pr = p[key(k, 1)]
    print(f"    kappa={k:g}: 1/(3n)={1/(3*pr['n']):.3e}  3mu/(16R2)={3*pr['mu']/(16*pr['R2']):.3e}  ->",
          "memory" if 1 / (3 * pr["n"]) < 3 * pr["mu"] / (16 * pr["R2"]) else "parameter")

print("== (C2) factors of eq. (saga-repr) from the window geometric means (kappa>=1e4)")
f1s, f2s, rat = [], [], []
for k in (10000, 100000):
    for al in A:
        v, pr = a[key(k, al)], p[key(k, al)]
        f1 = (1 - v["L_eff_geomean"] / pr["R2"] * (v["r_sq_geomean"] / 4 + 3 / 8)) / (3 / 8)
        f2 = v["mu_eff_geomean"] / pr["mu"]
        f1s.append(f1); f2s.append(f2); rat.append(f1 * f2 / v["speedup"])
        print(f"    {key(k, al):22s} first={f1:.4f} second={f2:.3f} product={f1*f2:.2f} speedup={v['speedup']:.2f} ratio={f1*f2/v['speedup']:.4f}")
print("  first factor :", rng(f1s), " paper: 2.45-2.62")
print("  second factor max:", max(f2s), " paper: reaches 81")
print("  product/speed-up:", rng(rat), " paper: 91-96 %")

print("== order A (undo I4, I2, I3, I1, then I5), same runs")
for kk, v in a.items():
    o = v["shares_orderA"]
    print(f"    {kk:22s} I1 {o['I1']:.3f} I2 {o['I2']:.3f} I3 {o['I3']:.3f} I4 {o['I4']:.3f} I5 {o['I5']:.3f} | I2+I4 A {o['I2']+o['I4']:.3f}  B {v['shares']['I2']+v['shares']['I4']:.3f}")
print("  kappa>=1e4 I2+I4 order A:", rng([v["shares_orderA"]["I2"] + v["shares_orderA"]["I4"] for v in big]))
print("  kappa<=1e2 I2 order A   :", rng([v["shares_orderA"]["I2"] for v in small]),
      " I5 order A:", rng([v["shares_orderA"]["I5"] for v in small]))
print("  I3 order A, all 15      :", rng([v["shares_orderA"]["I3"] for v in a.values()]))
