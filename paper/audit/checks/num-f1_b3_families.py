"""Audit num-f1: the sentence of (C2) on the designs of Na, Derezinski, Mahoney ("eight data sets ... 64-98 % ...
13 to 198 at kappa = 1e5") recomputed from results/B3_families_summary.json (read only)."""
import json

s = json.load(open("results/B3_families_summary.json"))
a = s["attribution"]
fam = lambda k: k.rsplit(",kappa=", 1)[0]
kap = lambda k: float(k.rsplit(",kappa=", 1)[1])
ndm = sorted({fam(k) for k in a if k[0] in "PLR"})
ours = sorted({fam(k) for k in a if k[0] == "C"})
print("NDM-design data sets:", len(ndm), ndm)
print("own-generator data sets (data seeds 1, 2):", len(ours))


def rng(v):
    return f"[{min(v):.4f}, {max(v):.4f}]"


for name, fams in (("NDM (P, L, R)", ndm), ("P only", [f for f in ndm if f[0] == "P"]),
                   ("P+R (labels of the design)", [f for f in ndm if f[0] in "PR"]), ("own generator, seeds 1-2", ours)):
    for sel, lab in (((1e4, 1e5), "kappa>=1e4"), ((1e3, 1e4, 1e5), "kappa>=1e3"), ((10., 100., 1e3, 1e4, 1e5), "all kappa")):
        v = [a[k]["shares"]["I2"] + a[k]["shares"]["I4"] for k in a if fam(k) in fams and kap(k) in sel]
        vA = [a[k]["shares_orderA"]["I2"] + a[k]["shares_orderA"]["I4"] for k in a if fam(k) in fams and kap(k) in sel]
        print(f"  {name:28s} {lab:11s} I2+I4 order B {rng(v)}   order A {rng(vA)}  ({len(v)} problems)")
    sp = {fam(k): a[k]["speedup"] for k in a if fam(k) in fams and kap(k) == 1e5}
    print(f"  {name:28s} speed-up at kappa=1e5: {rng(list(sp.values()))}")
    sp2 = [a[k]["speedup"] for k in a if fam(k) in fams and kap(k) <= 100]
    print(f"  {name:28s} speed-up at kappa<=1e2: {rng(sp2)}")
print("per data set (kappa=1e4, 1e5): speed-up, I2+I4 (B)")
for f in ndm + ours:
    print(f"  {f:22s} " + "  ".join(f"k={kk:g}: {a[f+',kappa='+format(kk,'g')]['speedup']:7.2f}, "
                                      f"{a[f+',kappa='+format(kk,'g')]['shares']['I2']+a[f+',kappa='+format(kk,'g')]['shares']['I4']:.4f}"
                                      for kk in (1e4, 1e5)))
print("largest I3 share (order B, seed mean) over the 70 problems:", max(v["shares"]["I3"] for v in a.values()))
