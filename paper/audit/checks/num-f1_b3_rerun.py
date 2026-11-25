"""Audit num-f1: independent small reproduction of the B3 numbers behind (C1)-(C3) and Figure 1(b),(c).
results/B3.pkl is not in the repository (git-ignored), so the per-seed quantities are recomputed here: SAGA with the
textbook constants on the 15 problems (n=1000, d=50, data seed 0), sampling seeds 0-4, first 20100 iterations only
(the attribution window is passes 2-20; the index stream of numpy's Generator has the prefix property, checked).
Read only; about 1 minute of CPU, one process.
Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 paper/audit/checks/num-f1_b3_rerun.py
"""
import json
import os
import sys

sys.dont_write_bytecode = True
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import numpy as np  # noqa: E402
import common  # noqa: E402
from vrlab import algorithms, analysis  # noqa: E402

N, D = 1000, 50
KAPPAS = [10.0, 100.0, 1000.0, 10000.0, 100000.0]
ALPHAS = [0.0, 1.0, 2.0]
SEEDS = range(5)
ref = json.load(open(os.path.join(ROOT, "results", "B3_summary.json")))["attribution"]

a = np.random.default_rng(3).integers(0, N, size=100000)[:20100]
b = np.random.default_rng(3).integers(0, N, size=20100)
assert np.array_equal(a, b), "prefix property of the index stream fails"

worst = 0.0
rows = {}
for alpha in ALPHAS:
    for kappa in KAPPAS:
        key = f"kappa={kappa:g},alpha={alpha:g}"
        prob = common.logistic_problem(N, D, kappa, alpha, seed=0)
        sp, shB, shA, nrec, f1_rec, f2_rec, slopeD, slopeV, f1g, f2g = [], [], [], [], [], [], [], [], [], []
        for s in SEEDS:
            r = algorithms.saga(prob, np.zeros(D), n_iters=20100, seed=s, record_every=N // 20,
                                instrument_every=N // 10, stop_D=1e-26)
            g, Dl, R2, mu = r["gamma"], r["Delta"], r["R2"], r["mu"]
            att = analysis.attribution(r["instr"], g, Dl, R2, mu, N, t_min=2 * N, t_max=20 * N, v_min=1e-18)
            attA = analysis.attribution_A(r["instr"], g, Dl, R2, mu, N, t_min=2 * N, t_max=20 * N, v_min=1e-18)
            sp.append(att["speedup_total"]); shB.append(att["shares"]); shA.append(attA["shares"])
            nrec.append(len(att["t"]))
            recs = [q for q in r["instr"] if 2 * N <= q["t"] <= 20 * N and q["V"] >= 1e-18]
            f1 = np.array([(1 - q["L_eff"] / R2 * (q["r_sq"] / 4 + 3 / 8)) / (3 / 8) for q in recs])
            f2 = np.array([q["mu_eff"] / mu for q in recs])
            f1_rec.append((f1.min(), f1.max())); f2_rec.append((f2.min(), f2.max()))
            f1g.append(np.log(f1).mean()); f2g.append(np.log(f2).mean())
            it = np.asarray(r["iters"]); Dt = np.asarray(r["D"]); Vt = np.asarray(r["V"])
            i0, i1 = int(np.argmin(abs(it - 2 * N))), int(np.argmin(abs(it - 20 * N)))
            cb = att["c_bound"]
            if Dt[i1] > 1e-20:
                slopeD.append(np.log(Dt[i0] / Dt[i1]) / (it[i1] - it[i0]) / cb)
                slopeV.append(np.log(Vt[i0] / Vt[i1]) / (it[i1] - it[i0]) / cb)
        sp = np.array(sp); shB = np.array(shB); shA = np.array(shA)
        v = ref[key]
        # order of analysis.STEP_KEYS[1:] = I1, I3, I2, I4, I5 ; STEP_KEYS_A = I4, I2, I3, I1, I5
        mB = dict(zip(["I1", "I3", "I2", "I4", "I5"], shB.mean(0)))
        mA = dict(zip(["I4", "I2", "I3", "I1", "I5"], shA.mean(0)))
        dev = [abs(sp.mean() / v["speedup"] - 1)] + [abs(mB[k] - v["shares"][k]) for k in mB] + \
              [abs(mA[k] - v["shares_orderA"][k]) for k in mA]
        worst = max(worst, max(dev))
        i24 = shB[:, 2] + shB[:, 3]
        rows[key] = dict(sp=sp, mB=mB, mA=mA, i24=i24)
        print(f"{key:22s} speed-up mean {sp.mean():8.3f} (json {v['speedup']:8.3f}) seeds [{sp.min():.3f}, {sp.max():.3f}] | "
              f"I2+I4 B mean {i24.mean():.4f} seeds [{i24.min():.4f}, {i24.max():.4f}] | I1 {mB['I1']:.4f} I2 {mB['I2']:.4f} "
              f"I3 {mB['I3']:.4f} (seed max {shB[:,1].max():.4f}) I4 {mB['I4']:.4f} I5 {mB['I5']:.4f} | "
              f"A: I4+I2 {mA['I4']+mA['I2']:.4f} | records/seed {min(nrec)}-{max(nrec)} | max dev vs json {max(dev):.2e}")
        print(f"{'':22s} first factor per record [{min(x[0] for x in f1_rec):.3f}, {max(x[1] for x in f1_rec):.3f}] "
              f"geomean {np.exp(np.mean(f1g)):.4f}; mu_eff/mu per record [{min(x[0] for x in f2_rec):.3f}, "
              f"{max(x[1] for x in f2_rec):.3f}] geomean {np.exp(np.mean(f2g)):.3f}; product of geomeans / speed-up "
              f"{np.exp(np.mean(f1g)) * np.exp(np.mean(f2g)) / sp.mean():.4f}"
              + (f"; realized slope of -log D over passes 2-20 / certified: [{min(slopeD):.2f}, {max(slopeD):.2f}], of -log V: "
                 f"[{min(slopeV):.2f}, {max(slopeV):.2f}]" if slopeD else "; D below 1e-20 before pass 20"))
print(f"\nworst deviation of the recomputed seed means from results/B3_summary.json (speed-up relative, shares absolute): {worst:.2e}")
lo = [rows[f'kappa={k:g},alpha={al:g}']['sp'] for k in (10., 100.) for al in ALPHAS]
hi = [rows[f'kappa=100000,alpha={al:g}']['sp'] for al in ALPHAS]
allr = [r_['sp'] for r_ in rows.values()]
print("per-seed speed-up, kappa<=1e2: [%.3f, %.3f]; kappa=1e5: [%.3f, %.3f]; all: [%.3f, %.3f]" % (
    min(x.min() for x in lo), max(x.max() for x in lo), min(x.min() for x in hi), max(x.max() for x in hi),
    min(x.min() for x in allr), max(x.max() for x in allr)))
big = [rows[f'kappa={k:g},alpha={al:g}']['i24'] for k in (10000., 100000.) for al in ALPHAS]
print("per-seed I2+I4 (order B), kappa>=1e4: [%.4f, %.4f]" % (min(x.min() for x in big), max(x.max() for x in big)))
