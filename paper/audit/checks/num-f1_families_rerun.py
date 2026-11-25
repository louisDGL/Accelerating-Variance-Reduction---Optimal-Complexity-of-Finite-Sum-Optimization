"""Audit num-f1, (C2): spot reproduction of the extreme values of the sentence on the designs of Na, Derezinski and
Mahoney (results/B3_families.pkl is absent). Designs P (kap = 0.5, low coherence, data seed 0) and R (kap = 1, low
coherence, data seed 1) at kappa = 1e5 and P (kap = 1, high coherence) at kappa = 1e5; SAGA seeds 0-4, 20100 iterations,
attribution over passes 2-20 as in experiments/exp_B3_families.py. (Design L, with rescaled labels, is not rerun.)
Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 paper/audit/checks/num-f1_families_rerun.py
"""
import json
import os
import sys

sys.dont_write_bytecode = True
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import numpy as np  # noqa: E402
from vrlab import algorithms, analysis, problems, solvers  # noqa: E402

N, D = 1000, 50
ref = json.load(open(os.path.join(ROOT, "results", "B3_families_summary.json")))["attribution"]
for tag, kap, hc, dseed in (("P,lc,kap=0.5,dseed=0", 0.5, False, 0), ("R,lc,kap=1,dseed=1", 1.0, False, 1),
                            ("P,hc,kap=1,dseed=0", 1.0, True, 0)):
    prob = problems.make_logistic_ndm(N, D, 1e5, kap=kap, high_coherence=hc, seed=dseed)
    solvers.solve_optimum(prob)
    sp, sh = [], []
    for s in range(5):
        r = algorithms.saga(prob, np.zeros(D), n_iters=20100, seed=s, record_every=N // 20, instrument_every=N // 10,
                            stop_D=1e-26)
        att = analysis.attribution(r["instr"], r["gamma"], r["Delta"], r["R2"], r["mu"], N, t_min=2 * N, t_max=20 * N,
                                   v_min=1e-18)
        sp.append(att["speedup_total"]); sh.append(att["shares"])
    sh = np.mean(sh, 0)                       # I1, I3, I2, I4, I5
    v = ref[tag + ",kappa=100000"]
    print(f"{tag}, kappa=1e5: speed-up {np.mean(sp):.3f} (json {v['speedup']:.3f}); I2+I4 {sh[2]+sh[3]:.4f} "
          f"(json {v['shares']['I2']+v['shares']['I4']:.4f})")
