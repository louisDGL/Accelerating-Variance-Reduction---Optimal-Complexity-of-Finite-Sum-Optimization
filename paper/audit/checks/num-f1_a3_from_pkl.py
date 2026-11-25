"""Audit num-f1, (C3): 'Applied to loopless Katyusha, the same instrument finds a smaller gap (2.8 and 4.1 at kappa = 1e3,
1e4, against 5.2 and 15.7 for Proposition 5.9)'. Recomputed from results/A3.pkl (written with numpy >= 2: the module
names are aliased for numpy 1.22); also on the window of the SAGA measurement (passes 2-20).
Run: OPENBLAS_NUM_THREADS=1 python3 paper/audit/checks/num-f1_a3_from_pkl.py
"""
import json
import pickle
import sys

import numpy as np
import numpy.core as nc
import numpy.core.multiarray as ncm
import numpy.core.numeric as ncn

sys.modules.setdefault("numpy._core", nc)
sys.modules.setdefault("numpy._core.multiarray", ncm)
sys.modules.setdefault("numpy._core.numeric", ncn)

res = pickle.load(open("results/A3.pkl", "rb"))
ref = json.load(open("results/A3_summary.json"))
b3 = json.load(open("results/B3_summary.json"))["attribution"]
print("top-level keys:", list(res.keys()) if isinstance(res, dict) else type(res))
runs_by = res.get("runs", res)
for name, runs in runs_by.items():
    if not isinstance(runs, list):
        continue
    n = runs[0]["n"]
    pref, win = [], []
    for r in runs:
        q = []
        for x in r["instr"]:
            if not (x["Psi"] >= 1e-10):
                break
            q.append(x)
        pref += q
        win += [x for x in q if 2 * n <= x["t"] <= 20 * n] if "t" in q[0] else []
    lem = np.log(pref[0]["rho_lemma"])
    sp = np.mean([np.log(x["rho_exact"]) for x in pref]) / lem
    spw = np.mean([np.log(x["rho_exact"]) for x in win]) / lem if win else float("nan")
    print(f"{name}: {len(runs)} seeds, {len(pref)} records with Psi >= 1e-10 (last at pass "
          f"{max(x.get('t', 0) for x in pref)/n:.1f}); speed-up exact / Lemma 10 = {sp:.4f} (json "
          f"{ref[name]['speedup_exact_over_lemma']:.4f}); on passes 2-20 only: {spw:.4f}; against the rate of Theorem 11: "
          f"{ref[name]['speedup_exact_over_rho']:.3f}")
print("SAGA, Prop. 5.9, alpha = 1: kappa = 1e3:", round(b3["kappa=1000,alpha=1"]["speedup"], 4), " kappa = 1e4:",
      round(b3["kappa=10000,alpha=1"]["speedup"], 4))
