"""Audit num-f1, Figure 1(a): the three curves recomputed (results/B3.pkl is absent, so the figure script cannot be run
as is). kappa = 1e4, alpha = 1, n = 1000, d = 50, seeds 0-4, 100 passes, records every n/20, instrumentation every n/10;
curves computed with the functions used by paper/scripts/make_figR1_classical.py.
Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 paper/audit/checks/num-f1_fig1a.py
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
from vrlab import algorithms, analysis, bounds  # noqa: E402

N, D = 1000, 50
prob = common.logistic_problem(N, D, 1e4, 1.0, seed=0)
runs = [algorithms.saga(prob, np.zeros(D), passes=100, seed=s, record_every=N // 20, instrument_every=N // 10,
                        stop_D=1e-26) for s in range(5)]
it, meanD, allD = analysis.mean_over_seeds(runs, "D")
r0 = runs[0]
prop59 = bounds.saga_bound(it, N, r0["R2"], r0["mu"], r0["D0"])
tb = analysis.trajectory_bounds(r0, which=("mu_eff+L_eff", "mu_eff", "exact"), v_min=1e-18)
traj = tb["mu_eff+L_eff"]
p = it / N
print(f"D0 = {r0['D0']:.6g}, V0 = {r0['V'][0]:.6g}, (1 + 3n/16) D0 = {(1 + 3 * N / 16) * r0['D0']:.6g}; stopped: {[r['stopped'] for r in runs]}")
print("passes | Prop. 5.9 bound | bound with mu_eff, L_eff (seed 0) | mean of D over 5 seeds [min, max] | realized V seed 0")
for q in (0, 2, 20, 50, 100):
    j = int(np.argmin(abs(p - q)))
    print(f"{q:6d} | {prop59[j]:.4e} | {traj[j]:.4e} | {meanD[j]:.4e} [{allD[:, j].min():.4e}, {allD[:, j].max():.4e}] | {r0['V'][j]:.4e}")
ref = json.load(open(os.path.join(ROOT, "results", "B3_summary.json")))["attribution"]["kappa=10000,alpha=1"]
print(f"final mean D = {meanD[-1]:.6e} (json final_D {ref['final_D']:.6e})")
print(f"trajectory bound / mean D along the run: min {np.nanmin(traj / meanD):.3f}, max {np.nanmax(traj / meanD):.3f}")
print(f"trajectory bound / realized V of seed 0: min {np.nanmin(traj / r0['V']):.3f}, max {np.nanmax(traj / r0['V']):.3f}")
for a, b, lab in ((0, 100, "passes 0-100"), (2, 20, "passes 2-20"), (20, 100, "passes 20-100")):
    ja, jb = int(np.argmin(abs(p - a))), int(np.argmin(abs(p - b)))
    e = lambda y: np.log(y[ja] / y[jb]) / (b - a)
    eb, et, em = e(prop59), e(traj), e(meanD)
    print(f"{lab}: decrease per pass: bound {eb:.5f}, a-posteriori {et:.4f}, mean D {em:.4f}; "
          f"share of the gap of exponents recovered = {(et - eb) / (em - eb):.3f}; observed / certified = {em / eb:.2f}")
