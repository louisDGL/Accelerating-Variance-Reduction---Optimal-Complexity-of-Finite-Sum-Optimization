"""Audit num-f1, (C4), complement: does SAGA converge at gamma R^2 = 4 (kappa = 1e3, alpha = 1, n = 1000, d = 50)?
gamma R^2 = 2 for 60 passes (3 seeds) and gamma R^2 = 4 for 2000 passes (seed 0) and 1000 passes (seeds 1, 2).
Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 paper/audit/checks/num-f1_gamma_delta_long.py
"""
import os
import sys

sys.dont_write_bytecode = True
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import numpy as np  # noqa: E402
import common  # noqa: E402
from vrlab import algorithms  # noqa: E402

prob = common.logistic_problem(1000, 50, 1e3, 1.0, seed=0)
for g, passes, seeds in ((2.0, 60, (0, 1, 2)), (4.0, 2000, (0,)), (4.0, 1000, (1, 2))):
    for seed in seeds:
        r = algorithms.saga(prob, np.zeros(50), gamma=g / prob.R2, passes=passes, seed=seed, record_every=1000,
                            record_F=False, stop_D=1e-26)
        it = np.asarray(r["iters"], float) / 1000; y = np.asarray(r["D"], float)
        pick = [p for p in (0, 2, 20, 40, 60, 100, 200, 400, 1000, 2000) if p <= it[-1]]
        vals = ", ".join(f"{p}: {y[int(np.argmin(abs(it - p)))]:.3e}" for p in pick)
        tail = y[it >= min(100, it[-1] / 2)]
        print(f"gamma R^2 = {g:g}, seed {seed}: stopped={r['stopped']}, last pass {it[-1]:g}; D at passes {vals}; "
              f"over the second part of the run: min {tail.min():.3e}, median {np.median(tail):.3e}, max {tail.max():.3e}")
