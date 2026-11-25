"""Audit num-m5, check 5 (complement of check 4): seed-to-seed range of the ratio quoted at lines 579-580,
E_j ||omega_j - grad F||_* / ||grad F||_*, at the states t = p n, p = 2, 6, ..., 26 (the states of the archived test
_archive/muon/tests/test_facts.py::test_F13_limits_on_the_benchmark, which used ONE run per instance, seed 10 of the
lab's generator), with the independent loop of num-m5_independent_recipe.py and six index sequences.

    PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        python3 paper/audit/checks/num-m5_estimator_error_ratio.py
"""
import importlib.util
import io
import os
import contextlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("rec", os.path.join(HERE, "num-m5_independent_recipe.py"))
src = open(os.path.join(HERE, "num-m5_independent_recipe.py")).read().split("\nfor key, s in SPECS.items():")[0]
ns = {"__file__": os.path.join(HERE, "num-m5_independent_recipe.py")}
exec(compile(src, "num-m5_independent_recipe.py", "exec"), ns)       # definitions only (run, SPECS, C), no runs
run, SPECS, C = ns["run"], ns["SPECS"], ns["C"]

for key, s in SPECS.items():
    prob = C.problem(s)
    snaps = [p * prob.n for p in (2, 6, 10, 14, 18, 22, 26)]
    for lr0 in ((0.1,) if key == "soft-cond" else (0.1, 0.3)):
        allv = []
        for seed in range(10, 16):
            rec, noise = run(prob, o="polar", q=0.8, lr0=lr0, passes=28, seed=seed, snapshots=snaps)
            allv.append(noise)
            print(f"{key}, polar, q = 0.8, lr_0 = {lr0}, seed {seed}: " + ", ".join(f"{v:.2f}" for v in noise)
                  + f"   (rel dist2 at the end: {rec[-1, 1]:.1e})", flush=True)
        allv = np.array(allv)
        print(f"   -> range over 6 seeds x 7 states: [{allv.min():.2f}, {allv.max():.2f}];"
              f" per-seed ranges: " + "; ".join(f"[{v.min():.2f}, {v.max():.2f}]" for v in allv))
