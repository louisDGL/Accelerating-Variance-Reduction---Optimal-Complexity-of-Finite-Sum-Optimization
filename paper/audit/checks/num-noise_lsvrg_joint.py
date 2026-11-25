"""Audit num-noise, check 5: does the nuclear-scaled polar step reach rounding precision with ANY finite-sum estimator?

exp_M1 (muon/results/M1_summary.json, digest lines 55, 111, 153, 211, 257, 304, 336) reports that the nuclear-scaled
polar step of the loopless-SVRG estimator ("L-SVRG-Muon (joint)", tuned gamma) misses the target on the six instances.
This script (i) prints those stored rows next to the SAGA-table rows and (ii) re-runs the two memory-limited instances
of (M1) with a longer budget (150 passes instead of 40) at the step selected by M1, to see whether the run is slow or
stalls. Lab code is used read only; nothing is written to the repository.

Run from the project root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 paper/audit/checks/num-noise_lsvrg_joint.py
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "muon", "experiments"))
import common as C  # noqa: E402

m1 = json.load(open(os.path.join(ROOT, "muon", "results", "M1_summary.json")))
print("stored (M1_summary.json): passes to the target, median over 10 confirmation seeds; seeds reaching it; median final "
      "relative squared distance")
best = {}
for key, block in m1.items():
    tol = f"{block['tol']:g}"
    for fam in ("SAGA-Muon (joint)", "L-SVRG-Muon (joint)", "L-SVRG", "MISO"):
        r = next((x for x in block["rows"] if x["family"] == fam), None)
        if r is None:
            continue
        s = r["summary"][tol]
        best[(key, fam)] = r["best"]
        print(f"  {key:9s} (target {tol}, budget {block['passes']:3d}) {fam:20s} median {s['median']!s:>8} reached "
              f"{s['reached']}/{s['runs']}  final {r['final_rel']['median']:.1e}  step {r['best']}")

for key in ("soft-mem", "ls-mem"):
    s = m1[key]["spec"]
    cfg = dict(estimator="lsvrg", **best[(key, "L-SVRG-Muon (joint)")])
    t0 = time.perf_counter()
    r = C.run_one({"spec": s, "method": "vr", "cfg": cfg, "passes": 150, "seed": 10, "stop_rel": 1e-30})
    p, rel = np.array(r["trace"]["passes"]), np.array(r["trace"]["rel"])
    print(f"re-run {key}, L-SVRG estimator, nuclear-scaled length, {cfg}, 150 passes, seed 10 "
          f"({time.perf_counter() - t0:.0f} s): "
          + "  ".join(f"{q}: {rel[np.argmin(np.abs(p - q))]:.1e}" for q in (10, 20, 40, 80, 120, 150))
          + f"; min {rel.min():.1e}; diverged={r['diverged']}", flush=True)
