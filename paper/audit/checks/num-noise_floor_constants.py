"""Audit num-noise, check 1: constants of (M1).

(a) ||W_*||_F^2 of the two instances of exp_M3 (soft-mem, ls-mem), to convert the RELATIVE squared distances stored in
    muon/results/M3_summary.json into the ABSOLUTE constants "3.8 eta^2" and "6.0 eta^2" quoted by the paper.
(b) Independent short re-run of the polar step of the SAGA estimator with a fixed length (lab code, read only):
    32 passes, eta = 1e-2, seed 0; the level over the last 10 passes is compared with the stored one.
(c) Ratio of the stationary levels of Muon (b = 32) at lr 1e-1 and 1e-2 ("linear in its learning rate").

Run from the project root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 paper/audit/checks/num-noise_floor_constants.py
Nothing is written to the repository.
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "muon", "experiments"))
import common as C  # noqa: E402

m3 = json.load(open(os.path.join(ROOT, "muon", "results", "M3_summary.json")))
for key in ("soft-mem", "ls-mem"):
    s = m3["instances"][key]["spec"]
    prob = C.problem(s)
    D0 = float(np.vdot(prob.W_star, prob.W_star))
    rank = min(prob.K, prob.d) - (1 if prob.loss == "softmax" else 0)
    print(f"\n=== {key}: ||W_*||_F^2 = {D0:.4f}; lam = {prob.lam:.3e}, R2 = {prob.R2:.4f}, R2_op = {prob.R2_op:.4f}, "
          f"L_op = {prob.L_op:.4f}; rank of a polar factor = {rank}")
    for lr in ("1e-1", "1e-2", "1e-3"):
        e = m3["results"][key][f"SAGA est., fixed length {lr}"]
        for which in ("reached", "stationary"):
            v = np.array(e[which], dtype=float)
            print(f"  SAGA est., fixed length {lr}: {which:10s} rel = {np.median(v):.3e}  ->  absolute / eta^2 = "
                  f"{np.median(v) * D0 / float(lr) ** 2:.3f}  (per seed: "
                  + ", ".join(f"{x * D0 / float(lr) ** 2:.3f}" for x in v) + ")")
    print(f"  Frobenius form of the length floor, rank * eta^2 / 4 = {rank / 4:.2f} eta^2")
    a = np.median(m3["results"][key]["Muon b=32, lr 1e-1"]["stationary"])
    b = np.median(m3["results"][key]["Muon b=32, lr 1e-2"]["stationary"])
    a0 = np.median(m3["results"][key]["Muon b=32, lr 1e-1"]["reached"])
    b0 = np.median(m3["results"][key]["Muon b=32, lr 1e-2"]["reached"])
    print(f"  Muon b=32: stationary level lr 1e-1 / lr 1e-2 = {a:.3e} / {b:.3e} = {a / b:.2f};  reached from W_0 = 0: "
          f"{a0:.3e} / {b0:.3e} = {a0 / b0:.2f}")
    # (b) independent short re-run
    t0 = time.perf_counter()
    r = C.run_one({"spec": s, "method": "vr", "cfg": dict(estimator="saga", length="const", lr=1e-2), "seed": 0,
                   "stop_rel": 1e-30, "passes": 32})
    p, rel = np.array(r["trace"]["passes"]), np.array(r["trace"]["rel"])
    tail = rel[p >= 22]
    print(f"  re-run (32 passes, eta = 1e-2, seed 0, {time.perf_counter() - t0:.0f} s): median rel over passes 22-32 = "
          f"{np.median(tail):.3e} [{tail.min():.2e}, {tail.max():.2e}]  ->  absolute / eta^2 = "
          f"{np.median(tail) * D0 / 1e-4:.3f}")
