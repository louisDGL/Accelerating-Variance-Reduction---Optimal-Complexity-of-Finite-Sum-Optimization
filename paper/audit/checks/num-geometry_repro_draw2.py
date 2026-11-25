"""Audit num-geometry, check 3: is the lower end of '0.20--0.37 of the passes of SAGA' and of '0.78--1.08' ((M3),
l. 379-380) an effect of the factor-2 step grid on the third draw of the softmax instance (data seed 2)?

M8 selects for SAGA gamma = 2/R2 on data seed 2 (136.1 passes) against 4/R2 on seeds 0 and 1 (69.5, 67.9 passes), and
for B-SAGA / IS-B-SAGA gamma = 8/R2 against 16/R2. This script runs the Euclidean families of the repository (same
code, same instance softmax n=200 d=40 K=8 kappa=1e4 alpha=2 seed=2, same target 1e-6, confirmation seeds) at steps
BETWEEN the points of the factor-2 grid. Small reproduction only: 2 or 3 confirmation seeds per point.

Run from the project root, one part at a time (each well under 3 minutes of CPU):
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 paper/audit/checks/num-geometry_repro_draw2.py saga
    ... bsaga     ... isbsaga
Read-only on the repository (common.run_one writes nothing).
"""
import os
import statistics as st
import sys

sys.dont_write_bytecode = True      # importing the repository must not refresh its __pycache__
ROOT = os.getcwd()
sys.path.insert(0, os.path.join(ROOT, "muon", "experiments"))
import common as C  # noqa: E402

part = sys.argv[1] if len(sys.argv) > 1 else "saga"
data_seed = int(sys.argv[2]) if len(sys.argv) > 2 else 2
s = C.spec("softmax", 200, 40, 8, 1e4, 2.0, seed=data_seed)
TOL = 1e-6
Q = 2 ** 0.25


def run(cfg, passes, seeds):
    out = []
    for sd in seeds:
        r = C.run_one({"spec": s, "method": "vr", "cfg": cfg, "passes": passes, "seed": sd, "stop_rel": 0.5 * TOL})
        out.append(r["passes_to"]["1e-06"])
    return out


if part == "saga":
    print(f"SAGA, softmax n=200 data seed {data_seed}, passes to 1e-6 (budget 160), confirmation seeds 10, 11")
    for c in (2.0, 2.0 * Q, 2.0 * Q ** 2, 2.0 * Q ** 3, 4.0):
        p = run(dict(estimator="saga", placement="euclid", gamma=C.over("R2", c)), 160, (10, 11))
        print(f"   gamma = {c:6.3f}/R2: {p}  median {st.median(p):.1f}", flush=True)
elif part == "bsaga":
    print(f"B-SAGA, softmax n=200 data seed {data_seed}, passes to 1e-6 (budget 80), confirmation seeds 10, 11, 12")
    for th in (0.15, 0.2):
        for c in (8.0, 8.0 * Q, 8.0 * Q ** 2, 8.0 * Q ** 3, 16.0):
            p = run(dict(estimator="saga", placement="euclid", theta=th, gamma=C.over("R2", c)), 80, (10, 11, 12))
            print(f"   theta = {th}, gamma = {c:6.3f}/R2: {p}  median {st.median(p):.1f}", flush=True)
elif part == "isbsaga":
    print(f"IS-B-SAGA (p_i ~ L_i^(1/2)), softmax n=200 data seed {data_seed}, passes to 1e-6 (budget 80), seeds 10-12")
    for th in (0.2, 0.3):
        for c in (8.0, 8.0 * Q, 8.0 * Q ** 2, 8.0 * Q ** 3, 16.0):
            p = run(dict(estimator="saga", placement="euclid", sampling=0.5, theta=th, gamma=C.over("R2", c)), 80,
                    (10, 11, 12))
            print(f"   theta = {th}, gamma = {c:6.3f}/R2: {p}  median {st.median(p):.1f}", flush=True)
