"""Audit num-m5, check 2: is "without a table the same schedule freezes the iterate at 8% to 93%" a property of the
estimator, or of a schedule that was selected for another method?

(A) Path-length control. The no-table controls of M9 part C run at the point selected for Muon's recipe WITH the table
    at batch one (soft-cond: q = 0.7, lr_0 = 0.3; ls-cond: q = 0.75, lr_0 = 0.3). At b = 32 the total length of that
    schedule, sum_t lr_t = lr_0 / (1 - q^(b/n)), is below ||W_*||_op: no estimator can reach W_* from W_0 = 0.
    We run the SAGA TABLE at b = 32 on that same schedule: if it freezes at the same level, the b = 32 controls say
    nothing about the table.
(B) Tuning control at b = 1 (and b = 32): the minibatch gradient through Muon's recipe (ns5, Nesterov 0.9) and MARS-M
    on a small grid of (q, lr_0), 300 passes, one seed: lowest level reachable without a table when the schedule
    is chosen for it.

Uses the lab code of the repository through muon/experiments/common.run_one (no cache is read or written).
Run from the project root (about 2 minutes):
    PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        python3 paper/audit/checks/num-m5_no_table_fairness.py [A|B1|B1M|B32] [soft-cond|ls-cond]
"""
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "muon", "experiments"))
import common as C                                                    # noqa: E402

SPECS = {"soft-cond": C.spec("softmax", 200, 40, 8, 1e4, 2.0), "ls-cond": C.spec("ls", 200, 40, 8, 1e4, 2.0)}
SELECTED = {"soft-cond": (0.7, 0.3), "ls-cond": (0.75, 0.3)}          # M9_summary.json, C/<inst>/controls_schedule
PASSES = 300
part = sys.argv[1] if len(sys.argv) > 1 else "A"
if len(sys.argv) > 2:                                                 # optional: one instance only
    SPECS = {sys.argv[2]: SPECS[sys.argv[2]]}


def run(spec, method, cfg, seed=10):
    t0 = time.perf_counter()
    r = C.run_one({"spec": spec, "method": method, "cfg": cfg, "passes": PASSES, "seed": seed, "stop_rel": 0.0})
    return r["final_rel"], r["min_rel"], time.perf_counter() - t0


def recipe(estimator, b, q, lr0):
    return dict(estimator=estimator, batch=b, length="const", orth="ns5", beta=0.9, nesterov=True,
                schedule="geometric", lr_decay=q, lr=lr0)


if part == "A":
    for key, s in SPECS.items():
        q, lr0 = SELECTED[key]
        prob = C.problem(s)
        import numpy as np
        wop = float(np.linalg.norm(prob.W_star, 2))
        n = s["n"]
        print(f"== {key}: schedule of the no-table controls q = {q}, lr_0 = {lr0}; ||W*||_op = {wop:.2f}")
        for b in (1, 32):
            path = lr0 / (1 - q ** (b / n))
            for est in ("sgd", "saga"):
                for seed in (10, 11):
                    f, m, w = run(s, "vr", recipe(est, b, q, lr0), seed)
                    print(f"   b = {b:2d}, sum lr_t = {path:7.2f} ({path / wop:5.2f} ||W*||_op), estimator {est:4s},"
                          f" seed {seed}: final rel dist2 {f:.3g} (lowest {m:.3g})  [{w:.1f}s]", flush=True)
elif part == "B1":
    for key, s in SPECS.items():
        print(f"== {key}, b = 1, no table (minibatch gradient, ns5 + Nesterov 0.9), 300 passes, seed 10:"
              f" final relative squared distance")
        qs, lrs = (SELECTED[key][0], 0.9, 0.95, 0.98), (0.03, 0.1, 0.3)
        print("      q \\ lr_0 " + " ".join(f"{v:>9g}" for v in lrs))
        for q in qs:
            row = [run(s, "vr", recipe("sgd", 1, q, lr0))[0] for lr0 in lrs]
            print(f"      {q:<9g} " + " ".join(f"{v:9.2e}" for v in row), flush=True)
elif part == "B1M":
    for key, s in SPECS.items():
        n = s["n"]
        print(f"== {key}, b = 1, no table, MARS-M (g = 0.025, clip 1, beta 0.95, ns5; 2 gradients per step),"
              f" 300 passes, seed 10: final relative squared distance")
        qs, lrs = (SELECTED[key][0], 0.9, 0.95, 0.98), (0.1, 0.3)
        print("      q \\ lr_0 " + " ".join(f"{v:>9g}" for v in lrs))
        for q in qs:
            row = [run(s, "mvr", dict(batch=1, lr=lr0, beta=0.95, same_batch=True, orth="ns5", g=0.025, clip=1.0,
                                      schedule=["geometric", q, n / 2.0]))[0] for lr0 in lrs]
            print(f"      {q:<9g} " + " ".join(f"{v:9.2e}" for v in row), flush=True)
elif part == "B32":
    for key, s in SPECS.items():
        n = s["n"]
        print(f"== {key}, b = 32, 300 passes, seed 10: final relative squared distance")
        qs, lrs = (0.75, 0.9, 0.95, 0.98), (0.3, 1.0, 3.0, 10.0)
        for name, method, mk in (
                ("minibatch gradient (ns5 + Nesterov 0.9)", "vr", lambda q, lr0: recipe("sgd", 32, q, lr0)),
                ("MARS-M (g = 0.025, clip 1, beta 0.95)", "mvr",
                 lambda q, lr0: dict(batch=32, lr=lr0, beta=0.95, same_batch=True, orth="ns5", g=0.025, clip=1.0,
                                     schedule=["geometric", q, n / 64.0])),
                ("SAGA table (ns5 + Nesterov 0.9)", "vr", lambda q, lr0: recipe("saga", 32, q, lr0))):
            print(f"   {name}")
            print("      q \\ lr_0 " + " ".join(f"{v:>9g}" for v in lrs))
            for q in qs:
                row = [run(s, method, mk(q, lr0))[0] for lr0 in lrs]
                print(f"      {q:<9g} " + " ".join(f"{v:9.2e}" for v in row), flush=True)
