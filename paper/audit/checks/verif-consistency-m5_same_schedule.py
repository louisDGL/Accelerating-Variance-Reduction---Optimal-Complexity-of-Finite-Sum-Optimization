"""Counter-check of finding C-02 (dimension consistency): what "the same schedule" of (M5) means in the data.

 part K : the cache M9_law.pkl of M9 part C holds exactly the runs listed in M9_summary.json (no run without a table
          at q = 0.8, lr_0 = 0.1).
 part P : ||W_*||_op of the two instances and the total step length sum_t lr_t of every schedule of part C
          (lr_0 / (1 - q^(b/n)) for the 'vr' runs, lr_0 / (1 - q^(2b/n)) for MARS-M / STORM-Muon, 2 b gradients a step).
 part S : the no-table controls of part C (minibatch gradient through ns5 + Nesterov 0.9, MARS-M, STORM-Muon) rerun
          at the schedule of the WITH-table runs quoted just before in (M5), q = 0.8, lr_0 = 0.1 (300 passes, no stop
          rule), b = 1 and b = 32: final relative squared distance.
 part T : the SAGA table through Muon's recipe at b = 32 on the schedule of the no-table controls (the only batch
          size at which the path length of that schedule is short).

Uses common.run_one of the repository; no cache is read (except in K) or written.
Run from the project root, one part and one instance per call (each well under 3 minutes of CPU):
    PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        python3 paper/audit/checks/verif-consistency-m5_same_schedule.py K|P|S|T [soft-cond|ls-cond] [seeds]
"""
import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "muon", "experiments"))
import common as C                                                    # noqa: E402

SPECS = {"soft-cond": C.spec("softmax", 200, 40, 8, 1e4, 2.0), "ls-cond": C.spec("ls", 200, 40, 8, 1e4, 2.0)}
PASSES = 300
part = sys.argv[1]
keys = [sys.argv[2]] if len(sys.argv) > 2 else list(SPECS)
seeds = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [10, 11]
m9 = json.load(open(os.path.join(ROOT, "muon", "results", "M9_summary.json")))


def tkey(task):
    return hashlib.sha1(json.dumps(task, sort_keys=True).encode()).hexdigest()


def recipe(est, b, q, lr0):
    return dict(estimator=est, batch=b, length="const", orth="ns5", beta=0.9, nesterov=True, schedule="geometric",
                lr_decay=q, lr=lr0)


def mvr(b, q, lr0, n, **kw):
    return dict(batch=b, lr=lr0, beta=0.95, same_batch=True, orth="ns5", schedule=["geometric", q, n / (2.0 * b)], **kw)


def run(s, method, cfg, seed):
    r = C.run_one({"spec": s, "method": method, "cfg": cfg, "passes": PASSES, "seed": seed, "stop_rel": 0.0})
    return r["final_rel"], r["min_rel"]


if part == "K":
    cache = pickle.load(open(os.path.join(ROOT, "muon", "results", "M9_law.pkl"), "rb"))
    listed = set()
    for key, e in m9["C"].items():
        for kind in ("deep", "controls"):
            for d in e[kind].values():
                for seed in e["seeds"]:
                    listed.add(tkey({"spec": e["spec"], "method": d.get("method", "vr"), "cfg": d["cfg"],
                                     "passes": e["passes"], "seed": seed, "stop_rel": 0.0}))
    print(f"M9_law.pkl: {len(cache)} runs; listed in the summary: {len(listed)}; listed and cached: "
          f"{len(listed & set(cache))}; cached and not listed: {len(set(cache) - listed)}")
    for key in SPECS:
        e = m9["C"][key]
        for seed in e["seeds"]:
            t = {"spec": e["spec"], "method": "vr", "cfg": recipe("sgd", 1, 0.8, 0.1), "passes": e["passes"],
                 "seed": seed, "stop_rel": 0.0}
            print(f"   {key}, no table at q = 0.8, lr_0 = 0.1, b = 1, seed {seed}: in cache = {tkey(t) in cache}")
elif part == "P":
    for key in keys:
        s = SPECS[key]
        n = s["n"]
        wop = float(np.linalg.norm(C.problem(s).W_star, 2))
        qc, lc = m9["C"][key]["controls_schedule"]["lr_decay"], m9["C"][key]["controls_schedule"]["lr"]
        print(f"== {key}: ||W_*||_op = {wop:.3f}; schedule of the no-table controls q = {qc}, lr_0 = {lc}")
        for (q, lr0) in ((qc, lc), (0.8, 0.1), (0.8, 0.3)):
            for b in (1, 32):
                vr, mv = lr0 / (1 - q ** (b / n)), lr0 / (1 - q ** (2 * b / n))
                print(f"   q = {q}, lr_0 = {lr0}, b = {b:2d}: sum lr_t = {vr:7.2f} = {vr / wop:6.2f} ||W_*||_op (vr);"
                      f" {mv:7.2f} = {mv / wop:6.2f} ||W_*||_op (MARS-M, STORM-Muon)")
elif part == "S":
    for key in keys:
        s = SPECS[key]
        n = s["n"]
        print(f"== {key}: no table, schedule q = 0.8, lr_0 = 0.1, {PASSES} passes, seeds {seeds}: final rel. dist2")
        for b in (1, 32):
            t0 = time.perf_counter()
            rows = {"minibatch gradient, ns5 + Nesterov": [run(s, "vr", recipe("sgd", b, 0.8, 0.1), sd)[0] for sd in seeds],
                    "MARS-M": [run(s, "mvr", mvr(b, 0.8, 0.1, n, g=0.025, clip=1.0), sd)[0] for sd in seeds],
                    "STORM-Muon": [run(s, "mvr", mvr(b, 0.8, 0.1, n, g=1.0), sd)[0] for sd in seeds]}
            for name, v in rows.items():
                print(f"   b = {b:2d} {name:36s} " + ", ".join(f"{x:.3g}" for x in v), flush=True)
            print(f"   [{time.perf_counter() - t0:.0f}s]")
elif part == "T":
    for key in keys:
        s = SPECS[key]
        q, lr0 = m9["C"][key]["controls_schedule"]["lr_decay"], m9["C"][key]["controls_schedule"]["lr"]
        print(f"== {key}: b = 32, schedule of the no-table controls q = {q}, lr_0 = {lr0}, seeds {seeds}")
        for est in ("sgd", "saga"):
            v = [run(s, "vr", recipe(est, 32, q, lr0), sd) for sd in seeds]
            print(f"   estimator {est:4s}: final " + ", ".join(f"{f:.3g}" for f, _ in v), flush=True)
