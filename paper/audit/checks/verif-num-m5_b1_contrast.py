"""Counter-check of finding M5-b32-path-length (paper lines 538-540), part 3: at b = 1, where the path length is
4.7 to 11.5 ||W_*||_op (2.4 to 5.7 for MARS-M / STORM-Muon), is "without a table the cooldown freezes the iterate"
established?

  part S <inst>: the schedule of the paper (C/<inst>/controls_schedule), seeds 10 and 11: no table / table / MARS-M /
                 STORM-Muon (lab), the SAGA table at b = 2 (the step schedule and the cost per step of MARS-M at
                 b = 1), and an independent loop: exact gradient of F on the step schedules of b = 1 and b = 2 (does
                 the path suffice in practice?), own SAGA table, own minibatch (same index stream as the lab:
                 Generator.choice(n, 1, replace=False, shuffle=False) and Generator.integers(0, n) coincide, so the
                 own loop checks the algorithm, not the randomness).
  part T <inst>: slower cooldowns and other lr_0 without a table (seed 10): is the freeze an artefact of a schedule
                 selected for the method with a table?
Run: PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 <this file> S|T soft-cond|ls-cond
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("vb32", os.path.join(HERE, "verif-num-m5_b32_table.py"))
vb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vb)                     # own helpers of part 2 (build, own_run, lab_vr, lab_mvr, recipe)

LOSS = {"soft-cond": "softmax", "ls-cond": "ls"}


def part_S(S, key):
    prob = vb.build(LOSS[key])
    sc = S[key]["controls_schedule"]
    q, lr0 = sc["lr_decay"], sc["lr"]
    wop = float(np.linalg.norm(prob.W_star, 2))
    js = S[key]["controls"]
    print(f"== {key}, b = 1, schedule of the controls q = {q}, lr_0 = {lr0}, 300 passes; ||W_*||_op = {wop:.3f}")
    print("   final relative squared distance (lowest along the run), seeds 10 and 11, lab:")

    def show(label, runs, ref=None):
        txt = ", ".join(f"{r[0]:.3e} ({r[1]:.1e})" for r in runs)
        print(f"   {label:58s} {txt}" + (f"   json values {ref[0]:.4g}, {ref[1]:.4g}" if ref else ""), flush=True)

    show("NO table, minibatch b = 1, ns5 + Nesterov", [vb.lab_vr(prob, vb.recipe("sgd", 1, q, lr0), s) for s in (10, 11)],
         js["no table (minibatch gradient), ns5 + Nesterov, b=1"]["values"])
    show("SAGA TABLE b = 1, ns5 + Nesterov", [vb.lab_vr(prob, vb.recipe("saga", 1, q, lr0), s) for s in (10, 11)])
    show("MARS-M b = 1 (2 gradients per step)", [vb.lab_mvr(prob, 1, q, lr0, s, g=0.025, clip=1.0) for s in (10, 11)],
         js["MARS-M, b=1"]["values"])
    show("STORM-Muon b = 1 (2 gradients per step)", [vb.lab_mvr(prob, 1, q, lr0, s, g=1.0) for s in (10, 11)],
         js["STORM-Muon, b=1"]["values"])
    show("SAGA TABLE b = 2 (step schedule of MARS-M b = 1), ns5 + Nest.",
         [vb.lab_vr(prob, vb.recipe("saga", 2, q, lr0), s) for s in (10, 11)])
    print("   independent loop:")
    for b in (1, 2):
        r = vb.own_run(prob, "exact", b, q, lr0, 0)
        print(f"   EXACT gradient of F on the step schedule of b = {b}: rel {r['rel']:.3e} ({r['steps']} steps, path "
              f"{r['path_op']:.1f} = {r['path_op'] / wop:.2f} ||W_*||_op)", flush=True)
    a = vb.own_run(prob, "saga", 1, q, lr0, 10)
    c = vb.own_run(prob, "sgd", 1, q, lr0, 10)
    print(f"   own SAGA table b = 1 (seed 10): rel {a['rel']:.3e} | own minibatch b = 1: rel {c['rel']:.4g}"
          f" (path {c['path_op']:.1f} = {c['path_op'] / wop:.2f} ||W_*||_op, ||W_T||_op = {c['W_op']:.2f})")


def part_T(S, key):
    prob = vb.build(LOSS[key])
    print(f"== {key}, b = 1, 300 passes, seed 10, WITHOUT a table: final relative squared distance")
    print(f"   {'q':>5s} {'lr_0':>5s} {'lr_T':>9s} | {'minibatch ns5+Nesterov':>22s} | {'MARS-M':>10s} | {'STORM-Muon':>10s}")
    best = {}
    for q, lr0 in ((0.9, 0.3), (0.95, 0.3), (0.98, 0.3), (0.98, 1.0)):
        a = vb.lab_vr(prob, vb.recipe("sgd", 1, q, lr0), 10)[0]
        m = vb.lab_mvr(prob, 1, q, lr0, 10, g=0.025, clip=1.0)[0]
        st = vb.lab_mvr(prob, 1, q, lr0, 10, g=1.0)[0] if (q, lr0) == (0.98, 0.3) else float("nan")
        for k, v in (("minibatch", a), ("MARS-M", m), ("STORM-Muon", st)):
            if np.isfinite(v):
                best[k] = min(best.get(k, np.inf), v)
        print(f"   {q:5.2f} {lr0:5.1f} {lr0 * q ** 300:9.1e} | {a:22.4g} | {m:10.4g} | {st:10.4g}", flush=True)
    print("   lowest over these points: " + ", ".join(f"{k} {v:.4g}" for k, v in best.items()))


if __name__ == "__main__":
    t0 = time.process_time()
    S = json.load(open(os.path.join(vb.ROOT, "muon", "results", "M9_summary.json")))["C"]
    (part_S if sys.argv[1] == "S" else part_T)(S, sys.argv[2])
    print(f"CPU {time.process_time() - t0:.1f} s")
