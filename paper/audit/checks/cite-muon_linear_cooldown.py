"""Audit cite-muon: what does the schedule of Muon's REFERENCE runs give with a SAGA table?

The paper calls "Muon's own recipe" fixed-length steps on a GEOMETRIC cooldown lr_t = lr_0 q^(t b / n) (Eq. (4)).
The reference runs of Muon (jordan2024muon: modded-nanogpt train_gpt2.py at the commit linked from the blog post,
cifar10-airbench research/clean_muon.py) use a LINEAR cooldown over a fixed horizon: constant then linear to zero
('wsd' below, last 600 of 1530 iterations = decay fraction 0.39), or linear to zero from the start ('linear').
Small reproduction with the repository's own code (read-only): instance ls-cond of experiment M9 (n = 200, 8 x 40),
batch one, exact polar factor, SAGA table, 300 passes, seeds 10-12; schedules geometric (the paper's selected point
q = 0.8, lr_0 = 0.3), linear and wsd (lr_0 on a small grid). Reported: final relative squared distance.
Run from the project root:
  PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python3 paper/audit/checks/cite-muon_linear_cooldown.py
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "muon", "experiments"))
sys.path.insert(0, os.path.join(ROOT, "muon"))
import numpy as np            # noqa: E402
import common as C            # noqa: E402

S = C.spec("ls", 200, 40, 8, 1e4, 2.0)          # 'ls-cond' in muon/experiments/exp_M9_schedule.py
BASE = dict(estimator="saga", batch=1, length="const", orth="svd")
PASSES = 300


def run(cfg, seed, passes=PASSES):
    r = C.run_one({"spec": S, "method": "vr", "cfg": {**BASE, **cfg}, "passes": passes, "seed": seed,
                   "stop_rel": 0.0})
    return r["final_rel"], r["min_rel"], r["wall"]


def main():
    seeds = (10, 11, 12)
    rows = [("geometric q=0.8 lr0=0.3 (paper, Fig. 4a)", dict(schedule="geometric", lr_decay=0.8, lr=0.3))]
    for lr0 in (0.03, 0.1, 0.3, 1.0):
        rows.append((f"linear to zero, lr0={lr0:g} (airbench)", dict(schedule="linear", lr=lr0)))
    for lr0 in (0.03, 0.1, 0.3):
        rows.append((f"constant then linear to zero over last 39%, lr0={lr0:g} (nanogpt)",
                     dict(schedule="wsd", decay_frac=0.39, lr=lr0)))
    print(f"instance ls-cond, batch 1, exact polar, SAGA table, {PASSES} passes; final relative squared distance")
    for label, cfg in rows:
        out = [run(cfg, s) for s in seeds]
        fin = [o[0] for o in out]
        print(f"  {label:66s} final_rel median {np.median(fin):.2e} [min {min(fin):.2e}, max {max(fin):.2e}]"
              f"  (wall {sum(o[2] for o in out):.0f}s)", flush=True)


if __name__ == "__main__":
    main()
