"""M3: which ingredient removes which error floor.

Two obstructions prevent a Muon-type method from converging to the minimizer of a finite sum with constant parameters:
  (a) the step length: a step of fixed spectral length eta cannot stay within eta / 2 of W_* (two consecutive iterates
      are eta apart in spectral norm), whatever the direction and whatever the estimator;
  (b) the estimator: the noise of a minibatch gradient, or of a momentum-based (STORM / MARS) buffer run with constant
      parameters, does not vanish at W_*.
The SAGA estimator removes (b); a length rule that vanishes with the estimator (nuclear-norm scaling, or its clipped
version) removes (a); a bounded-gain orthogonalization (offset eps) with a fixed step also removes (a) because the map
is linear near zero. This script runs the length-rule x estimator design and the published methods at fixed, untuned
parameters, in two ways:
  'from 0'      W_0 = 0, the usual run (traces for the figure; the level reached within the budget);
  'stationary'  W_0 = W_* + 1e-7 ||W_*||_F P (a point at relative squared distance 1e-14 of the minimizer, memory of
                the estimator initialized there): a method without floor stays at that level or below; a method with
                a floor leaves the minimizer and settles at its floor. The reported stationary level is the median
                of ||W - W_*||_F^2 / ||W_*||_F^2 over the second half of the run.

Outputs: muon/results/M3.pkl (cache), muon/results/M3_summary.json, a table on stdout.
Usage: python muon/experiments/exp_M3_floors.py [--quick] [--jobs N]
"""
from __future__ import annotations

import sys
import time

import numpy as np

import common as C
from common import over

SEEDS = (0, 1, 2)


def configs(batch, c_sharp=0.25):
    R = "R2_op"
    vr = lambda **kw: ("vr", dict(estimator="saga", **kw))                                           # noqa: E731
    out = [
        # SAGA estimator x length rule
        ("SAGA est., fixed length 1e-1", *vr(length="const", lr=1e-1), "est=saga,len=fixed"),
        ("SAGA est., fixed length 1e-2", *vr(length="const", lr=1e-2), "est=saga,len=fixed"),
        ("SAGA est., fixed length 1e-3", *vr(length="const", lr=1e-3), "est=saga,len=fixed"),
        ("SAGA est., nuclear-scaled", *vr(gamma=over(R, c_sharp)), "est=saga,len=sharp"),
        ("SAGA est., clipped (rho = 1e-1)", *vr(length="clip", rho=1e-1, gamma=over(R, c_sharp)),
         "est=saga,len=clip"),
        ("SAGA est., clipped (rho = 1e-2)", *vr(length="clip", rho=1e-2, gamma=over(R, c_sharp)),
         "est=saga,len=clip"),
        ("SAGA est., bounded gain (eps = 0.4), fixed length 1e-1",
         *vr(length="const", lr=1e-1, orth="soft", eps=0.4), "est=saga,len=fixed,soft"),
        ("SAGA est., bounded gain (eps = 0.04), fixed length 1e-2",
         *vr(length="const", lr=1e-2, orth="soft", eps=0.04), "est=saga,len=fixed,soft"),
        # minibatch gradient (no variance reduction) x length rule
        (f"minibatch b={batch}, nuclear-scaled", "vr",
         dict(estimator="sgd", batch=batch, gamma=over("L_op", 0.5)), "est=sgd,len=sharp"),
        (f"minibatch b={batch}, fixed length 1e-2", "vr",
         dict(estimator="sgd", batch=batch, length="const", lr=1e-2), "est=sgd,len=fixed"),
        # published methods
        (f"Muon b={batch}, lr 1e-1", "muon_sgd", dict(batch=batch, lr=1e-1, beta=0.95, nesterov=True, orth="ns5"),
         "muon"),
        (f"Muon b={batch}, lr 1e-2", "muon_sgd", dict(batch=batch, lr=1e-2, beta=0.95, nesterov=True, orth="ns5"),
         "muon"),
        (f"Muon b={batch}, lr 1e-1, cosine", "muon_sgd",
         dict(batch=batch, lr=1e-1, beta=0.95, nesterov=True, orth="ns5", schedule="cosine"), "muon"),
        (f"MARS-M b={batch}, lr 1e-2", "mvr",
         dict(batch=batch, lr=1e-2, beta=0.95, g=0.025, clip=1.0, same_batch=True, orth="ns5"), "mvr"),
        (f"STORM-Muon b={batch}, lr 1e-2", "mvr",
         dict(batch=batch, lr=1e-2, beta=0.95, g=1.0, same_batch=True, orth="ns5"), "mvr"),
        (f"STORM-Muon b={batch}, lr 1e-1, cosine", "mvr",
         dict(batch=batch, lr=1e-1, beta=0.95, g=1.0, same_batch=True, orth="ns5", schedule="cosine"), "mvr"),
        # momentum-based buffer with the nuclear-scaled step: no length floor, but a noise floor
        (f"STORM buffer b={batch}, nuclear-scaled", "mvr",
         dict(batch=batch, beta=0.95, g=1.0, same_batch=True, orth="svd", length="sharp", gamma=over("L_op", 0.3)),
         "mvr,len=sharp"),
        ("STORM buffer b=1, nuclear-scaled", "mvr",
         dict(batch=1, beta=0.95, g=1.0, same_batch=True, orth="svd", length="sharp", gamma=over(R, c_sharp)),
         "mvr,len=sharp"),
        # Euclidean reference
        ("SAGA (Euclidean)", "vr", dict(estimator="saga", placement="euclid", gamma=over("R2", 0.25)), "euclid"),
    ]
    return out


def main():
    quick = "--quick" in sys.argv
    jobs = None
    for i, a in enumerate(sys.argv):
        if a == "--jobs":
            jobs = int(sys.argv[i + 1])
    if quick:
        insts = {"soft": (C.spec("softmax", 150, 12, 5, 1e2, 1.0), 30, 8)}
        seeds = SEEDS[:2]
    else:
        insts = {"soft-mem": (C.spec("softmax", 1000, 50, 10, 1e3, 1.0), 60, 32),
                 "ls-mem": (C.spec("ls", 1000, 50, 10, 1e3, 1.0), 60, 32)}
        seeds = SEEDS
    name = "M3_quick" if quick else "M3"
    tasks, index = [], []
    for key, (s, passes, batch) in insts.items():
        for label, method, cfg, kind in configs(batch):
            for seed in seeds:
                for start in ("zero", "near_opt"):
                    t = {"spec": s, "method": method, "cfg": cfg, "seed": seed, "stop_rel": 1e-30,
                         "passes": passes if start == "zero" else max(4, passes // 2)}
                    if start == "near_opt":
                        t["start"] = "near_opt"
                    tasks.append(t)
                    index.append((key, label, kind, seed, start))
    t0 = time.perf_counter()
    res = C.run_tasks(tasks, name, jobs)
    summary = {}
    for (key, label, kind, seed, start), r in zip(index, res):
        e = summary.setdefault(key, {}).setdefault(label, {"kind": kind, "reached": [], "stationary": [],
                                                           "diverged": 0, "trace": None, "trace_stat": None})
        rel = np.asarray(r["trace"]["rel"]) if "trace" in r else np.array([np.inf])
        e["diverged"] += int(r["diverged"])
        if start == "zero":
            e["reached"].append(float(np.median(rel[-max(1, len(rel) // 6):])))
            if seed == seeds[0] and "trace" in r:
                e["trace"] = {"passes": r["trace"]["passes"], "rel": r["trace"]["rel"]}
        else:
            e["stationary"].append(float(np.median(rel[len(rel) // 2:])))
            if seed == seeds[0] and "trace" in r:
                e["trace_stat"] = {"passes": r["trace"]["passes"], "rel": r["trace"]["rel"]}
    for key in summary:
        (s, passes, batch) = insts[key]
        print(f"\n=== {key}: {C.spec_tag(s)}; {len(seeds)} seeds, median [min, max] of the relative squared distance")
        print(f"  {'configuration':58s} {'reached from W_0 = 0 (' + str(passes) + ' passes)':>34s}   "
              f"{'stationary level (start at 1e-14)':>36s}")
        for label, e in summary[key].items():
            f, g = np.array(e["reached"]), np.array(e["stationary"])
            print(f"  {label:58s} {np.median(f):12.1e} [{f.min():.1e}, {f.max():.1e}]   "
                  f"{np.median(g):12.1e} [{g.min():.1e}, {g.max():.1e}]" + ("   DIVERGED" if e["diverged"] else ""))
    C.save_json(f"{name}_summary", {"instances": {k: {"spec": v[0], "passes": v[1], "batch": v[2]}
                                                  for k, v in insts.items()}, "seeds": list(seeds),
                                    "results": summary})
    print(f"\nM3 done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
