"""M5: batch size of the variance-reduced estimator, Euclidean versus orthogonalized steps.

At batch size one the innovation of a variance-reduced estimator is a rank-one matrix that dominates the estimator, so
its polar factor is mostly determined by the sampled index. With a minibatch of b indices the innovation has rank
min(b, r) and its noise is divided by about sqrt(b): the estimator approaches the full gradient, whose polar factor is
the Muon direction. This script measures where the orthogonalized SAGA methods overtake Euclidean minibatch SAGA, in
gradient evaluations: for b in a geometric range, each of the three families
    SAGA b            Euclidean minibatch SAGA
    B-SAGA b          Euclidean minibatch SAGA with damped innovation (theta in {0.6, 0.3, 0.2, 0.1, 0.05}): the
                      Euclidean control for the decoupling of the step on the predictable part from the step on the
                      innovation
    SAGA-Muon b       joint placement, nuclear-scaled polar step of the minibatch SAGA estimator
is tuned and confirmed with the protocol of common.py (passes of per-sample gradients to the target; one iteration
costs b gradients and, for the orthogonalized methods, one factorization, so that the number of factorizations per pass
is n / b).

Outputs: muon/results/M5_<key>_{tune,confirm}.pkl (caches), muon/results/M5_summary.json, tables on stdout.
Usage: python muon/experiments/exp_M5_batch.py [--quick] [--only key1,key2] [--jobs N]
"""
from __future__ import annotations

import sys
import time

import common as C
from common import over

INSTANCES = {
    "ls-orth": dict(spec=C.spec("ls", 500, 20, 20, 1e5, 2.0, noise=0.1, planted="orthogonal"), passes=100, tol=1e-6),
    "soft-cond": dict(spec=C.spec("softmax", 200, 40, 8, 1e4, 2.0), passes=300, tol=1e-6),
    "ls-cond": dict(spec=C.spec("ls", 200, 40, 8, 1e4, 2.0), passes=300, tol=1e-6),
}
QUICK = {"ls-orth": dict(spec=C.spec("ls", 150, 8, 8, 1e4, 2.0, noise=0.1, planted="orthogonal"), passes=40, tol=1e-6)}
BATCHES = (1, 4, 16, 64)


def families(b, quick=False):
    st = 2 if quick else 1
    g = lambda unit, lo, hi, step=1: [over(unit, 2.0 ** k) for k in range(lo, hi + 1, step)]          # noqa: E731
    return [
        dict(name=f"SAGA b={b}", method="vr", base=dict(estimator="saga", placement="euclid", batch=b),
             grid=dict(gamma=g("R2", -3, 10, st))),
        dict(name=f"B-SAGA b={b}", method="vr", base=dict(estimator="saga", placement="euclid", batch=b),
             grid=dict(theta=[0.6, 0.3, 0.2, 0.1, 0.05] if not quick else [0.3, 0.1], gamma=g("R2", -1, 10, st))),
        dict(name=f"SAGA-Muon b={b}", method="vr", base=dict(estimator="saga", batch=b),
             grid=dict(gamma=g("R2_op", -9, 8, st))),
    ]


def main():
    quick = "--quick" in sys.argv
    only = None
    jobs = None
    for i, a in enumerate(sys.argv):
        if a == "--only":
            only = sys.argv[i + 1].split(",")
        if a == "--jobs":
            jobs = int(sys.argv[i + 1])
    insts = QUICK if quick else INSTANCES
    batches = (1, 4, 16) if quick else BATCHES
    tag = "M5_quick" if quick else "M5"
    summary = C.load_json(f"{tag}_summary") or {}
    t0 = time.perf_counter()
    for key, inst in insts.items():
        if only and key not in only:
            continue
        s = inst["spec"]
        fams = [f for b in batches if b <= s["n"] // 2 for f in families(b, quick)]
        print(f"\n=== {key}: {C.spec_tag(s)}, budget {inst['passes']} passes, target {inst['tol']:g}", flush=True)
        recs = C.tune_and_confirm([s], fams, inst["passes"], inst["tol"], f"{tag}_{key}", n_jobs=jobs)
        rows = [{"family": r["family"], "batch": int(r["base"]["batch"]),
                 "kind": r["family"].split(" b=")[0], "best": r["best"], "edge": r["edge"],
                 "summary": r["summary"], "final_rel": r["final_rel"], "stop_level": r["stop_level"],
                 "diverged": r["diverged"]} for r in recs]
        summary[key] = {"spec": s, "passes": inst["passes"], "tol": inst["tol"], "rows": rows,
                        "code": C.code_fingerprint()}
        C.save_json(f"{tag}_summary", summary)
        tol = inst["tol"]
        kinds = ["SAGA", "B-SAGA", "SAGA-Muon"]
        print(f"  passes to {tol:g}: median [min, max] over the confirmation seeds (selected step)")
        print(f"  {'b':>4s}  " + "  ".join(f"{k:^46s}" for k in kinds))
        for b in batches:
            cells = []
            for k in kinds:
                row = next((r for r in rows if r["batch"] == b and r["kind"] == k), None)
                if row is None:
                    cells.append(f"{'':46s}")
                    continue
                txt = f"{C.fmt_passes(row['summary'], tol)} ({C.fmt_point(row['best'])})" + ("*" if row["edge"] else "")
                cells.append(f"{txt:46s}")
            print(f"  {b:4d}  " + "  ".join(cells), flush=True)
        print("  (* = a selected value on the edge of its grid)")
    print(f"\nM5 done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
