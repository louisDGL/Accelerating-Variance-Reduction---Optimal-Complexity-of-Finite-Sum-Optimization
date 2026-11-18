"""Audit of the result caches: recompute a random sample of cached runs with the current code.

The caches muon/results/*.pkl are keyed by the task (instance, method, configuration, budget, seed) and not by the
code. This script rebuilds the tasks of the tune-then-confirm experiments (M1, M5, M8) from their definitions, draws
at random, in every cache file, a few tasks that are present in the cache, runs them again with the current code
(single-threaded BLAS, as the experiments) and compares the passes to every tolerance, the final relative squared
distance and the divergence flag with the cached values. Exact equality is expected: the runs are deterministic given
the seed, including the rounding-sensitive exact-polar configurations.

Outputs: muon/results/audit_summary.json, a report on stdout.
Usage: python muon/experiments/audit_cache.py [--per-file 6] [--seed 0] [--jobs 4]
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

import common as C


def stage_tasks(name, s, fams, passes, tol):
    """(cache name, task) for every task that tune_and_confirm may have run for these families on the instance s."""
    short = max(1, int(math.ceil(C.SCREEN_FRACTION * passes)))
    for fam in fams:
        det = fam["method"] in C.DETERMINISTIC
        pts = C.grid_points(fam.get("grid", {}))
        if C._screenable(fam, len(pts)):
            for t in C.make_tasks(s, fam, short, (C.TUNE_SEEDS[0],), stop_rel=C.SCREEN_STOP * tol)[1]:
                yield name + "_screen", t
        seeds = (0,) if det else tuple(C.TUNE_SEEDS) + tuple(C.REFINE_SEEDS)
        for t in C.make_tasks(s, fam, passes, seeds, stop_rel=C.TUNE_STOP * tol)[1]:
            yield name + "_tune", t
        seeds = (0,) if det else tuple(C.CONFIRM_SEEDS)
        for t in C.make_tasks(s, fam, passes, seeds, stop_rel=C.CONFIRM_STOP * tol)[1]:
            yield name + "_confirm", t


def all_tasks():
    from families import catalogue, minibatch_families, streaming_families
    import exp_M1_headtohead as M1
    import exp_M5_batch as M5
    import exp_M8_dataseeds as M8
    for key, inst in M1.INSTANCES.items():
        s = inst["spec"]
        fams = catalogue(s["n"]) + minibatch_families(inst["batch"]) + streaming_families(inst["stream"])
        yield from stage_tasks(f"M1_{key}", s, fams, inst["passes"], inst["tol"])
    for key, inst in M5.INSTANCES.items():
        s = inst["spec"]
        fams = [f for b in M5.BATCHES if b <= s["n"] // 2 for f in M5.families(b)]
        yield from stage_tasks(f"M5_{key}", s, fams, inst["passes"], inst["tol"])
    for key, inst in M8.INSTANCES.items():
        for sd in M8.DATA_SEEDS:
            s = inst["make"](sd)
            fams = [f for f in catalogue(s["n"]) if f["name"] in M8.FAMILIES]
            yield from stage_tasks(f"M8_{key}_s{sd}", s, fams, inst["passes"], inst["tol"])


def same(a, b):
    if a.get("stopped") == "error" or b.get("stopped") == "error":
        return a.get("stopped") == b.get("stopped")
    if bool(a["diverged"]) != bool(b["diverged"]):
        return False
    if a["diverged"]:
        return True                                 # diverged runs: the flag is what the protocol uses
    return a["passes_to"] == b["passes_to"] and a["final_rel"] == b["final_rel"]


def main():
    per_file, seed, jobs = 6, 0, int(os.environ.get("MUON_JOBS", "4"))
    for i, a in enumerate(sys.argv):
        if a == "--per-file":
            per_file = int(sys.argv[i + 1])
        if a == "--seed":
            seed = int(sys.argv[i + 1])
        if a == "--jobs":
            jobs = int(sys.argv[i + 1])
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    rng = np.random.default_rng(seed)
    by_file = {}
    for name, task in all_tasks():
        by_file.setdefault(name, []).append(task)
    picked = []
    n_cached = 0
    for name in sorted(by_file):
        cache = C._load_cache(os.path.join(C.RESULTS, f"{name}.pkl"))
        if not cache:
            continue
        seen, present = set(), []
        for t in by_file[name]:
            k = C.task_key(t)
            if k in cache and k not in seen:
                seen.add(k)
                present.append((k, t))
        n_cached += len(present)
        idx = rng.choice(len(present), size=min(per_file, len(present)), replace=False) if present else []
        picked += [(name, present[i][1], cache[present[i][0]]) for i in idx]
    print(f"{len(by_file)} cache files defined, {n_cached} cached runs matched to a task; recomputing {len(picked)} "
          f"of them with {jobs} processes", flush=True)
    t0 = time.perf_counter()
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        fresh = list(ex.map(C.run_one, [t for _, t, _ in picked], chunksize=1))
    details, ok = [], 0
    methods = {}
    for (name, task, old), new in zip(picked, fresh):
        methods[task["method"]] = methods.get(task["method"], 0) + 1
        if same(old, new):
            ok += 1
        else:
            details.append({"file": name, "method": task["method"], "cfg": C.fmt_point(task["cfg"]),
                            "seed": task["seed"],
                            "what": f"final {old['final_rel']:.6e} -> {new['final_rel']:.6e}; passes "
                                    f"{old['passes_to']} -> {new['passes_to']}"})
    files = sorted({name for name, _, _ in picked})
    out = {"checked": len(picked), "identical": ok, "mismatch": len(details), "details": details, "files": files,
           "seed": seed, "per_file": per_file, "methods": methods, "matched_cached_runs": n_cached,
           "code": C.code_fingerprint()}
    C.save_json("audit_summary", out)
    print(f"recomputed {len(picked)} runs in {time.perf_counter() - t0:.0f}s: {ok} identical, {len(details)} differ "
          f"(by method: {methods})")
    for d in details:
        print(f"  DIFF {d['file']} {d['method']} seed {d['seed']} {d['cfg']}: {d['what']}")


if __name__ == "__main__":
    main()
