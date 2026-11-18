"""M8: robustness of the main comparison to the data seed.

Every instance of M1 is one draw of the synthetic data (seed 0). Here the four conditioning-limited instances
(soft-cond, ls-cond: n = 200, d = 40, K = 8, alpha = 2, kappa = 1e4; ls-orth, soft-sq: n = 500, K = d = 20) are
regenerated with other data seeds, and the families that carry the conclusions of the study are tuned and confirmed
again on each draw with the protocol of common.py (same grids as M1: the families are taken from families.catalogue):

    SAGA, B-SAGA, IS-B-SAGA          Euclidean: plain, damped, damped with importance sampling
    cov-metric SAGA                  Euclidean, preconditioned by the feature covariance
    SAGA-Muon (joint), soft SAGA-Muon, split SAGA-Muon, IS soft SAGA-Muon      orthogonalized
    one-shot polar-metric SAGA, one-shot polar-metric B-SAGA                   frozen polar metric

Outputs: muon/results/M8_<key>_s<seed>_{screen,tune,confirm}.pkl (caches), muon/results/M8_summary.json, tables on
stdout. Usage: python muon/experiments/exp_M8_dataseeds.py [--quick] [--jobs N]
"""
from __future__ import annotations

import sys
import time

import common as C
from families import catalogue

FAMILIES = ["SAGA", "B-SAGA", "IS-B-SAGA", "cov-metric SAGA", "SAGA-Muon (joint)", "soft SAGA-Muon", "split SAGA-Muon",
            "IS soft SAGA-Muon", "one-shot polar-metric SAGA", "one-shot polar-metric B-SAGA"]
INSTANCES = {
    "soft-cond": dict(make=lambda sd: C.spec("softmax", 200, 40, 8, 1e4, 2.0, seed=sd), passes=300, tol=1e-6),
    "ls-cond": dict(make=lambda sd: C.spec("ls", 200, 40, 8, 1e4, 2.0, seed=sd), passes=300, tol=1e-6),
    "ls-orth": dict(make=lambda sd: C.spec("ls", 500, 20, 20, 1e5, 2.0, noise=0.1, planted="orthogonal", seed=sd),
                    passes=100, tol=1e-6),
    "soft-sq": dict(make=lambda sd: C.spec("softmax", 500, 20, 20, 1e4, 2.0, seed=sd), passes=120, tol=1e-6),
}
QUICK = {"ls-cond": dict(make=lambda sd: C.spec("ls", 80, 12, 4, 1e3, 2.0, seed=sd), passes=60, tol=1e-6)}
DATA_SEEDS = (1, 2)


def main():
    quick = "--quick" in sys.argv
    jobs = None
    for i, a in enumerate(sys.argv):
        if a == "--jobs":
            jobs = int(sys.argv[i + 1])
    from exp_M1_headtohead import instance_info
    insts = QUICK if quick else INSTANCES
    tag = "M8_quick" if quick else "M8"
    summary = C.load_json(f"{tag}_summary") or {}
    t0 = time.perf_counter()
    for key, inst in insts.items():
        for sd in (DATA_SEEDS[:1] if quick else DATA_SEEDS):
            s = inst["make"](sd)
            fams = [f for f in catalogue(s["n"], quick) if f["name"] in FAMILIES]
            assert len(fams) == len(FAMILIES), [f["name"] for f in fams]
            info = instance_info(s)
            print(f"\n=== {key}, data seed {sd}: {C.spec_tag(s)}  R2/mu_true = {info['R2_over_mu_true']:.0f} "
                  f"(n = {info['n']}), cond(H*) = {info['cond_true']:.0f}, budget {inst['passes']} passes, "
                  f"target {inst['tol']:g}", flush=True)
            recs = C.tune_and_confirm([s], fams, inst["passes"], inst["tol"], f"{tag}_{key}_s{sd}", n_jobs=jobs)
            tk = f"{inst['tol']:g}"
            rows = [{"family": r["family"], "best": r["best"], "edge": r["edge"], "summary": r["summary"],
                     "final_rel": r["final_rel"], "stop_level": r["stop_level"], "diverged": r["diverged"],
                     "confirm_passes": [c["passes_to"][tk] for c in r["confirm"]]} for r in recs]
            summary.setdefault(key, {})[str(sd)] = {"spec": s, "info": info, "passes": inst["passes"],
                                                     "tol": inst["tol"], "rows": rows, "code": C.code_fingerprint()}
            C.save_json(f"{tag}_summary", summary)
            for row in sorted(rows, key=lambda r: r["summary"][tk]["median"]):
                sm, sl = row["summary"][tk], row["stop_level"]
                print(f"  {row['family']:24s} {C.fmt_point(row['best']):44s} {C.fmt_passes(row['summary'], inst['tol']):>20s} "
                      f"hit {sm['reached']}/{sm['runs']} deep {sl['reached']}/{sl['runs']}  {','.join(row['edge'])}",
                      flush=True)
    print(f"\nM8 done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
