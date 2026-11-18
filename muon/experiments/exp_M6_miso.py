"""M6: the joint nuclear-scaled SAGA-Muon step as an implicit MISO iteration.

Fact F4 (notes_theory.md): the predictable part m = zbar + lam W of the SAGA estimator is the gradient of an isotropic
surrogate minimized at W_B = -zbar / lam, and m_{t+1} = m_t - lam S_t + (delta / n) c_t for any step S_t (delta = 1:
full replacement of the sampled table entry). The joint step W - gamma ||omega||_* polar(omega) moves W along the polar
factor of (the part of) m (outside the rank-one innovation) with a length set by the innovation: if this drives m to
zero, the iterate is slaved to W_B and the method IS the MISO / Finito iteration (the Euclidean member
theta = 1/n, gamma = 1/lam of the family), whatever gamma. MISO converges at rate about 1/n when n is of the order of
kappa or larger and is unstable when kappa >> n; relaxing the table by the dual-free-SDCA fraction
delta = lam n / (R2 + lam n) makes it stable for every kappa / n.

Test (fixed, untuned parameters; three kappa / n ratios at n = 1000; softmax and least squares): if the reading is
right, (i) joint SAGA-Muon at the transferred textbook step c = 1/4 needs as many passes as MISO where MISO converges
and fails where MISO fails, for c = 1/4 and c = 1 alike; (ii) with the relaxed table it follows dual-free SDCA
instead, on all instances; (iii) along the joint runs the predictable part is small compared with the gradient
(nuclear norms; the ratio is about 1 for Euclidean SAGA).

Outputs: muon/results/M6.pkl (cache), muon/results/M6_summary.json, a table on stdout.
Usage: python muon/experiments/exp_M6_miso.py [--quick] [--jobs N]
"""
from __future__ import annotations

import sys
import time

import numpy as np

import common as C
from common import over

SEEDS = (0, 1, 2)
TOL = 1e-6


def configs():
    return [
        ("SAGA (Euclidean), c = 1/4", dict(estimator="saga", placement="euclid", gamma=over("R2", 0.25))),
        ("MISO (delta = 1)", dict(estimator="saga", placement="euclid", miso_delta=1.0)),
        ("joint SAGA-Muon, c = 1/4", dict(estimator="saga", gamma=over("R2_op", 0.25))),
        ("joint SAGA-Muon, c = 1", dict(estimator="saga", gamma=over("R2_op", 1.0))),
        ("dual-free SDCA (delta = sdca)", dict(estimator="saga", placement="euclid", miso_delta="sdca")),
        ("joint SAGA-Muon, relaxed table, c = 1/4", dict(estimator="saga", gamma=over("R2_op", 0.25),
                                                         mem_relax="sdca")),
        ("joint SAGA-Muon, relaxed table, c = 1", dict(estimator="saga", gamma=over("R2_op", 1.0),
                                                       mem_relax="sdca")),
    ]


def main():
    quick = "--quick" in sys.argv
    jobs = None
    for i, a in enumerate(sys.argv):
        if a == "--jobs":
            jobs = int(sys.argv[i + 1])
    if quick:
        insts = {f"softmax k/n={k / 150:g}": C.spec("softmax", 150, 12, 5, k, 1.0) for k in (15.0, 150.0, 1500.0)}
        seeds, passes = SEEDS[:1], 30
    else:
        insts = {}
        for loss in ("softmax", "ls"):
            for kappa in (1e2, 1e3, 1e4):
                insts[f"{loss}, kappa/n = {kappa / 1000:g}"] = C.spec(loss, 1000, 50, 10, kappa, 1.0)
        seeds, passes = SEEDS, 40
    name = "M6_quick" if quick else "M6"
    tasks, index = [], []
    for key, s in insts.items():
        prob = C.problem(s)
        delta = prob.lam * prob.n / (prob.R2 + prob.lam * prob.n)
        for label, cfg in configs():
            cfg = dict(cfg)
            if cfg.get("miso_delta") == "sdca":
                cfg["miso_delta"] = float(delta)
            for seed in seeds:
                tasks.append({"spec": s, "method": "vr", "cfg": cfg, "passes": passes, "seed": seed,
                              "stop_rel": 1e-20, "instr": 1.0})
                index.append((key, label, seed, float(delta)))
    t0 = time.perf_counter()
    res = C.run_tasks(tasks, name, jobs)
    summary = {}
    for (key, label, seed, delta), r in zip(index, res):
        e = summary.setdefault(key, {"delta_sdca": delta, "rows": {}})["rows"].setdefault(
            label, {"passes": [], "final": [], "base": [], "diverged": 0, "trace": None})
        e["passes"].append(r["passes_to"][f"{TOL:g}"])
        e["final"].append(r["final_rel"])
        e["diverged"] += int(r["diverged"])
        ins = r.get("instr")
        if ins and "nuc_m" in ins:
            b = np.array(ins["nuc_m"][1:]) / np.maximum(np.array(ins["nuc_grad"][1:]), 1e-300)
            d2 = np.array(ins["dist2_F"][1:])
            ok = np.isfinite(b) & (d2 > 1e-16 * ins["dist2_F"][0])            # above the rounding floor
            e["base"].append(float(np.median(b[ok])) if ok.any() else float("nan"))
        if seed == seeds[0] and "trace" in r:
            e["trace"] = {"passes": r["trace"]["passes"], "rel": r["trace"]["rel"]}

    def cell(v):
        v = np.array(v, dtype=float)
        if not np.isfinite(v).any():
            return "not reached"
        f = lambda x: f"{x:.1f}" if np.isfinite(x) else "inf"                                        # noqa: E731
        return f"{f(np.median(v))} [{f(v.min())}, {f(v.max())}]"

    for key, block in summary.items():
        print(f"\n=== {key} (delta_sdca = {block['delta_sdca']:.3f}); passes to {TOL:g}: median [min, max] over "
              f"{len(seeds)} seeds; final relative squared distance (median); ||m||_* / ||grad F||_* (median over the "
              "run)")
        for label, e in block["rows"].items():
            base = np.nanmedian(e["base"]) if e["base"] else float("nan")
            print(f"  {label:42s} {cell(e['passes']):24s} {np.median(e['final']):9.1e}   {base:8.3f}"
                  + ("   DIVERGED" if e["diverged"] else ""))
    C.save_json(f"{name}_summary", {"tol": TOL, "passes": passes, "seeds": list(seeds),
                                    "instances": insts, "results": summary})
    print(f"\nM6 done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
