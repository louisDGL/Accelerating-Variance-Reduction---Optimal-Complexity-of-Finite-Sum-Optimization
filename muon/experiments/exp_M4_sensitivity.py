"""M4: sensitivity of the trajectories to a rounding-size perturbation.

Two copies of the same configuration are run with the SAME sampled indices from W_0 and from W_0 + delta P (P a fixed
random direction of unit Frobenius norm in the gauge subspace, delta = 1e-12 ||W_0 - W_*||_F). The script records the
relative separation ||W_a - W_b||_F / ||W_a - W_*||_F along the run. For a contraction in the Euclidean metric (SAGA)
the separation relative to the distance to the optimum stays at the level of delta; a map that is discontinuous or has
a large gain on the small singular directions of its argument (the exact polar factor: gain 1 / s_min, see maps.py)
can amplify it, in which case individual trajectories are not reproducible to rounding and only statistics over seeds
are meaningful.

Reported per configuration: the median over seeds of the relative separation at fixed numbers of passes, of its maximum
over the run, and the first pass at which it exceeds 1e-3.
Outputs: muon/results/M4_summary.json and a table on stdout.
Usage: python muon/experiments/exp_M4_sensitivity.py [--quick]
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

import common as C
from common import over

SEEDS = (0, 1, 2)
DELTA = 1e-12


def configs():
    return [
        ("SAGA (Euclidean), c = 1/4", dict(estimator="saga", placement="euclid", gamma=over("R2", 0.25))),
        ("joint, exact polar, c = 1/64", dict(estimator="saga", gamma=over("R2_op", 1.0 / 64))),
        ("joint, exact polar, c = 1/16", dict(estimator="saga", gamma=over("R2_op", 1.0 / 16))),
        ("joint, exact polar, c = 1/4", dict(estimator="saga", gamma=over("R2_op", 0.25))),
        ("joint, ns5, c = 1/4", dict(estimator="saga", gamma=over("R2_op", 0.25), orth="ns5")),
        ("joint, bounded gain (eps_innov = 1), c = 1/4", dict(estimator="saga", gamma=over("R2_op", 0.25),
                                                              orth="soft", eps_innov=1.0)),
        ("joint, damped (theta = 0.1), c = 1/4", dict(estimator="saga", gamma=over("R2_op", 0.25), theta=0.1)),
        ("split, c = 1/4", dict(estimator="saga", placement="split", gamma=over("R2_op", 0.25),
                                gamma_c=over("R2", 0.25))),
    ]


def paired(task):
    """Run the two copies; returns passes, relative separation and relative squared distance of copy a."""
    from lab import maps, vrmuon as vm
    s, cfg, seed, passes, per_pass = task["spec"], task["cfg"], task["seed"], task["passes"], task["per_pass"]
    prob = C.problem(s)
    n = prob.n
    cfg = C.resolve(cfg, prob)
    rng_p = np.random.default_rng(12345)
    P = rng_p.standard_normal((prob.K, prob.d))
    Q = maps.gauge_basis(prob)
    if Q is not None:
        P = Q @ (Q.T @ P)
    P /= np.linalg.norm(P)
    D0 = float(np.sqrt(np.vdot(prob.W_star, prob.W_star)))
    Ws = [np.zeros((prob.K, prob.d)), DELTA * D0 * P]
    algos = [vm.VRMuon(prob, **cfg) for _ in range(2)]
    sts = [a.init_state(W) for a, W in zip(algos, Ws)]
    rng = np.random.default_rng(seed)
    T = passes * n
    every = max(1, n // per_pass)
    out_p, out_sep, out_rel = [], [], []
    t = 0
    for B, coin in vm._draws(rng, n, 1, T, algos[0].needs_coin):
        for k in range(2):
            m, c, aux = algos[k].pieces(Ws[k], sts[k], B, coin)
            S, info = algos[k].step(Ws[k], m, c, sts[k], t)
            algos[k].commit(Ws[k], sts[k], B, coin, aux, info)
            Ws[k] = Ws[k] - S
        t += 1
        if t % every == 0:
            Ea = Ws[0] - prob.W_star
            da = float(np.sqrt(np.vdot(Ea, Ea)))
            sep = float(np.linalg.norm(Ws[0] - Ws[1]))
            if not np.isfinite(da) or da > 1e15 * D0:
                break
            out_p.append(t / n)
            out_sep.append(sep / da if da > 0 else float("nan"))
            out_rel.append((da / D0) ** 2)
            if da < 1e-13 * D0:                                     # rounding floor of the iterate itself
                break
    return {"passes": out_p, "sep": out_sep, "rel": out_rel}


def main():
    quick = "--quick" in sys.argv
    if quick:
        insts = {"soft": (C.spec("softmax", 150, 12, 5, 1e2, 1.0), 10)}
        seeds = SEEDS[:1]
    else:
        insts = {"soft-mem": (C.spec("softmax", 1000, 50, 10, 1e3, 1.0), 15),
                 "ls-mem": (C.spec("ls", 1000, 50, 10, 1e3, 1.0), 15)}
        seeds = SEEDS
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    tasks, index = [], []
    for key, (s, passes) in insts.items():
        for label, cfg in configs():
            for seed in seeds:
                tasks.append({"spec": s, "cfg": cfg, "seed": seed, "passes": passes, "per_pass": 10})
                index.append((key, label, seed))
    t0 = time.perf_counter()
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=int(os.environ.get("MUON_JOBS", "4"))) as ex:
        res = list(ex.map(paired, tasks))
    marks = (1, 2, 5, 10)
    summary = {}
    for (key, label, seed), r in zip(index, res):
        e = summary.setdefault(key, {}).setdefault(label, {"at": {str(m): [] for m in marks}, "max": [],
                                                           "first_1e-3": [], "trace": None})
        p, sep = np.asarray(r["passes"]), np.asarray(r["sep"])
        for m in marks:
            k = np.nonzero(p >= m - 1e-9)[0]
            e["at"][str(m)].append(float(sep[k[0]]) if len(k) else float("nan"))
        e["max"].append(float(np.nanmax(sep)) if len(sep) else float("nan"))
        k = np.nonzero(sep > 1e-3)[0]
        e["first_1e-3"].append(float(p[k[0]]) if len(k) else float("inf"))
        if seed == seeds[0]:
            e["trace"] = r
    for key in summary:
        print(f"\n=== {key}: {C.spec_tag(insts[key][0])}; relative separation ||W_a - W_b||_F / ||W_a - W_*||_F of two "
              f"runs started {DELTA:g} apart (relative), median over {len(seeds)} seeds")
        print(f"  {'configuration':40s} " + " ".join(f"{'pass ' + str(m):>10s}" for m in marks)
              + f" {'max':>10s} {'first > 1e-3':>13s}")
        for label, e in summary[key].items():
            print(f"  {label:40s} " + " ".join(f"{np.nanmedian(e['at'][str(m)]):10.1e}" for m in marks)
                  + f" {np.nanmedian(e['max']):10.1e} {np.median(e['first_1e-3']):13.1f}")
    C.save_json("M4_quick_summary" if quick else "M4_summary",
                {"delta": DELTA, "seeds": list(seeds), "instances": {k: {"spec": v[0], "passes": v[1]}
                                                                     for k, v in insts.items()}, "results": summary})
    print(f"\nM4 done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
