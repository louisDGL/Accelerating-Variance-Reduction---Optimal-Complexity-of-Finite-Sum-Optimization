"""M2: mechanism of the orthogonalized variance-reduced steps, from exact conditional expectations.

For the single-index SAGA-memory configurations of M1, at the steps selected by M1, this script evaluates the exact
one-step statistics of lab/exact.py (enumeration of the n possible indices, no Monte-Carlo error)
  (a) at states of the configuration's OWN trajectory (what the method does where it actually goes), and
  (b) at states of the Euclidean SAGA trajectory (all configurations compared from the same iterate and memory).
The comparison of (a) and (b) separates what a step rule gains in one iteration from a given state from what it can
sustain: the memory error obeys E[M_{t+1} | past] >= (1 - 1/n) M_t for every step rule, so a large one-step decrease of
the distance obtained from a state with a small memory error cannot be repeated.

Reported per configuration and state: n * eff_dist2 and n * eff_F (expected relative decrease of ||W - W_*||_F^2 and of
F - F_* per pass), the realized contraction per pass of the run, the estimator diagnostics (noise_nuc, noise_F,
smin_ratio, nuclear ranks) and the alignment of the mean step (cos_grad, dual_align, cos_sharp, cos_dist).
Requires muon/results/M1_summary.json (selected steps). Outputs: muon/results/M2_summary.json, tables on stdout.
Usage: python muon/experiments/exp_M2_mechanism.py [--quick] [--only key1,key2]
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

import common as C

FAMILIES = ["SAGA", "B-SAGA", "SAGA-Muon (joint)", "damped SAGA-Muon", "mean-scaled SAGA-Muon", "soft SAGA-Muon",
            "split SAGA-Muon", "tangent SAGA-Muon", "radial SAGA-Muon", "polar-metric SAGA",
            "one-shot polar-metric SAGA", "cov-metric SAGA"]
MARKS = {"soft-mem": [1, 2, 5, 10, 20], "ls-mem": [1, 2, 5, 10, 20], "soft-cond": [2, 10, 30, 80, 160],
         "ls-cond": [2, 10, 30, 80, 160], "ls-orth": [1, 5, 20, 50, 100], "soft-sq": [1, 5, 20, 50, 100]}
KEYS = ("eff_dist2", "eff_F", "rel_dec_dist2", "rel_dec_F", "frac_increase_F", "cos_grad", "dual_align", "cos_sharp",
        "cos_dist", "noise_nuc", "noise_F", "bias_F", "stale_F", "base_ratio", "surrogate_gap", "cos_m_grad",
        "smin_ratio", "nrank_G", "nrank_m", "nrank_omega",
        "srank_E", "dist2_F", "Fgap", "S2_F", "var_F", "mean_norm_F", "lin")


def job(task):
    from lab import exact, maps, vrmuon as vm
    s, cfg, marks, ref_cfg = task["spec"], task["cfg"], task["marks"], task["ref_cfg"]
    prob = C.problem(s)
    n = prob.n
    W0 = np.zeros((prob.K, prob.d))
    D0 = float(np.vdot(prob.W_star, prob.W_star))
    cfg_r = C.resolve(cfg, prob)
    at = [int(m * n) for m in marks]
    out = {"own": [], "cross": [], "marks": marks}
    floor = 1e-18 * D0
    # (a) own trajectory
    states = exact.collect_states(prob, W0, max(at), at, seed=0, **cfg_r)
    for t, W, st in states:
        if not np.isfinite(W).all() or np.vdot(W - prob.W_star, W - prob.W_star) < floor:
            continue
        stats = exact.one_step_stats(prob, vm.VRMuon(prob, **cfg_r), W, st, t)
        rec = {k: stats[k] for k in KEYS if k in stats}
        rec["pass"] = t / n
        out["own"].append(rec)
    # realized contraction per pass between consecutive marks (own trajectory)
    d = [(t / n, float(np.vdot(W - prob.W_star, W - prob.W_star))) for t, W, _ in states]
    out["realized"] = [{"from": a[0], "to": b[0], "rate_per_pass": float(1.0 - (b[1] / a[1]) ** (1.0 / (b[0] - a[0])))
                        if a[1] > 0 and b[1] > 0 else float("nan")} for a, b in zip(d[:-1], d[1:])]
    # (b) states of the reference (Euclidean SAGA) trajectory. A metric placement evaluated there uses the metric
    # its own run would hold: the one of the full gradient at W_0 when it is never refreshed (one-shot), otherwise the
    # one of the current predictable part (computed on demand by the step).
    ref_r = C.resolve(ref_cfg, prob)
    one_shot = cfg_r.get("placement") == "metric" and cfg_r.get("metric_every", 0) >= 10 ** 8
    for t, W, st in exact.collect_states(prob, W0, max(at), at, seed=0, **ref_r):
        if not np.isfinite(W).all() or np.vdot(W - prob.W_star, W - prob.W_star) < floor:
            continue
        algo_x = vm.VRMuon(prob, **cfg_r)
        if one_shot:
            st.metric = {"svd": algo_x.anchor_svd(prob.grad(W0)), "gauged": algo_x.basis is not None}
        stats = exact.one_step_stats(prob, algo_x, W, st, t, with_state=False)
        rec = {k: stats[k] for k in KEYS if k in stats}
        rec["pass"] = t / n
        out["cross"].append(rec)
    if task.get("yardstick"):
        out["yardstick"] = []
        for t, W, st in exact.collect_states(prob, W0, max(at), at, seed=0, **ref_r):
            if np.vdot(W - prob.W_star, W - prob.W_star) < floor:
                continue
            ref = exact.reference_steps(prob, W, basis=maps.gauge_basis(prob))
            out["yardstick"].append({"pass": t / n, **{k: v["rel_dec_dist2"] for k, v in ref.items()},
                                     **{k + "_F": v["rel_dec_F"] for k, v in ref.items()}})
    return out


def main():
    quick = "--quick" in sys.argv
    only = None
    for i, a in enumerate(sys.argv):
        if a == "--only":
            only = sys.argv[i + 1].split(",")
    m1 = C.load_json("M1_quick_summary" if quick else "M1_summary")
    if not m1:
        raise SystemExit("run exp_M1_headtohead.py first (M1_summary.json not found)")
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    from families import catalogue
    tasks, index = [], []
    for key, block in m1.items():
        if only and key not in only:
            continue
        s = block["spec"]
        bases = {f["name"]: f["base"] for f in catalogue(s["n"], quick)}
        best = {r["family"]: r["best"] for r in block["rows"]}
        if "SAGA" not in best:
            continue
        ref_cfg = dict(bases["SAGA"], **best["SAGA"])
        marks = MARKS.get(key, [1, 2, 5, 10]) if not quick else [1, 3, 8]
        marks = [m for m in marks if m <= block["passes"]]
        for fam in FAMILIES:
            if fam not in best:
                continue
            cfg = dict(bases[fam], **best[fam])
            tasks.append({"spec": s, "cfg": cfg, "marks": marks, "ref_cfg": ref_cfg, "yardstick": fam == "SAGA"})
            index.append((key, fam, cfg))
    t0 = time.perf_counter()
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=int(os.environ.get("MUON_JOBS", "4"))) as ex:
        res = list(ex.map(job, tasks))
    summary = {}
    for (key, fam, cfg), r in zip(index, res):
        summary.setdefault(key, {})[fam] = {"cfg": cfg, **r}
    C.save_json("M2_quick_summary" if quick else "M2_summary", summary)

    def med(recs, k, scale=1.0):
        v = [r[k] * scale for r in recs if k in r and np.isfinite(r[k])]
        return float(np.median(v)) if v else float("nan")

    for key, block in summary.items():
        n = m1[key]["spec"]["n"]
        print(f"\n=== {key} (n = {n}): medians over the recorded states; 'n eff' = expected relative decrease per pass")
        print(f"  {'family':24s} {'own: n eff_d2':>13s} {'realized':>9s} {'SAGA states: n eff_d2':>22s} "
              f"{'cos_grad':>9s} {'dual_al':>8s} {'noise_nuc':>10s} {'smin_ratio':>11s} {'nr_omega':>9s} {'nr_G':>6s} "
              f"{'frac_up':>8s} {'|m|/|G|':>8s} {'surr_gap':>9s}")
        for fam, r in block.items():
            real = [x["rate_per_pass"] for x in r["realized"] if np.isfinite(x["rate_per_pass"])]
            print(f"  {fam:24s} {med(r['own'], 'eff_dist2', n):13.3f} {np.median(real) if real else float('nan'):9.3f} "
                  f"{med(r['cross'], 'eff_dist2', n):22.3f} {med(r['own'], 'cos_grad'):9.2f} "
                  f"{med(r['own'], 'dual_align'):8.2f} {med(r['own'], 'noise_nuc'):10.2f} "
                  f"{med(r['own'], 'smin_ratio'):11.1f} {med(r['own'], 'nrank_omega'):9.2f} "
                  f"{med(r['own'], 'nrank_G'):6.2f} {med(r['own'], 'frac_increase_F'):8.2f} "
                  f"{med(r['own'], 'base_ratio'):8.3f} {med(r['own'], 'surrogate_gap'):9.3f}")
        if "SAGA" in block and block["SAGA"].get("yardstick"):
            y = block["SAGA"]["yardstick"]
            print("  full-gradient steps at the SAGA states (relative decrease of dist2 per pass): "
                  + ", ".join(f"{k} {np.median([q[k] for q in y]):.3f}" for k in ("gd", "gd_ls", "sharp", "sharp_ls")
                              if k in y[0]))
    print(f"\nM2 done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
