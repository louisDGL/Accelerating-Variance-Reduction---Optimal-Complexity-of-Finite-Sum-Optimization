"""M1: head-to-head comparison of variance-reduced Muon configurations at tuned steps.

Protocol of common.py (tune on the tuning seeds, report on fresh confirmation seeds) for every family of families.py on
six instances chosen to separate the regimes:

  key        instance                                              regime
  soft-mem   softmax, n = 1000, d = 50, K = 10, alpha = 1, kappa = 1e3   memory-limited (R2 / mu_true < n): every
  ls-mem     least squares, same sizes                                   single-index method is capped by the refresh
                                                                         of its memory, about (1 - 1/n) per iteration
  soft-cond  softmax, n = 200, d = 40, K = 8, alpha = 2, kappa = 1e4     conditioning-limited (R2 / mu_true >> n)
  ls-cond    least squares, same sizes
  ls-orth    least squares, n = 500, d = 20, K = 20, alpha = 2, kappa = 1e5, planted matrix with flat singular values:
             the polar factor of the full gradient points at the minimizer whatever the feature covariance (the
             situation that favours Muon-type steps), noise 0.1 in the targets
  soft-sq    softmax, n = 500, d = 20, K = 20 (square), alpha = 2, kappa = 1e4      conditioning-limited, square shape

Outputs: muon/results/M1_<key>_{tune,confirm}.pkl (caches), muon/results/M1_summary.json, a text table on stdout.
Usage: python muon/experiments/exp_M1_headtohead.py [--quick] [--only key1,key2] [--jobs N]
"""
from __future__ import annotations

import sys
import time

import numpy as np

import common as C
from families import catalogue, minibatch_families, streaming_families

INSTANCES = {
    "soft-mem": dict(spec=C.spec("softmax", 1000, 50, 10, 1e3, 1.0), passes=40, tol=1e-9, batch=10, stream=32),
    "ls-mem": dict(spec=C.spec("ls", 1000, 50, 10, 1e3, 1.0), passes=40, tol=1e-9, batch=10, stream=32),
    "soft-cond": dict(spec=C.spec("softmax", 200, 40, 8, 1e4, 2.0), passes=300, tol=1e-6, batch=8, stream=16),
    "ls-cond": dict(spec=C.spec("ls", 200, 40, 8, 1e4, 2.0), passes=300, tol=1e-6, batch=8, stream=16),
    "ls-orth": dict(spec=C.spec("ls", 500, 20, 20, 1e5, 2.0, noise=0.1, planted="orthogonal"), passes=100, tol=1e-6,
                    batch=20, stream=32),
    "soft-sq": dict(spec=C.spec("softmax", 500, 20, 20, 1e4, 2.0), passes=120, tol=1e-6, batch=20, stream=32),
}
QUICK = {
    "soft-mem": dict(spec=C.spec("softmax", 150, 12, 5, 1e2, 1.0), passes=25, tol=1e-9, batch=5, stream=8),
    "ls-orth": dict(spec=C.spec("ls", 150, 8, 8, 1e4, 2.0, noise=0.1, planted="orthogonal"), passes=40, tol=1e-6,
                    batch=8, stream=8),
}


def instance_info(s):
    """Constants of the instance, including the true conditioning at the optimum."""
    from lab import maps
    prob = C.problem(s)
    H = prob.hessian(prob.W_star)
    Q = maps.gauge_basis(prob)
    if Q is not None:                                           # softmax: restrict to the gauge subspace
        P = np.kron(Q, np.eye(prob.d))
        H = P.T @ H @ P
    ev = np.linalg.eigvalsh(H)
    sv = np.linalg.svd(prob.W_star, compute_uv=False)
    G0 = prob.grad(np.zeros((prob.K, prob.d)))
    sg = np.linalg.svd(G0 if Q is None else Q.T @ G0, compute_uv=False)
    info = C.constants(prob)
    info.update(mu_true=float(ev[0]), L_true=float(ev[-1]), cond_true=float(ev[-1] / ev[0]),
                R2_over_mu_true=float(prob.R2 / ev[0]), n=prob.n, d=prob.d, K=prob.K,
                srank_Wstar=float((sv ** 2).sum() / sv[0] ** 2), nrank_G0=float(sg.sum() ** 2 / (sg ** 2).sum()),
                Lop_over_LF=float(prob.L_op / prob.L_F))
    # one full-gradient step from W_0 = 0: relative decrease of the squared distance, and alignment of the polar
    # factor of the gradient with the error W_0 - W_*
    from lab import exact
    W0 = np.zeros((prob.K, prob.d))
    ref = exact.reference_steps(prob, W0, basis=Q)
    O0 = maps.polar(G0, basis=Q)
    E0 = W0 - prob.W_star
    info["step0"] = {k: float(v["rel_dec_dist2"]) for k, v in ref.items()}
    info["cos_polarG0_E0"] = float(np.vdot(O0, E0) / np.sqrt(np.vdot(O0, O0) * np.vdot(E0, E0)))
    info["cos_G0_E0"] = float(np.vdot(G0, E0) / np.sqrt(np.vdot(G0, G0) * np.vdot(E0, E0)))
    return info


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
    tag = "M1_quick" if quick else "M1"
    summary = C.load_json(f"{tag}_summary") or {}
    t0 = time.perf_counter()
    for key, inst in insts.items():
        if only and key not in only:
            continue
        s = inst["spec"]
        fams = (catalogue(s["n"], quick) + minibatch_families(inst["batch"], quick)
                + streaming_families(inst["stream"], quick))
        info = instance_info(s)
        print(f"\n=== {key}: {C.spec_tag(s)}  R2/mu_true = {info['R2_over_mu_true']:.0f} (n = {info['n']}), "
              f"cond(H*) = {info['cond_true']:.0f}, L_op/L_F = {info['Lop_over_LF']:.2f}, "
              f"nrank(G0) = {info['nrank_G0']:.2f}, budget {inst['passes']} passes, target {inst['tol']:g}", flush=True)
        recs = C.tune_and_confirm([s], fams, inst["passes"], inst["tol"], f"{tag}_{key}", n_jobs=jobs)
        meta = {f["name"]: {k: f[k] for k in ("group", "status", "what")} for f in fams}
        rows = []
        for r in recs:
            best_trace = None
            if r["confirm"]:
                order = np.argsort([c["final_rel"] for c in r["confirm"]])
                best_trace = r["confirm"][int(order[len(order) // 2])]["trace"]        # median run (final error)
            tk = f"{inst['tol']:g}"
            rows.append({"family": r["family"], **meta[r["family"]], "best": r["best"], "edge": r["edge"],
                         "best_score": r["best_score"], "summary": r["summary"], "final_rel": r["final_rel"],
                         "stop_level": r["stop_level"], "refined": r["refined"], "finalists": r["finalists"],
                         "n_points": r["n_points"], "n_kept": r["n_kept"],
                         "confirm_passes": [c["passes_to"][tk] for c in r["confirm"]],
                         "confirm_final": [c["final_rel"] for c in r["confirm"]],
                         "diverged": r["diverged"], "trace": best_trace,
                         "tune": [{"point": t["point"], "score": t["score"], "final_rel": t["final_rel"],
                                   "score_refined": t.get("score_refined"),
                                   "screen_rel": t["screen_rel"], "kept": t["kept"]} for t in r["tune"]],
                         "wall": float(np.mean([c["wall"] for c in r["confirm"]])) if r["confirm"] else None})
        summary[key] = {"spec": s, "info": info, "passes": inst["passes"], "tol": inst["tol"], "rows": rows,
                        "code": C.code_fingerprint(),
                        "protocol": {"tune_seeds": list(C.TUNE_SEEDS), "refine_seeds": list(C.REFINE_SEEDS),
                                     "refine_top": C.REFINE_TOP, "confirm_seeds": list(C.CONFIRM_SEEDS),
                                     "screen_fraction": C.SCREEN_FRACTION, "screen_keep": C.SCREEN_KEEP,
                                     "stops": [C.SCREEN_STOP, C.TUNE_STOP, C.CONFIRM_STOP]}}
        C.save_json(f"{tag}_summary", summary)
        tol = inst["tol"]
        print(f"{'family':32s} {'status':6s} {'selected step':44s} {'passes to 1e-3':>18s} "
              f"{'to ' + format(tol, 'g'):>20s} {'hit':>6s} {'deep':>6s} {'final rel. dist2':>17s}  edge")
        for row in sorted(rows, key=lambda r: (r["summary"][f"{tol:g}"]["median"], r["final_rel"]["median"])):
            sm, sl = row["summary"][f"{tol:g}"], row["stop_level"]
            print(f"{row['family']:32s} {row['status']:6s} {C.fmt_point(row['best']):44s} "
                  f"{C.fmt_passes(row['summary'], 1e-3):>18s} {C.fmt_passes(row['summary'], tol):>20s} "
                  f"{str(sm['reached']) + '/' + str(sm['runs']):>6s} {str(sl['reached']) + '/' + str(sl['runs']):>6s} "
                  f"{row['final_rel']['median']:17.1e}  {','.join(row['edge'])}", flush=True)
    print(f"\nM1 done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
