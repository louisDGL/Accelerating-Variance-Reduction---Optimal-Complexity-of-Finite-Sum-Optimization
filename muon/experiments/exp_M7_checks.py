"""M7: three checks behind statements of muon/findings.md.

(A) Conditioning of the problem in the frozen polar metric (facts F10, F11). For an anchor matrix D and a floor f, the
    metric C = V diag(nu / max(s_k, f s_1)) V^T + r (I - V V^T) turns the problem into Fhat(What) = F(What C^{1/2});
    the quantity that governs SAGA / L-SVRG in that metric is kappa_P = R2P / mu_P, with R2P the certified per-sample
    smoothness. Two values of mu_P are reported: the LOCAL one, the smallest eigenvalue of the Hessian of Fhat at the
    optimum (restricted to the gauge subspace for softmax), and the GLOBAL strong-convexity constant, the one a
    theorem needs (F11): the same number for least squares (constant Hessian), lam lambda_min(C) for softmax. The
    anchor grad F(W_0) is examined on the six instances of M1; on the two square instances (ls-orth, where the polar
    factor of the first gradient points at the optimum, and soft-sq) anchors are also taken along a Euclidean SAGA
    run (exact gradient and predictable part of SAGA after 0.5, 2 and 10 passes). Euclidean reference:
    kappa = R2 / mu_true (local) and R2 / lam (global, softmax). Control: the same constants for the metric of the
    feature covariance, C proportional to (c_l Sigma_hat + lam I)^-1 with the same floors on its gains
    (metric_source='cov').

(B) Weight of the spectral y step in Katyusha-Muon. At the (L, mu) selected by M1 for L-Katyusha, the loopless
    Katyusha scheme is run on the confirmation seeds with the published coupling step, with no y step at all
    (y = x), and with the nuclear-scaled polar y step for several gamma_y.

(C) Spread of the per-sample smoothness constants of the six instances (what importance sampling exploits).

Requires muon/results/M1_summary.json for (B). Outputs: muon/results/M7_summary.json, muon/results/M7_katyusha.pkl
(cache), tables on stdout. Usage: python muon/experiments/exp_M7_checks.py [--jobs N]
"""
from __future__ import annotations

import sys
import time

import numpy as np

import common as C
from common import over

FLOORS = (0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
ANCHOR_PASSES = (0.5, 2.0, 10.0)
LATER_ANCHORS = ("ls-orth", "soft-sq")
COV_LABEL = "feature covariance, (c_l Sigma_hat + lam I)^-1"
GAMMA_Y = (1.0 / 9.0, 1.0, 8.0, 32.0)


def metric_conditioning(s, saga_gamma, later_anchors=True):
    """Rows {anchor, floor, kappa_P (local), kappa_P_global, cond_P, cond_C, floored, rank} for the instance s (see
    the module docstring)."""
    from lab import exact, maps, vrmuon as vm
    prob = C.problem(s)
    K, d, n = prob.K, prob.d, prob.n
    Q = maps.gauge_basis(prob)
    H = prob.hessian(prob.W_star)
    if Q is not None:
        P = np.kron(Q, np.eye(d))
        H = P.T @ H @ P
    Kg = H.shape[0] // d
    ev = np.linalg.eigvalsh(H)
    W0 = np.zeros((K, d))
    anchors = [("grad F(W_0)", prob.grad(W0))]
    if later_anchors:
        gam = C.resolve({"gamma": saga_gamma}, prob)["gamma"] if saga_gamma is not None else None
        at = [int(round(p * n)) for p in ANCHOR_PASSES]
        states = exact.collect_states(prob, W0, max(at), at, seed=0, estimator="saga", placement="euclid", gamma=gam)
        for p, (t, W, st) in zip(ANCHOR_PASSES, states):
            anchors.append((f"grad F(W_t), SAGA after {p:g} passes", prob.grad(W)))
            anchors.append((f"SAGA predictable part m_t after {p:g} passes", st.zbar + prob.lam * W))
    anchors.append((COV_LABEL, None))                 # control: the metric of the feature covariance (no anchor)
    rows = []
    for label, D in anchors:
        cov = D is None
        if cov:
            sD = np.linalg.eigvalsh(prob.c_loss * prob.Sigma_hat + prob.lam * np.eye(d))[::-1]
        else:
            sD = np.linalg.svd(D if Q is None else Q.T @ D, compute_uv=False)
        for f in FLOORS:
            if cov:
                algo = vm.VRMuon(prob, estimator="saga", placement="metric", metric_source="cov", metric_floor=f,
                                 c_metric=1.0)
                V, gains, r, _, R2P = algo.polar_metric(svd=algo.covariance_factors())
            else:
                algo = vm.VRMuon(prob, estimator="saga", placement="metric", metric_floor=max(f, 1e-14),
                                 c_metric=1.0)
                V, gains, r, _, R2P = algo.polar_metric(D)
            Cm = V @ np.diag(gains) @ V.T + r * (np.eye(d) - V @ V.T)
            w, U = np.linalg.eigh(Cm)
            Ch = (U * np.sqrt(w)) @ U.T
            T = np.kron(np.eye(Kg), Ch)
            evP = np.linalg.eigvalsh(T @ H @ T)
            mu_glob = float(evP[0]) if prob.loss == "ls" else float(prob.lam * w[0])
            rows.append({"anchor": label, "floor": f, "kappa_P": float(R2P / evP[0]),
                         "kappa_P_global": float(R2P / mu_glob), "cond_P": float(evP[-1] / evP[0]),
                         "cond_C": float(w[-1] / w[0]), "floored": int(np.sum(sD < f * sD[0])), "rank": int(len(sD)),
                         "smin_over_smax": float(sD[-1] / sD[0])})
    Hinv_lev = None
    if prob.loss == "ls":
        Hd = prob.c_loss * prob.Sigma_hat + prob.lam * np.eye(d)
        Hinv_lev = float(prob.c_loss * np.einsum("id,de,ie->i", prob.X, np.linalg.inv(Hd), prob.X).max())
    return {"rows": rows, "kappa_euclid": float(prob.R2 / ev[0]), "cond_euclid": float(ev[-1] / ev[0]),
            "kappa_euclid_global": float(prob.R2 / ev[0]) if prob.loss == "ls" else float(prob.R2 / prob.lam),
            "loss": prob.loss, "n": int(n), "d_inf": Hinv_lev, "saga_gamma": saga_gamma}


def spread(s):
    prob = C.problem(s)
    L = np.asarray(prob.L_i)
    p = np.sqrt(L) / np.sqrt(L).sum()
    return {"R2_over_mean_L": float(L.max() / L.mean()), "max_over_min_L": float(L.max() / L.min()),
            "n_pmax_sqrt": float(prob.n * p.max()), "n_pmin_sqrt": float(prob.n * p.min()),
            "n_pmin_lin": float(prob.n * (L / L.sum()).min())}


def main():
    jobs = None
    for i, a in enumerate(sys.argv):
        if a == "--jobs":
            jobs = int(sys.argv[i + 1])
    from exp_M1_headtohead import INSTANCES
    m1 = C.load_json("M1_summary") or {}
    t0 = time.perf_counter()
    out = {"metric": {}, "katyusha": {}, "spread": {}, "floors": list(FLOORS), "code": C.code_fingerprint()}

    # ---- (C) spread of the per-sample constants
    print("(C) per-sample smoothness constants L_i = c ||x_i||^2 + lam")
    for key, inst in INSTANCES.items():
        out["spread"][key] = sp = spread(inst["spec"])
        print(f"  {key:10s} R2 / mean L_i = {sp['R2_over_mean_L']:.2f}, max / min = {sp['max_over_min_L']:.1f}, "
              f"n p_i for p ~ L^(1/2) in [{sp['n_pmin_sqrt']:.2f}, {sp['n_pmax_sqrt']:.2f}], "
              f"smallest n p_i for p ~ L: {sp['n_pmin_lin']:.3f}", flush=True)

    # ---- (A) conditioning in the polar metric
    for key, inst in INSTANCES.items():
        s = inst["spec"]
        row = next((r for r in m1.get(key, {}).get("rows", []) if r["family"] == "SAGA"), None)
        res = metric_conditioning(s, row["best"]["gamma"] if row else None, later_anchors=key in LATER_ANCHORS)
        out["metric"][key] = res
        soft = res["loss"] != "ls"
        print(f"\n(A) {key}: kappa = R2 / mu_true = {res['kappa_euclid']:.0f} in the Euclidean geometry"
              + (f" (global: R2 / lam = {res['kappa_euclid_global']:.0f})" if soft else "")
              + (f", d_inf = {res['d_inf']:.1f}" if res["d_inf"] else "") + "; kappa_P = R2P / mu_P by floor"
              + (" (local constant; second line: global constant)" if soft else "") + ":")
        print(f"  {'anchor':46s} " + " ".join(f"{'f=' + format(f, 'g'):>10s}" for f in FLOORS) + "   floored directions")
        labels = []
        for r in res["rows"]:
            if r["anchor"] not in labels:
                labels.append(r["anchor"])
        for lab in labels:
            rr = [r for r in res["rows"] if r["anchor"] == lab]
            print(f"  {lab:46s} " + " ".join(f"{r['kappa_P']:10.0f}" for r in rr) + "   "
                  + "/".join(str(r["floored"]) for r in rr) + f" of {rr[0]['rank']}", flush=True)
            if soft:
                print(f"  {'':46s} " + " ".join(f"{r['kappa_P_global']:10.0f}" for r in rr), flush=True)
    C.save_json("M7_summary", out)

    # ---- (B) weight of the y step in loopless Katyusha
    tasks, plan = [], []
    for key, inst in INSTANCES.items():
        row = next((r for r in m1.get(key, {}).get("rows", []) if r["family"] == "L-Katyusha"), None)
        if row is None:
            continue
        base = {"L": row["best"]["L"], "mu": row["best"]["mu"]}
        variants = [("coupling (L-Katyusha)", dict(ystep="coupling")),
                    ("no y step (y = x)", dict(ystep="euclid", gamma_y=0.0)),
                    ("Euclidean y step, gamma_y = 1/(3L)", dict(ystep="euclid"))]
        variants += [(f"sharp y step, gamma_y = {c:.3g}/R2_op", dict(ystep="sharp", gamma_y=over("R2_op", c)))
                     for c in GAMMA_Y]
        for label, kw in variants:
            cfg = dict(base)
            cfg.update(kw)
            lo = len(tasks)
            for seed in C.CONFIRM_SEEDS:
                tasks.append({"spec": inst["spec"], "method": "katyusha", "cfg": cfg, "passes": inst["passes"],
                              "seed": int(seed), "stop_rel": C.CONFIRM_STOP * inst["tol"]})
            plan.append((key, label, lo, len(tasks), inst["tol"]))
        out["katyusha"][key] = {"L": row["best"]["L"], "mu": row["best"]["mu"], "tol": inst["tol"], "rows": {}}
    res = C.run_tasks(tasks, "M7_katyusha", jobs) if tasks else []
    for key, label, lo, hi, tol in plan:
        v = [r["passes_to"][f"{tol:g}"] for r in res[lo:hi]]
        out["katyusha"][key]["rows"][label] = {"passes": v, "final": [r["final_rel"] for r in res[lo:hi]],
                                              "diverged": int(sum(r["diverged"] for r in res[lo:hi]))}
    for key, blk in out["katyusha"].items():
        print(f"\n(B) {key}: L = {C.fmt_point({'L': blk['L'], 'mu': blk['mu']})}, passes to {blk['tol']:g}")
        for label, e in blk["rows"].items():
            v = np.array(e["passes"], dtype=float)
            txt = (f"{np.median(v):.1f} [{v.min():.1f}, {v.max():.1f}]" if np.isfinite(v).all() else
                   f"reached {int(np.isfinite(v).sum())}/{len(v)}, final {np.median(e['final']):.1e}")
            print(f"  {label:42s} {txt}", flush=True)
    C.save_json("M7_summary", out)
    print(f"\nM7 done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
