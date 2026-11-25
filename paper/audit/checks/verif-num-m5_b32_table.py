"""Counter-check of finding M5-b32-path-length (paper lines 538-540), part 2: at b = 32, on exactly the schedule
of the controls without a table (muon/results/M9_summary.json, C/<inst>/controls_schedule), does the SAGA table
freeze too? And does an exact (full) gradient freeze too?

Own driver (not the auditor's script, not experiments/common.run_one). Two implementations:
  lab   : lab.vrmuon.vr_muon / lab.mvr.mvr_muon called directly;
  own   : an independent loop (own ns5, own Nesterov buffer, SAGA as a table of the n per-sample gradients
          K x d, or the exact gradient of F as an oracle that is charged b gradients per step).
Run: PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 <this file> [part]
  part A (default): the schedule of the paper;  part B: schedules at b = 32 whose path length is sufficient.
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "muon"))
sys.path.insert(0, ROOT)
from lab import mvr, optimum, problems, vrmuon  # noqa: E402

N, PASSES = 200, 300
SEEDS = (10, 11, 12, 13, 14)
NS = (3.4445, -4.7750, 2.0315)


def build(loss):
    mk = problems.make_softmax if loss == "softmax" else problems.make_matrix_ls
    prob = mk(n=N, d=40, K=8, kappa=1e4, alpha=2.0, seed=0)
    optimum.solve_optimum(prob)
    return prob


def own_ns5(M, eps=1e-7):
    a, b, c = NS
    X = M / (np.sqrt((M * M).sum()) + eps)            # K <= d: no transposition
    for _ in range(5):
        G = X @ X.T
        X = a * X + (b * G + c * (G @ G)) @ X
    return X


def own_run(prob, est, b, q, lr0, seed, beta=0.9, passes=PASSES):
    """Fixed-length recipe (4) with Nesterov momentum beta, lr_t = lr0 q^(t b / n), W_0 = 0.
    est: 'saga' (table of per-sample loss gradients), 'sgd' (minibatch gradient), 'exact' (grad F, charged b)."""
    n, lam = prob.n, prob.lam
    W = np.zeros((prob.K, prob.d))
    buf = np.zeros_like(W)
    rng = np.random.default_rng(seed)
    budget, ge = passes * n, 0
    if est == "saga":
        table = np.stack([prob.grad_i(W, i) - lam * W for i in range(n)])
        tbar = table.mean(axis=0)
        ge = n
    t, path = 0, 0.0
    while ge < budget:
        B = rng.choice(n, size=b, replace=False, shuffle=False)
        if est == "exact":
            omega = prob.grad(W)
        elif est == "sgd":
            omega = prob.grad_batch(W, B)
        else:
            gB = np.stack([prob.grad_i(W, i) - lam * W for i in B])
            diff = gB - table[B]
            omega = tbar + diff.mean(axis=0) + lam * W
            tbar = tbar + diff.sum(axis=0) / n
            table[B] = gB
        buf = beta * buf + omega
        D = (1.0 - beta) * (omega + beta * buf)
        O = own_ns5(prob.remove_gauge(D))
        lr = lr0 * q ** (t * b / n)
        W = W - lr * O
        path += lr * float(np.linalg.norm(O, 2))
        ge += b
        t += 1
    Ws = prob.W_star
    return {"rel": float(((W - Ws) ** 2).sum() / (Ws * Ws).sum()), "steps": t, "path_op": path,
            "W_op": float(np.linalg.norm(W, 2))}


def lab_vr(prob, cfg, seed, passes=PASSES):
    n = prob.n
    budget = passes * n
    init = 0 if cfg["estimator"] == "sgd" else n
    n_it = max(1, (budget - init) // cfg["batch"])
    r = vrmuon.vr_muon(prob, np.zeros((prob.K, prob.d)), n_it, seed=seed, max_grad_evals=budget, stop_dist2=0.0,
                       **cfg)
    D0 = float(np.vdot(prob.W_star, prob.W_star))
    rel = np.asarray(r["dist2_F"], dtype=float) / D0
    return float(rel[-1]), float(np.nanmin(rel)), r["stopped"]


def lab_mvr(prob, b, q, lr0, seed, passes=PASSES, **kw):
    n = prob.n
    budget = passes * n
    r = mvr.mvr_muon(prob, np.zeros((prob.K, prob.d)), max(1, budget // (2 * b)), seed=seed, stop_dist2=0.0,
                     max_grad_evals=budget, batch=b, lr=lr0, beta=0.95, same_batch=True, orth="ns5",
                     schedule=["geometric", q, n / (2.0 * b)], **kw)
    D0 = float(np.vdot(prob.W_star, prob.W_star))
    rel = np.asarray(r["dist2_F"], dtype=float) / D0
    return float(rel[-1]), float(np.nanmin(rel)), r["stopped"]


def med(v):
    v = np.asarray(v, dtype=float)
    return f"median {np.median(v):.4g} [{v.min():.4g}, {v.max():.4g}]"


def recipe(est, b, q, lr0, orth="ns5", mom=True):
    cfg = dict(estimator=est, batch=b, length="const", orth=orth, schedule="geometric", lr_decay=q, lr=lr0)
    if mom:
        cfg.update(beta=0.9, nesterov=True)
    return cfg


def part_A(S):
    for key, loss in (("soft-cond", "softmax"), ("ls-cond", "ls")):
        prob = build(loss)
        sc = S[key]["controls_schedule"]
        q, lr0 = sc["lr_decay"], sc["lr"]
        wop = float(np.linalg.norm(prob.W_star, 2))
        print(f"\n== {key}: schedule of the controls q = {q}, lr_0 = {lr0}, {PASSES} passes, seeds {SEEDS}; "
              f"||W_*||_op = {wop:.3f}")
        js = S[key]["controls"]
        print("  final relative squared distance (lab), b = 32:")
        v = [lab_vr(prob, recipe("sgd", 32, q, lr0), s)[0] for s in SEEDS]
        print(f"    NO table, minibatch, ns5 + Nesterov      {med(v)}   json median "
              f"{js['no table (minibatch gradient), ns5 + Nesterov, b=32']['final_rel']['median']:.4g}")
        v = [lab_vr(prob, recipe("saga", 32, q, lr0), s)[0] for s in SEEDS]
        print(f"    SAGA TABLE, ns5 + Nesterov               {med(v)}")
        v = [lab_vr(prob, recipe("saga", 32, q, lr0, orth="svd", mom=False), s)[0] for s in SEEDS]
        print(f"    SAGA TABLE, exact polar, no momentum     {med(v)}")
        v = [lab_mvr(prob, 32, q, lr0, s, g=0.025, clip=1.0)[0] for s in SEEDS]
        print(f"    MARS-M b = 32 (64 gradients per step)    {med(v)}   json median "
              f"{js['MARS-M, b=32']['final_rel']['median']:.4g}")
        v = [lab_mvr(prob, 32, q, lr0, s, g=1.0)[0] for s in SEEDS]
        print(f"    STORM-Muon b = 32 (64 gradients per step) {med(v)}   json median "
              f"{js['STORM-Muon, b=32']['final_rel']['median']:.4g}")
        v = [lab_vr(prob, recipe("saga", 64, q, lr0), s)[0] for s in SEEDS]
        print(f"    SAGA TABLE b = 64 (step schedule of MARS-M b = 32), ns5 + Nesterov   {med(v)}")
        print("  independent loop (own ns5, own table), b = 32:")
        r = own_run(prob, "exact", 32, q, lr0, 0)
        print(f"    EXACT gradient of F, ns5 + Nesterov      rel {r['rel']:.4g}  ({r['steps']} steps, path "
              f"sum lr_t ||O_t||_op = {r['path_op']:.3f} = {r['path_op'] / wop:.3f} ||W_*||_op, "
              f"||W_T||_op = {r['W_op']:.3f})")
        r = own_run(prob, "exact", 64, q, lr0, 0)
        print(f"    EXACT gradient, step schedule of b = 64  rel {r['rel']:.4g}  (path {r['path_op']:.3f} = "
              f"{r['path_op'] / wop:.3f} ||W_*||_op)")
        for s in SEEDS[:2]:
            a = own_run(prob, "saga", 32, q, lr0, s)
            c = own_run(prob, "sgd", 32, q, lr0, s)
            print(f"    seed {s}: own SAGA table rel {a['rel']:.4g} (path {a['path_op']:.3f}, ||W_T||_op "
                  f"{a['W_op']:.3f}) | own minibatch rel {c['rel']:.4g} (path {c['path_op']:.3f}) | lab SAGA "
                  f"{lab_vr(prob, recipe('saga', 32, q, lr0), s)[0]:.4g}, lab minibatch "
                  f"{lab_vr(prob, recipe('sgd', 32, q, lr0), s)[0]:.4g}")


def part_B(S):
    """b = 32 with enough path: (q, lr_0) with sum_t lr_t >= 2 ||W_*||_op; table against no table, seeds 10, 11."""
    pts = [(0.9, 1.0), (0.9, 3.0), (0.8, 3.0), (0.95, 1.0)]
    for key, loss in (("soft-cond", "softmax"), ("ls-cond", "ls")):
        prob = build(loss)
        wop = float(np.linalg.norm(prob.W_star, 2))
        print(f"\n== {key}, b = 32, {PASSES} passes, seeds 10 and 11: final (lowest) relative squared distance")
        print(f"   {'q':>5s} {'lr_0':>5s} {'L/|W*|op':>9s} {'lr_T':>9s} | {'SAGA table ns5+Nesterov':>34s} | "
              f"{'minibatch ns5+Nesterov':>24s} | {'MARS-M (L/2)':>24s}")
        for q, lr0 in pts:
            L = lr0 / (1.0 - q ** (32 / N))
            a = [lab_vr(prob, recipe("saga", 32, q, lr0), s) for s in (10, 11)]
            c = [lab_vr(prob, recipe("sgd", 32, q, lr0), s) for s in (10, 11)]
            m = [lab_mvr(prob, 32, q, lr0, s, g=0.025, clip=1.0) for s in (10, 11)]
            fa = ", ".join(f"{x[0]:.2e} ({x[1]:.1e})" for x in a)
            fc = ", ".join(f"{x[0]:.3g}" for x in c)
            fm = ", ".join(f"{x[0]:.3g}" for x in m)
            print(f"   {q:5.2f} {lr0:5.1f} {L / wop:9.2f} {lr0 * q ** PASSES:9.1e} | {fa:>34s} | {fc:>24s} | {fm:>24s}",
                  flush=True)


if __name__ == "__main__":
    t0 = time.process_time()
    S = json.load(open(os.path.join(ROOT, "muon", "results", "M9_summary.json")))["C"]
    (part_B if (len(sys.argv) > 1 and sys.argv[1] == "B") else part_A)(S)
    print(f"\nCPU {time.process_time() - t0:.1f} s")
