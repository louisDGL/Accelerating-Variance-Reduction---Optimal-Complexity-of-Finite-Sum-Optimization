"""Counter-check of finding M5-b32-path-length (paper lines 538-540), part 1: path length available to the
controls without a table, from the recipe (4) lr_t = lr_0 q^(t g / n) alone (g = gradients per step).

No run of an optimizer here. Inputs: n, W_* (singular values) of the two instances, W_0 = 0, the schedule of
muon/results/M9_summary.json part C. For every control:
  L        = sum_t lr_t over the steps that 300 passes allow (spectral length of the path if ||o||_op = 1)
  smax     = sup over [0, 1] of the scalar map of ns5 (five steps of p(x) = a x + b x^3 + c x^5): ||ns5(M)||_op <= smax
  R        = smax L >= ||W_T - W_0||_op, whatever the estimator
  floor    = min over ||W||_op <= R of ||W - W_*||_F^2 / ||W_*||_F^2 = sum_i (s_i - R)_+^2 / sum_i s_i^2
             (s_i singular values of W_*; W_0 = 0): a lower bound on the final relative squared distance of ANY
             fixed-length method on that schedule.
Run: PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 <this file>
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "muon"))
sys.path.insert(0, ROOT)
from lab import optimum, problems  # noqa: E402

N, PASSES = 200, 300
A, B, Cc = 3.4445, -4.7750, 2.0315


def ns5_scalar(x, steps=5):
    x = np.array(x, dtype=float)
    for _ in range(steps):
        x2 = x * x
        x = x * (A + x2 * (B + Cc * x2))
    return x


def build(loss):
    mk = problems.make_softmax if loss == "softmax" else problems.make_matrix_ls
    prob = mk(n=N, d=40, K=8, kappa=1e4, alpha=2.0, seed=0)
    optimum.solve_optimum(prob)
    return prob


def main():
    grid = np.linspace(0.0, 1.0, 2_000_001)
    out5 = ns5_scalar(grid)
    smax = float(out5.max())
    print(f"sup of the ns5 scalar map on [0,1]: {smax:.4f} at x = {grid[out5.argmax()]:.4f} "
          f"(inf on [0.05, 1]: {out5[grid >= 0.05].min():.4f})")
    S = json.load(open(os.path.join(ROOT, "muon", "results", "M9_summary.json")))["C"]
    for key, loss in (("soft-cond", "softmax"), ("ls-cond", "ls")):
        prob = build(loss)
        Ws = prob.W_star
        sv = np.linalg.svd(Ws, compute_uv=False)
        D0, wop = float((sv ** 2).sum()), float(sv[0])
        sc = S[key]["controls_schedule"]
        q, lr0 = sc["lr_decay"], sc["lr"]
        print(f"\n== {key}: n = {prob.n}, K x d = {prob.K} x {prob.d}, ||W_*||_op = {wop:.4f} "
              f"(json {S[key]['norms']['W_star_op']:.4f}), ||W_*||_F^2 = {D0:.2f} (json {S[key]['norms']['D0']:.2f})")
        print(f"   singular values of W_*: {np.array2string(sv, precision=2)}")
        print(f"   schedule of the controls (json C/{key}/controls_schedule): q = {q}, lr_0 = {lr0}; W_0 = 0")
        print(f"   {'control':34s} {'g':>3s} {'steps':>6s} {'L=sum lr_t':>10s} {'L/|W*|op':>9s} {'smax L/|W*|op':>13s} "
              f"{'floor(L)':>9s} {'floor(smax L)':>13s} {'observed median':>15s}")
        for label, cfg in S[key]["controls"].items():
            c = cfg["cfg"]
            b = c["batch"]
            g = b if cfg["method"] == "vr" else 2 * b          # MARS-M / STORM-Muon: same_batch=True, 2 b gradients
            if cfg["method"] == "mvr":
                assert c["same_batch"] and c["schedule"][0] == "geometric" and c["schedule"][1] == q
                assert abs(c["schedule"][2] - N / (2.0 * b)) < 1e-12 and c["lr"] == lr0
            else:
                assert c["lr_decay"] == q and c["lr"] == lr0 and c["estimator"] == "sgd"
            T = (PASSES * N) // g
            lrs = lr0 * q ** (np.arange(T) * g / N)
            L = float(lrs.sum())
            Linf = lr0 / (1.0 - q ** (g / N))
            assert abs(L - Linf) < 1e-6 * Linf                  # 300 passes exhaust the geometric series
            fl = [float((np.clip(sv - R, 0.0, None) ** 2).sum() / D0) for R in (L, smax * L)]
            print(f"   {label[:34]:34s} {g:3d} {T:6d} {L:10.3f} {L / wop:9.3f} {smax * L / wop:13.3f} "
                  f"{fl[0]:9.4f} {fl[1]:13.4f} {cfg['final_rel']['median']:15.4f}")


if __name__ == "__main__":
    main()
