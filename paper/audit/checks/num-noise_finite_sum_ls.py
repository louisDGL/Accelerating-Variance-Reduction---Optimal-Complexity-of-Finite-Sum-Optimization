"""Audit num-noise, check 4: the finite-sum least-squares problem of Figure 3(b)-(c) and of "Numerical checks".

Uses the functions of muon/theory_checks/make_paper_figures.py (imported read only; the cache is not touched) with
NEW seeds, and compares with the cached numbers quoted by the paper:
  (a) the problem: n, K, d, intercept, skewness of the residuals, ||W_LAD - W_LS||_F (paper: 0.170), and the exact
      mean field of the batch-one polar step at W_LAD (enumeration of the n indices; must vanish) and at W_LS;
  (b) batch one, eta_t = 0.3 (1 + t/200)^(-3/4), 32 runs, average over (t/2, t]: distance of the minibatch iterate to
      W_LAD (paper: 2.6e-3) and of the SAGA-estimator iterate to W_LS (paper: 2.1e-4), other seeds;
  (c) zero of the mean field for b = 2 and b = 32 from 200000 fresh batches (paper: b x ||grad F||_* = 0.240, 0.308)
      and a 64-run check at b = 2 (paper: within 5 % of the zero);
  (d) SAGA estimator with the nuclear-scaled length (paper: "ends at rounding level").

Run from the project root (parts are separate to stay under 3 minutes each):
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 paper/audit/checks/num-noise_finite_sum_ls.py [a|b1|b2|c|d]
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
spec = importlib.util.spec_from_file_location("mpf", os.path.join(ROOT, "muon", "theory_checks", "make_paper_figures.py"))
PF = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PF)
cache = json.load(open(os.path.join(ROOT, "muon", "theory_checks", "paper_figures_data.json")))
tr, rows = cache["noise_trace"], cache["noise_offsets"]["rows"]
parts = sys.argv[1:] or ["a"]

X, Y, W_ls, Sig = PF.ls_problem()
n, d = X.shape
K = Y.shape[1]
W_lad = PF.weighted_lad(X, Y, W_ls)
gnorm = lambda W: PF.nuc((W - W_ls) @ Sig)                               # noqa: E731

if "a" in parts:
    R = X @ W_ls.T - Y
    skew = np.mean((R - R.mean(0)) ** 3, axis=0) / np.std(R, axis=0) ** 3
    print(f"(a) n = {n}, K = {K}, d = {d}; first column of X constant: {bool(np.all(X[:, 0] == 1.0))}; no ridge term "
          f"(W_LS solves the normal equations: ||grad F(W_LS)||_F = {np.linalg.norm((X @ W_ls.T - Y).T @ X / n):.1e})")
    print(f"    skewness of the least-squares residuals per output: {skew.round(2)} (Exp(1) - 1 has skewness 2)")
    print(f"    ||W_LAD - W_LS||_F = {np.linalg.norm(W_lad - W_ls):.5f} (cache {tr['dist_lad_ls']:.5f}; paper 0.170); "
          f"||grad F(W_LAD)||_* = {gnorm(W_lad):.5f} (cache {tr['offset_b1_exact']:.5f}; paper 0.17)")

    def field1(W):                                                       # exact mean field of the batch-one polar step
        Rr = X @ W.T - Y
        w = 1.0 / (np.linalg.norm(Rr, axis=1) * np.linalg.norm(X, axis=1))
        return (Rr * w[:, None]).T @ X / n
    print(f"    exact batch-one mean field (n rank-one polar factors): ||Phi(W_LAD)||_F = {np.linalg.norm(field1(W_lad)):.1e}, "
          f"||Phi(W_LS)||_F = {np.linalg.norm(field1(W_ls)):.3f}")
    lam = lambda W: float(np.mean(np.linalg.norm(X @ W.T - Y, axis=1) / np.linalg.norm(X, axis=1)))   # noqa: E731
    rng = np.random.default_rng(0)
    worse = min(lam(W_lad + 1e-3 * rng.standard_normal((K, d))) - lam(W_lad) for _ in range(200))
    print(f"    Lambda(W_LAD) = {lam(W_lad):.6f} < Lambda(W_LS) = {lam(W_ls):.6f}; smallest increase of Lambda over 200 "
          f"random perturbations of W_LAD: {worse:.2e} (> 0: W_LAD minimizes Lambda)")
    print(f"    length schedule: eta_1 = {PF.length(1):.4f}, eta_T = {PF.length(100000):.5f}; sum eta_t^2 over 1e5 steps = "
          f"{sum(PF.length(np.arange(1, 100001)) ** 2):.3f} (exponent 3/4: square summable, not summable)")

cps = [int(c) for c in np.unique(np.round(np.geomspace(200, 100000, 25)).astype(int))]
if "b1" in parts:
    t0 = time.time()
    mb = PF.run_minibatch(X, Y, 1, 100000, 32, 1010, cps)
    dl = [float(np.linalg.norm(mb[c] - W_lad)) for c in cps]
    ds = [float(np.linalg.norm(mb[c] - W_ls)) for c in cps]
    print(f"(b1) minibatch, batch one, seed 1010 ({time.time() - t0:.0f} s): distance to W_LAD at the last five checkpoints "
          f"{[f'{v:.2e}' for v in dl[-5:]]} (cache: {[f'{v:.2e}' for v in tr['mb_to_lad'][-5:]]}); to W_LS at the end "
          f"{ds[-1]:.4f} (cache {tr['mb_to_ls'][-1]:.4f})")
if "b2" in parts:
    t0 = time.time()
    sg = PF.run_saga(X, Y, 1, 100000, 32, 1011, cps)
    ds = [float(np.linalg.norm(sg[c] - W_ls)) for c in cps]
    print(f"(b2) SAGA estimator, same decaying length, batch one, seed 1011 ({time.time() - t0:.0f} s): distance to W_LS at "
          f"the last five checkpoints {[f'{v:.2e}' for v in ds[-5:]]} (cache: {[f'{v:.2e}' for v in tr['saga_to_ls'][-5:]]}); "
          f"largest value over the run {max(ds):.2f} (cache {max(tr['saga_to_ls']):.2f})")
if "c" in parts:
    for b, W_init in ((2, W_lad), (32, W_ls)):
        t0 = time.time()
        Wb, res = PF.mean_field_root(X, Y, b, 200000, 7000 + b, W_init)
        row = next(r for r in rows if r["b"] == b)
        print(f"(c) b = {b}: zero of the mean field from 200000 fresh batches: ||grad F||_* = {gnorm(Wb):.5f}, b x = "
              f"{b * gnorm(Wb):.4f} (cache {row['root_mean']:.5f}, b x {b * row['root_mean']:.4f}); residual of the field "
              f"{res:.1e} ({time.time() - t0:.0f} s)")
    t0 = time.time()
    end = PF.run_minibatch(X, Y, 2, 20000, 64, 2002, [20000])[20000]
    row = next(r for r in rows if r["b"] == 2)
    print(f"    b = 2, 64 runs of 20000 steps, seed 2002: ||grad F||_* of the averaged iterate = {gnorm(end):.5f} (cache "
          f"{row['run']:.5f}); gap to the cached zero {100 * (gnorm(end) - row['root_mean']) / row['root_mean']:+.1f} % "
          f"({time.time() - t0:.0f} s)")
    print("    cache, all b: " + "; ".join(f"b={int(r['b'])}: run/root - 1 = {100 * (r['run'] / r['root_mean'] - 1):+.1f} %"
                                            for r in rows))
if "d" in parts:
    gamma = 1.0 / (3.0 * np.max(np.sum(X * X, axis=1)))
    for b in (1, 32):
        t0 = time.time()
        Ts = 200000 // b + 2000
        out = PF.run_saga(X, Y, b, Ts, 2, 9000 + b, [Ts // 4, Ts // 2, Ts], nuclear_scaled=True, gamma=gamma)
        print(f"(d) SAGA estimator, nuclear-scaled length (gamma = 1/(3 max ||x_i||^2) = {gamma:.4f}), b = {b}, 2 runs, "
              f"{Ts} steps: max ||grad F||_* at T/4, T/2, T = "
              + ", ".join(f"{max(gnorm(Wc) for Wc in out[c]):.1e}" for c in (Ts // 4, Ts // 2, Ts))
              + f" (cache at T: {next(r for r in rows if r['b'] == b)['saga']:.1e}) ({time.time() - t0:.0f} s)")
