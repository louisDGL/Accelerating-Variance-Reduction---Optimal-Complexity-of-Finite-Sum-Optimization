#!/usr/bin/env python3
"""Audit math-order, check 2: the figure "single runs under independent sampling lose about 1.6 (measured, n = 1000)".

(a) reads the stored values (results/B4b_summary.json, typical.n=1000.<c>; results/B4_summary.json if present);
(b) small independent reproduction on the scalar model of note B4 (own generator, own code): n = 1000, i.i.d. indices,
    8 runs per step, 260 passes, exponent = minus the least-squares slope over passes 60..260 of
      - log max(e^2, max_j psi_j^2)  (the quantity fitted by experiments/exp_B4b_refinements.py), and
      - log e^2 at pass ends          (the squared error itself, which is what the paper's sentence is about);
    the same for a fresh permutation per pass (c = 2, 32, 250), against the exact mean-square exponent.

Run from the project root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 paper/audit/checks/math-order_typical_runs.py
"""
import json
import math
import os
import sys

import numpy as np

sys.dont_write_bytecode = True          # read-only audit: no bytecode cache written into the repository
sys.path.insert(0, os.getcwd())

N, PASSES, BURN, SEEDS = 1000, 260, 60, 8
CS = [0.25, 1.0, 2.0, 4.0, 8.0, 32.0, 125.0, 250.0]


def T_iid(n, g):
    T = np.zeros((4, 4))
    T[0] = [(1 - g) ** 2, g * g, -g * g, 0.0]
    T[1] = [1 / n, 1 - 1 / n, 0, 0]
    T[2] = [1 / n ** 2, 1 / n ** 2, 1 - 2 / n, 2 / n - 2 / n ** 2]
    T[3] = [(1 - g) / n, -g / n, g / n, (1 - g) * (1 - 1 / n)]
    return T


def simulate(n, cs, seeds, passes, order, rng):
    """R = len(cs) * seeds runs in parallel; state rescaled at pass ends. Returns (log size^2, log e^2) at pass ends."""
    R = len(cs) * seeds
    g = np.repeat(np.asarray(cs, dtype=float), seeds) / n
    e, psi = np.ones(R), np.ones((R, n))
    ar = np.arange(R)
    logscale = np.zeros(R)
    log_size2 = np.zeros((passes + 1, R))
    log_e2 = np.zeros((passes + 1, R))
    for p in range(passes):
        if order == "iid":
            idx = rng.integers(0, n, size=(n, R))
        else:
            idx = np.argsort(rng.random((R, n)), axis=1).T          # one fresh permutation per run and per pass
        pbar = psi.mean(axis=1)
        for s in range(n):
            i = idx[s]
            d = e - psi[ar, i]
            en = e - g * (d + pbar)
            pbar += d / n
            psi[ar, i] = e
            e = en
        size = np.maximum(np.abs(e), np.abs(psi).max(axis=1))
        e /= size
        psi /= size[:, None]
        logscale += np.log(size)
        log_size2[p + 1] = 2 * logscale
        with np.errstate(divide="ignore"):
            log_e2[p + 1] = 2 * (logscale + np.log(np.abs(e)))
    return log_size2, log_e2


def slope(y, burn):
    x = np.arange(y.shape[0], dtype=float)
    sel = x >= burn
    out = []
    for r in range(y.shape[1]):
        ok = sel & np.isfinite(y[:, r])
        out.append(-np.polyfit(x[ok], y[ok, r], 1)[0])
    return np.array(out)


def main():
    print("(a) stored values, results/B4b_summary.json, typical.n=1000 (12 seeds, passes 60-260)")
    d = json.load(open("results/B4b_summary.json"))["typical"]["n=1000"]
    print("    c        median    min     max   expectation(exact)   permutation(exact)")
    for c, v in d.items():
        print(f"    {c:>5s}   {v['median']:7.3f} {v['min']:7.3f} {v['max']:7.3f}   {v['expectation']:10.4f}   {v['rr']:14.4f}")
    med = [v["median"] for v in d.values()]
    print(f"    range of the medians over the 9 steps: {min(med):.3f} .. {max(med):.3f};"
          f"  'about 1.6' holds for c in {{1, 2, 4}} only ({d['1']['median']:.2f}, {d['2']['median']:.2f}, {d['4']['median']:.2f})")
    try:
        b4 = json.load(open("results/B4_summary.json"))["model"]["by_c"]
        row = {c: b4[c]["n=1000"]["iid_typical"]["median"] for c in b4 if "n=1000" in b4[c]}
        print("    results/B4_summary.json, model.by_c.<c>.n=1000.iid_typical.median (9 seeds, passes 30-150):")
        print("    " + ", ".join(f"c={c}: {v:.3f}" for c, v in row.items()))
    except Exception as exc:
        print("    (B4_summary.json not read:", repr(exc), ")")

    print(f"(b) reproduction, n = {N}, {SEEDS} runs per step, {PASSES} passes, fit on passes {BURN}..{PASSES}")
    rng = np.random.default_rng(987654321)
    ls, le = simulate(N, CS, SEEDS, PASSES, "iid", rng)
    s_size, s_e = slope(ls, BURN), slope(le, BURN)
    print("    i.i.d. indices")
    print("    c        state size: median [min, max]     e^2: median [min, max]     exact expectation   min{2c,1}")
    for k, c in enumerate(CS):
        a = s_size[k * SEEDS:(k + 1) * SEEDS]
        b = s_e[k * SEEDS:(k + 1) * SEEDS]
        ex = -N * math.log(np.abs(np.linalg.eigvals(T_iid(N, c / N))).max())
        print(f"    {c:>6g}   {np.median(a):6.3f} [{a.min():.3f}, {a.max():.3f}]          {np.median(b):6.3f} [{b.min():.3f}, {b.max():.3f}]"
              f"        {ex:8.4f}        {min(2*c,1):.1f}")
    # expectation against runs at 20 passes, c = 2 (the note: exact 3.7e-12, median of 9 runs 7.3e-17)
    k = CS.index(2.0)
    v20 = np.exp(le[20, k * SEEDS:(k + 1) * SEEDS])
    ex20 = (np.linalg.matrix_power(T_iid(N, 2.0 / N), 20 * N) @ np.ones(4))[0]
    print(f"    c = 2, after 20 passes: exact E e^2 = {ex20:.2e}; e^2 of the {SEEDS} runs: median {np.median(v20):.2e}, mean {v20.mean():.2e}, max {v20.max():.2e}")

    cs_rr = [2.0, 32.0, 250.0]
    ls, le = simulate(N, cs_rr, SEEDS, 120, "perm", rng)
    s_size = slope(ls, 30)
    print("    fresh permutation per pass (120 passes, fit on 30..120)")
    try:
        from vrlab import order_model as om
        exact = [om.rr_ms_exponent(N, c) for c in cs_rr]
    except Exception:
        exact = [float("nan")] * len(cs_rr)
    for k, c in enumerate(cs_rr):
        a = s_size[k * SEEDS:(k + 1) * SEEDS]
        print(f"    c = {c:>5g} (g = {c/N:.3f}): runs median {np.median(a):.3f} [{a.min():.3f}, {a.max():.3f}]; exact mean-square exponent {exact[k]:.4f}")


if __name__ == "__main__":
    main()
