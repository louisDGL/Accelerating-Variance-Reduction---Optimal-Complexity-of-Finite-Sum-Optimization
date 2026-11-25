"""Audit num-noise, check 3: scope of (M2) "fails where MISO fails (kappa >> n): it is not a spectral method", and of
"reaches rounding precision" in (M1). Small re-runs with the lab code (read only; nothing is written to the repository).

(a) Instances of exp_M6 with kappa / n = 10 (n = 1000, d = 50, K = 10): joint SAGA-Muon at gamma = c / R2_op for the two
    constants of M6 (c = 1/4, 1) and for smaller ones (c = 1/16, 1/64), against MISO (delta = 1); 40 passes, seed 0.
(b) The two non-square conditioning-limited instances of the paper (soft-cond, ls-cond: kappa / n = 48 and 36): MISO
    (delta = 1) against the joint SAGA-Muon at the step selected by M1 (gamma = 0.5 / R2_op and 0.125 / R2_op);
    100 passes, confirmation seed 10.
(c) ls-mem, SAGA estimator with the nuclear-scaled length of exp_M3 (gamma = 0.25 / R2_op): 110 passes instead of 60,
    stop at 1e-30: where does the run end?

Run from the project root (about 2 minutes):
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 paper/audit/checks/num-noise_miso_regime.py [a|b|c]
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "muon", "experiments"))
import common as C  # noqa: E402
from common import over  # noqa: E402

parts = sys.argv[1:] or ["a", "b", "c"]


def run(s, cfg, passes, seed, stop=1e-20):
    t0 = time.perf_counter()
    r = C.run_one({"spec": s, "method": "vr", "cfg": cfg, "passes": passes, "seed": seed, "stop_rel": stop})
    p, rel = np.array(r["trace"]["passes"]), np.array(r["trace"]["rel"])
    return r, p, rel, time.perf_counter() - t0


def line(label, r, p, rel, wall, marks):
    at = "  ".join(f"{q:g}: {rel[np.argmin(np.abs(p - q))]:.1e}" for q in marks if q <= p[-1] + 0.5)
    print(f"  {label:44s} passes to 1e-6: {r['passes_to']['1e-06']:6.1f}   final {r['final_rel']:.1e} at pass "
          f"{r['passes_used']:.0f}   diverged={r['diverged']}   [{at}]  ({wall:.0f} s)", flush=True)


if "a" in parts:
    for loss in ("softmax", "ls"):
        s = C.spec(loss, 1000, 50, 10, 1e4, 1.0)
        prob = C.problem(s)
        print(f"\n(a) {loss}, kappa/n = 10 (instance of exp_M6): R2 = {prob.R2:.3f}, lam = {prob.lam:.2e}, 40 passes, seed 0")
        line("MISO (delta = 1)", *run(s, dict(estimator="saga", placement="euclid", miso_delta=1.0), 40, 0), (10, 20, 40))
        for c in (1.0, 0.25, 1 / 16, 1 / 64):
            line(f"joint SAGA-Muon, gamma = {c:g} / R2_op", *run(s, dict(estimator="saga", gamma=over("R2_op", c)), 40, 0),
                 (10, 20, 40))
        line("SAGA (Euclidean), gamma = 0.25 / R2", *run(s, dict(estimator="saga", placement="euclid",
                                                               gamma=over("R2", 0.25)), 40, 0), (10, 20, 40))

if "b" in parts:
    for key, s, c in (("soft-cond", C.spec("softmax", 200, 40, 8, 1e4, 2.0), 0.5),
                      ("ls-cond", C.spec("ls", 200, 40, 8, 1e4, 2.0), 0.125)):
        prob = C.problem(s)
        print(f"\n(b) {key}: n = {prob.n}, R2/lam = {prob.R2 / prob.lam:.0f} = {prob.R2 / prob.lam / prob.n:.1f} n; "
              f"100 passes, seed 10")
        line("MISO (delta = 1)", *run(s, dict(estimator="saga", placement="euclid", miso_delta=1.0), 100, 10),
             (10, 30, 60, 100))
        line(f"joint SAGA-Muon, gamma = {c:g} / R2_op (M1)", *run(s, dict(estimator="saga", gamma=over("R2_op", c)),
                                                                 100, 10), (10, 30, 60, 100))
        line("joint SAGA-Muon, gamma = 0.25 / R2_op", *run(s, dict(estimator="saga", gamma=over("R2_op", 0.25)),
                                                           100, 10), (10, 30, 60, 100))

if "c" in parts:
    s = C.spec("ls", 1000, 50, 10, 1e3, 1.0)
    print("\n(c) ls-mem, SAGA estimator, nuclear-scaled length gamma = 0.25 / R2_op (exp_M3), 110 passes, stop 1e-30")
    for seed in (0, 1):
        r, p, rel, wall = run(s, dict(estimator="saga", gamma=over("R2_op", 0.25)), 110, seed, stop=1e-30)
        line(f"seed {seed}", r, p, rel, wall, (20, 40, 60, 70, 80, 90, 100, 110))
        print(f"      minimum over the run: {rel.min():.2e}; median over the last 10 passes: {np.median(rel[p >= p[-1] - 10]):.2e}")
    s = C.spec("softmax", 1000, 50, 10, 1e3, 1.0)
    print("    soft-mem, same configuration, 80 passes")
    r, p, rel, wall = run(s, dict(estimator="saga", gamma=over("R2_op", 0.25)), 80, 0, stop=1e-30)
    line("seed 0", r, p, rel, wall, (20, 40, 60, 70, 80))
    print(f"      minimum over the run: {rel.min():.2e}; median over the last 10 passes: {np.median(rel[p >= p[-1] - 10]):.2e}")
