# Variance reduction for Muon-type (orthogonalized) methods on matrix finite sums

Side study of the project "Accelerating Variance Reduction: Optimal Complexity of Finite-Sum Optimization": to what
extent can finite-sum variance reduction (SAGA, SVRG, PAGE, Katyusha) be applied to Muon, the optimizer that replaces a
matrix gradient by its polar factor? The study keeps the standards of the main project: exact optimum, exact
constants, gradient accounting, multi-seed runs with a tune-then-confirm protocol, and exact conditional expectations
obtained by enumerating the sampled index.

**Results and discussion: [`findings.md`](findings.md).** This file describes the code.

## Layout

| Path | Content |
|---|---|
| `lab/problems.py` | l2-regularized softmax regression and multi-output least squares with a matrix parameter W in R^{K x d}; features generated as in `vrlab` (controlled n, kappa, spectrum decay alpha); exact Euclidean and spectral-norm constants; accurate `F(W1) - F(W0)`; `planted='orthogonal'` gives the instance where the polar factor of the full gradient points at the minimizer |
| `lab/optimum.py` | high-precision minimizer (closed form for least squares, damped Newton + polish for softmax) |
| `lab/polar.py` | exact polar factor (SVD), Keller Jordan's quintic Newton-Schulz (`ns5`; its normalization constant `eps` sets a floor under a cooldown, fact F14), cubic Newton-Schulz, bfloat16 emulation; two control maps that are *not* orthogonalizations, the Frobenius normalization (`frob`) and the coordinatewise sign (`sign`, gauge-projected for softmax), used by M9 |
| `lab/maps.py` | spectral maps and their Frechet derivatives: `sharp` (nuclear-norm-scaled polar factor, the dual map of the spectral norm), `soft_polar` (bounded gain), `SpectralDerivative` (derivative of the polar factor, of `sharp`, of any generalized matrix function), softmax gauge basis |
| `lab/algorithms.py` | GD, Nesterov's accelerated gradient, L-BFGS (scipy), full-gradient Muon / PolarGrad, SGD, Muon (Keller Jordan), STORM-Muon, SAGA, L-SVRG, and the first two orthogonalized variance-reduced methods (`saga_muon`, `lsvrg_muon`) |
| `lab/vrmuon.py` | **generalized variance-reduced Muon**: estimator (SAGA, L-SVRG, PAGE, minibatch) x placement of the nonlinear map (joint, split, tangent, radial, frozen polar metric, Euclidean) x step-length rule (nuclear-scaled, fixed, clipped) x orthogonalization (exact, Newton-Schulz, bounded gain), with damping, minibatches, importance sampling of the index, momentum before or after the map, lazy factorization at the SVRG reference point, anchor PolarGrad steps, relaxed memory (MISO / dual-free SDCA as members of the family); for experiment M9, all off by default: sampling order of the batches (`order`), geometric cooldown of a fixed length (`schedule='geometric'`, `lr_decay`), control maps (`orth='frob'`, `'sign'`), extra refreshes of the SAGA table (`refresh_extra`), see "Options of experiment M9" below. The metric placement takes its metric from the polar decomposition of a gradient (refreshed, or computed once: "one-shot") or, as a control, from the feature covariance (`metric_source='cov'`, the preconditioned SAGA of the main project). Options that a placement does not use are refused, not ignored |
| `lab/exact.py` | exact conditional expectations of one iteration (enumeration of the n indices): expected decrease of the distance and of the objective, bias and alignment of the mean step, estimator diagnostics; reference full-gradient steps |
| `lab/mvr.py` | momentum-based variance reduction for Muon, the published state of the art (Nesterov / MARS-M / STORM in one parameterization), with fixed-length or nuclear-scaled steps |
| `lab/katyusha.py` | loopless Katyusha on matrix problems, with the spectral (nuclear-scaled polar) step on the y sequence |
| `lab/instrumentation.py` | effective constants in the Euclidean and spectral geometries at a state |
| `experiments/common.py` | protocol: budgets in passes, tune-then-confirm (screening, tuning seeds, refinement of the finalists, fresh confirmation seeds), parallel runner with an incremental cache, code fingerprint |
| `experiments/families.py` | catalogue of the compared method families and their tuning grids |
| `experiments/exp_M1_headtohead.py` | M1: all families at tuned steps on six instances (memory-limited, conditioning-limited, Muon-favourable) |
| `experiments/exp_M2_mechanism.py` | M2: exact one-step statistics along the trajectories (own states and common states) |
| `experiments/exp_M3_floors.py` | M3: which ingredient removes which error floor (length rule x estimator) |
| `experiments/exp_M4_sensitivity.py` | M4: sensitivity of the trajectories to a rounding-size perturbation (script archived locally in `_archive/`, not versioned; its summary `results/M4_summary.json` is kept) |
| `experiments/exp_M5_batch.py` | M5: batch size of the variance-reduced estimator, Euclidean versus orthogonalized (factor-2 step grids; superseded by M5b where the two differ) |
| `experiments/exp_M5b_batch_fine.py` | M5b: M5 at resolved steps (grids of ratio 2^(1/4) centred on M5's steps), with a factor-2 control that reproduces M5, a grid shifted by half a step, the step window of the polar step, and two other draws of the data; needs `M5_summary.json`; `--tables` prints the tables of the existing summary |
| `experiments/exp_M6_miso.py` | M6: the joint step as an implicit MISO iteration (Euclidean counterparts, relaxed table) |
| `experiments/exp_M7_checks.py` | M7: conditioning of the problem in the frozen polar metric (local and global constants), weight of the y step in Katyusha-Muon, spread of the per-sample constants |
| `experiments/exp_M8_dataseeds.py` | M8: the main comparison on other draws of the data (the four conditioning-limited instances) |
| `experiments/exp_M9_schedule.py` | M9: Muon's own recipe with a SAGA table (fixed-length step, geometric cooldown lr_0 q^passes). A: maps (polar, `ns5`, and the controls `frob`, `sign`; with and without Nesterov) by batch size, with same-batch references on 2^(1/4) grids; B: batch one on the six instances and at one fixed point; C: the schedule law (depth, floor) and the controls without a table; D: sampling order and memory of the table. `--parts`, `--batches`, `--only`; the summary is updated part by part |
| `experiments/exp_M9b_order_model.py` | M9b: the sampling-order part of M9 (part D) on a sum with identical Hessians, the solvable model of hypothesis B4 of the main study, where the polar factor of the estimator is its normalization. D: the design and the code of part D on a model instance built for the lab (thresholds next to those of `soft-mem` and `ls-mem`); L: long horizons at batch one with a simulator in error coordinates (`model_run`, checked step by step against the lab), n = 100 to 10^4 |
| `experiments/audit_cache.py` | recomputes a random sample of cached runs with the current code and compares (archived locally in `_archive/`, not versioned) |
| `experiments/make_digest.py` | `results/digest.md`: every table of the study, generated from the JSON summaries |
| `experiments/make_figures.py` | figures from the JSON summaries |
| `notes_theory.md` | the structural facts F1-F16 with proofs (F12-F16, added 2026-11-19: polar steps fed by a SAGA table: staleness lemma without rank factor, recursion / floor / certified cooldown of fixed-length steps, scale covariance on least squares, a Lyapunov function for the nuclear-scaled step, the frozen polar factor of the SAG direction) |
| `literature_check.md` | record of the second prior-art search of 2026-11-12 (verdict, closest work, what was read and the queries, per configuration); long, generated from the reports of the search agents; at the end, an addendum for the check of 2026-11-18 on the additions of M9, M5b and F12-F16 (verdict "close") |
| `../archive/muon/notes_spectral_theory.tex` / `.pdf` | theory notes of 2026-11-16 (condition numbers in the spectral geometry, polar steps as mirror descent on the nuclear norm of the gradient, the least-squares model, stochastic polar steps and the nuclear median), archived on 2026-11-20 without update; the paper cites them for the proofs of its Theorems 1-2 and Propositions 3-4 |
| `../paper/` | the research paper of the study, now at the root of the project (`paper/paper_saga_to_muon.pdf`, 12 pages, revised and condensed on 2026-11-20; see `paper/README.md`) |
| `../archive/muon/` | earlier write-ups of the study, superseded by the paper and kept as they were (PDFs included): the theory notes `notes_spectral_theory`, the condensed paper `paper_spectral_muon`, the short report `report_spectral_muon` and the extended report `report_spectral_muon_new`; see [`../archive/README.md`](../archive/README.md) |
| `theory_checks/` | the two numerical checks whose numbers the paper quotes (`check_offset_first_order.py`: the constant of the minibatch offset; `check_minibatch_offset_exact.py`: the offset against the batch size) and `make_paper_figures.py`, which draws the figures of the archived notes (`figures/figT1_ls_model`, `figT2_noise`, `figT2s_noise`, `figR3_theory`) and computes the cache `paper_figures_data.json` read by figure 3 of the paper; the other checks of the notes are archived locally in `_archive/` |
| `results/` | `*_summary.json` (all reported numbers), caches `*.pkl` (rebuilt by the scripts) |
| `figures/` | figures (PNG + PDF) |

## Conventions

* Problem: F(W) = (1/n) sum_i l(W x_i; y_i) + (lam/2) ||W||_F^2, W in R^{K x d}; per-sample loss gradients are rank
  one, g_i'(W) = r_i x_i^T. `R2 = max_i` per-sample Euclidean smoothness, `mu = lam`, `kappa = R2 / mu` (set by the
  generators); `R2_op`, `L_op` are the smoothness constants with respect to the spectral norm; `L_F` the Euclidean
  smoothness of F. The true strong convexity at the optimum can be much larger than `lam` (the experiments report
  `R2 / mu_true`, which decides whether a run is limited by its memory or by the conditioning).
* Gradient accounting: one per-sample gradient = 1, a full gradient = n. SAGA: n + b per iteration (b + m per iteration
  with m extra refreshes of the table, `refresh_extra`). L-SVRG, PAGE,
  L-Katyusha: n + 2b per iteration + n per refresh (the convention of the main project; with the residuals stored at
  the reference point, as the implementation does, one could charge b per iteration: about 2/3 of the passes).
  L-BFGS: n per evaluation of (F, grad F). A *pass* is n per-sample gradient evaluations.
* Variance-reduced estimators are split into a predictable part `m` (memory mean + lam W) and an innovation `c`
  (depends on the sampled batch); the regularizer gradient is always exact.
* Softmax gauge: all factorizations are done in the subspace orthogonal to the all-ones vector (`maps.gauge_basis`);
  iterates start at W_0 = 0 (a starting point outside the gauge subspace is refused by the mapped placements).
* Step parameters are expressed in problem units: `gamma = c / R2_op` for the orthogonalized methods, `c / R2` for
  the Euclidean ones.
* Randomness: algorithm seeds drive the sampled indices; the instance has its own seed. Trajectories of the methods
  that use the exact polar factor are sensitive to rounding (experiment M4), so individual runs are reproducible only
  with the same BLAS threading; the experiments run with single-threaded BLAS and report medians and ranges over
  seeds.

## Running

```bash
python muon/experiments/exp_M1_headtohead.py --quick      # smoke test (a few minutes)
python muon/experiments/exp_M1_headtohead.py              # full comparison (several hours on 4 cores from scratch)
python muon/experiments/exp_M2_mechanism.py               # needs M1_summary.json
python muon/experiments/exp_M3_floors.py
python muon/experiments/exp_M5_batch.py
python muon/experiments/exp_M5b_batch_fine.py             # needs M5_summary.json; about 40 minutes on 3 cores from scratch
python muon/experiments/exp_M6_miso.py
python muon/experiments/exp_M7_checks.py                  # needs M1_summary.json
python muon/experiments/exp_M8_dataseeds.py
python muon/experiments/exp_M9_schedule.py                # about one hour on 5 cores from scratch (see its docstring)
python muon/experiments/exp_M9b_order_model.py            # reads M9_summary.json; about 15 minutes with 6 processes
python muon/experiments/make_digest.py && python muon/experiments/make_figures.py
python muon/experiments/make_figures.py M5b M9 M9b        # only the three figures added on 2026-11-19
python paper/scripts/make_figP4_recipe.py                 # figure 4 of the paper (see paper/README.md)
```

Use the project virtual environment (`.venv`). `MUON_JOBS` sets the number of worker processes (default 4). Cached
runs are reused (the cache is keyed by the task, not by the code: delete `muon/results/<name>.pkl` to recompute, or
run `audit_cache.py`, archived locally; every summary records the fingerprint of the code that wrote it). The grids of
`experiments/families.py` were widened and refined in several passes; the summaries of the earlier passes are kept as
`results/M1_pass*_summary.json` for the record and are superseded by `results/M1_summary.json`.

## Options of experiment M9

Added on 2026-11-19 to `lab/vrmuon.py` (and to the task dictionaries of `experiments/common.run_one`). All are off
by default, and the default behaviour is unchanged (checked by golden-value tests, archived locally in `_archive/`). An option that a configuration does not
use is refused with `ValueError`.

| option | values | meaning |
|---|---|---|
| `schedule='geometric'`, `lr_decay=q` | q in (0, 1] | fixed-length rule only: lr_t = lr q^(t b / n), the learning rate loses the factor q per pass; the schedule does not depend on the length of the run, and `warmup` is refused. For the loops of `lab/algorithms.py` and `lab/mvr.py`: `schedule=('geometric', q, period)` |
| `order` | `'iid'` (default), `'reshuffle'`, `'shuffle_once'` | sampling of the batches for the SAGA and minibatch estimators: independent, a fresh permutation per pass, one permutation read cyclically |
| `orth` | `'frob'`, `'sign'` | control maps (not orthogonalizations): M / \|\|M\|\|_F and the coordinatewise sign; learning rates are not comparable across maps |
| `orth_kw={'eps': ...}` | e.g. `1e-30` | normalization constant of `ns5` (default 1e-7, which sets a floor of the relative squared distance near 1e-18 under a cooldown, M9 part C) |
| `refresh_extra=m` | integer >= 0 | SAGA table: m entries outside the batch are refreshed at the new iterate after each step, and charged (b + m gradients per iteration) |

```python
# Muon's recipe with a SAGA table: fixed length, Newton-Schulz, Nesterov 0.9, lr_t = 0.1 * 0.8^passes
res = vrmuon.vr_muon(prob, W0, 60 * prob.n, estimator="saga", length="const", lr=0.1, schedule="geometric",
                     lr_decay=0.8, orth="ns5", beta=0.9, nesterov=True)
```

## Minimal example

```python
import sys; sys.path.insert(0, "muon")
import numpy as np
from lab import problems, optimum, vrmuon, exact

prob = problems.make_softmax(n=1000, d=50, K=10, kappa=1e3, alpha=1.0, seed=0)
optimum.solve_optimum(prob)
W0 = np.zeros((prob.K, prob.d))

# Euclidean SAGA, the joint orthogonalized step, and the split placement
saga = vrmuon.vr_muon(prob, W0, 30 * prob.n, estimator="saga", placement="euclid", gamma=1.0 / prob.R2)
joint = vrmuon.vr_muon(prob, W0, 30 * prob.n, estimator="saga", gamma=0.25 / prob.R2_op)
split = vrmuon.vr_muon(prob, W0, 30 * prob.n, estimator="saga", placement="split",
                       gamma=1.0 / prob.R2_op, gamma_c=0.25 / prob.R2)
print(saga["dist2_F"][-1], joint["dist2_F"][-1], split["dist2_F"][-1])

# exact conditional expectations of the next step at a state of the SAGA trajectory
t, W, st = exact.collect_states(prob, W0, 5 * prob.n, [5 * prob.n], estimator="saga", placement="euclid",
                                gamma=1.0 / prob.R2)[0]
stats = exact.one_step_stats(prob, vrmuon.VRMuon(prob, estimator="saga", gamma=0.25 / prob.R2_op), W, st)
print(stats["rel_dec_dist2"], stats["noise_nuc"], stats["dual_align"])
```
