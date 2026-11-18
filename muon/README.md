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
| `lab/polar.py` | exact polar factor (SVD), Keller Jordan's quintic Newton-Schulz (`ns5`), cubic Newton-Schulz, bfloat16 emulation |
| `lab/maps.py` | spectral maps and their Frechet derivatives: `sharp` (nuclear-norm-scaled polar factor, the dual map of the spectral norm), `soft_polar` (bounded gain), `SpectralDerivative` (derivative of the polar factor, of `sharp`, of any generalized matrix function), softmax gauge basis |
| `lab/algorithms.py` | GD, Nesterov's accelerated gradient, L-BFGS (scipy), full-gradient Muon / PolarGrad, SGD, Muon (Keller Jordan), STORM-Muon, SAGA, L-SVRG, and the first two orthogonalized variance-reduced methods (`saga_muon`, `lsvrg_muon`) |
| `lab/vrmuon.py` | **generalized variance-reduced Muon**: estimator (SAGA, L-SVRG, PAGE, minibatch) x placement of the nonlinear map (joint, split, tangent, radial, frozen polar metric, Euclidean) x step-length rule (nuclear-scaled, fixed, clipped) x orthogonalization (exact, Newton-Schulz, bounded gain), with damping, minibatches, importance sampling of the index, momentum before or after the map, lazy factorization at the SVRG reference point, anchor PolarGrad steps, relaxed memory (MISO / dual-free SDCA as members of the family). The metric placement takes its metric from the polar decomposition of a gradient (refreshed, or computed once: "one-shot") or, as a control, from the feature covariance (`metric_source='cov'`, the preconditioned SAGA of the main project). Options that a placement does not use are refused, not ignored |
| `lab/exact.py` | exact conditional expectations of one iteration (enumeration of the n indices): expected decrease of the distance and of the objective, bias and alignment of the mean step, estimator diagnostics; reference full-gradient steps |
| `lab/mvr.py` | momentum-based variance reduction for Muon, the published state of the art (Nesterov / MARS-M / STORM in one parameterization), with fixed-length or nuclear-scaled steps |
| `lab/katyusha.py` | loopless Katyusha on matrix problems, with the spectral (nuclear-scaled polar) step on the y sequence |
| `lab/instrumentation.py` | effective constants in the Euclidean and spectral geometries at a state |
| `tests/` | unit tests (`test_lab.py`, `test_maps.py`, `test_vrmuon.py`, `test_placements.py`, `test_mvr.py`, `test_katyusha.py`, `test_facts.py`, `test_protocol.py`): derivatives, constants, Frechet derivatives against finite differences, unbiasedness and exact step expectations by enumeration, reproduction of the earlier implementations, gradient accounting, convergence, argument validation, the facts F1-F11 of `notes_theory.md` by exact enumeration, the experiment protocol |
| `experiments/common.py` | protocol: budgets in passes, tune-then-confirm (screening, tuning seeds, refinement of the finalists, fresh confirmation seeds), parallel runner with an incremental cache, code fingerprint |
| `experiments/families.py` | catalogue of the compared method families and their tuning grids |
| `experiments/exp_M1_headtohead.py` | M1: all families at tuned steps on six instances (memory-limited, conditioning-limited, Muon-favourable) |
| `experiments/exp_M2_mechanism.py` | M2: exact one-step statistics along the trajectories (own states and common states) |
| `experiments/exp_M3_floors.py` | M3: which ingredient removes which error floor (length rule x estimator) |
| `experiments/exp_M4_sensitivity.py` | M4: sensitivity of the trajectories to a rounding-size perturbation |
| `experiments/exp_M5_batch.py` | M5: batch size of the variance-reduced estimator, Euclidean versus orthogonalized |
| `experiments/exp_M6_miso.py` | M6: the joint step as an implicit MISO iteration (Euclidean counterparts, relaxed table) |
| `experiments/exp_M7_checks.py` | M7: conditioning of the problem in the frozen polar metric (local and global constants), weight of the y step in Katyusha-Muon, spread of the per-sample constants |
| `experiments/exp_M8_dataseeds.py` | M8: the main comparison on other draws of the data (the four conditioning-limited instances) |
| `experiments/audit_cache.py` | recomputes a random sample of cached runs with the current code and compares |
| `experiments/make_digest.py` | `results/digest.md`: every table of the study, generated from the JSON summaries |
| `experiments/make_figures.py` | figures from the JSON summaries |
| `notes_theory.md` | the structural facts F1-F11 with proofs |
| `literature_check.md` | record of the second prior-art search of 2026-11-12 (verdict, closest work, what was read and the queries, per configuration); long, generated from the reports of the search agents |
| `notes_spectral_theory.tex` / `.pdf` | theory notes (2026-11-16, revised after a six-referee review): condition numbers in the spectral geometry, polar steps as mirror descent on the nuclear norm of the gradient, the least-squares model (published dynamics and additions), stochastic polar steps and the nuclear median; every statement carries its status (proved / known / formal / numerical) and Appendix B records the review |
| `paper_spectral_muon.tex` / `.bib` / `.pdf` | condensed version of the notes in the format of a short paper, with only the main findings (build: `pdflatex`, `bibtex`, `pdflatex` twice, from `muon/`) |
| `report_spectral_muon.tex` / `.pdf` | short report (8 pages, references included) with the main findings of the theory only: no norm improves the worst-case condition number of ridge-regularized classification models, stochastic polar steps converge to a nuclear median of the gradient noise (least-distances limit for batch one, formal `1/b` offset, fixed point of variance-reduced estimators), three complements on the least-squares dynamics; same build, shares `paper_spectral_muon.bib` |
| `report_spectral_muon_new.tex` / `.pdf` | extended short report (about 9 pages + 2 of references, 2026-11-17) covering the whole project, structured as a research paper: Introduction (with Table 1, what was published and what the project adds), Methodology (problems and algorithms, measurement of a proof along trajectories, protocol of the Muon experiments, analysis of stochastic polar steps, status of the claims), Results (classical findings on SAGA and loopless Katyusha from the main report, variance reduction inside Muon-type steps, the two theorems), Discussion (the line of reasoning from SAGA to Muon-type steps with Table 2, relation to prior work, limitations, conclusion); bibliography `report_spectral_muon.bib`, generated by `_scratch/muon_checks/make_report_bib.py muon/report_spectral_muon_new.tex` from `paper_spectral_muon.bib` and `bibliography/references.bib`; build: `pdflatex`, `bibtex`, `pdflatex` twice, from `muon/` |
| `paper_saga_to_muon.tex` / `.bib` / `.pdf` | research paper (12 pages: about 10.5 of text + references, 2026-11-17) on the two main findings only, with the SAGA-to-Muon link made explicit (Table 2): (1) certified rates lose at global constants (SAGA measured, also on two other data designs, with the constants of the proof, the memory regime, the refuted alignment prediction and loopless Katyusha; the same exact anatomy for the nuclear-scaled polar step), and no norm repairs the logistic worst case (no free lunch for every norm; certified vs local constants on the benchmark; Euclidean devices match orthogonalization); (2) through the polar map, variance reduction acts on the fixed point (with additive noise and deterministic decreasing lengths, the gradient of stochastic polar steps converges to minus a nuclear median of the noise; finite-sum estimators have the minimizer as only fixed point; offset formally of order 1/b; floors and MISO). Reviewed by four referees, each verified (workflow of 2026-11-17); the text is the reviewed snapshot plus the merged fix list `_scratch/muon_checks/paper2_fixes.py` (verified fixes and SAGA additions; collisions resolved there). Bibliography generated by `_scratch/muon_checks/make_report_bib.py muon/paper_saga_to_muon.tex --out muon/paper_saga_to_muon.bib`; build as above |
| `report_figures/` | `make_figR1_classical.py` (SAGA: certified and a-posteriori bounds, exact/certified exponent, attribution of the gap; from `results/B3.pkl` and `results/B3_summary.json` of the main project) and `make_figR2_muon_vr.py` (floors, SAGA-Muon against MISO, tuned passes; from `results/M3_summary.json`, `M6_summary.json`, `M1_summary.json`, checked against `results/digest.md`): the figures `figures/figR1_classical` and `figures/figR2_muon_vr` of the extended report; `make_figP2_geometry.py` (certified and local per-sample condition numbers of three fixed metrics on the six instances, from `results/M7_summary.json` checked against the M7 tables of `results/digest.md`, and the tuned passes of figR2) and `make_figP3_noise.py` (floors, batch-one least squares, b x offset, reusing the panels of figR2 and of `theory_checks/make_paper_figures.py`): the figures `figures/figP2_geometry` and `figures/figP3_noise` of `paper_saga_to_muon` |
| `theory_checks/` | numerical checks of the notes, one script per group of statements (listed in Appendix A of the notes); `make_paper_figures.py` draws `figures/figT1_ls_model` and `figures/figT2_noise` from the cache `paper_figures_data.json` (and, with the argument `T2s`, the two-panel `figures/figT2s_noise` of the short reports; with `R3`, the three-panel variant `figures/figR3_theory`, not used) |
| `results/` | `*_summary.json` (all reported numbers), caches `*.pkl` (rebuilt by the scripts) |
| `figures/` | figures (PNG + PDF) |

## Conventions

* Problem: F(W) = (1/n) sum_i l(W x_i; y_i) + (lam/2) ||W||_F^2, W in R^{K x d}; per-sample loss gradients are rank
  one, g_i'(W) = r_i x_i^T. `R2 = max_i` per-sample Euclidean smoothness, `mu = lam`, `kappa = R2 / mu` (set by the
  generators); `R2_op`, `L_op` are the smoothness constants with respect to the spectral norm; `L_F` the Euclidean
  smoothness of F. The true strong convexity at the optimum can be much larger than `lam` (the experiments report
  `R2 / mu_true`, which decides whether a run is limited by its memory or by the conditioning).
* Gradient accounting: one per-sample gradient = 1, a full gradient = n. SAGA: n + b per iteration. L-SVRG, PAGE,
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
python -m pytest muon/tests -q                            # 8 files, about one minute (each file also runs alone)
python muon/experiments/exp_M1_headtohead.py --quick      # smoke test (a few minutes)
python muon/experiments/exp_M1_headtohead.py              # full comparison (several hours on 4 cores from scratch)
python muon/experiments/exp_M2_mechanism.py               # needs M1_summary.json
python muon/experiments/exp_M3_floors.py
python muon/experiments/exp_M4_sensitivity.py
python muon/experiments/exp_M5_batch.py
python muon/experiments/exp_M6_miso.py
python muon/experiments/exp_M7_checks.py                  # needs M1_summary.json
python muon/experiments/exp_M8_dataseeds.py
python muon/experiments/audit_cache.py                    # optional: recompute a sample of the cached runs
python muon/experiments/make_digest.py && python muon/experiments/make_figures.py
```

Use the project virtual environment (`.venv`). `MUON_JOBS` sets the number of worker processes (default 4). Cached
runs are reused (the cache is keyed by the task, not by the code: delete `muon/results/<name>.pkl` to recompute, or
run `audit_cache.py`; every summary records the fingerprint of the code that wrote it). The grids of
`experiments/families.py` were widened and refined in several passes; the summaries of the earlier passes are kept as
`results/M1_pass*_summary.json` for the record and are superseded by `results/M1_summary.json`.

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
