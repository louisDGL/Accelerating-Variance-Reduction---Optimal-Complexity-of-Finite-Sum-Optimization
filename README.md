# Accelerating Variance Reduction: Optimal Complexity of Finite-Sum Optimization

Numerical and theoretical pipeline for the Statistical Learning Theory project of Beline Yuan, Paul Ruiz Alliata and
Louis Darrigol (UC Berkeley, Nov. 2026), built on Chapter 5 of F. Bach, *Learning Theory from First Principles*
(Section 5.4.4, Proposition 5.9, the SAGA convergence proof).

The project asks two questions.
**Hypothesis A** — can variance reduction be accelerated à la Nesterov once the momentum is corrected?
(A1 lower bound, A2 naive acceleration, A3 loopless Katyusha with a Lyapunov proof, A4 empirical exponents
κ¹ vs κ^½ on the hard instance.)
**Hypothesis B** — can the variance-reduced algorithm itself be optimised, and which steps of the proof of
Proposition 5.9 are loose? (B1 quadratic-metric / preconditioned SAGA, B2 trajectory-dependent constants
μ_eff, L_eff, B3 experimental diagnosis of every inequality of the proof, B4 sampling order: the ceiling of one nat per
pass set by the memory term belongs to the expectation under sampling with replacement — exact second moments of
i.i.d. and reshuffled SAGA and the exact recurrence of fixed-order SAGA on sums with identical Hessians, in the two
regimes of a fixed step per pass and of a fixed step per iteration; what typical i.i.d. runs do; an effective
curvature for unequal Hessians and the orthogonal extreme; tested on logistic regression and against the lower bounds
on the instance of the A1 note; analysis in `results/findings.md`, Section 9.)

## Layout

| Path | Content |
|---|---|
| `vrlab/` | Python package: problems (logistic regression with controlled n, κ, anisotropy; Nesterov-chain hard instance), high-precision optimum, algorithms (GD, AGD, SGD, SAGA, preconditioned SAGA, L-SVRG, L-Katyusha, naive Nesterov+SVRG; `saga(..., sampling=)` takes the index order `"iid"` (default, unchanged), `"rr"` (fresh permutation every pass), `"so"` (one permutation reused) or `"cyclic"`), exact instrumentation of the proofs, bounds (including the lower bounds of the A1 note on its instance for i.i.d. and for arbitrary index sequences), gap attribution, and `order_model.py` (the solvable model of hypothesis B4: exact second-moment recursion for i.i.d. indices, fixed-order recurrence and its critical step, exact first- and second-moment pass maps of the reshuffled order and their limits at fixed γnλ and at fixed γλ, the effective curvature of a sum with unequal Hessians, the orthogonal-components extreme, the incremental-aggregated-gradient threshold, a heuristic for typical i.i.d. runs, simulator, identical-Hessian problems) |
| `experiments/` | one script per hypothesis (`exp_B3_tightness.py` produces the three key figures of the proposal; `exp_A4_hard_instance.py` the κ sweep on the Lan–Zhou-normalised chains; `exp_A4b_instances.py` the shared-regularizer instance of the A1 note and the aligned instance; `exp_validation_book.py` reproduces the book's Figure 5.4; `check_B2_fp_heuristic.py` the floating-point sanity check of the B2 note; `exp_B4_sampling_order.py` the sampling orders: model, logistic regression, instance of the A1 note, Figure B4; `exp_B4b_refinements.py`, run after it, the two regimes of the model, typical i.i.d. runs for n up to 10⁵, the two-term law on the logistic problems of B4, the orthogonal extreme and every number printed in the B4 note, Figure B4b; both take `VRLAB_WORKERS=k` for k worker processes and need a few minutes with 4); `run_all.py` runs all of them and then `make_digest.py`; `--quick` for a smoke test (quick figures get a `_quick` suffix); `make_digest.py` builds `results/digest.md` |
| `results/` | machine-readable summaries (`*_summary.json`), `digest.md` (all numbers), `findings.md` (analysis), cached runs (`*.pkl`: only the small, costly ones are versioned, i.e. A3, A4, A4b, B2, GammaDelta, validation_book; the others are rebuilt by their script) |
| `figures/` | all figures (PNG + PDF) |
| `theory/` | LaTeX notes with full proofs: `B3_prop59_tightness.tex`, `B2_effective_curvature.tex`, `B1_preconditioned_saga.tex`, `A1_lower_bound.tex`, `A3_lkatyusha_lyapunov.tex`, `B4_sampling_order.tex`; `main.tex` compiles them |
| `bibliography/` | `literature_review.md` (annotated state of the art organised by hypothesis), `references.bib`, `refs.json` |
| `archive/` | earlier deliverables, superseded by the paper and kept as they were (sources and PDFs): `archive/report/`, the final report of the course (`main.pdf`, with the five theory notes as appendices, and `synthesis.pdf`; built by `build.py`, which restates the results of the notes through `make_restated.py`), and `archive/muon/`, the write-ups of the Muon side study (theory notes, condensed paper, short and extended reports); rebuild instructions in `archive/README.md` |
| `paper/` | **the research paper of the project**, `paper_saga_to_muon.tex` / `.bib` / `.pdf` (12 pages): from SAGA to Muon-type steps, built on the main study and on the side study in `muon/`; `paper/figures/` holds its four figures and `paper/scripts/` the scripts that draw them; five explanatory figures form an optional appendix (`paper/appendix_explanatory_figures.tex`, off by default) and `paper/talk/` holds figures for a lecture on the paper with a catalogue of them on slides (see `paper/README.md`, `paper/talk/README.md`) |
| `muon/` | side study: finite-sum variance reduction inside Muon-type (orthogonalized) steps; code in `muon/lab/`, experiments in `muon/experiments/`, results and analysis in `muon/results/` and `muon/findings.md`, earlier write-ups in `muon/archive/` (see `muon/README.md`) |
| `*/bias_law/` | **bias-law study** (where orthogonalized stochastic steps converge), spread over the folders above: `theory/bias_law/` (master theorem document `master.pdf`, shared statements and proofs, verification record, original notes in `notes/`), `paper/bias_law/` (two anonymous papers built from the same sources: `icml/twomoment`, `colt/discrete`; see its `README.md`), `experiments/bias_law/` (checks, exact lattice computations, computer-assisted proof, figure scripts), `results/bias_law/` (logs, plotted data, `NUMBERS_*.md`), `figures/bias_law/`, `bibliography/bias_law_literature_*.md` |
| `external/` | local clones of companion repositories of the team, kept for study (not versioned here; see `external/README.md` in a local copy) |
| `_scratch/` | local working files, not versioned (extracted book text, downloaded papers, transcription of the proof of Prop. 5.9) |
| `_archive/` | local archive, not versioned: unit tests, numerical checks of the archived theory notes, exploratory scripts and maintenance utilities that are not needed to reproduce the report or the paper |

## Conventions

* Notation of the book: F = (1/n) Σ f_i, f'_i = ∇f_i, η* minimiser, each f_i is R²-smooth, F is μ-strongly
  convex, κ = R²/μ. For logistic regression `R2 = max_i (‖x_i‖²/4 + λ)` (the per-sample smoothness constant;
  the book's "features bounded by R" is loose by the factor 4 of the logistic second derivative) and `mu = λ`.
* SAGA memory z_i, Lyapunov V_t = ‖θ_t − η*‖² + Δ Σ_i ‖z_i − f'_i(η*)‖², textbook constants γ = 1/(4R²), Δ = 3γ².
* Gradient accounting: one component gradient = 1, a full gradient = n. SAGA pays n initially and 1 per iteration.
  L-SVRG, L-Katyusha and the naive Nesterov+SVRG scheme (`count_reference_grad=True`, used in all experiments) pay n
  initially, 2 per iteration (f'_i at the query point and f'_i(w), charged as recomputed although the code stores the
  table of the f'_i(w)) and n per refresh of the reference point (n·p = 1 per iteration in expectation): 3 per
  iteration in expectation. The bounds of the A1 note assume stored reference gradients (1 + n·1[refresh], 2 per
  iteration in expectation), so measured counts are about 3/2 of stored-convention counts; slopes in κ are
  unaffected.
* Every "expectation" used in the tightness analysis is the exact conditional expectation over the sampled index
  given the state (computed in O(nd), or O(n²d) for the Lyapunov function of L-Katyusha), not a Monte-Carlo estimate.
* Trajectories are analysed only above the rounding floors: 10⁻¹⁸ on the Lyapunov value V_t for the SAGA
  instrumentation (B3: every attribution window and Figure 2; the B1 windows, passes 2–20, stay above it) and on
  ‖θ−η*‖² in B2, Ψ ≥ 10⁻¹⁰ for L-Katyusha.
* Hard instances: `make_hard_instance` (Lan–Zhou normalisation, block-local regularizer, Nesterov chain ζ = 2,
  κ = R²/μ solved exactly, requires κ > n), `make_hard_instance_shared` (the instance of the A1 note: shared
  regularizer, ζ = ζ_Q, closed-form geometric minimizer, any κ > 1) and `make_aligned_instance` (all components
  equal to one chain, L_F = R²). The dimension condition m ≥ ¼√(κ/n)·log(1/ε) (single chain: m ≥ ¼√κ·log(1/ε))
  delimits the window where the exponent ½ is certified optimal; outside it, use κ_eff = R²/λ_min(F″) = `prob.kappa`.
* Complexity exponents: least-squares slope of log(gradients) against log(κ), fitted to the per-κ geometric means
  over the seeds (κ ≥ 10n), reported as `slope ± h` with h the half-width of the 95% t-interval with K − 2 degrees
  of freedom (`vrlab.analysis.loglog_slope`). The seeds are not used as independent replicates (GD and AGD are
  deterministic on a fixed instance), so h measures the departure from a single power law over the window, not
  sampling noise.

## Running

```bash
pip install -r requirements.txt
python experiments/run_all.py --quick      # smoke test (a few minutes)
python experiments/run_all.py              # full experiments (many CPU hours, dominated by the κ sweeps A4/A4b)
cd theory && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex   # theory notes
cd paper && pdflatex paper_saga_to_muon && bibtex paper_saga_to_muon && pdflatex paper_saga_to_muon && pdflatex paper_saga_to_muon   # paper
```

Cached results are reused: delete `results/<TAG>.pkl` to recompute an experiment.

The two B4 experiments run on `VRLAB_WORKERS` worker processes (default 1; the results do not depend on it) and refuse
a cache computed with another configuration. Their caches `results/B4.pkl` and `results/B4b.pkl` are not versioned and are
rebuilt by the scripts (about 12 and 8 minutes with 4 workers and single-threaded BLAS; the run times are not recorded in
the summaries). `exp_B4b_refinements.py` reads `results/B4_summary.json`: run it after `exp_B4_sampling_order.py`
(`run_all.py` does):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VRLAB_WORKERS=4 python experiments/exp_B4_sampling_order.py --quick   # smoke test
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VRLAB_WORKERS=4 python experiments/exp_B4_sampling_order.py           # results/B4_summary.json, figures/figB4_sampling_order.*
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VRLAB_WORKERS=4 python experiments/exp_B4b_refinements.py             # results/B4b_summary.json, figures/figB4b_refinements.*
```

The B4 results were produced with numpy 1.22.4 and scipy 1.9.3 (older than the versions of `requirements.txt`).

Each solver run of the κ sweeps is capped by `VRLAB_MAX_TIME` seconds of **CPU time** (`time.process_time`, so a
suspended machine consumes no budget; default 600). The reported sweeps were run with single-threaded BLAS and a
4000 s cap (8000 s for the A4b runs completed last, at κ = 10⁶ on the shared-regularizer instance and on the aligned
instance); no reported run reached its cap (statuses in `results/A4.pkl`, `results/A4b.pkl`):

```bash
python experiments/retry_timeouts.py A4 A4b   # drop the entries that hit the cap
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VRLAB_MAX_TIME=4000 python experiments/exp_A4_hard_instance.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VRLAB_MAX_TIME=4000 python experiments/exp_A4b_instances.py
```
