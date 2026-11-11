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
μ_eff, L_eff, B3 experimental diagnosis of every inequality of the proof.)

## Layout

| Path | Content |
|---|---|
| `vrlab/` | Python package: problems (logistic regression with controlled n, κ, anisotropy; Nesterov-chain hard instance), high-precision optimum, algorithms (GD, AGD, SGD, SAGA, preconditioned SAGA, L-SVRG, L-Katyusha, naive Nesterov+SVRG), exact instrumentation of the proofs, bounds, gap attribution |
| `tests/test_vrlab.py` | unit tests: derivatives, optimum, exact one-step identities (checked by enumerating all indices), tightness ratios in (0,1], P-metric, Lemma 4.5 of L-Katyusha, convergence of every method |
| `experiments/` | one script per hypothesis (`exp_B3_tightness.py` produces the three key figures of the proposal; `exp_A4_hard_instance.py` the κ sweep on the Lan–Zhou-normalised chains; `exp_A4b_instances.py` the shared-regularizer instance of the A1 note and the aligned instance; `exp_validation_book.py` reproduces the book's Figure 5.4); `run_all.py` runs everything; `--quick` for a smoke test (quick figures get a `_quick` suffix); `make_digest.py` builds `results/digest.md` |
| `results/` | machine-readable summaries (`*_summary.json`), `digest.md` (all numbers), `findings.md` (analysis), cached runs (`*.pkl`: only the small, costly ones are versioned, i.e. A3, A4, A4b, B2, GammaDelta, validation_book; the others are rebuilt by their script) |
| `figures/` | all figures (PNG + PDF) |
| `theory/` | LaTeX notes with full proofs: `B3_prop59_tightness.tex`, `B2_effective_curvature.tex`, `B1_preconditioned_saga.tex`, `A1_lower_bound.tex`, `A3_lkatyusha_lyapunov.tex`; `main.tex` compiles them |
| `bibliography/` | `literature_review.md` (annotated state of the art organised by hypothesis), `references.bib`, `refs.json` |
| `report/` | the final report (`main.tex` → `main.pdf`) |
| `_scratch/` | local working files, not versioned (extracted book text, downloaded papers, transcription of the proof of Prop. 5.9) |

## Conventions

* Notation of the book: F = (1/n) Σ f_i, f'_i = ∇f_i, η* minimiser, each f_i is R²-smooth, F is μ-strongly
  convex, κ = R²/μ. For logistic regression `R2 = max_i (‖x_i‖²/4 + λ)` (the per-sample smoothness constant;
  the book's "features bounded by R" is loose by the factor 4 of the logistic second derivative) and `mu = λ`.
* SAGA memory z_i, Lyapunov V_t = ‖θ_t − η*‖² + Δ Σ_i ‖z_i − f'_i(η*)‖², textbook constants γ = 1/(4R²), Δ = 3γ².
* Gradient accounting: one component gradient = 1, a full gradient = n; SVRG-type methods pay 2 per iteration.
* Every "expectation" used in the tightness analysis is the exact conditional expectation over the sampled index
  given the state (computed in O(nd), or O(n²d) for the Lyapunov function of L-Katyusha), not a Monte-Carlo estimate.
* Trajectories are analysed only above the rounding floors (‖θ−η*‖² ≥ 10⁻¹⁸ for SAGA, Ψ ≥ 10⁻¹⁰ for L-Katyusha).
* Hard instances: `make_hard_instance` (Lan–Zhou normalisation, block-local regularizer, Nesterov chain ζ = 2,
  κ = R²/μ solved exactly, requires κ > n), `make_hard_instance_shared` (the instance of the A1 note: shared
  regularizer, ζ = ζ_Q, closed-form geometric minimizer, any κ > 1) and `make_aligned_instance` (all components
  equal to one chain, L_F = R²). The dimension condition m ≥ ¼√(κ/n)·log(1/ε) (single chain: m ≥ ¼√κ·log(1/ε))
  delimits the window where the exponent ½ is certified optimal; outside it, use κ_eff = R²/λ_min(F″) = `prob.kappa`.

## Running

```bash
pip install -r requirements.txt
python tests/test_vrlab.py                 # or: python -m pytest tests -q
python experiments/run_all.py --quick      # smoke test (a few minutes)
python experiments/run_all.py              # full experiments (about two hours, dominated by the κ sweep)
cd theory && pdflatex main.tex             # theory notes
cd report && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

Cached results are reused: delete `results/<TAG>.pkl` to recompute an experiment.
