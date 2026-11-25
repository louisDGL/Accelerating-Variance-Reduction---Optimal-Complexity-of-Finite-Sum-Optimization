# The paper of the project

**From SAGA to Muon-type steps: no norm repairs the logistic worst case, and variance reduction acts on the fixed
point** (`paper_saga_to_muon.pdf`, 12 pages, references included; 17 November 2026, revised and condensed on 20 November
2026).

The paper follows the two properties behind the linear rate of SAGA, a geometry (global constants) and an estimator
(unbiased, with an error that vanishes at the minimizer), from SAGA to steps along the polar factor of a matrix
gradient. Finding 1: certified rates lose at the global constants, and no norm repairs the worst case of ridge logistic
and softmax models. Finding 2: through the polar map, variance reduction acts on the fixed point. Section 3.3: Muon's
own recipe (fixed lengths on a geometric cooldown of the learning rate) fed by a SAGA table converges to rounding error
where minibatch and momentum-based estimators freeze; with minibatches the spectral map pays; a fresh permutation per
pass lifts the memory rate of the table.

## Layout

| Path | Content |
|---|---|
| `paper_saga_to_muon.tex` | the source (one file) |
| `paper_saga_to_muon.bib` | its bibliography (55 entries, all cited) |
| `paper_saga_to_muon.pdf` | the compiled paper |
| `figures/` | the four figures of the paper, `figR1`, `figP2`-`figP4` (PDF, included by LaTeX, and PNG), and five explanatory figures `figE1`-`figE5` for its optional appendix |
| `scripts/` | the scripts that draw them; each reads JSON summaries, prints every number it plots and, for figures 2 and 3, stops if a number disagrees with `muon/results/digest.md`; `make_figE*.py` and `make_figT*.py` draw the explanatory figures with the shared style `explain_style.py` |
| `appendix_explanatory_figures.tex` | an appendix of five explanatory figures, off by default (the paper keeps its 12 pages); see "Explanatory figures" below |
| `paper_saga_to_muon_with_appendix.pdf` | the paper followed by that appendix (15 pages) |
| `talk/` | figures for a lecture on the paper, not meant for the paper, and a catalogue of them on 16:9 slides (see `talk/README.md`) |

| Figure | File | Script | Data |
|---|---|---|---|
| 1 | `figures/figR1_classical` | `scripts/make_figR1_classical.py` | `results/B3_summary.json` and `results/B3.pkl` of the main study (the pickle is not versioned: `python experiments/exp_B3_tightness.py` rebuilds it) |
| 2 | `figures/figP2_geometry` | `scripts/make_figP2_geometry.py` | `muon/results/M7_summary.json`, `M1_summary.json` (M3, M6 for the digest check) |
| 3 | `figures/figP3_noise` | `scripts/make_figP3_noise.py` | `muon/results/M3_summary.json`, `M6_summary.json`, `muon/theory_checks/paper_figures_data.json` |
| 4 | `figures/figP4_recipe` | `scripts/make_figP4_recipe.py` | `muon/results/M9_summary.json`, `M5b_summary.json`, `M9b_summary.json` and the cache `muon/results/M9_law.pkl` (part C of M9; rebuilt by `python muon/experiments/exp_M9_schedule.py --parts C`) |

`scripts/make_figR2_muon_vr.py` holds functions used by the scripts of figures 2 and 3; run on its own it draws
`muon/figures/figR2_muon_vr`, the figure of the archived extended report.

## Build

```bash
cd paper
pdflatex paper_saga_to_muon && bibtex paper_saga_to_muon && pdflatex paper_saga_to_muon && pdflatex paper_saga_to_muon
```

The figures are redrawn from the project root with `python paper/scripts/make_fig<name>.py` (single-threaded BLAS, as
everywhere in the project).

## Explanatory figures

Added on 2026-11-25. `figE2` to `figE4` add no result: `figE2` and `figE3` illustrate Theorem 1, Theorem 2 and
Proposition 3 on exact instances in the plane (in `figE3` the polar factor of a vector is its normalization), and
`figE4` lays out the methods, with one random matrix; the numbers they show are those of these toys. `figE1` lays out
the proof of Section 2.2 and recuts the measured shares of Figure 1(c), with those of a second order of attribution
computed on the same runs. `figE5` redraws cached runs of experiment M9 (part C) and adds an exact enumeration along
one of them and a cached configuration that Section 3.3 does not show (the SAGA table on the schedule of the runs
without a table).
Each script states its toy or its data and the status (proved, measured, illustration) of what it draws, prints every
number and stops if a number it relies on disagrees with the repository or with the paper.

| Figure of the appendix | File | Script | What it explains |
|---|---|---|---|
| 5 | `figures/figE1_proof_chain` | `scripts/make_figE1_proof_chain.py` | the proof of Proposition 5.9 as an exact identity and five substitutions, and the share of the gap carried by each inequality in the two orders of attribution (Section 2.2, (C1)-(C2); data: `results/B3_summary.json`) |
| 6 | `figures/figE2_nfl_plane` | `scripts/make_figE2_nfl_plane.py` | Theorem 1 in the plane: ridge logistic regression with K = 1, d = 2; curvature along a ray, the two sets that bound every norm, condition numbers in five norms and a scan |
| 7 | `figures/figE3_polar_median` | `scripts/make_figE3_polar_median.py` | Theorem 2 and Proposition 3 in the plane (K x d = 2 x 1): polar steps go to the geometric median of the data, with a SAGA table to the minimizer; the offset against the batch size |
| 8 | `figures/figE4_recipe_pipeline` | `scripts/make_figE4_recipe_pipeline.py` (drawing function in `make_figT04_recipe_pipeline.py`) | SAGA, Muon and the recipe (4) on one skeleton: estimator, map, step |
| 9 | `figures/figE5_recipe_table` | `scripts/make_figE5_recipe_table.py` (data functions in `recipe_data.py`) | why the recipe needs a table: the error of the estimator, the schedule law against lr_t^2, and two cached configurations that differ only by the estimator (data: `muon/results/M9_summary.json`, `M9_law.pkl`) |

The appendix is read only when `\WithExplanatoryFigures` is defined, so the default build is unchanged. Version with the
appendix, from `paper/`:

```bash
J=paper_saga_to_muon_with_appendix; C='\def\WithExplanatoryFigures{}\input{paper_saga_to_muon}'
pdflatex -jobname=$J "$C" && bibtex $J && pdflatex -jobname=$J "$C" && pdflatex -jobname=$J "$C"
```

The figures are redrawn from the project root with
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/scripts/make_figE<name>.py` (each takes less than a minute).
The lecture figures of `talk/` (`figT01` to `figT25`) follow the same conventions; they were built from written
specifications, each checked by an independent reviewer that recomputed its numbers.

## Where the numbers come from

| Part of the paper | Source |
|---|---|
| Finding 1, (C1)-(C5), Theorem 1 | main report `archive/report/main.pdf`; `results/B3_summary.json`; theory notes, archived in `archive/muon/notes_spectral_theory.pdf` |
| (M1)-(M4), Theorem 2, Propositions 3-4 | `muon/findings.md` (Sections 5.1-5.10), `muon/results/digest.md`, the archived theory notes; the numbers of Proposition 4 from `muon/theory_checks/` |
| Section 3.3: (M5)-(M7), Proposition 5, Figure 4 | `muon/findings.md`, Sections 5.11-5.13 (experiments M9, M5b, M9b); `muon/notes_theory.md`, facts F12-F16; `theory/B4_sampling_order.tex` and `results/findings.md`, Section 9 (sampling order of linear SAGA, main study) |

The condensation of 2026-11-20 (17 to 12 pages) shortened proofs, remarks and table cells and dropped eight secondary
citations; it changed no statement and no number. The write-ups that preceded the paper are in
[`../archive/`](../archive/README.md).
