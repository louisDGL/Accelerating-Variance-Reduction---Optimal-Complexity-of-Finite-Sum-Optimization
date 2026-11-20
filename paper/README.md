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
| `figures/` | the four figures of the paper (PDF, included by LaTeX, and PNG) |
| `scripts/` | the scripts that draw them; each reads JSON summaries, prints every number it plots and, for figures 2 and 3, stops if a number disagrees with `muon/results/digest.md` |

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

## Where the numbers come from

| Part of the paper | Source |
|---|---|
| Finding 1, (C1)-(C5), Theorem 1 | main report `archive/report/main.pdf`; `results/B3_summary.json`; theory notes, archived in `archive/muon/notes_spectral_theory.pdf` |
| (M1)-(M4), Theorem 2, Propositions 3-4 | `muon/findings.md` (Sections 5.1-5.10), `muon/results/digest.md`, the archived theory notes; the numbers of Proposition 4 from `muon/theory_checks/` |
| Section 3.3: (M5)-(M7), Proposition 5, Figure 4 | `muon/findings.md`, Sections 5.11-5.13 (experiments M9, M5b, M9b); `muon/notes_theory.md`, facts F12-F16; `theory/B4_sampling_order.tex` and `results/findings.md`, Section 9 (sampling order of linear SAGA, main study) |

The condensation of 2026-11-20 (17 to 12 pages) shortened proofs, remarks and table cells and dropped eight secondary
citations; it changed no statement and no number. The write-ups that preceded the paper are in
[`../archive/`](../archive/README.md).
