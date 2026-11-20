# Archived write-ups of the Muon study

These documents preceded the paper [`../../paper/paper_saga_to_muon.pdf`](../../paper/README.md) and were moved on
2026-11-20 to `muon/archive/`, then here, when the paper became the single write-up of the project. They are kept as they were, PDFs included; their
sources were not edited.

| Document | What it is | Superseded by |
|---|---|---|
| `notes_spectral_theory.tex` / `.pdf` (27 pages) | theory notes of 2026-11-16, revised after a six-referee review: condition numbers in the spectral geometry, polar steps as mirror descent on the nuclear norm of the gradient, the least-squares model, stochastic polar steps and the nuclear median; every statement carries its status, and Appendix A lists numerical checks (archived locally in `_archive/muon/theory_checks/`, not versioned, except the two kept in `muon/theory_checks/`) | still the place of the full proofs of Theorems 1-2 and Propositions 3-4 of the paper |
| `paper_spectral_muon.tex` / `.bib` / `.pdf` (15 pages) | condensed version of the notes in the format of a short paper | the paper |
| `report_spectral_muon.tex` / `.pdf` (8 pages) | short report on the theory only; shares `paper_spectral_muon.bib` | the paper |
| `report_spectral_muon_new.tex` / `.pdf` (11 pages) | extended report of 2026-11-17 covering the whole project (classical findings on SAGA and loopless Katyusha, variance reduction inside Muon-type steps, the two theorems); bibliography `report_spectral_muon.bib` | the paper, which reuses its labels (C1)-(C5), (M1)-(M4) |

**Statements that later results correct.** These documents predate experiments M5b, M9, M9b and facts F12-F16
(`../../muon/findings.md`, Sections 5.11-5.13, and its section "Statements of the write-ups affected"). In short: "a fixed
length cannot converge" and "variance reduction does not remove the floor" hold at a constant length only, not when the
length is cooled geometrically with a SAGA table; "no convergence of a variance-reduced polar method is proved" is no
longer true in the restricted settings of F13 and F15; their statements on speed concern batch one, whereas with
minibatches the polar step pulls ahead (M5b, M9). The paper carries the corrections.

## Rebuild

The documents include figures from `muon/figures/` and, for `figR1_classical`, from `paper/figures/`; this search path
finds them without editing the sources (checked on 2026-11-20: same page counts as the archived PDFs):

```bash
cd archive/muon
export TEXINPUTS=.:../../muon:../../paper:
for d in notes_spectral_theory paper_spectral_muon report_spectral_muon report_spectral_muon_new; do
  pdflatex $d && bibtex $d; pdflatex $d && pdflatex $d
done
```

(`notes_spectral_theory` has no bibliography file; `bibtex` only warns for it.) The header comments of the sources still
give the build directory `muon/` and the old script paths (`muon/report_figures/`, now `paper/scripts/`).
